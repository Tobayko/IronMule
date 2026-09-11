"""Conservative CPU learning and hidden-holdout replay for DATA1.

No estimator dependency is imported at module import time.  Pilot or
insufficiently separated data returns a structured ``no_learning_claim`` and
does not fit or persist a model.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import math
from pathlib import Path
import random
import statistics
import time
from typing import Any

from friday_evidence.canonical import canonical_sha256

from .contracts import ContractError, file_sha256, load_json, write_json_new
from .corpus import _build_training_dataset, _state_root
from .objectives import runtime_objective


LEARNING_SCHEMA = "ironmule.learning.v2"
_MIN_GROUPS = 3
_MIN_SESSIONS = 3
_BOOTSTRAPS = 2_000
_QUALITY_LIMIT = 1.0
_COST_RATIO_LIMIT = 0.80


class LearningError(ContractError):
    """The learning/replay boundary could not preserve its guarantees."""


def _no_claim(dataset_sha: str, reasons: list[str], *, coverage: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema": LEARNING_SCHEMA,
        "status": "no_learning_claim",
        "reasons": sorted(set(reasons)),
        "dataset_sha256": dataset_sha,
        "coverage": coverage or {},
        "model": None,
        "evaluation": None,
        "ood_policy": "no_recommendation",
        "replay_gate_passed": False,
        "requires_live_confirmation": "native_end_to_end_search_policy",
        "performance_claim": False,
        "runtime_objective": runtime_objective(),
        "runtime_goal_met": False,
    }


def _valid(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if record["status"] == "valid" and isinstance(record.get("label"), dict)]


def _coverage(records: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    valid = _valid(records)
    partitions = ("train", "validation", "holdout")
    partition_rows = {name: [record for record in valid if record["partition"] == name] for name in partitions}
    candidates = sorted({record["candidate"] for record in valid})
    strata = sorted({record["stratum_sha256"] for record in valid})
    details: dict[str, Any] = {"candidate_count": len(candidates), "stratum_count": len(strata), "partitions": {}}
    reasons: list[str] = []
    if len(candidates) < 2:
        reasons.append("candidate_support_below_two")
    for partition, rows in partition_rows.items():
        groups = {record["group_sha256"] for record in rows}
        sessions = {record["capture_session"] for record in rows}
        supported = {record["candidate"] for record in rows}
        partition_strata = {record["stratum_sha256"] for record in rows}
        details["partitions"][partition] = {
            "valid_records": len(rows),
            "groups": len(groups),
            "sessions": len(sessions),
            "candidates": len(supported),
            "strata": len(partition_strata),
        }
        if len(groups) < _MIN_GROUPS:
            reasons.append(f"{partition}_requires_three_independent_groups")
        if len(sessions) < _MIN_SESSIONS:
            reasons.append(f"{partition}_requires_three_independent_sessions")
        if supported != set(candidates):
            reasons.append(f"{partition}_candidate_support_incomplete")
    for stratum in strata:
        stratum_rows = [record for record in valid if record["stratum_sha256"] == stratum]
        present = {record["partition"] for record in stratum_rows}
        if present != set(partitions):
            reasons.append("hardware_model_strata_not_shared_across_splits")
            continue
        for partition in partitions:
            rows = [record for record in stratum_rows if record["partition"] == partition]
            if len({record["group_sha256"] for record in rows}) < _MIN_GROUPS:
                reasons.append("smallest_hardware_model_stratum_requires_three_groups_per_split")
            if len({record["capture_session"] for record in rows}) < _MIN_SESSIONS:
                reasons.append("smallest_hardware_model_stratum_requires_three_sessions_per_split")
            if {record["candidate"] for record in rows} != set(candidates):
                reasons.append("hardware_model_stratum_candidate_support_incomplete")
    # Each workload panel needs the same arms so the hidden oracle and every
    # comparator have genuinely equal support.
    panel_support: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for record in valid:
        panel_support[(record["partition"], record["run_id"], record["case_id"])].add(record["candidate"])
    if any(support != set(candidates) for support in panel_support.values()):
        reasons.append("incomplete_candidate_panel")
    details["complete_panels"] = sum(support == set(candidates) for support in panel_support.values())
    return details, reasons


def _features(record: Mapping[str, Any]) -> dict[str, Any]:
    m, k, n = record["shape"]
    return {
        "backend": record["backend"],
        "candidate": record["candidate"],
        "model_id": record["model_id"],
        "hardware": record["hardware_sha256"],
        "dtype": record["dtype"],
        "log_m": math.log2(m),
        "log_k": math.log2(k),
        "log_n": math.log2(n),
        "log_flops": math.log2(2 * m * k * n),
        "aspect_mk": m / k,
        "aspect_kn": k / n,
    }


def _model_payload(name: str, vectorizer: Any, estimator: Any) -> dict[str, Any]:
    common = {
        "kind": name,
        "feature_names": list(vectorizer.get_feature_names_out()),
        "vocabulary": dict(sorted(vectorizer.vocabulary_.items())),
    }
    if name == "ridge":
        return {
            **common,
            "intercept": float(estimator.intercept_),
            "coefficients": [float(value) for value in estimator.coef_],
            "alpha": float(estimator.alpha),
        }
    trees = []
    for stage in estimator.estimators_:
        tree = stage[0].tree_
        trees.append({
            "children_left": [int(value) for value in tree.children_left],
            "children_right": [int(value) for value in tree.children_right],
            "feature": [int(value) for value in tree.feature],
            "threshold": [float(value) for value in tree.threshold],
            "value": [float(value[0][0]) for value in tree.value],
        })
    init_constant = float(estimator.init_.constant_.ravel()[0])
    return {
        **common,
        "learning_rate": float(estimator.learning_rate),
        "init_constant": init_constant,
        "trees": trees,
    }


def _seal_model(root: Path, payload: dict[str, Any]) -> str:
    model_sha = canonical_sha256(payload)
    path = root / "models" / f"{payload['dataset_sha256']}.json"
    if path.exists() or path.is_symlink():
        from .contracts import load_json

        try:
            existing = load_json(path)
        except (OSError, ValueError) as exc:
            raise LearningError("persisted_model_invalid") from exc
        if canonical_sha256(existing) != model_sha:
            raise LearningError("dataset_hash_already_bound_to_different_model")
    else:
        try:
            write_json_new(path, payload)
        except (OSError, ValueError) as exc:
            raise LearningError("cannot_persist_model") from exc
    return model_sha


def _holdout_state(
    root: Path, claim: dict[str, Any]
) -> tuple[str, dict[str, Any] | None, Path]:
    """Seal holdout use before evaluation and return an exact cached result."""
    holdout_sha = claim["holdout_identity_sha256"]
    directory = root / "holdout-seals"
    claim_sha = canonical_sha256(claim)
    for group_sha in claim.get("holdout_group_sha256s", []):
        group_claim = {
            "schema": "ironmule.holdout-group-claim.v1",
            "group_sha256": group_sha,
            "holdout_claim_sha256": claim_sha,
        }
        group_path = directory / "groups" / f"{group_sha}.json"
        if group_path.exists() or group_path.is_symlink():
            try:
                existing_group = load_json(group_path)
            except (OSError, ValueError) as exc:
                raise LearningError("holdout_group_claim_invalid") from exc
            if canonical_sha256(existing_group) != canonical_sha256(group_claim):
                return "consumed_by_different_policy", None, directory / f"{holdout_sha}.result.json"
        else:
            try:
                write_json_new(group_path, group_claim)
            except (OSError, ValueError) as exc:
                raise LearningError("cannot_seal_holdout_group_claim") from exc
    claim_path = directory / f"{holdout_sha}.claim.json"
    result_path = directory / f"{holdout_sha}.result.json"
    if claim_path.exists() or claim_path.is_symlink():
        try:
            existing = load_json(claim_path)
        except (OSError, ValueError) as exc:
            raise LearningError("holdout_claim_invalid") from exc
        if canonical_sha256(existing) != canonical_sha256(claim):
            return "consumed_by_different_policy", None, result_path
        if not result_path.exists():
            return "claimed_without_result", None, result_path
        try:
            cached = load_json(result_path)
        except (OSError, ValueError) as exc:
            raise LearningError("holdout_result_invalid") from exc
        if cached.get("holdout_claim_sha256") != canonical_sha256(claim):
            raise LearningError("holdout_result_claim_mismatch")
        return "cached", cached, result_path
    try:
        write_json_new(claim_path, claim)
    except (OSError, ValueError) as exc:
        raise LearningError("cannot_seal_holdout_claim") from exc
    return "new", None, result_path


def _rmse(actual: list[float], predicted: list[float]) -> float:
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(actual, predicted)) / len(actual))


def _numeric_cost(value: Any, prefix: str = "") -> list[tuple[str, float]]:
    result: list[tuple[str, float]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            result.extend(_numeric_cost(child, child_prefix))
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        result.append((prefix.lower(), float(value)))
    return result


def _discovery_cost(costs: dict[str, Any]) -> float:
    values = _numeric_cost(costs)
    preferred = [number for name, number in values if name.endswith("discovery_wall_seconds")]
    if preferred:
        return sum(preferred)
    wall = [number for name, number in values if name.endswith("wall_seconds")]
    if wall:
        return sum(wall)
    selected = [
        number for name, number in values
        if any(marker in name for marker in ("discover", "search", "compile", "measure"))
        and name.endswith(("seconds", "second", "secs"))
    ]
    return sum(selected)


def _panels(records: list[dict[str, Any]], partition: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in _valid(records):
        if record["partition"] == partition:
            grouped[(record["run_id"], record["case_id"])].append(record)
    return [
        {"group": rows[0]["group_sha256"], "stratum": rows[0]["stratum_sha256"], "rows": rows}
        for _, rows in sorted(grouped.items())
    ]


def _validation_runtime_score(rows: list[dict[str, Any]], predictions: list[float]) -> float:
    """Prefer decisions that select the fastest measured plan, not just low RMSE."""
    if len(rows) != len(predictions) or not rows:
        raise LearningError("invalid_validation_predictions")
    panels: dict[tuple[str, str], list[tuple[dict, float]]] = defaultdict(list)
    for row, prediction in zip(rows, predictions):
        if not math.isfinite(float(prediction)):
            raise LearningError("nonfinite_validation_prediction")
        panels[(row["run_id"], row["case_id"])].append((row, float(prediction)))
    strata: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for panel in panels.values():
        selected, _ = min(panel, key=lambda item: (item[1], item[0]["candidate"]))
        oracle = min(float(row["label"]["median_ratio"]) for row, _ in panel)
        regret = float(selected["label"]["median_ratio"]) / oracle
        strata[selected["stratum_sha256"]][selected["group_sha256"]].append(regret)
    return max(statistics.median(statistics.median(values) for values in groups.values())
               for groups in strata.values())


def _bootstrap_upper(group_values: dict[str, list[float]], seed: int) -> tuple[float, float, float]:
    reduced = [statistics.median(values) for _, values in sorted(group_values.items()) if values]
    if not reduced:
        return math.inf, math.inf, math.inf
    point = statistics.median(reduced)
    rng = random.Random(seed)
    draws = []
    for _ in range(_BOOTSTRAPS):
        sample = [reduced[rng.randrange(len(reduced))] for _ in reduced]
        draws.append(statistics.median(sample))
    draws.sort()
    return point, draws[int(0.025 * _BOOTSTRAPS)], draws[int(0.975 * _BOOTSTRAPS)]


def _cost_ratio_upper(
    learned: dict[str, list[float]], comparator: dict[str, list[float]], seed: int
) -> tuple[float, float, float]:
    groups = sorted(set(learned) & set(comparator))
    pairs = [(sum(learned[group]), sum(comparator[group])) for group in groups]
    if not pairs:
        return math.inf, math.inf, math.inf

    def ratio(rows: list[tuple[float, float]]) -> float:
        denominator = sum(right for _, right in rows)
        return sum(left for left, _ in rows) / denominator if denominator > 0 else math.inf

    point = ratio(pairs)
    rng = random.Random(seed)
    draws = [ratio([pairs[rng.randrange(len(pairs))] for _ in pairs]) for _ in range(_BOOTSTRAPS)]
    draws.sort()
    return point, draws[int(0.025 * _BOOTSTRAPS)], draws[int(0.975 * _BOOTSTRAPS)]


def _evaluate(
    records: list[dict[str, Any]], vectorizer: Any, estimator: Any,
    train_rows: list[dict[str, Any]], gp_template: Any, x_train: Any, y_train: list[float],
    training_wall_seconds: float,
) -> dict[str, Any]:
    from sklearn.base import clone

    static_started = time.perf_counter()
    train_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in train_rows:
        train_scores[(row["stratum_sha256"], row["candidate"])].append(row["label"]["median_ratio"])
    static_training_seconds = time.perf_counter() - static_started
    panels = _panels(records, "holdout")
    policies = ("learned", "static", "seeded_random", "grid", "bo")
    quality: dict[str, dict[str, list[float]]] = {policy: defaultdict(list) for policy in policies}
    cost: dict[str, dict[str, list[float]]] = {policy: defaultdict(list) for policy in policies}
    quality_by_stratum: dict[str, dict[str, dict[str, list[float]]]] = {
        policy: defaultdict(lambda: defaultdict(list)) for policy in policies
    }
    cost_by_stratum: dict[str, dict[str, dict[str, list[float]]]] = {
        policy: defaultdict(lambda: defaultdict(list)) for policy in policies
    }
    ood = 0
    ood_by_stratum: dict[str, int] = defaultdict(int)
    train_strata = {row["stratum_sha256"] for row in train_rows}
    holdout_strata = sorted({panel["stratum"] for panel in panels})
    training_share = training_wall_seconds / max(1, len(holdout_strata))
    static_training_share = static_training_seconds / max(1, len(holdout_strata))
    charged_training: set[str] = set()
    charged_static_training: set[str] = set()
    bounds: dict[str, list[tuple[int, int]]] = {}
    for stratum in train_strata:
        stratum_rows = [row for row in train_rows if row["stratum_sha256"] == stratum]
        bounds[stratum] = [
            (
                min(row["shape"][axis] for row in stratum_rows),
                max(row["shape"][axis] for row in stratum_rows),
            )
            for axis in range(3)
        ]
    for panel in panels:
        rows = panel["rows"]
        shape = rows[0]["shape"]
        stratum_bounds = bounds.get(panel["stratum"])
        if stratum_bounds is None or any(
            not low <= shape[index] <= high
            for index, (low, high) in enumerate(stratum_bounds)
        ):
            ood += 1
            ood_by_stratum[panel["stratum"]] += 1
            continue
        actual = {row["candidate"]: float(row["label"]["median_ratio"]) for row in rows}
        costs = {row["candidate"]: _discovery_cost(row["costs"]) for row in rows}
        oracle = min(actual.values())
        learned_started = time.perf_counter()
        transformed = vectorizer.transform([_features(row) for row in rows])
        predictions = estimator.predict(transformed)
        predicted = {row["candidate"]: float(value) for row, value in zip(rows, predictions)}
        learned_choice = min(predicted, key=lambda candidate: (predicted[candidate], candidate))
        learned_cpu_seconds = time.perf_counter() - learned_started
        if panel["stratum"] not in charged_training:
            learned_cpu_seconds += training_share
            charged_training.add(panel["stratum"])
        static_started = time.perf_counter()
        static_choice = min(
            actual,
            key=lambda candidate: (
                statistics.mean(train_scores.get((panel["stratum"], candidate), [math.inf])), candidate
            ),
        )
        static_cpu_seconds = time.perf_counter() - static_started
        if panel["stratum"] not in charged_static_training:
            static_cpu_seconds += static_training_share
            charged_static_training.add(panel["stratum"])
        random_started = time.perf_counter()
        ordered = sorted(actual)
        rng = random.Random(int(canonical_sha256([panel["group"], "random"])[:16], 16))
        rng.shuffle(ordered)
        random_seen = ordered[: max(1, math.ceil(math.sqrt(len(ordered))))]
        random_choice = min(random_seen, key=lambda candidate: (actual[candidate], candidate))
        random_cpu_seconds = time.perf_counter() - random_started
        grid_started = time.perf_counter()
        grid_choice = min(actual, key=lambda candidate: (actual[candidate], candidate))
        grid_cpu_seconds = time.perf_counter() - grid_started
        # Actual fixed-kernel Gaussian-process BO.  The prior sees training
        # only; a hidden outcome enters the GP only after that arm has been
        # selected and charged to discovery cost.
        bo_started = time.perf_counter()
        bo_seen: list[str] = []
        bo_model = clone(gp_template)
        bo_x = x_train
        bo_y = list(y_train)
        bo_budget = max(1, math.ceil(math.sqrt(len(rows))))
        for _ in range(bo_budget):
            remaining = [row for row in rows if row["candidate"] not in bo_seen]
            bo_model.fit(bo_x, bo_y)
            remaining_x = vectorizer.transform([_features(row) for row in remaining])
            means, deviations = bo_model.predict(remaining_x, return_std=True)
            choice_index = min(
                range(len(remaining)),
                key=lambda index: (float(means[index] - deviations[index]), remaining[index]["candidate"]),
            )
            chosen_row = remaining[choice_index]
            bo_seen.append(chosen_row["candidate"])
            import numpy as np

            bo_x = np.vstack([bo_x, vectorizer.transform([_features(chosen_row)])])
            bo_y.append(math.log(actual[chosen_row["candidate"]]))
        bo_choice = min(bo_seen, key=lambda candidate: (actual[candidate], candidate))
        bo_cpu_seconds = time.perf_counter() - bo_started
        choices = {
            "learned": (learned_choice, [], learned_cpu_seconds),
            "static": (static_choice, [], static_cpu_seconds),
            "seeded_random": (random_choice, random_seen, random_cpu_seconds),
            "grid": (grid_choice, sorted(actual), grid_cpu_seconds),
            "bo": (bo_choice, bo_seen, bo_cpu_seconds),
        }
        for policy, (choice, explored, cpu_seconds) in choices.items():
            quality_value = actual[choice] / oracle
            cost_value = cpu_seconds + sum(costs[candidate] for candidate in explored)
            quality[policy][panel["group"]].append(quality_value)
            cost[policy][panel["group"]].append(cost_value)
            quality_by_stratum[policy][panel["stratum"]][panel["group"]].append(quality_value)
            cost_by_stratum[policy][panel["stratum"]][panel["group"]].append(cost_value)
    stratum_results: dict[str, Any] = {}
    for stratum_index, stratum in enumerate(holdout_strata):
        policy_results: dict[str, Any] = {}
        for policy_index, policy in enumerate(policies):
            median, low, high = _bootstrap_upper(
                quality_by_stratum[policy][stratum], 20261000 + stratum_index * 10 + policy_index
            )
            policy_results[policy] = {
                "oracle_runtime_ratio": median if math.isfinite(median) else None,
                "oracle_runtime_ratio_ci_low": low if math.isfinite(low) else None,
                "oracle_runtime_ratio_ci_high": high if math.isfinite(high) else None,
                "discovery_wall_seconds": sum(
                    sum(values) for values in cost_by_stratum[policy][stratum].values()
                ),
                "qualified_quality": (
                    ood_by_stratum[stratum] == 0 and math.isfinite(high) and high <= _QUALITY_LIMIT
                ),
            }
        qualified = [
            policy for policy in policies
            if policy != "learned" and policy_results[policy]["qualified_quality"]
        ]
        strongest = min(
            qualified,
            key=lambda policy: (
                policy_results[policy]["oracle_runtime_ratio"],
                policy_results[policy]["discovery_wall_seconds"],
                policy,
            ),
        ) if qualified else None
        if strongest is None:
            ratio, ratio_low, ratio_high = math.inf, math.inf, math.inf
        else:
            ratio, ratio_low, ratio_high = _cost_ratio_upper(
                cost_by_stratum["learned"][stratum],
                cost_by_stratum[strongest][stratum],
                20262000 + stratum_index,
            )
        stratum_results[stratum] = {
            "ood_panels": ood_by_stratum[stratum],
            "policies": policy_results,
            "strongest_qualified_comparator": strongest,
            "learned_to_comparator_discovery_cost_ratio": ratio if math.isfinite(ratio) else None,
            "learned_to_comparator_discovery_cost_ratio_ci_low": (
                ratio_low if math.isfinite(ratio_low) else None
            ),
            "learned_to_comparator_discovery_cost_ratio_ci_high": (
                ratio_high if math.isfinite(ratio_high) else None
            ),
            "replay_gate_passed": (
                policy_results["learned"]["qualified_quality"]
                and strongest is not None
            ),
            "secondary_discovery_cost_target_met": strongest is not None and ratio_high < _COST_RATIO_LIMIT,
        }
    summaries: dict[str, Any] = {}
    for index, policy in enumerate(policies):
        median, low, high = _bootstrap_upper(quality[policy], 20260907 + index)
        summaries[policy] = {
            "oracle_runtime_ratio": median if math.isfinite(median) else None,
            "oracle_runtime_ratio_ci_low": low if math.isfinite(low) else None,
            "oracle_runtime_ratio_ci_high": high if math.isfinite(high) else None,
            "discovery_wall_seconds": sum(sum(values) for values in cost[policy].values()),
            "qualified_quality": (
                ood == 0
                and math.isfinite(high)
                and high <= _QUALITY_LIMIT
                and all(
                    value["policies"][policy]["qualified_quality"]
                    for value in stratum_results.values()
                )
            ),
        }
        if policy == "bo":
            summaries[policy]["method"] = "fixed_kernel_gaussian_process_lcb"
    replay_gate_passed = bool(stratum_results) and ood == 0 and all(
        value["replay_gate_passed"] for value in stratum_results.values()
    )
    return {
        "hidden_holdout_groups_evaluated": len(quality["learned"]),
        "ood_no_recommendation_panels": ood,
        "policies": summaries,
        "strata": stratum_results,
        "training_wall_seconds": training_wall_seconds,
        "quality_limit": _QUALITY_LIMIT,
        "discovery_cost_ratio_limit": _COST_RATIO_LIMIT,
        "replay_gate_passed": replay_gate_passed,
        "requires_live_confirmation": "native_end_to_end_search_policy",
        "performance_claim": False,
        "runtime_objective": runtime_objective(),
        "runtime_goal_met": False,
    }


def train_and_evaluate(state_dir: str | Path) -> dict[str, Any]:
    """Fit ridge then a small GBDT and evaluate once on grouped hidden data."""
    root = _state_root(state_dir)
    dataset = _build_training_dataset(root)
    dataset_sha = dataset["dataset_sha256"]
    records = dataset["records"]
    coverage, reasons = _coverage(records)
    if reasons:
        return _no_claim(dataset_sha, reasons, coverage=coverage)
    training_started = time.perf_counter()
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
        from sklearn.linear_model import Ridge
    except (ImportError, OSError) as exc:
        return _no_claim(dataset_sha, [f"learning_dependency_unavailable:{type(exc).__name__}"], coverage=coverage)
    train_rows = [record for record in _valid(records) if record["partition"] == "train"]
    validation_rows = [record for record in _valid(records) if record["partition"] == "validation"]
    vectorizer = DictVectorizer(sparse=False, sort=True)
    x_train = vectorizer.fit_transform([_features(record) for record in train_rows])
    y_train = [math.log(float(record["label"]["median_ratio"])) for record in train_rows]
    x_validation = vectorizer.transform([_features(record) for record in validation_rows])
    y_validation = [math.log(float(record["label"]["median_ratio"])) for record in validation_rows]
    ridge = Ridge(alpha=1.0)
    ridge.fit(x_train, y_train)
    gbdt = GradientBoostingRegressor(
        n_estimators=48, learning_rate=0.05, max_depth=2, min_samples_leaf=2,
        loss="huber", random_state=20260907,
    )
    gbdt.fit(x_train, y_train)
    predictions = {"ridge": [float(value) for value in ridge.predict(x_validation)],
                   "small_gbdt": [float(value) for value in gbdt.predict(x_validation)]}
    candidates = {"ridge": (ridge, _rmse(y_validation, predictions["ridge"])),
                  "small_gbdt": (gbdt, _rmse(y_validation, predictions["small_gbdt"]))}
    runtime_scores = {name: _validation_runtime_score(validation_rows, values)
                      for name, values in predictions.items()}
    selected_name = min(candidates, key=lambda name: (runtime_scores[name], candidates[name][1], name))
    selected, selected_rmse = candidates[selected_name]
    gp_template = GaussianProcessRegressor(
        kernel=DotProduct(sigma_0=1.0) + WhiteKernel(noise_level=0.05),
        optimizer=None,
        normalize_y=True,
        random_state=20260907,
    )
    training_wall_seconds = time.perf_counter() - training_started
    estimator_payload = _model_payload(selected_name, vectorizer, selected)
    policy_code_sha = file_sha256(Path(__file__))
    frozen_policy = {
        "schema": "ironmule.offline-policy.v2",
        "runtime_objective": runtime_objective(),
        "validation_runtime_scores": runtime_scores,
        "dataset_sha256": dataset_sha,
        "selected_model": selected_name,
        "validation_log_rmse": selected_rmse,
        "estimator": estimator_payload,
        "policy_code_sha256": policy_code_sha,
        "quality_limit": _QUALITY_LIMIT,
        "discovery_cost_ratio_limit": _COST_RATIO_LIMIT,
        "bootstraps": _BOOTSTRAPS,
    }
    claim = {
        "schema": "ironmule.holdout-claim.v1",
        "holdout_identity_sha256": dataset["holdout"]["projection_sha256"],
        "dataset_sha256": dataset_sha,
        "frozen_policy_sha256": canonical_sha256(frozen_policy),
        "policy_code_sha256": policy_code_sha,
        "holdout_group_sha256s": sorted({
            record["group_sha256"]
            for record in records if record["partition"] == "holdout"
        }),
    }
    holdout_status, cached, result_path = _holdout_state(root, claim)
    if holdout_status == "cached":
        assert cached is not None
        return {**cached, "status": "cached_offline_replay", "cache_reused": True}
    if holdout_status != "new":
        result = _no_claim(dataset_sha, [f"holdout_{holdout_status}"], coverage=coverage)
        result["holdout_identity_sha256"] = claim["holdout_identity_sha256"]
        return result
    evaluation = _evaluate(
        records, vectorizer, selected, train_rows, gp_template, x_train, y_train,
        training_wall_seconds,
    )
    persisted = {
        "schema": "ironmule.runtime-ranking-model.v2",
        "dataset_sha256": dataset_sha,
        "split_protocol": dataset["split_protocol"],
        "selected_on": "validation",
        "selected_model": selected_name,
        "validation_log_rmse": selected_rmse,
        "coverage": coverage,
        "ood_policy": "no_recommendation",
        "estimator": estimator_payload,
        "evaluation": evaluation,
        "performance_claim": False,
        "runtime_objective": runtime_objective(),
        "runtime_goal_met": False,
    }
    model_sha = _seal_model(root, persisted)
    result = {
        "schema": LEARNING_SCHEMA,
        "status": (
            "offline_replay_gate_passed" if evaluation["replay_gate_passed"] else "evaluated"
        ),
        "reasons": (
            ["native_end_to_end_search_policy_confirmation_required"]
            if evaluation["replay_gate_passed"]
            else ["preregistered_offline_replay_gate_not_met"]
        ),
        "dataset_sha256": dataset_sha,
        "holdout_identity_sha256": claim["holdout_identity_sha256"],
        "holdout_claim_sha256": canonical_sha256(claim),
        "coverage": coverage,
        "model": {
            "schema": persisted["schema"],
            "sha256": model_sha,
            "kind": selected_name,
            "validation_log_rmse": selected_rmse,
            "bound_dataset_sha256": dataset_sha,
        },
        "evaluation": evaluation,
        "ood_policy": "no_recommendation",
        "replay_gate_passed": evaluation["replay_gate_passed"],
        "requires_live_confirmation": "native_end_to_end_search_policy",
        "performance_claim": False,
        "runtime_objective": runtime_objective(),
        "runtime_goal_met": False,
    }
    try:
        write_json_new(result_path, result)
    except (OSError, ValueError) as exc:
        raise LearningError("cannot_seal_holdout_result") from exc
    return result


__all__ = ["LearningError", "train_and_evaluate"]
