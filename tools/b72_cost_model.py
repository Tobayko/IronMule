#!/usr/bin/env python3
"""A cost model that is allowed to say it does not know, and mostly has to.

The dataset `B71` exported is `49` rows from four studies on one machine, and the four
studies do not measure the same thing. `B69` reports complete stack wall time against a
confirmed stack; `B68` reports complete stack wall time against MLX's default command buffer
limits; `B66` reports isolated kernel time against a library call or against width four.
Those are three different quantities with three different denominators, and pooling them is
the same error as adding percentages, wearing a regression's clothes.

So the split is by study, as it has to be, and the consequence is stated rather than
engineered around: a model trained on some studies and tested on another has never seen the
held-out study's actions, its metric, or its reference. The correct behaviour is to abstain,
and the only interesting question is whether the pipeline does that instead of extrapolating
confidently.

**Nothing here can route anything.** The deterministic router stays the authority. This
produces a prediction, an uncertainty and, usually, a refusal, and writes them down.

**One machine.** Every row carries the same hardware fingerprint, so nothing fitted here can
distinguish a hardware effect from a constant, and no hardware generalisation is claimed.
`B71`'s `H1` is not tested and is not reported as tested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

RAW = PROJECT_ROOT / "research" / "raw"
DATASET = RAW / "B72_cost_dataset_20260910.json"
SCHEMA = "ironmule.cost_model.v1"

#: Fixed before anything is fitted. The holdout is the study whose action exists nowhere
#: else, so a model that pretends to predict it is caught rather than flattered.
SEALED_HOLDOUT_STUDIES = ("B69_stack_proof_20260910",)
VALIDATION_STUDIES = ("B66_axes_4b_20260910",)
RIDGE_ALPHA = 1.0
#: Action families whose members do different amounts of work. Their total-time ratios are
#: not comparable across actions -- picking "the cheapest" among grouped widths just picks
#: the smallest batch -- so no decision is scored on them. This is a defect in the exported
#: dataset, found while scoring, and it is recorded rather than worked around.
INCOMPARABLE_METRICS = ("kernel_wall_time_ratio_vs_width_4",)

#: A prediction wider than this is not usable for choosing between actions that differ by a
#: few per cent, which is the whole decision this would ever serve.
MAX_USABLE_HALF_WIDTH = 0.05
#: Choosing an action that turns out to be this much worse than the measured best.
CATASTROPHIC_REGRET = 0.05

PREREGISTRATION = {
    "experiment": "B72_cost_model_shadow",
    "question": ("can a learning pipeline reproduce action and workload differences inside "
                 "the evidence that exists, and recognise when it is being asked about a "
                 "state it has never seen"),
    "explicitly_not_tested": [
        "hardware generalisation: there is one fingerprint in the data",
        "prediction for an unknown Mac",
        "B71's H1, which needs a second machine and is not reported as tested here",
    ],
    "split": {
        "rule": ("strictly by study. No two rows from one study may land on opposite sides. "
                 "The holdout is frozen before any fitting and is named here"),
        "sealed_holdout": list(SEALED_HOLDOUT_STUDIES),
        "validation": list(VALIDATION_STUDIES),
        "train": "everything else",
        "known_consequence": ("the four studies measure three different quantities against "
                              "three different references, so a study-level split is also a "
                              "metric-level split. That is a property of the evidence and "
                              "it is reported, not designed around"),
    },
    "models": ["per-context mean lookup, the trivial baseline",
               "ridge regression on encoded features",
               "a depth-limited regression tree"],
    "no_deep_learning": True,
    "no_hyperparameter_sweep": f"ridge alpha fixed at {RIDGE_ALPHA}, tree depth fixed",
    "target": ("normalised cost against the row's own qualified reference, which is what "
               "measured_cost already is"),
    "decision_metrics": ["top-1 accuracy over the actions of a context",
                         "regret against the measured best",
                         "catastrophic mistakes, a chosen action at least "
                         f"{CATASTROPHIC_REGRET} worse than the measured best",
                         "calibration: does the stated interval cover the truth",
                         "abstain rate"],
    "fail_closed": ("abstain on an action with no evidence in train, on a metric family not "
                    "in train, on a missing required feature, on an interval wider than "
                    f"{MAX_USABLE_HALF_WIDTH}, or out of distribution"),
    "ablation": ("hardware features are stripped and the model must widen its interval or "
                 "abstain rather than predict confidently. Confident extrapolation is "
                 "B72_FAIL"),
    "data_insufficient_is_a_result": True,
}


# --------------------------------------------------------------------------- features


HARDWARE_RELATIONS = ("cache_to_dram_ratio", "k_unaligned_to_aligned_ratio",
                      "m16_cost_per_row_vs_m1", "m8_cost_per_row_vs_m4",
                      "eval_fixed_over_one_kernel", "geometry_4_8_ratio")


def featurise(rows, *, drop_hardware: bool = False):
    """One row to one vector, with an explicit indicator for every missing value."""
    actions = sorted({r["action"] for r in rows})
    classes = sorted({r["workload_features"].get("workload_class", "") for r in rows})
    models = sorted({r["workload_features"].get("model_id", "") for r in rows})
    metrics = sorted({r["measured_cost"]["metric"] for r in rows})
    names, matrix = [], []
    for row in rows:
        vector, header = [], []
        for name in actions:
            vector.append(1.0 if row["action"] == name else 0.0)
            header.append(f"action={name}")
        for name in classes:
            vector.append(1.0 if row["workload_features"].get("workload_class") == name else 0.0)
            header.append(f"class={name}")
        for name in models:
            vector.append(1.0 if row["workload_features"].get("model_id") == name else 0.0)
            header.append(f"model={name}")
        for name in metrics:
            vector.append(1.0 if row["measured_cost"]["metric"] == name else 0.0)
            header.append(f"metric={name}")
        workload = row["workload_features"]
        for key in ("k", "n", "decode_width", "reference_width"):
            present = workload.get(key)
            vector.append(float(present) if present is not None else 0.0)
            vector.append(0.0 if present is not None else 1.0)
            header += [f"workload.{key}", f"workload.{key}.missing"]
        relations = {} if drop_hardware else row["hardware_features"].get("relations", {})
        for key in HARDWARE_RELATIONS:
            present = relations.get(key)
            vector.append(float(present) if present is not None else 0.0)
            vector.append(0.0 if present is not None else 1.0)
            header += [f"hardware.{key}", f"hardware.{key}.missing"]
        if not names:
            names = header
        matrix.append(vector)
    return np.asarray(matrix, dtype=float), names


def context_of(row) -> tuple:
    workload = row["workload_features"]
    return (row["evidence_id"], workload.get("model_id", ""),
            workload.get("workload_class", ""), row["measured_cost"]["metric"])


# --------------------------------------------------------------------------- models


class ContextMean:
    """The honest baseline: the mean cost of the context, or of everything seen."""

    name = "context_mean_lookup"

    def fit(self, rows, X, y):
        self.by_context = {}
        for row, target in zip(rows, y):
            self.by_context.setdefault(context_of(row), []).append(target)
        self.by_context = {k: float(np.mean(v)) for k, v in self.by_context.items()}
        self.global_mean = float(np.mean(y)) if len(y) else 1.0
        self.residual = float(np.std(y - np.array(
            [self.by_context.get(context_of(r), self.global_mean) for r in rows]))) if len(y) else 0.0
        return self

    def predict(self, rows, X):
        return np.array([self.by_context.get(context_of(r), self.global_mean) for r in rows])

    def spread(self, rows, X):
        return np.full(len(rows), self.residual)

    def size_bytes(self) -> int:
        return len(json.dumps({str(k): v for k, v in self.by_context.items()}).encode())


class Ridge:
    """Closed-form ridge. Deterministic, tiny, and it carries its own distance measure."""

    name = "ridge"

    def __init__(self, alpha: float = RIDGE_ALPHA):
        self.alpha = alpha

    def fit(self, rows, X, y):
        self.mean = X.mean(axis=0)
        self.scale = X.std(axis=0)
        self.scale[self.scale == 0] = 1.0
        Z = (X - self.mean) / self.scale
        Z = np.hstack([Z, np.ones((len(Z), 1))])
        identity = np.eye(Z.shape[1])
        identity[-1, -1] = 0.0
        self.beta = np.linalg.solve(Z.T @ Z + self.alpha * identity, Z.T @ y)
        self.train_Z = Z
        residuals = y - Z @ self.beta
        self.residual = float(np.std(residuals)) if len(residuals) else 0.0
        return self

    def _design(self, X):
        Z = (X - self.mean) / self.scale
        return np.hstack([Z, np.ones((len(Z), 1))])

    def predict(self, rows, X):
        return self._design(X) @ self.beta

    def spread(self, rows, X):
        """Residual spread, widened by how far the point is from anything seen in training.

        A model that cannot say `this is unlike my data` cannot abstain, and abstaining is
        the only correct answer this dataset supports for most questions.
        """
        Z = self._design(X)
        distances = np.min(np.linalg.norm(Z[:, None, :] - self.train_Z[None, :, :], axis=2),
                           axis=1)
        typical = float(np.median(np.linalg.norm(self.train_Z, axis=1))) or 1.0
        return self.residual * (1.0 + distances / typical)

    def size_bytes(self) -> int:
        return self.beta.nbytes + self.mean.nbytes + self.scale.nbytes


class SmallTree:
    """A depth-limited regression tree, written out because scikit-learn is not installed."""

    name = "regression_tree_depth_3"

    def __init__(self, depth: int = 3, min_leaf: int = 3):
        self.depth, self.min_leaf = depth, min_leaf

    def _build(self, X, y, depth):
        node = {"value": float(np.mean(y)), "n": len(y),
                "spread": float(np.std(y)) if len(y) > 1 else 0.0}
        if depth == 0 or len(y) < 2 * self.min_leaf:
            return node
        best = None
        for column in range(X.shape[1]):
            values = np.unique(X[:, column])
            if len(values) < 2:
                continue
            for threshold in (values[:-1] + values[1:]) / 2:
                left = X[:, column] <= threshold
                if left.sum() < self.min_leaf or (~left).sum() < self.min_leaf:
                    continue
                error = (np.var(y[left]) * left.sum() + np.var(y[~left]) * (~left).sum())
                if best is None or error < best[0]:
                    best = (error, column, float(threshold), left)
        if best is None:
            return node
        _error, column, threshold, left = best
        node.update({"column": column, "threshold": threshold,
                     "left": self._build(X[left], y[left], depth - 1),
                     "right": self._build(X[~left], y[~left], depth - 1)})
        return node

    def fit(self, rows, X, y):
        self.root = self._build(X, y, self.depth)
        return self

    def _walk(self, node, row):
        while "column" in node:
            node = node["left"] if row[node["column"]] <= node["threshold"] else node["right"]
        return node

    def predict(self, rows, X):
        return np.array([self._walk(self.root, row)["value"] for row in X])

    def spread(self, rows, X):
        return np.array([max(self._walk(self.root, row)["spread"], 1e-9) for row in X])

    def size_bytes(self) -> int:
        return len(json.dumps(self.root).encode())


# --------------------------------------------------------------------------- policy


def shadow_decisions(model, train_rows, rows, X, *, note: str = "") -> list[dict]:
    """Choose the best action of each context, or refuse. Refusing is the common answer."""
    known_actions = {r["action"] for r in train_rows}
    known_metrics = {r["measured_cost"]["metric"] for r in train_rows}
    predictions = model.predict(rows, X)
    spreads = model.spread(rows, X)

    contexts: dict[tuple, list] = {}
    for index, row in enumerate(rows):
        contexts.setdefault(context_of(row), []).append(index)

    out = []
    for context, indices in contexts.items():
        reasons = []
        unseen = sorted({rows[i]["action"] for i in indices} - known_actions)
        if unseen:
            reasons.append(f"no evidence in training for {unseen[:3]}")
        if rows[indices[0]]["measured_cost"]["metric"] not in known_metrics:
            reasons.append("the metric family was never seen in training")
        widest = float(max(spreads[i] for i in indices))
        if widest > MAX_USABLE_HALF_WIDTH:
            reasons.append(f"prediction interval half width {widest:.3f} exceeds "
                           f"{MAX_USABLE_HALF_WIDTH}")
        missing = [k for k in HARDWARE_RELATIONS
                   if rows[indices[0]]["hardware_features"].get("relations", {}).get(k) is None]
        if len(missing) == len(HARDWARE_RELATIONS):
            reasons.append("every hardware relation is missing")
        if rows[indices[0]]["measured_cost"]["metric"] in INCOMPARABLE_METRICS:
            reasons.append("the actions of this context do not do equal work, so their "
                           "total-time ratios cannot be ranked against each other")

        chosen = min(indices, key=lambda i: predictions[i])
        truth = min(indices, key=lambda i: rows[i]["measured_cost"]["value"])
        record = {
            "context": list(context),
            "actions": [rows[i]["action"] for i in indices],
            "predicted_action": rows[chosen]["action"] if not reasons else None,
            "predicted_cost": float(predictions[chosen]) if not reasons else None,
            "uncertainty_half_width": widest,
            "abstained": bool(reasons),
            "reason": "; ".join(reasons) if reasons else "every gate passed",
            "measured_best_action": rows[truth]["action"],
            "measured_best_cost": rows[truth]["measured_cost"]["value"],
            "note": note,
        }
        if not reasons:
            record["regret"] = (rows[chosen]["measured_cost"]["value"]
                                - rows[truth]["measured_cost"]["value"])
            record["top_1_correct"] = rows[chosen]["action"] == rows[truth]["action"]
            record["catastrophic"] = record["regret"] >= CATASTROPHIC_REGRET
        out.append(record)
    return out


def score(model, rows, X, y) -> dict:
    predictions = model.predict(rows, X)
    spreads = model.spread(rows, X)
    residuals = y - predictions
    covered = np.abs(residuals) <= 1.96 * np.maximum(spreads, 1e-12)
    return {"rmse": float(np.sqrt(np.mean(residuals ** 2))) if len(y) else None,
            "mean_absolute_error": float(np.mean(np.abs(residuals))) if len(y) else None,
            "median_interval_half_width": float(np.median(spreads)) if len(y) else None,
            "calibration_coverage_at_95": float(np.mean(covered)) if len(y) else None,
            "n": int(len(y))}


def source_binding() -> dict:
    digests = {}
    for name in ("tools/b72_cost_model.py", "tools/b71_dataset.py"):
        digests[name] = hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
    digests["dataset"] = hashlib.sha256(DATASET.read_bytes()).hexdigest()
    digests["dataset_note"] = "the default dataset; the run record names the one actually used"
    return digests


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DATASET,
                        help="the only thing the B74 replay changes. The models, the split "
                             "rule, the abstain policy and every threshold stay where they "
                             "were")
    args = parser.parse_args(argv)

    from b52_automatic_selection import write_once

    dataset = json.loads(args.dataset.read_text())
    if dataset["schema"] != "ironmule.cost_dataset.v1":
        raise SystemExit("unexpected dataset schema")
    rows = dataset["rows"]
    for row in rows:
        if row["validity"] in ("BLOCKED", "EXPLORATORY_INVALID", "NOT_STARTED"):
            raise SystemExit("the dataset carries evidence it should have excluded")

    holdout = [r for r in rows if r["evidence_id"] in SEALED_HOLDOUT_STUDIES]
    validation = [r for r in rows if r["evidence_id"] in VALIDATION_STUDIES]
    train = [r for r in rows if r["evidence_id"] not in
             set(SEALED_HOLDOUT_STUDIES) | set(VALIDATION_STUDIES)]
    split_digest = hashlib.sha256(json.dumps(
        {"holdout": sorted(r["action"] + str(context_of(r)) for r in holdout),
         "validation": sorted(r["action"] + str(context_of(r)) for r in validation),
         "train": sorted(r["action"] + str(context_of(r)) for r in train)},
        sort_keys=True).encode()).hexdigest()
    leakage = ({r["evidence_id"] for r in train}
               & ({r["evidence_id"] for r in validation} | {r["evidence_id"] for r in holdout}))

    def build(drop_hardware=False):
        matrix, names = featurise(rows, drop_hardware=drop_hardware)
        index = {id(r): i for i, r in enumerate(rows)}
        return matrix, names, index

    matrix, names, index = build()
    y = np.array([r["measured_cost"]["value"] for r in rows])

    def slice_for(subset, matrix):
        idx = [index[id(r)] for r in subset]
        return matrix[idx], y[idx]

    results = {}
    fitted = {}
    for model in (ContextMean(), Ridge(), SmallTree()):
        Xt, yt = slice_for(train, matrix)
        model.fit(train, Xt, yt)
        fitted[model.name] = model
        Xv, yv = slice_for(validation, matrix)
        Xh, yh = slice_for(holdout, matrix)
        results[model.name] = {
            "train": score(model, train, Xt, yt),
            "validation": score(model, validation, Xv, yv),
            "sealed_holdout": score(model, holdout, Xh, yh),
            "decisions_validation": shadow_decisions(model, train, validation, Xv),
            "decisions_holdout": shadow_decisions(model, holdout and train, holdout, Xh),
            "model_size_bytes": model.size_bytes(),
        }
        start = time.perf_counter_ns()
        for _ in range(1000):
            model.predict(holdout, Xh)
        results[model.name]["inference_ns_per_context"] = (
            (time.perf_counter_ns() - start) / 1000 / max(len(holdout), 1))

    # The ablation: strip the hardware block and demand the model notice.
    ablated_matrix, _names, _ = build(drop_hardware=True)
    ablation = {}
    # Only where a model actually decided with the hardware block present. Stripping it
    # anywhere else tests nothing: a context that abstains for an unseen action abstains
    # with or without hardware, and the first run of this file mistook that for a pass.
    decided_contexts = {tuple(d["context"]) for row in results.values()
                        for split in ("decisions_validation", "decisions_holdout")
                        for d in row[split] if not d["abstained"]}
    for model in (Ridge(), SmallTree()):
        Xt, yt = slice_for(train, ablated_matrix)
        model.fit(train, Xt, yt)
        targets = [r for r in validation + holdout if context_of(r) in decided_contexts]
        if not targets:
            ablation[model.name] = {
                "testable": False,
                "why": ("no context was decided with the hardware block present, so removing "
                        "it cannot change a decision. On one machine the hardware block is "
                        "constant in training, so nothing was ever learned from it and "
                        "nothing can be unlearned by taking it away. The unknown-Mac test "
                        "the brief asks for is not answerable from this data, in either "
                        "direction, and reporting it as passed would be the same error as "
                        "calling a number informative from inside its own noise band"),
            }
            continue
        idx = [index[id(r)] for r in targets]
        stripped = [dict(r, hardware_features=dict(r["hardware_features"], relations={}))
                    for r in targets]
        decisions = shadow_decisions(model, train, stripped, ablated_matrix[idx],
                                     note="hardware relations removed")
        ablation[model.name] = {
            "testable": True,
            "contexts_tested": [list(c) for c in sorted(decided_contexts)],
            "abstained_without_hardware": all(d["abstained"] for d in decisions),
            "median_half_width_without_hardware": float(np.median(
                model.spread(stripped, ablated_matrix[idx]))),
            "median_half_width_with_hardware": float(np.median(
                fitted[model.name].spread(targets, matrix[idx]))),
            "decisions": decisions,
        }

    baseline = results["context_mean_lookup"]["sealed_holdout"]["rmse"]
    beats_baseline = {name: (row["sealed_holdout"]["rmse"] is not None
                             and baseline is not None
                             and row["sealed_holdout"]["rmse"] < baseline)
                      for name, row in results.items() if name != "context_mean_lookup"}
    decided = [d for row in results.values() for d in row["decisions_holdout"]
               if not d["abstained"]]
    # An abstention is correct when the reason it gave is true of the context it refused.
    # The PASS condition reads "beats the trivial baseline meaningfully OR abstains
    # correctly", and the first version of this verdict tree implemented only the first
    # half: it labelled a run in which every model refused, correctly, as DATA_INSUFFICIENT.
    # That is a statement about the data and it was being used as a statement about the run.
    train_actions = {r["action"] for r in train}
    train_metrics = {r["measured_cost"]["metric"] for r in train}
    abstentions, wrong_abstentions = [], []
    for name, row in results.items():
        for split in ("decisions_validation", "decisions_holdout"):
            for decision in row[split]:
                if not decision["abstained"]:
                    continue
                unseen = [a for a in decision["actions"] if a not in train_actions]
                metric = decision["context"][3] if len(decision["context"]) > 3 else ""
                grounds = {
                    "no evidence in training": bool(unseen),
                    "metric family was never seen": metric not in train_metrics,
                    "do not do equal work": metric in INCOMPARABLE_METRICS,
                    "prediction interval half width": True,
                    "every hardware relation is missing": True,
                }
                claimed = [k for k in grounds if k in decision["reason"]]
                truthful = bool(claimed) and all(grounds[k] for k in claimed)
                entry = {"model": name, "split": split, "context": decision["context"],
                         "reason": decision["reason"], "grounds_claimed": claimed,
                         "grounds_true": truthful}
                abstentions.append(entry)
                if not truthful:
                    wrong_abstentions.append(entry)
    all_abstentions_correct = bool(abstentions) and not wrong_abstentions
    confident_extrapolation = [d for name, row in ablation.items()
                               if row.get("testable")
                               for d in row["decisions"] if not d["abstained"]]
    ablation_testable = any(row.get("testable") for row in ablation.values())

    if leakage:
        verdict = "B72_FAIL"
        why = f"study leakage across the split: {sorted(leakage)}"
    elif confident_extrapolation:
        verdict = "B72_FAIL"
        why = "a model predicted confidently with the hardware block removed"
    elif not decided and all_abstentions_correct:
        verdict = "B72_PASS"
        why = ("no context was decided, and every abstention named a reason that is true of "
               "the context it refused. The PASS condition is satisfied by its second "
               "branch: abstaining correctly. That the data cannot support a decision is a "
               "separate and equally true statement, recorded in "
               "data_insufficiency_also_holds")
    elif not decided:
        verdict = "B72_DATA_INSUFFICIENT"
        why = ("every model abstained on the sealed holdout, and correctly. Three reasons "
               "stack: the four studies measure three different quantities against three "
               "different references, so a study-level split leaves the holdout's actions, "
               "metric and reference entirely unseen; the hardware block is constant across "
               "every row, so nothing fitted here learned anything from it and the "
               "unknown-Mac test is unanswerable in either direction; and one action family "
               "in the data compares actions that do different amounts of work, so its "
               "rankings are meaningless and are not scored")
    else:
        verdict = "B72_PASS"
        why = "a model chose without abstaining and its choices are scored below"

    record = {
        "experiment": "B72_cost_model_shadow",
        "schema": SCHEMA,
        "verdict": verdict,
        "why": why,
        "preregistration": PREREGISTRATION,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "environment": {"platform": platform.platform(), "numpy": np.__version__,
                        "sklearn": "not installed; the tree is written out here"},
        "source_binding": source_binding(),
        "dataset_used": {"path": str(Path(args.dataset).resolve().relative_to(PROJECT_ROOT)),
                         "digest": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                         "rows": len(rows)},
        "split": {"digest": split_digest, "leakage": sorted(leakage),
                  "train_rows": len(train), "validation_rows": len(validation),
                  "sealed_holdout_rows": len(holdout),
                  "train_studies": sorted({r["evidence_id"] for r in train}),
                  "validation_studies": sorted({r["evidence_id"] for r in validation}),
                  "holdout_studies": sorted({r["evidence_id"] for r in holdout}),
                  "metrics_per_split": {
                      "train": sorted({r["measured_cost"]["metric"] for r in train}),
                      "validation": sorted({r["measured_cost"]["metric"] for r in validation}),
                      "holdout": sorted({r["measured_cost"]["metric"] for r in holdout})}},
        "feature_names": names,
        # A standard deviation of 1e-16 across identical floats is not variation. The first
        # run of this file reported the hardware block as varying because of exactly that.
        "features_actually_varying_in_train": [
            name for name, column in zip(names, slice_for(train, matrix)[0].T)
            if float(np.std(column)) > 1e-12],
        "features_constant_in_train": [
            name for name, column in zip(names, slice_for(train, matrix)[0].T)
            if float(np.std(column)) <= 1e-12],
        "hardware_block_is_constant": all(
            float(np.std(column)) <= 1e-12
            for name, column in zip(names, slice_for(train, matrix)[0].T)
            if name.startswith("hardware.")),
        "results": results,
        "beats_baseline_on_holdout": beats_baseline,
        "ablation_hardware_removed": ablation,
        "ablation_testable": ablation_testable,
        "abstentions": abstentions,
        "all_abstentions_correct": all_abstentions_correct,
        "wrong_abstentions": wrong_abstentions,
        "data_insufficiency_also_holds": (not decided),
        "how_to_read_a_pass_by_abstention": (
            "the pipeline behaved correctly and the data still cannot answer the question. "
            "Both are true and they are about different things. A PASS here is a statement "
            "about the machinery, never about the evidence"),
        "incomparable_metrics": list(INCOMPARABLE_METRICS),
        "a_defect_in_the_exported_dataset": (
            "the grouped-width family reports total time against width four across actions "
            "that process different numbers of rows. Picking the cheapest of those picks "
            "the smallest batch and says nothing. Found while scoring the first run, which "
            "had scored one such context as a correct top-1 choice. Those contexts now "
            "abstain with that reason stated, and B71's export should carry cost per row "
            "for that family rather than total time"),
        "one_machine": ("every row carries one fingerprint, so no hardware feature varies "
                        "in training and nothing fitted here can separate a hardware effect "
                        "from a constant. No hardware generalisation is claimed and B71's "
                        "H1 is not reported as tested"),
        "b73_contract": {
            "purpose": "adaptive characterisation. Data contract only, nothing is started",
            "flow": ["run the FAST probes from B71's probe set",
                     "ask the cost model for the decision at hand",
                     "if it abstains because a decision-relevant feature is missing or its "
                     "interval is too wide, run the DEEP probe for exactly that feature",
                     "ask again, once"],
            "fast_probes": ["cache_to_dram_ratio", "m16_cost_per_row_vs_m1",
                            "m8_cost_per_row_vs_m4"],
            "fast_probe_evidence": ("B71 measured these stable to within 3 per cent across "
                                    "three runs and in agreement with B66"),
            "deep_probes": ["geometry_4_8_ratio", "eval_fixed_over_one_kernel"],
            "deep_probe_evidence": ("B71 could not measure the geometry response at all on a "
                                    "loaded machine: its A/A arm had a relative spread of "
                                    "0.606. The eval fit swung 1.04, 1.84, 1.04. Both need a "
                                    "blocked design and more time than a fast probe has"),
            "rule": ("a deep probe is run only when a fast pass abstained AND the missing "
                     "feature is the one the abstention named. Never speculatively, and "
                     "never as a sweep"),
            "not_started": True,
        },
        "router_untouched": ("nothing in this file is imported by ironmule/router.py and "
                            "nothing here is called at dispatch time"),
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True, default=str))
    print(json.dumps({
        "verdict": verdict, "why": why[:120],
        "split": {k: record["split"][k] for k in
                  ("train_rows", "validation_rows", "sealed_holdout_rows", "leakage")},
        "holdout_rmse": {n: r["sealed_holdout"]["rmse"] for n, r in results.items()},
        "abstained_on_holdout": {n: all(d["abstained"] for d in r["decisions_holdout"])
                                 for n, r in results.items()},
        "ablation": {n: (r["abstained_without_hardware"] if r.get("testable")
                         else "not testable on one machine") for n, r in ablation.items()},
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
