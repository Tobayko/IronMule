"""Compare sealed B27 pre/post engineering baselines without promoting evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import tempfile
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "ironmule.post_change_comparison.v1"
ARMS = ("interactive", "throughput")
METRICS = ("outer_wall_ms", "physical_tokens_per_second")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path.name}")
    return value


def _samples(record: dict[str, Any], arm: str, metric: str) -> list[float]:
    rows = record["benchmark"]["arms"][arm]["raw"]
    values = [float(row["snapshot"][metric]) for row in rows if row.get("phase") == "measure"]
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError(f"invalid measured samples for {arm}.{metric}")
    return values


def bootstrap_ratio(
    after: list[float], before: list[float], *, resamples: int = 10_000,
    seed: int = 20260828,
) -> dict[str, Any]:
    if not after or not before or resamples < 100:
        raise ValueError("bootstrap needs non-empty samples and at least 100 resamples")
    denominator = statistics.median(before)
    if denominator <= 0:
        raise ValueError("bootstrap denominator must be positive")
    rng = random.Random(seed)
    ratios = []
    for _ in range(resamples):
        after_draw = [after[rng.randrange(len(after))] for _ in after]
        before_draw = [before[rng.randrange(len(before))] for _ in before]
        base = statistics.median(before_draw)
        if base <= 0:
            raise ValueError("bootstrap draw denominator must be positive")
        ratios.append(statistics.median(after_draw) / base)
    ratios.sort()
    low = ratios[int(0.025 * (len(ratios) - 1))]
    high = ratios[int(0.975 * (len(ratios) - 1))]
    return {
        "median_ratio": statistics.median(after) / denominator,
        "ci_low": low,
        "ci_high": high,
        "before_median": denominator,
        "after_median": statistics.median(after),
        "before_n": len(before),
        "after_n": len(after),
        "resamples": resamples,
        "seed": seed,
    }


def _binding(record: dict[str, Any]) -> dict[str, Any]:
    model = record["model_binding"]
    environment = record["environment"]
    protocol = record["protocol"]
    fingerprint = record["benchmark"]["arms"]["interactive"]["runtime_fingerprint"]
    system_before = record["system_before"]
    free_percent = system_before.get("memory_free_percent")
    free_class = ("gte_80" if free_percent is not None and free_percent >= 80
                  else "gte_60" if free_percent is not None and free_percent >= 60
                  else "lt_60" if free_percent is not None else "missing")
    return {
        "apple_chip": fingerprint["chip"],
        "machine": fingerprint["machine"],
        "memory_bytes": fingerprint["memory_bytes"],
        "gpu_cores": fingerprint["gpu_cores"],
        "hardware_fingerprint": fingerprint["hardware_fingerprint"],
        "runtime_version": fingerprint["runtime_version"],
        "model_id": model["model_id"],
        "revision": model["revision"],
        "model_manifest_sha256": model["model_manifest_sha256"],
        "architecture": model["architecture"],
        "quantisation": model["quantisation"],
        "python": environment["python"],
        "mlx": environment["mlx"],
        "mlx_lm": environment["mlx_lm"],
        "os": environment["os"],
        "power_source": environment["power_source"],
        "low_power_mode": environment["low_power_mode"],
        "thermal": environment["thermal"],
        "swap_preflight_class": (
            "zero" if system_before.get("swap_used_bytes") == 0
            else "nonzero" if system_before.get("swap_used_bytes") is not None else "missing"
        ),
        "memory_free_class": free_class,
        "requests": protocol["requests"],
        "max_tokens": protocol["max_tokens"],
        "warmup": protocol["warmup"],
        "repeats": protocol["repeats"],
        "plan": protocol["plan"],
        "knobs": protocol["knobs"],
        "offline_cached_snapshot_only": protocol["offline_cached_snapshot_only"],
    }


def _hard_failures(record: dict[str, Any]) -> list[str]:
    failures = []
    comparison = record.get("benchmark", {}).get("comparison", {})
    resources = record.get("resource_summary", {})
    if record.get("status") != "BASELINE_CAPTURED":
        failures.append("status")
    if comparison.get("token_identity") is not True:
        failures.append("token_identity")
    if resources.get("fallbacks") != 0:
        failures.append("fallbacks")
    if resources.get("correctness_errors") != 0:
        failures.append("correctness_errors")
    if resources.get("swap_delta_bytes") is None or resources["swap_delta_bytes"] > 256 * 1024**2:
        failures.append("swap_delta")
    return failures


def compare(
    before_records: Iterable[tuple[str, dict[str, Any]]],
    after_records: Iterable[tuple[str, dict[str, Any]]],
    *,
    regression_threshold: float = 0.05,
    resamples: int = 10_000,
    seed: int = 20260828,
) -> dict[str, Any]:
    if not 0 < regression_threshold < 1:
        raise ValueError("regression threshold must be between zero and one")
    before = {record["model_binding"]["model_id"]: (digest, record)
              for digest, record in before_records}
    after = {record["model_binding"]["model_id"]: (digest, record)
             for digest, record in after_records}
    errors = []
    if set(before) != set(after):
        errors.append(f"model sets differ: before={sorted(before)!r} after={sorted(after)!r}")

    cells = []
    any_drift = False
    any_correctness_failure = False
    any_performance_miss = False
    for index, model_id in enumerate(sorted(set(before) & set(after))):
        before_digest, old = before[model_id]
        after_digest, new = after[model_id]
        old_binding = _binding(old)
        new_binding = _binding(new)
        drift = sorted(key for key in old_binding if old_binding[key] != new_binding[key])
        hard_failures = _hard_failures(new)
        any_drift = any_drift or bool(drift)
        any_correctness_failure = any_correctness_failure or bool(hard_failures)
        comparisons = {}
        misses = []
        for arm in ARMS:
            comparisons[arm] = {}
            wall = bootstrap_ratio(
                _samples(new, arm, "outer_wall_ms"),
                _samples(old, arm, "outer_wall_ms"),
                resamples=resamples, seed=seed + index * 10 + (0 if arm == "interactive" else 1),
            )
            rate = bootstrap_ratio(
                _samples(new, arm, "physical_tokens_per_second"),
                _samples(old, arm, "physical_tokens_per_second"),
                resamples=resamples, seed=seed + index * 10 + (2 if arm == "interactive" else 3),
            )
            comparisons[arm]["outer_wall_post_over_pre"] = wall
            comparisons[arm]["physical_rate_post_over_pre"] = rate
            if wall["ci_high"] > 1.0 + regression_threshold:
                misses.append(f"{arm}.outer_wall")
            if rate["ci_low"] < 1.0 - regression_threshold:
                misses.append(f"{arm}.physical_rate")
        any_performance_miss = any_performance_miss or bool(misses)
        cells.append({
            "model_id": model_id,
            "model_revision": new["model_binding"]["revision"],
            "model_manifest_sha256": new["model_binding"]["model_manifest_sha256"],
            "before_raw_sha256": before_digest,
            "after_raw_sha256": after_digest,
            "before_commit": old["runtime_binding"]["git_head"],
            "after_commit": new["runtime_binding"]["git_head"],
            "before_runtime_tree_sha256": old["runtime_binding"]["runtime_tree_sha256"],
            "after_runtime_tree_sha256": new["runtime_binding"]["runtime_tree_sha256"],
            "domain_drift": drift,
            "hard_failures": hard_failures,
            "performance_misses": misses,
            "comparisons": comparisons,
            "after_resources": new["resource_summary"],
        })

    if errors:
        classification = "INCONCLUSIVE_INCOMPLETE"
        regression_kind = "INCONCLUSIVE"
    elif any_drift:
        classification = "REVALIDATION_REQUIRED"
        regression_kind = "EVIDENCE_DRIFT"
    elif any_correctness_failure:
        classification = "CODE_REGRESSION"
        regression_kind = "HARD_CORRECTNESS_OR_RESOURCE_REGRESSION"
    elif any_performance_miss:
        classification = "INCONCLUSIVE_POTENTIAL_REGRESSION"
        regression_kind = "POTENTIAL_CODE_REGRESSION"
    else:
        classification = "NO_REGRESSION_OBSERVED"
        regression_kind = "NONE"
    return {
        "schema": SCHEMA,
        "classification": classification,
        "regression_kind": regression_kind,
        "regression_threshold": regression_threshold,
        "valid_for_qualification": False,
        "activation_allowed": False,
        "errors": errors,
        "cells": cells,
        "limitations": [
            "Separate-session engineering comparison, not a qualification experiment.",
            "Bootstrap intervals compare independent pre/post repeat samples.",
            "No stock mlx_lm arm and no runtime activation.",
        ],
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-raw", type=Path, action="append", required=True)
    parser.add_argument("--after-raw", type=Path, action="append", required=True)
    parser.add_argument("--threshold-pct", type=float, default=5.0)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    before = [(_sha256(path), _load(path)) for path in args.before_raw]
    after = [(_sha256(path), _load(path)) for path in args.after_raw]
    result = compare(
        before, after, regression_threshold=args.threshold_pct / 100.0,
        resamples=args.resamples, seed=args.seed,
    )
    _atomic_json(args.output, result)
    print(json.dumps({
        "classification": result["classification"],
        "regression_kind": result["regression_kind"],
        "output": str(args.output),
    }, sort_keys=True))
    return 0 if result["classification"] == "NO_REGRESSION_OBSERVED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
