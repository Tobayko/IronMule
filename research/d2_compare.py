"""Independently validate D2 identity wiring and compare sealed pre/post runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


SCHEMA = "ironmule.d2_post_comparison.v1"
IDENTITY_SCHEMA = "ironmule.model_identity.v1"
FINGERPRINT_SCHEMA = "ironmule.runtime_fingerprint.v2"
ARMS = ("interactive", "throughput")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOKENIZER_NAMES = {
    "added_tokens.json", "chat_template.json", "merges.txt", "special_tokens_map.json",
    "tokenizer.json", "tokenizer.model", "tokenizer_config.json", "vocab.json",
    "vocab.txt", "tiktoken.model",
}


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


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


def derive_identity(binding: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct ModelIdentity without importing the implementation under test."""
    required = {
        "model_id", "revision", "model_manifest_sha256", "manifest", "architecture",
        "quantisation",
    }
    if not required <= set(binding):
        raise ValueError("model binding is incomplete")
    for field in ("model_id", "revision", "architecture"):
        if not isinstance(binding[field], str) or not binding[field].strip():
            raise ValueError(f"model binding {field} is invalid")
    rows = binding["manifest"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("model manifest is empty")
    normalized = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
            raise ValueError("model manifest row is invalid")
        relative = row["path"]
        pure = PurePosixPath(relative) if isinstance(relative, str) else None
        if (pure is None or pure.is_absolute() or ".." in pure.parts
                or relative in seen or not relative):
            raise ValueError("model manifest path is invalid or duplicated")
        size = row["bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("model manifest byte count is invalid")
        if not isinstance(row["sha256"], str) or not _SHA256.fullmatch(row["sha256"]):
            raise ValueError("model manifest digest is invalid")
        seen.add(relative)
        normalized.append({"path": relative, "bytes": size, "sha256": row["sha256"]})
    normalized.sort(key=lambda row: row["path"])
    manifest_sha256 = canonical_sha256(normalized)
    if binding["model_manifest_sha256"] != manifest_sha256:
        raise ValueError("declared model manifest digest does not match manifest")
    tokenizer_rows = [
        row for row in normalized
        if PurePosixPath(row["path"]).name in _TOKENIZER_NAMES
        or PurePosixPath(row["path"]).suffix == ".tiktoken"
    ]
    if not tokenizer_rows:
        raise ValueError("model binding contains no tokenizer artifacts")
    quantisation = binding["quantisation"]
    if not isinstance(quantisation, dict):
        raise ValueError("model quantisation is invalid")
    for field in ("bits", "group_size"):
        value = quantisation.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"model quantisation {field} is invalid")
    quantisation_sha256 = canonical_sha256(quantisation)
    semantic = {
        "schema": IDENTITY_SCHEMA,
        "model_id": binding["model_id"],
        "revision": binding["revision"],
        "model_manifest_sha256": manifest_sha256,
        "architecture": binding["architecture"],
        "quantisation": quantisation,
        "quantisation_sha256": quantisation_sha256,
        "tokenizer_sha256": canonical_sha256(tokenizer_rows),
        "manifest_file_count": len(normalized),
        "manifest_bytes": sum(row["bytes"] for row in normalized),
        "tokenizer_file_count": len(tokenizer_rows),
    }
    return {**semantic, "identity_sha256": canonical_sha256(semantic)}


def _identity_failures(record: dict[str, Any]) -> tuple[list[str], dict[str, Any] | None]:
    failures = []
    try:
        expected = derive_identity(record["model_binding"])
    except (KeyError, TypeError, ValueError) as exc:
        return [f"model_binding_identity:{type(exc).__name__}"], None
    if record.get("runtime_model_identity") != expected:
        failures.append("runtime_model_identity")
    expected_fingerprint = {
        "fingerprint_schema": FINGERPRINT_SCHEMA,
        "model_id": expected["model_id"],
        "model_revision": expected["revision"],
        "model_manifest_sha256": expected["model_manifest_sha256"],
        "model_architecture": expected["architecture"],
        "quantisation": expected["quantisation"],
        "quantisation_sha256": expected["quantisation_sha256"],
        "tokenizer_sha256": expected["tokenizer_sha256"],
        "model_identity_sha256": expected["identity_sha256"],
    }
    arms = record.get("benchmark", {}).get("arms", {})
    for arm in ARMS:
        fingerprint = arms.get(arm, {}).get("runtime_fingerprint")
        if not isinstance(fingerprint, dict):
            failures.append(f"{arm}.runtime_fingerprint")
            continue
        mismatches = [
            field for field, value in expected_fingerprint.items()
            if fingerprint.get(field) != value
        ]
        if mismatches:
            failures.append(f"{arm}.runtime_fingerprint_identity")
    return failures, expected


def _samples(record: dict[str, Any], arm: str, metric: str) -> list[float]:
    rows = record["benchmark"]["arms"][arm]["raw"]
    values = [float(row["snapshot"][metric]) for row in rows if row.get("phase") == "measure"]
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError(f"invalid measured samples for {arm}.{metric}")
    return values


def bootstrap_ratio(
    after: list[float], before: list[float], *, resamples: int = 10_000,
    seed: int = 20260829,
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
    return {
        "median_ratio": statistics.median(after) / denominator,
        "ci_low": ratios[int(0.025 * (len(ratios) - 1))],
        "ci_high": ratios[int(0.975 * (len(ratios) - 1))],
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
        "apple_chip": fingerprint["chip"], "machine": fingerprint["machine"],
        "memory_bytes": fingerprint["memory_bytes"], "gpu_cores": fingerprint["gpu_cores"],
        "hardware_fingerprint": fingerprint["hardware_fingerprint"],
        "runtime_version": fingerprint["runtime_version"],
        "model_id": model["model_id"], "revision": model["revision"],
        "model_manifest_sha256": model["model_manifest_sha256"],
        "architecture": model["architecture"], "quantisation": model["quantisation"],
        "python": environment["python"], "mlx": environment["mlx"],
        "mlx_lm": environment["mlx_lm"], "os": environment["os"],
        "power_source": environment["power_source"],
        "low_power_mode": environment["low_power_mode"], "thermal": environment["thermal"],
        "swap_preflight_class": (
            "zero" if system_before.get("swap_used_bytes") == 0
            else "nonzero" if system_before.get("swap_used_bytes") is not None else "missing"
        ),
        "memory_free_class": free_class,
        "requests": protocol["requests"], "max_tokens": protocol["max_tokens"],
        "warmup": protocol["warmup"], "repeats": protocol["repeats"],
        "plan": protocol["plan"], "knobs": protocol["knobs"],
        "stored_profile_reuse": protocol.get("stored_profile_reuse"),
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
    identity_failures, _expected = _identity_failures(record)
    return failures + identity_failures


def compare(
    before_records: Iterable[tuple[str, dict[str, Any]]],
    after_records: Iterable[tuple[str, dict[str, Any]]], *,
    regression_threshold: float = 0.05, resamples: int = 10_000,
    seed: int = 20260829,
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
    any_drift = any_hard = any_miss = False
    for index, model_id in enumerate(sorted(set(before) & set(after))):
        before_digest, old = before[model_id]
        after_digest, new = after[model_id]
        old_binding, new_binding = _binding(old), _binding(new)
        drift = sorted(key for key in old_binding if old_binding[key] != new_binding[key])
        hard_failures = _hard_failures(new)
        _identity_errors, expected_identity = _identity_failures(new)
        comparisons: dict[str, Any] = {}
        misses = []
        for arm_index, arm in enumerate(ARMS):
            wall = bootstrap_ratio(
                _samples(new, arm, "outer_wall_ms"),
                _samples(old, arm, "outer_wall_ms"),
                resamples=resamples, seed=seed + index * 10 + arm_index,
            )
            rate = bootstrap_ratio(
                _samples(new, arm, "physical_tokens_per_second"),
                _samples(old, arm, "physical_tokens_per_second"),
                resamples=resamples, seed=seed + index * 10 + 2 + arm_index,
            )
            comparisons[arm] = {
                "outer_wall_post_over_pre": wall,
                "physical_rate_post_over_pre": rate,
            }
            if wall["ci_high"] > 1.0 + regression_threshold:
                misses.append(f"{arm}.outer_wall")
            if rate["ci_low"] < 1.0 - regression_threshold:
                misses.append(f"{arm}.physical_rate")
        any_drift = any_drift or bool(drift)
        any_hard = any_hard or bool(hard_failures)
        any_miss = any_miss or bool(misses)
        cells.append({
            "model_id": model_id,
            "model_revision": new["model_binding"]["revision"],
            "model_manifest_sha256": new["model_binding"]["model_manifest_sha256"],
            "expected_model_identity": expected_identity,
            "before_raw_sha256": before_digest, "after_raw_sha256": after_digest,
            "before_commit": old["runtime_binding"]["git_head"],
            "after_commit": new["runtime_binding"]["git_head"],
            "before_runtime_tree_sha256": old["runtime_binding"]["runtime_tree_sha256"],
            "after_runtime_tree_sha256": new["runtime_binding"]["runtime_tree_sha256"],
            "domain_drift": drift, "hard_failures": hard_failures,
            "performance_misses": misses, "comparisons": comparisons,
            "after_resources": new["resource_summary"],
        })
    if errors:
        classification, regression_kind = "INCONCLUSIVE_INCOMPLETE", "INCONCLUSIVE"
    elif any_drift:
        classification, regression_kind = "REVALIDATION_REQUIRED", "EVIDENCE_DRIFT"
    elif any_hard:
        classification = "CODE_REGRESSION"
        regression_kind = "HARD_IDENTITY_CORRECTNESS_OR_RESOURCE_REGRESSION"
    elif any_miss:
        classification = "INCONCLUSIVE_POTENTIAL_REGRESSION"
        regression_kind = "POTENTIAL_CODE_REGRESSION"
    else:
        classification, regression_kind = "NO_REGRESSION_OBSERVED", "NONE"
    return {
        "schema": SCHEMA, "classification": classification,
        "regression_kind": regression_kind,
        "regression_threshold": regression_threshold,
        "valid_for_qualification": False, "activation_allowed": False,
        "errors": errors, "cells": cells,
        "limitations": [
            "Separate-session engineering comparison, not a qualification experiment.",
            "Identity is reconstructed independently from each complete post manifest.",
            "Bootstrap intervals compare independent pre/post repeat samples.",
            "No stock mlx_lm arm, profile activation, routing or strategy selection.",
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
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = compare(
        [(_sha256(path), _load(path)) for path in args.before_raw],
        [(_sha256(path), _load(path)) for path in args.after_raw],
        regression_threshold=args.threshold_pct / 100.0,
        resamples=args.resamples, seed=args.seed,
    )
    _atomic_json(args.output, result)
    print(json.dumps({
        "classification": result["classification"],
        "regression_kind": result["regression_kind"], "output": str(args.output),
    }, sort_keys=True))
    return 0 if result["classification"] == "NO_REGRESSION_OBSERVED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
