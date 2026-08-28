"""Fail-closed verifier for the path-free B27 public baseline summary.

The verifier reads local raw records, but emits only hashes and bounded counts.  It
does not execute MLX, alter evidence, or upgrade an engineering baseline into a
qualification result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "ironmule.public_summary_verification.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _get(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(path)
        current = current[part]
    return current


def _equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    return left == right


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _walk_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_strings(child)


def _check(errors: list[str], label: str, actual: Any, expected: Any) -> None:
    if not _equal(actual, expected):
        errors.append(f"{label}: public={actual!r} raw={expected!r}")


def _check_path(errors: list[str], label: str, public: dict[str, Any], public_path: str,
                raw: dict[str, Any], raw_path: str) -> None:
    try:
        actual = _get(public, public_path)
    except KeyError:
        errors.append(f"{label}: missing public field {public_path}")
        return
    try:
        expected = _get(raw, raw_path)
    except KeyError:
        errors.append(f"{label}: missing raw field {raw_path}")
        return
    _check(errors, label, actual, expected)


def verify(summary: dict[str, Any], raw_records: list[tuple[str, dict[str, Any]]],
           failure_records: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    errors: list[str] = []
    if summary.get("schema") != "ironmule.main_baseline.public.v1":
        errors.append("unsupported public summary schema")
    if summary.get("status") != "BASELINE_CAPTURED":
        errors.append("public top status is not BASELINE_CAPTURED")
    if summary.get("activation_allowed") is not False:
        errors.append("public activation_allowed must be false")
    if summary.get("valid_for_qualification") is not False:
        errors.append("public valid_for_qualification must be false")

    forbidden = ("/users/", "file://", "snapshot_path", "source_root")
    leaked = sorted({text for text in _walk_strings(summary)
                     if any(term in text.lower() for term in forbidden)})
    if leaked:
        errors.append(f"public summary contains local-path material: {leaked!r}")

    raw_by_model: dict[str, tuple[str, dict[str, Any]]] = {}
    for digest, raw in raw_records:
        try:
            model_id = str(_get(raw, "model_binding.model_id"))
        except KeyError:
            errors.append(f"raw {digest} has no model binding")
            continue
        if model_id in raw_by_model:
            errors.append(f"duplicate raw model cell: {model_id}")
        raw_by_model[model_id] = (digest, raw)

    cells = summary.get("cells")
    if not isinstance(cells, list) or not cells:
        errors.append("public summary has no cells")
        cells = []
    public_models = [str(cell.get("model", {}).get("id")) for cell in cells]
    if len(public_models) != len(set(public_models)):
        errors.append("duplicate public model cell")
    if set(public_models) != set(raw_by_model):
        errors.append(
            f"public/raw model set differs: public={sorted(public_models)!r} "
            f"raw={sorted(raw_by_model)!r}"
        )

    mappings = [
        ("model revision", "model.revision", "model_binding.revision"),
        ("model manifest", "model.manifest_sha256", "model_binding.model_manifest_sha256"),
        ("model architecture", "model.architecture", "model_binding.architecture"),
        ("quantisation bits", "model.bits", "model_binding.quantisation.bits"),
        ("quantisation group", "model.group_size", "model_binding.quantisation.group_size"),
        ("environment Python", "environment.python", "environment.python"),
        ("environment MLX", "environment.mlx", "environment.mlx"),
        ("environment mlx_lm", "environment.mlx_lm", "environment.mlx_lm"),
        ("environment OS", "environment.os", "environment.os"),
        ("environment power", "environment.power", "environment.power_source"),
        ("interactive wall median", "interactive.outer_wall_ms_median", "benchmark.arms.interactive.summary.outer_wall_ms.median"),
        ("interactive rate median", "interactive.physical_tokens_per_second_median", "benchmark.arms.interactive.summary.physical_tokens_per_second.median"),
        ("throughput wall median", "throughput.outer_wall_ms_median", "benchmark.arms.throughput.summary.outer_wall_ms.median"),
        ("throughput rate median", "throughput.physical_tokens_per_second_median", "benchmark.arms.throughput.summary.physical_tokens_per_second.median"),
        ("wall ratio median", "comparison.wall_ratio_throughput_over_interactive.median", "benchmark.comparison.primary_wall_ratio_throughput_over_interactive.median_ratio"),
        ("wall ratio CI low", "comparison.wall_ratio_throughput_over_interactive.ci_low", "benchmark.comparison.primary_wall_ratio_throughput_over_interactive.ci_low"),
        ("wall ratio CI high", "comparison.wall_ratio_throughput_over_interactive.ci_high", "benchmark.comparison.primary_wall_ratio_throughput_over_interactive.ci_high"),
        ("rate ratio median", "comparison.physical_rate_ratio_throughput_over_interactive.median", "benchmark.comparison.primary_rate_ratio_throughput_over_interactive.median_ratio"),
        ("rate ratio CI low", "comparison.physical_rate_ratio_throughput_over_interactive.ci_low", "benchmark.comparison.primary_rate_ratio_throughput_over_interactive.ci_low"),
        ("rate ratio CI high", "comparison.physical_rate_ratio_throughput_over_interactive.ci_high", "benchmark.comparison.primary_rate_ratio_throughput_over_interactive.ci_high"),
        ("token identity", "comparison.token_identity", "benchmark.comparison.token_identity"),
        ("MLX peak", "resources.mlx_peak_memory_bytes", "resource_summary.mlx_peak_memory_bytes"),
        ("swap delta", "resources.swap_delta_bytes", "resource_summary.swap_delta_bytes"),
        ("fallbacks", "resources.fallbacks", "resource_summary.fallbacks"),
        ("correctness errors", "resources.correctness_errors", "resource_summary.correctness_errors"),
    ]
    for cell in cells:
        model_id = str(cell.get("model", {}).get("id"))
        matched = raw_by_model.get(model_id)
        if matched is None:
            continue
        raw_digest, raw = matched
        if raw.get("schema") != "ironmule.main_baseline.v1":
            errors.append(f"{model_id}: unsupported raw schema")
        if raw.get("status") != "BASELINE_CAPTURED":
            errors.append(f"{model_id}: raw status is not BASELINE_CAPTURED")
        if raw.get("activation_allowed") is not False:
            errors.append(f"{model_id}: raw activation_allowed must be false")
        if raw.get("valid_for_performance") is not True:
            errors.append(f"{model_id}: raw valid_for_performance must be true")
        _check(errors, f"{model_id}: raw SHA-256", cell.get("raw_sha256"), raw_digest)
        _check(errors, f"{model_id}: model ID", cell.get("model", {}).get("id"),
               raw.get("model_binding", {}).get("model_id"))
        _check(errors, f"{model_id}: runtime tree", summary.get("runtime_tree_sha256"),
               raw.get("runtime_binding", {}).get("runtime_tree_sha256"))
        _check(errors, f"{model_id}: base commit", summary.get("base_commit"),
               raw.get("runtime_binding", {}).get("git_head"))
        _check(errors, f"{model_id}: chip", cell.get("environment", {}).get("apple_chip"),
               _get(raw, "benchmark.arms.interactive.runtime_fingerprint.chip"))
        for label, public_path, raw_path in mappings:
            _check_path(errors, f"{model_id}: {label}", cell, public_path, raw, raw_path)
        for field in ("requests", "max_tokens", "warmup", "repeats"):
            _check(errors, f"{model_id}: protocol {field}", summary.get("protocol", {}).get(field),
                   raw.get("protocol", {}).get(field))
        _check(errors, f"{model_id}: protocol knobs", summary.get("protocol", {}).get("knobs"),
               raw.get("protocol", {}).get("knobs"))
        _check(errors, f"{model_id}: cached-only", summary.get("protocol", {}).get("offline_cached_snapshots_only"),
               raw.get("protocol", {}).get("offline_cached_snapshot_only"))
        _check(errors, f"{model_id}: fresh process per model", summary.get("protocol", {}).get("fresh_process_per_model"),
               raw.get("protocol", {}).get("fresh_process_per_model"))

    failure_by_hash = {digest: raw for digest, raw in failure_records}
    public_failures = summary.get("premeasurement_failures", [])
    if not isinstance(public_failures, list):
        errors.append("premeasurement_failures is not a list")
        public_failures = []
    if {row.get("raw_sha256") for row in public_failures} != set(failure_by_hash):
        errors.append("public/raw premeasurement failure hash set differs")
    for row in public_failures:
        digest = row.get("raw_sha256")
        raw = failure_by_hash.get(digest)
        if raw is None:
            continue
        _check(errors, f"failure {digest}: stage", row.get("stage"), raw.get("stage"))
        _check(errors, f"failure {digest}: type", row.get("type"),
               raw.get("error", {}).get("type"))
        if raw.get("status") != "INCONCLUSIVE":
            errors.append(f"failure {digest}: raw status must be INCONCLUSIVE")
        if raw.get("benchmark") is not None:
            errors.append(f"failure {digest}: benchmark evidence must be absent")

    return {
        "schema": SCHEMA,
        "ok": not errors,
        "errors": errors,
        "checked_cells": len(cells),
        "checked_failures": len(public_failures),
        "activation_allowed": False,
        "qualification_changed": False,
        "raw_hashes": sorted(digest for digest, _ in raw_records),
        "failure_hashes": sorted(digest for digest, _ in failure_records),
    }


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path.name}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--raw", type=Path, action="append", required=True)
    parser.add_argument("--failure-raw", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = _load(args.summary)
    raw_records = [(_sha256(path), _load(path)) for path in args.raw]
    failure_records = [(_sha256(path), _load(path)) for path in args.failure_raw]
    result = verify(summary, raw_records, failure_records)
    result["summary_sha256"] = _sha256(args.summary)
    _atomic_json(args.output, result)
    print(json.dumps({"ok": result["ok"], "errors": len(result["errors"]),
                      "output": str(args.output)}, sort_keys=True))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
