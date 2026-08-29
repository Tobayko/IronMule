"""Build deterministic path-free D2 pre/post engineering summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "ironmule.d2_baseline_summary.v1"


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


def summarize(
    records: Iterable[tuple[str, dict[str, Any]]], *, phase: str, experiment_id: str,
) -> dict[str, Any]:
    if phase not in {"pre", "post"}:
        raise ValueError("phase must be pre or post")
    cells = []
    errors = []
    seen = set()
    for raw_sha256, record in records:
        model = record.get("model_binding", {})
        model_id = model.get("model_id")
        if not isinstance(model_id, str) or not model_id or model_id in seen:
            errors.append(f"invalid or duplicate model cell: {model_id!r}")
            continue
        seen.add(model_id)
        comparison = record.get("benchmark", {}).get("comparison", {})
        resources = record.get("resource_summary", {})
        failures = []
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
        cells.append({
            "model": {
                "model_id": model_id,
                "revision": model.get("revision"),
                "model_manifest_sha256": model.get("model_manifest_sha256"),
                "architecture": model.get("architecture"),
                "quantisation": model.get("quantisation"),
                "runtime_identity": record.get("runtime_model_identity"),
            },
            "raw_sha256": raw_sha256,
            "status": record.get("status"),
            "hard_failures": failures,
            "commit": record.get("runtime_binding", {}).get("git_head"),
            "runtime_tree_sha256": record.get("runtime_binding", {}).get("runtime_tree_sha256"),
            "environment": {
                key: record.get("environment", {}).get(key)
                for key in ("python", "mlx", "mlx_lm", "os", "power_source", "low_power_mode")
            },
            "protocol": {
                key: record.get("protocol", {}).get(key)
                for key in ("requests", "max_tokens", "warmup", "repeats", "plan", "knobs",
                            "stored_profile_reuse", "offline_cached_snapshot_only")
            },
            "interactive": record.get("benchmark", {}).get("arms", {}).get("interactive", {}).get("summary"),
            "throughput": record.get("benchmark", {}).get("arms", {}).get("throughput", {}).get("summary"),
            "comparison": comparison,
            "resources": resources,
            "system_before": {
                key: record.get("system_before", {}).get(key)
                for key in ("memory_free_percent", "swap_used_bytes")
            },
        })
    cells.sort(key=lambda cell: cell["model"]["model_id"])
    if len(cells) != 2:
        errors.append(f"expected two model cells, got {len(cells)}")
    if any(cell["hard_failures"] for cell in cells):
        errors.append("one or more cells failed hard gates")
    result = {
        "schema": SCHEMA,
        "phase": phase,
        "experiment_id": experiment_id,
        "classification": "BASELINE_CAPTURED" if not errors else "INCONCLUSIVE",
        "valid_for_qualification": False,
        "activation_allowed": False,
        "errors": errors,
        "cells": cells,
    }
    forbidden = ("/users/", "file://", "snapshot_path", "source_root")
    leaks = [text for text in _walk_strings(result)
             if any(term in text.lower() for term in forbidden)]
    if leaks:
        raise ValueError("path-free D2 summary contains local path material")
    return result


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["pre", "post"], required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--raw", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = summarize(
        [(_sha256(path), _load(path)) for path in args.raw],
        phase=args.phase, experiment_id=args.experiment_id,
    )
    _atomic_json(args.output, result)
    print(json.dumps({
        "classification": result["classification"], "cells": len(result["cells"]),
        "output": str(args.output),
    }, sort_keys=True))
    return 0 if result["classification"] == "BASELINE_CAPTURED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
