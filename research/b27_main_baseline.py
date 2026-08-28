"""Engineering baseline capture for B27.

This harness deliberately reuses IronMule's public balanced benchmark protocol while
forcing ``Knobs()`` instead of loading a stored tuned profile.  It resolves only an
already-cached model snapshot, runs one model per fresh process, and writes raw evidence
atomically.  The result is an engineering baseline, not a qualification claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "ironmule.main_baseline.v1"
DEFAULT_EXPERIMENT_ID = "B27a"
MIB = 1024 * 1024


def _command(args: list[str], timeout: int = 30) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "returncode": None, "error": type(exc).__name__}
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def parse_swap_used_bytes(text: str) -> int | None:
    match = re.search(r"\bused\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGTP])", text, re.I)
    if not match:
        return None
    powers = {"K": 1, "M": 2, "G": 3, "T": 4, "P": 5}
    value = float(match.group(1)) * (1024 ** powers[match.group(2).upper()])
    return int(round(value)) if math.isfinite(value) else None


def parse_memory_free_percent(text: str) -> int | None:
    match = re.search(r"memory free percentage:\s*([0-9]+)%", text, re.I)
    return int(match.group(1)) if match else None


def system_state() -> dict[str, Any]:
    swap = _command(["sysctl", "vm.swapusage"])
    pressure = _command(["memory_pressure"])
    power = _command(["pmset", "-g", "batt"])
    return {
        "swap_used_bytes": parse_swap_used_bytes(swap.get("stdout", "")),
        "memory_free_percent": parse_memory_free_percent(pressure.get("stdout", "")),
        "power_source": power.get("stdout") or None,
        "probes": {
            "swap_ok": swap["ok"],
            "memory_pressure_ok": pressure["ok"],
            "power_ok": power["ok"],
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_digest(root: Path, paths: Iterable[Path]) -> tuple[str, list[dict[str, Any]]]:
    rows = []
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        if not path.is_file():
            continue
        rows.append({
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest(), rows


def runtime_binding(root: Path) -> dict[str, Any]:
    files = [path for path in (root / "ironmule").rglob("*.py")]
    files.extend([root / "ironmule_cli.py", root / "pyproject.toml"])
    digest, rows = tree_digest(root, files)
    head = _command(["git", "-C", str(root), "rev-parse", "HEAD"])
    return {
        "git_head": head.get("stdout") or None,
        "runtime_tree_sha256": digest,
        "files": rows,
    }


def select_cached_snapshot(cache: Any, model_id: str, revision: str) -> Path:
    """Resolve one exact cached revision without invoking snapshot download logic."""
    repositories = [repo for repo in cache.repos if repo.repo_id == model_id]
    matches = [
        cached_revision
        for repo in repositories
        for cached_revision in repo.revisions
        if cached_revision.commit_hash == revision
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one cached snapshot for {model_id}@{revision}, found {len(matches)}"
        )
    snapshot = Path(matches[0].snapshot_path).resolve()
    if not snapshot.is_dir():
        raise RuntimeError(f"cached snapshot directory is unavailable for {model_id}@{revision}")
    return snapshot


def model_binding(model_id: str, revision: str) -> tuple[Path, dict[str, Any]]:
    from huggingface_hub import scan_cache_dir

    snapshot = select_cached_snapshot(scan_cache_dir(), model_id, revision)
    digest, rows = tree_digest(snapshot, [path for path in snapshot.rglob("*") if path.is_file()])
    config_path = snapshot / "config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    quantisation = config.get("quantization") or config.get("quantization_config")
    return snapshot, {
        "model_id": model_id,
        "revision": revision,
        "model_manifest_sha256": digest,
        "manifest": rows,
        "architecture": config.get("model_type") or config.get("architectures"),
        "quantisation": quantisation,
    }


def _fallback_counts(result: dict[str, Any]) -> dict[str, int]:
    snapshots = [
        row.get("snapshot", {})
        for arm in result.get("arms", {}).values()
        for row in arm.get("raw", [])
        if row.get("phase") == "measure"
    ]
    return {
        "fallbacks": sum(int(row.get("fallbacks", 0) or 0) for row in snapshots),
        "correctness_errors": sum(int(row.get("correctness_errors", 0) or 0) for row in snapshots),
    }


def classify(result: dict[str, Any], before: dict[str, Any], after: dict[str, Any],
             *, swap_delta_ceiling_bytes: int) -> tuple[str, list[str]]:
    failures = []
    comparison = result.get("comparison", {})
    if not comparison.get("token_identity"):
        failures.append("token_identity")
    counts = _fallback_counts(result)
    if counts["fallbacks"]:
        failures.append("fallbacks")
    if counts["correctness_errors"]:
        failures.append("correctness_errors")
    before_swap = before.get("swap_used_bytes")
    after_swap = after.get("swap_used_bytes")
    if before_swap is None or after_swap is None:
        failures.append("swap_unavailable")
    elif after_swap - before_swap > swap_delta_ceiling_bytes:
        failures.append("swap_delta")
    return ("INCONCLUSIVE", failures) if failures else ("BASELINE_CAPTURED", [])


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = (args.runtime_root.resolve() if getattr(args, "runtime_root", None)
            else Path(__file__).resolve().parents[1])
    started = time.monotonic()
    before = system_state()
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment_id": args.experiment_id,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "status": "INCONCLUSIVE",
        "valid_for_performance": False,
        "activation_allowed": False,
        "stage": "preflight",
        "protocol": {
            "model": args.model,
            "revision": args.revision,
            "plan": "strict",
            "requests": args.requests,
            "max_tokens": args.max_tokens,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "knobs": "BASELINE",
            "stored_profile_reuse": False,
            "fresh_process_per_model": True,
            "shared_loaded_model_between_arms": True,
            "offline_cached_snapshot_only": True,
            "swap_preflight_ceiling_bytes": args.swap_preflight_ceiling_mib * MIB,
            "swap_delta_ceiling_bytes": args.swap_delta_ceiling_mib * MIB,
        },
        "system_before": before,
        "runtime_binding": runtime_binding(root),
        "limitations": [
            "Engineering baseline only; no fresh process per arm.",
            "No stock mlx_lm arm.",
            "No qualification, promotion, activation, or cross-domain claim.",
        ],
    }
    _atomic_json(args.output, result)

    ceiling = args.swap_preflight_ceiling_mib * MIB
    if before.get("swap_used_bytes") is None or before["swap_used_bytes"] > ceiling:
        result["failures"] = ["swap_preflight"]
        result["elapsed_seconds"] = time.monotonic() - started
        _atomic_json(args.output, result)
        return result

    try:
        result["stage"] = "model_binding"
        snapshot, binding = model_binding(args.model, args.revision)
        result["model_binding"] = binding
        _atomic_json(args.output, result)

        import mlx.core as mx
        import ironmule
        from ironmule.bench import environment
        from ironmule.benchmark import run_protocol
        from ironmule.runtime import BASELINE
        from ironmule.service import Runtime
        from ironmule.tune import _eos_ids, gpu_busy, load_engine

        busy = gpu_busy()
        if busy:
            result["failures"] = ["model_process_busy"]
            result["busy_process"] = busy
            return result

        result["stage"] = "model_load"
        engine, tokenizer = load_engine(str(snapshot), BASELINE, offline=True)
        rt = Runtime(
            engine,
            tokenizer,
            model_id=args.model,
            quantisation=binding.get("quantisation"),
        )
        result["environment"] = environment()
        result["stage"] = "benchmark"
        _atomic_json(args.output, result)
        benchmark = run_protocol(
            rt,
            ironmule,
            requests=args.requests,
            max_tokens=args.max_tokens,
            plan_name="strict",
            warmup=args.warmup,
            repeats=args.repeats,
        )
        after = system_state()
        status, failures = classify(
            benchmark,
            before,
            after,
            swap_delta_ceiling_bytes=args.swap_delta_ceiling_mib * MIB,
        )
        result.update({
            "stage": "complete",
            "status": status,
            "failures": failures,
            "valid_for_performance": status == "BASELINE_CAPTURED",
            "benchmark": benchmark,
            "system_after": after,
            "resource_summary": {
                "mlx_peak_memory_bytes": int(mx.get_peak_memory()),
                "swap_delta_bytes": (
                    after["swap_used_bytes"] - before["swap_used_bytes"]
                    if after.get("swap_used_bytes") is not None
                    and before.get("swap_used_bytes") is not None
                    else None
                ),
                **_fallback_counts(benchmark),
            },
        })
        return result
    except BaseException as exc:  # raw evidence must survive any terminal failure
        result.update({
            "status": "INCONCLUSIVE",
            "valid_for_performance": False,
            "failures": ["exception"],
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "system_after": system_state(),
        })
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        _atomic_json(args.output, result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--swap-preflight-ceiling-mib", type=int, default=256)
    parser.add_argument("--swap-delta-ceiling-mib", type=int, default=256)
    args = parser.parse_args(argv)
    if min(args.requests, args.max_tokens, args.warmup, args.repeats) < 1:
        parser.error("requests, max-tokens, warmup and repeats must be positive")
    if not args.experiment_id.strip():
        parser.error("experiment-id must be non-empty")
    try:
        result = run(args)
    except BaseException:
        return 3
    print(json.dumps({
        "output": str(args.output),
        "status": result["status"],
        "model": args.model,
        "failures": result.get("failures", []),
        "elapsed_seconds": result.get("elapsed_seconds"),
    }, sort_keys=True))
    return 0 if result["status"] == "BASELINE_CAPTURED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
