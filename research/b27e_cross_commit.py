"""Mirrored fresh-process cross-commit control for B27e.

The parent never imports MLX.  Each child uses this checkout's frozen baseline harness
but imports the actual IronMule package from one exact detached target worktree.  The
four child order is fixed to OLD/D1 then D1/OLD and all hardware work is serial.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "ironmule.b27e_cross_commit.v1"
PUBLIC_SCHEMA = "ironmule.b27e_cross_commit.public.v1"
MIB = 1024 * 1024
EXECUTION_SURFACE = (
    "ironmule/__init__.py",
    "ironmule/_version.py",
    "ironmule/ab.py",
    "ironmule/bench.py",
    "ironmule/benchmark.py",
    "ironmule/executor.py",
    "ironmule/fast.py",
    "ironmule/fingerprint.py",
    "ironmule/hw.py",
    "ironmule/plans.py",
    "ironmule/runtime.py",
    "ironmule/service.py",
    "ironmule/telemetry.py",
    "ironmule/tune.py",
    "ironmule_cli.py",
    "pyproject.toml",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def execution_surface_digest(root: Path, files: Iterable[str] = EXECUTION_SURFACE) -> str:
    rows = []
    for relative in files:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"execution-surface file is missing: {relative}")
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": _sha256(path)})
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True,
        timeout=30, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed for target")
    return result.stdout.strip()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path.name}")
    return value


def _command(args: list[str], timeout: int = 30) -> dict[str, Any]:
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "returncode": None, "error": type(exc).__name__}
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _swap_bytes(text: str) -> int | None:
    match = re.search(r"\bused\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGTP])", text, re.I)
    if not match:
        return None
    powers = {"K": 1, "M": 2, "G": 3, "T": 4, "P": 5}
    return int(round(float(match.group(1)) * 1024 ** powers[match.group(2).upper()]))


def _memory_percent(text: str) -> int | None:
    match = re.search(r"memory free percentage:\s*([0-9]+)%", text, re.I)
    return int(match.group(1)) if match else None


def _artifact_date(value: str) -> str:
    if not re.fullmatch(r"[0-9]{8}", value):
        raise argparse.ArgumentTypeError("artifact-date must be YYYYMMDD")
    return value


def preflight(*, memory_min_percent: int, swap_ceiling_bytes: int) -> dict[str, Any]:
    swap = _command(["sysctl", "vm.swapusage"])
    memory = _command(["memory_pressure"])
    power = _command(["pmset", "-g", "batt"])
    processes = _command(["ps", "-Ao", "pid=,rss=,args="])
    swap_used = _swap_bytes(swap.get("stdout", ""))
    free = _memory_percent(memory.get("stdout", ""))
    busy = []
    if processes["ok"]:
        for line in processes["stdout"].splitlines():
            parts = line.strip().split(None, 2)
            if len(parts) != 3:
                continue
            pid, rss, command = parts
            lowered = command.lower()
            if (int(pid) != os.getpid() and int(rss) >= 1_000_000
                    and any(term in lowered for term in ("gemma", "qwen", "mlx_lm"))):
                busy.append({"pid": int(pid), "rss_kib": int(rss)})
    failures = []
    if swap_used is None or swap_used > swap_ceiling_bytes:
        failures.append("swap")
    if free is None or free < memory_min_percent:
        failures.append("memory_free")
    if not power["ok"] or "AC Power" not in power.get("stdout", ""):
        failures.append("power")
    if busy:
        failures.append("model_process")
    return {
        "ok": not failures,
        "failures": failures,
        "swap_used_bytes": swap_used,
        "memory_free_percent": free,
        "power_source": "AC" if power["ok"] and "AC Power" in power.get("stdout", "") else "unknown",
        "busy_processes": busy,
    }


def _binding(record: dict[str, Any]) -> dict[str, Any]:
    model = record["model_binding"]
    environment = record["environment"]
    protocol = record["protocol"]
    fingerprint = record["benchmark"]["arms"]["interactive"]["runtime_fingerprint"]
    return {
        "chip": fingerprint["chip"], "machine": fingerprint["machine"],
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
        "requests": protocol["requests"], "max_tokens": protocol["max_tokens"],
        "warmup": protocol["warmup"], "repeats": protocol["repeats"],
        "plan": protocol["plan"], "knobs": protocol["knobs"],
        "offline_cached_snapshot_only": protocol["offline_cached_snapshot_only"],
    }


def _hard_failures(record: dict[str, Any]) -> list[str]:
    result = []
    comparison = record.get("benchmark", {}).get("comparison", {})
    resources = record.get("resource_summary", {})
    if record.get("status") != "BASELINE_CAPTURED":
        result.append("status")
    if comparison.get("token_identity") is not True:
        result.append("token_identity")
    if resources.get("fallbacks") != 0:
        result.append("fallbacks")
    if resources.get("correctness_errors") != 0:
        result.append("correctness_errors")
    if resources.get("swap_delta_bytes") is None or resources["swap_delta_bytes"] > 256 * MIB:
        result.append("swap_delta")
    return result


def _median(record: dict[str, Any], arm: str, metric: str) -> float:
    values = [
        float(row["snapshot"][metric])
        for row in record["benchmark"]["arms"][arm]["raw"]
        if row.get("phase") == "measure"
    ]
    if len(values) < 2 or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError(f"invalid samples for {arm}.{metric}")
    return statistics.median(values)


def analyze(
    entries: list[dict[str, Any]], *, old_commit: str, d1_commit: str,
    execution_surface_sha256: str, threshold: float = 0.05,
) -> dict[str, Any]:
    errors = []
    if len(entries) != 4:
        errors.append(f"expected four children, got {len(entries)}")
    blocks = []
    domain_drift = []
    hard_failures = []
    for block_index in (0, 1):
        block_entries = [entry for entry in entries if entry.get("block") == block_index]
        by_label = {entry.get("label"): entry for entry in block_entries}
        if set(by_label) != {"old", "d1"}:
            errors.append(f"block {block_index} labels are incomplete")
            continue
        old = by_label["old"]["record"]
        d1 = by_label["d1"]["record"]
        if old.get("b27e_target", {}).get("commit") != old_commit:
            errors.append(f"block {block_index} old commit mismatch")
        if d1.get("b27e_target", {}).get("commit") != d1_commit:
            errors.append(f"block {block_index} D1 commit mismatch")
        old_binding = _binding(old)
        d1_binding = _binding(d1)
        drift = sorted(key for key in old_binding if old_binding[key] != d1_binding[key])
        if drift:
            domain_drift.append({"block": block_index, "fields": drift})
        for label, record in (("old", old), ("d1", d1)):
            failures = _hard_failures(record)
            if failures:
                hard_failures.append({"block": block_index, "label": label, "failures": failures})
        ratios = {}
        for arm in ("interactive", "throughput"):
            old_wall = _median(old, arm, "outer_wall_ms")
            d1_wall = _median(d1, arm, "outer_wall_ms")
            old_rate = _median(old, arm, "physical_tokens_per_second")
            d1_rate = _median(d1, arm, "physical_tokens_per_second")
            ratios[arm] = {
                "d1_over_old_wall": d1_wall / old_wall,
                "d1_over_old_rate": d1_rate / old_rate,
                "old_wall_median": old_wall, "d1_wall_median": d1_wall,
                "old_rate_median": old_rate, "d1_rate_median": d1_rate,
            }
        blocks.append({
            "block": block_index,
            "order": [entry["label"] for entry in sorted(block_entries, key=lambda row: row["position"])],
            "ratios": ratios,
            "child_sha256": {label: by_label[label]["sha256"] for label in ("old", "d1")},
        })

    all_ratios = [value for block in blocks for value in block["ratios"].values()]
    within = all(
        1 - threshold <= row["d1_over_old_wall"] <= 1 + threshold
        and 1 - threshold <= row["d1_over_old_rate"] <= 1 + threshold
        for row in all_ratios
    ) if all_ratios else False
    slower = all(
        row["d1_over_old_wall"] > 1 + threshold
        and row["d1_over_old_rate"] < 1 - threshold
        for row in all_ratios
    ) if all_ratios else False
    faster = all(
        row["d1_over_old_wall"] < 1 - threshold
        and row["d1_over_old_rate"] > 1 + threshold
        for row in all_ratios
    ) if all_ratios else False

    if errors:
        classification = "INCONCLUSIVE_INCOMPLETE"
        consequence = "B27D_REMAINS_INCONCLUSIVE"
    elif domain_drift:
        classification = "REVALIDATION_REQUIRED"
        consequence = "EVIDENCE_DRIFT"
    elif hard_failures:
        classification = "CODE_REGRESSION"
        consequence = "HARD_CORRECTNESS_OR_RESOURCE_REGRESSION"
    elif within:
        classification = "COMMITS_INDISTINGUISHABLE"
        consequence = "COMMON_MODE_TEMPORAL_DRIFT_SUPPORTED"
    elif slower:
        classification = "D1_SLOWER_REPRODUCED"
        consequence = "D1_CAUSAL_ASSOCIATION_REPRODUCED"
    elif faster:
        classification = "D1_FASTER_REPRODUCED"
        consequence = "D1_CAUSAL_ASSOCIATION_REPRODUCED"
    else:
        classification = "ORDER_OR_TEMPORAL_DRIFT"
        consequence = "B27D_REMAINS_INCONCLUSIVE"
    return {
        "schema": PUBLIC_SCHEMA,
        "classification": classification,
        "b27d_consequence": consequence,
        "old_commit": old_commit,
        "d1_commit": d1_commit,
        "execution_surface_sha256": execution_surface_sha256,
        "threshold": threshold,
        "orders": [["old", "d1"], ["d1", "old"]],
        "blocks": blocks,
        "domain_drift": domain_drift,
        "hard_failures": hard_failures,
        "errors": errors,
        "valid_for_qualification": False,
        "activation_allowed": False,
        "limitations": [
            "Two mirrored blocks are a mechanism control, not qualification evidence.",
            "Only new B27e samples are used; B27a2/B27d values are not pooled.",
            "No stock mlx_lm arm and no runtime activation.",
        ],
    }


def run_child(args: argparse.Namespace) -> int:
    target = args.target_root.resolve()
    commit = _git(target, "rev-parse", "HEAD")
    if commit != args.expected_commit:
        raise RuntimeError("target commit does not match the sealed B27e selector")
    surface = execution_surface_digest(target)
    sys.path.insert(0, str(target))
    harness_path = Path(__file__).with_name("b27_main_baseline.py")
    spec = importlib.util.spec_from_file_location("b27e_frozen_harness", harness_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen B27 baseline harness")
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    run_args = argparse.Namespace(
        model=args.model, revision=args.revision, experiment_id="B27e",
        output=args.output, requests=6, max_tokens=48, warmup=2, repeats=6,
        swap_preflight_ceiling_mib=256, swap_delta_ceiling_mib=256,
        runtime_root=target,
    )
    result = harness.run(run_args)
    import ironmule  # imported only after target root is first on sys.path
    package_root = Path(ironmule.__file__).resolve().parents[1]
    if package_root != target:
        raise RuntimeError("child imported IronMule from the wrong target root")
    result["b27e_target"] = {
        "label": args.label, "commit": commit,
        "execution_surface_sha256": surface,
    }
    harness._atomic_json(args.output, result)
    print(json.dumps({"status": result["status"], "label": args.label}, sort_keys=True))
    return 0 if result["status"] == "BASELINE_CAPTURED" else 2


def run_parent(args: argparse.Namespace) -> int:
    old_root = args.old_root.resolve()
    d1_root = args.d1_root.resolve()
    targets = {"old": old_root, "d1": d1_root}
    commits = {"old": _git(old_root, "rev-parse", "HEAD"),
               "d1": _git(d1_root, "rev-parse", "HEAD")}
    if commits != {"old": args.old_commit, "d1": args.d1_commit}:
        raise RuntimeError("target worktree commits do not match B27e preregistration")
    surfaces = {label: execution_surface_digest(root) for label, root in targets.items()}
    if len(set(surfaces.values())) != 1:
        raise RuntimeError("old and D1 execution surfaces are not byte-identical")
    if (old_root / "ironmule" / "evidence.py").exists():
        raise RuntimeError("old target unexpectedly contains D1")
    d1_evidence = d1_root / "ironmule" / "evidence.py"
    if not d1_evidence.is_file() or _sha256(d1_evidence) != args.d1_evidence_sha256:
        raise RuntimeError("D1 evidence module does not match the sealed hash")

    orders = (("old", "d1"), ("d1", "old"))
    parent = {
        "schema": SCHEMA,
        "experiment_id": "B27e",
        "orders": [list(order) for order in orders],
        "target_commits": commits,
        "execution_surface_sha256": surfaces["old"],
        "children": [],
        "status": "RUNNING",
        "activation_allowed": False,
    }
    _atomic_json(args.raw_output, parent)
    entries = []
    stop = False
    for block, order in enumerate(orders):
        for position, label in enumerate(order):
            gate = preflight(
                memory_min_percent=args.memory_min_percent,
                swap_ceiling_bytes=args.swap_ceiling_mib * MIB,
            )
            child_name = f"B27e_block{block}_pos{position}_{label}_{args.artifact_date}.json"
            output = args.output_dir / child_name
            row = {"block": block, "position": position, "label": label, "preflight": gate,
                   "artifact_id": child_name}
            parent["children"].append(row)
            _atomic_json(args.raw_output, parent)
            if not gate["ok"]:
                row["returncode"] = None
                row["failure"] = "preflight"
                stop = True
                break
            command = [
                sys.executable, str(Path(__file__).resolve()), "child",
                "--target-root", str(targets[label]), "--expected-commit", commits[label],
                "--label", label, "--model", args.model, "--revision", args.revision,
                "--output", str(output),
            ]
            try:
                process = subprocess.run(
                    command, capture_output=True, text=True,
                    timeout=args.child_timeout_seconds, check=False,
                )
            except subprocess.TimeoutExpired as exc:
                row.update({
                    "returncode": None, "failure": "timeout",
                    "stdout": (exc.stdout or "") if isinstance(exc.stdout, str) else "",
                    "stderr": (exc.stderr or "") if isinstance(exc.stderr, str) else "",
                })
                _atomic_json(args.raw_output, parent)
                stop = True
                break
            row.update({
                "returncode": process.returncode,
                "stdout": process.stdout.strip(),
                "stderr": process.stderr.strip(),
            })
            if output.is_file():
                record = _load(output)
                digest = _sha256(output)
                row["sha256"] = digest
                entries.append({
                    "block": block, "position": position, "label": label,
                    "sha256": digest, "record": record,
                })
            _atomic_json(args.raw_output, parent)
            if process.returncode != 0:
                stop = True
                break
        if stop:
            break
    public = analyze(
        entries, old_commit=args.old_commit, d1_commit=args.d1_commit,
        execution_surface_sha256=surfaces["old"], threshold=args.threshold_pct / 100.0,
    )
    parent["status"] = "COMPLETE" if len(entries) == 4 else "INCONCLUSIVE"
    parent["public_result"] = public
    _atomic_json(args.raw_output, parent)
    _atomic_json(args.public_output, public)
    print(json.dumps({
        "classification": public["classification"],
        "b27d_consequence": public["b27d_consequence"],
        "children": len(entries),
    }, sort_keys=True))
    return 0 if len(entries) == 4 and not public["errors"] else 2


def entries_from_parent(parent_path: Path) -> list[dict[str, Any]]:
    parent = _load(parent_path)
    entries = []
    for row in parent.get("children", []):
        if not row.get("sha256"):
            continue
        artifact = parent_path.parent / row["artifact_id"]
        if not artifact.is_file() or _sha256(artifact) != row["sha256"]:
            raise RuntimeError(f"child artifact hash mismatch: {row['artifact_id']}")
        entries.append({
            "block": row["block"], "position": row["position"], "label": row["label"],
            "sha256": row["sha256"], "record": _load(artifact),
        })
    return entries


def run_reanalyze(args: argparse.Namespace) -> int:
    result = analyze(
        entries_from_parent(args.parent_raw),
        old_commit=args.old_commit,
        d1_commit=args.d1_commit,
        execution_surface_sha256=args.execution_surface_sha256,
        threshold=args.threshold_pct / 100.0,
    )
    _atomic_json(args.output, result)
    print(json.dumps({
        "classification": result["classification"],
        "b27d_consequence": result["b27d_consequence"],
    }, sort_keys=True))
    return 0 if not result["errors"] else 2


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    commands = result.add_subparsers(dest="command", required=True)
    child = commands.add_parser("child")
    child.add_argument("--target-root", type=Path, required=True)
    child.add_argument("--expected-commit", required=True)
    child.add_argument("--label", choices=["old", "d1"], required=True)
    child.add_argument("--model", required=True)
    child.add_argument("--revision", required=True)
    child.add_argument("--output", type=Path, required=True)

    parent = commands.add_parser("run")
    parent.add_argument("--old-root", type=Path, required=True)
    parent.add_argument("--d1-root", type=Path, required=True)
    parent.add_argument("--old-commit", required=True)
    parent.add_argument("--d1-commit", required=True)
    parent.add_argument("--d1-evidence-sha256", required=True)
    parent.add_argument("--model", required=True)
    parent.add_argument("--revision", required=True)
    parent.add_argument("--output-dir", type=Path, required=True)
    parent.add_argument("--raw-output", type=Path, required=True)
    parent.add_argument("--public-output", type=Path, required=True)
    parent.add_argument("--threshold-pct", type=float, default=5.0)
    parent.add_argument("--memory-min-percent", type=int, default=80)
    parent.add_argument("--swap-ceiling-mib", type=int, default=256)
    parent.add_argument("--child-timeout-seconds", type=int, default=600)
    parent.add_argument("--artifact-date", type=_artifact_date, required=True)

    reanalyze = commands.add_parser("reanalyze")
    reanalyze.add_argument("--parent-raw", type=Path, required=True)
    reanalyze.add_argument("--old-commit", required=True)
    reanalyze.add_argument("--d1-commit", required=True)
    reanalyze.add_argument("--execution-surface-sha256", required=True)
    reanalyze.add_argument("--threshold-pct", type=float, default=5.0)
    reanalyze.add_argument("--output", type=Path, required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "child":
        return run_child(args)
    if args.command == "reanalyze":
        return run_reanalyze(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
