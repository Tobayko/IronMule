"""Native PROD4 load-only screen; generation is intentionally impossible here."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import threading
import time
import uuid
from typing import Any

from friday_evidence.budget import BudgetGuard
from friday_evidence.canonical import canonical_json_bytes, canonical_sha256
from friday_evidence.events import EventJournal
from friday_evidence.identity import runtime_identity
from ironmule_product.calibration import model_lease
from ironmule_product.memory import LoadMemoryGuard, MemoryGuardError, SWAP_DELTA_LIMIT_BYTES
from ironmule_product.readiness import ReadinessPolicy, ReadinessWindow, hardware_identity, probe
from ironmule_product.state import ProductStore


JOURNAL_NAME = "load-screen.sqlite3"
SCHEMA = "ironmule.prod4_load_memory.v1"
MAX_OUTPUT_BYTES = 2 * 1024 * 1024
MAX_SOURCE_FILE_BYTES = 16 * 1024 * 1024
DEFAULT_WAIT_READY = 120.0
READINESS_POLICY = ReadinessPolicy(stable_samples=3, sample_interval_s=5.0, max_gap_s=15.0, max_sample_age_s=10.0)


class LoadScreenFailure(RuntimeError):
    def __init__(self, code: str, detail: str | None = None):
        self.code = code
        super().__init__(detail or code)


_EXPECTED_ABORT_CODES = frozenset(("cancelled", "paused", "job_deadline", "readiness_timeout"))


def _memory_guard_code(exc: BaseException) -> str | None:
    """Recover a typed memory failure wrapped by startup guard cleanup."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, MemoryGuardError):
            return current.code
        current = current.__cause__ or current.__context__
    return None


def classify_worker_exit(
    returncode: int | None,
    forced_error: str | None = None,
    controlled_error: str | None = None,
) -> str:
    """Classify the child's actual exit without masking an active guard error."""
    if returncode == 0:
        return "clean"
    if forced_error is not None:
        return "forced_abort"
    if controlled_error in _EXPECTED_ABORT_CODES:
        return "expected_abort"
    return "crashed"


def _source_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = [
        Path(__file__).resolve(),
        root / "docs" / "PROD4_LOAD_MEMORY_SPEC.md",
        root / "ironmule_product" / "backend.py",
        root / "ironmule_product" / "worker.py",
        root / "ironmule_product" / "memory.py",
        root / "ironmule_product" / "readiness.py",
        root / "friday_evidence" / "identity.py",
        root / "friday_evidence" / "events.py",
        root / "friday_evidence" / "budget.py",
    ]
    result = {}
    for path in paths:
        if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_SOURCE_FILE_BYTES:
            raise LoadScreenFailure("source_manifest_invalid")
        result[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _exclusive_write(path: Path, value: dict) -> None:
    encoded = canonical_json_bytes(value) + b"\n"
    if len(encoded) > MAX_OUTPUT_BYTES:
        raise LoadScreenFailure("output_too_large")
    path = path.expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.parent.is_dir():
        raise LoadScreenFailure("output_parent_unavailable")
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise LoadScreenFailure("output_exists") from exc
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _sleep_cancel(seconds: float, check, cancel) -> None:
    end = time.monotonic() + seconds
    while True:
        check()
        remaining = end - time.monotonic()
        if remaining <= 0:
            return
        cancel.wait(min(0.25, remaining))


def _wait_ready(*, deadline: float, global_deadline: float, check, sleep, journal, run_id: str, phase: str,
                memory_guard: LoadMemoryGuard | None = None, sample_sink: list[dict] | None = None) -> tuple[dict, list[dict]]:
    window = ReadinessWindow(READINESS_POLICY)
    observations: list[dict] = []
    while time.monotonic() < min(deadline, global_deadline):
        check()
        observation = probe()
        decision = window.observe(observation)
        row = {"phase": phase, "observation": observation, "decision": decision}
        observations.append(row)
        if sample_sink is not None:
            sample_sink.append(row)
        journal.append(run_id, "readiness", row)
        if decision["stable"]:
            if memory_guard is not None and memory_guard._bound_pid is not None:
                try:
                    memory_guard(memory_guard._bound_pid, force=True)
                except MemoryGuardError as exc:
                    raise LoadScreenFailure(exc.code) from exc
            elif memory_guard is not None:
                raise LoadScreenFailure("load_memory_sample_missing")
            check()
            return observation, observations
        end = min(deadline, global_deadline, time.monotonic() + READINESS_POLICY.sample_interval_s)
        while time.monotonic() < end:
            sleep(min(0.25, end - time.monotonic()))
    raise LoadScreenFailure(f"{phase}_readiness_timeout")


def _versions() -> dict[str, str | None]:
    from importlib import metadata
    result: dict[str, str | None] = {}
    for name in ("mlx", "mlx-lm", "numpy", "transformers"):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def _installed_package_proof() -> dict[str, Any]:
    import importlib

    project_root = Path(__file__).resolve().parents[1]
    names = ("ironmule_product.backend", "ironmule_product.memory", "friday_evidence.events", "friday_evidence.identity")
    module_hashes: dict[str, str] = {}
    outside = True
    for name in names:
        module = importlib.import_module(name)
        raw_origin = getattr(module, "__file__", None)
        if not isinstance(raw_origin, str):
            raise LoadScreenFailure("installed_package_origin_missing")
        origin = Path(raw_origin).resolve()
        try:
            origin.relative_to(project_root)
            outside = False
        except ValueError:
            pass
        if not origin.is_file() or origin.stat().st_size > MAX_OUTPUT_BYTES:
            raise LoadScreenFailure("installed_package_origin_invalid")
        module_hashes[name] = hashlib.sha256(origin.read_bytes()).hexdigest()
    if not outside:
        raise LoadScreenFailure("source_tree_package_in_use")
    versions = _versions()
    return {
        "outside_project_root": True,
        "module_hash": canonical_sha256(module_hashes),
        "versions_hash": canonical_sha256(versions),
        "versions": versions,
    }


def run(state_dir: Path, model_id: str, output: Path, wait_ready: float) -> dict:
    if not isinstance(model_id, str) or not model_id or len(model_id) > 4096:
        raise LoadScreenFailure("invalid_model_id")
    if isinstance(wait_ready, bool) or not isinstance(wait_ready, (int, float)) or not math.isfinite(wait_ready) or not 0 <= wait_ready <= 900:
        raise LoadScreenFailure("invalid_wait_ready")
    output = output.expanduser()
    if not output.is_absolute():
        output = Path.cwd() / output
    if output.exists() or output.is_symlink():
        raise LoadScreenFailure("output_exists")
    state_dir = state_dir.expanduser().absolute()
    store = ProductStore(state_dir)
    store.settings()
    spec = store.model(model_id)
    run_id = uuid.uuid4().hex
    report: dict[str, Any] = {
        "schema": SCHEMA, "run_id": run_id, "status": "started",
        "model_id": spec.model_id, "revision": spec.revision,
        "performance_claim": False, "activation_allowed": False,
        "samples": [], "readiness_samples": [], "load_memory_samples": [],
        "worker": None, "source_manifest_before": _source_manifest(),
        "platform": platform.platform(), "package_import": None,
    }
    cancel = threading.Event()
    guard: BudgetGuard | None = None
    load_guard: LoadMemoryGuard | None = None
    process = None
    journal: EventJournal | None = None
    job_deadline = time.monotonic() + 20 * 60

    def check() -> None:
        if cancel.is_set():
            raise LoadScreenFailure("cancelled")
        if time.monotonic() >= job_deadline:
            raise LoadScreenFailure("job_deadline")
        if store.settings()["optimization_paused"]:
            raise LoadScreenFailure("paused")
        if guard is not None:
            guard.check_wall()
        if load_guard is not None and process is not None:
            try:
                load_guard(process.pid)
            except MemoryGuardError as exc:
                raise LoadScreenFailure(exc.code) from exc

    def sleep(seconds: float) -> None:
        _sleep_cancel(seconds, check, cancel)

    try:
        with EventJournal(state_dir / JOURNAL_NAME) as journal_handle:
            journal = journal_handle
            journal.append(run_id, "run_started", {
                "model_id": spec.model_id, "revision": spec.revision,
                "source_manifest_sha256": canonical_sha256(report["source_manifest_before"]),
                "plan_sha256": canonical_sha256({
                    "schema": SCHEMA,
                    "spec_sha256": hashlib.sha256((Path(__file__).resolve().parents[1] / "docs" / "PROD4_LOAD_MEMORY_SPEC.md").read_bytes()).hexdigest(),
                    "policy": asdict(READINESS_POLICY), "wait_ready": float(wait_ready),
                    "swap_delta_limit_bytes": SWAP_DELTA_LIMIT_BYTES,
                    "rss_limit_fraction": 0.60,
                }),
            })
            report["package_import"] = _installed_package_proof()
            report["versions"] = report["package_import"]["versions"]
            with model_lease(store):
                initial_deadline = min(job_deadline, time.monotonic() + float(wait_ready))
                initial, _ = _wait_ready(
                    deadline=initial_deadline, global_deadline=job_deadline,
                    check=check, sleep=sleep, journal=journal, run_id=run_id,
                    phase="before_load", sample_sink=report["readiness_samples"],
                )
                hardware = hardware_identity()
                report["hardware"] = hardware
                report["memory_total_bytes"] = initial["memory_total_bytes"]
                report["swap_baseline_bytes"] = initial["swap_used_bytes"]
                if type(report["memory_total_bytes"]) is not int or report["memory_total_bytes"] <= 0:
                    raise LoadScreenFailure("memory_total_invalid")
                if type(report["swap_baseline_bytes"]) is not int or report["swap_baseline_bytes"] < 0:
                    raise LoadScreenFailure("swap_baseline_invalid")
                host = {**initial, **hardware, "chip_name": hardware.get("chip_name"),
                        "memory_total_bytes": report["memory_total_bytes"]}
                report["identity_before"] = runtime_identity(spec.as_dict(), host)
                journal.append(run_id, "validation", {
                    "state": "identity_before",
                    "identity": report["identity_before"],
                })
                after_identity, _ = _wait_ready(
                    deadline=min(job_deadline, time.monotonic() + float(wait_ready)),
                    global_deadline=job_deadline, check=check, sleep=sleep,
                    journal=journal, run_id=run_id, phase="after_identity",
                    sample_sink=report["readiness_samples"],
                )
                if type(after_identity.get("swap_used_bytes")) is not int:
                    raise LoadScreenFailure("swap_telemetry_missing")
                if after_identity["swap_used_bytes"] - report["swap_baseline_bytes"] > SWAP_DELTA_LIMIT_BYTES:
                    raise LoadScreenFailure("swap_growth")
                guard = BudgetGuard(sleeper=sleep)

                def on_load_memory_sample(sample: dict[str, Any]) -> None:
                    report["load_memory_samples"].append(dict(sample))
                    journal.append(run_id, "validation", {
                        "state": "load_memory_sample", "observation": dict(sample),
                    })

                load_guard = LoadMemoryGuard(
                    swap_baseline_bytes=report["swap_baseline_bytes"],
                    memory_total_bytes=report["memory_total_bytes"],
                    on_sample=on_load_memory_sample,
                )
                started = time.monotonic()
                worker_record = {
                    "started_monotonic": started, "pid": None, "returncode": None,
                    "ready_monotonic": None, "closed_monotonic": None,
                }
                report["worker"] = worker_record

                def startup_guard(pid: int) -> None:
                    nonlocal process
                    process = client._process
                    if process is None or process.pid != pid:
                        raise LoadScreenFailure("worker_start_handle_missing")
                    worker_record["pid"] = pid
                    check()

                from ironmule_product.backend import MLXWorkerClient
                client = MLXWorkerClient(spec, startup_timeout=max(0.1, float(wait_ready)))
                forced_error: str | None = None
                controlled_error: str | None = None
                worker_exception = False
                try:
                    ready = client.start(startup_guard=startup_guard)
                    if process is None:
                        raise LoadScreenFailure("worker_start_handle_missing")
                    ready_metrics = {
                        key: ready.get(key) for key in (
                            "startup_wall_seconds", "process_peak_rss_bytes", "mlx_active_bytes",
                            "mlx_peak_bytes", "mlx_cache_bytes", "recommended_working_set_bytes",
                        )
                    } if isinstance(ready, dict) else {}
                    report["ready_telemetry"] = ready_metrics
                    journal.append(run_id, "validation", {
                        "state": "load_memory_ready", "ready": ready_metrics,
                    })
                    ready_observation = load_guard(process.pid, force=True)
                    ready_checked = load_guard.validate_ready(ready)
                    worker_record["ready_monotonic"] = time.monotonic()
                    report["ready_telemetry_checked"] = ready_checked
                    report["load_memory_ready_observation"] = ready_observation
                    journal.append(run_id, "worker_started", {
                        "pid": process.pid, "ready_monotonic": worker_record["ready_monotonic"],
                    })
                    guard.required_break()
                    after, _ = _wait_ready(
                        deadline=min(job_deadline, time.monotonic() + float(wait_ready)),
                        global_deadline=job_deadline, check=check, sleep=sleep,
                        journal=journal, run_id=run_id, phase="after_load",
                        memory_guard=load_guard, sample_sink=report["readiness_samples"],
                    )
                except MemoryGuardError as exc:
                    forced_error = exc.code
                    controlled_error = exc.code
                    raise LoadScreenFailure(exc.code) from exc
                except BaseException as exc:
                    worker_exception = True
                    forced_error = _memory_guard_code(exc)
                    controlled_error = getattr(exc, "code", None)
                    raise
                finally:
                    active_process = process
                    process = None
                    load_guard = None
                    client.close()
                    worker_record["closed_monotonic"] = time.monotonic()
                    worker_record["returncode"] = active_process.poll() if active_process is not None else None
                    worker_record["exit_class"] = classify_worker_exit(
                        worker_record["returncode"], forced_error, controlled_error,
                    )
                    journal.append(run_id, "validation", {
                        "state": "worker_cleanup", "worker": dict(worker_record),
                    })
                    if active_process is not None and active_process.poll() is None:
                        raise LoadScreenFailure("worker_cleanup_failed")
                    if worker_record["exit_class"] == "crashed" and not worker_exception:
                        raise LoadScreenFailure("worker_exit_nonzero")
                report["samples"] = report["readiness_samples"] + report["load_memory_samples"]
                report["identity_after"] = runtime_identity(
                    spec.as_dict(), {**after, **hardware, "chip_name": hardware.get("chip_name"),
                                     "memory_total_bytes": report["memory_total_bytes"]},
                )
                if report["identity_after"]["identity_sha256"] != report["identity_before"]["identity_sha256"]:
                    raise LoadScreenFailure("identity_changed")
                journal.append(run_id, "validation", {
                    "state": "identity_after", "identity": report["identity_after"],
                })
                check()
                report["status"] = "passed"
                report["budget"] = guard.summary()
            report["source_manifest_after"] = _source_manifest()
            if report["source_manifest_after"] != report["source_manifest_before"]:
                raise LoadScreenFailure("source_changed")
            journal.append(run_id, "run_finished", {
                "status": report["status"], "worker": report["worker"],
                "identity_sha256": report.get("identity_after", {}).get("identity_sha256"),
            })
            return report
    except BaseException as exc:
        report["status"] = "failed"
        report["error_code"] = getattr(exc, "code", type(exc).__name__.lower())
        report["error_type"] = type(exc).__name__
        report["samples"] = report["readiness_samples"] + report["load_memory_samples"]
        if guard is not None:
            report["budget"] = guard.summary()
        report.setdefault("source_manifest_after", _source_manifest())
        try:
            with EventJournal(state_dir / JOURNAL_NAME) as failed_journal:
                failed_journal.append(run_id, "run_finished", {
                    "status": "failed", "error_code": report["error_code"],
                    "error_type": report["error_type"], "worker": report["worker"],
                })
        except Exception:
            report["journal_terminal_error"] = True
        return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--wait-ready", type=float, default=DEFAULT_WAIT_READY)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"state": "not_released", "hint": "pass --execute"}))
        return 78
    try:
        report = run(args.state_dir, args.model, args.output, args.wait_ready)
        _exclusive_write(args.output, report)
        print(json.dumps({"status": report["status"], "run_id": report["run_id"], "output": str(args.output)}))
        return 0 if report["status"] == "passed" else 1
    except BaseException as exc:
        print(json.dumps({"status": "failed", "error_code": getattr(exc, "code", type(exc).__name__.lower())}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
