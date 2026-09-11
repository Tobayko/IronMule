"""Fail-closed, process-isolated DATA1 native-matmul measurement runner.

The parent process validates and persists control evidence only.  It never
imports MLX, PyTorch, or JAX.  Accelerator imports and tensor operations happen
solely in a spawned worker which has both a parent-enforced server timeout and a
shorter in-guest work deadline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import queue
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..budget import BudgetError, BudgetGuard
from ..canonical import canonical_json
from ..registry import BudgetPolicy
from ..statistics import paired_ratio, summarise
from .contracts import code_digest, file_sha256, load_json, safe_file, spec_digest, validate_case, validate_spec, write_json_new


REPORT_SCHEMA = "ironmule.collection.v1"
EVENT_SCHEMA = "ironmule.collection-event.v1"


class RunnerError(RuntimeError):
    """A portable-runner input, persistence, or control-plane error."""


class WorkDeadlineExceeded(TimeoutError):
    """The in-guest DATA1 work budget has expired."""


def _error(exc: BaseException) -> dict[str, str]:
    """Persist a stable failure class without untrusted path/error text."""
    known = {
        "WorkDeadlineExceeded": "work_deadline_exceeded",
        "BudgetError": "budget_guard_rejected",
        "UnsupportedCandidate": "candidate_unsupported",
        "BackendError": "backend_unavailable",
        "RunnerError": "runner_input_or_execution_error",
    }
    return {
        "type": type(exc).__name__,
        "code": known.get(type(exc).__name__, "worker_exception"),
    }


def _code_sha256() -> str:
    """Bind every report to the common portable evidence bundle."""
    return code_digest()


def _load_operands(case: Mapping[str, Any], data_dir: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Hash-check and load one captured tensor pair without pickle support."""
    started = time.perf_counter()
    root = data_dir.resolve(strict=True)
    validated = validate_case(dict(case))
    try:
        a_path = safe_file(root, str(validated["a_file"]))
        b_path = safe_file(root, str(validated["b_file"]))
    except ValueError as exc:
        raise RunnerError("case data file is not an approved local regular file") from exc
    if file_sha256(a_path) != validated["a_sha256"] or file_sha256(b_path) != validated["b_sha256"]:
        raise RunnerError("captured tensor SHA-256 does not match the sealed case")
    try:
        left = np.load(a_path, allow_pickle=False)
        right = np.load(b_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise RunnerError("captured tensor file is not a safe .npy array") from exc
    shape = validated["shape"]
    expected_a = (shape[0], shape[1])
    expected_b = (shape[1], shape[2])
    if left.shape != expected_a or right.shape != expected_b:
        raise RunnerError("captured tensor shapes differ from sealed case shape")
    if left.dtype != np.dtype("float32") or right.dtype != np.dtype("float32"):
        raise RunnerError("captured tensors must be float32")
    if not bool(np.isfinite(left).all() and np.isfinite(right).all()):
        raise RunnerError("captured tensors contain non-finite values")
    # The backend gets private device copies; host captures cannot be modified.
    left = np.ascontiguousarray(left)
    right = np.ascontiguousarray(right)
    left.setflags(write=False)
    right.setflags(write=False)
    return left, right, time.perf_counter() - started


def _deadline_for(spec: Mapping[str, Any]) -> float:
    declared = int(spec["work_seconds"])
    ceiling = 60 if spec["mode"] == "smoke" else 720
    return time.monotonic() + min(declared, ceiling)


def _check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise WorkDeadlineExceeded("in-guest work deadline exceeded")


def _pair_order(spec: Mapping[str, Any], case_id: str, candidate: str, index: int) -> str:
    """Seed the balanced AB/BA sequence without changing its pair counts."""
    material = f"{spec['seed']}:{case_id}:{candidate}".encode("ascii")
    starts_ab = (hashlib.sha256(material).digest()[0] & 1) == 0
    return "AB" if starts_ab == (index % 2 == 0) else "BA"


def aa_noise_gate(trial: Mapping[str, Any], pairs: int) -> dict[str, Any]:
    """Freeze A/A resolution before examining any non-reference candidate."""
    interval = trial["statistics"]["paired"]
    half_width = max(abs(float(interval["ci_low"]) - 1), abs(float(interval["ci_high"]) - 1))
    # 5% target effect, at most 2.5% A/A uncertainty. Recommendations never
    # change a sealed run's pair count or trigger a retry.
    required = max(12, math.ceil(pairs * (half_width / 0.025)**2))
    return {"target_effect": 0.05, "maximum_aa_uncertainty": 0.025,
            "observed_aa_uncertainty": half_width, "passed": half_width <= 0.025,
            "suggested_pairs": required if required <= 48 else None,
            "automatic_retry": False}


def _budget_for(spec: Mapping[str, Any]) -> BudgetGuard:
    # This DATA1 worker must fit inside its own short guest budget.  The same
    # guard still enforces continuous-load and duty-cycle limits on actual GPU
    # operation time; it never estimates hardware time from CPU fallback work.
    work = float(spec["work_seconds"])
    policy = BudgetPolicy(
        gpu_work_limit_s=work,
        continuous_gpu_limit_s=6.0,
        required_break_s=4.0,
        duty_window_s=60.0,
        duty_cycle_limit=0.25,
        wall_limit_s=work,
        candidate_cooldown_s=60.0,
    )
    return BudgetGuard(policy)


def _resource_snapshot(backend: Any) -> dict[str, Any]:
    """Get explicit resource availability without letting telemetry vanish."""
    try:
        raw = backend.resources()
    except Exception:
        return {"availability": "unknown", "reason": "resource_probe_failed"}
    if not isinstance(raw, dict):
        return {"availability": "unknown", "reason": "resource_probe_invalid"}
    return dict(raw)


def _require_bounded_resources(sample: Mapping[str, Any]) -> None:
    """Fail closed when live memory cannot be bounded before/after device work."""
    if sample.get("availability") != "available":
        raise RunnerError("resource availability is unknown")
    total = sample.get("total_memory_bytes")
    if type(total) is not int or total <= 0:
        raise RunnerError("resource total memory is unavailable")
    for key, value in sample.items():
        if key.endswith("memory_bytes") and key != "total_memory_bytes":
            if type(value) is not int or value < 0 or value > total:
                raise RunnerError("resource memory limit is violated")


def _trial(
    *,
    spec: Mapping[str, Any], case: Mapping[str, Any], candidate_name: str,
    backend: Any, left: np.ndarray, right: np.ndarray, input_setup_seconds: float,
    deadline: float, guard: BudgetGuard, code_sha: str, spec_sha: str,
) -> dict[str, Any]:
    """Compile, warm, pair and exact-check a single candidate on native tensors."""
    from .backends import UnsupportedCandidate, candidate_rows

    hardware, environment = backend.fingerprint()
    base: dict[str, Any] = {
        "run_id": spec["run_id"], "case_id": case["case_id"], "backend": spec["backend"],
        "candidate": candidate_name, "partition": spec["partition"], "case": dict(case),
        "hardware": hardware, "environment": environment, "code_sha256": code_sha,
        "spec_sha256": spec_sha, "status": "failed", "samples": [],
        "resources": {"before": {"availability": "unknown", "reason": "not_sampled"}, "after": {"availability": "unknown", "reason": "not_sampled"}},
    }
    costs: dict[str, Any] = {
        "input_setup_seconds": input_setup_seconds,
        "compile_seconds": 0.0,
        "compile_accelerator_busy_upper_bound_seconds": 0.0,
        "correctness_seconds": 0.0,
        "warmup_seconds": 0.0,
        "trial_seconds": 0.0,
        "wall_seconds": 0.0,
    }
    started = time.perf_counter()
    try:
        _check_deadline(deadline)
        guard.check_wall()
        base["resources"]["before"] = _resource_snapshot(backend)
        _require_bounded_resources(base["resources"]["before"])
        setup_started = time.perf_counter()
        device_left, device_right = backend.prepare_inputs(left, right)
        backend.synchronize()
        costs["input_setup_seconds"] += time.perf_counter() - setup_started
        guard.before_candidate()
        compile_started = time.perf_counter()
        candidate = backend.compile_candidate(candidate_name, candidate_rows(candidate_name))
        # Torch and JAX compilation is commonly lazy.  Prime it before the
        # recorded warmups, include that real native-device work in compile
        # cost, and reject a non-finite generated tensor immediately.
        if candidate_name != "native":
            primed, compile_gpu_seconds = backend._timed(lambda: candidate(device_left, device_right))
            # Compilation may include substantial host compiler time.  Retain
            # that conservative accelerator-busy upper bound as cost evidence,
            # but never mislabel it as measured GPU work for BudgetGuard.
            costs["compile_accelerator_busy_upper_bound_seconds"] = compile_gpu_seconds
            if not bool(np.isfinite(np.ascontiguousarray(backend.to_host(primed), dtype=np.float32)).all()):
                raise RunnerError("compiled candidate produced non-finite output during compilation")
        costs["compile_seconds"] = time.perf_counter() - compile_started
        for index in range(int(spec["warmup"])):
            _check_deadline(deadline)
            guard.check_wall()
            warmup_started = time.perf_counter()
            baseline, observed, exact, check_seconds = backend.pair(
                device_left, device_right, candidate, _pair_order(spec, str(case["case_id"]), candidate_name, index)
            )
            costs["warmup_seconds"] += time.perf_counter() - warmup_started
            costs["correctness_seconds"] += check_seconds
            guard.record_gpu(baseline + observed)
            if not exact:
                base["status"] = "incorrect"
                base["error"] = {"type": "ByteExactMismatch", "message": "candidate differs from native output during warmup"}
                return base
        for index in range(int(spec["pairs"])):
            _check_deadline(deadline)
            guard.check_wall()
            order = _pair_order(spec, str(case["case_id"]), candidate_name, index)
            sample_started = time.perf_counter()
            baseline, observed, exact, check_seconds = backend.pair(device_left, device_right, candidate, order)
            costs["trial_seconds"] += time.perf_counter() - sample_started
            costs["correctness_seconds"] += check_seconds
            guard.record_gpu(baseline + observed)
            base["samples"].append({
                "order": order, "baseline_seconds": baseline, "candidate_seconds": observed, "exact": exact,
            })
            if not exact:
                base["status"] = "incorrect"
                base["error"] = {"type": "ByteExactMismatch", "message": "candidate differs from native output during sample"}
                return base
        base["status"] = "valid"
        baseline_times = [sample["baseline_seconds"] for sample in base["samples"]]
        candidate_times = [sample["candidate_seconds"] for sample in base["samples"]]
        base["statistics"] = {
            "baseline": summarise(baseline_times), "candidate": summarise(candidate_times),
            "paired": paired_ratio(candidate_times, baseline_times),
        }
    except UnsupportedCandidate as exc:
        base["status"] = "unsupported"
        base["error"] = _error(exc)
    except (WorkDeadlineExceeded, BudgetError) as exc:
        base["status"] = "censored"
        base["error"] = _error(exc)
    except BaseException as exc:  # Worker must emit the failure rather than hide it.
        base["status"] = "failed"
        base["error"] = _error(exc)
    finally:
        base["resources"]["after"] = _resource_snapshot(backend)
        try:
            _require_bounded_resources(base["resources"]["after"])
        except RunnerError as exc:
            if base["status"] == "valid":
                base["status"] = "failed"
                base["error"] = _error(exc)
        costs["wall_seconds"] = time.perf_counter() - started
        costs["budget"] = guard.summary()
        base["costs"] = costs
        try:
            guard.finish_candidate()
        except BudgetError as exc:
            if base["status"] == "valid":
                base["status"] = "censored"
                base["error"] = _error(exc)
    return base


def _smoke_trial(
    *, spec: Mapping[str, Any], case: Mapping[str, Any], backend: Any, left: np.ndarray,
    right: np.ndarray, input_setup_seconds: float, deadline: float, guard: BudgetGuard,
    code_sha: str, spec_sha: str,
) -> dict[str, Any]:
    """Run one native operation to prove device execution, never performance."""
    hardware, environment = backend.fingerprint()
    trial: dict[str, Any] = {
        "run_id": spec["run_id"], "case_id": case["case_id"], "backend": spec["backend"],
        "candidate": "native", "partition": spec["partition"], "case": dict(case),
        "hardware": hardware, "environment": environment, "code_sha256": code_sha,
        "spec_sha256": spec_sha, "status": "valid", "samples": [],
        "resources": {"before": {"availability": "unknown", "reason": "not_sampled"}, "after": {"availability": "unknown", "reason": "not_sampled"}},
        "smoke_only": True,
        "costs": {
            "input_setup_seconds": input_setup_seconds, "compile_seconds": 0.0,
            "compile_accelerator_busy_upper_bound_seconds": 0.0, "correctness_seconds": 0.0,
            "warmup_seconds": 0.0, "trial_seconds": 0.0, "wall_seconds": 0.0,
        },
    }
    started = time.perf_counter()
    try:
        _check_deadline(deadline)
        guard.check_wall()
        trial["resources"]["before"] = _resource_snapshot(backend)
        _require_bounded_resources(trial["resources"]["before"])
        setup_started = time.perf_counter()
        device_left, device_right = backend.prepare_inputs(left, right)
        backend.synchronize()
        trial["costs"]["input_setup_seconds"] += time.perf_counter() - setup_started
        operation_started = time.perf_counter()
        output, seconds = backend._timed(lambda: backend.native(device_left, device_right))
        trial["costs"]["trial_seconds"] = time.perf_counter() - operation_started
        check_started = time.perf_counter()
        finite = bool(np.isfinite(np.ascontiguousarray(backend.to_host(output), dtype=np.float32)).all())
        trial["costs"]["correctness_seconds"] = time.perf_counter() - check_started
        guard.record_gpu(seconds)
        if not finite:
            trial["status"] = "incorrect"
            trial["error"] = {"type": "NonFiniteOutput", "code": "native_smoke_nonfinite"}
    except (WorkDeadlineExceeded, BudgetError) as exc:
        trial["status"] = "censored"
        trial["error"] = _error(exc)
    except BaseException as exc:
        trial["status"] = "failed"
        trial["error"] = _error(exc)
    finally:
        trial["resources"]["after"] = _resource_snapshot(backend)
        try:
            _require_bounded_resources(trial["resources"]["after"])
        except RunnerError as exc:
            if trial["status"] == "valid":
                trial["status"] = "failed"
                trial["error"] = _error(exc)
        trial["costs"]["wall_seconds"] = time.perf_counter() - started
        trial["costs"]["budget"] = guard.summary()
    return trial


def _worker_main(
    spec: dict[str, Any], data_dir_text: str, code_sha: str,
    event_queue: Any, result_queue: Any,
) -> None:
    """Spawn target: all framework imports occur below this line."""
    spec_sha = spec_digest(spec)
    trials: list[dict[str, Any]] = []
    hardware: dict[str, Any] = {}
    environment: dict[str, Any] = {}
    deadline = _deadline_for(spec)
    try:
        from .backends import make_backend

        backend = make_backend(spec["backend"])
        hardware, environment = backend.fingerprint()
        event_queue.put({"kind": "started", "hardware": hardware, "environment": environment})
        data_dir = Path(data_dir_text)
        guard = _budget_for(spec)
        for raw_case in spec["cases"]:
            _check_deadline(deadline)
            case = validate_case(dict(raw_case))
            left, right, load_seconds = _load_operands(case, data_dir)
            if spec["mode"] == "smoke":
                trial = _smoke_trial(
                    spec=spec, case=case, backend=backend, left=left, right=right,
                    input_setup_seconds=load_seconds, deadline=deadline, guard=guard,
                    code_sha=code_sha, spec_sha=spec_sha,
                )
                trials.append(trial)
                event_queue.put({"kind": "trial", "trial": trial})
                if trial["status"] in {"censored", "failed"}:
                    raise WorkDeadlineExceeded("smoke stopped before completion")
                break
            for candidate in spec["candidates"]:
                _check_deadline(deadline)
                guard.check_wall()
                trial = _trial(
                    spec=spec, case=case, candidate_name=candidate, backend=backend,
                    left=left, right=right, input_setup_seconds=load_seconds,
                    deadline=deadline, guard=guard, code_sha=code_sha, spec_sha=spec_sha,
                )
                if candidate == "native" and trial["status"] == "valid":
                    trial["aa_calibration"] = aa_noise_gate(trial, spec["pairs"])
                    if not trial["aa_calibration"]["passed"]:
                        trial["status"] = "censored"
                        trial["error"] = {"type": "NoiseGate", "code": "aa_resolution_insufficient"}
                trials.append(trial)
                event_queue.put({"kind": "trial", "trial": trial})
                if trial["status"] == "censored":
                    result_queue.put({"status": "censored", "hardware": hardware,
                        "environment": environment, "trials": trials, "error": trial.get("error")})
                    return
                if trial["status"] == "failed":
                    raise RunnerError("measurement stopped after failed trial")
        status = "smoke" if spec["mode"] == "smoke" else "complete"
        result_queue.put({"status": status, "hardware": hardware, "environment": environment, "trials": trials})
    except WorkDeadlineExceeded as exc:
        result_queue.put({"status": "censored", "hardware": hardware, "environment": environment, "trials": trials, "error": _error(exc)})
    except BaseException as exc:
        result_queue.put({"status": "failed", "hardware": hardware, "environment": environment, "trials": trials, "error": _error(exc)})


def _event_path(output_dir: Path, run_id: str) -> Path:
    return output_dir / f"{run_id}.events.jsonl"


def _append_event(path: Path, event: Mapping[str, Any]) -> None:
    """Append one canonical, fsynced event; existing lines are never rewritten."""
    payload = canonical_json({"schema": EVENT_SCHEMA, "observed_at_unix_ns": time.time_ns(), **dict(event)}) + "\n"
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "a", encoding="utf-8", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _drain_events(events: Any, event_path: Path) -> list[dict[str, Any]]:
    """Persist every child progress message already delivered to the parent."""
    drained: list[dict[str, Any]] = []
    try:
        while True:
            event = events.get_nowait()
            _append_event(event_path, event)
            if isinstance(event, dict):
                drained.append(event)
    except queue.Empty:
        return drained


class _DirectEventSink:
    """Persist direct-worker progress at each ``put`` so a kill loses no trial."""

    def __init__(self, event_path: Path):
        self._event_path = event_path

    def put(self, event: Mapping[str, Any]) -> None:
        _append_event(self._event_path, event)

    def get_nowait(self) -> Any:
        raise queue.Empty


def _empty_report(spec: Mapping[str, Any], spec_sha: str, code_sha: str) -> dict[str, Any]:
    return {
        "schema": REPORT_SCHEMA, "run_id": spec["run_id"], "spec_sha256": spec_sha,
        "backend": spec["backend"], "partition": spec["partition"], "hardware": {},
        "environment": {}, "code_sha256": code_sha, "status": "failed", "trials": [],
        "performance_claim": False,
    }


def run_experiment(spec: dict[str, Any], data_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Run a sealed portable experiment in a bounded worker and persist evidence.

    This API is intentionally the only execution surface.  It verifies the
    supplied spec before spawning and writes a one-time report plus an
    append-only event stream suitable for a read-only dashboard.
    """
    if not isinstance(spec, dict):
        raise RunnerError("spec must be an object")
    sealed = validate_spec(dict(spec))
    spec_sha = spec_digest(sealed)
    code_sha = _code_sha256()
    if sealed.get("code_sha256") not in (None, code_sha):
        raise RunnerError("sealed code_sha256 differs from the executing portable bundle")
    source_dir = Path(data_dir)
    if not source_dir.is_dir():
        raise RunnerError("data_dir must be an existing directory")
    destination = Path(output_dir)
    if destination.is_symlink():
        raise RunnerError("output_dir must not be a symlink")
    destination.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise RunnerError("output_dir must not become a symlink")
    event_path = _event_path(destination, sealed["run_id"])
    report_path = destination / f"{sealed['run_id']}.report.json"
    if event_path.exists() or report_path.exists():
        raise RunnerError("run_id already has immutable portable output")
    # Create the stream before the child so a kill/timeout has an immutable
    # parent-side record even if the child never finishes framework import.
    descriptor = os.open(event_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    os.close(descriptor)
    _append_event(event_path, {"kind": "accepted", "spec_sha256": spec_sha, "code_sha256": code_sha})
    report = _empty_report(sealed, spec_sha, code_sha)
    partial_trials: list[dict[str, Any]] = []
    direct = os.environ.get("FRIDAY_PORTABLE_DIRECT_WORKER") == "1"
    process: Any = None
    if direct:
        # The supervisor's CLI child is the only accelerator-owning process in
        # this mode.  Its PID is therefore exactly the PID sampled by the
        # parent memory guard; library callers retain the spawned isolation.
        events: Any = _DirectEventSink(event_path)
        results: Any = queue.Queue(maxsize=1)
        _worker_main(sealed, str(source_dir), code_sha, events, results)
        result: dict[str, Any] | None = results.get_nowait()
        for event in _drain_events(events, event_path):
            if event.get("kind") == "trial" and isinstance(event.get("trial"), dict):
                partial_trials.append(event["trial"])
    else:
        context = mp.get_context("spawn")
        events = context.Queue()
        results = context.Queue(maxsize=1)
        process = context.Process(target=_worker_main, args=(sealed, str(source_dir), code_sha, events, results), daemon=True)
        process.start()
        started = time.monotonic()
        server_limit = float(sealed["server_seconds"])
        result = None
    try:
        while not direct and process.is_alive() and time.monotonic() - started < server_limit:
            for event in _drain_events(events, event_path):
                if event.get("kind") == "trial" and isinstance(event.get("trial"), dict):
                    partial_trials.append(event["trial"])
            try:
                result = results.get_nowait()
                break
            except queue.Empty:
                pass
            process.join(0.05)
        if not direct and result is None:
            try:
                result = results.get_nowait()
            except queue.Empty:
                pass
        if not direct and result is None and process.is_alive():
            process.terminate()
            process.join(5)
            result = {"status": "censored", "hardware": {}, "environment": {}, "trials": partial_trials, "error": {"type": "ServerTimeout", "code": "server_deadline_exceeded"}}
        if result is None:
            exit_code = None if direct else process.exitcode
            result = {"status": "failed", "hardware": {}, "environment": {}, "trials": partial_trials, "error": {"type": "WorkerExit", "code": "worker_exited_without_result", "exit_code": exit_code}}
        for event in _drain_events(events, event_path):
            if event.get("kind") == "trial" and isinstance(event.get("trial"), dict):
                partial_trials.append(event["trial"])
        report.update({
            "status": result["status"], "hardware": result.get("hardware", {}),
            "environment": result.get("environment", {}), "trials": result.get("trials", []),
        })
        if "error" in result:
            report["error"] = result["error"]
        _append_event(event_path, {"kind": "finished", "status": report["status"], "trials": len(report["trials"]), "error": report.get("error")})
    finally:
        if not direct and process.is_alive():
            process.terminate()
            process.join(5)
    write_json_new(report_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="run a sealed DATA1 native portable measurement")
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_experiment(load_json(args.spec), args.data_dir, args.output_dir)
    except (RunnerError, ValueError, OSError) as exc:
        print(f"portable runner refused input: {type(exc).__name__}", file=sys.stderr)
        return 2
    print(json.dumps({"run_id": report["run_id"], "status": report["status"], "report_schema": report["schema"]}, sort_keys=True))
    return 0 if report["status"] in {"complete", "smoke"} else 1


if __name__ == "__main__":  # pragma: no cover - CLI process entrypoint
    raise SystemExit(main())


__all__ = ["REPORT_SCHEMA", "RunnerError", "main", "run_experiment"]
