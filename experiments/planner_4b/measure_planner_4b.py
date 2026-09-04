#!/usr/bin/env python3
"""Prospective one-shot validation of Gemma 4B as a closed evidence planner."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from _bench import require_ac_power, resolve_local_model_snapshot  # noqa: E402
from friday_evidence.budget import BudgetError, BudgetGuard  # noqa: E402
from friday_evidence.registry import BudgetPolicy  # noqa: E402

STUDY_ID = "gemma-4b-evidence-planner-20260824-01"
RUN_ID = "planner-4b-validation-20260824-01"
MODEL_ID = "mlx-community/gemma-3-4b-it-4bit"
MODEL_REVISION = "93724907d4ed1745d2fe50baadf3b0b01a65abf2"
EXPECTED_CANDIDATE = "persistent_service_qualification"
EXPECTED_CPU_BRAND = "Apple M1 Max"
EXPECTED_MACHINE = "arm64"
EXPECTED_MEMORY_BYTES = 32 * 1024**3
REQUIRED_PACKAGES = {"mlx": "0.32.0", "mlx-lm": "0.31.3"}
FROZEN_PREREGISTRATION_SHA256 = (
    "0fa346db7985cdd4dfa49015b395ee0f9d56a097a06f3828b0c161c45e53e5ec"
)
PREREGISTRATION = Path(__file__).with_name("PREREGISTRATION.md")
WORKER = Path(__file__).with_name("worker.py")
RESULT_PATH = Path(__file__).with_name("results.json")
ATTEMPT_DIR = PROJECT_ROOT / ".friday-data" / "planner-4b"
ATTEMPT_PATH = ATTEMPT_DIR / "attempt.json"
MAX_EVENT_BYTES = 1_000_000
WORKER_TIMEOUT_SECONDS = 90.0
RUN_COUNT = 3
MAX_OUTPUT_TOKENS = 32
MAX_RESPONSE_BYTES = 512
MAX_MEMORY_BYTES = 5 * 1024**3
PACING_TARGET = 0.075
WORKER_ENVIRONMENT = {
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "PYTHONNOUSERSITE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
UNSAFE_PYTHON_ENVIRONMENT = (
    "PYTHONHOME",
    "PYTHONINSPECT",
    "PYTHONPATH",
    "PYTHONSTARTUP",
)

POLICY = BudgetPolicy(
    gpu_work_limit_s=120.0,
    continuous_gpu_limit_s=6.0,
    required_break_s=4.0,
    duty_window_s=60.0,
    duty_cycle_limit=0.15,
    wall_limit_s=1_200.0,
    candidate_cooldown_s=60.0,
)


class StudyError(RuntimeError):
    """The one-shot study cannot continue safely or reproducibly."""


class WorkerError(StudyError):
    """A fixed worker violated the closed process protocol."""


def _load_worker_module() -> Any:
    specification = importlib.util.spec_from_file_location("planner_4b_worker", WORKER)
    if specification is None or specification.loader is None:
        raise StudyError("worker module is unavailable")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _token_sha256(tokens: list[int]) -> str:
    return hashlib.sha256(_canonical_json(tokens)).hexdigest()


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
    )
    return completed.stdout.decode("utf-8", errors="strict").strip()


def _require_clean_worktree() -> str:
    revision = _git("rev-parse", "HEAD")
    status = _git(
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        ".",
        ":(exclude)ProjectAtlas",
    )
    if status:
        raise StudyError("project worktree is dirty; commit the study before execution")
    return revision


def _swap_used_bytes() -> int | None:
    try:
        import psutil

        value = psutil.swap_memory().used
    except Exception:
        return None
    return value if type(value) is int and value >= 0 else None


def _sysctl(name: str) -> str | None:
    try:
        completed = subprocess.run(
            ["/usr/sbin/sysctl", "-n", name],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.decode("utf-8", errors="replace").strip()
    return value or None


def _exclusive_json(path: Path, value: dict[str, Any], mode: int) -> None:
    payload = _canonical_json(value) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _atomic_result(value: dict[str, Any]) -> None:
    if RESULT_PATH.exists() or RESULT_PATH.is_symlink():
        raise StudyError("result path already exists")
    temporary = RESULT_PATH.with_name(f".{RESULT_PATH.name}.tmp-{os.getpid()}")
    _exclusive_json(temporary, value, 0o600)
    os.chmod(temporary, 0o644)
    os.replace(temporary, RESULT_PATH)


def _require_private_directory(path: Path) -> None:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise StudyError("attempt directory cannot be inspected") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or path.resolve(strict=True) != path
    ):
        raise StudyError("attempt directory is unsafe")


def _worker_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in UNSAFE_PYTHON_ENVIRONMENT:
        environment.pop(name, None)
    environment.update(WORKER_ENVIRONMENT)
    environment["FRIDAY_PLANNER_4B_PARENT_PID"] = str(os.getpid())
    environment["FRIDAY_PLANNER_4B_RUN_ID"] = RUN_ID
    return environment


def _require_target_environment() -> None:
    """Fail before the one-shot marker unless the registered target is active."""

    if platform.machine() != EXPECTED_MACHINE:
        raise StudyError("machine architecture is not the registered arm64 target")
    if _sysctl("machdep.cpu.brand_string") != EXPECTED_CPU_BRAND:
        raise StudyError("CPU is not the registered Apple M1 Max target")
    if _sysctl("hw.memsize") != str(EXPECTED_MEMORY_BYTES):
        raise StudyError("memory size is not the registered 32 GiB target")
    for package, expected in REQUIRED_PACKAGES.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise StudyError(f"required package is unavailable: {package}") from exc
        if actual != expected:
            raise StudyError(f"required package version changed: {package}")
    import mlx.core as mx

    if str(mx.default_device()) != "Device(gpu, 0)":
        raise StudyError("MLX default device is not the registered GPU")


def _stderr_tail(stream: Any) -> str:
    try:
        stream.flush()
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(max(0, size - 4_000))
        return stream.read().decode("utf-8", errors="replace")[-4_000:]
    except Exception:
        return ""


def _decode_event(payload: bytes) -> dict[str, Any]:
    if not payload or len(payload) > MAX_EVENT_BYTES:
        raise WorkerError("worker output size is invalid")
    lines = payload.splitlines()
    if len(lines) != 1 or not lines[0]:
        raise WorkerError("worker must emit exactly one JSON event")
    try:
        value = json.loads(
            lines[0].decode("utf-8", errors="strict"),
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {item}")
            ),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise WorkerError("worker emitted invalid JSON") from exc
    if not isinstance(value, dict):
        raise WorkerError("worker event is not an object")
    if value.get("event") == "error":
        raise WorkerError(
            f"worker failed: {value.get('error_type', 'unknown')}: "
            f"{str(value.get('message', ''))[:300]}"
        )
    return value


def _validate_event(event: dict[str, Any], process_pid: int) -> dict[str, Any]:
    worker = _load_worker_module()
    expected_fields = {
        "compute_ns",
        "event",
        "finish_reason",
        "load_count",
        "mlx_peak_bytes",
        "model_id",
        "output_tokens",
        "pid",
        "prompt_sha256",
        "prompt_tokens",
        "rss_peak_bytes",
        "snapshot_revision",
        "text",
        "tokens",
    }
    tokens = event.get("tokens")
    text = event.get("text")
    if (
        set(event) != expected_fields
        or event.get("event") != "complete"
        or event.get("model_id") != MODEL_ID
        or event.get("snapshot_revision") != MODEL_REVISION
        or event.get("prompt_sha256") != worker.PROMPT_SHA256
        or event.get("load_count") != 1
        or event.get("pid") != process_pid
        or type(event.get("prompt_tokens")) is not int
        or event["prompt_tokens"] <= 0
        or type(event.get("compute_ns")) is not int
        or event["compute_ns"] <= 0
        or type(event.get("rss_peak_bytes")) is not int
        or event["rss_peak_bytes"] <= 0
        or type(event.get("mlx_peak_bytes")) is not int
        or event["mlx_peak_bytes"] <= 0
        or not isinstance(tokens, list)
        or not 1 <= len(tokens) <= MAX_OUTPUT_TOKENS
        or event.get("output_tokens") != len(tokens)
        or any(type(token) is not int for token in tokens)
        or not isinstance(text, str)
        or not text
        or len(text.encode("utf-8")) > MAX_RESPONSE_BYTES
        or event.get("finish_reason") not in {"stop", "length"}
    ):
        raise WorkerError("worker completion event is invalid")
    try:
        candidate = worker.parse_choice(text)
    except worker.WorkerError:
        candidate = None
    return {
        **event,
        "candidate_id": candidate,
        "token_sha256": _token_sha256(tokens),
    }


def _run_worker(index: int) -> dict[str, Any]:
    del index  # The subprocess receives no variable prompt or candidate input.
    started_ns = time.perf_counter_ns()
    with tempfile.TemporaryFile(mode="w+b") as stdout, tempfile.TemporaryFile(
        mode="w+b"
    ) as stderr:
        process = subprocess.Popen(
            [sys.executable, str(WORKER), "--worker"],
            cwd=PROJECT_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            env=_worker_environment(),
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=WORKER_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as exc:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
            raise WorkerError("worker timed out") from exc
        completed_ns = time.perf_counter_ns()
        stdout.seek(0, os.SEEK_END)
        size = stdout.tell()
        stdout.seek(0)
        if size > MAX_EVENT_BYTES:
            raise WorkerError("worker output exceeded the size limit")
        payload = stdout.read()
        if code != 0:
            tail = _stderr_tail(stderr)
            event_error = ""
            try:
                _decode_event(payload)
            except WorkerError as inner:
                event_error = str(inner)
            detail = "; ".join(value for value in (event_error, tail) if value)
            raise WorkerError(f"worker exited with {code}: {detail[:500]}")
        value = _validate_event(_decode_event(payload), process.pid)
        process_wall_ns = completed_ns - started_ns
        if value["compute_ns"] > process_wall_ns:
            raise WorkerError("worker compute duration exceeds parent wall duration")
        value["process_wall_ns"] = process_wall_ns
        return value


def _pace(guard: BudgetGuard, compute_ns: int) -> None:
    # The worker duration has already stopped before this function can sleep.
    seconds = compute_ns / 1_000_000_000
    if not math.isfinite(seconds) or seconds <= 0:
        raise BudgetError("worker compute duration is invalid")
    guard.record_gpu(seconds)
    required = seconds * (1.0 - PACING_TARGET) / PACING_TARGET
    for _ in range(max(1, math.ceil(required / POLICY.required_break_s))):
        guard.required_break()


def _identity_gate(runs: list[dict[str, Any]]) -> bool:
    if len(runs) != RUN_COUNT:
        return False
    first = runs[0]
    pids = [run.get("pid") for run in runs]
    return bool(
        len(pids) == len(set(pids))
        and all(type(pid) is int and pid > 0 for pid in pids)
        and all(run.get("load_count") == 1 for run in runs)
        and all(run.get("finish_reason") == "stop" for run in runs)
        and all(run.get("tokens") == first.get("tokens") for run in runs)
        and all(run.get("text") == first.get("text") for run in runs)
        and all(run.get("prompt_tokens") == first.get("prompt_tokens") for run in runs)
    )


def _contract_gate(runs: list[dict[str, Any]]) -> bool:
    allowed = _load_worker_module().ALLOWED_CANDIDATES
    return bool(
        len(runs) == RUN_COUNT
        and all(run.get("candidate_id") in allowed for run in runs)
    )


def _priority_gate(runs: list[dict[str, Any]]) -> bool:
    return bool(
        len(runs) == RUN_COUNT
        and all(run.get("candidate_id") == EXPECTED_CANDIDATE for run in runs)
    )


def _resource_summary(
    runs: list[dict[str, Any]], swap_before: int, swap_after: int | None
) -> dict[str, Any]:
    rss_values = [run.get("rss_peak_bytes") for run in runs]
    mlx_values = [run.get("mlx_peak_bytes") for run in runs]
    valid_values = bool(
        len(runs) == RUN_COUNT
        and all(type(value) is int and value > 0 for value in rss_values + mlx_values)
    )
    peak_rss = max(rss_values) if valid_values else None
    peak_mlx = max(mlx_values) if valid_values else None
    swap_delta = swap_after - swap_before if swap_after is not None else None
    gate = bool(
        valid_values
        and peak_rss is not None
        and peak_rss <= MAX_MEMORY_BYTES
        and peak_mlx is not None
        and peak_mlx <= MAX_MEMORY_BYTES
        and swap_delta is not None
        and swap_delta <= 0
    )
    return {
        "gate_passed": gate,
        "max_mlx_peak_bytes": peak_mlx,
        "max_rss_peak_bytes": peak_rss,
        "swap_after_bytes": swap_after,
        "swap_before_bytes": swap_before,
        "swap_delta_bytes": swap_delta,
    }


def decision_for(
    *, identity: bool, contract: bool, priority: bool, resources: bool, budget: bool
) -> str:
    """Apply the immutable preregistered decision table."""

    if not identity:
        return "correctness_failed"
    if not contract:
        return "planner_contract_failed"
    if not priority:
        return "planner_priority_failed"
    if not resources or not budget:
        return "resource_or_budget_failed"
    return "planner_4b_qualified_exact_case"


def _provenance(revision: str, power_source: str, snapshot: Any) -> dict[str, Any]:
    code_files = {
        path.relative_to(PROJECT_ROOT).as_posix(): _sha256(path)
        for path in (Path(__file__), WORKER, PREREGISTRATION)
    }
    packages: dict[str, str | None] = {}
    for name in ("mlx", "mlx-lm", "numpy", "psutil"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "code_files": code_files,
        "code_sha256": hashlib.sha256(_canonical_json(code_files)).hexdigest(),
        "environment": {
            "executable": str(Path(sys.executable).resolve()),
            "packages": packages,
            "python": platform.python_version(),
            "worker": dict(WORKER_ENVIRONMENT),
        },
        "git_revision": revision,
        "hardware": {
            "cpu_brand": _sysctl("machdep.cpu.brand_string"),
            "machine": platform.machine(),
            "macos": platform.mac_ver()[0],
            "memory_bytes": _sysctl("hw.memsize"),
            "model": _sysctl("hw.model"),
        },
        "model": snapshot.report_identity(),
        "power_source": power_source,
        "preregistration_sha256": FROZEN_PREREGISTRATION_SHA256,
    }


def _preflight(run_id: str) -> tuple[str, str, Any, int]:
    if run_id != RUN_ID:
        raise StudyError("run id is not registered")
    expected_python = (PROJECT_ROOT / ".venv" / "bin" / "python").resolve(strict=True)
    if Path(sys.executable).resolve() != expected_python:
        raise StudyError("study must use the project virtual environment")
    if _sha256(PREREGISTRATION) != FROZEN_PREREGISTRATION_SHA256:
        raise StudyError("preregistration changed")
    if RESULT_PATH.exists() or RESULT_PATH.is_symlink():
        raise StudyError("result already exists")
    if ATTEMPT_PATH.exists() or ATTEMPT_PATH.is_symlink():
        raise StudyError("the one-shot hardware attempt was already started")
    revision = _require_clean_worktree()
    _require_target_environment()
    power_source = require_ac_power()
    snapshot = resolve_local_model_snapshot(MODEL_ID)
    if snapshot.revision != MODEL_REVISION:
        raise StudyError("model revision changed")
    swap_before = _swap_used_bytes()
    if swap_before is None:
        raise StudyError("swap usage is unavailable")
    return revision, power_source, snapshot, swap_before


def execute(run_id: str) -> dict[str, Any]:
    revision, power_source, snapshot, swap_before = _preflight(run_id)
    ATTEMPT_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    _require_private_directory(ATTEMPT_DIR)
    provenance = _provenance(revision, power_source, snapshot)
    _exclusive_json(
        ATTEMPT_PATH,
        {
            "formal_claim": False,
            "provenance": provenance,
            "run_id": run_id,
            "started_at_unix_ns": time.time_ns(),
            "study_id": STUDY_ID,
        },
        0o600,
    )

    guard = BudgetGuard(POLICY)
    state: dict[str, Any] = {
        "decision": "hardware_run_failed",
        "formal_claim": False,
        "provenance": provenance,
        "run_id": run_id,
        "runs": [],
        "schema_version": 1,
        "study_id": STUDY_ID,
    }
    error: dict[str, str] | None = None
    try:
        guard.before_candidate()
        for index in range(RUN_COUNT):
            value = _run_worker(index)
            state["runs"].append(value)
            _pace(guard, value["compute_ns"])
    except Exception as exc:
        error = {"message": str(exc)[:500], "type": type(exc).__name__}
    finally:
        try:
            guard.finish_candidate()
        except Exception as exc:
            if error is None:
                error = {"message": str(exc)[:500], "type": type(exc).__name__}

    runs = state["runs"]
    identity_gate = _identity_gate(runs)
    contract_gate = _contract_gate(runs)
    priority_gate = _priority_gate(runs)
    swap_after = _swap_used_bytes()
    resources = _resource_summary(runs, swap_before, swap_after)
    resource_gate = bool(resources["gate_passed"])
    budget = guard.summary()
    budget_gate = bool(
        error is None
        and budget["duty_cycle_limit"] == 0.15
        and budget["gpu_work_seconds"] <= POLICY.gpu_work_limit_s
        and budget["max_continuous_gpu_seconds"] <= POLICY.continuous_gpu_limit_s
        and budget["wall_seconds"] <= POLICY.wall_limit_s
    )
    if error is None:
        decision = decision_for(
            identity=identity_gate,
            contract=contract_gate,
            priority=priority_gate,
            resources=resource_gate,
            budget=budget_gate,
        )
    else:
        decision = (
            "resource_or_budget_failed"
            if error["type"] == "BudgetError"
            else "hardware_run_failed"
        )
    state.update(
        {
            "budget": budget,
            "completed_at_unix_ns": time.time_ns(),
            "decision": decision,
            "error": error,
            "gates": {
                "H1_greedy_identity": identity_gate,
                "H2_response_contract": contract_gate,
                "H3_expected_priority": priority_gate,
                "H4_resources": resource_gate,
                "H5_budget": budget_gate,
            },
            "metrics": {
                "candidate_ids": [run.get("candidate_id") for run in runs],
                "compute_seconds": [run["compute_ns"] / 1_000_000_000 for run in runs],
                "process_wall_seconds": [
                    run["process_wall_ns"] / 1_000_000_000 for run in runs
                ],
                "prompt_tokens": [run.get("prompt_tokens") for run in runs],
                "runs_completed": len(runs),
                "token_hashes": [run.get("token_sha256") for run in runs],
            },
            "resources": resources,
            "thresholds": {
                "max_memory_bytes": MAX_MEMORY_BYTES,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
                "required_runs": RUN_COUNT,
            },
        }
    )
    _atomic_result(state)
    return state


def _self_check() -> int:
    if _sha256(PREREGISTRATION) != FROZEN_PREREGISTRATION_SHA256:
        raise StudyError("preregistration hash mismatch")
    assert decision_for(
        identity=False, contract=True, priority=True, resources=True, budget=True
    ) == "correctness_failed"
    assert decision_for(
        identity=True, contract=False, priority=True, resources=True, budget=True
    ) == "planner_contract_failed"
    assert decision_for(
        identity=True, contract=True, priority=False, resources=True, budget=True
    ) == "planner_priority_failed"
    assert decision_for(
        identity=True, contract=True, priority=True, resources=False, budget=True
    ) == "resource_or_budget_failed"
    assert decision_for(
        identity=True, contract=True, priority=True, resources=True, budget=False
    ) == "resource_or_budget_failed"
    assert decision_for(
        identity=True, contract=True, priority=True, resources=True, budget=True
    ) == "planner_4b_qualified_exact_case"
    assert POLICY.duty_cycle_limit == 0.15
    assert PACING_TARGET < POLICY.duty_cycle_limit
    print(json.dumps({"checks": 9, "self_check": "pass"}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="measure_planner_4b", allow_abbrev=False)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--self-check", action="store_true")
    modes.add_argument("--show", action="store_true")
    parser.add_argument("--run-id", default=RUN_ID)
    args = parser.parse_args(argv)
    if args.self_check:
        return _self_check()
    if args.show:
        if not RESULT_PATH.is_file() or RESULT_PATH.is_symlink():
            raise SystemExit("result is unavailable")
        print(RESULT_PATH.read_text(encoding="utf-8"), end="")
        return 0
    if not args.execute:
        print(json.dumps({"state": "not_released", "required_flag": "--execute"}))
        return 78
    try:
        report = execute(args.run_id)
    except StudyError as exc:
        print(json.dumps({"error": str(exc), "state": "not_started"}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "formal_claim": False,
                "result": RESULT_PATH.relative_to(PROJECT_ROOT).as_posix(),
                "run_id": report["run_id"],
                "runs_completed": report["metrics"]["runs_completed"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["decision"] == "planner_4b_qualified_exact_case" else 1


if __name__ == "__main__":
    raise SystemExit(main())
