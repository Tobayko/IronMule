#!/usr/bin/env python3
"""Prospective, one-shot study of a persistent local model process.

The parent owns timing, process lifetime, evidence, and BudgetGuard pacing.  The
worker owns only the fixed model computation and reports the first token before it
finishes the remaining fixed horizon.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import select
import stat
import statistics
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

STUDY_ID = "persistent-process-20260824-03"
RUN_ID = "persistent-process-validation-20260824-01"
MODEL_ID = "mlx-community/gemma-3-4b-it-4bit"
MODEL_REVISION = "93724907d4ed1745d2fe50baadf3b0b01a65abf2"
FROZEN_PREREGISTRATION_SHA256 = (
    "a9fa83438b7ab30fb85e8cae76a90627b908c469159141286658ac0cc7f6ad9f"
)
PREREGISTRATION = Path(__file__).with_name("PREREGISTRATION.md")
WORKER = Path(__file__).with_name("worker.py")
RESULT_PATH = Path(__file__).with_name("results.json")
ATTEMPT_DIR = PROJECT_ROOT / ".friday-data" / "persistent-process"
ATTEMPT_PATH = ATTEMPT_DIR / "attempt.json"
MAX_EVENT_BYTES = 1_000_000
EVENT_TIMEOUT_SECONDS = 90.0
PROCESS_EXIT_TIMEOUT_SECONDS = 10.0
OUTPUT_TOKENS = 32
EXPECTED_PROMPT_TOKENS = 897
MAX_WARM_RSS_BYTES = 5 * 1024**3
MAX_WARM_GROWTH_BYTES = 256 * 1024**2
AA_PAIR_LOW = 0.80
AA_PAIR_HIGH = 1.25
AA_MEDIAN_LOW = 0.90
AA_MEDIAN_HIGH = 1.10
GAIN_RATIO_MAX = 0.50
PAIR_RATIO_MAX = 0.65
PACING_TARGET = 0.12
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
    """A fixed worker violated the process protocol."""


class JsonLineReader:
    """Bounded binary line reader that cannot deadlock on TextIO buffering."""

    def __init__(self, stream: Any) -> None:
        self._fd = stream.fileno()
        self._buffer = bytearray()

    def read(self, timeout_seconds: float = EVENT_TIMEOUT_SECONDS) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        while b"\n" not in self._buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WorkerError("worker event timed out")
            ready, _, _ = select.select([self._fd], [], [], remaining)
            if not ready:
                raise WorkerError("worker event timed out")
            chunk = os.read(self._fd, 65_536)
            if not chunk:
                raise WorkerError("worker output closed early")
            self._buffer.extend(chunk)
            if len(self._buffer) > MAX_EVENT_BYTES:
                raise WorkerError("worker event exceeded the size limit")
        line, _, remainder = self._buffer.partition(b"\n")
        self._buffer = bytearray(remainder)
        try:
            value = json.loads(
                line.decode("utf-8", errors="strict"),
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


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _token_sha256(tokens: list[int]) -> str:
    return hashlib.sha256(_canonical_json(tokens)).hexdigest()


def _median(values: list[float]) -> float:
    if not values or any(not math.isfinite(value) for value in values):
        raise StudyError("cannot summarize empty or non-finite measurements")
    return float(statistics.median(values))


def _mad(values: list[float]) -> float:
    center = _median(values)
    return _median([abs(value - center) for value in values])


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


def _stderr_tail(stream: Any) -> str:
    try:
        stream.flush()
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(max(0, size - 4_000))
        return stream.read().decode("utf-8", errors="replace")[-4_000:]
    except Exception:
        return ""


def _worker_environment() -> dict[str, str]:
    """Return a fixed offline child environment without Python path injection."""

    environment = os.environ.copy()
    for name in UNSAFE_PYTHON_ENVIRONMENT:
        environment.pop(name, None)
    environment.update(WORKER_ENVIRONMENT)
    return environment


class ManagedWorker:
    """Own exactly one fixed child and terminate only that child on failure."""

    def __init__(self, *, server: bool, prompt_key: str | None = None, request_id: str = "") -> None:
        arguments = [sys.executable, str(WORKER)]
        if server:
            arguments.append("--server")
        else:
            if prompt_key is None:
                raise StudyError("cold worker requires a prompt key")
            arguments.extend(["--once", prompt_key, "--request-id", request_id])
        self.server = server
        self._stderr = tempfile.TemporaryFile(mode="w+b")
        self.process = subprocess.Popen(
            arguments,
            cwd=PROJECT_ROOT,
            stdin=subprocess.PIPE if server else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            bufsize=0,
            env=_worker_environment(),
            start_new_session=True,
        )
        if self.process.stdout is None:
            raise WorkerError("worker stdout is unavailable")
        self.reader = JsonLineReader(self.process.stdout)
        self.ready: dict[str, Any] | None = None

    def read(self) -> dict[str, Any]:
        try:
            return self.reader.read()
        except WorkerError as exc:
            tail = _stderr_tail(self._stderr)
            if tail:
                raise WorkerError(f"{exc}; stderr: {tail}") from exc
            raise

    def read_ready(self) -> dict[str, Any]:
        event = self.read()
        expected = {
            "event",
            "load_count",
            "model_id",
            "pid",
            "prompt_tokens",
            "snapshot_revision",
        }
        if set(event) != expected or event.get("event") != "ready":
            raise WorkerError("worker did not emit the exact ready event")
        if (
            event.get("load_count") != 1
            or event.get("model_id") != MODEL_ID
            or event.get("snapshot_revision") != MODEL_REVISION
            or event.get("prompt_tokens") != {key: EXPECTED_PROMPT_TOKENS for key in "PQRS"}
            or type(event.get("pid")) is not int
            or event["pid"] <= 0
        ):
            raise WorkerError("worker ready identity is invalid")
        self.ready = event
        return event

    def send(self, value: dict[str, Any]) -> None:
        if not self.server or self.process.stdin is None:
            raise WorkerError("cannot send to a one-shot worker")
        payload = _canonical_json(value) + b"\n"
        try:
            self.process.stdin.write(payload)
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise WorkerError("worker input closed early") from exc

    def require_exit(self) -> None:
        try:
            code = self.process.wait(timeout=PROCESS_EXIT_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as exc:
            self.abort()
            raise WorkerError("one-shot worker did not exit") from exc
        if code != 0:
            raise WorkerError(f"worker exited with {code}: {_stderr_tail(self._stderr)}")

    def shutdown(self) -> None:
        if not self.server:
            return
        code = self.process.poll()
        if code is not None:
            raise WorkerError(
                f"server worker exited before shutdown with {code}: "
                f"{_stderr_tail(self._stderr)}"
            )
        self.send({"command": "shutdown"})
        event = self.read()
        if (
            set(event) != {"event", "pid", "requests"}
            or event.get("event") != "stopped"
            or event.get("pid") != self.process.pid
        ):
            raise WorkerError("worker did not acknowledge shutdown")
        self.require_exit()

    def abort(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)

    def close(self) -> None:
        self.abort()
        if self.process.stdout is not None:
            self.process.stdout.close()
        if self.process.stdin is not None:
            self.process.stdin.close()
        self._stderr.close()

    def __enter__(self) -> "ManagedWorker":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            if exc_type is None and self.server:
                self.shutdown()
        finally:
            self.close()


def _validate_answer_events(
    worker: ManagedWorker,
    *,
    request_id: str,
    prompt_key: str,
    started_ns: int,
) -> dict[str, Any]:
    first = worker.read()
    first_received_ns = time.perf_counter_ns()
    if (
        set(first) != {"event", "first_compute_ns", "request_id", "token_id"}
        or first.get("event") != "first_token"
        or first.get("request_id") != request_id
        or type(first.get("token_id")) is not int
        or type(first.get("first_compute_ns")) is not int
        or first["first_compute_ns"] <= 0
    ):
        raise WorkerError("first-token event is invalid")
    complete = worker.read()
    completed_ns = time.perf_counter_ns()
    expected_complete = {
        "cache_instances",
        "compute_ns",
        "event",
        "load_count",
        "mlx_peak_bytes",
        "pid",
        "prompt_key",
        "prompt_tokens",
        "request_count",
        "request_id",
        "rss_peak_bytes",
        "tokens",
    }
    tokens = complete.get("tokens")
    if (
        set(complete) != expected_complete
        or complete.get("event") != "complete"
        or complete.get("request_id") != request_id
        or complete.get("prompt_key") != prompt_key
        or complete.get("prompt_tokens") != EXPECTED_PROMPT_TOKENS
        or complete.get("cache_instances") != 1
        or complete.get("load_count") != 1
        or complete.get("pid") != worker.process.pid
        or type(complete.get("compute_ns")) is not int
        or complete["compute_ns"] <= 0
        or first["first_compute_ns"] > complete["compute_ns"]
        or type(complete.get("request_count")) is not int
        or complete["request_count"] <= 0
        or type(complete.get("rss_peak_bytes")) is not int
        or complete["rss_peak_bytes"] <= 0
        or type(complete.get("mlx_peak_bytes")) is not int
        or complete["mlx_peak_bytes"] <= 0
        or not isinstance(tokens, list)
        or len(tokens) != OUTPUT_TOKENS
        or any(type(token) is not int for token in tokens)
        or tokens[0] != first["token_id"]
    ):
        raise WorkerError("completion event is invalid")
    return {
        "compute_ns": complete["compute_ns"],
        "first_compute_ns": first["first_compute_ns"],
        "mlx_peak_bytes": complete["mlx_peak_bytes"],
        "pid": complete["pid"],
        "prompt_key": prompt_key,
        "prompt_tokens": complete["prompt_tokens"],
        "request_count": complete["request_count"],
        "request_id": request_id,
        "rss_peak_bytes": complete["rss_peak_bytes"],
        "token_sha256": _token_sha256(tokens),
        "tokens": tokens,
        "total_wall_ns": completed_ns - started_ns,
        "ttft_ns": first_received_ns - started_ns,
    }


def _run_cold(prompt_key: str, request_id: str) -> dict[str, Any]:
    started_ns = time.perf_counter_ns()
    with ManagedWorker(server=False, prompt_key=prompt_key, request_id=request_id) as worker:
        ready = worker.read_ready()
        result = _validate_answer_events(
            worker,
            request_id=request_id,
            prompt_key=prompt_key,
            started_ns=started_ns,
        )
        worker.require_exit()
        result["load_count"] = ready["load_count"]
        result["mode"] = "cold"
        return result


def _run_warm(worker: ManagedWorker, prompt_key: str, request_id: str) -> dict[str, Any]:
    started_ns = time.perf_counter_ns()
    worker.send({"command": "request", "request_id": request_id, "prompt_key": prompt_key})
    result = _validate_answer_events(
        worker,
        request_id=request_id,
        prompt_key=prompt_key,
        started_ns=started_ns,
    )
    result["load_count"] = worker.ready["load_count"] if worker.ready else None
    result["mode"] = "warm"
    return result


def _pace(guard: BudgetGuard, compute_ns: int) -> None:
    seconds = compute_ns / 1_000_000_000
    if not math.isfinite(seconds) or seconds <= 0:
        raise BudgetError("worker compute duration is invalid")
    guard.record_gpu(seconds)
    required = seconds * (1.0 - PACING_TARGET) / PACING_TARGET
    for _ in range(max(4, math.ceil(required / POLICY.required_break_s))):
        guard.required_break()


def _record_and_pace(guard: BudgetGuard, value: dict[str, Any]) -> dict[str, Any]:
    # The worker and parent durations are complete before BudgetGuard sleeps.
    _pace(guard, value["compute_ns"])
    return value


def _calibration(guard: BudgetGuard, state: dict[str, Any]) -> bool:
    calibration: dict[str, Any] = {
        "gate_passed": False,
        "median_ratio": None,
        "pairs": [],
        "ratios": [],
    }
    state["calibration"] = calibration
    guard.before_candidate()
    pairs: list[dict[str, Any]] = calibration["pairs"]
    try:
        for index, prompt_key in enumerate(("P", "Q")):
            first = _record_and_pace(
                guard, _run_cold(prompt_key, f"aa-{index}-first")
            )
            second = _record_and_pace(
                guard, _run_cold(prompt_key, f"aa-{index}-second")
            )
            state["cold_pids"].extend([first["pid"], second["pid"]])
            token_identical = first["tokens"] == second["tokens"]
            ratio = second["ttft_ns"] / first["ttft_ns"]
            pairs.append(
                {
                    "first": first,
                    "prompt_key": prompt_key,
                    "ratio": ratio,
                    "second": second,
                    "token_identical": token_identical,
                }
            )
            if not token_identical:
                break
    finally:
        guard.finish_candidate()
    ratios = [pair["ratio"] for pair in pairs]
    gate = (
        len(pairs) == 2
        and all(pair["token_identical"] for pair in pairs)
        and all(AA_PAIR_LOW <= ratio <= AA_PAIR_HIGH for ratio in ratios)
        and AA_MEDIAN_LOW <= _median(ratios) <= AA_MEDIAN_HIGH
    )
    calibration.update(
        {
            "gate_passed": gate,
            "median_ratio": _median(ratios) if ratios else None,
            "ratios": ratios,
        }
    )
    return gate


def _phase_gate(pairs: list[dict[str, Any]]) -> tuple[bool, dict[str, Any]]:
    ratios = [pair["ratio"] for pair in pairs]
    token_identical = len(pairs) == 3 and all(pair["token_identical"] for pair in pairs)
    median_ratio = _median(ratios) if ratios else None
    gate = bool(
        token_identical
        and median_ratio is not None
        and median_ratio <= GAIN_RATIO_MAX
        and all(ratio <= PAIR_RATIO_MAX for ratio in ratios)
    )
    return gate, {
        "gate_passed": gate,
        "median_ratio": median_ratio,
        "ratio_mad": _mad(ratios) if ratios else None,
        "ratios": ratios,
        "token_identical": token_identical,
    }


def _run_phase(
    guard: BudgetGuard,
    state: dict[str, Any],
    *,
    name: str,
    prompts: tuple[str, str, str],
    orders: tuple[str, str, str],
) -> bool:
    phase: dict[str, Any] = {
        "gate_passed": False,
        "median_ratio": None,
        "pairs": [],
        "ratio_mad": None,
        "ratios": [],
        "ready": None,
        "token_identical": False,
        "warmup": None,
    }
    state[name] = phase
    guard.before_candidate()
    pairs: list[dict[str, Any]] = phase["pairs"]
    try:
        with ManagedWorker(server=True) as worker:
            ready = worker.read_ready()
            phase["ready"] = ready
            state["warm_pids"].append(ready["pid"])
            warmup = _record_and_pace(
                guard, _run_warm(worker, "S", f"{name}-warmup")
            )
            phase["warmup"] = warmup
            for index, (prompt_key, order) in enumerate(zip(prompts, orders, strict=True)):
                arms: dict[str, dict[str, Any]] = {}
                for arm in order:
                    request_id = f"{name}-{index}-{arm.lower()}"
                    if arm == "A":
                        value = _run_cold(prompt_key, request_id)
                        state["cold_pids"].append(value["pid"])
                        arms["cold"] = _record_and_pace(guard, value)
                    elif arm == "B":
                        arms["warm"] = _record_and_pace(
                            guard, _run_warm(worker, prompt_key, request_id)
                        )
                    else:
                        raise StudyError("phase order is invalid")
                cold = arms["cold"]
                warm = arms["warm"]
                token_identical = cold["tokens"] == warm["tokens"]
                pair = {
                    "cold": cold,
                    "order": order,
                    "prompt_key": prompt_key,
                    "ratio": warm["ttft_ns"] / cold["ttft_ns"],
                    "token_identical": token_identical,
                    "warm": warm,
                }
                pairs.append(pair)
                if not token_identical:
                    break
            gate, summary = _phase_gate(pairs)
            phase.update(summary)
            return gate
    finally:
        guard.finish_candidate()


def decision_for(
    *,
    calibration: bool,
    correctness: bool,
    characterization: bool | None,
    validation: bool | None,
    resources: bool,
    budget: bool,
) -> str:
    """Apply the immutable preregistered decision table."""

    if not calibration:
        return "calibration_failed"
    if not correctness:
        return "correctness_failed"
    if characterization is not True:
        return "candidate_characterized_no_gain"
    if validation is not True:
        return "candidate_not_confirmed"
    if not resources or not budget:
        return "resource_or_budget_failed"
    return "engineering_gain_confirmed_exact_scope"


def _path_and_correctness(state: dict[str, Any]) -> bool:
    characterization = state.get("characterization")
    validation = state.get("validation")
    if not isinstance(characterization, dict):
        return False
    if validation is not None and not isinstance(validation, dict):
        return False
    phases = [characterization]
    if isinstance(validation, dict):
        phases.append(validation)
    if any(
        not isinstance(phase.get("pairs"), list)
        or len(phase["pairs"]) != 3
        or not isinstance(phase.get("ready"), dict)
        or not isinstance(phase.get("warmup"), dict)
        for phase in phases
    ):
        return False
    pairs = [pair for phase in phases for pair in phase["pairs"]]
    cold_pids = state.get("cold_pids")
    warm_pids = state.get("warm_pids")
    if not isinstance(cold_pids, list) or not isinstance(warm_pids, list):
        return False
    expected_cold_pids = 4 + 3 * len(phases)
    ready_pids = [phase["ready"].get("pid") for phase in phases]
    return bool(
        all(isinstance(pair, dict) and pair.get("token_identical") is True for pair in pairs)
        and len(cold_pids) == expected_cold_pids
        and len(cold_pids) == len(set(cold_pids))
        and len(warm_pids) == len(phases)
        and len(warm_pids) == len(set(warm_pids))
        and warm_pids == ready_pids
        and not set(cold_pids).intersection(warm_pids)
        and all(phase["ready"].get("load_count") == 1 for phase in phases)
        and all(
            phase["warmup"].get("load_count") == 1
            and phase["warmup"].get("pid") == phase["ready"].get("pid")
            and phase["warmup"].get("request_count") == 1
            for phase in phases
        )
        and all(
            isinstance(pair.get(arm), dict)
            and pair[arm].get("load_count") == 1
            for pair in pairs
            for arm in ("cold", "warm")
        )
        and all(pair["cold"].get("request_count") == 1 for pair in pairs)
        and all(
            pair["warm"].get("pid") == phase["ready"].get("pid")
            and pair["warm"].get("request_count") == index + 2
            for phase in phases
            for index, pair in enumerate(phase["pairs"])
        )
    )


def _resource_summary(state: dict[str, Any], swap_before: int, swap_after: int | None) -> dict[str, Any]:
    phases = [
        phase
        for phase in (state.get("characterization"), state.get("validation"))
        if isinstance(phase, dict)
    ]
    warm_peaks: list[int] = []
    growths: list[int] = []
    for phase in phases:
        warmup = phase.get("warmup")
        pairs = phase.get("pairs")
        if not isinstance(warmup, dict) or not isinstance(pairs, list):
            continue
        warmup_rss = warmup.get("rss_peak_bytes")
        if type(warmup_rss) is not int or warmup_rss <= 0:
            continue
        measured = [
            pair["warm"]["rss_peak_bytes"]
            for pair in pairs
            if isinstance(pair, dict)
            and isinstance(pair.get("warm"), dict)
            and type(pair["warm"].get("rss_peak_bytes")) is int
            and pair["warm"]["rss_peak_bytes"] > 0
        ]
        if measured:
            warm_peaks.extend(measured)
            growths.append(max(0, max(measured) - warmup_rss))
    swap_delta = swap_after - swap_before if swap_after is not None else None
    peak = max(warm_peaks) if warm_peaks else None
    growth = max(growths) if growths else None
    gate = bool(
        len(phases) == 2
        and peak is not None
        and peak <= MAX_WARM_RSS_BYTES
        and growth is not None
        and growth <= MAX_WARM_GROWTH_BYTES
        and swap_delta is not None
        and swap_delta <= 0
    )
    return {
        "gate_passed": gate,
        "swap_after_bytes": swap_after,
        "swap_before_bytes": swap_before,
        "swap_delta_bytes": swap_delta,
        "warm_peak_rss_bytes": peak,
        "warm_rss_growth_bytes": growth,
    }


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
    power_source = require_ac_power()
    snapshot = resolve_local_model_snapshot(MODEL_ID)
    if snapshot.revision != MODEL_REVISION:
        raise StudyError("model revision changed")
    swap_before = _swap_used_bytes()
    if swap_before is None:
        raise StudyError("swap usage is unavailable")
    return revision, power_source, snapshot, swap_before


def _require_private_directory(path: Path) -> None:
    """Reject redirects, foreign ownership, and unexpectedly broad permissions."""

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
        "calibration": None,
        "characterization": None,
        "cold_pids": [],
        "decision": "hardware_run_failed",
        "formal_claim": False,
        "provenance": provenance,
        "run_id": run_id,
        "schema_version": 1,
        "study_id": STUDY_ID,
        "validation": None,
        "warm_pids": [],
    }
    calibration_gate = False
    characterization_gate: bool | None = None
    validation_gate: bool | None = None
    correctness_gate = False
    resource_gate = False
    budget_gate = False
    error: dict[str, str] | None = None
    try:
        calibration_gate = _calibration(guard, state)
        if calibration_gate:
            characterization_gate = _run_phase(
                guard,
                state,
                name="characterization",
                prompts=("P", "Q", "R"),
                orders=("AB", "BA", "AB"),
            )
        if calibration_gate and characterization_gate:
            validation_gate = _run_phase(
                guard,
                state,
                name="validation",
                prompts=("R", "P", "Q"),
                orders=("BA", "AB", "BA"),
            )
        correctness_gate = _path_and_correctness(state)
    except Exception as exc:
        error = {"message": str(exc)[:500], "type": type(exc).__name__}

    swap_after = _swap_used_bytes()
    resources = _resource_summary(state, swap_before, swap_after)
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
        state["decision"] = decision_for(
            calibration=calibration_gate,
            correctness=correctness_gate,
            characterization=characterization_gate,
            validation=validation_gate,
            resources=resource_gate,
            budget=budget_gate,
        )
    else:
        state["decision"] = (
            "resource_or_budget_failed"
            if isinstance(error.get("type"), str) and error["type"] == "BudgetError"
            else "hardware_run_failed"
        )

    measured_pairs = [
        pair
        for phase_name in ("characterization", "validation")
        if isinstance(state.get(phase_name), dict)
        for pair in state[phase_name]["pairs"]
    ]
    all_ratios = [pair["ratio"] for pair in measured_pairs]
    state.update(
        {
            "budget": budget,
            "completed_at_unix_ns": time.time_ns(),
            "error": error,
            "gates": {
                "H0_calibration": calibration_gate,
                "H1_correctness_and_path": correctness_gate,
                "H2_characterization": characterization_gate,
                "H2_validation": validation_gate,
                "H3_resources": resource_gate,
                "H4_budget": budget_gate,
            },
            "metrics": {
                "all_pair_mad": _mad(all_ratios) if all_ratios else None,
                "all_pair_median_ratio": _median(all_ratios) if all_ratios else None,
                "effect_percent": (
                    100.0 * (_median(all_ratios) - 1.0) if all_ratios else None
                ),
                "pairs_measured": len(all_ratios),
            },
            "resources": resources,
            "thresholds": {
                "aa_median_high": AA_MEDIAN_HIGH,
                "aa_median_low": AA_MEDIAN_LOW,
                "aa_pair_high": AA_PAIR_HIGH,
                "aa_pair_low": AA_PAIR_LOW,
                "gain_ratio_max": GAIN_RATIO_MAX,
                "max_warm_growth_bytes": MAX_WARM_GROWTH_BYTES,
                "max_warm_rss_bytes": MAX_WARM_RSS_BYTES,
                "pair_ratio_max": PAIR_RATIO_MAX,
            },
        }
    )
    _atomic_result(state)
    return state


def _self_check() -> int:
    if _sha256(PREREGISTRATION) != FROZEN_PREREGISTRATION_SHA256:
        raise StudyError("preregistration hash mismatch")
    assert decision_for(
        calibration=False,
        correctness=True,
        characterization=True,
        validation=True,
        resources=True,
        budget=True,
    ) == "calibration_failed"
    assert decision_for(
        calibration=True,
        correctness=False,
        characterization=True,
        validation=True,
        resources=True,
        budget=True,
    ) == "correctness_failed"
    assert decision_for(
        calibration=True,
        correctness=True,
        characterization=False,
        validation=None,
        resources=True,
        budget=True,
    ) == "candidate_characterized_no_gain"
    assert decision_for(
        calibration=True,
        correctness=True,
        characterization=True,
        validation=False,
        resources=True,
        budget=True,
    ) == "candidate_not_confirmed"
    assert decision_for(
        calibration=True,
        correctness=True,
        characterization=True,
        validation=True,
        resources=False,
        budget=True,
    ) == "resource_or_budget_failed"
    assert decision_for(
        calibration=True,
        correctness=True,
        characterization=True,
        validation=True,
        resources=True,
        budget=True,
    ) == "engineering_gain_confirmed_exact_scope"
    assert _median([0.31, 0.33, 0.35]) == 0.33
    assert math.isclose(_mad([0.31, 0.33, 0.35]), 0.02)
    print(json.dumps({"self_check": "pass", "checks": 9}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="measure_persistent_process", allow_abbrev=False)
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
                "effect_percent": report["metrics"]["effect_percent"],
                "formal_claim": False,
                "result": RESULT_PATH.relative_to(PROJECT_ROOT).as_posix(),
                "run_id": report["run_id"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["decision"] == "engineering_gain_confirmed_exact_scope" else 1


if __name__ == "__main__":
    raise SystemExit(main())
