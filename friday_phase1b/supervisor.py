"""Bounded parent watchdog for the fixed Phase-1B worker."""

from __future__ import annotations

import hashlib
import math
import os
import selectors
import shutil
import signal
import stat
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .canonical import CanonicalError, strict_json_loads
from .constants import (
    AA_MODE,
    AA_FIXTURE_SEEDS,
    AA_ORDER_SEEDS,
    AB_MODE,
    AB_FIXTURE_SEEDS,
    AB_ORDER_SEEDS,
    BASELINE_NAMES,
    CHARACTERIZE_BLOCKS,
    CHARACTERIZE_FIXTURE_SEEDS,
    CHARACTERIZE_MODE,
    CHARACTERIZE_ORDER_SEEDS,
    CLEANUP_TIMEOUT_SECONDS,
    CONFIRM_BLOCKS,
    CONTRACT_ID,
    EXPECTED_DEVICE_NAME,
    EXPECTED_MLX_VERSION,
    QUALIFICATION_CASES,
)
from .constants import (
    PROJECT_ROOT,
    RESULT_LIMIT_BYTES,
    RSS_POLL_SECONDS,
    SCHEMA_VERSION,
    STREAM_LIMIT_BYTES,
    WORKER_MODES,
    WORKER_RSS_LIMIT_BYTES,
    WORKER_TIMEOUT_SECONDS,
)
from .kernel_source import KERNEL_NAME, KERNEL_SOURCE_SHA256
from .provenance import ProvenanceError, verify_source_snapshot


class SupervisorError(RuntimeError):
    """The fixed worker could not be supervised within its closed contract."""


@dataclass(frozen=True)
class _ExecutableIdentity:
    lexical: Path
    resolved: Path
    lexical_stat: tuple[int, int, int, int, int, int]
    target_stat: tuple[int, int, int, int, int, int]


def _stat_identity(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_uid,
        info.st_size,
        info.st_mtime_ns,
    )


def _exact_dict(value: Any, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise SupervisorError(f"worker {name} keys differ")
    return value


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SupervisorError(f"worker {name} is not numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise SupervisorError(f"worker {name} is not finite and positive")
    return result


def _validate_timing(value: Any, *, arms: set[str], blocks: int) -> None:
    timing = _exact_dict(value, {"samples_ns", "orders"}, "timing")
    samples = _exact_dict(timing["samples_ns"], arms, "timing samples")
    if not all(
        isinstance(values, list)
        and len(values) == blocks
        and all(_positive_number(item, "timing sample") > 0 for item in values)
        for values in samples.values()
    ):
        raise SupervisorError("worker timing sample geometry differs")
    orders = timing["orders"]
    if not isinstance(orders, list) or len(orders) != blocks or any(
        not isinstance(order, list)
        or len(order) != len(arms)
        or set(order) != arms
        for order in orders
    ):
        raise SupervisorError("worker timing order geometry differs")
    for arm in arms:
        position_counts = [
            sum(order[position] == arm for order in orders)
            for position in range(len(arms))
        ]
        if max(position_counts) - min(position_counts) > 1:
            raise SupervisorError("worker timing orders are not position-balanced")


def _validate_passed_evidence(
    value: Any, *, mode: str, index: int, baseline: str | None
) -> None:
    if not isinstance(value, dict) or value.get("passed") is not True:
        raise SupervisorError("passed worker evidence is incomplete")
    if mode == "qualification":
        evidence = _exact_dict(
            value,
            {"compile_first_eval_ns", "cases", "memory", "gates", "passed"},
            "qualification evidence",
        )
        _positive_number(evidence["compile_first_eval_ns"], "compile-first-eval")
        cases = evidence["cases"]
        if (
            not isinstance(cases, list)
            or any(not isinstance(case, dict) for case in cases)
            or [case.get("name") for case in cases] != list(QUALIFICATION_CASES)
            or any(case.get("passed") is not True for case in cases)
        ):
            raise SupervisorError("qualification correctness matrix differs")
        memory = _exact_dict(
            evidence["memory"], {"active_bytes", "cache_bytes", "peak_bytes"}, "qualification memory"
        )
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in memory.values()
        ):
            raise SupervisorError("qualification memory metric differs")
        gates = _exact_dict(evidence["gates"], {"correctness", "mlx_peak"}, "qualification gates")
        if any(value is not True for value in gates.values()):
            raise SupervisorError("qualification resource or correctness gate failed")
        return
    common = {
        "baseline",
        "fixture_sha256",
        "fixture_seed",
        "order_seed",
        "correctness",
        "timing",
        "passed",
    }
    if mode == CHARACTERIZE_MODE:
        common.remove("baseline")
        evidence = _exact_dict(value, common, "characterization evidence")
        expected_fixture = CHARACTERIZE_FIXTURE_SEEDS[index]
        expected_order = CHARACTERIZE_ORDER_SEEDS[index]
        arms = set(BASELINE_NAMES)
        blocks = CHARACTERIZE_BLOCKS
    elif mode == AA_MODE:
        evidence = _exact_dict(value, common, "A/A evidence")
        expected_fixture = AA_FIXTURE_SEEDS[index]
        expected_order = AA_ORDER_SEEDS[index]
        arms = {"a", "b"}
        blocks = CONFIRM_BLOCKS
    else:
        common.add("memory")
        evidence = _exact_dict(value, common, "A/B evidence")
        expected_fixture = AB_FIXTURE_SEEDS[index]
        expected_order = AB_ORDER_SEEDS[index]
        arms = {"baseline", "candidate"}
        blocks = CONFIRM_BLOCKS
        memory = _exact_dict(evidence["memory"], arms, "memory")
        for arm in arms:
            values = _exact_dict(
                memory[arm], {"active_bytes", "cache_bytes", "peak_bytes"}, "memory arm"
            )
            if any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in values.values()
            ):
                raise SupervisorError("worker memory metric differs")
    if evidence.get("fixture_seed") != expected_fixture or evidence.get("order_seed") != expected_order:
        raise SupervisorError("worker fixture or order seed differs")
    if mode in {AA_MODE, AB_MODE} and evidence.get("baseline") != baseline:
        raise SupervisorError("worker evidence baseline differs")
    fixture_hash = evidence.get("fixture_sha256")
    if (
        not isinstance(fixture_hash, str)
        or len(fixture_hash) != 64
        or any(character not in "0123456789abcdef" for character in fixture_hash)
    ):
        raise SupervisorError("worker fixture hash differs")
    correctness = evidence.get("correctness")
    if not isinstance(correctness, dict) or correctness.get("passed") is not True:
        raise SupervisorError("worker correctness guard failed")
    _validate_timing(evidence["timing"], arms=arms, blocks=blocks)


def _interpreter() -> _ExecutableIdentity:
    lexical = PROJECT_ROOT / ".venv" / "bin" / "python"
    try:
        lexical_info = lexical.lstat()
        resolved = lexical.resolve(strict=True)
        target_info = resolved.lstat()
    except (OSError, RuntimeError) as exc:
        raise SupervisorError("project interpreter is unavailable") from exc
    allowed_owners = {0, os.getuid()}
    if (
        not (stat.S_ISREG(lexical_info.st_mode) or stat.S_ISLNK(lexical_info.st_mode))
        or not stat.S_ISREG(target_info.st_mode)
        or lexical_info.st_uid not in allowed_owners
        or target_info.st_uid not in allowed_owners
        or target_info.st_mode & 0o022
        or not os.access(lexical, os.X_OK)
    ):
        raise SupervisorError("project interpreter identity is unsafe")
    return _ExecutableIdentity(
        lexical=lexical,
        resolved=resolved,
        lexical_stat=_stat_identity(lexical_info),
        target_stat=_stat_identity(target_info),
    )


def _verify_interpreter(identity: _ExecutableIdentity) -> None:
    try:
        lexical_info = identity.lexical.lstat()
        resolved = identity.lexical.resolve(strict=True)
        target_info = resolved.lstat()
    except (OSError, RuntimeError) as exc:
        raise SupervisorError("project interpreter changed before spawn") from exc
    if (
        resolved != identity.resolved
        or _stat_identity(lexical_info) != identity.lexical_stat
        or _stat_identity(target_info) != identity.target_stat
    ):
        raise SupervisorError("project interpreter identity changed before spawn")


def _environment(run_dir: Path) -> dict[str, str]:
    return {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "PYTHONPATH": str(PROJECT_ROOT),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": str(run_dir / "pycache"),
        "PYTHONHASHSEED": "0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "NO_COLOR": "1",
        "LC_ALL": "C",
        "LANG": "C",
        "TMPDIR": str(run_dir),
    }


def _kill_group(pid: int, sig: signal.Signals) -> None:
    if pid <= 0:
        return
    try:
        os.killpg(pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _failure(
    *,
    code: str,
    message: str,
    started_ns: int,
    peak_rss: int | None,
    stdout: bytes,
    stderr: bytes,
    exit_code: int | None,
) -> dict[str, Any]:
    return {
        "ok": False,
        "failure": {"code": code, "message": message[:512]},
        "result": None,
        "supervisor": {
            "wall_ns": time.monotonic_ns() - started_ns,
            "rss_peak_bytes": peak_rss,
            "exit_code": exit_code,
            "stdout_bytes": len(stdout),
            "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
            "stderr_bytes": len(stderr),
            "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
            "stderr_tail": stderr.decode("utf-8", errors="replace")[-2048:],
        },
    }


def _validate_result(
    value: Any, *, mode: str, session_index: int, baseline: str | None
) -> dict[str, Any]:
    required = {
        "schema_version",
        "contract_id",
        "mode",
        "session_index",
        "baseline",
        "status",
        "source_sha256",
        "kernel_name",
        "limits",
        "device",
        "evidence",
        "error",
        "process",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise SupervisorError("worker result keys differ")
    if (
        value["schema_version"] != SCHEMA_VERSION
        or value["contract_id"] != CONTRACT_ID
        or value["mode"] != mode
        or value["session_index"] != session_index
        or value["baseline"] != baseline
        or value["source_sha256"] != KERNEL_SOURCE_SHA256
        or value["kernel_name"] != KERNEL_NAME
        or value["status"] not in {"passed", "failed"}
        or not isinstance(value["process"], dict)
    ):
        raise SupervisorError("worker result identity differs")
    if value["status"] == "passed":
        if value["error"] is not None or not isinstance(value["evidence"], dict):
            raise SupervisorError("passed worker result is incomplete")
        limits = _exact_dict(
            value["limits"],
            {
                "resource",
                "mlx_memory_bytes",
                "mlx_cache_bytes",
                "previous_mlx_memory_bytes",
                "previous_mlx_cache_bytes",
            },
            "limits",
        )
        if not isinstance(limits["resource"], dict):
            raise SupervisorError("worker resource limits differ")
        device = _exact_dict(
            value["device"],
            {"mlx_version", "metal_available", "device_info", "python", "macos"},
            "device",
        )
        if (
            device["mlx_version"] != EXPECTED_MLX_VERSION
            or device["metal_available"] is not True
            or not isinstance(device["device_info"], dict)
            or device["device_info"].get("device_name") != EXPECTED_DEVICE_NAME
        ):
            raise SupervisorError("worker device identity differs")
        _validate_passed_evidence(
            value["evidence"], mode=mode, index=session_index, baseline=baseline
        )
    elif not isinstance(value["error"], dict):
        raise SupervisorError("failed worker result lacks a bounded error")
    process = _exact_dict(
        value["process"],
        {"wall_ns", "cpu_ns", "rss_peak_bytes", "pid", "power_source"},
        "process",
    )
    for key in ("wall_ns", "cpu_ns", "rss_peak_bytes", "pid"):
        _positive_number(process[key], f"process {key}")
    return value


def run_worker(
    mode: str,
    session_index: int,
    baseline: str | None = None,
    *,
    controller_deadline_ns: int | None = None,
    expected_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if mode not in WORKER_MODES or isinstance(session_index, bool):
        raise SupervisorError("worker request is not registered")
    if (mode in {AA_MODE, AB_MODE}) != (baseline in BASELINE_NAMES):
        raise SupervisorError("worker baseline binding differs")
    if mode not in {AA_MODE, AB_MODE} and baseline is not None:
        raise SupervisorError("worker mode cannot accept a baseline")
    if (mode == "qualification" and session_index != 0) or (
        mode != "qualification" and not 0 <= session_index < 3
    ):
        raise SupervisorError("worker session index differs")
    if expected_provenance is None:
        raise SupervisorError("worker requires frozen provenance")
    try:
        verify_source_snapshot(expected_provenance)
    except ProvenanceError as exc:
        raise SupervisorError("worker source snapshot differs before spawn") from exc

    try:
        import psutil
    except ImportError as exc:
        raise SupervisorError("psutil is required for the parent RSS watchdog") from exc

    started = time.monotonic_ns()
    own_deadline = started + int(WORKER_TIMEOUT_SECONDS * 1e9)
    deadline = min(own_deadline, controller_deadline_ns or own_deadline)
    temp_root = Path(tempfile.mkdtemp(prefix="friday-phase1b-worker-"))
    run_dir = temp_root / "cwd"
    process: subprocess.Popen[bytes] | None = None
    try:
        run_dir.mkdir(mode=0o700)
        executable = _interpreter()
        argv = [
            str(executable.lexical),
            "-P",
            "-s",
            "-B",
            "-m",
            "friday_phase1b.worker",
            "--mode",
            mode,
            "--session-index",
            str(session_index),
        ]
        if baseline is not None:
            argv.extend(["--baseline", baseline])
        # Darwin has no fd-bound Popen executable API.  Double inspection narrows
        # the path-to-exec window and a post-spawn check detects replacement.
        _verify_interpreter(executable)
        process = subprocess.Popen(
            argv,
            cwd=run_dir,
            env=_environment(run_dir),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            close_fds=True,
            pass_fds=(),
            shell=False,
        )
        try:
            _verify_interpreter(executable)
        except SupervisorError:
            _kill_group(process.pid, signal.SIGKILL)
            process.wait(timeout=CLEANUP_TIMEOUT_SECONDS)
            raise
        watched = psutil.Process(process.pid)
        selector = selectors.DefaultSelector()
        streams = {"stdout": process.stdout, "stderr": process.stderr}
        for name, stream in streams.items():
            if stream is not None:
                selector.register(stream, selectors.EVENT_READ, name)
        buffers = {"stdout": bytearray(), "stderr": bytearray()}
        peak_rss: int | None = None
        failure_code: str | None = None
        failure_message = ""
        terminated_at: int | None = None
        killed = False
        while True:
            now = time.monotonic_ns()
            if now >= deadline and failure_code is None and (
                process.poll() is None or bool(selector.get_map())
            ):
                failure_code = "timeout"
                failure_message = "worker exceeded its monotonic deadline"
                _kill_group(process.pid, signal.SIGTERM)
                terminated_at = now
            if process.poll() is None:
                try:
                    rss = int(watched.memory_info().rss)
                except (psutil.Error, OSError):
                    rss = 0
                if rss > 0:
                    peak_rss = max(peak_rss or 0, rss)
                    if rss >= WORKER_RSS_LIMIT_BYTES and failure_code is None:
                        failure_code = "rss_limit"
                        failure_message = "worker exceeded the parent RSS limit"
                if failure_code is not None and terminated_at is None:
                    _kill_group(process.pid, signal.SIGTERM)
                    terminated_at = now
                if (
                    terminated_at is not None
                    and not killed
                    and now - terminated_at >= int(0.5e9)
                ):
                    _kill_group(process.pid, signal.SIGKILL)
                    killed = True
            events = selector.select(timeout=RSS_POLL_SECONDS)
            for key, _mask in events:
                stream = key.fileobj
                chunk = os.read(stream.fileno(), 65536)
                if not chunk:
                    selector.unregister(stream)
                    stream.close()
                    continue
                name = key.data
                stream_limit = RESULT_LIMIT_BYTES + 1 if name == "stdout" else STREAM_LIMIT_BYTES
                remaining = max(0, stream_limit - len(buffers[name]))
                buffers[name].extend(chunk[:remaining])
                if len(chunk) > remaining and failure_code is None:
                    failure_code = "stream_limit"
                    failure_message = f"worker {name} exceeded its byte limit"
            if process.poll() is not None and not selector.get_map():
                break
            if terminated_at is not None and now - terminated_at >= int(CLEANUP_TIMEOUT_SECONDS * 1e9):
                _kill_group(process.pid, signal.SIGKILL)
                killed = True
                if process.poll() is None:
                    failure_code = "termination_unconfirmed"
                    failure_message = "worker process-group termination was not confirmed"
                    break
        try:
            exit_code = process.wait(timeout=CLEANUP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            _kill_group(process.pid, signal.SIGKILL)
            return _failure(
                code="termination_unconfirmed",
                message="worker remained alive after SIGKILL",
                started_ns=started,
                peak_rss=peak_rss,
                stdout=bytes(buffers["stdout"][:STREAM_LIMIT_BYTES]),
                stderr=bytes(buffers["stderr"][:STREAM_LIMIT_BYTES]),
                exit_code=None,
            )
        stdout = bytes(buffers["stdout"][: RESULT_LIMIT_BYTES + 1])
        stderr = bytes(buffers["stderr"][:STREAM_LIMIT_BYTES])
        try:
            verify_source_snapshot(expected_provenance)
        except ProvenanceError as exc:
            return _failure(
                code="source_changed",
                message=str(exc),
                started_ns=started,
                peak_rss=peak_rss,
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
            )
        if failure_code is not None:
            return _failure(
                code=failure_code,
                message=failure_message,
                started_ns=started,
                peak_rss=peak_rss,
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
            )
        if exit_code not in {0, 2}:
            return _failure(
                code="worker_exit",
                message="worker exited outside the closed status contract",
                started_ns=started,
                peak_rss=peak_rss,
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
            )
        payload = stdout.strip()
        if len(payload) > RESULT_LIMIT_BYTES:
            return _failure(
                code="result_limit",
                message="worker result exceeded its byte limit",
                started_ns=started,
                peak_rss=peak_rss,
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
            )
        try:
            result = _validate_result(
                strict_json_loads(payload, maximum=RESULT_LIMIT_BYTES),
                mode=mode,
                session_index=session_index,
                baseline=baseline,
            )
        except (CanonicalError, SupervisorError) as exc:
            return _failure(
                code="invalid_result",
                message=str(exc),
                started_ns=started,
                peak_rss=peak_rss,
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
            )
        expected_exit = 0 if result["status"] == "passed" else 2
        if exit_code != expected_exit:
            return _failure(
                code="status_exit_mismatch",
                message="worker status and exit code differ",
                started_ns=started,
                peak_rss=peak_rss,
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
            )
        return {
            "ok": result["status"] == "passed",
            "failure": None if result["status"] == "passed" else result["error"],
            "result": result,
            "supervisor": {
                "wall_ns": time.monotonic_ns() - started,
                "rss_peak_bytes": peak_rss,
                "exit_code": exit_code,
                "stdout_bytes": len(stdout),
                "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
                "stderr_bytes": len(stderr),
                "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
                "stderr_tail": stderr.decode("utf-8", errors="replace")[-2048:],
            },
        }
    finally:
        if process is not None and process.poll() is None:
            _kill_group(process.pid, signal.SIGKILL)
            try:
                process.wait(timeout=CLEANUP_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                pass
        shutil.rmtree(temp_root, ignore_errors=True)
