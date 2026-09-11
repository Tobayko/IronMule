"""Parent-side, non-accelerator supervisor for local DATA1 workers.

The supervisor deliberately owns only readiness, leases, process lifetime and
resource observation.  The allowed child invokes the portable CLI directly in
``FRIDAY_PORTABLE_DIRECT_WORKER`` mode, so its PID (not an intermediate Python
parent) is the PID monitored by :class:`LoadMemoryGuard`.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
import fcntl
import importlib.util
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import threading
import time
import uuid
from typing import Any, Iterator

from friday_evidence.canonical import canonical_sha256
from friday_evidence.events import EventJournal, EventJournalError
from ironmule_product.memory import LoadMemoryGuard, MemoryGuardError
from ironmule_product.readiness import ReadinessWindow, probe

from .contracts import code_digest, file_sha256, write_json_new


_ALLOWED_MODULES = frozenset({"friday_evidence.portable.runner", "friday_evidence.portable.capture"})
_MAX_READY_SECONDS = 30.0
_POLL_SECONDS = 1.0
_LOG_LIMIT_BYTES = 1 * 1024 * 1024
_THREAD_HOLD = threading.Lock()


class SupervisorError(RuntimeError):
    """A local portable worker cannot be safely supervised."""


def _private_directory(path: Path) -> Path:
    path = Path(path)
    if path.is_symlink():
        raise SupervisorError("unsafe_directory")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise SupervisorError("unsafe_directory")
    return path.resolve(strict=True)


def _module_hash(module: str) -> str:
    if module not in _ALLOWED_MODULES:
        raise SupervisorError("module_not_allowed")
    found = importlib.util.find_spec(module)
    origin = None if found is None else found.origin
    if not isinstance(origin, str) or not origin.endswith(".py"):
        raise SupervisorError("module_unavailable")
    path = Path(origin)
    package = Path(__file__).resolve().parent
    try:
        path.resolve(strict=True).relative_to(package)
    except (OSError, ValueError) as exc:
        raise SupervisorError("module_outside_portable_bundle") from exc
    return file_sha256(path)


@contextmanager
def _lease(path: Path) -> Iterator[None]:
    """Acquire the persistent cross-process DATA1 hold without waiting."""
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    held = False
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise SupervisorError("unsafe_data1_lock")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            held = True
        except BlockingIOError as exc:
            raise SupervisorError("data1_lock_busy") from exc
        yield
    finally:
        if held:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _wait_ready(journal: EventJournal, run_id: str) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    window = ReadinessWindow()
    observations: list[dict[str, Any]] = []
    deadline = time.monotonic() + _MAX_READY_SECONDS
    while time.monotonic() < deadline:
        snapshot = probe()
        decision = window.observe(snapshot)
        compact = {
            "eligible": decision["eligible"], "stable": decision["stable"],
            "consecutive": decision["consecutive"], "reasons": list(decision["reasons"]),
            "observed_unix_ns": snapshot.get("observed_unix_ns"),
            "observed": {key: snapshot.get(key) for key in (
                "load_ratio", "load_1m", "cpu_count", "memory_free_percent",
                "memory_total_bytes", "swap_used_bytes", "power_source",
                "low_power_mode", "thermal_state", "errors")},
        }
        observations.append(compact)
        journal.append(run_id, "readiness", compact)
        if decision["stable"]:
            return snapshot, observations
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(5.0, remaining))
    return None, observations


def _open_log(output_dir: Path, run_id: str) -> tuple[Path, Any]:
    path = output_dir / f"{run_id}.worker.log"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    return path, os.fdopen(descriptor, "wb", buffering=0)


def _terminate_group(process: subprocess.Popen[bytes]) -> dict[str, Any]:
    """Stop only the process group created for this child, then verify exit."""
    telemetry: dict[str, Any] = {"pid": process.pid, "term_sent": False, "kill_sent": False, "group_gone": False}
    def group_gone() -> bool:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        return False

    def await_group_exit(seconds: float) -> bool:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if group_gone():
                return True
            time.sleep(0.05)
        return group_gone()

    if process.poll() is not None:
        if not group_gone():
            try:
                os.killpg(process.pid, signal.SIGKILL)
                telemetry["kill_sent"] = True
            except ProcessLookupError:
                pass
        telemetry["group_gone"] = await_group_exit(5)
        return telemetry
    try:
        os.killpg(process.pid, signal.SIGTERM)
        telemetry["term_sent"] = True
    except ProcessLookupError:
        telemetry["group_gone"] = True
        return telemetry
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
            telemetry["kill_sent"] = True
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            telemetry["group_gone"] = False
            return telemetry
    if not group_gone():
        try:
            os.killpg(process.pid, signal.SIGKILL)
            telemetry["kill_sent"] = True
        except ProcessLookupError:
            pass
    telemetry["group_gone"] = await_group_exit(5)
    return telemetry


def _report(
    *, run_id: str, module: str, status: str, code_before: dict[str, str],
    telemetry: dict[str, Any], error_code: str | None = None,
) -> dict[str, Any]:
    try:
        code_after = {"bundle": code_digest(), "module": _module_hash(module)} if module in _ALLOWED_MODULES else dict(code_before)
    except (OSError, ValueError, SupervisorError):
        code_after = {"bundle": "unavailable", "module": "unavailable"}
    value: dict[str, Any] = {
        "schema": "ironmule.portable-supervisor.v1", "run_id": run_id, "module": module,
        "status": status, "code_before": code_before, "code_after": code_after,
        "code_unchanged": code_before == code_after, "telemetry": telemetry,
        "performance_claim": False,
    }
    if error_code is not None:
        value["error_code"] = error_code
    if status == "finished" and code_before != code_after:
        value["status"] = "failed"
        value["error_code"] = "code_changed_during_worker"
    return value


def run_local_worker(
    module: str, argv: list[str], output_dir: Path, *, timeout_seconds: int,
    state_dir: Path, product_state_dir: Path | None = None,
) -> dict[str, Any]:
    """Run one approved portable CLI module under readiness/resource supervision.

    No shell is used and the supervisor never imports an accelerator framework.
    A result's status is based only on observed readiness, resource checks, and
    child process exit—not on any report emitted by the child.
    """
    run_id = uuid.uuid4().hex
    if not isinstance(argv, list) or any(not isinstance(item, str) or "\x00" in item for item in argv):
        raise SupervisorError("argv_invalid")
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 900:
        raise SupervisorError("timeout_invalid")
    output = _private_directory(Path(output_dir))
    state = _private_directory(Path(state_dir))
    supervisor_path = output / "supervisor.json"
    if supervisor_path.exists():
        raise SupervisorError("supervisor_output_exists")
    telemetry: dict[str, Any] = {"readiness": [], "resource_samples": [], "worker": None}
    code_before: dict[str, str]
    try:
        code_before = {"bundle": code_digest(), "module": _module_hash(module)}
        if argv.count("--output-dir") != 1:
            raise SupervisorError("worker_output_flag_required")
        position = argv.index("--output-dir")
        if position + 1 >= len(argv):
            raise SupervisorError("worker_output_path_required")
        child_output = Path(argv[position + 1])
        if child_output.is_symlink() or not child_output.resolve().is_relative_to(output.resolve()):
            raise SupervisorError("worker_output_outside_supervision")
    except SupervisorError as exc:
        report = _report(run_id=run_id, module=module, status="rejected", code_before={}, telemetry=telemetry, error_code=str(exc))
        write_json_new(supervisor_path, report)
        return report
    if not _THREAD_HOLD.acquire(blocking=False):
        report = _report(run_id=run_id, module=module, status="deferred", code_before=code_before, telemetry=telemetry, error_code="data1_thread_busy")
        write_json_new(supervisor_path, report)
        return report
    try:
        try:
            with ExitStack() as stack:
                stack.enter_context(_lease(state / ".data1.lock"))
                with EventJournal(state / "supervisor-events.sqlite3") as journal:
                    journal.append(run_id, "run_started", {"module": module, "timeout_seconds": timeout_seconds})
                    ready, observations = _wait_ready(journal, run_id)
                    telemetry["readiness"] = observations
                    if ready is None:
                        journal.append(run_id, "deferred", {"reason": "readiness_not_stable"})
                        report = _report(run_id=run_id, module=module, status="deferred", code_before=code_before, telemetry=telemetry, error_code="readiness_not_stable")
                        write_json_new(supervisor_path, report)
                        return report
                    memory_total = ready.get("memory_total_bytes")
                    swap_baseline = ready.get("swap_used_bytes")
                    if type(memory_total) is not int or type(swap_baseline) is not int:
                        raise SupervisorError("readiness_memory_missing")
                    guard = LoadMemoryGuard(swap_baseline_bytes=swap_baseline, memory_total_bytes=memory_total)
                    if product_state_dir is not None:
                        from ironmule_product.calibration import model_lease
                        from ironmule_product.state import ProductStore
                        store = ProductStore(Path(product_state_dir))
                        # model_lease calls settings() and therefore refuses an
                        # uninitialized/unsafe state instead of creating one.
                        stack.enter_context(model_lease(store))
                    log_path, log_handle = _open_log(output, run_id)
                    child_env = {**os.environ, "FRIDAY_PORTABLE_DIRECT_WORKER": "1"}
                    command = [sys.executable, "-m", module, *argv]
                    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log_handle, stderr=log_handle, env=child_env, start_new_session=True)
                    telemetry["worker"] = {"pid": process.pid, "log": log_path.name,
                        "module": module, "argument_flags": [arg for arg in argv if arg.startswith("--")]}
                    journal.append(run_id, "worker_started", {"pid": process.pid, "module": module})
                    started = time.monotonic()
                    failure: str | None = None
                    termination: dict[str, Any] | None = None
                    try:
                        while process.poll() is None:
                            if time.monotonic() - started >= timeout_seconds:
                                failure = "worker_timeout"
                                break
                            current = probe()
                            decision = ReadinessWindow().observe(current)
                            if not decision["eligible"]:
                                failure = "runtime_readiness_failed"
                                break
                            try:
                                sample = guard(process.pid, force=True)
                            except MemoryGuardError as exc:
                                failure = exc.code
                                break
                            telemetry["resource_samples"].append(sample)
                            journal.append(run_id, "sample", {"pid": process.pid, "rss_bytes": sample["rss_bytes"], "swap_delta_bytes": sample["swap_delta_bytes"], "errors": sample["errors"]})
                            if log_path.stat().st_size > _LOG_LIMIT_BYTES:
                                failure = "worker_log_limit_exceeded"
                                break
                            time.sleep(_POLL_SECONDS)
                    finally:
                        if process.poll() is None:
                            termination = _terminate_group(process)
                        log_handle.close()
                    telemetry["worker"].update({"exit_code": process.returncode, "termination": termination})
                    if failure is not None:
                        journal.append(run_id, "run_finished", {"status": "failed", "reason": failure, "exit_code": process.returncode})
                        report = _report(run_id=run_id, module=module, status="failed", code_before=code_before, telemetry=telemetry, error_code=failure)
                    elif process.returncode == 0:
                        journal.append(run_id, "run_finished", {"status": "finished", "exit_code": 0})
                        report = _report(run_id=run_id, module=module, status="finished", code_before=code_before, telemetry=telemetry)
                    else:
                        journal.append(run_id, "run_finished", {"status": "failed", "reason": "worker_nonzero_exit", "exit_code": process.returncode})
                        report = _report(run_id=run_id, module=module, status="failed", code_before=code_before, telemetry=telemetry, error_code="worker_nonzero_exit")
                    write_json_new(supervisor_path, report)
                    return report
        except SupervisorError as exc:
            report = _report(run_id=run_id, module=module, status="deferred", code_before=code_before, telemetry=telemetry, error_code=str(exc))
            write_json_new(supervisor_path, report)
            return report
        except RuntimeError:
            report = _report(run_id=run_id, module=module, status="deferred", code_before=code_before, telemetry=telemetry, error_code="product_model_lease_unavailable")
            write_json_new(supervisor_path, report)
            return report
        except (EventJournalError, OSError, ValueError) as exc:
            report = _report(run_id=run_id, module=module, status="failed", code_before=code_before, telemetry=telemetry, error_code=type(exc).__name__)
            write_json_new(supervisor_path, report)
            return report
    finally:
        _THREAD_HOLD.release()


__all__ = ["SupervisorError", "run_local_worker"]
