"""Stdlib-only supervisor for the fixed H0 worker entrypoint."""

from __future__ import annotations

import errno
import hashlib
import os
import selectors
import signal
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .protocol import (
    INTERNAL_CONTROL_SLEEP_ENV,
    MANIFEST_FILENAME,
    MANIFEST_SHA_ENV,
    PRODUCTION_CLEANUP_S,
    PRODUCTION_MANIFEST_BYTES,
    PRODUCTION_RESULT_BYTES,
    PRODUCTION_STDERR_BYTES,
    PRODUCTION_STDOUT_BYTES,
    PRODUCTION_TOTAL_S,
    RESULT_FILENAME,
    RSS_SAMPLE_INTERVAL_S,
    ClosedManifest,
    ProtocolError,
    ensure_directory_0700,
    fallback_result,
    read_capped_json,
    validate_result,
    write_json_atomic,
)


@dataclass(frozen=True)
class SupervisorLimits:
    """Internal-only test seam; production defaults are immutable constants."""

    total_s: float
    cleanup_s: float
    stdout_bytes: int
    stderr_bytes: int
    result_bytes: int
    manifest_bytes: int
    control_sleep_s: float
    _test_only: bool = False

    @classmethod
    def for_tests(
        cls,
        *,
        total_s: float = 0.25,
        cleanup_s: float = 0.25,
        stdout_bytes: int = 64 * 1024,
        stderr_bytes: int = 64 * 1024,
        result_bytes: int = 1 * 1024 * 1024,
        manifest_bytes: int = 64 * 1024,
        control_sleep_s: float | None = None,
    ) -> "SupervisorLimits":
        """Create a bounded test-only override; never used by production CLI."""

        if total_s <= 0 or cleanup_s <= 0 or control_sleep_s is not None and control_sleep_s <= 0:
            raise ValueError("test limits must be positive")
        return cls(
            total_s=total_s,
            cleanup_s=cleanup_s,
            stdout_bytes=stdout_bytes,
            stderr_bytes=stderr_bytes,
            result_bytes=result_bytes,
            manifest_bytes=manifest_bytes,
            control_sleep_s=control_sleep_s if control_sleep_s is not None else total_s * 2.0,
            _test_only=True,
        )


PRODUCTION_LIMITS = SupervisorLimits(
    total_s=PRODUCTION_TOTAL_S,
    cleanup_s=PRODUCTION_CLEANUP_S,
    stdout_bytes=PRODUCTION_STDOUT_BYTES,
    stderr_bytes=PRODUCTION_STDERR_BYTES,
    result_bytes=PRODUCTION_RESULT_BYTES,
    manifest_bytes=PRODUCTION_MANIFEST_BYTES,
    control_sleep_s=PRODUCTION_TOTAL_S + 1.0,
)


@dataclass(frozen=True)
class _PathIdentity:
    path: str
    device: int
    inode: int
    uid: int
    mode: int
    link_target: str | None = None


@dataclass(frozen=True)
class _ExecutableIdentity:
    lexical: str
    resolved: str
    lexical_object: _PathIdentity
    links: tuple[_PathIdentity, ...]
    target: _PathIdentity


class _DrainState:
    def __init__(self, *, stdout_limit: int, stderr_limit: int) -> None:
        self.buffers: dict[int, bytearray] = {}
        self.limits: dict[int, int] = {}
        self.names: dict[int, str] = {}
        self.eof_fds: set[int] = set()
        self.counts: dict[int, int] = {}
        self.hashes: dict[int, Any] = {}
        self.previews: dict[int, bytearray] = {}
        self.overflow_fds: set[int] = set()
        self.closed: dict[str, tuple[int, str, str, bool, bool]] = {}
        self.overflow: str | None = None
        self.stdout_limit = stdout_limit
        self.stderr_limit = stderr_limit

    def register(self, fileobj: Any, name: str) -> None:
        fd = fileobj.fileno()
        os.set_blocking(fd, False)
        self.buffers[fd] = bytearray()
        self.limits[fd] = self.stdout_limit if name == "stdout" else self.stderr_limit
        self.names[fd] = name
        self.counts[fd] = 0
        self.hashes[fd] = hashlib.sha256()
        self.previews[fd] = bytearray()

    def read(self, fd: int) -> bool:
        remaining = self.limits[fd] - len(self.buffers[fd])
        try:
            chunk = os.read(fd, max(1, min(16 * 1024, remaining + 1)))
        except BlockingIOError:
            return False
        except OSError as exc:
            if exc.errno not in {errno.EBADF, errno.EIO}:
                self.overflow = self.names[fd] + "_read_error"
            self.eof_fds.add(fd)
            return True
        if not chunk:
            self.eof_fds.add(fd)
            return True
        self.counts[fd] += len(chunk)
        self.hashes[fd].update(chunk)
        preview_remaining = 4096 - len(self.previews[fd])
        if preview_remaining > 0:
            self.previews[fd].extend(chunk[:preview_remaining])
        if len(chunk) > remaining:
            self.buffers[fd].extend(chunk[:remaining])
            self.overflow = self.names[fd] + "_overflow"
            self.overflow_fds.add(fd)
        else:
            self.buffers[fd].extend(chunk)
        return False

    def close(self, fd: int) -> None:
        if fd in self.names:
            name = self.names[fd]
            self.closed[name] = (
                self.counts[fd],
                self.hashes[fd].hexdigest(),
                bytes(self.previews[fd]).decode("utf-8", errors="replace"),
                self.counts[fd] > 4096,
                fd in self.overflow_fds,
            )
        self.buffers.pop(fd, None)
        self.limits.pop(fd, None)
        self.names.pop(fd, None)
        self.eof_fds.discard(fd)
        self.counts.pop(fd, None)
        self.hashes.pop(fd, None)
        self.previews.pop(fd, None)
        self.overflow_fds.discard(fd)

    def evidence(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name in ("stdout", "stderr"):
            fd = next((candidate for candidate, stream_name in self.names.items() if stream_name == name), None)
            if fd is None:
                count, digest, preview, truncated, overflow = self.closed.get(
                    name, (0, hashlib.sha256(b"").hexdigest(), "", False, False)
                )
                result[f"{name}_bytes"] = count
                result[f"{name}_sha256"] = digest
                result[f"{name}_preview"] = preview
                result[f"{name}_truncated"] = truncated
                result[f"{name}_overflow"] = overflow
                continue
            result[f"{name}_bytes"] = self.counts[fd]
            result[f"{name}_sha256"] = self.hashes[fd].hexdigest()
            result[f"{name}_preview"] = bytes(self.previews[fd]).decode("utf-8", errors="replace")
            result[f"{name}_truncated"] = self.counts[fd] > 4096
            result[f"{name}_overflow"] = fd in self.overflow_fds
        return result

    def text(self, name: str) -> str:
        for fd, stream_name in self.names.items():
            if stream_name == name:
                return bytes(self.buffers[fd]).decode("utf-8", errors="replace")
        return ""


def _rss_bytes(pid: int) -> int | None:
    """Best-effort Darwin RSS sample for exactly one worker PID."""

    if pid <= 0:
        return None
    try:
        completed = subprocess.run(
            ["/bin/ps", "-o", "rss=", "-p", str(pid)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=0.10,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"},
            close_fds=True,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    try:
        value = int(completed.stdout.decode("ascii", errors="strict").strip())
    except (UnicodeDecodeError, ValueError):
        return None
    return value * 1024 if value >= 0 else None


def _rss_evidence(peak: int | None, reason: str | None) -> dict[str, Any]:
    if peak is None:
        return {"rss_peak_bytes": None, "rss_missing_reason": reason or "unavailable"}
    return {"rss_peak_bytes": peak, "rss_missing_reason": None}


def _empty_stream_evidence() -> dict[str, Any]:
    digest = hashlib.sha256(b"").hexdigest()
    return {
        "stdout_bytes": 0,
        "stdout_sha256": digest,
        "stdout_preview": "",
        "stdout_truncated": False,
        "stdout_overflow": False,
        "stderr_bytes": 0,
        "stderr_sha256": digest,
        "stderr_preview": "",
        "stderr_truncated": False,
        "stderr_overflow": False,
    }


def _controlled_environment(*, manifest: ClosedManifest, test_limits: SupervisorLimits | None) -> dict[str, str]:
    project_root = str(Path(__file__).resolve().parent.parent)
    environment = {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": project_root,
        "PYTHONSAFEPATH": "1",
        "PYTHONUNBUFFERED": "1",
        "LC_ALL": "C",
        "LANG": "C",
        MANIFEST_SHA_ENV: manifest.sha256,
    }
    if test_limits is not None:
        environment[INTERNAL_CONTROL_SLEEP_ENV] = f"{test_limits.control_sleep_s:.9f}"
    return environment


def _current_uid() -> int:
    getter = getattr(os, "geteuid", None) or getattr(os, "getuid", None)
    if getter is None:
        raise ProtocolError("interpreter ownership checks are unavailable")
    return int(getter())


def _path_identity(path: Path, info: os.stat_result, *, link_target: str | None = None) -> _PathIdentity:
    return _PathIdentity(
        path=os.fspath(path),
        device=int(info.st_dev),
        inode=int(info.st_ino),
        uid=int(info.st_uid),
        mode=int(info.st_mode),
        link_target=link_target,
    )


def _identity_fields(info: os.stat_result) -> tuple[int, int, int, int]:
    return int(info.st_dev), int(info.st_ino), int(info.st_uid), int(info.st_mode)


def _resolve_executable_links(path: Path, *, allowed_owners: frozenset[int]) -> tuple[Path, tuple[_PathIdentity, ...]]:
    """Resolve every path component while binding each encountered symlink."""

    current = Path(path.anchor)
    pending = list(path.parts[1:])
    links: list[_PathIdentity] = []
    while pending:
        component = pending.pop(0)
        if component in {"", "."}:
            continue
        if component == "..":
            current = current.parent
            continue
        candidate = current / component
        try:
            before = os.lstat(candidate)
        except (OSError, ValueError) as exc:
            raise ProtocolError("interpreter symlink chain is missing or inaccessible") from exc
        if not stat.S_ISLNK(before.st_mode):
            current = candidate
            continue
        if before.st_uid not in allowed_owners:
            raise ProtocolError("interpreter symlink owner is not trusted")
        # ``Path.resolve(strict=True)`` already rejects cycles and dangling
        # links before this walk.  Do not keep a global inode set here: a
        # valid absolute/relative target can legitimately traverse the same
        # symlink again after routing through another path component.
        if len(links) >= 40:
            raise ProtocolError("interpreter symlink chain is cyclic or too deep")
        try:
            link_target = os.readlink(candidate)
            after = os.lstat(candidate)
        except (OSError, ValueError) as exc:
            raise ProtocolError("interpreter symlink changed while being inspected") from exc
        if _identity_fields(before) != _identity_fields(after) or not stat.S_ISLNK(after.st_mode):
            raise ProtocolError("interpreter symlink changed while being inspected")
        links.append(_path_identity(candidate, before, link_target=link_target))
        target = Path(link_target)
        combined = target if target.is_absolute() else current / target
        normalized = Path(os.path.normpath(os.fspath(combined)))
        if not normalized.is_absolute():
            raise ProtocolError("interpreter symlink target is not absolute after resolution")
        current = Path(normalized.anchor)
        pending = list(normalized.parts[1:]) + pending
    return current, tuple(links)


def _capture_executable_identity(lexical_value: str) -> _ExecutableIdentity:
    if not isinstance(lexical_value, str) or not lexical_value:
        raise ProtocolError("sys.executable is empty or not a string")
    lexical = Path(lexical_value)
    if not lexical.is_absolute():
        raise ProtocolError("sys.executable is not an absolute path")
    try:
        lexical_info = os.lstat(lexical)
    except (OSError, ValueError) as exc:
        raise ProtocolError("sys.executable is missing or inaccessible") from exc
    if not (stat.S_ISREG(lexical_info.st_mode) or stat.S_ISLNK(lexical_info.st_mode)):
        raise ProtocolError("sys.executable is neither a regular file nor a symlink")

    current_uid = _current_uid()
    allowed_owners = frozenset({0, current_uid})
    try:
        resolved = lexical.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ProtocolError("interpreter symlink chain is dangling or cyclic") from exc
    walked, links = _resolve_executable_links(lexical, allowed_owners=allowed_owners)
    if walked != resolved:
        raise ProtocolError("interpreter symlink resolution is inconsistent")
    if stat.S_ISLNK(lexical_info.st_mode) and lexical_info.st_uid not in allowed_owners:
        raise ProtocolError("interpreter symlink owner is not trusted")

    try:
        target_info = os.lstat(resolved)
    except (OSError, ValueError) as exc:
        raise ProtocolError("resolved interpreter target is missing") from exc
    if not stat.S_ISREG(target_info.st_mode):
        raise ProtocolError("resolved interpreter target is not a regular file")
    if target_info.st_uid not in allowed_owners:
        raise ProtocolError("resolved interpreter target owner is not trusted")
    if stat.S_IMODE(target_info.st_mode) & 0o022:
        raise ProtocolError("resolved interpreter target is group- or other-writable")
    if not target_info.st_mode & 0o111 or not os.access(resolved, os.X_OK):
        raise ProtocolError("resolved interpreter target is not executable")

    identity = _ExecutableIdentity(
        lexical=lexical_value,
        resolved=os.fspath(resolved),
        lexical_object=_path_identity(lexical, lexical_info),
        links=links,
        target=_path_identity(resolved, target_info),
    )
    try:
        lexical_after = os.lstat(lexical)
        target_after = os.lstat(resolved)
    except (OSError, ValueError) as exc:
        raise ProtocolError("interpreter identity changed while being inspected") from exc
    if (
        _identity_fields(lexical_after) != _identity_fields(lexical_info)
        or _identity_fields(target_after) != _identity_fields(target_info)
    ):
        raise ProtocolError("interpreter identity changed while being inspected")
    return identity


def _trusted_executable() -> _ExecutableIdentity:
    """Bind the selected lexical interpreter and its trusted resolved target."""

    return _capture_executable_identity(sys.executable)


def _verify_executable_identity(expected: _ExecutableIdentity) -> None:
    if _capture_executable_identity(expected.lexical) != expected:
        raise ProtocolError("interpreter identity changed before spawn")


def _kill_group(pid: int) -> None:
    if pid <= 0:
        return
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        pass


def _fallback_with_rss(
    manifest: ClosedManifest,
    *,
    status: str,
    classification: str,
    code: str,
    message: str,
    peak: int | None,
    reason: str | None,
    streams: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return fallback_result(
        manifest=manifest,
        status=status,
        classification=classification,
        code=code,
        message=message,
        evidence={**_empty_stream_evidence(), **(streams or {}), **_rss_evidence(peak, reason)},
    )


def run_supervised(manifest: ClosedManifest, limits: SupervisorLimits | None = None) -> dict[str, Any]:
    """Run only the fixed worker with bounded pipes, files, time, and fallback.

    ``manifest`` must be a :class:`ClosedManifest`; accepting an arbitrary mapping
    here would reopen the protocol boundary.  ``limits`` is only accepted from the
    internal test seam produced by :meth:`SupervisorLimits.for_tests`.
    """

    if not isinstance(manifest, ClosedManifest):
        raise TypeError("run_supervised accepts only a ClosedManifest")
    if limits is not None and not limits._test_only:
        raise TypeError("custom supervisor limits are test-seam only")
    active_limits = limits if limits is not None else PRODUCTION_LIMITS
    temp_root = Path(tempfile.mkdtemp(prefix="friday-h0-worker-"))
    run_dir = temp_root / "cwd"
    termination_unconfirmed = False
    try:
        ensure_directory_0700(run_dir)
        manifest_path = run_dir / MANIFEST_FILENAME
        result_path = run_dir / RESULT_FILENAME
        write_json_atomic(manifest_path, manifest.value, limit=active_limits.manifest_bytes)
        try:
            executable = _trusted_executable()
        except ProtocolError as exc:
            return _fallback_with_rss(
                manifest,
                status="worker_exit",
                classification="worker_exit",
                code="invalid_executable",
                message=str(exc),
                peak=None,
                reason="worker_not_started",
            )
        argv = (executable.lexical, "-P", "-s", "-B", "-m", "friday_h0.worker")
        environment = _controlled_environment(manifest=manifest, test_limits=limits)
        deadline = time.monotonic() + active_limits.total_s
        try:
            # Popen has no portable fd-bound executable API on Darwin.  This
            # immediate identity recheck narrows, but cannot eliminate, the
            # remaining path-to-exec TOCTOU interval.
            _verify_executable_identity(executable)
            process = subprocess.Popen(
                argv,
                cwd=run_dir,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                close_fds=True,
                pass_fds=(),
                shell=False,
            )
        except ProtocolError as exc:
            return _fallback_with_rss(
                manifest,
                status="worker_exit",
                classification="worker_exit",
                code="invalid_executable",
                message=str(exc),
                peak=None,
                reason="worker_not_started",
            )
        except (OSError, ValueError) as exc:
            return _fallback_with_rss(
                manifest,
                status="worker_exit",
                classification="worker_exit",
                code="spawn_failed",
                message=str(exc),
                peak=None,
                reason="worker_not_started",
            )

        selector = selectors.DefaultSelector()
        drain = _DrainState(stdout_limit=active_limits.stdout_bytes, stderr_limit=active_limits.stderr_bytes)
        for stream, name in ((process.stdout, "stdout"), (process.stderr, "stderr")):
            if stream is not None:
                drain.register(stream, name)
                selector.register(stream, selectors.EVENT_READ)
        peak_rss: int | None = None
        rss_reason: str | None = "no_sample"
        timed_out = False
        overflow = False
        cleanup_deadline: float | None = None
        next_rss = time.monotonic()
        try:
            while True:
                now = time.monotonic()
                if now >= deadline and process.poll() is None and not timed_out:
                    timed_out = True
                    _kill_group(process.pid)
                    cleanup_deadline = now + active_limits.cleanup_s
                if drain.overflow is not None and process.poll() is None and not overflow:
                    overflow = True
                    _kill_group(process.pid)
                    cleanup_deadline = now + active_limits.cleanup_s
                if now >= next_rss:
                    sample = _rss_bytes(process.pid)
                    if sample is not None:
                        peak_rss = max(peak_rss or 0, sample)
                        rss_reason = None
                    next_rss = now + RSS_SAMPLE_INTERVAL_S
                if process.poll() is not None and not selector.get_map():
                    break
                if cleanup_deadline is not None and now >= cleanup_deadline:
                    if process.poll() is None:
                        _kill_group(process.pid)
                    break
                wait_until = cleanup_deadline if cleanup_deadline is not None else deadline
                wait_for = max(0.0, min(0.05, wait_until - now))
                events = selector.select(wait_for)
                for key, _ in events:
                    fd = key.fileobj.fileno()
                    eof = drain.read(fd)
                    if drain.overflow is not None and not overflow:
                        overflow = True
                        _kill_group(process.pid)
                        cleanup_deadline = time.monotonic() + active_limits.cleanup_s
                    if eof:
                        try:
                            selector.unregister(key.fileobj)
                        except Exception:
                            pass
                        try:
                            key.fileobj.close()
                        except OSError:
                            pass
                        drain.close(fd)
                for key in list(selector.get_map().values()):
                    fd = key.fileobj.fileno()
                    if fd not in drain.buffers:
                        try:
                            selector.unregister(key.fileobj)
                        except Exception:
                            pass
                        drain.close(fd)
            if process.poll() is None:
                try:
                    process.wait(timeout=max(0.0, active_limits.cleanup_s))
                except subprocess.TimeoutExpired:
                    _kill_group(process.pid)
                    try:
                        process.wait(timeout=0.05)
                    except subprocess.TimeoutExpired:
                        termination_unconfirmed = True
            else:
                process.wait()
            if process.poll() is None:
                termination_unconfirmed = True
        finally:
            selector.close()
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
        if peak_rss is None and rss_reason == "no_sample":
            rss_reason = "unavailable"
        stream_evidence = drain.evidence()
        if termination_unconfirmed:
            return _fallback_with_rss(
                manifest,
                status="invalid",
                classification="invalid",
                code="termination_unconfirmed",
                message="worker remained alive after SIGKILL and bounded waits",
                peak=peak_rss,
                reason=rss_reason,
                streams=stream_evidence,
            )
        evidence = _rss_evidence(peak_rss, rss_reason)
        evidence.update(stream_evidence)
        if timed_out:
            return _fallback_with_rss(
                manifest,
                status="timeout",
                classification="timeout",
                code="deadline_exceeded",
                message="worker exceeded the monotonic total deadline",
                peak=peak_rss,
                reason=rss_reason,
                streams=stream_evidence,
            )
        if overflow:
            return _fallback_with_rss(
                manifest,
                status="invalid",
                classification="invalid",
                code="stream_overflow",
                message=drain.overflow or "worker stream exceeded its byte budget",
                peak=peak_rss,
                reason=rss_reason,
                streams=stream_evidence,
            )
        if process.returncode != 0:
            return _fallback_with_rss(
                manifest,
                status="worker_exit",
                classification="worker_exit",
                code=f"exit_{process.returncode}",
                message="worker exited without a promotable result",
                peak=peak_rss,
                reason=rss_reason,
                streams=stream_evidence,
            )
        try:
            result, _ = read_capped_json(result_path, limit=active_limits.result_bytes)
            validated = validate_result(result, manifest=manifest)
            validated["evidence"] = dict(validated["evidence"])
            validated["evidence"].update(evidence)
            validated = validate_result(validated, manifest=manifest)
            write_json_atomic(result_path, validated, limit=active_limits.result_bytes)
            return validated
        except ProtocolError as exc:
            return _fallback_with_rss(
                manifest,
                status="invalid",
                classification="invalid",
                code="invalid_result",
                message=str(exc),
                peak=peak_rss,
                reason=rss_reason,
                streams=stream_evidence,
            )
    finally:
        if not termination_unconfirmed:
            try:
                for directory in (run_dir, temp_root):
                    if not directory.exists():
                        continue
                    for child in directory.iterdir():
                        try:
                            child.unlink()
                        except OSError:
                            pass
                    try:
                        directory.rmdir()
                    except OSError:
                        pass
            except OSError:
                pass


__all__ = ["PRODUCTION_LIMITS", "SupervisorLimits", "run_supervised"]
