"""Manual, deadline-bound session state machine for the optimizer control plane.

The controller owns orchestration only.  A real runner is never constructed
here; callers must inject an adapter implementing the small protocol below.
"""

from __future__ import annotations

import time
import math
import os
import signal
import subprocess
import selectors
import threading
import hashlib
import shutil
import stat
import tempfile
import re
from collections import deque
from types import MappingProxyType
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol

from .readiness import HardwareLease, ProbeSnapshot, ReadinessDecision, ReadinessGate, ReadinessPolicy, check_readiness
from .canonical import loads_strict


class SessionError(RuntimeError):
    pass


class InvalidTransition(SessionError):
    pass


class SessionState(str, Enum):
    REQUESTED = "requested"
    WAITING = "waiting"
    CALIBRATING = "calibrating"
    TESTING = "testing"
    QUALIFIED = "qualified"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"
    ACTIVATION_PENDING = "activation_pending"
    CANARY = "canary"
    ACTIVE = "active"
    ROLLBACK_LATCHED = "rollback_latched"
    BASELINE = "baseline"
    CANCELLED = "cancelled"
    ACTIVATION_UNCERTAIN = "activation_uncertain"
    MANUAL_RECOVERY_REQUIRED = "manual_recovery_required"


@dataclass(frozen=True)
class TransitionAudit:
    sequence: int
    from_state: SessionState | None
    to_state: SessionState
    at: float
    reason: str

    @property
    def old_state(self) -> SessionState | None:
        return self.from_state


@dataclass(frozen=True)
class AdapterResult:
    """Bounded, untrusted result supplied by an injected adapter."""

    outcome: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    reason: str = ""

    ALLOWED_OUTCOMES = frozenset({
        "ok", "pass", "qualified", "rejected", "fail", "failed",
        "inconclusive", "activated", "healthy", "cancelled", "timeout", "error",
    })

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, str) or self.outcome not in self.ALLOWED_OUTCOMES:
            raise ValueError("adapter outcome is not allowed")
        if not isinstance(self.reason, str) or len(self.reason) > 512:
            raise ValueError("adapter reason is invalid")
        frozen = _freeze_value(self.payload, depth=0)
        if not isinstance(frozen, MappingProxyType):
            raise TypeError("adapter payload must be a mapping")
        object.__setattr__(self, "payload", frozen)


def _freeze_value(value: Any, *, depth: int) -> Any:
    if depth > 6:
        raise ValueError("adapter payload too deep")
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str) and len(value) > 4096:
            raise ValueError("adapter payload string too long")
        if isinstance(value, int) and not isinstance(value, bool) and abs(value) > 10**18:
            raise ValueError("adapter payload integer out of range")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("adapter payload number is not finite")
        return value
    if isinstance(value, Mapping):
        if len(value) > 128 or any(not isinstance(key, str) or len(key) > 128 for key in value):
            raise ValueError("adapter payload mapping is unbounded")
        return MappingProxyType({key: _freeze_value(item, depth=depth + 1) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        if len(value) > 128:
            raise ValueError("adapter payload sequence is unbounded")
        return tuple(_freeze_value(item, depth=depth + 1) for item in value)
    raise TypeError("adapter payload contains a non-canonical type")


@dataclass(frozen=True)
class PromotionAuthorization:
    """Separate, explicit authorization required before activation."""

    approved: bool
    session_id: str
    nonce: str
    fingerprint: str
    issued_at: float
    expires_at: float
    scope: str = "optimizer"

    def __post_init__(self) -> None:
        if not isinstance(self.approved, bool) or not isinstance(self.session_id, str) or not self.session_id or len(self.session_id) > 256 or not isinstance(self.nonce, str) or not self.nonce or len(self.nonce) > 256 or not isinstance(self.fingerprint, str) or not self.fingerprint or len(self.fingerprint) > 512:
            raise ValueError("invalid promotion authorization")
        if not isinstance(self.scope, str) or self.scope != "optimizer":
            raise ValueError("invalid promotion scope")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in (self.issued_at, self.expires_at)) or self.expires_at <= self.issued_at:
            raise ValueError("invalid promotion authorization window")

    def valid(self, *, now: float, fingerprint: str, session_id: str) -> bool:
        return (
            self.approved is True and self.scope == "optimizer" and
            self.session_id == session_id and self.fingerprint == fingerprint and self.issued_at <= now < self.expires_at
        )


_USED_PROMOTION_NONCES: set[str] = set()
_USED_PROMOTION_NONCE_ORDER: deque[str] = deque()
_PROMOTION_NONCE_LOCK = threading.Lock()


class PromotionGate(Protocol):
    def authorize(self, *, session_id: str, deadline: float) -> PromotionAuthorization: ...


class ProfileContract(Protocol):
    def validate_activation(self, *, profile_id: str, fingerprint: str, session_id: str) -> bool: ...
    def activate(self, profile_id: str, *, fingerprint: str, session_id: str, expected_version: int) -> Any: ...
    def rollback(self, *, reason: str, expected_version: int) -> Any: ...
    def current_version(self) -> int: ...


@dataclass(frozen=True)
class StageSpec:
    executable: str
    args: tuple[str, ...] = ()
    cwd: str | None = None
    env: Mapping[str, str] | None = None
    execute_authorized: bool = False
    stage: str = ""
    candidate_id: str = "baseline"
    parameters: Mapping[str, Any] = field(default_factory=dict)
    authorization_session_id: str | None = None
    authorization_nonce: str | None = None
    authorization_tag: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.executable, str) or not os.path.isabs(self.executable) or len(self.executable) > 4096:
            raise ValueError("stage executable must be an absolute bounded path")
        if not isinstance(self.args, tuple) or len(self.args) > 128 or any(not isinstance(item, str) or len(item) > 4096 for item in self.args):
            raise ValueError("stage arguments are invalid")
        if self.cwd is not None and (not isinstance(self.cwd, str) or not os.path.isabs(self.cwd) or len(self.cwd) > 4096 or not os.path.isdir(self.cwd) or os.path.islink(self.cwd)):
            raise ValueError("stage cwd must be absolute")
        if self.env is not None and (not isinstance(self.env, Mapping) or len(self.env) > 128 or any(not isinstance(k, str) or not isinstance(v, str) or len(k) > 256 or len(v) > 4096 for k, v in self.env.items())):
            raise ValueError("stage environment is invalid")
        if type(self.execute_authorized) is not bool:
            raise TypeError("execute_authorized must be bool")
        if not isinstance(self.stage, str) or len(self.stage) > 128 or (self.stage and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", self.stage)):
            raise ValueError("stage name is invalid")
        if not isinstance(self.candidate_id, str) or not self.candidate_id or len(self.candidate_id) > 256 or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/+-]*", self.candidate_id):
            raise ValueError("candidate id is invalid")
        if not isinstance(self.parameters, Mapping) or len(self.parameters) > 128:
            raise TypeError("stage parameters are invalid")
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))
        for name in ("authorization_session_id", "authorization_nonce", "authorization_tag"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value or len(value) > 512):
                raise ValueError(f"{name} is invalid")

    @property
    def argv(self) -> tuple[str, ...]:
        return (self.executable,) + self.args


class StageRunner(Protocol):
    verified: bool
    def run(self, spec: StageSpec, *, deadline: float) -> AdapterResult: ...


class StageAuthorizationGate(Protocol):
    """Concrete gate that verifies and consumes an adapter-issued stage token."""

    def verify_and_consume_authorization(self, spec: StageSpec, session_id: str) -> bool: ...


class VerifiedStageRunner:
    """Marker base: only explicitly injected verified runners are trusted."""
    verified = True


#: Copying an Apple-signed binary out of these prefixes and running the copy is
#: killed by macOS code signing enforcement (verified: cp /usr/bin/python3 then
#: running the copy gives rc=137). Staging one can therefore never succeed.
SYSTEM_EXECUTABLE_PREFIXES = ("/usr/bin/", "/usr/sbin/", "/usr/libexec/", "/bin/", "/sbin/", "/System/")


class SubprocessStageRunner(VerifiedStageRunner):
    """Run a bounded stage with hard process cleanup, never a shell."""

    def __init__(self, *, allowlisted_executables: Mapping[str, str] | None = None, allowed_cwd_root: str | None = None, allowed_env: Mapping[str, str] | None = None, fixed_env: Mapping[str, str] | None = None, max_output_bytes: int = 64 * 1024, clock: Any = None) -> None:
        if not 1024 <= max_output_bytes <= 4 * 1024 * 1024:
            raise ValueError("invalid stage output bound")
        self.max_output_bytes = max_output_bytes
        self.clock = clock
        self.allowlisted_executables = dict(allowlisted_executables or {})
        if any(not isinstance(path, str) or not os.path.isabs(path) or not isinstance(digest, str) or len(digest) != 64 for path, digest in self.allowlisted_executables.items()):
            raise ValueError("executable allowlist is invalid")
        self.allowed_cwd_root = os.path.abspath(allowed_cwd_root) if allowed_cwd_root is not None else None
        if self.allowed_cwd_root is not None and (not os.path.isdir(self.allowed_cwd_root) or os.path.islink(self.allowed_cwd_root)):
            raise ValueError("allowed cwd root is invalid")
        self._cwd_identity = os.stat(self.allowed_cwd_root) if self.allowed_cwd_root else None
        self.allowed_env = dict(allowed_env or {})
        self.fixed_env = dict(fixed_env or {})
        if any(not isinstance(k, str) or not isinstance(v, str) for k, v in tuple(self.allowed_env.items()) + tuple(self.fixed_env.items())) or not set(self.fixed_env).issubset(self.allowed_env):
            raise ValueError("environment policy is invalid")

    @staticmethod
    def _file_sha256(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    def _stage_executable(self, path: str, expected: str) -> tuple[str, str, int]:
        """Copy bytes verified from an open inode into a private immutable dir."""
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0))
        directory = tempfile.mkdtemp(prefix="friday-stage-")
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or not (info.st_mode & 0o111):
                raise SessionError("stage executable is not executable regular file")
            digest = hashlib.sha256()
            os.lseek(fd, 0, os.SEEK_SET)
            staged = os.path.join(directory, "runner")
            with open(staged, "wb", buffering=0) as output:
                while True:
                    chunk = os.read(fd, 1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            if digest.hexdigest() != expected:
                raise SessionError("stage executable identity changed")
            after = os.fstat(fd)
            if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                raise SessionError("stage executable changed during copy")
            os.chmod(staged, 0o500)
            staged_info = os.stat(staged)
            if staged_info.st_mode & 0o077 or not stat.S_ISREG(staged_info.st_mode):
                raise SessionError("staged executable permissions invalid")
            directory_fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            staged_fd = os.open(staged, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
            return staged, directory, staged_fd
        except Exception:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        finally:
            os.close(fd)

    def _validate_spec(self, spec: StageSpec) -> None:
        expected = self.allowlisted_executables.get(spec.executable)
        if expected is None:
            raise SessionError("stage executable is not allowlisted")
        if not os.path.isfile(spec.executable) or os.path.islink(spec.executable) or not os.access(spec.executable, os.X_OK):
            raise SessionError("stage executable is not a stable executable")
        # This runner executes a *copy* of the allowlisted binary, and macOS
        # kills the copy of an Apple-signed system binary with SIGKILL. The
        # failure only surfaces as exit:-9 after the process has started, so
        # without this check a scarce approved measurement block is spent on a
        # cryptic error. A real session always stages the project interpreter.
        if os.path.realpath(spec.executable).startswith(SYSTEM_EXECUTABLE_PREFIXES):
            raise SessionError("stage executable is a system binary and cannot be staged")
        if self._file_sha256(spec.executable) != expected:
            raise SessionError("stage executable identity changed")
        if spec.cwd is not None:
            if self.allowed_cwd_root is None:
                raise SessionError("stage cwd is not allowlisted")
            root = os.path.realpath(self.allowed_cwd_root)
            cwd = os.path.realpath(spec.cwd)
            if os.path.commonpath((root, cwd)) != root:
                raise SessionError("stage cwd is outside allowlisted root")
            if self._cwd_identity is None or os.stat(self.allowed_cwd_root).st_ino != self._cwd_identity.st_ino or os.stat(self.allowed_cwd_root).st_dev != self._cwd_identity.st_dev:
                raise SessionError("allowed cwd identity changed")
        if spec.env is None:
            raise SessionError("stage environment must be explicitly fixed")
        if spec.env is not None:
            if set(spec.env) - set(self.allowed_env):
                raise SessionError("stage environment key is not allowlisted")
            for key, value in self.allowed_env.items():
                if spec.env.get(key) != value:
                    raise SessionError("stage environment value is not fixed")

    def run(self, spec: StageSpec | tuple[str, ...], *, deadline: float) -> AdapterResult:
        if isinstance(spec, tuple):
            if not spec or any(not isinstance(item, str) or not item for item in spec):
                raise ValueError("stage argv must be non-empty strings")
            spec = StageSpec(spec[0], tuple(spec[1:]))
        if not isinstance(spec, StageSpec):
            raise TypeError("stage runner requires StageSpec")
        if spec.execute_authorized is not True:
            raise SessionError("explicit stage execution authorization is required")
        if not spec.authorization_session_id or not spec.authorization_nonce or not spec.authorization_tag:
            raise SessionError("cryptographic stage authorization is required")
        self._validate_spec(spec)
        now = self.clock() if callable(self.clock) else time.monotonic()
        remaining = deadline - now
        if remaining <= 0:
            return AdapterResult("timeout", reason="deadline")
        cwd_identity = os.stat(spec.cwd) if spec.cwd is not None else None
        # Re-hash immediately before exec as a TOCTOU guard; a changed binary
        # is rejected even when its path remains the same.
        self._validate_spec(spec)
        staged_executable, stage_directory, staged_fd = self._stage_executable(spec.executable, self.allowlisted_executables[spec.executable])
        argv = (staged_executable,) + spec.args
        try:
            process = subprocess.Popen(list(argv), shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=spec.cwd, env=dict(spec.env), start_new_session=True, pass_fds=(staged_fd,))
        except Exception:
            os.close(staged_fd)
            shutil.rmtree(stage_directory, ignore_errors=True)
            raise
        os.close(staged_fd)
        if spec.cwd is not None:
            after_identity = os.stat(spec.cwd)
            if (cwd_identity.st_dev, cwd_identity.st_ino) != (after_identity.st_dev, after_identity.st_ino):
                _terminate_process(process)
                shutil.rmtree(stage_directory, ignore_errors=True)
                return AdapterResult("error", reason="cwd_identity_changed")
        assert process.stdout is not None and process.stderr is not None
        for stream in (process.stdout, process.stderr):
            os.set_blocking(stream.fileno(), False)
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        streams: dict[str, bytearray] = {"stdout": bytearray(), "stderr": bytearray()}
        combined = 0
        try:
            while selector.get_map():
                now = self.clock() if callable(self.clock) else time.monotonic()
                remaining = deadline - now
                if remaining <= 0:
                    _terminate_process(process)
                    return AdapterResult("timeout", reason="deadline")
                events = selector.select(min(remaining, 0.1))
                if not events and process.poll() is not None:
                    break
                for key, _ in events:
                    try:
                        chunk = os.read(key.fileobj.fileno(), 8192)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    streams[key.data].extend(chunk)
                    combined += len(chunk)
                    if len(streams[key.data]) > self.max_output_bytes or combined > self.max_output_bytes * 2:
                        _terminate_process(process)
                        return AdapterResult("error", reason="output_truncated")
            now = self.clock() if callable(self.clock) else time.monotonic()
            process.wait(timeout=max(0.0, deadline - now))
        except subprocess.TimeoutExpired:
            _terminate_process(process)
            return AdapterResult("timeout", reason="deadline")
        finally:
            selector.close()
            for stream in (process.stdout, process.stderr):
                stream.close()
            shutil.rmtree(stage_directory, ignore_errors=True)
        if process.returncode != 0:
            return AdapterResult("error", reason="exit:" + str(process.returncode))
        raw_stdout = bytes(streams["stdout"]).strip()
        if raw_stdout.startswith(b"{"):
            try:
                decoded = loads_strict(raw_stdout, max_bytes=min(self.max_output_bytes, 64 * 1024))
                if isinstance(decoded, dict) and isinstance(decoded.get("outcome"), str):
                    return AdapterResult(decoded["outcome"], decoded.get("payload", {}), decoded.get("reason", ""))
            except (ValueError, TypeError):
                return AdapterResult("error", reason="invalid_stage_result")
        return AdapterResult("ok", payload={"stdout_bytes": len(streams["stdout"]), "stderr_bytes": len(streams["stderr"])})


def _terminate_process(process: subprocess.Popen[Any]) -> None:
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=0.5)
    except (OSError, subprocess.TimeoutExpired):
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait(timeout=0.5)
        except (OSError, subprocess.TimeoutExpired):
            pass
    finally:
        # communicate() in the caller may have been interrupted by timeout;
        # close any remaining pipe explicitly after the process is dead.
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass


class OptimizerAdapter(Protocol):
    def calibrate(self, *, deadline: float, session_id: str) -> AdapterResult: ...
    def test(self, *, deadline: float, session_id: str) -> AdapterResult: ...
    def canary(self, *, deadline: float, session_id: str) -> AdapterResult: ...
    def activate(self, *, deadline: float, session_id: str) -> AdapterResult: ...
    def rollback(self, *, deadline: float, session_id: str) -> AdapterResult: ...
    def deactivate(self, *, deadline: float, session_id: str) -> AdapterResult: ...


def _now(clock: Any) -> float:
    if clock is None:
        return time.monotonic()
    if callable(clock):
        return float(clock())
    method = getattr(clock, "monotonic", None) or getattr(clock, "now", None)
    if method is None:
        raise TypeError("clock must be callable or expose monotonic()/now()")
    return float(method())


class SessionController:
    """Drive one explicitly requested run, never exceeding its total deadline."""

    _TERMINAL = frozenset({
        SessionState.QUALIFIED, SessionState.REJECTED, SessionState.INCONCLUSIVE,
        SessionState.ACTIVE, SessionState.BASELINE, SessionState.ROLLBACK_LATCHED,
        SessionState.CANCELLED, SessionState.ACTIVATION_UNCERTAIN,
        SessionState.MANUAL_RECOVERY_REQUIRED,
    })

    def __init__(
        self,
        *,
        probe: Any = None,
        lease: HardwareLease,
        adapter: OptimizerAdapter | None = None,
        readiness_policy: ReadinessPolicy | None = None,
        readiness: ReadinessGate | None = None,
        sleeper: Any = None,
        clock: Any = None,
        session_id: str = "session",
        auto_activate: bool = False,
        promotion_authorization: PromotionAuthorization | None = None,
        promotion_gate: PromotionGate | None = None,
        profile_contract: ProfileContract | None = None,
        stage_runner: StageRunner | None = None,
        stage_specs: Mapping[str, StageSpec] | None = None,
        stage_authorization_gate: StageAuthorizationGate | None = None,
    ) -> None:
        if stage_runner is None and adapter is not None:
            raise SessionError("in-process adapter is test-only; use an approved stage runner")
        if stage_runner is not None and type(stage_runner) is not SubprocessStageRunner:
            raise SessionError("stage runner is not the controlled implementation")
        self.probe = probe
        self.readiness = readiness
        self.lease = lease
        self.adapter = adapter
        self.readiness_policy = readiness_policy or ReadinessPolicy()
        self.sleeper = sleeper
        self.clock = clock
        self.session_id = session_id
        self.auto_activate = auto_activate
        self.promotion_authorization = promotion_authorization
        self.promotion_gate = promotion_gate
        self.profile_contract = profile_contract
        if stage_runner is None:
            raise SessionError("a controlled stage runner is required")
        self.stage_runner = stage_runner
        self.stage_specs = dict(stage_specs or {})
        if stage_authorization_gate is not None and not callable(getattr(stage_authorization_gate, "verify_and_consume_authorization", None)):
            raise SessionError("a concrete stage authorization gate is required")
        self.stage_authorization_gate = stage_authorization_gate
        if auto_activate or promotion_authorization is not None or promotion_gate is not None:
            required_stages = {"activate", "canary", "rollback", "deactivate"}
            if any(not isinstance(self.stage_specs.get(name), StageSpec) for name in required_stages):
                raise SessionError("promotion requires activate/canary/rollback/deactivate stage specs")
        self.release_error: str | None = None
        self.audit_errors: list[str] = []
        self.profile_id: str | None = None
        self.profile_fingerprint: str | None = None
        self.profile_version: int | None = None
        self._profile_activated = False
        self._runtime_activated = False
        self._runtime_activation_attempted = False
        self.state: SessionState | None = None
        self.deadline: float | None = None
        self.duration_minutes: int | None = None
        self.audit: list[TransitionAudit] = []
        self.cancel_requested = False
        self.last_decision: ReadinessDecision | None = None
        self.result: AdapterResult | None = None

    @property
    def transitions(self) -> tuple[TransitionAudit, ...]:
        return tuple(self.audit)

    @property
    def no_recommendation(self) -> bool:
        return self.state in {
            SessionState.BASELINE,
            SessionState.REJECTED,
            SessionState.INCONCLUSIVE,
            SessionState.CANCELLED,
            SessionState.ROLLBACK_LATCHED,
            SessionState.ACTIVATION_UNCERTAIN,
            SessionState.MANUAL_RECOVERY_REQUIRED,
        }

    def request(self, duration_minutes: int | float, *, user_started: bool = False) -> None:
        if not user_started:
            raise SessionError("an explicit user start is required")
        if self.state is not None:
            raise SessionError("session already requested")
        if int(duration_minutes) != duration_minutes or not 5 <= duration_minutes <= 30:
            raise ValueError("duration must be an integer from 5 through 30 minutes")
        self.duration_minutes = int(duration_minutes)
        started = _now(self.clock)
        self.deadline = started + self.duration_minutes * 60
        self._transition(SessionState.REQUESTED, "explicit_user_start")

    def cancel(self, reason: str = "user_cancelled") -> None:
        self.cancel_requested = True
        if self.state not in (None, SessionState.CANCELLED, SessionState.BASELINE, SessionState.ACTIVATION_UNCERTAIN, SessionState.MANUAL_RECOVERY_REQUIRED, SessionState.ROLLBACK_LATCHED):
            self._transition(SessionState.CANCELLED, reason)

    def _transition(self, target: SessionState, reason: str) -> None:
        source = self.state
        allowed = {
            None: {SessionState.REQUESTED},
            SessionState.REQUESTED: {SessionState.WAITING, SessionState.CANCELLED, SessionState.BASELINE, SessionState.ACTIVATION_UNCERTAIN},
            SessionState.WAITING: {SessionState.CALIBRATING, SessionState.CANCELLED, SessionState.BASELINE, SessionState.ACTIVATION_UNCERTAIN},
            SessionState.CALIBRATING: {SessionState.TESTING, SessionState.INCONCLUSIVE, SessionState.BASELINE, SessionState.CANCELLED, SessionState.ACTIVATION_UNCERTAIN},
            SessionState.TESTING: {SessionState.QUALIFIED, SessionState.REJECTED, SessionState.INCONCLUSIVE, SessionState.BASELINE, SessionState.CANCELLED, SessionState.ACTIVATION_UNCERTAIN},
            SessionState.QUALIFIED: {SessionState.ACTIVATION_PENDING, SessionState.BASELINE, SessionState.CANCELLED, SessionState.ACTIVATION_UNCERTAIN, SessionState.ROLLBACK_LATCHED},
            SessionState.ACTIVATION_PENDING: {SessionState.CANARY, SessionState.BASELINE, SessionState.CANCELLED, SessionState.ACTIVATION_UNCERTAIN, SessionState.ROLLBACK_LATCHED},
            SessionState.CANARY: {SessionState.ACTIVE, SessionState.ROLLBACK_LATCHED, SessionState.BASELINE, SessionState.CANCELLED, SessionState.ACTIVATION_UNCERTAIN},
            SessionState.ROLLBACK_LATCHED: {SessionState.BASELINE},
            SessionState.REJECTED: {SessionState.BASELINE, SessionState.ACTIVATION_UNCERTAIN},
            SessionState.INCONCLUSIVE: {SessionState.BASELINE, SessionState.ACTIVATION_UNCERTAIN},
            SessionState.ACTIVE: {SessionState.ROLLBACK_LATCHED, SessionState.BASELINE, SessionState.CANCELLED, SessionState.ACTIVATION_UNCERTAIN},
            SessionState.BASELINE: set(),
            SessionState.CANCELLED: set(),
            SessionState.ACTIVATION_UNCERTAIN: set(),
            SessionState.MANUAL_RECOVERY_REQUIRED: set(),
        }
        if target not in allowed.get(source, set()):
            raise InvalidTransition(f"{source!s} -> {target!s}")
        at = _now(self.clock)
        self.state = target
        self.audit.append(TransitionAudit(len(self.audit), source, target, at, reason))

    def _deadline_or_baseline(self, stage: str) -> bool:
        if self.deadline is None or _now(self.clock) >= self.deadline:
            if self.state not in {SessionState.BASELINE, SessionState.CANCELLED}:
                self._failure(stage + ":deadline", rollback=self._profile_activated or stage in {"activation", "canary"})
            return True
        if self.cancel_requested:
            if self.state not in {SessionState.BASELINE, SessionState.CANCELLED}:
                self._failure(stage + ":cancelled", rollback=self._profile_activated or stage in {"activation", "canary"})
            return True
        return False

    def _ready_checkpoint(self, stage: str, *, fail_closed: bool = True) -> bool:
        if self._deadline_or_baseline(stage):
            return False
        # The caller may use a fake lease in offline tests; it must still expose
        # the same ownership contract.  No block may run without both checks.
        if not self.lease.validate():
            if self.state not in {SessionState.BASELINE, SessionState.CANCELLED}:
                self._failure(stage + ":lease_invalid", rollback=self._profile_activated or stage in {"activation", "canary"})
            return False
        remaining = self.deadline - _now(self.clock) if self.deadline is not None else 0
        if self.readiness is not None:
            self.last_decision = self.readiness.check(deadline=self.deadline)
        else:
            self.last_decision = check_readiness(
                self.probe,
                self.readiness_policy,
                sleeper=self.sleeper,
                clock=self.clock,
                deadline=min(self.deadline or remaining, _now(self.clock) + remaining),
            )
        if not self.last_decision.ready or not self.lease.validate():
            if not fail_closed:
                return False
            if self.state not in {SessionState.BASELINE, SessionState.CANCELLED}:
                self._failure(stage + ":readiness_failed", rollback=self._profile_activated or stage in {"activation", "canary"})
            return False
        return True

    def _wait_until_ready(self) -> bool:
        """Poll while waiting; transient foreign work is not terminal."""
        while True:
            if self._deadline_or_baseline("waiting"):
                return False
            if not self.lease.validate():
                self._transition(SessionState.BASELINE, "waiting:lease_invalid")
                return False
            if self._ready_checkpoint("waiting", fail_closed=False):
                return True
            if self.deadline is None:
                return False
            remaining = self.deadline - _now(self.clock)
            if remaining <= 0:
                self._transition(SessionState.BASELINE, "waiting:deadline")
                return False
            # A bounded poll prevents a busy loop while retaining the hard
            # session deadline (waiting time is part of the selected duration).
            interval = min(max(self.readiness_policy.sample_interval_seconds, 0.25), 2.0, remaining)
            if self.sleeper is None:
                time.sleep(interval)
            elif callable(self.sleeper):
                self.sleeper(interval)
            else:
                self.sleeper.sleep(interval)

    def _call(self, name: str) -> AdapterResult:
        if self.deadline is None:
            raise SessionError("session not requested")
        if self.stage_runner is not None:
            spec = self.stage_specs.get(name)
            if spec is None:
                raise SessionError("stage spec missing: " + name)
            gate = self.stage_authorization_gate
            if gate is None:
                return AdapterResult("inconclusive", {"status": "blocked", "stage": name}, reason="stage_authorization_gate_missing")
            try:
                authorized = gate.verify_and_consume_authorization(spec, self.session_id)
            except Exception as exc:
                self.audit_errors.append("stage_authorization_error:" + type(exc).__name__)
                return AdapterResult("inconclusive", {"status": "blocked", "stage": name}, reason="stage_authorization_rejected")
            if authorized is not True:
                return AdapterResult("inconclusive", {"status": "blocked", "stage": name}, reason="stage_authorization_rejected")
            value = self.stage_runner.run(spec, deadline=self.deadline)
        else:
            raise SessionError("controlled stage runner is required")
        if not isinstance(value, AdapterResult):
            raise SessionError("adapter must return AdapterResult")
        return value

    def _failure(self, reason: str, *, rollback: bool = False) -> None:
        if rollback:
            rollback_error: Exception | None = None
            try:
                if self._runtime_activation_attempted:
                    for stage in ("rollback", "deactivate"):
                        runtime_result = self._call(stage)
                        if runtime_result.outcome not in {"ok", "pass", "healthy"}:
                            raise SessionError("runtime_" + stage + "_failed")
                if self.profile_contract is None or self.profile_id is None or not isinstance(self.profile_version, int) or isinstance(self.profile_version, bool):
                    raise SessionError("rollback CAS version unavailable")
                # Rollback always carries the post-activation CAS version.
                self.profile_contract.rollback(reason=reason, expected_version=self.profile_version)
            except Exception as exc:
                rollback_error = exc
            if rollback_error is not None:
                self.audit_errors.append(reason + ":rollback_error:" + type(rollback_error).__name__)
                if self.state not in {SessionState.ACTIVATION_UNCERTAIN, SessionState.MANUAL_RECOVERY_REQUIRED}:
                    self._transition(SessionState.ACTIVATION_UNCERTAIN, reason + ":manual_recovery_required")
                return
            if self.state not in {SessionState.ROLLBACK_LATCHED, SessionState.BASELINE, SessionState.CANCELLED}:
                self._transition(SessionState.ROLLBACK_LATCHED, reason)
                self._transition(SessionState.BASELINE, "rollback_baseline")
            return
        if self.state not in {SessionState.BASELINE, SessionState.CANCELLED}:
            self._transition(SessionState.BASELINE, reason)

    def _bind_qualified_profile(self, result: AdapterResult) -> bool:
        profile_id = result.payload.get("profile_id")
        fingerprint = result.payload.get("fingerprint")
        version = result.payload.get("profile_version")
        if not isinstance(profile_id, str) or not profile_id or len(profile_id) > 256:
            return False
        if not isinstance(fingerprint, str) or fingerprint != getattr(self.lease, "fingerprint", None):
            return False
        if isinstance(version, bool) or not isinstance(version, int) or version < 0:
            return False
        self.profile_id, self.profile_fingerprint, self.profile_version = profile_id, fingerprint, version
        return True

    def run(self, duration_minutes: int | float | None = None, *, user_started: bool = False) -> SessionState:
        if self.state is None:
            if duration_minutes is None:
                raise SessionError("duration is required")
            self.request(duration_minutes, user_started=user_started)
        elif duration_minutes is not None or user_started:
            raise SessionError("session already requested")
        if self._deadline_or_baseline("requested"):
            return self.state or SessionState.BASELINE
        self._transition(SessionState.WAITING, "awaiting_stable_readiness")
        try:
            self.lease.acquire()
        except Exception:
            self._transition(SessionState.BASELINE, "lease_unavailable")
            return self.state
        try:
            if not self._wait_until_ready():
                return self.state or SessionState.BASELINE
            self._transition(SessionState.CALIBRATING, "readiness_confirmed")
            if not self._ready_checkpoint("calibrating"):
                return self.state or SessionState.BASELINE
            calibration = self._call("calibrate")
            if self._deadline_or_baseline("calibration_block") or not self.lease.validate():
                if self.state not in {SessionState.BASELINE, SessionState.CANCELLED}:
                    self._failure("calibration_block:lease_lost")
                return self.state or SessionState.BASELINE
            if calibration.outcome.lower() not in {"ok", "qualified", "pass"}:
                self.result = calibration
                self._transition(SessionState.INCONCLUSIVE, "calibration:" + calibration.outcome)
                self._transition(SessionState.BASELINE, "calibration_no_recommendation")
                return self.state
            self._transition(SessionState.TESTING, "calibration_passed")
            if not self._ready_checkpoint("testing"):
                return self.state or SessionState.BASELINE
            result = self._call("test")
            if self._deadline_or_baseline("testing_block") or not self.lease.validate():
                if self.state not in {SessionState.BASELINE, SessionState.CANCELLED}:
                    self._failure("testing_block:lease_lost")
                return self.state or SessionState.BASELINE
            self.result = result
            outcome = result.outcome.lower()
            if outcome in {"qualified", "pass", "ok"}:
                if not self._bind_qualified_profile(result):
                    self._failure("qualified_profile_binding_invalid")
                    return self.state or SessionState.BASELINE
                self._transition(SessionState.QUALIFIED, "test_qualified")
            elif outcome in {"rejected", "fail", "failed"}:
                self._transition(SessionState.REJECTED, "test_rejected")
                self._transition(SessionState.BASELINE, "rejected_no_recommendation")
                return self.state
            else:
                self._transition(SessionState.INCONCLUSIVE, "test_inconclusive")
                self._transition(SessionState.BASELINE, "inconclusive_no_recommendation")
                return self.state
            self._transition(SessionState.ACTIVATION_PENDING, "qualified_activation_gate")
            if not self.auto_activate or not self._promotion_allowed():
                return self.state
            if not self._ready_checkpoint("activation"):
                return self.state or SessionState.BASELINE
            self._runtime_activation_attempted = True
            activation = self._call("activate")
            if self._deadline_or_baseline("activation_block") or not self.lease.validate():
                if self.state not in {SessionState.BASELINE, SessionState.CANCELLED}:
                    self._failure("activation_block:lease_lost", rollback=True)
                return self.state or SessionState.BASELINE
            if activation.outcome.lower() not in {"ok", "pass", "activated"}:
                self._failure("activation_failed", rollback=True)
                return self.state
            self._runtime_activated = True
            if self.profile_contract is None:
                self._failure("profile_contract_missing", rollback=True)
                return self.state or SessionState.BASELINE
            try:
                self.profile_contract.activate(profile_id=self.profile_id, fingerprint=self.profile_fingerprint, session_id=self.session_id, expected_version=self.profile_version)
                self._profile_activated = True
                current_version = getattr(self.profile_contract, "current_version", None)
                if callable(current_version):
                    self.profile_version = current_version()
                else:
                    self.profile_version += 1
            except Exception:
                self._failure("profile_activation_failed", rollback=True)
                return self.state or SessionState.BASELINE
            self._transition(SessionState.CANARY, "activation_started")
            if not self._ready_checkpoint("canary"):
                return self.state or SessionState.BASELINE
            canary = self._call("canary")
            if self._deadline_or_baseline("canary_block") or not self.lease.validate():
                if self.state not in {SessionState.BASELINE, SessionState.CANCELLED}:
                    self._failure("canary_block:lease_lost", rollback=True)
                return self.state or SessionState.BASELINE
            if canary.outcome.lower() in {"ok", "pass", "healthy"}:
                self._transition(SessionState.ACTIVE, "canary_passed")
            else:
                self._failure("canary_failed", rollback=True)
            return self.state
        except Exception as exc:
            if self.state not in {SessionState.BASELINE, SessionState.CANCELLED}:
                self._failure("exception:" + type(exc).__name__, rollback=self._profile_activated or self._runtime_activation_attempted)
            return self.state or SessionState.BASELINE
        finally:
            try:
                self.lease.release()
            except Exception as exc:
                self.release_error = type(exc).__name__
                self.audit_errors.append("lease_release_error:" + self.release_error)
                if self.state not in {SessionState.BASELINE, SessionState.CANCELLED}:
                    try:
                        self._failure("lease_release_error", rollback=self._profile_activated or self._runtime_activation_attempted)
                    except InvalidTransition:
                        pass

    def _promotion_allowed(self) -> bool:
        if not self.auto_activate or self.deadline is None:
            return False
        authorization = self.promotion_authorization
        if self.promotion_gate is not None:
            authorization = self.promotion_gate.authorize(session_id=self.session_id, deadline=self.deadline)
        if not isinstance(authorization, PromotionAuthorization):
            return False
        if not authorization.valid(now=_now(self.clock), fingerprint=getattr(self.lease, "fingerprint", ""), session_id=self.session_id):
            return False
        if self.profile_contract is not None:
            try:
                if self.profile_id is None or not self.profile_contract.validate_activation(profile_id=self.profile_id, fingerprint=self.lease.fingerprint, session_id=self.session_id):
                    return False
            except Exception:
                return False
        with _PROMOTION_NONCE_LOCK:
            if authorization.nonce in _USED_PROMOTION_NONCES:
                return False
            _USED_PROMOTION_NONCES.add(authorization.nonce)
            _USED_PROMOTION_NONCE_ORDER.append(authorization.nonce)
            if len(_USED_PROMOTION_NONCE_ORDER) > 4096:
                _USED_PROMOTION_NONCES.discard(_USED_PROMOTION_NONCE_ORDER.popleft())
        return True

    start = run


__all__ = [
    "AdapterResult", "InvalidTransition", "OptimizerAdapter", "ProfileContract",
    "PromotionAuthorization", "PromotionGate", "SessionController", "SessionError",
    "SessionState", "StageAuthorizationGate", "StageRunner", "StageSpec", "SubprocessStageRunner",
    "TransitionAudit", "VerifiedStageRunner",
]
