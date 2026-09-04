"""Fail-closed, offline binding to the existing IronMule command line.

This module is intentionally a *control-plane boundary*, not another tuner.  It
does not import IronMule or MLX, and it never accepts caller supplied command
flags, paths, source, or model output.  A checkout, interpreter and exact
``ExactFingerprint`` are bound once and every later operation is checked against
that binding.

The adapter can describe a future stage invocation, but execution is disabled by
default.  In particular, ``calibrate`` and ``test`` return a blocked
``AdapterResult`` unless the caller explicitly opts in.  Activation stages are
not described at all: there is no safe rollback/deactivate command in the
current IronMule CLI.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
import secrets
import sys
import sysconfig
import threading
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Mapping

from .candidates import CandidateError, CandidateRegistry
from .canonical import canonical_bytes, loads_strict
from .fingerprint import ExactFingerprint
from .session import AdapterResult, StageSpec, SubprocessStageRunner
from .ironmule_stage_worker import TUNE_SEARCH_CONTRACT_SHA256, WorkerError, validate_hub_model_id


WORKER_RELATIVE_PATH = "friday_ironmule_stage_worker.py"
SPEC_RELATIVE_PATH = "stage_spec.json"


GIT = "/usr/bin/git"
MAX_GIT_OUTPUT = 64 * 1024
MAX_RESULT_BYTES = 256 * 1024
MAX_RESULT_DEPTH = 10
MAX_RESULT_ITEMS = 10_000
MAX_WORKER_RESOURCE_BYTES = 12 * 1024**3
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HEAD = re.compile(r"^[0-9a-f]{40}$")
_EMPTY_PYTHONPATH_SHA256 = hashlib.sha256(b"").hexdigest()

# This is deliberately explicit.  The active Claude checkout is never a
# production adapter target, even when a caller forgets to pass a forbidden
# path list.  Tests can pass a different explicit list for their temporary
# repositories.
CURRENT_IRONMULE_WORKTREE = "/Users/tobiasburandt/Project_Friday/.worktrees/ironmule-b7"
DEFAULT_FORBIDDEN_CHECKOUTS = (CURRENT_IRONMULE_WORKTREE,)

# Files whose bytes affect the command and result contract.  A production
# binding must either use this complete list or provide a separately reviewed
# fixed list for a test fixture; arbitrary recursive source hashing is not used.
DEFAULT_EXECUTION_FILES = (
    "ironmule_cli.py",
    "ironmule/_version.py",
    "ironmule/__init__.py",
    "ironmule/ab.py",
    "ironmule/bench.py",
    "ironmule/benchmark.py",
    "ironmule/evidence.py",
    "ironmule/fast.py",
    "ironmule/fingerprint.py",
    "ironmule/hw.py",
    "ironmule/tune.py",
    "ironmule/service.py",
    "ironmule/runtime.py",
    "ironmule/executor.py",
    "ironmule/plans.py",
    "ironmule/model_identity.py",
    "ironmule/telemetry.py",
    "pyproject.toml",
)


def _registry_hash(files: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(files).encode("utf-8")).hexdigest()


def _purelib_binding(interpreter: str) -> tuple[str | None, tuple[int, int, int, int] | None]:
    """Return a verified venv purelib, or no override for system Python."""
    if sys.prefix == sys.base_prefix:
        return None, None
    if os.path.realpath(sys.executable) != os.path.realpath(interpreter):
        raise CheckoutValidationError("running interpreter does not match bound interpreter")
    prefix = _absolute(sys.prefix, "sys.prefix")
    purelib_raw = sysconfig.get_path("purelib")
    if not isinstance(purelib_raw, str):
        raise CheckoutValidationError("purelib path is unavailable")
    purelib = _absolute(purelib_raw, "purelib")
    _reject_symlink_ancestors(prefix)
    _reject_symlink_ancestors(purelib)
    try:
        Path(purelib).relative_to(Path(prefix))
    except ValueError as exc:
        raise CheckoutValidationError("purelib is outside the active venv") from exc
    _regular_file(os.path.join(prefix, "pyvenv.cfg"), "pyvenv.cfg")
    info = os.stat(purelib)
    if not stat.S_ISDIR(info.st_mode):
        raise CheckoutValidationError("purelib is not a directory")
    return purelib, (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


EXECUTION_FILE_REGISTRY_HASH = _registry_hash(DEFAULT_EXECUTION_FILES)

# No inherited environment is allowed.  In particular, these values prevent a
# stage from silently reaching a model hub or a proxy.  SubprocessStageRunner
# requires every key to be fixed in both its allowlist and each StageSpec.
OFFLINE_ENV = MappingProxyType({
    "PATH": "/usr/bin:/bin",
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONUNBUFFERED": "1",
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HTTP_PROXY": "",
    "HTTPS_PROXY": "",
    "ALL_PROXY": "",
    "http_proxy": "",
    "https_proxy": "",
    "all_proxy": "",
    "NO_PROXY": "*",
})


class IronMuleAdapterError(ValueError):
    """Base error for an invalid or unsafe adapter input."""


class CheckoutValidationError(IronMuleAdapterError):
    """The bound checkout/interpreter/source identity is not valid."""


class ResultValidationError(IronMuleAdapterError):
    """A stage result is malformed, stale, or insufficiently evidenced."""


class UnsupportedStage(IronMuleAdapterError):
    """The current IronMule CLI has no allowlisted operation for this stage."""


def _absolute(value: str | os.PathLike[str], field: str) -> str:
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str) or not value or not os.path.isabs(value):
        raise ValueError(f"{field} must be an absolute path")
    if len(value) > 4096 or "\x00" in value:
        raise ValueError(f"{field} is invalid")
    return os.path.abspath(value)


def _path_key(value: str) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(value)))


def _reject_symlink_ancestors(path: str, *, allow_missing: bool = False) -> None:
    """Reject links in every ancestor without resolving them first."""

    absolute = Path(_absolute(path, "path"))
    current = Path(absolute.anchor or os.sep)
    parts = absolute.parts[1:] if absolute.anchor else absolute.parts
    for index, component in enumerate(parts):
        current /= component
        try:
            info = current.lstat()
        except FileNotFoundError:
            if allow_missing and index == len(parts) - 1:
                return
            raise CheckoutValidationError(f"path component does not exist: {current}")
        # macOS exposes the system temporary directory through the stable
        # compatibility aliases /var and /tmp.  They are OS-owned aliases, not
        # caller-controlled checkout links; every descendant is still checked.
        if stat.S_ISLNK(info.st_mode) and str(current) not in {"/var", "/tmp"}:
            raise CheckoutValidationError(f"symlink path component refused: {current}")


def _regular_file(path: str, field: str, *, executable: bool = False) -> os.stat_result:
    _reject_symlink_ancestors(path)
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise CheckoutValidationError(f"{field} is not readable") from exc
    if not stat.S_ISREG(info.st_mode) or os.path.islink(path):
        raise CheckoutValidationError(f"{field} must be a regular non-symlink file")
    if executable and not (info.st_mode & 0o111):
        raise CheckoutValidationError(f"{field} is not executable")
    return info


def _file_digest(path: str, *, field: str, executable: bool = False) -> tuple[str, tuple[int, int, int, int]]:
    """Hash through one no-follow descriptor and detect replacement/TOCTOU."""

    _regular_file(path, field, executable=executable)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise CheckoutValidationError(f"cannot open {field}") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise CheckoutValidationError(f"{field} is not regular")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(fd)
        before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if before_identity != after_identity:
            raise CheckoutValidationError(f"{field} changed while being read")
        return digest.hexdigest(), after_identity
    finally:
        os.close(fd)


def _validate_relative(path: str, field: str) -> str:
    if not isinstance(path, str) or not path or len(path) > 512:
        raise ValueError(f"{field} contains an invalid path")
    candidate = Path(path)
    if candidate.is_absolute() or "\x00" in path or ".." in candidate.parts:
        raise ValueError(f"{field} must be a relative fixed path")
    return candidate.as_posix()


def _source_digest(checkout: str, files: tuple[str, ...], registry_hash: str | None = None) -> str:
    digest = hashlib.sha256()
    digest.update(b"execution-registry\0")
    digest.update((registry_hash or _registry_hash(files)).encode("ascii"))
    digest.update(b"\0")
    for relative in files:
        path = os.path.join(checkout, relative)
        file_hash, _ = _file_digest(path, field=f"execution file {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_hash))
    return digest.hexdigest()


def _compose_source_digest(files: tuple[str, ...], hashes: Mapping[str, str], registry_hash: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"execution-registry\0")
    digest.update(registry_hash.encode("ascii"))
    digest.update(b"\0")
    for relative in files:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(hashes[relative]))
    return digest.hexdigest()


def _validate_sha(value: str | None, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _git(checkout: str, args: tuple[str, ...], *, timeout: float = 5.0) -> str:
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 30:
        raise ValueError("git timeout must be between 0 and 30 seconds")
    if any(not isinstance(arg, str) or not arg or len(arg) > 4096 or "\x00" in arg for arg in args):
        raise ValueError("invalid git argument")
    try:
        completed = subprocess.run(
            [GIT, "-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null", *args],
            cwd=checkout,
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={
                "PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C",
                "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_NOGLOBAL": "1",
                "GIT_OPTIONAL_LOCKS": "0",
            },
            timeout=float(timeout),
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CheckoutValidationError("git invocation failed") from exc
    stdout = completed.stdout or b""
    stderr = completed.stderr or b""
    if len(stdout) > MAX_GIT_OUTPUT or len(stderr) > MAX_GIT_OUTPUT:
        raise CheckoutValidationError("git output exceeds bound")
    if completed.returncode != 0:
        detail = stderr.decode("utf-8", "replace")[:512]
        raise CheckoutValidationError(f"git command failed: {detail}")
    try:
        return stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CheckoutValidationError("git output is not UTF-8") from exc


def _freeze(value: Any, depth: int = 0) -> Any:
    if depth > MAX_RESULT_DEPTH:
        raise ResultValidationError("result is too deeply nested")
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str) and len(value) > 4096:
            raise ResultValidationError("result string exceeds bound")
        if isinstance(value, int) and not isinstance(value, bool) and abs(value) > 10**18:
            raise ResultValidationError("result integer exceeds bound")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ResultValidationError("result number is not finite")
        return value
    if isinstance(value, Mapping):
        if len(value) > 256 or any(not isinstance(k, str) or len(k) > 256 for k in value):
            raise ResultValidationError("result object is unbounded")
        return MappingProxyType({k: _freeze(v, depth + 1) for k, v in value.items()})
    if isinstance(value, (tuple, list)):
        if len(value) > MAX_RESULT_ITEMS:
            raise ResultValidationError("result array is unbounded")
        return tuple(_freeze(v, depth + 1) for v in value)
    raise ResultValidationError("result contains a non-JSON value")


@dataclass(frozen=True, slots=True)
class CheckoutValidation:
    checkout: str
    head: str
    source_digest: str
    interpreter_sha256: str
    interpreter_identity: tuple[int, int, int, int]
    clean: bool = True

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkout": self.checkout,
            "head": self.head,
            "source_digest": self.source_digest,
            "interpreter_sha256": self.interpreter_sha256,
            "interpreter_identity": self.interpreter_identity,
            "clean": self.clean,
        }


@dataclass(frozen=True, slots=True, init=False)
class IronMuleCheckoutBinding:
    """Immutable identity required before an IronMule stage can be planned."""

    checkout: str
    expected_head: str
    interpreter: str
    fingerprint: ExactFingerprint
    fixed_execution_files: tuple[str, ...]
    execution_registry_hash: str
    forbidden_checkouts: tuple[str, ...]
    source_digest: str | None
    interpreter_sha256: str
    interpreter_identity: tuple[int, int, int, int]

    def __init__(
        self,
        checkout: str | os.PathLike[str] | None = None,
        expected_head: str | None = None,
        interpreter: str | os.PathLike[str] | None = None,
        fingerprint: ExactFingerprint | None = None,
        *,
        checkout_path: str | os.PathLike[str] | None = None,
        model_fingerprint: ExactFingerprint | None = None,
        source_digest: str | None = None,
        interpreter_sha256: str | None = None,
        expected_interpreter_sha256: str | None = None,
        interpreter_identity: tuple[int, int, int, int] | None = None,
    ) -> None:
        if checkout is None:
            checkout = checkout_path
        elif checkout_path is not None and _absolute(checkout) != _absolute(checkout_path):
            raise ValueError("checkout and checkout_path disagree")
        if fingerprint is None:
            fingerprint = model_fingerprint
        elif model_fingerprint is not None and fingerprint != model_fingerprint:
            raise ValueError("fingerprint and model_fingerprint disagree")
        if expected_interpreter_sha256 is not None:
            if interpreter_sha256 is not None and interpreter_sha256 != expected_interpreter_sha256:
                raise ValueError("interpreter hash arguments disagree")
            interpreter_sha256 = expected_interpreter_sha256
        if checkout is None or expected_head is None or interpreter is None or fingerprint is None:
            raise TypeError("checkout, expected_head, interpreter and fingerprint are required")
        checkout_value = _absolute(checkout, "checkout")
        interpreter_value = _absolute(interpreter, "interpreter")
        if not _HEAD.fullmatch(expected_head):
            raise ValueError("expected_head must be a 40-character lowercase commit")
        if not isinstance(fingerprint, ExactFingerprint) or not fingerprint.recommendation_allowed:
            raise ValueError("an exact complete ExactFingerprint is required")
        files = DEFAULT_EXECUTION_FILES
        forbidden = DEFAULT_FORBIDDEN_CHECKOUTS
        if source_digest is not None:
            source_digest = _validate_sha(source_digest, "source_digest")
        if interpreter_sha256 is not None:
            interpreter_sha256 = _validate_sha(interpreter_sha256, "interpreter_sha256")
        # Capture interpreter/source identities now when safely possible.  An
        # invalid checkout is still representable so validate_checkout() can
        # report the precise failure; no identity is guessed on failure.
        captured_hash = interpreter_sha256
        captured_identity = interpreter_identity
        try:
            actual_hash, actual_identity = _file_digest(interpreter_value, field="interpreter", executable=True)
            if captured_hash is None:
                captured_hash = actual_hash
            if captured_identity is None:
                captured_identity = actual_identity
        except (CheckoutValidationError, OSError):
            if captured_hash is None:
                captured_hash = ""
            if captured_identity is None:
                captured_identity = (0, 0, 0, 0)
        if not isinstance(captured_identity, tuple) or len(captured_identity) != 4 or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in captured_identity
        ):
            raise ValueError("interpreter_identity is invalid")
        if source_digest is None:
            try:
                source_digest = _source_digest(checkout_value, files, _registry_hash(files))
            except (CheckoutValidationError, OSError):
                source_digest = ""
        object.__setattr__(self, "checkout", checkout_value)
        object.__setattr__(self, "expected_head", expected_head)
        object.__setattr__(self, "interpreter", interpreter_value)
        object.__setattr__(self, "fingerprint", fingerprint)
        object.__setattr__(self, "fixed_execution_files", files)
        object.__setattr__(self, "execution_registry_hash", EXECUTION_FILE_REGISTRY_HASH)
        object.__setattr__(self, "forbidden_checkouts", forbidden)
        object.__setattr__(self, "source_digest", source_digest)
        object.__setattr__(self, "interpreter_sha256", captured_hash)
        object.__setattr__(self, "interpreter_identity", captured_identity)

    @classmethod
    def _for_testing(
        cls,
        *,
        checkout: str | os.PathLike[str],
        expected_head: str,
        interpreter: str | os.PathLike[str],
        fingerprint: ExactFingerprint,
        fixed_execution_files: tuple[str, ...],
        forbidden_checkouts: tuple[str, ...] = (),
        source_digest: str | None = None,
        interpreter_sha256: str | None = None,
        interpreter_identity: tuple[int, int, int, int] | None = None,
    ) -> "IronMuleCheckoutBinding":
        """Construct a fixture binding; never used by production code."""
        # Re-run the public validation/capture path with fixture-only values,
        # then replace the sealed registry/forbidden values before returning.
        base = cls(
            checkout=checkout, expected_head=expected_head, interpreter=interpreter,
            fingerprint=fingerprint, source_digest=source_digest,
            interpreter_sha256=interpreter_sha256, interpreter_identity=interpreter_identity,
        )
        files = tuple(_validate_relative(item, "fixed_execution_files") for item in fixed_execution_files)
        if not files or len(files) != len(set(files)):
            raise ValueError("fixed execution files must be unique and non-empty")
        forbidden = tuple(_absolute(item, "forbidden_checkouts") for item in forbidden_checkouts)
        if len(forbidden) != len(set(_path_key(item) for item in forbidden)):
            raise ValueError("forbidden checkouts must be unique")
        object.__setattr__(base, "fixed_execution_files", files)
        object.__setattr__(base, "execution_registry_hash", _registry_hash(files))
        object.__setattr__(base, "forbidden_checkouts", forbidden)
        # Recompute a fixture digest when the caller did not supply one.
        if source_digest is None:
            try:
                object.__setattr__(base, "source_digest", _source_digest(base.checkout, files, _registry_hash(files)))
            except CheckoutValidationError:
                object.__setattr__(base, "source_digest", "")
        return base

    @property
    def checkout_path(self) -> str:
        return self.checkout

    @property
    def expected_commit(self) -> str:
        return self.expected_head

    @property
    def head(self) -> str:
        return self.expected_head

    @property
    def model_fingerprint(self) -> ExactFingerprint:
        return self.fingerprint

    @property
    def source_tree_digest(self) -> str:
        return self.source_digest

    @property
    def execution_file_registry_hash(self) -> str:
        return self.execution_registry_hash

    @property
    def interpreter_hash(self) -> str:
        return self.interpreter_sha256


@dataclass(frozen=True, slots=True)
class ParsedIronMuleResult:
    stage: str
    commit: str
    fingerprint: str
    candidate_id: str
    token_identity: bool
    token_count: int
    stop_reason: str
    response_hash: str
    resources: Mapping[str, Any]
    screening: Mapping[str, Any] | None
    confirmation: Mapping[str, Any] | None
    confirmed: bool
    selected_ratio: float | None = None
    profile_id: str | None = None
    profile_version: int | None = None
    source_digest: str | None = None
    registry_hash: str | None = None
    worker_sha256: str | None = None
    session_id: str | None = None
    status: str | None = None
    calibration: Mapping[str, Any] | None = None
    evidence: Mapping[str, Any] | None = None

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]

    def as_dict(self) -> dict[str, Any]:
        result = {
            "schema": "friday.ironmule.result.v1",
            "stage": self.stage,
            "commit": self.commit,
            "fingerprint": self.fingerprint,
            "candidate": self.candidate_id,
            "token_identity": self.token_identity,
            "token_count": self.token_count,
            "stop_reason": self.stop_reason,
            "response_hash": self.response_hash,
            "resources": dict(self.resources),
            "screening": None if self.screening is None else dict(self.screening),
            "confirmation": None if self.confirmation is None else dict(self.confirmation),
            "confirmed": self.confirmed,
            "selected_ratio": self.selected_ratio,
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "source_digest": self.source_digest,
            "registry_hash": self.registry_hash,
            "worker_sha256": self.worker_sha256,
            "session_id": self.session_id,
            "status": self.status,
            "calibration": None if self.calibration is None else dict(self.calibration),
        }
        if self.evidence is not None:
            result.update(dict(self.evidence))
        return result


@dataclass(frozen=True, slots=True)
class StagedSourceEntry:
    """Canonical identity of one file in a staged source snapshot."""

    relative_path: str
    sha256: str
    size_bytes: int
    st_dev: int
    st_ino: int
    mode: int

    def __post_init__(self) -> None:
        path = Path(self.relative_path)
        if not isinstance(self.relative_path, str) or not self.relative_path or path.is_absolute() or ".." in path.parts:
            raise ValueError("staged relative path is invalid")
        if not isinstance(self.sha256, str) or not _SHA256.fullmatch(self.sha256):
            raise ValueError("staged source hash is invalid")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int) or self.size_bytes < 0:
            raise ValueError("staged source size is invalid")
        for name in ("st_dev", "st_ino", "mode"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"staged source {name} is invalid")

    def as_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "st_dev": self.st_dev,
            "st_ino": self.st_ino,
            "mode": self.mode,
        }


@dataclass(frozen=True, slots=True)
class StagedStageSpec(StageSpec):
    """A StageSpec whose executable source lives in a private snapshot.

    It remains an actual ``StageSpec`` for ``SubprocessStageRunner`` while
    carrying the immutable operation metadata needed by SessionController.
    The owner must call ``cleanup`` or use it as a context manager.
    """

    stage_directory: str = ""
    source_manifest: tuple[StagedSourceEntry, ...] = ()
    source_digest: str = ""
    source_registry_hash: str = ""
    worker_sha256: str = ""
    spec_relative_path: str = SPEC_RELATIVE_PATH
    spec_sha256: str = ""
    spec_bytes: bytes = b""

    def __post_init__(self) -> None:
        StageSpec.__post_init__(self)
        if not isinstance(self.stage_directory, str) or not os.path.isabs(self.stage_directory):
            raise ValueError("staged directory must be absolute")
        if not os.path.basename(self.stage_directory).startswith("friday-ironmule-stage-"):
            raise ValueError("staged directory is not adapter-owned")
        if not isinstance(self.stage, str) or not self.stage:
            raise ValueError("stage name is required")
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("candidate id is required")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("stage parameters must be a mapping")
        if not isinstance(self.execute_authorized, bool):
            raise TypeError("execute_authorized must be bool")
        if not isinstance(self.source_digest, str) or not _SHA256.fullmatch(self.source_digest):
            raise ValueError("staged source digest is invalid")
        if not isinstance(self.source_registry_hash, str) or not _SHA256.fullmatch(self.source_registry_hash):
            raise ValueError("staged source registry hash is invalid")
        if not isinstance(self.worker_sha256, str) or not _SHA256.fullmatch(self.worker_sha256):
            raise ValueError("staged worker hash is invalid")
        if self.spec_relative_path != SPEC_RELATIVE_PATH:
            raise ValueError("staged spec path is not fixed")
        if not isinstance(self.spec_sha256, str) or not _SHA256.fullmatch(self.spec_sha256):
            raise ValueError("staged spec hash is invalid")
        if not isinstance(self.spec_bytes, bytes) or len(self.spec_bytes) > 64 * 1024:
            raise ValueError("staged spec bytes are invalid")
        if not isinstance(self.source_manifest, tuple) or not self.source_manifest:
            raise ValueError("staged source manifest is required")
        if any(not isinstance(entry, StagedSourceEntry) for entry in self.source_manifest):
            raise TypeError("staged source manifest entries are invalid")
        paths = tuple(entry.relative_path for entry in self.source_manifest)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("staged source manifest must be sorted and unique")

    @property
    def spec(self) -> "StagedStageSpec":
        return self

    def cleanup(self) -> None:
        path = self.stage_directory
        if path and os.path.basename(path).startswith("friday-ironmule-stage-"):
            shutil.rmtree(path, ignore_errors=True)

    def __enter__(self) -> "StagedStageSpec":
        return self

    def __exit__(self, *_: Any) -> None:
        self.cleanup()


def _finite_number(value: Any, field: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < minimum:
        raise ResultValidationError(f"{field} must be finite and >= {minimum}")
    return float(value)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or isinstance(value, (str, bytes, bytearray)):
        raise ResultValidationError(f"{field} must be an object")
    return value


def _ratio(confirmation: Mapping[str, Any]) -> float | None:
    value = confirmation.get("ratio")
    if isinstance(value, Mapping) and isinstance(value.get("total_ns"), Mapping):
        value = value["total_ns"]
    if isinstance(value, Mapping):
        low = value.get("ci_low")
        high = value.get("ci_high")
        if low is None or high is None:
            raise ResultValidationError("confirmation ratio needs a confidence interval")
        low_value = _finite_number(low, "confirmation ratio ci_low", minimum=0.000001)
        high_value = _finite_number(high, "confirmation ratio ci_high", minimum=0.000001)
        if low_value > high_value:
            raise ResultValidationError("confirmation ratio interval is inverted")
        value = value.get("median_ratio")
    if value is None:
        return None
    return _finite_number(value, "confirmation ratio", minimum=0.000001)


_RESOURCE_KEYS = (
    "ttft_ms", "decode_tokens_per_second", "peak_memory_bytes",
    "peak_rss_bytes", "swap_delta_bytes",
)


def _validate_resources(value: Any, field: str, *, require_gate: bool = True) -> Mapping[str, Any]:
    resources = _mapping(value, field)
    for key in _RESOURCE_KEYS:
        if key not in resources:
            raise ResultValidationError(f"{field} metric missing: {key}")
        # A negative swap delta is a useful diagnostic (swap fell during the
        # block), unlike negative memory/time/throughput values.
        minimum = -float("inf") if key == "swap_delta_bytes" else 0.0
        _finite_number(resources[key], f"{field}.{key}", minimum=minimum)
    if require_gate and resources.get("resource_gate_passed") is not True:
        raise ResultValidationError(f"{field}.resource_gate_passed must be true")
    if not isinstance(resources.get("resource_gate_passed"), bool):
        raise ResultValidationError(f"{field}.resource_gate_passed must be boolean")
    return resources


class IronMuleTuneAdapter:
    """Strict adapter for the currently available IronMule CLI."""

    supports_promotion = False
    _READ_ONLY_STAGES = frozenset({"doctor", "status"})
    _EXECUTION_STAGES = frozenset({"calibrate", "test"})
    # IronMule's public CLI has no flag selecting individual knobs.  ``tune``
    # therefore maps only to the preregistered combined profile it actually
    # searches; mapping every Friday candidate to the same command would make
    # unsupported candidates look executable.  Baseline maps to the real,
    # read-only ``status`` command.  All other candidates remain blocked.
    _CANDIDATE_COMMANDS = MappingProxyType({
        "baseline": "status",
        "combined_core_profile": "tune",
    })

    def __init__(
        self,
        binding: IronMuleCheckoutBinding,
        *,
        runner: SubprocessStageRunner | None = None,
        max_output_bytes: int = 64 * 1024,
    ) -> None:
        if not isinstance(binding, IronMuleCheckoutBinding):
            raise TypeError("binding must be IronMuleCheckoutBinding")
        self.binding = binding
        self.registry = CandidateRegistry()
        self._max_output_bytes = max_output_bytes
        self._pythonpath, self._pythonpath_identity = _purelib_binding(binding.interpreter)
        if runner is not None and type(runner) is not SubprocessStageRunner:
            raise TypeError("only the controlled SubprocessStageRunner is accepted")
        self.stage_runner = runner
        self._validation: CheckoutValidation | None = None
        self._staged_specs: dict[str, StagedStageSpec] = {}
        self._authorization_secret = secrets.token_bytes(32)
        self._authorization_lock = threading.Lock()
        self._consumed_authorizations: set[str] = set()

    def _runner_for(self, cwd: str, env: Mapping[str, str] | None = None) -> SubprocessStageRunner:
        effective_env = dict(OFFLINE_ENV)
        if env is not None:
            effective_env.update(dict(env))
        runner = self.stage_runner
        if (
            runner is not None
            and getattr(runner, "allowed_cwd_root", None) == os.path.abspath(cwd)
            and dict(getattr(runner, "allowed_env", {})) == effective_env
        ):
            return runner
        runner = SubprocessStageRunner(
            allowlisted_executables={self.binding.interpreter: self.binding.interpreter_sha256},
            allowed_cwd_root=cwd,
            allowed_env=effective_env,
            fixed_env=effective_env,
            max_output_bytes=self._max_output_bytes,
        )
        self.stage_runner = runner
        return runner

    @property
    def stage_specs(self) -> Mapping[str, StageSpec]:
        # Deliberately no activate/canary/rollback/deactivate entries.  This
        # prevents SessionController's promotion path from being constructed.
        return MappingProxyType({
            "doctor": self.plan_stage("doctor", candidate_id="baseline"),
            "status": self.plan_stage("status", candidate_id="baseline"),
            "calibrate": self.plan_stage("calibrate", candidate_id="combined_core_profile", qualified=("fixed_compiled_cache", "head_skip_prefill")),
            "test": self.plan_stage("test", candidate_id="combined_core_profile", qualified=("fixed_compiled_cache", "head_skip_prefill")),
        })

    @staticmethod
    def _authorization_material(spec: StagedStageSpec, *, session_id: str, nonce: str) -> bytes:
        return canonical_bytes({
            "schema": "friday.ironmule.stage-authorization.v1",
            "session_id": session_id,
            "nonce": nonce,
            "executable": spec.executable,
            "args": list(spec.args),
            "cwd": spec.cwd,
            "env": dict(spec.env or {}),
            "stage": spec.stage,
            "candidate_id": spec.candidate_id,
            "parameters": dict(spec.parameters),
            "source_digest": spec.source_digest,
            "source_registry_hash": spec.source_registry_hash,
            "worker_sha256": spec.worker_sha256,
            "spec_relative_path": spec.spec_relative_path,
            "spec_sha256": spec.spec_sha256,
            "source_manifest": [entry.as_dict() for entry in spec.source_manifest],
            "execute_authorized": True,
        }, max_bytes=64 * 1024)

    def validate_checkout(self) -> CheckoutValidation:
        checkout = self.binding.checkout
        expected_registry_hash = _registry_hash(self.binding.fixed_execution_files)
        if self.binding.execution_registry_hash != expected_registry_hash:
            raise CheckoutValidationError("execution file registry identity changed")
        _reject_symlink_ancestors(checkout)
        info = os.lstat(checkout)
        if not stat.S_ISDIR(info.st_mode) or os.path.islink(checkout):
            raise CheckoutValidationError("checkout must be a regular non-symlink directory")
        if _path_key(checkout) in {_path_key(item) for item in self.binding.forbidden_checkouts}:
            raise CheckoutValidationError("checkout is explicitly forbidden")
        head = _git(checkout, ("rev-parse", "--verify", "HEAD")).strip()
        if not _HEAD.fullmatch(head) or head != self.binding.expected_head:
            raise CheckoutValidationError("checkout HEAD does not match expected commit")
        status = _git(checkout, ("status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none"))
        status_lines = tuple(line for line in status.splitlines() if line)
        tracked_changes = tuple(line for line in status_lines if not line.startswith("??"))
        untracked_changes = tuple(line for line in status_lines if line.startswith("??"))
        if tracked_changes:
            raise CheckoutValidationError("checkout has tracked changes")
        if untracked_changes:
            raise CheckoutValidationError("checkout has untracked files")
        interpreter_hash, interpreter_identity = _file_digest(self.binding.interpreter, field="interpreter", executable=True)
        if interpreter_hash != self.binding.interpreter_sha256 or interpreter_identity != self.binding.interpreter_identity:
            raise CheckoutValidationError("interpreter identity changed")
        if self._pythonpath is not None:
            if os.path.realpath(sys.executable) != os.path.realpath(self.binding.interpreter):
                raise CheckoutValidationError("running interpreter does not match bound interpreter")
            try:
                _reject_symlink_ancestors(self._pythonpath)
                info = os.stat(self._pythonpath)
            except (OSError, CheckoutValidationError) as exc:
                raise CheckoutValidationError("purelib is unavailable") from exc
            current = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
            if current != self._pythonpath_identity:
                raise CheckoutValidationError("purelib identity changed")
        source_digest = _source_digest(checkout, self.binding.fixed_execution_files, self.binding.execution_registry_hash)
        if source_digest != self.binding.source_digest:
            raise CheckoutValidationError("fixed execution source changed")
        result = CheckoutValidation(checkout, head, source_digest, interpreter_hash, interpreter_identity, not bool(tracked_changes or untracked_changes))
        self._validation = result
        return result

    def doctor(self) -> Mapping[str, Any]:
        """Return an offline binding report; never execute IronMule or inspect MLX."""
        try:
            validation = self.validate_checkout()
        except CheckoutValidationError as exc:
            return MappingProxyType({
                "ok": False,
                "reason": str(exc),
                "checkout": self.binding.checkout,
                "expected_head": self.binding.expected_head,
                "fingerprint": self.binding.fingerprint.fingerprint_hash,
                "supports_promotion": False,
                "offline": True,
            })
        return MappingProxyType({
            "ok": True,
            "checkout": validation.checkout,
            "head": validation.head,
            "source_digest": validation.source_digest,
            "interpreter_sha256": validation.interpreter_sha256,
            "fingerprint": self.binding.fingerprint.fingerprint_hash,
            "supports_promotion": False,
            "stages": tuple(sorted(self._READ_ONLY_STAGES | self._EXECUTION_STAGES)),
            "offline": True,
        })

    def _model_id(self) -> str:
        model_id = self.binding.fingerprint.model.model_id
        try:
            return validate_hub_model_id(model_id)
        except WorkerError as exc:
            raise IronMuleAdapterError("unsupported_model_source") from exc

    def _max_tokens(self) -> str:
        value = self.binding.fingerprint.workload.max_tokens
        if isinstance(value, bool) or not isinstance(value, int) or value < 1 or value > 1_000_000:
            raise IronMuleAdapterError("bound max_tokens is invalid")
        return str(value)

    @staticmethod
    def _read_stage_file(path: str) -> tuple[bytes, str, tuple[int, int, int, int]]:
        _regular_file(path, "execution file")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        try:
            fd = os.open(path, flags)
        except OSError as exc:
            raise CheckoutValidationError("cannot open execution file") from exc
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_size > 16 * 1024 * 1024:
                raise CheckoutValidationError("execution file is invalid or too large")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(fd)
            identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            if identity != after_identity:
                raise CheckoutValidationError("execution file changed while staging")
            raw = b"".join(chunks)
            return raw, hashlib.sha256(raw).hexdigest(), identity
        finally:
            os.close(fd)

    def _stage_checkout(self) -> tuple[str, tuple[StagedSourceEntry, ...], str]:
        """Copy the verified fixed source subset into a private immutable tree."""
        files = self.binding.fixed_execution_files
        hashes: dict[str, str] = {}
        contents: dict[str, bytes] = {}
        for relative in files:
            raw, digest, _ = self._read_stage_file(os.path.join(self.binding.checkout, relative))
            contents[relative] = raw
            hashes[relative] = digest
        aggregate = _compose_source_digest(files, hashes, self.binding.execution_registry_hash)
        if aggregate != self.binding.source_digest:
            raise CheckoutValidationError("fixed execution source changed before staging")
        stage_root = tempfile.mkdtemp(prefix="friday-ironmule-stage-")
        try:
            os.chmod(stage_root, 0o700)
            for relative in files:
                target = os.path.join(stage_root, relative)
                parent = os.path.dirname(target)
                os.makedirs(parent, mode=0o700, exist_ok=True)
                # The staging root is private and freshly created; O_EXCL plus
                # no-follow prevents a future path substitution from turning a
                # source file into an arbitrary destination.
                fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o500)
                try:
                    remaining = memoryview(contents[relative])
                    while remaining:
                        written = os.write(fd, remaining)
                        if written <= 0:
                            raise CheckoutValidationError("staged file write made no progress")
                        remaining = remaining[written:]
                    os.fsync(fd)
                finally:
                    os.close(fd)
                os.chmod(target, 0o500)
            directories = [stage_root]
            package_dir = os.path.join(stage_root, "ironmule")
            if os.path.isdir(package_dir):
                directories.append(package_dir)
            for directory in directories:
                directory_fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            manifest: list[StagedSourceEntry] = []
            for relative in sorted(files):
                target = os.path.join(stage_root, relative)
                info = os.lstat(target)
                manifest.append(StagedSourceEntry(
                    relative_path=relative,
                    sha256=hashes[relative],
                    size_bytes=info.st_size,
                    st_dev=info.st_dev,
                    st_ino=info.st_ino,
                    mode=stat.S_IMODE(info.st_mode),
                ))
            return stage_root, tuple(manifest), aggregate
        except Exception:
            shutil.rmtree(stage_root, ignore_errors=True)
            raise

    @staticmethod
    def _write_private_file(path: str, raw: bytes, *, mode: int) -> str:
        """Create one adapter-owned file without following a destination link."""
        if len(raw) > 16 * 1024 * 1024:
            raise CheckoutValidationError("staged file exceeds bound")
        parent = os.path.dirname(path)
        os.makedirs(parent, mode=0o700, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), mode)
        try:
            view = memoryview(raw)
            while view:
                count = os.write(fd, view)
                if count <= 0:
                    raise CheckoutValidationError("staged file write made no progress")
                view = view[count:]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.chmod(path, mode)
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _replace_private_file(path: str, raw: bytes, *, mode: int) -> str:
        """Rewrite a staged regular file without following a substituted link."""
        if len(raw) > 64 * 1024:
            raise CheckoutValidationError("staged file exceeds bound")
        try:
            _reject_symlink_ancestors(path)
            os.chmod(path, 0o600)
            fd = os.open(path, os.O_WRONLY | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0))
        except OSError as exc:
            raise CheckoutValidationError("staged file replacement failed") from exc
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise CheckoutValidationError("staged replacement is not regular")
            view = memoryview(raw)
            while view:
                count = os.write(fd, view)
                if count <= 0:
                    raise CheckoutValidationError("staged file write made no progress")
                view = view[count:]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.chmod(path, mode)
        return hashlib.sha256(raw).hexdigest()

    def _stage_worker(self, stage_root: str) -> str:
        """Copy this exact worker source into the private stage."""
        source = Path(__file__).with_name("ironmule_stage_worker.py")
        raw, digest, _ = self._read_stage_file(str(source))
        target = os.path.join(stage_root, WORKER_RELATIVE_PATH)
        observed = self._write_private_file(target, raw, mode=0o500)
        if observed != digest:
            raise CheckoutValidationError("worker source changed while staging")
        return digest

    def _worker_spec(
        self,
        *,
        stage: str,
        candidate_id: str,
        source_digest: str,
        worker_sha256: str,
        source_manifest: tuple[StagedSourceEntry, ...],
        source_registry_hash: str,
        session_id: str = "",
    ) -> dict[str, Any]:
        model = self.binding.fingerprint.model.as_dict()
        workload = self.binding.fingerprint.workload.as_dict()
        ram = self.binding.fingerprint.environment.ram_bytes or 1
        return {
            "schema": "friday.ironmule.stage-spec.v1",
            "stage": stage,
            "candidate": candidate_id,
            "model": model,
            "workload": workload,
            "expected": {
                "commit": self.binding.expected_head,
                "source_digest": source_digest,
                "registry_hash": source_registry_hash,
                "fingerprint": self.binding.fingerprint.fingerprint_hash,
                "worker_sha256": worker_sha256,
                "tune_search_contract_sha256": TUNE_SEARCH_CONTRACT_SHA256,
                "pythonpath_sha256": hashlib.sha256((self._pythonpath or "").encode("utf-8")).hexdigest(),
            },
            "limits": {
                "max_seconds": 1800.0,
                "max_output_bytes": min(self._max_output_bytes, 256 * 1024),
                "max_rss_bytes": min(MAX_WORKER_RESOURCE_BYTES, max(ram, 1)),
                "max_peak_memory_bytes": min(MAX_WORKER_RESOURCE_BYTES, max(ram, 1)),
                "max_swap_delta_bytes": 0,
                "ac_connected": self.binding.fingerprint.workload.power_mode == "ac",
                "low_power": False,
                "processes": 6,
                "repeats": 7,
                "warmup": 2,
                "ttft_contract": "engine_prefill_to_first_token",
            },
            "session": {"session_id": session_id},
            "source_manifest": [
                {"relative_path": entry.relative_path, "sha256": entry.sha256,
                 "size_bytes": entry.size_bytes}
                for entry in source_manifest
            ],
        }

    def _write_worker_spec(self, stage_root: str, spec: Mapping[str, Any]) -> str:
        try:
            raw = canonical_bytes(spec, max_bytes=64 * 1024)
            return self._write_private_file(os.path.join(stage_root, SPEC_RELATIVE_PATH), raw, mode=0o400)
        except Exception as exc:
            raise CheckoutValidationError("worker spec could not be staged") from exc

    def cleanup(self) -> None:
        for spec in tuple(self._staged_specs.values()):
            spec.cleanup()
        self._staged_specs.clear()

    def __enter__(self) -> "IronMuleTuneAdapter":
        return self

    def __exit__(self, *_: Any) -> None:
        self.cleanup()

    def plan_stage(self, stage: str, *, candidate_id: str = "baseline", parameters: Mapping[str, Any] | None = None, qualified: tuple[str, ...] = ()) -> StagedStageSpec:
        """Build one exact argv; caller cannot append flags or paths."""
        if not isinstance(stage, str) or stage not in self._READ_ONLY_STAGES | self._EXECUTION_STAGES:
            raise UnsupportedStage(f"stage is not allowlisted: {stage!r}")
        if not isinstance(qualified, tuple) or any(not isinstance(item, str) for item in qualified):
            raise TypeError("qualified candidates must be a tuple of IDs")
        command = self._CANDIDATE_COMMANDS.get(candidate_id)
        if command is None:
            raise UnsupportedStage(f"candidate has no supported IronMule CLI command: {candidate_id!r}")
        if (stage in self._READ_ONLY_STAGES and candidate_id != "baseline") or (
            stage in self._EXECUTION_STAGES and candidate_id != "combined_core_profile"
        ):
            raise UnsupportedStage(f"candidate {candidate_id!r} has no semantic command for stage {stage!r}")
        try:
            self.registry.validate(candidate_id, fingerprint=self.binding.fingerprint, parameters=parameters, qualified=qualified)
        except (CandidateError, TypeError, ValueError) as exc:
            raise IronMuleAdapterError(str(exc)) from exc
        self._validation = self.validate_checkout()
        if stage in self._EXECUTION_STAGES:
            # Reject local paths before creating any staged directory or
            # reaching IronMule's cache resolver (which could otherwise fall
            # through to a Hub/network lookup).
            self._model_id()
        previous = self._staged_specs.pop(stage, None)
        if previous is not None:
            previous.cleanup()
        stage_root, source_manifest, source_digest = self._stage_checkout()
        # Every executable stage now enters through one fixed worker protocol.
        # The IronMule CLI is deliberately never placed on argv: that surface
        # accepts free flags and cannot expose the post-tune MLX peak safely.
        worker_sha256 = ""
        spec_bytes = b""
        spec_sha256 = ""
        source_registry_hash = self.binding.execution_registry_hash
        if stage in {"calibrate", "test"}:
            worker_sha256 = self._stage_worker(stage_root)
            worker_spec = self._worker_spec(
                stage=stage, candidate_id=candidate_id, source_digest=source_digest,
                worker_sha256=worker_sha256, source_manifest=source_manifest,
                source_registry_hash=source_registry_hash, session_id="pending",
            )
            spec_bytes = canonical_bytes(worker_spec, max_bytes=64 * 1024)
            spec_sha256 = self._write_worker_spec(stage_root, worker_spec)
            args = (WORKER_RELATIVE_PATH, "tune")
            if stage == "calibrate":
                args += ("--no-confirm",)
            args += ("--spec-file", SPEC_RELATIVE_PATH)
        elif stage == "doctor":
            # Read-only diagnostics retain their existing fixed CLI mapping.
            args = (os.path.join(stage_root, "ironmule_cli.py"), "doctor")
        elif stage == "status" or command == "status":
            args = (os.path.join(stage_root, "ironmule_cli.py"), "status", "--model", self._model_id())
        else:  # pragma: no cover - guarded above
            raise UnsupportedStage(stage)
        stage_env = dict(OFFLINE_ENV)
        if self._pythonpath is not None:
            stage_env["PYTHONPATH"] = self._pythonpath
        planned = StagedStageSpec(
            self.binding.interpreter, args, cwd=stage_root, env=stage_env,
            stage_directory=stage_root, stage=stage, candidate_id=candidate_id,
            parameters={} if parameters is None else parameters,
            source_manifest=source_manifest, source_digest=source_digest,
            source_registry_hash=source_registry_hash, worker_sha256=worker_sha256 or hashlib.sha256(b"doctor").hexdigest(),
            spec_relative_path=SPEC_RELATIVE_PATH,
            spec_sha256=spec_sha256 or hashlib.sha256(b"doctor").hexdigest(),
            spec_bytes=spec_bytes,
        )
        self._staged_specs[stage] = planned
        return planned

    def authorize_stage(self, spec: StagedStageSpec, session_id: str) -> StagedStageSpec:
        """Return a new immutable authorized plan; never mutate a staged plan."""
        if not isinstance(spec, StagedStageSpec) or self._staged_specs.get(spec.stage) is not spec:
            raise IronMuleAdapterError("stage does not belong to this adapter")
        if spec.execute_authorized:
            raise IronMuleAdapterError("stage is already authorized")
        if not isinstance(session_id, str) or not session_id or len(session_id) > 256:
            raise ValueError("session_id is invalid")
        # Bind the actual session id into the worker document before issuing the
        # one-time HMAC.  A plan with the ``pending`` placeholder can never be
        # executed directly and is not useful as an authorization token.
        authorized_spec = spec
        if spec.stage in {"calibrate", "test"}:
            try:
                worker_spec = loads_strict(spec.spec_bytes, max_bytes=64 * 1024)
                if not isinstance(worker_spec, Mapping):
                    raise ValueError("worker spec is not an object")
                worker_spec = dict(worker_spec)
                worker_spec["session"] = {"session_id": session_id}
                updated_bytes = canonical_bytes(worker_spec, max_bytes=64 * 1024)
                path = os.path.join(spec.stage_directory, spec.spec_relative_path)
                updated_sha = self._replace_private_file(path, updated_bytes, mode=0o400)
                authorized_spec = replace(
                    spec, spec_sha256=updated_sha, spec_bytes=updated_bytes
                )
            except (OSError, ValueError, TypeError) as exc:
                raise IronMuleAdapterError("worker spec authorization failed") from exc
        nonce = secrets.token_urlsafe(32)
        material = self._authorization_material(authorized_spec, session_id=session_id, nonce=nonce)
        tag = hmac.new(self._authorization_secret, material, hashlib.sha256).hexdigest()
        authorized = replace(
            authorized_spec, execute_authorized=True,
            authorization_session_id=session_id,
            authorization_nonce=nonce,
            authorization_tag=tag,
        )
        self._staged_specs[spec.stage] = authorized
        return authorized

    def _validate_staged_manifest(self, spec: StagedStageSpec) -> bool:
        """Re-open every staged source file immediately before token consume."""
        root = spec.stage_directory
        try:
            _reject_symlink_ancestors(root)
            root_info = os.lstat(root)
            if not stat.S_ISDIR(root_info.st_mode) or os.path.islink(root):
                return False
            expected = {entry.relative_path: entry for entry in spec.source_manifest}
            expected_paths = set(expected)
            if spec.stage in {"calibrate", "test"}:
                expected_paths |= {WORKER_RELATIVE_PATH, spec.spec_relative_path}
            observed: set[str] = set()

            def walk(directory: str, prefix: str = "") -> bool:
                try:
                    entries = list(os.scandir(directory))
                except OSError:
                    return False
                for item in entries:
                    relative = f"{prefix}/{item.name}" if prefix else item.name
                    info = item.stat(follow_symlinks=False)
                    if stat.S_ISLNK(info.st_mode):
                        return False
                    if stat.S_ISDIR(info.st_mode):
                        if not walk(item.path, relative):
                            return False
                    elif stat.S_ISREG(info.st_mode):
                        observed.add(relative)
                    else:
                        return False
                return True

            if not walk(root) or observed != expected_paths:
                return False
            hashes: dict[str, str] = {}
            for relative, entry in expected.items():
                raw, digest, identity = self._read_stage_file(os.path.join(root, relative))
                info = os.lstat(os.path.join(root, relative))
                if (
                    digest != entry.sha256 or len(raw) != entry.size_bytes
                    or identity != (entry.st_dev, entry.st_ino, entry.size_bytes, info.st_mtime_ns)
                    or stat.S_IMODE(info.st_mode) != entry.mode
                ):
                    return False
                hashes[relative] = digest
            aggregate = _compose_source_digest(self.binding.fixed_execution_files, hashes, self.binding.execution_registry_hash)
            if aggregate != spec.source_digest:
                return False
            if spec.stage in {"calibrate", "test"}:
                worker_raw, worker_digest, _ = self._read_stage_file(os.path.join(root, WORKER_RELATIVE_PATH))
                if worker_digest != spec.worker_sha256 or not worker_raw:
                    return False
                spec_raw, spec_digest, _ = self._read_stage_file(os.path.join(root, spec.spec_relative_path))
                if spec_digest != spec.spec_sha256 or spec_raw != self._expected_spec_bytes(spec):
                    return False
            return True
        except (OSError, CheckoutValidationError, ValueError):
            return False

    @staticmethod
    def _expected_spec_bytes(spec: StagedStageSpec) -> bytes:
        """Read the canonical spec bytes represented by a staged immutable spec.

        The spec itself is additionally covered by the authorization HMAC.  The
        worker revalidates its semantic fields; this method only checks that the
        file was not replaced between authorization and spawn.
        """
        return spec.spec_bytes

    def verify_and_consume_authorization(self, spec: StagedStageSpec, session_id: str) -> bool:
        """Verify an adapter-issued token and consume it exactly once."""
        if not isinstance(spec, StagedStageSpec) or self._staged_specs.get(spec.stage) is not spec:
            return False
        if spec.execute_authorized is not True or spec.authorization_session_id != session_id:
            return False
        nonce = spec.authorization_nonce
        tag = spec.authorization_tag
        if not isinstance(nonce, str) or not isinstance(tag, str):
            return False
        # This check must precede the nonce lock/consume.  A transient or
        # malicious staged-file mutation therefore leaves no consumed token and
        # cannot reach the runner.
        if not self._validate_staged_manifest(spec):
            return False
        with self._authorization_lock:
            if nonce in self._consumed_authorizations:
                return False
            expected = hmac.new(
                self._authorization_secret,
                self._authorization_material(spec, session_id=session_id, nonce=nonce),
                hashlib.sha256,
            ).hexdigest()
            if not hmac.compare_digest(expected, tag):
                return False
            self._consumed_authorizations.add(nonce)
            return True

    def _blocked(self, stage: str, reason: str = "explicit_execution_authorization_required") -> AdapterResult:
        return AdapterResult("inconclusive", {
            "status": "blocked",
            "stage": stage,
            "supports_promotion": False,
            "reason": reason,
        }, reason=reason)

    def run_stage(self, spec: StagedStageSpec, *, deadline: float, session_id: str) -> AdapterResult:
        """Run only an adapter-owned, explicitly authorized staged spec."""
        if not isinstance(spec, StagedStageSpec) or self._staged_specs.get(spec.stage) is not spec:
            raise IronMuleAdapterError("stage does not belong to this adapter")
        if self.verify_and_consume_authorization(spec, session_id) is not True:
            return self._blocked(spec.stage, "stage_authorization_rejected")
        if not os.path.isdir(spec.stage_directory) or os.path.islink(spec.stage_directory):
            return AdapterResult("error", reason="staged_checkout_missing")
        return self._runner_for(spec.stage_directory, spec.env).run(spec, deadline=deadline)

    def _run(self, stage: str, *, deadline: float, session_id: str, candidate_id: str, parameters: Mapping[str, Any] | None, qualified: tuple[str, ...] = ()) -> AdapterResult:
        # Revalidate the complete immutable binding immediately before any
        # future execution.  Planning alone is harmless, but a caller must not
        # be able to mutate the checkout between planning and spawning.
        self._validation = self.validate_checkout()
        spec = self.plan_stage(
            stage, candidate_id=candidate_id, parameters=parameters,
            qualified=qualified,
        )
        try:
            raw = self.run_stage(spec, deadline=deadline, session_id=session_id)
        finally:
            spec.cleanup()
            self._staged_specs.pop(stage, None)
        if not isinstance(raw, AdapterResult):
            raise ResultValidationError("stage runner returned an invalid result")
        if raw.payload.get("status") == "blocked":
            return raw
        if raw.outcome in {"timeout", "error", "failed", "fail"}:
            return raw
        # A worker that could not import MLX, re-check power, or complete its
        # bounded preflight is already an honest inconclusive result.  Do not
        # turn that diagnostic envelope into a parser error merely because no
        # model evidence exists to populate correctness fields.
        if raw.outcome == "inconclusive" and "correctness" not in raw.payload:
            return raw
        try:
            parsed = self.parse_result(raw.payload, stage=stage, candidate_id=candidate_id)
        except ResultValidationError as exc:
            return AdapterResult("error", reason=f"invalid_result:{exc}")
        if stage == "calibrate":
            calibration = parsed.calibration
            if (
                calibration is not None
                and calibration.get("complete") is True
                and parsed.resources.get("resource_gate_passed") is True
            ):
                return AdapterResult("ok", parsed.as_dict(), reason="calibration_complete")
            return AdapterResult("inconclusive", parsed.as_dict(), reason="calibration_incomplete")
        outcome = "qualified" if parsed.confirmed and parsed.resources.get("resource_gate_passed") is True else "inconclusive"
        reason = "" if outcome == "qualified" else "confirmation_or_resource_gate_required"
        return AdapterResult(outcome, parsed.as_dict(), reason=reason)

    def calibrate(self, *, deadline: float, session_id: str = "", candidate_id: str = "combined_core_profile", parameters: Mapping[str, Any] | None = None, qualified: tuple[str, ...] = ("fixed_compiled_cache", "head_skip_prefill")) -> AdapterResult:
        return self._run("calibrate", deadline=deadline, session_id=session_id, candidate_id=candidate_id, parameters=parameters, qualified=qualified)

    def test(self, *, deadline: float, session_id: str = "", candidate_id: str = "combined_core_profile", parameters: Mapping[str, Any] | None = None, qualified: tuple[str, ...] = ("fixed_compiled_cache", "head_skip_prefill")) -> AdapterResult:
        return self._run("test", deadline=deadline, session_id=session_id, candidate_id=candidate_id, parameters=parameters, qualified=qualified)

    def canary(self, *, deadline: float, session_id: str = "", **_: Any) -> AdapterResult:
        del deadline, session_id
        return self._blocked("canary", "promotion_unsupported")

    def activate(self, *, deadline: float, session_id: str = "", **_: Any) -> AdapterResult:
        del deadline, session_id
        return self._blocked("activate", "promotion_unsupported")

    def rollback(self, *, deadline: float, session_id: str = "", **_: Any) -> AdapterResult:
        del deadline, session_id
        return self._blocked("rollback", "promotion_unsupported")

    def deactivate(self, *, deadline: float, session_id: str = "", **_: Any) -> AdapterResult:
        del deadline, session_id
        return self._blocked("deactivate", "promotion_unsupported")

    def parse_result(self, raw: bytes | str | Mapping[str, Any], *, stage: str = "test", candidate_id: str = "baseline") -> ParsedIronMuleResult:
        if stage not in self._EXECUTION_STAGES:
            raise ResultValidationError("result stage is not an executable shadow stage")
        if isinstance(raw, bytes):
            if len(raw) > MAX_RESULT_BYTES:
                raise ResultValidationError("result exceeds byte bound")
            try:
                value = loads_strict(raw, max_bytes=MAX_RESULT_BYTES, max_depth=MAX_RESULT_DEPTH, max_items=MAX_RESULT_ITEMS)
            except Exception as exc:
                raise ResultValidationError("result JSON is malformed") from exc
        elif isinstance(raw, str):
            encoded = raw.encode("utf-8")
            if len(encoded) > MAX_RESULT_BYTES:
                raise ResultValidationError("result exceeds byte bound")
            try:
                value = loads_strict(encoded, max_bytes=MAX_RESULT_BYTES, max_depth=MAX_RESULT_DEPTH, max_items=MAX_RESULT_ITEMS)
            except Exception as exc:
                raise ResultValidationError("result JSON is malformed") from exc
        elif isinstance(raw, Mapping):
            value = raw
        else:
            raise ResultValidationError("result must be JSON bytes, text, or object")
        value = _freeze(value)
        envelope = False
        envelope_outcome: str | None = None
        data = _mapping(value, "result")
        if "payload" in data:
            envelope = True
            if set(data) - {"outcome", "reason", "payload"}:
                raise ResultValidationError("result envelope contains unknown fields")
            envelope_outcome = data.get("outcome")
            if envelope_outcome not in {"ok", "pass", "qualified", "inconclusive", "rejected", "error", "timeout"}:
                raise ResultValidationError("result envelope outcome is invalid")
            if not isinstance(data.get("reason", ""), str) or len(data.get("reason", "")) > 512:
                raise ResultValidationError("result envelope reason is invalid")
            data = _mapping(data.get("payload"), "result envelope payload")
        allowed_fields = {
            "schema", "stage", "commit", "fingerprint", "candidate", "parameters",
            "correctness", "resources", "screening", "confirmation", "confirmed",
            "profile_id", "profile_version", "source_digest", "registry_hash",
            "worker_sha256", "session_id", "status", "profile_artifact_sha256",
            "captured_output_bytes",
            "calibration",
            "tune_search_contract_sha256",
            "baseline_samples", "candidate_samples", "aa_baseline_samples",
            "aa_control_samples", "raw_pairs", "orders", "pair_count",
            "evidence_sha256",
            "baseline_correctness", "candidate_correctness",
        }
        unknown = set(data) - allowed_fields
        if unknown:
            raise ResultValidationError("result contains unknown fields")
        schema = data.get("schema")
        if schema != "friday.ironmule.result.v1":
            raise ResultValidationError("unsupported result schema")
        if data.get("stage") != stage:
            raise ResultValidationError("result stage mismatch")
        commit = data.get("commit")
        fingerprint = data.get("fingerprint")
        if commit != self.binding.expected_head:
            raise ResultValidationError("result commit is stale or mismatched")
        if fingerprint != self.binding.fingerprint.fingerprint_hash:
            raise ResultValidationError("result fingerprint is stale or mismatched")
        candidate = data.get("candidate")
        if candidate != candidate_id or not isinstance(candidate, str):
            raise ResultValidationError("result candidate mismatch")
        try:
            qualified_candidates = ("fixed_compiled_cache", "head_skip_prefill") if candidate == "combined_core_profile" else ()
            self.registry.validate(candidate, fingerprint=self.binding.fingerprint, parameters=data.get("parameters"), qualified=qualified_candidates)
        except (CandidateError, TypeError, ValueError) as exc:
            raise ResultValidationError(f"result candidate is outside registry: {exc}") from exc
        correctness = _mapping(data.get("correctness"), "correctness")
        token_identity = correctness.get("token_identity")
        if not isinstance(token_identity, bool):
            raise ResultValidationError("token_identity is required")
        token_count = correctness.get("token_count")
        if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 0:
            raise ResultValidationError("token_count is invalid")
        stop_reason = correctness.get("stop_reason")
        if not isinstance(stop_reason, str) or not stop_reason or len(stop_reason) > 128:
            raise ResultValidationError("stop_reason is invalid")
        response_hash = correctness.get("response_hash")
        if not isinstance(response_hash, str) or not _SHA256.fullmatch(response_hash):
            raise ResultValidationError("response_hash is invalid")
        resources = _validate_resources(data.get("resources"), "resources", require_gate=False)
        source_digest = data.get("source_digest")
        if source_digest is not None and source_digest != self.binding.source_digest:
            raise ResultValidationError("result source digest is stale or mismatched")
        registry_hash = data.get("registry_hash")
        if registry_hash is not None and registry_hash != self.binding.execution_registry_hash:
            raise ResultValidationError("result registry hash is stale or mismatched")
        worker_sha256 = data.get("worker_sha256")
        if worker_sha256 is not None and (not isinstance(worker_sha256, str) or not _SHA256.fullmatch(worker_sha256)):
            raise ResultValidationError("result worker hash is invalid")
        tune_contract_hash = data.get("tune_search_contract_sha256")
        if tune_contract_hash is not None and tune_contract_hash != TUNE_SEARCH_CONTRACT_SHA256:
            raise ResultValidationError("result tune search contract is stale or mismatched")
        session_id = data.get("session_id")
        if session_id is not None and (not isinstance(session_id, str) or not session_id or len(session_id) > 256):
            raise ResultValidationError("result session id is invalid")
        status = data.get("status")
        if status is not None and (not isinstance(status, str) or not status or len(status) > 128):
            raise ResultValidationError("result status is invalid")
        screening_raw = data.get("screening")
        confirmation_raw = data.get("confirmation")
        calibration_raw = data.get("calibration")
        screening = None if screening_raw is None else _mapping(screening_raw, "screening")
        confirmation = None if confirmation_raw is None else _mapping(confirmation_raw, "confirmation")
        calibration = None if calibration_raw is None else _mapping(calibration_raw, "calibration")
        if calibration is not None:
            required_calibration = {"complete", "trial_count", "trials", "baseline", "candidate", "evidence_sha256"}
            if set(calibration) != required_calibration:
                raise ResultValidationError("calibration fields are missing or unknown")
            if not isinstance(calibration["complete"], bool):
                raise ResultValidationError("calibration.complete must be boolean")
            trial_count = calibration["trial_count"]
            trials = calibration["trials"]
            if isinstance(trial_count, bool) or not isinstance(trial_count, int) or trial_count < 1 or not isinstance(trials, (tuple, list)) or len(trials) != trial_count:
                raise ResultValidationError("calibration trials are invalid")
            if not isinstance(calibration["baseline"], Mapping) or not isinstance(calibration["candidate"], Mapping):
                raise ResultValidationError("calibration diagnostics are invalid")
            if not isinstance(calibration["evidence_sha256"], str) or not _SHA256.fullmatch(calibration["evidence_sha256"]):
                raise ResultValidationError("calibration evidence hash is invalid")
            if stage != "calibrate":
                raise ResultValidationError("calibration is only valid for calibrate stage")
        ratio = None if confirmation is None else _ratio(confirmation)
        confirmed = False
        bound_worker_contract = (
            source_digest == self.binding.source_digest
            and registry_hash == self.binding.execution_registry_hash
            and isinstance(worker_sha256, str)
            and _SHA256.fullmatch(worker_sha256)
        )
        derived_confirmation = False
        if confirmation is not None and not confirmation.get("confirmed"):
            ratio_value = confirmation.get("ratio")
            if isinstance(ratio_value, Mapping) and isinstance(ratio_value.get("total_ns"), Mapping):
                ratio_value = ratio_value["total_ns"]
            derived_confirmation = (
                bound_worker_contract
                and confirmation.get("token_identity") is True
                and isinstance(ratio_value, Mapping)
                and isinstance(ratio_value.get("ci_high"), (int, float))
                and not isinstance(ratio_value.get("ci_high"), bool)
                and math.isfinite(float(ratio_value["ci_high"]))
                and float(ratio_value["ci_high"]) < 1.0
            )
        if confirmation is not None and (confirmation.get("confirmed") is True or derived_confirmation):
            if confirmation.get("token_identity") is not True:
                raise ResultValidationError("confirmation token_identity must be true")
            if ratio is None:
                raise ResultValidationError("confirmed result needs a confirmation ratio")
            pair_count = confirmation.get("pair_count")
            pairs = confirmation.get("pairs")
            if not isinstance(pairs, (tuple, list)):
                ratio_source = confirmation.get("ratio")
                if isinstance(ratio_source, Mapping) and isinstance(ratio_source.get("total_ns"), Mapping):
                    pairs = ratio_source["total_ns"].get("pairs")
            if isinstance(pair_count, bool) or not isinstance(pair_count, int) or pair_count < 3:
                if not isinstance(pairs, (tuple, list)) or len(pairs) < 3:
                    raise ResultValidationError("confirmed result needs paired evidence")
                pair_count = len(pairs)
            baseline_count = confirmation.get("baseline_count", pair_count)
            candidate_count = confirmation.get("candidate_count", pair_count)
            if baseline_count != pair_count or candidate_count != pair_count:
                raise ResultValidationError("confirmation pair counts do not match")
            # IronMule's current profile does not repeat resource evidence in
            # its confirmation object.  The worker-captured top-level resource
            # record is the authority; never require a fabricated duplicate.
            confirmation_resources = confirmation.get("resources")
            if not isinstance(confirmation_resources, Mapping) or not confirmation_resources:
                confirmation_resources = resources
            _validate_resources(confirmation_resources, "confirmation.resources", require_gate=False)
            confirmed = True
        if envelope and envelope_outcome == "qualified" and not confirmed:
            raise ResultValidationError("envelope claims qualified without confirmed evidence")
        profile_id = data.get("profile_id")
        if profile_id is not None and (not isinstance(profile_id, str) or not profile_id or len(profile_id) > 256):
            raise ResultValidationError("profile_id is invalid")
        profile_version = data.get("profile_version")
        if profile_version is not None and (isinstance(profile_version, bool) or not isinstance(profile_version, int) or profile_version < 0):
            raise ResultValidationError("profile_version is invalid")
        if stage == "calibrate" and envelope and envelope_outcome == "qualified":
            raise ResultValidationError("calibration cannot claim qualified")
        evidence_keys = ("baseline_samples", "candidate_samples", "aa_baseline_samples", "aa_control_samples", "raw_pairs", "orders", "pair_count", "evidence_sha256", "baseline_correctness", "candidate_correctness", "tune_search_contract_sha256")
        evidence = {key: data[key] for key in evidence_keys if key in data}
        return ParsedIronMuleResult(stage, commit, fingerprint, candidate, token_identity, token_count, stop_reason, response_hash, resources, screening, confirmation, confirmed, ratio, profile_id, profile_version, source_digest, registry_hash, worker_sha256, session_id, status, calibration, evidence or None)


def validate_checkout(binding: IronMuleCheckoutBinding) -> CheckoutValidation:
    """Validate one binding without constructing or running a model stage."""
    return IronMuleTuneAdapter(binding).validate_checkout()


def doctor(binding: IronMuleCheckoutBinding) -> Mapping[str, Any]:
    """Return the offline adapter doctor report."""
    return IronMuleTuneAdapter(binding).doctor()


def plan_stage(binding: IronMuleCheckoutBinding, stage: str, *, candidate_id: str = "baseline", parameters: Mapping[str, Any] | None = None, qualified: tuple[str, ...] = ()) -> StageSpec:
    """Build an exact, offline StageSpec from a binding."""
    return IronMuleTuneAdapter(binding).plan_stage(stage, candidate_id=candidate_id, parameters=parameters, qualified=qualified)


def parse_result(binding: IronMuleCheckoutBinding, raw: bytes | str | Mapping[str, Any], *, stage: str = "test", candidate_id: str = "baseline") -> ParsedIronMuleResult:
    """Parse one bounded result against a freshly validated binding."""
    return IronMuleTuneAdapter(binding).parse_result(raw, stage=stage, candidate_id=candidate_id)


__all__ = [
    "CURRENT_IRONMULE_WORKTREE",
    "DEFAULT_EXECUTION_FILES",
    "DEFAULT_FORBIDDEN_CHECKOUTS",
    "EXECUTION_FILE_REGISTRY_HASH",
    "OFFLINE_ENV",
    "MAX_WORKER_RESOURCE_BYTES",
    "TUNE_SEARCH_CONTRACT_SHA256",
    "SPEC_RELATIVE_PATH",
    "WORKER_RELATIVE_PATH",
    "CheckoutValidation",
    "CheckoutValidationError",
    "doctor",
    "IronMuleAdapterError",
    "IronMuleCheckoutBinding",
    "IronMuleTuneAdapter",
    "ParsedIronMuleResult",
    "parse_result",
    "plan_stage",
    "ResultValidationError",
    "StagedSourceEntry",
    "StagedStageSpec",
    "UnsupportedStage",
    "validate_checkout",
]
