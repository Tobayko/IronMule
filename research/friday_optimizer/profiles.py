"""Versioned, fingerprint-bound profile storage with atomic rollback."""

from __future__ import annotations

import hashlib
import fcntl
import math
import os
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .canonical import CanonicalJSONError, canonical_bytes, loads_strict


class ProfileError(RuntimeError):
    """A profile cannot be trusted or safely used."""


_NO_WRITE = object()


class ProfileMode(str, Enum):
    AUTO = "auto"
    BASELINE = "baseline"
    PINNED = "pinned"


@dataclass(frozen=True)
class OptimizerProfile:
    profile_id: str
    fingerprint: str
    candidate: str
    metrics: Mapping[str, Any] = field(default_factory=dict)
    qualified: bool = False
    version: int = 1
    profile_hash: str = ""

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value or len(value) > 256 for value in (self.profile_id, self.fingerprint, self.candidate)):
            raise ValueError("profile identity is invalid")
        if isinstance(self.qualified, bool) is False:
            raise TypeError("qualified must be bool")
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 1:
            raise ValueError("profile version is invalid")
        _validate_json(self.metrics, depth=0)
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))

    def canonical_body(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "fingerprint": self.fingerprint,
            "candidate": self.candidate,
            "metrics": dict(self.metrics),
            "qualified": self.qualified,
            "version": self.version,
        }

    def with_hash(self) -> "OptimizerProfile":
        body = _canonical_bytes(self.canonical_body())
        return replace(self, profile_hash=hashlib.sha256(body).hexdigest())

    def verify(self) -> bool:
        return bool(self.profile_hash) and self.profile_hash == self.with_hash().profile_hash

    def as_dict(self) -> dict[str, Any]:
        value = self.canonical_body()
        value["profile_hash"] = self.profile_hash
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OptimizerProfile":
        if not isinstance(value, Mapping):
            raise ProfileError("malformed profile")
        try:
            required = ("profile_id", "fingerprint", "candidate", "metrics", "qualified", "version", "profile_hash")
            if set(value) != set(required):
                raise ProfileError("malformed profile")
            if any(not isinstance(value[key], str) for key in ("profile_id", "fingerprint", "candidate", "profile_hash")):
                raise ProfileError("profile types invalid")
            if not isinstance(value["metrics"], dict) or not isinstance(value["qualified"], bool) or isinstance(value["version"], bool) or not isinstance(value["version"], int):
                raise ProfileError("profile types invalid")
            profile = cls(
                profile_id=value["profile_id"],
                fingerprint=value["fingerprint"],
                candidate=value["candidate"],
                metrics=value["metrics"],
                qualified=value["qualified"],
                version=value["version"],
                profile_hash=value["profile_hash"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProfileError("malformed profile") from exc
        if not profile.profile_id or not profile.fingerprint or not profile.candidate or not profile.verify():
            raise ProfileError("profile hash or identity invalid")
        return profile


@dataclass(frozen=True)
class ProfileSelection:
    mode: ProfileMode
    profile: OptimizerProfile | None
    no_recommendation: bool
    reason: str = ""
    rollback_latched: bool = False


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return canonical_bytes(value, max_depth=8, max_bytes=1024 * 1024, max_items=256, max_string_bytes=4096)
    except (TypeError, ValueError, CanonicalJSONError) as exc:
        raise ProfileError("profile contains non-canonical data") from exc


def _validate_json(value: Any, *, depth: int) -> None:
    if depth > 8:
        raise ProfileError("profile data too deep")
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str) and len(value) > 4096:
            raise ProfileError("profile string too long")
        if isinstance(value, int) and not isinstance(value, bool) and abs(value) > 10**18:
            raise ProfileError("profile integer out of range")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProfileError("profile number is not finite")
        return
    if isinstance(value, Mapping):
        if len(value) > 256 or any(not isinstance(key, str) or len(key) > 256 for key in value):
            raise ProfileError("profile mapping is unbounded")
        for item in value.values():
            _validate_json(item, depth=depth + 1)
        return
    if isinstance(value, (list, tuple)):
        if len(value) > 256:
            raise ProfileError("profile sequence is unbounded")
        for item in value:
            _validate_json(item, depth=depth + 1)
        return
    raise ProfileError("profile contains non-canonical data")


def _regular_or_missing(path: Path) -> None:
    if path.is_symlink():
        raise ProfileError("profile store symlink refused")
    if path.exists() and not path.is_file():
        raise ProfileError("profile store must be a regular file")


def _path_identity(path: Path) -> tuple[int, int] | None:
    if not path.exists():
        if path.is_symlink():
            raise ProfileError("profile store symlink refused")
        return None
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ProfileError("profile store must be a regular file")
    return info.st_dev, info.st_ino


class AtomicProfileStore:
    """Single JSON store; all mutations use temp+fsync+replace+dir fsync."""

    SCHEMA_VERSION = 1

    def __init__(self, path: str | os.PathLike[str], *, max_bytes: int = 1024 * 1024) -> None:
        if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or not 4096 <= max_bytes <= 16 * 1024 * 1024:
            raise ValueError("invalid profile store size bound")
        self.path = Path(path)
        self.max_bytes = max_bytes
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self._check_parent()
        _regular_or_missing(self.path)
        _regular_or_missing(self.lock_path)

    def _check_parent(self) -> None:
        parent = self.path.parent
        if not parent.exists() or not parent.is_dir():
            raise ProfileError("profile parent is unsafe")
        current = parent
        while True:
            if current.is_symlink() and str(current) not in {"/var", "/tmp"}:
                raise ProfileError("profile ancestor is a symlink")
            if current.parent == current:
                break
            current = current.parent

    def _empty(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "version": 0,
            "mode": ProfileMode.AUTO.value,
            "profiles": {},
            "baseline": None,
            "previous_baselines": [],
            "active": None,
            "pinned": None,
            "rollback_latched": False,
        }

    def _read(self) -> dict[str, Any]:
        _regular_or_missing(self.path)
        if not self.path.exists():
            return self._empty()
        before = _path_identity(self.path)
        try:
            if self.path.stat().st_mode & 0o077:
                raise ProfileError("profile store permissions are broader than 0600")
        except OSError as exc:
            raise ProfileError("profile store stat failed") from exc
        try:
            raw = self.path.read_bytes()
            if len(raw) > self.max_bytes:
                raise ProfileError("profile store exceeds size bound")
            value = loads_strict(raw, max_depth=8, max_bytes=self.max_bytes, max_items=256, max_string_bytes=4096)
        except (OSError, UnicodeDecodeError, CanonicalJSONError) as exc:
            raise ProfileError("profile store unreadable") from exc
        if _path_identity(self.path) != before:
            raise ProfileError("profile store identity changed during read")
        allowed_root = {"schema_version", "version", "mode", "profiles", "baseline", "previous_baselines", "active", "pinned", "rollback_latched", "rollback_reason"}
        if not isinstance(value, dict) or not set(value).issubset(allowed_root) or value.get("schema_version") != self.SCHEMA_VERSION:
            raise ProfileError("unsupported profile store")
        if isinstance(value.get("version"), bool) or not isinstance(value.get("version"), int) or not 0 <= value["version"] <= 10**12:
            raise ProfileError("store version invalid")
        profiles = value.get("profiles")
        if not isinstance(profiles, dict):
            raise ProfileError("profile index malformed")
        if len(profiles) > 1024:
            raise ProfileError("profile index exceeds bound")
        # Validate every stored profile, not just the selected one: a tampered
        # historical entry must not remain a covert activation path.
        for key, item in profiles.items():
            if not isinstance(item, dict) or str(key) != str(item.get("profile_id")):
                raise ProfileError("profile index identity invalid")
            OptimizerProfile.from_dict(item)
        for field_name in ("baseline", "active", "pinned"):
            item = value.get(field_name)
            if item is not None and item not in profiles:
                raise ProfileError("profile pointer invalid")
        if not isinstance(value.get("previous_baselines", []), list):
            raise ProfileError("baseline history malformed")
        if any(not isinstance(item, str) or len(item) > 256 for item in value.get("previous_baselines", [])):
            raise ProfileError("baseline history invalid")
        if value.get("mode") not in {mode.value for mode in ProfileMode}:
            raise ProfileError("profile mode invalid")
        for pointer in ("baseline", "active", "pinned"):
            if value.get(pointer) is not None and not isinstance(value.get(pointer), str):
                raise ProfileError("profile pointer invalid")
        if not isinstance(value.get("rollback_latched"), bool):
            raise ProfileError("rollback latch invalid")
        return value

    @contextmanager
    def _exclusive(self):
        _regular_or_missing(self.lock_path)
        try:
            fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0), 0o600)
        except OSError as exc:
            raise ProfileError("profile lock unavailable") from exc
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode) or os.fstat(fd).st_mode & 0o077:
                raise ProfileError("profile lock must be regular and 0600")
            os.fchmod(fd, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
            except OSError as exc:
                raise ProfileError("profile lock failed") from exc
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def _write(self, value: Mapping[str, Any], *, expected_identity: tuple[int, int] | None = None) -> None:
        self._check_parent()
        _regular_or_missing(self.path)
        if _path_identity(self.path) != expected_identity:
            raise ProfileError("profile store identity changed before write")
        try:
            payload = canonical_bytes(value, max_depth=8, max_bytes=self.max_bytes, max_items=256, max_string_bytes=4096)
        except (TypeError, ValueError, CanonicalJSONError) as exc:
            raise ProfileError("profile store exceeds canonical bounds") from exc
        fd, tmp_name = tempfile.mkstemp(prefix=".profiles.", dir=str(self.path.parent))
        tmp = Path(tmp_name)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, self.path)
            os.chmod(self.path, 0o600)
            _path_identity(self.path)
            dir_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except Exception:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise

    def load(self) -> Mapping[str, Any]:
        return self._read()

    def current_version(self) -> int:
        with self._exclusive():
            return self._read()["version"]

    def _next(self, value: dict[str, Any], *, expected_identity: tuple[int, int] | None = None) -> None:
        value["version"] = int(value.get("version", 0)) + 1
        self._write(value, expected_identity=expected_identity)

    def _transaction(self, operation, *, expected_version: int | None = None):
        with self._exclusive():
            value = self._read()
            identity = _path_identity(self.path)
            if expected_version is not None:
                if isinstance(expected_version, bool) or not isinstance(expected_version, int):
                    raise ProfileError("expected version invalid")
                if value["version"] != expected_version:
                    raise ProfileError("profile version conflict")
            result = operation(value)
            if result is not _NO_WRITE:
                self._next(value, expected_identity=identity)
            return result

    def save(self, profile: OptimizerProfile, *, baseline: bool = False, current_fingerprint: str | None = None, expected_version: int | None = None) -> OptimizerProfile:
        if current_fingerprint is not None and profile.fingerprint != current_fingerprint:
            raise ProfileError("profile fingerprint mismatch")
        stored = profile.with_hash()
        def operation(value):
            profiles = dict(value["profiles"])
            old_baseline = value.get("baseline")
            if stored.profile_id in profiles:
                existing = OptimizerProfile.from_dict(profiles[stored.profile_id])
                if existing.as_dict() != stored.as_dict():
                    raise ProfileError("profile IDs are immutable")
                if baseline and old_baseline != stored.profile_id:
                    value["baseline"] = stored.profile_id
                    if old_baseline is not None:
                        history = list(value.get("previous_baselines", []))
                        history.append(old_baseline)
                        value["previous_baselines"] = history[-16:]
                    return existing
                return _NO_WRITE
            profiles[stored.profile_id] = stored.as_dict()
            value["profiles"] = profiles
            if baseline:
                if old_baseline is not None and old_baseline != stored.profile_id:
                    history = list(value.get("previous_baselines", []))
                    history.append(old_baseline)
                    value["previous_baselines"] = history[-16:]
                value["baseline"] = stored.profile_id
            return stored
        result = self._transaction(operation, expected_version=expected_version)
        return stored if result is _NO_WRITE else result

    def set_mode(self, mode: ProfileMode | str, *, pinned_id: str | None = None, expected_version: int | None = None) -> None:
        try:
            mode = ProfileMode(mode)
        except ValueError as exc:
            raise ProfileError("profile mode invalid") from exc
        def operation(value):
            if mode is ProfileMode.PINNED:
                if not pinned_id or pinned_id not in value["profiles"]:
                    raise ProfileError("pinned profile unavailable")
                value["pinned"] = pinned_id
            elif pinned_id is not None:
                raise ProfileError("pinned_id only valid in pinned mode")
            value["mode"] = mode.value
            return None
        self._transaction(operation, expected_version=expected_version)

    def _profile(self, value: Mapping[str, Any], profile_id: str | None) -> OptimizerProfile | None:
        if profile_id is None:
            return None
        return OptimizerProfile.from_dict(value["profiles"][profile_id])

    def select(self, fingerprint: str, *, mode: ProfileMode | str | None = None) -> ProfileSelection:
        if not fingerprint:
            raise ProfileError("fingerprint required")
        value = self._read()
        try:
            chosen_mode = ProfileMode(mode or value["mode"])
        except ValueError as exc:
            raise ProfileError("profile mode invalid") from exc
        if chosen_mode is ProfileMode.BASELINE:
            profile = self._profile(value, value.get("baseline"))
            if profile is None or profile.fingerprint != fingerprint:
                return ProfileSelection(chosen_mode, None, True, "baseline_missing_or_incompatible")
            return ProfileSelection(chosen_mode, profile, False)
        if value.get("rollback_latched"):
            baseline = self._profile(value, value.get("baseline"))
            if baseline is None or baseline.fingerprint != fingerprint:
                return ProfileSelection(chosen_mode, None, True, "rollback_latched_baseline_incompatible", True)
            return ProfileSelection(chosen_mode, baseline, True, "rollback_latched_baseline", True)
        pointer = value.get("active") if chosen_mode is ProfileMode.AUTO else value.get("pinned")
        profile = self._profile(value, pointer)
        if profile is None or profile.fingerprint != fingerprint or not profile.qualified:
            return ProfileSelection(chosen_mode, None, True, "profile_missing_or_incompatible")
        return ProfileSelection(chosen_mode, profile, False)

    def validate_activation(self, *, profile_id: str, fingerprint: str, session_id: str) -> bool:
        if not isinstance(session_id, str) or not session_id or not isinstance(profile_id, str):
            return False
        value = self._read()
        profile = self._profile(value, profile_id)
        return profile is not None and profile.fingerprint == fingerprint and profile.qualified and not value.get("rollback_latched")

    def activate(self, profile_id: str, *, fingerprint: str, mode: ProfileMode | str = ProfileMode.AUTO, expected_version: int | None = None, session_id: str | None = None) -> OptimizerProfile:
        if session_id is not None and (not isinstance(session_id, str) or not session_id):
            raise ProfileError("session identity invalid")
        try:
            chosen_mode = ProfileMode(mode)
        except ValueError as exc:
            raise ProfileError("profile mode invalid") from exc
        def operation(value):
            if value.get("rollback_latched"):
                raise ProfileError("rollback latch is set")
            profile = self._profile(value, profile_id)
            if profile is None or profile.fingerprint != fingerprint or not profile.qualified:
                raise ProfileError("profile is not safely activatable")
            value["active"] = profile_id
            if chosen_mode is ProfileMode.PINNED:
                value["pinned"] = profile_id
            return profile
        return self._transaction(operation, expected_version=expected_version)

    def rollback(self, *, reason: str = "rollback", expected_version: int | None = None) -> None:
        if not isinstance(reason, str) or len(reason) > 240:
            raise ProfileError("rollback reason invalid")
        def operation(value):
            if value.get("baseline") is None:
                raise ProfileError("cannot rollback without baseline")
            value["active"] = value["baseline"]
            value["rollback_latched"] = True
            value["rollback_reason"] = reason
            return None
        self._transaction(operation, expected_version=expected_version)

    def clear_rollback_latch(self, *, fingerprint: str, qualified_profile_id: str, expected_version: int | None = None) -> None:
        def operation(value):
            profile = self._profile(value, qualified_profile_id)
            if profile is None or profile.fingerprint != fingerprint or not profile.qualified:
                raise ProfileError("only a new compatible qualified session clears rollback")
            value["rollback_latched"] = False
            return None
        self._transaction(operation, expected_version=expected_version)


# Friendly aliases used by callers that do not want to depend on the concrete
# storage class name.
Profile = OptimizerProfile
ProfileStore = AtomicProfileStore

__all__ = [
    "AtomicProfileStore", "OptimizerProfile", "Profile", "ProfileError",
    "ProfileMode", "ProfileSelection", "ProfileStore",
]
