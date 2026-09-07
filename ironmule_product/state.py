"""Small, local, dependency-free product state store.

This module stores configuration and model *references* only.  It never owns,
copies, or removes model weights.  Every mutation is protected by a process
lock and committed with an fsync'd replace so concurrent callers cannot lose a
registry update or observe a partially written JSON file.
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Iterator

from .errors import InvalidRequest, ModelNotFound, StateError
from .types import ModelSpec


_SCHEMA = 1
_MAX_MODELS = 256
_MAX_JSON_BYTES = 2 * 1024 * 1024
_MODES = {"desktop": 8, "server": 64}
_SETTINGS = "settings.json"
_MODELS = "models.json"
_LOCK = ".lock"


def _default_root() -> Path:
    explicit = os.environ.get("IRONMULE_HOME")
    if explicit:
        return Path(explicit).expanduser()
    return Path.home() / ".ironmule" / "product"


def _validate_model_id(model_id: str) -> str:
    if not isinstance(model_id, str) or not model_id or len(model_id) > 4096:
        raise StateError("model_id must be a bounded non-empty string")
    if "\x00" in model_id:
        raise StateError("model_id contains a NUL byte")
    return model_id


def _ensure_directory(path: Path) -> None:
    """Create a private root, or validate an already-existing private root.

    Existing directories are never repaired in place: changing permissions on a
    caller-owned path could silently alter unrelated state.
    """
    try:
        try:
            info = os.lstat(path)
        except FileNotFoundError:
            path.mkdir(parents=True, exist_ok=False, mode=0o700)
            info = os.lstat(path)
        if stat.S_ISLNK(info.st_mode):
            raise StateError("product state root must not be a symlink")
        if not stat.S_ISDIR(info.st_mode):
            raise StateError("product state root is not a directory")
        if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise StateError("product state root must be a private directory owned by the current user")
    except StateError:
        raise
    except FileExistsError:
        # A concurrent creator wins only if it created a safe root.
        _ensure_directory(path)
    except OSError as exc:
        raise StateError("product state root is unavailable") from exc


def _regular_or_missing(path: Path) -> None:
    try:
        info = os.lstat(path)
        if stat.S_ISLNK(info.st_mode):
            raise StateError(f"product state file must not be a symlink: {path.name}")
        if not stat.S_ISREG(info.st_mode):
            raise StateError(f"product state path is not a regular file: {path.name}")
        if info.st_uid != os.geteuid():
            raise StateError(f"product state file has a foreign owner: {path.name}")
        if info.st_nlink != 1:
            raise StateError(f"product state file must not be hard-linked: {path.name}")
        if stat.S_IMODE(info.st_mode) & 0o077:
            raise StateError(f"product state file must be private: {path.name}")
    except FileNotFoundError:
        return
    except StateError:
        raise
    except OSError as exc:
        raise StateError(f"cannot inspect product state file: {path.name}") from exc


def _open_state_file(path: Path, flags: int, mode: int = 0o600) -> int:
    """Open a state file without following a symlink and recheck its identity."""
    try:
        fd = os.open(path, flags | getattr(os, "O_NOFOLLOW", 0), mode)
    except OSError as exc:
        raise StateError(f"cannot open product state file: {path.name}") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise StateError(f"product state path is not a regular file: {path.name}")
        if info.st_uid != os.geteuid() or info.st_nlink != 1:
            raise StateError(f"product state file has an unsafe owner or link count: {path.name}")
        if stat.S_IMODE(info.st_mode) & 0o077:
            raise StateError(f"product state file must be private: {path.name}")
        return fd
    except Exception:
        os.close(fd)
        raise


@contextmanager
def _locked(root: Path) -> Iterator[None]:
    _ensure_directory(root)
    lock_path = root / _LOCK
    _regular_or_missing(lock_path)
    try:
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    except OSError as exc:
        raise StateError("product state lock is unavailable") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise StateError("product state lock is not a regular file")
        if info.st_uid != os.geteuid() or info.st_nlink != 1:
            raise StateError("product state lock has an unsafe owner or link count")
        if stat.S_IMODE(info.st_mode) & 0o077:
            raise StateError("product state lock must be private")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
        except OSError as exc:
            raise StateError("product state lock failed") from exc
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


@contextmanager
def _read_locked(root: Path) -> Iterator[None]:
    """Read under an existing lock without creating root or lock state."""
    try:
        info = os.lstat(root)
    except FileNotFoundError:
        yield
        return
    except OSError as exc:
        raise StateError("product state root is unavailable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise StateError("product state root must be a private directory owned by the current user")
    if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise StateError("product state root must be a private directory owned by the current user")
    lock_path = root / _LOCK
    try:
        _regular_or_missing(lock_path)
        lock_info = os.lstat(lock_path)
    except FileNotFoundError:
        yield
        return
    if lock_info is None:
        yield
        return
    try:
        fd = _open_state_file(lock_path, os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_SH)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
    except StateError:
        raise
    except OSError as exc:
        raise StateError("product state lock failed") from exc


def _read_json(path: Path) -> Any:
    _regular_or_missing(path)
    try:
        os.lstat(path)
    except FileNotFoundError:
        return None
    try:
        fd = _open_state_file(path, os.O_RDONLY)
        try:
            size = os.fstat(fd).st_size
            if size > _MAX_JSON_BYTES:
                raise StateError(f"product state file is too large: {path.name}")
            chunks: list[bytes] = []
            remaining = _MAX_JSON_BYTES + 1
            while remaining:
                chunk = os.read(fd, min(65536, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            encoded = b"".join(chunks)
            if len(encoded) > _MAX_JSON_BYTES:
                raise StateError(f"product state file is too large: {path.name}")
        finally:
            os.close(fd)
        return json.loads(encoded.decode("utf-8"))
    except StateError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StateError(f"product state file is invalid: {path.name}") from exc


def _atomic_write(path: Path, value: Any) -> None:
    _regular_or_missing(path)
    try:
        encoded = json.dumps(
            value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        ).encode("utf-8")
        if len(encoded) > _MAX_JSON_BYTES:
            raise StateError(f"product state file is too large: {path.name}")
        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
        temporary = Path(temporary_name)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(encoded)
                stream.write(b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except Exception:
            try:
                temporary.unlink()
            except OSError:
                pass
            raise
    except StateError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise StateError(f"cannot atomically write product state: {path.name}") from exc


def _settings_value(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise StateError("settings state is not an object")
    expected = {
        "schema",
        "mode",
        "exact",
        "max_pending",
        "request_timeout_s",
        "optimization_paused",
    }
    if set(value) != expected:
        raise StateError("settings state schema is invalid")
    if type(value["schema"]) is not int or value["schema"] != _SCHEMA:
        raise StateError("settings state schema or mode is invalid")
    if not isinstance(value["mode"], str) or value["mode"] not in _MODES:
        raise StateError("settings state schema or mode is invalid")
    if value["exact"] is not True or type(value["request_timeout_s"]) is not int or value["request_timeout_s"] != 120:
        raise StateError("settings state has invalid exact or timeout policy")
    if value["max_pending"] != _MODES[value["mode"]] or type(value["max_pending"]) is not int:
        raise StateError("settings state has invalid pending limit")
    if type(value["optimization_paused"]) is not bool:
        raise StateError("settings optimization_paused must be a boolean")
    return dict(value)


def _models_value(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"schema", "models"}:
        raise StateError("model registry schema is invalid")
    models = value.get("models")
    if type(value.get("schema")) is not int or value.get("schema") != _SCHEMA or not isinstance(models, list) or len(models) > _MAX_MODELS:
        raise StateError("model registry is invalid")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in models:
        if not isinstance(item, dict) or set(item) != {"model_id", "revision", "snapshot_path", "weight_bytes"}:
            raise StateError("model registry contains an invalid model")
        try:
            spec = ModelSpec.from_dict(item)
        except (InvalidRequest, TypeError, ValueError) as exc:
            raise StateError("model registry contains an invalid model") from exc
        if spec.model_id in seen:
            raise StateError("model registry contains duplicate model ids")
        if any("\x00" in value for value in (spec.model_id, spec.revision, spec.snapshot_path)):
            raise StateError("model registry contains an invalid model")
        seen.add(spec.model_id)
        result.append(spec.as_dict())
    result.sort(key=lambda item: item["model_id"])
    return {"schema": _SCHEMA, "models": result}


def _empty_models() -> dict[str, Any]:
    return {"schema": _SCHEMA, "models": []}


class ProductStore:
    """Atomic local product configuration and model-reference registry."""

    def __init__(self, root: Path | None = None):
        self.root = (Path(root).expanduser() if root is not None else _default_root())
        if not self.root.is_absolute():
            self.root = self.root.absolute()

    @property
    def _settings_path(self) -> Path:
        return self.root / _SETTINGS

    @property
    def _models_path(self) -> Path:
        return self.root / _MODELS

    def setup(self, mode: str = "desktop") -> dict[str, Any]:
        if not isinstance(mode, str) or mode not in _MODES:
            raise StateError("mode must be 'desktop' or 'server'")
        with _locked(self.root):
            current = _read_json(self._settings_path)
            models = _read_json(self._models_path)
            paused = False
            if current is not None:
                paused = _settings_value(current)["optimization_paused"]
            if models is not None:
                _models_value(models)
            value = {
                "schema": _SCHEMA,
                "mode": mode,
                "exact": True,
                "max_pending": _MODES[mode],
                "request_timeout_s": 120,
                "optimization_paused": paused,
            }
            _atomic_write(self._settings_path, value)
            if models is None:
                _atomic_write(self._models_path, _empty_models())
            return dict(value)

    def settings(self) -> dict[str, Any]:
        with _read_locked(self.root):
            value = _read_json(self._settings_path)
            if value is None:
                raise StateError("product settings are not initialized")
            return _settings_value(value)

    def register_model(self, spec: ModelSpec) -> None:
        if not isinstance(spec, ModelSpec):
            raise StateError("register_model requires ModelSpec")
        # Validate the complete metadata contract before touching the filesystem.
        _models_value({"schema": _SCHEMA, "models": [spec.as_dict()]})
        try:
            snapshot = Path(spec.snapshot_path)
            if not snapshot.is_absolute() or snapshot.is_symlink():
                raise StateError("model snapshot path must be an absolute non-symlink directory")
            if not snapshot.exists() or not snapshot.is_dir():
                raise StateError("model snapshot path is unavailable")
            if snapshot.resolve(strict=True) != snapshot:
                raise StateError("model snapshot path must resolve without symlinks")
        except StateError:
            raise
        except OSError as exc:
            raise StateError("model snapshot path cannot be inspected") from exc
        _validate_model_id(spec.model_id)
        with _locked(self.root):
            raw = _read_json(self._models_path)
            registry = _empty_models() if raw is None else _models_value(raw)
            models = [item for item in registry["models"] if item["model_id"] != spec.model_id]
            if len(models) >= _MAX_MODELS and all(
                item["model_id"] != spec.model_id for item in registry["models"]
            ):
                raise StateError("model registry is full")
            models.append(spec.as_dict())
            _atomic_write(self._models_path, _models_value({"schema": _SCHEMA, "models": models}))

    def remove_model(self, model_id: str) -> bool:
        _validate_model_id(model_id)
        with _locked(self.root):
            raw = _read_json(self._models_path)
            if raw is None:
                return False
            registry = _models_value(raw)
            models = [item for item in registry["models"] if item["model_id"] != model_id]
            if len(models) == len(registry["models"]):
                return False
            _atomic_write(self._models_path, _models_value({"schema": _SCHEMA, "models": models}))
            return True

    def models(self) -> list[ModelSpec]:
        with _read_locked(self.root):
            raw = _read_json(self._models_path)
            if raw is None:
                return []
            return [ModelSpec.from_dict(item) for item in _models_value(raw)["models"]]

    def model(self, model_id: str) -> ModelSpec:
        _validate_model_id(model_id)
        for spec in self.models():
            if spec.model_id == model_id:
                return spec
        raise ModelNotFound(f"model is not registered: {model_id}")

    def optimization_status(self) -> dict[str, Any]:
        value = self.settings()
        return {
            "schema": _SCHEMA,
            "paused": value["optimization_paused"],
            "optimization_paused": value["optimization_paused"],
            "configuration_only": True,
            "engine_started": False,
        }

    def set_optimization_paused(self, paused: bool) -> dict[str, Any]:
        if type(paused) is not bool:
            raise StateError("paused must be a boolean")
        with _locked(self.root):
            raw = _read_json(self._settings_path)
            if raw is None:
                raise StateError("product settings are not initialized")
            value = _settings_value(raw)
            value["optimization_paused"] = paused
            _atomic_write(self._settings_path, value)
            return {
                "schema": _SCHEMA,
                "paused": paused,
                "optimization_paused": paused,
                "configuration_only": True,
                "engine_started": False,
            }


__all__ = ["ProductStore"]
