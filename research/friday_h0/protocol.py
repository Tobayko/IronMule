"""Closed, bounded JSON and result protocol for the H0 worker.

This module deliberately contains no MLX or GPU code.  It is the trust boundary
between a parent process and the fixed ``friday_h0.worker`` module.
"""

from __future__ import annotations

import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .canonical import canonical_json_bytes, canonical_sha256
from .manifest import ManifestError, canonical_manifest_bytes, validate_manifest


PRODUCTION_TOTAL_S = 120.0
PRODUCTION_CLEANUP_S = 2.0
PRODUCTION_STDOUT_BYTES = 64 * 1024
PRODUCTION_STDERR_BYTES = 64 * 1024
PRODUCTION_RESULT_BYTES = 1 * 1024 * 1024
PRODUCTION_MANIFEST_BYTES = 64 * 1024
PRODUCTION_JSON_DEPTH = 16
RSS_SAMPLE_INTERVAL_S = 0.050

MANIFEST_FILENAME = "manifest.json"
RESULT_FILENAME = "result.json"
MANIFEST_SHA_ENV = "FRIDAY_H0_MANIFEST_SHA256"
INTERNAL_CONTROL_SLEEP_ENV = "FRIDAY_H0_INTERNAL_CONTROL_SLEEP_S"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_RESULT_KEYS = frozenset(
    {"schema_version", "run_id", "mode", "manifest_sha256", "status", "classification", "action", "error", "evidence"}
)
_ERROR_KEYS = frozenset({"code", "message"})
_STATUSES = frozenset({"completed", "invalid", "timeout", "worker_exit"})
_CLASSIFICATIONS = frozenset(
    {
        "runtime_unavailable",
        "measurement_complete",
        "promoted",
        "regression",
        "invalid",
        "invalid: correctness",
        "invalid: missing_required_field",
        "timeout",
        "worker_exit",
    }
)
_ACTIONS = frozenset({"promoted", "baseline_fallback", "not_run"})
_H0_MEASUREMENT_MODES = frozenset({"eager_baseline", "compile_comparison", "aa_gpu"})


class ProtocolError(ValueError):
    """Raised for malformed, unbounded, or semantically closed-protocol-invalid data."""


class _DuplicateKeyError(ValueError):
    pass


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _depth(value: Any, current: int = 0) -> int:
    if current > PRODUCTION_JSON_DEPTH:
        return current
    if isinstance(value, Mapping):
        return max((_depth(child, current + 1) for child in value.values()), default=current)
    if isinstance(value, list):
        return max((_depth(child, current + 1) for child in value), default=current)
    return current


def parse_capped_json(payload: bytes, *, limit: int, depth_limit: int = PRODUCTION_JSON_DEPTH) -> Any:
    """Parse exactly one strict UTF-8 JSON value within byte and depth limits."""

    if not isinstance(payload, bytes):
        raise ProtocolError("JSON payload must be bytes")
    if len(payload) > limit:
        raise ProtocolError(f"JSON payload exceeds {limit} bytes")
    try:
        text = payload.decode("utf-8", errors="strict")
        decoder = json.JSONDecoder(object_pairs_hook=_pairs_no_duplicates, parse_constant=_reject_constant)
        value, end = decoder.raw_decode(text)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, _DuplicateKeyError) as exc:
        raise ProtocolError(f"invalid strict JSON: {exc}") from exc
    if end != len(text):
        raise ProtocolError("trailing JSON bytes are not allowed")
    if _depth(value) > depth_limit:
        raise ProtocolError(f"JSON depth exceeds {depth_limit}")
    try:
        canonical = canonical_json_bytes(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"JSON is not canonicalizable: {exc}") from exc
    if len(canonical) > limit:
        raise ProtocolError(f"canonical JSON exceeds {limit} bytes")
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(child) for child in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw(child) for child in value]
    return value


@dataclass(frozen=True)
class ClosedManifest:
    """Immutable validated manifest accepted by the public supervisor function."""

    _frozen: Mapping[str, Any]
    canonical_bytes: bytes
    sha256: str

    @property
    def value(self) -> dict[str, Any]:
        return _thaw(self._frozen)

    @property
    def run_id(self) -> str:
        return str(self._frozen["run_id"])

    @property
    def mode(self) -> str:
        return str(self._frozen["mode"])


def close_manifest(manifest: Mapping[str, Any]) -> ClosedManifest:
    """Validate, canonicalize, hash, and freeze one manifest before launch."""

    try:
        validated = validate_manifest(manifest)
        payload = canonical_manifest_bytes(validated)
    except (ManifestError, TypeError, ValueError) as exc:
        raise ProtocolError(f"manifest validation failed: {exc}") from exc
    if len(payload) > PRODUCTION_MANIFEST_BYTES:
        raise ProtocolError("manifest exceeds the production byte limit")
    return ClosedManifest(_freeze(validated), payload, canonical_sha256(payload))


def _safe_open_flags() -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def read_capped_json(path: Path, *, limit: int = PRODUCTION_RESULT_BYTES) -> tuple[Any, bytes]:
    """Read one regular, non-symlink file and parse it under strict caps."""

    try:
        info = os.lstat(path)
    except OSError as exc:
        raise ProtocolError(f"cannot stat protocol file: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ProtocolError("protocol file must be a regular non-symlink file")
    if info.st_size > limit:
        raise ProtocolError(f"protocol file exceeds {limit} bytes")
    try:
        fd = os.open(path, _safe_open_flags())
        with os.fdopen(fd, "rb") as handle:
            payload = handle.read(limit + 1)
    except OSError as exc:
        raise ProtocolError(f"cannot read protocol file: {exc}") from exc
    if len(payload) > limit:
        raise ProtocolError(f"protocol file exceeds {limit} bytes")
    return parse_capped_json(payload, limit=limit), payload


def write_json_atomic(path: Path, value: Any, *, limit: int) -> bytes:
    """Write bounded canonical JSON via a private temporary file and replace."""

    try:
        payload = canonical_json_bytes(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"cannot canonicalize protocol JSON: {exc}") from exc
    if len(payload) > limit:
        raise ProtocolError(f"canonical JSON exceeds {limit} bytes")
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = -1
    temporary: str | None = None
    try:
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb", closefd=True) as handle:
            fd = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = -1
        if directory_fd >= 0:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except OSError as exc:
        raise ProtocolError(f"cannot atomically write protocol file: {exc}") from exc
    finally:
        if fd >= 0:
            os.close(fd)
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass
    return payload


def validate_result(result: Mapping[str, Any], *, manifest: ClosedManifest) -> dict[str, Any]:
    """Validate the closed common result v1 and bind it to one manifest."""

    if not isinstance(result, Mapping):
        raise ProtocolError("result must be an object")
    if set(result) != _RESULT_KEYS:
        raise ProtocolError("result has unknown or missing top-level keys")
    if result["schema_version"] != 1 or isinstance(result["schema_version"], bool):
        raise ProtocolError("result.schema_version must be exactly 1")
    if result["run_id"] != manifest.run_id or not _RUN_ID_RE.fullmatch(str(result["run_id"])):
        raise ProtocolError("result.run_id does not match the manifest")
    if result["mode"] != manifest.mode:
        raise ProtocolError("result.mode does not match the manifest")
    if not isinstance(result["manifest_sha256"], str) or not _SHA256_RE.fullmatch(result["manifest_sha256"]):
        raise ProtocolError("result.manifest_sha256 is not lowercase SHA-256")
    if result["manifest_sha256"] != manifest.sha256:
        raise ProtocolError("result manifest hash mismatch")
    for field, allowed in (("status", _STATUSES), ("classification", _CLASSIFICATIONS), ("action", _ACTIONS)):
        if result[field] not in allowed:
            raise ProtocolError(f"result.{field} is not allowlisted")
    error = result["error"]
    if error is not None:
        if not isinstance(error, Mapping) or set(error) != _ERROR_KEYS:
            raise ProtocolError("result.error must be null or {code,message}")
        for field in _ERROR_KEYS:
            if not isinstance(error[field], str) or not error[field] or len(error[field]) > 256:
                raise ProtocolError(f"result.error.{field} is not bounded")
    evidence = result["evidence"]
    if not isinstance(evidence, Mapping):
        raise ProtocolError("result.evidence must be an object")
    if _depth(result) > PRODUCTION_JSON_DEPTH:
        raise ProtocolError(f"result JSON depth exceeds {PRODUCTION_JSON_DEPTH}")
    try:
        evidence_bytes = canonical_json_bytes(evidence)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"result.evidence is not canonical JSON: {exc}") from exc
    if len(evidence_bytes) > PRODUCTION_RESULT_BYTES:
        raise ProtocolError("result.evidence exceeds the result limit")
    if result["status"] == "completed" and result["error"] is not None:
        raise ProtocolError("completed result cannot carry an error")
    classification = result["classification"]
    status = result["status"]
    action = result["action"]
    if classification == "measurement_complete":
        if manifest.mode not in _H0_MEASUREMENT_MODES:
            raise ProtocolError(
                "measurement_complete is only valid for H0 measurement modes"
            )
        if (status, action, error) != ("completed", "baseline_fallback", None):
            raise ProtocolError(
                "measurement_complete is only valid for completed/baseline_fallback without an error"
            )
    if classification == "promoted" and (status, action, error) != ("completed", "promoted", None):
        raise ProtocolError("promoted is only valid for completed/promoted results without an error")
    if classification == "regression" and (status, action, error) != ("completed", "baseline_fallback", None):
        raise ProtocolError("regression must be a completed baseline fallback without an error")
    if classification == "runtime_unavailable" and (status, action) != ("invalid", "baseline_fallback"):
        raise ProtocolError("runtime_unavailable must be an invalid baseline fallback")
    if classification in {"timeout", "worker_exit"} and (status, action) != (classification, "baseline_fallback"):
        raise ProtocolError(f"{classification} has an invalid status/action combination")
    if classification.startswith("invalid") and (status, action) != ("invalid", "baseline_fallback"):
        raise ProtocolError("invalid classifications must be invalid baseline fallbacks")
    if classification in {"runtime_unavailable", "timeout", "worker_exit", "invalid", "invalid: correctness", "invalid: missing_required_field"} and error is None:
        raise ProtocolError(f"{classification} requires a bounded error object")
    return dict(result)


def fallback_result(
    *,
    manifest: ClosedManifest,
    status: str,
    classification: str,
    code: str,
    message: str,
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create and validate a bounded baseline-fallback result."""

    result = {
        "schema_version": 1,
        "run_id": manifest.run_id,
        "mode": manifest.mode,
        "manifest_sha256": manifest.sha256,
        "status": status,
        "classification": classification,
        "action": "baseline_fallback",
        "error": {"code": code[:256], "message": message[:256]},
        "evidence": dict(evidence or {}),
    }
    return validate_result(result, manifest=manifest)


def validate_manifest_bytes(payload: bytes, *, expected: ClosedManifest) -> dict[str, Any]:
    """Require raw manifest bytes to be exactly the parent's canonical bytes."""

    value = parse_capped_json(payload, limit=PRODUCTION_MANIFEST_BYTES)
    if not isinstance(value, Mapping):
        raise ProtocolError("manifest JSON must be an object")
    try:
        canonical = canonical_manifest_bytes(value)
    except (ManifestError, TypeError, ValueError) as exc:
        raise ProtocolError(f"worker manifest validation failed: {exc}") from exc
    if canonical != expected.canonical_bytes or canonical_sha256(canonical) != expected.sha256:
        raise ProtocolError("manifest bytes or hash differ from the closed parent manifest")
    return dict(value)


def ensure_directory_0700(path: Path) -> None:
    """Create and verify a private worker CWD."""

    path.mkdir(mode=0o700, parents=False, exist_ok=False)
    os.chmod(path, 0o700)
    mode = stat.S_IMODE(os.lstat(path).st_mode)
    if mode != 0o700:
        raise ProtocolError(f"worker CWD has unexpected mode {oct(mode)}")
