"""Exact, immutable bindings for the offline Gemma optimizer.

The optimizer is deliberately conservative at this boundary.  A fingerprint is
not a best-effort description of a machine: it is the identity of the exact
runtime on which an observation may be reused.  Missing identity is represented
explicitly and makes a fingerprint out-of-distribution (OOD).

This module has no runtime or model dependency.  It only uses the standard
library and produces a versioned canonical JSON representation before hashing.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Mapping


MAX_STRING = 256
MAX_HASH = 128
SCHEMA_VERSION = 1
_MISSING = object()


class FingerprintError(ValueError):
    """A fingerprint value is malformed, unbounded, or ambiguously typed."""


def _text(value: Any, field: str, *, required: bool = False, maximum: int = MAX_STRING) -> str | None:
    if value is None:
        if required:
            raise FingerprintError(f"{field} is required")
        return None
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise FingerprintError(f"{field} must be a non-empty bounded string")
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise FingerprintError(f"{field} contains an invalid surrogate")
    return value


def _integer(value: Any, field: str, *, required: bool = False) -> int | None:
    if value is None:
        if required:
            raise FingerprintError(f"{field} is required")
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise FingerprintError(f"{field} must be a positive integer")
    # Keep hashes and canonical payloads bounded even when called by an
    # untrusted importer.
    if value > 2**63 - 1:
        raise FingerprintError(f"{field} is out of range")
    return value


def _boolean(value: Any, field: str, *, required: bool = False) -> bool | None:
    if value is None:
        if required:
            raise FingerprintError(f"{field} is required")
        return None
    if not isinstance(value, bool):
        raise FingerprintError(f"{field} must be a boolean")
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _canonical(value: Mapping[str, Any]) -> bytes:
    try:
        encoded = json.dumps(
            _thaw(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise FingerprintError("fingerprint cannot be canonically encoded") from exc
    if len(encoded) > 64 * 1024:
        raise FingerprintError("fingerprint exceeds the canonical size bound")
    return encoded


def _mapping(value: Mapping[str, Any], allowed: set[str], field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or isinstance(value, (str, bytes, bytearray)):
        raise FingerprintError(f"{field} must be an object mapping")
    if any(not isinstance(key, str) for key in value):
        raise FingerprintError(f"{field} keys must be strings")
    unknown = set(value) - allowed
    if unknown:
        raise FingerprintError(f"unknown {field} field(s): {', '.join(sorted(map(str, unknown)))}")
    return dict(value)


@dataclass(frozen=True, slots=True)
class EnvironmentFingerprint:
    """Identity of the host and software runtime.

    The short field names are intentional and are the canonical schema.  The
    longer names used by older inventory records are accepted by
    :meth:`from_mapping` and normalised into this shape.
    """

    chip: str | None = None
    gpu: str | None = None
    ram_bytes: int | None = None
    cpu_cores: int | None = None
    macos: str | None = None
    mlx: str | None = None
    mlx_lm: str | None = None
    python: str | None = None
    runtime_commit: str | None = None

    _FIELDS: ClassVar[tuple[str, ...]] = (
        "chip", "gpu", "ram_bytes", "cpu_cores", "macos", "mlx", "mlx_lm", "python", "runtime_commit"
    )

    def __post_init__(self) -> None:
        for field in ("chip", "gpu", "macos", "mlx", "mlx_lm", "python", "runtime_commit"):
            object.__setattr__(self, field, _text(getattr(self, field), field))
        object.__setattr__(self, "ram_bytes", _integer(self.ram_bytes, "ram_bytes"))
        object.__setattr__(self, "cpu_cores", _integer(self.cpu_cores, "cpu_cores"))

    @property
    def chip_family(self) -> str | None:
        return self.chip

    @property
    def gpu_family(self) -> str | None:
        return self.gpu

    @property
    def ram_total_bytes(self) -> int | None:
        return self.ram_bytes

    @property
    def core_count(self) -> int | None:
        return self.cpu_cores

    @property
    def macos_version(self) -> str | None:
        return self.macos

    @property
    def mlx_version(self) -> str | None:
        return self.mlx

    @property
    def mlx_lm_version(self) -> str | None:
        return self.mlx_lm

    @property
    def python_version(self) -> str | None:
        return self.python

    @property
    def complete(self) -> bool:
        return all(getattr(self, field) is not None for field in self._FIELDS)

    @property
    def is_exact(self) -> bool:
        return self.complete

    def as_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self._FIELDS}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EnvironmentFingerprint":
        aliases = {
            "chip_family": "chip", "gpu_family": "gpu", "ram_total_bytes": "ram_bytes",
            "core_count": "cpu_cores", "macos_version": "macos", "mlx_version": "mlx",
            "mlx_lm_version": "mlx_lm", "python_version": "python", "code_commit": "runtime_commit",
        }
        raw = dict(value)
        for old, new in aliases.items():
            if old in raw:
                if new in raw:
                    raise FingerprintError(f"duplicate environment identity: {old}/{new}")
                raw[new] = raw.pop(old)
        raw = _mapping(raw, set(cls._FIELDS), "environment")
        return cls(**raw)


@dataclass(frozen=True, slots=True)
class ModelFingerprint:
    """Exact local Gemma model, revision, architecture and tokenizer binding."""

    model_id: str | None = None
    revision: str | None = None
    manifest: str | None = None
    architecture: str | None = None
    quant_bits: int | None = None
    quant_group_size: int | None = None
    tokenizer: str | None = None

    _FIELDS: ClassVar[tuple[str, ...]] = (
        "model_id", "revision", "manifest", "architecture", "quant_bits", "quant_group_size", "tokenizer"
    )

    def __post_init__(self) -> None:
        for field in ("model_id", "revision", "manifest", "architecture", "tokenizer"):
            object.__setattr__(self, field, _text(getattr(self, field), field))
        object.__setattr__(self, "quant_bits", _integer(self.quant_bits, "quant_bits"))
        object.__setattr__(self, "quant_group_size", _integer(self.quant_group_size, "quant_group_size"))

    @property
    def manifest_hash(self) -> str | None:
        return self.manifest

    @property
    def tokenizer_version(self) -> str | None:
        return self.tokenizer

    @property
    def complete(self) -> bool:
        return all(getattr(self, field) is not None for field in self._FIELDS)

    @property
    def is_exact(self) -> bool:
        return self.complete

    def as_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self._FIELDS}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ModelFingerprint":
        aliases = {
            "model_revision": "revision", "manifest_hash": "manifest", "model_manifest": "manifest",
            "quantization_bits": "quant_bits", "group_size": "quant_group_size",
            "tokenizer_version": "tokenizer", "tokenizer_hash": "tokenizer",
        }
        raw = dict(value)
        for old, new in aliases.items():
            if old in raw:
                if new in raw:
                    raise FingerprintError(f"duplicate model identity: {old}/{new}")
                raw[new] = raw.pop(old)
        raw = _mapping(raw, set(cls._FIELDS), "model")
        return cls(**raw)


@dataclass(frozen=True, slots=True)
class WorkloadFingerprint:
    """Prompt/generator and serving-mode identity."""

    prompt_family: str | None = None
    tokenizer: str | None = None
    generator: str | None = None
    context_bucket: str | None = None
    batch: int | None = None
    concurrency: int | None = None
    max_tokens: int | None = None
    greedy: bool | None = None
    prompt_logprobs: bool | None = None
    power_mode: str | None = None
    mode: str | None = None

    _FIELDS: ClassVar[tuple[str, ...]] = (
        "prompt_family", "tokenizer", "generator", "context_bucket", "batch", "concurrency",
        "max_tokens", "greedy", "prompt_logprobs", "power_mode", "mode"
    )

    def __post_init__(self) -> None:
        for field in ("prompt_family", "tokenizer", "generator", "context_bucket", "power_mode", "mode"):
            object.__setattr__(self, field, _text(getattr(self, field), field))
        for field in ("batch", "concurrency", "max_tokens"):
            object.__setattr__(self, field, _integer(getattr(self, field), field))
        object.__setattr__(self, "greedy", _boolean(self.greedy, "greedy"))
        object.__setattr__(self, "prompt_logprobs", _boolean(self.prompt_logprobs, "prompt_logprobs"))

    @property
    def workload_mode(self) -> str | None:
        return self.mode

    @property
    def complete(self) -> bool:
        return all(getattr(self, field) is not None for field in self._FIELDS)

    @property
    def is_exact(self) -> bool:
        return self.complete

    def as_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self._FIELDS}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "WorkloadFingerprint":
        aliases = {"generator_version": "generator", "workload_mode": "mode", "context": "context_bucket"}
        raw = dict(value)
        for old, new in aliases.items():
            if old in raw:
                if new in raw:
                    raise FingerprintError(f"duplicate workload identity: {old}/{new}")
                raw[new] = raw.pop(old)
        raw = _mapping(raw, set(cls._FIELDS), "workload")
        return cls(**raw)


@dataclass(frozen=True, slots=True)
class ExactFingerprint:
    """The complete environment/model/workload identity used by the gates."""

    environment: EnvironmentFingerprint
    model: ModelFingerprint
    workload: WorkloadFingerprint

    def __post_init__(self) -> None:
        if not isinstance(self.environment, EnvironmentFingerprint):
            raise TypeError("environment must be EnvironmentFingerprint")
        if not isinstance(self.model, ModelFingerprint):
            raise TypeError("model must be ModelFingerprint")
        if not isinstance(self.workload, WorkloadFingerprint):
            raise TypeError("workload must be WorkloadFingerprint")

    @property
    def complete(self) -> bool:
        return self.environment.complete and self.model.complete and self.workload.complete

    @property
    def is_exact(self) -> bool:
        return self.complete

    @property
    def ood(self) -> bool:
        return not self.complete

    @property
    def recommendation_allowed(self) -> bool:
        return self.complete

    @property
    def ood_reason(self) -> str | None:
        missing: list[str] = []
        for name, item in (("environment", self.environment), ("model", self.model), ("workload", self.workload)):
            missing.extend(f"{name}.{field}" for field in item._FIELDS if getattr(item, field) is None)
        return None if not missing else "missing_identity:" + ",".join(missing)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "environment": self.environment.as_dict(),
            "model": self.model.as_dict(),
            "workload": self.workload.as_dict(),
        }

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical(self.as_dict())

    @property
    def fingerprint_hash(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    @property
    def digest(self) -> str:
        return self.fingerprint_hash

    @property
    def hash(self) -> str:
        return self.fingerprint_hash

    def matches(self, other: "ExactFingerprint | str") -> bool:
        if isinstance(other, ExactFingerprint):
            return self.complete and other.complete and self.fingerprint_hash == other.fingerprint_hash
        return self.complete and isinstance(other, str) and self.fingerprint_hash == other

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExactFingerprint":
        raw = _mapping(value, {"schema_version", "environment", "model", "workload"}, "fingerprint")
        version = raw.get("schema_version", SCHEMA_VERSION)
        if isinstance(version, bool) or not isinstance(version, int) or version != SCHEMA_VERSION:
            raise FingerprintError("unsupported fingerprint schema version")
        try:
            return cls(
                EnvironmentFingerprint.from_mapping(raw["environment"]),
                ModelFingerprint.from_mapping(raw["model"]),
                WorkloadFingerprint.from_mapping(raw["workload"]),
            )
        except KeyError as exc:
            raise FingerprintError(f"fingerprint missing {exc.args[0]}") from exc


# Friendly aliases used by callers and earlier design notes.
Fingerprint = ExactFingerprint
RuntimeFingerprint = ExactFingerprint
HardwareFingerprint = EnvironmentFingerprint

__all__ = [
    "SCHEMA_VERSION", "FingerprintError", "EnvironmentFingerprint", "HardwareFingerprint",
    "ModelFingerprint", "WorkloadFingerprint", "ExactFingerprint", "Fingerprint", "RuntimeFingerprint",
]
