"""Fail-closed, read-only identity collection for the Friday optimizer.

The collector is deliberately a boring boundary.  It observes public macOS
commands and package metadata, verifies already materialised local identity
documents, and constructs the project's :class:`ExactFingerprint`.  It never
imports MLX, starts a model, scans a hub, measures GPU utilisation, or writes
state.  An unknown value is an error, not a guessed default.

This module is kept dependency-light on purpose.  The only project import is
the stdlib-only fingerprint value object used by the existing optimizer gates.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform as _platform
import re
import selectors
import signal
import stat
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from .fingerprint import (
    EnvironmentFingerprint,
    ExactFingerprint,
    FingerprintError,
    ModelFingerprint,
    WorkloadFingerprint,
)


SCHEMA_VERSION = 1
WORKLOAD_SCHEMA = "friday.workload_contract.v1"
Q2_PROFILE_CONTRACT_ID = "friday.q2_profiles.confirmation_ratio.v1"
Q2_PROFILE_CONTRACT_VERSION = 1
DEFAULT_MAX_OUTPUT_BYTES = 64 * 1024
DEFAULT_MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_STRING = 512
MAX_INTEGER = 2**63 - 1
_SHA_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40,64}$")


class CollectorError(ValueError):
    """Malformed identity, unavailable public fact, or unsafe source."""


class CommandError(CollectorError):
    """A bounded command could not provide one unambiguous result."""


class OutputTruncated(CommandError):
    """A command or JSON source exceeded its explicit byte bound."""


class _Runner(Protocol):
    def __call__(self, argv: Sequence[str], **kwargs: Any) -> Any:
        ...


def _canonical(value: Any) -> bytes:
    """Encode JSON deterministically and reject non-JSON values."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise CollectorError("value cannot be canonically encoded") from exc
    if len(encoded) > DEFAULT_MAX_JSON_BYTES:
        raise OutputTruncated("canonical value exceeds the JSON bound")
    return encoded


def _strict_loads(data: bytes | str, *, maximum: int = DEFAULT_MAX_JSON_BYTES) -> Any:
    """Load one bounded JSON document with duplicate-key/constant rejection."""

    raw = data.encode("utf-8") if isinstance(data, str) else data
    if not isinstance(raw, bytes) or len(raw) > maximum:
        raise OutputTruncated("JSON source exceeds its byte bound")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise CollectorError("duplicate JSON object key")
            result[key] = value
        return result

    def constant(value: str) -> Any:
        raise CollectorError(f"non-finite JSON constant: {value}")

    try:
        return json.loads(
            raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CollectorError("JSON source is invalid") from exc


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or isinstance(value, (str, bytes, bytearray)):
        raise CollectorError(f"{field} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise CollectorError(f"{field} keys must be strings")
    return dict(value)


def _text(value: Any, field: str, *, maximum: int = MAX_STRING) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise CollectorError(f"{field} must be a bounded non-empty string")
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise CollectorError(f"{field} contains an invalid surrogate")
    return value


def _positive(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= MAX_INTEGER:
        raise CollectorError(f"{field} must be a positive bounded integer")
    return value


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise CollectorError(f"{field} must be boolean")
    return value


def _sha(value: Any, field: str) -> str:
    text = _text(value, field, maximum=64).lower()
    if _SHA_RE.fullmatch(text) is None:
        raise CollectorError(f"{field} must be a 64-character SHA-256")
    return text


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_bounded_file(path: str | os.PathLike[str], *, maximum: int) -> bytes:
    """Read a regular local file through one identity-bound descriptor.

    The descriptor is opened without following symlinks and is never read past
    ``maximum + 1`` bytes.  ``fstat`` is performed both before and after the
    bounded loop, so replacement, growth, shrinkage, and metadata races all
    fail closed.  In particular, a huge file is rejected before the first read.
    """
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    cloexec = getattr(os, "O_CLOEXEC", 0)
    if not nofollow or not cloexec:
        raise CollectorError("secure_file_flags_unavailable")
    try:
        fd = os.open(os.fspath(path), os.O_RDONLY | nofollow | cloexec)
    except OSError as exc:
        raise CollectorError("source_open_failed") from exc
    try:
        try:
            before = os.fstat(fd)
        except OSError as exc:
            raise CollectorError("source_stat_failed") from exc
        if not stat.S_ISREG(before.st_mode):
            raise CollectorError("source_not_regular_file")
        if before.st_size < 0 or before.st_size > maximum:
            raise OutputTruncated("source_oversized")
        before_identity = (
            before.st_dev, before.st_ino, before.st_size,
            getattr(before, "st_mtime_ns", int(before.st_mtime * 1_000_000_000)),
        )
        chunks: list[bytes] = []
        total = 0
        while total <= maximum:
            # The extra byte is intentional: it detects growth even when the
            # initial stat reported an in-bound file.
            want = min(64 * 1024, maximum + 1 - total)
            if want <= 0:
                break
            try:
                chunk = os.read(fd, want)
            except OSError as exc:
                raise CollectorError("source_read_failed") from exc
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum:
                raise OutputTruncated("source_grew_beyond_bound")
        try:
            after = os.fstat(fd)
        except OSError as exc:
            raise CollectorError("source_stat_failed") from exc
        after_identity = (
            after.st_dev, after.st_ino, after.st_size,
            getattr(after, "st_mtime_ns", int(after.st_mtime * 1_000_000_000)),
        )
        if before_identity != after_identity:
            raise CollectorError("source_changed_during_read")
        if total != before.st_size:
            raise CollectorError("source_truncated_during_read")
        return b"".join(chunks)
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def _read_source(source: Any, *, field: str) -> tuple[dict[str, Any], str, str | None]:
    """Read a mapping/path/bytes without following an external network path."""

    path_text: str | None = None
    if isinstance(source, Mapping):
        raw = _canonical(dict(source))
        return _mapping(_strict_loads(raw), field), _sha256(raw), path_text
    if isinstance(source, (bytes, bytearray)):
        raw = bytes(source)
    elif isinstance(source, (str, os.PathLike)):
        path = Path(source)
        raw = _read_bounded_file(path, maximum=DEFAULT_MAX_JSON_BYTES)
        path_text = str(path)
    else:
        raise CollectorError(f"{field} must be a mapping, bytes, or local JSON path")
    parsed = _strict_loads(raw)
    return _mapping(parsed, field), _sha256(raw), path_text


def _normalize_power(value: Any, field: str = "power_mode") -> str:
    value = _text(value, field).strip()
    normalized = value.lower().replace("_", " ").replace("-", " ")
    aliases = {
        "ac": "AC",
        "ac power": "AC",
        "battery": "Battery",
        "battery power": "Battery",
    }
    if normalized not in aliases:
        raise CollectorError(f"{field} is unknown")
    return aliases[normalized]


def _verify_commit(value: Any) -> str:
    commit = _text(value, "runtime_commit", maximum=64).lower()
    if _COMMIT_RE.fullmatch(commit) is None:
        raise CollectorError("runtime_commit must be an exact 40- or 64-character SHA")
    return commit


@dataclass(frozen=True, slots=True)
class WorkloadContract:
    """Strict, content-free serving identity used in an exact fingerprint."""

    prompt_family: str
    tokenizer: str
    generator: str
    context_bucket: str
    batch: int
    concurrency: int
    max_tokens: int
    greedy: bool
    prompt_logprobs: bool
    power_mode: str
    mode: str

    _FIELDS = (
        "prompt_family", "tokenizer", "generator", "context_bucket", "batch",
        "concurrency", "max_tokens", "greedy", "prompt_logprobs", "power_mode", "mode",
    )

    def __post_init__(self) -> None:
        for field in ("prompt_family", "tokenizer", "generator", "context_bucket", "mode"):
            object.__setattr__(self, field, _text(getattr(self, field), field))
        object.__setattr__(self, "power_mode", _normalize_power(self.power_mode))
        for field in ("batch", "concurrency", "max_tokens"):
            object.__setattr__(self, field, _positive(getattr(self, field), field))
        object.__setattr__(self, "greedy", _boolean(self.greedy, "greedy"))
        object.__setattr__(self, "prompt_logprobs", _boolean(self.prompt_logprobs, "prompt_logprobs"))

    @property
    def schema(self) -> str:
        return WORKLOAD_SCHEMA

    def as_dict(self) -> dict[str, Any]:
        return {"schema": WORKLOAD_SCHEMA, **{field: getattr(self, field) for field in self._FIELDS}}

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical(self.as_dict())

    @property
    def contract_hash(self) -> str:
        return _sha256(self.canonical_bytes)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "WorkloadContract":
        raw = _mapping(value, "workload_contract")
        allowed = {"schema", "schema_version", "version", *cls._FIELDS}
        if set(raw) - allowed:
            raise CollectorError("workload_contract has unknown fields")
        schema = raw.pop("schema", None)
        version = raw.pop("schema_version", raw.pop("version", None))
        # Accept either spelling used by archived contracts, but never infer a
        # version from an otherwise unversioned object.
        if schema is not None and schema != WORKLOAD_SCHEMA:
            raise CollectorError("unsupported workload contract schema")
        if version is None and schema is None:
            raise CollectorError("workload_contract version is required")
        if version not in (None, 1):
            raise CollectorError("unsupported workload contract schema")
        if set(raw) != set(cls._FIELDS):
            missing = sorted(set(cls._FIELDS) - set(raw))
            raise CollectorError("workload_contract missing: " + ",".join(missing))
        return cls(**raw)

    @classmethod
    def from_json(cls, source: Any) -> "WorkloadContract":
        value, _, _ = _read_source(source, field="workload_contract")
        return cls.from_mapping(value)


@dataclass(frozen=True, slots=True)
class CurrentSnapshot:
    """One public, static observation; no private GPU counters are retained."""

    environment: EnvironmentFingerprint
    power_mode: str
    gpu_cores: int | None = None
    low_power_mode: bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.environment, EnvironmentFingerprint):
            raise TypeError("environment must be EnvironmentFingerprint")
        object.__setattr__(self, "power_mode", _normalize_power(self.power_mode))
        if self.gpu_cores is not None:
            object.__setattr__(self, "gpu_cores", _positive(self.gpu_cores, "gpu_cores"))
        if self.low_power_mode is not None:
            object.__setattr__(self, "low_power_mode", _boolean(self.low_power_mode, "low_power_mode"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "environment": self.environment.as_dict(),
            "power_mode": self.power_mode,
            "gpu_cores": self.gpu_cores,
            "low_power_mode": self.low_power_mode,
        }


@dataclass(frozen=True, slots=True)
class CollectorReport:
    """Canonical, redacted result of one collection attempt."""

    current_snapshot: CurrentSnapshot | None
    fingerprint: ExactFingerprint | None
    model_source_sha256: str | None
    workload_contract_sha256: str | None
    profile_contract_sha256: str | None
    profile_source_sha256: str | None = None
    ood_reasons: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    model_source_path: str | None = None
    workload_source_path: str | None = None
    profile_source_path: str | None = None

    @property
    def ready(self) -> bool:
        return not self.errors and self.current_snapshot is not None and self.fingerprint is not None and self.fingerprint.complete

    @property
    def blocked(self) -> bool:
        return not self.ready

    @property
    def ood(self) -> bool:
        return bool(self.ood_reasons) or not self.ready

    @property
    def fingerprint_hash(self) -> str | None:
        return self.fingerprint.fingerprint_hash if self.fingerprint is not None else None

    @property
    def source_sha256(self) -> str | None:
        """SHA of the identity source actually selected for this report."""
        return self.model_source_sha256 or self.profile_source_sha256

    @property
    def recommendation_allowed(self) -> bool:
        return self.ready and not self.ood

    @property
    def ood_reason(self) -> str | None:
        return ";".join(self.ood_reasons) if self.ood_reasons else None

    def to_dict(self) -> dict[str, Any]:
        """Return only bounded identity facts; paths and raw command output stay out."""
        return {
            "schema_version": SCHEMA_VERSION,
            "ready": self.ready,
            "blocked": self.blocked,
            "ood": self.ood,
            "recommendation_allowed": self.recommendation_allowed,
            "fingerprint_hash": self.fingerprint_hash,
            "fingerprint": self.fingerprint.as_dict() if self.fingerprint else None,
            "current_snapshot": self.current_snapshot.as_dict() if self.current_snapshot else None,
            "model_source_sha256": self.model_source_sha256,
            "source_sha256": self.source_sha256,
            "workload_contract_sha256": self.workload_contract_sha256,
            "profile_contract_sha256": self.profile_contract_sha256,
            "profile_source_sha256": self.profile_source_sha256,
            "ood_reasons": list(self.ood_reasons),
            "errors": list(self.errors),
        }

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical(self.to_dict())

    @property
    def canonical(self) -> bytes:
        return self.canonical_bytes

    @property
    def canonical_dict(self) -> dict[str, Any]:
        """Canonical redacted payload before byte encoding."""
        return self.to_dict()

    @property
    def canonical_json(self) -> str:
        return self.canonical_bytes.decode("utf-8")

    def safe_redacted(self) -> dict[str, Any]:
        result = self.to_dict()
        fingerprint = result.get("fingerprint")
        if isinstance(fingerprint, dict):
            model = fingerprint.get("model")
            if isinstance(model, dict) and isinstance(model.get("model_id"), str):
                model_id = model["model_id"]
                if "/" in model_id or "\\" in model_id:
                    model["model_id"] = "<local-model>"
        return result


def _q2_contract_hash() -> str:
    """Match the reviewed Q2 contract hash without importing the corpus module."""
    payload = {
        "contract_id": Q2_PROFILE_CONTRACT_ID,
        "version": Q2_PROFILE_CONTRACT_VERSION,
        "source_basename": "Q2_profiles.json",
        "feature_paths": [
            "*.conditions.chip", "*.conditions.fingerprint", "*.conditions.model_id",
            "*.conditions.model_identity_sha256", "*.conditions.execution_plan",
            "*.conditions.max_tokens", "*.conditions.prompt_tokens", "*.conditions.mlx",
            "*.conditions.mlx_lm", "*.conditions.quantisation.bits",
            "*.conditions.quantisation.group_size", "*.knobs.*",
        ],
        "label_paths": [
            "*.confirmation.ratio.decode_ns.median_ratio",
            "*.confirmation.ratio.prefill_ns.median_ratio",
            "*.confirmation.ratio.total_ns.median_ratio",
        ],
        "identity_paths": {
            "hardware": "*.conditions.fingerprint",
            "model": ["*.conditions.model_id", "*.conditions.model_identity_sha256", "*.conditions.quantisation_sha256"],
            "workload": ["*.conditions.execution_plan", "*.conditions.max_tokens", "*.conditions.prompt_tokens"],
            "prompt": "*.conditions.max_tokens", "time": "*.tuned_at",
        },
        "bound_constants": {"study": "Q2", "run": "q2-profile-confirmation-v1"},
        "identity_all_of": {
            "model": ["*.conditions.model_id", "*.conditions.model_revision", "*.conditions.model_identity_sha256", "*.conditions.model_manifest_sha256", "*.conditions.quantisation_sha256", "*.conditions.tokenizer_sha256"],
            "hardware": ["*.conditions.fingerprint", "*.conditions.chip"],
            "workload": ["*.conditions.execution_plan", "*.conditions.max_tokens", "*.conditions.prompt_tokens"],
        },
        "identity_one_of": {"prompt": ["*.conditions.prompt_tokens", "*.conditions.max_tokens"]},
        "identity_rules": {
            "*.conditions.chip": {"type": "string", "nonempty": True},
            "*.conditions.fingerprint": {"type": "string", "format": r"[0-9a-f]{16,64}"},
            "*.conditions.model_id": {"type": "string", "nonempty": True},
            "*.conditions.model_revision": {"type": "string", "nonempty": True},
            "*.conditions.model_identity_sha256": {"type": "string", "format": r"[0-9a-f]{64}"},
            "*.conditions.model_manifest_sha256": {"type": "string", "format": r"[0-9a-f]{64}"},
            "*.conditions.quantisation_sha256": {"type": "string", "format": r"[0-9a-f]{64}"},
            "*.conditions.tokenizer_sha256": {"type": "string", "format": r"[0-9a-f]{64}"},
            "*.conditions.execution_plan": {"type": "string", "nonempty": True},
            "*.conditions.max_tokens": {"type": "integer", "min": 1},
            "*.conditions.prompt_tokens": {"type": "integer", "min": 1},
        },
    }
    return _sha256(_canonical(payload))


Q2_PROFILE_CONTRACT_SHA256 = _q2_contract_hash()


def _identity_semantic(identity: Mapping[str, Any]) -> dict[str, Any]:
    quant = _mapping(identity.get("quantisation"), "model_identity.quantisation")
    bits = _positive(quant.get("bits"), "quantisation.bits")
    group = _positive(quant.get("group_size"), "quantisation.group_size")
    if set(quant) != {"bits", "group_size"}:
        raise CollectorError("quantisation must contain exactly bits and group_size")
    return {
        "schema": "ironmule.model_identity.v1",
        "model_id": _text(identity.get("model_id"), "model_identity.model_id"),
        "revision": _text(identity.get("revision"), "model_identity.revision"),
        "model_manifest_sha256": _sha(identity.get("model_manifest_sha256"), "model_manifest_sha256"),
        "architecture": _text(identity.get("architecture"), "model_identity.architecture"),
        "quantisation": {"bits": bits, "group_size": group},
        "quantisation_sha256": _sha(identity.get("quantisation_sha256"), "quantisation_sha256"),
        "tokenizer_sha256": _sha(identity.get("tokenizer_sha256"), "tokenizer_sha256"),
        "manifest_file_count": _positive(identity.get("manifest_file_count"), "manifest_file_count"),
        "manifest_bytes": _positive(identity.get("manifest_bytes"), "manifest_bytes"),
        "tokenizer_file_count": _positive(identity.get("tokenizer_file_count"), "tokenizer_file_count"),
    }


def _parse_model_identity(source: Any) -> tuple[ModelFingerprint, str, str | None, Mapping[str, Any]]:
    data, source_sha, path = _read_source(source, field="model_identity")
    identity = data.get("model_identity", data)
    identity = _mapping(identity, "model_identity")
    expected = {
        "schema", "model_id", "revision", "model_manifest_sha256", "architecture",
        "quantisation", "quantisation_sha256", "tokenizer_sha256", "manifest_file_count",
        "manifest_bytes", "tokenizer_file_count", "identity_sha256",
    }
    # IronMule's full, self-authenticating identity is the preferred source.
    # A compact seven-field identity is also accepted when supplied explicitly;
    # its source SHA still binds the complete local JSON document.
    compact_expected = {
        "model_id", "revision", "manifest", "architecture", "quant_bits",
        "quant_group_size", "tokenizer",
    }
    if set(identity) == compact_expected:
        model = ModelFingerprint.from_mapping(identity)
        return model, source_sha, path, {
            "identity_sha256": model_fingerprint_sha256(model),
            "model_id": model.model_id,
            "revision": model.revision,
            "model_manifest_sha256": model.manifest,
            "architecture": model.architecture,
            "quantisation": {"bits": model.quant_bits, "group_size": model.quant_group_size},
            "tokenizer_sha256": model.tokenizer,
        }
    if set(identity) != expected or identity.get("schema") != "ironmule.model_identity.v1":
        raise CollectorError("model_identity fields are missing, unknown, or wrong schema")
    semantic = _identity_semantic(identity)
    quant_digest = _sha256(_canonical(semantic["quantisation"]))
    if quant_digest != semantic["quantisation_sha256"]:
        raise CollectorError("model_identity quantisation digest does not match content")
    computed = _sha256(_canonical(semantic))
    if _sha(identity.get("identity_sha256"), "identity_sha256") != computed:
        raise CollectorError("model_identity digest does not match content")
    model = ModelFingerprint(
        model_id=semantic["model_id"], revision=semantic["revision"],
        manifest=semantic["model_manifest_sha256"], architecture=semantic["architecture"],
        quant_bits=semantic["quantisation"]["bits"],
        quant_group_size=semantic["quantisation"]["group_size"],
        tokenizer=semantic["tokenizer_sha256"],
    )
    return model, source_sha, path, identity


def model_fingerprint_sha256(model: ModelFingerprint) -> str:
    """Stable digest for the compact identity representation."""
    return _sha256(_canonical(model.as_dict()))


def _profile_entry(data: Mapping[str, Any]) -> dict[str, Any]:
    if "conditions" in data:
        return dict(data)
    entries = [value for value in data.values() if isinstance(value, Mapping)]
    if len(entries) != 1:
        raise CollectorError("Q2 profile source is missing or ambiguous")
    return dict(entries[0])


def _parse_profile(source: Any) -> tuple[ModelFingerprint, str, str | None, dict[str, Any], dict[str, Any]]:
    data, source_sha, path = _read_source(source, field="profile")
    entry = _profile_entry(data)
    conditions = _mapping(entry.get("conditions"), "profile.conditions")
    model, _, _, identity = _parse_model_identity(entry.get("model_identity"))
    # Conditions are an independently stored binding and must agree with the
    # verified identity.  Confirmation ratios are deliberately ignored.
    checks = {
        "model_id": model.model_id,
        "model_revision": model.revision,
        "model_identity_sha256": identity["identity_sha256"],
        "model_manifest_sha256": identity["model_manifest_sha256"],
        "tokenizer_sha256": identity["tokenizer_sha256"],
    }
    for name, expected in checks.items():
        if conditions.get(name) != expected:
            raise CollectorError(f"profile.conditions.{name} disagrees with verified model identity")
    quant = _mapping(conditions.get("quantisation"), "profile.conditions.quantisation")
    if quant != identity["quantisation"]:
        raise CollectorError("profile quantisation disagrees with verified model identity")
    for name in ("chip", "execution_plan", "mlx", "mlx_lm", "os", "power_source"):
        _text(conditions.get(name), f"profile.conditions.{name}")
    for name in ("max_tokens", "prompt_tokens"):
        _positive(conditions.get(name), f"profile.conditions.{name}")
    hardware = _mapping(entry.get("hardware"), "profile.hardware")
    static = _mapping(hardware.get("static"), "profile.hardware.static")
    # Keep the profile environment as a plain, bounded map.  It is compared to
    # the current snapshot below, without trusting its old fingerprint string.
    environment = {
        "chip": conditions["chip"],
        "ram_bytes": static.get("memory_bytes", conditions.get("memory_bytes")),
        "cpu_cores": static.get("cpu_logical", conditions.get("cpu_cores")),
        "macos": conditions["os"],
        "mlx": conditions["mlx"],
        "mlx_lm": conditions["mlx_lm"],
        "python": static.get("python", conditions.get("python")),
        "gpu_cores": static.get("gpu_cores", conditions.get("gpu_cores")),
        "power_mode": conditions["power_source"],
    }
    for name in ("ram_bytes", "cpu_cores", "gpu_cores"):
        _positive(environment[name], f"profile.environment.{name}")
    _text(environment["python"], "profile.environment.python")
    return model, source_sha, path, environment, conditions


class BoundedCommandRunner:
    """Run one absolute command while bounding both output streams.

    A convenience subprocess helper is intentionally not used here because it
    would buffer an untrusted child indefinitely before the caller could inspect
    its size. Pipes are drained incrementally and the complete process group is
    terminated on timeout or output-limit breach.
    """

    def __init__(
        self,
        *,
        timeout_seconds: float = 1.0,
        max_stdout_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        max_stderr_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        max_combined_bytes: int | None = None,
        popen: Callable[..., Any] = subprocess.Popen,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("command timeout must be positive")
        if min(max_stdout_bytes, max_stderr_bytes) <= 0:
            raise ValueError("command output bounds must be positive")
        combined = max_combined_bytes
        if combined is None:
            combined = max_stdout_bytes + max_stderr_bytes
        if combined <= 0:
            raise ValueError("combined command output bound must be positive")
        self.timeout_seconds = timeout_seconds
        self.max_stdout_bytes = max_stdout_bytes
        self.max_stderr_bytes = max_stderr_bytes
        self.max_combined_bytes = combined
        self._popen = popen
        self._clock = clock

    @staticmethod
    def _kill_group(process: Any, sig: int) -> None:
        try:
            os.killpg(int(process.pid), sig)
        except (AttributeError, OSError, TypeError, ValueError):
            try:
                process.send_signal(sig)
            except (AttributeError, OSError):
                pass

    def _terminate(self, process: Any) -> None:
        if process.poll() is not None:
            return
        self._kill_group(process, signal.SIGTERM)
        try:
            process.wait(timeout=0.2)
            return
        except (subprocess.TimeoutExpired, TimeoutError, OSError):
            pass
        self._kill_group(process, signal.SIGKILL)
        try:
            process.wait(timeout=0.5)
        except (subprocess.TimeoutExpired, TimeoutError, OSError):
            # A child that cannot be reaped is unsafe to reuse; report the
            # failure rather than pretending the process was cleaned up.
            pass

    def run(self, argv: Sequence[str]) -> str:
        if not argv or not isinstance(argv[0], str) or not argv[0].startswith("/"):
            raise CommandError("collector command must be absolute")
        process = None
        selector: selectors.BaseSelector | None = None
        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []
        stdout_total = stderr_total = 0
        try:
            try:
                process = self._popen(
                    list(argv), stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    shell=False, close_fds=True, start_new_session=True,
                )
            except (OSError, subprocess.SubprocessError, TypeError) as exc:
                raise CommandError(f"command_unavailable:{argv[0]}") from exc
            if process.stdout is None or process.stderr is None:
                raise CommandError("command_pipes_unavailable")
            selector = selectors.DefaultSelector()
            for stream in (process.stdout, process.stderr):
                try:
                    os.set_blocking(stream.fileno(), False)
                except (AttributeError, OSError) as exc:
                    raise CommandError("command_nonblocking_unavailable") from exc
                selector.register(stream, selectors.EVENT_READ)
            deadline = self._clock() + self.timeout_seconds
            while selector.get_map() or process.poll() is None:
                remaining = deadline - self._clock()
                if remaining <= 0:
                    self._terminate(process)
                    raise CommandError(f"command_timeout:{argv[0]}")
                events = selector.select(min(remaining, 0.05))
                if not events and process.poll() is not None:
                    # EOF readiness is normally delivered, but retain a short
                    # bounded drain opportunity for platform selector quirks.
                    continue
                for key, _ in events:
                    stream = key.fileobj
                    try:
                        chunk = os.read(stream.fileno(), 64 * 1024)
                    except (BlockingIOError, InterruptedError):
                        continue
                    except OSError as exc:
                        raise CommandError("command_pipe_read_failed") from exc
                    if not chunk:
                        try:
                            selector.unregister(stream)
                        except (KeyError, ValueError):
                            pass
                        continue
                    if stream is process.stdout:
                        stdout_total += len(chunk)
                        if stdout_total > self.max_stdout_bytes:
                            self._terminate(process)
                            raise OutputTruncated("stdout_output_truncated")
                        stdout_chunks.append(chunk)
                    else:
                        stderr_total += len(chunk)
                        if stderr_total > self.max_stderr_bytes:
                            self._terminate(process)
                            raise OutputTruncated("stderr_output_truncated")
                        stderr_chunks.append(chunk)
                    if stdout_total + stderr_total > self.max_combined_bytes:
                        self._terminate(process)
                        raise OutputTruncated("combined_output_truncated")
            try:
                returncode = process.wait(timeout=max(0.0, deadline - self._clock()))
            except subprocess.TimeoutExpired as exc:
                self._terminate(process)
                raise CommandError(f"command_timeout:{argv[0]}") from exc
            if returncode != 0:
                raise CommandError(f"command_failed:{argv[0]}")
            if stderr_total:
                raise CommandError(f"stderr_not_empty:{argv[0]}")
            try:
                return b"".join(stdout_chunks).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise CommandError("stdout_not_utf8") from exc
        finally:
            if selector is not None:
                try:
                    selector.close()
                except OSError:
                    pass
            if process is not None and process.poll() is None:
                self._terminate(process)


class Collector:
    """Read public Apple-Silicon facts and assemble one exact report."""

    COMMANDS = {
        "sysctl": "/usr/sbin/sysctl",
        "system_profiler": "/usr/sbin/system_profiler",
        "pmset": "/usr/bin/pmset",
        "sw_vers": "/usr/bin/sw_vers",
    }

    def __init__(
        self,
        *,
        runner: _Runner | None = None,
        metadata_version: Callable[[str], str] | None = None,
        platform_module: Any = _platform,
        timeout_seconds: float = 1.0,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        max_stderr_bytes: int | None = None,
        max_combined_output_bytes: int | None = None,
        popen: Callable[..., Any] = subprocess.Popen,
    ) -> None:
        if timeout_seconds <= 0 or max_output_bytes <= 0:
            raise ValueError("collector bounds must be positive")
        self._runner = runner
        self._command_runner = BoundedCommandRunner(
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=max_output_bytes,
            max_stderr_bytes=max_stderr_bytes or max_output_bytes,
            max_combined_bytes=max_combined_output_bytes,
            popen=popen,
        )
        self._metadata_version = metadata_version or importlib.metadata.version
        self._platform = platform_module
        self._timeout = timeout_seconds
        self._max_output = max_output_bytes

    def _run(self, argv: Sequence[str]) -> str:
        if not argv or not str(argv[0]).startswith("/"):
            raise CommandError("collector command must be absolute")
        if self._runner is None:
            return self._command_runner.run(argv)
        try:
            result = self._runner(
                list(argv), shell=False, check=False,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=self._timeout,
            )
        except (OSError, subprocess.SubprocessError, TypeError) as exc:
            raise CommandError(f"command_unavailable:{argv[0]}") from exc
        if isinstance(result, tuple):
            if len(result) < 2:
                raise CommandError("command result is malformed")
            returncode, stdout = result[0], result[1]
            stderr = result[2] if len(result) > 2 else ""
        else:
            returncode, stdout = getattr(result, "returncode", 0), getattr(result, "stdout", result)
            stderr = getattr(result, "stderr", "")
        if returncode != 0 or not isinstance(stdout, str) or not isinstance(stderr, str):
            raise CommandError(f"command_failed:{argv[0]}")
        if stderr:
            raise CommandError(f"stderr_not_empty:{argv[0]}")
        if len(stdout.encode("utf-8", "replace")) > self._max_output:
            raise OutputTruncated(f"output_truncated:{argv[0]}")
        return stdout

    @staticmethod
    def _one_line(value: str, field: str) -> str:
        lines = [line.strip() for line in value.splitlines() if line.strip()]
        if len(lines) != 1:
            raise CommandError(f"{field}_ambiguous")
        return _text(lines[0], field)

    def _sysctl(self, name: str) -> str:
        return self._one_line(self._run((self.COMMANDS["sysctl"], "-n", name)), name)

    def _macos(self) -> str:
        try:
            version = self._platform.mac_ver()[0]
        except (AttributeError, TypeError, IndexError):
            version = ""
        if isinstance(version, str) and version.strip():
            return _text(version.strip(), "macos")
        return self._one_line(self._run((self.COMMANDS["sw_vers"], "-productVersion")), "macos")

    def _metadata(self, distribution: str) -> str:
        try:
            return _text(self._metadata_version(distribution), distribution)
        except Exception as exc:
            raise CollectorError(f"package_version_unavailable:{distribution}") from exc

    def _gpu(self) -> tuple[str, int]:
        text = self._run((self.COMMANDS["system_profiler"], "-json", "SPDisplaysDataType"))
        payload = _strict_loads(text, maximum=self._max_output)
        root = _mapping(payload, "system_profiler")
        rows = root.get("SPDisplaysDataType")
        if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], Mapping):
            raise CommandError("gpu_ambiguous")
        row = dict(rows[0])
        vendor = row.get("spdisplays_vendor") or row.get("sppci_vendor")
        model = row.get("sppci_model") or row.get("spdisplays_device_type")
        cores = row.get("sppci_cores") or row.get("spdisplays_ndrvs_cores")
        if not isinstance(vendor, str) or "apple" not in vendor.lower():
            raise CommandError("gpu_not_apple")
        gpu = _text(model if isinstance(model, str) and model.strip() else "Apple GPU", "gpu")
        if isinstance(cores, str):
            core_match = re.fullmatch(r"\s*(\d+)(?:\s+.*)?", cores)
            cores = int(core_match.group(1)) if core_match else None
        if isinstance(cores, bool) or not isinstance(cores, int) or cores <= 0:
            raise CommandError("gpu_cores_unknown")
        return gpu, cores

    def _power(self) -> str:
        text = self._run((self.COMMANDS["pmset"], "-g", "batt"))
        matches = re.findall(r"^\s*Now drawing from ['\"]([^'\"]+)['\"]\s*$", text, re.M)
        if len(matches) != 1:
            raise CommandError("power_source_ambiguous")
        return _normalize_power(matches[0], "power_source")

    def snapshot(self, *, runtime_commit: str) -> CurrentSnapshot:
        commit = _verify_commit(runtime_commit)
        system = self._platform.system()
        machine = self._platform.machine()
        if system != "Darwin":
            raise CollectorError("unsupported_platform:non-Darwin")
        if machine not in {"arm64", "aarch64"}:
            raise CollectorError("unsupported_platform:non-Apple-Silicon")
        chip = self._sysctl("machdep.cpu.brand_string")
        try:
            ram = _positive(int(self._sysctl("hw.memsize")), "ram_bytes")
            cores = _positive(int(self._sysctl("hw.logicalcpu")), "cpu_cores")
        except (TypeError, ValueError) as exc:
            raise CommandError("hardware_value_malformed") from exc
        gpu, gpu_cores = self._gpu()
        environment = EnvironmentFingerprint(
            chip=chip, gpu=gpu, ram_bytes=ram, cpu_cores=cores,
            macos=self._macos(), mlx=self._metadata("mlx"),
            mlx_lm=self._metadata("mlx-lm"), python=_text(self._platform.python_version(), "python"),
            runtime_commit=commit,
        )
        return CurrentSnapshot(environment=environment, power_mode=self._power(), gpu_cores=gpu_cores)

    @staticmethod
    def _compare(snapshot: CurrentSnapshot, profile_environment: Mapping[str, Any]) -> tuple[str, ...]:
        current = snapshot.environment.as_dict()
        reasons: list[str] = []
        pairs = (
            ("chip", current["chip"], profile_environment.get("chip")),
            ("ram_bytes", current["ram_bytes"], profile_environment.get("ram_bytes")),
            ("cpu_cores", current["cpu_cores"], profile_environment.get("cpu_cores")),
            ("macos", current["macos"], profile_environment.get("macos")),
            ("mlx", current["mlx"], profile_environment.get("mlx")),
            ("mlx_lm", current["mlx_lm"], profile_environment.get("mlx_lm")),
            ("python", current["python"], profile_environment.get("python")),
            ("gpu_cores", snapshot.gpu_cores, profile_environment.get("gpu_cores")),
            ("power_mode", snapshot.power_mode, profile_environment.get("power_mode")),
        )
        for name, actual, expected in pairs:
            if expected is None:
                reasons.append(f"profile_environment_missing:{name}")
            elif actual != expected and not (name == "power_mode" and _normalize_power(actual) == _normalize_power(expected)):
                reasons.append(f"environment_mismatch:{name}")
        return tuple(reasons)

    def collect(
        self,
        *,
        runtime_commit: str,
        workload_contract: Any,
        model_identity: Any | None = None,
        profile: Any | None = None,
    ) -> CollectorReport:
        errors: list[str] = []
        snapshot: CurrentSnapshot | None = None
        model: ModelFingerprint | None = None
        workload: WorkloadContract | None = None
        model_sha: str | None = None
        workload_sha: str | None = None
        profile_sha: str | None = None
        model_path = workload_path = profile_path = None
        profile_env: Mapping[str, Any] | None = None
        try:
            snapshot = self.snapshot(runtime_commit=runtime_commit)
        except (CollectorError, FingerprintError, ValueError, TypeError) as exc:
            errors.append(str(exc))
        try:
            if profile is not None:
                model, profile_sha, profile_path, profile_env, _ = _parse_profile(profile)
            elif model_identity is not None:
                model, model_sha, model_path, _ = _parse_model_identity(model_identity)
            else:
                raise CollectorError("model_identity_source_missing")
        except (CollectorError, FingerprintError, ValueError, TypeError) as exc:
            errors.append(str(exc))
        try:
            raw_workload, workload_sha, workload_path = _read_source(workload_contract, field="workload_contract")
            workload = WorkloadContract.from_mapping(raw_workload)
        except (CollectorError, FingerprintError, ValueError, TypeError) as exc:
            errors.append(str(exc))
        reasons: tuple[str, ...] = ()
        if snapshot is not None and profile_env is not None:
            try:
                reasons = self._compare(snapshot, profile_env)
            except (CollectorError, ValueError, TypeError) as exc:
                errors.append(str(exc))
        if snapshot is not None and workload is not None:
            if snapshot.power_mode != workload.power_mode:
                reasons = tuple(dict.fromkeys((*reasons, "workload_power_mismatch")))
        exact: ExactFingerprint | None = None
        if snapshot is not None and model is not None and workload is not None:
            try:
                exact = ExactFingerprint(
                    snapshot.environment,
                    model,
                    WorkloadFingerprint.from_mapping(
                        {field: getattr(workload, field) for field in WorkloadContract._FIELDS}
                    ),
                )
            except (CollectorError, FingerprintError, ValueError, TypeError) as exc:
                errors.append(str(exc))
        return CollectorReport(
            current_snapshot=snapshot, fingerprint=exact,
            model_source_sha256=model_sha, workload_contract_sha256=workload_sha,
            profile_contract_sha256=Q2_PROFILE_CONTRACT_SHA256 if profile is not None else None,
            profile_source_sha256=profile_sha,
            ood_reasons=tuple(dict.fromkeys(reasons)), errors=tuple(dict.fromkeys(errors)),
            model_source_path=model_path, workload_source_path=workload_path,
            profile_source_path=profile_path,
        )


def collect_fingerprint(**kwargs: Any) -> CollectorReport:
    """Convenience wrapper preserving the fail-closed report boundary."""
    return Collector().collect(**kwargs)


# Friendly names for callers that want to use the collector as a pure
# environment probe.  They intentionally do not add a CLI or runtime hook.
EnvironmentCollector = Collector


def collect_environment(*, runtime_commit: str, **kwargs: Any) -> CurrentSnapshot:
    """Convenience read-only snapshot helper."""
    return Collector(**kwargs).snapshot(runtime_commit=runtime_commit)


__all__ = [
    "Collector", "BoundedCommandRunner", "CollectorError", "CommandError", "OutputTruncated", "CurrentSnapshot",
    "WorkloadContract", "CollectorReport", "WORKLOAD_SCHEMA", "Q2_PROFILE_CONTRACT_SHA256",
    "collect_fingerprint", "EnvironmentCollector", "collect_environment",
]
