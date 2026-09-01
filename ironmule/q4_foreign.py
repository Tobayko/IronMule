"""Offline verification boundary for foreign-machine Q4 evidence.

This module deliberately has no execution surface.  It accepts an exact
``ForeignBundleMetadata`` record from :mod:`ironmule.q4_contracts`, verifies a
user-approved Ed25519 key and the referenced local raw files, and returns a
calibration-only result.  It never imports a model/runtime, writes a profile,
or turns foreign observations into Q4 data.

The standard library does not provide Ed25519.  ``CryptographyEd25519Verifier``
therefore detects the already-installed ``cryptography`` package lazily.  If
it is not available, verification is ``VERIFIER_UNAVAILABLE``; this module intentionally
does not contain a home-grown cryptographic implementation.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import stat
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from .q4_contracts import ForeignBundleMetadata, Q4ValidationError, canonical_json


SCHEMA = "ironmule.q4_foreign_bundle.v1"
ENVELOPE_SCHEMA = "ironmule.q4_foreign_bundle_envelope.v1"
TRUST_STORE_SCHEMA = "ironmule.q4_foreign_trust_store.v1"
MAX_TRUST_STORE_BYTES = 64 * 1024
MAX_ARTIFACT_BYTES = 128 * 1024 * 1024
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_NONCE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_PUBLIC_KEY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@/-]{0,255}$")


class ForeignEvidenceStatus(str, Enum):
    """Closed statuses; only accepted ``REVALIDATION_REQUIRED`` may create a prior."""

    VERIFIED = "VERIFIED"
    MISSING = "MISSING"
    UNTRUSTED_FOREIGN_EVIDENCE = "UNTRUSTED_FOREIGN_EVIDENCE"
    OUT_OF_DOMAIN = "OUT_OF_DOMAIN"
    REVALIDATION_REQUIRED = "REVALIDATION_REQUIRED"
    REPLAY_REJECTED = "REPLAY_REJECTED"
    EXPIRED = "EXPIRED"
    PATH_RACE = "PATH_RACE"
    DUPLICATE = "DUPLICATE"
    VERIFIER_UNAVAILABLE = "VERIFIER_UNAVAILABLE"
    INCOMPLETE = "INCOMPLETE"


class ForeignVerificationError(ValueError):
    """Raised only for caller errors such as an unapproved trust store."""


class VerifierUnavailable(RuntimeError):
    """The host has no already-installed Ed25519 verifier."""


class Ed25519Verifier(Protocol):
    """Small injection boundary used by tests and by an installed verifier."""

    def verify(self, public_key: bytes, signature: bytes, message: bytes) -> bool:
        ...


class ForeignAttestationValidator(Protocol):
    """Evaluator-owned gate for repeat-level correctness/resource evidence."""

    def validate(
        self,
        envelope: "ForeignBundleEnvelope",
        artifact_root: Path,
        reviewer_record_path: Path,
    ) -> bool:
        ...


class CryptographyEd25519Verifier:
    """Use ``cryptography`` when it is already installed; never install it."""

    def verify(self, public_key: bytes, signature: bytes, message: bytes) -> bool:
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # type: ignore[import-not-found]
                Ed25519PublicKey,
            )
            from cryptography.exceptions import InvalidSignature  # type: ignore[import-not-found]
        except ImportError as exc:
            raise VerifierUnavailable("no installed Ed25519 verifier") from exc
        if len(public_key) != 32 or len(signature) != 64:
            return False
        try:
            Ed25519PublicKey.from_public_bytes(public_key).verify(signature, message)
        except (InvalidSignature, ValueError, TypeError):
            return False
        return True


def _digest(name: str, value: Any) -> str:
    if type(value) is not str or not _DIGEST.fullmatch(value):
        raise ForeignVerificationError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _text(name: str, value: Any) -> str:
    if type(value) is not str or not value.strip():
        raise ForeignVerificationError(f"{name} must be a non-empty string")
    value = value.strip()
    if "\x00" in value or value.startswith(("/", "\\")) or value.lower().startswith(("file:", "http:", "https:")):
        raise ForeignVerificationError(f"{name} contains a path or URL")
    return value


def _nonce(value: Any) -> str:
    if type(value) is not str or not _NONCE.fullmatch(value):
        raise ForeignVerificationError("nonce must be a bounded non-empty token")
    return value


def _utc(value: Any, name: str) -> datetime:
    if type(value) is not str:
        raise ForeignVerificationError(f"{name} must be an ISO-8601 UTC timestamp")
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ForeignVerificationError(f"{name} must be an ISO-8601 UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ForeignVerificationError(f"{name} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _b64decode(name: str, value: Any) -> bytes:
    if type(value) is not str:
        raise ForeignVerificationError(f"{name} must be base64")
    try:
        result = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ForeignVerificationError(f"{name} must be base64") from exc
    return result


def _key_bytes(value: Any) -> bytes:
    """Decode a trust-store key from base64, or a clearly marked hex value."""
    if type(value) is not str:
        raise ForeignVerificationError("public_key must be base64")
    try:
        result = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error):
        if len(value) == 64 and re.fullmatch(r"[0-9a-fA-F]{64}", value):
            result = bytes.fromhex(value)
        else:
            raise ForeignVerificationError("public_key must be base64")
    if len(result) != 32:
        raise ForeignVerificationError("Ed25519 public_key must be 32 bytes")
    return result


def _canonical_mapping(value: Mapping[str, Any]) -> str:
    """Canonicalize plain JSON data without allowing executable fields."""
    if not isinstance(value, Mapping):
        raise ForeignVerificationError("bundle payload must be an object")
    forbidden = {
        "callable", "callback", "command", "cmd", "argv", "exec", "executable",
        "program", "script", "shell", "subprocess", "payload", "code", "source_code",
        "python", "module", "import", "entrypoint", "url", "uri",
    }
    for key in value:
        if type(key) is not str:
            raise ForeignVerificationError("bundle payload keys must be strings")
        if key.strip().lower() in forbidden:
            raise ForeignVerificationError(f"executable field is not permitted: {key!r}")
    try:
        return canonical_json(dict(value))
    except (Q4ValidationError, TypeError, ValueError) as exc:
        raise ForeignVerificationError("bundle payload is not canonical JSON") from exc


def _metadata(bundle: ForeignBundleMetadata | Mapping[str, Any]) -> ForeignBundleMetadata:
    if isinstance(bundle, ForeignBundleMetadata):
        return bundle
    try:
        return ForeignBundleMetadata.from_dict(bundle)
    except (Q4ValidationError, TypeError, KeyError, ValueError, AttributeError) as exc:
        raise ForeignVerificationError("malformed foreign bundle metadata") from exc


def canonical_bundle_payload(
    bundle: ForeignBundleMetadata | Mapping[str, Any],
    *,
    nonce: str,
    expires_at_utc: str,
) -> bytes:
    """Return the exact bytes that the exporter must sign.

    The signature itself is excluded.  The payload includes the complete
    metadata and every artifact ID/hash pair, plus the transport nonce and
    expiry.  Artifact *bytes* are checked separately against those hashes.
    """

    metadata = _metadata(bundle)
    nonce = _nonce(nonce)
    expires = _utc(expires_at_utc, "expires_at_utc")
    values = metadata.to_dict()
    values.pop("signature", None)
    values["raw_artifacts"] = [
        {"artifact_id": item["artifact_id"], "sha256": item["sha256"], "quality": item["quality"]}
        for item in values["raw_artifacts"]
    ]
    values["nonce"] = nonce
    values["expires_at_utc"] = expires.isoformat().replace("+00:00", "Z")
    return _canonical_mapping(values).encode("utf-8")


def canonical_bundle_identity_payload(
    bundle: ForeignBundleMetadata | Mapping[str, Any],
    *,
    nonce: str,
    expires_at_utc: str,
) -> bytes:
    """Return the unsigned identity bytes used to derive ``bundle_id``.

    ``bundle_id`` is deliberately removed as well as ``signature`` so the ID
    has no circular definition.  The signature still covers the full bundle
    payload (which contains the already-derived ID), but never itself.
    """

    values = json.loads(
        canonical_bundle_payload(bundle, nonce=nonce, expires_at_utc=expires_at_utc).decode("utf-8")
    )
    values.pop("bundle_id", None)
    return _canonical_mapping(values).encode("utf-8")


def bundle_id_sha256(
    bundle: ForeignBundleMetadata | Mapping[str, Any],
    *,
    nonce: str,
    expires_at_utc: str,
) -> str:
    """Compute the non-circular ID expected in the metadata."""

    return hashlib.sha256(
        canonical_bundle_identity_payload(bundle, nonce=nonce, expires_at_utc=expires_at_utc)
    ).hexdigest()


def bundle_payload_sha256(
    bundle: ForeignBundleMetadata | Mapping[str, Any],
    *,
    nonce: str,
    expires_at_utc: str,
) -> str:
    return hashlib.sha256(
        canonical_bundle_payload(bundle, nonce=nonce, expires_at_utc=expires_at_utc)
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class TrustedPublicKey:
    """One key explicitly approved by the local user."""

    public_key_id: str
    public_key: bytes
    approved_by_user: bool
    fingerprint: str = ""
    host_classes: tuple[str, ...] = ()
    hardware_digests: tuple[str, ...] = ()
    model_digests: tuple[str, ...] = ()
    runtime_digests: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.public_key_id) is not str or not _PUBLIC_KEY_ID.fullmatch(self.public_key_id):
            raise ForeignVerificationError("public_key_id is not a safe identifier")
        if type(self.public_key) is not bytes or len(self.public_key) != 32:
            raise ForeignVerificationError("Ed25519 public_key must be 32 bytes")
        if self.approved_by_user is not True:
            raise ForeignVerificationError("trust key is not explicitly user-approved")
        computed = hashlib.sha256(self.public_key).hexdigest()
        if self.fingerprint and self.fingerprint != computed:
            raise ForeignVerificationError("trust key fingerprint does not match public key")
        object.__setattr__(self, "fingerprint", computed)
        hosts = tuple(_text("host_class", host) for host in self.host_classes)
        object.__setattr__(self, "host_classes", tuple(sorted(set(hosts))))
        for field in ("hardware_digests", "model_digests", "runtime_digests"):
            values = tuple(_digest(field, item) for item in getattr(self, field))
            object.__setattr__(self, field, tuple(sorted(set(values))))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, approved_by_user: bool) -> "TrustedPublicKey":
        expected = {
            "public_key_id", "public_key", "fingerprint", "host_classes", "hardware_digests",
            "model_digests", "runtime_digests",
        }
        if set(data) != expected:
            raise ForeignVerificationError("trust key fields differ from the frozen schema")
        for field in ("host_classes", "hardware_digests", "model_digests", "runtime_digests"):
            if not isinstance(data[field], (list, tuple)):
                raise ForeignVerificationError(f"trust key {field} must be an array")
        return cls(
            public_key_id=data["public_key_id"],
            public_key=_key_bytes(data["public_key"]),
            approved_by_user=approved_by_user,
            fingerprint=data["fingerprint"],
            host_classes=tuple(data["host_classes"]),
            hardware_digests=tuple(data["hardware_digests"]),
            model_digests=tuple(data["model_digests"]),
            runtime_digests=tuple(data["runtime_digests"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "public_key_id": self.public_key_id,
            "public_key": base64.b64encode(self.public_key).decode("ascii"),
            "fingerprint": self.fingerprint,
            "host_classes": list(self.host_classes),
            "hardware_digests": list(self.hardware_digests),
            "model_digests": list(self.model_digests),
            "runtime_digests": list(self.runtime_digests),
        }


@dataclass(frozen=True, slots=True)
class UserApprovedTrustStore:
    """Immutable, in-memory view of a local user-approved trust store."""

    keys: tuple[TrustedPublicKey, ...]
    approved_by_user: bool
    store_digest: str

    def __post_init__(self) -> None:
        if self.approved_by_user is not True:
            raise ForeignVerificationError("trust store requires explicit user approval")
        keys = tuple(self.keys)
        if not keys or any(not isinstance(item, TrustedPublicKey) for item in keys):
            raise ForeignVerificationError("trust store must contain trusted Ed25519 keys")
        if len({item.public_key_id for item in keys}) != len(keys):
            raise ForeignVerificationError("trust store has duplicate public_key_id values")
        object.__setattr__(self, "keys", tuple(sorted(keys, key=lambda item: item.public_key_id)))
        expected = hashlib.sha256(
            _canonical_mapping({"schema": TRUST_STORE_SCHEMA, "keys": [item.to_dict() for item in self.keys]}).encode()
        ).hexdigest()
        if self.store_digest and self.store_digest != expected:
            raise ForeignVerificationError("trust store digest does not match content")
        object.__setattr__(self, "store_digest", expected)

    @classmethod
    def from_keys(cls, keys: Sequence[TrustedPublicKey], *, approved_by_user: bool) -> "UserApprovedTrustStore":
        return cls(tuple(keys), approved_by_user, "")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, approved_by_user: bool) -> "UserApprovedTrustStore":
        expected = {"schema", "approved_by_user", "keys", "store_digest"}
        if set(data) != expected or data["schema"] != TRUST_STORE_SCHEMA or data["approved_by_user"] is not True:
            raise ForeignVerificationError("unsupported or unapproved trust-store schema")
        if not isinstance(data["keys"], list):
            raise ForeignVerificationError("trust store keys must be an array")
        keys = tuple(TrustedPublicKey.from_dict(item, approved_by_user=approved_by_user) for item in data["keys"])
        return cls(keys, approved_by_user, data["store_digest"])

    @classmethod
    def from_file(cls, path: str | os.PathLike[str], *, approved_by_user: bool) -> "UserApprovedTrustStore":
        if approved_by_user is not True:
            raise ForeignVerificationError("reading a trust store requires explicit user approval")
        data = _read_json_file(path, max_bytes=MAX_TRUST_STORE_BYTES, label="trust store")
        if not isinstance(data, Mapping):
            raise ForeignVerificationError("trust store must be a JSON object")
        return cls.from_dict(data, approved_by_user=True)

    def resolve(self, public_key_id: str, fingerprint: str) -> TrustedPublicKey | None:
        for key in self.keys:
            if key.public_key_id == public_key_id and key.fingerprint == fingerprint:
                return key
        return None


@dataclass(frozen=True, slots=True)
class ForeignBundleEnvelope:
    """Transport wrapper adding replay/expiry fields to exact Q4 metadata."""

    metadata: ForeignBundleMetadata
    nonce: str
    expires_at_utc: str

    SCHEMA = ENVELOPE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "nonce", _nonce(self.nonce))
        expires = _utc(self.expires_at_utc, "expires_at_utc")
        if expires <= _utc(self.metadata.exported_at_utc, "exported_at_utc"):
            raise ForeignVerificationError("bundle expiry must be after export time")
        object.__setattr__(self, "expires_at_utc", expires.isoformat().replace("+00:00", "Z"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ForeignBundleEnvelope":
        expected = {"schema", "metadata", "nonce", "expires_at_utc"}
        if set(data) != expected or data["schema"] != ENVELOPE_SCHEMA:
            raise ForeignVerificationError("unsupported foreign bundle envelope schema")
        if not isinstance(data["metadata"], Mapping):
            raise ForeignVerificationError("envelope metadata must be an object")
        return cls(ForeignBundleMetadata.from_dict(data["metadata"]), data["nonce"], data["expires_at_utc"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "metadata": self.metadata.to_dict(),
            "nonce": self.nonce,
            "expires_at_utc": self.expires_at_utc,
        }


@dataclass(frozen=True, slots=True)
class ForeignIdentity:
    """Exact identity binding for a local calibration consumer."""

    host_class: str
    hardware_digest: str
    model_digest: str
    model_manifest_digest: str
    runtime_digest: str
    code_digest: str
    workload_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "host_class", _text("host_class", self.host_class))
        for name in (
            "hardware_digest", "model_digest", "model_manifest_digest", "runtime_digest",
            "code_digest", "workload_digest",
        ):
            object.__setattr__(self, name, _digest(name, getattr(self, name)))

    @classmethod
    def from_bundle(cls, bundle: ForeignBundleMetadata) -> "ForeignIdentity":
        return cls(
            host_class=bundle.host_class,
            hardware_digest=bundle.hardware_digest,
            model_digest=bundle.model_digest,
            model_manifest_digest=bundle.model_manifest_digest,
            runtime_digest=bundle.runtime_digest,
            code_digest=bundle.code_digest,
            workload_digest=bundle.workload_digest,
        )


@dataclass(frozen=True, slots=True)
class ForeignVerificationResult:
    status: ForeignEvidenceStatus
    bundle_id: str
    reason: str
    accepted: bool
    hardware_group: str | None = None
    revalidation_required: bool = False
    can_calibrate: bool = False
    can_replace_base: bool = False
    can_enter_q4_split: bool = False

    @property
    def calibration_only(self) -> bool:
        return self.accepted and self.can_calibrate and not self.can_replace_base and not self.can_enter_q4_split


@dataclass(frozen=True, slots=True)
class ForeignCalibrationPrior:
    """Safe output of foreign verification: ordering only, never performance."""

    status: ForeignEvidenceStatus
    hardware_group: str
    ordered_action_ids: tuple[str, ...]
    requires_local_revalidation: bool
    may_replace_base: bool = False
    may_enter_q4_split: bool = False


class ReplayRegistry:
    """Process-local replay authority shared by all verifier instances.

    The registry is intentionally in-memory and never writes persistence.  A
    caller that wants replay protection across process restarts must provide a
    separately reviewed durable registry; this offline module will not invent
    one.  The convenience wrapper refuses to verify without this object (or a
    long-lived :class:`ForeignBundleVerifier`).
    """

    def __init__(self) -> None:
        self._bundles: dict[str, str] = {}
        self._nonces: set[str] = set()
        self._lock = threading.RLock()

    def lookup_bundle(self, bundle_id: str) -> str | None:
        with self._lock:
            return self._bundles.get(bundle_id)

    def nonce_seen(self, nonce: str) -> bool:
        with self._lock:
            return nonce in self._nonces

    def reserve(self, bundle_id: str, payload_digest: str, nonce: str) -> ForeignEvidenceStatus | None:
        """Atomically claim a verified bundle, returning a replay status if used."""

        with self._lock:
            existing = self._bundles.get(bundle_id)
            if existing is not None:
                return ForeignEvidenceStatus.DUPLICATE if existing == payload_digest else ForeignEvidenceStatus.REPLAY_REJECTED
            if nonce in self._nonces:
                return ForeignEvidenceStatus.REPLAY_REJECTED
            self._bundles[bundle_id] = payload_digest
            self._nonces.add(nonce)
            return None


def _safe_artifact_id(value: str) -> Path:
    if type(value) is not str or not value or "\x00" in value or "\\" in value:
        raise ForeignVerificationError("artifact_id is not a safe relative path")
    path = Path(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ForeignVerificationError("artifact_id must be a normalized relative path")
    return path


def _secure_root(root: str | os.PathLike[str]) -> Path:
    raw = Path(root)
    if not raw.is_absolute():
        raise ForeignVerificationError("artifact_root must be an absolute local path")
    try:
        resolved = raw.resolve(strict=True)
        stat_result = raw.lstat()
    except OSError as exc:
        raise ForeignVerificationError("artifact_root is missing") from exc
    # Parent aliases such as macOS ``/var -> /private/var`` are harmless; the
    # final root component itself must not be a symlink.
    if not stat.S_ISDIR(stat_result.st_mode) or raw.is_symlink():
        raise ForeignVerificationError("artifact_root must be a non-symlink directory")
    return resolved


def _read_json_file(path: str | os.PathLike[str], *, max_bytes: int, label: str) -> Any:
    raw = Path(path)
    if not raw.is_absolute():
        raise ForeignVerificationError(f"{label} path must be absolute")
    try:
        before = raw.lstat()
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise ForeignVerificationError(f"{label} is missing") from exc
    if raw.is_symlink() or not stat.S_ISREG(before.st_mode):
        raise ForeignVerificationError(f"{label} must be a non-symlink regular file")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(raw, flags)
    except OSError as exc:
        raise ForeignVerificationError(f"{label} cannot be opened safely") from exc
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (
            before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
        ):
            raise ForeignVerificationError(f"{label} changed during open")
        if opened.st_size > max_bytes:
            raise ForeignVerificationError(f"{label} exceeds the size limit")
        with os.fdopen(fd, "rb") as stream:
            fd = -1
            content = stream.read(max_bytes + 1)
            after = os.fstat(stream.fileno())
        if len(content) > max_bytes or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
            opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns
        ):
            raise ForeignVerificationError(f"{label} changed while reading")
    finally:
        if fd != -1:
            os.close(fd)
    try:
        return json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ForeignVerificationError(f"{label} is not valid UTF-8 JSON") from exc


def _hash_artifacts(bundle: ForeignBundleMetadata, root: str | os.PathLike[str]) -> None:
    secure_root = _secure_root(root)
    seen_ids: set[str] = set()
    for ref in bundle.raw_artifacts:
        artifact_id = _safe_artifact_id(ref.artifact_id)
        if ref.artifact_id in seen_ids:
            raise ForeignVerificationError("bundle contains duplicate artifact IDs")
        seen_ids.add(ref.artifact_id)
        candidate = secure_root / artifact_id
        try:
            before = candidate.lstat()
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise ForeignVerificationError(f"raw artifact is missing: {ref.artifact_id}") from exc
        if candidate.is_symlink() or not stat.S_ISREG(before.st_mode) or secure_root not in resolved.parents:
            raise ForeignVerificationError(f"raw artifact path is unsafe: {ref.artifact_id}")
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(candidate, flags)
        except OSError as exc:
            raise ForeignVerificationError(f"raw artifact cannot be opened safely: {ref.artifact_id}") from exc
        try:
            opened = os.fstat(fd)
            if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (
                before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
            ):
                raise ForeignVerificationError(f"raw artifact changed during open: {ref.artifact_id}")
            if opened.st_size > MAX_ARTIFACT_BYTES:
                raise ForeignVerificationError(f"raw artifact exceeds size limit: {ref.artifact_id}")
            digest = hashlib.sha256()
            try:
                while True:
                    chunk = os.read(fd, 1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
            except OSError as exc:
                raise ForeignVerificationError(f"raw artifact cannot be read: {ref.artifact_id}") from exc
            after = os.fstat(fd)
            if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
                opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns
            ):
                raise ForeignVerificationError(f"raw artifact changed while reading: {ref.artifact_id}")
            if digest.hexdigest() != ref.sha256:
                raise ForeignVerificationError(f"raw artifact hash mismatch: {ref.artifact_id}")
        finally:
            os.close(fd)


def _hash_referenced_file(
    root: str | os.PathLike[str], relative_path: str, expected_digest: str, *, label: str
) -> Path:
    """Hash one explicitly named record beneath ``root`` with TOCTOU checks."""

    secure_root = _secure_root(root)
    relative = _safe_artifact_id(relative_path)
    candidate = secure_root / relative
    try:
        before = candidate.lstat()
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ForeignVerificationError(f"{label} is missing") from exc
    if candidate.is_symlink() or not stat.S_ISREG(before.st_mode) or secure_root not in resolved.parents:
        raise ForeignVerificationError(f"{label} path is unsafe")
    if before.st_size > MAX_ARTIFACT_BYTES:
        raise ForeignVerificationError(f"{label} exceeds size limit")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(candidate, flags)
    except OSError as exc:
        raise ForeignVerificationError(f"{label} cannot be opened safely") from exc
    try:
        opened = os.fstat(fd)
        signature = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != signature:
            raise ForeignVerificationError(f"{label} changed during open")
        digest = hashlib.sha256()
        try:
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        except OSError as exc:
            raise ForeignVerificationError(f"{label} cannot be read") from exc
        after = os.fstat(fd)
        if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != signature:
            raise ForeignVerificationError(f"{label} changed while reading")
        if digest.hexdigest() != expected_digest:
            raise ForeignVerificationError(f"{label} hash mismatch")
    finally:
        os.close(fd)
    return candidate


class ForeignBundleVerifier:
    """Verify signed bundles once, with in-memory replay/deduplication state."""

    def __init__(
        self,
        trust_store: UserApprovedTrustStore,
        *,
        signature_verifier: Ed25519Verifier | Callable[[bytes, bytes, bytes], bool] | None = None,
        attestation_validator: ForeignAttestationValidator | Callable[[ForeignBundleEnvelope, Path, Path], bool] | None = None,
        replay_registry: ReplayRegistry | None = None,
        max_age: timedelta = timedelta(days=30),
        clock_skew: timedelta = timedelta(minutes=5),
    ) -> None:
        if not isinstance(trust_store, UserApprovedTrustStore) or trust_store.approved_by_user is not True:
            raise ForeignVerificationError("a user-approved trust store is required")
        if max_age <= timedelta(0) or clock_skew < timedelta(0):
            raise ForeignVerificationError("invalid expiry policy")
        self._trust_store = trust_store
        self._signature_verifier = signature_verifier or CryptographyEd25519Verifier()
        self._attestation_validator = attestation_validator
        self._max_age = max_age
        self._clock_skew = clock_skew
        self._replay_registry = replay_registry or ReplayRegistry()
        self._lock = threading.RLock()

    @property
    def replay_registry(self) -> ReplayRegistry:
        """The registry used by this verifier, for explicit instance sharing."""

        return self._replay_registry

    def verify(
        self,
        bundle: ForeignBundleEnvelope | ForeignBundleMetadata | Mapping[str, Any],
        *,
        artifact_root: str | os.PathLike[str] | None,
        reviewer_record_path: str | os.PathLike[str] | None = None,
        expected_identity: ForeignIdentity | Mapping[str, Any] | None = None,
        now: datetime | None = None,
    ) -> ForeignVerificationResult:
        with self._lock:
            return self._verify_locked(
                bundle,
                artifact_root=artifact_root,
                reviewer_record_path=reviewer_record_path,
                expected_identity=expected_identity,
                now=now,
            )

    def _verify_locked(
        self,
        bundle: ForeignBundleEnvelope | ForeignBundleMetadata | Mapping[str, Any],
        *,
        artifact_root: str | os.PathLike[str] | None,
        reviewer_record_path: str | os.PathLike[str] | None,
        expected_identity: ForeignIdentity | Mapping[str, Any] | None,
        now: datetime | None,
    ) -> ForeignVerificationResult:
        bundle_id = ""
        try:
            envelope = self._coerce_envelope(bundle)
            metadata = envelope.metadata
            bundle_id = metadata.bundle_id
            payload = canonical_bundle_payload(
                metadata, nonce=envelope.nonce, expires_at_utc=envelope.expires_at_utc
            )
            payload_digest = hashlib.sha256(payload).hexdigest()
            expected_bundle_id = bundle_id_sha256(
                metadata, nonce=envelope.nonce, expires_at_utc=envelope.expires_at_utc
            )
            if metadata.bundle_id != expected_bundle_id:
                raise ForeignVerificationError("bundle_id does not match canonical unsigned payload")
        except (ForeignVerificationError, Q4ValidationError, TypeError, KeyError, ValueError, AttributeError) as exc:
            return ForeignVerificationResult(ForeignEvidenceStatus.UNTRUSTED_FOREIGN_EVIDENCE, bundle_id, str(exc), False)

        seen_payload_digest = self._replay_registry.lookup_bundle(bundle_id)
        if seen_payload_digest is not None:
            if seen_payload_digest == payload_digest:
                status = ForeignEvidenceStatus.DUPLICATE
            else:
                status = ForeignEvidenceStatus.REPLAY_REJECTED
            return ForeignVerificationResult(status, bundle_id, "bundle was already seen", False)
        if self._replay_registry.nonce_seen(envelope.nonce):
            return ForeignVerificationResult(ForeignEvidenceStatus.REPLAY_REJECTED, bundle_id, "nonce was already seen", False)

        current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        exported = _utc(metadata.exported_at_utc, "exported_at_utc")
        expires = _utc(envelope.expires_at_utc, "expires_at_utc")
        if exported - self._clock_skew > current or current - exported > self._max_age:
            return ForeignVerificationResult(ForeignEvidenceStatus.EXPIRED, bundle_id, "bundle export time is outside the acceptance window", False)
        if current > expires:
            return ForeignVerificationResult(ForeignEvidenceStatus.EXPIRED, bundle_id, "bundle has expired", False)
        if expires <= exported:
            return ForeignVerificationResult(ForeignEvidenceStatus.UNTRUSTED_FOREIGN_EVIDENCE, bundle_id, "expiry is not after export", False)
        if artifact_root is None:
            return ForeignVerificationResult(ForeignEvidenceStatus.MISSING, bundle_id, "raw artifact root is required", False)
        identity = ForeignIdentity.from_bundle(metadata)
        if expected_identity is not None:
            try:
                expected = expected_identity if isinstance(expected_identity, ForeignIdentity) else ForeignIdentity(**dict(expected_identity))
            except (TypeError, ValueError, ForeignVerificationError) as exc:
                return ForeignVerificationResult(ForeignEvidenceStatus.OUT_OF_DOMAIN, bundle_id, str(exc), False)
            if identity != expected:
                return ForeignVerificationResult(ForeignEvidenceStatus.OUT_OF_DOMAIN, bundle_id, "bundle identity does not match the expected identity", False)

        key = self._trust_store.resolve(metadata.public_key_id, metadata.signer_key_fingerprint)
        if key is None:
            return ForeignVerificationResult(ForeignEvidenceStatus.UNTRUSTED_FOREIGN_EVIDENCE, bundle_id, "signing key is not user-approved", False)
        if key.host_classes and metadata.host_class not in key.host_classes:
            return ForeignVerificationResult(ForeignEvidenceStatus.OUT_OF_DOMAIN, bundle_id, "signing key is not bound to this host class", False)
        if key.hardware_digests and metadata.hardware_digest not in key.hardware_digests:
            return ForeignVerificationResult(ForeignEvidenceStatus.OUT_OF_DOMAIN, bundle_id, "signing key is not bound to this hardware", False)
        if key.model_digests and metadata.model_digest not in key.model_digests:
            return ForeignVerificationResult(ForeignEvidenceStatus.OUT_OF_DOMAIN, bundle_id, "signing key is not bound to this model", False)
        if key.runtime_digests and metadata.runtime_digest not in key.runtime_digests:
            return ForeignVerificationResult(ForeignEvidenceStatus.OUT_OF_DOMAIN, bundle_id, "signing key is not bound to this runtime", False)

        try:
            signature = _b64decode("signature", metadata.signature)
            if len(signature) != 64:
                raise ForeignVerificationError("Ed25519 signature must be 64 bytes")
            verifier = self._signature_verifier
            valid = verifier.verify(key.public_key, signature, payload) if hasattr(verifier, "verify") else verifier(key.public_key, signature, payload)  # type: ignore[misc]
        except VerifierUnavailable as exc:
            return ForeignVerificationResult(ForeignEvidenceStatus.VERIFIER_UNAVAILABLE, bundle_id, str(exc), False)
        except (ForeignVerificationError, TypeError, ValueError) as exc:
            return ForeignVerificationResult(ForeignEvidenceStatus.UNTRUSTED_FOREIGN_EVIDENCE, bundle_id, str(exc), False)
        if valid is not True:
            return ForeignVerificationResult(ForeignEvidenceStatus.UNTRUSTED_FOREIGN_EVIDENCE, bundle_id, "Ed25519 signature verification failed", False)

        if reviewer_record_path is None or self._attestation_validator is None:
            return ForeignVerificationResult(
                ForeignEvidenceStatus.INCOMPLETE,
                bundle_id,
                "independent evaluator/reviewer attestation is required",
                False,
            )

        try:
            _hash_artifacts(metadata, artifact_root)
            reviewer_path = _hash_referenced_file(
                artifact_root,
                str(reviewer_record_path),
                metadata.reviewer_record_sha256,
                label="reviewer record",
            )
        except ForeignVerificationError as exc:
            reason = str(exc)
            status = ForeignEvidenceStatus.PATH_RACE if "changed" in reason else ForeignEvidenceStatus.UNTRUSTED_FOREIGN_EVIDENCE
            return ForeignVerificationResult(status, bundle_id, reason, False)

        if not metadata.raw_artifacts or any(item.quality.value != "RAW_SAMPLES" for item in metadata.raw_artifacts):
            return ForeignVerificationResult(ForeignEvidenceStatus.UNTRUSTED_FOREIGN_EVIDENCE, bundle_id, "bundle needs complete raw-sample artifact references", False)

        try:
            attestor = self._attestation_validator
            if hasattr(attestor, "validate"):
                attested = attestor.validate(envelope, _secure_root(artifact_root), reviewer_path)  # type: ignore[union-attr]
            else:
                attested = attestor(envelope, _secure_root(artifact_root), reviewer_path)  # type: ignore[misc]
        except Exception as exc:
            return ForeignVerificationResult(ForeignEvidenceStatus.INCOMPLETE, bundle_id, f"evaluator attestation failed: {exc}", False)
        if attested is not True:
            return ForeignVerificationResult(ForeignEvidenceStatus.INCOMPLETE, bundle_id, "evaluator attestation did not confirm complete gates", False)

        replay_status = self._replay_registry.reserve(bundle_id, payload_digest, envelope.nonce)
        if replay_status is not None:
            return ForeignVerificationResult(replay_status, bundle_id, "bundle was already accepted", False)
        return ForeignVerificationResult(
            ForeignEvidenceStatus.REVALIDATION_REQUIRED,
            bundle_id,
            "verified foreign evidence is calibration-only pending a local exact-identity probe",
            True,
            hardware_group=metadata.hardware_digest,
            revalidation_required=True,
            can_calibrate=True,
            can_replace_base=False,
            can_enter_q4_split=False,
        )

    @staticmethod
    def _coerce_envelope(
        bundle: ForeignBundleEnvelope | ForeignBundleMetadata | Mapping[str, Any],
    ) -> ForeignBundleEnvelope:
        if isinstance(bundle, ForeignBundleEnvelope):
            return bundle
        if isinstance(bundle, Mapping):
            if set(bundle) == {"schema", "metadata", "nonce", "expires_at_utc"}:
                return ForeignBundleEnvelope.from_dict(bundle)
            raise ForeignVerificationError("foreign bundle needs nonce and expiry envelope")
        raise ForeignVerificationError("foreign bundle needs nonce and expiry envelope")

    def calibration_prior(
        self,
        result: ForeignVerificationResult,
        *,
        ordered_action_ids: Sequence[str] = (),
    ) -> ForeignCalibrationPrior:
        if not result.accepted or not result.calibration_only or not result.hardware_group:
            raise ForeignVerificationError("only a verified foreign result can create a calibration prior")
        ids = tuple(_text("ordered_action_id", item) for item in ordered_action_ids)
        if len(set(ids)) != len(ids):
            raise ForeignVerificationError("calibration action ordering contains duplicates")
        return ForeignCalibrationPrior(
            status=ForeignEvidenceStatus.REVALIDATION_REQUIRED,
            hardware_group=result.hardware_group,
            ordered_action_ids=ids,
            requires_local_revalidation=True,
        )


def verify_foreign_bundle(
    bundle: ForeignBundleEnvelope | Mapping[str, Any],
    trust_store: UserApprovedTrustStore | None = None,
    *,
    artifact_root: str | os.PathLike[str] | None,
    expected_identity: ForeignIdentity | Mapping[str, Any] | None = None,
    signature_verifier: Ed25519Verifier | Callable[[bytes, bytes, bytes], bool] | None = None,
    attestation_validator: ForeignAttestationValidator | Callable[[ForeignBundleEnvelope, Path, Path], bool] | None = None,
    reviewer_record_path: str | os.PathLike[str] | None = None,
    replay_registry: ReplayRegistry | None = None,
    verifier: ForeignBundleVerifier | None = None,
    now: datetime | None = None,
) -> ForeignVerificationResult:
    """Verify through a caller-owned replay authority.

    A fresh verifier per call is unsafe because it forgets previously accepted
    nonces.  Callers must therefore provide either a long-lived ``verifier``
    or a shared ``replay_registry``; otherwise the wrapper returns a closed
    ``REPLAY_REJECTED`` result without attempting acceptance.
    """

    if verifier is None and replay_registry is None:
        return ForeignVerificationResult(
            ForeignEvidenceStatus.REPLAY_REJECTED,
            "",
            "one-shot verification requires a caller-owned replay registry or long-lived verifier",
            False,
        )
    if verifier is not None and (replay_registry is not None or trust_store is not None or signature_verifier is not None or attestation_validator is not None):
        raise ForeignVerificationError("pass either a configured verifier or registry-based verifier arguments")
    if verifier is None:
        if trust_store is None:
            raise ForeignVerificationError("trust_store is required with replay_registry")
        verifier = ForeignBundleVerifier(
            trust_store,
            signature_verifier=signature_verifier,
            attestation_validator=attestation_validator,
            replay_registry=replay_registry,
        )
    return verifier.verify(
        bundle,
        artifact_root=artifact_root,
        reviewer_record_path=reviewer_record_path,
        expected_identity=expected_identity,
        now=now,
    )


__all__ = [
    "SCHEMA", "ENVELOPE_SCHEMA", "TRUST_STORE_SCHEMA", "ForeignEvidenceStatus", "ForeignVerificationError",
    "VerifierUnavailable", "Ed25519Verifier", "ForeignAttestationValidator", "CryptographyEd25519Verifier", "TrustedPublicKey",
    "UserApprovedTrustStore", "ForeignBundleEnvelope", "ForeignIdentity", "ForeignVerificationResult",
    "ForeignCalibrationPrior", "canonical_bundle_payload", "canonical_bundle_identity_payload",
    "bundle_id_sha256", "bundle_payload_sha256", "ReplayRegistry", "ForeignBundleVerifier",
    "verify_foreign_bundle",
]
