"""Offline/read-only portfolio inventory for the four Gemma model cells.

The portfolio is deliberately an evidence *view*, not another tuner.  It
accepts already materialised cache identities and quality-labelled records and
returns a deterministic status plus the next safe, user-started measurement.
No model, cache resolver, network client, activation path, or writer is used.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

from .canonical import canonical_bytes, loads_strict
from .candidates import CandidateRegistry
from .ironmule_adapter import EXECUTION_FILE_REGISTRY_HASH


MANIFEST_SCHEMA = "friday.optimizer.portfolio-manifest.v1"
SNAPSHOT_SCHEMA = "friday.optimizer.portfolio.v1"
EVIDENCE_SCHEMA = "friday.optimizer.portfolio-evidence.v1"
MODEL_SIZES = ("1b", "4b", "12b", "27b")
MODEL_IDS = MappingProxyType({size: f"mlx-community/gemma-3-{size}-it-4bit" for size in MODEL_SIZES})
STATUSES = (
    "ready_for_experiment",
    "waiting_readiness",
    "missing_local_model",
    "insufficient_evidence",
    "unsupported",
)
QUALITY_CLASSES = ("formal", "engineering", "exploratory", "legacy_summary", "invalid", "quarantined")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_HEX_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_PRIVATE_PARTS = {"path", "prompt", "prompts", "text", "tokens", "token_ids", "output", "response", "log", "logs", "stdout", "stderr", "pid"}


class PortfolioError(ValueError):
    """Malformed, ambiguous, or unsafe portfolio input."""


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise PortfolioError(f"{field}_invalid")
    return value


def _text(value: Any, field: str, *, pattern: re.Pattern[str] | None = None) -> str:
    if not isinstance(value, str) or not value or len(value) > 512 or "\x00" in value:
        raise PortfolioError(f"{field}_invalid")
    if pattern is not None and pattern.fullmatch(value) is None:
        raise PortfolioError(f"{field}_invalid")
    return value


def _strict_mapping(value: Any, field: str, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise PortfolioError(f"{field}_must_be_object")
    unknown = set(value) - allowed
    if unknown:
        raise PortfolioError(f"{field}_unknown_field")
    return dict(value)


def _private_free(value: Any, *, depth: int = 0) -> None:
    if depth > 8:
        raise PortfolioError("portfolio_input_too_deep")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise PortfolioError("portfolio_input_key_invalid")
            parts = {part for part in re.split(r"[^a-z0-9]+", key.lower()) if part}
            is_digest = key.lower().endswith(("_hash", "_sha256", "hash", "sha256"))
            if parts & _PRIVATE_PARTS and not is_digest:
                raise PortfolioError("portfolio_private_field")
            _private_free(item, depth=depth + 1)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _private_free(item, depth=depth + 1)


def _read_source(source: Any) -> tuple[Mapping[str, Any], bytes]:
    if isinstance(source, Mapping):
        raw = canonical_bytes(dict(source), max_bytes=4 * 1024 * 1024, max_depth=32, max_items=100_000)
        return dict(source), raw
    if isinstance(source, (bytes, bytearray)):
        raw = bytes(source)
    elif isinstance(source, (str, os.PathLike)):
        path = Path(source)
        current = Path(path.anchor or os.sep)
        parts = path.parts[1:] if path.anchor else path.parts
        for part in parts[:-1]:
            current /= part
            try:
                if current.is_symlink():
                    raise PortfolioError("portfolio_manifest_source_invalid")
            except OSError as exc:
                raise PortfolioError("portfolio_manifest_source_invalid") from exc
        if path.suffix.lower() != ".json" or path.is_symlink() or not path.is_file():
            raise PortfolioError("portfolio_manifest_source_invalid")
        try:
            before = path.lstat()
            if not stat.S_ISREG(before.st_mode) or before.st_size > 4 * 1024 * 1024:
                raise PortfolioError("portfolio_manifest_source_invalid")
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0))
        except (OSError, PortfolioError) as exc:
            if isinstance(exc, PortfolioError):
                raise
            raise PortfolioError("portfolio_manifest_source_invalid") from exc
        try:
            opened = os.fstat(fd)
            if not stat.S_ISREG(opened.st_mode) or opened.st_size > 4 * 1024 * 1024:
                raise PortfolioError("portfolio_manifest_source_invalid")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(fd, min(1024 * 1024, 4 * 1024 * 1024 - sum(map(len, chunks)) + 1))
                if not chunk:
                    break
                chunks.append(chunk)
                if sum(map(len, chunks)) > 4 * 1024 * 1024:
                    raise PortfolioError("portfolio_manifest_source_invalid")
            after = os.fstat(fd)
            current = path.lstat()
            identity = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
            if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or identity != (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns):
                raise PortfolioError("portfolio_manifest_source_changed")
            raw = b"".join(chunks)
        finally:
            os.close(fd)
    else:
        raise PortfolioError("portfolio_manifest_source_invalid")
    try:
        value = loads_strict(raw, max_bytes=4 * 1024 * 1024, max_depth=32, max_items=100_000)
    except Exception as exc:
        raise PortfolioError("portfolio_manifest_json_invalid") from exc
    if not isinstance(value, Mapping):
        raise PortfolioError("portfolio_manifest_must_be_object")
    if canonical_bytes(value, max_bytes=4 * 1024 * 1024, max_depth=32, max_items=100_000) != raw:
        raise PortfolioError("portfolio_manifest_not_canonical")
    return value, raw


def _readiness(value: Any) -> str:
    if isinstance(value, Mapping):
        value = value.get("status")
    if value not in {"ready", "blocked", "unknown"}:
        raise PortfolioError("readiness_invalid")
    return str(value)


def _identity(value: Any, *, size: str) -> dict[str, Any]:
    allowed = {"revision", "manifest_sha256", "manifest", "tokenizer_sha256", "tokenizer", "architecture", "quant_bits", "quant_group_size", "identity_sha256", "identity_document_sha256"}
    identity = _strict_mapping(value, "identity", allowed)
    required = ("revision", "architecture", "identity_sha256", "identity_document_sha256")
    if any(key not in identity for key in required):
        raise PortfolioError("identity_incomplete")
    result = {
        "revision": _text(identity["revision"], "identity.revision"),
        "manifest_sha256": _sha(identity.get("manifest_sha256") or identity.get("manifest"), "identity.manifest_sha256"),
        "tokenizer_sha256": _sha(identity.get("tokenizer_sha256") or identity.get("tokenizer"), "identity.tokenizer_sha256"),
        "architecture": _text(identity["architecture"], "identity.architecture"),
        "quant_bits": identity.get("quant_bits", 4),
        "quant_group_size": identity.get("quant_group_size", 64),
        "identity_sha256": _sha(identity["identity_sha256"], "identity.identity_sha256"),
        "identity_document_sha256": _sha(identity["identity_document_sha256"], "identity.identity_document_sha256"),
    }
    if (size == "1b" and result["architecture"] not in {"gemma3", "gemma3_text"}) or (size != "1b" and result["architecture"] != "gemma3"):
        raise PortfolioError("identity_architecture_unsupported")
    if result["quant_bits"] != 4 or result["quant_group_size"] != 64:
        raise PortfolioError("identity_quantization_unsupported")
    if size not in MODEL_SIZES:
        raise PortfolioError("model_size_unsupported")
    return result


def _evidence(value: Any, *, model: Mapping[str, Any], model_hash: str) -> tuple[dict[str, Any], ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)) or len(value) > 10_000:
        raise PortfolioError("evidence_must_be_bounded_array")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        allowed = {"sha256", "quality", "model_identity_hash", "hardware", "hardware_hash", "workload", "workload_hash", "identity", "correctness", "resources", "raw_pairs", "status"}
        row = _strict_mapping(item, f"evidence_{index}", allowed)
        digest = _sha(row.get("sha256"), f"evidence_{index}.sha256")
        quality = row.get("quality")
        if quality not in QUALITY_CLASSES:
            raise PortfolioError(f"evidence_{index}_quality_invalid")
        identity_hash = row.get("model_identity_hash")
        if identity_hash is not None:
            _sha(identity_hash, f"evidence_{index}.model_identity_hash")
        if identity_hash is not None and identity_hash != model_hash:
            # Keep the row in the audit, but it can never be usable for this
            # model cell.  The mismatch is represented explicitly below.
            row["_identity_mismatch"] = True
        for name in ("hardware", "hardware_hash", "workload", "workload_hash"):
            if name in row and row[name] is not None:
                _sha(row[name], f"evidence_{index}.{name}")
        correctness = row.get("correctness")
        resources = row.get("resources")
        raw_pairs = row.get("raw_pairs")
        if correctness is not None and not isinstance(correctness, (bool, Mapping)):
            raise PortfolioError(f"evidence_{index}_correctness_invalid")
        if resources is not None and not isinstance(resources, (bool, Mapping)):
            raise PortfolioError(f"evidence_{index}_resources_invalid")
        if raw_pairs is not None and not isinstance(raw_pairs, (int, list, tuple)):
            raise PortfolioError(f"evidence_{index}_raw_pairs_invalid")
        result.append({key: item for key, item in row.items() if not key.startswith("_")})
        result[-1]["_usable"] = _usable_evidence(row, model=model, model_hash=model_hash)
        result[-1]["sha256"] = digest
    return tuple(result)


def _usable_evidence(row: Mapping[str, Any], *, model: Mapping[str, Any], model_hash: str) -> bool:
    if row.get("quality") not in {"formal", "engineering"}:
        return False
    if row.get("_identity_mismatch"):
        return False
    if row.get("model_identity_hash") != model_hash:
        return False
    # Hardware/workload hashes must be explicit and equal to the model cell's
    # bindings; the manifest supplies these without storing paths or prose.
    if not model.get("hardware_hash") or row.get("hardware_hash", row.get("hardware")) != model.get("hardware_hash"):
        return False
    if not model.get("workload_hash") or row.get("workload_hash", row.get("workload")) != model.get("workload_hash"):
        return False
    correctness = row.get("correctness")
    resources = row.get("resources")
    if correctness is not True and not (isinstance(correctness, Mapping) and correctness.get("passed") is True):
        return False
    if resources is not True and not (isinstance(resources, Mapping) and resources.get("passed") is True):
        return False
    pairs = row.get("raw_pairs")
    if isinstance(pairs, bool) or not isinstance(pairs, (int, list, tuple)):
        return False
    return (pairs >= 3 if isinstance(pairs, int) else 3 <= len(pairs) <= 6)


@dataclass(frozen=True, slots=True)
class PortfolioModel:
    size: str
    model_id: str
    cache_status: str
    identity: Mapping[str, Any] | None
    evidence: tuple[Mapping[str, Any], ...]
    preregistration: Mapping[str, Any] | None
    readiness: str
    hardware_hash: str | None = None
    workload_hash: str | None = None
    identity_error: bool = False
    readiness_error: bool = False
    evidence_error: bool = False
    preregistration_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class PortfolioEntry:
    size: str
    model_id: str
    status: str
    identity_hash: str | None
    identity_document_sha256: str | None
    preregistration_sha256: str | None
    revision: str | None
    manifest_sha256: str | None
    tokenizer_sha256: str | None
    hardware_hash: str | None
    workload_hash: str | None
    evidence_counts: Mapping[str, int]
    usable_records: int
    evidence_sha256: str
    blocked_reasons: tuple[str, ...]
    next_safe_measurement: Mapping[str, Any] | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "model_id": self.model_id,
            "status": self.status,
            "identity_hash": self.identity_hash,
            "identity_document_sha256": self.identity_document_sha256,
            "preregistration_sha256": self.preregistration_sha256,
            "revision": self.revision,
            "manifest_sha256": self.manifest_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "hardware_hash": self.hardware_hash,
            "workload_hash": self.workload_hash,
            "evidence_counts": dict(self.evidence_counts),
            "usable_records": self.usable_records,
            "evidence_sha256": self.evidence_sha256,
            "blocked_reasons": list(self.blocked_reasons),
            "next_safe_measurement": None if self.next_safe_measurement is None else dict(self.next_safe_measurement),
        }


@dataclass(frozen=True, slots=True)
class PortfolioManifest:
    models: tuple[PortfolioModel, ...]
    manifest_sha256: str
    cache_inventory_sha256: str
    evidence_inventory_sha256: str
    readiness_evidence_sha256: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PortfolioManifest":
        raw = _strict_mapping(value, "portfolio_manifest", {"schema", "version", "models", "candidate_id", "registry_hash", "cache_inventory_sha256", "evidence_inventory_sha256", "readiness_evidence_sha256"})
        if raw.get("schema") != MANIFEST_SCHEMA or raw.get("version", 1) != 1:
            raise PortfolioError("portfolio_manifest_schema_invalid")
        models_raw = raw.get("models")
        if not isinstance(models_raw, (list, tuple)) or len(models_raw) != len(MODEL_SIZES):
            raise PortfolioError("portfolio_manifest_models_invalid")
        seen: set[str] = set()
        models: list[PortfolioModel] = []
        registry = CandidateRegistry()
        if raw.get("candidate_id") != "combined_core_profile":
            raise PortfolioError("portfolio_candidate_unsupported")
        registry_hash = raw.get("registry_hash")
        if registry_hash is None:
            raise PortfolioError("portfolio_registry_missing")
        if registry_hash != registry.registry_hash:
            raise PortfolioError("portfolio_registry_mismatch")
        cache_inventory_hash = _sha(raw.get("cache_inventory_sha256"), "cache_inventory_sha256")
        evidence_inventory_hash = _sha(raw.get("evidence_inventory_sha256"), "evidence_inventory_sha256")
        readiness_inventory_hash = raw.get("readiness_evidence_sha256")
        if readiness_inventory_hash is not None:
            readiness_inventory_hash = _sha(readiness_inventory_hash, "readiness_evidence_sha256")
        for index, item in enumerate(models_raw):
            allowed = {"size", "model_id", "cache_status", "identity", "identity_document_sha256", "evidence", "preregistration", "preregistration_sha256", "readiness", "hardware_hash", "workload_hash"}
            row = _strict_mapping(item, f"model_{index}", allowed)
            size = row.get("size")
            if size not in MODEL_SIZES or size in seen:
                raise PortfolioError("portfolio_model_size_invalid")
            seen.add(size)
            if row.get("model_id") != MODEL_IDS[size]:
                raise PortfolioError("portfolio_model_id_invalid")
            cache_status = row.get("cache_status")
            cache_status_invalid = cache_status not in {"verified", "missing", "ambiguous"}
            if cache_status_invalid:
                cache_status = "invalid"
            identity = None
            identity_error = False
            if row.get("identity") is not None:
                try:
                    identity = _identity(row["identity"], size=size)
                except PortfolioError:
                    identity_error = True
            readiness_error = False
            try:
                readiness = _readiness(row.get("readiness", "unknown"))
            except PortfolioError:
                readiness = "unknown"
                readiness_error = True
            hardware_hash = row.get("hardware_hash")
            workload_hash = row.get("workload_hash")
            if hardware_hash is not None:
                hardware_hash = _sha(hardware_hash, "hardware_hash")
            if workload_hash is not None:
                workload_hash = _sha(workload_hash, "workload_hash")
            if identity is not None:
                identity["hardware_hash"] = hardware_hash
                identity["workload_hash"] = workload_hash
            model_hash = None if identity is None else identity["identity_sha256"]
            evidence_error = False
            try:
                # Preserve coverage even when the model identity is missing;
                # the evidence row remains permanently unusable in that case.
                evidence = _evidence(row.get("evidence", []), model=identity or {}, model_hash=model_hash or "")
            except PortfolioError:
                evidence = ()
                evidence_error = True
            prereg = row.get("preregistration")
            if prereg is not None:
                prereg = _strict_mapping(prereg, f"model_{index}.preregistration", {"schema", "status", "fingerprint_hash", "model_identity_sha256", "model_manifest_sha256", "tokenizer_sha256", "workload_contract_sha256", "candidate_id", "optimizer_head", "code_manifest_sha256", "registry_sha256", "result_schema", "start_authorization_schema", "duration_minutes", "stages", "preregistration_sha256"})
            identity_doc = row.get("identity_document_sha256") or (None if identity is None else identity.get("identity_document_sha256"))
            if identity_doc is not None:
                try:
                    identity_doc = _sha(identity_doc, f"model_{index}.identity_document_sha256")
                except PortfolioError:
                    identity_error = True
            elif identity is not None:
                identity_error = True
            prereg_doc = row.get("preregistration_sha256") or (None if prereg is None else prereg.get("preregistration_sha256"))
            if prereg_doc is not None:
                try:
                    prereg_doc = _sha(prereg_doc, f"model_{index}.preregistration_sha256")
                except PortfolioError:
                    evidence_error = True
            elif prereg is not None:
                evidence_error = True
            models.append(PortfolioModel(size, row["model_id"], cache_status, None if identity is None else MappingProxyType(identity), evidence, None if prereg is None else MappingProxyType(dict(prereg)), readiness, hardware_hash, workload_hash, identity_error or cache_status_invalid, readiness_error, evidence_error, prereg_doc))
        if seen != set(MODEL_SIZES):
            raise PortfolioError("portfolio_model_sizes_incomplete")
        # The input identity is canonicalized as its own source commitment; it
        # is never emitted, and thus cannot leak any local source path.
        _private_free(raw)
        canonical = canonical_bytes(raw, max_bytes=4 * 1024 * 1024, max_depth=32, max_items=100_000)
        return cls(tuple(sorted(models, key=lambda item: MODEL_SIZES.index(item.size))), hashlib.sha256(canonical).hexdigest(), cache_inventory_hash, evidence_inventory_hash, readiness_inventory_hash)

    @classmethod
    def from_json(cls, source: Any) -> "PortfolioManifest":
        value, _ = _read_source(source)
        return cls.from_mapping(value)

    load = from_json


def _prereg_ready(model: PortfolioModel) -> bool:
    p = model.preregistration
    if not isinstance(p, Mapping) or p.get("status") != "SEALED" or p.get("schema") != "friday.optimizer.preregistration.v1":
        return False
    if p.get("candidate_id") != "combined_core_profile" or p.get("result_schema") != "friday.optimizer.session-result.v1":
        return False
    try:
        for key in ("fingerprint_hash", "model_identity_sha256", "workload_contract_sha256", "code_manifest_sha256", "preregistration_sha256"):
            _sha(p.get(key), "preregistration." + key)
    except PortfolioError:
        return False
    if p.get("model_identity_sha256") != (model.identity or {}).get("identity_sha256") or not model.identity.get("identity_document_sha256") or model.preregistration_sha256 is None or p.get("preregistration_sha256") != model.preregistration_sha256:
        return False
    if p.get("workload_contract_sha256") != model.workload_hash:
        return False
    if p.get("model_manifest_sha256") != (model.identity or {}).get("manifest_sha256") or p.get("tokenizer_sha256") != (model.identity or {}).get("tokenizer_sha256"):
        return False
    if p.get("start_authorization_schema") != "friday.optimizer.start-authorization.v1":
        return False
    try:
        _sha(p.get("registry_sha256"), "preregistration.registry_sha256")
    except PortfolioError:
        return False
    if p.get("registry_sha256") != EXECUTION_FILE_REGISTRY_HASH:
        return False
    duration = p.get("duration_minutes")
    if isinstance(duration, bool) or not isinstance(duration, int) or not 5 <= duration <= 30:
        return False
    stages = p.get("stages")
    return isinstance(stages, (list, tuple)) and tuple(stages) == ("calibrate", "test") and isinstance(p.get("optimizer_head"), str) and bool(_HEX_COMMIT.fullmatch(p["optimizer_head"])) and isinstance(p.get("code_manifest_sha256"), str) and bool(_SHA.fullmatch(p["code_manifest_sha256"]))


def _entry(model: PortfolioModel) -> PortfolioEntry:
    counts = {quality: 0 for quality in QUALITY_CLASSES}
    for row in model.evidence:
        counts[str(row["quality"])] += 1
    usable = sum(1 for row in model.evidence if row.get("_usable") is True)
    reasons: list[str] = []
    identity = model.identity
    if model.cache_status == "missing":
        status = "missing_local_model"
        reasons.append("local_model_missing")
    elif model.cache_status == "ambiguous" or model.identity_error or model.readiness_error or model.evidence_error:
        status = "unsupported"
        reasons.append("local_cache_ambiguous" if model.cache_status == "ambiguous" else "model_input_malformed")
    elif identity is None:
        status = "unsupported"
        reasons.append("exact_identity_missing")
    elif not _prereg_ready(model):
        status = "insufficient_evidence"
        reasons.append("sealed_preregistration_missing_or_mismatched")
    elif model.readiness != "ready":
        status = "waiting_readiness"
        reasons.append("readiness_" + model.readiness)
    else:
        status = "ready_for_experiment"
    next_point: Mapping[str, Any] | None
    if status == "waiting_readiness":
        next_point = {"action": "recheck_readiness", "size": model.size, "requires_user_start": False}
    elif status == "insufficient_evidence":
        next_point = {"action": "new_preregistered_evidence", "size": model.size, "requires_user_start": True}
    elif status == "ready_for_experiment":
        next_point = {"action": "aa_calibration", "size": model.size, "requires_user_start": True}
    else:
        next_point = None
    evidence_hash = hashlib.sha256(canonical_bytes([{"sha256": row["sha256"], "quality": row["quality"], "usable": bool(row.get("_usable"))} for row in model.evidence], max_bytes=256 * 1024)).hexdigest()
    return PortfolioEntry(model.size, model.model_id, status, None if identity is None else identity["identity_sha256"], None if identity is None else identity["identity_document_sha256"], model.preregistration_sha256, None if identity is None else identity["revision"], None if identity is None else identity["manifest_sha256"], None if identity is None else identity["tokenizer_sha256"], model.hardware_hash, model.workload_hash, counts, usable, evidence_hash, tuple(reasons), next_point)


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    entries: tuple[PortfolioEntry, ...]
    registry_hash: str
    manifest_sha256: str
    snapshot_sha256: str
    next_safe_measurement: Mapping[str, Any] | None
    cache_inventory_sha256: str
    evidence_inventory_sha256: str
    readiness_evidence_sha256: str | None = None

    @property
    def hash(self) -> str:
        return self.snapshot_sha256

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SNAPSHOT_SCHEMA,
            "models": [entry.as_dict() for entry in self.entries],
            "registry_hash": self.registry_hash,
            "manifest_sha256": self.manifest_sha256,
            "cache_inventory_sha256": self.cache_inventory_sha256,
            "evidence_inventory_sha256": self.evidence_inventory_sha256,
            "readiness_evidence_sha256": self.readiness_evidence_sha256,
            "snapshot_sha256": self.snapshot_sha256,
            "next_safe_measurement": None if self.next_safe_measurement is None else dict(self.next_safe_measurement),
            "no_model_load": True,
            "no_download": True,
            "no_activation": True,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.as_dict(), max_bytes=4 * 1024 * 1024)


def build_portfolio(source: PortfolioManifest | Mapping[str, Any] | bytes | bytearray | str | os.PathLike[str]) -> PortfolioSnapshot:
    manifest = source if isinstance(source, PortfolioManifest) else PortfolioManifest.from_json(source)
    registry = CandidateRegistry()
    entries = tuple(_entry(model) for model in manifest.models)
    candidates = [entry for entry in entries if entry.next_safe_measurement is not None]
    candidates.sort(key=lambda entry: (0 if entry.status == "waiting_readiness" else 1 if entry.status == "ready_for_experiment" else 2, MODEL_SIZES.index(entry.size)))
    next_point = candidates[0].next_safe_measurement if candidates else None
    body = {
        "schema": SNAPSHOT_SCHEMA,
        "models": [entry.as_dict() for entry in entries],
        "registry_hash": registry.registry_hash,
        "manifest_sha256": manifest.manifest_sha256,
        "cache_inventory_sha256": manifest.cache_inventory_sha256,
        "evidence_inventory_sha256": manifest.evidence_inventory_sha256,
        "readiness_evidence_sha256": manifest.readiness_evidence_sha256,
        "next_safe_measurement": next_point,
        "no_model_load": True,
        "no_download": True,
        "no_activation": True,
    }
    snapshot_hash = hashlib.sha256(canonical_bytes(body, max_bytes=4 * 1024 * 1024)).hexdigest()
    return PortfolioSnapshot(entries, registry.registry_hash, manifest.manifest_sha256, snapshot_hash, next_point, manifest.cache_inventory_sha256, manifest.evidence_inventory_sha256, manifest.readiness_evidence_sha256)


__all__ = [
    "EVIDENCE_SCHEMA", "MANIFEST_SCHEMA", "MODEL_IDS", "MODEL_SIZES", "PortfolioEntry",
    "PortfolioError", "PortfolioManifest", "PortfolioModel", "PortfolioSnapshot",
    "QUALITY_CLASSES", "SNAPSHOT_SCHEMA", "STATUSES", "build_portfolio",
]
