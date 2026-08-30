"""Safe fingerprint and real-shadow session boundary.

This module is the only orchestration boundary for a manually started local
IronMule shadow session.  It deliberately has no model-loading code.  The
existing collector, adapter, stage worker, readiness probe and memory/history
implementations remain the authorities; this module only binds their immutable
identities and persists a compact, redacted terminal record.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from .canonical import canonical_bytes, loads_strict
from .collector import Collector, CollectorError, CollectorReport, WorkloadContract
from .evaluator import CorrectnessResult, Evaluator, MetricSample, ResourceResult
from .fingerprint import ExactFingerprint
from .history import HistoryWriter, SessionEvent
from .ironmule_adapter import (
    EXECUTION_FILE_REGISTRY_HASH,
    IronMuleCheckoutBinding,
    IronMuleTuneAdapter,
)
from .memory import OptimizationMemoryV2, ReadOnlyMemoryView
from .readiness import HardwareLease, LeaseBusy, MacSystemProbe, ReadinessDecision, ReadinessGate, ReadinessPolicy


SCHEMA = "friday.optimizer.real-session.v1"
FINGERPRINT_SCHEMA = "friday.optimizer.fingerprint-report.v1"
PREREG_SCHEMA = "friday.optimizer.preregistration.v1"
PLAN_SCHEMA = "friday.optimizer.session-plan.v1"
RESULT_SCHEMA = "friday.optimizer.session-result.v1"
ALLOWED_STAGES = ("calibrate", "test")
ALLOWED_CANDIDATE = "combined_core_profile"
_SHA = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_MAX_JSON = 4 * 1024 * 1024


def _is_local_model_id(value: Any) -> bool:
    """Recognize local path spellings without treating Hub IDs as paths."""
    if not isinstance(value, str) or not value:
        return False
    return (
        value.startswith(("/", "\\\\", "./", "../", ".\\", "..\\"))
        or value == "~"
        or value.startswith(("~/", "~\\"))
        or bool(re.match(r"^~[^/\\]*[\\/]", value))
        or bool(re.match(r"^[A-Za-z]:[\\/]", value))
        or "\\" in value
    )


class RealSessionError(ValueError):
    """A user input, identity, or safety precondition was not accepted."""


class RealSessionUnavailable(RealSessionError):
    """The current host or runtime cannot provide a safe observation."""


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise RealSessionError(f"{field}_invalid")
    return value


def _non_placeholder_sha(value: Any, field: str) -> str:
    result = _sha(value, field)
    if len(set(result)) == 1:
        raise RealSessionError(f"{field}_placeholder")
    return result


def _commit(value: Any, field: str = "runtime_commit") -> str:
    if not isinstance(value, str) or not _COMMIT.fullmatch(value):
        raise RealSessionError(f"{field}_invalid")
    return value


def _non_placeholder_commit(value: Any, field: str) -> str:
    result = _commit(value, field)
    if len(set(result)) == 1:
        raise RealSessionError(f"{field}_placeholder")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def optimizer_source_manifest(root: str | os.PathLike[str] | None = None) -> tuple[dict[str, Any], ...]:
    """Canonical relative manifest for every bound ``friday_optimizer`` module."""
    base = _bounded_path(root, "optimizer_root") if root is not None else Path(__file__).resolve().parent
    package = base / "friday_optimizer" if base.name != "friday_optimizer" else base
    rows: list[dict[str, Any]] = []
    for path in sorted(package.glob("*.py"), key=lambda item: item.name):
        if path.is_symlink() or not path.is_file():
            raise RealSessionError("optimizer_manifest_file_invalid")
        raw = _read_stable(path, maximum=16 * 1024 * 1024)
        rows.append({"path": "friday_optimizer/" + path.name, "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    if not rows:
        raise RealSessionError("optimizer_manifest_empty")
    return tuple(rows)


def optimizer_identity(root: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Return current optimizer HEAD and its content-addressed source identity."""
    package_root = _bounded_path(root, "optimizer_root") if root is not None else Path(__file__).resolve().parent.parent
    try:
        completed = subprocess.run(
            ["/usr/bin/git", "-C", str(package_root), "rev-parse", "--verify", "HEAD"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, timeout=2, check=False, shell=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RealSessionError("optimizer_head_unavailable") from exc
    head = (completed.stdout or "").strip()
    if completed.returncode != 0 or not _COMMIT.fullmatch(head):
        raise RealSessionError("optimizer_head_unavailable")
    try:
        status = subprocess.run(
            ["/usr/bin/git", "-C", str(package_root), "status", "--porcelain=v1", "--untracked-files=all"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, timeout=2, check=False, shell=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RealSessionError("optimizer_checkout_unavailable") from exc
    if status.returncode != 0:
        raise RealSessionError("optimizer_checkout_unavailable")
    if (status.stdout or "").strip():
        raise RealSessionError("optimizer_checkout_dirty")
    manifest = optimizer_source_manifest(package_root)
    manifest_bytes = canonical_bytes(list(manifest), max_bytes=_MAX_JSON)
    tree_bytes = canonical_bytes({"schema": "friday.optimizer.source-tree.v1", "files": list(manifest)}, max_bytes=_MAX_JSON)
    return {
        "head": head,
        "manifest": manifest,
        "code_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "optimizer_tree_sha256": hashlib.sha256(tree_bytes).hexdigest(),
        "adapter_sha256": _file_sha256(Path(__file__).with_name("ironmule_adapter.py")),
        "worker_sha256": _worker_hash(),
        "registry_sha256": EXECUTION_FILE_REGISTRY_HASH,
    }


def _model_artifact_hashes(source: Any) -> tuple[str | None, str | None, str | None]:
    if not isinstance(source, (str, os.PathLike)):
        return None, None, None
    path = _existing_file(source, "model_identity")
    data = _strict_json(path)
    identity = data.get("model_identity", data)
    if not isinstance(identity, Mapping):
        return None, None, None
    def first(*names: str) -> str | None:
        value = next((identity.get(name) for name in names if identity.get(name) is not None), None)
        return value if isinstance(value, str) and _SHA.fullmatch(value) else None
    return first("model_identity_sha256", "identity_sha256"), first("model_manifest_sha256", "model_manifest_hash"), first("tokenizer_sha256", "tokenizer_hash", "tokenizer")


def _workload_hashes(source: Any) -> tuple[str | None, str | None]:
    if not isinstance(source, (str, os.PathLike)):
        return None, None
    path = _existing_file(source, "workload_contract")
    source_hash = hashlib.sha256(_read_stable(path)).hexdigest()
    try:
        contract = WorkloadContract.from_json(path)
    except Exception:
        return source_hash, None
    return source_hash, contract.contract_hash


def _session_id(value: Any) -> str:
    if not isinstance(value, str) or not _SESSION_ID.fullmatch(value):
        raise RealSessionError("session_id_invalid")
    return value


def _bounded_path(value: str | os.PathLike[str], field: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise RealSessionError(f"{field}_invalid")
    try:
        path = Path(os.path.abspath(os.fspath(value)))
    except (TypeError, ValueError, OSError) as exc:
        raise RealSessionError(f"{field}_invalid") from exc
    if len(str(path).encode("utf-8", "surrogatepass")) > 4096:
        raise RealSessionError(f"{field}_invalid")
    return path


def _reject_symlink_path(path: Path, *, leaf: bool = True) -> None:
    current = Path(path.anchor or os.sep)
    parts = path.parts[1:] if path.anchor else path.parts
    for index, part in enumerate(parts):
        current /= part
        if not leaf and index == len(parts) - 1:
            break
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise RealSessionError("path_unreadable") from exc
        if stat.S_ISLNK(info.st_mode):
            raise RealSessionError("symlink_path_refused")


def _existing_file(value: str | os.PathLike[str], field: str, *, maximum: int = _MAX_JSON) -> Path:
    path = _bounded_path(value, field)
    _reject_symlink_path(path)
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise RealSessionError(f"{field}_missing") from exc
    except OSError as exc:
        raise RealSessionError(f"{field}_unreadable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise RealSessionError(f"{field}_must_be_regular")
    if info.st_size > maximum:
        raise RealSessionError(f"{field}_too_large")
    return path


def _read_stable(path: Path, *, maximum: int = _MAX_JSON) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise RealSessionError("input_unreadable") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
        raise RealSessionError("input_invalid")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise RealSessionError("input_unreadable") from exc
    try:
        opened = os.fstat(fd)
        if stat.S_ISLNK(opened.st_mode) or not stat.S_ISREG(opened.st_mode) or opened.st_size > maximum:
            raise RealSessionError("input_invalid")
        raw = bytearray()
        while True:
            chunk = os.read(fd, min(1024 * 1024, maximum - len(raw) + 1))
            if not chunk:
                break
            raw.extend(chunk)
            if len(raw) > maximum:
                raise RealSessionError("input_too_large")
        after = os.fstat(fd)
        current = path.lstat()
        identity = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
        if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or identity != (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns):
            raise RealSessionError("input_changed")
        return bytes(raw)
    finally:
        os.close(fd)


def _strict_json(path: Path) -> Mapping[str, Any]:
    try:
        value = loads_strict(_read_stable(path), max_bytes=_MAX_JSON, max_depth=32, max_items=100_000)
    except Exception as exc:
        raise RealSessionError("invalid_json") from exc
    if not isinstance(value, Mapping):
        raise RealSessionError("json_must_be_object")
    return value


def _atomic_new(path: Path, raw: bytes) -> None:
    """Create exactly one new 0600 file without following a path race."""
    if len(raw) > _MAX_JSON:
        raise RealSessionError("output_too_large")
    parent = path.parent
    _reject_symlink_path(parent)
    try:
        pinfo = parent.lstat()
    except OSError as exc:
        raise RealSessionError("output_parent_unreadable") from exc
    if stat.S_ISLNK(pinfo.st_mode) or not stat.S_ISDIR(pinfo.st_mode):
        raise RealSessionError("output_parent_invalid")
    try:
        path.lstat()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise RealSessionError("output_unreadable") from exc
    else:
        raise RealSessionError("output_exists")
    temp: Path | None = None
    fd = -1
    try:
        for index in range(16):
            candidate = parent / f".{path.name}.tmp-{os.getpid()}-{index}"
            try:
                fd = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
                temp = candidate
                break
            except FileExistsError:
                continue
        if fd < 0 or temp is None:
            raise RealSessionError("temporary_file_unavailable")
        view = memoryview(raw)
        while view:
            count = os.write(fd, view)
            if count <= 0:
                raise RealSessionError("output_write_failed")
            view = view[count:]
        os.fsync(fd)
        os.close(fd)
        fd = -1
        try:
            path.lstat()
        except FileNotFoundError:
            pass
        else:
            raise RealSessionError("output_exists")
        now = parent.lstat()
        if (now.st_dev, now.st_ino) != (pinfo.st_dev, pinfo.st_ino):
            raise RealSessionError("output_parent_changed")
        os.replace(temp, path)
        temp = None
        os.chmod(path, 0o600, follow_symlinks=False)
        dfd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except RealSessionError:
        raise
    except OSError as exc:
        raise RealSessionError("atomic_write_failed") from exc
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        if temp is not None:
            try:
                temp.unlink()
            except OSError:
                pass


def _redacted_report(report: CollectorReport) -> dict[str, Any]:
    # Collector's public representation already excludes source paths and raw
    # command output.  Preserve only that representation at this boundary.
    # ``safe_redacted`` intentionally replaces public repository IDs containing
    # a slash.  That is useful for an untrusted display surface but cannot be
    # used to reconstruct the exact fingerprint for a later binding.  The
    # collector's canonical ``to_dict`` is already path-free and preserves the
    # verified public model identity.
    result = report.to_dict()
    for field in ("errors", "ood_reasons"):
        values = result.get(field, [])
        if isinstance(values, list):
            result[field] = [
                re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(value)).lower().strip("_")[:256]
                for value in values
            ]
    return result


@dataclass(frozen=True, slots=True)
class FingerprintReport:
    report: Mapping[str, Any]
    fingerprint_hash: str | None
    report_hash: str
    ood: bool
    recommendation_allowed: bool
    source_sha256: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": FINGERPRINT_SCHEMA,
            "report": dict(self.report),
            "fingerprint_hash": self.fingerprint_hash,
            "report_hash": self.report_hash,
            "ood": self.ood,
            "recommendation_allowed": self.recommendation_allowed,
            "source_sha256": self.source_sha256,
        }

    def public_dict(self) -> dict[str, Any]:
        """Return a display/file projection without changing exact internals."""
        result = deepcopy(self.as_dict())
        report = result.get("report")
        if isinstance(report, dict):
            fingerprint = report.get("fingerprint")
            if isinstance(fingerprint, dict):
                model = fingerprint.get("model")
                if isinstance(model, dict) and _is_local_model_id(model.get("model_id")):
                    model["model_id"] = "<local-model>"
        return result

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.as_dict(), max_bytes=_MAX_JSON)

    @classmethod
    def from_collector(cls, report: CollectorReport) -> "FingerprintReport":
        safe = _redacted_report(report)
        encoded = canonical_bytes(safe, max_bytes=_MAX_JSON)
        return cls(safe, report.fingerprint_hash, hashlib.sha256(encoded).hexdigest(), report.ood, report.recommendation_allowed, report.source_sha256)


def collect_fingerprint(
    *,
    model_identity: str | os.PathLike[str] | Mapping[str, Any],
    workload_contract: str | os.PathLike[str] | Mapping[str, Any],
    runtime_commit: str,
    collector: Collector | None = None,
) -> FingerprintReport:
    """Collect one exact report through the real environment collector."""
    _commit(runtime_commit)
    try:
        report = (collector or Collector()).collect(
            runtime_commit=runtime_commit,
            model_identity=model_identity,
            workload_contract=workload_contract,
        )
    except (CollectorError, ValueError, TypeError) as exc:
        raise RealSessionError("fingerprint_collection_failed") from exc
    return FingerprintReport.from_collector(report)


@dataclass(frozen=True, slots=True)
class Preregistration:
    raw: Mapping[str, Any]
    sha256: str
    fingerprint_hash: str
    code_hash: str
    dataset_hash: str
    checkout_head: str
    source_digest: str
    worker_sha256: str
    model_identity_sha256: str | None = None
    workload_contract_sha256: str | None = None
    optimizer_head: str | None = None
    candidate_id: str | None = None
    duration_minutes: int | None = None
    experiment: str = ""
    optimizer_tree_sha256: str = ""
    code_manifest_sha256: str = ""
    adapter_sha256: str = ""
    registry_sha256: str = ""
    workload_file_sha256: str = ""
    model_manifest_sha256: str = ""
    tokenizer_sha256: str = ""
    model_identity_source_sha256: str | None = None
    stages: tuple[str, ...] = ()
    result_schema: str = RESULT_SCHEMA
    start_authorization_schema: str = "friday.optimizer.start-authorization.v1"

    @classmethod
    def load(cls, source: str | os.PathLike[str] | Mapping[str, Any]) -> "Preregistration":
        if isinstance(source, Mapping):
            raw = dict(source)
            encoded = canonical_bytes(raw, max_bytes=_MAX_JSON)
        else:
            path = _existing_file(source, "preregistration")
            if path.suffix.lower() != ".json":
                raise RealSessionError("preregistration_json_required")
            encoded = _read_stable(path)
            try:
                raw = loads_strict(encoded, max_bytes=_MAX_JSON, max_depth=32, max_items=100_000)
            except Exception as exc:
                raise RealSessionError("invalid_preregistration") from exc
        if not isinstance(raw, Mapping):
            raise RealSessionError("invalid_preregistration")
        # Accept the reviewed nested JSON layout, but normalize it to one
        # canonical internal vocabulary.  Unknown fields are rejected before
        # any identity is trusted; Markdown preregistrations never enter this
        # path.
        raw = dict(raw)
        nested_aliases = {
            "optimizer": {"optimizer_head": "optimizer_head", "head": "optimizer_head", "tree_sha256": "optimizer_tree_sha256", "tree": "optimizer_tree_sha256", "code_manifest_sha256": "code_manifest_sha256", "code_manifest": "code_manifest_sha256"},
            "checkout": {"head": "checkout_head", "expected_head": "checkout_head", "source_digest": "source_digest", "source_sha256": "source_digest", "registry_hash": "registry_sha256", "registry_sha256": "registry_sha256"},
            "workload": {"file_sha256": "workload_file_sha256", "file_hash": "workload_file_sha256", "contract_sha256": "workload_contract_sha256", "canonical_sha256": "workload_contract_sha256"},
            "model": {"identity_sha256": "model_identity_sha256", "identity_hash": "model_identity_sha256", "manifest_sha256": "model_manifest_sha256", "manifest_hash": "model_manifest_sha256", "tokenizer_sha256": "tokenizer_sha256", "tokenizer_hash": "tokenizer_sha256"},
        }
        for container, aliases in nested_aliases.items():
            value = raw.pop(container, None)
            if value is None:
                continue
            if not isinstance(value, Mapping):
                raise RealSessionError(f"preregistration_{container}_invalid")
            if set(value) - set(aliases):
                raise RealSessionError(f"preregistration_{container}_unknown_field")
            for key, target in aliases.items():
                if key in value:
                    if target in raw:
                        raise RealSessionError(f"preregistration_duplicate_{target}")
                    raw[target] = value[key]
        allowed_fields = {
            "schema", "schema_version", "status", "experiment", "fingerprint_hash", "fingerprint", "code_hash", "code_sha256", "runtime_code_hash", "dataset_hash", "dataset_sha256",
            "checkout_head", "expected_head", "ironmule_head", "optimizer_head", "optimizer_commit", "optimizer_tree_sha256", "optimizer_tree", "code_manifest_sha256", "code_manifest",
            "adapter_sha256", "adapter_sha", "worker_sha256", "worker_sha", "adapter_worker_sha256", "registry_sha256", "registry_hash", "source_digest", "source_sha256", "adapter_source_digest",
            "workload_file_sha256", "workload_file_hash", "workload_contract_sha256", "workload_sha256", "model_identity_sha256", "model_sha256", "model_manifest_sha256", "model_manifest_hash", "tokenizer_sha256", "tokenizer_hash", "model_identity_source_sha256",
            "candidate_id", "candidate", "duration_minutes", "duration", "stages", "allowed_stages", "result_schema", "session_result_schema", "start_authorization_schema", "authorization_schema",
        }
        if set(raw) - allowed_fields:
            raise RealSessionError("preregistration_unknown_field")
        schema = raw.get("schema")
        if schema is None and raw.get("schema_version") == 1:
            schema = PREREG_SCHEMA
        if schema != PREREG_SCHEMA:
            raise RealSessionError("preregistration_schema_invalid")
        status = raw.get("status")
        if status != "SEALED":
            raise RealSessionError("preregistration_not_sealed")

        def pick(*names: str, required: bool = True) -> Any:
            for name in names:
                if name in raw:
                    return raw[name]
            if required:
                raise RealSessionError("preregistration_field_missing")
            return None

        experiment = pick("experiment")
        if not isinstance(experiment, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", experiment):
            raise RealSessionError("preregistration_experiment_invalid")
        head = _non_placeholder_commit(pick("checkout_head", "expected_head", "ironmule_head"), "preregistration_head")
        optimizer_head = _non_placeholder_commit(pick("optimizer_head", "optimizer_commit"), "preregistration_optimizer_head")
        optimizer_tree = _non_placeholder_sha(pick("optimizer_tree_sha256", "optimizer_tree"), "preregistration_optimizer_tree")
        manifest = _non_placeholder_sha(pick("code_manifest_sha256", "code_manifest"), "preregistration_code_manifest")
        adapter_sha = _non_placeholder_sha(pick("adapter_sha256", "adapter_sha"), "preregistration_adapter")
        worker = _non_placeholder_sha(pick("worker_sha256", "worker_sha", "adapter_worker_sha256"), "preregistration_worker")
        registry = _non_placeholder_sha(pick("registry_sha256", "registry_hash"), "preregistration_registry")
        source_digest = _non_placeholder_sha(pick("source_digest", "source_sha256", "adapter_source_digest"), "preregistration_source")
        workload_file = _non_placeholder_sha(pick("workload_file_sha256", "workload_file_hash"), "preregistration_workload_file")
        workload_sha = _non_placeholder_sha(pick("workload_contract_sha256", "workload_sha256"), "preregistration_workload")
        model_sha = _non_placeholder_sha(pick("model_identity_sha256", "model_sha256"), "preregistration_model")
        model_manifest = _non_placeholder_sha(pick("model_manifest_sha256", "model_manifest_hash"), "preregistration_model_manifest")
        tokenizer_sha = _non_placeholder_sha(pick("tokenizer_sha256", "tokenizer_hash"), "preregistration_tokenizer")
        fp = _non_placeholder_sha(pick("fingerprint_hash", "fingerprint"), "preregistration_fingerprint")
        code_value = pick("code_hash", "code_sha256", "runtime_code_hash", required=False) or manifest
        code = _non_placeholder_sha(code_value, "preregistration_code")
        dataset_value = pick("dataset_hash", "dataset_sha256", required=False) or workload_sha
        dataset = _non_placeholder_sha(dataset_value, "preregistration_dataset")
        model_identity_source = pick("model_identity_source_sha256", required=False)
        if model_identity_source is not None:
            model_identity_source = _non_placeholder_sha(model_identity_source, "preregistration_model_source")
        candidate_id = pick("candidate_id", "candidate", required=False)
        if candidate_id is not None and (not isinstance(candidate_id, str) or not candidate_id):
            raise RealSessionError("preregistration_candidate_invalid")
        duration_minutes = pick("duration_minutes", "duration", required=False)
        if duration_minutes is not None:
            duration_minutes = _duration(duration_minutes)
        stages_raw = pick("stages", "allowed_stages")
        if not isinstance(stages_raw, list) or tuple(stages_raw) != ALLOWED_STAGES:
            raise RealSessionError("preregistration_stages_invalid")
        result_schema = pick("result_schema", "session_result_schema")
        auth_schema = pick("start_authorization_schema", "authorization_schema")
        if result_schema != RESULT_SCHEMA or auth_schema != "friday.optimizer.start-authorization.v1":
            raise RealSessionError("preregistration_protocol_schema_invalid")
        return cls(raw, hashlib.sha256(encoded).hexdigest(), fp, code, dataset, head, source_digest, worker, model_sha, workload_sha, optimizer_head, candidate_id, duration_minutes, experiment, optimizer_tree, manifest, adapter_sha, registry, workload_file, model_manifest, tokenizer_sha, model_identity_source, tuple(stages_raw), result_schema, auth_schema)


def _field_match(prereg: Preregistration, report: FingerprintReport, binding: Any) -> list[str]:
    reasons: list[str] = []
    if report.fingerprint_hash != prereg.fingerprint_hash:
        reasons.append("fingerprint_mismatch")
    if prereg.checkout_head != binding.expected_head:
        reasons.append("preregistration_head_mismatch")
    if prereg.source_digest != binding.source_digest:
        reasons.append("preregistration_source_mismatch")
    if prereg.worker_sha256 != _worker_hash():
        reasons.append("preregistration_worker_mismatch")
    if prereg.model_identity_sha256 and report.report.get("model_source_sha256") != prereg.model_identity_sha256:
        reasons.append("model_identity_source_mismatch")
    if prereg.workload_contract_sha256 and report.report.get("workload_contract_sha256") != prereg.workload_contract_sha256:
        reasons.append("workload_contract_mismatch")
    return reasons


def _worker_hash() -> str:
    from . import ironmule_stage_worker
    return hashlib.sha256(Path(ironmule_stage_worker.__file__).read_bytes()).hexdigest()


def _checkout_details(binding: IronMuleCheckoutBinding, adapter: IronMuleTuneAdapter) -> tuple[Any, list[str]]:
    try:
        validation = adapter.validate_checkout()
        return validation, []
    except Exception as exc:
        return None, ["checkout_invalid:" + type(exc).__name__]


@dataclass(frozen=True, slots=True)
class SessionPlan:
    fingerprint: FingerprintReport
    preregistration: Preregistration
    candidate_id: str
    duration_minutes: int
    checkout_head: str | None
    source_digest: str | None
    interpreter_sha256: str | None
    allowed_stages: tuple[str, ...] = ALLOWED_STAGES
    blocked_reasons: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return not self.blocked_reasons

    @property
    def duration(self) -> int:
        return self.duration_minutes

    @property
    def candidate(self) -> str:
        return self.candidate_id

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": PLAN_SCHEMA,
            "ready": self.ready,
            "candidate_id": self.candidate_id,
            "duration_minutes": self.duration_minutes,
            "allowed_stages": list(self.allowed_stages),
            "blocked_reasons": list(self.blocked_reasons),
            "fingerprint_hash": self.fingerprint.fingerprint_hash,
            "fingerprint_report_hash": self.fingerprint.report_hash,
            "model_identity_source_sha256": self.fingerprint.report.get("model_source_sha256"),
            "workload_contract_source_sha256": self.fingerprint.report.get("workload_contract_sha256"),
            "preregistration_hash": self.preregistration.sha256,
            "checkout_head": self.checkout_head,
            "source_digest": self.source_digest,
            "interpreter_sha256": self.interpreter_sha256,
            "execution_registry_hash": EXECUTION_FILE_REGISTRY_HASH,
            "optimizer_head": self.preregistration.optimizer_head,
            "optimizer_tree_sha256": self.preregistration.optimizer_tree_sha256,
            "code_manifest_sha256": self.preregistration.code_manifest_sha256,
            "adapter_sha256": self.preregistration.adapter_sha256,
            "worker_sha256": self.preregistration.worker_sha256,
            "no_hardware_readiness": True,
            "no_stage_planned": True,
            "no_write": True,
        }


def _duration(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 5 <= value <= 30:
        raise RealSessionError("duration_invalid")
    return value


def build_session_plan(
    *,
    checkout: str | os.PathLike[str],
    expected_head: str,
    interpreter: str | os.PathLike[str],
    model_identity: str | os.PathLike[str] | Mapping[str, Any],
    workload_contract: str | os.PathLike[str] | Mapping[str, Any],
    runtime_commit: str,
    candidate_id: str,
    duration_minutes: int,
    preregistration: str | os.PathLike[str] | Mapping[str, Any],
    collector: Collector | None = None,
    adapter_factory: Callable[[IronMuleCheckoutBinding], IronMuleTuneAdapter] = IronMuleTuneAdapter,
    binding_factory: Callable[..., IronMuleCheckoutBinding] | None = None,
    optimizer_root: str | os.PathLike[str] | None = None,
    identity_provider: Callable[[], Mapping[str, Any]] | None = None,
) -> SessionPlan:
    """Validate all static bindings; never run readiness commands or stages."""
    duration = _duration(duration_minutes)
    expected = _commit(expected_head, "expected_head")
    if candidate_id != ALLOWED_CANDIDATE:
        raise RealSessionError("candidate_not_supported")
    prereg = Preregistration.load(preregistration)
    fp = collect_fingerprint(model_identity=model_identity, workload_contract=workload_contract, runtime_commit=runtime_commit, collector=collector)
    blocked: list[str] = []
    if not fp.recommendation_allowed:
        blocked.append("fingerprint_ood")
    if prereg.checkout_head != expected:
        blocked.append("preregistration_head_mismatch")
    if prereg.fingerprint_hash != fp.fingerprint_hash:
        blocked.append("fingerprint_mismatch")
    if prereg.optimizer_head is not None and prereg.optimizer_head != runtime_commit:
        blocked.append("optimizer_head_mismatch")
    if prereg.candidate_id is not None and prereg.candidate_id != candidate_id:
        blocked.append("preregistration_candidate_mismatch")
    if prereg.duration_minutes is not None and prereg.duration_minutes != duration:
        blocked.append("preregistration_duration_mismatch")
    try:
        current_optimizer = dict(identity_provider() if identity_provider is not None else optimizer_identity(optimizer_root))
        if current_optimizer.get("head") != prereg.optimizer_head or current_optimizer.get("optimizer_tree_sha256") != prereg.optimizer_tree_sha256:
            blocked.append("optimizer_source_identity_mismatch")
        if current_optimizer.get("code_manifest_sha256") != prereg.code_manifest_sha256:
            blocked.append("optimizer_code_manifest_mismatch")
        if current_optimizer.get("adapter_sha256") != prereg.adapter_sha256:
            blocked.append("adapter_identity_mismatch")
        if current_optimizer.get("worker_sha256") != prereg.worker_sha256:
            blocked.append("worker_identity_mismatch")
        if current_optimizer.get("registry_sha256") != prereg.registry_sha256:
            blocked.append("registry_identity_mismatch")
    except RealSessionError as exc:
        blocked.append(str(exc))
    model_source = fp.report.get("model_source_sha256")
    workload_source = fp.report.get("workload_contract_sha256")
    workload_file_hash, workload_contract_hash = _workload_hashes(workload_contract)
    if prereg.workload_file_sha256 != workload_file_hash:
        blocked.append("workload_file_mismatch")
    if prereg.workload_contract_sha256 != workload_contract_hash:
        blocked.append("workload_contract_mismatch")
    model_identity_hash, model_manifest_hash, tokenizer_hash = _model_artifact_hashes(model_identity)
    if prereg.model_identity_sha256 not in {model_identity_hash, model_source}:
        blocked.append("model_identity_hash_mismatch")
    if model_manifest_hash != prereg.model_manifest_sha256:
        blocked.append("model_manifest_mismatch")
    if tokenizer_hash != prereg.tokenizer_sha256:
        blocked.append("tokenizer_mismatch")
    binding: IronMuleCheckoutBinding | None = None
    validation: Any = None
    try:
        exact = fp.report.get("fingerprint")
        if not isinstance(exact, Mapping):
            raise RealSessionError("fingerprint_missing")
        fingerprint = ExactFingerprint.from_mapping(exact)
        make_binding = binding_factory or IronMuleCheckoutBinding
        binding = make_binding(checkout=checkout, expected_head=expected, interpreter=interpreter, fingerprint=fingerprint)
        adapter = adapter_factory(binding)
        validation, checkout_reasons = _checkout_details(binding, adapter)
        blocked.extend(checkout_reasons)
        if validation is not None:
            if prereg.source_digest != validation.source_digest:
                blocked.append("preregistration_source_mismatch")
            if prereg.worker_sha256 != _worker_hash():
                blocked.append("preregistration_worker_mismatch")
    except Exception as exc:
        blocked.append("binding_invalid:" + type(exc).__name__)
    return SessionPlan(fp, prereg, candidate_id, duration, None if validation is None else validation.head, None if validation is None else validation.source_digest, None if validation is None else validation.interpreter_sha256, blocked_reasons=tuple(dict.fromkeys(blocked)))


def _safe_value(value: Any, depth: int = 0) -> Any:
    """Keep only bounded, path-/prompt-free evidence summaries."""
    # AdapterResult is intentionally immutable but is not itself a Mapping.
    # Normalize it at the boundary so the persisted worker envelope retains
    # its outcome/reason while still filtering the untrusted payload.
    if hasattr(value, "outcome") and hasattr(value, "payload"):
        return {
            "outcome": str(getattr(value, "outcome", "inconclusive")),
            "reason": re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(getattr(value, "reason", ""))).lower().strip("_")[:128],
            "payload": _safe_value(getattr(value, "payload", {}), depth + 1),
        }
    if depth > 4:
        return "<omitted>"
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                continue
            low = key.lower()
            if key in {"token_hash", "response_hash", "text_hash", "stop_hash", "count_hash", "evidence_hash", "profile_artifact_sha256"}:
                result[key] = item
                continue
            if key in {"session_id", "session", "session_id_hash"} or any(part in low for part in ("prompt", "output", "response", "text", "log", "path", "stdout", "stderr", "token_ids")):
                continue
            elif key.endswith(("_hash", "_sha256")):
                result[key] = item
            elif isinstance(item, (str, int, float, bool)) or item is None:
                if key.lower() in {"reason", "error", "status"} and isinstance(item, str):
                    result[key] = re.sub(r"[^A-Za-z0-9_.:-]+", "_", item).lower().strip("_")[:256]
                else:
                    result[key] = item
            elif isinstance(item, Mapping):
                result[key] = _safe_value(item, depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        return [_safe_value(item, depth + 1) for item in value[:32]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return "<omitted>"


def _result_event(result: Mapping[str, Any], session_id: str, *, code_hash: str, dataset_hash: str, prereg: str, fingerprint: str, candidate: str) -> SessionEvent:
    for supplied, key in ((code_hash, "code_hash"), (dataset_hash, "dataset_hash"), (prereg, "preregistration_hash"), (fingerprint, "fingerprint_hash"), (candidate, "candidate_id")):
        if key in result and result.get(key) != supplied:
            raise RealSessionError("history_result_binding_mismatch")
    state = str(result.get("state", "baseline"))
    status = str(result.get("status", "inconclusive"))
    reason = str(result.get("reason", "no_recommendation"))
    reason = re.sub(r"[^a-zA-Z0-9_.:-]+", "_", reason).lower().strip("_")[:128] or "no_recommendation"
    evidence = str(result.get("result_hash") or hashlib.sha256(canonical_bytes(_safe_value(result), max_bytes=_MAX_JSON)).hexdigest())
    payload = {
        "status": status,
        "qualified": bool(result.get("recommendation_available")),
        "no_activation": True,
        "fingerprint_hash": fingerprint,
        "dataset_hash": dataset_hash,
        "candidate_id": candidate,
        "evidence_hash": evidence,
        "code": "real_session",
        "reasons": [reason],
    }
    created_at = result.get("created_at_utc")
    if not isinstance(created_at, str):
        created_at = _now_rfc3339()
    return SessionEvent(event_id="real-session:" + evidence[:48], kind="benchmark", session_id=session_id, fingerprint_hash=fingerprint, dataset_hash=dataset_hash, candidate_id=candidate, code_hash=code_hash, evidence_hash=evidence, state=state, reason=reason, payload=payload, created_at=created_at)


def _memory_binding(path: Path) -> tuple[int, int, int, int, str]:
    info = path.lstat()
    raw = _read_stable(path, maximum=64 * 1024 * 1024)
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, hashlib.sha256(raw).hexdigest()


def _memory_preflight(path: Path) -> tuple[int, int, int, int, str]:
    try:
        binding = _memory_binding(path)
        view = OptimizationMemoryV2.open_read_only(path)
        try:
            if not view.schema_ok or not view.integrity().ok:
                raise RealSessionError("memory_preflight_failed")
        finally:
            view.close()
        if _memory_binding(path) != binding:
            raise RealSessionError("memory_changed")
        return binding
    except RealSessionError:
        raise
    except Exception as exc:
        raise RealSessionError("memory_preflight_failed") from exc


def _lease_path(memory: Path) -> Path:
    base = memory.parent.parent if memory.parent.name == ".friday-data" else memory.parent
    return base / ".friday-data" / "optimizer-hardware.lock"


def _now_rfc3339() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _clock_now(clock: Any) -> float:
    if clock is None:
        return time.monotonic()
    if callable(clock):
        return float(clock())
    method = getattr(clock, "monotonic", None) or getattr(clock, "now", None)
    if method is None:
        raise RealSessionError("clock_invalid")
    return float(method())


def start_authorization_hash(session_id: str, duration_minutes: int, candidate_id: str, execute: bool, schema: str = "friday.optimizer.start-authorization.v1", preregistration_hash: str = "") -> str:
    """Bind the explicit user start request, not a model or hardware identity."""
    _session_id(session_id)
    duration = _duration(duration_minutes)
    if candidate_id != ALLOWED_CANDIDATE or execute is not True:
        raise RealSessionError("start_authorization_invalid")
    material = canonical_bytes({
        "schema": schema,
        "session_id": session_id,
        "duration_minutes": duration,
        "candidate_id": candidate_id,
        "execute": True,
        "preregistration_hash": preregistration_hash,
    }, max_bytes=4096)
    return hashlib.sha256(material).hexdigest()


def _transition(state: list[str], transitions: list[dict[str, Any]], target: str, reason: str) -> None:
    old = state[0] if state else None
    state[0] = target
    transitions.append({"sequence": len(transitions), "from": old, "to": target, "reason": re.sub(r"[^a-zA-Z0-9_.:-]+", "_", reason).lower()[:128]})


def _unwrap_stage_payload(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        payload = value.get("payload")
        return payload if isinstance(payload, Mapping) else value
    payload = getattr(value, "payload", None)
    return payload if isinstance(payload, Mapping) else {}


def _metric_samples(value: Any) -> tuple[MetricSample, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    parsed: list[MetricSample] = []
    for item in value:
        if isinstance(item, MetricSample):
            parsed.append(item)
        elif isinstance(item, Mapping):
            try:
                allowed = {"session_id", "pair_id", "arm", "order", "fingerprint", "workload", "ttft_seconds", "decode_tps", "tokens", "status", "error"}
                if any(key not in allowed for key in item if not isinstance(key, str)):
                    return ()
                parsed.append(MetricSample(**{key: item[key] for key in allowed if key in item}))
            except Exception:
                return ()
    return tuple(parsed)


def _correctness_value(value: Any) -> CorrectnessResult | None:
    if isinstance(value, CorrectnessResult):
        return value
    if not isinstance(value, Mapping):
        return None
    try:
        allowed = {"token_ids", "text", "stop_reason", "physical_tokens", "visible_tokens", "response_hash", "passed", "error"}
        data = {key: value[key] for key in allowed if key in value}
        if isinstance(data.get("token_ids"), list):
            data["token_ids"] = tuple(data["token_ids"])
        return CorrectnessResult(**data)
    except Exception:
        return None


def _resource_values(value: Any) -> tuple[ResourceResult, ...]:
    if isinstance(value, Mapping):
        value = (value,)
    if not isinstance(value, (list, tuple)):
        return ()
    parsed: list[ResourceResult] = []
    for item in value:
        if isinstance(item, ResourceResult):
            if not item.passed:
                return ()
            parsed.append(item)
        elif isinstance(item, Mapping):
            try:
                raw = dict(item)
                required = {"peak_memory_bytes", "peak_rss_bytes", "swap_delta_bytes", "resource_gate_passed"}
                if not required.issubset(raw) or raw.get("resource_gate_passed") is not True:
                    return ()
                # Worker resource envelopes use the adapter's metric names;
                # no defaults or unit coercion are permitted here.
                if any(isinstance(raw.get(key), bool) or not isinstance(raw.get(key), int) or raw.get(key) < 0 for key in ("peak_memory_bytes", "peak_rss_bytes", "swap_delta_bytes")):
                    return ()
                if raw.get("swap_delta_bytes") != 0:
                    return ()
                if raw.get("status", "ok") != "ok":
                    return ()
                booleans = ("timed_out", "crashed", "foreign_load")
                if any(key in raw and not isinstance(raw[key], bool) for key in booleans):
                    return ()
                parsed.append(ResourceResult(
                    peak_memory_bytes=raw["peak_memory_bytes"], peak_rss_bytes=raw["peak_rss_bytes"],
                    swap_delta_bytes=raw["swap_delta_bytes"], timed_out=raw.get("timed_out", False),
                    crashed=raw.get("crashed", False), foreign_load=raw.get("foreign_load", False),
                    status="ok", error=raw.get("error", ""),
                ))
            except Exception:
                return ()
    return tuple(parsed)


def _bounded_series(value: Any) -> list[int | float] | None:
    if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= 7:
        return None
    result: list[int | float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        if isinstance(item, float) and not (item == item and abs(item) != float("inf")):
            return None
        if float(item) < 0:
            return None
        result.append(item)
    return result


def _evidence_hash(value: Any) -> str | None:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        return None
    return value


def _measurement_arm(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    series: dict[str, list[int | float]] = {}
    for key in ("total_ns", "prefill_ns", "decode_ns"):
        parsed = _bounded_series(value.get(key))
        if parsed is None:
            return None
        series[key] = parsed
    decode_steps = value.get("decode_steps")
    peak = value.get("mlx_peak_bytes")
    if isinstance(decode_steps, bool) or not isinstance(decode_steps, int) or decode_steps < 0:
        return None
    if isinstance(peak, bool) or not isinstance(peak, int) or peak < 0:
        return None
    deterministic = value.get("deterministic")
    if not isinstance(deterministic, bool):
        return None
    hashes: dict[str, str] = {}
    for output, names in {
        "token_hash": ("token_hash", "response_hash"),
        "count_hash": ("count_hash",),
        "text_equivalence_hash": ("text_equivalence_hash", "text_hash"),
    }.items():
        candidate = next((value.get(name) for name in names if value.get(name) is not None), None)
        digest = _evidence_hash(candidate)
        if digest is None:
            return None
        hashes[output] = digest
    return {**series, "decode_steps": decode_steps, "deterministic": deterministic, "mlx_peak_bytes": peak, **hashes}


def _measurement_stage(value: Any, *, left_name: str, right_name: str) -> dict[str, Any] | None:
    payload = _unwrap_stage_payload(value)
    # ParsedIronMuleResult carries evaluator-facing MetricSample lists and the
    # actual timing series separately in raw_pairs[].  Only raw_pairs can
    # support a persisted measurement claim.  A clear evaluation container is
    # accepted when the root does not contain raw_pairs; a diagnostic
    # ``calibration`` mapping is never interpreted as evidence.
    source = payload
    if "raw_pairs" not in source and isinstance(source.get("evaluation"), Mapping):
        source = source["evaluation"]
    raw_pairs = source.get("raw_pairs")
    if "raw_pairs" in source and not isinstance(raw_pairs, (list, tuple)):
        return None
    if isinstance(raw_pairs, (list, tuple)):
        if not 1 <= len(raw_pairs) <= 6:
            return None
        pairs: list[dict[str, Any]] = []
        seen: set[str] = set()
        for pair in raw_pairs:
            if not isinstance(pair, Mapping):
                return None
            pair_id = pair.get("pair_id")
            order = pair.get("order")
            if not isinstance(pair_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", pair_id) or pair_id in seen:
                return None
            if order not in {"AB", "BA"}:
                return None
            left_raw, right_raw = pair.get("left"), pair.get("right")
            left_arm, right_arm = _measurement_arm(left_raw), _measurement_arm(right_raw)
            if left_arm is None or right_arm is None:
                return None
            if any(len(left_arm[key]) != len(right_arm[key]) for key in ("total_ns", "prefill_ns", "decode_ns")):
                return None
            seen.add(pair_id)
            pairs.append({"pair_id": pair_id, "order": order, "arms": {left_name: left_arm, right_name: right_arm}})
        orders = [row["order"] for row in pairs]
        if set(orders) != {"AB", "BA"}:
            return None
        body = {"pairs": pairs, "pair_count": len(pairs)}
        return {**body, "evidence_sha256": hashlib.sha256(canonical_bytes(body, max_bytes=64 * 1024)).hexdigest()}

    # Legacy/fake envelopes may provide full raw-shaped arm lists directly.
    # This path is retained only when raw_pairs is absent; if raw_pairs exists
    # but has the wrong type, fail closed above rather than silently downgrading.
    left = source.get(left_name)
    right = source.get(right_name)
    if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)) or len(left) != len(right) or not 1 <= len(left) <= 6:
        return None
    pairs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for one, two in zip(left, right):
        if not isinstance(one, Mapping) or not isinstance(two, Mapping):
            return None
        pair_id = one.get("pair_id")
        order = one.get("order")
        if not isinstance(pair_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", pair_id) or pair_id in seen:
            return None
        if order not in {"AB", "BA"} or two.get("pair_id") != pair_id or two.get("order") != order:
            return None
        seen.add(pair_id)
        left_arm = _measurement_arm(one)
        right_arm = _measurement_arm(two)
        if left_arm is None or right_arm is None:
            return None
        if any(len(left_arm[key]) != len(right_arm[key]) for key in ("total_ns", "prefill_ns", "decode_ns")):
            return None
        pairs.append({"pair_id": pair_id, "order": order, "arms": {left_name: left_arm, right_name: right_arm}})
    orders = [row["order"] for row in pairs]
    if set(orders) != {"AB", "BA"}:
        return None
    body = {"pairs": pairs, "pair_count": len(pairs)}
    return {**body, "evidence_sha256": hashlib.sha256(canonical_bytes(body, max_bytes=64 * 1024)).hexdigest()}


def _measurement_evidence(calibration: Any, test: Any) -> dict[str, Any]:
    calibration_stage = _measurement_stage(calibration, left_name="aa_baseline_samples", right_name="aa_control_samples")
    test_stage = _measurement_stage(test, left_name="baseline_samples", right_name="candidate_samples")
    if calibration_stage is None or test_stage is None:
        reason = "measurement_evidence_invalid_or_missing"
        body = {"schema": "friday.optimizer.measurement-evidence.v1", "status": "unavailable", "reason": reason}
        return {**body, "evidence_sha256": hashlib.sha256(canonical_bytes(body, max_bytes=64 * 1024)).hexdigest()}
    body = {"schema": "friday.optimizer.measurement-evidence.v1", "status": "complete", "calibration": calibration_stage, "test": test_stage}
    return {**body, "evidence_sha256": hashlib.sha256(canonical_bytes(body, max_bytes=256 * 1024)).hexdigest()}


def _evaluate_worker(
    value: Any,
    *,
    exact: ExactFingerprint,
    candidate_id: str,
    evaluator: Evaluator,
    calibration: Any = None,
) -> tuple[Any | None, dict[str, Any]]:
    payload = _unwrap_stage_payload(value)
    source = payload.get("evaluation") if isinstance(payload.get("evaluation"), Mapping) else payload
    # AB observations belong exclusively to the test stage.  A/A observations
    # are read exclusively from the calibration envelope; never reinterpret
    # test baseline/candidate rows as A/A controls.
    baseline = _metric_samples(source.get("baseline_samples"))
    candidate = _metric_samples(source.get("candidate_samples"))
    calibration_payload = _unwrap_stage_payload(calibration) if calibration is not None else {}
    # ParsedIronMuleResult currently exposes the raw A/A arrays at its root
    # while also carrying a diagnostic ``calibration`` summary.  The summary
    # is not measurement evidence.  Only an explicit root pair, or (when the
    # root has no pair fields) an explicit evaluation container, is trusted.
    if "aa_baseline_samples" in calibration_payload or "aa_control_samples" in calibration_payload or "aa_left" in calibration_payload or "aa_right" in calibration_payload:
        calibration_source = calibration_payload
    else:
        candidate_container = calibration_payload.get("evaluation")
        calibration_source = candidate_container if isinstance(candidate_container, Mapping) else {}
    aa_left = _metric_samples(calibration_source.get("aa_baseline_samples", calibration_source.get("aa_left")))
    aa_right = _metric_samples(calibration_source.get("aa_control_samples", calibration_source.get("aa_right")))
    resources = _resource_values(source.get("resources"))
    baseline_correctness = _correctness_value(source.get("baseline_correctness"))
    candidate_correctness = _correctness_value(source.get("candidate_correctness"))
    correctness = source.get("correctness")
    pair_correctness = None
    if isinstance(correctness, (list, tuple)) and len(correctness) == 2:
        left, right = _correctness_value(correctness[0]), _correctness_value(correctness[1])
        if left is not None and right is not None:
            pair_correctness = (left, right)
    try:
        decision = evaluator.evaluate(
            exact, candidate_id, baseline, candidate,
            aa_baseline_samples=aa_left, aa_control_samples=aa_right,
            correctness=pair_correctness,
            baseline_correctness=baseline_correctness,
            candidate_correctness=candidate_correctness,
            resources=resources,
            qualified=("fixed_compiled_cache", "head_skip_prefill"),
        )
    except Exception as exc:
        return None, {"status": "inconclusive", "qualified": False, "reasons": ["evaluator_failed:" + type(exc).__name__], "pair_count": len(aa_left)}
    missing: list[str] = []
    if not baseline or not candidate:
        missing.append("raw_ab_evidence_missing")
    if not aa_left or not aa_right:
        missing.append("raw_aa_evidence_missing")
    if not resources:
        missing.append("resource_evidence_missing")
    summary = {
        "status": decision.status,
        "qualified": bool(decision.qualified) and not missing,
        "reasons": list(dict.fromkeys((*missing, *decision.reasons))),
        "evidence_hash": decision.evidence_hash,
        "ratios": dict(decision.baseline_ratios),
        "confidence_intervals": {key: list(value) for key, value in decision.confidence_intervals.items()},
        "pair_count": len(aa_left),
        "resource_count": len(resources),
        "resources": [_safe_value(item.as_dict()) for item in resources],
        "correctness": {
            "baseline_response_hash": None if baseline_correctness is None else baseline_correctness.response_hash,
            "candidate_response_hash": None if candidate_correctness is None else candidate_correctness.response_hash,
            "baseline_passed": None if baseline_correctness is None else baseline_correctness.passed,
            "candidate_passed": None if candidate_correctness is None else candidate_correctness.passed,
        },
    }
    return decision, summary


@dataclass(slots=True)
class RealSessionController:
    plan: SessionPlan
    checkout: str | os.PathLike[str]
    expected_head: str
    interpreter: str | os.PathLike[str]
    model_identity: Any
    workload_contract: Any
    runtime_commit: str
    session_id: str
    memory: str | os.PathLike[str]
    result_out: str | os.PathLike[str]
    execute: bool = False
    collector: Collector | None = None
    probe: Any = None
    adapter_factory: Callable[[IronMuleCheckoutBinding], IronMuleTuneAdapter] = IronMuleTuneAdapter
    binding_factory: Callable[..., IronMuleCheckoutBinding] | None = None
    readiness_policy: ReadinessPolicy = field(default_factory=ReadinessPolicy)
    clock: Any = None
    optimizer_root: str | os.PathLike[str] | None = None
    identity_provider: Callable[[], Mapping[str, Any]] | None = None
    evaluator: Evaluator | Any | None = None

    @staticmethod
    def _stage_result(adapter: Any, stage: str, *, deadline: float, session_id: str) -> Any:
        """Plan, authorize and consume exactly one private adapter stage."""
        planner = getattr(adapter, "plan_stage", None)
        authorizer = getattr(adapter, "authorize_stage", None)
        runner = getattr(adapter, "run_stage", None)
        if callable(planner) and callable(authorizer) and callable(runner):
            spec = planner(stage, candidate_id=ALLOWED_CANDIDATE, qualified=("fixed_compiled_cache", "head_skip_prefill"))
            try:
                authorized = authorizer(spec, session_id)
                return runner(authorized, deadline=deadline, session_id=session_id)
            finally:
                cleanup = getattr(spec, "cleanup", None)
                if callable(cleanup):
                    cleanup()
        method = getattr(adapter, stage, None)
        if not callable(method):
            raise RealSessionError("stage_unavailable")
        return method(deadline=deadline, session_id=session_id)

    def _revalidate_before_stage(self, adapter: Any, exact: ExactFingerprint) -> None:
        current = dict(self.identity_provider() if self.identity_provider is not None else optimizer_identity(self.optimizer_root))
        prereg = self.plan.preregistration
        checks = {
            "head": prereg.optimizer_head,
            "optimizer_tree_sha256": prereg.optimizer_tree_sha256,
            "code_manifest_sha256": prereg.code_manifest_sha256,
            "adapter_sha256": prereg.adapter_sha256,
            "worker_sha256": prereg.worker_sha256,
            "registry_sha256": prereg.registry_sha256,
        }
        if any(current.get(key) != value for key, value in checks.items()):
            raise RealSessionError("optimizer_identity_changed")
        validation = adapter.validate_checkout()
        if validation.head != prereg.checkout_head or validation.source_digest != prereg.source_digest:
            raise RealSessionError("checkout_identity_changed")
        fresh = collect_fingerprint(model_identity=self.model_identity, workload_contract=self.workload_contract, runtime_commit=prereg.optimizer_head, collector=self.collector)
        workload_file_hash, workload_contract_hash = _workload_hashes(self.workload_contract)
        if fresh.fingerprint_hash != exact.fingerprint_hash or workload_file_hash != prereg.workload_file_sha256 or workload_contract_hash != prereg.workload_contract_sha256 or (prereg.model_identity_source_sha256 is not None and fresh.report.get("model_source_sha256") != prereg.model_identity_source_sha256):
            raise RealSessionError("fingerprint_changed")

    def run(self) -> "SessionExecutionOutcome":
        if not self.execute:
            raise RealSessionError("explicit_execute_required")
        _session_id(self.session_id)
        result_path = _bounded_path(self.result_out, "result_out")
        memory_path = _existing_file(self.memory, "memory", maximum=64 * 1024 * 1024)
        try:
            result_path.lstat()
        except FileNotFoundError:
            pass
        else:
            raise RealSessionError("result_exists")
        memory_before = _memory_preflight(memory_path)
        if not self.plan.ready:
            raise RealSessionError("session_plan_blocked")
        exact_raw = self.plan.fingerprint.report.get("fingerprint")
        if not isinstance(exact_raw, Mapping):
            raise RealSessionError("fingerprint_missing")
        exact = ExactFingerprint.from_mapping(exact_raw)
        make_binding = self.binding_factory or IronMuleCheckoutBinding
        binding = make_binding(checkout=self.checkout, expected_head=_commit(self.expected_head, "expected_head"), interpreter=self.interpreter, fingerprint=exact)
        adapter = self.adapter_factory(binding)
        validation = adapter.validate_checkout()
        if validation.head != self.plan.preregistration.checkout_head or validation.source_digest != self.plan.preregistration.source_digest:
            raise RealSessionError("checkout_binding_mismatch")
        worker_hash = _worker_hash()
        if self.plan.preregistration.worker_sha256 != worker_hash:
            raise RealSessionError("worker_binding_mismatch")
        probe = self.probe or MacSystemProbe()
        started = _clock_now(self.clock)
        deadline = started + self.plan.duration_minutes * 60
        lease_parent = _lease_path(memory_path).parent
        if not lease_parent.exists():
            lease_parent.mkdir(mode=0o700)
            os.chmod(lease_parent, 0o700)
        lease = HardwareLease(_lease_path(memory_path), fingerprint=exact.fingerprint_hash, clock=self.clock)
        state = ["requested"]
        transitions: list[dict[str, Any]] = []
        _transition(state, transitions, "waiting", "awaiting_stable_readiness")
        outcomes: dict[str, Any] = {}
        raw_outcomes: dict[str, Any] = {}
        evaluation_summary: dict[str, Any] = {"status": "inconclusive", "qualified": False, "reasons": ["evaluator_not_run"], "pair_count": 0}
        readiness: ReadinessDecision | None = None
        status = "inconclusive"
        reason = "no_recommendation"
        cleanup_error = False
        try:
            lease.acquire()
        except LeaseBusy as exc:
            # A competing owner means no hardware work started; do not create
            # a terminal result that could be mistaken for an attempted run.
            raise RealSessionError("lease_busy") from exc
        except Exception as exc:
            raise RealSessionError("lease_unavailable") from exc
        try:
            gate = ReadinessGate(probe, self.readiness_policy, clock=self.clock)
            readiness = gate.check(deadline=deadline)
            if not readiness.ready or not lease.validate():
                _transition(state, transitions, "baseline", "readiness_failed")
                reason = "readiness_failed"
            else:
                _transition(state, transitions, "calibrating", "readiness_confirmed")
                # Re-check immediately before every model stage.  A clean
                # initial sample is not evidence that AC/load stayed safe.
                readiness = gate.check(deadline=deadline)
                if not readiness.ready or _clock_now(self.clock) >= deadline or not lease.validate():
                    reason = "readiness_failed" if not readiness.ready else "timeout"
                    _transition(state, transitions, "baseline", reason)
                else:
                    self._revalidate_before_stage(adapter, exact)
                    raw_outcomes["calibrate"] = self._stage_result(adapter, "calibrate", deadline=deadline, session_id=self.session_id)
                    outcomes["calibrate"] = _safe_value(raw_outcomes["calibrate"])
                    if isinstance(outcomes["calibrate"], Mapping):
                        outcomes["calibrate"] = dict(outcomes["calibrate"], evidence_hash=hashlib.sha256(canonical_bytes(outcomes["calibrate"], max_bytes=64 * 1024)).hexdigest())
                    cal_outcome = outcomes["calibrate"].get("outcome") if isinstance(outcomes["calibrate"], Mapping) else None
                    if cal_outcome not in {"ok", "pass", "qualified"}:
                        _transition(state, transitions, "inconclusive", "calibration_inconclusive")
                        _transition(state, transitions, "baseline", "no_recommendation")
                        reason = "calibration_inconclusive"
                    else:
                        _transition(state, transitions, "testing", "calibration_passed")
                        readiness = gate.check(deadline=deadline)
                        if not readiness.ready or _clock_now(self.clock) >= deadline or not lease.validate():
                            reason = "readiness_failed" if not readiness.ready else "timeout"
                            _transition(state, transitions, "baseline", reason)
                        else:
                            self._revalidate_before_stage(adapter, exact)
                            raw_outcomes["test"] = self._stage_result(adapter, "test", deadline=deadline, session_id=self.session_id)
                            outcomes["test"] = _safe_value(raw_outcomes["test"])
                            if isinstance(outcomes["test"], Mapping):
                                outcomes["test"] = dict(outcomes["test"], evidence_hash=hashlib.sha256(canonical_bytes(outcomes["test"], max_bytes=64 * 1024)).hexdigest())
                            # A final post-test readiness check closes the
                            # window in which foreign load or power can appear
                            # after the worker exits but before evaluation.
                            readiness = gate.check(deadline=deadline)
                            test_outcome = outcomes["test"].get("outcome") if isinstance(outcomes["test"], Mapping) else None
                            if not readiness.ready or not lease.validate():
                                reason = "readiness_failed"
                                _transition(state, transitions, "inconclusive", reason)
                                _transition(state, transitions, "baseline", "no_recommendation")
                            elif test_outcome in {"qualified", "pass", "ok"}:
                                decision, evaluation_summary = _evaluate_worker(raw_outcomes["test"], calibration=raw_outcomes.get("calibrate"), exact=exact, candidate_id=self.plan.candidate_id, evaluator=self.evaluator or Evaluator())
                                if decision is not None and bool(evaluation_summary.get("qualified")):
                                    _transition(state, transitions, "qualified", "evaluator_qualified")
                                    status = "qualified"
                                    reason = "qualified_no_activation"
                                else:
                                    _transition(state, transitions, "inconclusive", "evaluator_no_recommendation")
                                    _transition(state, transitions, "baseline", "no_recommendation")
                                    reason = "evaluator_no_recommendation"
                            else:
                                _transition(state, transitions, "inconclusive", "test_inconclusive")
                                _transition(state, transitions, "baseline", "no_recommendation")
                                reason = "test_inconclusive"
        except (LeaseBusy, Exception) as exc:
            if state[0] not in {"baseline", "inconclusive"}:
                _transition(state, transitions, "baseline", "session_error")
            reason = "lease_busy" if isinstance(exc, LeaseBusy) else "session_error"
        finally:
            try:
                lease.release()
            except Exception:
                cleanup_error = True
        run_ok = bool(
            readiness is not None and readiness.ready
            and isinstance(outcomes.get("calibrate"), Mapping)
            and outcomes["calibrate"].get("outcome") in {"ok", "pass", "qualified"}
            and isinstance(outcomes.get("test"), Mapping)
            and outcomes["test"].get("outcome") in {"ok", "pass", "qualified"}
            and evaluation_summary.get("status") in {"qualified", "inconclusive", "rejected"}
        )
        # A stage exception/timeout/readiness failure is not a completed shadow
        # even though it still receives a durable terminal fallback record.
        if reason in {"readiness_failed", "timeout", "session_error", "lease_unavailable", "lease_busy"}:
            run_ok = False
        if cleanup_error:
            run_ok = False
            reason = "cleanup_error"
        created_at = _now_rfc3339()
        measurement_evidence = _measurement_evidence(raw_outcomes.get("calibrate"), raw_outcomes.get("test"))
        if measurement_evidence.get("status") != "complete":
            run_ok = False
            evaluation_summary["qualified"] = False
            reasons = list(evaluation_summary.get("reasons", ()))
            if "measurement_evidence_missing_or_invalid" not in reasons:
                reasons.append("measurement_evidence_missing_or_invalid")
            evaluation_summary["reasons"] = reasons
            # If an earlier evaluator result was favorable, revoke that state
            # before constructing the terminal record.  A result may never
            # claim qualified and no-recommendation simultaneously.
            if state[0] == "qualified":
                _transition(state, transitions, "inconclusive", "measurement_evidence_missing_or_invalid")
                _transition(state, transitions, "baseline", "no_recommendation")
                status = "inconclusive"
                reason = "measurement_evidence_missing_or_invalid"
        result: dict[str, Any] = {
            "schema": RESULT_SCHEMA,
            "experiment": self.plan.preregistration.experiment,
            "classification": reason,
            "run_ok": run_ok,
            "recommendation_available": False,
            "session_id_hash": hashlib.sha256(self.session_id.encode("utf-8")).hexdigest(),
            "created_at_utc": created_at,
            "status": status,
            "state": state[0],
            "reason": reason,
            "decision": "shadow_recommendation" if evaluation_summary.get("qualified") else "no_recommendation",
            "activation": "disabled",
            "no_recommendation": not bool(evaluation_summary.get("qualified")),
            "no_activation": True,
            "start_authorization_hash": start_authorization_hash(self.session_id, self.plan.duration_minutes, self.plan.candidate_id, True, self.plan.preregistration.start_authorization_schema, self.plan.preregistration.sha256),
            "transitions": transitions,
            "states": [item["to"] for item in transitions],
            "fingerprint_hash": self.plan.fingerprint.fingerprint_hash,
            "fingerprint_report_hash": self.plan.fingerprint.report_hash,
            "source": {
                "source_digest": validation.source_digest,
                "registry_hash": EXECUTION_FILE_REGISTRY_HASH,
                "worker_sha256": worker_hash,
                "adapter_sha256": self.plan.preregistration.adapter_sha256,
            },
            "preregistration_hash": self.plan.preregistration.sha256,
            "optimizer_head": self.plan.preregistration.optimizer_head,
            "optimizer_tree_sha256": self.plan.preregistration.optimizer_tree_sha256,
            "code_manifest_sha256": self.plan.preregistration.code_manifest_sha256,
            "adapter_sha256": self.plan.preregistration.adapter_sha256,
            "worker_sha256": self.plan.preregistration.worker_sha256,
            "checkout_head": validation.head,
            "source_digest": validation.source_digest,
            "registry_sha256": self.plan.preregistration.registry_sha256,
            "workload_file_sha256": self.plan.preregistration.workload_file_sha256,
            "workload_contract_sha256": self.plan.preregistration.workload_contract_sha256,
            "model_identity_sha256": self.plan.preregistration.model_identity_sha256,
            "model_manifest_sha256": self.plan.preregistration.model_manifest_sha256,
            "tokenizer_sha256": self.plan.preregistration.tokenizer_sha256,
            "preregistration": {
                "experiment": self.plan.preregistration.experiment,
                "optimizer_head": self.plan.preregistration.optimizer_head,
                "optimizer_tree_sha256": self.plan.preregistration.optimizer_tree_sha256,
                "code_manifest_sha256": self.plan.preregistration.code_manifest_sha256,
                "adapter_sha256": self.plan.preregistration.adapter_sha256,
                "worker_sha256": self.plan.preregistration.worker_sha256,
                "registry_sha256": self.plan.preregistration.registry_sha256,
                "result_schema": self.plan.preregistration.result_schema,
                "start_authorization_schema": self.plan.preregistration.start_authorization_schema,
            },
            "checkout": {"head": validation.head, "source_digest": validation.source_digest, "interpreter_sha256": validation.interpreter_sha256},
            "dataset_hash": self.plan.preregistration.dataset_hash,
            "code_hash": self.plan.preregistration.code_hash,
            "candidate_id": self.plan.candidate_id,
            "duration_minutes": self.plan.duration_minutes,
            "readiness": None if readiness is None else {"ready": readiness.ready, "reasons": list(readiness.reasons), "sample_count": len(readiness.samples)},
            "worker": outcomes,
            "evaluation": evaluation_summary,
            "metrics": evaluation_summary.get("ratios", {}),
            "resources": evaluation_summary.get("resources", []),
            "confirmation_pair_count": evaluation_summary.get("pair_count", 0),
            "measurement_evidence": measurement_evidence,
        }
        result["recommendation_available"] = bool(evaluation_summary["qualified"])
        result["no_recommendation"] = not result["recommendation_available"]
        result["result_hash"] = hashlib.sha256(canonical_bytes(result, max_bytes=_MAX_JSON)).hexdigest()
        SessionResult.from_mapping(result)
        encoded = canonical_bytes(result, max_bytes=_MAX_JSON)
        _atomic_new(result_path, encoded)
        history_written = False
        history_error: str | None = None
        try:
            if _memory_binding(memory_path) != memory_before:
                raise RealSessionError("memory_changed")
            with OptimizationMemoryV2(memory_path) as writable:
                HistoryWriter(writable).append(_result_event(result, result["session_id_hash"], code_hash=self.plan.preregistration.code_hash, dataset_hash=self.plan.preregistration.dataset_hash, prereg=self.plan.preregistration.sha256, fingerprint=self.plan.fingerprint.fingerprint_hash or "0" * 64, candidate=self.plan.candidate_id))
            history_written = True
        except Exception as exc:
            # The terminal result is durable; memory is never silently created
            # or repaired after the session.  Wrapper metadata remains outside
            # the immutable persisted map.
            history_error = type(exc).__name__
        return SessionExecutionOutcome(result, history_written, history_error, persistence_ok=history_written)


@dataclass(frozen=True, slots=True)
class SessionExecutionOutcome:
    """Execution metadata kept separate from the immutable result file."""

    result: Mapping[str, Any]
    history_written: bool
    history_error: str | None
    persistence_ok: bool

    def __post_init__(self) -> None:
        validated = SessionResult.from_mapping(self.result)
        object.__setattr__(self, "result", validated.payload)
        if not isinstance(self.history_written, bool) or (self.history_error is not None and not isinstance(self.history_error, str)):
            raise RealSessionError("session_execution_outcome_invalid")

    def as_dict(self) -> dict[str, Any]:
        return {
            "result": dict(self.result),
            "history_written": self.history_written,
            "history_error": self.history_error,
            "persistence_ok": self.persistence_ok,
        }


def run_session(**kwargs: Any) -> SessionExecutionOutcome:
    controller = RealSessionController(**kwargs)
    return controller.run()


# Friendly names for integrations that call this boundary a shadow session.
RealFingerprintReport = FingerprintReport
RealSessionPlan = SessionPlan


@dataclass(frozen=True, slots=True)
class SessionResult:
    """Immutable view/validator for a persisted terminal session record."""

    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.payload, Mapping) or self.payload.get("schema") != RESULT_SCHEMA:
            raise RealSessionError("session_result_schema_invalid")
        required = {"experiment", "classification", "run_ok", "recommendation_available", "no_activation", "session_id_hash", "start_authorization_hash", "created_at_utc", "transitions", "fingerprint_hash", "preregistration_hash", "candidate_id", "measurement_evidence", "result_hash"}
        if not required.issubset(self.payload):
            raise RealSessionError("session_result_field_missing")
        def public(value: Any, depth: int = 0) -> None:
            if depth > 8:
                raise RealSessionError("session_result_too_deep")
            if isinstance(value, Mapping):
                for key, item in value.items():
                    if isinstance(key, str) and key.lower() in {"session_id", "prompt", "prompts", "stdout", "stderr", "raw_log", "absolute_path"}:
                        raise RealSessionError("session_result_private_field")
                    public(item, depth + 1)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    public(item, depth + 1)
        public(self.payload)
        _sha(self.payload["result_hash"], "result_hash")
        body = dict(self.payload)
        claimed = body.pop("result_hash")
        if hashlib.sha256(canonical_bytes(body, max_bytes=_MAX_JSON)).hexdigest() != claimed:
            raise RealSessionError("session_result_hash_mismatch")
        _sha(self.payload["session_id_hash"], "session_id_hash")
        created = self.payload["created_at_utc"]
        if not isinstance(created, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", created):
            raise RealSessionError("session_result_timestamp_invalid")
        measurement = self.payload["measurement_evidence"]
        if not isinstance(measurement, Mapping) or measurement.get("schema") != "friday.optimizer.measurement-evidence.v1" or measurement.get("status") not in {"complete", "unavailable"}:
            raise RealSessionError("session_result_measurement_evidence_invalid")
        evidence_hash = measurement.get("evidence_sha256")
        if not isinstance(evidence_hash, str) or not _SHA.fullmatch(evidence_hash):
            raise RealSessionError("session_result_measurement_hash_invalid")
        evidence_body = dict(measurement)
        evidence_body.pop("evidence_sha256", None)
        if hashlib.sha256(canonical_bytes(evidence_body, max_bytes=256 * 1024)).hexdigest() != evidence_hash:
            raise RealSessionError("session_result_measurement_hash_mismatch")
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))

    @property
    def result_hash(self) -> str:
        return str(self.payload["result_hash"])

    def as_dict(self) -> dict[str, Any]:
        return dict(self.payload)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SessionResult":
        return cls(value)
RealSession = RealSessionController
session_plan = build_session_plan
fingerprint = collect_fingerprint


__all__ = [
    "ALLOWED_CANDIDATE", "ALLOWED_STAGES", "FINGERPRINT_SCHEMA", "FingerprintReport",
    "RealFingerprintReport",
    "PREREG_SCHEMA", "PLAN_SCHEMA", "RESULT_SCHEMA", "Preregistration", "RealSessionController",
    "RealSessionError", "RealSessionUnavailable", "SessionPlan", "RealSessionPlan", "SessionResult", "SessionExecutionOutcome", "RealSession", "build_session_plan", "session_plan",
    "collect_fingerprint", "fingerprint", "run_session", "optimizer_source_manifest", "optimizer_identity", "start_authorization_hash",
]
