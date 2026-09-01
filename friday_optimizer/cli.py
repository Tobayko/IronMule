"""Closed, bounded command line boundary for the Friday optimizer.

The CLI is intentionally a policy boundary, not another optimizer.  Reads are
read-only by default; the only operations which can create or modify state are
guarded by explicit ``--execute`` (and shadow history additionally requires
``--write-history``).  All machine-readable output is canonical JSON and does
not disclose absolute local paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import stat
import sys
import tempfile
import threading
from enum import IntEnum
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical import canonical_bytes, canonical_dumps, loads_strict
from .corpus import EVIDENCE_CONTRACTS, CorpusAuditor
from .dashboard import DashboardError, serve
from .dataset import DatasetBuilder
from .evaluator import CorrectnessResult, MetricSample, ResourceResult
from .decisions import CENSORING, REWARD_METRICS, SELECTION_RULES, DecisionError, SelectionPolicy
from .fingerprint import ExactFingerprint
from .memory import OptimizationMemoryV2, ReadOnlyMemoryView
from .orchestrator import (
    OptimizerConfig,
    OptimizerOrchestrator,
    ShadowRequest,
    _inventory_identity_hash,
)
from .portfolio import PortfolioError, build_portfolio
from .readiness import MacSystemProbe
from .replay import DEFAULT_MIN_SAMPLES, ReplayError
from .collector import Collector
from .real_session import (
    RealSessionError,
    build_session_plan,
    collect_fingerprint,
    run_session,
)


MAX_REQUEST_BYTES = 4 * 1024 * 1024
MAX_OUTPUT_BYTES = 1024 * 1024
MAX_PATH_BYTES = 4096


class ExitCode(IntEnum):
    OK = 0
    UNAVAILABLE = 1
    USAGE = 64
    DATA = 65
    INTERNAL = 70
    NOT_AUTHORIZED = 78


class CLIError(Exception):
    """An expected, user-facing CLI failure with a stable exit code."""

    def __init__(self, code: ExitCode, reason: str = "error") -> None:
        super().__init__(reason)
        self.code = code
        self.reason = reason


class HelpRequested(Exception):
    pass


class _Parser(argparse.ArgumentParser):
    """Argparse which maps every parser error to the CLI's stable code."""

    def error(self, message: str) -> None:  # pragma: no cover - argparse calls this
        raise CLIError(ExitCode.USAGE, "usage_error")

    def exit(self, status: int = 0, message: str | None = None) -> None:
        if status == 0:
            if message:
                self._print_message(message, sys.stdout)
            raise HelpRequested
        raise CLIError(ExitCode.USAGE, "usage_error")


def _bounded_path(value: str, field: str) -> Path:
    if not isinstance(value, str) or not value or len(value.encode("utf-8", "surrogatepass")) > MAX_PATH_BYTES:
        raise CLIError(ExitCode.USAGE, "invalid_path")
    try:
        return Path(os.path.abspath(value))
    except (TypeError, ValueError, OSError):
        raise CLIError(ExitCode.USAGE, "invalid_path") from None


def _reject_symlink_ancestors(path: Path, *, include_leaf: bool = True) -> None:
    current = Path(path.anchor or os.sep)
    parts = path.parts[1:] if path.anchor else path.parts
    for index, component in enumerate(parts):
        current /= component
        if not include_leaf and index == len(parts) - 1:
            break
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        except OSError:
            raise CLIError(ExitCode.DATA, "path_unreadable") from None
        if stat.S_ISLNK(info.st_mode):
            raise CLIError(ExitCode.DATA, "symlink_path_refused")


def _existing_directory(value: str, field: str) -> Path:
    path = _bounded_path(value, field)
    _reject_symlink_ancestors(path)
    try:
        info = path.lstat()
    except FileNotFoundError:
        raise CLIError(ExitCode.DATA, f"{field}_missing") from None
    except OSError:
        raise CLIError(ExitCode.DATA, f"{field}_unreadable") from None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise CLIError(ExitCode.DATA, f"{field}_must_be_directory")
    return path


def _safe_existing_file(value: str, field: str, *, max_bytes: int = MAX_REQUEST_BYTES) -> Path:
    path = _bounded_path(value, field)
    _reject_symlink_ancestors(path)
    try:
        info = path.lstat()
    except FileNotFoundError:
        raise CLIError(ExitCode.DATA, f"{field}_missing") from None
    except OSError:
        raise CLIError(ExitCode.DATA, f"{field}_unreadable") from None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise CLIError(ExitCode.DATA, f"{field}_must_be_regular")
    if info.st_size > max_bytes:
        raise CLIError(ExitCode.DATA, f"{field}_too_large")
    return path


def _safe_parent(path: Path) -> Path:
    parent = path.parent
    _reject_symlink_ancestors(parent)
    try:
        info = parent.lstat()
    except FileNotFoundError:
        raise CLIError(ExitCode.DATA, "parent_missing") from None
    except OSError:
        raise CLIError(ExitCode.DATA, "parent_unreadable") from None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise CLIError(ExitCode.DATA, "parent_must_be_directory")
    return parent


def _target_new_file(value: str, field: str) -> tuple[Path, Path]:
    path = _bounded_path(value, field)
    parent = _safe_parent(path)
    try:
        path.lstat()
    except FileNotFoundError:
        return path, parent
    except OSError:
        raise CLIError(ExitCode.DATA, "output_unreadable") from None
    raise CLIError(ExitCode.DATA, "output_exists")


def _relative_or_external(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix() or "."
    except ValueError:
        return "<external>"


def _stable_read(path: Path, *, max_bytes: int = MAX_REQUEST_BYTES) -> tuple[bytes, tuple[int, int, int, int]]:
    """Read one regular file through an O_NOFOLLOW descriptor and recheck it."""

    try:
        before = path.lstat()
    except OSError:
        raise CLIError(ExitCode.DATA, "input_unreadable") from None
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise CLIError(ExitCode.DATA, "input_must_be_regular")
    if before.st_size > max_bytes:
        raise CLIError(ExitCode.DATA, "input_too_large")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise CLIError(ExitCode.DATA, "input_unreadable") from None
    try:
        opened = os.fstat(descriptor)
        if stat.S_ISLNK(opened.st_mode) or not stat.S_ISREG(opened.st_mode):
            raise CLIError(ExitCode.DATA, "input_must_be_regular")
        if opened.st_size > max_bytes:
            raise CLIError(ExitCode.DATA, "input_too_large")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise CLIError(ExitCode.DATA, "input_too_large")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise CLIError(ExitCode.DATA, "input_changed")
        try:
            current = path.lstat()
        except OSError:
            raise CLIError(ExitCode.DATA, "input_changed") from None
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns):
            raise CLIError(ExitCode.DATA, "input_changed")
        return b"".join(chunks), (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
    finally:
        os.close(descriptor)


def _metadata_signature(paths: Sequence[Path]) -> tuple[tuple[str, int, int, int, int, str], ...]:
    result: list[tuple[str, int, int, int, int, str]] = []
    for path in paths:
        try:
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            result.append((str(path), info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, digest))
        except OSError:
            result.append((str(path), -1, -1, -1, -1, "unreadable"))
    return tuple(result)


def _memory_binding(path: Path) -> tuple[int, int, int, int, int, str]:
    """Bind a memory leaf's bytes and filesystem identity without writing.

    The writer class intentionally secures an existing database with chmod.
    Therefore this preflight must happen through the read-only view first, and
    the binding is checked once more immediately before that writer is opened.
    """

    _safe_existing_file(str(path), "memory", max_bytes=64 * 1024 * 1024)
    try:
        info = path.lstat()
    except OSError:
        raise CLIError(ExitCode.DATA, "memory_unreadable") from None
    raw, identity = _stable_read(path, max_bytes=64 * 1024 * 1024)
    return (
        int(identity[0]),
        int(identity[1]),
        int(identity[2]),
        stat.S_IMODE(info.st_mode),
        int(identity[3]),
        hashlib.sha256(raw).hexdigest(),
    )


def _memory_preflight(path: Path) -> tuple[int, int, int, int, int, str] | None:
    """Validate an existing DB entirely read-only before any writer exists."""

    if not (path.exists() or path.is_symlink()):
        return None
    binding = _memory_binding(path)
    try:
        view = OptimizationMemoryV2.open_read_only(path)
        try:
            integrity = view.integrity()
            if not integrity.ok or not view.schema_ok:
                raise CLIError(ExitCode.DATA, "memory_preflight_failed")
        finally:
            view.close()
    except CLIError:
        raise
    except Exception:
        raise CLIError(ExitCode.DATA, "memory_preflight_failed") from None
    current = _memory_binding(path)
    if current != binding:
        raise CLIError(ExitCode.DATA, "memory_changed")
    return binding


def _assert_memory_binding(path: Path, binding: tuple[int, int, int, int, int, str] | None) -> None:
    """Fail closed if a memory target changed between preflight and writer."""

    if binding is None:
        if path.exists() or path.is_symlink():
            raise CLIError(ExitCode.DATA, "memory_changed")
        return
    if _memory_binding(path) != binding:
        raise CLIError(ExitCode.DATA, "memory_changed")


def _json_file(path: Path) -> Any:
    raw, _ = _stable_read(path)
    try:
        return loads_strict(raw, max_bytes=MAX_REQUEST_BYTES, max_depth=32, max_items=100_000, max_string_bytes=1_048_576)
    except Exception:
        raise CLIError(ExitCode.DATA, "invalid_json") from None


def _hash(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise CLIError(ExitCode.DATA, f"invalid_{field}")
    return value


def _object(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise CLIError(ExitCode.DATA, f"{field}_must_be_object")
    return value


def _strict_object(value: Any, field: str, allowed: set[str]) -> Mapping[str, Any]:
    mapping = _object(value, field)
    if set(mapping) - allowed:
        raise CLIError(ExitCode.DATA, f"unknown_{field}_field")
    return mapping


_SAMPLE_FIELDS = {"session_id", "pair_id", "arm", "order", "fingerprint", "workload", "ttft_seconds", "decode_tps", "tokens", "status", "error", "ttft", "ttft_ms", "tokens_per_second"}
_CORRECTNESS_FIELDS = {"token_ids", "text", "stop_reason", "physical_tokens", "visible_tokens", "response_hash", "passed", "error"}
_RESOURCE_FIELDS = {"peak_memory_bytes", "peak_rss_bytes", "swap_delta_bytes", "timed_out", "crashed", "foreign_load", "status", "error"}
_REQUEST_FIELDS = {"fingerprint", "candidate_id", "baseline_samples", "candidate_samples", "aa_baseline_samples", "aa_control_samples", "aa_pairs", "resources", "baseline_correctness", "candidate_correctness", "correctness", "parameters", "qualified", "session_id", "dataset_hash", "code_hash"}


def _sample(value: Any, field: str) -> MetricSample:
    return MetricSample(**dict(_strict_object(value, field, _SAMPLE_FIELDS)))


def _samples(value: Any, field: str) -> tuple[MetricSample, ...]:
    if not isinstance(value, list) or len(value) > 10_000:
        raise CLIError(ExitCode.DATA, f"{field}_must_be_bounded_array")
    return tuple(_sample(item, field) for item in value)


def _correctness(value: Any, field: str) -> CorrectnessResult:
    return CorrectnessResult(**dict(_strict_object(value, field, _CORRECTNESS_FIELDS)))


def _resources(value: Any, field: str) -> tuple[ResourceResult, ...]:
    if not isinstance(value, list) or len(value) > 10_000:
        raise CLIError(ExitCode.DATA, f"{field}_must_be_bounded_array")
    return tuple(ResourceResult(**dict(_strict_object(item, field, _RESOURCE_FIELDS))) for item in value)


def _shadow_request(value: Any, *, write_history: bool) -> ShadowRequest:
    raw = _strict_object(value, "request", _REQUEST_FIELDS)
    if "fingerprint" not in raw or "candidate_id" not in raw or "dataset_hash" not in raw or "code_hash" not in raw:
        raise CLIError(ExitCode.DATA, "shadow_identity_incomplete")
    try:
        fingerprint = ExactFingerprint.from_mapping(_object(raw["fingerprint"], "fingerprint"))
        baseline = _samples(raw.get("baseline_samples", []), "baseline_samples")
        candidate = _samples(raw.get("candidate_samples", []), "candidate_samples")
        aa_baseline = _samples(raw.get("aa_baseline_samples", []), "aa_baseline_samples")
        aa_control = _samples(raw.get("aa_control_samples", []), "aa_control_samples")
        raw_pairs = raw.get("aa_pairs", [])
        if not isinstance(raw_pairs, list) or len(raw_pairs) > 10_000:
            raise CLIError(ExitCode.DATA, "aa_pairs_must_be_bounded_array")
        aa_pairs: list[tuple[MetricSample, MetricSample]] = []
        for pair in raw_pairs:
            if not isinstance(pair, list) or len(pair) != 2:
                raise CLIError(ExitCode.DATA, "aa_pair_must_have_two_samples")
            aa_pairs.append((_sample(pair[0], "aa_pair"), _sample(pair[1], "aa_pair")))
        resources = _resources(raw.get("resources", []), "resources")
        baseline_correctness = _correctness(raw["baseline_correctness"], "baseline_correctness") if "baseline_correctness" in raw else None
        candidate_correctness = _correctness(raw["candidate_correctness"], "candidate_correctness") if "candidate_correctness" in raw else None
        correctness = None
        if "correctness" in raw:
            item = raw["correctness"]
            if not isinstance(item, list) or len(item) != 2:
                raise CLIError(ExitCode.DATA, "correctness_must_have_two_results")
            correctness = (_correctness(item[0], "correctness"), _correctness(item[1], "correctness"))
        parameters = raw.get("parameters", {})
        _strict_object(parameters, "parameters", set(parameters) if isinstance(parameters, Mapping) else set())
        qualified = raw.get("qualified", [])
        if not isinstance(qualified, list) or len(qualified) > 256 or any(not isinstance(item, str) or not item or len(item) > 128 for item in qualified):
            raise CLIError(ExitCode.DATA, "qualified_must_be_bounded_array")
        session_id = raw.get("session_id", "shadow")
        if not isinstance(session_id, str) or len(session_id) > 256:
            raise CLIError(ExitCode.DATA, "invalid_session_id")
        return ShadowRequest(
            fingerprint=fingerprint,
            candidate_id=raw["candidate_id"],
            baseline_samples=baseline,
            candidate_samples=candidate,
            aa_baseline_samples=aa_baseline,
            aa_control_samples=aa_control,
            aa_pairs=tuple(aa_pairs),
            resources=resources,
            baseline_correctness=baseline_correctness,
            candidate_correctness=candidate_correctness,
            correctness=correctness,
            parameters=parameters,
            qualified=tuple(qualified),
            session_id=session_id,
            dataset_hash=_hash(raw["dataset_hash"], "dataset_hash"),
            code_hash=_hash(raw["code_hash"], "code_hash"),
            write_history=write_history,
        )
    except CLIError:
        raise
    except Exception:
        raise CLIError(ExitCode.DATA, "invalid_shadow_request") from None


def _inventory_summary(inventory: Any, *, root: Path) -> dict[str, Any]:
    contracts = [
        {
            "contract_id": contract.contract_id,
            "version": contract.version,
            "source_basename": contract.source_basename,
            "contract_hash": contract.contract_hash,
            "feature_fields": len(contract.feature_paths),
            "label_fields": len(contract.label_paths),
        }
        for contract in EVIDENCE_CONTRACTS
    ]
    source_paths = [file.path for file in inventory.files]
    return {
        "root": ".",
        "files": len(inventory.files),
        "records": len(inventory.records),
        "usable_records": len(inventory.usable_records),
        "quality_counts": dict(sorted(inventory.quality_counts.items())),
        "duplicate_count": inventory.duplicate_count,
        "exclusions": dict(sorted(inventory.exclusions.items())),
        "issues": [
            {"path": str(issue.path), "code": issue.code, "terminal": bool(issue.terminal)}
            for issue in inventory.issues
        ],
        "contracts": contracts,
        "inventory_hash": _inventory_identity_hash(inventory),
        "source_metadata_unchanged": True,
        "source_metadata_count": len(_metadata_signature(source_paths)),
    }


def _dataset_summary(snapshot: Any, *, written: bool = False, output: str | None = None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "sha256": snapshot.sha256,
        "bytes": len(snapshot.canonical_bytes),
        "records": len(snapshot.records),
        "splits": {name: len(snapshot.splits[name]) for name in ("train", "validation", "holdout")},
        "card": snapshot.card.as_dict(),
        "written": written,
        "output": output,
    }


def _atomic_new_bytes(path: Path, payload: bytes) -> None:
    parent = _safe_parent(path)
    try:
        parent_before = parent.lstat()
    except OSError:
        raise CLIError(ExitCode.DATA, "parent_unreadable") from None
    if stat.S_ISLNK(parent_before.st_mode) or not stat.S_ISDIR(parent_before.st_mode):
        raise CLIError(ExitCode.DATA, "parent_must_be_directory")
    descriptor = -1
    temporary: Path | None = None
    try:
        for _ in range(8):
            name = f".{path.name}.tmp-{os.getpid()}-{next(tempfile._get_candidate_names())}"
            temporary = parent / name
            try:
                descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
                break
            except FileExistsError:
                continue
        if descriptor < 0 or temporary is None:
            raise CLIError(ExitCode.INTERNAL, "temporary_file_unavailable")
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise CLIError(ExitCode.INTERNAL, "write_failed")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            path.lstat()
        except FileNotFoundError:
            pass
        else:
            raise CLIError(ExitCode.DATA, "output_exists")
        current_parent = parent.lstat()
        if (current_parent.st_dev, current_parent.st_ino) != (parent_before.st_dev, parent_before.st_ino):
            raise CLIError(ExitCode.DATA, "parent_changed")
        os.replace(temporary, path)
        temporary = None
        try:
            directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            raise CLIError(ExitCode.INTERNAL, "directory_sync_failed") from None
    except CLIError:
        raise
    except OSError:
        raise CLIError(ExitCode.INTERNAL, "atomic_write_failed") from None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                pass


def _doctor(args: argparse.Namespace) -> tuple[dict[str, Any], ExitCode]:
    root = _existing_directory(args.root or ".", "root")
    memory = _bounded_path(args.memory, "memory") if args.memory else root / ".friday-data" / "optimizer-memory.sqlite3"
    profiles = _bounded_path(args.profiles, "profiles") if args.profiles else root / ".friday-data" / "optimizer-profiles.json"
    before = _metadata_signature([memory, profiles])
    config = OptimizerConfig(root, memory_path=memory, profile_path=profiles)
    probe_data: dict[str, Any] | None = None
    probe_ok = True
    if args.probe_system:
        try:
            sample = MacSystemProbe().sample()
            probe_data = {
                "known": sample.known,
                "ac_connected": sample.ac_connected,
                "low_power": sample.low_power,
                "memory_available_bytes": sample.memory_available_bytes,
                "memory_total_bytes": sample.memory_total_bytes,
                "swap_used_bytes": sample.swap_used_bytes,
                "load_1m": sample.load_1m,
                "cpu_percent": sample.cpu_percent,
                "workload_active": sample.workload_active,
                "process_tree_readable": sample.process_tree_readable,
                "errors": list(sample.errors),
            }
            probe_ok = sample.known
        except Exception:
            probe_data = {"known": False, "errors": ["probe_failed"]}
            probe_ok = False
    report = OptimizerOrchestrator(config).doctor()
    after = _metadata_signature([memory, profiles])
    unchanged = before == after
    payload = {
        "command": "doctor",
        "schema_version": 1,
        "ok": bool(report.ok and probe_ok and unchanged),
        "root": ".",
        "paths": {"memory": _relative_or_external(memory, root), "profiles": _relative_or_external(profiles, root)},
        "schemas": report.schemas,
        "fingerprint": report.fingerprint,
        "fingerprint_exact": report.fingerprint_exact,
        "readiness": report.readiness,
        "probe": probe_data,
        "reasons": list(report.reasons) + ([] if unchanged else ["source_metadata_changed"]),
        "source_metadata_unchanged": unchanged,
    }
    return payload, ExitCode.OK if payload["ok"] else ExitCode.UNAVAILABLE


def _audit(args: argparse.Namespace) -> tuple[dict[str, Any], ExitCode]:
    root = _existing_directory(args.root, "root")
    auditor = CorpusAuditor(root)
    discovered = auditor.discover()
    before = _metadata_signature([item.path for item in discovered])
    inventory = auditor.audit()
    after = _metadata_signature([item.path for item in discovered])
    unchanged = before == after
    payload = _inventory_summary(inventory, root=root)
    payload["command"] = "audit"
    payload["schema_version"] = 1
    payload["source_metadata_unchanged"] = unchanged
    if not unchanged:
        payload["issues"] = list(payload["issues"]) + [{"path": "<source>", "code": "source_metadata_changed", "terminal": True}]
    terminal = any(bool(issue.terminal) for issue in inventory.issues) or not unchanged
    payload["ok"] = not terminal
    return payload, ExitCode.OK if payload["ok"] else ExitCode.UNAVAILABLE


def _import(args: argparse.Namespace) -> tuple[dict[str, Any], ExitCode]:
    root = _existing_directory(args.root, "root")
    memory = _bounded_path(args.memory, "memory")
    if not args.execute:
        return {"command": "import", "schema_version": 1, "ok": False, "authorized": False, "reason": "explicit_execute_required", "memory_created": False}, ExitCode.NOT_AUTHORIZED
    _safe_parent(memory)
    binding = _memory_preflight(memory)
    config = OptimizerConfig(root, memory_path=memory)
    orchestrator = OptimizerOrchestrator(config)
    inventory = orchestrator.audit(root)
    _assert_memory_binding(memory, binding)
    report = orchestrator.import_inventory(inventory)
    payload = report.as_dict()
    # ImportReport carries the source root for Python callers; the process
    # boundary deliberately replaces that machine-layout disclosure.
    payload["source_root"] = "."
    payload.update({"command": "import", "schema_version": 1, "authorized": True, "memory_created": binding is None})
    return payload, ExitCode.OK if report.ok else ExitCode.DATA


def _dataset(args: argparse.Namespace) -> tuple[dict[str, Any], ExitCode]:
    root = _existing_directory(args.root, "root")
    output: Path | None = None
    parent: Path | None = None
    if args.out:
        if not args.execute:
            return {"command": "dataset", "schema_version": 1, "ok": False, "authorized": False, "reason": "explicit_execute_required", "written": False}, ExitCode.NOT_AUTHORIZED
        output, parent = _target_new_file(args.out, "out")
    elif args.execute:
        raise CLIError(ExitCode.USAGE, "execute_requires_out")
    orchestrator = OptimizerOrchestrator(root=root)
    inventory = orchestrator.audit(root)
    snapshot = DatasetBuilder(inventory).build()
    if output is not None:
        _atomic_new_bytes(output, snapshot.canonical_bytes)
    payload = _dataset_summary(snapshot, written=output is not None, output=None if output is None else _relative_or_external(output, root))
    payload.update({"command": "dataset", "ok": True, "authorized": output is None or bool(args.execute)})
    return payload, ExitCode.OK


def _status(args: argparse.Namespace) -> tuple[dict[str, Any], ExitCode]:
    memory_candidate = _bounded_path(args.memory, "memory")
    memory = _safe_existing_file(args.memory, "memory", max_bytes=64 * 1024 * 1024) if memory_candidate.exists() else memory_candidate
    profiles_candidate = _bounded_path(args.profiles, "profiles") if args.profiles else None
    profiles = _safe_existing_file(args.profiles, "profiles", max_bytes=16 * 1024 * 1024) if profiles_candidate is not None and profiles_candidate.exists() else profiles_candidate
    before = _metadata_signature([memory] + ([] if profiles is None else [profiles]))
    # The dashboard service is a query-only presentation boundary and handles
    # missing/corrupt history without exposing SQLite errors.
    from .dashboard import DashboardService
    result = DashboardService(memory, profile_path=profiles).status()
    after = _metadata_signature([memory] + ([] if profiles is None else [profiles]))
    unchanged = before == after
    payload = {"command": "status", "schema_version": 1, **result, "source_metadata_unchanged": unchanged}
    if not unchanged:
        payload["data_state"] = "unavailable"
        payload["reason"] = "source_metadata_changed"
    return payload, ExitCode.OK if unchanged and result.get("data_state") not in {"unavailable", "empty"} else ExitCode.UNAVAILABLE


def _shadow(args: argparse.Namespace) -> tuple[dict[str, Any], ExitCode]:
    if args.write_history and not args.execute:
        return {"command": "shadow", "schema_version": 1, "ok": False, "authorized": False, "reason": "explicit_execute_required", "error": "explicit_execute_required", "history_written": False}, ExitCode.NOT_AUTHORIZED
    request_path = _safe_existing_file(args.request, "request")
    before = _metadata_signature([request_path])
    raw = _json_file(request_path)
    after = _metadata_signature([request_path])
    if before != after:
        raise CLIError(ExitCode.DATA, "request_changed")
    write_history = bool(args.write_history and args.execute)
    request = _shadow_request(raw, write_history=write_history)
    memory = _bounded_path(args.memory, "memory") if args.memory else None
    if write_history:
        if memory is None:
            raise CLIError(ExitCode.USAGE, "history_requires_memory")
        _safe_parent(memory)
    binding = _memory_preflight(memory) if write_history and memory is not None else None
    config = OptimizerConfig(Path.cwd(), memory_path=memory) if memory is not None else OptimizerConfig(Path.cwd())
    orchestrator = OptimizerOrchestrator(config)
    if write_history and memory is not None:
        _assert_memory_binding(memory, binding)
    decision = orchestrator.shadow(request)
    payload = {
        "command": "shadow",
        "schema_version": 1,
        "fingerprint": decision.fingerprint,
        "candidate_id": decision.candidate_id,
        "status": decision.status,
        "baseline_ratios": dict(decision.baseline_ratios),
        "confidence_intervals": {key: list(value) for key, value in decision.confidence_intervals.items()},
        "qualified": decision.qualified,
        "reasons": list(decision.reasons),
        "evidence_hash": decision.evidence_hash,
        "no_activation": True,
        "history_requested": bool(args.write_history),
        "history_written": write_history,
        "source_metadata_unchanged": True,
    }
    return payload, ExitCode.OK if decision.qualified else ExitCode.UNAVAILABLE


def _dashboard(args: argparse.Namespace) -> tuple[dict[str, Any], ExitCode]:
    memory = _bounded_path(args.memory, "memory")
    profiles = _bounded_path(args.profiles, "profiles") if args.profiles else None
    dataset = _bounded_path(args.dataset, "dataset") if args.dataset else None
    portfolio = _bounded_path(args.portfolio, "portfolio") if args.portfolio else None
    if memory.exists() or memory.is_symlink():
        _safe_existing_file(str(memory), "memory", max_bytes=64 * 1024 * 1024)
    if profiles is not None and (profiles.exists() or profiles.is_symlink()):
        _safe_existing_file(str(profiles), "profiles", max_bytes=16 * 1024 * 1024)
    if dataset is not None and (dataset.exists() or dataset.is_symlink()):
        _safe_existing_file(str(dataset), "dataset", max_bytes=4 * 1024 * 1024)
    if portfolio is not None and (portfolio.exists() or portfolio.is_symlink()):
        _safe_existing_file(str(portfolio), "portfolio", max_bytes=4 * 1024 * 1024)
    try:
        server = serve(memory, int(args.port), profile_path=profiles, dataset_path=dataset, portfolio_path=portfolio)
    except (DashboardError, OSError):
        raise CLIError(ExitCode.UNAVAILABLE, "dashboard_unavailable") from None
    stopped = False
    def stop(_signum: int, _frame: Any) -> None:
        nonlocal stopped
        stopped = True
        # ``BaseServer.shutdown`` waits for ``serve_forever`` to observe its
        # flag and therefore must not be called synchronously by a signal
        # handler running on that same serving thread.
        threading.Thread(target=server.shutdown, name="friday-dashboard-stop", daemon=True).start()
    old_handlers = {name: signal.getsignal(name) for name in (signal.SIGINT, signal.SIGTERM)}
    try:
        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        payload = {"command": "dashboard", "schema_version": 1, "ok": True, "read_only": True, "host": "127.0.0.1", "port": server.server_port}
        _emit(payload)
        server.serve_forever(poll_interval=0.2)
    finally:
        for name, handler in old_handlers.items():
            signal.signal(name, handler)
        server.server_close()
    return {"command": "dashboard", "schema_version": 1, "ok": True, "read_only": True, "stopped": stopped}, ExitCode.OK


def _real_error(exc: RealSessionError) -> CLIError:
    reason = str(exc) or "real_session_failed"
    # Input/precondition failures are data errors; an explicit execute gate is
    # intentionally distinct so callers can safely ask for authorization.
    if reason == "explicit_execute_required":
        return CLIError(ExitCode.NOT_AUTHORIZED, reason)
    return CLIError(ExitCode.DATA, reason)


def _fingerprint(args: argparse.Namespace) -> tuple[dict[str, Any], ExitCode]:
    model = _safe_existing_file(args.model_identity, "model_identity")
    workload = _safe_existing_file(args.workload_contract, "workload_contract")
    if args.out and not args.execute:
        return {"command": "fingerprint", "schema_version": 1, "ok": False, "authorized": False, "reason": "explicit_execute_required", "written": False}, ExitCode.NOT_AUTHORIZED
    try:
        report = collect_fingerprint(
            model_identity=model,
            workload_contract=workload,
            runtime_commit=args.runtime_commit,
            collector=Collector(),
        )
    except RealSessionError as exc:
        raise _real_error(exc) from exc
    output: Path | None = None
    if args.out:
        output, _ = _target_new_file(args.out, "out")
        try:
            from .real_session import _atomic_new
            _atomic_new(output, canonical_bytes(report.public_dict(), max_bytes=4 * 1024 * 1024))
        except RealSessionError as exc:
            raise _real_error(exc) from exc
    payload = report.public_dict()
    payload.update({"command": "fingerprint", "schema_version": 1, "ok": report.recommendation_allowed, "authorized": output is not None, "written": output is not None})
    # Never serialize the caller's model/workload path; only a root-relative
    # marker is returned for the optional output target.
    payload["output"] = None if output is None else "<new>"
    return payload, ExitCode.OK if report.recommendation_allowed else ExitCode.UNAVAILABLE


def _session_plan(args: argparse.Namespace) -> tuple[dict[str, Any], ExitCode]:
    try:
        plan = build_session_plan(
            checkout=_existing_directory(args.checkout, "checkout"),
            expected_head=args.expected_head,
            interpreter=_safe_existing_file(args.interpreter, "interpreter"),
            model_identity=_safe_existing_file(args.model_identity, "model_identity"),
            workload_contract=_safe_existing_file(args.workload_contract, "workload_contract"),
            runtime_commit=args.runtime_commit,
            candidate_id=args.candidate,
            duration_minutes=args.duration,
            preregistration=_safe_existing_file(args.prereg, "preregistration"),
            collector=Collector(),
        )
    except RealSessionError as exc:
        raise _real_error(exc) from exc
    payload = plan.as_dict()
    payload.update({"command": "session-plan", "schema_version": 1, "ok": plan.ready, "authorized": False})
    return payload, ExitCode.OK if plan.ready else ExitCode.UNAVAILABLE


def _session(args: argparse.Namespace) -> tuple[dict[str, Any], ExitCode]:
    if not args.execute:
        return {"command": "session", "schema_version": 1, "ok": False, "authorized": False, "reason": "explicit_execute_required", "no_activation": True}, ExitCode.NOT_AUTHORIZED
    try:
        plan = build_session_plan(
            checkout=_existing_directory(args.checkout, "checkout"),
            expected_head=args.expected_head,
            interpreter=_safe_existing_file(args.interpreter, "interpreter"),
            model_identity=_safe_existing_file(args.model_identity, "model_identity"),
            workload_contract=_safe_existing_file(args.workload_contract, "workload_contract"),
            runtime_commit=args.runtime_commit,
            candidate_id=args.candidate,
            duration_minutes=args.duration,
            preregistration=_safe_existing_file(args.prereg, "preregistration"),
            collector=Collector(),
        )
        execution = run_session(
            plan=plan,
            checkout=_existing_directory(args.checkout, "checkout"),
            expected_head=args.expected_head,
            interpreter=_safe_existing_file(args.interpreter, "interpreter"),
            model_identity=_safe_existing_file(args.model_identity, "model_identity"),
            workload_contract=_safe_existing_file(args.workload_contract, "workload_contract"),
            runtime_commit=args.runtime_commit,
            session_id=args.session_id,
            memory=_safe_existing_file(args.memory, "memory", max_bytes=64 * 1024 * 1024),
            result_out=args.result_out,
            execute=True,
        )
    except RealSessionError as exc:
        raise _real_error(exc) from exc
    result = dict(execution.result)
    result.update({"command": "session", "schema_version": 1, "ok": bool(result.get("run_ok")) and execution.persistence_ok, "authorized": True, "history_written": execution.history_written, "history_error": execution.history_error, "persistence_ok": execution.persistence_ok})
    return result, ExitCode.OK if result.get("ok") else ExitCode.UNAVAILABLE


def _emit(payload: Mapping[str, Any]) -> None:
    output_error = False
    try:
        encoded = canonical_bytes(payload, max_bytes=MAX_OUTPUT_BYTES, max_items=100_000, max_depth=32)
    except Exception:
        encoded = b'{"ok":false,"error":"output_unavailable"}'
        output_error = True
    stream = getattr(sys.stdout, "buffer", None)
    try:
        if stream is not None:
            stream.write(encoded + b"\n")
            stream.flush()
        else:  # Useful for embedders/tests that provide a text-only stream.
            sys.stdout.write(encoded.decode("utf-8") + "\n")
            sys.stdout.flush()
    except (BrokenPipeError, OSError):
        raise CLIError(ExitCode.UNAVAILABLE, "output_unavailable") from None
    if output_error:
        raise CLIError(ExitCode.INTERNAL, "output_unavailable")


def _portfolio(args: argparse.Namespace) -> tuple[dict[str, Any], ExitCode]:
    try:
        snapshot = build_portfolio(_safe_existing_file(args.manifest, "manifest"))
    except PortfolioError as exc:
        raise CLIError(ExitCode.DATA, str(exc)) from exc
    payload = snapshot.as_dict()
    payload.update({"command": "portfolio", "schema_version": 1, "ok": True})
    return payload, ExitCode.OK


def _policy(args: argparse.Namespace) -> SelectionPolicy:
    try:
        return SelectionPolicy(args.policy_id, rule=args.rule, epsilon=args.epsilon)
    except DecisionError as exc:
        raise CLIError(ExitCode.USAGE, "invalid_policy") from exc


def _fingerprint_document(path: Path) -> ExactFingerprint:
    """Accept a plain fingerprint document or a fingerprint report."""

    raw = _json_file(path)
    body = _object(raw, "fingerprint")
    for key in ("report", "fingerprint"):
        while isinstance(body, Mapping) and key in body and isinstance(body[key], Mapping):
            body = body[key]
    try:
        return ExactFingerprint.from_mapping(body)
    except Exception as exc:
        raise CLIError(ExitCode.DATA, "fingerprint_invalid") from exc


def _write_memory_path(args: argparse.Namespace) -> Path:
    memory = _bounded_path(args.memory, "memory")
    _safe_parent(memory)
    return memory


def _decide(args: argparse.Namespace) -> tuple[dict[str, Any], ExitCode]:
    if not args.execute:
        return {"command": "decide", "schema_version": 1, "ok": False, "authorized": False,
                "reason": "explicit_execute_required", "written": False}, ExitCode.NOT_AUTHORIZED
    policy = _policy(args)
    fingerprint = _fingerprint_document(_safe_existing_file(args.fingerprint, "fingerprint"))
    memory = _write_memory_path(args)
    binding = _memory_preflight(memory)
    orchestrator = OptimizerOrchestrator(OptimizerConfig(Path.cwd(), memory_path=memory))
    _assert_memory_binding(memory, binding)
    try:
        event = orchestrator.select(fingerprint, policy=policy, hints=tuple(args.hint or ()),
                                    qualified=tuple(args.qualified or ()), seed=args.seed, write=True)
    except (DecisionError, ValueError, TypeError) as exc:
        raise CLIError(ExitCode.DATA, "decision_rejected") from exc
    payload = event.payload()
    payload.update({"command": "decide", "schema_version": 1, "ok": True, "authorized": True,
                    "written": True, "no_activation": True})
    return payload, ExitCode.OK


def _outcome(args: argparse.Namespace) -> tuple[dict[str, Any], ExitCode]:
    if not args.execute:
        return {"command": "outcome", "schema_version": 1, "ok": False, "authorized": False,
                "reason": "explicit_execute_required", "written": False}, ExitCode.NOT_AUTHORIZED
    memory = _write_memory_path(args)
    binding = _memory_preflight(memory)
    orchestrator = OptimizerOrchestrator(OptimizerConfig(Path.cwd(), memory_path=memory))
    _assert_memory_binding(memory, binding)
    try:
        outcome = orchestrator.record_outcome(
            args.decision, args.censoring, reward=args.reward, reward_metric=args.metric,
            evidence_hash=args.evidence_hash, notes=args.notes or "", write=True,
        )
    except (DecisionError, ValueError, TypeError) as exc:
        raise CLIError(ExitCode.DATA, "outcome_rejected") from exc
    payload = outcome.payload()
    payload.update({"command": "outcome", "schema_version": 1, "ok": True, "authorized": True,
                    "written": True, "no_activation": True})
    return payload, ExitCode.OK


def _replay(args: argparse.Namespace) -> tuple[dict[str, Any], ExitCode]:
    policy = _policy(args)
    memory = _safe_existing_file(args.memory, "memory", max_bytes=64 * 1024 * 1024)
    before = _metadata_signature([memory])
    orchestrator = OptimizerOrchestrator(OptimizerConfig(Path.cwd(), memory_path=memory))
    try:
        estimates = orchestrator.evaluate_policy(policy, min_samples=args.min_samples, seed=args.seed)
        steps = len(orchestrator.replay())
    except (ReplayError, DecisionError, ValueError, TypeError) as exc:
        raise CLIError(ExitCode.DATA, "replay_unavailable") from exc
    except OSError as exc:
        raise CLIError(ExitCode.UNAVAILABLE, "memory_unreadable") from exc
    unchanged = before == _metadata_signature([memory])
    conclusive = all(estimate.conclusive for estimate in estimates.values()) and bool(estimates)
    payload = {
        "command": "replay", "schema_version": 1, "policy": policy.as_dict(),
        "labelled_steps": steps, "min_samples": args.min_samples,
        "estimates": {name: estimate.as_dict() for name, estimate in estimates.items()},
        "conclusive": conclusive, "learning_claim": False, "no_activation": True,
        "source_metadata_unchanged": unchanged,
        "ok": unchanged and conclusive,
    }
    if not unchanged:
        payload["reason"] = "source_metadata_changed"
    return payload, ExitCode.OK if payload["ok"] else ExitCode.UNAVAILABLE


def _build_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="python -m friday_optimizer", allow_abbrev=False, description="Closed offline Friday optimizer control plane")
    parser.add_argument("--json", action="store_true", help="emit canonical JSON (the default)")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", allow_abbrev=False)
    doctor.add_argument("--root")
    doctor.add_argument("--memory")
    doctor.add_argument("--profiles")
    doctor.add_argument("--probe-system", action="store_true")
    doctor.add_argument("--json", action="store_true", help=argparse.SUPPRESS)

    audit = sub.add_parser("audit", allow_abbrev=False)
    audit.add_argument("--root", required=True)
    audit.add_argument("--json", action="store_true", help=argparse.SUPPRESS)

    imp = sub.add_parser("import", allow_abbrev=False)
    imp.add_argument("--root", required=True)
    imp.add_argument("--memory", required=True)
    imp.add_argument("--execute", action="store_true")
    imp.add_argument("--json", action="store_true", help=argparse.SUPPRESS)

    dataset = sub.add_parser("dataset", allow_abbrev=False)
    dataset.add_argument("--root", required=True)
    dataset.add_argument("--out")
    dataset.add_argument("--execute", action="store_true")
    dataset.add_argument("--json", action="store_true", help=argparse.SUPPRESS)

    status = sub.add_parser("status", allow_abbrev=False)
    status.add_argument("--memory", required=True)
    status.add_argument("--profiles")
    status.add_argument("--json", action="store_true", help=argparse.SUPPRESS)

    shadow = sub.add_parser("shadow", allow_abbrev=False)
    shadow.add_argument("--request", required=True)
    shadow.add_argument("--memory")
    shadow.add_argument("--write-history", action="store_true")
    shadow.add_argument("--execute", action="store_true")
    shadow.add_argument("--json", action="store_true", help=argparse.SUPPRESS)

    dashboard = sub.add_parser("dashboard", allow_abbrev=False)
    dashboard.add_argument("--memory", required=True)
    dashboard.add_argument("--profiles")
    dashboard.add_argument("--dataset")
    dashboard.add_argument("--portfolio")
    dashboard.add_argument("--port", type=int, default=0)
    dashboard.add_argument("--json", action="store_true", help=argparse.SUPPRESS)

    fingerprint = sub.add_parser("fingerprint", allow_abbrev=False)
    fingerprint.add_argument("--model-identity", required=True)
    fingerprint.add_argument("--workload-contract", required=True)
    fingerprint.add_argument("--runtime-commit", required=True)
    fingerprint.add_argument("--out")
    fingerprint.add_argument("--execute", action="store_true")
    fingerprint.add_argument("--json", action="store_true", help=argparse.SUPPRESS)

    def _policy_arguments(target: argparse.ArgumentParser) -> None:
        target.add_argument("--policy-id", default="deterministic_order_v1")
        target.add_argument("--rule", default="deterministic_order", choices=list(SELECTION_RULES))
        target.add_argument("--epsilon", type=float, default=0.0)

    decide_parser = sub.add_parser("decide", allow_abbrev=False)
    decide_parser.add_argument("--memory", required=True)
    decide_parser.add_argument("--fingerprint", required=True)
    _policy_arguments(decide_parser)
    decide_parser.add_argument("--hint", action="append", default=[])
    decide_parser.add_argument("--qualified", action="append", default=[])
    decide_parser.add_argument("--seed", type=int)
    decide_parser.add_argument("--execute", action="store_true")
    decide_parser.add_argument("--json", action="store_true", help=argparse.SUPPRESS)

    outcome_parser = sub.add_parser("outcome", allow_abbrev=False)
    outcome_parser.add_argument("--memory", required=True)
    outcome_parser.add_argument("--decision", required=True)
    outcome_parser.add_argument("--censoring", required=True, choices=list(CENSORING))
    outcome_parser.add_argument("--reward", type=float)
    outcome_parser.add_argument("--metric", default="ratio_median", choices=list(REWARD_METRICS))
    outcome_parser.add_argument("--evidence-hash", dest="evidence_hash")
    outcome_parser.add_argument("--notes", default="")
    outcome_parser.add_argument("--execute", action="store_true")
    outcome_parser.add_argument("--json", action="store_true", help=argparse.SUPPRESS)

    replay_parser = sub.add_parser("replay", allow_abbrev=False)
    replay_parser.add_argument("--memory", required=True)
    _policy_arguments(replay_parser)
    replay_parser.add_argument("--min-samples", dest="min_samples", type=int, default=DEFAULT_MIN_SAMPLES)
    replay_parser.add_argument("--seed", type=int, default=0)
    replay_parser.add_argument("--json", action="store_true", help=argparse.SUPPRESS)

    portfolio = sub.add_parser("portfolio", allow_abbrev=False)
    portfolio.add_argument("--manifest", required=True)
    portfolio.add_argument("--json", action="store_true", help=argparse.SUPPRESS)

    plan = sub.add_parser("session-plan", allow_abbrev=False)
    plan.add_argument("--checkout", required=True)
    plan.add_argument("--expected-head", required=True)
    plan.add_argument("--interpreter", required=True)
    plan.add_argument("--model-identity", required=True)
    plan.add_argument("--workload-contract", required=True)
    plan.add_argument("--runtime-commit", required=True)
    plan.add_argument("--candidate", required=True)
    plan.add_argument("--duration", "--duration-minutes", dest="duration", type=int, required=True)
    plan.add_argument("--prereg", "--preregistration", dest="prereg", required=True)
    plan.add_argument("--json", action="store_true", help=argparse.SUPPRESS)

    session = sub.add_parser("session", allow_abbrev=False)
    session.add_argument("--checkout", required=True)
    session.add_argument("--expected-head", required=True)
    session.add_argument("--interpreter", required=True)
    session.add_argument("--model-identity", required=True)
    session.add_argument("--workload-contract", required=True)
    session.add_argument("--runtime-commit", required=True)
    session.add_argument("--candidate", required=True)
    session.add_argument("--duration", "--duration-minutes", dest="duration", type=int, required=True)
    session.add_argument("--prereg", "--preregistration", dest="prereg", required=True)
    session.add_argument("--memory", required=True)
    session.add_argument("--result-out", required=True)
    session.add_argument("--session-id", required=True)
    session.add_argument("--execute", action="store_true")
    session.add_argument("--json", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _build_parser().parse_args(list(argv) if argv is not None else None)
        if not 0 <= args.port <= 65535 if args.command == "dashboard" else False:
            raise CLIError(ExitCode.USAGE, "invalid_port")
        handler = {"doctor": _doctor, "audit": _audit, "import": _import, "dataset": _dataset, "status": _status, "shadow": _shadow, "dashboard": _dashboard, "fingerprint": _fingerprint, "portfolio": _portfolio, "session-plan": _session_plan, "session": _session, "decide": _decide, "outcome": _outcome, "replay": _replay}[args.command]
        payload, code = handler(args)
        if args.command != "dashboard":
            _emit(payload)
        return int(code)
    except HelpRequested:
        return int(ExitCode.OK)
    except CLIError as exc:
        # _emit has already written a bounded sentinel when the output itself
        # was unavailable; emitting again would duplicate a partial report.
        if exc.reason != "output_unavailable":
            try:
                _emit({"ok": False, "error": exc.reason})
            except CLIError:
                pass
        try:
            sys.stderr.write("friday_optimizer: " + ("usage error\n" if exc.code == ExitCode.USAGE else "operation failed\n"))
        except OSError:
            pass
        return int(exc.code)
    except (BrokenPipeError, KeyboardInterrupt):
        return int(ExitCode.UNAVAILABLE)
    except Exception:
        _emit({"ok": False, "error": "internal_error"})
        try:
            sys.stderr.write("friday_optimizer: internal error\n")
        except OSError:
            pass
        return int(ExitCode.INTERNAL)


__all__ = ["ExitCode", "main"]
