"""Fixed offline parent orchestration for H0 analysis and control fixtures."""

from __future__ import annotations

import hashlib
import contextlib
import json
import math
import os
import random
import stat
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .benchmark import (
    H0_BATCH_MAX_NS,
    H0_BATCH_MIN_NS,
    H0_INITIAL_WARMUPS,
    H0_MAX_REPETITIONS,
    H0_MAX_WARMUPS,
)
from .canonical import canonical_json_bytes, canonical_sha256
from .correctness_contract import (
    CORRECTNESS_CASES,
    CORRECTNESS_CASE_NAMES,
    CORRECTNESS_FULL_KEYS,
    CORRECTNESS_HARD_CAPS,
    SIGN_INVARIANT_KEYS,
    CorrectnessContractError,
    MEMORY_LIMIT_KEYS,
    MEMORY_MAX_INT,
    fixture_digest,
    missing_reason_allowed,
    validate_memory_limit_contract,
    validate_sign_invariant_case,
    validate_fixed_case,
    validate_performance_case,
)
from .constants import (
    ALLOWED_MODES,
    ANALYSIS_MODES,
    AA_BOOTSTRAP_SEEDS,
    AA_SESSION_SEEDS,
    CONTROL_MODES,
    EAGER_COMPILE_SESSION_SEEDS,
    MLX_MODES,
    PHASE_H0,
    WRONG_FIXTURE_SEED,
    WRONG_FIXTURE_SIZE,
)
from .manifest import validate_manifest
from .protocol import ClosedManifest, ProtocolError, close_manifest, fallback_result, validate_result
from .provenance import Provenance, collect_provenance
from .storage import PersistenceOutcome, Storage, StorageError
from .supervisor import SupervisorLimits, run_supervised


OFFLINE_MODES = frozenset(ANALYSIS_MODES | CONTROL_MODES)
_DATA_DIR_NAME = ".friday-data"
_DATABASE_NAME = "h0.sqlite3"
_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
_TEST_ROOT_TOKEN = object()
_SCALAR_FIELDS = (
    ("rss_peak_bytes", "bytes"),
    ("stdout_bytes", "bytes"),
    ("stderr_bytes", "bytes"),
)
_MAX_SCALAR_BYTES = (1 << 63) - 1
_PROJECTION_VERSION = 1
_PROJECTION_HASH_DOMAIN = "friday_h0.sqlite_projection.v1"


class RunnerError(RuntimeError):
    """Raised for fixed runner policy or parent infrastructure failures."""


@dataclass(frozen=True)
class _TestRoot:
    """Private, token-bound alternate root; never accepted by the CLI."""

    root: Path
    token: object


def _test_root(root: str | Path) -> _TestRoot:
    candidate = Path(root).resolve()
    if not candidate.is_dir():
        raise RunnerError("test project root must be an existing directory")
    return _TestRoot(candidate, _TEST_ROOT_TOKEN)


@dataclass(frozen=True)
class RunOutcome:
    """Validated result plus atomic storage outcome."""

    manifest: ClosedManifest
    result: dict[str, Any]
    persistence: PersistenceOutcome


def database_path(project_root: str | Path | None = None) -> Path:
    """Return only the real repository's fixed database path.

    The optional argument remains solely to fail clearly for legacy callers;
    alternate roots require the private token-bound test factory below.
    """

    if project_root is not None:
        raise RunnerError("alternate database roots require the private test factory")
    return _REPOSITORY_ROOT / _DATA_DIR_NAME / _DATABASE_NAME


def _database_path_for_tests(context: _TestRoot) -> Path:
    if not isinstance(context, _TestRoot) or context.token is not _TEST_ROOT_TOKEN:
        raise RunnerError("invalid private database test context")
    return context.root / _DATA_DIR_NAME / _DATABASE_NAME


def _current_uid() -> int:
    getter = getattr(os, "geteuid", os.getuid)
    return int(getter())


def _open_flags(*, directory: bool = False) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise RunnerError("secure no-follow file opening is unavailable")
    flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0)
    if directory:
        directory_flag = getattr(os, "O_DIRECTORY", None)
        if directory_flag is None:
            raise RunnerError("secure directory opening is unavailable")
        flags |= directory_flag
    return flags


def _verify_fd(fd: int, *, mode: int, kind: str, label: str) -> os.stat_result:
    info = os.fstat(fd)
    if kind == "file" and not stat.S_ISREG(info.st_mode):
        raise RunnerError(f"{label} must be a regular file")
    if kind == "directory" and not stat.S_ISDIR(info.st_mode):
        raise RunnerError(f"{label} must be a directory")
    if info.st_uid != _current_uid():
        raise RunnerError(f"{label} owner is not the current user")
    if stat.S_IMODE(info.st_mode) != mode:
        raise RunnerError(f"{label} permissions are not {mode:o}")
    return info


def _open_private_data_dir(root: Path) -> tuple[Path, int, int, os.stat_result, os.stat_result]:
    """Open the fixed data directory without following a symlink.

    The directory descriptor protects the object being checked.  sqlite3 only
    accepts a pathname, not this descriptor, so the path is compared again
    immediately before/after connection; this is fail-closed detection, not a
    claim of perfect fd-bound sqlite3 TOCTOU freedom.
    """

    root = root.resolve()
    try:
        root_fd = os.open(root, _open_flags(directory=True))
        root_info = os.fstat(root_fd)
        if not stat.S_ISDIR(root_info.st_mode) or root_info.st_uid != _current_uid():
            raise RunnerError("project root is not a secure directory")
    except RunnerError:
        if "root_fd" in locals():
            os.close(root_fd)
        raise
    except OSError as exc:
        if "root_fd" in locals():
            os.close(root_fd)
        raise RunnerError("project root is not a secure directory") from exc
    path = root / _DATA_DIR_NAME
    try:
        try:
            os.mkdir(_DATA_DIR_NAME, 0o700, dir_fd=root_fd)
        except FileExistsError:
            pass
        fd = os.open(_DATA_DIR_NAME, _open_flags(directory=True), dir_fd=root_fd)
        try:
            os.fchmod(fd, 0o700)
            info = _verify_fd(fd, mode=0o700, kind="directory", label="database directory")
            return path, root_fd, fd, root_info, info
        except Exception:
            os.close(fd)
            raise
    except RunnerError:
        os.close(root_fd)
        raise
    except (OSError, NotImplementedError) as exc:
        os.close(root_fd)
        raise RunnerError("cannot create or secure the database directory") from exc


def _open_private_database(path: Path, *, dir_fd: int | None = None) -> tuple[int, os.stat_result]:
    flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    if not hasattr(os, "O_NOFOLLOW"):
        raise RunnerError("secure no-follow database opening is unavailable")
    fd = -1
    target: str | Path = path.name if dir_fd is not None else path
    try:
        try:
            fd = os.open(target, flags, dir_fd=dir_fd)
        except FileNotFoundError:
            fd = os.open(target, flags | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=dir_fd)
        os.fchmod(fd, 0o600)
        info = _verify_fd(fd, mode=0o600, kind="file", label="database file")
        return fd, info
    except RunnerError:
        if fd >= 0:
            os.close(fd)
        raise
    except (OSError, NotImplementedError) as exc:
        if fd >= 0:
            os.close(fd)
        raise RunnerError("cannot securely open the database file") from exc


def _same_path_identity(path: Path, expected: os.stat_result, *, label: str) -> None:
    try:
        observed = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise RunnerError(f"{label} disappeared or was replaced") from exc
    if not stat.S_ISREG(observed.st_mode) and label == "database file":
        raise RunnerError(f"{label} is not a regular file")
    if not stat.S_ISDIR(observed.st_mode) and label in {"project root", "database directory"}:
        raise RunnerError(f"{label} is not a directory")
    if (observed.st_dev, observed.st_ino) != (expected.st_dev, expected.st_ino):
        raise RunnerError(f"{label} was replaced while opening SQLite")


@contextlib.contextmanager
def _storage_for_root(root: Path, *, read_only: bool = False) -> Iterator[Storage]:
    data_path, root_fd, data_fd, root_info, data_info = _open_private_data_dir(root)
    db_path = data_path / _DATABASE_NAME
    try:
        db_fd, db_info = _open_private_database(db_path, dir_fd=data_fd)
    except Exception:
        os.close(data_fd)
        os.close(root_fd)
        raise
    try:
        _same_path_identity(root, root_info, label="project root")
        _same_path_identity(data_path, data_info, label="database directory")
        _same_path_identity(db_path, db_info, label="database file")
        with Storage.open(db_path, read_only=read_only) as storage:
            hook = _TEST_AFTER_CONNECT_HOOK
            if hook is not None:
                hook(db_path)
            _same_path_identity(root, root_info, label="project root")
            _same_path_identity(data_path, data_info, label="database directory")
            _same_path_identity(db_path, db_info, label="database file")
            yield storage
        _same_path_identity(data_path, data_info, label="database directory")
        _same_path_identity(db_path, db_info, label="database file")
    finally:
        os.close(db_fd)
        os.close(data_fd)
        os.close(root_fd)


_TEST_AFTER_CONNECT_HOOK: Any = None


def _initialize_database_for_tests(context: _TestRoot) -> Path:
    if not isinstance(context, _TestRoot) or context.token is not _TEST_ROOT_TOKEN:
        raise RunnerError("invalid private database test context")
    with _storage_for_root(context.root):
        pass
    return _database_path_for_tests(context)


def initialize_database(project_root: str | Path | None = None) -> Path:
    """Create or verify the fixed repository-local SQLite-v1 database."""

    if project_root is not None:
        raise RunnerError("alternate database roots require the private test factory")
    with _storage_for_root(_REPOSITORY_ROOT):
        pass
    return database_path()


def _process_for_mode(mode: str) -> tuple[str, int]:
    if mode in ANALYSIS_MODES:
        return "analysis", 0
    if mode in CONTROL_MODES:
        return "control", 0
    raise RunnerError("mode is not an offline mode")


def _seeds_for_mode(mode: str, process_set: str, process_index: int) -> dict[str, int]:
    if mode == "analysis_wrong_fixture":
        return {"fixture": WRONG_FIXTURE_SEED, "order": 0}
    if mode in ANALYSIS_MODES or mode in CONTROL_MODES:
        return {"fixture": 0, "order": 0}
    table = AA_SESSION_SEEDS if mode == "aa_gpu" else EAGER_COMPILE_SESSION_SEEDS
    seeds = {
        "fixture": table[f"{process_set}_fixture"] + process_index,
        "order": table[f"{process_set}_order"] + process_index,
    }
    if mode == "aa_gpu":
        seeds["bootstrap_seed"] = AA_BOOTSTRAP_SEEDS[process_set]
    return seeds


def run_id_for(mode: str, process_set: str, process_index: int, provenance: Provenance) -> str:
    if not isinstance(mode, str) or not isinstance(process_set, str):
        raise RunnerError("mode and process set must be strings")
    if mode not in ALLOWED_MODES:
        raise RunnerError("mode is not allowlisted")
    if isinstance(process_index, bool) or not isinstance(process_index, int):
        raise RunnerError("process index must be an integer")
    if mode in MLX_MODES:
        if process_set not in {"characterization", "confirmation"} or not 0 <= process_index <= 2:
            raise RunnerError("MLX process tuple is not registered")
    elif mode in ANALYSIS_MODES:
        if process_set != "analysis" or process_index != 0:
            raise RunnerError("analysis process tuple is fixed")
    elif mode in CONTROL_MODES:
        if process_set != "control" or process_index != 0:
            raise RunnerError("control process tuple is fixed")
    else:
        raise RunnerError("mode has no registered process tuple")
    material = canonical_json_bytes(provenance.as_manifest())
    prefix = hashlib.sha256(material).hexdigest()
    value = f"h0-{mode}-{process_set}-{process_index}-{prefix}"
    if len(value) > 128:
        raise RunnerError("deterministic run_id exceeds the manifest limit")
    return value


def build_manifest(mode: str, provenance: Provenance, *, process_set: str | None = None, process_index: int = 0) -> dict[str, Any]:
    """Construct and validate one registered manifest; no arbitrary fields are accepted."""

    if mode in MLX_MODES:
        expected_set = "characterization" if process_set is None else process_set
        expected_index = process_index
        if expected_set not in {"characterization", "confirmation"} or not 0 <= expected_index <= 2:
            raise RunnerError("MLX process tuple is not registered")
    elif mode in OFFLINE_MODES:
        expected_set, expected_index = _process_for_mode(mode)
    else:
        raise RunnerError("mode is not registered")
    if process_set is not None and process_set != expected_set:
        raise RunnerError("offline process set is fixed")
    if process_index != expected_index:
        raise RunnerError("offline process index is fixed")
    process_set, process_index = expected_set, expected_index
    size = WRONG_FIXTURE_SIZE if mode == "analysis_wrong_fixture" else 2048
    manifest = {
        "schema_version": 1,
        "phase": PHASE_H0,
        "run_id": run_id_for(mode, process_set, process_index, provenance),
        "mode": mode,
        "workload": {
            "operation": "matmul",
            "a_shape": [size, size],
            "b_shape": [size, size],
            "y_shape": [size, size],
            "dtype": "float16",
            "layout": "C-contiguous",
            "generator": "PCG64",
            "distribution": "uniform[-1,1)",
        },
        "seeds": _seeds_for_mode(mode, process_set, process_index),
        "limits": {"first_eval_s": 10, "synchronize_s": 5, "total_s": 120},
        "process": {"set": process_set, "index": process_index},
        "provenance": provenance.as_manifest(),
    }
    return validate_manifest(manifest)


def _metric_rows(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, unit in _SCALAR_FIELDS:
        value = evidence.get(name)
        reason = evidence.get(f"{name}_missing_reason")
        if value is None:
            if not isinstance(reason, str) or not reason or len(reason) > 256 or "\x00" in reason:
                reason = "invalid_source_value"
            rows.append({"metric_name": name, "value": None, "unit": unit, "scope": "run", "missing_reason": reason or "not_recorded"})
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            rows.append({"metric_name": name, "value": None, "unit": unit, "scope": "run", "missing_reason": "invalid_source_value"})
        else:
            try:
                numeric = float(value)
            except (OverflowError, TypeError, ValueError):
                numeric = math.nan
            if not math.isfinite(numeric) or numeric < 0 or numeric > _MAX_SCALAR_BYTES:
                rows.append({"metric_name": name, "value": None, "unit": unit, "scope": "run", "missing_reason": "invalid_source_value"})
            else:
                rows.append({"metric_name": name, "value": value, "unit": unit, "scope": "run"})
    return rows


def _artifact_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    decision = result.get("evidence", {}).get("decision")
    if not isinstance(decision, dict) or not isinstance(decision.get("decision_hash"), str):
        return []
    digest = decision["decision_hash"]
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        return []
    return [{
        "artifact_name": "analysis_decision",
        "artifact_kind": "analytical_fixture",
        "sha256": digest,
        "metadata": {"analytical_only": True, "classification": result["classification"]},
    }]


# The following adapter is intentionally kept in the parent runner instead of
# the worker.  It consumes only a result that has already crossed
# ``validate_result`` and never imports benchmark/MLX code.  The worker's
# terminal common-result event remains the lossless source of truth; these
# arrays are a bounded, explicitly allowlisted SQLite projection.
_MLX_MEASUREMENT_MODES = frozenset({"eager_baseline", "compile_comparison", "aa_gpu"})
_COMMON_MODE_BINDINGS = {
    "eager_baseline": ("baseline_reference", "not_run", False),
    "compile_comparison": ("measurement_complete", "aggregation_required", True),
    "aa_gpu": ("measurement_complete", "aggregation_required", True),
}
_COMMON_EVIDENCE_KEYS = frozenset({
    "rss_peak_bytes", "rss_missing_reason", "stdout_bytes", "stdout_sha256",
    "stdout_preview", "stdout_truncated", "stdout_overflow", "stderr_bytes",
    "stderr_sha256", "stderr_preview", "stderr_truncated", "stderr_overflow",
    "benchmark_classification", "benchmark_action", "aggregation_required",
    "adapter_contract", "benchmark_evidence", "domain_evidence_bytes",
    "domain_evidence_sha256", "total_ns",
})
_AGGREGATION_BRIDGE_DROP_KEYS = frozenset({
    "stdout_bytes", "stdout_sha256", "stdout_preview", "stdout_truncated", "stdout_overflow",
    "stderr_bytes", "stderr_sha256", "stderr_preview", "stderr_truncated", "stderr_overflow",
})
_DOMAIN_EVIDENCE_KEYS = frozenset({
    "fixture", "correctness", "memory", "memory_limit", "memory_gate", "cache_state",
    "fresh_process_required", "aggregation_required", "compile_wrapper_setup_ns",
    "first_eval_compile_inclusive_ns", "total_elapsed_ns", "arms", "comparison",
    "raw_samples",
})
_ARM_KEYS = frozenset({
    "warmup", "repetitions", "batches", "statistics", "calibration_samples",
    "raw_samples", "memory", "first_eval_compile_inclusive_ns",
})
_RAW_SOURCE_BASE_KEYS = frozenset({"phase", "sample_index", "value", "unit", "arm", "position"})
_WARMUP_DECLARED_KEYS = frozenset({"phase", "sample_index", "value", "unit"})
_WARMUP_BLOCK_KEYS = frozenset({
    "block_index", "evaluations", "block_ns", "per_eval_ns",
    "median_eval_ns", "min_eval_ns", "max_eval_ns",
})
_RAW_SOURCE_PROBE_KEYS = _RAW_SOURCE_BASE_KEYS | {"repetitions"}
_RAW_SOURCE_MEASUREMENT_KEYS = _RAW_SOURCE_PROBE_KEYS | {"sample_kind", "block_index"}
_RAW_STORAGE_KINDS = frozenset({
    "warmup_baseline", "warmup_candidate", "repetition_calibration_baseline",
    "repetition_calibration_candidate", "performance_baseline", "performance_candidate",
    "pair_performance_baseline", "pair_performance_candidate",
})
_CORRECTNESS_NAMES = (*CORRECTNESS_CASE_NAMES, "performance_fixture", "sign_invariant")
_CORRECTNESS_CASE_SPECS = {
    name: (tuple(shape_a) + tuple(shape_b), seed, zero_rhs)
    for name, shape_a, shape_b, seed, _low, _high, zero_rhs in CORRECTNESS_CASES
}
_CORRECTNESS_FULL_KEYS = CORRECTNESS_FULL_KEYS
_SIGN_INVARIANT_KEYS = SIGN_INVARIANT_KEYS
_CORRECTNESS_HARD_CAPS = CORRECTNESS_HARD_CAPS
_CORRECTNESS_METRIC_NAMES = (
    "abs_q50", "abs_q95", "abs_q99", "abs_max", "rel_q50", "rel_q95", "rel_q99",
    "rel_max", "rel_q99_abs_oracle_ge_1", "normalized_l2", "scaled_normalized_inf",
)
_MEMORY_NAMES = frozenset({
    "mlx_active_memory", "mlx_peak_memory", "mlx_cache_memory", "rss", "custom", "memory_metrics",
})
_MEMORY_LIMIT_KEYS = frozenset({"attempted", "hard_limit", "applied", "missing_reason"})
_H0_MEMORY_LIMIT_BYTES = 1 << 30
_MEMORY_PHASES = frozenset({"before_correctness", "after_calibration", "after_measurement", "after_timing"})
_MEMORY_ARMS = frozenset({"all", "baseline", "candidate"})
_SCALAR_UNITS = {
    "rss_peak_bytes": "bytes", "stdout_bytes": "bytes", "stderr_bytes": "bytes",
    "compile_wrapper_setup_ns": "ns", "first_eval_compile_inclusive_ns": "ns",
    "total_elapsed_ns": "ns", "baseline_warmup_count": "count",
    "candidate_warmup_count": "count", "baseline_warmup_median_ns": "ns",
    "candidate_warmup_median_ns": "ns", "baseline_repetitions": "count",
    "candidate_repetitions": "count", "baseline_calibration_batch_ns": "ns",
    "candidate_calibration_batch_ns": "ns", "baseline_batch_count": "count",
    "candidate_batch_count": "count", "baseline_median_ns": "ns",
    "candidate_median_ns": "ns", "baseline_mad_ns": "ns", "candidate_mad_ns": "ns",
    "baseline_iqr_ns": "ns", "candidate_iqr_ns": "ns", "baseline_min_ns": "ns",
    "candidate_min_ns": "ns", "baseline_max_ns": "ns", "candidate_max_ns": "ns",
    "ratio_count": "count", "ratio_median": "ratio", "ratio_mad": "ratio",
    "ratio_iqr": "ratio", "ratio_min": "ratio", "ratio_max": "ratio",
}
_SUPERVISOR_SCALARS = ("rss_peak_bytes", "stdout_bytes", "stderr_bytes")
_MISSING_REASON_RE = frozenset()
_WORKER_ADAPTER_CONTRACT = {
    "common_result_ready": False,
    "reason": "single-process measurements require aggregation before any global decision",
    "mapping": {
        "runtime_unavailable": "invalid/baseline_fallback",
        "invalid*": "invalid/baseline_fallback",
        "measurement_complete": "aggregation_required",
        "baseline_reference": "not_run",
    },
}


def _validate_projection_adapter_contract(evidence: Mapping[str, Any]) -> None:
    contract = evidence.get("adapter_contract")
    if not isinstance(contract, Mapping) or set(contract) != {"common_result_ready", "reason", "mapping"}:
        raise RunnerError("result.evidence.adapter_contract is missing or not closed")
    if contract["common_result_ready"] is not False or type(contract["common_result_ready"]) is not bool:
        raise RunnerError("result.evidence.adapter_contract.common_result_ready is invalid")
    if contract["reason"] != _WORKER_ADAPTER_CONTRACT["reason"] or not isinstance(contract["reason"], str) or len(contract["reason"]) > 256:
        raise RunnerError("result.evidence.adapter_contract.reason is invalid")
    if contract["mapping"] != _WORKER_ADAPTER_CONTRACT["mapping"]:
        raise RunnerError("result.evidence.adapter_contract.mapping is invalid")
def _missing_reason_allowed(reason: Any) -> bool:
    return missing_reason_allowed(reason)


def _median(values: Sequence[int | float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise RunnerError("empty measurement sequence")
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0


def _mad(values: Sequence[int | float]) -> float:
    center = _median(values)
    return _median([abs(float(value) - center) for value in values])


def _iqr(values: Sequence[int | float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise RunnerError("empty measurement sequence")
    def quantile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        low, high = math.floor(position), math.ceil(position)
        if low == high:
            return ordered[low]
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)
    return quantile(0.75) - quantile(0.25)


def _registered_balanced_order(seed: int, count: int = 30) -> list[str]:
    if count <= 0 or count % 2:
        raise RunnerError("paired block count is not positive and even")
    values = ["baseline"] * (count // 2) + ["candidate"] * (count // 2)
    random.Random(seed).shuffle(values)
    return values


def _projection_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RunnerError(f"{name} must be an object")
    return value


def _projection_text(value: Any, name: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum or "\x00" in value:
        raise RunnerError(f"{name} must be a bounded non-empty string")
    return value


def _projection_int(value: Any, name: str, *, nonnegative: bool = True) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < -(1 << 63) or value > _MAX_SCALAR_BYTES:
        raise RunnerError(f"{name} must be a signed 64-bit integer")
    if nonnegative and value < 0:
        raise RunnerError(f"{name} must be non-negative")
    return value


def _projection_number(value: Any, name: str, *, nonnegative: bool = True) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RunnerError(f"{name} must be numeric")
    if isinstance(value, int):
        return _projection_int(value, name, nonnegative=nonnegative)
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise RunnerError(f"{name} must be finite and in range")
    return value


def _projection_positive_number(value: Any, name: str) -> int | float:
    """Validate a measured duration; zero is not a valid elapsed time."""

    normalized = _projection_number(value, name, nonnegative=False)
    if normalized <= 0:
        raise RunnerError(f"{name} must be finite and strictly positive")
    return normalized


def _projection_missing(value: Any, reason: Any, name: str) -> tuple[int | float | None, str | None]:
    if value is None:
        if not _missing_reason_allowed(reason):
            raise RunnerError(f"{name} missing value has no registered reason")
        return None, reason
    if reason is not None:
        raise RunnerError(f"{name} has both value and missing reason")
    return _projection_number(value, name), None


def _projection_scalar(
    rows: list[dict[str, Any]], name: str, value: Any, unit: str, *, reason: str | None = None,
    nonnegative: bool = True,
) -> None:
    if name not in _SCALAR_UNITS and not name.startswith("memory_"):
        raise RunnerError(f"scalar metric {name} is not allowlisted")
    if _SCALAR_UNITS.get(name, unit) != unit:
        raise RunnerError(f"scalar metric {name} has an invalid unit")
    normalized, missing_reason = _projection_missing(value, reason, name)
    if normalized is not None:
        normalized = _projection_number(normalized, name, nonnegative=nonnegative)
    rows.append({
        "metric_name": name, "value": normalized, "missing_reason": missing_reason,
        "unit": unit, "scope": "run",
    })


def _validate_arm_for_projection(arm: Mapping[str, Any], name: str) -> None:
    if set(arm) - _ARM_KEYS or not {"warmup", "repetitions", "batches", "statistics", "calibration_samples", "raw_samples"}.issubset(arm):
        raise RunnerError(f"{name} has unknown or missing fields")
    warmup = _projection_mapping(arm["warmup"], f"{name}.warmup")
    if set(warmup) != {"count", "durations_ns", "stable", "median_ns", "samples", "blocks"}:
        raise RunnerError(f"{name}.warmup has unknown or missing fields")
    count = _projection_int(warmup.get("count"), f"{name}.warmup.count")
    if not H0_INITIAL_WARMUPS <= count <= H0_MAX_WARMUPS or warmup.get("stable") is not True:
        raise RunnerError(f"{name}.warmup is not a stable registered warmup")
    durations = warmup.get("durations_ns")
    if not isinstance(durations, Sequence) or isinstance(durations, (str, bytes, bytearray)) or len(durations) != count:
        raise RunnerError(f"{name}.warmup durations are invalid")
    for index, value in enumerate(durations):
        duration = _projection_int(value, f"{name}.warmup.durations_ns[{index}]")
        if duration <= 0:
            raise RunnerError(f"{name}.warmup.durations_ns[{index}] must be strictly positive")
    warmup_samples = warmup["samples"]
    if not isinstance(warmup_samples, Sequence) or isinstance(warmup_samples, (str, bytes, bytearray)) or len(warmup_samples) != count:
        raise RunnerError(f"{name}.warmup samples are not bounded")
    for index, sample in enumerate(warmup_samples):
        item = _projection_mapping(sample, f"{name}.warmup.samples[{index}]")
        if set(item) != _WARMUP_DECLARED_KEYS or item["phase"] != "warmup" or item["sample_index"] != index or item["unit"] != "ns":
            raise RunnerError(f"{name}.warmup.samples[{index}] is not canonical")
        value = _projection_positive_number(item["value"], f"{name}.warmup.samples[{index}].value")
        if value != durations[index]:
            raise RunnerError(f"{name}.warmup.samples[{index}] differs from durations_ns")
    blocks = warmup["blocks"]
    if not isinstance(blocks, Sequence) or isinstance(blocks, (str, bytes, bytearray)) or len(blocks) != count:
        raise RunnerError(f"{name}.warmup blocks are not bounded")
    for index, block in enumerate(blocks):
        item = _projection_mapping(block, f"{name}.warmup.blocks[{index}]")
        if set(item) != _WARMUP_BLOCK_KEYS:
            raise RunnerError(f"{name}.warmup.blocks[{index}] is not closed")
        block_index = _projection_int(item["block_index"], f"{name}.warmup.blocks[{index}].block_index")
        evaluations = _projection_int(item["evaluations"], f"{name}.warmup.blocks[{index}].evaluations")
        block_ns = _projection_int(item["block_ns"], f"{name}.warmup.blocks[{index}].block_ns")
        per_eval_ns = _projection_int(item["per_eval_ns"], f"{name}.warmup.blocks[{index}].per_eval_ns")
        median_eval_ns = _projection_int(item["median_eval_ns"], f"{name}.warmup.blocks[{index}].median_eval_ns")
        min_eval_ns = _projection_int(item["min_eval_ns"], f"{name}.warmup.blocks[{index}].min_eval_ns")
        max_eval_ns = _projection_int(item["max_eval_ns"], f"{name}.warmup.blocks[{index}].max_eval_ns")
        if (
            block_index != index
            or not 1 <= evaluations <= H0_MAX_REPETITIONS
            or block_ns < H0_BATCH_MIN_NS
            or per_eval_ns <= 0
            or not min_eval_ns <= median_eval_ns <= max_eval_ns
            or per_eval_ns != max(1, int(round(block_ns / evaluations)))
            or per_eval_ns != durations[index]
        ):
            raise RunnerError(f"{name}.warmup.blocks[{index}] is inconsistent")
    warmup_median = _projection_positive_number(warmup.get("median_ns"), f"{name}.warmup.median_ns")
    if warmup_median != _median(durations[-5:]):
        raise RunnerError(f"{name}.warmup median is not reconstructed from the last five gate values")
    repetition = _projection_mapping(arm["repetitions"], f"{name}.repetitions")
    if set(repetition) != {"repetitions", "batch_ns", "probe_timings", "calibration_samples"}:
        raise RunnerError(f"{name}.repetitions has unknown or missing fields")
    repetitions = _projection_int(repetition.get("repetitions"), f"{name}.repetitions.repetitions")
    if repetitions < 1 or repetitions > H0_MAX_REPETITIONS or repetitions & (repetitions - 1):
        raise RunnerError(f"{name}.repetitions is not a power of two in range")
    batch_ns = _projection_positive_number(repetition.get("batch_ns"), f"{name}.repetitions.batch_ns")
    if not H0_BATCH_MIN_NS <= batch_ns <= H0_BATCH_MAX_NS:
        raise RunnerError(f"{name}.repetitions window is invalid")
    calibration = repetition.get("calibration_samples")
    if not isinstance(calibration, Sequence) or isinstance(calibration, (str, bytes, bytearray)) or not calibration:
        raise RunnerError(f"{name}.repetitions calibration is missing")
    for index, sample in enumerate(calibration):
        item = _projection_mapping(sample, f"{name}.repetitions.calibration[{index}]")
        if set(item) != {"phase", "repetitions", "sample_index", "value", "unit"}:
            raise RunnerError(f"{name}.repetitions calibration fields are invalid")
        if item["phase"] != "repetition_probe" or item["repetitions"] != repetitions or item["unit"] != "ns":
            raise RunnerError(f"{name}.repetitions calibration binding is invalid")
        _projection_int(item["sample_index"], f"{name}.repetitions.calibration index")
        _projection_positive_number(item["value"], f"{name}.repetitions.calibration value")
    probe_timings = repetition["probe_timings"]
    if not isinstance(probe_timings, Sequence) or isinstance(probe_timings, (str, bytes, bytearray)) or len(probe_timings) != repetitions:
        raise RunnerError(f"{name}.repetitions probe_timings are invalid")
    for index, value in enumerate(probe_timings):
        _projection_positive_number(value, f"{name}.repetitions.probe_timings[{index}]")
        if value != calibration[index]["value"]:
            raise RunnerError(f"{name}.repetitions probe_timings differ from calibration samples")
    batches = arm["batches"]
    if not isinstance(batches, Sequence) or isinstance(batches, (str, bytes, bytearray)) or len(batches) != 30:
        raise RunnerError(f"{name}.batches must contain exactly 30 blocks")
    for index, batch in enumerate(batches):
        item = _projection_mapping(batch, f"{name}.batches[{index}]")
        allowed = {"block_index", "batch_ns", "per_eval_ns", "repetitions", "position", "evaluation_ns", "synchronize_ns"}
        if set(item) != allowed or item["block_index"] != index or item["repetitions"] != repetitions:
            raise RunnerError(f"{name}.batches[{index}] is not deterministic")
        if item["position"] not in {"single", "first", "second"}:
            raise RunnerError(f"{name}.batches[{index}] has an invalid position")
        _projection_positive_number(item["batch_ns"], f"{name}.batches[{index}].batch_ns")
        per_eval = _projection_positive_number(item["per_eval_ns"], f"{name}.batches[{index}].per_eval_ns")
        if per_eval * repetitions != item["batch_ns"]:
            raise RunnerError(f"{name}.batches[{index}] repetition binding is inconsistent")
        for field in ("evaluation_ns", "synchronize_ns"):
            values = item[field]
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)) or len(values) != repetitions:
                raise RunnerError(f"{name}.batches[{index}].{field} is invalid")
            for subindex, value in enumerate(values):
                _projection_positive_number(value, f"{name}.batches[{index}].{field}[{subindex}]")
    stats = _projection_mapping(arm["statistics"], f"{name}.statistics")
    if set(stats) != {"count", "median_ns", "mad_ns", "iqr_ns", "min_ns", "max_ns"} or stats["count"] != 30:
        raise RunnerError(f"{name}.statistics is invalid")
    for key, value in stats.items():
        if key != "count":
            if key in {"median_ns", "min_ns", "max_ns"}:
                _projection_positive_number(value, f"{name}.statistics.{key}")
            else:
                _projection_number(value, f"{name}.statistics.{key}")
    per_eval_values = [batch["per_eval_ns"] for batch in batches]
    expected_stats = {
        "median_ns": _median(per_eval_values), "mad_ns": _mad(per_eval_values),
        "iqr_ns": _iqr(per_eval_values), "min_ns": min(per_eval_values), "max_ns": max(per_eval_values),
    }
    if any(stats[key] != value for key, value in expected_stats.items()):
        raise RunnerError(f"{name}.statistics are not reconstructed from batches")


def _normalize_arm_samples(
    manifest: ClosedManifest, arm_name: str, arm: Mapping[str, Any], *, paired: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    _validate_arm_for_projection(arm, f"arms.{arm_name}")
    rows = arm["raw_samples"]
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)) or len(rows) > 10_000:
        raise RunnerError(f"arms.{arm_name}.raw_samples exceeds the bound")
    count = arm["warmup"]["count"]
    repetitions = arm["repetitions"]["repetitions"]
    warmups: list[Mapping[str, Any]] = []
    probes: list[Mapping[str, Any]] = []
    measurements: list[Mapping[str, Any]] = []
    seen_measurement_indices: set[int] = set()
    for source_index, sample in enumerate(rows):
        item = _projection_mapping(sample, f"arms.{arm_name}.raw_samples[{source_index}]")
        phase = item.get("phase")
        if phase == "warmup":
            if set(item) != _RAW_SOURCE_BASE_KEYS or item["arm"] != arm_name or item["position"] != "calibration" or item["unit"] != "ns":
                raise RunnerError(f"arms.{arm_name} warmup sample is not closed")
            sample_index = _projection_int(item["sample_index"], "warmup sample index")
            _projection_positive_number(item["value"], "warmup sample value")
            warmups.append(item)
            if sample_index != len(warmups) - 1:
                raise RunnerError(f"arms.{arm_name} warmup sample index is not deterministic")
        elif phase == "repetition_probe":
            if set(item) != _RAW_SOURCE_PROBE_KEYS or item["arm"] != arm_name or item["position"] != "calibration" or item["unit"] != "ns" or item["repetitions"] != repetitions:
                raise RunnerError(f"arms.{arm_name} repetition sample is not closed")
            sample_index = _projection_int(item["sample_index"], "repetition sample index")
            _projection_positive_number(item["value"], "repetition sample value")
            probes.append(item)
            if sample_index != len(probes) - 1:
                raise RunnerError(f"arms.{arm_name} repetition sample index is not deterministic")
        elif phase == "measurement":
            if set(item) != _RAW_SOURCE_MEASUREMENT_KEYS or item["arm"] != arm_name or item["unit"] != "ns" or item["repetitions"] != repetitions:
                raise RunnerError(f"arms.{arm_name} measurement sample is not closed")
            if item["sample_kind"] != "timing_batch":
                raise RunnerError(f"arms.{arm_name} measurement kind is not registered")
            block_index = _projection_int(item["block_index"], "measurement block index")
            sample_index = _projection_int(item["sample_index"], "measurement sample index")
            if block_index != sample_index or not 0 <= block_index < 30:
                raise RunnerError(f"arms.{arm_name} measurement index is not deterministic")
            if block_index in seen_measurement_indices:
                raise RunnerError(f"arms.{arm_name} measurement index is duplicated")
            if item["position"] not in ({"first", "second"} if paired else {"single"}):
                raise RunnerError(f"arms.{arm_name} measurement position is invalid")
            value = _projection_positive_number(item["value"], "measurement sample value")
            batch = arm["batches"][block_index]
            if item["position"] != batch["position"] or value != batch["batch_ns"]:
                raise RunnerError(f"arms.{arm_name} measurement does not match its batch position/value")
            measurements.append(item)
            seen_measurement_indices.add(block_index)
        else:
            raise RunnerError(f"arms.{arm_name} has an unknown raw sample phase")
    if len(warmups) != count or len(probes) != len(arm["repetitions"]["calibration_samples"]) or len(measurements) != 30:
        raise RunnerError(f"arms.{arm_name} raw sample counts are incomplete")
    declared_warmups = [
        {key: sample[key] for key in ("phase", "sample_index", "value", "unit")}
        for sample in warmups
    ]
    if declared_warmups != list(arm["warmup"]["samples"]):
        raise RunnerError(f"arms.{arm_name} warmup samples differ from raw samples")
    if [sample["value"] for sample in warmups] != list(arm["warmup"]["durations_ns"]):
        raise RunnerError(f"arms.{arm_name} warmup durations differ from raw samples")
    calibration = list(warmups) + list(probes)
    if canonical_json_bytes(arm["calibration_samples"]) != canonical_json_bytes(calibration):
        raise RunnerError(f"arms.{arm_name} calibration projection differs from raw samples")
    kind_prefix = "pair_performance" if paired else "performance"
    normalized: list[dict[str, Any]] = []
    for sample in warmups:
        normalized.append({"session_id": manifest.run_id, "sample_kind": f"warmup_{arm_name}", "sample_index": sample["sample_index"], "block_index": 0, "arm": arm_name, "value": sample["value"], "unit": "ns", "observed_at_ns": None})
    for sample in probes:
        normalized.append({"session_id": manifest.run_id, "sample_kind": f"repetition_calibration_{arm_name}", "sample_index": sample["sample_index"], "block_index": 0, "arm": arm_name, "value": sample["value"], "unit": "ns", "observed_at_ns": None})
    for sample in measurements:
        normalized.append({"session_id": manifest.run_id, "sample_kind": f"{kind_prefix}_{arm_name}", "sample_index": sample["sample_index"], "block_index": sample["block_index"], "arm": arm_name, "value": sample["value"], "unit": "ns", "observed_at_ns": None})
    if any(row["sample_kind"] not in _RAW_STORAGE_KINDS for row in normalized):
        raise RunnerError("normalized sample kind is not allowlisted")
    return normalized, [f"arms.{arm_name}.raw_samples" for _ in ()]


def _validate_timing_contract(
    domain: Mapping[str, Any], arms: Mapping[str, Any], mode: str,
) -> None:
    """Reconstruct the producer's mode-specific compile/evaluation contract."""

    setup = domain.get("compile_wrapper_setup_ns")
    first_eval = domain.get("first_eval_compile_inclusive_ns")
    if mode in {"eager_baseline", "aa_gpu"}:
        if setup is not None or first_eval is not None:
            raise RunnerError(f"{mode} must not report compile timing")
        for arm_name, arm in arms.items():
            if "first_eval_compile_inclusive_ns" in arm:
                raise RunnerError(f"{mode} arm {arm_name} must not report first evaluation compile timing")
        return
    if mode != "compile_comparison":
        raise RunnerError("unknown MLX timing mode")
    _projection_positive_number(setup, "benchmark_evidence.compile_wrapper_setup_ns")
    _projection_positive_number(first_eval, "benchmark_evidence.first_eval_compile_inclusive_ns")
    baseline = _projection_mapping(arms.get("baseline"), "benchmark_evidence.arms.baseline")
    candidate = _projection_mapping(arms.get("candidate"), "benchmark_evidence.arms.candidate")
    if "first_eval_compile_inclusive_ns" in baseline:
        raise RunnerError("compile baseline must not report first evaluation compile timing")
    if set(candidate) - _ARM_KEYS or "first_eval_compile_inclusive_ns" not in candidate:
        raise RunnerError("compile candidate must report first evaluation compile timing")
    candidate_first = _projection_positive_number(
        candidate["first_eval_compile_inclusive_ns"],
        "benchmark_evidence.arms.candidate.first_eval_compile_inclusive_ns",
    )
    if candidate_first != first_eval:
        raise RunnerError("compile candidate first evaluation timing differs from domain")


def _validate_memory_limit_contract(value: Any) -> None:
    """Validate the exact best-effort limit envelope emitted by the producer."""
    try:
        validate_memory_limit_contract(value)
    except CorrectnessContractError as exc:
        raise RunnerError(f"benchmark_evidence.memory_limit:{exc}") from exc


def _validate_memory_contract(domain: Mapping[str, Any]) -> None:
    _validate_memory_limit_contract(domain.get("memory_limit"))
    gate = domain.get("memory_gate")
    if not isinstance(gate, str) or gate not in {"aggregation_required", "not_evaluable_missing_required_metric"}:
        raise RunnerError("benchmark memory gate is not registered")
    memory = domain.get("memory")
    if not isinstance(memory, Sequence) or isinstance(memory, (str, bytes, bytearray)):
        raise RunnerError("benchmark_evidence.memory is not a sequence")
    latest: dict[str, tuple[Any, str | None]] = {}
    for index, entry in enumerate(memory):
        item = _projection_mapping(entry, f"benchmark_evidence.memory[{index}]")
        allowed = {"name", "api", "unit", "value", "missing_reason", "measurement_phase", "arm", "measured_at_ns", "reset_state"}
        if set(item) != allowed:
            raise RunnerError("memory evidence is not closed")
        phase = item["measurement_phase"]
        arm = item["arm"]
        if phase not in _MEMORY_PHASES or arm not in _MEMORY_ARMS:
            raise RunnerError("memory phase/arm is not registered")
            if phase in {"before_correctness", "after_timing"} and arm != "all":
                raise RunnerError("memory phase/arm semantics are invalid")
        if phase in {"before_correctness", "after_timing"} and arm != "all":
            raise RunnerError("memory phase/arm semantics are invalid")
        measured_at = item["measured_at_ns"]
        if isinstance(measured_at, bool) or not isinstance(measured_at, int) or not 0 < measured_at <= MEMORY_MAX_INT:
            raise RunnerError("memory evidence measurement time is invalid")
        if item["name"] in {"mlx_peak_memory", "rss"}:
            value, reason = _projection_missing(item["value"], item["missing_reason"], f"memory.{item['name']}")
            latest[item["name"]] = (value, reason)
            if value is None and reason is None:
                raise RunnerError("memory missing value has no reason")
    expected_gate = (
        "aggregation_required"
        if latest.get("mlx_peak_memory", (None, None))[0] is not None and latest.get("rss", (None, None))[0] is not None
        else "not_evaluable_missing_required_metric"
    )
    if gate != expected_gate:
        raise RunnerError("benchmark memory gate is not reconstructed from memory rows")


def _normalize_correctness(
    domain: Mapping[str, Any], fixture: Mapping[str, Any], manifest: ClosedManifest,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    workload = manifest.value["workload"]
    correctness = _projection_mapping(domain.get("correctness"), "benchmark_evidence.correctness")
    if set(correctness) != {"cases", "passed", "performance", "sign_invariant"} or correctness["passed"] is not True:
        raise RunnerError("correctness evidence is incomplete or failed")
    cases = correctness["cases"]
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes, bytearray)) or len(cases) != 9:
        raise RunnerError("correctness must contain exactly nine cases")
    by_name: dict[str, Mapping[str, Any]] = {}
    for case in cases:
        item = _projection_mapping(case, "correctness.case")
        name = _projection_text(item.get("name"), "correctness.case.name", maximum=64)
        if name in by_name or name not in _CORRECTNESS_NAMES:
            raise RunnerError("correctness case names are not unique and registered")
        expected_keys = _SIGN_INVARIANT_KEYS if name == "sign_invariant" else _CORRECTNESS_FULL_KEYS
        if set(item) != expected_keys:
            raise RunnerError(f"correctness case {name} has unknown or missing fields")
        if item["passed"] is not True:
            raise RunnerError(f"correctness case {name} failed")
        seed = _projection_int(item["seed"], f"correctness.{name}.seed")
        if name == "sign_invariant":
            try:
                validate_sign_invariant_case(
                    item,
                    manifest.value["seeds"]["fixture"],
                    fixture_digest(fixture["a_sha256"], fixture["b_sha256"], fixture["fixture_seed"]),
                )
            except CorrectnessContractError as exc:
                raise RunnerError(str(exc)) from exc
            by_name[name] = item
            continue
        elif name != "performance_fixture":
            try:
                validate_fixed_case(item)
            except CorrectnessContractError as exc:
                raise RunnerError(str(exc)) from exc
        else:
            try:
                validate_performance_case(
                    item,
                    {
                        **fixture,
                        "a_shape": workload["a_shape"],
                        "b_shape": workload["b_shape"],
                        "dtype": workload["dtype"],
                        "layout": workload["layout"],
                    },
                )
            except CorrectnessContractError as exc:
                raise RunnerError(str(exc)) from exc
        metrics = _projection_mapping(item["metrics"], f"correctness.{name}.metrics")
        if set(metrics) != set(_CORRECTNESS_METRIC_NAMES):
            raise RunnerError(f"correctness case {name} metrics are not complete")
        for metric_name, metric_value in metrics.items():
            if metric_name == "rel_q99_abs_oracle_ge_1":
                metric_item = _projection_mapping(metric_value, f"correctness.{name}.{metric_name}")
                if set(metric_item) != {"value", "missing_reason"}:
                    raise RunnerError(f"correctness.{name}.{metric_name} missing contract is invalid")
                _projection_missing(metric_item["value"], metric_item["missing_reason"], f"correctness.{name}.{metric_name}")
            else:
                _projection_number(metric_value, f"correctness.{name}.{metric_name}")
        by_name[name] = item
    if tuple(by_name) != _CORRECTNESS_NAMES:
        raise RunnerError("correctness case order is not the registered order")
    performance = by_name["performance_fixture"]
    workload = manifest.value["workload"]
    expected_performance_shape = list(workload["a_shape"]) + list(workload["b_shape"])
    if (
        performance["shape"] != expected_performance_shape
        or performance["dtype"] != workload["dtype"]
        or performance["layout"] != workload["layout"]
        or performance["seed"] != fixture["fixture_seed"]
        or performance["zero_rhs"] is not False
        or performance["a_sha256"] != fixture["a_sha256"]
        or performance["b_sha256"] != fixture["b_sha256"]
    ):
        raise RunnerError("performance correctness case is not bound to the manifest fixture")
    expected_performance_digest = fixture_digest(
        performance["a_sha256"], performance["b_sha256"], performance["seed"]
    )
    if performance["fixture_digest"] != expected_performance_digest:
        raise RunnerError("performance correctness fixture digest is not bound to its inputs")
    for linked_name, case_name in (("performance", "performance_fixture"), ("sign_invariant", "sign_invariant")):
        linked = _projection_mapping(correctness.get(linked_name), f"correctness.{linked_name}")
        if canonical_json_bytes(linked) != canonical_json_bytes(by_name[case_name]):
            raise RunnerError(f"correctness.{linked_name} is not an exact case link")
    artifacts: list[dict[str, Any]] = []
    omitted: list[str] = []
    for name in ("a_sha256", "b_sha256", "metadata_sha256", "fixture_sha256"):
        digest = fixture.get(name)
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise RunnerError(f"fixture.{name} is not a lowercase SHA-256")
        artifacts.append({"artifact_name": f"fixture_{name.removesuffix('_sha256')}", "artifact_kind": "performance_fixture", "sha256": digest, "metadata": {"mode": manifest.mode, "source": f"benchmark_evidence.fixture.{name}"}})
    rows: list[dict[str, Any]] = []
    for name in _CORRECTNESS_NAMES:
        case = by_name[name]
        digest = case.get("fixture_digest")
        derived = False
        if name == "sign_invariant":
            digest = canonical_sha256({"case": name, "seed": case["seed"], "fixture_sha256": fixture["fixture_sha256"]})
            derived = True
        artifacts.append({"artifact_name": f"correctness_fixture_{name}", "artifact_kind": "correctness_fixture", "sha256": digest, "metadata": {"case": name, "seed": case["seed"], "derived": derived, "source": "benchmark_evidence.correctness.cases"}})
        if name == "sign_invariant":
            continue
        for metric_name in _CORRECTNESS_METRIC_NAMES:
            metric = _projection_mapping(case["metrics"][metric_name], f"correctness.{name}.{metric_name}") if metric_name == "rel_q99_abs_oracle_ge_1" else {"value": case["metrics"][metric_name], "missing_reason": None}
            if set(metric) != {"value", "missing_reason"}:
                raise RunnerError(f"correctness.{name}.{metric_name} missing contract is invalid")
            value, reason = _projection_missing(metric["value"], metric["missing_reason"], f"correctness.{name}.{metric_name}")
            if value is None:
                omitted.append(f"correctness.{name}.{metric_name}:{reason}")
                continue
            rows.append({"case_name": name, "metric_name": metric_name, "value": value, "unit": "ratio", "passed": case["passed"], "detail": {"seed": case["seed"], "fixture_digest": digest, "derived_fixture_digest": derived}})
    return rows, artifacts, omitted


def _normalize_scalar_projection(
    evidence: Mapping[str, Any], domain: Mapping[str, Any] | None, mode: str, *, invalid: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in _SUPERVISOR_SCALARS:
        value = evidence.get(name)
        reason = evidence.get(f"{name}_missing_reason")
        if value is None and reason is None:
            reason = "not_recorded"
        _projection_scalar(rows, name, value, _SCALAR_UNITS[name], reason=reason)
    if invalid or domain is None:
        if "total_ns" in evidence:
            if evidence["total_ns"] is not None:
                _projection_positive_number(evidence["total_ns"], "total_ns")
            _projection_scalar(rows, "total_elapsed_ns", evidence["total_ns"], "ns")
        return sorted(rows, key=lambda row: row["metric_name"])
    for name in ("compile_wrapper_setup_ns", "first_eval_compile_inclusive_ns", "total_elapsed_ns"):
        value = domain.get(name)
        reason = "not_applicable" if value is None else None
        if value is not None:
            _projection_positive_number(value, f"benchmark_evidence.{name}")
        _projection_scalar(rows, name, value, _SCALAR_UNITS[name], reason=reason)
    arms = _projection_mapping(domain.get("arms"), "benchmark_evidence.arms")
    for arm_name in ("baseline", "candidate"):
        if arm_name not in arms:
            continue
        arm = _projection_mapping(arms[arm_name], f"benchmark_evidence.arms.{arm_name}")
        warmup = _projection_mapping(arm["warmup"], f"benchmark_evidence.arms.{arm_name}.warmup")
        repetition = _projection_mapping(arm["repetitions"], f"benchmark_evidence.arms.{arm_name}.repetitions")
        stats = _projection_mapping(arm["statistics"], f"benchmark_evidence.arms.{arm_name}.statistics")
        prefix = arm_name
        _projection_scalar(rows, f"{prefix}_warmup_count", warmup["count"], "count")
        _projection_scalar(rows, f"{prefix}_warmup_median_ns", warmup["median_ns"], "ns")
        _projection_scalar(rows, f"{prefix}_repetitions", repetition["repetitions"], "count")
        _projection_scalar(rows, f"{prefix}_calibration_batch_ns", repetition["batch_ns"], "ns")
        for source, target in (("count", "batch_count"), ("median_ns", "median_ns"), ("mad_ns", "mad_ns"), ("iqr_ns", "iqr_ns"), ("min_ns", "min_ns"), ("max_ns", "max_ns")):
            _projection_scalar(rows, f"{prefix}_{target}", stats[source], "count" if source == "count" else "ns")
    comparison = domain.get("comparison")
    if isinstance(comparison, Mapping) and "ratio_statistics" in comparison:
        ratio = _projection_mapping(comparison.get("ratio_statistics"), "benchmark_evidence.comparison.ratio_statistics")
        for source, target in (("count", "ratio_count"), ("median_ratio", "ratio_median"), ("mad_ratio", "ratio_mad"), ("iqr_ratio", "ratio_iqr"), ("min_ratio", "ratio_min"), ("max_ratio", "ratio_max")):
            _projection_scalar(rows, target, ratio[source], "count" if source == "count" else "ratio")
    memory = domain.get("memory")
    if not isinstance(memory, Sequence) or isinstance(memory, (str, bytes, bytearray)):
        raise RunnerError("benchmark_evidence.memory is not a sequence")
    for index, entry in enumerate(memory):
        item = _projection_mapping(entry, f"benchmark_evidence.memory[{index}]")
        allowed = {"name", "api", "unit", "value", "missing_reason", "measurement_phase", "arm", "measured_at_ns", "reset_state"}
        if set(item) != allowed or item["name"] not in _MEMORY_NAMES or item["measurement_phase"] not in _MEMORY_PHASES or item["arm"] not in _MEMORY_ARMS or item["unit"] != "bytes":
            raise RunnerError("memory evidence is not closed or allowlisted")
        if not isinstance(item["api"], str) or not item["api"] or item["reset_state"] != "not_reset_or_api_unavailable":
            raise RunnerError("memory evidence API/reset semantics are invalid")
        if isinstance(item["measured_at_ns"], bool) or not isinstance(item["measured_at_ns"], int) or not 0 < item["measured_at_ns"] <= MEMORY_MAX_INT:
            raise RunnerError("memory evidence measurement time is invalid")
        if (item["measurement_phase"] in {"after_calibration", "after_measurement"}) != (item["arm"] in {"baseline", "candidate"}):
            raise RunnerError("memory evidence phase/arm semantics are invalid")
        if item["measurement_phase"] not in {"after_calibration", "after_measurement"} and item["arm"] != "all":
            raise RunnerError("memory evidence phase/arm semantics are invalid")
        metric_name = f"memory_{item['name']}_{item['measurement_phase']}_{item['arm']}"
        if len(metric_name) > 128:
            raise RunnerError("memory metric name exceeds the bound")
        _projection_scalar(rows, metric_name, item["value"], "bytes", reason=item["missing_reason"])
    names = [row["metric_name"] for row in rows]
    if len(names) != len(set(names)):
        raise RunnerError("normalized scalar metric names are duplicated")
    return sorted(rows, key=lambda row: row["metric_name"])


def normalize_mlx_common_result(manifest: ClosedManifest, result: Mapping[str, Any]) -> dict[str, Any]:
    """Project one protocol-valid MLX H0 result into exact SQLite-v1 arrays.

    The full validated result is persisted by the caller as the terminal event.
    This projection deliberately omits correctness metrics whose SQLite v1 row
    has no missing-value column, and returns an explicit reason instead of
    manufacturing zeroes.
    """

    if not isinstance(manifest, ClosedManifest) or manifest.mode not in _MLX_MEASUREMENT_MODES:
        raise RunnerError("MLX projection requires a closed H0 measurement manifest")
    try:
        validated = validate_result(result, manifest=manifest)
    except (ProtocolError, TypeError, ValueError) as exc:
        raise RunnerError("MLX projection requires a protocol-valid Common Result") from exc
    evidence = _projection_mapping(validated["evidence"], "result.evidence")
    if set(evidence) - _COMMON_EVIDENCE_KEYS:
        raise RunnerError("result.evidence has unknown adapter fields")
    projection_omissions: list[str] = []
    is_measurement = validated["status"] == "completed" and validated["classification"] == "measurement_complete" and validated["action"] == "baseline_fallback" and validated["error"] is None
    is_fallback = validated["status"] in {"invalid", "timeout", "worker_exit"} or validated["classification"] in {"runtime_unavailable", "invalid", "invalid: correctness", "invalid: missing_required_field"}
    if not (is_measurement or is_fallback):
        raise RunnerError("H0 MLX projection rejects promotion/regression or an unknown result class")
    if is_measurement or "adapter_contract" in evidence:
        _validate_projection_adapter_contract(evidence)
    source_evidence_sha256 = canonical_sha256(evidence)
    result_sha256 = canonical_sha256(validated)
    if is_fallback:
        scalar_metrics = _normalize_scalar_projection(evidence, None, manifest.mode, invalid=True)
        projection_omissions.append(f"benchmark_evidence:{validated['classification']}")
        core = {"raw_samples": [], "scalar_metrics": scalar_metrics, "correctness_metrics": [], "artifacts": []}
        projection_shell = {"projection_schema": "friday_h0.normalization_projection.v1", "manifest_sha256": manifest.sha256, "result_sha256": result_sha256, "source_evidence_sha256": source_evidence_sha256, "arrays": core}
        projection_sha256 = canonical_sha256(projection_shell)
        projection_artifact = {"artifact_name": "normalization_projection_v1", "artifact_kind": "normalization_projection", "sha256": projection_sha256, "metadata": {"manifest_sha256": manifest.sha256, "result_sha256": result_sha256, "source_evidence_sha256": source_evidence_sha256, "counts": {key: len(value) for key, value in core.items()}}}
        core["artifacts"].append(projection_artifact)
        return {**core, "projection_schema": "friday_h0.normalization_projection.v1", "manifest_sha256": manifest.sha256, "result_sha256": result_sha256, "source_evidence_sha256": source_evidence_sha256, "projection_sha256": projection_sha256, "counts": {key: len(value) for key, value in core.items()}, "omitted_normalized_fields": sorted(projection_omissions)}
    if "benchmark_evidence" not in evidence or evidence.get("benchmark_classification") not in {"baseline_reference", "measurement_complete"}:
        raise RunnerError("measurement result lacks a registered benchmark evidence domain")
    domain = _projection_mapping(evidence["benchmark_evidence"], "result.evidence.benchmark_evidence")
    if set(domain) != _DOMAIN_EVIDENCE_KEYS:
        raise RunnerError("benchmark evidence schema is not closed")
    expected_binding = _COMMON_MODE_BINDINGS[manifest.mode]
    actual_binding = (
        evidence.get("benchmark_classification"),
        evidence.get("benchmark_action"),
        evidence.get("aggregation_required"),
    )
    if type(actual_binding[2]) is not bool or actual_binding != expected_binding:
        raise RunnerError("common result mode binding is invalid")
    if type(domain.get("aggregation_required")) is not bool or domain["aggregation_required"] is not expected_binding[2]:
        raise RunnerError("benchmark evidence aggregation binding is invalid")
    fixture = _projection_mapping(domain.get("fixture"), "benchmark_evidence.fixture")
    if set(fixture) != {"fixture_seed", "a_sha256", "b_sha256", "metadata_sha256", "fixture_sha256"}:
        raise RunnerError("benchmark fixture fields are not closed")
    if fixture["fixture_seed"] != manifest.value["seeds"]["fixture"]:
        raise RunnerError("benchmark fixture seed is not bound to the manifest")
    arms = _projection_mapping(domain.get("arms"), "benchmark_evidence.arms")
    if manifest.mode == "eager_baseline":
        if set(arms) != {"baseline"}:
            raise RunnerError("eager baseline must contain exactly one arm")
        baseline_arm = _projection_mapping(arms["baseline"], "baseline")
        raw_samples, _ = _normalize_arm_samples(manifest, "baseline", baseline_arm, paired=False)
        source_raw_samples = list(baseline_arm["raw_samples"])
        if "raw_samples" not in domain:
            raise RunnerError("eager baseline domain raw samples are missing")
        comparison = _projection_mapping(domain.get("comparison"), "benchmark_evidence.comparison")
        required = {"raw_samples", "benchmark_classification", "action", "aggregation_required", "cache_state"}
        if set(comparison) != required:
            raise RunnerError("eager baseline comparison evidence is not closed")
        if (
            comparison["benchmark_classification"] != "baseline_reference"
            or comparison["action"] != "not_run"
            or comparison["aggregation_required"] is not False
            or domain.get("cache_state") != "unknown"
            or comparison["cache_state"] != domain["cache_state"]
        ):
            raise RunnerError("eager baseline comparison semantics are invalid")
        if canonical_json_bytes(comparison["raw_samples"]) != canonical_json_bytes(domain["raw_samples"]):
            raise RunnerError("eager baseline comparison raw samples differ from domain evidence")
        if canonical_json_bytes(comparison["raw_samples"]) != canonical_json_bytes(source_raw_samples):
            raise RunnerError("eager baseline comparison raw samples differ from baseline arm")
    else:
        if set(arms) != {"baseline", "candidate"}:
            raise RunnerError("comparison mode must contain exactly baseline and candidate arms")
        baseline_arm = _projection_mapping(arms["baseline"], "baseline")
        candidate_arm = _projection_mapping(arms["candidate"], "candidate")
        raw_baseline, _ = _normalize_arm_samples(manifest, "baseline", baseline_arm, paired=True)
        raw_candidate, _ = _normalize_arm_samples(manifest, "candidate", candidate_arm, paired=True)
        raw_samples = raw_baseline + raw_candidate
        source_raw_samples = list(baseline_arm["raw_samples"]) + list(candidate_arm["raw_samples"])
        comparison = _projection_mapping(domain.get("comparison"), "benchmark_evidence.comparison")
        required = {"order", "blocks", "raw_samples", "ratio_statistics", "benchmark_classification", "action", "aggregation_required", "aggregation_gate", "global_decision", "comparison_kind"}
        expected_kind = "aa_gpu_null_control" if manifest.mode == "aa_gpu" else "eager_vs_compiled"
        expected_gate = "aa_gate" if manifest.mode == "aa_gpu" else "phase1_promotion_gate"
        if set(comparison) != required or comparison["benchmark_classification"] != "session_observation" or comparison["aggregation_required"] is not True or comparison["global_decision"] is not None or comparison["action"] != "aggregation_required" or comparison["comparison_kind"] != expected_kind or comparison["aggregation_gate"] != expected_gate:
            raise RunnerError("comparison evidence is not an aggregation observation")
        order = comparison["order"]
        expected_order = _registered_balanced_order(manifest.value["seeds"]["order"])
        if order != expected_order:
            raise RunnerError("comparison order is not reproducible from the manifest seed")
        ratios: list[float] = []
        blocks = comparison["blocks"]
        if not isinstance(blocks, Sequence) or len(blocks) != 30:
            raise RunnerError("comparison must contain exactly 30 paired blocks")
        for index, block in enumerate(blocks):
            item = _projection_mapping(block, f"comparison.blocks[{index}]")
            if set(item) != {"block_index", "first", "second", "baseline_per_eval_ns", "candidate_per_eval_ns", "ratio"} or item["block_index"] != index:
                raise RunnerError("comparison block is not closed")
            if (item["first"], item["second"]) != (order[index], "candidate" if order[index] == "baseline" else "baseline"):
                raise RunnerError("comparison order is not reproducible")
            baseline_value = _projection_number(item["baseline_per_eval_ns"], "comparison baseline")
            candidate_value = _projection_number(item["candidate_per_eval_ns"], "comparison candidate")
            if baseline_value != baseline_arm["batches"][index]["per_eval_ns"] or candidate_value != candidate_arm["batches"][index]["per_eval_ns"]:
                raise RunnerError("comparison block values differ from original arm batches")
            if baseline_arm["batches"][index]["position"] != ("first" if order[index] == "baseline" else "second") or candidate_arm["batches"][index]["position"] != ("first" if order[index] == "candidate" else "second"):
                raise RunnerError("comparison order differs from arm batch positions")
            ratio = _projection_number(item["ratio"], "comparison ratio")
            expected_ratio = candidate_value / baseline_value
            if ratio != expected_ratio:
                raise RunnerError("comparison ratio is inconsistent")
            ratios.append(expected_ratio)
        ratio_statistics = _projection_mapping(comparison["ratio_statistics"], "comparison.ratio_statistics")
        if set(ratio_statistics) != {"count", "median_ratio", "mad_ratio", "iqr_ratio", "min_ratio", "max_ratio"}:
            raise RunnerError("comparison ratio statistics are not closed")
        expected_ratio_statistics = {
            "count": 30, "median_ratio": _median(ratios), "mad_ratio": _mad(ratios),
            "iqr_ratio": _iqr(ratios), "min_ratio": min(ratios), "max_ratio": max(ratios),
        }
        if any(ratio_statistics[key] != value for key, value in expected_ratio_statistics.items()):
            raise RunnerError("comparison ratio statistics are not reconstructed from blocks")
        if canonical_json_bytes(comparison["raw_samples"]) != canonical_json_bytes(domain["raw_samples"]):
            raise RunnerError("comparison raw sample envelope differs")
    if canonical_json_bytes(domain["raw_samples"]) != canonical_json_bytes(source_raw_samples):
        raise RunnerError("benchmark raw sample envelope differs from arm projection")
    _validate_timing_contract(domain, arms, manifest.mode)
    if _projection_positive_number(domain["total_elapsed_ns"], "benchmark_evidence.total_elapsed_ns") <= 0:
        raise RunnerError("benchmark total elapsed time is invalid")
    _validate_memory_contract(domain)
    correctness_metrics, artifacts, correctness_omissions = _normalize_correctness(domain, fixture, manifest)
    projection_omissions.extend(correctness_omissions)
    scalar_metrics = _normalize_scalar_projection(evidence, domain, manifest.mode, invalid=False)
    core = {"raw_samples": sorted(raw_samples, key=lambda row: (row["sample_kind"], row["sample_index"])), "scalar_metrics": scalar_metrics, "correctness_metrics": sorted(correctness_metrics, key=lambda row: (row["case_name"], row["metric_name"])), "artifacts": sorted(artifacts, key=lambda row: (row["artifact_name"], row["artifact_kind"]))}
    projection_shell = {"projection_schema": "friday_h0.normalization_projection.v1", "manifest_sha256": manifest.sha256, "result_sha256": result_sha256, "source_evidence_sha256": source_evidence_sha256, "arrays": core}
    projection_sha256 = canonical_sha256(projection_shell)
    projection_artifact = {"artifact_name": "normalization_projection_v1", "artifact_kind": "normalization_projection", "sha256": projection_sha256, "metadata": {"manifest_sha256": manifest.sha256, "result_sha256": result_sha256, "source_evidence_sha256": source_evidence_sha256, "counts": {key: len(value) for key, value in core.items()}}}
    core["artifacts"].append(projection_artifact)
    return {**core, "projection_schema": "friday_h0.normalization_projection.v1", "manifest_sha256": manifest.sha256, "result_sha256": result_sha256, "source_evidence_sha256": source_evidence_sha256, "projection_sha256": projection_sha256, "counts": {key: len(value) for key, value in core.items()}, "omitted_normalized_fields": sorted(projection_omissions)}


def _persist_mlx_common_result(
    manifest: ClosedManifest, result: Mapping[str, Any], *, _test_context: _TestRoot | None = None,
    created_at_unix_ns: int | None = None, recorded_at_ns: int | None = None,
) -> dict[str, Any]:
    """Private later-worker seam: normalize and atomically persist one result."""

    if not isinstance(manifest, ClosedManifest):
        raise RunnerError("private MLX persistence requires a ClosedManifest")
    projection = normalize_mlx_common_result(manifest, result)
    root = _test_context.root if _test_context is not None else _REPOSITORY_ROOT
    if _test_context is not None and (not isinstance(_test_context, _TestRoot) or _test_context.token is not _TEST_ROOT_TOKEN):
        raise RunnerError("invalid private database test context")
    with _storage_for_root(root) as storage:
        persistence = storage.persist_common_result(
            manifest, result, created_at_unix_ns=time.time_ns() if created_at_unix_ns is None else created_at_unix_ns,
            raw_samples=projection["raw_samples"], scalar_metrics=projection["scalar_metrics"],
            correctness_metrics=projection["correctness_metrics"], artifacts=projection["artifacts"],
            recorded_at_ns=recorded_at_ns,
        )
    return {"manifest": manifest, "result": dict(result), "projection": projection, "persistence": persistence}


def load_aa_sessions(*, _test_context: _TestRoot | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read exactly the current-provenance 3+3 A/A sessions through SQLite mode=ro."""

    if _test_context is not None and (not isinstance(_test_context, _TestRoot) or _test_context.token is not _TEST_ROOT_TOKEN):
        raise RunnerError("invalid private database test context")
    provenance = collect_provenance()
    expected_provenance = provenance.as_manifest()
    root = _test_context.root if _test_context is not None else _REPOSITORY_ROOT
    grouped: dict[str, list[dict[str, Any]]] = {"characterization": [], "confirmation": []}
    with _storage_for_root(root, read_only=True) as storage:
        runs = storage.connection.execute("SELECT run_id FROM runs WHERE phase=? AND mode=? ORDER BY run_id", (PHASE_H0, "aa_gpu")).fetchall()
        if len(runs) != 6:
            raise RunnerError("A/A loader requires exactly six aa_gpu runs")
        seen: set[tuple[str, int]] = set()
        for row in runs:
            run_id = _projection_text(row["run_id"], "stored run_id", maximum=128)
            stored = storage.get_run(run_id)
            if stored is None:
                raise RunnerError("stored A/A run disappeared")
            manifest_value = validate_manifest(stored["manifest"])
            closed = close_manifest(manifest_value)
            process = _projection_mapping(manifest_value.get("process"), "stored manifest process")
            process_set = _projection_text(process.get("set"), "stored process set", maximum=32)
            process_index = _projection_int(process.get("index"), "stored process index")
            if process_set not in grouped or process_index not in range(3) or (process_set, process_index) in seen:
                raise RunnerError("stored A/A process tuple is missing or duplicated")
            if canonical_json_bytes(manifest_value["provenance"]) != canonical_json_bytes(expected_provenance):
                raise RunnerError("stored A/A provenance does not match current provenance")
            if run_id != run_id_for("aa_gpu", process_set, process_index, provenance):
                raise RunnerError("stored A/A run_id is not deterministic for current provenance")
            events = [event for event in storage.rows("status_events", run_id) if event["event_kind"] == "common_result"]
            if len(events) != 1:
                raise RunnerError("stored A/A run must have exactly one common_result")
            try:
                payload = json.loads(events[0]["payload_json"])
                result = payload["result"]
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RunnerError("stored A/A common_result payload is invalid") from exc
            try:
                validated = validate_result(result, manifest=closed)
            except (ProtocolError, TypeError, ValueError) as exc:
                raise RunnerError("stored A/A common_result is not protocol-valid") from exc
            if (validated["status"], validated["classification"], validated["action"], validated["error"]) != ("completed", "measurement_complete", "baseline_fallback", None):
                raise RunnerError("stored A/A run is not a neutral measurement result")
            projection = normalize_mlx_common_result(closed, validated)
            verification = storage.verify_common_result_bundle(
                closed,
                validated,
                raw_samples=projection["raw_samples"],
                scalar_metrics=projection["scalar_metrics"],
                correctness_metrics=projection["correctness_metrics"],
                artifacts=projection["artifacts"],
            )
            if verification != "verified":
                raise RunnerError("stored A/A common_result bundle was not verified")
            grouped[process_set].append({"manifest": manifest_value, "result": validated})
            seen.add((process_set, process_index))
    for process_set in grouped:
        grouped[process_set].sort(key=lambda session: session["manifest"]["process"]["index"])
        if [session["manifest"]["process"]["index"] for session in grouped[process_set]] != [0, 1, 2]:
            raise RunnerError(f"stored A/A {process_set} sessions are incomplete")
    return grouped["characterization"], grouped["confirmation"]


def load_and_aggregate_h0_aa(*, _test_context: _TestRoot | None = None) -> dict[str, Any]:
    """Load the six read-only sessions and pass their session forms to the pure aggregator."""

    characterization, confirmation = load_aa_sessions(_test_context=_test_context)
    # Lazy import keeps normalization/persistence free of benchmark and MLX imports.
    from .aggregation import aggregate_h0_aa

    def bridge(session: Mapping[str, Any]) -> dict[str, Any]:
        # The complete Common Result remains in the terminal SQLite event and
        # is returned unchanged by load_aa_sessions.  The pure A/A aggregator
        # consumes only its registered statistical evidence contract; stream
        # capture metadata belongs to the supervisor diagnostic projection.
        result = _projection_mapping(session.get("result"), "A/A result")
        evidence = _projection_mapping(result.get("evidence"), "A/A result evidence")
        bridged_result = dict(result)
        bridged_result["evidence"] = {
            key: value for key, value in evidence.items() if key not in _AGGREGATION_BRIDGE_DROP_KEYS
        }
        return {"manifest": session["manifest"], "result": bridged_result}

    return aggregate_h0_aa(
        [bridge(session) for session in characterization],
        [bridge(session) for session in confirmation],
    )


def run_offline(
    mode: str,
    *,
    project_root: str | Path | None = None,
    _test_context: _TestRoot | None = None,
    limits: SupervisorLimits | None = None,
    now_ns: int | None = None,
) -> RunOutcome:
    """Run a fixed analysis/control worker and persist its complete Common Result."""

    if mode not in OFFLINE_MODES:
        raise RunnerError("mode is not allowlisted for offline execution")
    if project_root is not None:
        raise RunnerError("alternate database roots require the private test factory")
    if _test_context is not None and (
        not isinstance(_test_context, _TestRoot) or _test_context.token is not _TEST_ROOT_TOKEN
    ):
        raise RunnerError("invalid private database test context")
    if limits is not None and _test_context is None:
        raise RunnerError("custom supervisor limits require the private test context")
    provenance = collect_provenance()
    manifest = close_manifest(build_manifest(mode, provenance))
    # The closed manifest is the only object crossing the supervisor boundary.
    try:
        result = run_supervised(manifest, limits)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        # Parent setup failures must remain observable and persistable, without
        # echoing paths or exception text into the Common Result.
        result = fallback_result(
            manifest=manifest,
            status="invalid",
            classification="invalid",
            code="parent_setup_failure",
            message=f"{type(exc).__name__} before worker result",
            evidence={"rss_peak_bytes": None, "rss_missing_reason": "parent_setup_failure"},
        )
    try:
        result = validate_result(result, manifest=manifest)
    except (ProtocolError, OSError, ValueError) as exc:
        raise RunnerError("parent could not obtain a validated worker result") from exc
    evidence = result.get("evidence")
    if not isinstance(evidence, dict):
        raise RunnerError("validated result evidence is not an object")
    created = time.time_ns() if now_ns is None else now_ns
    try:
        root = _test_context.root if _test_context is not None else _REPOSITORY_ROOT
        path = _database_path_for_tests(_test_context) if _test_context is not None else database_path()
        with _storage_for_root(root) as storage:
            persistence = storage.persist_common_result(
                manifest,
                result,
                created_at_unix_ns=created,
                scalar_metrics=_metric_rows(evidence),
                artifacts=_artifact_rows(result),
            )
    except (RunnerError, StorageError, OSError, ValueError) as exc:
        raise StorageError(f"offline result was not persisted: {exc}") from exc
    return RunOutcome(manifest, result, persistence)


def run_mlx(
    mode: str,
    process_set: str,
    process_index: int,
    *,
    _test_context: _TestRoot | None = None,
    limits: SupervisorLimits | None = None,
    now_ns: int | None = None,
) -> RunOutcome:
    """Run one registered MLX mode through Option A and persist its result.

    The CLI's explicit ``--execute`` gate is an execution interlock/confirmation
    flag, not authentication or proof of caller authorization.  This function still repeats the closed mode/process validation
    and accepts alternate roots or limits only through the private test seam.
    The worker may return a runtime-unavailable fallback; that result remains
    persisted as a non-promoted baseline fallback until a runtime is present.
    """

    if mode not in MLX_MODES:
        raise RunnerError("mode is not an allowlisted MLX mode")
    if process_set not in {"characterization", "confirmation"}:
        raise RunnerError("MLX process set is not registered")
    if isinstance(process_index, bool) or not isinstance(process_index, int) or not 0 <= process_index <= 2:
        raise RunnerError("MLX process index is not registered")
    if _test_context is not None and (
        not isinstance(_test_context, _TestRoot) or _test_context.token is not _TEST_ROOT_TOKEN
    ):
        raise RunnerError("invalid private database test context")
    if limits is not None and _test_context is None:
        raise RunnerError("custom supervisor limits require the private test context")

    provenance = collect_provenance()
    manifest = close_manifest(
        build_manifest(mode, provenance, process_set=process_set, process_index=process_index)
    )
    try:
        result = run_supervised(manifest, limits)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        result = fallback_result(
            manifest=manifest,
            status="invalid",
            classification="invalid",
            code="parent_setup_failure",
            message=f"{type(exc).__name__} before worker result",
            evidence={"rss_peak_bytes": None, "rss_missing_reason": "parent_setup_failure"},
        )
    try:
        result = validate_result(result, manifest=manifest)
    except (ProtocolError, OSError, ValueError) as exc:
        raise RunnerError("parent could not obtain a validated MLX worker result") from exc

    created = time.time_ns() if now_ns is None else now_ns
    try:
        persisted = _persist_mlx_common_result(
            manifest,
            result,
            _test_context=_test_context,
            created_at_unix_ns=created,
        )
    except (RunnerError, StorageError, OSError, ValueError) as exc:
        raise StorageError(f"MLX result was not persisted: {exc}") from exc
    return RunOutcome(manifest, result, persisted["persistence"])


def result_exit_code(result: dict[str, Any]) -> int:
    """Map only validated Common Results to the closed CLI result codes."""

    return 0 if result.get("action") == "promoted" and result.get("status") == "completed" else 10


__all__ = [
    "OFFLINE_MODES", "RunOutcome", "RunnerError", "build_manifest", "database_path",
    "initialize_database", "load_aa_sessions", "load_and_aggregate_h0_aa",
    "normalize_mlx_common_result", "result_exit_code", "run_id_for", "run_mlx", "run_offline",
]
