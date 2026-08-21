"""Closed recursive H0 manifest v1 validation and hashing."""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .canonical import canonical_json_bytes, canonical_sha256
from .constants import (
    ALLOWED_MODES,
    ANALYSIS_MODES,
    AA_BOOTSTRAP_SEEDS,
    AA_SESSION_SEEDS,
    MLX_MODES,
    PHASE_H0,
    SCHEMA_VERSION,
    CONTROL_MODES,
    EAGER_COMPILE_SESSION_SEEDS,
    WRONG_FIXTURE_SEED,
    WRONG_FIXTURE_SIZE,
)


class ManifestError(ValueError):
    """Raised when a manifest violates the closed H0 v1 contract."""


_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_MISSING_REASON = "project root is not a Git repository"
_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version", "phase", "run_id", "mode", "workload", "seeds", "limits",
        "process", "provenance",
    }
)
_WORKLOAD_KEYS = frozenset(
    {"operation", "a_shape", "b_shape", "y_shape", "dtype", "layout", "generator", "distribution"}
)
_SEED_KEYS = frozenset({"fixture", "order"})
_AA_SEED_KEYS = frozenset({"fixture", "order", "bootstrap_seed"})
_LIMIT_KEYS = frozenset({"first_eval_s", "synchronize_s", "total_s"})
_PROCESS_KEYS = frozenset({"set", "index"})
_PROVENANCE_KEYS = frozenset({"code_sha256", "spec_sha256", "environment_sha256", "revision"})
_REVISION_KEYS = frozenset({"value", "missing_reason"})


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], path: str) -> None:
    actual = set(value)
    unknown = sorted(actual - expected)
    missing = sorted(expected - actual)
    if unknown or missing:
        parts = []
        if unknown:
            parts.append(f"unknown keys {unknown}")
        if missing:
            parts.append(f"missing keys {missing}")
        raise ManifestError(f"{path}: " + "; ".join(parts))


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestError(f"{path} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ManifestError(f"{path} has a non-string key")
    return value


def _int(value: Any, path: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestError(f"{path} must be an integer, not bool or another type")
    if value < minimum or (maximum is not None and value > maximum):
        raise ManifestError(f"{path} outside [{minimum}, {maximum}]")
    return value


def _finite_number(value: Any, path: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestError(f"{path} must be a finite number")
    if not math.isfinite(float(value)) or float(value) < minimum:
        raise ManifestError(f"{path} must be finite and >= {minimum}")
    return float(value)


def _bounded_string(value: Any, path: str, *, maximum: int = 128) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum or "\x00" in value:
        raise ManifestError(f"{path} must be a bounded non-empty string")
    return value


def _shape(value: Any, path: str) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ManifestError(f"{path} must be a two-element array")
    if len(value) != 2:
        raise ManifestError(f"{path} must contain exactly two dimensions")
    return [_int(dimension, f"{path}[{index}]", minimum=1, maximum=8192) for index, dimension in enumerate(value)]


def validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a deep copy of the closed manifest.

    The v1 contract intentionally has no generic metadata, path, source, module,
    command, code, or flag field. This is the narrowest representation that can
    identify the H0 arm and its fixed workload without authorizing execution.
    """

    top = _mapping(manifest, "manifest")
    _exact_keys(top, _TOP_LEVEL_KEYS, "manifest")
    if _int(top["schema_version"], "manifest.schema_version", minimum=1, maximum=1) != SCHEMA_VERSION:
        raise ManifestError("manifest.schema_version must be 1")
    if top["phase"] != PHASE_H0:
        raise ManifestError("manifest.phase must be exactly H0")
    run_id = top["run_id"]
    if not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id):
        raise ManifestError("manifest.run_id must be a bounded identifier without paths")
    mode = top["mode"]
    if not isinstance(mode, str) or mode not in ALLOWED_MODES:
        raise ManifestError(f"manifest.mode must be one of {sorted(ALLOWED_MODES)}")

    process = _mapping(top["process"], "manifest.process")
    _exact_keys(process, _PROCESS_KEYS, "manifest.process")
    process_set = process["set"]
    if not isinstance(process_set, str):
        raise ManifestError("manifest.process.set must be a string")
    process_index = _int(process["index"], "manifest.process.index", maximum=2)
    if mode in {"aa_gpu", "eager_baseline", "compile_comparison"}:
        if process_set not in {"characterization", "confirmation"}:
            raise ManifestError("MLX/A-A process.set must be characterization or confirmation")
    elif mode in ANALYSIS_MODES:
        if process_set != "analysis" or process_index != 0:
            raise ManifestError("analysis process must be {set: analysis, index: 0}")
    elif mode in CONTROL_MODES:
        if process_set != "control" or process_index != 0:
            raise ManifestError("control process must be {set: control, index: 0}")
    if mode in {"aa_gpu", "eager_baseline", "compile_comparison"} and process_index > 2:
        raise ManifestError("characterization/confirmation process.index must be in 0..2")

    provenance = _mapping(top["provenance"], "manifest.provenance")
    _exact_keys(provenance, _PROVENANCE_KEYS, "manifest.provenance")
    for field in ("code_sha256", "spec_sha256", "environment_sha256"):
        value = provenance[field]
        if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
            raise ManifestError(f"manifest.provenance.{field} must be lowercase 64-hex")
    revision = _mapping(provenance["revision"], "manifest.provenance.revision")
    _exact_keys(revision, _REVISION_KEYS, "manifest.provenance.revision")
    revision_value = revision["value"]
    revision_reason = revision["missing_reason"]
    if revision_value is not None:
        _bounded_string(revision_value, "manifest.provenance.revision.value")
    if revision_reason is not None:
        _bounded_string(revision_reason, "manifest.provenance.revision.missing_reason")
    if (revision_value is None) == (revision_reason is None):
        raise ManifestError("manifest.provenance.revision requires exactly one non-null field")
    if revision_value is None and revision_reason != _REVISION_MISSING_REASON:
        raise ManifestError("manifest.provenance.revision.missing_reason is not the registered root reason")

    workload = _mapping(top["workload"], "manifest.workload")
    _exact_keys(workload, _WORKLOAD_KEYS, "manifest.workload")
    if workload["operation"] != "matmul":
        raise ManifestError("manifest.workload.operation must be matmul")
    a_shape = _shape(workload["a_shape"], "manifest.workload.a_shape")
    b_shape = _shape(workload["b_shape"], "manifest.workload.b_shape")
    y_shape = _shape(workload["y_shape"], "manifest.workload.y_shape")
    if a_shape[1] != b_shape[0] or y_shape != [a_shape[0], b_shape[1]]:
        raise ManifestError("manifest.workload shapes are not matmul-compatible")
    fixed_strings = {
        "dtype": "float16",
        "layout": "C-contiguous",
        "generator": "PCG64",
        "distribution": "uniform[-1,1)",
    }
    for field, expected in fixed_strings.items():
        if workload[field] != expected:
            raise ManifestError(f"manifest.workload.{field} must be {expected!r}")

    seeds = _mapping(top["seeds"], "manifest.seeds")
    expected_seed_keys = _AA_SEED_KEYS if mode == "aa_gpu" else _SEED_KEYS
    _exact_keys(seeds, expected_seed_keys, "manifest.seeds")
    for field in expected_seed_keys:
        _int(seeds[field], f"manifest.seeds.{field}", maximum=(1 << 64) - 1)

    limits = _mapping(top["limits"], "manifest.limits")
    _exact_keys(limits, _LIMIT_KEYS, "manifest.limits")
    for field in _LIMIT_KEYS:
        _finite_number(limits[field], f"manifest.limits.{field}", minimum=0.0)
    if float(limits["first_eval_s"]) != 10.0:
        raise ManifestError("manifest.limits.first_eval_s must be the registered 10 seconds")
    if float(limits["synchronize_s"]) != 5.0:
        raise ManifestError("manifest.limits.synchronize_s must be the registered 5 seconds")
    if float(limits["total_s"]) != 120.0:
        raise ManifestError("manifest.limits.total_s must be the registered 120 seconds")

    expected_size = WRONG_FIXTURE_SIZE if mode == "analysis_wrong_fixture" else 2048
    if a_shape != [expected_size, expected_size] or b_shape != [expected_size, expected_size]:
        raise ManifestError(f"manifest.workload shapes must be the registered {expected_size}² shape for {mode}")
    if mode == "analysis_wrong_fixture" and seeds["fixture"] != WRONG_FIXTURE_SEED:
        raise ManifestError("analysis_wrong_fixture requires seed 0xBAD02026")
    expected_fixture = 0
    expected_order = 0
    if mode == "aa_gpu":
        expected_fixture = AA_SESSION_SEEDS[
            f"{process_set}_fixture"
        ] + process_index
        expected_order = AA_SESSION_SEEDS[f"{process_set}_order"] + process_index
        if seeds["bootstrap_seed"] != AA_BOOTSTRAP_SEEDS[process_set]:
            raise ManifestError("manifest.seeds.bootstrap_seed does not match the registered A/A set")
    elif mode in {"eager_baseline", "compile_comparison"}:
        expected_fixture = EAGER_COMPILE_SESSION_SEEDS[f"{process_set}_fixture"] + process_index
        expected_order = EAGER_COMPILE_SESSION_SEEDS[f"{process_set}_order"] + process_index
    elif mode in ANALYSIS_MODES | CONTROL_MODES:
        expected_fixture = WRONG_FIXTURE_SEED if mode == "analysis_wrong_fixture" else 0
        expected_order = 0
    if seeds["fixture"] != expected_fixture or seeds["order"] != expected_order:
        raise ManifestError("manifest.seeds do not match the registered process arm")

    # Deep-copy prevents a caller from mutating the validated object after hashing.
    return copy.deepcopy(dict(manifest))


def canonical_manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    """Validate and canonicalize one manifest."""

    return canonical_json_bytes(validate_manifest(manifest))


def manifest_hash(manifest: Mapping[str, Any]) -> str:
    """Return the SHA-256 of the canonical validated manifest."""

    return canonical_sha256(canonical_manifest_bytes(manifest))
