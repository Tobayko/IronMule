"""Pure-stdlib H0 measurement, decision, and append-only storage primitives."""

from .canonical import CanonicalizationError, canonical_json, canonical_json_bytes, canonical_sha256
from .constants import (
    AA_BOOTSTRAP_SEEDS,
    AA_SESSION_SEEDS,
    ALLOWED_MODES,
    BOOTSTRAP_REPLICATES,
    PHASE_H0,
)
from .decision import (
    DecisionError,
    decision_hash,
    evaluate_analysis_fixture,
    make_analysis_fixture,
)
from .manifest import ManifestError, canonical_manifest_bytes, manifest_hash, validate_manifest
from .statistics import (
    StatisticsError,
    aa_gate,
    hierarchical_bootstrap,
    iqr,
    mad,
    median,
    percentile,
    sample_standard_deviation,
    session_ratio,
    set_ratio,
)
from .storage import Storage, StorageError

__all__ = [
    "AA_BOOTSTRAP_SEEDS",
    "AA_SESSION_SEEDS",
    "ALLOWED_MODES",
    "BOOTSTRAP_REPLICATES",
    "PHASE_H0",
    "CanonicalizationError",
    "DecisionError",
    "ManifestError",
    "StatisticsError",
    "Storage",
    "StorageError",
    "aa_gate",
    "canonical_json",
    "canonical_json_bytes",
    "canonical_sha256",
    "canonical_manifest_bytes",
    "decision_hash",
    "evaluate_analysis_fixture",
    "hierarchical_bootstrap",
    "iqr",
    "mad",
    "make_analysis_fixture",
    "manifest_hash",
    "median",
    "percentile",
    "sample_standard_deviation",
    "session_ratio",
    "set_ratio",
    "validate_manifest",
]
