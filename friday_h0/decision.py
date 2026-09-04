"""Pure deterministic H0 analysis fixtures and decision hashes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .canonical import canonical_sha256
from .constants import (
    ANALYSIS_BASELINE_NS,
    ANALYSIS_CLUSTERS,
    ANALYSIS_KNOWN_WIN_FACTOR,
    ANALYSIS_PAIRS_PER_CLUSTER,
    ANALYSIS_SLOW_FACTOR,
    ANALYSIS_STEP_NS,
    REQUIRED_MEMORY_FIELD,
    WRONG_FIXTURE_SEED,
    WRONG_FIXTURE_SIZE,
)
from .statistics import median


class DecisionError(ValueError):
    """Raised for unsupported or structurally invalid decision inputs."""


def baseline_ns(cluster: int, block: int) -> int:
    """Return the exact registered deterministic baseline formula."""

    if isinstance(cluster, bool) or not isinstance(cluster, int) or not 0 <= cluster < ANALYSIS_CLUSTERS:
        raise DecisionError("cluster must be 0, 1, or 2")
    if isinstance(block, bool) or not isinstance(block, int) or not 0 <= block < ANALYSIS_PAIRS_PER_CLUSTER:
        raise DecisionError("block must be in 0..29")
    return ANALYSIS_BASELINE_NS + ANALYSIS_STEP_NS * (((17 * cluster + 13 * block) % 11) - 5)


def make_analysis_fixture(kind: str) -> dict[str, Any]:
    """Create a non-GPU fixture; no sleep, timer, or runtime call is performed."""

    if kind not in {"slow", "known_win", "wrong", "missing"}:
        raise DecisionError("kind must be slow, known_win, wrong, or missing")
    if kind == "wrong":
        return {
            "fixture_version": 1,
            "kind": "wrong",
            "shape": [WRONG_FIXTURE_SIZE, WRONG_FIXTURE_SIZE],
            "seed": WRONG_FIXTURE_SEED,
            "correctness_passed": False,
            "correctness_reason": "zeros_like(matmul)",
            "timed": False,
        }
    if kind == "missing":
        return {
            "fixture_version": 1,
            "kind": "missing",
            "required_fields": ["run_id", "rss_peak_bytes"],
            "fields": {"run_id": "analysis-missing"},
            "missing_reason": None,
            "timed": False,
        }
    candidate_factor = ANALYSIS_SLOW_FACTOR if kind == "slow" else ANALYSIS_KNOWN_WIN_FACTOR
    rows = []
    for cluster in range(ANALYSIS_CLUSTERS):
        for block in range(ANALYSIS_PAIRS_PER_CLUSTER):
            baseline = baseline_ns(cluster, block)
            candidate = int(baseline * (11 if kind == "slow" else 9) // 10)
            rows.append({"cluster": cluster, "block": block, "baseline_ns": baseline, "candidate_ns": candidate})
    return {
        "fixture_version": 1,
        "kind": kind,
        "factor": candidate_factor,
        "rows": rows,
        "timed": False,
    }


def decision_hash(decision: Mapping[str, Any]) -> str:
    """Hash a decision without recursively including a prior hash field."""

    payload = {key: value for key, value in decision.items() if key != "decision_hash"}
    return canonical_sha256(payload)


def _with_hash(decision: dict[str, Any]) -> dict[str, Any]:
    decision["decision_hash"] = decision_hash(decision)
    return decision


def evaluate_analysis_fixture(kind: str) -> dict[str, Any]:
    """Return a deterministic classification and fallback/promotion action."""

    fixture = make_analysis_fixture(kind)
    if kind == "wrong":
        return _with_hash(
            {
                "schema_version": 1,
                "kind": kind,
                "classification": "invalid: correctness",
                "action": "baseline_fallback",
                "timed": False,
                "fixture": fixture,
            }
        )
    if kind == "missing":
        return _with_hash(
            {
                "schema_version": 1,
                "kind": kind,
                "classification": "invalid: missing_required_field",
                "action": "baseline_fallback",
                "timed": False,
                "missing_field": REQUIRED_MEMORY_FIELD,
                "fixture": fixture,
            }
        )
    rows = fixture["rows"]
    ratios = [row["candidate_ns"] / row["baseline_ns"] for row in rows]
    ratio = median(ratios)
    promoted = kind == "known_win" and ratio < 1.0
    return _with_hash(
        {
            "schema_version": 1,
            "kind": kind,
            "classification": "promoted" if promoted else "regression",
            "action": "promoted" if promoted else "baseline_fallback",
            "timed": False,
            "ratio": ratio,
            "fixture": fixture,
        }
    )


def evaluate_slow_fixture() -> dict[str, Any]:
    """Convenience wrapper for the registered 1.10x analytical regression."""

    return evaluate_analysis_fixture("slow")


def evaluate_known_win_fixture() -> dict[str, Any]:
    """Convenience wrapper for the registered 0.90x analytical promotion."""

    return evaluate_analysis_fixture("known_win")


def evaluate_wrong_fixture() -> dict[str, Any]:
    """Convenience wrapper for the never-timed wrong-result fixture."""

    return evaluate_analysis_fixture("wrong")


def evaluate_missing_data_fixture() -> dict[str, Any]:
    """Convenience wrapper for the required-field failure fixture."""

    return evaluate_analysis_fixture("missing")

