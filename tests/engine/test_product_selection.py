"""Pure contract tests: these records are not hardware or model measurements."""

from __future__ import annotations

import pytest

from ironmule_product.selection import (
    IntegrationEvidence,
    RequestContext,
    RewardObservation,
    SelectionContractError,
    propose_ranks,
    select_candidate,
)


SHA = "a" * 64


def request(**changes: object) -> RequestContext:
    value = dict(model_id="local/model", model_revision="immutable/rev", model_sha256=SHA,
                 hardware_sha256=SHA,
                 environment_sha256=SHA, code_sha256=SHA,
                 source_manifest_sha256=SHA, profile="interactive", session_requests=1,
                 context_tokens=100, max_new_tokens=16, stream=True,
                 group_width=1, prefix_hit=True)
    value.update(changes)
    return RequestContext.from_mapping(value)


def evidence(**changes: object) -> IntegrationEvidence:
    value = dict(audit_id="audit-1", candidate_id="prefix_reuse",
                 baseline_candidate_id="reference",
                 model_id="local/model", model_revision="immutable/rev", model_sha256=SHA,
                 hardware_sha256=SHA,
                 environment_sha256=SHA, code_sha256=SHA,
                 source_manifest_sha256=SHA, heldout=True, quality_exact=True,
                 profile="interactive", min_session_requests=1, max_session_requests=1,
                 min_context_tokens=1, max_context_tokens=200,
                 min_new_tokens=1, max_new_tokens=32, streamable=True,
                 group_widths=[1], prefix_hit_capable=True, prefix_hit_required=True,
                 cold_setup_included=False,
                 estimated_e2e_ms=8.0, pair_ratio=0.8, confidence_upper=0.88,
                 frozen_min_effect=0.05)
    value.update(changes)
    return IntegrationEvidence.from_mapping(value)


def test_empty_store_and_legacy_candidate_names_do_not_authorize_an_optimization() -> None:
    decision = select_candidate(request(), [])
    assert decision.candidate_id == "reference"
    assert decision.evidence_audit_id is None
    assert decision.explored is False


def test_fastest_matching_independently_proven_e2e_candidate_wins() -> None:
    rows = [evidence(audit_id="slow", candidate_id="core_interactive", estimated_e2e_ms=9.0),
            evidence(audit_id="fast", estimated_e2e_ms=8.0)]
    decision = select_candidate(request(), rows)
    assert (decision.candidate_id, decision.evidence_audit_id) == ("prefix_reuse", "fast")


@pytest.mark.parametrize("change", [
    {"heldout": False}, {"quality_exact": False}, {"confidence_upper": 0.96},
    {"model_revision": "other"}, {"model_id": "other/model"},
    {"model_sha256": "b" * 64}, {"hardware_sha256": "b" * 64},
    {"environment_sha256": "b" * 64}, {"code_sha256": "b" * 64},
    {"source_manifest_sha256": "b" * 64}, {"streamable": False},
    {"group_widths": [2]},
])
def test_missing_uncertain_or_context_mismatched_evidence_falls_back(change: dict[str, object]) -> None:
    assert select_candidate(request(), [evidence(**change)]).candidate_id == "reference"


def test_switch_from_compatible_incumbent_requires_direct_proof() -> None:
    row = evidence(baseline_candidate_id="reference")
    decision = select_candidate(request(), [row], incumbent_candidate_id="core_interactive")
    assert decision.candidate_id == "reference"


def test_throughput_has_its_own_reference() -> None:
    assert select_candidate(request(profile="throughput"), []).candidate_id == "reference"


def test_available_prefix_does_not_disqualify_a_normal_candidate() -> None:
    row = evidence(candidate_id="core_interactive", prefix_hit_capable=False,
                   prefix_hit_required=False, cold_setup_included=False)
    assert select_candidate(request(prefix_hit=True), [row]).candidate_id == "core_interactive"


def test_session_scope_must_match_evidence_bounds() -> None:
    row = evidence(min_session_requests=2, max_session_requests=8)
    assert select_candidate(request(session_requests=1), [row]).candidate_id == "reference"


def test_cold_prefix_evidence_must_include_setup() -> None:
    with pytest.raises(SelectionContractError, match="include setup"):
        evidence(prefix_hit_required=False, cold_setup_included=False)


@pytest.mark.parametrize("change", [
    {"pair_ratio": float("nan")}, {"confidence_upper": float("inf")},
    {"hardware_sha256": "not-a-hash"}, {"group_widths": [1, 1]},
    {"candidate_id": "arbitrary"}, {"max_context_tokens": 0},
    {"prefix_hit_capable": False, "prefix_hit_required": False},
])
def test_malformed_pure_json_evidence_is_rejected(change: dict[str, object]) -> None:
    with pytest.raises(SelectionContractError):
        evidence(**change)


def observation(audit: str, candidate: str, split: str, cluster: str, reward: float) -> RewardObservation:
    return RewardObservation.from_mapping(dict(audit_id=audit, candidate_id=candidate,
                                               split=split, cluster_id=cluster,
                                               model_id="local/model", model_sha256=SHA,
                                               profile="interactive", session_requests=1,
                                               prefix_hit=True,
                                               actual_reward=reward))


def test_empirical_learner_uses_actual_rewards_but_cannot_promote() -> None:
    rows = [observation("a", "prefix_reuse", "train", "process-1", 3.0),
            observation("b", "prefix_reuse", "validation", "process-2", 2.0),
            observation("c", "core_interactive", "train", "process-3", 1.0),
            observation("h", "core_interactive", "heldout", "process-4", 100.0)]
    proposals = propose_ranks(rows)
    assert proposals[0].candidate_id == "prefix_reuse"
    assert all(item.candidate_id != "reference" for item in proposals)
    assert select_candidate(request(), []).candidate_id == "reference"


def test_clusters_must_be_disjoint_across_splits() -> None:
    rows = [observation("a", "prefix_reuse", "train", "same-run", 1.0),
            observation("b", "prefix_reuse", "heldout", "same-run", 2.0)]
    with pytest.raises(SelectionContractError, match="cluster crosses"):
        propose_ranks(rows)


def test_duplicate_reward_audit_ids_are_rejected() -> None:
    rows = [observation("a", "prefix_reuse", "train", "p1", 1.0),
            observation("a", "prefix_reuse", "train", "p1", 2.0)]
    with pytest.raises(SelectionContractError, match="audit_id"):
        propose_ranks(rows)
