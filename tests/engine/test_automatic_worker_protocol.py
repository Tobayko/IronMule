from __future__ import annotations

import pytest

from ironmule_product.backend import MAX_PROTOCOL_LINE, MLXWorkerClient
from ironmule_product.types import GenerationRequest, ModelSpec


def _spec() -> ModelSpec:
    return ModelSpec("local/model", "revision", "/nonexistent/model", 1)


def _requests() -> list[GenerationRequest]:
    return [GenerationRequest("local/model", (("user", "hello"),), request_id=value)
            for value in ("a", "b")]


def test_automatic_evidence_is_bounded_json_and_not_a_request_field() -> None:
    evidence = [{"audit_id": "record"}]
    client = MLXWorkerClient(_spec(), execution_variant="automatic", selection_evidence=evidence)
    evidence[0]["audit_id"] = "mutated"
    assert client.selection_evidence == [{"audit_id": "record"}]
    assert "selection_evidence" not in _requests()[0].as_dict()
    with pytest.raises(ValueError, match="JSON records"):
        MLXWorkerClient(_spec(), execution_variant="automatic", selection_evidence=[{"bad": object()}])
    with pytest.raises(ValueError, match="bounded JSON"):
        MLXWorkerClient(_spec(), execution_variant="automatic",
                        selection_evidence=[{"large": "x" * MAX_PROTOCOL_LINE}])


def test_automatic_batch_requires_worker_reported_current_qualification() -> None:
    client = MLXWorkerClient(_spec(), execution_variant="automatic")
    assert client.batch_capacity == 1
    assert not client.can_batch_requests(_requests())
    client._ready_payload = {"automatic_batch_qualified": False}  # noqa: SLF001 - readiness contract
    assert client.batch_capacity == 1
    # A readiness claim is considered only for a live worker; stale metadata
    # cannot activate grouped transport.
    assert not client.can_batch_requests(_requests())


def test_curated_selection_metadata_has_exact_bounded_shape() -> None:
    value = {
        "candidate_id": "reference",
        "reason": "no_matching_independent_heldout_promotion",
        "evidence_audit_id": None,
        "explored": False,
        "rejected_evidence_records": 0,
        "actual_backend": "reference",
        "actual_group_width": 1,
    }
    assert MLXWorkerClient._valid_selection_metadata(value)  # noqa: SLF001
    assert not MLXWorkerClient._valid_selection_metadata({**value, "prompt": "secret"})  # noqa: SLF001
    assert not MLXWorkerClient._valid_selection_metadata({**value, "actual_group_width": 33})  # noqa: SLF001


def test_empty_evidence_is_valid_and_does_not_start_worker() -> None:
    client = MLXWorkerClient(_spec(), execution_variant="automatic", selection_evidence=[])
    assert client.selection_evidence == []
    assert not client.ready
