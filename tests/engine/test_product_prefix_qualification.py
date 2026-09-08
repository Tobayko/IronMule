"""Pure JSON controller-contract tests; no MLX model, cache, or GPU is faked.

The complete fixture below is only a protocol-shaped metadata example. It is
not evidence that a native qualification ran or that any hardware claim holds.
"""

from copy import deepcopy

import pytest

from tools.product_prefix_qualification import (
    CHECKS, CHILD_SCHEMA, EXPECTED_PROMPT_TOKENS, QualificationFailure,
    digest_json, validate_child_report,
)


_HASH = "a" * 64


def _output_record():
    return {"output_sha256": _HASH, "token_sha256": _HASH,
            "text_sha256": _HASH, "logprobs_sha256": _HASH,
            "completion_tokens": 8, "prompt_tokens": EXPECTED_PROMPT_TOKENS,
            "finish_reason": "length"}


def protocol_example():
    """Return validator-shaped metadata, never a fabricated native result."""
    layer = {"class": "ExampleCache", "meta_state": ["metadata-only"],
             "state": [{"shape": [1], "dtype": "float16", "sha256": _HASH,
                        "bytes": 1}], "nbytes": 1}
    checkpoint = {"layers": [layer], "nbytes": 1}
    checkpoint["sha256"] = digest_json(checkpoint["layers"])
    cold = {"status": "completed", "failure_type": "none", "cache_hit": False,
            "cache_stored": True, "reused_tokens": 0, "fallback_reason": "none"}
    hit = {**cold, "cache_hit": True, "cache_stored": False,
           "reused_tokens": EXPECTED_PROMPT_TOKENS - 1}
    output = _output_record()
    return {"schema": CHILD_SCHEMA, "status": "passed", "correctness_gate": True,
            "model_id": "m", "revision": "r", "device": "Device(gpu, 0)",
            "prompt_tokens": EXPECTED_PROMPT_TOKENS, "performance_claim": False,
            "activation_allowed": False, "mlx_active_bytes": 1, "mlx_peak_bytes": 1,
            "checks": {name: True for name in CHECKS},
            "stock": [deepcopy(output) for _ in range(4)],
            "candidate": [deepcopy(output) for _ in range(4)],
            "traces": [cold, deepcopy(hit), deepcopy(hit), deepcopy(hit)],
            "recovery_cold": deepcopy(output), "recovery_hit": deepcopy(output),
            "checkpoint": checkpoint, "canonical_after_hits": deepcopy(checkpoint)}


def test_digest_is_canonical_and_does_not_persist_input():
    assert digest_json({"b": 2, "a": 1}) == digest_json({"a": 1, "b": 2})


def test_validator_accepts_complete_metadata_protocol_example_only():
    assert validate_child_report(protocol_example(), "m", "r")["correctness_gate"] is True


def test_previous_minimal_success_shape_is_not_a_native_qualification():
    minimal = {"schema": CHILD_SCHEMA, "status": "passed", "correctness_gate": True,
               "model_id": "m", "revision": "r", "prompt_tokens": EXPECTED_PROMPT_TOKENS,
               "performance_claim": False, "activation_allowed": False}
    with pytest.raises(QualificationFailure):
        validate_child_report(minimal, "m", "r")


@pytest.mark.parametrize("field", ["checks", "stock", "candidate", "traces", "checkpoint"])
def test_validator_rejects_missing_required_protocol_sections(field):
    row = protocol_example(); del row[field]
    with pytest.raises(QualificationFailure): validate_child_report(row, "m", "r")


@pytest.mark.parametrize(("field", "value"), [
    ("device", "Device(cpu, 0)"), ("performance_claim", True),
    ("activation_allowed", True), ("mlx_active_bytes", 0), ("mlx_peak_bytes", False),
])
def test_validator_rejects_non_native_or_promotional_claims(field, value):
    row = protocol_example(); row[field] = value
    with pytest.raises(QualificationFailure, match="child_device_or_claim_invalid"):
        validate_child_report(row, "m", "r")


@pytest.mark.parametrize("path", [
    ("stock", 0, "text"), ("traces", 1, "scope_nonce"),
    ("checkpoint", "layers", 0, "state", 0, "tokens"),
])
def test_validator_rejects_nested_raw_payload_fields(path):
    row = protocol_example(); target = row
    for key in path[:-1]: target = target[key]
    target[path[-1]] = "must-not-persist"
    with pytest.raises(QualificationFailure, match="raw_data_exposed"):
        validate_child_report(row, "m", "r")


def test_validator_rejects_incomplete_or_nonidentical_protocol_records():
    row = protocol_example(); row["checks"]["wrong_scope"] = False
    with pytest.raises(QualificationFailure, match="child_checks_incomplete"):
        validate_child_report(row, "m", "r")

    row = protocol_example(); row["candidate"][3]["output_sha256"] = "b" * 64
    with pytest.raises(QualificationFailure, match="child_outputs_invalid"):
        validate_child_report(row, "m", "r")

    row = protocol_example(); row["traces"][2]["reused_tokens"] = 0
    with pytest.raises(QualificationFailure, match="child_cache_trace_invalid"):
        validate_child_report(row, "m", "r")

    row = protocol_example(); row["canonical_after_hits"]["sha256"] = "b" * 64
    with pytest.raises(QualificationFailure, match="child_checkpoint_invalid"):
        validate_child_report(row, "m", "r")


@pytest.mark.parametrize(("index", "field", "value"), [(0, "reused_tokens", 1), (1, "cache_stored", True)])
def test_validator_rejects_inconsistent_cold_or_hit_accounting(index, field, value):
    row = protocol_example(); row["traces"][index][field] = value
    with pytest.raises(QualificationFailure, match="child_cache_trace_invalid"):
        validate_child_report(row, "m", "r")
