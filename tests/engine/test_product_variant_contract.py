"""Pure option/JSON protocol checks, not model or hardware simulations."""

from copy import deepcopy

import pytest

from ironmule_product.backend import MLXWorkerClient
from ironmule_product.errors import BackendUnavailable, InvalidRequest
from ironmule_product.types import GenerationRequest, ModelSpec


def spec():
    return ModelSpec("protocol/model", "revision", "/protocol/snapshot", 1)


def prefix_metadata():
    return {
        "status": "accepted", "commit_status": "stored",
        "trace": {"status": "completed", "failure_type": "none", "cache_hit": False,
                  "cache_stored": True, "reused_tokens": 0, "fallback_reason": "none",
                  "restore_host_ns": 1, "capture_host_ns": 1, "commit_host_ns": 1},
        "stats": {"hits": 0, "misses": 1, "inserts": 1, "skipped_oversize": 0,
                  "evictions": 0, "entries": 1, "bytes": 1},
    }


def test_experimental_variants_are_opt_in_without_starting_a_worker():
    reference = MLXWorkerClient(spec())
    request = GenerationRequest(spec().model_id, (("user", "protocol example"),))
    assert reference.execution_variant == "reference" and reference._process is None
    for variant in ("prefix_reuse", "current_engine"):
        with pytest.raises(InvalidRequest, match="not enabled"):
            reference.stream(request, variant=variant)
        client = MLXWorkerClient(spec(), execution_variant=variant)
        assert client._process is None
        # Merely constructing the iterator is still model-free.
        client.stream(request)
        assert client._process is None


@pytest.mark.parametrize("variant", [[], {}, "unknown", True])
def test_invalid_variant_is_rejected_before_model_start(variant):
    with pytest.raises(ValueError):
        MLXWorkerClient(spec(), execution_variant=variant)


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5, 2**63])
def test_cache_capacity_is_validated_as_metadata_not_model_admission(capacity):
    with pytest.raises(ValueError):
        MLXWorkerClient(spec(), prefix_cache_max_bytes=capacity)


def test_cache_reset_never_implicitly_loads_a_model():
    client = MLXWorkerClient(spec(), execution_variant="prefix_reuse")
    with pytest.raises(BackendUnavailable):
        client.clear_prefix_cache()
    assert client._process is None


def test_prefix_frame_rejects_raw_or_malformed_metadata():
    example = prefix_metadata()
    MLXWorkerClient._validate_prefix_metadata(example)
    for field, value in (("status", []), ("trace", {}), ("stats", {"tokens": [1]})):
        malformed = deepcopy(example); malformed[field] = value
        with pytest.raises(BackendUnavailable):
            MLXWorkerClient._validate_prefix_metadata(malformed)
    malformed = deepcopy(example); malformed["trace"]["prompt"] = "private example"
    with pytest.raises(BackendUnavailable):
        MLXWorkerClient._validate_prefix_metadata(malformed)


def test_long_startup_duration_is_valid_telemetry_not_a_new_timeout():
    # A JSON metadata boundary only; this does not claim a day-long model run.
    MLXWorkerClient._validate_startup_telemetry({
        "startup_wall_seconds": 86401.0, "process_peak_rss_bytes": 1,
        "mlx_active_bytes": 0, "mlx_peak_bytes": 0, "mlx_cache_bytes": 0,
        "recommended_working_set_bytes": 1,
    })
