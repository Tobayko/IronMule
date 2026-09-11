from __future__ import annotations

import threading

import pytest

from ironmule_product.backend import MLXWorkerClient
from ironmule_product.errors import InvalidRequest
from ironmule_product.types import GenerationRequest, ModelSpec


def _spec() -> ModelSpec:
    return ModelSpec("local/model", "revision", "/nonexistent/model", 1)


def _request(request_id: str, *, model: str = "local/model", stream: bool = False) -> GenerationRequest:
    return GenerationRequest(model=model, messages=(("user", "hello"),), max_tokens=4,
                             stream=stream, request_id=request_id)


@pytest.mark.parametrize("configuration", ["baseline_throughput", "core_throughput"])
def test_only_engine_throughput_workers_advertise_grouping(configuration: str) -> None:
    client = MLXWorkerClient(_spec(), execution_variant="current_engine",
                            engine_configuration=configuration)
    assert client.batch_capacity == 32
    assert client.can_batch_requests([_request("a"), _request("b")])
    assert not client.ready  # Metadata-only admission must not start a model worker.


@pytest.mark.parametrize(
    ("variant", "configuration"),
    [("reference", "current_profile"), ("bounded_prefetch", "current_profile"),
     ("prefix_reuse", "current_profile"), ("current_engine", "current_profile"),
     ("current_engine", "baseline_interactive"), ("current_engine", "core_interactive")],
)
def test_unqualified_worker_families_cannot_batch(variant: str, configuration: str) -> None:
    client = MLXWorkerClient(_spec(), execution_variant=variant, engine_configuration=configuration)
    assert client.batch_capacity == 1
    assert not client.can_batch_requests([_request("a"), _request("b")])


def test_batch_admission_is_bounded_unique_same_model_and_nonstreaming() -> None:
    client = MLXWorkerClient(_spec(), execution_variant="current_engine",
                            engine_configuration="core_throughput")
    assert not client.can_batch_requests([_request("only")])
    assert not client.can_batch_requests([_request(str(i)) for i in range(33)])
    assert not client.can_batch_requests([_request("same"), _request("same")])
    assert not client.can_batch_requests([_request("a"), _request("b", model="other")])
    assert not client.can_batch_requests([_request("a"), _request("b", stream=True)])
    assert not client.ready


def test_complete_batch_rejects_before_start_and_validates_cancel_shape() -> None:
    client = MLXWorkerClient(_spec(), execution_variant="current_engine",
                            engine_configuration="core_throughput")
    with pytest.raises(InvalidRequest, match="eligible"):
        client.complete_batch([_request("only")])
    with pytest.raises(TypeError, match="cancels"):
        client.complete_batch([_request("a"), _request("b")], cancels=[threading.Event()])
    assert not client.ready


def test_oversized_group_is_rejected_without_worker_start() -> None:
    client = MLXWorkerClient(_spec(), execution_variant="current_engine",
                            engine_configuration="baseline_throughput")
    requests = [_request("a"), GenerationRequest(
        model="local/model", messages=(("user", "x" * (1024 * 1024)),), max_tokens=4,
        request_id="b",
    )]
    assert not client.can_batch_requests(requests)
    assert not client.ready
