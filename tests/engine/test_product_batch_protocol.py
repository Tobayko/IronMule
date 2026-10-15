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
     ("prefix_reuse", "current_profile"),
     ("current_engine", "baseline_interactive"), ("current_engine", "core_interactive")],
)
def test_unqualified_worker_families_cannot_batch(variant: str, configuration: str) -> None:
    client = MLXWorkerClient(_spec(), execution_variant=variant, engine_configuration=configuration)
    assert client.batch_capacity == 1
    assert not client.can_batch_requests([_request("a"), _request("b")])


def test_the_current_profile_engine_groups_simultaneous_streams() -> None:
    """SERVE4: `--engine tuned` admits up to 8 simultaneous requests, streaming ones included,
    into one grouped engine session (grouped batch-1, exact; refused on recurrent caches)."""
    client = MLXWorkerClient(_spec(), execution_variant="current_engine", engine_configuration="current_profile")
    assert client.batch_capacity == 8 and client.streams_batches
    streaming = [_request("a", stream=True), _request("b", stream=True)]
    assert client.can_batch_requests(streaming)
    assert not client.ready
    throughput = MLXWorkerClient(_spec(), execution_variant="current_engine", engine_configuration="core_throughput")
    assert not throughput.streams_batches and not throughput.can_batch_requests(streaming)


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


def test_every_configuration_that_advertises_grouping_is_accepted_by_its_worker() -> None:
    """The client's batch admission and the worker's dispatch must agree (SERVE4's first T4 run
    failed here: the client grouped current-profile streams, the worker refused the batch)."""
    from ironmule_product.backend import ENGINE_CONFIGURATIONS
    from ironmule_product.worker import BATCH_CONFIGURATIONS

    grouping = {configuration for configuration in ENGINE_CONFIGURATIONS
                if MLXWorkerClient(_spec(), execution_variant="current_engine",
                                   engine_configuration=configuration).batch_capacity > 1}
    assert grouping == set(BATCH_CONFIGURATIONS)


def test_a_request_handed_over_mid_batch_is_sent_as_a_join_and_its_events_accepted(monkeypatch) -> None:
    """SERVE8 at the protocol boundary: join_batch goes out once, the joiner's frames are
    validated like the originals', and batch_done must name everyone in send order."""
    client = MLXWorkerClient(_spec(), execution_variant="current_engine", engine_configuration="current_profile")
    monkeypatch.setattr(MLXWorkerClient, "ready", property(lambda self: True))
    sent, delivered = [], []
    monkeypatch.setattr(client, "_send", lambda command, deadline: sent.append(command))
    batch_id = {}

    def frames():
        batch_id["id"] = sent[0]["batch_id"]
        for rid in ("a", "b"):
            yield {"type": "token", "batch_id": batch_id["id"], "request_id": rid, "text": "x", "token_id": 1,
                   "prompt_tokens": 1, "completion_tokens": 1}
        assert sent[-1]["type"] == "join_batch" and [r["request_id"] for r in sent[-1]["requests"]] == ["c"]
        for rid in ("c", "a", "b", "c"):
            done = rid in ("a", "b") or len([e for e in delivered if e[0] == "c"]) == 1
            yield ({"type": "done", "batch_id": batch_id["id"], "request_id": rid, "finish_reason": "length",
                    "prompt_tokens": 1, "completion_tokens": 1, "variant": "current_engine", "engine": {}}
                   if done else
                   {"type": "token", "batch_id": batch_id["id"], "request_id": rid, "text": "y", "token_id": 2,
                    "prompt_tokens": 1, "completion_tokens": 1})
        yield {"type": "batch_done", "batch_id": batch_id["id"], "request_ids": ["a", "b", "c"]}

    stream = frames()
    monkeypatch.setattr(client, "_read_event", lambda deadline, on_tick=None: next(stream))
    handed = [[], [(_request("c", stream=True), None)]]
    rows = client.complete_batch([_request("a", stream=True), _request("b", stream=True)],
                                 on_event=lambda rid, event: delivered.append((rid, event["type"])),
                                 joins=lambda: handed.pop() if handed else [])
    assert [row["request_id"] for row in rows] == ["a", "b", "c"]
    assert [command["type"] for command in sent] == ["generate_batch", "join_batch"]
    assert ("c", "token") in delivered and ("c", "done") in delivered
