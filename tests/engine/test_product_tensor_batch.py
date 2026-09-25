"""PERF1-K: the opt-in tensor-batch worker variant, at its protocol and admission boundaries.

A fake BatchGenerator stands in for mlx-lm's, so these tests check which rows reach it, what
the worker emits and when rows leave the batch; they make no model, arithmetic or speed claim.
"""

from __future__ import annotations

import sys
import threading
import types

import pytest

from ironmule_product import worker
from ironmule_product.backend import MLXWorkerClient
from ironmule_product.errors import BackendUnavailable
from ironmule_product.service import _execution_label
from ironmule_product.types import GenerationRequest, ModelSpec

EOS = 99


def _spec() -> ModelSpec:
    return ModelSpec("local/model", "revision", "/nonexistent/model", 1)


def _request(request_id: str, *, stream: bool = False, max_tokens: int = 4) -> GenerationRequest:
    return GenerationRequest(model="local/model", messages=(("user", request_id),), max_tokens=max_tokens,
                             stream=stream, request_id=request_id)


def test_tensor_batch_advertises_its_width_and_admits_only_buffered_requests() -> None:
    client = MLXWorkerClient(_spec(), execution_variant="tensor_batch", batch_width=3)
    assert client.batch_capacity == 3
    assert client.can_batch_requests([_request("a"), _request("b"), _request("c")])
    assert not client.can_batch_requests([_request(str(i)) for i in range(4)])
    assert not client.can_batch_requests([_request("a"), _request("b", stream=True)])
    assert not client.ready  # admission is metadata only


@pytest.mark.parametrize(("variant", "width"), [("tensor_batch", None), ("tensor_batch", 1),
                                                ("tensor_batch", 9), ("reference", 4)])
def test_a_batch_width_belongs_to_tensor_batch_and_is_bounded(variant, width) -> None:
    with pytest.raises(ValueError, match="batch_width"):
        MLXWorkerClient(_spec(), execution_variant=variant, batch_width=width)


def test_tensor_batch_starts_its_worker_with_width_and_plan(monkeypatch) -> None:
    seen = []

    def popen(arguments, **_kwargs):
        seen.append(arguments)
        raise OSError("not started in this test")

    monkeypatch.setattr("ironmule_product.backend.subprocess.Popen", popen)
    client = MLXWorkerClient(_spec(), execution_variant="tensor_batch", batch_width=8, compute_dtype="native")
    with pytest.raises(BackendUnavailable):
        client.start()
    arguments = seen[0]
    assert arguments[arguments.index("--execution-variant") + 1] == "tensor_batch"
    assert arguments[arguments.index("--batch-width") + 1] == "8"
    assert arguments[arguments.index("--compute-dtype") + 1] == "native"


def test_health_names_batched_execution_and_its_plan() -> None:
    backend = types.SimpleNamespace
    assert _execution_label(backend(execution_variant="reference", compute_dtype=None)) == "exact"
    assert _execution_label(backend(execution_variant="reference", compute_dtype="native")) == "exact@native"
    assert _execution_label(backend(execution_variant="tensor_batch", compute_dtype=None)) == "batched"
    assert _execution_label(backend(execution_variant="tensor_batch", compute_dtype="native")) == "batched@native"


@pytest.mark.parametrize("width", ["1", "9"])
def test_serve_refuses_a_width_outside_two_to_eight(width, tmp_path) -> None:
    from ironmule_product import cli

    with pytest.raises(SystemExit):
        cli.serve(["--no-model", "--batch-width", width, "--state-dir", str(tmp_path / "state")])
    assert not (tmp_path / "state").exists(), "refused before any state is touched"


class _Tokenizer:
    eos_token_ids = {EOS}

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        return [1, 2, 3]

    def decode(self, tokens):
        return " ".join(str(token) for token in tokens)


class _Generator:
    """Scripted rows: each prompt's tokens by request order; EOS stops, the limit ends."""

    scripts: list[list[int]] = []
    instances: list["_Generator"] = []
    after_step = None

    def __init__(self, model, *, max_tokens, stop_tokens, completion_batch_size, prefill_batch_size):
        self.stop = {row[0] for row in stop_tokens}
        self.sizes = (completion_batch_size, prefill_batch_size)
        self.rows: dict[int, dict] = {}
        self.removed: list[int] = []
        self.closed = False
        _Generator.instances.append(self)

    def insert(self, prompts, max_tokens):
        uids = [10 + index for index in range(len(prompts))]
        self.rows = {uid: {"script": list(self.scripts[index]), "limit": max_tokens[index], "count": 0}
                     for index, uid in enumerate(uids)}
        return uids

    def remove(self, uids):
        self.removed.extend(uids)
        for uid in uids:
            self.rows.pop(uid)

    def next_generated(self):
        responses = []
        for uid, row in list(self.rows.items()):
            token = row["script"][row["count"]]
            row["count"] += 1
            finish = "stop" if token in self.stop else "length" if row["count"] >= row["limit"] else None
            responses.append(types.SimpleNamespace(uid=uid, token=token, finish_reason=finish))
            if finish:
                self.rows.pop(uid)
        if _Generator.after_step is not None:  # a plain function on the class, not a method
            _Generator.after_step()
        return responses

    def close(self):
        self.closed = True


@pytest.fixture()
def harness(monkeypatch):
    frames: list[dict] = []
    monkeypatch.setattr(worker, "_emit", frames.append)
    monkeypatch.setitem(sys.modules, "mlx_lm.generate", types.SimpleNamespace(BatchGenerator=_Generator))
    _Generator.instances.clear()
    _Generator.after_step = None
    state = types.SimpleNamespace(frames=frames, cancellations={}, pending=set(), lock=threading.Lock())

    def run(requests, scripts, *, width=8, variant="tensor_batch"):
        _Generator.scripts = scripts
        command = {"type": "generate_batch", "batch_id": "batch-1", "variant": variant,
                   "requests": [request.as_dict() for request in requests]}
        worker._run_tensor_batch(command, "local/model", object(), _Tokenizer(), state.cancellations,
                                 state.pending, state.lock, width)
        return frames

    state.run = run
    return state


def test_rows_stop_at_eos_or_limit_like_the_reference_path(harness) -> None:
    frames = harness.run([_request("a", max_tokens=8), _request("b", max_tokens=2)], [[5, 6, EOS], [7, 8, 9]])
    generator = _Generator.instances[0]
    assert generator.sizes == (2, 2) and generator.closed
    by_request = {rid: [f for f in frames if f.get("request_id") == rid] for rid in ("a", "b")}
    a_tokens = [f for f in by_request["a"] if f["type"] == "token"]
    assert [f["token_id"] for f in a_tokens] == [5, 6, EOS], "the EOS token is counted, as stream_generate does"
    assert [f["text"] for f in a_tokens] == ["", "", "5 6"], "text once, without the EOS token"
    assert [f["completion_tokens"] for f in a_tokens] == [1, 2, 3]
    done_a, done_b = by_request["a"][-1], by_request["b"][-1]
    assert (done_a["finish_reason"], done_a["completion_tokens"]) == ("stop", 3)
    assert (done_b["finish_reason"], done_b["completion_tokens"]) == ("length", 2)
    assert [f["text"] for f in by_request["b"] if f["type"] == "token"] == ["", "7 8"]
    assert done_a["variant"] == done_b["variant"] == "tensor_batch"
    assert done_a["engine"] == {"delivery": "buffered_completion", "batch_rows": 2, "computed_tokens": 3}
    assert frames[-1] == {"type": "batch_done", "batch_id": "batch-1", "request_ids": ["a", "b"]}
    assert harness.cancellations == {}, "every registered cancellation is released"


def test_a_cancelled_row_leaves_the_batch_and_the_others_finish(harness) -> None:
    def cancel_b_after_first_step():
        harness.cancellations["b"].set()
        _Generator.after_step = None

    _Generator.after_step = cancel_b_after_first_step
    frames = harness.run([_request("a", max_tokens=3), _request("b", max_tokens=3)], [[5, 6, 7], [8, 9, 4]])
    assert _Generator.instances[0].removed == [11]
    done_b = [f for f in frames if f.get("request_id") == "b"]
    assert done_b == [{**done_b[0], "type": "done", "finish_reason": "cancelled", "completion_tokens": 1}]
    assert [f["token_id"] for f in frames if f.get("request_id") == "a" and f["type"] == "token"] == [5, 6, 7]


def test_a_request_cancelled_before_dispatch_never_runs_a_step(harness) -> None:
    harness.pending.add("b")
    frames = harness.run([_request("a", max_tokens=1), _request("b", max_tokens=1)], [[5], [6]])
    assert _Generator.instances[0].removed == [11]
    assert [f["finish_reason"] for f in frames if f["type"] == "done"] == ["length", "cancelled"]
    assert harness.pending == set()


@pytest.mark.parametrize(("variant", "width", "count"), [("current_engine", 8, 2), ("tensor_batch", 2, 3)])
def test_a_batch_outside_the_variant_or_width_is_refused(harness, variant, width, count) -> None:
    frames = harness.run([_request(str(i)) for i in range(count)], [[5]] * count, width=width, variant=variant)
    assert frames == [{"type": "error", "code": "invalid_request", "message": "grouped engine backend unavailable",
                       "batch_id": "batch-1"}]
    assert _Generator.instances == [], "refused before any generator exists"
