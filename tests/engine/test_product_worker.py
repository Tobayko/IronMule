"""Process/protocol tests for the product worker; no fake inference is used."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

import ironmule_product.backend as backend_module
from ironmule_product.backend import MLXWorkerClient
from ironmule_product.errors import BackendUnavailable, InvalidRequest, RequestTimeout
from ironmule_product.types import GenerationRequest, ModelSpec


ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "ironmule_product" / "worker.py"


def _spec(tmp_path: Path) -> ModelSpec:
    return ModelSpec("local/test", "revision", str(tmp_path), 1)


def test_parent_rejects_wrong_model_without_starting_worker(tmp_path: Path) -> None:
    client = MLXWorkerClient(_spec(tmp_path))
    request = GenerationRequest("other/model", (("user", "hello"),))
    with pytest.raises(InvalidRequest):
        list(client.stream(request))
    assert not client.ready


def test_unavailable_device_or_missing_snapshot_is_clean_backend_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # An empty snapshot is rejected before any device operation. This checks
    # the real worker's metadata failure, not GPU availability.
    client = MLXWorkerClient(_spec(tmp_path), startup_timeout=2)
    launched: list[list[str]] = []
    real_popen = backend_module.subprocess.Popen

    def recording_popen(args, *popen_args, **popen_kwargs):
        launched.append(list(args))
        return real_popen(args, *popen_args, **popen_kwargs)

    monkeypatch.setattr(backend_module.subprocess, "Popen", recording_popen)
    with pytest.raises(BackendUnavailable):
        client.start()
    assert not client.ready
    assert launched and launched[0][1:3] == ["-I", "-u"]


def test_worker_emits_bounded_protocol_error_for_malformed_input() -> None:
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-u", str(WORKER), "--spec", "{}"],
        capture_output=True,
        timeout=2,
        check=False,
    )
    assert result.returncode == 1, result.stderr.decode(errors="replace")
    event = json.loads(result.stdout)
    assert event["type"] == "error"
    assert event["code"] == "backend_unavailable"
    assert "prompt" not in json.dumps(event).lower()
    assert b"GenericAlias" not in result.stderr


def test_timeout_marks_client_unusable_without_restart(tmp_path: Path) -> None:
    client = MLXWorkerClient(_spec(tmp_path / "missing"), startup_timeout=0.1)
    with pytest.raises(BackendUnavailable):
        client.start()
    with pytest.raises(BackendUnavailable):
        client.start()
    assert not client.ready


def _attached_transport(tmp_path: Path, body: str) -> MLXWorkerClient:
    """Attach a real subprocess that exercises only the JSON pipe transport.

    Children in this file start with ``-S`` like the product worker's ``-I -S``: a host
    ``sitecustomize`` (Debian's writes to stderr on Kaggle) must not leak into
    stderr-exact assertions or slow the start past a sub-second timeout.
    """
    process = subprocess.Popen(
        [sys.executable, "-S", "-u", "-c", body],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    client = MLXWorkerClient(_spec(tmp_path), startup_timeout=1)
    client._process = process  # noqa: SLF001 - transport fixture, no model path
    client._ready_payload = {
        "type": "ready", "protocol_version": 2, "model_id": "local/test",
        "revision": "revision", "device": "gpu", "stop_handling": "parent", "context_limit": 8192,
    }
    return client


def _request(request_id: str = "request-1", *, max_tokens: int = 2) -> GenerationRequest:
    return GenerationRequest("local/test", (("user", "hello"),), max_tokens=max_tokens, request_id=request_id)


def _cleanup_process(process: subprocess.Popen) -> None:
    try:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
    finally:
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()


def test_transport_preserves_fragmented_and_batched_frames(tmp_path: Path) -> None:
    body = (
        "import json, os, sys, time; "
        "sys.stdin.buffer.readline(); "
        "token=json.dumps({'type':'token','request_id':'request-1','text':'hel','token_id':7,'prompt_tokens':3,'completion_tokens':1},separators=(',',':')).encode()+b'\\n'; "
        "done=json.dumps({'type':'done','request_id':'request-1','finish_reason':'length','prompt_tokens':3,'completion_tokens':2,'metrics':{},'variant':'reference'},separators=(',',':')).encode()+b'\\n'; "
        "os.write(1,token[:2]); time.sleep(.02); os.write(1,token[2:]+done)"
    )
    client = _attached_transport(tmp_path, body)
    try:
        events = list(client.stream(_request()))
    finally:
        client.close()
    assert [event["type"] for event in events] == ["token", "done"]
    assert events[0]["text"] == "hel"


def test_transport_deadline_does_not_wait_for_a_partial_frame(tmp_path: Path) -> None:
    client = _attached_transport(tmp_path, "import os, sys, time; sys.stdin.buffer.readline(); os.write(1, b'{'); time.sleep(1)")
    try:
        with pytest.raises(RequestTimeout):
            list(client.stream(_request(), timeout=.05))
    finally:
        client.close()
    assert not client.ready


def test_transport_rejects_oversized_unterminated_frame(tmp_path: Path) -> None:
    body = "import os, sys; sys.stdin.buffer.readline(); os.write(1, b'x' * (1024 * 1024 + 1))"
    client = _attached_transport(tmp_path, body)
    try:
        with pytest.raises(BackendUnavailable):
            list(client.stream(_request()))
    finally:
        client.close()
    assert not client.ready


def test_transport_write_backpressure_obeys_absolute_deadline(tmp_path: Path) -> None:
    client = _attached_transport(tmp_path, "import time; time.sleep(1)")
    try:
        with pytest.raises(RequestTimeout):
            client._send({"type": "bulk", "payload": "x" * 900_000}, time.monotonic() + .05)  # noqa: SLF001
    finally:
        client.close()
    assert not client.ready


def test_cancellation_is_polled_while_waiting_and_finishes_once(tmp_path: Path) -> None:
    body = (
        "import json, select, sys; sys.stdin.buffer.readline(); "
        "readable,_,_=select.select([sys.stdin],[],[],1); "
        "json.loads(sys.stdin.buffer.readline()); "
        "sys.stdout.write(json.dumps({'type':'done','request_id':'request-1','finish_reason':'cancelled','prompt_tokens':0,'completion_tokens':0,'variant':'reference'})+'\\n'); sys.stdout.flush()"
    )
    client = _attached_transport(tmp_path, body)
    cancel = threading.Event()
    timer = threading.Timer(.03, cancel.set)
    try:
        timer.start()
        events = list(client.stream(_request(max_tokens=1), cancel=cancel, timeout=.5))
    finally:
        timer.cancel()
        client.close()
    assert events == [{"type": "done", "request_id": "request-1", "finish_reason": "cancelled", "prompt_tokens": 0, "completion_tokens": 0, "variant": "reference"}]


def test_none_generation_id_is_rejected(tmp_path: Path) -> None:
    client = MLXWorkerClient(_spec(tmp_path))
    request = GenerationRequest("local/test", (("user", "hello"),), request_id=None)  # type: ignore[arg-type]
    with pytest.raises(InvalidRequest):
        list(client.stream(request))


def test_worker_invalid_request_does_not_discard_warm_transport(tmp_path: Path) -> None:
    body = (
        "import json, sys; command=json.loads(sys.stdin.buffer.readline()); "
        "print(json.dumps({'type':'error','code':'invalid_request','request_id':command['request_id']}), flush=True); "
        "sys.stdin.buffer.readline()"
    )
    client = _attached_transport(tmp_path, body)
    try:
        with pytest.raises(InvalidRequest):
            list(client.stream(_request()))
        assert client.ready
    finally:
        client.close()


def test_private_variant_and_forward_trace_round_trip(tmp_path: Path) -> None:
    body = (
        "import json, sys; command=json.loads(sys.stdin.buffer.readline()); "
        "assert command['variant']=='bounded_prefetch' and command['trace_forwards'] is True; "
        "print(json.dumps({'type':'done','request_id':command['request_id'],'finish_reason':'length','prompt_tokens':0,'completion_tokens':0,'variant':'bounded_prefetch','model_forward_invocations':8}), flush=True)"
    )
    client = _attached_transport(tmp_path, body)
    try:
        events = list(client.stream(_request(max_tokens=1), variant="bounded_prefetch", trace_forwards=True))
    finally:
        client.close()
    assert events[-1]["variant"] == "bounded_prefetch"
    assert events[-1]["model_forward_invocations"] == 8


@pytest.mark.parametrize("value", [float("inf"), float("nan"), -1, 0, True])
def test_transport_rejects_nonfinite_or_invalid_deadlines(tmp_path, value):
    with pytest.raises(ValueError):
        MLXWorkerClient(_spec(tmp_path), startup_timeout=value)
    client = MLXWorkerClient(_spec(tmp_path))
    with pytest.raises(ValueError):
        client.stream(_request(), timeout=value)
    assert client._process is None


def test_buffered_transport_frame_does_not_bypass_deadline(tmp_path):
    client = _attached_transport(tmp_path, "import time; time.sleep(1)")
    client._stdout_buffer.extend(b'{"type":"control"}\n')
    try:
        with pytest.raises(RequestTimeout):
            client._read_event(time.monotonic() - 1)
    finally:
        client.close()


def test_buffered_transport_frame_still_polls_cancellation(tmp_path):
    client = _attached_transport(tmp_path, "import time; time.sleep(1)")
    client._stdout_buffer.extend(b'{"type":"control"}\n')
    cancel = threading.Event()
    cancel.set()
    observed = []
    try:
        event = client._read_event(time.monotonic() + 1, cancel=cancel,
                                  on_cancel=lambda: observed.append(True))
        assert event == {"type": "control"}
        assert observed == [True]
    finally:
        client.close()


def test_protocol_size_rejection_keeps_worker_unstarted(tmp_path):
    client = MLXWorkerClient(_spec(tmp_path))
    # Legal text length can expand in JSON; framing must reject it as input,
    # not turn it into a model/Metal failure.
    request = GenerationRequest("local/test", (("user", "\x00" * 200_000),))
    with pytest.raises(InvalidRequest):
        client.stream(request)
    assert client._process is None
    assert client._usable is True


def test_receiver_shutdown_returns_without_daemon_finalization_abort(tmp_path: Path) -> None:
    code = (
        "import queue, threading; from ironmule_product.worker import _receiver; "
        "q=queue.Queue(maxsize=8); c={}; p=set(); lock=threading.Lock(); "
        "t=threading.Thread(target=_receiver,args=(q,c,p,lock),daemon=True); t.start(); "
        "assert q.get()['type']=='shutdown'"
    )
    process = subprocess.Popen(
        [sys.executable, "-S", "-u", "-c", code], cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT)}, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        assert process.stdin is not None and process.stderr is not None
        process.stdin.write(b'{"type":"shutdown"}\n')
        process.stdin.flush()
        assert process.wait(timeout=3) == 0
        assert process.stderr.read() == b""
    finally:
        _cleanup_process(process)


def test_receiver_full_queue_shutdown_is_bounded(tmp_path: Path) -> None:
    code = (
        "import queue, threading; from ironmule_product.worker import _receiver; "
        "q=queue.Queue(maxsize=1); q.put({'type':'work'}); c={}; p=set(); lock=threading.Lock(); "
        "t=threading.Thread(target=_receiver,args=(q,c,p,lock),daemon=True); t.start(); "
        "t.join(timeout=2); assert not t.is_alive(); assert q.get_nowait()['type']=='shutdown'"
    )
    process = subprocess.Popen(
        [sys.executable, "-S", "-u", "-c", code], cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT)}, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        assert process.stdin is not None and process.stderr is not None
        process.stdin.write(b'{"type":"shutdown"}\n')
        process.stdin.flush()
        assert process.wait(timeout=3) == 0
        assert process.stderr.read() == b""
    finally:
        _cleanup_process(process)


def test_startup_guard_exception_is_preserved_and_child_is_reaped(tmp_path, monkeypatch):
    processes = []
    real_popen = backend_module.subprocess.Popen

    def fake_popen(*args, **kwargs):
        process = real_popen(
            [sys.executable, "-S", "-u", "-c", "import time; time.sleep(5)"],
            stdin=kwargs.get("stdin"), stdout=kwargs.get("stdout"), stderr=kwargs.get("stderr"),
            env=kwargs.get("env"),
        )
        processes.append(process)
        return process

    monkeypatch.setattr(backend_module.subprocess, "Popen", fake_popen)
    client = MLXWorkerClient(_spec(tmp_path), startup_timeout=1)

    class GuardFailure(KeyboardInterrupt):
        pass

    with pytest.raises(GuardFailure):
        client.start(startup_guard=lambda _pid: (_ for _ in ()).throw(GuardFailure()))
    assert processes and processes[0].poll() is not None
    assert not client.ready


def test_startup_guard_is_polled_until_timeout_and_child_is_reaped(tmp_path, monkeypatch):
    processes = []
    calls = []
    real_popen = backend_module.subprocess.Popen

    def fake_popen(*args, **kwargs):
        process = real_popen(
            [sys.executable, "-S", "-u", "-c", "import time; time.sleep(5)"],
            stdin=kwargs.get("stdin"), stdout=kwargs.get("stdout"), stderr=kwargs.get("stderr"),
            env=kwargs.get("env"),
        )
        processes.append(process)
        return process

    monkeypatch.setattr(backend_module.subprocess, "Popen", fake_popen)
    client = MLXWorkerClient(_spec(tmp_path), startup_timeout=.12)
    with pytest.raises(BackendUnavailable):
        client.start(startup_guard=lambda pid: calls.append(pid))
    assert processes and processes[0].poll() is not None
    assert len(calls) >= 2 and all(pid == processes[0].pid for pid in calls)


def test_slow_startup_guard_cannot_bypass_absolute_deadline(tmp_path, monkeypatch):
    processes = []
    real_popen = backend_module.subprocess.Popen
    spec = _spec(tmp_path)

    def fake_popen(*args, **kwargs):
        body = ("import json, sys, time; print(json.dumps({'type':'ready','protocol_version':2,"
                f"'model_id':{spec.model_id!r},'revision':{spec.revision!r},"
                "'device':'gpu','stop_handling':'parent'}), flush=True); time.sleep(5)")
        process = real_popen(
            [sys.executable, "-S", "-u", "-c", body],
            stdin=kwargs.get("stdin"), stdout=kwargs.get("stdout"), stderr=kwargs.get("stderr"),
            env=kwargs.get("env"),
        )
        processes.append(process)
        return process

    monkeypatch.setattr(backend_module.subprocess, "Popen", fake_popen)
    client = MLXWorkerClient(spec, startup_timeout=.05)

    calls = []

    def slow_guard(_pid):
        calls.append(True)
        if len(calls) == 2:
            time.sleep(.1)

    with pytest.raises(BackendUnavailable):
        client.start(startup_guard=slow_guard)
    assert processes and processes[0].poll() is not None
    assert len(calls) == 2
    assert not client.ready


@pytest.mark.parametrize("event", [
    {"type": "ready", "startup_wall_seconds": float("nan")},
    {"type": "ready", "startup_wall_seconds": 1},
    {"type": "ready", "startup_wall_seconds": -1, "process_peak_rss_bytes": 1,
     "mlx_active_bytes": 1, "mlx_peak_bytes": 1, "mlx_cache_bytes": 1,
     "recommended_working_set_bytes": 1},
    {"type": "ready", "startup_wall_seconds": 1, "process_peak_rss_bytes": 1.0,
     "mlx_active_bytes": 1, "mlx_peak_bytes": 1, "mlx_cache_bytes": 1,
     "recommended_working_set_bytes": 1},
    {"type": "ready", "startup_wall_seconds": 1, "process_peak_rss_bytes": 1,
     "mlx_active_bytes": 2, "mlx_peak_bytes": 1, "mlx_cache_bytes": 1,
     "recommended_working_set_bytes": 1},
    {"type": "ready", "startup_wall_seconds": 10**1000, "process_peak_rss_bytes": 1,
     "mlx_active_bytes": 1, "mlx_peak_bytes": 1, "mlx_cache_bytes": 1,
     "recommended_working_set_bytes": 1},
])
def test_startup_telemetry_contract_rejects_invalid_payloads(event):
    with pytest.raises(BackendUnavailable):
        MLXWorkerClient._validate_startup_telemetry(event)


def test_ready_start_with_guard_is_explicitly_rejected(tmp_path):
    client = _attached_transport(tmp_path, "import time; time.sleep(1)")
    observed = []
    try:
        with pytest.raises(ValueError, match="already-ready"):
            client.start(startup_guard=lambda pid: observed.append(pid))
        assert observed == []
        assert client.ready
    finally:
        client.close()


def test_worker_accepts_metal_or_cuda_device_facts() -> None:
    from types import SimpleNamespace

    from ironmule_product.worker import _gpu_available, _working_set_bytes

    def fake(metal: bool, cuda, info: dict):
        return SimpleNamespace(metal=SimpleNamespace(is_available=lambda: metal),
                               cuda=None if cuda is None else SimpleNamespace(is_available=lambda: cuda),
                               device_info=lambda: info)

    metal = fake(True, None, {"max_recommended_working_set_size": 7, "total_memory": 9})
    assert _gpu_available(metal) and _working_set_bytes(metal) == 7
    # MLX's CUDA backend (Kaggle T4, DATA3) has no recommended working set.
    cuda = fake(False, True, {"total_memory": 15_000_000_000})
    assert _gpu_available(cuda) and _working_set_bytes(cuda) == 15_000_000_000
    cpu = fake(False, False, {})
    assert not _gpu_available(cpu)
    # MLX's CPU backend reports no memory; the worker serves there with installed RAM (CPU4).
    from ironmule.hw import installed_memory_bytes
    assert _working_set_bytes(cpu) == installed_memory_bytes() > 0


def test_chat_end_of_turn_marker_stops_generation() -> None:
    from ironmule_product.worker import stop_at_end_of_turn

    class Tokenizer:
        def __init__(self, vocab):
            self.vocab, self.eos_token_ids = vocab, {1}

        def get_vocab(self):
            return self.vocab

        def add_eos_token(self, token):
            self.eos_token_ids.add(self.vocab[token])

    gemma = Tokenizer({"<eos>": 1, "<end_of_turn>": 106, "hello": 7})
    stop_at_end_of_turn(gemma)
    assert gemma.eos_token_ids == {1, 106}
    plain = Tokenizer({"<eos>": 1, "hello": 7})
    stop_at_end_of_turn(plain)
    assert plain.eos_token_ids == {1}


class _StreamTokenizer:
    """mlx-lm's streaming detokenizer contract, one character per token."""

    eos_token_ids = {0}

    class _Detokenizer:
        def __init__(self):
            self.pending = ""

        def reset(self):
            self.pending = ""

        def add_token(self, token):
            self.pending += chr(token)

        def finalize(self):
            pass

        @property
        def last_segment(self):
            segment, self.pending = self.pending, ""
            return segment

    def __init__(self):
        self.detokenizer = self._Detokenizer()


class _StreamBridge:
    def __init__(self, tokens, result_tokens=None):
        self.tokens, self.result_tokens = tokens, result_tokens or tokens

    def generate(self, prompt_ids, max_tokens, on_token=None):
        from types import SimpleNamespace

        for token in self.tokens:
            on_token(token)
        return SimpleNamespace(tokens=list(self.result_tokens), finish_reason="stop", token_count=len(self.result_tokens),
                               text="", metadata={})


def _collect(stop_after=None):
    rows = []

    def consume(response):
        rows.append(response)
        if stop_after is not None and len(rows) == stop_after:
            return {"finish_reason": "cancelled"}
        return {"finish_reason": response.finish_reason} if response.finish_reason else None
    return rows, consume


def test_engine_stream_hands_each_token_on_with_text_and_finishes_on_the_last():
    from ironmule_product import worker

    holder, (rows, consume) = {}, _collect()
    terminal = worker._engine_stream(_StreamBridge([72, 105, 0]), _StreamTokenizer(), [1, 2], 8, 2, holder, consume)
    assert terminal == {"finish_reason": "stop"}
    assert [row.token for row in rows] == [72, 105, 0]
    assert "".join(row.text for row in rows) == "Hi"
    assert [row.generation_tokens for row in rows] == [1, 2, 3]
    assert [row.finish_reason for row in rows] == [None, None, "stop"]
    assert holder["result"].tokens == [72, 105, 0]



def test_engine_stream_sends_each_token_before_the_engine_reads_the_next():
    """KP1 run 1: holding each token for the next delayed the first by a whole chained readback (56 ms at
    readback 4). The token that ends the answer, EOS or the limit, still carries the finish reason."""
    from types import SimpleNamespace

    from ironmule_product import worker

    for tokens, limit, finish in (([72, 105, 0], 8, "stop"), ([72, 105], 2, "length")):
        rows, consume = _collect()
        consumed = []

        class Bridge:
            def generate(self, prompt_ids, max_tokens, on_token=None):
                for token in tokens:
                    on_token(token)
                    consumed.append(len(rows))  # frames handed on by the time the engine reads the next
                return SimpleNamespace(tokens=list(tokens), finish_reason=finish, token_count=len(tokens), text="",
                                       metadata={})

        terminal = worker._engine_stream(Bridge(), _StreamTokenizer(), [1, 2], limit, 2, {}, consume)
        assert consumed == list(range(1, len(tokens) + 1)) and terminal == {"finish_reason": finish}
        assert [row.finish_reason for row in rows] == [None] * (len(tokens) - 1) + [finish]
        assert "".join(row.text for row in rows) == "Hi"

def test_engine_stream_stops_sending_after_a_cancellation():
    from ironmule_product import worker

    rows, consume = _collect(stop_after=1)
    terminal = worker._engine_stream(_StreamBridge([72, 105, 33, 0]), _StreamTokenizer(), [1], 8, 1, {}, consume)
    assert terminal == {"finish_reason": "cancelled"} and [row.token for row in rows] == [72]


def test_engine_stream_refuses_tokens_that_differ_from_the_result():
    from ironmule_product import worker

    rows, consume = _collect()
    with pytest.raises(RuntimeError, match="differ"):
        worker._engine_stream(_StreamBridge([72, 105], [72, 106]), _StreamTokenizer(), [1], 8, 1, {}, consume)


def test_the_streaming_detokenizer_is_built_once_and_reset_per_request():
    from ironmule_product import worker

    built = []

    class Detokenizer:
        def __init__(self):
            built.append(self)
            self.text = ""

        def reset(self):
            self.text = ""

    class Tokenizer:
        @property
        def detokenizer(self):
            return Detokenizer()

    tokenizer = Tokenizer()
    worker.reuse_detokenizer(tokenizer)
    first = tokenizer.detokenizer
    first.text = "left over"
    second = tokenizer.detokenizer
    assert second is first and second.text == "" and len(built) == 1
    assert isinstance(tokenizer, Tokenizer)


def test_a_grouped_stream_sends_tokens_interleaved_as_the_engine_reads_them(monkeypatch):
    import threading
    from types import SimpleNamespace

    from ironmule_product import worker

    class Tokenizer:
        eos_token_ids = {0}

        @property
        def detokenizer(self):
            return _StreamTokenizer._Detokenizer()

        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            return [1, 2, 3]

    scripts = {0: [72, 105, 0], 1: [79, 107, 0]}

    class Bridge:
        def metadata(self):
            return {"configuration": "current_profile"}

        def generate_many(self, ids, limits, on_tokens=None, admit=None, cancelled=None):
            assert admit(64) == []  # nothing joined
            for step in range(3):  # the group interleaves its requests' steps
                for index, emit in enumerate(on_tokens):
                    emit(scripts[index][step])
            return [SimpleNamespace(tokens=scripts[i], text="", finish_reason="stop", token_count=3, metadata={})
                    for i in range(2)]

    sent = []
    monkeypatch.setattr(worker, "_emit", sent.append)
    command = {"type": "generate_batch", "batch_id": "batch-1", "variant": "current_engine",
               "requests": [{"request_id": rid, "model": "m", "messages": [{"role": "user", "content": "x"}],
                             "max_tokens": 8, "stream": True} for rid in ("a", "b")]}
    worker._run_engine_batch(command, "m", Tokenizer(), Bridge(), {}, set(), threading.Lock())
    tokens = [(event["request_id"], event["token_id"]) for event in sent if event["type"] == "token"]
    assert tokens == [("a", 72), ("b", 79), ("a", 105), ("b", 107), ("a", 0), ("b", 0)]
    assert "".join(e["text"] for e in sent if e.get("request_id") == "a" and e["type"] == "token") == "Hi"
    done = [event for event in sent if event["type"] == "done"]
    assert [d["request_id"] for d in done] == ["a", "b"] and all(d["engine"]["delivery"] == "streamed" for d in done)
    assert sent[-1]["type"] == "batch_done"


def test_no_thinking_reaches_the_chat_template_only_when_set(monkeypatch):
    from ironmule_product import worker

    seen = []

    class Tokenizer:
        def apply_chat_template(self, messages, add_generation_prompt, tokenize, **extra):
            seen.append(extra)
            return [1]

    worker._render(Tokenizer(), [{"role": "user", "content": "x"}])
    monkeypatch.setattr(worker, "THINKING", False)
    worker._render(Tokenizer(), [{"role": "user", "content": "x"}])
    assert seen == [{}, {"enable_thinking": False}]


def _join_world(monkeypatch, capacity):
    """A grouped batch of two that admits a third request sent while it runs (SERVE8)."""
    import threading
    from types import SimpleNamespace

    from ironmule_product import worker

    built = []

    class Tokenizer:
        eos_token_ids = {0}

        @property
        def detokenizer(self):
            built.append(_StreamTokenizer._Detokenizer())
            return built[-1]

        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            return [1] * len(messages[0]["content"])

    scripts = {"a": [72, 0], "b": [79, 0], "c": [33, 0]}
    calls = []

    class Bridge:
        def metadata(self):
            return {"configuration": "current_profile"}

        def generate_many(self, ids, limits, on_tokens=None, admit=None, cancelled=None):
            calls.append(len(ids))
            names = ["c"] if len(ids) == 1 else ["a", "b"]
            emitters = list(on_tokens)
            worker.JOINS.put({"type": "join_batch", "batch_id": "batch-1", "requests": [
                {"request_id": "c", "model": "m", "messages": [{"role": "user", "content": "xyz"}],
                 "max_tokens": 8, "stream": True}]}) if len(calls) == 1 else None
            joined = admit(capacity)
            names += ["c"] * len(joined)
            emitters += [emit for _, _, emit, _ in joined]
            for step in range(2):
                for name, emit in zip(names, emitters):
                    emit(scripts[name][step])
            return [SimpleNamespace(tokens=scripts[n], text="", finish_reason="stop", token_count=2, metadata={})
                    for n in names]

    sent = []
    monkeypatch.setattr(worker, "_emit", sent.append)
    monkeypatch.setattr(worker, "_DETOKENIZER_POOLS", {})
    command = {"type": "generate_batch", "batch_id": "batch-1", "variant": "current_engine",
               "requests": [{"request_id": rid, "model": "m", "messages": [{"role": "user", "content": "x"}],
                             "max_tokens": 8, "stream": True} for rid in ("a", "b")]}
    worker._run_engine_batch(command, "m", Tokenizer(), Bridge(), {}, set(), threading.Lock())
    return sent, calls, built


def test_a_request_sent_while_a_batch_runs_joins_it(monkeypatch):
    sent, calls, built = _join_world(monkeypatch, capacity=64)
    assert calls == [2]  # one engine run served all three
    tokens = [(e["request_id"], e["token_id"]) for e in sent if e["type"] == "token"]
    assert tokens == [("a", 72), ("b", 79), ("c", 33), ("a", 0), ("b", 0), ("c", 0)]
    assert [e["request_id"] for e in sent if e["type"] == "done"] == ["a", "b", "c"]
    memory = sent[-1].pop("memory")  # MEM2: MLX counters at the group boundary
    assert sent[-1] == {"type": "batch_done", "batch_id": "batch-1", "request_ids": ["a", "b", "c"]}
    assert set(memory) == {"active", "cache", "peak"} and all(type(v) is int for v in memory.values())
    assert len(built) == 3


def test_a_joined_request_too_long_for_the_run_is_served_right_after_it(monkeypatch):
    sent, calls, built = _join_world(monkeypatch, capacity=4)  # 3 prompt + 8 tokens + 8 do not fit 4
    assert calls == [2, 1]
    assert [e["request_id"] for e in sent if e["type"] == "done"] == ["a", "b", "c"]
    assert sent[-1]["request_ids"] == ["a", "b", "c"]
    assert len(built) == 2  # c reused a detokenizer a or b gave back


def test_chat_reuse_reaches_the_engine_in_the_grouped_path(monkeypatch):
    """A lone chat runs as a group under --engine tuned (SERVE8), so --chat-reuse must work there too."""
    import threading
    from types import SimpleNamespace

    from ironmule_product import worker

    class Tokenizer:
        eos_token_ids = {0}

        @property
        def detokenizer(self):
            return _StreamTokenizer._Detokenizer()

        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            ids = []
            for message in messages:
                ids += [{"user": 2, "assistant": 3}[message["role"]], len(message["content"])]
            return ids + ([3] if add_generation_prompt else [])

    seen = {}

    class Bridge:
        def metadata(self):
            return {"configuration": "current_profile"}

        def generate_many(self, ids, limits, on_tokens=None, admit=None, boundaries=None, cancelled=None):
            seen["boundaries"] = boundaries
            on_tokens[0](0)
            return [SimpleNamespace(tokens=[0], text="", finish_reason="stop", token_count=1, metadata={})]

    monkeypatch.setattr(worker, "_emit", lambda event: None)
    monkeypatch.setattr(worker, "CHAT_REUSE", True)
    conversation = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"},
                    {"role": "user", "content": "more"}]
    command = {"type": "generate_batch", "batch_id": "batch-1", "variant": "current_engine",
               "requests": [{"request_id": "a", "model": "m", "messages": conversation, "max_tokens": 8,
                             "stream": True}]}
    worker._run_engine_batch(command, "m", Tokenizer(), Bridge(), {}, set(), threading.Lock())
    assert seen["boundaries"] == [[3]]


@pytest.mark.parametrize("device", ["gpu", "cpu", "tpu", None])
def test_ready_handshake_reports_the_actual_device_and_refuses_unknown_ones(tmp_path, monkeypatch, device):
    """OPS1: a CPU worker says cpu and is served; a missing or unknown device is refused."""
    processes = []
    real_popen = backend_module.subprocess.Popen
    spec = _spec(tmp_path)
    ready = {"type": "ready", "protocol_version": 2, "model_id": spec.model_id, "revision": spec.revision,
             "stop_handling": "parent", **({"device": device} if device else {})}

    def fake_popen(*args, **kwargs):
        body = f"import json, sys, time; print(json.dumps({ready!r}), flush=True); time.sleep(5)"
        process = real_popen([sys.executable, "-S", "-u", "-c", body], stdin=kwargs.get("stdin"),
                             stdout=kwargs.get("stdout"), stderr=kwargs.get("stderr"), env=kwargs.get("env"))
        processes.append(process)
        return process

    monkeypatch.setattr(backend_module.subprocess, "Popen", fake_popen)
    client = MLXWorkerClient(spec, startup_timeout=5)
    try:
        if device in ("gpu", "cpu"):
            client.start()
            assert client.ready and client.device == device
            assert client.ready_payload["device"] == device
        else:
            with pytest.raises(BackendUnavailable):
                client.start()
            assert client.device is None
    finally:
        client.close()
        for process in processes:
            _cleanup_process(process)


def test_a_stream_cancelled_during_a_fallback_rerun_may_end_short_of_what_it_streamed():
    """RSI3-LIVE-P2: a fallback restarts a member; cancelled during the re-run it ends shorter than
    the tokens it already streamed. Its output is never delivered, so a matching prefix passes;
    without a cancellation any difference is still an error."""
    from types import SimpleNamespace

    from ironmule_product import worker

    class Detokenizer:
        last_segment = ""

        def add_token(self, _token):
            pass

        def finalize(self):
            pass

    class Pool:
        def acquire(self):
            return Detokenizer()

        def release(self, _detokenizer):
            pass

    events = {"r": threading.Event()}
    stream = worker._BatchStream("b", "r", 3, 8, Pool(), events, threading.Lock(), SimpleNamespace(eos_token_ids=()))
    for token in (5, 6, 7):
        stream.on_token(token)
    short = SimpleNamespace(tokens=[5], finish_reason="cancelled")
    with pytest.raises(RuntimeError, match="differ"):
        stream.finish(short)
    events["r"].set()
    assert stream.finish(short) >= 0
    with pytest.raises(RuntimeError, match="differ"):
        stream.finish(SimpleNamespace(tokens=[9], finish_reason="cancelled"))


@pytest.mark.parametrize("digest,ok", [("0" * 64, True), ("not-a-digest", False), (5, False)])
def test_ready_handshake_carries_a_valid_source_identity(tmp_path, monkeypatch, digest, ok):
    """RSI3-LIVE (f): the worker reports the digest of every source file it loaded."""
    processes = []
    real_popen = backend_module.subprocess.Popen
    spec = _spec(tmp_path)
    ready = {"type": "ready", "protocol_version": 2, "model_id": spec.model_id, "revision": spec.revision,
             "stop_handling": "parent", "device": "gpu", "sources_sha256": digest}

    def fake_popen(*args, **kwargs):
        body = f"import json, sys, time; print(json.dumps({ready!r}), flush=True); time.sleep(5)"
        process = real_popen([sys.executable, "-S", "-u", "-c", body], stdin=kwargs.get("stdin"),
                             stdout=kwargs.get("stdout"), stderr=kwargs.get("stderr"), env=kwargs.get("env"))
        processes.append(process)
        return process

    monkeypatch.setattr(backend_module.subprocess, "Popen", fake_popen)
    client = MLXWorkerClient(spec, startup_timeout=5)
    try:
        if ok:
            client.start()
            assert client.ready_payload["sources_sha256"] == digest
        else:
            with pytest.raises(BackendUnavailable):
                client.start()
    finally:
        client.close()
        for process in processes:
            _cleanup_process(process)


class _FakeMx:
    """MEM2: just the allocator controls the worker touches."""

    def __init__(self, cuda: bool) -> None:
        self.cuda = type("cuda", (), {"is_available": staticmethod(lambda: cuda)})()
        self.limits: list[int] = []

    def set_cache_limit(self, value: int) -> int:
        self.limits.append(value)
        return 0


def test_the_cuda_cache_bound_is_set_on_cuda_only_and_can_be_turned_off():
    """MEM2-D: on a 15 GiB T4 MLX's default cache (0.95 of the device) held 11 GiB of freed buffers
    beside about 3 GiB that MLX does not count, and the device ran out."""
    from ironmule_product.worker import CUDA_CACHE_LIMIT_ENV, _apply_cuda_cache_limit

    cuda, metal = _FakeMx(True), _FakeMx(False)
    assert _apply_cuda_cache_limit(cuda, {}) == 1024 << 20 and cuda.limits == [1024 << 20]
    assert _apply_cuda_cache_limit(metal, {}) is None and metal.limits == []  # Metal keeps MLX's default
    off = _FakeMx(True)
    assert _apply_cuda_cache_limit(off, {CUDA_CACHE_LIMIT_ENV: "0"}) is None and off.limits == []
    assert _apply_cuda_cache_limit(_FakeMx(True), {CUDA_CACHE_LIMIT_ENV: "256"}) == 256 << 20
    for bad in ("-1", "1.5", "lots"):
        with pytest.raises(ValueError):
            _apply_cuda_cache_limit(_FakeMx(True), {CUDA_CACHE_LIMIT_ENV: bad})



def test_a_grouped_stream_sends_each_token_before_the_engine_reads_the_next(monkeypatch):
    """KP1 run 1, the grouped path: every token leaves as the engine reads it back; the last carries the final
    text, and nothing waits for a token that comes a chained readback later."""
    import threading
    from types import SimpleNamespace

    from ironmule_product import worker

    class Tokenizer:
        eos_token_ids = {0}

        @property
        def detokenizer(self):
            return _StreamTokenizer._Detokenizer()

        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            return [1, 2, 3]

    sent, emitted_by_then = [], []
    monkeypatch.setattr(worker, "_emit", sent.append)
    monkeypatch.setattr(worker, "_DETOKENIZER_POOLS", {})

    class Bridge:
        def metadata(self):
            return {"configuration": "current_profile"}

        def generate_many(self, ids, limits, on_tokens=None, admit=None, cancelled=None):
            for token in (72, 105, 0):
                on_tokens[0](token)
                emitted_by_then.append(sum(event["type"] == "token" for event in sent))
            return [SimpleNamespace(tokens=[72, 105, 0], text="", finish_reason="stop", token_count=3, metadata={})]

    command = {"type": "generate_batch", "batch_id": "batch-1", "variant": "current_engine",
               "requests": [{"request_id": "a", "model": "m", "messages": [{"role": "user", "content": "x"}],
                             "max_tokens": 8, "stream": True}]}
    worker._run_engine_batch(command, "m", Tokenizer(), Bridge(), {}, set(), threading.Lock())
    assert emitted_by_then == [1, 2, 3]
    tokens = [event for event in sent if event["type"] == "token"]
    assert "".join(event["text"] for event in tokens) == "Hi" and [e["completion_tokens"] for e in tokens] == [1, 2, 3]
    done = [event for event in sent if event["type"] == "done"]
    assert done[0]["completion_tokens"] == 3 and done[0]["finish_reason"] == "stop"


def test_a_grouped_member_is_done_with_its_last_token_not_when_the_run_ends(monkeypatch):
    """H9: a member's done follows its last token at once, so its caller is answered then and the service sees
    room in the group; the streamed tokens are still checked against the run's result afterwards."""
    import threading
    from types import SimpleNamespace

    from ironmule_product import worker

    class Tokenizer:
        eos_token_ids = {0}

        @property
        def detokenizer(self):
            return _StreamTokenizer._Detokenizer()

        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            return [1, 2, 3]

    sent = []
    monkeypatch.setattr(worker, "_emit", sent.append)
    monkeypatch.setattr(worker, "_DETOKENIZER_POOLS", {})
    scripts = {0: [72, 0], 1: [79, 107, 33, 0]}  # a ends at its second token, b runs on

    class Bridge:
        def metadata(self):
            return {"configuration": "current_profile"}

        def eos_ids(self):
            return (0,)

        def generate_many(self, ids, limits, on_tokens=None, admit=None, cancelled=None):
            for step in range(4):
                for index, emit in enumerate(on_tokens):
                    if step < len(scripts[index]):
                        emit(scripts[index][step])
            return [SimpleNamespace(tokens=scripts[i], text="", finish_reason="stop", token_count=len(scripts[i]),
                                    metadata={}) for i in range(2)]

    command = {"type": "generate_batch", "batch_id": "batch-1", "variant": "current_engine",
               "requests": [{"request_id": rid, "model": "m", "messages": [{"role": "user", "content": "x"}],
                             "max_tokens": 8, "stream": True} for rid in ("a", "b")]}
    worker._run_engine_batch(command, "m", Tokenizer(), Bridge(), {}, set(), threading.Lock())
    order = [(event["type"], event.get("request_id")) for event in sent]
    assert order.index(("done", "a")) < [i for i, e in enumerate(order) if e == ("token", "b")][2]
    done = {event["request_id"]: event for event in sent if event["type"] == "done"}
    assert sorted(done) == ["a", "b"] and len([e for e in sent if e["type"] == "done"]) == 2  # once each
    assert done["a"]["completion_tokens"] == 2 and done["b"]["completion_tokens"] == 4
    assert done["a"]["engine"]["delivery"] == "streamed" and sent[-1]["type"] == "batch_done"


def test_a_join_left_over_from_an_ended_batch_is_dropped_and_the_next_batch_runs(monkeypatch):
    """A join that reached the worker after its own batch ended stays queued; the next batch drops it (the client
    handed that request back) instead of failing as a join for another batch."""
    import queue
    import threading
    from types import SimpleNamespace

    from ironmule_product import worker

    class Tokenizer:
        eos_token_ids = {0}

        @property
        def detokenizer(self):
            return _StreamTokenizer._Detokenizer()

        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            return [1, 2, 3]

    sent = []
    monkeypatch.setattr(worker, "_emit", sent.append)
    monkeypatch.setattr(worker, "_DETOKENIZER_POOLS", {})
    monkeypatch.setattr(worker, "JOINS", queue.Queue())
    worker.JOINS.put({"type": "join_batch", "batch_id": "batch-0", "requests": [
        {"request_id": "late", "model": "m", "messages": [{"role": "user", "content": "x"}], "max_tokens": 4,
         "stream": True}]})

    class Bridge:
        def metadata(self):
            return {"configuration": "current_profile"}

        def generate_many(self, ids, limits, on_tokens=None, admit=None, cancelled=None):
            assert admit(64) == []  # the stale join is not admitted
            for token in (72, 0):
                on_tokens[0](token)
            return [SimpleNamespace(tokens=[72, 0], text="", finish_reason="stop", token_count=2, metadata={})]

    command = {"type": "generate_batch", "batch_id": "batch-1", "variant": "current_engine",
               "requests": [{"request_id": "a", "model": "m", "messages": [{"role": "user", "content": "x"}],
                             "max_tokens": 8, "stream": True}]}
    worker._run_engine_batch(command, "m", Tokenizer(), Bridge(), {}, set(), threading.Lock())
    assert not [event for event in sent if event["type"] == "error"], sent
    assert sent[-1] == {**sent[-1], "type": "batch_done", "request_ids": ["a"]}


def test_a_grouped_request_beyond_the_context_limit_is_rejected_alone_on_either_batch_path(monkeypatch):
    """Protocol 2: a prompt plus output limit beyond the context limit failed its whole group, and the client
    retired the worker over it. The worker now rejects that request alone, before any token, on the engine path
    and the automatic one, and names it in batch_done in the order it came."""
    import queue
    import threading
    from types import SimpleNamespace

    from ironmule_product import worker

    class Tokenizer:
        eos_token_ids = {0}

        @property
        def detokenizer(self):
            return _StreamTokenizer._Detokenizer()

        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            return [1] * len(messages[0]["content"])

    def chat(rid, length, max_tokens=8):
        return {"request_id": rid, "model": "m", "messages": [{"role": "user", "content": "x" * length}],
                "max_tokens": max_tokens, "stream": True}

    class Bridge:
        def metadata(self):
            return {"configuration": "current_profile"}

        def generate_many(self, ids, limits, on_tokens=None, admit=None, cancelled=None):
            assert len(ids) == 2  # the over-long request never reaches the engine
            for stream in on_tokens:
                for token in (72, 0):
                    stream(token)
            return [SimpleNamespace(tokens=[72, 0], text="", finish_reason="stop", token_count=2, metadata={})
                    for _ in ids]

    sent = []
    monkeypatch.setattr(worker, "_emit", sent.append)
    monkeypatch.setattr(worker, "_DETOKENIZER_POOLS", {})
    monkeypatch.setattr(worker, "JOINS", queue.Queue())
    command = {"type": "generate_batch", "batch_id": "batch-1", "variant": "current_engine",
               "requests": [chat("a", 3), chat("b", worker.CONTEXT_LIMIT), chat("c", 3)]}
    worker._run_engine_batch(command, "m", Tokenizer(), Bridge(), {}, set(), threading.Lock())
    assert not [event for event in sent if event["type"] == "error"], sent
    assert [(e["request_id"], e["reason"], e["limit"]) for e in sent if e["type"] == "rejected"] == [
        ("b", "context_limit", worker.CONTEXT_LIMIT)]
    assert sorted(e["request_id"] for e in sent if e["type"] == "done") == ["a", "c"]
    assert sent[-1] == {**sent[-1], "type": "batch_done", "request_ids": ["a", "b", "c"]}

    sent.clear()
    runtime = SimpleNamespace(choose_many=lambda ids, limits, **_: SimpleNamespace(candidate_id="reference"),
                              last_decision={})

    def stream_generate(model, tokenizer, prompt_ids, max_tokens):
        yield SimpleNamespace(generation_tokens=1, token=72, text="h", finish_reason="stop")

    command = {"type": "generate_batch", "batch_id": "batch-2", "variant": "automatic",
               "requests": [{**chat("a", 3), "stream": False}, {**chat("b", worker.CONTEXT_LIMIT), "stream": False}]}
    worker._run_automatic_batch(command, "m", None, Tokenizer(), stream_generate, runtime, {}, set(),
                                threading.Lock())
    assert not [event for event in sent if event["type"] == "error"], sent
    assert [e["request_id"] for e in sent if e["type"] == "rejected"] == ["b"]
    assert [e["request_id"] for e in sent if e["type"] == "done"] == ["a"]
    assert sent[-1] == {"type": "batch_done", "batch_id": "batch-2", "request_ids": ["a", "b"]}
