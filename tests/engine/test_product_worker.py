"""Process/protocol tests for the product worker; no fake inference is used."""

from __future__ import annotations

import json
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
    request = GenerationRequest("local/test", (("user", "hello"),))
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
    """Attach a real subprocess that exercises only the JSON pipe transport."""
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", body],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    client = MLXWorkerClient(_spec(tmp_path), startup_timeout=1)
    client._process = process  # noqa: SLF001 - transport fixture, no model path
    client._ready_payload = {
        "type": "ready", "protocol_version": 1, "model_id": "local/test",
        "revision": "revision", "device": "gpu", "stop_handling": "parent", "context_limit": 8192,
    }
    return client


def _request(request_id: str = "request-1", *, max_tokens: int = 2) -> GenerationRequest:
    return GenerationRequest("local/test", (("user", "hello"),), max_tokens=max_tokens, request_id=request_id)


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
