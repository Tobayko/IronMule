"""Deadline-policy tests; these exercise scheduling and real pipe transport, not inference."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import threading

import pytest

from ironmule_product.backend import MLXWorkerClient
from ironmule_product.errors import RequestCancelled, StateError
from ironmule_product.service import ProductService
from ironmule_product.state import ProductStore
from ironmule_product.types import GenerationRequest, ModelSpec


def _spec(tmp_path: Path) -> ModelSpec:
    return ModelSpec("local/test", "revision", str(tmp_path), 1)


def _request(request_id: str = "request-1") -> GenerationRequest:
    return GenerationRequest(
        "local/test", (("user", "hello"),), max_tokens=1, request_id=request_id
    )


def _transport(tmp_path: Path, body: str, *, startup_timeout=None) -> MLXWorkerClient:
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", body],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    client = MLXWorkerClient(_spec(tmp_path), startup_timeout=startup_timeout)
    client._process = process  # noqa: SLF001 - real protocol transport fixture
    client._ready_payload = {  # noqa: SLF001
        "type": "ready",
        "protocol_version": 1,
        "model_id": "local/test",
        "revision": "revision",
        "device": "gpu",
        "stop_handling": "parent",
        "context_limit": 8192,
    }
    return client


def test_state_accepts_explicit_unbounded_request_policy_and_positive_slo(tmp_path: Path) -> None:
    store = ProductStore(tmp_path / "state")
    store.setup()

    assert store.set_request_timeout(None)["request_timeout_s"] is None
    assert store.settings()["request_timeout_s"] is None
    assert store.set_request_timeout(7)["request_timeout_s"] == 7
    with pytest.raises(StateError):
        store.set_request_timeout(0)
    with pytest.raises(StateError):
        store.set_request_timeout(True)


def test_transport_none_timeout_has_no_replacement_generation_cap(tmp_path: Path) -> None:
    body = (
        "import json,sys,time; command=json.loads(sys.stdin.buffer.readline()); time.sleep(.15); "
        "print(json.dumps({'type':'done','request_id':command['request_id'],"
        "'finish_reason':'length','prompt_tokens':1,'completion_tokens':0,"
        "'metrics':{},'variant':'reference'}),flush=True)"
    )
    client = _transport(tmp_path, body, startup_timeout=0.01)
    try:
        assert list(client.stream(_request(), timeout=None))[-1]["type"] == "done"
    finally:
        client.close()


def test_transport_none_timeout_remains_cancel_responsive(tmp_path: Path) -> None:
    body = (
        "import json,sys; command=json.loads(sys.stdin.buffer.readline()); "
        "json.loads(sys.stdin.buffer.readline()); "
        "print(json.dumps({'type':'done','request_id':command['request_id'],"
        "'finish_reason':'cancelled','prompt_tokens':0,'completion_tokens':0,"
        "'metrics':{},'variant':'reference'}),flush=True)"
    )
    client = _transport(tmp_path, body)
    cancel = threading.Event()
    timer = threading.Timer(0.05, cancel.set)
    try:
        timer.start()
        events = list(client.stream(_request(), cancel=cancel, timeout=None))
        assert events[-1]["finish_reason"] == "cancelled"
    finally:
        timer.cancel()
        client.close()


class _Store:
    def __init__(self, spec: ModelSpec, timeout: int | None):
        self.spec = spec
        self.timeout = timeout

    def settings(self):
        return {"mode": "desktop", "max_pending": 1, "request_timeout_s": self.timeout}

    def model(self, _model_id):
        return self.spec

    def optimization_status(self):
        return {"stage": "inactive", "engine_started": False, "activation_allowed": False}


class _BlockingBackend:
    ready = True

    def __init__(self):
        self.observed_timeout = "unset"

    def stream(self, request, cancel, timeout):
        self.observed_timeout = timeout
        while not cancel.wait(0.01):
            pass
        yield {
            "type": "done",
            "request_id": request.request_id,
            "finish_reason": "cancelled",
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    def close(self):
        pass


def test_service_propagates_unbounded_policy_and_still_cancels(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    backend = _BlockingBackend()
    service = ProductService(_Store(spec, None), backend=backend, spec=spec)
    events = service.stream(_request())
    timer = threading.Timer(0.05, lambda: service.cancel("request-1"))
    try:
        timer.start()
        with pytest.raises(RequestCancelled):
            list(events)
        assert backend.observed_timeout is None
    finally:
        timer.cancel()
        service.close()
