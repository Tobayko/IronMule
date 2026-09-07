"""Service disconnect recovery tests; no model or MLX runtime is involved."""

from __future__ import annotations

from pathlib import Path
import threading
import time

from ironmule_product.errors import BackendUnavailable
from ironmule_product.service import ProductService
from ironmule_product.types import GenerationRequest, ModelSpec


def _spec(tmp_path: Path) -> ModelSpec:
    return ModelSpec("local/test", "revision", str(tmp_path), 1)


def _request(request_id: str) -> GenerationRequest:
    return GenerationRequest(
        "local/test", (("user", "hello"),), max_tokens=2, request_id=request_id
    )


class _Store:
    def __init__(self, spec: ModelSpec):
        self.spec = spec

    def settings(self):
        return {"mode": "server", "max_pending": 4, "request_timeout_s": None}

    def model(self, _model_id):
        return self.spec

    def optimization_status(self):
        return {"stage": "inactive", "engine_started": False, "activation_allowed": False}


class _Backend:
    def __init__(self, *, fail_drain: bool = False):
        self.ready = True
        self.fail_drain = fail_drain
        self.cancel_observed = threading.Event()
        self.terminal_observed = threading.Event()
        self.close_calls = 0

    def stream(self, request, cancel, timeout):
        yield {
            "type": "token",
            "request_id": request.request_id,
            "text": "visible",
            "token_id": 1,
            "prompt_tokens": 1,
            "completion_tokens": 1,
        }
        if request.request_id == "disconnect":
            while not cancel.wait(0.01):
                pass
            self.cancel_observed.set()
            if self.fail_drain:
                raise BackendUnavailable("transport failed during cancellation")
            self.terminal_observed.set()
            yield {
                "type": "done",
                "request_id": request.request_id,
                "finish_reason": "cancelled",
                "prompt_tokens": 1,
                "completion_tokens": 1,
            }
            return
        yield {
            "type": "done",
            "request_id": request.request_id,
            "finish_reason": "length",
            "prompt_tokens": 1,
            "completion_tokens": 1,
        }

    def close(self):
        self.close_calls += 1
        self.ready = False


class _CloseInterruptBackend(_Backend):
    def __init__(self):
        super().__init__()
        self.closed = threading.Event()

    def stream(self, request, cancel, timeout):
        yield {
            "type": "token",
            "request_id": request.request_id,
            "text": "visible",
            "token_id": 1,
            "prompt_tokens": 1,
            "completion_tokens": 1,
        }
        self.cancel_observed.set()
        self.closed.wait()
        raise BackendUnavailable("backend closed")

    def close(self):
        super().close()
        self.closed.set()


def _wait_idle(service: ProductService) -> None:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        health = service.health()
        if health["active_requests"] == 0 and health["queued_requests"] == 0:
            return
        time.sleep(0.01)
    raise AssertionError("service did not become idle")


def test_consumer_disconnect_drains_cancel_and_preserves_warm_backend(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    backend = _Backend()
    service = ProductService(_Store(spec), backend=backend, spec=spec)
    disconnected = service.stream(_request("disconnect"))
    try:
        assert next(disconnected)["type"] == "token"
        disconnected.close()
        _wait_idle(service)

        assert backend.cancel_observed.is_set()
        assert backend.terminal_observed.is_set()
        assert backend.ready is True
        assert service.health()["cancelled_requests"] == 1
        assert [event["type"] for event in service.stream(_request("recovery"))] == [
            "token",
            "done",
        ]
    finally:
        service.close()


def test_failed_cancel_drain_retires_backend_without_changing_cancel_reason(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    backend = _Backend(fail_drain=True)
    service = ProductService(_Store(spec), backend=backend, spec=spec)
    disconnected = service.stream(_request("disconnect"))
    try:
        next(disconnected)
        disconnected.close()
        _wait_idle(service)

        health = service.health()
        assert health["cancelled_requests"] == 1
        assert health["failed_requests"] == 0
        assert health["ready"] is False
        assert backend.close_calls == 1
    finally:
        service.close()


def test_service_close_interrupts_a_cancel_drain(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    backend = _CloseInterruptBackend()
    service = ProductService(_Store(spec), backend=backend, spec=spec)
    disconnected = service.stream(_request("disconnect"))
    next(disconnected)
    disconnected.close()
    assert backend.cancel_observed.wait(1)

    started = time.monotonic()
    service.close()

    assert time.monotonic() - started < 2
    assert not service._thread.is_alive()  # noqa: SLF001 - lifecycle assertion
