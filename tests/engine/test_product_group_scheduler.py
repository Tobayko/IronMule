"""Queue and transport-contract coverage for opt-in product grouping.

These fixtures model only the JSON transport boundary.  They make no model,
MLX, hardware, latency, or throughput assertion.
"""

from __future__ import annotations

from collections import deque
import queue
import threading

from ironmule_product.service import ProductService, _Session
from ironmule_product.types import GenerationRequest


def _request(request_id: str) -> GenerationRequest:
    return GenerationRequest("local/test", (("user", request_id),), max_tokens=2,
                             request_id=request_id)


class _Transport:
    ready = True
    batch_capacity = 4

    def __init__(self, *, allowed: bool = True, invalid_rows: bool = False) -> None:
        self.allowed = allowed
        self.invalid_rows = invalid_rows
        self.batch_calls: list[list[str]] = []
        self.stream_calls = 0
        self.close_calls = 0

    def can_batch_requests(self, requests):
        return self.allowed

    def complete_batch(self, requests, *, cancels, timeout):
        self.batch_calls.append([request.request_id for request in requests])
        rows = [{
            "request_id": request.request_id,
            "events": [
                {"type": "token", "request_id": request.request_id, "text": request.request_id,
                 "token_id": 1, "prompt_tokens": 1, "completion_tokens": 1},
                {"type": "done", "request_id": request.request_id, "finish_reason": "length",
                 "prompt_tokens": 1, "completion_tokens": 1},
            ],
            "metadata": {},
        } for request in requests]
        return rows[:-1] if self.invalid_rows else rows

    def stream(self, request, *_args, **_kwargs):
        self.stream_calls += 1
        return iter([
            {"type": "token", "request_id": request.request_id, "text": request.request_id,
             "token_id": 1, "prompt_tokens": 1, "completion_tokens": 1},
            {"type": "done", "request_id": request.request_id, "finish_reason": "length",
             "prompt_tokens": 1, "completion_tokens": 1},
        ])

    def close(self):
        self.close_calls += 1


def _service(backend: _Transport) -> ProductService:
    """A scheduler shell for deterministic queue-planning/unit transport tests."""
    service = object.__new__(ProductService)
    service.backend = backend
    service._pending = queue.Queue()
    service._carry = deque()
    service._sessions = {}
    service._lock = threading.RLock()
    service._closed = threading.Event()
    service._active = set()
    service._completed = service._failed = service._cancelled = 0
    service._last_batch_size = None
    service._batch_count = 0
    return service


def test_metadata_rejection_runs_first_single_and_keeps_later_fifo_in_carry():
    service = _service(_Transport(allowed=False))
    first, second, third = (_Session(_request(value), None) for value in ("a", "b", "c"))
    service._pending.put(first)
    service._pending.put(second)
    service._pending.put(third)

    selected = service._group_for(service._pending.get_nowait())

    assert [item.request.request_id for item in selected] == ["a"]
    assert [item.request.request_id for item in service._carry] == ["b", "c"]
    assert service._next_session() is second
    assert service._next_session() is third
    # The three acquired queue tasks remain balanced by scheduler retirement.
    for _ in range(3):
        service._pending.task_done()


def test_grouping_drains_earlier_carry_before_newer_pending_work():
    service = _service(_Transport())
    first, carried, pending = (_Session(_request(value), None) for value in ("a", "b", "c"))
    service._carry.append(carried)
    service._pending.put(pending)

    selected = service._group_for(first)

    assert [item.request.request_id for item in selected] == ["a", "b", "c"]
    service._pending.task_done()


def test_buffered_batch_routes_each_row_through_normal_event_delivery_once():
    backend = _Transport()
    service = _service(backend)
    first, second = _Session(_request("a"), None), _Session(_request("b"), None)

    service._perform_batch([first, second])

    assert backend.batch_calls == [["a", "b"]]
    assert backend.stream_calls == 0
    assert [first.events.get_nowait()["type"], first.events.get_nowait()["type"]] == ["token", "done"]
    assert [second.events.get_nowait()["type"], second.events.get_nowait()["type"]] == ["token", "done"]
    assert service._completed == 2
    assert service._failed == service._cancelled == 0
    assert service._last_batch_size == 2
    assert service._batch_count == 1


def test_filtered_group_with_one_live_session_uses_single_path_before_dispatch():
    backend = _Transport()
    service = _service(backend)
    first, cancelled = _Session(_request("a"), None), _Session(_request("b"), None)
    cancelled.cancelled.set()

    service._perform_batch([first, cancelled])

    assert backend.batch_calls == []
    assert backend.stream_calls == 1
    assert service._completed == 1
    assert service._cancelled == 1


def test_invalid_batch_rows_do_not_publish_batch_health_metadata():
    backend = _Transport(invalid_rows=True)
    service = _service(backend)

    service._perform_batch([_Session(_request("a"), None), _Session(_request("b"), None)])

    assert backend.batch_calls == [["a", "b"]]
    assert service._last_batch_size is None
    assert service._batch_count == 0
    assert service._failed == 2


def test_queued_cancellation_does_not_retire_an_undispatched_backend():
    backend = _Transport()
    service = _service(backend)
    session = _Session(_request("a"), None)
    session.cancelled.set()

    service._perform(session)

    assert backend.close_calls == 0
    assert service._cancelled == 1


def test_done_queue_time_uses_dispatch_time_not_buffered_delivery_time():
    service = _service(_Transport())
    session = _Session(_request("a"), None, submitted_ns=100, dispatched_ns=250)

    service._consume_events(session, iter([
        {"type": "done", "request_id": "a", "finish_reason": "length",
         "prompt_tokens": 1, "completion_tokens": 0},
    ]))

    assert session.events.get_nowait()["queue_ns"] == 150
