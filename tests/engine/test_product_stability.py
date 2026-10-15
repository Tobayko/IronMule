"""STAB1: worker failures under the real scheduler, transport and process lifecycle (no model).

The worker is `stab1_fake_worker.py`, started through `MLXWorkerClient.start()` and speaking the real
protocol; faults are chosen per request. Nothing here measures speed or model output.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

import ironmule_product.backend as backend_module
import ironmule_product.service as service_module
from ironmule_product.backend import MLXWorkerClient
from ironmule_product.errors import ProductError
from ironmule_product.service import STABILITY_LOG, ProductService
from ironmule_product.state import ProductStore
from ironmule_product.types import GenerationRequest, ModelSpec

FAKE = Path(__file__).with_name("stab1_fake_worker.py")


def _setup(tmp_path: Path, monkeypatch, timeout: int | None = 120) -> tuple[ProductStore, ModelSpec]:
    real_popen = subprocess.Popen

    def fake_popen(args, *popen_args, **popen_kwargs):
        if len(args) > 3 and str(args[3]).endswith("worker.py"):
            args = [sys.executable, "-u", str(FAKE), *args[4:]]
        return real_popen(args, *popen_args, **popen_kwargs)

    monkeypatch.setattr(backend_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(service_module, "RESTART_BACKOFF_S", (0.0, 0.2, 0.4))  # production: 0, 5, 20 s
    store = ProductStore(tmp_path / "state")
    store.setup("server")
    store.set_request_timeout(timeout)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    spec = ModelSpec("local/test", "revision", str(snapshot.resolve()), 1)
    store.register_model(spec)
    return store, spec


def _service(store: ProductStore, spec: ModelSpec) -> ProductService:
    backend = MLXWorkerClient(spec, execution_variant="current_engine")
    backend.start()
    return ProductService(store, backend=backend, spec=spec)


def _request(request_id: str, content: str, max_tokens: int = 4) -> GenerationRequest:
    return GenerationRequest("local/test", (("user", content),), max_tokens=max_tokens, request_id=request_id,
                             stream=True)


def _call(service: ProductService, request: GenerationRequest) -> list[dict] | ProductError:
    events = []
    try:
        for event in service.stream(request):
            events.append(event)
    except ProductError as exc:
        exc.events = events  # what reached the caller before the error
        return exc
    return events


def _recovering(store: ProductStore, spec: ModelSpec) -> ProductService:
    service = _service(store, spec)
    service.make_backend = lambda: MLXWorkerClient(spec, execution_variant="current_engine")
    return service


def _text(events: list[dict]) -> str:
    return "".join(event.get("text", "") for event in events if event.get("type") == "token")


def _until(predicate, timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return bool(predicate())


def _pid(service: ProductService) -> int | None:
    return service.health()["worker"]["pid"]


def _grouped(service: ProductService, requests: list[GenerationRequest], gap_s: float = 0.0,
             blocker: tuple[str, int] = ("slow:0.5", 2)) -> list:
    """Hold the scheduler on a blocker group so `requests` wait together and form one group."""
    results: list = [None] * (len(requests) + 1)
    blocker = _request("r-blocker", blocker[0], max_tokens=blocker[1])

    def run(index: int, request: GenerationRequest) -> None:
        results[index] = _call(service, request)

    threads = [threading.Thread(target=run, args=(0, blocker))]
    threads[0].start()
    assert _until(lambda: service.health()["in_flight"] is not None)
    for index, request in enumerate(requests, 1):
        threads.append(threading.Thread(target=run, args=(index, request)))
        threads[-1].start()
        time.sleep(gap_s)
    for thread in threads:
        thread.join(timeout=30)
    return results[1:]


def _log(store: ProductStore) -> list[dict]:
    return [json.loads(line) for line in (store.root / STABILITY_LOG).read_text().splitlines()]


def test_a_worker_crash_is_recorded_with_its_exit_status_reason_and_last_stderr(tmp_path, monkeypatch):
    store, spec = _setup(tmp_path, monkeypatch)
    service = _service(store, spec)
    try:
        outcome = _call(service, _request("r-1", "crash"))
        assert isinstance(outcome, ProductError) and outcome.code == "backend_unavailable"
        worker = service.health()["worker"]
        assert worker["ready"] is False and worker["exit"]["reaped"] is True
        assert worker["exit"]["exit_status"] in (9, -15)  # exited, or terminated while exiting
        assert "exited before completing the protocol" in worker["exit"]["reason"]
        assert _until(lambda: _log(store)[-1]["event"] == "end")  # the caller can see its error first
        log = _log(store)
        events = [row["event"] for row in log]
        assert events[:2] == ["admit", "group"] and "worker_exit" in events and events[-1] == "end"
        exit_row = next(row for row in log if row["event"] == "worker_exit")
        assert "fatal out of memory" in exit_row["stderr_tail"]
        assert log[-1]["outcome"] == "failed" and log[-1]["error"] == "backend_unavailable"
        assert "crash" not in (store.root / STABILITY_LOG).read_text()  # no prompt text
    finally:
        service.close()


def test_one_members_deadline_cancels_only_that_member_and_keeps_the_worker(tmp_path, monkeypatch):
    store, spec = _setup(tmp_path, monkeypatch, timeout=6)
    service = _recovering(store, spec)
    try:
        pid = _pid(service)
        # One token per second. The group starts at 3 s; A's deadline passes 3 s into it, its run ends
        # at the next round (4 s), and B, whose last token and done the worker holds until the run
        # ends, meets its own deadline (5 s into the group). Before STAB1 the worker was retired at 3 s.
        late, early = _grouped(service, [_request("r-a", "slow:1.0", max_tokens=8),
                                         _request("r-b", "steady", max_tokens=3)], gap_s=2.0,
                               blocker=("slow:1.0", 3))
        assert isinstance(late, ProductError) and late.code == "request_timeout"
        assert not isinstance(early, ProductError) and early[-1]["type"] == "done"
        alone = _call(service, _request("r-c", "steady", max_tokens=3))
        assert _text(early) == _text(alone)  # the group member's output is the one it gets alone
        assert _pid(service) == pid and service.health()["worker"]["restarts"] == 0
        assert service.health()["worker"]["memory"] == {"active": 1, "cache": 2, "peak": 3}  # MEM2 counters
        assert any(row.get("memory") == {"active": 1, "cache": 2, "peak": 3}
                   for row in _log(store) if row["event"] == "group_end")
        assert not any(row["event"] == "worker_exit" for row in _log(store))
    finally:
        service.close()


def test_a_crashed_worker_is_replaced_and_no_partial_answer_is_continued_or_repeated(tmp_path, monkeypatch):
    store, spec = _setup(tmp_path, monkeypatch)
    service = _recovering(store, spec)
    try:
        before = _call(service, _request("r-0", "steady"))
        pid = _pid(service)
        crashed, neighbour = _grouped(service, [_request("r-a", "crash"), _request("r-b", "steady")])
        for result in (crashed, neighbour):
            assert isinstance(result, ProductError) and result.code == "backend_unavailable"
            assert not any(event["type"] == "done" for event in result.events)
        assert len(crashed.events) <= 1  # the one token it streamed, then the error; nothing after
        assert _until(lambda: service.health()["worker"]["state"] == "ready")
        assert _pid(service) not in (None, pid) and service.health()["worker"]["restarts"] == 1
        after = _call(service, _request("r-c", "steady"))
        assert _text(after) == _text(before)
        restart = [row for row in _log(store) if row["event"] == "worker_restart"]
        assert [row["outcome"] for row in restart] == ["ready"] and "exited" in restart[0]["reason"]
        ends = {row["id"]: row for row in _log(store) if row["event"] == "end"}
        assert ends["r-a"]["error"] == ends["r-b"]["error"] == "backend_unavailable"  # served once, not retried
    finally:
        service.close()


def test_a_worker_that_stops_answering_is_retired_after_the_deadline_and_replaced(tmp_path, monkeypatch):
    store, spec = _setup(tmp_path, monkeypatch, timeout=1)
    service = _recovering(store, spec)
    try:
        pid = _pid(service)
        results: list = [None, None]
        hung = threading.Thread(target=lambda: results.__setitem__(0, _call(service, _request("r-a", "hang"))))
        hung.start()
        assert _until(lambda: service.health()["in_flight"] is not None)
        results[1] = _call(service, _request("r-b", "steady"))  # queued behind the hung group
        hung.join(timeout=10)
        assert results[0].code == "request_timeout" and results[1].code == "request_timeout"
        assert _until(lambda: service.health()["worker"]["state"] == "ready" and _pid(service) != pid)
        assert _call(service, _request("r-c", "steady"))[-1]["type"] == "done"
        exits = [row for row in _log(store) if row["event"] == "worker_exit"]
        assert exits and "timed out" in exits[0]["reason"] and exits[0]["reaped"] is True
    finally:
        service.close()


def test_recovery_stops_after_three_failed_restarts_and_says_why(tmp_path, monkeypatch):
    store, spec = _setup(tmp_path, monkeypatch)
    control = tmp_path / "control.json"
    monkeypatch.setenv("FAKE_STAB1_CONTROL", str(control))
    service = _recovering(store, spec)
    try:
        control.write_text(json.dumps({"fail_start": True}))  # every replacement fails to start
        assert _call(service, _request("r-a", "crash")).code == "backend_unavailable"
        assert _until(lambda: service.health()["worker"]["state"] == "refused")
        health = service.health()
        assert health["ready"] is False and health["worker"]["restarts"] == 3
        assert "3 worker restarts" in health["worker"]["refused"]
        started = time.monotonic()
        assert _call(service, _request("r-b", "steady")).code == "backend_unavailable"
        assert time.monotonic() - started < 1.0  # an explicit error at once, not a wait
    finally:
        service.close()


@pytest.mark.parametrize("change, key", [({"digest": "b" * 64}, "sources_sha256"),
                                         ({"engine": {"configuration": "current_profile", "profile": 2}}, "engine")])
def test_a_replacement_with_another_identity_is_never_used(tmp_path, monkeypatch, change, key):
    """Changed sources on disk, or another tuned profile than the one the learner is bound to."""
    store, spec = _setup(tmp_path, monkeypatch)
    control = tmp_path / "control.json"
    monkeypatch.setenv("FAKE_STAB1_CONTROL", str(control))
    service = _recovering(store, spec)
    try:
        control.write_text(json.dumps(change))
        assert _call(service, _request("r-a", "crash")).code == "backend_unavailable"
        assert _until(lambda: service.health()["worker"]["state"] == "refused")
        assert service.health()["ready"] is False
        attempts = [row for row in _log(store) if row["event"] == "worker_restart" and row.get("attempt")]
        assert [row["outcome"] for row in attempts] == [f"failed: a different identity ({key})"] * 3
        assert all(row.get("reaped") for row in _log(store) if row["event"] == "worker_exit")
    finally:
        service.close()


def test_a_worker_that_dies_between_groups_is_retired_and_replaced(tmp_path, monkeypatch):
    import os
    import signal

    store, spec = _setup(tmp_path, monkeypatch)
    service = _recovering(store, spec)
    try:
        before = _call(service, _request("r-0", "steady"))
        pid = _pid(service)
        os.kill(pid, signal.SIGKILL)  # nothing in flight: no backend call notices
        assert _until(lambda: service.health()["worker"]["state"] == "ready" and _pid(service) != pid)
        assert _text(_call(service, _request("r-1", "steady"))) == _text(before)
        exits = [row for row in _log(store) if row["event"] == "worker_exit"]
        assert exits[0]["pid"] == pid and exits[0]["exit_status"] == -9 and exits[0]["reaped"] is True
    finally:
        service.close()


def test_a_worker_generation_error_is_named_as_such_and_the_worker_replaced(tmp_path, monkeypatch):
    """STAB1 run 1: the worker's out-of-memory report arrived as `generation_error` without a request id
    and was misread as a protocol identity violation."""
    store, spec = _setup(tmp_path, monkeypatch)
    service = _recovering(store, spec)
    try:
        assert _call(service, _request("r-a", "error")).code == "backend_unavailable"
        assert _until(lambda: service.health()["worker"]["state"] == "ready")
        exits = [row for row in _log(store) if row["event"] == "worker_exit"]
        assert exits[0]["reason"] == "BackendUnavailable: grouped engine generation failed (generation_error)"
        assert _call(service, _request("r-b", "steady"))[-1]["type"] == "done"
    finally:
        service.close()
