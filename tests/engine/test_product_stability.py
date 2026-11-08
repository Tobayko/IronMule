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
        # Queued in order behind the blocker before the next starts: under a loaded test run a late thread
        # could otherwise miss the group and run alone.
        _until(lambda index=index: service.health()["queued_requests"] >= index, timeout=5.0)
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
        # A three-second blocker: both must queue behind it to form one group, even on a loaded test machine.
        crashed, neighbour = _grouped(service, [_request("r-a", "crash"), _request("r-b", "steady")],
                                      blocker=("slow:1.0", 3))
        groups = [row["ids"] for row in _log(store) if row["event"] == "group"]
        # The premise: both ran in one group. If not, the scheduler grouped them otherwise, not a crash bug.
        assert any({"r-a", "r-b"} <= set(ids) for ids in groups), groups
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


# ---------------------------------------------------------------- STAB3: AG5-G run 6, a worker lost under a spike

def _ends(store: ProductStore) -> dict[str, list[dict]]:
    ends: dict[str, list[dict]] = {}
    for row in _log(store):
        if row["event"] == "end":
            ends.setdefault(row["id"], []).append(row)
    return ends


def _threads(service: ProductService, requests: list[GenerationRequest], results: dict) -> list[threading.Thread]:
    threads = []
    for request in requests:
        threads.append(threading.Thread(target=lambda r=request: results.__setitem__(r.request_id, _call(service, r))))
        threads[-1].start()
    return threads


def test_a_worker_lost_while_requests_join_ends_each_request_once_and_serves_the_rest_after_a_restart(
        tmp_path, monkeypatch):
    """The run-6 pattern: a group runs, newcomers join it, the worker dies. Each request reaches one terminal
    state, the ones in flight fail at once, and a request that came after the loss is served by the replacement,
    all far inside the 60 s request timeout."""
    store, spec = _setup(tmp_path, monkeypatch, timeout=60)

    def dense() -> MLXWorkerClient:  # the agent's stacked engine at width 6, which takes joiners (run 6)
        return MLXWorkerClient(spec, execution_variant="current_engine", engine_configuration="dense_stacked",
                               dense_width=6)

    backend = dense()
    backend.start()
    service = ProductService(store, backend=backend, spec=spec)
    service.make_backend = dense
    results: dict = {}
    try:
        started = time.monotonic()
        threads = _threads(service, [_request("r-a", "slow:0.3", max_tokens=8)], results)
        assert _until(lambda: service.health()["in_flight"] is not None)
        threads += _threads(service, [_request("r-b", "slow:0.3", max_tokens=8),
                                      _request("r-c", "slow:0.3", max_tokens=8)], results)
        assert _until(lambda: any(row["event"] == "join" for row in _log(store)))
        threads += _threads(service, [_request("r-d", "crash")], results)
        assert _until(lambda: any(row["event"] == "worker_exit" for row in _log(store)))
        threads += _threads(service, [_request("r-e", "steady")], results)
        for thread in threads:
            thread.join(timeout=30)
        assert time.monotonic() - started < 20, "nothing waited for the request timeout"
        assert results["r-e"][-1]["type"] == "done", "a request after the loss is served by the replacement"
        lost = [rid for rid in ("r-a", "r-b", "r-c", "r-d") if isinstance(results[rid], ProductError)]
        assert "r-d" in lost and all(results[rid].code == "backend_unavailable" for rid in lost)
        assert all(results[rid][-1]["type"] == "done" for rid in ("r-a", "r-b", "r-c") if rid not in lost)
        assert _until(lambda: set(_ends(store)) == {"r-a", "r-b", "r-c", "r-d", "r-e"})
        assert all(len(rows) == 1 for rows in _ends(store).values()), "exactly one terminal state each"
        assert service.health()["active_requests"] == 0 and service.health()["queued_requests"] == 0
    finally:
        service.close()


def test_a_live_worker_without_progress_is_retired_after_the_stall_limit_not_the_request_timeout(
        tmp_path, monkeypatch):
    """Before STAB3 a worker that stopped sending held its group until the deadline plus a full request timeout,
    and every request behind it waited out its own deadline (run 6: an hour each)."""
    store, spec = _setup(tmp_path, monkeypatch, timeout=60)
    monkeypatch.setattr(backend_module, "STALL_TIMEOUT_S", 1.0)
    service = _recovering(store, spec)
    results: dict = {}
    try:
        pid = _pid(service)
        started = time.monotonic()
        threads = _threads(service, [_request("r-a", "hang")], results)
        assert _until(lambda: service.health()["in_flight"] is not None)
        threads += _threads(service, [_request("r-b", "steady")], results)
        for thread in threads:
            thread.join(timeout=30)
        assert time.monotonic() - started < 15
        assert results["r-a"].code == "backend_unavailable", "the hung member is told, not left to time out"
        assert results["r-b"][-1]["type"] == "done" and _pid(service) != pid, "the one behind it is served"
        exits = [row for row in _log(store) if row["event"] == "worker_exit"]
        assert "no progress for 1 s" in exits[0]["reason"] and exits[0]["reaped"] is True
    finally:
        service.close()


def test_slow_failing_restarts_use_up_the_budget_and_waiting_requests_fail_at_once(tmp_path, monkeypatch):
    """Each attempt took longer than the restart window, so the window forgot the attempts before it and recovery
    never gave up: admission stayed open and every request waited out its deadline. Now three restarts without a
    served request end recovery whatever their duration, the request that waited is told at once, and new ones
    are refused immediately."""
    store, spec = _setup(tmp_path, monkeypatch, timeout=60)
    monkeypatch.setattr(service_module, "RESTART_WINDOW_S", 0.3)
    control = tmp_path / "control.json"
    monkeypatch.setenv("FAKE_STAB1_CONTROL", str(control))
    service = _recovering(store, spec)
    results: dict = {}
    try:
        control.write_text(json.dumps({"fail_start": True, "start_delay": 0.5}))
        started = time.monotonic()
        assert _call(service, _request("r-a", "crash")).code == "backend_unavailable"
        assert _until(lambda: service.health()["worker"]["state"] == "restarting", timeout=5)
        health = service.health()
        assert health["accepting"] is True and health["ready"] is False, "admitted while replaced, not ready"
        threads = _threads(service, [_request("r-b", "steady")], results)
        for thread in threads:
            thread.join(timeout=30)
        assert results["r-b"].code == "backend_unavailable" and time.monotonic() - started < 20
        health = service.health()
        assert health["worker"]["state"] == "refused" and health["accepting"] is False
        assert "no request served" in health["worker"]["refused"]
        attempts = [row for row in _log(store) if row["event"] == "worker_restart_start"]
        assert len(attempts) == 3
        quick = time.monotonic()
        assert _call(service, _request("r-c", "steady")).code == "backend_unavailable"
        assert time.monotonic() - quick < 1.0
        assert all(len(rows) == 1 for rows in _ends(store).values())
    finally:
        service.close()


def test_a_successful_restart_serves_new_requests_and_the_budget_returns_only_after_one_is_served(
        tmp_path, monkeypatch):
    """A replacement that starts but never serves does not prove recovery: with the window too short to count,
    four losses in a row with no request served end recovery; one served request in between gives it back."""
    store, spec = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(service_module, "RESTART_WINDOW_S", 0.01)
    service = _recovering(store, spec)
    try:
        assert _call(service, _request("r-1", "crash")).code == "backend_unavailable"
        assert _until(lambda: service.health()["worker"]["state"] == "ready")
        assert service._unproven_restarts == 1, "a replacement that is ready has not served yet"
        assert _call(service, _request("r-2", "steady"))[-1]["type"] == "done"
        assert service._unproven_restarts == 0, "a served request proves it"
        for index in range(3, 7):
            assert _call(service, _request(f"r-{index}", "crash")).code == "backend_unavailable"
            assert _until(lambda: service.health()["worker"]["state"] in ("ready", "refused"))
        assert service.health()["worker"]["state"] == "refused"
        assert "no request served" in service.health()["worker"]["refused"]
    finally:
        service.close()
