"""START1: idle-window tuning in the product server. No model, no GPU, no real tune process."""

from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace

from ironmule_product import cli


class _Process:
    def __init__(self, finishes_after: int | None, code: int = 0):
        self.polls, self.finishes_after, self.code, self.pid = 0, finishes_after, code, 4242
        self.returncode = None

    def poll(self):
        self.polls += 1
        if self.finishes_after is not None and self.polls > self.finishes_after:
            self.returncode = self.code
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = -15
        return self.returncode


class _Backend:
    def __init__(self):
        self.closed = self.started = False

    def close(self):
        self.closed = True

    def start(self):
        self.started = True


def _tuner(monkeypatch, process, confirmed=False):
    service = SimpleNamespace(backend=_Backend())
    made, logged, stopped = [], [], []
    tuner = cli.IdleTuner(service, SimpleNamespace(model_id="m"), lambda: made.append(_Backend()) or made[-1],
                          popen=lambda *_a, **_k: process, log=lambda line, **_k: logged.append(line))
    tuner.POLL_S = 0
    monkeypatch.setattr(tuner, "_confirmed", lambda: confirmed)
    monkeypatch.setattr(cli, "_stop_group", lambda p: stopped.append(p.pid))
    return tuner, service, made, logged, stopped


def test_a_completed_idle_tune_hands_the_service_a_fresh_worker_once(monkeypatch):
    tuner, service, made, logged, stopped = _tuner(monkeypatch, _Process(finishes_after=2))
    old = service.backend
    tuner(lambda: False)
    assert old.closed and service.backend is made[0] and made[0].started and not stopped
    assert tuner.done and '"done"' in logged[-1]
    tuner(lambda: False)  # nothing more to do once confirmed
    assert len(made) == 1


def test_a_waiting_request_stops_the_tune_and_gets_a_worker_back(monkeypatch):
    tuner, service, made, logged, stopped = _tuner(monkeypatch, _Process(finishes_after=None))
    tuner(lambda: True)
    assert stopped == [4242] and service.backend is made[0]
    assert not tuner.done and '"interrupted"' in logged[-1]


def test_failed_tunes_stop_retrying(monkeypatch):
    tuner, service, made, logged, _ = _tuner(monkeypatch, _Process(finishes_after=0, code=1))
    for _ in range(4):
        tuner(lambda: False)
    assert tuner.failures == 2 and len(made) == 2


def test_an_already_confirmed_profile_skips_the_tune_and_keeps_the_worker(monkeypatch):
    tuner, service, made, _, _ = _tuner(monkeypatch, _Process(finishes_after=0), confirmed=True)
    old = service.backend
    tuner(lambda: False)
    assert tuner.done and service.backend is old and not old.closed and not made


def test_the_scheduler_runs_idle_work_only_while_nothing_waits():
    from ironmule_product.service import ProductService

    runs = []
    service = object.__new__(ProductService)
    service._pending, service._carry, service._lock = queue.Queue(), __import__("collections").deque(), threading.RLock()
    service._closed = threading.Event()
    service.idle_after_s = 0.0
    service.idle_task = lambda preempted: runs.append(preempted())
    thread = threading.Thread(target=service._run, daemon=True)
    thread.start()
    time.sleep(0.3)
    service._closed.set()
    thread.join(timeout=2)
    assert runs and runs[0] is False


def test_a_request_during_idle_work_is_queued_not_refused():
    from ironmule_product.service import ProductService

    service = object.__new__(ProductService)
    service.backend = SimpleNamespace(ready=False)
    service._idle_busy = True
    assert service._backend_ready()
    service._idle_busy = False
    assert not service._backend_ready()
