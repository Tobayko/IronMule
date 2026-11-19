"""A GGUF worker learns from completed local cohorts while foreground inference has priority."""
import os
import threading
import time
from types import SimpleNamespace

from ironmule import compare_learning as core
from ironmule_product.admission_learning import AdmissionLearning, live_status
from ironmule_product.ggml_bridge import GgmlBridge


def controller(tmp_path, *, enabled=True):
    engine = SimpleNamespace()
    return AdmissionLearning(tmp_path, "local.gguf", engine, {"seq_ctx": 2048}, enabled=enabled)


def observed(controller):
    for index in range(2):
        controller.observe(index, (64, 64, 64, 64), 8, 12.0, 85.0)


def test_worker_uses_only_completed_local_observations_and_reloads_qualified_rule(tmp_path, monkeypatch):
    ctl = controller(tmp_path)
    observed(ctl)
    monkeypatch.setattr(ctl, "_room_for_child", lambda: True)
    held = []
    def trial(observations, *, binding, gguf, state, deadline, interrupted):
        assert all(type(r) is core.Observation for r in observations)
        assert len(observations) == 2
        assert time.monotonic() < deadline <= time.monotonic() + 60
        assert not interrupted()
        held.append(binding)
        return {"status": "qualified", "candidate_budget": 64, "incumbent_budget": 0}
    monkeypatch.setattr(core, "respond", trial)
    monkeypatch.setattr(core, "read_rules", lambda *_: {"4,4,4,4/8": 64})
    ctl.idle(lambda: False)
    assert ctl.engine.admission_budgets == {"4,4,4,4/8": 64}
    view = live_status(str(tmp_path), os.getpid())
    assert view["status"] == "qualified" and view["rules"] == 1 and view["observations"] == 2
    assert view["last"]["selection"] == {"focus": "admission", "samples": 2, "concurrency": 4, "max_tokens": 8}
    assert "prompt_lengths" not in repr(view)
    ctl.idle(lambda: False)
    assert len(held) == 1


def test_trial_requires_room_new_cohorts_and_idle_server(tmp_path, monkeypatch):
    ctl = controller(tmp_path)
    observed(ctl)
    monkeypatch.setattr(ctl, "_room_for_child", lambda: False)
    monkeypatch.setattr(core, "respond", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    ctl.idle(lambda: True)
    assert ctl.status == "waiting"
    ctl.idle(lambda: False)
    assert ctl.status == "deferred" and not ctl.engine.admission_budgets
    assert controller(tmp_path, enabled=False).engine.admission_observer is None


def test_foreground_never_waits_for_native_holdout_and_cancellation_is_latched(tmp_path, monkeypatch):
    ctl = controller(tmp_path)
    observed(ctl)
    class Stop(threading.Event):
        def wait(self, timeout=None):
            return super().wait(min(timeout or 0.005, 0.005))

    bridge = GgmlBridge.__new__(GgmlBridge)
    bridge._agent = None
    bridge._admission = ctl
    bridge._run_lock = threading.Lock()
    bridge._demand_lock = threading.Lock()
    bridge._arrival_epoch = bridge._demand = 0
    bridge._last_run = 0.0
    bridge._stop = Stop()
    bridge._engine = ctl.engine
    began, release, finished = threading.Event(), threading.Event(), threading.Event()
    def trial(*args, interrupted, **kwargs):
        began.set()
        assert release.wait(2)
        assert interrupted()  # even a request that completed between two child polls
        finished.set()
        return {"status": "deferred", "reason": "foreground work resumed"}
    monkeypatch.setattr(core, "respond", trial)
    monkeypatch.setattr(ctl, "_room_for_child", lambda: True)
    # Empty completion is sufficient here; the generation entry point takes its normal run lock.
    bridge._engine.run = lambda *_a, **_k: []
    worker = threading.Thread(target=bridge._idle_loop)
    worker.start()
    try:
        assert began.wait(2)
        assert bridge.generate_many([], 1) == []  # the independent native trial is still blocked
        release.set()
        assert finished.wait(2)
    finally:
        release.set()
        bridge._stop.set()
        worker.join(2)
    assert not worker.is_alive() and not ctl.engine.admission_budgets
