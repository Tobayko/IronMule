from contextlib import contextmanager
from types import SimpleNamespace

from ironmule import autopilot


class FakeController:
    def __init__(self, promote_after):
        self.promote_after, self.calls, self.grants = promote_after, 0, []
        self.action_contract = {"profiles": ["current.sequential.v1", "core.sequential.v1"]}
        self.config = SimpleNamespace(freeze_every=32, min_train=8)

    def snapshot(self):
        used = 2 * self.calls
        return {"promoted": int(self.calls >= self.promote_after), "rejected": 0, "updates": self.calls,
                "candidate_version": 0, "pairs_completed": 0,
                "offline_training": {"used": used, "remaining": max(0, self.grants[-1] - used) if self.grants else 0}}

    @contextmanager
    def offline_training(self, *, max_extra_comparison_executions):
        self.grants.append(max_extra_comparison_executions)
        yield self

    def flush(self):
        return True

    def hardware_knowledge(self):
        labels = self.calls // 2
        actions = [{"profile_id": "current.sequential.v1", "ew_cost_seconds_per_request": 0.4, "observations": labels},
                   {"profile_id": "core.sequential.v1", "ew_cost_seconds_per_request": 0.3, "observations": labels}]
        unseen = [dict(a, ew_cost_seconds_per_request=None, observations=0) for a in actions]
        trained = {"request_count": "1", "requested_tokens": "over_32", "active_action": 1, "actions": actions}
        return {"cells": [{**trained, "requested_tokens": "at_most_32", "active_action": 0, "actions": unseen}, trained,
                          {"request_count": "2", "requested_tokens": "at_most_32", "active_action": 0,
                           "actions": unseen}]}


class FakeRuntime:
    def __init__(self, controller):
        self.online_controller, self.prompts = controller, []
        self.engine = SimpleNamespace(knobs=SimpleNamespace(as_dict=lambda: {"readback_every": 2, "fused_argmax": False,
                                                                      "speculate_ngram": 3}))

    def generate(self, prompt, max_tokens):
        self.prompts.append((prompt, max_tokens))
        self.online_controller.calls += 1


def test_self_training_stops_at_the_first_decision_and_uses_one_grant():
    runtime = FakeRuntime(FakeController(promote_after=5))
    result = autopilot.self_train(runtime, max_tokens=16, seconds=60, budget=64, log=lambda _: None)
    assert result["calls"] == 5 and result["promoted"] == 1
    assert runtime.online_controller.grants == [64]
    assert {tokens for _, tokens in runtime.prompts} == {16}
    assert runtime.prompts[4][0] == autopilot.TRAINING_PROMPTS[0]


def test_self_training_stops_when_the_grant_is_spent():
    runtime = FakeRuntime(FakeController(promote_after=10**6))
    result = autopilot.self_train(runtime, max_tokens=4, seconds=60, budget=8, log=lambda _: None)
    assert result["calls"] == 4 and result["promoted"] == 0


def test_decisions_report_only_observed_cells_and_active_profiles():
    report = autopilot.decisions(FakeRuntime(FakeController(promote_after=1)))
    assert report["knobs"] == {"readback_every": 2}
    assert report["cells"] == [{"requests": "1", "tokens": "over_32", "serving": "core.sequential.v1",
                                "cost_s": {"current.sequential.v1": 0.4, "core.sequential.v1": 0.3}}]


def test_buffered_answers_use_the_completion_limit_for_visible_first_content():
    config = autopilot.controller_config(128)
    assert config.qualification_protocol == "sequential_noise_v4" and config.directed_training
    assert config.latency_view == "delivered" and config.comparison_floor_s == 0
    assert config.ttft_limit_s == config.latency_limit_s == 64.0
    assert autopilot.controller_config(8).latency_limit_s == 30.0


def test_self_training_stops_once_converged_without_a_better_profile():
    runtime = FakeRuntime(FakeController(promote_after=10**6))
    result = autopilot.self_train(runtime, max_tokens=64, seconds=60, budget=10**4, log=lambda _: None)
    assert result["converged"] and result["promoted"] == 0
    assert result["calls"] == 64  # two freeze rounds; both profiles already hold min_train labels


def test_failed_tuning_falls_back_to_the_untuned_engine(monkeypatch):
    import importlib
    tune = importlib.import_module("ironmule.tune")
    identity = SimpleNamespace(identity_sha256="a" * 64, model_id="m")
    monkeypatch.setattr(autopilot, "_probe_out_of_process", lambda probe: {"fingerprint": "f"})
    monkeypatch.setattr(tune, "resolve_local_model", lambda _: SimpleNamespace(identity=identity))
    monkeypatch.setattr(tune, "load_profile", lambda *a, **k: None)
    def broken(*_a, **_k):
        raise RuntimeError("no paired processes here")
    monkeypatch.setattr(autopilot, "_tune_out_of_process", broken)
    logged = []
    prepared = autopilot.prepare("m", log=logged.append)
    assert prepared["profile"] is None and not prepared["tuned_now"]
    assert "serving the untuned engine" in logged[-1]
    assert sum("tuning attempt" in line for line in logged) == 2


def _fake_prepare(monkeypatch, tmp_path, *, tuned_at, verdict):
    import importlib
    tune = importlib.import_module("ironmule.tune")
    identity = SimpleNamespace(identity_sha256="b" * 64, model_id="m")
    calls = []
    monkeypatch.setattr(autopilot, "STATE", tmp_path)
    monkeypatch.setattr(autopilot, "_probe_out_of_process", lambda probe: {"fingerprint": "f"})
    monkeypatch.setattr(tune, "resolve_local_model", lambda _: SimpleNamespace(identity=identity))
    monkeypatch.setattr(tune, "load_profile", lambda *a, **k: {"gain": 0.2, "tuned_at": tuned_at})
    monkeypatch.setattr(tune, "revalidate", lambda _, **k: calls.append("revalidate") or {"verdict": verdict})
    monkeypatch.setattr(autopilot, "_tune_out_of_process", lambda *_: calls.append("tune") or {"gain": 0.1})
    return calls


def test_recent_settings_are_used_without_any_measurement(monkeypatch, tmp_path):
    import time
    calls = _fake_prepare(monkeypatch, tmp_path, tuned_at=time.time(), verdict="still_valid")
    assert autopilot.prepare("m", log=lambda _: None)["profile"]["gain"] == 0.2 and calls == []


def test_old_settings_are_rechecked_and_retuned_only_when_they_no_longer_hold(monkeypatch, tmp_path):
    calls = _fake_prepare(monkeypatch, tmp_path, tuned_at=0, verdict="still_valid")
    assert autopilot.prepare("m", log=lambda _: None)["profile"]["gain"] == 0.2 and calls == ["revalidate"]
    assert autopilot.prepare("m", log=lambda _: None) and calls == ["revalidate"], "marked checked"
    calls = _fake_prepare(monkeypatch, tmp_path / "other", tuned_at=0, verdict="retune_required")
    prepared = autopilot.prepare("m", log=lambda _: None)
    assert calls == ["revalidate", "tune"] and prepared["tuned_now"] and prepared["profile"]["gain"] == 0.1


def test_a_failing_recheck_keeps_the_stored_settings(monkeypatch, tmp_path):
    import importlib
    calls = _fake_prepare(monkeypatch, tmp_path, tuned_at=0, verdict="unused")
    def broken(*_a, **_k):
        raise RuntimeError("child aborted")
    monkeypatch.setattr(importlib.import_module("ironmule.tune"), "revalidate", broken)
    logged = []
    prepared = autopilot.prepare("m", log=logged.append)
    assert prepared["profile"]["gain"] == 0.2 and calls == [] and "check_failed" in logged[-1]


def test_probe_runs_in_a_child_then_reads_the_cache(monkeypatch):
    import subprocess
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: calls.append(argv[-1]))
    import importlib
    monkeypatch.setattr(importlib.import_module("ironmule.hw"), "apply_cuda_graph_defaults", lambda: {})
    def probe(allow_measure=True):
        calls.append(allow_measure)
        return {"fingerprint": "f"}
    assert autopilot._probe_out_of_process(probe) == {"fingerprint": "f"}
    assert calls == ["from ironmule.hw import probe; probe()", False]
    def failing(argv, **kw):
        raise subprocess.CalledProcessError(1, argv)
    monkeypatch.setattr(subprocess, "run", failing)
    calls.clear()
    autopilot._probe_out_of_process(probe)
    assert calls == [True], "a failed child falls back to measuring here"


def test_unusable_cache_after_the_child_falls_back_to_measuring_here(monkeypatch):
    import importlib, subprocess
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: None)
    monkeypatch.setattr(importlib.import_module("ironmule.hw"), "apply_cuda_graph_defaults", lambda: {})
    calls = []
    def probe(allow_measure=True):
        calls.append(allow_measure)
        if not allow_measure:
            raise RuntimeError("probe_binding_mismatch")
        return {"fingerprint": "f"}
    assert autopilot._probe_out_of_process(probe) == {"fingerprint": "f"} and calls == [False, True]


def test_learning_speculation_offers_the_profile_and_compares_offsets():
    plain, learning = autopilot.controller_config(64), autopilot.controller_config(64, True)
    assert (plain.speculative_profiles, plain.state_signature) == (False, "hash")
    assert (learning.speculative_profiles, learning.state_signature) == (True, "offset")


def test_idle_learner_trains_on_recent_prompts_only_while_nobody_waits():
    import time
    controller = FakeController(promote_after=10**6)
    runtime = FakeRuntime(controller)
    learner = autopilot.IdleLearner(runtime, max_tokens=8, idle_s=0.0, budget=10, keep=2)
    with learner:
        time.sleep(1.0)
        assert learner.calls == 0, "no prompts yet: nothing to learn from"
        learner.seen("first question")
        learner.seen("second question")
        deadline = time.monotonic() + 5
        while learner.calls < 4 and time.monotonic() < deadline:
            time.sleep(0.05)
    assert learner.calls >= 4
    assert {prompt for prompt, _ in runtime.prompts} == {"first question", "second question"}
    assert controller.grants and all(g == 10 for g in controller.grants)


def test_idle_learner_pauses_while_a_user_request_holds_the_lock():
    import time
    runtime = FakeRuntime(FakeController(promote_after=10**6))
    learner = autopilot.IdleLearner(runtime, max_tokens=8, idle_s=0.0, budget=10**4)
    with learner:
        with learner.lock:
            learner.seen("question")
            before = learner.calls
            time.sleep(0.8)
            assert learner.calls == before, "training waits for the user's request"


def test_numeric_consent_is_explicit_remembered_and_withdrawable(monkeypatch, tmp_path):
    monkeypatch.setattr(autopilot, "STATE", tmp_path)
    assert autopilot.numeric_consent(None) is False, "never assumed"
    assert autopilot.numeric_consent(True) and autopilot.numeric_consent(None), "given once, remembered"
    assert autopilot.numeric_consent(False) is False and autopilot.numeric_consent(None) is False
    (tmp_path / "consent.json").write_text("not json")
    assert autopilot.numeric_consent(None) is False, "a broken record is no consent"


def test_a_too_slow_tune_is_not_retried_and_remembered_for_a_week(monkeypatch, tmp_path):
    calls = _fake_prepare(monkeypatch, tmp_path, tuned_at=0, verdict="unused")
    import importlib
    monkeypatch.setattr(importlib.import_module("ironmule.tune"), "load_profile", lambda *a, **k: None)

    def slow(*_):
        calls.append("tune")
        raise autopilot.TuneTooSlow("20 tokens took 147 s (TUNE1)")

    monkeypatch.setattr(autopilot, "_tune_out_of_process", slow)
    logged = []
    assert autopilot.prepare("m", log=logged.append)["profile"] is None
    assert calls == ["tune"], "no retry of the same answer"
    logged.clear()
    assert autopilot.prepare("m", log=logged.append)["profile"] is None and calls == ["tune"]
    assert any("too slow here within the last week" in line for line in logged)


def test_a_cpu_only_machine_has_a_device_identity(monkeypatch):
    import mlx.core as mx

    from ironmule import hw
    monkeypatch.setattr(mx, "device_info", lambda: {"architecture": "cpu"})
    monkeypatch.setattr(mx.metal, "is_available", lambda: False)
    monkeypatch.setattr(mx.cuda, "is_available", lambda: False)
    monkeypatch.setattr(hw, "static_facts", lambda: {"chip": "Intel Xeon", "machine": "x86_64", "memory_bytes": 2**35})
    assert hw.device_identity() == {"device_name": "cpu: Intel Xeon", "memory_size": 2**35}
