"""Own request cohorts direct isolated native admission trials. Only a locally qualified action serves."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from ironmule import compare_learning as learn


def binding():
    return {"gguf_sha256": "1" * 64, "machine": "fixture", "host_sha256": "2" * 64,
            "seq_ctx": 512, "prompt_budget": 0, "implementation": learn.implementation(), "provider": "cpu"}


def own(first=1, lengths=(32, 32, 64, 64), limit=6, ttft=40):
    return [learn.Observation(index, lengths, limit, 12.0, ttft) for index in (first, first + 1)]


def measured(payload, deadline, interrupted):
    incumbent = {"output_digest": "a" * 64, "execution_shape": "b" * 64, "correct": 4, "requests": 4,
                 "tokens_per_second": 10, "median_ttft_ms": 10, "p95_ttft_ms": 12,
                 "budget_applied": int(bool(payload["incumbent"]))}
    candidate = {**incumbent, "tokens_per_second": 11, "execution_shape": "c" * 64,
                 "budget_applied": int(bool(payload["candidate"]))}
    orders = [["incumbent", "candidate"], ["candidate", "incumbent"]] * 2
    return [{"order": order, "incumbent": dict(incumbent), "candidate": dict(candidate)} for order in orders]


def args(tmp_path):
    return dict(binding=binding(), gguf=Path("fixture.gguf"), state=tmp_path, deadline=time.monotonic() + 30)


def test_requires_recurring_own_native_load_and_ignores_external_inputs():
    b = binding()
    assert learn.select(own()[:1], binding=b) is None
    assert learn.select(own()[:1] * 3, binding=b) is None
    assert learn.select(own(lengths=(33, 35)), binding=b) is None
    assert learn.select(own(lengths=(32, 64)), binding=b)["samples"] == 2
    assert learn.select(own(lengths=(64,) * 4), binding=b)["samples"] == 2
    assert learn.select({"schema": "ironmule.compare.v1", "runs": [], "arms": []}, binding=b) is None
    assert learn.select([*own(), {"competitor": "untrusted"}], binding=b) == learn.select(own(), binding=b)
    assert learn.select([replace(row, ttft_p95_ms=float("nan")) for row in own()], binding=b) is None
    assert learn.select(own(lengths=(510,) * 4), binding=b) is None
    assert learn.select(own(), binding=b) == learn.select(list(reversed(own())), binding=b)
    rows = [*own(), *own(7, lengths=(48,) * 4, ttft=160)]
    assert learn.select(rows, binding=b)["prompt_lengths"] == [48] * 4


@pytest.mark.parametrize("outcome", ["win", "quality", "throughput_margin", "first_token_regression",
                                     "unchanged_execution", "invalid_measurement"])
def test_only_qualified_own_holdout_becomes_native_rule(tmp_path, monkeypatch, outcome):
    def child(payload, *limits):
        assert payload["candidate"] == 32 and payload["incumbent"] == 0
        pairs = measured(payload, *limits)
        b = pairs[-1]["candidate"]
        if outcome == "quality":
            b["output_digest"] = "f" * 64
        if outcome == "throughput_margin":
            b["tokens_per_second"] = 10.29
        if outcome == "first_token_regression":
            b["p95_ttft_ms"] = 13
        if outcome == "unchanged_execution":
            b["execution_shape"] = pairs[-1]["incumbent"]["execution_shape"]
        if outcome == "invalid_measurement":
            b["tokens_per_second"] = float("nan")
        return 0, json.dumps(pairs)

    monkeypatch.setattr(learn, "_run_child", child)
    kw = args(tmp_path)
    result = learn.respond(own(), **kw)
    rules = learn.read_rules(tmp_path, kw["binding"])
    assert rules == ({result["workload_key"]: 32} if outcome == "win" else {})
    if outcome == "invalid_measurement":
        assert result["status"] == "deferred" and result["reason"] == outcome
        return
    assert result["status"] == ("qualified" if outcome == "win" else "retained_incumbent")
    assert result["reasons"] == ([] if outcome == "win" else [outcome])
    assert len(result["holdout"]) == 4
    monkeypatch.setattr(learn, "_run_child", lambda *v: pytest.fail("counted the identical measurements twice"))
    assert learn.respond(own(), **kw) == result
    assert learn.read_rules(tmp_path, {**kw["binding"], "machine": "another"}) == {}


def test_competitor_qualification_history_cannot_seed_or_suppress_local_trials(tmp_path, monkeypatch):
    kw = args(tmp_path)
    b = kw["binding"]
    key = learn.shape([32, 32, 64, 64], 6)
    legacy = tmp_path / "competitor-response" / learn.digest(b)
    legacy.mkdir(parents=True)
    (legacy / "qualified.json").write_text(json.dumps({"binding": b, "budgets": {key: 32}}))
    stray = learn.rule_directory(tmp_path, b)
    (tmp_path / "self-admission").mkdir(mode=0o700)
    stray.mkdir(mode=0o700)
    (stray / "qualified.json").write_text((legacy / "qualified.json").read_text())
    (stray / "rival-loss.json").write_text(json.dumps({"schema": "ironmule.competitor-response.v1",
        "candidate_budget": 32, "incumbent_budget": 0, "workload_key": key}))
    assert learn.read_rules(tmp_path, b) == {}

    def child(payload, *limits):
        pairs = measured(payload, *limits)
        pairs[0]["candidate"]["tokens_per_second"] = 1
        return 0, json.dumps(pairs)

    monkeypatch.setattr(learn, "_run_child", child)
    first = learn.respond(own(), **kw)
    assert first["candidate_budget"] == 32 and first["incumbent_budget"] == 0
    second = learn.respond(own(first=4), **kw)
    assert second["candidate_budget"] == 64 and second["incumbent_budget"] == 0
    assert learn.respond(own(first=7), **kw)["reason"] == "candidate budgets already measured for this workload"
    assert learn.read_rules(tmp_path, b) == {}


def test_qualified_action_is_the_next_independent_incumbent(tmp_path, monkeypatch):
    seen = []
    def child(payload, *limits):
        seen.append((payload["incumbent"], payload["candidate"]))
        return 0, json.dumps(measured(payload, *limits))
    monkeypatch.setattr(learn, "_run_child", child)
    kw = args(tmp_path)
    learn.respond(own(), **kw)
    result = learn.respond(own(first=4), **kw)
    assert seen == [(0, 32), (32, 64)]
    assert learn.read_rules(tmp_path, kw["binding"])[result["workload_key"]] == 64


def test_own_holdout_reaps_child_on_foreground_traffic_and_does_not_publish(tmp_path, monkeypatch):
    real = subprocess.Popen
    interrupted = threading.Event()
    children = []
    def child(command, **kwargs):
        proc = real([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
        children.append(proc)
        return proc
    monkeypatch.setattr(learn.subprocess, "Popen", child)
    timer = threading.Timer(0.2, interrupted.set)
    timer.start()
    try:
        result = learn.respond(own(), interrupted=interrupted.is_set, **args(tmp_path))
    finally:
        timer.cancel()
    assert children and children[0].poll() is not None
    assert result["status"] == "deferred" and result["reason"] == "foreground work resumed"
    assert not list(tmp_path.rglob("workload-*.json"))


def test_deadline_and_changed_binding_do_not_publish(tmp_path, monkeypatch):
    kw = args(tmp_path)
    kw["deadline"] = time.monotonic() - 1
    assert learn.respond(own(), **kw)["reason"] == "local trial budget exhausted"
    kw["deadline"] = time.monotonic() + 1
    monkeypatch.setattr(learn, "_run_child", lambda *a: (3, '{"binding_mismatch":["machine","settings"]}'))
    assert learn.respond(own(), **kw)["binding_mismatch"] == ["machine", "settings"]
    assert not list(tmp_path.rglob("workload-*.json"))


def test_replay_uses_same_cpu_pilot_decision_with_isolated_weights_and_visits():
    from types import SimpleNamespace
    from ironmule.pilot_learning import ThreadPolicy

    original = {"2:4:2:0": -1}
    policy = ThreadPolicy(SimpleNamespace(cores=4), original)
    assert policy.choose(128, 0, 4, 2) == 3
    assert original == policy.weights and policy.weights is not original
    assert policy.choose(10, 0, 4, 2) == 4
