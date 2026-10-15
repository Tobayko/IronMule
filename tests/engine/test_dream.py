"""Model-free checks of the experiment planner (RSI1): Rust state, persistence, replay, the gate.

They need the planner library (`cargo build --release --manifest-path
native/experiment_planner/Cargo.toml`); without it they are skipped, and a skip is not a pass.
"""

from __future__ import annotations

import itertools
import json

import pytest

from ironmule import dream
from ironmule.runtime import BASELINE

pytestmark = pytest.mark.skipif(dream._lib() is None, reason="experiment planner library not built")


def _world(gain, bonus=None, unsupported=(), depth=2, seconds=10.0):
    """A recorded tree over every configuration up to ``depth`` changes from stock."""
    bonus = bonus or {}
    root = BASELINE.as_dict()
    changes = [(k, v) for k, values in dream.SPACE for v in values]
    nodes = {dream._key(root): {"knobs": root, "parent": None, "change": None, "depth": 0,
                                "status": "ok", "ratio": 1.0, "seconds": 0.0, "order": 0}}
    for size in range(1, depth + 1):
        for combo in itertools.combinations(changes, size):
            if len({k for k, _ in combo}) < size:
                continue
            names = {f"{k}={v}" for k, v in combo}
            ratio = 1.0
            for name in names:
                ratio *= 1.0 - gain.get(name, 0.0)
            ratio *= 1.0 - sum(b for pair, b in bonus.items() if pair <= names)
            bad = bool(names & set(unsupported))
            nodes[dream._key(dict(root, **dict(combo)))] = {
                "knobs": dict(root, **dict(combo)), "parent": dream._key(dict(root, **dict(combo[:-1]))),
                "change": list(combo[-1]), "depth": size, "status": "unsupported" if bad else "ok",
                "ratio": None if bad else ratio, "seconds": seconds, "order": len(nodes)}
    return {"schema": dream.SCHEMA_TREE, "device_class": "test", "nodes": nodes}


def _family(seed):
    # head skip pays a lot; readback 16, outside tune.SEARCH, pays as well; wired never runs here.
    return _world({"head_skip_prefill=True": 0.30 + seed / 100, "readback_every=16": 0.08,
                   "compiled_fixed_cache=True": 0.02}, unsupported=("wired_fraction=0.6",))


def test_planner_learns_incrementally_and_persists(tmp_path, monkeypatch):
    monkeypatch.setattr(dream, "STORE", tmp_path)
    planner, envelope = dream.load_planner("cls", dream.DEFAULT_CONFIG)
    assert envelope["observations"] == 0
    before = planner.status("readback_every=16")
    planner.observe("readback_every=16", -0.08, 12.0)
    after = planner.status("readback_every=16")
    assert after["observations"] == 1 and after["posterior_mean"] < before["posterior_mean"] < 1e-12
    assert after["posterior_sd"] < before["posterior_sd"] and after["cost_s"] == 12.0
    envelope["observations"] = 1
    dream.save_planner(planner, envelope)
    restored, stored = dream.load_planner("cls", dream.DEFAULT_CONFIG)
    assert restored.status("readback_every=16") == after and stored["observations"] == 1
    # corrupt or foreign state starts fresh instead of trusting it
    data = json.loads(dream.state_path("cls").read_text())
    data["state_sha256"] = "0" * 64
    dream.state_path("cls").write_text(json.dumps(data))
    fresh, envelope = dream.load_planner("cls", dream.DEFAULT_CONFIG)
    assert fresh.status("readback_every=16")["observations"] == 0 and envelope["observations"] == 0
    with pytest.raises(ValueError):
        planner.observe("readback_every=16", float("nan"), 1.0)


def test_replay_is_deterministic_and_prefix_only():
    tree = _family(0)
    first = dream.replay(tree, "planner")
    assert first == dream.replay(tree, "planner")
    assert first["opened"] <= len(tree["nodes"]) - 1
    root = next(k for k, n in tree["nodes"].items() if n["parent"] is None)
    planner = dream.Planner(dream.DEFAULT_CONFIG)
    chosen = dream.pick(planner, {root: tree["nodes"][root]}, allowed=tree["nodes"])
    assert chosen is not None and chosen[0] == root  # only the root's children are eligible first


def test_coordinate_replay_keeps_to_the_tune_search():
    result = dream.replay(_family(0), "coordinate")
    # it finds head skip and compiled cache but never tries readback 16
    assert result["best_ratio"] == pytest.approx(0.70 * 0.98, rel=1e-6)


def test_taught_planner_beats_fixed_and_random_order_held_out():
    trees = [_family(seed) for seed in range(4)]
    report = dream.held_out(trees, dream.DEFAULT_CONFIG, random_seeds=5)
    mean = report["mean_utility"]
    assert mean["planner"] > mean["coordinate"] and mean["planner"] > mean["random"]
    # experience transfers: wired_fraction failed in the other trees, so it is not tried again
    planned = dream.replay(trees[0], "planner", experience=trees[1:])
    assert planned["best_ratio"] < 0.70 * 0.98


def test_the_simple_rule_wins_when_the_planner_does_not_beat_it(tmp_path, monkeypatch):
    monkeypatch.setattr(dream, "STORE", tmp_path)
    dream.tree_dir().mkdir(parents=True)
    for seed in range(2):
        (dream.tree_dir() / f"t{seed}.json").write_text(json.dumps(_family(seed)))
    monkeypatch.setattr(dream, "held_out", lambda trees, config, random_seeds=20: {
        "rows": [], "mean_utility": {"planner": 1.0, "coordinate": 2.0, "random": 0.0}})
    assert dream.dream("test", log=lambda _line: None)["enabled"] is False
    assert dream.available("test") is False
    monkeypatch.setattr(dream, "held_out", lambda trees, config, random_seeds=20: {
        "rows": [], "mean_utility": {"planner": 3.0, "coordinate": 2.0, "random": 0.0}})
    assert dream.dream("test", log=lambda _line: None)["enabled"] is True
    assert dream.available("test") is True


def _crashing_child(monkeypatch, tmp_path, exit_code):
    """tune._screen_in_child with a child that checkpoints one completed test, then dies."""
    import importlib
    import subprocess
    import sys
    import time

    tune = importlib.import_module("ironmule.tune")

    monkeypatch.setattr(dream, "STORE", tmp_path)
    root, done = BASELINE.as_dict(), dict(BASELINE.as_dict(), head_skip_prefill=True)
    pending = dict(done, capacity_slack=64)
    nodes = {dream._key(root): {"knobs": root, "parent": None, "change": None, "depth": 0, "status": "ok",
                                "ratio": 1.0, "seconds": 0.0, "order": 0},
             dream._key(done): {"knobs": done, "parent": dream._key(root), "change": ["head_skip_prefill", True],
                                "depth": 1, "status": "ok", "ratio": 0.8, "seconds": 5.0, "order": 1}}
    checkpoint = {
        "pending": {"knobs": pending, "parent": dream._key(done), "change": ["capacity_slack", 64],
                    "depth": 2, "order": 2},
        "at": time.time(),
        "screen": {"prompt_tokens": 3, "base": {"total_ns": 10}, "best": done, "best_result": {"total_ns": 8},
                   "trials": [{"knob": "head_skip_prefill", "value": True, "ratio": 0.8, "depth": 1,
                               "disposition": "accepted", "verdict": "ok"}], "tree": "t"},
        "tree": {"schema": dream.SCHEMA_TREE, "device_class": "cls", "planner_config": dream.DEFAULT_CONFIG,
                 "explorer": "planner", "nodes": nodes},
    }
    death = "os.abort()" if exit_code is None else f"raise SystemExit({exit_code})"
    child = (f"import os; open(os.environ[{dream.CHECKPOINT_ENV!r}], 'w').write({json.dumps(checkpoint)!r})\n"
             + death)
    real = subprocess.Popen
    monkeypatch.setattr(tune.subprocess, "Popen",
                        lambda _args, **kwargs: real([sys.executable, "-c", child], **kwargs))
    return tune, done, pending


def test_a_native_abort_costs_the_pending_test_not_the_tune(tmp_path, monkeypatch):
    tune, done, pending = _crashing_child(monkeypatch, tmp_path, exit_code=None)
    screen = tune._screen_in_child("m", "p", 8, 1, None, {"class": "cls", "skip": []})
    assert screen["best"] == done and screen["best_result"] == {"total_ns": 8}
    assert [t["disposition"] for t in screen["trials"]] == ["accepted", "crashed"]
    tree = json.loads((dream.tree_dir() / "t.json").read_text())
    assert tree["nodes"][dream._key(pending)]["status"] == "crashed"
    assert tree["nodes"][dream._key(done)]["status"] == "ok"  # nothing completed is lost
    planner, envelope = dream.load_planner("cls", dream.DEFAULT_CONFIG)
    try:
        assert envelope["observations"] == 1 and envelope["log"][-1]["status"] == "crashed"
        assert planner.status("capacity_slack=64")["failures"] == 1
    finally:
        planner.close()


def test_a_python_error_in_the_screening_child_stays_loud(tmp_path, monkeypatch):
    tune, _, _ = _crashing_child(monkeypatch, tmp_path, exit_code=1)
    with pytest.raises(RuntimeError, match="exited with status 1"):
        tune._screen_in_child("m", "p", 8, 1, None, {"class": "cls", "skip": []})
    assert not dream.tree_dir().exists() or not list(dream.tree_dir().iterdir())
