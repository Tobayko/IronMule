"""RL2-H: the headroom plan, its analysis and one run through two pinned fake workers (no model, no GPU).

The analysis cases use scripted costs; the end-to-end case drives `stab1_fake_worker.py` through the
real `MLXWorkerClient`, `ProductService` and device pool, so arms, traces and digests are checked on
the real scheduler. Nothing here measures speed.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import ironmule_product.backend as backend_module
from ironmule_product.backend import MLXWorkerClient
from ironmule_product.device_pool import DevicePool, start_pool
from ironmule_product.service import ProductService
from ironmule_product.state import ProductStore
from ironmule_product.types import ModelSpec
from tools import rl2_headroom as rl2

FAKE = Path(__file__).with_name("stab1_fake_worker.py")


def test_the_arms_are_the_learner_catalogue_times_placement_with_the_reference_first():
    one, two = rl2.arms(1), rl2.arms(2)
    assert one[0] == two[0] == rl2.REFERENCE == "shortest/w8/g0"
    assert len(one) == len(set(one)) == 10 and len(two) == len(set(two)) == 20
    assert "shortest/w1/g0" in one and not any(name.startswith("shortest/w1/g") and name != "shortest/w1/g0"
                                               for name in one)  # width 1 never waits
    assert not any(name.startswith("fill/") for name in one)
    assert rl2.parse_arm("fill/w4/g150") == ("fill", 4, 150)


def test_every_arm_replays_the_same_trace_from_a_fixed_seed():
    first, again, other = rl2.trace(1, 2, 3.0), rl2.trace(1, 2, 3.0), rl2.trace(1, 3, 3.0)
    assert first == again and first != other and len(first) == rl2.REQUESTS
    offsets = [item["offset_s"] for item in first]
    assert offsets[0] == 0 and offsets == sorted(offsets)
    assert {item["max_tokens"] for item in first} <= set(rl2.MAX_TOKENS)


def test_a_block_brackets_every_arm_with_the_reference_and_rotates():
    names = rl2.arms(1)
    blocks = [rl2.block_order(names, block) for block in range(3)]
    for order in blocks:
        assert order[0] == (rl2.REFERENCE, 0) and order[-1] == (rl2.REFERENCE, 1)
        assert sorted(name for name, _ in order[1:-1]) == sorted(names[1:])
    assert blocks[0][1:-1] != blocks[1][1:-1] != blocks[2][1:-1]


def test_the_cost_is_the_live_learner_reward_and_a_failure_costs_more_than_the_deadline():
    assert rl2.request_cost({"outcome": "ok", "first_s": 0.5, "end_s": 2.0}, 120) == 2.5
    assert rl2.request_cost({"outcome": "ok", "first_s": None, "end_s": 2.0}, 120) == 4.0
    for outcome in ("rejected", "timeout", "failed"):
        assert rl2.request_cost({"outcome": outcome, "first_s": None, "end_s": 0.1}, 120) == 241.0


def _report(cost, names=("shortest/w8/g0", "shortest/w1/g0"), traces=4, aa=1.0, digest=lambda *k: "d"):
    """Rows whose every request costs `cost(level, trace, arm)` (end only), the reference's
    second run `aa` times its first."""
    rows = []
    for t in range(traces):
        for level in range(3):
            for name, run in [(names[0], 0), *((n, 0) for n in names[1:]), (names[0], 1)]:
                value = cost(level, t, name) * (aa if run else 1.0)
                requests = [{"index": i, "outcome": "ok", "first_s": 0.0, "end_s": value,
                             "digest": digest(level, t, name, i)} for i in range(2)]
                rows.append({"level": level, "trace": t, "arm": name, "run": run, "requests": requests})
    return {"arms": list(names), "rates": [1, 2, 3], "deadline_s": 120, "calibration": {"capacity_rps": 1},
            "rows": rows}


def test_one_setting_best_everywhere_leaves_no_headroom_and_ships_it():
    result = rl2.analyse(_report(lambda level, t, name: 1.0 if name.endswith("w1/g0") else 2.0 + 0.01 * t))
    assert result["valid"] and result["best_fixed"] == "shortest/w1/g0"
    joint = result["decisions"]["joint"]
    assert joint["headroom_in_sample"] == 0 and joint["headroom_held_out"] == 0
    assert result["verdict"] == "no headroom: the fixed policy shortest/w1/g0 ships, no learner"
    assert result["best_fixed_vs_reference"] < 1


def test_levels_that_want_different_settings_show_headroom_above_the_noise():
    def cost(level, t, name):  # light load wants width 1, heavy load width 8
        wide = name.endswith("w8/g0")
        return (1.0 if not wide else 2.0) if level == 0 else (1.0 if wide else 3.0)

    result = rl2.analyse(_report(cost, aa=1.01))
    joint = result["decisions"]["joint"]
    assert result["valid"] and abs(result["aa_noise"] - 0.01 / 1.005) < 1e-9
    assert joint["oracle"] == {"0": "shortest/w1/g0", "1": "shortest/w8/g0", "2": "shortest/w8/g0"}
    assert joint["headroom_held_out"] > 0.2 and joint["headroom_held_out_ci95"][0] > result["aa_noise"]
    assert result["verdict"] == "headroom for: joint"


def test_noise_larger_than_the_headroom_releases_no_learner():
    def cost(level, t, name):
        return 1.0 if (level == 0) == name.endswith("w1/g0") else 1.05

    result = rl2.analyse(_report(cost, aa=1.5))
    assert result["valid"] and result["decisions"]["joint"]["headroom_held_out"] > 0
    assert result["verdict"] == ("no headroom: the reference stays (shortest/w8/g0 is within the A/A noise of it), "
                                 "no learner")


def test_placement_and_schedule_are_judged_apart_with_two_gpus():
    names = ("shortest/w8/g0", "shortest/w1/g0", "fill/w8/g0", "fill/w1/g0")

    def cost(level, t, name):  # only placement matters: fill at light load, shortest at heavy load
        return 1.0 if name.startswith("fill") == (level == 0) else 2.0

    result = rl2.analyse(_report(cost, names=names))
    assert set(result["decisions"]) == {"joint", "placement", "schedule"}
    assert result["decisions"]["placement"]["verdict"].startswith("headroom")
    assert result["decisions"]["schedule"]["verdict"].startswith("no headroom")


def test_answers_that_differ_between_arms_or_too_few_traces_invalidate_the_run():
    differs = rl2.analyse(_report(lambda *k: 1.0, digest=lambda level, t, name, i: name))
    assert not differs["valid"] and "answered differently" in differs["problems"][0]
    assert differs["verdict"] == "invalid: no learner is released"
    short = rl2.analyse(_report(lambda *k: 1.0, traces=1))
    assert not short["valid"] and "complete traces" in short["problems"][0]
    failed = rl2.analyse({**_report(lambda *k: 1.0), "calibration": {"failed": "fewer than half"}})
    assert not failed["valid"]


class _Shard:
    def __init__(self, load):
        self.load, self.spec, self.store = load, "spec", SimpleNamespace(root=None)
        self.stability_log = self.serving_log = None

    def accepting(self):
        return True

    def outstanding(self):
        return self.load


def test_fill_places_on_the_lowest_gpu_with_room_and_otherwise_the_shortest_queue():
    pool = rl2._placement_pool(DevicePool([_Shard(3), _Shard(1)], ["0", "1"]))
    assert pool._order() == [1, 0]  # shortest, HW1's reference
    pool.placement, pool.fill_width = "fill", 4
    assert pool._order() == [0, 1]
    pool.fill_width = 3
    assert pool._order() == [1, 0]
    pool.fill_width = 1
    assert pool._order() == [1, 0]  # no room anywhere: the shortest queue


@pytest.mark.parametrize("plan", ["h1", "h2"])
def test_a_small_run_on_two_fake_workers_measures_every_arm_and_keeps_no_text(tmp_path, monkeypatch, plan):
    real_popen = subprocess.Popen

    def fake_popen(args, *popen_args, **popen_kwargs):
        if len(args) > 3 and str(args[3]).endswith("worker.py"):
            args = [sys.executable, "-u", str(FAKE), *args[4:]]
        return real_popen(args, *popen_args, **popen_kwargs)

    monkeypatch.setattr(backend_module.subprocess, "Popen", fake_popen)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(rl2, "WIDTHS", (1, 8))
    monkeypatch.setattr(rl2, "GATHERS_MS", (0,))
    store = ProductStore(tmp_path / "state")
    store.setup("server")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    spec = ModelSpec("local/test", "revision", str(snapshot.resolve()), 1)
    store.register_model(spec)
    pool = start_pool(store, spec, ["0", "1"],
                      lambda device: MLXWorkerClient(spec, execution_variant="current_engine", cuda_device=device),
                      ProductService)
    saved = []
    try:
        report = rl2.measure(pool, spec.model_id, deadline_s=30, traces=2, requests=3, max_tokens=(2, 3), plan=plan,
                             on_row=lambda partial: saved.append(len(partial["rows"])))
    finally:
        pool.close()
    assert report["arms"] == ["shortest/w8/g0", "shortest/w1/g0", "fill/w1/g0", "fill/w8/g0"]
    assert report["calibration"]["ok"] == 32 and report["calibration"]["capacity_rps"] > 0
    assert report["calibration"]["outcomes"] == {"ok": 32}
    assert len(report["rows"]) == 2 * 3 * (4 + 1) == saved[-1]
    assert all(r["outcome"] == "ok" for row in report["rows"] for r in row["requests"])
    result = rl2.analyse(report, resamples=50)
    assert result["valid"] and result["mismatched_requests"] == 0 and result["complete_traces"] == [0, 1]
    assert set(result["decisions"]) == {"joint", "placement", "schedule"}
    assert not any(prompt in json.dumps(report) for prompt in rl2.PROMPTS) and "Background" not in json.dumps(report)
    assert report["plan"] == plan and (report.get("states") == list(rl2.H2_STATES)) == (plan == "h2")


def test_run_two_keeps_every_shortest_schedule_and_fill_only_without_a_wait():
    two = rl2.arms_h2(2)
    assert two[0] == rl2.REFERENCE and len(two) == len(set(two)) == 14
    assert [n for n in two if n.startswith("fill/")] == ["fill/w1/g0", "fill/w2/g0", "fill/w4/g0", "fill/w8/g0"]
    assert set(rl2.arms(1)) <= set(two) and rl2.arms_h2(1) == rl2.arms(1)


def test_run_two_traffic_is_quiet_bursty_or_a_spike_with_mixed_lengths():
    quiet = rl2.trace_h2(0, 1, 2.0)
    assert quiet == [{**item, "long": False} for item in rl2.trace(0, 101, 0.4)]
    for state in (1, 2):
        traces = [rl2.trace_h2(state, index, 2.0) for index in range(5)]
        assert traces[0] == rl2.trace_h2(state, 0, 2.0) and traces[0] != traces[1]
        for items in traces:
            offsets = [item["offset_s"] for item in items]
            assert len(items) == rl2.REQUESTS and offsets[0] == 0 and offsets == sorted(offsets)
            assert {item["max_tokens"] for item in items} <= set(rl2.H2_MAX_TOKENS)
        flat = [item for items in traces for item in items]
        assert {item["long"] for item in flat} == {True, False}
        assert {item["max_tokens"] for item in flat} == set(rl2.H2_MAX_TOKENS)
    assert all(item["offset_s"] <= 1.0 for index in range(5) for item in rl2.trace_h2(2, index, 2.0))
    assert max(item["offset_s"] for item in rl2.trace_h2(1, 0, 2.0)) > 1.0  # bursts spread out in time
    assert len(rl2.CONTEXT.split()) > 600


def test_a_kernel_folder_pins_one_commit(tmp_path):
    commit = "0123456789abcdef0123456789abcdef01234567"
    folder = rl2.write_kernel(tmp_path / "kernel", "someone", commit)
    metadata = json.loads((folder / "kernel-metadata.json").read_text())
    assert metadata["id"] == "someone/ironmule-rl2-headroom" and metadata["is_private"] is True
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    script = (folder / "run.py").read_text()
    assert f"@{commit}" in script and "rl2_headroom_report.json" in script
    compile(script, "run.py", "exec")
    second = rl2.write_kernel(tmp_path / "kernel2", "someone", commit, "h2")
    assert json.loads((second / "kernel-metadata.json").read_text())["id"] == "someone/ironmule-rl2-headroom-h2"
    assert "'--plan', 'h2'" in (second / "run.py").read_text()
    for bad in ("main", commit.upper()):
        try:
            rl2.write_kernel(tmp_path / "bad", "someone", bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")
