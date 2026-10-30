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
import threading
import time
from types import SimpleNamespace

import pytest

import ironmule_product.backend as backend_module
from ironmule_product.backend import MLXWorkerClient
from ironmule_product.device_pool import DevicePool, start_pool
from ironmule_product.state import ProductStore
from ironmule_product.types import GenerationRequest, ModelSpec
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


@pytest.mark.parametrize("plan", ["h1", "h2", "h3", "h4", "h5", "h6"])
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
                      rl2.ordered_service())
    saved = []
    try:
        policy = rl2.LearnedPolicy([0.0, 0.0, 0.0, 0.0, 0.0, 5.0, -5.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
        report = rl2.measure(pool, spec.model_id, deadline_s=30, traces=2, requests=3, max_tokens=(2, 3), plan=plan,
                             names=(rl2.arms_h6() if plan == "h6" else
                                    rl2.arms_h4("spill/w8/g0/sjf") if plan in ("h4", "h5") else None),
                             policy=policy if plan in ("h4", "h5", "h6") else None,
                             on_row=lambda partial: saved.append(len(partial["rows"])))
    finally:
        pool.close()
    expected = (rl2.arms_h3(2) if plan == "h3" else rl2.arms_h6() if plan == "h6" else
                rl2.arms_h4("spill/w8/g0/sjf") if plan in ("h4", "h5") else
                ["shortest/w8/g0", "shortest/w1/g0", "fill/w1/g0", "fill/w8/g0"])
    assert report["arms"] == expected
    assert report["calibration"]["ok"] == 32 and report["calibration"]["capacity_rps"] > 0
    assert report["calibration"]["outcomes"] == {"ok": 32}
    assert len(report["rows"]) == 2 * {"h5": 1, "h6": 2}.get(plan, 3) * (len(expected) + 1) == saved[-1]
    assert all(r["outcome"] == "ok" for row in report["rows"] for r in row["requests"])
    result = rl2.analyse(report, resamples=50)
    assert result["valid"] and result["mismatched_requests"] == 0 and result["complete_traces"] == [0, 1]
    assert set(result["decisions"]) == ({"joint", "placement", "order"} if plan in ("h3", "h4", "h5")
                                        else {"joint", "placement", "schedule", "order"} if plan == "h6"
                                        else {"joint", "placement", "schedule"})
    assert (result["learned"] is not None) == (plan in ("h4", "h5", "h6"))
    assert (report["policy"] is not None) == (plan in ("h4", "h5", "h6"))
    if plan == "h6":
        assert set(result["learned"]["fixed_vs_reference"]) == set(rl2.arms_h6()[1:4])
    assert not any(prompt in json.dumps(report) for prompt in rl2.PROMPTS) and "Background" not in json.dumps(report)
    assert report["plan"] == plan and report.get("states") == {"h1": None, "h5": ["spike"], "h6": ["bursts", "spike"]
                                                                }.get(plan, list(rl2.H2_STATES))


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


def test_run_three_combines_length_placement_and_order_at_the_reference_schedule():
    assert rl2.arms_h3(2) == ["shortest/w8/g0", "shortest/w8/g0/sjf", "segregate/w8/g0", "segregate/w8/g0/sjf",
                              "spill/w8/g0", "spill/w8/g0/sjf"]
    assert rl2.arms_h3(1) == ["shortest/w8/g0", "shortest/w8/g0/sjf"]
    assert rl2.arm_order("spill/w8/g0/sjf") == "sjf" and rl2.arm_order("spill/w8/g0") == "fifo"
    assert rl2.parse_arm("spill/w8/g0/sjf") == ("spill", 8, 0)


def test_order_is_judged_as_its_own_decision():
    names = tuple(rl2.arms_h3(2)[:4])

    def cost(level, t, name):  # only order matters: sjf under the spike, fifo when quiet
        return 1.0 if (rl2.arm_order(name) == "sjf") == (level == 2) else 2.0

    result = rl2.analyse(_report(cost, names=names))
    assert set(result["decisions"]) == {"joint", "placement", "order"}
    assert result["decisions"]["order"]["verdict"].startswith("headroom")
    assert result["decisions"]["placement"]["verdict"].startswith("no headroom")


class _Request(SimpleNamespace):
    pass


def test_segregate_sends_long_requests_to_the_last_gpu_and_spill_leaves_a_crowded_one():
    pool = rl2._placement_pool(DevicePool([_Shard(0), _Shard(0)], ["0", "1"]))
    pool.placement = "segregate"
    pool._local.request = _Request(max_tokens=256)
    assert pool._order()[0] == 1
    pool._local.request = _Request(max_tokens=32)
    assert pool._order()[0] == 0
    crowded = rl2._placement_pool(DevicePool([_Shard(0), _Shard(rl2.SPILL_GAP)], ["0", "1"]))
    crowded._local.request = _Request(max_tokens=256)
    crowded.placement = "segregate"
    assert crowded._order()[0] == 1  # strict: stays on the long card however crowded
    crowded.placement = "spill"
    assert crowded._order()[0] == 0
    crowded.shards[1].load = rl2.SPILL_GAP - 1
    assert crowded._order()[0] == 1


def test_sjf_starts_the_smallest_output_limit_first_and_fifo_keeps_arrival_order(tmp_path, monkeypatch):
    from ironmule_product.serving_policy import ServingPolicy

    real_popen = subprocess.Popen

    def fake_popen(args, *popen_args, **popen_kwargs):
        if len(args) > 3 and str(args[3]).endswith("worker.py"):
            args = [sys.executable, "-u", str(FAKE), *args[4:]]
        return real_popen(args, *popen_args, **popen_kwargs)

    monkeypatch.setattr(backend_module.subprocess, "Popen", fake_popen)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    store = ProductStore(tmp_path / "state")
    store.setup("server")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    spec = ModelSpec("local/test", "revision", str(snapshot.resolve()), 1)
    store.register_model(spec)
    service_type = rl2.ordered_service()
    backend = MLXWorkerClient(spec, execution_variant="current_engine")
    backend.start()
    service = service_type(store, backend=backend, spec=spec)
    service.policy = ServingPolicy(max_width=1, source="test")  # one request at a time: order is visible
    starts = {}
    try:
        always_sjf = rl2.LearnedPolicy([0.0] * rl2.PLACEMENT_FEATURES, [10.0, 0.0, 0.0, 0.0])
        never_sjf = rl2.LearnedPolicy([0.0] * rl2.PLACEMENT_FEATURES, [-10.0, 0.0, 0.0, 0.0])
        for order in ("sjf", "fifo", "learned-sjf", "learned-fifo"):
            service.order = order.split("-")[0]
            service.learned_policy = always_sjf if order == "learned-sjf" else never_sjf
            ends: list[int] = []
            hold = GenerationRequest("local/test", (("user", "slow:0.05"),), max_tokens=12, stream=True)
            waiting = [GenerationRequest("local/test", (("user", f"q{limit}"),), max_tokens=limit, stream=True)
                       for limit in (9, 3, 6)]

            def consume(request):
                for event in service.stream(request):
                    if event.get("type") == "done":
                        ends.append(request.max_tokens)

            threads = [threading.Thread(target=consume, args=(hold,))]
            threads[0].start()
            time.sleep(0.2)  # the holding request is running; the rest queue behind it
            for request in waiting:
                threads.append(threading.Thread(target=consume, args=(request,)))
                threads[-1].start()
                time.sleep(0.02)
            for thread in threads:
                thread.join(30)
            starts[order] = ends
    finally:
        service.close()
    assert starts["sjf"] == starts["learned-sjf"] == [12, 3, 6, 9]
    assert starts["fifo"] == starts["learned-fifo"] == [12, 9, 3, 6]


class _LiveShard(_Shard):
    """A shard whose admitted sessions the learned placement reads."""

    def __init__(self, sessions):
        super().__init__(len(sessions))
        self._lock = threading.Lock()
        self._sessions = {f"s{i}": SimpleNamespace(request=request) for i, request in enumerate(sessions)}


def _live(max_tokens, content="q"):
    return GenerationRequest("local/test", (("user", content),), max_tokens=max_tokens)


def test_the_learned_placement_reads_each_cards_admitted_requests():
    segregating = rl2.LearnedPolicy([0.0, 0.0, 0.0, 0.0, 0.0, 5.0, -5.0, 0.0, 0.0, 0.0], [0.0] * rl2.ORDER_FEATURES)
    pool = rl2._placement_pool(DevicePool([_LiveShard([]), _LiveShard([_live(32)])], ["0", "1"]))
    pool.placement, pool.learned_policy = "learned", segregating
    pool._local.request = _live(256)
    assert pool._order()[0] == 1  # long to the last card although it is busier
    pool._local.request = _live(32)
    assert pool._order()[0] == 0
    assert rl2._View.of(_live(8, "x" * (rl2.LONG_PROMPT_CHARS + 1))).long_prompt
    assert not rl2._View.of(_live(8, "short")).long_prompt


def test_run_four_compares_the_learned_policy_with_the_best_fixed_arm():
    names = ("shortest/w8/g0", "spill/w8/g0/sjf", rl2.LEARNED)

    def cost(level, t, name):
        return {rl2.LEARNED: 1.0, "spill/w8/g0/sjf": 1.5}.get(name, 2.0)

    result = rl2.analyse(_report(cost, names=names))
    assert result["learned"]["rival"] == "spill/w8/g0/sjf"
    assert abs(result["learned"]["gain"] - 1 / 3) < 1e-9
    assert result["learned"]["verdict"] == "the learned policy beats the best fixed arm"
    even = rl2.analyse(_report(lambda level, t, name: 1.0 if name != "shortest/w8/g0" else 2.0, names=names))
    assert even["learned"]["verdict"] == "the learned policy does not beat the best fixed arm"


def test_a_run_four_kernel_carries_the_frozen_policy_and_refuses_a_tampered_one(tmp_path):
    commit = "0123456789abcdef0123456789abcdef01234567"
    policy = rl2.LearnedPolicy([0.5] * rl2.PLACEMENT_FEATURES, [0.25] * rl2.ORDER_FEATURES).to_json()
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy))
    folder = tmp_path / "kernel"
    assert rl2.main(["--write-kernel", str(folder), "--owner", "someone", "--commit", commit, "--plan", "h4",
                     "--best-fixed", "spill/w8/g0/sjf", "--policy-json", f"@{path}"]) == 0
    script = (folder / "run.py").read_text()
    assert policy["sha256"] in script and "'--best-fixed', 'spill/w8/g0/sjf'" in script
    assert json.loads((folder / "kernel-metadata.json").read_text())["id"] == "someone/ironmule-rl2-headroom-h4"
    compile(script, "run.py", "exec")
    policy["order"][0] = 3.0
    path.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="digest"):
        rl2.main(["--write-kernel", str(tmp_path / "bad"), "--owner", "someone", "--commit", commit, "--plan", "h4",
                  "--best-fixed", "spill/w8/g0/sjf", "--policy-json", f"@{path}"])


def _x1_device(alone=("a",) * 3, grouped=("a",) * 3, moved=()):
    sweep = [{"prompt": p, "own_capacity": 64, "digest": f"d{p}",
              "at": {str(c): {"equal": (p, c) not in moved, "first_difference": 9 if (p, c) in moved else None,
                              "max_logit_difference": 0.0, "digest": f"d{p}"} for c in rl2.X1_CAPACITIES}}
             for p in range(3)]
    return {"probe": rl2.X1_PROBE, "alone": list(alone), "grouped": list(grouped), "sweep": sweep, "long_capacity": 1088}


def test_x1_names_the_kv_capacity_when_padding_moves_a_token():
    verdict = rl2.x1_summary({"0": _x1_device(grouped=("b",) * 3, moved={(1, 1088)}), "1": _x1_device()})
    assert verdict["verdict"].startswith("the KV capacity moves tokens")
    assert verdict["capacity_moves"] == [["0", 1, "1088"]] and verdict["group_moves"] == ["0"]


def test_x1_separates_the_group_the_card_nondeterminism_and_a_failed_device():
    assert rl2.x1_summary({"0": _x1_device(grouped=("b",) * 3)})["verdict"].startswith("the group moves tokens")
    assert rl2.x1_summary({"0": _x1_device(alone=("a", "b", "a"))})["verdict"].startswith("nondeterministic")
    other = _x1_device()
    other["sweep"][0]["digest"] = "elsewhere"
    assert rl2.x1_summary({"0": _x1_device(), "1": other})["verdict"] == "the cards answer differently"
    assert rl2.x1_summary({"0": _x1_device(), "1": {"failed": "exit 1"}})["verdict"].startswith("incomplete")
    clean = rl2.x1_summary({"0": _x1_device(), "1": _x1_device()}, {"a": "shortest arms"})
    assert clean["verdict"].startswith("not reproduced") and clean["run3"]["0"]["alone"] == ["shortest arms"] * 3


def test_an_x1_kernel_embeds_its_own_source_for_the_per_device_children(tmp_path):
    commit = "0123456789abcdef0123456789abcdef01234567"
    folder = rl2.write_kernel(tmp_path / "kernel", "someone", commit, "x1")
    assert json.loads((folder / "kernel-metadata.json").read_text())["id"] == "someone/ironmule-rl2-headroom-x1"
    script = (folder / "run.py").read_text()
    assert "'--plan', 'x1'" in script
    compile(script, "run.py", "exec")
    embedded = script.split("\n_EMBEDDED_SOURCE = ", 1)[1].split("\n", 1)[0]
    assert eval(embedded, {"__import__": __import__}) == Path(rl2.__file__).read_text()
    other = rl2.write_kernel(tmp_path / "h3", "someone", commit, "h3") / "run.py"
    assert "\n_EMBEDDED_SOURCE = " not in other.read_text()


def test_rl4_s_replays_only_spike_traffic_on_traces_no_earlier_run_or_training_used(tmp_path):
    assert rl2.H2_STATES[rl2.H5_STATE] == "spike" and len(rl2.H5_TRACES) == 30
    earlier = set(range(rl2.H2_TRACES))
    assert not earlier & set(rl2.H5_TRACES)
    seeds = {2000 * rl2.H5_STATE + t for t in rl2.H5_TRACES}  # trace_h2's seed rule
    from tools import rl4_agent as rl4
    assert max(seeds) < rl4.TRAIN_SEED0  # training, validation and held-out simulator seeds start above
    commit = "0123456789abcdef0123456789abcdef01234567"
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps(rl2.LearnedPolicy.zero().to_json()))
    folder = tmp_path / "kernel"
    assert rl2.main(["--write-kernel", str(folder), "--owner", "someone", "--commit", commit, "--plan", "h5",
                     "--best-fixed", "shortest/w8/g0/sjf", "--policy-json", f"@{policy}"]) == 0
    assert json.loads((folder / "kernel-metadata.json").read_text())["id"] == "someone/ironmule-rl2-headroom-h5"
    assert "'--plan', 'h5'" in (folder / "run.py").read_text()


def test_a_width_policy_sets_each_groups_width_in_the_scheduler():
    import collections
    import queue

    from ironmule_product.serving_policy import ServingPolicy

    service = object.__new__(rl2.ordered_service())
    service._lock, service._carry, service._pending = threading.Lock(), collections.deque(), queue.Queue()
    service.policy = ServingPolicy(max_width=8, source="test")
    service.backend = SimpleNamespace(batch_capacity=8, can_batch_requests=lambda requests: True)
    service.order = "learned"
    sessions = [SimpleNamespace(request=_live(32), submitted_ns=time.monotonic_ns() + i) for i in range(6)]
    for choice, expected in ((1, 1), (4, 4), (8, 6)):
        policy = rl2.LearnedPolicy.zero(widths=True)
        policy.width[rl2.WIDTH_CHOICES.index(choice) * rl2.ORDER_FEATURES] = 5.0
        service.learned_policy = policy
        service._carry.extend(sessions[1:])
        group = service._group_for(sessions[0])
        assert len(group) == expected, choice
        service._carry.clear()
    service.learned_policy = rl2.LearnedPolicy.zero()  # a v1 policy: the arm's width
    service._carry.extend(sessions[1:])
    assert len(service._group_for(sessions[0])) == 6


def test_ld1_v_runs_the_products_own_learned_dispatch_against_its_reference(tmp_path, monkeypatch):
    from ironmule_product import learned_dispatch as ld

    real_popen = subprocess.Popen

    def fake_popen(args, *popen_args, **popen_kwargs):
        if len(args) > 3 and str(args[3]).endswith("worker.py"):
            args = [sys.executable, "-u", str(FAKE), *args[4:]]
        return real_popen(args, *popen_args, **popen_kwargs)

    monkeypatch.setattr(backend_module.subprocess, "Popen", fake_popen)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    store = ProductStore(tmp_path / "state")
    store.setup("server")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    spec = ModelSpec("local/test", "revision", str(snapshot.resolve()), 1)
    store.register_model(spec)
    policy = rl2.LearnedPolicy([0.0, 0.0, 0.0, 0.0, 0.0, 5.0, -5.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                               [0.0] * (len(rl2.WIDTH_CHOICES) * rl2.ORDER_FEATURES)).to_json()
    dispatch = ld.LearnedDispatch(ld.Policy.from_json(policy), {"experiment": "test"}, None)
    pool = start_pool(store, spec, ["0", "1"],
                      lambda device: MLXWorkerClient(spec, execution_variant="current_engine", cuda_device=device),
                      rl2.ordered_service())
    try:
        with pytest.raises(ValueError, match="active learned dispatch"):
            rl2.measure(pool, spec.model_id, deadline_s=30, traces=1, requests=2, max_tokens=(2,), plan="h7",
                        dispatch=ld.LearnedDispatch(None, None, "off"))
        report = rl2.measure(pool, spec.model_id, deadline_s=30, traces=2, requests=3, max_tokens=(2, 3), plan="h7",
                             dispatch=dispatch)
    finally:
        pool.close()
    assert report["arms"] == rl2.arms_h7() and report["states"] == list(rl2.H2_STATES)
    assert len(report["rows"]) == 2 * 3 * (len(rl2.arms_h7()) + 1)
    product_rows = [row for row in report["rows"] if row["arm"] == rl2.PRODUCT]
    assert product_rows and all(row["dispatch"]["active"] for row in product_rows)
    assert product_rows[-1]["dispatch"]["decisions"]["placement"] > 0
    result = rl2.analyse(report, resamples=50)
    assert result["valid"] and result["learned"]["rival"] in rl2.arms_h7()[:1] + rl2.arms_h7()[2:]
    assert set(result["learned"]["per_state"]) == {"0", "1", "2"} and "vs_reference" in result["learned"]
    product_rows[0]["dispatch"] = {"active": False}
    assert "without an active learned dispatch" in rl2.analyse(report, resamples=10)["problems"][0]


def test_h8_runs_both_engines_pools_by_round_and_judges_the_dense_plan_against_todays_default(tmp_path, monkeypatch):
    real_popen = subprocess.Popen

    def fake_popen(args, *popen_args, **popen_kwargs):
        if len(args) > 3 and str(args[3]).endswith("worker.py"):
            args = [sys.executable, "-u", str(FAKE), *args[4:]]
        return real_popen(args, *popen_args, **popen_kwargs)

    monkeypatch.setattr(backend_module.subprocess, "Popen", fake_popen)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    store = ProductStore(tmp_path / "state")
    store.setup("server")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    spec = ModelSpec("local/test", "revision", str(snapshot.resolve()), 1)
    store.register_model(spec)
    started = []

    def start(engine):
        started.append(engine)
        return start_pool(store, spec, ["0", "1"], lambda device: MLXWorkerClient(
            spec, execution_variant="current_engine", engine_configuration=rl2.ENGINE_CONFIGURATIONS[engine],
            cuda_device=device), rl2.ordered_service())

    report = rl2.measure_h8(start, spec.model_id, deadline_s=30, traces=[55, 56], rounds=2, requests=3,
                            max_tokens=(2, 3))
    assert started == ["tuned", "dense", "dense", "tuned"]  # the first engine alternates by round
    assert {row["engine"] for row in report["rows"]} == {"tuned", "dense"}
    assert all(row["execution"] == "numeric@dense" for row in report["rows"] if row["engine"] == "dense")
    assert len(report["rows"]) == 2 * 3 * (2 + 3)  # per block: tuned A/A pair; dense w8, w8/sjf, w8 again
    result = rl2.analyse_h8(report, resamples=50)
    assert result["valid"], result
    assert result["best_dense"] in rl2.arms_h8()[1:] and set(result["(2) no harm in any state"]["per_state"]) == {"0", "1", "2"}
    assert all(e["warmup_ok"] == 16 and e["warmup_failed"] == 0 and len(e["engine"]) == 2
               for rounds in report["engines"].values() for e in rounds)  # eight per card, both engines
    saved = dict(report["rows"][0]["requests"][0])
    report["rows"][0]["requests"][0].update(outcome="failed", error="backend_unavailable")  # run 1's defect
    problems = rl2.analyse_h8(report, resamples=50)["problems"]
    assert problems == [f"1 requests failed: {{'{report['rows'][0]['engine']}': {{'backend_unavailable': 1}}}}"]
    report["rows"][0]["requests"][0] = {}  # a client that ended without an outcome counts too
    assert "no outcome" in rl2.analyse_h8(report, resamples=50)["problems"][0]
    report["rows"][0]["requests"][0] = saved
    report["rows"][-1]["execution"] = "exact"  # a dense cell that did not run the plan invalidates the run
    for row in report["rows"]:
        if row["engine"] == "dense":
            row["execution"] = "exact"
            break
    assert not rl2.analyse_h8(report, resamples=50)["valid"]


def test_h8_stops_when_a_pool_answers_no_warm_up_request(monkeypatch):
    class Pool:
        devices = ["0"]

        @property
        def shards(self):
            return [self]

        def health(self):
            return {"execution": "numeric@dense", "ready": False}

        def close(self):
            pass

    monkeypatch.setattr(rl2, "_placement_pool", lambda pool: pool)
    monkeypatch.setattr(rl2, "replay", lambda pool, model, items, deadline_s: [
        {"index": i, "outcome": "failed", "error": "backend_unavailable"} for i in range(len(items))])
    report = rl2.measure_h8(lambda engine: Pool(), "m", deadline_s=30, traces=[55], rounds=1, requests=3,
                            max_tokens=(2, 3))
    assert report["stopped"] == "the tuned pool failed a warm-up request" and not report["rows"]
    assert report["stopped_health"] == [{"execution": "numeric@dense", "ready": False}]
    assert not rl2.analyse_h8(report, resamples=10)["valid"]


def test_the_dense_preflight_reports_why_the_plan_did_not_install(tmp_path):
    result = rl2.dense_preflight(str(tmp_path / "no-snapshot"), None, timeout_s=300)
    assert result["ok"] is False and result["error"]


def test_kp1_runs_the_default_and_its_class_a_knobs_by_round_and_judges_them_as_fixed(tmp_path, monkeypatch):
    pytest.importorskip("mlx.core")  # the fake worker reports the baseline knobs from ironmule.runtime
    from ironmule_product.knob_profile import kp1_knobs

    real_popen = subprocess.Popen
    commands = []

    def fake_popen(args, *popen_args, **popen_kwargs):
        if len(args) > 3 and str(args[3]).endswith("worker.py"):
            commands.append(list(args))
            args = [sys.executable, "-u", str(FAKE), *args[4:]]
        return real_popen(args, *popen_args, **popen_kwargs)

    monkeypatch.setattr(backend_module.subprocess, "Popen", fake_popen)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("FAKE_REPORT_KNOBS", "1")
    store = ProductStore(tmp_path / "state")
    store.setup("server")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    spec = ModelSpec("local/test", "revision", str(snapshot.resolve()), 1)
    store.register_model(spec)
    knobs = kp1_knobs()

    def start(engine):
        extra = {"engine_knobs": {"knobs": knobs, "source": "KP1-candidate"}} if engine == "kp" else {}
        return start_pool(store, spec, ["0", "1"], lambda device: MLXWorkerClient(
            spec, execution_variant="current_engine", cuda_device=device, **extra), rl2.ordered_service())

    def stress(round_, capacity):  # the shape of `trace_stress`, small enough for the fake
        return [{**item, "max_tokens": 2 + i} for i, item in enumerate(rl2.trace_stress(round_, capacity)[:3])]

    report = rl2.measure_h8(start, spec.model_id, deadline_s=30, traces=[63, 64], rounds=2, requests=3,
                            max_tokens=(2, 3), names=rl2.arms_kp1(), engines=("tuned", "kp"), plan="kp1",
                            stress=stress)
    assert sum("--engine-knobs" in c for c in commands) == 2 * 2  # two cards, kp's pool in both rounds
    assert len(report["rows"]) == 2 * 3 * 4 and len(report["stress_rows"]) == 2 * 2  # A/A for both engines
    from ironmule.runtime import BASELINE

    assert all(card["selected_knobs"] == (knobs if engine == "kp" else BASELINE.as_dict())
               for engine, rounds in report["engines"].items() for r in rounds for card in r["engine"])
    report["preflight"] = {"ok": True}
    result = rl2.analyse_kp1(report, resamples=50)
    assert result["valid"], result
    assert set(result["(2) no harm in any state"]["per_state"]) == {"0", "1", "2"}
    assert result["stress_cost"]["kp"] is not None and result["second_token_s"]["kp"] is not None
    request = report["rows"][-1]["requests"][0]
    saved = dict(request)
    request["digest"] = "0" * 64  # one answer that differs anywhere kills KP1
    assert result["kill 2 (answers differ on the product path)"] is False
    assert rl2.analyse_kp1(report, resamples=50)["kill 2 (answers differ on the product path)"] is True
    report["rows"][-1]["requests"][0] = saved
    report["engines"]["kp"][0]["engine"][0]["selected_knobs"] = knobs | {"compiled_fixed_cache": False}
    assert "a kp pool did not run its arm's knobs" in rl2.analyse_kp1(report, resamples=50)["problems"]
    report["engines"]["kp"][0]["engine"][0]["selected_knobs"] = knobs
    del report["preflight"]
    assert "the preflight did not pass (kill 1)" in rl2.analyse_kp1(report, resamples=50)["problems"]


def test_the_kp1_stress_cell_holds_more_capacities_than_the_engine_keeps_bodies():
    items = rl2.trace_stress(0, 1.0)
    assert sorted(item["max_tokens"] for item in items) == list(rl2.STRESS_MAX_TOKENS)
    capacities = {((24 + item["max_tokens"] + 8 + 63) // 64) * 64 for item in items}
    assert len(capacities) == 16 > 8 and not any(item["long"] for item in items)


def test_h9_runs_three_engines_on_the_same_traffic_and_judges_both_parts_as_fixed(tmp_path, monkeypatch):
    pytest.importorskip("mlx.core")  # the fake worker reports the baseline knobs from ironmule.runtime
    from ironmule_product.knob_profile import kp1_knobs

    real_popen = subprocess.Popen

    def fake_popen(args, *popen_args, **popen_kwargs):
        if len(args) > 3 and str(args[3]).endswith("worker.py"):
            args = [sys.executable, "-u", str(FAKE), *args[4:]]
        return real_popen(args, *popen_args, **popen_kwargs)

    monkeypatch.setattr(backend_module.subprocess, "Popen", fake_popen)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("FAKE_REPORT_KNOBS", "1")
    store = ProductStore(tmp_path / "state")
    store.setup("server")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    spec = ModelSpec("local/test", "revision", str(snapshot.resolve()), 1)
    store.register_model(spec)
    started = []

    def start(engine):
        started.append(engine)
        extra = {"engine_knobs": {"knobs": kp1_knobs(), "source": "KP1-candidate"}} if engine == "kp" else {}
        return start_pool(store, spec, ["0", "1"], lambda device: MLXWorkerClient(
            spec, execution_variant="current_engine", engine_configuration=rl2.ENGINE_CONFIGURATIONS[engine],
            cuda_device=device, **extra), rl2.ordered_service())

    report = rl2.measure_h8(start, spec.model_id, deadline_s=30, traces=[75, 76, 77], rounds=3, requests=3,
                            max_tokens=(2, 3), names=rl2.arms_h9(), engines=("tuned", "kp", "dense"), plan="h9")
    assert started == ["tuned", "kp", "dense", "kp", "dense", "tuned", "dense", "tuned", "kp"]  # a Latin square
    assert len(report["rows"]) == 3 * 3 * 6  # per block: an A/A pair for each of the three engines
    report.update(preflight={"ok": True}, dense_preflight={"ok": True})
    result = rl2.analyse_h9(report, resamples=50)
    assert result["kp"]["valid"] and result["dense"]["valid"], (result["kp"].get("problems"), result["dense"].get("problems"))
    assert result["dense_reference"] == ("kp:shortest/w2/g0/sjf" if result["kp_ships"] else "tuned:shortest/w2/g0/sjf")
    assert result["dense"]["best_dense"] == "dense:shortest/w8/g0/sjf"
    assert set(result["dense_against_each"]) == {"tuned:shortest/w2/g0/sjf", "kp:shortest/w2/g0/sjf"}
    report["dense_preflight"] = {"ok": False}
    assert not rl2.analyse_h9(report, resamples=50)["dense"]["valid"]


def test_h10_measures_the_dense_plan_with_slot_reuse_and_requires_that_joins_happened(tmp_path, monkeypatch):
    real_popen = subprocess.Popen

    def fake_popen(args, *popen_args, **popen_kwargs):
        if len(args) > 3 and str(args[3]).endswith("worker.py"):
            args = [sys.executable, "-u", str(FAKE), *args[4:]]
        return real_popen(args, *popen_args, **popen_kwargs)

    monkeypatch.setattr(backend_module.subprocess, "Popen", fake_popen)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    store = ProductStore(tmp_path / "state")
    store.setup("server")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    spec = ModelSpec("local/test", "revision", str(snapshot.resolve()), 1)
    store.register_model(spec)

    def start(engine):
        return start_pool(store, spec, ["0", "1"], lambda device: MLXWorkerClient(
            spec, execution_variant="current_engine", engine_configuration=rl2.ENGINE_CONFIGURATIONS[engine],
            cuda_device=device), rl2.ordered_service())

    report = rl2.measure_h8(start, spec.model_id, deadline_s=30, traces=[87, 88], rounds=2, requests=6,
                            max_tokens=(2, 12), names=rl2.arms_h10(), engines=("tuned", "dense"), plan="h10")
    assert all("joined" in row for row in report["rows"])
    assert sum(row["joined"] for row in report["rows"] if row["engine"] == "tuned") == 0  # join stays off there
    report["preflight"] = {"ok": True}
    result = rl2.analyse_h10(report, resamples=50)
    assert result["joined_per_state"][1] + result["joined_per_state"][2] > 0, result["joined_per_state"]
    assert result["valid"], result.get("problems")
    assert result["recommended_width"] in ("w4", "w8") and "w4_over_w8" in result
    for row in report["rows"]:
        row["joined"] = 0  # a run in which slot reuse never engaged measures the wrong thing
    assert "slot reuse did not engage" in " ".join(rl2.analyse_h10(report, resamples=50)["problems"])


def test_h11_serves_long_prompts_through_both_engines_and_requires_slots_that_hold_the_whole_context(tmp_path,
                                                                                                    monkeypatch):
    real_popen = subprocess.Popen

    def fake_popen(args, *popen_args, **popen_kwargs):
        if len(args) > 3 and str(args[3]).endswith("worker.py"):
            args = [sys.executable, "-u", str(FAKE), *args[4:]]
        return real_popen(args, *popen_args, **popen_kwargs)

    monkeypatch.setattr(backend_module.subprocess, "Popen", fake_popen)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    store = ProductStore(tmp_path / "state")
    store.setup("server")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    spec = ModelSpec("local/test", "revision", str(snapshot.resolve()), 1)
    store.register_model(spec)

    def start(engine):
        return start_pool(store, spec, ["0", "1"], lambda device: MLXWorkerClient(
            spec, execution_variant="current_engine", engine_configuration=rl2.ENGINE_CONFIGURATIONS[engine],
            cuda_device=device), rl2.ordered_service())

    assert rl2.prompt_text({"prompt": 2, "long": True}) == rl2.CONTEXT + "\n\n" + rl2.PROMPTS[2]  # as H10 sent it
    assert all(sum(1 for item in rl2.trace_h11(level, 99, 2.0) if item.get("context")) == 2 for level in range(3))

    def measure():
        report = rl2.measure_h8(start, spec.model_id, deadline_s=30, traces=[99, 100], rounds=2, requests=6,
                                max_tokens=(2, 12), names=rl2.arms_h11(), engines=("tuned", "dense"), plan="h11",
                                traffic=rl2.trace_h11,
                                warm_extra=[{"offset_s": 0.0, "prompt": 0, "context": 7, "max_tokens": 8}])
        report["preflight"] = {"ok": True}
        return report

    report = measure()
    assert all(e["warmup_ok"] == 18 for rounds in report["engines"].values() for e in rounds)  # the long one too
    result = rl2.analyse_h11(report, resamples=50)
    assert result["valid"], result.get("problems")
    assert result["dense_slots"] == [8192] and result["long_requests"]["dense"]["ok"] > 0
    assert "recommended_up_to_context_limit" in result
    monkeypatch.setenv("FAKE_DENSE_SLOTS", "2048")  # H10's slots reject the long requests (alone, protocol 2)
    problems = " ".join(rl2.analyse_h11(measure(), resamples=50)["problems"])
    assert "slots were [2048]" in problems and "long requests were not all answered" in problems
