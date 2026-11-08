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
from training import rl2_headroom as rl2

FAKE = Path(__file__).resolve().parents[1] / "engine" / "stab1_fake_worker.py"  # shared with the engine suite


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
    from training import rl4_agent as rl4
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
    assert result["dense_slots"] == [8192] and result["long_outcomes"]["dense"].get("ok", 0) > 0
    assert "recommended_up_to_context_limit" in result
    assert set(result["short_and_long"]["dense"]["0"]) == {"short", "long"} and result["offered_work_per_state"]
    timed_out = next(r for row in report["rows"] if row["engine"] == "dense" for r in row["requests"] if r["context"])
    timed_out.update(outcome="timeout", digest=None)  # a long request that times out is a cost, not a defect
    assert rl2.analyse_h11(report, resamples=50)["valid"]
    monkeypatch.setenv("FAKE_DENSE_SLOTS", "2048")  # H10's slots reject the long requests (alone, protocol 2)
    problems = " ".join(rl2.analyse_h11(measure(), resamples=50)["problems"])
    assert "slots were [2048]" in problems and "long requests did not all reach an outcome" in problems


# ---- AG1, V: the agent learning online on hardware, then frozen against the reference ----

def _ag1_report(agent_factor: float, aa_factor: float = 1.0, mismatch: bool = False, failed: bool = False,
                fixed_factor: float | None = None):
    rows, block = [], 0
    arms = list(rl2.AG1_ARMS) if fixed_factor is None else ["reference", "fixed", "agent"]
    for trace in (143, 144, 145, 146):
        for level, base in enumerate((2.0, 15.0, 30.0)):
            for name, run in rl2.block_order(arms, block):
                factor = (agent_factor if name == "agent" else fixed_factor if name == "fixed" else
                          (aa_factor if run == 1 else 1.0))
                requests = [{"outcome": "ok", "first_s": base * factor * 0.2 * (1 + 0.01 * (trace % 3)),
                             "end_s": base * factor * (1 + 0.01 * (trace % 3)), "digest": f"d{i}"} for i in range(4)]
                if mismatch and name == "agent":
                    requests[0] = {**requests[0], "digest": "other"}
                if failed and name == "agent":
                    requests[1] = {**requests[1], "outcome": "failed"}
                rows.append({"level": level, "trace": trace, "arm": name, "run": run, "block": block,
                             "requests": requests})
            block += 1
    learning = [{"agent": {"serving": "reference"}}, {"agent": {"serving": "learned:abc"}}]
    return {"plan": "ag1", "deadline_s": 120, "rows": rows, "learning": learning, "arms": arms,
            "frozen": {"policy_sha256": "abc"}, "agent_final": {"counters": {"qualified": 1}, "decision_us": 7.5}}


def test_ag1_v2_reports_three_tiers_default_fixed_agent():
    result = rl2.analyse_ag1(_ag1_report(0.8, fixed_factor=0.9), resamples=200)
    tiers = result["tiers"]
    assert result["valid"] and result["verdict"] == "the agent beats the reference"
    assert tiers["fixed over the default"]["gain"] == pytest.approx(0.1, abs=0.002)
    assert tiers["agent over the default"]["gain"] == pytest.approx(0.2, abs=0.002)
    assert tiers["agent over fixed"]["gain"] == pytest.approx(1 - 0.8 / 0.9, abs=0.002)
    unqualified = {**_ag1_report(1.0, fixed_factor=0.9), "frozen": None}
    result = rl2.analyse_ag1(unqualified, resamples=200)
    assert result["verdict"] == "nothing qualified online"
    assert result["tiers"]["fixed over the default"]["gain"] == pytest.approx(0.1, abs=0.002)


def test_ag1_passes_a_resolved_gain_and_records_the_online_phase():
    result = rl2.analyse_ag1(_ag1_report(0.9, aa_factor=1.002), resamples=200)
    assert result["valid"] and result["verdict"] == "the agent beats the reference"
    assert result["(1) the agent beats the reference"]["gain"] == pytest.approx(0.1, abs=0.002)
    assert result["aa_floor"] == pytest.approx(0.002, abs=1e-4)
    assert result["online"]["first_learned_trace"] == 1 and result["online"]["decision_us"] == 7.5


def test_ag1_a_gain_inside_the_aa_floor_or_a_different_answer_does_not_pass():
    assert rl2.analyse_ag1(_ag1_report(0.995, aa_factor=1.02), resamples=200)["verdict"] == \
        "the agent does not beat the reference"
    mismatched = rl2.analyse_ag1(_ag1_report(0.9, mismatch=True), resamples=200)
    assert mismatched["verdict"] == "invalid" and mismatched["mismatched_requests"] == 12
    assert rl2.analyse_ag1(_ag1_report(0.9, failed=True), resamples=200)["verdict"] == "invalid"


def test_ag1_without_a_qualified_schedule_reports_that_and_compares_nothing():
    report = {"plan": "ag1", "rows": [], "learning": [{"agent": {"serving": "reference"}}], "frozen": None,
              "stopped": "the agent qualified no schedule in the online phase; nothing to compare"}
    result = rl2.analyse_ag1(report, resamples=50)
    assert result["verdict"] == "nothing qualified online" and result["online"]["first_learned_trace"] is None


def test_ag1_kernel_runs_the_plan_at_the_pinned_commit(tmp_path):
    folder = rl2.write_kernel(tmp_path / "k", "owner", "a" * 40, plan="ag1")
    assert "'--plan', 'ag1'" in (folder / "run.py").read_text()
    assert json.loads((folder / "kernel-metadata.json").read_text())["id"] == "owner/ironmule-rl2-headroom-ag1"


def test_ag1_kernel_for_another_profile_is_its_own_kernel_and_runs_that_profile(tmp_path, capsys):
    assert rl2.main(["--write-kernel", str(tmp_path / "k"), "--owner", "owner", "--commit", "b" * 40,
                     "--plan", "ag1", "--profile", "t4b"]) == 0
    assert "'--profile', 't4b'" in (tmp_path / "k" / "run.py").read_text()
    assert json.loads((tmp_path / "k" / "kernel-metadata.json").read_text())["id"] == \
        "owner/ironmule-rl2-headroom-ag1-t4b"


def test_ag1_profiles_differ_only_where_backlog_ag1_says():
    t4, cpu = rl2.AG1_PROFILES["t4"], rl2.AG1_PROFILES["cpu"]
    assert t4["model"] == rl2.MODEL and t4["variant"] == "current_engine" and t4["pause_s"] == 120.0
    assert list(t4["learn"]) == list(range(107, 143)) and list(t4["evaluate"]) == list(range(143, 151))
    assert cpu == {"model": "mlx-community/SmolLM2-135M-Instruct-8bit", "variant": "reference",
                   "compute_dtype": "dequantize", "max_tokens": (4, 32), "learn": range(107, 122),
                   "evaluate": range(143, 147), "pause_s": 360.0, "timeout_s": 1800, "short_prompts": True}
    assert rl2.AG1_PROFILES["cpu2"] == {**cpu, "learn": range(151, 191), "evaluate": range(191, 195)}
    assert rl2.AG1_PROFILES["t4b"] == {**t4, "requests": 64, "learn": range(195, 219), "evaluate": range(219, 223),
                                       "pause_s": 480.0, "fixed": (2, True)}
    assert rl2.AG1_PROFILES["t4c"] == {**rl2.AG1_PROFILES["t4b"], "learn": range(223, 239),
                                       "evaluate": range(239, 242), "pause_s": 1200.0, "timeout_s": 900,
                                       "stop_on_qualify": True}
    used = [set(p["learn"]) | set(p["evaluate"]) for name, p in rl2.AG1_PROFILES.items() if name != "t4c"]
    assert not (set(rl2.AG1_PROFILES["t4c"]["learn"]) | set(rl2.AG1_PROFILES["t4c"]["evaluate"])) & set().union(*used)
    assert rl2.AG1_PROFILES["t4d"] == {**rl2.AG1_PROFILES["t4c"], "learn": range(242, 254),
                                       "evaluate": range(254, 256), "pause_s": 800.0, "fixed": (8, True)}
    used = [set(p["learn"]) | set(p["evaluate"]) for name, p in rl2.AG1_PROFILES.items() if name != "t4d"]
    assert not (set(rl2.AG1_PROFILES["t4d"]["learn"]) | set(rl2.AG1_PROFILES["t4d"]["evaluate"])) & set().union(*used)
    assert rl2.AG1_PROFILES["cpu3"] == {**cpu, "learn": range(256, 286), "evaluate": range(286, 290),
                                        "stop_on_qualify": True, "fixed": (1, True), "accelerator": "cpu"}
    used = [set(p["learn"]) | set(p["evaluate"]) for name, p in rl2.AG1_PROFILES.items() if name != "cpu3"]
    assert not (set(rl2.AG1_PROFILES["cpu3"]["learn"]) | set(rl2.AG1_PROFILES["cpu3"]["evaluate"])) & set().union(*used)


def test_ag1_cpu_profile_kernel_is_a_cpu_session_that_installs_the_cpu_build(tmp_path):
    assert rl2.main(["--write-kernel", str(tmp_path / "k"), "--owner", "owner", "--commit", "c" * 40,
                     "--plan", "ag1", "--profile", "cpu3"]) == 0
    metadata = json.loads((tmp_path / "k" / "kernel-metadata.json").read_text())
    assert metadata["id"] == "owner/ironmule-rl2-headroom-ag1-cpu3"
    assert metadata["enable_gpu"] is False and "machine_shape" not in metadata
    source = (tmp_path / "k" / "run.py").read_text()
    assert "ironmule[cpu] @ git+" in source and "ironmule[cuda]" not in source
    assert "'--profile', 'cpu3'" in source
    gpu = json.loads((rl2.write_kernel(tmp_path / "g", "owner", "c" * 40, plan="ag1") /
                      "kernel-metadata.json").read_text())
    assert gpu["enable_gpu"] is True and gpu["machine_shape"] == "NvidiaTeslaT4"


def test_ag2_cpu_profile_uses_real_prompts_one_limit_and_fresh_seeds():
    lp = rl2.AG1_PROFILES["cpu-lp"]
    assert lp == {**rl2.AG1_PROFILES["cpu3"], "max_tokens": (128,), "prompts": "dolly", "learn": range(290, 320),
                  "evaluate": range(320, 324), "short_prompts": False}
    used = [set(p["learn"]) | set(p["evaluate"]) for name, p in rl2.AG1_PROFILES.items() if name != "cpu-lp"]
    assert not (set(lp["learn"]) | set(lp["evaluate"])) & set().union(*used)


def test_ag2_items_carry_real_prompts_with_one_limit_and_held_out_prompts_are_unseen():
    prompts = [f"prompt {k}" for k in range(15_000)]
    items = rl2.trace_h2(1, 300, 0.2, 16)
    online = rl2.lp_items(items, prompts, 300, False, 128)
    held = rl2.lp_items(items, prompts, 320, True, 128)
    assert all(i["max_tokens"] == 128 and not i["long"] and rl2.prompt_text(i) == i["text"] for i in online + held)
    assert [i["offset_s"] for i in online] == [i["offset_s"] for i in items]
    assert {i["text"] for i in online} <= set(prompts[:12_000])
    assert not {i["text"] for i in held} & set(prompts[:12_000])
    assert rl2.lp_items(items, prompts, 300, False, 128) == online  # fixed by the trace index


def test_ag2_held_out_arm_orders_by_the_frozen_length_model():
    from ironmule_product import agent as ag
    from ironmule_product.learned_dispatch import LearnedDispatch, Policy
    from ironmule_product.length_model import LengthModel, features
    from ironmule_product.types import GenerationRequest

    lengths = LengthModel()
    for _ in range(300):
        for text, tokens in (("Answer yes or no: is it red?", 3), ("Write a long essay about the sea.", 400)):
            request = GenerationRequest("m", (("user", text),), max_tokens=512)
            lengths.learn(features(request), tokens)
    predicted = rl2._PredictingDispatch(LearnedDispatch(Policy.from_json(ag.fixed_policy(1, True, "predicted")),
                                                        {}, None), lengths)
    short = GenerationRequest("m", (("user", "Answer yes or no: is it red?"),), max_tokens=512)
    long = GenerationRequest("m", (("user", "Write a long essay about the sea."),), max_tokens=512)
    assert predicted.active and predicted.sjf([short, long]) is True
    assert predicted.order_key(short) < 20 < 200 < predicted.order_key(long)
    by_limit = rl2._PredictingDispatch(LearnedDispatch(Policy.from_json(ag.fixed_policy(1, True)), {}, None), lengths)
    assert by_limit.order_key(long) is None


def test_ag2_t4_gate_is_the_owners_gemma_under_its_native_plan_on_fresh_seeds():
    g12 = rl2.AG1_PROFILES["t4-g12-lp"]
    assert g12["model"] == "mlx-community/gemma-3-12b-it-4bit" and g12["variant"] == "reference"
    assert g12["compute_dtype"] == "native" and g12["max_tokens"] == (512,) and g12["prompts"] == "dolly"
    assert g12["fixed"] == (1, True) and g12["stop_on_qualify"] and g12["timeout_s"] == 1800 and g12["cards"] == 2
    assert g12.get("accelerator") != "cpu"
    used = [set(p["learn"]) | set(p["evaluate"]) for name, p in rl2.AG1_PROFILES.items() if name != "t4-g12-lp"]
    assert not (set(g12["learn"]) | set(g12["evaluate"])) & set().union(*used)


def test_ag2_full_t4_run_is_the_gate_with_the_sizes_it_fixed():
    v = rl2.AG1_PROFILES["t4-g12-v"]
    assert v == {**rl2.AG1_PROFILES["t4-g12-lp"], "max_tokens": (256,), "requests": 32, "learn": range(358, 398),
                 "evaluate": range(398, 400), "pause_s": 360.0}
    used = [set(p["learn"]) | set(p["evaluate"]) for name, p in rl2.AG1_PROFILES.items() if name != "t4-g12-v"]
    assert not (set(v["learn"]) | set(v["evaluate"])) & set().union(*used)


def test_ag4r_runs_three_arms_on_the_same_two_workers_and_judges_them(tmp_path, monkeypatch):
    """AG4-R: the single path forced, the agent's router and the stacked engine forced, on two fake workers that
    stand in for the single path and the stacked engine on one card."""
    from ironmule_product.device_pool import start_paths
    from ironmule_product.path_router import PathRouter
    from ironmule_product.service import ProductService

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
    setup = {"stacked": {"speed": {"short": {"base_step_s": 0.0173, "base_first_s": 0.06, "plan_first_s": 0.3,
                                             "plan_step_s": {"1": 0.021, "8": 0.028}}}}}
    router = PathRouter.from_setup(setup)
    makers = [lambda: MLXWorkerClient(spec, execution_variant="current_engine", cuda_device="0")] * 2  # the fake
    # worker speaks the current engine's protocol; on the card the first is the reference worker with NUM1's plan
    pool = start_paths(store, spec, makers, ["0:single", "0:stacked"], ProductService, router)
    saved = []
    try:
        report = rl2.measure_ag4r(pool, spec.model_id, router, None, 30.0, loads=(3.0,), traces=[0, 1], requests=3,
                                  limit=3, on_row=lambda partial: saved.append(len(partial["rows"])))
    finally:
        pool.close()
    assert report["calibration"]["ok"] == 16 and report["calibration"]["capacity_rps"] > 0
    assert len(report["rows"]) == 3 * 2 * 5 == saved[-1]  # 3 states x 2 traces x (single, 3 arms rotated, single)
    assert {row["arm"] for row in report["rows"]} == set(rl2.AG4R_ARMS)
    middles = [tuple(r["arm"] for r in report["rows"] if r["block"] == b)[1:4] for b in range(3)]
    assert len(set(middles)) == 3 and all(set(m) == {"router", "stacked", "frozen"} for m in middles)  # rotated
    assert report["router"]["decisions"]["single"] + report["router"]["decisions"]["stacked"] > 0
    assert report["frozen"]["frozen"] and all(path["requests"] == 0 for path in report["frozen"]["paths"].values())
    assert router.costs[0].seen + router.costs[1].seen > 0  # the router learned; the frozen copy did not
    result = rl2.analyse_ag4r(report, resamples=50)
    assert result["valid"] and result["complete_blocks"] == 6
    assert set(result["passes"]) == {"tokens_per_s", "ttft_no_worse"} and result["verdict"]
    assert set(result["learning"]["passes"]) == {"tokens_per_s", "ttft_no_worse"}
    assert result["learning"]["verdict"].startswith(("pass", "fail"))
    old = {**report, "arms": ["single", "router", "stacked"],
           "rows": [r for r in report["rows"] if r["arm"] != "frozen"]}  # a report from before the frozen arm
    assert rl2.analyse_ag4r(old, resamples=50)["complete_blocks"] == 6 and "learning" not in rl2.analyse_ag4r(old)


def test_ag4c_compares_the_agents_path_with_the_default_on_the_same_arrivals(tmp_path, monkeypatch):
    from ironmule_product.device_pool import start_paths
    from ironmule_product.service import ProductService

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
    makers = [lambda: MLXWorkerClient(spec, execution_variant="current_engine")] * 2
    pool = start_paths(store, spec, makers, ["cpu:default", "cpu:agent"], ProductService, rl2.ForcedRoute(0))
    try:
        report = rl2.measure_ag4c(pool, spec.model_id, 30.0, traces=[0, 1], requests=3, limit=3)
    finally:
        pool.close()
    assert report["calibration"]["ok"] == 3 and len(report["rows"]) == 3 * 2 * 3
    assert [row["arm"] for row in report["rows"][:3]] == ["default", "agent", "default"]
    result = rl2.analyse_ag4c(report, resamples=50)
    assert result["valid"] and result["complete_blocks"] == 6 and set(result["passes"]) == {"tokens_per_s",
                                                                                           "ttft_no_worse"}
    assert rl2.analyse_ag4c({"stopped": "the agent kept reference with None"})["verdict"].startswith("stopped")


def test_ag4c_run3_censors_only_the_default_paths_failures_and_only_in_its_favour():
    """AG4-C run 3: a default-path request that misses the deadline counts as its whole token limit, first token and
    last, at the deadline; an agent-path failure still voids its block."""
    def row(block, arm, run, requests):
        return {"block": block, "arm": arm, "run": run, "level": 1, "trace": block, "requests": requests}

    def req(first, end, offset=0.0, outcome="ok"):
        return ({"outcome": outcome, "first_s": first, "end_s": end, "offset_s": offset, "tokens": 16}
                if outcome == "ok" else {"outcome": outcome, "first_s": None, "end_s": 100.0, "offset_s": offset,
                                         "tokens": 0})

    slow = [req(50, 60), req(None, None, outcome="timeout")]
    fast = [req(2, 3), req(4, 5)]
    rows = [r for b in range(3) for r in (row(b, "default", 0, slow), row(b, "agent", 0, fast),
                                          row(b, "default", 1, slow))]
    assert rl2._cell(slow) is None
    cell = rl2._cell(slow, {"deadline_s": 100.0, "limit": 16})
    assert cell["tokens_per_s"] == 32 / 100 and cell["ttft_p95_s"] == 100.0
    censored = rl2.analyse_ag4c({"rows": rows, "censor_default": {"deadline_s": 100.0, "limit": 16}}, resamples=50)
    assert censored["valid"] and censored["complete_blocks"] == 3 and censored["censored_default_requests"] == 6
    assert censored["verdict"].startswith("pass")
    assert rl2.analyse_ag4c({"rows": rows}, resamples=50)["verdict"].startswith("invalid")  # runs 1-2
    broken = rows[:1] + [row(0, "agent", 0, [req(2, 3), req(None, None, outcome="timeout")])] + rows[2:]
    voided = rl2.analyse_ag4c({"rows": broken, "censor_default": {"deadline_s": 100.0, "limit": 16}}, resamples=50)
    assert voided["verdict"] == "invalid" and voided["complete_blocks"] == 2
    assert voided["subset"].startswith("descriptive only: 2 complete blocks of 3"), "a subset is labelled so"
    assert "subset" not in censored


def test_ag5g_runs_both_phases_with_swapped_cards_and_the_agents_width(tmp_path, monkeypatch):
    """AG5-G with every outside piece stubbed: the agent decides first, the default path and the agent's stacked
    engine (at the agent's width) share a phase on two cards, phase 2 swaps them, calibration runs once per phase,
    prompts are short, and the blocks are numbered across loads and phases."""
    import importlib

    import huggingface_hub

    from ironmule import plan_choice
    from ironmule_product import device_pool, state

    tune = importlib.import_module("ironmule.tune")  # `ironmule.tune` the package attribute is the function

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(tmp_path / "86cc6a8"))
    monkeypatch.setattr(tune, "resolve_local_model", lambda model, revision=None: SimpleNamespace(
        identity=SimpleNamespace(identity_sha256="e" * 64)))
    monkeypatch.setattr(plan_choice, "choose_setup", lambda model, fingerprint, identity: {
        "engine": "dense", "width": 6, "compute_dtype": None, "reason": "2.01x at width 6",
        "decision": {"tokens_per_second": {"by_width": {"6": 42.5, "8": 20.3}}}})
    monkeypatch.setattr(device_pool, "cuda_devices", lambda: ["0", "1"])
    monkeypatch.setattr(rl2, "_ironmule", lambda *a, **k: {"returncode": 0})

    class Store:
        def __init__(self, root):
            pass

        def set_request_timeout(self, seconds):
            self.timeout = seconds

        def model(self, model_id):
            return ModelSpec("local/test", "revision", str(tmp_path), 1)

    monkeypatch.setattr(state, "ProductStore", Store)
    pools, clients, replayed = [], [], []

    class Pool:
        def __init__(self, makers, labels):
            self.labels, self.router = labels, None
            clients.append([make() for make in makers])

        def health(self):
            return {"execution": "fake"}

        def close(self):
            pass

    def start_paths(store, spec, makers, labels, service, router):
        pools.append(Pool(makers, labels))
        return pools[-1]

    monkeypatch.setattr(device_pool, "start_paths", start_paths)

    def replay(pool, model, items, deadline_s):
        replayed.append((pool.labels[pool.router.index], [dict(item) for item in items]))
        return [{"outcome": "ok", "first_s": 0.1, "end_s": 1.0 + i, "tokens": 8} for i, _ in enumerate(items)]

    monkeypatch.setattr(rl2, "replay", replay)
    monkeypatch.setattr(rl2, "wait_idle", lambda pool, timeout_s=None: True)
    report = rl2.run_ag5g(rl2.AG5G_MODEL, tmp_path, smoke=False)
    assert "stopped" not in report and report["width"] == 6
    assert [pool.labels for pool in pools] == [["0:default", "1:agent"], ["1:default", "0:agent"]]
    default, agent = clients[0]
    assert default.compute_dtype == "native" and default.cuda_device == "0" and default.dense_width is None
    assert agent.engine_configuration == "dense_stacked" and agent.dense_width == 6 and agent.cuda_device == "1"
    assert clients[1][1].cuda_device == "0"
    calibrations = [labels for labels, items in replayed if all(item["offset_s"] == 0.0 for item in items)
                    and len(items) == rl2.AG5G_REQUESTS and labels.endswith("default")]
    assert len(report["phases"]) == 2 and len(calibrations) >= 2
    assert all(not item.get("long") for _, items in replayed for item in items), "short prompts only"
    blocks = sorted({row["block"] for row in report["rows"]})
    assert blocks == list(range(8)) and len(report["rows"]) == 8 * 3
    assert {(row["phase"], row["load"]) for row in report["rows"]} == {(p, l) for p in (0, 1) for l in rl2.AG5G_LOADS}
    analysis = rl2.analyse_ag5g(report, resamples=50)
    assert analysis["complete_blocks"] == 8 and analysis["width"] == 6
    assert set(analysis["recorded"]) == {"load 0.8", "load 3.0", "phase 0", "phase 1"}
    assert analysis["curve_tokens_per_s"] == {"6": 42.5, "8": 20.3}


def test_ag5g_stops_when_the_agent_does_not_choose_the_stacked_engine(tmp_path, monkeypatch):
    import importlib

    import huggingface_hub

    from ironmule import plan_choice
    from ironmule_product import device_pool

    tune = importlib.import_module("ironmule.tune")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(tmp_path / "86cc6a8"))
    monkeypatch.setattr(device_pool, "cuda_devices", lambda: ["0"])
    assert "two cards" in rl2.run_ag5g(rl2.AG5G_MODEL, tmp_path, smoke=False)["stopped"]
    monkeypatch.setattr(device_pool, "cuda_devices", lambda: ["0", "1"])
    monkeypatch.setattr(tune, "resolve_local_model", lambda model, revision=None: SimpleNamespace(
        identity=SimpleNamespace(identity_sha256="e" * 64)))
    monkeypatch.setattr(plan_choice, "choose_setup", lambda model, fingerprint, identity: {
        "engine": "reference", "width": None, "compute_dtype": "native", "reason": "the quality gate"})
    saved = []
    stopped = rl2.run_ag5g(rl2.AG5G_MODEL, tmp_path, smoke=False, on_row=lambda partial: saved.append(dict(partial)))
    assert "not the stacked engine" in stopped["stopped"]
    assert saved and saved[0]["setup"]["engine"] == "reference", "the decision is saved before anything else runs"
    assert rl2.analyse_ag5g(stopped)["verdict"].startswith("stopped")


def test_ag5g_records_a_path_that_does_not_start(tmp_path, monkeypatch):
    import importlib

    import huggingface_hub

    from ironmule import plan_choice
    from ironmule_product import device_pool, state

    tune = importlib.import_module("ironmule.tune")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(tmp_path / "86cc6a8"))
    monkeypatch.setattr(tune, "resolve_local_model", lambda model, revision=None: SimpleNamespace(
        identity=SimpleNamespace(identity_sha256="e" * 64)))
    monkeypatch.setattr(plan_choice, "choose_setup", lambda model, fingerprint, identity: {
        "engine": "dense", "width": 6, "compute_dtype": None, "reason": "2.01x at width 6"})
    monkeypatch.setattr(device_pool, "cuda_devices", lambda: ["0", "1"])
    monkeypatch.setattr(rl2, "_ironmule", lambda *a, **k: {"returncode": 0})
    monkeypatch.setattr(state, "ProductStore", lambda root: SimpleNamespace(
        set_request_timeout=lambda seconds: None, model=lambda model_id: ModelSpec("local/t", "r", str(tmp_path), 1)))

    def start_paths(*args, **kwargs):
        raise RuntimeError("worker exited before ready")

    monkeypatch.setattr(device_pool, "start_paths", start_paths)
    report = rl2.run_ag5g(rl2.AG5G_MODEL, tmp_path, smoke=False)
    assert "phase 0" in report["failed"] and "worker exited" in report["failed"] and report["setup"]["width"] == 6


def test_ag6_check_names_the_mode_the_dreamer_would_serve_and_scores_it_by_the_fixed_rule():
    """AG6's check on a constructed AG4-R report: a spike block, where the stacked engine was measured best, and a
    trickle block, where every arm is alike. The dreamer, from the setup's startup prices, names stacking for the
    spike and keeps routing for the trickle; both count as hits."""
    row = {"base_first_s": 0.1, "base_step_s": 0.0174, "plan_first_s": 0.1, "plan_step_s": {"1": 0.0229, "8": 0.034}}
    setup = {"stacked": {"speed": {"short": row, "long": dict(row)}}}

    def cell(block, arm, run, offsets, span):
        return {"block": block, "arm": arm, "run": run, "load": 0.8, "level": 2 if block == 0 else 0, "trace": 1,
                "requests": [{"offset_s": o, "end_s": span - o, "first_s": 0.2, "tokens": 200, "outcome": "ok"}
                             for o in offsets]}

    spike, trickle = [0.001 * i for i in range(32)], [30.0 * i for i in range(32)]
    rows = [cell(0, "single", 0, spike, 110.0), cell(0, "single", 1, spike, 110.0), cell(0, "router", 0, spike, 55.0),
            cell(0, "stacked", 0, spike, 30.0)]
    rows += [cell(1, arm, run, trickle, 935.0) for arm, run in (("single", 0), ("single", 1), ("router", 0),
                                                                 ("stacked", 0))]
    result = rl2.check_ag6({"setup": setup, "rows": rows})
    assert result["blocks"] == 2 and [r["named"] for r in result["rows"]] == ["stacked", "router"]
    assert result["named_mode_hits"] == 2 and result["share"] == 1.0
    assert set(result) >= {"replayed_stacked_over_single", "measured_stacked_over_single", "verdict"}


_AG6_SETUP = {"stacked": {"speed": {"short": {"base_step_s": 0.0173, "base_first_s": 0.06, "plan_first_s": 0.3,
                                              "plan_step_s": {"1": 0.021, "8": 0.028}}}}}


def _ag6h_pool(tmp_path, monkeypatch):
    from ironmule_product.device_pool import start_paths
    from ironmule_product.path_router import PathRouter, PolicyRouter
    from ironmule_product.service import ProductService

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
    router, agent = PathRouter.from_setup(_AG6_SETUP), PolicyRouter.from_setup(_AG6_SETUP)
    makers = [lambda: MLXWorkerClient(spec, execution_variant="current_engine", cuda_device="0")] * 2
    pool = start_paths(store, spec, makers, ["0:single", "0:stacked"], ProductService, router)
    return pool, spec, router, agent


def test_ag6h_runs_four_arms_trace_by_trace_and_records_the_agents_mode(tmp_path, monkeypatch):
    """AG6-H: the single path, the router, the learning agent and the stacked engine on the same two fake workers;
    every load and state of a trace before the next trace; the agent's mode around each of its cells."""
    pool, spec, router, agent = _ag6h_pool(tmp_path, monkeypatch)
    try:
        report = rl2.measure_ag6h(pool, spec.model_id, router, agent, None, 30.0, loads=(0.8, 3.0), levels=(1, 2),
                                  traces=[0, 1], requests=3, limit=3)
    finally:
        pool.close()
    assert len(report["rows"]) == 2 * 2 * 2 * 5 and "time_cut" not in report
    assert [(r["trace"], r["load"], r["level"]) for r in report["rows"] if r["arm"] == "single" and r["run"] == 0][:4] \
        == [(0, 0.8, 1), (0, 0.8, 2), (0, 3.0, 1), (0, 3.0, 2)], "a trace's loads and states before the next trace"
    middles = [tuple(r["arm"] for r in report["rows"] if r["block"] == b)[1:4] for b in range(3)]
    assert len(set(middles)) == 3 and all(set(m) == {"router", "agent", "stacked"} for m in middles)
    agent_rows = [r for r in report["rows"] if r["arm"] == "agent"]
    assert all(set(r["agent"]) == {"mode_before", "mode_after", "switches", "dreams"} for r in agent_rows)
    assert report["agent"]["mode"] in ("route", "stacked", "single") and report["fallbacks"] == 0
    analysis = rl2.analyse_ag6h(report, resamples=50)
    assert analysis["complete_blocks"] == 8 and analysis["valid"]
    assert set(analysis["recorded"]["by_load_and_state"]) == {"load 0.8, bursts", "load 0.8, spike",
                                                             "load 3.0, bursts", "load 3.0, spike"}


def test_ag6h_starts_no_block_after_its_budget(tmp_path, monkeypatch):
    pool, spec, router, agent = _ag6h_pool(tmp_path, monkeypatch)
    try:
        report = rl2.measure_ag6h(pool, spec.model_id, router, agent, None, 30.0, loads=(0.8,), levels=(1,),
                                  traces=[0, 1, 2], requests=2, limit=2, budget_s=0.0)
    finally:
        pool.close()
    assert report["time_cut"]["after_blocks"] == 0 and report["rows"] == []
    assert rl2.analyse_ag6h(report)["verdict"].startswith("invalid")


def test_ag6h_decision_reads_the_fixed_rule():
    def requests(span, first=0.2):
        return [{"offset_s": 0.0, "end_s": span, "first_s": first, "tokens": 100, "outcome": "ok"} for _ in range(4)]

    def report(agent_span, agent_first=0.2, blocks=4, failed_block=None):
        rows = []
        for b in range(blocks):
            for arm, run, span in (("single", 0, 100.0), ("router", 0, 50.0), ("agent", 0, agent_span),
                                   ("stacked", 0, 30.0), ("single", 1, 100.0)):
                cell = requests(span, agent_first if arm == "agent" else 0.2)
                if b == failed_block and arm == "agent":
                    cell[0] = {**cell[0], "outcome": "failed", "first_s": None}
                rows.append({"block": b, "arm": arm, "run": run, "load": 0.8, "level": 2, "trace": b,
                             "requests": cell, **({"agent": {"mode_before": "route", "mode_after": "stacked",
                                                            "switches": 1, "dreams": 2}} if arm == "agent" else {})})
        return {"arms": list(rl2.AG6H_ARMS), "loads": [0.8], "levels": [2], "rows": rows}

    good = rl2.analyse_ag6h(report(30.0), resamples=200)
    assert good["verdict"].startswith("pass") and good["agent_over_router_tokens_per_s"]["ratio"] == 50.0 / 30.0
    assert good["recorded"]["agent_over_single"]["ratio"] == 100.0 / 30.0 and len(good["recorded"]["agent_modes"]) == 4
    assert rl2.analyse_ag6h(report(50.0), resamples=200)["verdict"].startswith("fail"), "no gain over the router"
    assert rl2.analyse_ag6h(report(30.0, agent_first=0.4), resamples=200)["verdict"].startswith("fail"), "later TTFT"
    voided = rl2.analyse_ag6h(report(30.0, blocks=5, failed_block=2), resamples=200)
    assert voided["voided_blocks"] == [2] and voided["complete_blocks"] == 4 and voided["valid"]
    assert rl2.analyse_ag6h(report(30.0, blocks=3), resamples=50)["verdict"].startswith("invalid")


def test_ag6q_trace_is_quiet_then_a_spike_then_quiet_again_as_fixed():
    """AG6-Q's arrivals as fixed in BACKLOG: 32 quiet requests 3 s apart, 3 s later 48 within 0.5 s, and from 90 s
    after the spike's first arrival 80 quiet ones 3 s apart, all with one 64-token limit; the seed sets the
    spike's spread and nothing else of the timing."""
    items = rl2.trace_ag6q(450)
    phases = [[i for i in items if i["phase"] == p] for p in range(3)]
    assert [len(p) for p in phases] == [32, 48, 80] and {i["max_tokens"] for i in items} == {64}
    assert [i["offset_s"] for i in phases[0]] == [3.0 * k for k in range(32)]
    spike = [i["offset_s"] for i in phases[1]]
    assert spike[0] == 96.0 and spike == sorted(spike) and spike[-1] - spike[0] < 0.5
    assert [i["offset_s"] for i in phases[2]] == [186.0 + 3.0 * k for k in range(80)]
    other = rl2.trace_ag6q(451)
    assert [i["offset_s"] for i in other if i["phase"] != 1] == [i["offset_s"] for i in items if i["phase"] != 1]
    assert [i["offset_s"] for i in other if i["phase"] == 1] != spike and rl2.trace_ag6q(450) == items


def test_the_busy_rule_serves_stacked_while_anything_is_outstanding_and_single_otherwise():
    rule = rl2.BusyRule()
    assert rule.choose([0, 0], 64) == 0 and rule.choose([1, 0], 64) == 1 and rule.choose([0, 3], 64) == 1
    assert list(rule.watch(0, 1, iter([{"type": "done"}]))) == [{"type": "done"}], "it watches nothing"


def _ag6q(tmp_path, monkeypatch, **kwargs):
    from ironmule_product.path_router import PathRouter, PolicyRouter

    pool, spec, _, _ = _ag6h_pool(tmp_path, monkeypatch)
    made = []

    def agent():
        made.append(PolicyRouter.from_setup(_AG6_SETUP))
        return made[-1]

    try:
        report = rl2.measure_ag6q(pool, spec.model_id, lambda: PathRouter.from_setup(_AG6_SETUP), agent, None, 30.0,
                                  **kwargs)
    finally:
        pool.close()
    return report, made


def test_ag6q_runs_five_fresh_arms_a_repetition_rotated_and_records_every_decision(tmp_path, monkeypatch):
    """AG6-Q on two fake workers: both paths warmed first; per repetition the five arms in a rotated order on the
    same arrivals, a fresh agent each time; every decision recorded with its time, mode and path."""
    report, made = _ag6q(tmp_path, monkeypatch, traces=[0, 1], sizes=(3, 4, 3), gap_s=0.05, spike_s=0.01,
                         recover_s=0.3, limit=3, warmup=2)
    assert report["warmup"] == {"single": {"ok": 2, "makespan_s": report["warmup"]["single"]["makespan_s"]},
                                "stacked": {"ok": 2, "makespan_s": report["warmup"]["stacked"]["makespan_s"]}}
    assert [r["arm"] for r in report["rows"]] == ["agent", "route", "stacked", "single", "busy",
                                                  "route", "stacked", "single", "busy", "agent"]
    assert len(made) == 2 and made[0] is not made[1], "a fresh agent every repetition"
    for row in report["rows"]:
        assert [r["phase"] for r in row["requests"]] == [0] * 3 + [1] * 4 + [2] * 3
        assert all(r["outcome"] == "ok" and r["prompt_key"] for r in row["requests"])
        assert len(row["picks"]) == 10 and all(at > -0.01 for at, _, _ in row["picks"])
    stacked = [r for r in report["rows"] if r["arm"] == "stacked"][0]
    assert {index for _, _, index in stacked["picks"]} == {1} and {m for _, m, _ in stacked["picks"]} == {None}
    agent = [r for r in report["rows"] if r["arm"] == "agent"][0]
    assert {m for _, m, _ in agent["picks"]} <= {"route", "stacked", "single"} and "switches" in agent
    assert {"decision_us_mean", "decision_us_max", "dream_ms_mean", "dream_ms_max"} <= set(agent["router"])
    assert report["fallbacks"] == 0 and report["router_faults"] == 0
    analysis = rl2.analyse_ag6q(report, resamples=50)
    assert analysis["valid"] and analysis["valid_repetitions"] == [0, 1]
    assert set(analysis["phases"]) == {"quiet", "spike", "quiet again"}
    assert analysis["correctness"]["with_more_than_two"] == [] and analysis["verdict"].split(":")[0] in ("pass", "fail")


def test_ag6q_starts_no_repetition_after_its_budget(tmp_path, monkeypatch):
    report, made = _ag6q(tmp_path, monkeypatch, traces=[0, 1], sizes=(1, 1, 1), gap_s=0.05, recover_s=0.1, limit=2,
                         warmup=0, budget_s=0.0)
    assert report["time_cut"]["after_repetitions"] == 0 and report["rows"] == [] and made == []
    assert rl2.analyse_ag6q(report)["verdict"].startswith("invalid")


def _ag6q_report(cost, reps=3, failed=None, answers=lambda arm, rep, k: "single" if arm == "single" else "stacked",
                 switches=()):
    """Synthetic AG6-Q rows: `cost(arm, phase)` gives (first token, end) per request; phase starts at 0, 10, 20."""
    rows = []
    for rep in range(reps):
        for arm in rl2.AG6Q_ARMS:
            requests = []
            for k in range(12):
                phase = k // 4
                first, end = cost(arm, phase)
                requests.append({"offset_s": 10.0 * phase + 0.5 * (k % 4), "phase": phase, "first_s": first,
                                 "end_s": end, "tokens": 10, "outcome": "ok", "digest": answers(arm, rep, k),
                                 "prompt_key": f"p{k}"})
            if (rep, arm) == failed:
                requests[5] = {**requests[5], "outcome": "failed", "first_s": None}
            row = {"repetition": rep, "arm": arm, "requests": requests,
                   "picks": [[r["offset_s"] + 0.001, "route" if arm == "agent" else None, 0] for r in requests],
                   "router": {"decision_us_mean": 1.0, "decision_us_max": 3.0, "mode": "route"}}
            if arm == "agent":
                row["router"].update(dreams=4, dream_ms_mean=2.0, dream_ms_max=5.0)
                row["switches"] = [dict(s) for s in switches]
            rows.append(row)
    return {"plan": "ag6q", "arms": list(rl2.AG6Q_ARMS), "fixed": list(rl2.AG6Q_FIXED),
            "phases": list(rl2.AG6Q_PHASES), "rows": rows}


def test_ag6q_decision_reads_the_fixed_rule():
    """Per phase, the agent against the best fixed mode for first tokens and for tokens per second, pooled; a failed
    request voids its repetition; fewer than two valid repetitions or a third answer make the run invalid."""
    def cost(agent_first=None, agent_end=None):
        def at(arm, phase):
            first, end = {"single": (0.1, 1.0), "stacked": (0.5, 1.2), "route": (0.1, 1.0), "busy": (0.1, 1.0)}[arm] \
                if arm != "agent" else (0.1, 1.0)
            if phase == 1:  # the spike: the stacked engine serves it sooner and earlier
                first, end = {"single": (3.0, 6.0), "stacked": (0.5, 2.0)}.get(arm, (first, end))
                if arm in ("route", "busy", "agent"):
                    first, end = (1.0, 3.0) if arm == "route" else (0.5, 2.0)
            if arm == "agent":
                first = agent_first(phase) if agent_first else first
                end = agent_end(phase) if agent_end else end
            return first, end
        return at

    good = rl2.analyse_ag6q(_ag6q_report(cost()), resamples=200)
    assert good["valid"] and good["verdict"].startswith("pass")
    assert good["phases"]["quiet"]["best_fixed"] == {"ttft_mean": "single", "tokens_per_s": "single"}
    assert good["phases"]["spike"]["best_fixed"] == {"ttft_mean": "stacked", "tokens_per_s": "stacked"}
    assert good["phases"]["spike"]["agent"]["ttft_mean"]["ratio"] == pytest.approx(1.0)
    assert good["phases"]["spike"]["busy"]["tokens_per_s"]["mean_of_repetitions"] == pytest.approx(1.0)
    assert good["recorded"]["decision_us"]["agent"] == {"mean": 1.0, "max": 3.0}
    assert good["recorded"]["dream_ms"] == {"dreams": [4, 4, 4], "mean": 2.0, "max": 5.0}
    late = rl2.analyse_ag6q(_ag6q_report(cost(agent_first=lambda p: 0.5 if p == 2 else (0.5 if p == 1 else 0.1)))
                            , resamples=200)
    assert late["verdict"].startswith("fail: quiet again: ttft"), "stacked first tokens after the spike fail it"
    assert late["phases"]["quiet again"]["agent"]["ttft_mean"]["ratio"] == pytest.approx(5.0)
    slow = rl2.analyse_ag6q(_ag6q_report(cost(agent_first=lambda p: 1.0 if p == 1 else 0.1,
                                              agent_end=lambda p: 3.0 if p == 1 else 1.0)), resamples=200)
    assert slow["verdict"].startswith("fail: spike: ttft and tokens_per_s"), "staying in route through the spike"
    voided = rl2.analyse_ag6q(_ag6q_report(cost(), failed=(1, "busy")), resamples=50)
    assert voided["valid"] and voided["voided_repetitions"] == [1] and voided["valid_repetitions"] == [0, 2]
    assert rl2.analyse_ag6q(_ag6q_report(cost(), reps=2, failed=(0, "agent")))["verdict"].startswith(
        "invalid: fewer than 2"), "one valid repetition is not enough"
    third = rl2.analyse_ag6q(_ag6q_report(cost(), answers=lambda arm, rep, k: arm if k == 3 else "same"))
    assert third["verdict"] == "invalid: 1 prompts answered more than two ways"


def test_ag6q_times_the_agents_adaptation_from_each_phases_first_arrival_and_censors_it():
    def cost(arm, phase):
        return (0.1, 1.0)

    moved = rl2.analyse_ag6q(_ag6q_report(cost, switches=(
        {"at_request": 6, "from": "route", "to": "stacked", "rule": "tokens per second", "at_s": 10.75},
        {"at_request": 10, "from": "stacked", "to": "single", "rule": "first token", "at_s": 21.25})), resamples=20)
    agent = moved["recorded"]["agent"][0]
    assert agent["adaptation"]["spike"] == {"after_s": 0.75, "censored": False}
    assert agent["adaptation"]["quiet again"] == {"after_s": 1.25, "censored": False}
    assert agent["picks_by_phase"]["spike"]["paths"] == {"0": 4}
    assert sum(agent["picks_by_phase"]["quiet"]["modes"].values()) == 4
    never = rl2.analyse_ag6q(_ag6q_report(cost), resamples=20)["recorded"]["agent"][0]["adaptation"]
    assert never["spike"] == {"after_s": pytest.approx(2.5), "censored": True}, "no switch into stacked: censored"
    assert never["quiet again"] == {"after_s": 0.0, "censored": False}, "already in route when that phase began"
    assert json.loads(json.dumps(moved))["recorded"]["agent"][0]["adaptation"]["spike"]["after_s"] == 0.75


def test_the_ag6q_plan_runs_from_the_command_line_and_writes_its_own_kernel(tmp_path, monkeypatch):
    def run(model, work, *, smoke, on_row=None):
        on_row({"plan": "ag6q", "setup": {"engine": "single"}})
        return {"plan": "ag6q", "setup": {"engine": "single", "reason": "test"}, "stopped": "not route"}

    monkeypatch.setattr(rl2, "run_ag6q", run)
    monkeypatch.setattr(rl2, "environment", lambda: {"machine": "test"})
    out = tmp_path / "report.json"
    assert rl2.main(["--out", str(out), "--plan", "ag6q"]) == 0
    report = json.loads(out.read_text())
    assert report["analysis"]["verdict"] == "stopped: not route" and report["rl2h"]["stopped"] == "not route"
    folder = tmp_path / "kernel"
    assert rl2.main(["--write-kernel", str(folder), "--owner", "someone", "--commit", "c" * 40, "--plan", "ag6q"]) == 0
    metadata = json.loads((folder / "kernel-metadata.json").read_text())
    assert metadata["id"] == "someone/ironmule-rl2-headroom-ag6q" and metadata["enable_gpu"]
    assert "'--plan', 'ag6q'" in (folder / "run.py").read_text()


def test_a_run_that_fails_still_leaves_its_report_and_ends_in_order(tmp_path, monkeypatch):
    """AG5-G run 2: the agent's measurement timed out, the exception ended the session, and its report held only
    the environment. A failing run now records what it measured, and where and why it stopped."""
    def run(model, work, *, smoke, on_row=None):
        on_row({"plan": "ag6h", "setup": {"engine": "route"}})
        raise subprocess.TimeoutExpired(["python"], 7200)

    monkeypatch.setattr(rl2, "run_ag6h", run)
    monkeypatch.setattr(rl2, "environment", lambda: {"machine": "test"})
    out = tmp_path / "report.json"
    assert rl2.main(["--out", str(out), "--plan", "ag6h"]) == 0
    report = json.loads(out.read_text())
    assert report["rl2h"] == {"plan": "ag6h", "setup": {"engine": "route"}}, "what it measured is kept"
    assert report["failed"]["error"].startswith("TimeoutExpired") and "Traceback" in report["failed"]["traceback"]
    assert "finished_unix" in report


def test_diagnostics_reach_the_output_folder_when_the_worker_is_lost_and_the_run_then_aborts(tmp_path, monkeypatch):
    """STAB3: run 6's cause was lost because the product's logs stayed in a temporary folder. A deliberate failure:
    the worker dies under a request, then the run itself fails. The output folder still holds the harness events,
    the product's stability log (admission, group, end, the worker's exit with its last stderr, each restart's
    start and outcome), the commit and the traceback."""
    from ironmule_product.service import ProductService

    real_popen = subprocess.Popen

    def fake_popen(args, *popen_args, **popen_kwargs):
        if len(args) > 3 and str(args[3]).endswith("worker.py"):
            args = [sys.executable, "-u", str(FAKE), *args[4:]]
        return real_popen(args, *popen_args, **popen_kwargs)

    monkeypatch.setattr(backend_module.subprocess, "Popen", fake_popen)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    def run(model, work, *, smoke, on_row=None):
        store = ProductStore(rl2._state_dir(work))
        store.setup("server")
        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        spec = ModelSpec("local/test", "revision", str(snapshot.resolve()), 1)
        store.register_model(spec)
        backend = MLXWorkerClient(spec, execution_variant="current_engine")
        backend.start()
        service = ProductService(store, backend=backend, spec=spec)
        service.make_backend = lambda: MLXWorkerClient(spec, execution_variant="current_engine")
        service.label = "1:agent"
        rl2._diag("phase_start", phase=0, agent_card="1")
        try:
            request = GenerationRequest("local/test", (("user", "crash"),), max_tokens=4, request_id="r-0", stream=True)
            with pytest.raises(Exception):
                list(service.stream(request))
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline and service.health()["worker"]["state"] != "ready":
                time.sleep(0.05)
        finally:
            service.close()
        raise RuntimeError("the run stops here, after the worker was lost")

    monkeypatch.setattr(rl2, "run_ag5g", run)
    monkeypatch.setattr(rl2, "environment", lambda: {"machine": "test", "versions": {"mlx": "x"}})
    out = tmp_path / "out" / "report.json"
    out.parent.mkdir()
    assert rl2.main(["--out", str(out), "--plan", "ag5g", "--commit", "c" * 40]) == 0
    report = json.loads(out.read_text())
    assert report["commit"] == "c" * 40 and report["failed"]["error"].startswith("RuntimeError")
    harness = [json.loads(line) for line in (out.parent / "diagnostics" / "harness.jsonl").read_text().splitlines()]
    assert [row["event"] for row in harness] == ["run_start", "phase_start", "run_failed"]
    assert harness[0]["commit"] == "c" * 40 and harness[0]["env"]["versions"] == {"mlx": "x"}
    assert "RuntimeError" in harness[-1]["traceback"]
    log = [json.loads(line) for line in
           (out.parent / "diagnostics" / "ironmule-state" / "stability-log.jsonl").read_text().splitlines()]
    events = [row["event"] for row in log]
    for event in ("admit", "group", "end", "worker_exit", "worker_restart_start", "worker_restart"):
        assert event in events, event
    exit_ = next(row for row in log if row["event"] == "worker_exit")
    assert "fatal out of memory" in exit_["stderr_tail"] and exit_["exit_status"] == 9 and exit_["reaped"]
    end = next(row for row in log if row["event"] == "end")
    assert end["id"] == "r-0" and end["error"] == "backend_unavailable"
    assert all(row["shard"] == "1:agent" and "unix" in row for row in log)
    assert rl2.DIAGNOSTICS is None, "the folder ends with the run"


def test_ag5r_rebuilds_run_6s_spike_exactly_and_adds_a_group_larger_than_the_plans_slots():
    blocks = rl2.ag5r_blocks()
    names = [name for name, _ in blocks]
    assert names == ["phase0-block0", "phase0-block1"] + [f"phase0-block1-again-{k}" for k in range(1, 5)] + \
        ["eight-at-once"]
    spike = dict(blocks)["phase0-block1"]
    recorded = [0.0, 0.0695, 0.0728, 0.2177, 0.4498, 0.4755, 0.5094, 0.6037, 0.655, 0.8254, 0.8715, 0.8747]
    assert [round(item["offset_s"], 4) for item in spike] == recorded, "run 6's report, phase 0, block 1"
    assert {item["max_tokens"] for item in spike} == {256} and not any(item["long"] for item in spike)
    eight = dict(blocks)["eight-at-once"]
    assert len(eight) == 8 > rl2.AG5R_WIDTH and {item["offset_s"] for item in eight} == {0.0}


def test_ag5r_runs_every_block_through_one_path_and_reports_each_as_it_ends(tmp_path, monkeypatch):
    pool, spec, _, _ = _ag6h_pool(tmp_path, monkeypatch)
    seen = []
    try:
        blocks = rl2.ag5r_blocks(requests=3, limit=3, repeats=1)
        report = rl2.measure_ag5r(pool, spec.model_id, blocks, 30.0, on_row=lambda r: seen.append(len(r["blocks"])))
    finally:
        pool.close()
    assert [b["block"] for b in report["blocks"]] == [name for name, _ in blocks] and seen == [1, 2, 3, 4]
    assert rl2.analyse_ag5r(report)["verdict"].startswith("accepted")


def test_ag5r_tells_an_accepted_run_from_a_controlled_failure_and_a_hang():
    def report(*outcomes):
        return {"blocks": [{"block": "b", "requests": [{"outcome": o, "error": e, "end_s": 1.0} for o, e in outcomes],
                            "shards": [{"worker": {"state": "ready"}}]}]}

    assert rl2.analyse_ag5r(report(("ok", None)))["verdict"].startswith("accepted")
    controlled = rl2.analyse_ag5r(report(("ok", None), ("failed", "backend_unavailable")))
    assert controlled["verdict"].startswith("controlled") and controlled["blocks"][0]["errors"] == ["backend_unavailable"]
    assert rl2.analyse_ag5r(report(("failed", "backend_unavailable"), ("timeout", None)))["verdict"].startswith("hung")
    assert rl2.analyse_ag5r({"failed": "no worker"})["verdict"] == "not run: no worker"
