"""RL4: the pool model, its fit and fidelity gate, and the REINFORCE agent (no model, no GPU)."""

from __future__ import annotations

import json

import pytest

from tools import rl2_headroom as rl2
from tools import rl4_agent as rl4

MODEL = rl4.PoolModel(prefill_s=0.04, prefill_s_per_token=3e-4, step_s=0.03, step_s_per_session=0.006)


def _items(*rows):
    """(offset_s, max_tokens, long_prompt) rows in rl2_headroom's item format."""
    return [{"offset_s": t, "prompt": 0, "max_tokens": n, "long": long} for t, n, long in rows]


def test_a_group_streams_first_tokens_after_each_prefill_and_releases_every_end_together():
    members = [rl4._Request(0, 0.0, 3, False), rl4._Request(1, 0.0, 1, True)]
    end = rl4.group_timeline(MODEL, members, 1.0)
    assert members[0].first == pytest.approx(1.0 + MODEL.prefill(False))
    assert members[1].first == pytest.approx(members[0].first + MODEL.prefill(True))
    assert end == pytest.approx(members[1].first + 2 * MODEL.step(1))  # only the first member decodes on
    assert members[0].end == members[1].end == end


def test_decode_rounds_serve_at_most_four_sessions_and_rotate():
    members = [rl4._Request(i, 0.0, 2, False) for i in range(6)]
    end = rl4.group_timeline(MODEL, members, 0.0)
    prefill = 6 * MODEL.prefill(False)
    assert end == pytest.approx(prefill + MODEL.step(4) + MODEL.step(2))


def test_segregating_lengths_frees_short_requests_from_long_groups():
    spike = _items(*[(0.01 * i, 256 if i < 8 else 32, False) for i in range(16)])  # the long ones first
    shortest = rl4.cost(rl4.simulate(MODEL, spike, rl4.fixed_placer("shortest"), rl4.fixed_orderer("fifo")))
    segregate = rl4.cost(rl4.simulate(MODEL, spike, rl4.fixed_placer("segregate"), rl4.fixed_orderer("fifo")))
    assert segregate < shortest


def test_sjf_starts_the_shortest_waiting_requests_first():
    one_card = rl4.PoolModel(cards=1)
    trace = _items((0.0, 64, False), (0.001, 256, False), (0.002, 8, False), (0.003, 16, False))
    sjf = rl4.simulate(one_card, trace, rl4.fixed_placer("shortest"), rl4.fixed_orderer("sjf"), capacity=1)
    fifo = rl4.simulate(one_card, trace, rl4.fixed_placer("shortest"), rl4.fixed_orderer("fifo"), capacity=1)
    assert [r.max_tokens for r in sorted(sjf, key=lambda r: r.end)] == [64, 8, 16, 256]
    assert [r.max_tokens for r in sorted(fifo, key=lambda r: r.end)] == [64, 256, 8, 16]
    assert rl4.cost(sjf) < rl4.cost(fifo)


def _report(model, arms=rl4.FIXED_ARMS, traces=3, capacity=1.0):
    """A run-3-shaped report whose times come from `model` itself."""
    rows = []
    for t in range(traces):
        for state in range(3):
            items = rl2.trace_h2(state, t, capacity)
            for arm in arms:
                place, order, width = rl4.arm_policy(arm)
                done = rl4.simulate(model, items, place, order, width)
                rows.append({"level": state, "trace": t, "arm": arm, "run": 0, "requests": [
                    {"index": r.index, "outcome": "ok", "first_s": r.first - r.arrive, "end_s": r.end - r.arrive}
                    for r in done]})
    return {"rl2h": {"plan": "h3", "arms": list(arms), "calibration": {"capacity_rps": capacity}, "rows": rows}}


def test_the_fit_recovers_the_timing_that_produced_the_measurements():
    report = _report(MODEL, traces=2)
    fitted, loss = rl4.fit([report], start=rl4.PoolModel())
    assert loss < 1e-3
    for name in ("prefill_s_per_token", "step_s", "step_s_per_session"):
        assert getattr(fitted, name) == pytest.approx(getattr(MODEL, name), rel=0.25), name
    assert rl4.fidelity(fitted, report)["passes"]


def test_the_fidelity_gate_refuses_a_model_that_misranks_or_mispredicts():
    report = _report(MODEL, traces=2)
    assert rl4.fidelity(MODEL, report)["passes"]
    wrong = rl4.PoolModel(prefill_s=2.0, prefill_s_per_token=0.0, step_s=0.001, step_s_per_session=0.0)
    gate = rl4.fidelity(wrong, report)
    assert not gate["passes"] and any(state["max_error"] > 0.25 for state in gate["states"].values())


def test_only_known_plans_without_gather_waits_are_fitted():
    with pytest.raises(ValueError, match="plans h1 to h4"):
        rl4._cells({"rl2h": {"plan": "h9", "rows": [], "calibration": {"capacity_rps": 1.0}}})
    report = _report(MODEL, arms=("shortest/w8/g0",), traces=1)
    report["rl2h"]["rows"].append({**report["rl2h"]["rows"][0], "arm": "shortest/w8/g50"})
    assert {arm for _, arm in rl4._cells(report)} == {"shortest/w8/g0"}


def test_reinforce_learns_a_policy_that_beats_its_untrained_start_in_the_model():
    policy, curve = rl4.train(MODEL, 1.0, episodes=900, lr=0.1, seed=1)
    untrained = rl4.Policy.zero().greedy()
    learned = policy.greedy()
    held_out = rl4.evaluate(MODEL, 1.0, {"untrained": (*untrained, 8), "learned": (*learned, 8)}, traces=40)
    weighted = {name: sum(w * c for w, c in zip(rl2.WEIGHTS, costs)) for name, costs in held_out.items()}
    assert weighted["learned"] < weighted["untrained"]
    assert len(curve) == 900


def test_the_simulator_gate_compares_the_learned_policy_with_every_fixed_arm():
    result = rl4.judge(MODEL, 1.0, rl4.Policy.zero(), noise=0.035, traces=10)
    assert set(result["weighted"]) == set(rl4.FIXED_ARMS) | set(rl4.SIMULATOR_ONLY_ARMS) | {"learned"}
    assert result["best_simulator_only_rule"] in rl4.SIMULATOR_ONLY_ARMS
    assert result["best_fixed"] in rl4.FIXED_ARMS
    assert result["passes"] == (result["gain_vs_best_fixed"] > 0.035)


def test_a_policy_file_carries_a_digest_of_its_weights():
    policy = rl4.Policy([0.1] * rl4.PLACEMENT_SIZE, [0.2] * rl4.ORDER_SIZE)
    data = policy.to_json()
    assert rl4.Policy.from_json(json.loads(json.dumps(data))) == policy
    data["order"][0] = 9.0
    with pytest.raises(ValueError, match="digest"):
        rl4.Policy.from_json(data)
