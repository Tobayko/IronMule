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
        rl4._cells({"rl2h": {"plan": "x1", "rows": [], "calibration": {"capacity_rps": 1.0}}})
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


def test_training_keeps_the_seed_that_does_best_on_validation_traffic():
    policy, summary = rl4.train_best(MODEL, 1.0, seeds=2, episodes=150, validation_traces=8)
    costs = summary["validation_cost_per_seed"]
    assert len(costs) == 2 and summary["chosen_seed"] == min(range(2), key=lambda i: (costs[i], i))
    assert isinstance(policy, rl4.Policy)


def test_a_width_policy_picks_each_groups_width_and_keeps_old_policies_digests():
    old = rl4.Policy([0.1] * rl4.PLACEMENT_SIZE, [0.2] * rl4.ORDER_SIZE)
    assert old.to_json()["schema"] == "ironmule.rl4.policy.v1" and old.greedy_width() is None
    wide = rl4.Policy.zero(widths=True)
    wide.width[rl2.WIDTH_CHOICES.index(2) * rl4.ORDER_SIZE] = 5.0  # bias towards width 2
    data = json.loads(json.dumps(wide.to_json()))
    assert data["schema"] == "ironmule.rl4.policy.v2" and rl4.Policy.from_json(data) == wide
    queue = [rl4._Request(i, 0.0, 32, False) for i in range(6)]
    assert wide.best_width(queue) == 2
    trace = _items(*[(0.0, 32, False)] * 6)
    groups = []
    rl4.simulate(MODEL, trace, rl4.fixed_placer("shortest"), rl4.fixed_orderer("fifo"),
                 width=lambda waiting, now: groups.append(min(2, len(waiting))) or 2)
    assert groups and max(groups) <= 2


def test_reinforce_can_learn_widths_and_the_gate_then_faces_every_fixed_width():
    policy, curve = rl4.train(MODEL, 1.0, episodes=60, lr=0.1, seed=2, widths=True)
    assert policy.width is not None and len(curve) == 60
    result = rl4.judge(MODEL, 1.0, policy, noise=0.035, traces=4)
    assert set(rl4.WIDTH_ARMS) <= set(result["weighted"]) and result["best_fixed"] in rl4.FIXED_ARMS + rl4.WIDTH_ARMS



def test_member_release_ends_each_member_at_its_own_last_token_and_the_card_when_the_group_ends():
    from dataclasses import replace

    member = replace(MODEL, release="member")
    members = [rl4._Request(0, 0.0, 3, False), rl4._Request(1, 0.0, 1, True), rl4._Request(2, 0.0, 2, False)]
    end = rl4.group_timeline(member, members, 0.0)
    first_round = members[2].first + MODEL.step(2)
    assert members[1].end == members[1].first  # one token: done at its prefill
    assert members[2].end == pytest.approx(first_round)  # its second token comes in the first round
    assert members[0].end == pytest.approx(first_round + MODEL.step(1)) == end  # the card frees with the last
    grouped = [rl4._Request(i, 0.0, n, long) for i, (n, long) in enumerate(((3, False), (1, True), (2, False)))]
    assert rl4.group_timeline(MODEL, grouped, 0.0) == pytest.approx(end)  # same card time, later ends
    spike = _items(*[(0.01 * i, 256 if i % 2 else 32, False) for i in range(16)])
    by_rule = {rule: rl4.cost(rl4.simulate(replace(MODEL, release=rule), spike, rl4.fixed_placer("shortest"),
                                           rl4.fixed_orderer("fifo"))) for rule in ("group", "member")}
    assert by_rule["member"] < by_rule["group"]  # short members no longer wait for their long neighbours


def _product_report(model, plan, arms=("tuned:shortest/w2/g0/sjf", "tuned:shortest/w8/g0", "kp:shortest/w2/g0/sjf"),
                    traces=2, capacity=1.0):
    """A product-run-shaped report (arms naming their engine, A/A runs), times from `model` under the release
    rule the product had for `plan`."""
    from dataclasses import replace

    sim = replace(model, release="member" if plan in rl4.MEMBER_RELEASE_PLANS else "group")
    rows = []
    for t in range(traces):
        for state in range(3):
            items = rl2.trace_h2(state, t, capacity)
            for arm in arms:
                place, order, width = rl4.arm_policy(arm)
                done = rl4.simulate(sim, items, place, order, width)
                for run in (0, 1):
                    rows.append({"level": state, "trace": t, "arm": arm, "run": run, "engine": rl2.arm_engine(arm),
                                 "requests": [{"index": r.index, "outcome": "ok", "first_s": r.first - r.arrive,
                                               "end_s": r.end - r.arrive} for r in done]})
    return {"rl2h": {"plan": plan, "arms": list(arms), "calibration": {"capacity_rps": capacity}, "rows": rows,
                     "deadline_s": 120}}


def test_runs_before_and_after_the_release_change_fit_one_set_of_timings():
    before, after = _product_report(MODEL, "h7"), _product_report(MODEL, "h9")
    assert rl4.release_of(before) == "group" and rl4.release_of(after) == "member"
    assert {arm for _, arm in rl4._cells(after)} == {"tuned:shortest/w2/g0/sjf", "tuned:shortest/w8/g0"}
    assert {arm for _, arm in rl4._cells(after, engine="kp")} == {"kp:shortest/w2/g0/sjf"}
    fitted, loss = rl4.fit([before, after], start=rl4.PoolModel())
    assert loss < 1e-3
    for name in ("prefill_s_per_token", "step_s", "step_s_per_session"):
        assert getattr(fitted, name) == pytest.approx(getattr(MODEL, name), rel=0.25), name
    gate = rl4.fidelity(fitted, after)
    assert gate["passes"] and gate["noise"] == 0.0  # identical A/A runs in this synthetic report



def test_a_harm_weight_pushes_learning_toward_losing_no_state_to_its_best_fixed_arm():
    guards = rl4.guard_arms(MODEL, 1.0, rl4.WIDTH_ARMS, traces=10)
    assert set(guards) == {0, 1, 2} and all(arm in rl4.WIDTH_ARMS for arm in guards.values())
    plain, _ = rl4.train(MODEL, 1.0, episodes=600, lr=0.1, seed=2, widths=True)
    guarded, _ = rl4.train(MODEL, 1.0, episodes=600, lr=0.1, seed=2, widths=True, guards=guards, harm_weight=5.0)

    def worst_loss(policy):
        verdict = rl4.judge(MODEL, 1.0, policy, noise=0.03, traces=20)
        return min(verdict["per_state_gain_vs_best_fixed_there"]), verdict

    worst_guarded, verdict = worst_loss(guarded)
    assert set(verdict) >= {"per_state_gain_vs_best_fixed_there", "no_state_lost"}
    assert worst_guarded >= worst_loss(plain)[0] - 0.01  # the guard does not make the worst state worse



def test_a_joining_card_admits_newcomers_into_freed_places_between_rounds():
    from dataclasses import replace

    dense = replace(MODEL, release="member", engine_width=8, joins=True, cards=1, step_s_per_session=0.001)
    alone_then_newcomer = _items((0.0, 40, False), (0.01, 3, False))
    joined = sorted(rl4.simulate(dense, alone_then_newcomer, rl4.fixed_placer("shortest"), rl4.fixed_orderer("fifo"),
                                 capacity=2), key=lambda r: r.index)
    waited = sorted(rl4.simulate(replace(dense, joins=False), alone_then_newcomer, rl4.fixed_placer("shortest"),
                                 rl4.fixed_orderer("fifo"), capacity=2), key=lambda r: r.index)
    assert joined[1].first < joined[1].end < joined[0].end  # it took a free place while the long one ran
    assert waited[1].first > waited[0].end  # without joins it waits for the group to end
    full = _items((0.0, 2, False), (0.0, 40, False), (0.01, 3, False))  # both places taken until one ends
    short, long_, newcomer = sorted(rl4.simulate(dense, full, rl4.fixed_placer("shortest"), rl4.fixed_orderer("fifo"),
                                                 capacity=2), key=lambda r: r.index)
    assert short.end <= newcomer.first < long_.end
    spike = _items(*[(0.01 * i, 256 if i % 3 == 0 else 32, i % 2 == 0) for i in range(16)])
    by_mode = {joins: rl4.cost(rl4.simulate(replace(dense, joins=joins, cards=2), spike, rl4.fixed_placer("shortest"),
                                            rl4.fixed_orderer("sjf"))) for joins in (False, True)}
    assert by_mode[True] < by_mode[False]
