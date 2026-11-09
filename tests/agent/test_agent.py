"""AG1: the IronMule agent's world model, planner, qualification, decisions, lifecycle and its place in the
real scheduler (with a timed grouped fake backend; the hardware result belongs to BACKLOG AG1)."""

from __future__ import annotations

import json
import random
import threading
import time

import pytest

from ironmule_product import agent as ag
from ironmule_product.agent import Agent, Job, WorldModel, decode_rounds, judge, plan, simulate
from ironmule_product.learned_dispatch import Policy, View
from ironmule_product.state import ProductStore
from ironmule_product.service import ProductService
from ironmule_product.types import GenerationRequest, ModelSpec

TRUTH = {"prefill_s": 0.15, "prefill_s_per_ktoken": 2.2, "step_s": 0.019, "step_s_per_session": 0.016}
BINDING = {"hardware": {"chip_name": "test", "gpu_devices": 1}, "device": "gpu", "model": {"id": "m", "revision": "r"}}
REFERENCE = {"width": 8, "sjf": False, "gather_ms": 0}


def _group(params, rng, width, start=10.0, prompts=(24, 1030)):
    """One group as the pool model runs it, with 2 % timing noise: what the service would report."""
    members = [{"prompt_tokens": rng.choice(prompts), "tokens": rng.choice((8, 32, 64))} for _ in range(width)]
    t = start
    for m in members:
        t += (params["prefill_s"] + params["prefill_s_per_ktoken"] * m["prompt_tokens"] / 1000) * rng.uniform(0.98, 1.02)
        m["first"] = t
    remaining = {i: m["tokens"] - 1 for i, m in enumerate(members)}
    active = [i for i in remaining if remaining[i] > 0]
    for i, m in enumerate(members):
        if remaining[i] == 0:
            m["end"] = m["first"]
    while active:
        served = active[:4]
        t += (params["step_s"] + params["step_s_per_session"] * (len(served) - 1)) * rng.uniform(0.98, 1.02)
        for i in served:
            remaining[i] -= 1
            if remaining[i] == 0:
                members[i]["end"] = t
        active = active[len(served):] + [i for i in served if remaining[i] > 0]
    for m in members:
        m.update(start=start, arrive=start - 1.0, ok=True, max_tokens=m["tokens"], prompt_chars=m["prompt_tokens"] * 4)
    return members


def test_decode_rounds_rotate_at_most_four_sessions_a_round():
    assert decode_rounds([1, 1]) == (0, 0)
    assert decode_rounds([3]) == (2, 0)
    assert decode_rounds([2, 2]) == (1, 1)
    # five members of two tokens: rounds of four then one
    assert decode_rounds([2] * 5) == (2, 3)


def test_the_world_model_recovers_the_machine_from_its_own_groups():
    rng = random.Random(0)
    world = WorldModel()
    for k in range(200):
        assert world.observe(_group(TRUTH, rng, rng.choice((1, 2, 4, 8))))
    fitted = world.params()
    for key, value in TRUTH.items():
        assert fitted[key] == pytest.approx(value, rel=0.1), key
    assert world.fidelity() < 0.05
    assert world.supported(1) and world.supported(8)


def test_thousands_of_groups_keep_the_fit_well_conditioned():
    """AG1 S run 1: after about 2,600 groups, many of them eight long prompts, the covariance lost its positive
    diagonal and the prefill fit ran off (2.3 to 1,824 s per thousand tokens)."""
    rng = random.Random(7)
    world = WorldModel()
    for k in range(4000):
        if k % 3 == 0:
            world.observe(_group(TRUTH, rng, rng.choice((4, 8)), prompts=(1030,)))
        else:
            world.observe(_group(TRUTH, rng, 1, prompts=(24,)))
        assert all(world.prefill.p[i][i] > 0 for i in range(2)) and all(world.decode.p[i][i] > 0 for i in range(2))
    for key, value in TRUTH.items():
        assert world.params()[key] == pytest.approx(value, rel=0.1), key


def test_a_changed_machine_is_relearned():
    rng = random.Random(8)
    world = WorldModel()
    for _ in range(100):
        world.observe(_group(TRUTH, rng, rng.choice((1, 2, 4, 8))))
    cpu = {"prefill_s": 0.4, "prefill_s_per_ktoken": 6.0, "step_s": 0.06, "step_s_per_session": 0.06}
    for _ in range(300):
        world.observe(_group(cpu, rng, rng.choice((1, 2, 4, 8))))
    for key, value in cpu.items():
        assert world.params()[key] == pytest.approx(value, rel=0.1), key
    assert world.fidelity() < ag.FIDELITY


def test_a_failed_or_inconsistent_group_teaches_nothing():
    world = WorldModel()
    group = _group(TRUTH, random.Random(1), 2)
    group[0]["ok"] = False
    assert not world.observe(group)
    group = _group(TRUTH, random.Random(1), 2)
    group[1]["end"] = group[1]["first"] - 1
    assert not world.observe(group)
    assert world.groups == {}


def test_shortest_first_lowers_the_cost_of_a_mixed_queue_in_the_model():
    jobs = lambda: [Job(i, 0.0, t, False, 24, t) for i, t in enumerate((256, 32, 256, 32, 32, 256))]  # noqa: E731
    fifo = simulate(TRUTH, jobs(), lambda w: False, lambda w: 1, capacity=8)
    sjf = simulate(TRUTH, jobs(), lambda w: True, lambda w: 1, capacity=8)
    assert sjf < fifo


def _arrivals(seed=0, n=160):
    rng = random.Random(seed)
    out, t = [], 0.0
    while len(out) < n:  # bursts of four mixed requests
        for _ in range(4):
            limit = rng.choice((32, 256))
            out.append([t + rng.uniform(0, 0.2), 24, False, limit, limit])
        t += rng.expovariate(0.15)
    return sorted(out)


def test_the_planner_is_deterministic_and_proposes_only_what_the_model_prices_lower():
    payload = {"model": TRUTH, "arrivals": _arrivals(), "capacity": 8, "seed": 0, "reference": REFERENCE,
               "incumbent": None, "settings": {"episodes": 200, "seeds": 1, "validation": 10, "held_out": 20}}
    first, second = plan(payload), plan(json.loads(json.dumps(payload)))
    assert first["candidate"] == second["candidate"] and first["model_costs"] == second["model_costs"]
    assert first["candidate"] is not None
    assert first["model_costs"][first["choice"]] < first["model_costs"]["reference"]
    Policy.from_json(first["candidate"])  # the LD1 format the server decides with
    # once the candidate serves, the same plan finds nothing the model prices lower than it
    again = plan({**payload, "incumbent": first["candidate"]})
    assert again["candidate"] is None


def test_a_schedule_rejected_against_the_incumbent_is_never_proposed_again():
    payload = {"model": TRUTH, "arrivals": _arrivals(), "capacity": 8, "seed": 0, "reference": REFERENCE,
               "incumbent": None, "settings": {"episodes": 150, "seeds": 1, "validation": 10, "held_out": 20}}
    first = plan(payload)
    lost = ag.behaviour(first["candidate"])
    again = plan({**payload, "excluded": [lost]})
    assert again["candidate"] is None or ag.behaviour(again["candidate"]) != lost
    assert lost in again["excluded"]


def test_the_agent_remembers_what_lost_to_its_incumbent_until_the_machine_changes(tmp_path):
    agent = _agent(tmp_path)
    agent.candidate = {"policy": ag.fixed_policy(4, True), "plan": {}, "since": 0.0, "triplets": [], "used": 0}
    agent._reject("lost")
    assert agent._excluded() == [ag.behaviour(ag.fixed_policy(4, True))]
    _promote(agent, ag.fixed_policy(8, True))
    assert agent._excluded() == []  # a new incumbent: everything may challenge it
    agent.candidate = {"policy": ag.fixed_policy(4, True), "plan": {}, "since": 0.0, "triplets": [], "used": 0}
    agent._reject("lost again")
    agent.world.reopened += 1  # the machine changed
    assert agent._excluded() == []


def test_a_width_the_backend_cannot_run_is_never_proposed():
    payload = {"model": TRUTH, "arrivals": _arrivals(), "capacity": 2, "seed": 0, "reference": {"width": 2, "sjf": False},
               "incumbent": None, "widths": [1, 2],
               "settings": {"episodes": 100, "seeds": 1, "validation": 5, "held_out": 10}}
    result = plan(payload)
    assert set(result["arm_costs"]) <= {"w1/fifo", "w1/sjf", "w2/fifo", "w2/sjf"}
    if result["candidate"] is not None:
        policy = Policy.from_json(result["candidate"])
        assert policy.best_width([View(32, False)] * 8) in (1, 2)


def test_judge_needs_the_bound_above_zero_and_the_aa_floor_and_gives_up_at_the_limit():
    rng = random.Random(4)
    clear = [{"a": 10.0 * f, "b": 9.5 * f, "aa": 10.05 * f} for f in (rng.uniform(1, 3) for _ in range(ag.MIN_TRIPLETS))]
    result = judge(clear)
    assert result["verdict"] == "pass" and result["gain"] == pytest.approx(0.0525, abs=0.002)
    assert result["floor"] == pytest.approx(0.005, abs=1e-6)
    assert judge(clear[:-1])["verdict"] == "open"  # too few triplets, whatever the gain
    # the incumbent's two runs differ by more than the candidate gains: not resolved
    drifting = [{"a": 10.0, "b": 9.6, "aa": 9.0}] * ag.MIN_TRIPLETS
    assert judge(drifting)["verdict"] == "fail"
    noisy = [{"a": 10.0, "b": 10.0 + (2.0 if k % 2 else -1.9), "aa": 10.0} for k in range(ag.MAX_TRIPLETS)]
    assert judge(noisy)["verdict"] == "fail"
    worse = [{"a": 10.0, "b": 11.0, "aa": 10.0}] * ag.MIN_TRIPLETS
    assert judge(worse)["verdict"] == "fail"


def _agent(tmp_path, **kwargs):
    """An agent for AG1's mechanics, which these tests price by AG1's cost (AG8 has its own tests below)."""
    kwargs.setdefault("objective", "latency")
    return Agent(tmp_path / "agent", BINDING, capacity=8, reference=REFERENCE, **kwargs)


def _replay_factory(costs, texts=None):
    """A replay stand-in: each arm's mean cost from `costs(arm_index)`, with identical answers unless `texts`."""
    calls = []

    def replay(trace, choose, preempted):
        index = len(calls)
        calls.append(choose(len(trace), [], 0.0, 0))
        cost = costs(index)
        rows = [{"ttft_s": cost / 4, "e2e_s": cost * 3 / 4, "tokens": 4, "outcome": "ok"} for _ in trace]
        return {"rows": rows, "texts": (texts or (lambda i: ["same"] * len(trace)))(index), "span_s": 1.0}
    return replay, calls


def _feed(agent, n=64, start=0.0, gap=1.0):
    rng = random.Random(3)
    for k in range(n):
        group = _group(TRUTH, rng, 1, start=start + k * gap)
        group[0]["arrive"] = start + k * gap
        agent.observe_group(group)


def test_a_candidate_serves_only_after_it_wins_replayed_triplets_on_held_out_traffic(tmp_path):
    agent = _agent(tmp_path)
    _feed(agent, 32)
    agent.candidate = {"policy": ag.fixed_policy(2, True), "plan": {"choice": "best_fixed"}, "since": 31.5,
                       "triplets": [], "used": 0}
    assert agent._trace(agent.traffic, since=31.5) is None  # nothing recorded after the plan yet
    _feed(agent, 16 * ag.MIN_TRIPLETS, start=100.0, gap=10.0)  # the server empties between arrivals
    # A, B, AA (or B, A, AA): the candidate (the arm that sorts shortest-first) costs 20 % less
    replay, calls = _replay_factory(lambda i: 0.0)

    def costs_by_arm(trace, choose, preempted):
        sjf = choose(len(trace), [], 0.0, 0)[2]
        cost = 8.0 if sjf else 10.0
        rows = [{"ttft_s": cost / 4, "e2e_s": cost * 3 / 4, "tokens": 4, "outcome": "ok"} for _ in trace]
        return {"rows": rows, "texts": ["same"] * len(trace), "span_s": 1.0}

    for _ in range(ag.MIN_TRIPLETS):
        assert not agent.active
        agent._qualify(costs_by_arm, lambda: False)
    assert agent.active and agent.evidence["gain"] == pytest.approx(0.2)
    assert agent.sjf([GenerationRequest("m", (("user", "x"),), max_tokens=32)]) is True
    assert agent.width([GenerationRequest("m", (("user", "x"),), max_tokens=32)]) == 2
    assert agent.status()["decision_us"] is not None


def test_a_different_replayed_answer_vetoes_a_candidate_whatever_its_speed(tmp_path):
    agent = _agent(tmp_path)
    _feed(agent, 32)
    agent.candidate = {"policy": ag.fixed_policy(2, True), "plan": {}, "since": 0.0, "triplets": [], "used": 0}
    replay, _ = _replay_factory(lambda i: 1.0 if i == 1 else 10.0,
                                texts=lambda i: ["other" if i == 1 else "same"] * ag.TRACE)
    agent._qualify(replay, lambda: False)
    assert agent.candidate is None and not agent.active
    assert agent.counters["rejected"] == 1


def test_an_arm_demand_cut_runs_again_in_the_next_idle_spell_and_finished_arms_are_kept(tmp_path):
    agent = _agent(tmp_path)
    _feed(agent, 32)
    agent.candidate = {"policy": ag.fixed_policy(2, True), "plan": {}, "since": 0.0, "triplets": [], "used": 0}
    runs = []

    def cut_after(n):
        def replay(trace, choose, preempted):
            runs.append(choose(len(trace), [], 0.0, 0)[2])  # the arm's order: True only for the candidate
            if len(runs) > n:
                return None
            rows = [{"ttft_s": 1.0, "e2e_s": 3.0, "tokens": 4, "outcome": "ok"} for _ in trace]
            return {"rows": rows, "texts": ["same"] * len(trace), "span_s": 1.0}
        return replay

    agent._qualify(cut_after(1), lambda: False)  # A runs, B is cut
    assert "a" in agent.candidate["partial"] and "b" not in agent.candidate["partial"]
    assert agent.candidate["triplets"] == []
    runs.clear()
    agent._qualify(cut_after(5), lambda: False)  # the next spell runs only B and AA
    assert runs == [True, False]
    assert len(agent.candidate["triplets"]) == 1 and agent.candidate["partial"] == {} and agent.candidate["used"] == 1


def test_a_window_cut_again_and_again_gives_way_to_the_next(tmp_path):
    agent = _agent(tmp_path)
    _feed(agent, 48)
    agent.candidate = {"policy": ag.fixed_policy(2, True), "plan": {}, "since": 0.0, "triplets": [], "used": 0}
    for _ in range(ag.MAX_PREEMPTIONS):
        agent._qualify(lambda trace, choose, preempted: None, lambda: True)
    assert agent.candidate["used"] == 1 and agent.candidate["partial"] == {} and agent.candidate["triplets"] == []


def _promote(agent, policy=None):
    agent.candidate = {"policy": policy or ag.fixed_policy(2, True), "plan": {}, "since": 0.0, "triplets": [], "used": 0}
    agent._promote(agent.candidate, {"gain": 0.1, "gain_ci95": [0.05, 0.15], "floor": 0.01, "noise": 0.02})


def test_errors_failures_and_the_operator_withdraw_and_the_withdrawal_holds(tmp_path):
    agent = _agent(tmp_path)
    _promote(agent)
    agent.policy = type("Broken", (), {"uses_sjf": lambda self, v: 1 / 0, "sha256": agent.policy.sha256})()
    for _ in range(ag.MAX_ERRORS):
        assert agent.sjf([]) is None
    assert not agent.active and agent.withdrawn[-1]["reason"].startswith("3 errors")

    _promote(agent, ag.fixed_policy(4, True))
    bad = _group(TRUTH, random.Random(5), 1)
    bad[0]["ok"] = False
    for _ in range(2 * ag.MONITOR_WINDOW):
        agent.observe_group(bad)
    assert not agent.active and "failed requests" in agent.withdrawn[-1]["reason"]

    _promote(agent, ag.fixed_policy(1, True))
    status = agent.control("off")
    assert status["paused"] and not status["active"]
    agent.control("on")
    assert not agent.active  # resuming learning never restores a withdrawn policy
    # a withdrawn policy is never proposed again under this binding
    agent._planner_files = (tmp_path / "in.json", tmp_path / "out.json")
    (tmp_path / "out.json").write_text(json.dumps({"schema": ag.PLAN_SCHEMA, "candidate": ag.fixed_policy(1, True)}))
    agent._planner = type("Done", (), {"poll": lambda self: 0, "returncode": 0})()
    agent._collect_plan()
    assert agent.candidate is None


def test_restart_resumes_this_binding_and_sets_aside_a_tampered_or_foreign_state(tmp_path):
    agent = _agent(tmp_path)
    _feed(agent, 20)
    _promote(agent)
    agent.close()
    again = _agent(tmp_path)
    assert again.active and again.policy.sha256 == agent.policy.sha256
    assert sum(again.world.groups.values()) == 20 and len(again.traffic) == 20

    other = Agent(tmp_path / "agent", {**BINDING, "device": "cpu"}, capacity=8, reference=REFERENCE)
    assert not other.active and not other.traffic  # another setup never reads this one's learning

    state = json.loads(again.path.read_text())
    state["evidence"]["gain"] = 0.9  # an edited verdict no longer matches its digest
    again.path.write_text(json.dumps(state))
    tampered = _agent(tmp_path)
    assert not tampered.active and "set aside" in tampered.reason
    assert list(again.path.parent.glob(again.path.name + ".rejected-*"))


def test_exploration_measures_an_unpriced_width_with_every_arrival_queued_at_once(tmp_path):
    agent = _agent(tmp_path)
    _feed(agent, 20)
    seen = []
    agent._explore(lambda trace, choose, preempted: seen.append((trace, choose(16, [], 0.0, 0))) or {"rows": []},
                   lambda: False)
    trace, decision = seen[0]
    assert {row[0] for row in trace} == {0.0} and decision == (2, 0, False)
    assert agent.explored == {2: 1}


def test_the_planner_process_runs_while_idle_and_its_candidate_waits_for_qualification(tmp_path, monkeypatch):
    monkeypatch.setattr(ag, "MIN_GROUPS", 4)
    monkeypatch.setattr(ag, "MIN_ARRIVALS", 16)
    clock = [0.0]
    agent = _agent(tmp_path, clock=lambda: clock[0],
                   planner_settings={"episodes": 50, "seeds": 1, "validation": 5, "held_out": 5})
    rng = random.Random(2)
    for k, arrival in enumerate(_arrivals(n=48)):
        group = _group(TRUTH, rng, rng.choice((1, 2, 4, 8)), start=arrival[0])
        for member in group:
            member["arrive"] = arrival[0]
        agent.observe_group(group)
    clock[0] += 100
    agent.idle(lambda: False, replay=None)
    assert agent.status()["planner"] == "running"
    deadline = time.time() + 120
    while agent._planner is not None and time.time() < deadline:
        time.sleep(0.2)
        agent.idle(lambda: False, replay=None)
    assert agent.counters["plans"] == 1 and agent.last_plan["episodes"] == 50
    assert agent.history[-1]["event"] == "planned"


# ---- the real scheduler with a timed grouped fake backend ----

class TimedBackend:
    """Grouped streaming worker shape (SERVE4); each answer depends only on its request."""

    ready = True
    streams_batches = True
    batch_capacity = 8
    device = "gpu"
    execution_variant = "current_engine"

    def __init__(self):
        self.groups: list[list[int]] = []

    def can_batch_requests(self, requests):
        return 1 <= len(requests) <= self.batch_capacity

    def complete_batch(self, requests, *, cancels, timeout, on_event=None, joins=None):
        requests = list(requests)
        self.groups.append([r.max_tokens for r in requests])
        for _ in requests:
            time.sleep(0.002)
        left = {r.request_id: r.max_tokens for r in requests}
        produced = {r.request_id: 0 for r in requests}
        while left:
            time.sleep(0.001 + 0.0005 * len(left))
            for request in requests:
                rid = request.request_id
                if rid not in left:
                    continue
                on_event(rid, {"type": "token", "request_id": rid, "text": f"{len(request.messages[-1][1])}.{produced[rid]} ",
                               "token_id": produced[rid], "prompt_tokens": 7})
                produced[rid] += 1
                left[rid] -= 1
                if left[rid] == 0:
                    del left[rid]
                    on_event(rid, {"type": "done", "request_id": rid, "finish_reason": "length",
                                   "prompt_tokens": 7, "completion_tokens": produced[rid]})
        return [{"request_id": r.request_id, "events": [], "metadata": {}} for r in requests]

    def close(self):
        pass


def _service(tmp_path):
    store = ProductStore(tmp_path / "state")
    store.setup()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    spec = ModelSpec("local/model", "revision", str(snapshot.resolve()), 1)
    store.register_model(spec)
    backend = TimedBackend()
    service = ProductService(store, backend=backend, spec=spec)
    return service, backend, spec


def test_the_scheduler_feeds_every_finished_group_to_the_agent_with_its_timings(tmp_path):
    service, backend, spec = _service(tmp_path)
    agent = _agent(tmp_path)
    service.agent = service.dispatch = agent
    try:
        threads = [threading.Thread(target=lambda n=n: list(service.stream(
            GenerationRequest(spec.model_id, (("user", "x" * (n + 1)),), max_tokens=4 + n)))) for n in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        deadline = time.time() + 5  # a caller's stream ends before the scheduler retires its group
        while len(agent.traffic) < 6 and time.time() < deadline:
            time.sleep(0.01)
        assert len(agent.traffic) == 6 and agent.counters["groups"] >= 1
        assert all(row[1] == 7 for row in agent.traffic)  # the worker's prompt tokens
        assert sorted(row[3] for row in agent.traffic) == [4, 5, 6, 7, 8, 9]
        assert service.health()["agent"]["serving"] == "reference"
    finally:
        service.close()


def test_replay_runs_the_agents_order_and_teaches_the_world_model(tmp_path):
    service, backend, spec = _service(tmp_path)
    agent = _agent(tmp_path)
    service.agent = service.dispatch = agent
    try:
        # the scheduler idles beside this call: with no traffic recorded, the agent starts no replay itself
        trace = [(0.0, 10, 64), (0.0, 10, 4), (0.0, 10, 32), (0.0, 10, 2)]
        result = service._replay_window(trace, lambda count, waiting, rate, now: (2, 0, True), lambda: False)
        assert result is not None and all(row["outcome"] == "ok" for row in result["rows"])
        # shortest output limit first: the two short requests ran together, then the two long ones
        assert backend.groups == [[2, 4], [32, 64]]
        assert agent.counters["replayed_groups"] == 2 and not agent.traffic  # replay never becomes traffic
    finally:
        service.close()


def test_the_cli_refuses_the_agent_beside_another_decision_owner():
    from ironmule_product.cli import serve
    with pytest.raises(SystemExit):
        serve(["--model", "m", "--agent", "on", "--learning", "active"])


def test_an_agent_replay_skips_idle_stretches_without_changing_a_cost(tmp_path):
    service, backend, spec = _service(tmp_path)
    try:
        trace = [(0.0, 10, 4), (30.0, 10, 4)]  # 30 s apart: the server is idle in between
        started = time.monotonic()
        result = service._replay_window(trace, lambda count, waiting, rate, now: (1, 0, False), lambda: False,
                                        skip_idle=True)
        assert time.monotonic() - started < 10 and result is not None
        assert all(row["outcome"] == "ok" and row["e2e_s"] < 5 for row in result["rows"])
    finally:
        service.close()


def test_a_learned_twin_of_a_rejected_schedule_is_excluded_too():
    fixed = ag.fixed_policy(4, True)
    twin_order = [30.0, 0.0, 0.0, 0.0]  # other weights, the same decisions everywhere
    twin_width = []
    for choice in ag.WIDTH_CHOICES:
        twin_width += [33.0 if choice == 4 else 1.0, 0.0, 0.0, 0.0]
    twin = ag.policy_json([0.0] * ag.PLACEMENT_FEATURES, twin_order, twin_width)
    assert twin["sha256"] != fixed["sha256"] and ag.behaviour(twin) == ag.behaviour(fixed)
    assert ag.behaviour(ag.fixed_policy(8, True)) != ag.behaviour(fixed)


def _arm_cost_replay(costs, fail_arm=None):
    """A replay stand-in pricing each arm by what it decides: the reference (no sjf), or a policy by its width."""
    calls = []

    def replay(trace, choose, preempted):
        width, _, sjf = choose(len(trace), [], 0.0, 0)
        index = len(calls)
        calls.append((width, sjf))
        cost = costs(width, sjf)
        rows = [{"ttft_s": cost / 4, "e2e_s": cost * 3 / 4, "tokens": 4, "outcome": "ok"} for _ in trace]
        if fail_arm is not None and index % 3 == fail_arm:
            rows[0] = {**rows[0], "outcome": "failed"}
        return {"rows": rows, "texts": ["same"] * len(trace), "span_s": 1.0}
    return replay, calls


def test_a_failure_under_the_incumbent_never_counts_for_the_candidate(tmp_path):
    agent = _agent(tmp_path)
    _feed(agent, 16 * 3)
    agent.candidate = {"policy": ag.fixed_policy(2, True), "plan": {}, "since": 0.0, "triplets": [], "used": 0}
    replay, calls = _arm_cost_replay(lambda width, sjf: 10.0, fail_arm=0)  # arm A fails, all cost the same
    agent._qualify(replay, lambda: False)
    assert [sjf for _, sjf in calls] == [False, True, False]  # always A, B, AA
    assert agent.candidate["triplets"] == [] and agent.candidate["used"] == 1


def test_a_triplet_whose_incumbent_changed_meanwhile_is_dropped(tmp_path):
    agent = _agent(tmp_path)
    _feed(agent, 32)
    _promote(agent, ag.fixed_policy(8, True))
    agent.candidate = {"policy": ag.fixed_policy(2, True), "plan": {}, "since": 0.0, "triplets": [], "used": 0}
    replay, _ = _arm_cost_replay(lambda width, sjf: 10.0)
    agent._qualify(lambda trace, choose, preempted: replay(trace, choose, preempted) if not agent.candidate.get(
        "partial", {}).get("a") else None, lambda: False)  # A runs, B is cut
    assert "a" in agent.candidate["partial"]
    agent.control("off")
    agent.control("on")  # the incumbent was withdrawn meanwhile: the kept arm ran against another schedule
    seen = []
    agent._qualify(lambda trace, choose, preempted: seen.append(1) or replay(trace, choose, preempted), lambda: False)
    assert len(seen) == 3  # all three arms ran again


def test_an_audit_withdraws_a_serving_schedule_that_loses_to_the_reference(tmp_path):
    agent = _agent(tmp_path)
    _feed(agent, 8)
    _promote(agent, ag.fixed_policy(2, True))
    _feed(agent, 16 * max(ag.LOOKS), start=500.0, gap=10.0)  # the server empties between arrivals
    replay, _ = _arm_cost_replay(lambda width, sjf: 12.0 if sjf else 10.0)  # the schedule is 20 % slower
    for _ in range(ag.LOOKS[0]):
        agent._audit(replay, lambda: False)
    assert not agent.active and "lost to the reference" in agent.withdrawn[-1]["reason"]


def test_an_audit_confirms_a_schedule_that_holds_up(tmp_path):
    agent = _agent(tmp_path)
    _feed(agent, 8)
    _promote(agent, ag.fixed_policy(2, True))
    _feed(agent, 16 * ag.MAX_TRIPLETS, start=500.0, gap=10.0)  # the server empties between arrivals
    replay, _ = _arm_cost_replay(lambda width, sjf: 9.0 if sjf else 10.0)
    for _ in range(ag.MAX_TRIPLETS):
        agent._audit(replay, lambda: False)
    assert agent.active and agent.audit["verdict"] == "confirmed"
    assert agent.history[-1]["event"] == "audited"


def test_a_width_less_policy_keeps_the_services_width_and_close_ends_idle_work(tmp_path):
    agent = _agent(tmp_path)
    body = {"schema": "ironmule.rl4.policy.v1", "placement": [0.0] * ag.PLACEMENT_FEATURES,
            "order": [5.0, 0.0, 0.0, 0.0], "features": {"placement": ag.PLACEMENT_FEATURES, "order": ag.ORDER_FEATURES}}
    import hashlib as _h
    v1 = {**body, "sha256": _h.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()}
    agent.policy = Policy.from_json(v1)
    request = GenerationRequest("m", (("user", "x"),), max_tokens=32)
    assert agent.width([request]) is None and agent.errors == 0
    agent.close()
    agent._last_activity = -1e9
    agent.idle(lambda: False, replay=lambda *a: pytest.fail("idle work after close"))


def test_a_decode_only_change_reopens_only_the_decode_fit():
    """AG1 S2 run 2: after a change of the decode price, reopening the prefill fit too let it wander 10.4 %."""
    rng = random.Random(9)
    world = WorldModel()
    for _ in range(200):
        world.observe(_group(TRUTH, rng, rng.choice((1, 2, 4, 8))))
    narrow = {**TRUTH, "step_s": 0.010, "step_s_per_session": 0.040}
    for _ in range(400):
        world.observe(_group(narrow, rng, 1 if rng.random() < 0.8 else 2, prompts=(24,)))
        assert world.prefill.p[0][0] < 1.0  # never reopened (a reopened fit starts at 100)
    assert world.reopened >= 1
    for key, value in narrow.items():
        assert world.params()[key] == pytest.approx(value, rel=0.1), key


def test_the_compiled_table_decides_exactly_as_the_policy_on_every_queue():
    for body in (ag.fixed_policy(2, True), ag.fixed_policy(8, False),
                 ag.policy_json([0.0] * ag.PLACEMENT_FEATURES, [0.4, -1.2, 2.0, -0.7],
                                [0.1 * k - 0.5 for k in range(len(ag.WIDTH_CHOICES) * ag.ORDER_FEATURES)])):
        policy = Policy.from_json(body)
        table = ag.compile_table(policy)
        rng = random.Random(1)
        for _ in range(300):
            n = rng.randint(1, ag.TABLE_QUEUE)
            views = [View(rng.choice((8, 64, 128, 300)), rng.random() < 0.3) for _ in range(n)]
            longs = sum(v.max_tokens >= ag.LONG_TOKENS for v in views)
            assert table[n, longs] == (policy.uses_sjf(views), policy.best_width(views))


def test_a_queue_longer_than_the_table_is_decided_by_the_policy(tmp_path):
    agent = _agent(tmp_path)
    _promote(agent, ag.fixed_policy(4, True))
    queue = [GenerationRequest("m", (("user", "x"),), max_tokens=32)] * (ag.TABLE_QUEUE + 5)
    assert agent.sjf(queue) is True and agent.width(queue) == 4


V3_MODEL = {"prefill_s": 0.1056, "prefill_s_per_ktoken": 1.404, "group_s": 0.0189, "step_s": 0.0203,
            "step_s_per_session": 0.0180}  # the world model the agent ended AG1 V3 with (Kaggle T4, Qwen3-1.7B)


def _bursty_arrivals(seed, traces=6):
    """V3-like traffic: 64-request spikes inside one second, an idle gap, then quiet arrivals; outputs 32 or 256
    tokens, half the prompts long. [offset s, prompt tokens, long, output limit, tokens made]."""
    rng, out, t = random.Random(seed), [], 0.0
    for _ in range(traces):
        for offset in sorted(rng.uniform(0, 1.0) for _ in range(64)):
            long = rng.random() < 0.5
            limit = rng.choice((32, 256))
            out.append([t + offset, 1030 if long else 20, long, limit, limit])
        t += 1500.0
        for _ in range(16):
            limit = rng.choice((32, 256))
            out.append([t, 20, False, limit, limit])
            t += 30.0
        t += 1500.0
    return out


def test_a_busy_window_spans_the_whole_burst_and_stops_where_the_server_empties():
    arrivals = _bursty_arrivals(1, traces=1)
    jobs = ag._jobs(arrivals)
    assert ag.busy_size(V3_MODEL, jobs, REFERENCE, 8, 4) == 64  # the whole spike, not its first 16
    quiet = ag._jobs(arrivals[64:])
    assert ag.busy_size(V3_MODEL, quiet, REFERENCE, 8, 4) == ag.TRACE  # quiet traffic empties after every arrival
    assert ag.busy_size(V3_MODEL, jobs, REFERENCE, 8, 4, maximum=40) == 40  # a server that never empties is cut
    starts = ag.busy_starts(V3_MODEL, arrivals, REFERENCE, 8, 4)
    assert 0 in starts and not any(0 < k < 64 for k in starts)


def test_busy_windows_stop_the_planner_preferring_the_narrow_schedule_v3_served_slower():
    """AG1 V3: on 16-arrival windows the model priced `w2/sjf` below `w8/fifo`; served on whole traces it lost
    6.5 %. On whole busy periods the same model ranks it below the default, as served."""
    arrivals = _bursty_arrivals(2)
    payload = {"model": V3_MODEL, "arrivals": arrivals, "capacity": 8, "engine_width": 4, "seed": 0,
               "widths": [1, 2, 4, 8], "reference": REFERENCE,
               "settings": {"episodes": 20, "seeds": 1, "validation": 24, "held_out": 24}}
    fixed = plan({**payload, "settings": {**payload["settings"], "busy": False}})["arm_costs"]
    busy = plan(payload)["arm_costs"]
    assert fixed["w2/sjf"] < fixed["w8/fifo"]  # the V3 mistake, reproduced
    assert busy["w2/sjf"] > busy["w8/fifo"]
    assert min(busy, key=busy.get) in ("w8/sjf", "w8/fifo")


def test_a_replay_window_spans_the_backlog_and_the_next_window_starts_after_it(tmp_path):
    agent = _agent(tmp_path)
    _feed(agent, 32)  # one arrival a second, each about 3 s of work: the server never empties
    agent.candidate = {"policy": ag.fixed_policy(2, True), "plan": {}, "since": 0.0, "triplets": [], "used": 0}
    seen = []

    def replay(trace, choose, preempted):
        seen.append(len(trace))
        rows = [{"ttft_s": 1.0, "e2e_s": 3.0, "tokens": 4, "outcome": "ok"} for _ in trace]
        return {"rows": rows, "texts": ["same"] * len(trace), "span_s": 1.0}

    agent._qualify(replay, lambda: False)
    assert seen == [31] * 3  # every recorded arrival after the plan: the burst never ended
    assert agent.candidate["trace"] is None and agent.candidate["used"] == 1
    # the next window starts after this one's last recorded arrival (a time, not a count: review of 2026-11-08)
    last = max(agent.traffic, key=lambda a: (a[0], a[6]))
    assert agent.candidate["skip"] == 0 and agent.candidate["cursor"] == [last[0], last[6]]  # (time, id)


# ---------------------------------------------------------------- AG2: ordering by a predicted answer length

def _answers(rng, n):
    """Prompts whose kind sets the answer's length, all with one output limit: the chat traffic AG2 is for."""
    kinds = [("Answer yes or no: is {} red?", 3), ("Name the capital of {}.", 8),
             ("Write a long essay about {}.", 300), ("Explain in detail how {} works.", 180)]
    for k in range(n):
        text, tokens = rng.choice(kinds)
        yield text.format(f"thing{rng.randrange(50)}"), max(1, int(tokens * rng.uniform(0.8, 1.2)))


def test_the_length_model_learns_answer_lengths_online_and_keeps_no_text():
    from ironmule_product.length_model import LengthModel, features
    model = LengthModel()
    rng = random.Random(1)
    for prompt, tokens in _answers(rng, 600):
        request = GenerationRequest("m", (("user", prompt),), max_tokens=512)
        feats = features(request)
        model.learn(feats, tokens, model.predict(feats, 512))
    assert model.quality() > 0.8
    short = features(GenerationRequest("m", (("user", "Answer yes or no: is thing7 red?"),), max_tokens=512))
    long = features(GenerationRequest("m", (("user", "Write a long essay about thing7."),), max_tokens=512))
    assert model.predict(short, 512) < 20 < 150 < model.predict(long, 512)
    assert model.predict(long, 64) == 64  # never above the request's own limit
    saved = json.dumps(model.to_json())
    assert "essay" not in saved and "thing" not in saved
    assert LengthModel.from_json(json.loads(saved)).predict(long, 512) == pytest.approx(model.predict(long, 512))


def test_a_limit_ordered_policy_keeps_its_digest_and_a_predicted_one_is_another_schedule():
    limit, predicted = ag.fixed_policy(2, True), ag.fixed_policy(2, True, "predicted")
    assert "order_key" not in limit and Policy.from_json(limit).key == "limit"
    assert Policy.from_json(predicted).key == "predicted" and predicted["sha256"] != limit["sha256"]
    assert ag.behaviour(predicted) != ag.behaviour(limit)
    assert ag._policy_body(Policy.from_json(predicted)) == predicted
    tampered = {**predicted, "order_key": "oracle"}
    with pytest.raises(ValueError):
        Policy.from_json(tampered)


def test_ld1_refuses_a_policy_that_orders_by_prediction(tmp_path):
    from ironmule_product.learned_dispatch import EVIDENCE_SCHEMA, LearnedDispatch, _digest
    policy = ag.fixed_policy(2, True, "predicted")
    body = {"schema": EVIDENCE_SCHEMA, "policy_sha256": policy["sha256"], "verdict": "pass", "binding": {}}
    (tmp_path / "p.json").write_text(json.dumps(policy))
    (tmp_path / "e.json").write_text(json.dumps({**body, "sha256": _digest(body)}))
    dispatch = LearnedDispatch.load(tmp_path / "p.json", tmp_path / "e.json", {})
    assert not dispatch.active and "only the agent" in dispatch.reason


def test_the_agent_predicts_once_learns_from_live_answers_and_records_the_prediction(tmp_path):
    agent = _agent(tmp_path)
    rng = random.Random(2)
    for k, (prompt, tokens) in enumerate(_answers(rng, 400)):
        request = GenerationRequest("m", (("user", prompt),), max_tokens=512, request_id=f"r{k}")
        known = agent.length_of(request)
        assert agent.length_of(request) is known  # made once, when first seen
        agent.observe_group([{"arrive": float(k), "start": k + 0.1, "first": k + 0.2, "end": k + 0.2 + tokens * 0.01,
                              "prompt_tokens": 12, "prompt_chars": len(prompt), "max_tokens": 512, "tokens": tokens,
                              "ok": True, "length": known, "request_id": f"r{k}"}])
    assert agent.lengths.seen == 400 and agent.status()["lengths"]["spearman"] > 0.8
    assert all(len(row) == 7 and row[5] is not None for row in agent.traffic)  # the 7th field is the arrival's id
    assert not agent._predicted  # a finished request's prediction is not kept
    assert ag.predictions_recorded(list(agent.traffic))
    # no serving policy orders by prediction yet: the service sorts by the limit
    assert agent.order_key(GenerationRequest("m", (("user", "Write a long essay about x."),), max_tokens=512)) is None


def test_the_planner_proposes_shortest_predicted_first_where_limits_say_nothing():
    rng = random.Random(4)
    arrivals, t = [], 0.0
    for burst in range(30):  # bursts of eight, every limit 512, answers of 3 to 300 tokens
        for _ in range(8):
            tokens = rng.choice((3, 8, 180, 300))
            predicted = tokens * rng.uniform(0.7, 1.3)
            arrivals.append([t, 24, False, 512, tokens, predicted])
            t += 0.05
        t += 60.0
    result = plan({"model": TRUTH, "arrivals": arrivals, "capacity": 1, "reference": {"width": 1, "sjf": False},
                   "settings": {"episodes": 200, "seeds": 1, "validation": 10, "held_out": 10}})
    assert result["best_fixed"] == "w1/pred"
    assert Policy.from_json(result["candidate"]).key == "predicted"
    assert result["arm_costs"]["w1/pred"] < 0.8 * result["arm_costs"]["w1/sjf"]
    # without recorded predictions the arm is not offered
    bare = plan({"model": TRUTH, "arrivals": [a[:5] for a in arrivals], "capacity": 1,
                 "reference": {"width": 1, "sjf": False},
                 "settings": {"episodes": 200, "seeds": 1, "validation": 10, "held_out": 10}})
    assert "w1/pred" not in bare["arm_costs"]


def test_a_predicted_policy_orders_live_requests_by_their_prediction(tmp_path):
    agent = _agent(tmp_path)
    agent.policy = Policy.from_json(ag.fixed_policy(1, True, "predicted"))
    long = GenerationRequest("m", (("user", "a"),), max_tokens=512, request_id="long")
    short = GenerationRequest("m", (("user", "b"),), max_tokens=512, request_id="short")
    agent._predicted["long"], agent._predicted["short"] = (((), 0.0, 0.0), 300.0), (((), 0.0, 0.0), 5.0)
    assert agent.order_key(long) == 300.0 and agent.order_key(short) == 5.0
    assert agent.order_key(GenerationRequest("m", (("user", "c"),), max_tokens=4, request_id="long")) == 4.0
    agent.paused = True
    assert agent.order_key(short) is None


def test_a_replay_makes_each_answer_its_recorded_length_and_sorts_by_the_arms_key(tmp_path):
    service, backend, spec = _service(tmp_path)
    agent = _agent(tmp_path)
    service.agent = service.dispatch = agent
    try:
        # (offset, chars, tokens to make, recorded limit, recorded prediction): one limit for all
        trace = [(0.0, 10, 64, 512, 70.0), (0.0, 10, 4, 512, 3.0), (0.0, 10, 32, 512, 40.0), (0.0, 10, 2, 512, 2.5)]
        by_limit = service._replay_window(trace, lambda count, waiting, rate, now: (2, 0, True), lambda: False)
        assert by_limit is not None and backend.groups == [[64, 4], [32, 2]]  # equal limits: arrival order
        backend.groups.clear()
        by_prediction = service._replay_window(trace, lambda count, waiting, rate, now: (2, 0, True, "predicted"),
                                               lambda: False)
        assert by_prediction is not None and backend.groups == [[2, 4], [32, 64]]
    finally:
        service.close()


def test_the_agents_trace_replays_recorded_lengths_and_predictions(tmp_path):
    agent = _agent(tmp_path)
    for k in range(20):
        agent.traffic.append([100.0 + k * 50, 12, False, 512, 10 + k, 12.5 + k])
    trace = agent._trace(agent.traffic, since=None)
    assert trace[0][2:] == (10, 512, 12.5)
    agent.traffic.clear()
    for k in range(20):  # recorded before AG2: ordered by the limit under either key
        agent.traffic.append([100.0 + k * 50, 12, False, 600, 10 + k])
    assert agent._trace(agent.traffic, since=None)[0][2:] == (10, 512, 512.0)


def test_agents_on_several_cards_share_one_length_model_and_learn_from_each_other(tmp_path):
    first = Agent(tmp_path / "agent", {**BINDING, "card": "0"}, capacity=1, reference=REFERENCE)
    second = Agent(tmp_path / "agent", {**BINDING, "card": "1"}, capacity=1, reference=REFERENCE)
    second.lengths.seen = 5  # the copy with more lessons survives a restart
    survivor = second.lengths
    ag.share_lengths([first, second])
    assert first.lengths is second.lengths is survivor
    request = GenerationRequest("m", (("user", "Write a long essay about the sea."),), max_tokens=512, request_id="x")
    known = first.length_of(request)
    first.observe_group([{"arrive": 1.0, "start": 1.1, "first": 1.2, "end": 9.0, "prompt_tokens": 9,
                          "prompt_chars": 33, "max_tokens": 512, "tokens": 400, "ok": True, "length": known,
                          "request_id": "x"}])
    assert second.lengths.seen == 6


def test_the_agent_knows_its_hardware_and_reports_how_much_of_its_bandwidth_decode_uses(tmp_path):
    """AG3: a T4 (320 GB/s) reading 3.4 GB of weights per step has a 10.6 ms floor; the world model's measured
    step against it is the share of bandwidth used, and its inverse the most a bandwidth-bound kernel could gain."""
    from ironmule_product.hardware_profile import bandwidth_gb_s, profile, utilisation

    assert bandwidth_gb_s("Tesla T4") == 320.0 and bandwidth_gb_s("Apple M1 Max") == 400.0
    assert bandwidth_gb_s("NVIDIA A100-SXM4-80GB") == 2039.0 and bandwidth_gb_s("NVIDIA A100-PCIE-40GB") == 1555.0
    assert bandwidth_gb_s("Intel(R) Xeon(R) CPU") is None
    hardware = profile("Tesla T4", 3_400_000_000, None)
    assert hardware["decode_floor_ms"] == pytest.approx(10.625) and hardware["reason"] is None
    assert profile("Tesla T4", 3_400_000_000, "float32")["decode_floor_ms"] is None  # widened weights: no floor
    assert profile("Unknown GPU", 3_400_000_000, None)["reason"].startswith("the device's memory bandwidth")
    used = utilisation(hardware, 0.0187)
    assert used["bandwidth_share"] == pytest.approx(0.568, abs=1e-3) and used["kernel_headroom_x"] == pytest.approx(1.76)
    agent = Agent(tmp_path / "agent", BINDING, capacity=8, reference=REFERENCE, hardware=hardware)
    assert agent.status()["hardware"]["utilisation"] is None  # nothing measured yet
    _feed(agent, 40)
    reported = agent.status()["hardware"]
    assert reported["device"] == "Tesla T4" and reported["utilisation"]["decode_ms"] == pytest.approx(19.0, rel=0.1)
    assert _agent(tmp_path / "other").status()["hardware"] is None  # no profile, no claim


class _Stop(BaseException):
    """Stops `serve` at the first worker it builds; a BaseException, so the agent's start fallback lets it pass."""


def _serve_until_backend(monkeypatch, tmp_path, argv, setup):
    """Run `serve` until it builds the worker client; returns that client's keyword arguments."""
    import contextlib

    from ironmule_product import backend as backend_module, calibration, cli
    seen = {}

    class Store:
        root = tmp_path

        def __init__(self, *_):
            pass

        def settings(self):
            return {}

        def model(self, name):
            return type("Spec", (), {"model_id": name, "revision": "r"})()

    def client(spec, **kwargs):
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(cli, "ProductStore", Store)
    monkeypatch.setattr(calibration, "model_lease", lambda store: contextlib.nullcontext())
    monkeypatch.setattr(cli, "_agent_setup", lambda spec: seen.setdefault("setup", spec) and setup)
    monkeypatch.setattr(backend_module, "MLXWorkerClient", client)
    with pytest.raises(_Stop):
        cli.serve(["--model", "m", *argv])
    return seen


def test_with_the_agent_on_and_no_plan_named_the_agent_chooses_the_engine(monkeypatch, tmp_path):
    """AG4: `--agent on` without `--engine` and `--compute-dtype` takes the agent's own setup."""
    dense = {"engine": "dense", "compute_dtype": "native", "reason": "2.4x"}
    seen = _serve_until_backend(monkeypatch, tmp_path, ["--agent", "on"], dense)
    assert "setup" in seen and seen["execution_variant"] == "current_engine"
    assert seen["engine_configuration"] == "dense_stacked" and seen["compute_dtype"] is None
    plain = {"engine": "reference", "compute_dtype": "native", "reason": "no stacked engine"}
    seen = _serve_until_backend(monkeypatch, tmp_path, ["--agent", "on"], plain)
    assert seen["execution_variant"] == "reference" and seen["compute_dtype"] == "native"


def test_a_named_plan_is_kept_even_with_the_agent_on(monkeypatch, tmp_path):
    seen = _serve_until_backend(monkeypatch, tmp_path, ["--agent", "on", "--engine", "tuned"],
                                {"engine": "dense", "compute_dtype": None, "reason": "x"})
    assert "setup" not in seen and seen["execution_variant"] == "current_engine"
    assert "engine_configuration" not in seen
    seen = _serve_until_backend(monkeypatch, tmp_path, [], {"engine": "dense", "compute_dtype": None, "reason": "x"})
    assert "setup" not in seen and seen["execution_variant"] == "reference"


def test_the_agents_two_path_setup_starts_both_workers_on_one_card(monkeypatch, tmp_path):
    """AG4 step 2: `route` starts the single path (NUM1's plan) first; the stacked engine beside it."""
    setup = {"engine": "route", "compute_dtype": "float16", "reason": "routes each request",
             "stacked": {"speed": {"short": {"base_step_s": 0.0173, "base_first_s": 0.06, "plan_first_s": 0.3,
                                             "plan_step_s": {"1": 0.021, "8": 0.028}}}}}
    seen = _serve_until_backend(monkeypatch, tmp_path, ["--agent", "on"], setup)
    assert seen["compute_dtype"] == "float16" and seen.get("execution_variant", "reference") == "reference"


def _serve_recording(monkeypatch, tmp_path, argv, setup, start_paths=None, start_fails=0):
    """Run `serve` until it builds a worker that would start; returns every worker client's keyword arguments.
    `start_fails` workers fail to start first (as a card out of memory does)."""
    import contextlib

    from ironmule_product import backend as backend_module, calibration, cli, device_pool
    built, failing = [], [start_fails]

    class Store:
        root = tmp_path

        def __init__(self, *_):
            pass

        def settings(self):
            return {}

        def model(self, name):
            return type("Spec", (), {"model_id": name, "revision": "r"})()

    class Client:
        def __init__(self, spec, **kwargs):
            built.append(kwargs)

        def start(self):
            if failing[0]:
                failing[0] -= 1
                raise RuntimeError("cudaMalloc: out of memory")
            raise _Stop

        def close(self):
            pass

    monkeypatch.setattr(cli, "ProductStore", Store)
    monkeypatch.setattr(calibration, "model_lease", lambda store: contextlib.nullcontext())
    monkeypatch.setattr(cli, "_agent_setup", lambda spec: setup)
    monkeypatch.setattr(backend_module, "MLXWorkerClient", Client)
    if start_paths is not None:
        monkeypatch.setattr(device_pool, "start_paths", start_paths)
    with pytest.raises(_Stop):
        cli.serve(["--model", "m", *argv])
    return built


_ROUTE = {"engine": "route", "compute_dtype": "float16", "reason": "routes each request", "width": 8,
          "stacked": {"speed": {"short": {"base_step_s": 0.0173, "base_first_s": 0.06, "plan_first_s": 0.3,
                                          "plan_step_s": {"1": 0.021, "8": 0.028}}}}}


def test_when_the_agents_two_paths_do_not_start_its_single_path_serves(monkeypatch, tmp_path, capsys):
    """Stability (owner, 2026-11-07): a choice of the agent's that cannot start never stops the server. The
    two paths not starting (a card out of memory) leaves the agent's gated plan on the reference engine."""
    from ironmule_product.errors import BackendUnavailable

    def start_paths(*args, **kwargs):
        raise BackendUnavailable("the worker for CUDA device 0:stacked did not start")

    built = _serve_recording(monkeypatch, tmp_path, ["--agent", "on"], _ROUTE, start_paths=start_paths)
    assert built[-1]["compute_dtype"] == "float16" and built[-1]["execution_variant"] == "reference"
    assert "engine_configuration" not in built[-1]
    assert "the agent chose route, which did not start" in capsys.readouterr().out


def test_when_the_agents_stacked_engine_does_not_start_its_plan_then_stock_serve(monkeypatch, tmp_path, capsys):
    dense = {"engine": "dense", "compute_dtype": "native", "reason": "2.0x", "width": 6}
    built = _serve_recording(monkeypatch, tmp_path, ["--agent", "on"], dense, start_fails=2)
    assert [b.get("engine_configuration") for b in built] == ["dense_stacked", None, None]
    assert built[0]["dense_width"] == 6
    assert [b["compute_dtype"] for b in built] == [None, "native", None], "its gated plan first, then stock"
    assert capsys.readouterr().out.count('"fallback"') == 2


def test_a_plan_the_caller_named_is_kept_when_it_does_not_start(monkeypatch, tmp_path):
    """The caller's plan is preserved (AGENTS.md): `--engine dense` that cannot start is reported, not swapped."""
    import contextlib

    from ironmule_product import backend as backend_module, calibration, cli

    class Store:
        root = tmp_path

        def __init__(self, *_):
            pass

        def settings(self):
            return {}

        def model(self, name):
            return type("Spec", (), {"model_id": name, "revision": "r"})()

    built = []

    class Client:
        def __init__(self, spec, **kwargs):
            built.append(kwargs)

        def start(self):
            raise RuntimeError("cudaMalloc: out of memory")

        def close(self):
            pass

    monkeypatch.setattr(cli, "ProductStore", Store)
    monkeypatch.setattr(calibration, "model_lease", lambda store: contextlib.nullcontext())
    monkeypatch.setattr(backend_module, "MLXWorkerClient", Client)
    with pytest.raises(RuntimeError, match="out of memory"):
        cli.serve(["--model", "m", "--engine", "dense"])
    assert len(built) == 1 and built[0]["engine_configuration"] == "dense_stacked"


def test_the_agents_two_paths_on_several_cards_fall_back_to_its_plan_on_each(monkeypatch, tmp_path, capsys):
    from ironmule_product import device_pool

    monkeypatch.setattr(device_pool, "cuda_devices", lambda: ["0", "1"])
    monkeypatch.setattr(device_pool, "select_devices", lambda gpus, devices: ["0", "1"])

    def start_pool(store, spec, devices, make_backend, service_type):
        client = make_backend(devices[0])
        client.start()

    monkeypatch.setattr(device_pool, "start_pool", start_pool)
    built = _serve_recording(monkeypatch, tmp_path, ["--agent", "on", "--gpus", "all"], _ROUTE)
    assert built[-1]["compute_dtype"] == "float16" and built[-1]["execution_variant"] == "reference"
    assert "two paths share one card" in capsys.readouterr().out


def test_the_agents_setup_never_stops_the_server(monkeypatch):
    """Whatever fails in the agent's measurement (here the setup itself raising) leaves the reference engine."""
    import importlib
    import subprocess

    import ironmule.autopilot as autopilot
    import ironmule.plan_choice as plan_choice
    from ironmule_product import cli

    tune = importlib.import_module("ironmule.tune")  # `ironmule.tune` the attribute is a function

    monkeypatch.setattr(autopilot, "_probe_out_of_process", lambda probe: {"fingerprint": "f"})
    monkeypatch.setattr(tune, "resolve_local_model",
                        lambda model, revision=None: type("R", (), {"identity": type("I", (), {
                            "identity_sha256": "a" * 64})()})())

    def broken(*args, **kwargs):
        raise subprocess.TimeoutExpired(["python"], 3600)

    monkeypatch.setattr(plan_choice, "choose_setup", broken)
    setup = cli._agent_setup(type("Spec", (), {"model_id": "m", "revision": "r"})())
    assert setup["engine"] == "reference" and setup["compute_dtype"] is None
    assert setup["reason"].startswith("the agent's setup failed (TimeoutExpired")


def test_the_agents_two_path_setup_serves_through_the_learning_router(monkeypatch, tmp_path):
    """AG6: `route` hands both paths to the policy router, which starts in `route` at the agent's width."""
    from ironmule_product import device_pool
    from ironmule_product.path_router import PolicyRouter
    setup = {"engine": "route", "compute_dtype": "float16", "reason": "routes each request", "width": 6,
             "stacked": {"speed": {"short": {"base_step_s": 0.0173, "base_first_s": 0.06, "plan_first_s": 0.3,
                                             "plan_step_s": {"1": 0.021, "6": 0.026, "8": 0.028}}}}}
    taken = {}

    def start_paths(store, spec, makers, labels, service, router):
        taken.update(labels=labels, router=router)
        raise _Stop

    monkeypatch.setattr(device_pool, "start_paths", start_paths)
    _serve_until_backend(monkeypatch, tmp_path, ["--agent", "on"], setup)
    router = taken["router"]
    assert isinstance(router, PolicyRouter) and router.mode == "route"
    assert taken["labels"] == ["0:single", "0:stacked"] and router.costs[1].capacity == 6


# ---------------------------------------------------------------- review of 2026-11-08: qualification evidence

_REFERENCE = {"width": 8, "sjf": False, "gather_ms": 0}


def _review_result(trace, cost):
    return {"rows": [{"ttft_s": cost / 4, "e2e_s": cost * 3 / 4, "tokens": 64, "outcome": "ok"} for _ in trace],
            "texts": ["same"] * len(trace), "span_s": 1.0}


def test_qualification_keeps_finding_windows_after_the_traffic_memory_drops_old_arrivals(tmp_path):
    """The cursor counted used arrivals, but the memory keeps only MAX_TRAFFIC of them: once 2,048 were used, no
    window was ever found again, and the open candidate blocked replanning for good. It is now a recorded time."""
    agent = Agent(tmp_path, {"review": "sliding-window"}, capacity=8, reference=_REFERENCE, objective="latency")  # AG1's cost, as reviewed
    agent.traffic.extend([[1 + i * .001, 24, False, 64, 64, None] for i in range(ag.MAX_TRAFFIC)])
    agent.candidate = {"policy": ag.fixed_policy(2, True), "plan": {}, "since": 0., "triplets": [], "used": 0}
    calls = [0]

    def replay(trace, choose, preempted):
        index = calls[0]
        calls[0] += 1
        return _review_result(trace, 10. if index % 3 != 1 else 9. if (index // 3) % 2 == 0 else 11.)

    for _ in range(16):
        agent._qualify(replay, lambda: False)
    assert len(agent.candidate["triplets"]) == 16 and calls[0] == 48
    agent.traffic.extend([[100 + i * .001, 24, False, 64, 64, None] for i in range(4096)])
    before = calls[0]
    for _ in range(4):
        agent._qualify(replay, lambda: False)
    assert calls[0] > before, "new arrivals give new windows"
    assert agent.candidate is None or len(agent.candidate["triplets"]) > 16


def test_a_candidates_triplets_against_a_withdrawn_schedule_do_not_count_against_the_new_one(tmp_path):
    """Eleven triplets against a schedule that was then withdrawn, judged with one against the reference that
    serves now, promoted a candidate that costs 50 % more than that reference. Its evidence starts over."""
    agent = Agent(tmp_path, {"review": "incumbent-change"}, capacity=8, reference=_REFERENCE, objective="latency")  # AG1's cost, as reviewed
    agent.traffic.extend([[1 + i * 100, 24, False, 64, 64, None] for i in range(16 * 12)])
    incumbent = {"policy": ag.fixed_policy(8, True), "plan": {}, "since": 0., "triplets": [], "used": 0}
    agent._promote(incumbent, {"gain": .1, "gain_ci95": [.05, .15], "floor": 0., "noise": 0.})
    agent.candidate = {"policy": ag.fixed_policy(2, True), "plan": {}, "since": 0., "triplets": [], "used": 0}

    def replay(trace, choose, preempted):  # candidate 15, the old incumbent 20, the reference 10
        width, _, sjf = choose(len(trace), [], 0., 0)
        return _review_result(trace, 15. if width == 2 else 20. if sjf else 10.)

    for _ in range(11):
        agent._qualify(replay, lambda: False)
    assert len(agent.candidate["triplets"]) == 11
    agent.control("off")
    agent.control("on")
    agent._qualify(replay, lambda: False)
    assert not agent.active, "the candidate loses to the reference serving now"
    assert len(agent.candidate["triplets"]) == 1 and agent.candidate["incumbent"] == "reference"
    assert any(event["event"] == "restarted" for event in agent.history)


def test_rank_correlation_shares_ranks_between_ties_and_is_undefined_for_a_constant():
    from ironmule_product.length_model import spearman

    from training.rl4_agent import _spearman

    assert spearman([64.] * 8, list(range(1, 9))) is None
    assert spearman([64.] * 8, [64.] * 8) is None
    assert spearman([1., 1., 2., 3.], [1., 2., 3., 4.]) == pytest.approx(0.9486832980505138)  # SciPy's value
    assert spearman([1., 2., 3.], [3., 2., 1.]) == pytest.approx(-1.0)
    assert _spearman([64.] * 8, list(range(1, 9))) is None, "the training copy is the product's now"


def test_requests_that_fail_before_their_first_token_reach_the_live_monitor(tmp_path):
    """Review of 2026-11-08: a group with a request that failed before its first token was dropped whole, so a
    policy failing requests that early was never withdrawn; one failing them after their first token was."""
    from ironmule_product.service import _Session

    for before_first in (True, False):
        agent = Agent(tmp_path / str(before_first), {}, capacity=8, reference=_REFERENCE, objective="latency")  # AG1's cost, as reviewed
        agent.policy = Policy.from_json(ag.fixed_policy(2, True))
        service = object.__new__(ProductService)
        service.agent = agent
        for n in range(2 * ag.MONITOR_WINDOW):
            now = time.monotonic_ns()
            session = _Session(GenerationRequest("local/test", (("user", "hi"),), max_tokens=2, request_id=str(n)),
                               None, submitted_ns=now - 3_000_000, dispatched_ns=now - 2_000_000,
                               first_ns=None if before_first else now - 1_000_000, end_ns=now, outcome="failed")
            service._agent_observe([session])
        assert not agent.active and len(agent.withdrawn) == 1, before_first


# ---------------------------------------------------------------- the cursor at equal timestamps (owner, 2026-11-08)

def _marked_spike(agent, n, tied, at=10.0, order=None):
    """`n` arrivals in one spike a microsecond apart, each marked by its own output limit (1 + k, below the
    replay's cap), the ones in `tied` all at the first tied one's time, recorded in `order` (default: k); a spike
    keeps the server busy, so its windows are cut at MAX_WINDOW."""
    first = min(tied)
    for k in (order or range(n)):
        arrive = at + (first if k in tied else k) * 1e-6
        group = [{"arrive": arrive, "start": arrive, "first": arrive + 0.1, "end": arrive + 1.0,
                  "prompt_tokens": 24, "prompt_chars": 96, "max_tokens": 1 + k, "tokens": 1 + k, "ok": True}]
        agent.observe_group(group)


def _windows(agent, rounds):
    """The marks each replayed window held, and every arm's trace per window (all three must match)."""
    windows, arms = [], []

    def replay(trace, choose, preempted):
        arms.append(tuple(row[3] for row in trace))
        return {"rows": [{"ttft_s": 1.0, "e2e_s": 3.0, "tokens": 4, "outcome": "ok"} for _ in trace],
                "texts": ["same"] * len(trace), "span_s": 1.0}

    for _ in range(rounds):
        before = len(arms)
        agent._qualify(replay, lambda: False)
        if len(arms) > before:
            windows.append(arms[before])
            assert arms[before:] == [arms[before]] * 3, "every arm of a triplet replays the same window"
    return windows


def test_arrivals_sharing_a_time_across_a_window_cut_are_each_used_exactly_once(tmp_path):
    """The window is cut at MAX_WINDOW inside a spike, where twelve arrivals share one recorded time on both sides
    of the cut. A cursor of the cut's time alone skipped the tied arrivals that did not fit the first window."""
    agent = _agent(tmp_path)
    n = 2 * ag.MAX_WINDOW
    tied = set(range(ag.MAX_WINDOW - 6, ag.MAX_WINDOW + 6))
    _marked_spike(agent, n, tied)
    agent.candidate = {"policy": ag.fixed_policy(2, True), "plan": {}, "since": 0.0, "triplets": [], "used": 0}
    windows = _windows(agent, 4)
    used = [mark - 1 for window in windows for mark in window]
    assert len(windows[0]) == ag.MAX_WINDOW, "the first window is cut inside the tie"
    assert 0 < len(tied & {mark - 1 for mark in windows[0]}) < len(tied)
    assert sorted(used) == list(range(n)), "every arrival used once: none skipped at the tie, none counted twice"


def test_an_interrupted_triplet_resumes_on_its_window_across_a_restart_and_the_next_window_follows_it(tmp_path):
    """Demand cuts a triplet after its first arm and the server restarts: the saved window, cursor and kept arm are
    loaded, the other arms replay the same arrivals, and the next window starts after it, ties included."""
    n = 2 * ag.MAX_WINDOW
    tied = set(range(ag.MAX_WINDOW - 6, ag.MAX_WINDOW + 6))
    agent = _agent(tmp_path)
    _marked_spike(agent, n, tied)
    agent.candidate = {"policy": ag.fixed_policy(2, True), "plan": {}, "since": 0.0, "triplets": [], "used": 0}
    arms = []

    def replay(trace, choose, preempted, cut=[1]):
        if len(arms) == cut[0]:  # demand arrives during the second arm, once
            cut[0] = -1
            return None
        arms.append(tuple(row[3] for row in trace))
        return {"rows": [{"ttft_s": 1.0, "e2e_s": 3.0, "tokens": 4, "outcome": "ok"} for _ in trace],
                "texts": ["same"] * len(trace), "span_s": 1.0}

    agent._qualify(replay, lambda: False)
    assert len(arms) == 1 and agent.candidate["trace"] and agent.candidate["used"] == 0
    agent._dirty()
    agent.save(force=True)
    again = _agent(tmp_path)  # a restart: everything comes from the saved state
    assert again.candidate["trace_end"] == agent.candidate["trace_end"] and again._next_id == n
    again._qualify(replay, lambda: False)
    assert len(arms) == 3 and arms[1:] == [arms[0]] * 2, "the other arms replayed the saved window"
    again._qualify(replay, lambda: False)
    used = [mark - 1 for window in (arms[0], arms[3]) for mark in window]
    assert arms[3:] == [arms[3]] * 3 and sorted(used) == list(range(n))


def test_arrivals_the_memory_drops_before_use_are_counted_and_qualification_goes_on(tmp_path):
    agent = _agent(tmp_path)
    agent.candidate = {"policy": ag.fixed_policy(2, True), "plan": {}, "since": -1.0, "since_id": -1,
                       "cursor": [5.0, 10], "triplets": [], "used": 0}
    _marked_spike(agent, ag.MAX_TRAFFIC + 40, {0}, at=10.0)  # 40 of them pushed out unused
    assert agent.candidate["lost"] == 40 and len(agent.traffic) == ag.MAX_TRAFFIC
    windows = _windows(agent, 2)
    assert windows and windows[0][0] == 41, "it goes on with the oldest arrival it still holds"


def test_a_cursor_saved_as_a_time_alone_means_every_arrival_at_that_time_was_used(tmp_path):
    agent = _agent(tmp_path)
    _marked_spike(agent, 40, {10, 11, 12})
    tied_time = 10.0 + 10e-6
    agent.candidate = {"policy": ag.fixed_policy(2, True), "plan": {}, "since": -1.0, "cursor": tied_time,
                       "triplets": [], "used": 0}
    windows = _windows(agent, 1)
    assert windows and min(mark - 1 for mark in windows[0]) == 13, "a legacy cursor skips its whole time, as saved"


def test_arrivals_recorded_after_the_plan_at_the_plans_last_time_are_held_out_by_id(tmp_path):
    agent = _agent(tmp_path)
    _marked_spike(agent, 20, {18, 19})  # 18 and 19 share a time
    plan_id = 18  # the planner saw arrivals up to id 18 (mark 19); mark 20 came after it, at the same time
    agent.candidate = {"policy": ag.fixed_policy(2, True), "plan": {}, "since": 10.0 + 18e-6, "since_id": plan_id,
                       "triplets": [], "used": 0}
    _marked_spike(agent, 16, {0}, at=20.0)
    windows = _windows(agent, 1)
    assert windows and sorted(windows[0])[0] == 1 and 20 in windows[0], "id 19 (mark 20) is held out, ids <= 18 not"


def _simulate_by_rounds(params, jobs, sjf, width, capacity, engine_width=ag.ENGINE_WIDTH, key="limit"):
    """`simulate` as it was before A1 (2026-11-09), stepping every decode round through the lists: the reference
    the event-driven one must equal float for float."""
    import statistics

    pending = sorted(jobs, key=lambda j: (j.arrive, j.index))
    waiting, now, i = [], 0.0, 0
    while i < len(pending) or waiting:
        while i < len(pending) and pending[i].arrive <= now:
            waiting.append(pending[i])
            i += 1
        if not waiting:
            now = pending[i].arrive
            continue
        if sjf(waiting):
            waiting.sort(key=ag._sjf_key(now, key))
        size = max(1, min(capacity, width(waiting)))
        group, waiting = waiting[:size], waiting[size:]
        t = now + params.get("group_s", 0.0)
        remaining = {}
        for job in group:
            t += params["prefill_s"] + params["prefill_s_per_ktoken"] * job.prompt_tokens / 1000
            job.first = t
            remaining[job.index] = job.tokens - 1
            if remaining[job.index] <= 0:
                job.end = t
        active = [j for j in group if remaining[j.index] > 0]
        while active:
            served = active[:engine_width]
            t += params["step_s"] + params["step_s_per_session"] * (len(served) - 1)
            for job in served:
                remaining[job.index] -= 1
                if remaining[job.index] == 0:
                    job.end = t
            active = active[len(served):] + [j for j in served if remaining[j.index] > 0]
        now = t
    return statistics.fmean((j.end - j.arrive) + (j.first - j.arrive) for j in jobs) if jobs else 0.0


def test_the_event_driven_simulator_equals_the_round_by_round_one_float_for_float():
    """A1: jumping to the next round in which a session ends changes no number the planner sees: every job's first
    token, end and the mean cost are bit-identical, with fewer sessions than the engine's width, with more (the
    rotation), at every width and order, on random traffic with answers of 1 token and of many."""
    rng = random.Random(7)
    cases = 0
    for _ in range(400):
        n = rng.randint(1, 24)
        params = {"prefill_s": rng.uniform(0.01, 0.3), "prefill_s_per_ktoken": rng.uniform(0.0, 3.0),
                  "step_s": rng.uniform(0.005, 0.05), "step_s_per_session": rng.uniform(0.0, 0.02),
                  **({"group_s": rng.uniform(0, 0.05)} if rng.random() < 0.3 else {})}

        def jobs(n=n, seed=rng.randrange(1 << 30)):
            r, t, out = random.Random(seed), 0.0, []
            for k in range(n):
                t += r.expovariate(2.0) if r.random() < 0.7 else 0.0
                limit = r.choice((1, 2, 3, 16, 32, 256))
                out.append(Job(k, t, limit, r.random() < 0.3, r.choice((24, 1030)), r.randint(1, limit)))
            return out

        engine_width = rng.choice((1, 2, 3, 4, 8))
        capacity = rng.choice((1, 2, 4, 8, 16))
        wide = rng.choice((1, 2, 4, 8, 32))
        order = rng.random() < 0.5
        old, new = jobs(), jobs()
        a = _simulate_by_rounds(params, old, lambda w: order, lambda w: wide, capacity, engine_width)
        b = simulate(params, new, lambda w: order, lambda w: wide, capacity, engine_width)
        assert a == b and [(j.first, j.end) for j in old] == [(j.first, j.end) for j in new], (engine_width, capacity)
        cases += 1
    assert cases == 400
    payload = {"model": TRUTH, "arrivals": _arrivals(), "capacity": 8, "seed": 0, "reference": REFERENCE,
               "incumbent": None, "settings": {"episodes": 150, "seeds": 1, "validation": 10, "held_out": 20}}
    fast = plan(payload)
    original = ag.simulate
    try:
        ag.simulate = _simulate_by_rounds
        slow = plan(payload)
    finally:
        ag.simulate = original
    for key in ("candidate", "choice", "model_costs", "arm_costs", "curve"):
        assert fast[key] == slow[key], key


# ---- AG8: tokens per second first, the first token no worse ----

def _rate_triplets(n, b_spt=0.9, b_first=1.0, noise=0.01, seed=0):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        a, aa = 1.0 + rng.gauss(0, noise), 1.0 + rng.gauss(0, noise)
        fa, faa = 1.0 + rng.gauss(0, noise), 1.0 + rng.gauss(0, noise)
        out.append({"a": a, "aa": aa, "b": b_spt + rng.gauss(0, noise),
                    "fa": fa, "faa": faa, "fb": b_first + rng.gauss(0, noise)})
    return out


def test_ag8_qualification_needs_more_tokens_per_second_and_a_first_token_no_later():
    faster = judge(_rate_triplets(12), objective="throughput")
    assert faster["verdict"] == "pass" and faster["gain"] > 0.08 and faster["first_token"]["upper"] <= 0.02
    later = judge(_rate_triplets(12, b_first=1.1), objective="throughput")  # 10 % more tokens/s, 10 % later
    assert later["verdict"] == "fail" and later["first_token"]["lower"] > later["first_token"]["allowed"]
    slower = judge(_rate_triplets(12, b_spt=1.05), objective="throughput")
    assert slower["verdict"] == "fail"
    within = judge(_rate_triplets(12, b_first=1.005, noise=0.002), objective="throughput")
    assert within["verdict"] == "pass", within["first_token"]  # inside the 2 % tolerance
    latency = judge([{k: t[k] for k in ("a", "b", "aa")} for t in _rate_triplets(12, b_first=1.1)])
    assert latency["verdict"] == "pass" and "first_token" not in latency, "AG1's judge sees no first token"


def test_ag8_audit_withdraws_a_schedule_that_makes_the_first_token_later():
    assert ag.harm(_rate_triplets(12, b_spt=0.95, b_first=1.15), objective="throughput")["verdict"] == "harm"
    assert ag.harm(_rate_triplets(12, b_spt=1.1), objective="throughput")["verdict"] == "harm"
    assert ag.harm(_rate_triplets(12, b_spt=0.95), objective="throughput")["verdict"] == "open"


def _stacked_like(seed=0, n=160):
    """A queue that a wider group serves sooner: a further session costs a tenth of a step (a stacked engine)."""
    rng = random.Random(seed)
    out, t = [], 0.0
    while len(out) < n:
        for _ in range(6):
            out.append([t + rng.uniform(0, 0.3), 24, False, 64, rng.randint(32, 64)])
        t += rng.expovariate(0.4)
    return sorted(out)


def test_ag8_planner_picks_the_width_that_makes_more_tokens_per_second_within_the_first_token_bound():
    params = {"prefill_s": 0.02, "prefill_s_per_ktoken": 0.5, "step_s": 0.03, "step_s_per_session": 0.003}
    payload = {"model": params, "arrivals": _stacked_like(), "capacity": 8, "engine_width": 8, "seed": 0,
               "reference": {"width": 1, "sjf": False, "gather_ms": 0}, "incumbent": None,
               "settings": {"episodes": 200, "seeds": 1, "validation": 10, "held_out": 20}, "objective": "throughput"}
    out = plan(payload)
    assert out["objective"] == "throughput" and out["candidate"] is not None, out["model_costs"]
    assert out["model_costs"][out["choice"]] < out["model_costs"]["reference"], "fewer seconds per token"
    assert out["model_firsts"][out["choice"]] <= (1 + ag.TTFT_TOLERANCE) * out["model_firsts"]["reference"]
    policy = Policy.from_json(out["candidate"])
    assert (policy.best_width([View(64, False)] * 8) or 8) > 1, "a wider group than the reference's one"
    old = plan({k: v for k, v in payload.items() if k != "objective"})
    assert old["objective"] == "latency" and "model_firsts" not in old, "a payload from before AG8 keeps AG1's cost"


def test_ag8_the_objective_is_part_of_the_binding_and_reported(tmp_path):
    binding = {"machine": "test"}
    common = dict(capacity=8, reference=REFERENCE, planner_command=["true"])
    rate = Agent(tmp_path / "a", binding, **common)
    latency = Agent(tmp_path / "b", binding, objective="latency", **common)
    try:
        assert rate.objective == "throughput" and rate.status()["objective"] == "throughput"
        assert latency.binding == binding, "AG1's states resume under the binding they were saved with"
        assert rate.binding == {**binding, "objective": "throughput"} and rate.binding_sha256 != latency.binding_sha256
        with pytest.raises(ValueError):
            Agent(tmp_path / "c", binding, objective="speed", **common)
    finally:
        rate.close()
        latency.close()


def test_ag8_a_replayed_window_scores_its_seconds_per_token_and_first_token():
    rows = [{"ttft_s": 0.1, "e2e_s": 1.0, "tokens": 30}, {"ttft_s": None, "e2e_s": 0.3, "tokens": 10}]
    assert ag.window_rate({"rows": rows, "span_s": 2.0}) == {"spt": 2.0 / 40, "first": (0.1 + 0.3) / 2}


def test_ag8_a_candidate_that_ends_the_same_window_sooner_qualifies_and_one_that_only_reorders_does_not(tmp_path):
    """The replayed qualification under `throughput`: B makes the window's tokens 20 % sooner with the same first
    tokens and serves; a B that only lowers AG1's request cost, with the same span, never does."""
    def run(spans, firsts):
        agent = _agent(tmp_path / f"run-{spans[True]}", objective="throughput")
        _feed(agent, 32)
        agent.candidate = {"policy": ag.fixed_policy(2, True), "plan": {"choice": "best_fixed"}, "since": 31.5,
                           "triplets": [], "used": 0}
        _feed(agent, 16 * ag.MIN_TRIPLETS, start=100.0, gap=10.0)

        def replay(trace, choose, preempted):
            sjf = choose(len(trace), [], 0.0, 0)[2]
            rows = [{"ttft_s": firsts[sjf], "e2e_s": 2.0, "tokens": 4, "outcome": "ok"} for _ in trace]
            return {"rows": rows, "texts": ["same"] * len(trace), "span_s": spans[sjf]}
        for _ in range(ag.MIN_TRIPLETS):
            agent._qualify(replay, lambda: False)
        return agent

    sooner = run({True: 8.0, False: 10.0}, {True: 0.5, False: 0.5})
    assert sooner.active and sooner.evidence["gain"] == pytest.approx(0.2)
    assert sooner.evidence["objective"] == "throughput" and sooner.evidence["first_token"]["upper"] <= 0.02
    reordered = run({True: 10.0, False: 10.0}, {True: 0.3, False: 0.5})
    assert not reordered.active and reordered.candidate is None and reordered.counters["rejected"] == 1
    later = run({True: 8.0, False: 10.0}, {True: 0.6, False: 0.5})  # 20 % sooner, first tokens 20 % later
    assert not later.active and later.counters["rejected"] == 1
    for agent in (sooner, reordered, later):
        agent.close()
