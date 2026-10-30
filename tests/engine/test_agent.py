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


def _feed(agent, n=64, start=0.0):
    rng = random.Random(3)
    for k in range(n):
        group = _group(TRUTH, rng, 1, start=start + k)
        group[0]["arrive"] = start + k
        agent.observe_group(group)


def test_a_candidate_serves_only_after_it_wins_replayed_triplets_on_held_out_traffic(tmp_path):
    agent = _agent(tmp_path)
    _feed(agent, 32)
    agent.candidate = {"policy": ag.fixed_policy(2, True), "plan": {"choice": "best_fixed"}, "since": 31.5,
                       "triplets": [], "used": 0}
    assert agent._trace(agent.traffic, since=31.5) is None  # nothing recorded after the plan yet
    _feed(agent, 16 * ag.MIN_TRIPLETS, start=100.0)
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
    assert set(agent.candidate["partial"]) == {"a"} and agent.candidate["triplets"] == []
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
    assert {offset for offset, _, _ in trace} == {0.0} and decision == (2, 0, False)
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
