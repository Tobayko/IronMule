"""The routing contract: what may be chosen, on what facts, and what may never be.

No model is loaded here. The router is a pure function of dispatch-time facts, and the
point of these tests is that it stays one.
"""

from __future__ import annotations

import pytest

from ironmule.plans import ReusableSessionPlan, StrictOneShotPlan
from ironmule.router import (ROUTER_VERSION, ROUTES, SCHEDULING_RULE, AppleRuntime,
                             ExecutionRouter, RouteDecision)
from ironmule.service import AutomaticMode, InteractiveMode, Request
from ironmule.telemetry import RequestMetrics, Telemetry

QUALIFIED = dict(identity_sha256="a" * 64, fingerprint="dc652d66f24ac207",
                 mlx="0.32.0", mlx_lm="0.31.3")


def request(prompt_tokens=8, max_tokens=16, plan=None, arrival_ms=0.0):
    return Request(prompt_ids=list(range(prompt_tokens)), max_tokens=max_tokens,
                   plan=plan or StrictOneShotPlan(), arrival_ms=arrival_ms)


def known(profile=None):
    return ExecutionRouter(profile or {"knobs": {}}, **QUALIFIED)


class TestFailsClosed:
    def test_no_profile_is_the_reference_path(self):
        decision = ExecutionRouter(None).decide([request()])
        assert decision.route == "reference"
        assert not decision.qualified_profile

    def test_concurrency_does_not_unlock_an_unknown_fingerprint(self):
        decision = ExecutionRouter(None).decide([request(), request(), request()])
        assert decision.route == "reference", "grouping is qualified evidence, not a default"

    @pytest.mark.parametrize("missing", ["identity_sha256", "fingerprint"])
    def test_a_half_known_identity_is_unknown(self, missing):
        facts = dict(QUALIFIED, **{missing: None})
        assert not ExecutionRouter({"knobs": {}}, **facts).qualified
        assert ExecutionRouter({"knobs": {}}, **facts).decide([request()]).route == "reference"

    def test_a_model_name_alone_qualifies_nothing(self):
        router = ExecutionRouter({"model_id": "mlx-community/gemma-3-12b-it-qat-4bit"})
        assert not router.qualified


class TestSingleRequestNeverWaits:
    def test_one_request_takes_the_sequential_path(self):
        decision = known().decide([request()])
        assert decision.route == "interactive"
        assert decision.requests == 1

    def test_the_reason_names_the_rule(self):
        assert "waiting" in known().decide([request()]).reason

    def test_an_empty_dispatch_is_not_an_error(self):
        decision = known().decide([])
        assert decision.route == "interactive"
        assert decision.max_prompt_tokens == 0 and decision.max_new_tokens == 0


class TestGroupedDispatch:
    def test_two_requests_go_to_the_grouped_path(self):
        decision = known().decide([request(), request()])
        assert decision.route == "throughput" and decision.requests == 2

    def test_the_paired_split_is_left_to_the_profile_record(self):
        # The router must not re-derive the paired admission; `AutomaticMode` owns it.
        runtime = AppleRuntime.__new__(AppleRuntime)
        runtime.router = known()
        mode = runtime._mode_for(known().decide([request(), request()]))
        assert isinstance(mode, AutomaticMode) and mode.opt_in

    def test_a_single_request_never_reaches_the_automatic_mode(self):
        runtime = AppleRuntime.__new__(AppleRuntime)
        runtime.router = known()
        assert isinstance(runtime._mode_for(known().decide([request()])), InteractiveMode)


class TestObjective:
    """The latency/throughput trade is a caller's to make, and it is recorded."""

    def test_throughput_is_the_default(self):
        assert known().objective == "throughput"
        assert known().decide([request()]).objective == "throughput"

    def test_latency_keeps_a_group_on_the_sequential_path(self):
        router = ExecutionRouter({"knobs": {}}, objective="latency", **QUALIFIED)
        decision = router.decide([request(), request(), request(), request()])
        assert decision.route == "interactive"
        assert "latency" in decision.reason

    def test_the_reason_names_what_grouping_would_have_cost(self):
        router = ExecutionRouter({"knobs": {}}, objective="latency", **QUALIFIED)
        assert "B55" in router.decide([request(), request()]).as_dict()["evidence_run_ids"]

    def test_an_unknown_objective_is_refused_at_construction(self):
        with pytest.raises(ValueError, match="objective"):
            ExecutionRouter(None, objective="fastest")

    def test_the_objective_does_not_unlock_an_unknown_fingerprint(self):
        router = ExecutionRouter(None, objective="latency")
        assert router.decide([request(), request()]).route == "reference"


class TestPlansAreReportedNeverChosen:
    def test_a_session_plan_is_counted_when_it_matches(self):
        plan = ReusableSessionPlan([0, 1, 2], name="doc")
        decision = known().decide([request(plan=plan), request(plan=plan)])
        assert decision.session_plan_matches == 2
        assert decision.plan_kinds == ("reusable_session", "reusable_session")

    def test_a_session_plan_that_does_not_match_is_not_counted(self):
        plan = ReusableSessionPlan([99, 98], name="other")
        decision = known().decide([request(plan=plan)])
        assert decision.session_plan_matches == 0

    def test_the_route_does_not_change_because_a_prefix_matches(self):
        plan = ReusableSessionPlan([0, 1, 2], name="doc")
        with_plan = known().decide([request(plan=plan), request(plan=plan)])
        without = known().decide([request(), request()])
        assert with_plan.route == without.route

    def test_the_router_never_builds_a_plan(self):
        decision = known().decide([request()])
        assert decision.plan_kinds == ("strict_one_shot",)


class TestDecisionRecord:
    def test_it_names_its_version_and_evidence(self):
        record = known().decide([request()]).as_dict()
        assert record["router_version"] == ROUTER_VERSION
        assert record["evidence_run_ids"] == ["E15", "E16", "B55"]

    def test_the_reference_route_claims_no_evidence(self):
        assert ExecutionRouter(None).decide([request()]).as_dict()["evidence_run_ids"] == []

    def test_every_route_it_can_return_is_declared(self):
        router = known()
        routes = {router.decide(batch).route
                  for batch in ([], [request()], [request(), request()])}
        routes.add(ExecutionRouter(None).decide([request()]).route)
        assert routes <= set(ROUTES)

    def test_dispatch_facts_are_the_dispatch_facts(self):
        decision = known().decide([request(4, 10), request(31, 64)])
        assert decision.max_prompt_tokens == 31 and decision.max_new_tokens == 64

    def test_the_record_is_frozen(self):
        with pytest.raises(Exception):
            known().decide([request()]).route = "throughput"


class TestTelemetryCarriesTheRoute:
    def test_an_unrouted_dispatch_reports_no_route(self):
        assert Telemetry(mode="interactive").snapshot()["routing"] == {}

    def test_a_routed_dispatch_reports_the_whole_decision(self):
        telemetry = Telemetry(mode="interactive")
        telemetry.routing = known().decide([request()]).as_dict()
        snapshot = telemetry.snapshot()
        assert snapshot["routing"]["route"] == "interactive"
        assert snapshot["routing"]["reason"]

    def test_decode_time_and_rate_are_per_request(self):
        metrics = RequestMetrics(rid="r0", arrival_ns=0, engine_start_ns=1_000_000,
                                 first_token_ns=3_000_000, finished_ns=9_000_000,
                                 generated_tokens=3)
        record = metrics.as_dict()
        assert record["decode_ms"] == 6.0
        assert record["decode_tokens_per_second"] == pytest.approx(2 / 0.006)

    def test_a_request_that_produced_one_token_has_no_rate(self):
        metrics = RequestMetrics(rid="r0", arrival_ns=0, first_token_ns=1,
                                 finished_ns=2, generated_tokens=1)
        assert metrics.as_dict()["decode_tokens_per_second"] is None


class FakeRuntime:
    """Enough `Runtime` to test cohort dispatch without loading a model."""

    def __init__(self):
        self.mode = None
        self.telemetry = Telemetry()
        self.calls: list[tuple[str, tuple[str, ...], int | None]] = []
        self.engine = type("E", (), {"knobs": ironmule_knobs()})()

    def serve(self, requests, dispatch_ns=None):
        from ironmule.service import Result
        name = getattr(self.mode, "name", "unknown")
        self.calls.append((name, tuple(r.rid for r in requests), dispatch_ns))
        self.telemetry = Telemetry(mode=name)
        for request in requests:
            metrics = RequestMetrics(rid=request.rid, arrival_ns=dispatch_ns or 0,
                                     engine_start_ns=1, first_token_ns=2, finished_ns=3,
                                     prompt_tokens=len(request.prompt_ids),
                                     generated_tokens=2, stop_reason="length")
            self.telemetry.requests.append(metrics)
            self.telemetry.plan_kinds.append("strict_one_shot")
            self.telemetry.realised_widths.append(len(requests))
        self.telemetry.wall_ns = 1_000_000
        return [Result(rid=r.rid, tokens=[1, 2], text="", stop_reason="length",
                       metrics={}) for r in requests]

    def encode(self, text):
        return [1, 2, 3]


def ironmule_knobs():
    from ironmule.runtime import BASELINE
    return BASELINE


def routed(objective="throughput", profile=None):
    runtime = FakeRuntime()
    router = ExecutionRouter(profile or {"knobs": {}}, objective=objective, **QUALIFIED)
    return AppleRuntime(runtime, router), runtime


def tagged(objective=None, arrival=0.0, rid=""):
    return Request(prompt_ids=[1, 2, 3], max_tokens=8, plan=StrictOneShotPlan(),
                   objective=objective, arrival_ms=arrival, rid=rid)


class TestRequestObjective:
    """The field itself: it exists, it validates, and it defaults to saying nothing."""

    def test_an_unspecified_request_says_nothing(self):
        assert tagged().objective is None

    @pytest.mark.parametrize("objective", ["latency", "throughput"])
    def test_both_objectives_are_accepted(self, objective):
        assert tagged(objective).objective == objective

    @pytest.mark.parametrize("bad", ["balanced", "fast", "", "LATENCY"])
    def test_anything_else_is_refused_at_construction(self, bad):
        with pytest.raises(ValueError, match="objective"):
            Request(prompt_ids=[1], objective=bad)

    def test_precedence_is_request_then_dispatch_then_runtime(self):
        router = known()
        assert router.objective_for(tagged("latency"), "throughput") == "latency"
        assert router.objective_for(tagged(), "latency") == "latency"
        assert router.objective_for(tagged(), None) == "throughput"


class TestCohorts:
    def test_one_objective_is_one_cohort(self):
        plan = known().plan([tagged("throughput"), tagged("throughput")])
        assert len(plan) == 1 and plan[0].route == "throughput"

    def test_unspecified_requests_keep_the_pre_b56_behaviour(self):
        plan = known().plan([tagged(), tagged()])
        assert len(plan) == 1
        assert plan[0].objective == "throughput" and plan[0].route == "throughput"

    def test_latency_leads_a_tie(self):
        plan = known().plan([tagged("throughput"), tagged("latency")])
        assert [d.objective for d in plan] == ["latency", "throughput"]

    def test_a_cohort_that_has_not_arrived_does_not_lead(self):
        plan = known().plan([tagged("throughput", 0.0), tagged("latency", 50.0)])
        assert [d.objective for d in plan] == ["throughput", "latency"], \
            "leading with a request nobody has sent yet would be priority inversion"

    def test_a_latency_request_is_never_in_a_throughput_cohort(self):
        plan = known().plan([tagged("throughput"), tagged("latency"),
                             tagged("throughput"), tagged("latency")])
        latency = [d for d in plan if d.objective == "latency"][0]
        throughput = [d for d in plan if d.objective == "throughput"][0]
        assert latency.route == "interactive" and latency.requests == 2
        assert throughput.route == "throughput" and throughput.requests == 2
        assert not set(latency.request_ids) & set(throughput.request_ids)

    def test_every_request_lands_in_exactly_one_cohort(self):
        requests = [tagged("latency"), tagged("throughput"), tagged(),
                    tagged("latency", 10.0)]
        plan = known().plan(requests)
        placed = [rid for d in plan for rid in d.request_ids]
        assert sorted(placed) == sorted(r.rid for r in requests)
        assert len(placed) == len(set(placed)), "no request may be served twice"

    def test_the_plan_is_deterministic(self):
        requests = [tagged("throughput"), tagged("latency"), tagged("throughput")]
        first = known().plan(requests)
        for _ in range(5):
            assert known().plan(requests) == first

    def test_caller_order_survives_inside_a_cohort(self):
        a, b = tagged("throughput", rid="aaa"), tagged("throughput", rid="bbb")
        plan = known().plan([b, a])
        assert plan[0].request_ids == ("bbb", "aaa")

    def test_an_unqualified_fingerprint_routes_no_cohort(self):
        plan = ExecutionRouter(None).plan([tagged("latency"), tagged("throughput")])
        assert {d.route for d in plan} == {"reference"}

    def test_an_unknown_dispatch_objective_is_refused(self):
        with pytest.raises(ValueError, match="objective"):
            known().plan([tagged()], "balanced")

    def test_an_empty_dispatch_still_yields_one_decision(self):
        assert len(known().plan([])) == 1


class TestCohortDispatch:
    """`AppleRuntime.serve` over the fake runtime: order, modes, telemetry, records."""

    def test_a_single_objective_is_one_serve_call(self):
        runtime, fake = routed()
        runtime.serve([tagged("throughput"), tagged("throughput")])
        assert len(fake.calls) == 1
        assert fake.calls[0][0] == "automatic"

    def test_mixed_objectives_are_two_calls_in_cohort_order(self):
        runtime, fake = routed()
        runtime.serve([tagged("throughput"), tagged("latency"), tagged("throughput")])
        assert [call[0] for call in fake.calls] == ["interactive", "automatic"]

    def test_both_cohorts_share_one_dispatch_timestamp(self):
        runtime, fake = routed()
        runtime.serve([tagged("latency"), tagged("throughput"), tagged("throughput")])
        stamps = {call[2] for call in fake.calls}
        assert len(stamps) == 1 and None not in stamps, \
            "a second serve stamped at its own start would hide the wait before it"

    def test_results_come_back_in_the_caller_order(self):
        runtime, _ = routed()
        requests = [tagged("throughput", rid="r1"), tagged("latency", rid="r2"),
                    tagged("throughput", rid="r3")]
        assert [r.rid for r in runtime.serve(requests)] == ["r1", "r2", "r3"]

    def test_the_record_names_the_route_of_every_request(self):
        runtime, _ = routed()
        requests = [tagged("latency", rid="a"), tagged("throughput", rid="b"),
                    tagged("throughput", rid="c")]
        runtime.serve(requests)
        decision = runtime.last_decision
        assert decision["route_by_request"] == {"a": "interactive", "b": "throughput",
                                                "c": "throughput"}
        assert decision["objective_by_request"] == {"a": "latency", "b": "throughput",
                                                    "c": "throughput"}
        assert decision["mixed_objectives"] is True
        assert decision["cohort_order"] == ["latency", "throughput"]
        assert decision["scheduling_rule"] == SCHEDULING_RULE

    def test_a_dispatch_says_whether_it_could_protect_its_latency_requests(self):
        runtime, _ = routed()
        runtime.serve([tagged("latency"), tagged("throughput"), tagged("throughput")])
        assert runtime.last_decision["latency_protected"] is True

    def test_a_late_latency_request_is_reported_as_unprotected(self):
        runtime, _ = routed()
        runtime.serve([tagged("throughput", 0.0), tagged("throughput", 0.0),
                       tagged("latency", 60.0)])
        decision = runtime.last_decision
        assert decision["cohort_order"] == ["throughput", "latency"]
        assert decision["latency_protected"] is False, \
            "B56 measured this case at 2.10x; a caller must be able to see it"

    def test_a_dispatch_without_throughput_is_always_protected(self):
        runtime, _ = routed()
        runtime.serve([tagged("latency", 60.0), tagged("latency", 0.0)])
        assert runtime.last_decision["latency_protected"] is True

    def test_a_single_objective_dispatch_is_not_marked_mixed(self):
        runtime, _ = routed()
        runtime.serve([tagged("throughput"), tagged("throughput")])
        assert runtime.last_decision["mixed_objectives"] is False

    def test_telemetry_covers_every_request_of_every_cohort(self):
        runtime, _ = routed()
        runtime.serve([tagged("latency"), tagged("throughput"), tagged("throughput")])
        snapshot = runtime.telemetry.snapshot()
        assert snapshot["requests"] == 3
        assert snapshot["rounds"] == 3

    def test_merged_wall_time_adds_the_cohorts_rather_than_hiding_one(self):
        runtime, _ = routed()
        runtime.serve([tagged("latency"), tagged("throughput"), tagged("throughput")])
        # Two cohorts of 1 ms each ran one after the other, in one process.
        assert runtime.telemetry.wall_ns == 2_000_000

    def test_the_dispatch_objective_fills_in_only_where_nothing_was_said(self):
        runtime, fake = routed()
        runtime.serve([tagged(), tagged("throughput")], objective="latency")
        assert [call[0] for call in fake.calls] == ["interactive", "interactive"]
        assert runtime.last_decision["objective_by_request"] == {
            fake.calls[0][1][0]: "latency", fake.calls[1][1][0]: "throughput"}

    def test_two_requests_may_not_share_one_id(self):
        runtime, _ = routed()
        with pytest.raises(ValueError, match="same rid"):
            runtime.serve([tagged("latency", rid="dup"), tagged("throughput", rid="dup")])

    def test_generate_can_name_an_objective(self):
        runtime, fake = routed()
        runtime.generate("hello", objective="latency")
        assert fake.calls[0][0] == "interactive"


class TestDispatchArrivalIsTheCallersArrival:
    """A dispatch split into several `serve` calls must not lose the wait between them."""

    class Backend:
        eos_ids = ()

        def capacity_for(self, prompt_lens, max_tokens):
            return 64

        def prefill(self, prompt_ids, plan, capacity):
            return {"layers": []}, 7

        def reset_state(self, base_state, offset):
            return dict(base_state)

    def test_it_defaults_to_now(self):
        from ironmule.executor import build_sessions
        telemetry = Telemetry()
        sessions = build_sessions([tagged()], self.Backend(), telemetry, 64)
        assert sessions[0].metrics.arrival_ns > 0

    def test_an_explicit_stamp_is_used_verbatim(self):
        from ironmule.executor import build_sessions
        telemetry = Telemetry()
        build_sessions([tagged(), tagged()], self.Backend(), telemetry, 64,
                       dispatch_ns=1234)
        assert [m.arrival_ns for m in telemetry.requests] == [1234, 1234]

    def test_the_queue_wait_of_a_trailing_cohort_stays_visible(self):
        from ironmule.executor import build_sessions
        telemetry = Telemetry()
        build_sessions([tagged()], self.Backend(), telemetry, 64, dispatch_ns=0)
        metrics = telemetry.requests[0]
        assert metrics.queue_wait_ms > 0, \
            "engine start after a foreign cohort must show up as queue wait"


class TestSwapProbe:
    """The probe must be cheap enough to sit in a dispatch it also reports on."""

    def test_it_returns_bytes_or_nothing(self):
        from ironmule.hw import swap_used_bytes
        value = swap_used_bytes()
        assert value is None or (isinstance(value, int) and value >= 0)

    def test_it_agrees_with_sysctl(self):
        import subprocess
        from ironmule.hw import swap_used_bytes
        units = {"B": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3}
        try:
            fields = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True,
                                    text=True, timeout=5).stdout.split()
            amount = fields[fields.index("used") + 2]
            expected = float(amount[:-1]) * units[amount[-1].upper()]
        except (OSError, subprocess.SubprocessError, ValueError, IndexError, KeyError):
            pytest.skip("this machine does not report vm.swapusage")
        probed = swap_used_bytes()
        assert probed is not None
        # `sysctl(8)` rounds to two decimals of the unit it prints; swap also moves.
        assert probed == pytest.approx(expected, rel=0.02, abs=16 * 1024 ** 2)

    def test_it_costs_far_less_than_a_subprocess(self):
        import time
        from ironmule.hw import swap_used_bytes
        swap_used_bytes()
        started = time.perf_counter_ns()
        for _ in range(50):
            swap_used_bytes()
        each_ms = (time.perf_counter_ns() - started) / 50 / 1e6
        # B55 measured the subprocess it replaced at 8.7 ms per call under load.
        assert each_ms < 1.0, f"{each_ms:.3f} ms per probe belongs outside the dispatch"


def test_module_self_check_runs():
    from ironmule.router import _self_check
    _self_check()
