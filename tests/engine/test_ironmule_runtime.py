"""Runtime tests. No model needed: the scheduler talks to a scripted fake backend.

The matrix the runtime contract requires — token identity, counts, stop reasons,
ragged lengths, early finishers, reversed order, heterogeneous prompts, staggered
arrival, widths 1 to 4, sequential fallback, and absence of state aliasing — is
covered here deterministically. The real-model counterpart lives in
`tests/test_forge_runtime_integration.py` and is skipped unless the model is present.
"""

import json

import pytest

from ironmule.executor import AsyncGroupedB1Executor, SequentialExecutor, build_sessions
from ironmule.plans import ReusableSessionPlan, StrictOneShotPlan
from ironmule.service import InteractiveMode, Request, Runtime
from ironmule.telemetry import Telemetry

EOS = 99


class FakeBackend:
    """Deterministic scripted decoder. Every session gets its own state object."""

    eos_ids = (EOS,)

    def __init__(self, scripts, fail_on_group_call=None, prefill_tokens=None):
        self.scripts = scripts              # rid -> list of tokens after the first
        self.fail_on_group_call = fail_on_group_call
        self.prefill_tokens = prefill_tokens or {}
        self.group_calls = 0
        self.completed_widths = []
        self.states_seen = []

    def capacity_for(self, prompt_lens, max_tokens):
        return ((max(prompt_lens) + max_tokens + 8 + 63) // 64) * 64

    def prefill(self, prompt_ids, plan, capacity):
        rid = self._rid_for(prompt_ids)
        state = {"rid": rid, "step": 0, "touched_by": [rid]}
        self.states_seen.append(id(state))
        return state, self.prefill_tokens.get(rid, 0)

    def _rid_for(self, prompt_ids):
        return f"p{len(prompt_ids)}_{prompt_ids[0]}"

    def reset_state(self, base_state, offset):
        state = {"rid": base_state["rid"], "step": 0, "touched_by": list(base_state["touched_by"])}
        self.states_seen.append(id(state))
        return state

    def step(self, state, token, capacity):
        return {"state": state, "token": token}

    def complete(self, handles):
        self.group_calls += 1
        self.completed_widths.append(len(handles))
        if self.fail_on_group_call is not None and self.group_calls == self.fail_on_group_call:
            raise RuntimeError("simulated device failure")

    def read(self, handle):
        state = handle["state"]
        script = self.scripts[state["rid"]]
        index = state["step"]
        token = script[index] if index < len(script) else EOS
        new_state = {"rid": state["rid"], "step": index + 1,
                     "touched_by": state["touched_by"] + [state["rid"]]}
        self.states_seen.append(id(new_state))
        return token, new_state


def make(scripts, prompts, max_tokens=None, arrivals=None, **kwargs):
    backend = FakeBackend(scripts, **kwargs)
    telemetry = Telemetry()
    requests = []
    for index, ids in enumerate(prompts):
        requests.append(Request(prompt_ids=ids, rid=backend._rid_for(ids),
                                max_tokens=(max_tokens or [8] * len(prompts))[index],
                                plan=StrictOneShotPlan(),
                                arrival_ms=(arrivals or [0.0] * len(prompts))[index]))
    capacity = backend.capacity_for([len(r.prompt_ids) for r in requests],
                                    max(r.max_tokens for r in requests))
    sessions = build_sessions(requests, backend, telemetry, capacity)
    return backend, telemetry, sessions, capacity


PROMPTS = [[1] * 10, [2] * 12, [3] * 8, [4] * 20, [5] * 15, [6] * 9]
SCRIPTS = {
    "p10_1": [11, 12, 13, EOS],
    "p12_2": [21, EOS],
    "p8_3": [31, 32, 33, 34, 35, 36, 37, 38],
    "p20_4": [41, 42, EOS],
    "p15_5": [51, 52, 53, 54, 55, EOS],
    "p9_6": [61],
}


def run(executor_cls, prompts=PROMPTS, width=None, **kwargs):
    backend, telemetry, sessions, capacity = make(SCRIPTS, prompts, **kwargs)
    executor = (executor_cls(backend, telemetry, max_width=width)
                if width else executor_cls(backend, telemetry))
    executor.run(sessions, capacity)
    return backend, telemetry, sessions


def outputs(sessions):
    return {s.rid: (tuple(s.tokens), s.stop_reason) for s in sessions}


def test_sequential_is_the_reference():
    _, _, sessions = run(SequentialExecutor)
    assert outputs(sessions)["p10_1"] == ((0, 11, 12, 13, EOS), "eos")
    assert outputs(sessions)["p9_6"] == ((0, 61, EOS), "eos"), "early finisher stops at eos"
    assert all(s.done for s in sessions)


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_grouped_matches_sequential_at_every_width(width):
    _, _, reference = run(SequentialExecutor)
    _, telemetry, sessions = run(AsyncGroupedB1Executor, width=width)
    assert outputs(sessions) == outputs(reference), "token ids, counts and stop reasons"
    assert max(telemetry.realised_widths) <= width
    assert telemetry.fallbacks == 0


def test_realised_width_may_fall_below_the_maximum():
    _, telemetry, _ = run(AsyncGroupedB1Executor, width=4)
    assert min(telemetry.realised_widths) < 4, "groups thin out as requests finish"


def test_width_above_the_shipped_maximum_is_refused():
    backend, telemetry, _, _ = make(SCRIPTS, PROMPTS)
    with pytest.raises(ValueError):
        AsyncGroupedB1Executor(backend, telemetry, max_width=8)


def test_reversed_arrival_order_changes_nothing():
    _, _, reference = run(SequentialExecutor)
    _, _, sessions = run(AsyncGroupedB1Executor, prompts=list(reversed(PROMPTS)), width=4)
    assert outputs(sessions) == outputs(reference)


def test_staggered_arrival_changes_nothing():
    _, _, reference = run(SequentialExecutor)
    _, telemetry, sessions = run(AsyncGroupedB1Executor, width=4,
                                 arrivals=[0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    assert outputs(sessions) == outputs(reference)
    assert all(m.queue_wait_ms is not None for m in telemetry.requests)


def test_heterogeneous_lengths_and_caps():
    caps = [2, 8, 3, 8, 4, 8]
    _, _, reference = run(SequentialExecutor, max_tokens=caps)
    _, _, sessions = run(AsyncGroupedB1Executor, width=4, max_tokens=caps)
    assert outputs(sessions) == outputs(reference)
    got = outputs(sessions)
    assert got["p10_1"][1] == "length" and got["p10_1"][0] == (0, 11), "cap honoured"
    assert got["p12_2"][1] == "eos", "eos beats the cap when it comes first"


class CapacitySensitiveBackend(FakeBackend):
    """As CUDA measured in X1: a decoded token depends on the KV capacity it is decoded at."""

    def __init__(self, scripts, **kwargs):
        super().__init__(scripts, **kwargs)
        self.capacities = []

    def prefill(self, prompt_ids, plan, capacity):
        self.capacities.append((self._rid_for(prompt_ids), capacity))
        return super().prefill(prompt_ids, plan, capacity)

    def step(self, state, token, capacity):
        self.capacities.append((state["rid"], capacity))
        return {"state": state, "token": token, "capacity": capacity}

    def read(self, handle):
        token, state = super().read(handle)
        return (token if token == EOS else token + handle["capacity"]), state


def test_a_request_decodes_at_its_own_capacity_whatever_it_is_grouped_with():
    long_prompt = [7] * 900
    scripts = {**SCRIPTS, "p900_7": [71, 72, EOS]}
    alone = {}
    for ids in PROMPTS:
        backend, telemetry = CapacitySensitiveBackend(scripts), Telemetry()
        capacity = backend.capacity_for([len(ids)], 8)
        sessions = build_sessions([Request(prompt_ids=ids, rid=backend._rid_for(ids), max_tokens=8)],
                                  backend, telemetry, capacity)
        SequentialExecutor(backend, telemetry).run(sessions, capacity)
        alone.update(outputs(sessions))
    backend = CapacitySensitiveBackend(scripts)
    telemetry = Telemetry()
    requests = [Request(prompt_ids=ids, rid=backend._rid_for(ids), max_tokens=8) for ids in [long_prompt, *PROMPTS]]
    capacity = backend.capacity_for([len(r.prompt_ids) for r in requests], 8)
    sessions = build_sessions(requests, backend, telemetry, capacity)
    AsyncGroupedB1Executor(backend, telemetry, max_width=4).run(sessions, capacity)
    grouped = outputs(sessions)
    assert {rid: grouped[rid] for rid in alone} == alone, "a long prompt in the group moved no other answer"
    for rid, used in backend.capacities:
        prompt = len(long_prompt) if rid == "p900_7" else len(next(p for p in PROMPTS if backend._rid_for(p) == rid))
        assert used == backend.capacity_for([prompt], 8) <= capacity, rid


@pytest.mark.parametrize("executor_cls", [SequentialExecutor, AsyncGroupedB1Executor])
def test_prefill_token_eos_stops_and_is_counted(executor_cls):
    _, telemetry, sessions = run(
        executor_cls,
        prompts=PROMPTS[:1],
        prefill_tokens={"p10_1": EOS},
        width=1 if executor_cls is AsyncGroupedB1Executor else None,
    )
    session = sessions[0]
    metrics = telemetry.requests[0]
    assert session.tokens == [EOS]
    assert session.stop_reason == "eos"
    assert metrics.first_token_ns > metrics.arrival_ns
    assert metrics.generated_tokens == 1
    assert metrics.visible_generated_tokens == 0
    assert len(metrics.token_times_ns) == 1
    assert metrics.finished_ns == metrics.first_token_ns
    assert metrics.engine_start_ns <= metrics.first_token_ns
    assert telemetry.realised_widths == []
    assert telemetry.snapshot()["generated_tokens"] == 1
    assert telemetry.snapshot()["physical_generated_tokens"] == 1
    assert telemetry.snapshot()["visible_generated_tokens"] == 0
    assert metrics.as_dict()["physical_generated_tokens"] == 1
    assert metrics.as_dict()["visible_generated_tokens"] == 0


def test_grouped_all_prefill_eos_has_no_decode_round_or_width():
    prompts = PROMPTS[:3]
    prefill_tokens = {f"p{len(ids)}_{ids[0]}": EOS for ids in prompts}
    backend, telemetry, sessions, capacity = make(
        SCRIPTS, prompts, prefill_tokens=prefill_tokens)

    AsyncGroupedB1Executor(backend, telemetry, max_width=4).run(sessions, capacity)

    assert backend.completed_widths == []
    assert telemetry.realised_widths == []
    assert all(session.done and session.stop_reason == "eos" for session in sessions)
    assert all(session.tokens == [EOS] for session in sessions)
    assert all(metrics.generated_tokens == 1 and metrics.visible_generated_tokens == 0
               for metrics in telemetry.requests)
    assert all(metrics.finished_ns == metrics.first_token_ns
               for metrics in telemetry.requests)


@pytest.mark.parametrize("executor_cls", [SequentialExecutor, AsyncGroupedB1Executor])
def test_max_tokens_one_stops_after_prefill_token(executor_cls):
    backend, telemetry, sessions = run(
        executor_cls,
        prompts=PROMPTS[:1],
        max_tokens=[1],
        width=1 if executor_cls is AsyncGroupedB1Executor else None,
    )
    session = sessions[0]
    metrics = telemetry.requests[0]
    assert session.tokens == [0]
    assert session.stop_reason == "length"
    assert metrics.generated_tokens == 1
    assert metrics.visible_generated_tokens == 1
    assert metrics.first_token_ns > metrics.arrival_ns
    assert metrics.engine_start_ns <= metrics.first_token_ns
    assert backend.completed_widths == []
    assert telemetry.realised_widths == []


def test_grouped_prefill_and_first_token_timestamps_are_ordered():
    _, telemetry, sessions = run(AsyncGroupedB1Executor, width=4)
    assert all(m.engine_start_ns <= m.first_token_ns for m in telemetry.requests)
    assert all(m.engine_ttft_ms is not None and m.engine_ttft_ms >= 0
               for m in telemetry.requests)
    assert all(m.generated_tokens > 1 for m in telemetry.requests)


def test_telemetry_does_not_present_zero_as_a_correctness_check():
    telemetry = Telemetry()
    snapshot = telemetry.snapshot()
    assert snapshot["correctness_errors"] == 0  # compatibility field
    assert snapshot["correctness_check_performed"] is False
    assert snapshot["correctness_checked_requests"] == 0

    checked = Telemetry(correctness_errors=1, correctness_check_performed=True,
                        correctness_checked_requests=2)
    checked_snapshot = checked.snapshot()
    assert checked_snapshot["correctness_check_performed"] is True
    assert checked_snapshot["correctness_checked_requests"] == 2
    assert checked_snapshot["correctness_errors"] == 1


def test_runtime_serve_prefill_eos_result_contract():
    class FakeTokenizer:
        @staticmethod
        def decode(tokens):
            return " ".join(map(str, tokens))

    prompt = PROMPTS[0]
    backend = FakeBackend(SCRIPTS, prefill_tokens={"p10_1": EOS})
    runtime = Runtime.__new__(Runtime)
    runtime.backend = backend
    runtime.mode = InteractiveMode()
    runtime.tokenizer = FakeTokenizer()
    request = Request(prompt_ids=prompt, max_tokens=8, rid="p10_1",
                      plan=StrictOneShotPlan())

    result = runtime.serve([request])[0]

    assert result.text == ""
    assert result.tokens == [EOS]
    assert result.stop_reason == "eos"
    assert result.metrics["generated_tokens"] == 1
    assert result.metrics["physical_generated_tokens"] == 1
    assert result.metrics["visible_generated_tokens"] == 0
    assert backend.completed_widths == []


def test_no_state_aliasing_between_requests():
    _, _, sessions = run(AsyncGroupedB1Executor, width=4)
    for session in sessions:
        assert session.state["rid"] == session.rid, "a session must end on its own state"
        assert set(session.state["touched_by"]) == {session.rid}, "no foreign writer"
    identities = [id(s.state) for s in sessions]
    assert len(set(identities)) == len(identities), "no two sessions share a state object"


def test_fallback_on_device_failure_still_matches_sequential():
    _, _, reference = run(SequentialExecutor)
    backend, telemetry, sessions = run(AsyncGroupedB1Executor, width=4,
                                       fail_on_group_call=2)
    assert telemetry.fallbacks == 1
    assert "simulated device failure" in telemetry.fallback_reasons[0]
    assert outputs(sessions) == outputs(reference), "fallback must not change output"
    assert any(m.fell_back for m in telemetry.requests)


def test_fallback_discards_the_failed_group_rather_than_trusting_it():
    backend, telemetry, sessions = run(AsyncGroupedB1Executor, width=4,
                                       fail_on_group_call=3)
    for session in sessions:
        assert session.tokens[0] == 0, "every session still starts from its prefill token"
        assert session.done


def test_telemetry_separates_service_and_engine_ttft():
    _, telemetry, _ = run(AsyncGroupedB1Executor, width=4,
                          arrivals=[0.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    snap = telemetry.snapshot()
    assert snap["service_ttft_p50_ms"] is not None
    assert snap["engine_ttft_p50_ms"] is not None
    assert snap["rounds"] == len(telemetry.realised_widths)
    assert snap["plan_switch_attempts"] == 0
    assert all(m["service_ttft_ms"] >= m["engine_ttft_ms"] for m in snap["per_request"])


def test_plans_are_never_substituted():
    backend, telemetry, sessions, capacity = make(SCRIPTS, PROMPTS)
    plan = ReusableSessionPlan([1, 2, 3])
    for session in sessions:
        session.plan = plan
    AsyncGroupedB1Executor(backend, telemetry, max_width=4).run(sessions, capacity)
    assert all(s.plan is plan for s in sessions), "the executor may not rewrite a plan"
    assert telemetry.plan_switch_attempts == 0


def _streamed_run(executor_cls, width=None, **kwargs):
    """Every request streams; returns what each emitted, in order, plus the sessions."""
    from ironmule.executor import TokenStream

    backend = FakeBackend(SCRIPTS, **kwargs)
    telemetry, emitted = Telemetry(), {}
    requests = []
    for ids in PROMPTS:
        rid = backend._rid_for(ids)
        emitted[rid] = []
        requests.append(Request(prompt_ids=ids, rid=rid, max_tokens=8, plan=StrictOneShotPlan(),
                                stream=TokenStream(emitted[rid].append)))
    capacity = backend.capacity_for([len(r.prompt_ids) for r in requests], 8)
    sessions = build_sessions(requests, backend, telemetry, capacity)
    (executor_cls(backend, telemetry, max_width=width) if width else executor_cls(backend, telemetry)).run(
        sessions, capacity)
    return emitted, sessions, telemetry


@pytest.mark.parametrize("executor_cls,width", [(SequentialExecutor, None), (AsyncGroupedB1Executor, 4)])
def test_streamed_tokens_are_the_served_tokens_in_order(executor_cls, width):
    emitted, sessions, _ = _streamed_run(executor_cls, width)
    _, _, reference = run(SequentialExecutor)
    assert {rid: tuple(tokens) for rid, tokens in emitted.items()} == {
        rid: tokens for rid, (tokens, _) in outputs(reference).items()}
    assert outputs(sessions) == outputs(reference)


def test_a_fallback_restart_never_emits_a_token_twice():
    emitted, sessions, telemetry = _streamed_run(AsyncGroupedB1Executor, 4, fail_on_group_call=2)
    _, _, reference = run(SequentialExecutor)
    assert telemetry.fallbacks == 1
    assert {rid: tuple(tokens) for rid, tokens in emitted.items()} == {
        rid: tokens for rid, (tokens, _) in outputs(reference).items()}


def test_a_replay_that_diverges_from_what_was_streamed_raises():
    from ironmule.executor import TokenStream

    stream = TokenStream(lambda _token: None)
    stream.accept(0, 5)
    stream.accept(1, 6)
    stream.accept(1, 6)  # a replay of the same token only checks
    with pytest.raises(RuntimeError, match="diverged"):
        stream.accept(1, 7)
    with pytest.raises(RuntimeError, match="skipped"):
        stream.accept(3, 8)


def test_a_session_admitted_mid_run_joins_the_rotation_and_matches_sequential():
    backend, telemetry, sessions, capacity = make(SCRIPTS, PROMPTS[:2])
    late = build_sessions([Request(prompt_ids=PROMPTS[2], rid=backend._rid_for(PROMPTS[2]), max_tokens=8,
                                   plan=StrictOneShotPlan())], backend, telemetry, capacity)
    calls = []

    def admit():
        calls.append(len(calls))
        return late if len(calls) == 3 else []  # arrives after two rounds

    AsyncGroupedB1Executor(backend, telemetry, max_width=4).run(sessions, capacity, admit=admit)
    _, _, reference = run(SequentialExecutor, prompts=PROMPTS[:3])
    assert outputs(sessions + late) == outputs(reference)
    assert all(s.done for s in sessions + late) and len(calls) > 3


def test_the_opt_in_round_trace_records_each_round_with_its_join(tmp_path, monkeypatch):
    trace = tmp_path / "rounds.jsonl"
    monkeypatch.setenv("IRONMULE_TRACE_ROUNDS", str(trace))
    backend, telemetry, sessions, capacity = make(SCRIPTS, PROMPTS[:2])
    late = build_sessions([Request(prompt_ids=PROMPTS[2], rid=backend._rid_for(PROMPTS[2]), max_tokens=8,
                                   plan=StrictOneShotPlan())], backend, telemetry, capacity)
    calls = []

    def admit():
        calls.append(len(calls))
        return late if len(calls) == 3 else []

    AsyncGroupedB1Executor(backend, telemetry, max_width=4).run(sessions, capacity, admit=admit)
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    assert len(rows) == len(telemetry.realised_widths)
    assert [r["joined"] for r in rows][:3] == [0, 0, 1]
    assert [r["width"] for r in rows] == telemetry.realised_widths


def test_runtime_serves_requests_admitted_during_a_grouped_run_after_the_original_ones():
    from ironmule.service import ThroughputMode

    class Tokenizer:
        @staticmethod
        def decode(tokens):
            return " ".join(map(str, tokens))

    backend = FakeBackend(SCRIPTS)
    runtime = Runtime.__new__(Runtime)
    runtime.backend, runtime.mode, runtime.tokenizer = backend, ThroughputMode(max_width=4), Tokenizer()
    first = [Request(prompt_ids=ids, rid=backend._rid_for(ids), max_tokens=8, plan=StrictOneShotPlan())
             for ids in PROMPTS[:2]]
    waiting = [Request(prompt_ids=ids, rid=backend._rid_for(ids), max_tokens=8, plan=StrictOneShotPlan())
               for ids in PROMPTS[2:4]]
    offered = []

    def admit(capacity):
        offered.append(capacity)
        return [waiting.pop(0)] if len(offered) in (2, 4) and waiting else []

    results = runtime.serve(first, admit=admit)
    _, _, reference = run(SequentialExecutor, prompts=PROMPTS[:4])
    assert [r.rid for r in results] == [backend._rid_for(ids) for ids in PROMPTS[:4]]
    assert {r.rid: (tuple(r.tokens), r.stop_reason) for r in results} == outputs(reference)
    assert not waiting and len(set(offered)) == 1


def test_an_interactive_runtime_ignores_admission():
    class Tokenizer:
        @staticmethod
        def decode(tokens):
            return ""

    backend = FakeBackend(SCRIPTS)
    runtime = Runtime.__new__(Runtime)
    runtime.backend, runtime.mode, runtime.tokenizer = backend, InteractiveMode(), Tokenizer()
    request = Request(prompt_ids=PROMPTS[0], rid="p10_1", max_tokens=8, plan=StrictOneShotPlan())
    assert len(runtime.serve([request], admit=lambda capacity: 1 / 0)) == 1


@pytest.mark.parametrize("executor_cls,width", [(SequentialExecutor, None), (AsyncGroupedB1Executor, 4)])
def test_a_cancelled_stream_stops_early_and_the_others_keep_their_exact_tokens(executor_cls, width):
    """RSI3-LIVE-P2: a caller-cancelled request ends at its next token boundary; the group's other
    requests produce exactly the sequential reference; a stream without a predicate is unchanged."""
    from ironmule.executor import TokenStream

    backend = FakeBackend(SCRIPTS)
    telemetry, emitted, flags = Telemetry(), {}, {}
    requests = []
    for ids in PROMPTS:
        rid = backend._rid_for(ids)
        emitted[rid], flags[rid] = [], False
        requests.append(Request(prompt_ids=ids, rid=rid, max_tokens=8, plan=StrictOneShotPlan(),
                                stream=TokenStream(emitted[rid].append, lambda rid=rid: flags[rid])))
    victim = requests[1].rid
    original = requests[1].stream.emit

    def cancel_after_two(token):
        original(token)
        if len(emitted[victim]) == 2:
            flags[victim] = True
    requests[1].stream.emit = cancel_after_two
    capacity = backend.capacity_for([len(r.prompt_ids) for r in requests], 8)
    sessions = build_sessions(requests, backend, telemetry, capacity)
    (executor_cls(backend, telemetry, max_width=width) if width else executor_cls(backend, telemetry)).run(
        sessions, capacity)
    _, _, reference = run(SequentialExecutor)
    expected = outputs(reference)
    by_rid = {s.rid: s for s in sessions}
    assert by_rid[victim].stop_reason == "cancelled" and len(by_rid[victim].tokens) == 2
    assert tuple(by_rid[victim].tokens) == expected[victim][0][:2]
    for rid, session in by_rid.items():
        if rid != victim:
            assert (tuple(session.tokens), session.stop_reason) == expected[rid]


def test_a_stream_cancelled_before_prefill_computes_nothing():
    from ironmule.executor import TokenStream

    backend = FakeBackend(SCRIPTS)
    telemetry = Telemetry()
    request = Request(prompt_ids=PROMPTS[0], rid=backend._rid_for(PROMPTS[0]), max_tokens=8,
                      plan=StrictOneShotPlan(), stream=TokenStream(lambda _t: None, lambda: True))
    calls = []
    original = backend.prefill
    backend.prefill = lambda *a, **k: calls.append(1) or original(*a, **k)
    sessions = build_sessions([request], backend, telemetry, 64)
    AsyncGroupedB1Executor(backend, telemetry, max_width=4).run(sessions, 64)
    assert calls == [] and sessions[0].tokens == [] and sessions[0].stop_reason == "cancelled"
