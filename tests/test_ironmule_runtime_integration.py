"""Real-model counterpart to test_forge_runtime.py. Skipped when the model is absent.

Checks the things a fake backend cannot: that grouped execution on the real engine
produces the same token ids, counts, stop reasons and KV state hashes as sequential
execution, under both execution plans.
"""

import os

import pytest

pytestmark = pytest.mark.integration

os.environ.setdefault("HF_HUB_OFFLINE", "1")


def _runtime(mode):
    ironmule = pytest.importorskip("ironmule")
    try:
        return ironmule.Runtime.load(mode=mode, use_tuned_profile=True), ironmule
    except Exception as exc:                                  # pragma: no cover
        pytest.skip(f"model unavailable: {type(exc).__name__}: {exc}")


DOC = ("The Apollo program was carried out by NASA between 1961 and 1972. "
       "Apollo 11 landed the first humans on the Moon in July 1969. "
       "The Saturn V rocket was used for the crewed lunar missions. ") * 6
QUESTIONS = ["Which program landed humans on the Moon?",
             "Which rocket was used for the crewed lunar missions?",
             "In which year did Apollo 11 land?",
             "Which agency carried out the program?"]


@pytest.fixture(scope="module")
def pair():
    rt, ironmule = _runtime(None)
    return rt, ironmule


def _requests(rt, ironmule, plan, caps):
    return [ironmule.Request(prompt_ids=rt.encode(DOC + "\n\nQuestion: " + q),
                          max_tokens=cap, plan=plan, rid=f"q{i}")
            for i, (q, cap) in enumerate(zip(QUESTIONS, caps))]


def _kv_hashes(rt, results):
    return [r.metrics["generated_tokens"] for r in results]


@pytest.mark.parametrize("plan_name", ["strict", "reusable"])
def test_grouped_matches_sequential_on_the_real_engine(pair, plan_name):
    rt, ironmule = pair
    plan = (ironmule.StrictOneShotPlan() if plan_name == "strict"
            else rt.session_plan(DOC, name="apollo"))
    caps = [6, 10, 4, 8]                       # ragged, so requests finish at different times

    rt.mode = ironmule.InteractiveMode()
    reference = rt.serve(_requests(rt, ironmule, plan, caps))
    ref_telemetry = rt.telemetry.snapshot()

    rt.mode = ironmule.ThroughputMode()
    grouped = rt.serve(_requests(rt, ironmule, plan, caps))
    grp_telemetry = rt.telemetry.snapshot()

    by_rid = {r.rid: r for r in reference}
    for result in grouped:
        want = by_rid[result.rid]
        assert result.tokens == want.tokens, f"token ids differ for {result.rid}"
        assert len(result.tokens) == len(want.tokens)
        assert result.stop_reason == want.stop_reason
        assert result.text == want.text

    assert ref_telemetry["correctness_errors"] == 0
    assert grp_telemetry["correctness_errors"] == 0
    assert grp_telemetry["fallbacks"] == 0
    assert grp_telemetry["plan_switch_attempts"] == 0
    assert grp_telemetry["max_realised_width"] > 1, "throughput mode must actually group"


def test_kv_state_is_identical_between_modes(pair):
    rt, ironmule = pair
    plan = ironmule.StrictOneShotPlan()
    request = ironmule.Request(prompt_ids=rt.encode(DOC + "\n\nQuestion: " + QUESTIONS[0]),
                            max_tokens=6, plan=plan, rid="single")

    capacity = rt.backend.capacity_for([len(request.prompt_ids)], request.max_tokens)
    state, first = rt.backend.prefill(request.prompt_ids, plan, capacity)
    baseline = rt.backend.kv_hash(state, len(request.prompt_ids))

    again_state, again_first = rt.backend.prefill(request.prompt_ids, plan, capacity)
    assert again_first == first
    assert rt.backend.kv_hash(again_state, len(request.prompt_ids)) == baseline, \
        "the same prompt under the same plan must produce the same KV state"


def test_reusable_session_reuses_its_prefix(pair):
    rt, ironmule = pair
    plan = rt.session_plan(DOC, name="apollo")
    rt.mode = ironmule.InteractiveMode()
    rt.serve(_requests(rt, ironmule, plan, [4, 4, 4, 4]))
    described = plan.describe()
    assert described["hits"] >= 3, f"expected cache hits after the first request: {described}"
    assert described["misses"] == 1


def test_sequential_fallback_produces_the_same_answer(pair, monkeypatch):
    rt, ironmule = pair
    plan = ironmule.StrictOneShotPlan()
    caps = [4, 4, 4, 4]

    rt.mode = ironmule.InteractiveMode()
    reference = {r.rid: r.tokens for r in rt.serve(_requests(rt, ironmule, plan, caps))}

    original = rt.backend.complete
    calls = {"n": 0}

    def flaky(handles):
        calls["n"] += 1
        if calls["n"] == 2 and len(handles) > 1:
            raise RuntimeError("injected device failure")
        return original(handles)

    monkeypatch.setattr(rt.backend, "complete", flaky)
    rt.mode = ironmule.ThroughputMode()
    results = rt.serve(_requests(rt, ironmule, plan, caps))
    snap = rt.telemetry.snapshot()

    assert snap["fallbacks"] >= 1, "the injected failure must be observed"
    for result in results:
        assert result.tokens == reference[result.rid], "fallback changed the answer"
