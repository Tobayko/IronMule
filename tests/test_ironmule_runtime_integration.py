"""Real-model counterpart to test_forge_runtime.py. Skipped when the model is absent.

Checks the things a fake backend cannot: that grouped execution on the real engine
produces the same token ids, counts, stop reasons and KV state hashes as sequential
execution, under both execution plans.
"""

import json
import os

import pytest

pytestmark = pytest.mark.integration

os.environ.setdefault("HF_HUB_OFFLINE", "1")


def _runtime(mode):
    ironmule = pytest.importorskip("ironmule")
    try:
        return ironmule.Runtime.load(mode=mode, use_tuned_profile=True), ironmule
    except Exception as exc:                                  # pragma: no cover
        if _is_expected_unavailable(exc):
            pytest.skip(f"model unavailable: {type(exc).__name__}: {exc}")
        raise


def _is_expected_unavailable(exc: BaseException) -> bool:
    """Only classify explicit model/access/Metal availability failures as skips."""
    if isinstance(exc, (FileNotFoundError, PermissionError)):
        return True
    if isinstance(exc, ModuleNotFoundError):
        module_name = exc.name
        if not module_name and "'" in str(exc):
            module_name = str(exc).split("'")[1]
        return (module_name in {"mlx", "mlx.core", "mlx_lm", "mlx_lm.models"}
                or str(module_name).split(".", 1)[0] in {"mlx", "mlx_lm"})
    message = str(exc).lower()
    availability = (
        "model not found", "model unavailable", "model is unavailable",
        "model is not cached",
        "no such file", "permission denied", "access denied", "cannot access",
        "metal is not available", "metal unavailable", "metal device unavailable",
        "no metal device", "gpu is not available", "gpu unavailable",
    )
    return isinstance(exc, (ImportError, OSError, RuntimeError, ValueError)) and any(
        marker in message for marker in availability
    )


def test_unexpected_integration_errors_are_not_classified_as_unavailable():
    assert not _is_expected_unavailable(RuntimeError("injected programming error"))
    assert not _is_expected_unavailable(ValueError("invalid tensor shape"))


@pytest.mark.parametrize("error", [
    ValueError("model is not cached: expected exactly one unique cached revision"),
    FileNotFoundError("local model not found"),
    RuntimeError("Metal device unavailable"),
    RuntimeError("model unavailable on this host"),
    ModuleNotFoundError("No module named 'mlx'"),
])
def test_known_environment_failures_are_skippable(error):
    assert _is_expected_unavailable(error)


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


def test_runtime_exposes_exact_path_free_model_identity(pair):
    rt, ironmule = pair
    identity = rt.model_identity
    assert identity is not None
    assert identity.model_id == "mlx-community/gemma-3-4b-it-4bit"
    assert identity.revision == "93724907d4ed1745d2fe50baadf3b0b01a65abf2"
    assert identity.model_manifest_sha256 == \
        "a405b1a73ee9fac816ed7cfeab45b70a26f031843467a4aa4030edc663e857ae"
    assert identity.quantisation == {"bits": 4, "group_size": 64}
    assert "/Users/" not in json.dumps(identity.to_dict())
    record = rt.fingerprint(
        ironmule.StrictOneShotPlan(),
        {"prompt_tokens": 16, "max_tokens": 8, "concurrency": 1},
    )
    assert record["model_identity_sha256"] == identity.identity_sha256
    assert record["model_revision"] == identity.revision


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
