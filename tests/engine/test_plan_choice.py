"""AG4: the agent's own choice of plan and engine (ironmule/plan_choice.py), by the rule BACKLOG AG4 fixed."""

from __future__ import annotations

import pytest

from ironmule import plan_choice as pc


def _speed(base_step=0.052, plan_steps=None, base_first=0.5, plan_first=0.5):
    plan_steps = plan_steps or {"1": 0.053, "8": 0.17}
    row = {"base_first_s": base_first, "base_step_s": base_step, "plan_first_s": plan_first, "plan_step_s": plan_steps}
    return {"short": row, "long": dict(row)}


def _measured(passes=True, **speed):
    return {"eligible": True, "install": {}, "speed": _speed(**speed),
            "gate": {"passes": passes, "ratio": 1.001, "interval": [0.998, 1.004]}}


def test_the_stacked_engine_serves_when_it_passes_every_part_of_the_rule():
    decision = pc.decide(_measured())
    assert decision["stacked"] and decision["tokens_per_second"]["stacked_8"] == pytest.approx(8 / 0.17)
    assert "2.45x" in decision["reason"]


@pytest.mark.parametrize("measured, why", [
    ({"eligible": False, "reason": "Qwen 3 only"}, "no stacked engine"),
    (_measured(passes=False), "quality gate"),
    (_measured(plan_steps={"1": 0.053, "8": 0.5}), "no more tokens"),
    (_measured(plan_steps={"1": 0.06, "8": 0.17}), "one request alone"),  # 0.87x of the base path
    (_measured(plan_first=0.6), "first token"),                          # 1.2x later
])
def test_the_base_path_serves_otherwise(measured, why):
    decision = pc.decide(measured)
    assert not decision["stacked"] and why in decision["reason"]


def test_the_choice_is_made_once_and_shown_with_its_reasons(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "_store", lambda: tmp_path / "plan_choice.json")
    calls = []

    def numeric(model_id, fingerprint, identity, log):
        calls.append("numeric")
        return {"plan": "native", "reason": "gate passed"}

    def measure(model_id, compute_dtype, base_only=False):
        calls.append(("measure", compute_dtype))
        return _measured()

    setup = pc.choose_setup("m", "fp", "a" * 64, log=lambda _: None, numeric=numeric, measure=measure)
    assert setup["engine"] == "dense" and setup["compute_dtype"] == "native" and "tokens per second" in setup["reason"]
    assert setup["stacked"]["gate"]["passes"] and setup["numeric"]["reason"] == "gate passed"
    again = pc.choose_setup("m", "fp", "a" * 64, log=lambda _: None, numeric=numeric, measure=measure)
    assert again["engine"] == "dense" and calls == ["numeric", ("measure", "native")]  # stored, not re-measured


def test_a_float_plan_is_compared_against_not_stacked_on(tmp_path, monkeypatch):
    """AG4-Q run 1: NUM1 chose float16 for Qwen 3 bf16, and the dense plan refused float16 weights. The stacked
    engine runs on the checkpoint's bf16 instead and must beat the float16 path's own speed."""
    monkeypatch.setattr(pc, "_store", lambda: tmp_path / "plan_choice.json")
    calls = []

    def measure(model_id, compute_dtype, base_only=False):
        calls.append((compute_dtype, base_only))
        if base_only:  # float16 alone: 3.2x faster than bf16 emulation
            row = {"base_first_s": 0.2, "base_step_s": 0.016}
            return {"short": row, "long": dict(row)}
        return _measured(base_step=0.052, plan_steps={"1": 0.05, "8": 0.04}, base_first=0.5, plan_first=0.21)

    setup = pc.choose_setup("m", "fp", "b" * 64, log=lambda _: None,
                            numeric=lambda *a, **k: {"plan": "float16", "reason": "table"}, measure=measure)
    assert calls == [(None, False), ("float16", True)]
    row = setup["stacked"]["speed"]["short"]
    assert row["base_step_s"] == 0.016 and row["stacked_base_step_s"] == 0.052
    # 8 / 0.04 = 200 tokens/s against float16's 62.5, but alone 20 against 62.5: quiet traffic would regress
    assert setup["engine"] == "reference" and setup["compute_dtype"] == "float16"
    assert "one request alone" in setup["reason"]


mx = pytest.importorskip("mlx.core")


def test_measure_stacked_runs_on_a_cpu_stand_in():
    """Every number the rule reads, on a tiny random 4-bit Gemma 3 with FA2's semantics in plain MLX."""
    from ironmule import dense_gemma

    from tests.engine.test_dense_gemma import _Tok, tiny_model

    model = tiny_model()

    def make_plan(model_, info):
        plan = dense_gemma.GemmaDensePlan(model_, slots=4, capacity=96, attention=dense_gemma.reference_attention,
                                          native=False)
        return plan, dense_gemma.probe(plan)

    measured = pc.measure_stacked(model, _Tok(), None, chunks=4, size=16, steps=3, long_prompt="x" * 40,
                                  make_plan=make_plan)
    assert measured["eligible"] and set(measured["gate"]) >= {"ratio", "interval", "passes"}
    assert set(measured["speed"]) == {"short", "long"}
    assert set(measured["speed"]["short"]["plan_step_s"]) == {"1", "4"}
    assert "stacked" in pc.decide(measured)


def test_a_model_without_a_stacked_engine_is_measured_as_ineligible():
    def refuse(model, info):
        raise ValueError("the dense plan serves Qwen 3 checkpoints only")

    assert pc.measure_stacked(object(), None, None, make_plan=refuse) == {
        "eligible": False, "reason": "the dense plan serves Qwen 3 checkpoints only"}


def test_a_stacked_engine_that_wins_only_under_load_serves_beside_the_single_path_where_both_fit():
    """AG4 step 2: AG4-Q run 2's numbers (dense 5x at width 8, 0.82x alone); with room for two copies the agent
    serves both and routes each request, without room it keeps the single path."""
    measured = _measured(base_step=0.0173, plan_steps={"1": 0.0212, "8": 0.0275})
    fits = pc.decide({**measured, "memory": {"two_paths_fit": True}})
    assert not fits["stacked"] and fits["route"] and "routes each request" in fits["reason"]
    tight = pc.decide({**measured, "memory": {"two_paths_fit": False}})
    assert not tight["stacked"] and not tight["route"]


def test_the_product_serves_the_float16_plan_nums1_can_choose():
    """AG4-R's smoke on a T4: NUM1 chose `float16`, which the product's worker did not accept. It does now, in the
    client, the serve flag and the worker's own arguments."""
    import inspect

    from ironmule_product import worker
    from ironmule_product.backend import MLXWorkerClient
    from ironmule_product.types import ModelSpec

    client = MLXWorkerClient(ModelSpec("m", "r", "/tmp/m", 1), compute_dtype="float16")
    assert client.compute_dtype == "float16"
    source = inspect.getsource(worker.main)
    assert '"float16"' in source and "mx.float16" in source
    with pytest.raises(ValueError):
        MLXWorkerClient(ModelSpec("m", "r", "/tmp/m", 1), compute_dtype="int8")


def test_the_setup_lists_every_lever_the_agent_knows_built_or_not(tmp_path, monkeypatch):
    """HQ1: the levers read from HyperQwen are known to the agent and shown as not built yet; the built ones say
    what happened to them on this machine."""
    monkeypatch.setattr(pc, "_store", lambda: tmp_path / "plan_choice.json")
    measured = _measured(base_step=0.0173, plan_steps={"1": 0.0212, "8": 0.0275})
    measured["memory"] = {"two_paths_fit": True}
    setup = pc.choose_setup("m", "fp", "c" * 64, log=lambda _: None,
                            numeric=lambda *a, **k: {"plan": None, "reason": "none"},
                            measure=lambda *a, **k: measured)
    by_name = {lever["name"]: lever for lever in setup["levers"]}
    assert set(by_name) == {lever["name"] for lever in pc.LEVERS}
    assert by_name["routed_paths"]["here"] == "used" and by_name["stacked_engine"]["here"].startswith("used")
    assert by_name["numeric_plan"]["here"].startswith("measured, none chosen")
    for planned in ("tensor_core_rows", "context_speculation", "kv_cache_8bit", "int8_head"):
        assert by_name[planned]["status"] == "planned" and by_name[planned]["here"] == "not built yet"
        assert "HyperQwen" in by_name[planned]["what"]
