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

    def measure(model_id, compute_dtype):
        calls.append(("measure", compute_dtype))
        return _measured()

    setup = pc.choose_setup("m", "fp", "a" * 64, log=lambda _: None, numeric=numeric, measure=measure)
    assert setup["engine"] == "dense" and setup["compute_dtype"] == "native" and "tokens per second" in setup["reason"]
    assert setup["stacked"]["gate"]["passes"] and setup["numeric"]["reason"] == "gate passed"
    again = pc.choose_setup("m", "fp", "a" * 64, log=lambda _: None, numeric=numeric, measure=measure)
    assert again["engine"] == "dense" and calls == ["numeric", ("measure", "native")]  # stored, not re-measured


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
