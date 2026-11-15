"""CPU6-M's harness (`tools/cpu6_measure.py`): its rotation and its fixed analysis, on synthetic reports."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("cpu6_measure", ROOT / "tools" / "cpu6_measure.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def _cell(arm, tps, first, complete=True, text="x"):
    return {"arm": arm, "tokens_per_s": tps, "mean_first_s": first, "complete": complete,
            "requests": [{"text": text}] * 8}


def _report(b=5.0, c=12.0, first_b=0.4, first_c=0.1, gate_upper=1.001, reps=5):
    rows = []
    for k in range(reps):
        rows.append([_cell("A", 4.0, 30.0), _cell("B", 4.0 * b, 30.0 * first_b), _cell("C", 4.0 * c, 30.0 * first_c),
                     _cell("A", 4.1, 30.5)])
    return {"repetitions": rows, "gate": {"ran": True, "ratio": 1.0, "interval": [0.999, gate_upper], "passes": True}}


def test_every_arm_runs_once_per_repetition_and_a_twice_in_rotated_order():
    order = tool.rotation(5)
    assert order[0] == ["A", "B", "C", "A"] and order[1] == ["B", "C", "A", "A"]
    assert all(sorted(rep) == ["A", "A", "B", "C"] for rep in order)


def test_both_arms_pass_when_far_faster_first_tokens_sooner_and_the_gate_holds():
    out = tool.analyse(_report())
    assert out["valid"] and out["verdict"] == {"B": "pass", "C": "pass"}
    assert 4.8 < out["B"]["tokens_per_s"][0] < 5.0 and out["C"]["passes_first"]


def test_the_bounds_fixed_before_the_data_decide():
    assert tool.analyse(_report(b=1.4))["verdict"]["B"] == "fail"
    assert tool.analyse(_report(c=2.9))["verdict"]["C"] == "fail"
    assert tool.analyse(_report(first_c=1.2))["verdict"]["C"] == "fail"
    assert tool.analyse(_report(gate_upper=1.006))["verdict"] == {"B": "fail", "C": "fail"}


def test_a_short_or_incomplete_run_is_invalid():
    assert tool.analyse(_report(reps=4))["verdict"] == "invalid"
    report = _report()
    report["repetitions"][2][1]["complete"] = False
    assert tool.analyse(report)["problems"] == ["a request failed or returned nothing"]


def _agent_report(s=12.0, first=0.03, setup=None, reps=5):
    setup = setup or {"engine": "dense", "compute_dtype": None, "cpu_blas": "openblas", "width": 8}
    rows = [[_cell("A", 4.0, 50.0), _cell("S", 4.0 * s, 50.0 * first), _cell("A", 4.1, 51.0)] for _ in range(reps)]
    return {"protocol": "cpu6-a", "setup": setup, "repetitions": rows}


def test_cpu6a_passes_when_the_agent_chose_something_far_faster_by_itself():
    out = tool.analyse_agent(_agent_report())
    assert out["verdict"] == "pass" and out["chose_other_than_a"] and 11.5 < out["tokens_per_s"][0] < 12.5


def test_cpu6a_fails_when_the_agent_kept_a_or_gained_too_little_or_was_incomplete():
    kept = {"engine": "reference", "compute_dtype": "dequantize", "cpu_blas": None}
    assert tool.analyse_agent(_agent_report(setup=kept))["verdict"] == "fail"
    assert tool.analyse_agent(_agent_report(s=2.9))["verdict"] == "fail"
    assert tool.analyse_agent(_agent_report(first=1.2))["verdict"] == "fail"
    assert tool.analyse_agent(_agent_report(reps=3))["verdict"] == "invalid"
    assert tool.rotation(3, tool.ORDER_AGENT) == [["A", "S", "A"], ["S", "A", "A"], ["A", "A", "S"]]


def test_the_agents_setup_is_started_as_the_server_starts_it():
    dense = tool.agent_options({"engine": "dense", "width": 6, "cpu_blas": "openblas"})
    assert dense["engine_configuration"] == "dense_stacked" and dense["dense_width"] == 6 and dense["cpu_blas"] == "openblas"
    assert tool.agent_options({"engine": "reference", "compute_dtype": "dequantize"}) == {"compute_dtype": "dequantize"}
