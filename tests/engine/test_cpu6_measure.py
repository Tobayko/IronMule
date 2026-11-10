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
