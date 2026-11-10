"""GG1-H's harness (`tools/gg1_measure.py`): its fixed analysis on synthetic reports."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("gg1_measure", ROOT / "tools" / "gg1_measure.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def _cell(arm, rate, first, n):
    rows = [{"first_s": first, "end_s": first + 63 / rate, "tokens": 64, "text": "x", "error": None} for _ in range(n)]
    return {"arm": arm, "complete": True, "requests": rows}


def _report(quiet=1.3, load=1.3, first=0.9, passes=True):
    return {"gate": {"passes": passes},
            "load": [[_cell("L", 300.0, 0.01, 8), _cell("G", 300.0 * load, 0.01 * first, 8), _cell("L", 300.0, 0.01, 8)]
                     for _ in range(5)],
            "quiet": [[_cell("L", 160.0, 0.05, 16), _cell("G", 160.0 * quiet, 0.05 * first, 16),
                       _cell("L", 160.0, 0.05, 16)]]}


def test_the_goal_needs_both_halves_above_1_2_and_first_tokens_no_later():
    assert tool.analyse(_report())["verdict"] == "goal reached"
    out = tool.analyse(_report(load=1.1))
    assert out["verdict"] == "goal not reached" and out["checks"] == {
        "quiet_tokens_per_s": True, "load_tokens_per_s": False, "quiet_first": True, "load_first": True}
    assert tool.analyse(_report(first=1.2))["checks"]["load_first"] is False


def test_a_failed_gate_or_request_makes_the_run_invalid():
    assert tool.analyse(_report(passes=False))["problems"] == ["the gate did not pass"]
    report = _report()
    report["quiet"][0][1]["complete"] = False
    assert tool.analyse(report)["verdict"] == "invalid"
