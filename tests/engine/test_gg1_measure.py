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


def _gg3(l0, ls, g, first=0.9):
    def block(rate, n, at):
        return [_cell("L0", rate * l0, at, n), _cell("Ls", rate * ls, at, n), _cell("G", rate * g, at * first, n),
                _cell("Ls", rate * ls, at, n), _cell("L0", rate * l0, at, n)]
    return {"protocol": "gg3-h", "gate": {"passes": True, "settings": {"flash_attn": True}},
            "load": [block(300.0, 8, 0.01) for _ in range(5)], "quiet": [block(160.0, 16, 0.05)]}


def test_gg3_judges_g_against_l0_and_against_ls_and_leads_with_the_fair_one():
    out = tool.analyse(_gg3(l0=1.0, ls=1.3, g=1.3))
    assert out["against_l0"]["verdict"] == "goal reached" and out["against_ls"]["verdict"] == "goal not reached"
    assert out["verdict"] == "fair: goal not reached"
    assert tool.analyse(_gg3(l0=1.0, ls=1.0, g=1.3))["verdict"] == "goal reached"
    report = _gg3(1.0, 1.0, 1.0)
    report["load"][0].pop()
    assert tool.analyse(report)["problems"] == ["missing cells"]


def test_gg3_gives_llama_server_the_options_of_the_engine_s_setting():
    assert tool.llama_options({"flash_attn": False, "kv_unified": False, "kv_f32": False}) == ()
    assert tool.llama_options({"flash_attn": True, "kv_unified": True, "kv_f32": True}) == (
        "-fa", "-kvu", "-ctk", "f32", "-ctv", "f32")
