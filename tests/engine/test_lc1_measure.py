"""LC1's harness (`tools/lc1_measure.py`): its fixed, descriptive analysis on a synthetic report."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("lc1_measure", ROOT / "tools" / "lc1_measure.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def _cell(arm, rate, first, n, text="x"):
    rows = [{"first_s": first, "end_s": first + 63 / rate, "tokens": 64, "text": text, "error": None}
            for _ in range(n)]
    return {"arm": arm, "complete": True, "requests": rows}


def test_it_reports_llama_cpp_over_ironmule_with_ironmule_s_own_noise():
    report = {"load": [[_cell("I", 10.0, 1.0, 8), _cell("L", 20.0, 0.5, 8, "y"), _cell("I", 10.0, 1.0, 8)]
                       for _ in range(5)],
              "quiet": [[_cell("I", 10.0, 0.1, 16), _cell("L", 30.0, 0.05, 16), _cell("I", 10.0, 0.1, 16)]]}
    out = tool.analyse(report)
    assert out["complete"] and out["load_blocks"] == 5 and out["quiet"]["prompts"] == 16
    assert 1.9 < out["load"]["l_over_i_tokens_per_s"][0] < 2.1 and out["load"]["i_aa_tokens_per_s"] == 0
    assert 2.9 < out["quiet"]["l_over_i_tokens_per_s"][0] < 3.1 and out["quiet"]["l_over_i_first"][0] == 0.5
    assert out["answers_equal"] == 16 and out["answers"] == 56
