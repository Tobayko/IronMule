"""SP2-H's harness (`tools/sp2_measure.py`): its fixed analysis, on synthetic reports."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("sp2_measure", ROOT / "tools" / "sp2_measure.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def _row(rate, first, tokens=65, ids=(1, 2, 3)):
    return {"offered_s": 0.0, "first_s": first, "end_s": first + (tokens - 1) / rate, "tokens": tokens,
            "token_ids": list(ids), "text": "x", "error": None}


def _cell(arm, rate, first, n):
    return {"arm": arm, "complete": True, "requests": [_row(rate, first) for _ in range(n)]}


def _report(quiet_gain=1.6, quiet_first=1.0, load_gain=1.0, load_first=1.0):
    quiet = [[_cell("A", 10.0, 0.1, 16), _cell("S", 10.0 * quiet_gain, 0.1 * quiet_first, 16), _cell("A", 10.0, 0.1, 16)]
             for _ in range(3)]
    load = [[_cell("A", 10.0, 1.0, 8), _cell("S", 10.0 * load_gain, 1.0 * load_first, 8), _cell("A", 10.0, 1.0, 8)]
            for _ in range(3)]
    return {"quiet": quiet, "load": load}


def test_it_passes_when_quiet_is_far_faster_and_load_unharmed_with_every_answer_equal():
    out = tool.analyse(_report())
    assert out["verdict"] == "pass" and out["answers_unequal"] == 0 and 1.55 < out["quiet"]["tokens_per_s"][0] < 1.65


def test_the_bounds_fixed_before_the_data_decide():
    assert tool.analyse(_report(quiet_gain=1.2))["checks"]["quiet_tokens_per_s"] is False
    assert tool.analyse(_report(quiet_first=1.2))["checks"]["quiet_first"] is False
    assert tool.analyse(_report(load_gain=0.95))["checks"]["load_tokens_per_s"] is False
    assert tool.analyse(_report(load_first=1.1))["checks"]["load_first"] is False
    report = _report()
    report["load"][1][1]["requests"][3]["token_ids"] = [1, 2, 4]
    out = tool.analyse(report)
    assert out["answers_unequal"] == 1 and out["verdict"] == "fail"


def test_a_missing_block_or_failed_request_is_invalid():
    report = _report()
    report["quiet"].pop()
    assert tool.analyse(report)["verdict"] == "invalid"
    report = copy.deepcopy(_report())
    report["load"][0][2]["complete"] = False
    assert tool.analyse(report)["problems"] == ["a request failed or returned nothing"]
