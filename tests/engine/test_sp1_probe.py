"""SP1-0's probe (`tools/sp1_probe.py`): drafts, acceptance, the verify window's exactness on a tiny model, the rule."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("sp1_probe", ROOT / "tools" / "sp1_probe.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def test_a_draft_copies_what_followed_the_most_recent_earlier_occurrence_of_the_longest_suffix():
    assert tool.draft([5, 6, 7, 8, 1, 5, 6, 7], 3) == [8, 1, 5]
    assert tool.draft([1, 2, 9, 1, 2, 4, 2], 2) == [4, 2]  # 1-gram "2": most recent earlier one at index 4
    assert tool.draft([1, 2, 3], 4) == [] and tool.draft([4, 4], 0) == []


def test_acceptance_takes_the_agreeing_prefix_then_the_models_own_token():
    assert tool.accept([3, 4, 5], [3, 4, 9, 7]) == [3, 4, 9]
    assert tool.accept([3], [8, 1]) == [8] and tool.accept([], [6]) == [6]


@pytest.mark.parametrize("family", ["LLAMA", "QWEN3"])
def test_a_verify_window_answers_exactly_as_one_row_at_a_time(family):
    pytest.importorskip("mlx.core")
    helper = importlib.util.spec_from_file_location("dense_cpu_tests", ROOT / "tests" / "engine" / "test_dense_cpu.py")
    cpu = importlib.util.module_from_spec(helper)
    helper.loader.exec_module(cpu)
    plan = tool.spec_plan_class()(cpu.tiny(getattr(cpu, family)), slots=8, capacity=256)
    prompt = [7, 3, 9, 7, 3, 9, 7, 3, 9, 7, 3, 11, 4, 7, 3]  # repeats, so drafts are made
    sequential = tool.decode(plan, prompt, tool.prefill(plan, prompt), set(), 40)
    for k in tool.KS:
        speculative = tool.decode(plan, prompt, tool.prefill(plan, prompt), set(), 40, k)
        assert speculative["tokens"] == sequential["tokens"] and speculative["drafted"] > 0


def _arm(tps, tokens=(1, 2, 3), drafted=10, accepted=5):
    return {"tokens_per_s": tps, "tokens": list(tokens), "drafted": drafted, "accepted": accepted}


def _report(k_tps, n=10):
    return {"prompts": [{"index": i, "arms": {"seq": _arm(10.0, drafted=0, accepted=0),
                                              "seq2": _arm(10.0, drafted=0, accepted=0),
                                              **{f"k{k}": _arm(t) for k, t in zip(tool.KS, k_tps)}}}
                        for i in range(n)]}


def test_the_rule_fixed_before_the_data_decides():
    assert tool.analyse(_report((14.0, 16.0, 12.0)))["decision"].startswith("go: k4")
    assert tool.analyse(_report((11.0, 12.0, 10.0)))["decision"].startswith("kill")
    report = _report((14.0, 16.0, 12.0))
    report["prompts"][3]["arms"]["k7"]["tokens"] = [1, 2, 4]
    out = tool.analyse(report)
    assert out["decision"].startswith("not exact") and out["differing"][0] == {"prompt": 3, "arm": "k7", "position": 2}
