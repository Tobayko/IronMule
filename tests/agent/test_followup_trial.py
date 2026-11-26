import pytest

from lab.pk1 import followup_trial as trial
from lab.pk1.followup_contract import contract_digest


def test_contract_and_complete_profile_guardrails_are_frozen():
    assert contract_digest() == "9310c18a643c943d79a067221a55e43797cccf065fc387b660e6a8eba8b84b36"
    row = {
        "cold_speedup": 1.08,
        "warm_wall": 1.0,
        "warm_p95": 1.0,
        "warm_fact": 1.0,
        "warm_fact_p95": 1.0,
        "rss": 1.0,
        "device": 1.0,
        "factual_cold": 1.08,
        "factual_cold_p95": 1.0,
        "factual_warm": 1.0,
        "factual_warm_p95": 1.0,
    }
    assert trial.gate([row] * 6, [row] * 2, 0.03)
    assert not trial.gate([row] * 6, [{**row, "factual_warm_p95": 1.021}] * 2, 0.03)
    assert not trial.gate([row] * 6, [row] * 2, 0.081)
    assert trial.select_source("a", actual="b", admitted=True, parent_available=True)["source"] == "reference"
    assert trial.select_source("a", actual="a", admitted=False, parent_available=True)["source"] == "reference"
    assert (
        trial.select_source("a", actual="a", admitted=True, parent_available=False)["source"] == "preceding-deployment"
    )


def test_only_ordered_method_changes(tmp_path):
    src = tmp_path / "ref.py"
    cand = tmp_path / "proposal.py"
    text = (
        "class GgmlEngine:\n    def thing(self, arg):\n        return arg\n\n    def extra(self):\n        return 3\n"
    )
    src.write_text(text)
    cand.write_text(text.replace("return arg", "return arg + 1"))
    order = {"target": {"function": "thing", "line": 2}, "source_sha256": trial.prior.native.checksum(src)}
    trial.source_check(src, cand, order)
    cand.write_text(cand.read_text().replace("return 3", "return 4"))
    with pytest.raises(ValueError):
        trial.source_check(src, cand, order)
