import unittest

from friday_h0.decision import (
    baseline_ns,
    decision_hash,
    evaluate_analysis_fixture,
    make_analysis_fixture,
)


class DecisionTests(unittest.TestCase):
    def test_registered_baseline_formula_and_fixtures(self):
        self.assertEqual(baseline_ns(0, 0), 995_000)
        slow = make_analysis_fixture("slow")
        self.assertEqual(len(slow["rows"]), 90)
        self.assertEqual(slow["rows"][0]["candidate_ns"], 1_094_500)

    def test_analysis_outcomes(self):
        self.assertEqual(evaluate_analysis_fixture("slow")["classification"], "regression")
        self.assertEqual(evaluate_analysis_fixture("slow")["action"], "baseline_fallback")
        self.assertEqual(evaluate_analysis_fixture("known_win")["classification"], "promoted")
        wrong = evaluate_analysis_fixture("wrong")
        self.assertEqual(wrong["classification"], "invalid: correctness")
        self.assertFalse(wrong["timed"])
        missing = evaluate_analysis_fixture("missing")
        self.assertEqual(missing["classification"], "invalid: missing_required_field")

    def test_decision_hash_replays_without_self_reference(self):
        decision = evaluate_analysis_fixture("slow")
        self.assertEqual(decision["decision_hash"], decision_hash(decision))
        self.assertEqual(decision["decision_hash"], evaluate_analysis_fixture("slow")["decision_hash"])

