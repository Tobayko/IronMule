"""friday_optimizer.reward: one measured session becomes one label.

Fixtures only -- the shape of a session result is a contract, and checking that
the contract is read correctly needs no GPU. No number here is evidence about
any candidate.
"""

from __future__ import annotations

import unittest

from friday_optimizer.reward import (
    EVIDENCE_SCHEMA,
    RESULT_SCHEMA,
    RewardError,
    censoring_for,
    outcome_for,
    ratio_median,
)

DIGEST = "a" * 64


def _arm(*values: float) -> dict:
    return {"total_ns": list(values), "prefill_ns": [1.0], "decode_ns": [1.0]}


def _pair(pair_id: str, order: str, baseline, candidate) -> dict:
    return {
        "pair_id": pair_id,
        "order": order,
        "arms": {"baseline_samples": _arm(*baseline), "candidate_samples": _arm(*candidate)},
    }


def _result(**overrides) -> dict:
    base = {
        "schema": RESULT_SCHEMA,
        "run_ok": True,
        "reason": "measured",
        "measurement_evidence": {
            "schema": EVIDENCE_SCHEMA,
            "status": "complete",
            "evidence_sha256": DIGEST,
            "test": {
                "pair_count": 2,
                "pairs": [
                    _pair("p0", "AB", [100.0, 100.0], [90.0, 90.0]),
                    _pair("p1", "BA", [100.0, 100.0], [80.0, 80.0]),
                ],
            },
        },
    }
    base.update(overrides)
    return base


class RatioTest(unittest.TestCase):
    def test_the_ratio_is_paired_first_then_aggregated(self) -> None:
        # 0.90 and 0.80 per pair; the median across pairs is 0.85.
        self.assertAlmostEqual(ratio_median(_result()["measurement_evidence"]["test"]), 0.85)

    def test_a_missing_series_is_not_silently_completed(self) -> None:
        evidence = _result()["measurement_evidence"]["test"]
        evidence["pairs"][0]["arms"]["candidate_samples"]["total_ns"] = []
        self.assertIsNone(ratio_median(evidence))

    def test_a_nonpositive_duration_is_refused(self) -> None:
        evidence = _result()["measurement_evidence"]["test"]
        evidence["pairs"][0]["arms"]["baseline_samples"]["total_ns"] = [0.0, 100.0]
        self.assertIsNone(ratio_median(evidence))


class OutcomeTest(unittest.TestCase):
    def test_a_measured_run_becomes_an_observed_label(self) -> None:
        outcome = outcome_for(_result(), decision_id="camp.0001")
        self.assertTrue(outcome.observed)
        self.assertAlmostEqual(outcome.reward, 0.85)
        self.assertEqual(outcome.reward_metric, "ratio_median")
        self.assertEqual(outcome.evidence_hash, DIGEST)

    def test_the_metric_is_always_time_like(self) -> None:
        # replay.default_reward computes 1 - reward without reading the metric,
        # so a throughput metric would enter the estimators sign-flipped.
        self.assertEqual(outcome_for(_result(), decision_id="d1").reward_metric, "ratio_median")

    def test_a_failed_run_is_censored_not_dropped(self) -> None:
        outcome = outcome_for(
            _result(run_ok=False, reason="readiness_failed"), decision_id="camp.0002"
        )
        self.assertFalse(outcome.observed)
        self.assertEqual(outcome.censoring, "censored_gate_failed")
        self.assertIsNone(outcome.reward)

    def test_incomplete_evidence_is_censored(self) -> None:
        result = _result()
        result["measurement_evidence"] = {
            "schema": EVIDENCE_SCHEMA,
            "status": "unavailable",
            "reason": "measurement_evidence_invalid_or_missing",
        }
        outcome = outcome_for(result, decision_id="camp.0003")
        self.assertEqual(outcome.censoring, "censored_error")

    def test_evidence_that_claims_complete_but_carries_nothing_is_censored(self) -> None:
        result = _result()
        result["measurement_evidence"]["test"] = {"pairs": []}
        outcome = outcome_for(result, decision_id="camp.0004")
        self.assertEqual(outcome.censoring, "censored_error")
        self.assertEqual(outcome.notes, "measurement_evidence_unusable")

    def test_censoring_is_classified_by_cause(self) -> None:
        self.assertEqual(censoring_for("readiness_failed"), "censored_gate_failed")
        self.assertEqual(censoring_for("session_plan_blocked"), "censored_gate_failed")
        self.assertEqual(censoring_for("stage timeout: deadline"), "censored_timeout")
        self.assertEqual(censoring_for("worker crashed"), "censored_error")

    def test_a_foreign_document_is_refused(self) -> None:
        with self.assertRaises(RewardError):
            outcome_for({"schema": "something.else"}, decision_id="d1")

    def test_the_label_round_trips_into_a_memory_record(self) -> None:
        record = outcome_for(_result(), decision_id="camp.0005").as_record()
        self.assertEqual(record.record_id, "outcome:camp.0005")
        self.assertEqual(record.phase.value, "label")


if __name__ == "__main__":
    unittest.main()
