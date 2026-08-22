"""Offline replay, sealing, and decision tests for formal N10-v1."""

from __future__ import annotations

import copy
import hashlib
import unittest

from friday_n10.constants import (
    BOOTSTRAP_SEEDS,
    CALIBRATION,
    CANDIDATE_COOLDOWN_SECONDS,
    CONFIRMATION,
    CONTINUOUS_GPU_LIMIT_SECONDS,
    DUTY_CYCLE_LIMIT,
    GPU_WORK_LIMIT_SECONDS,
    MEASUREMENT_BLOCKS,
    MINIMUM_EFFECT_FLOOR,
    FIXTURE_SEED,
    OPERAND_SEED,
    REQUIRED_BREAK_SECONDS,
    SESSION_ORDER,
    SESSION_SPECS,
    WALL_LIMIT_SECONDS,
    WARMUP_PAIRS,
)
from friday_n10.protocol import (
    ProtocolError,
    build_calibration_summary,
    build_confirmation_seal,
    build_preregistration,
    build_session_payload,
    build_study_decision,
    orders_for,
    study_specification,
    validate_history,
    validate_session,
)

PROVENANCE = "a" * 64
OUTPUT = "b" * 64


def budget() -> dict[str, object]:
    return {
        "passed": True,
        "gpu_work_seconds": 1.0,
        "max_continuous_gpu_seconds": 1.0,
        "cooldown_seconds": 0.0,
        "required_break_seconds": 0.0,
        "wall_seconds": 2.0,
        "gpu_work_limit_seconds": GPU_WORK_LIMIT_SECONDS,
        "continuous_gpu_limit_seconds": CONTINUOUS_GPU_LIMIT_SECONDS,
        "duty_cycle_limit": DUTY_CYCLE_LIMIT,
        "wall_limit_seconds": WALL_LIMIT_SECONDS,
        "candidate_cooldown_seconds": CANDIDATE_COOLDOWN_SECONDS,
        "required_break_limit_seconds": REQUIRED_BREAK_SECONDS,
    }


def resources() -> dict[str, object]:
    return {
        "cpu_process_ns": 10,
        "rss_peak_bytes": 20,
        "mlx_active_memory_bytes": 30,
        "mlx_peak_memory_bytes": 40,
        "mlx_cache_memory_bytes": None,
    }


def correctness() -> dict[str, object]:
    return {
        "status": "byte_identical",
        "max_abs_error": 0.0,
        "reference_sha256": OUTPUT,
        "a_output_sha256": OUTPUT,
        "b_output_sha256": OUTPUT,
    }


def blocks(stage: str, session_id: str, *, warmup: bool, ratio: float) -> list[dict[str, object]]:
    count = WARMUP_PAIRS if warmup else MEASUREMENT_BLOCKS
    result = []
    for index, order in enumerate(orders_for(stage, session_id, warmup=warmup)):
        a_ns = 1_000_000 + index * 1_000
        result.append(
            {
                "block_index": index,
                "order": order,
                "a_ns": a_ns,
                "b_ns": max(1, round(a_ns * ratio)),
            }
        )
    assert len(result) == count
    return result


def session(
    stage: str,
    session_id: str,
    *,
    ratio: float,
    preregistration: dict[str, object],
    seal: dict[str, object] | None = None,
) -> dict[str, object]:
    return build_session_payload(
        stage=stage,
        session_id=session_id,
        preregistration_sha256=preregistration["preregistration_sha256"],
        confirmation_seal_sha256=None if seal is None else seal["confirmation_seal_sha256"],
        provenance_sha256=preregistration["provenance_sha256"],
        power_source="ac_power",
        warmups=blocks(stage, session_id, warmup=True, ratio=ratio),
        measurements=blocks(stage, session_id, warmup=False, ratio=ratio),
        correctness=correctness(),
        budget=budget(),
        resources=resources(),
    )


def calibration_phase() -> tuple[
    dict[str, object],
    list[dict[str, object]],
    dict[str, object],
    dict[str, object],
]:
    preregistration = build_preregistration(PROVENANCE)
    calibration_sessions = [
        session(CALIBRATION, item, ratio=1.0, preregistration=preregistration)
        for item in SESSION_ORDER
    ]
    calibration = build_calibration_summary(calibration_sessions)
    seal = build_confirmation_seal(calibration, calibration_sessions)
    return preregistration, calibration_sessions, calibration, seal


def complete_study(
    confirmation_ratio: float = 0.80,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    dict[str, object],
    dict[str, object],
    list[dict[str, object]],
    dict[str, object],
]:
    preregistration, calibration_sessions, calibration, seal = calibration_phase()
    confirmation_sessions = [
        session(
            CONFIRMATION,
            item,
            ratio=confirmation_ratio,
            preregistration=preregistration,
            seal=seal,
        )
        for item in SESSION_ORDER
    ]
    decision = build_study_decision(confirmation_sessions, seal)
    return preregistration, calibration_sessions, calibration, seal, confirmation_sessions, decision


class ScheduleAndSessionTest(unittest.TestCase):
    def test_all_frozen_seeds_replay_from_the_documented_domain(self) -> None:
        def derive(label: str) -> int:
            digest = hashlib.sha256(
                f"project-friday:h2-n10-v1:{label}".encode("ascii")
            ).digest()
            return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)

        self.assertEqual(FIXTURE_SEED, derive("fixture"))
        self.assertEqual(OPERAND_SEED, derive("operands"))
        for session_id in SESSION_ORDER:
            self.assertEqual(SESSION_SPECS[session_id][2], derive(f"session:{session_id}"))
        bootstrap_labels = {
            "calibration_all": "bootstrap:calibration",
            "confirmation_characterization": "bootstrap:characterization",
            "confirmation_validation": "bootstrap:validation",
            "confirmation_all": "bootstrap:all",
        }
        for key, label in bootstrap_labels.items():
            self.assertEqual(BOOTSTRAP_SEEDS[key], derive(label))

    def test_study_freezes_exactly_one_prior_selected_n10_candidate(self) -> None:
        specification = study_specification()
        self.assertEqual(specification["workload"]["rhs_count"], 10)
        self.assertEqual(
            specification["confirmation"]["candidate_family"],
            ["dispatch_n10_single_terminal_sync"],
        )
        hypothesis = specification["hypothesis"]
        self.assertEqual(hypothesis["selected_candidate"], 10)
        self.assertTrue(hypothesis["model_excluded_from_confirmation"])
        self.assertTrue(hypothesis["fresh_confirmation_data_required"])
        self.assertEqual(
            specification["schedule"]["session_seeds"],
            {
                session_id: SESSION_SPECS[session_id][2]
                for session_id in SESSION_ORDER
            },
        )

    def test_orders_are_balanced_deterministic_and_stage_separated(self) -> None:
        calibration = orders_for(CALIBRATION, "C0", warmup=False)
        self.assertEqual(calibration.count("ab"), MEASUREMENT_BLOCKS // 2)
        self.assertEqual(calibration.count("ba"), MEASUREMENT_BLOCKS // 2)
        self.assertEqual(calibration, orders_for(CALIBRATION, "C0", warmup=False))
        self.assertNotEqual(calibration, orders_for(CONFIRMATION, "C0", warmup=False))

    def test_session_replay_rejects_order_timing_correctness_and_hash_mutations(self) -> None:
        preregistration = build_preregistration(PROVENANCE)
        value = session(CALIBRATION, "C0", ratio=1.0, preregistration=preregistration)
        validate_session(value)
        mutations = []
        reordered = copy.deepcopy(value)
        reordered["measurements"][0]["order"] = (
            "ba" if reordered["measurements"][0]["order"] == "ab" else "ab"
        )
        mutations.append(reordered)
        boolean_duration = copy.deepcopy(value)
        boolean_duration["measurements"][0]["a_ns"] = True
        mutations.append(boolean_duration)
        wrong_output = copy.deepcopy(value)
        wrong_output["correctness"]["b_output_sha256"] = "c" * 64
        mutations.append(wrong_output)
        forged_hash = copy.deepcopy(value)
        forged_hash["session_sha256"] = "d" * 64
        mutations.append(forged_hash)
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with self.assertRaises((ProtocolError, ValueError)):
                    validate_session(mutation)


class TwoStageDecisionTest(unittest.TestCase):
    def test_clean_aa_derives_floor_mde_and_opens_confirmation(self) -> None:
        preregistration, sessions, calibration, seal = calibration_phase()
        self.assertEqual(calibration["status"], "calibration_passed")
        self.assertEqual(calibration["mde"], MINIMUM_EFFECT_FLOOR)
        self.assertEqual(seal["mde"], MINIMUM_EFFECT_FLOOR)
        self.assertEqual(seal["preregistration_sha256"], preregistration["preregistration_sha256"])
        validate_history([preregistration, *sessions, calibration, seal])

    def test_split_confirmation_requires_characterization_and_validation(self) -> None:
        preregistration, calibration_sessions, calibration, seal = calibration_phase()
        # Characterization keeps the real effect; validation becomes a null result.
        mixed = [
            session(
                CONFIRMATION,
                item,
                ratio=0.80 if item.startswith("C") else 1.0,
                preregistration=preregistration,
                seal=seal,
            )
            for item in SESSION_ORDER
        ]
        decision = build_study_decision(mixed, seal)
        self.assertEqual(decision["status"], "n10_inconclusive")
        self.assertFalse(decision["gates"]["gain_all_splits"])
        validate_history([preregistration, *calibration_sessions, calibration, seal, *mixed, decision])

    def test_consistent_twenty_percent_gain_is_formally_confirmed(self) -> None:
        preregistration, calibration_sessions, calibration, seal, sessions, decision = complete_study()
        self.assertEqual(decision["status"], "n10_gain_confirmed")
        self.assertEqual(decision["action"], "permit_bounded_n10_runtime_prototype")
        self.assertTrue(decision["formal_claim"])
        self.assertLess(decision["intervals"]["all"]["ci_high"], 0.95)
        validate_history(
            [preregistration, *calibration_sessions, calibration, seal, *sessions, decision]
        )

    def test_history_forbids_confirmation_before_seal_and_optional_reordering(self) -> None:
        preregistration, calibration_sessions, calibration, seal = calibration_phase()
        sessions = [
            session(
                CONFIRMATION,
                item,
                ratio=0.8,
                preregistration=preregistration,
                seal=seal,
            )
            for item in SESSION_ORDER
        ]
        with self.assertRaises(ProtocolError):
            validate_history([preregistration, sessions[0]])
        with self.assertRaises(ProtocolError):
            validate_history([preregistration, calibration_sessions[1]])
        with self.assertRaises(ProtocolError):
            validate_history([preregistration, *calibration_sessions, seal])
        with self.assertRaises(ProtocolError):
            validate_history([preregistration, *calibration_sessions, calibration_sessions[0]])


if __name__ == "__main__":
    unittest.main()
