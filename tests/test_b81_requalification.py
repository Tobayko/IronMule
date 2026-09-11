"""The only way back, and every way it must refuse to be one.

`B80` can take a qualified action away and cannot give it back. `B81` is the thing that can,
and the whole risk is that it becomes easy: a recovery path reachable by accident is worse
than no recovery path. These check that it starts only when the state requires it, that it
decides on its own numbers, and that every failure leaves the reference exactly where it was.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from ironmule import requalification as requalification_module
from ironmule.local_learner import (CANDIDATE_QUALIFIED, IntakeContext, LocalLearner,
                                    REFERENCE_ONLY, UNKNOWN, VALID, evidence_from)
from ironmule.monitoring import (BASELINE_MIN, DriftMonitor, REQUALIFICATION_REQUIRED, WINDOW,
                                 observation_from, requalification_path)
from ironmule.qmv_variant import QUALIFIED_ACTION_ID
from ironmule.requalification import (BLOCKS, SESSIONS, apply_outcome, check_preconditions,
                                      decide, lineage_path, requalify)

CLASS = "single_short"
CONTEXT = IntakeContext("fp", "applegpu_g13s", "0.32.0", "0.31.3", "model", "sha", "rev")


def evidence(index: int, ratio: float) -> dict:
    return {"evidence_id": f"old_{index}", "action_id": QUALIFIED_ACTION_ID,
            "workload_class": CLASS, "reference_stack": "A", "hardware_fingerprint": "fp",
            "gpu_architecture": "applegpu_g13s", "mlx": "0.32.0", "mlx_lm": "0.31.3",
            "model_id": "model", "model_identity_sha256": "sha", "model_revision": "rev",
            "ratio": ratio, "ci_low": ratio - 0.01, "ci_high": ratio + 0.01,
            "within_session_se": 0.005, "aa_median": 1.0, "aa_half_width": 0.005,
            "correctness_passed": True, "resource_gate_passed": True, "status": VALID,
            "measured_at": f"2026-09-11T{index:02d}:00:00+00:00"}


def qualified_state(tmp_path: Path) -> Path:
    learner = LocalLearner(CONTEXT)
    learner.observe_all([evidence_from(evidence(i, 0.955 + 0.002 * i)) for i in range(6)])
    assert learner.recommendation(QUALIFIED_ACTION_ID, CLASS).local_learning_state \
        == CANDIDATE_QUALIFIED
    return learner.save(tmp_path / "local_learning.json")


def session(ratio: float, half: float = 0.004, aa_ok: bool = True, **overrides) -> dict:
    row = {"session": 0, "complete_blocks": BLOCKS,
           "candidate": {"median": ratio, "ci_low": ratio - half, "ci_high": ratio + half,
                         "n": BLOCKS},
           "reference_aa": {"median": 1.0, "ci_low": 0.996, "ci_high": 1.004, "n": BLOCKS},
           "aa_half_width": 0.004 if aa_ok else 0.2, "aa_gate_passed": aa_ok,
           "disturbed_blocks": [], "within_session_se": 0.003,
           "fallbacks": 0, "correctness_errors": 0}
    row.update(overrides)
    return row


def comparison(ratio: float = 0.96, **overrides) -> dict:
    row = {"child_failures": [], "correctness_identical": True,
           "resource_gate_passed": True, "resource_gate_reasons": [],
           "children_run": SESSIONS * BLOCKS * 3, "wall_seconds": 700.0,
           "sessions": [{**session(ratio), "session": index} for index in range(SESSIONS)]}
    row.update(overrides)
    return row


# --------------------------------------------------------------------------- the state rule


def test_nothing_may_run_while_nothing_requires_it(tmp_path: Path) -> None:
    state = qualified_state(tmp_path)
    learner = LocalLearner.restore(state, CONTEXT)
    outcome = check_preconditions(state, CONTEXT, learner)
    assert outcome.allowed is False
    assert "nothing requires a requalification" in outcome.reason


def test_a_requalification_record_is_what_opens_the_door(tmp_path: Path) -> None:
    state = qualified_state(tmp_path)
    requalification_path(state).write_text(json.dumps({"reason": "drift"}), encoding="utf-8")
    learner = LocalLearner.restore(state, CONTEXT)
    assert check_preconditions(state, CONTEXT, learner).allowed is True


def test_ordinary_observations_never_clear_the_state(tmp_path: Path) -> None:
    state = tmp_path / "local_learning.json"
    monitor = DriftMonitor(state, min_material_shift=0.0226)
    raw = {"observed_at": "2026-09-11T00:00:00+00:00", "hardware_fingerprint": "fp",
           "gpu_architecture": "applegpu_g13s", "model_identity_sha256": "sha",
           "model_revision": "rev", "quantization_bits": 4, "quantization_group_size": 64,
           "mlx": "0.32.0", "mlx_lm": "0.31.3", "workload_class": CLASS,
           "action_id": QUALIFIED_ACTION_ID, "action_code_digest": "digest",
           "effective_action": "candidate", "end_to_end_ms": 921.0, "service_ttft_ms": 20.0,
           "tokens_per_second": 33.0, "new_tokens": 32, "prompt_tokens": 27, "fallbacks": 0,
           "correctness_errors": 0, "memory_pressure_level": 1, "swap_used_bytes": 0,
           "controller_digest": "c"}
    for index in range(BASELINE_MIN):
        monitor.observe(observation_from({**raw, "end_to_end_ms": 921.0 + index % 4}))
    for index in range(WINDOW):
        monitor.observe(observation_from({**raw, "end_to_end_ms": 921.0 * 1.25}))
    assert monitor.requalification_required
    for index in range(500):
        monitor.observe(observation_from({**raw, "end_to_end_ms": 600.0}))
    assert monitor.requalification_required
    assert requalification_path(state).exists()
    assert DriftMonitor(state).requalification_required


@pytest.mark.parametrize("field,value", [
    ("model_revision", "other"), ("mlx", "0.33.0"), ("mlx_lm", "0.32.0"),
    ("hardware_fingerprint", "other"), ("model_identity_sha256", "other"),
])
def test_a_changed_machine_or_build_refuses_to_requalify(tmp_path: Path, field, value) -> None:
    state = qualified_state(tmp_path)
    requalification_path(state).write_text(json.dumps({"reason": "drift"}), encoding="utf-8")
    moved = replace(CONTEXT, **{field: value})
    learner = LocalLearner.restore(state, moved)
    outcome = check_preconditions(state, moved, learner)
    assert outcome.allowed is False


def test_a_kill_record_is_not_cleared_by_a_measurement(tmp_path: Path) -> None:
    from ironmule.activation import write_kill

    state = qualified_state(tmp_path)
    requalification_path(state).write_text(json.dumps({"reason": "drift"}), encoding="utf-8")
    write_kill(state, "a fallback in the candidate path")
    learner = LocalLearner.restore(state, CONTEXT)
    outcome = check_preconditions(state, CONTEXT, learner)
    assert outcome.allowed is False
    assert "safety failure" in outcome.reason


def test_a_corrupt_state_leaves_nothing_to_requalify(tmp_path: Path) -> None:
    state = tmp_path / "local_learning.json"
    state.write_text("{not json", encoding="utf-8")
    requalification_path(state).write_text(json.dumps({"reason": "drift"}), encoding="utf-8")
    learner = LocalLearner.restore(state, CONTEXT)
    outcome = check_preconditions(state, CONTEXT, learner)
    assert outcome.allowed is False
    assert "never qualified" in outcome.reason


# --------------------------------------------------------------------------- the decision


def test_a_clean_winning_run_passes() -> None:
    assert decide(comparison(0.96))["outcome"] == "PASS"


@pytest.mark.parametrize("broken,fragment", [
    ({"correctness_identical": False}, "tokens or stop reasons"),
    ({"resource_gate_passed": False, "resource_gate_reasons": ["swap rose"]}, "swap rose"),
    ({"child_failures": [{"arm": "candidate"}]}, "a child failed"),
])
def test_a_run_that_fails_its_own_gates_is_invalid(broken, fragment) -> None:
    outcome = decide(comparison(0.96, **broken))
    assert outcome["outcome"] == "INVALID"
    assert fragment in outcome["reason"]
    assert outcome["requalification_stays_required"] is True


@pytest.mark.parametrize("field,value,fragment", [
    ("fallbacks", 1, "fallback"),
    ("correctness_errors", 1, "correctness error"),
    ("disturbed_blocks", [1], "drift gate"),
    ("complete_blocks", 1, "session and block"),
])
def test_a_session_defect_is_invalid(field, value, fragment) -> None:
    rows = [{**session(0.96), "session": i} for i in range(SESSIONS)]
    rows[1][field] = value
    outcome = decide(comparison(0.96, sessions=rows))
    assert outcome["outcome"] == "INVALID" and fragment in outcome["reason"]


def test_a_noisy_control_cannot_restore_anything() -> None:
    rows = [{**session(0.90, aa_ok=False), "session": i} for i in range(SESSIONS)]
    outcome = decide(comparison(0.90, sessions=rows))
    assert outcome["outcome"] == "INVALID"
    assert "A/A control" in outcome["reason"]
    assert outcome["requalification_stays_required"] is True


def test_an_interval_that_touches_one_is_no_gain() -> None:
    rows = [{**session(0.99, half=0.03), "session": i} for i in range(SESSIONS)]
    assert decide(comparison(sessions=rows))["outcome"] == "NO_GAIN"


def test_a_candidate_that_lost_is_worse() -> None:
    rows = [{**session(1.08), "session": i} for i in range(SESSIONS)]
    assert decide(comparison(sessions=rows))["outcome"] == "WORSE"


def test_one_bad_session_is_enough_to_withhold_a_pass() -> None:
    rows = [{**session(0.96), "session": i} for i in range(SESSIONS)]
    rows[2] = {**session(0.99, half=0.03), "session": 2}
    assert decide(comparison(sessions=rows))["outcome"] == "NO_GAIN"


def test_no_historical_ratio_appears_in_the_decision() -> None:
    import inspect

    source = inspect.getsource(requalification_module)
    for number in ("0.8469", "0.9624", "0.9648", "0.9647"):
        assert number not in source, f"{number} is history and must not decide a rerun"


# --------------------------------------------------------------------------- the transition


def test_a_pass_restores_the_candidate_on_new_evidence(tmp_path: Path) -> None:
    state = qualified_state(tmp_path)
    requalification_path(state).write_text(json.dumps({"reason": "drift"}), encoding="utf-8")
    learner = LocalLearner.restore(state, CONTEXT)
    run = comparison(0.96)
    applied = apply_outcome(state, CONTEXT, learner, run, decide(run))
    assert applied["state"] == CANDIDATE_QUALIFIED
    assert applied["requalification_record_cleared"] is True
    assert not requalification_path(state).exists()
    assert len(applied["new_evidence_ids"]) == SESSIONS
    restored = LocalLearner.restore(state, CONTEXT)
    stored_ids = restored.as_dict()["actions"][0]["evidence_ids"]
    assert all(name.startswith("B81_requalification_") for name in stored_ids)


def test_the_previous_epoch_is_archived_and_never_deleted(tmp_path: Path) -> None:
    state = qualified_state(tmp_path)
    before = state.read_bytes()
    requalification_path(state).write_text(json.dumps({"reason": "drift"}), encoding="utf-8")
    learner = LocalLearner.restore(state, CONTEXT)
    run = comparison(0.96)
    applied = apply_outcome(state, CONTEXT, learner, run, decide(run))
    archived = Path(applied["archived_epoch"])
    assert archived.exists() and archived.read_bytes() == before
    assert "old_0" in archived.read_text()


def test_a_no_gain_run_leaves_the_reference_and_clears_the_flag(tmp_path: Path) -> None:
    state = qualified_state(tmp_path)
    requalification_path(state).write_text(json.dumps({"reason": "drift"}), encoding="utf-8")
    learner = LocalLearner.restore(state, CONTEXT)
    rows = [{**session(0.99, half=0.03), "session": i} for i in range(SESSIONS)]
    run = comparison(sessions=rows)
    applied = apply_outcome(state, CONTEXT, learner, run, decide(run))
    assert applied["state"] == REFERENCE_ONLY
    assert LocalLearner.restore(state, CONTEXT).recommendation(
        QUALIFIED_ACTION_ID, CLASS).recommended_action == "reference"


def test_an_invalid_run_changes_nothing_at_all(tmp_path: Path) -> None:
    state = qualified_state(tmp_path)
    required = requalification_path(state)
    required.write_text(json.dumps({"reason": "drift"}), encoding="utf-8")
    learner = LocalLearner.restore(state, CONTEXT)
    run = comparison(0.96, correctness_identical=False)
    applied = apply_outcome(state, CONTEXT, learner, run, decide(run))
    assert applied["applied"] is False
    assert applied["state"] == REQUALIFICATION_REQUIRED
    assert required.exists(), "the requalification stays required"
    assert LocalLearner.restore(state, CONTEXT).recommendation(
        QUALIFIED_ACTION_ID, CLASS).local_learning_state == CANDIDATE_QUALIFIED


def test_the_lineage_reads_as_a_story(tmp_path: Path) -> None:
    state = qualified_state(tmp_path)
    requalification_path(state).write_text(json.dumps({"reason": "drift"}), encoding="utf-8")
    learner = LocalLearner.restore(state, CONTEXT)
    run = comparison(0.96)
    apply_outcome(state, CONTEXT, learner, run, decide(run))
    events = json.loads(lineage_path(state).read_text())["events"]
    assert [event["event"] for event in events] == ["requalification_pass"]
    assert events[0]["resulting_state"] == CANDIDATE_QUALIFIED
    assert events[0]["archived_epoch"]


# --------------------------------------------------------------------------- the command


def test_the_command_refuses_without_a_requalification(tmp_path: Path) -> None:
    record = requalify(state_path=tmp_path / "absent.json")
    assert record["outcome"] == "REFUSED"
    assert record["applied"] is False
    assert record["state"] == REQUALIFICATION_REQUIRED


def test_the_cli_knows_the_command(capsys) -> None:
    import ironmule_cli

    assert hasattr(ironmule_cli, "_run_requalify")
    ironmule_cli.main(["--help"])
    assert "requalify" in capsys.readouterr().out


def test_the_runtime_default_is_still_off() -> None:
    import inspect

    from ironmule.router import AppleRuntime

    assert inspect.signature(AppleRuntime.load).parameters[
        "enable_local_learned_dispatch"].default is False
