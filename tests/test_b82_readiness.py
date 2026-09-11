"""The cheap check in front of the expensive one, and what it is not allowed to be.

A readiness probe is a small idea with one large temptation: to become a predictor of which
action wins, or a knob that makes a `PASS` more likely. These check that it does neither. It
runs the reference against itself, it borrows every limit from the comparison it guards, and a
failure stops the spending rather than starting a retry.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from ironmule import readiness as readiness_module
from ironmule import requalification as requalification_module
from ironmule.readiness import (BLOCKS, CHILDREN_PER_BLOCK, PROTOCOL, Readiness,
                                evaluate_recorded_aa)
from ironmule.requalification import AA_MAX_HALF_WIDTH, AA_MAX_OFFSET, DRIFT_LIMIT

SOURCE = Path(readiness_module.__file__)


def _code(path: Path = SOURCE) -> str:
    """Everything after the module docstring.

    Cutting at the *last* statement, which is what earlier tests in this session did, keeps
    only the final function and silently passes checks the rest of the file would fail. The
    first statement after the docstring is the right boundary.
    """
    text = path.read_text()
    tree = ast.parse(text)
    rest = tree.body[1:] if ast.get_docstring(tree) else tree.body
    if not rest:
        return ""
    segment = ast.get_source_segment(text, rest[0]) or ""
    return text[text.index(segment):] if segment else text


# --------------------------------------------------------------------------- what it is not


def test_the_probe_never_runs_the_candidate() -> None:
    """Structural: every child it starts is told to be the reference."""
    code = _code()
    assert '"arm": "candidate"' not in code
    assert "install(" not in code and "qmv_variant" not in code
    assert code.count('"arm": "reference"') == 1, (
        "there is one child spec and it is the reference")


def test_it_invents_no_threshold_of_its_own() -> None:
    """Every limit is the comparison's. A bar set from the runs it filters is not a bar."""
    assert PROTOCOL["limits"]["aa_max_offset"] is AA_MAX_OFFSET
    assert PROTOCOL["limits"]["aa_max_half_width"] is AA_MAX_HALF_WIDTH
    assert PROTOCOL["limits"]["drift_limit"] is DRIFT_LIMIT
    tree = ast.parse(SOURCE.read_text())
    assigned = {target.id for node in tree.body if isinstance(node, ast.Assign)
                for target in node.targets if isinstance(target, ast.Name)}
    for forbidden in ("AA_MAX_OFFSET", "AA_MAX_HALF_WIDTH", "DRIFT_LIMIT"):
        assert forbidden not in assigned, f"{forbidden} must come from requalification"


def test_it_shares_the_workload_and_lifecycle_with_the_real_run() -> None:
    assert PROTOCOL["repeats"] == requalification_module.REPEATS
    assert PROTOCOL["warmups"] == requalification_module.WARMUPS
    assert str(requalification_module.MAX_NEW_TOKENS) in PROTOCOL["workload"]
    assert "one model per process" in PROTOCOL["lifecycle"]


def test_it_is_cheaper_than_what_it_guards() -> None:
    guarded = requalification_module.SESSIONS * requalification_module.BLOCKS * 3
    assert BLOCKS * CHILDREN_PER_BLOCK < guarded / 2


def test_it_promises_nothing_about_the_outcome() -> None:
    assert "not_a_prediction" in PROTOCOL
    assert "never runs the candidate" in PROTOCOL["not_a_prediction"]
    assert "not_a_guarantee" in PROTOCOL


def test_there_is_no_retry_and_no_waiting() -> None:
    """Checked on the syntax tree, not on the text.

    Searching the source for the word `retry` finds the paragraph promising there is none,
    which is how a check ends up measuring prose. A loop is a `While` node and a wait is a
    call to `sleep`; those are what to look for.
    """
    tree = ast.parse(SOURCE.read_text())
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.While)], (
        "a readiness probe with a loop in it is a waiting room")
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    names = {getattr(node.func, "attr", None) or getattr(node.func, "id", None)
             for node in calls}
    assert "sleep" not in names, "nothing here waits for a better moment"


# --------------------------------------------------------------------------- the rule


@pytest.mark.parametrize("median_ratio,half_width,ready", [
    (1.0000, 0.0069, True),
    (0.9988, 0.0069, True),
    (1.0067, 0.0095, True),
    (0.9673, 0.0164, True),
    (0.8508, 0.2297, False),
    (0.9228, 0.1996, False),
    (1.0829, 0.0374, False),
    (1.0000, 0.0501, False),
    (0.9499, 0.0010, False),
])
def test_the_rule_reads_a_control_the_way_the_real_run_will(median_ratio, half_width, ready) -> None:
    assert evaluate_recorded_aa(median_ratio, half_width)[0] is ready


def test_a_missing_half_width_is_not_ready() -> None:
    assert evaluate_recorded_aa(1.0, None)[0] is False


def test_a_refusal_always_says_which_limit_it_was() -> None:
    assert "half width" in evaluate_recorded_aa(1.0, 0.3)[1]
    assert "further than" in evaluate_recorded_aa(1.2, 0.001)[1]


# --------------------------------------------------------------------------- the wiring


def test_a_readiness_failure_spends_nothing(monkeypatch, tmp_path: Path) -> None:
    """The comparison must not start, and nothing may retry inside one call."""
    import json

    from ironmule.local_learner import LocalLearner, VALID, evidence_from
    from ironmule.monitoring import requalification_path
    from ironmule.qmv_variant import QUALIFIED_ACTION_ID

    context = requalification_module.IntakeContext(
        "fp", "applegpu_g13s", "0.32.0", "0.31.3", "model", "sha", "rev")
    learner = LocalLearner(context)
    for index in range(6):
        ratio = 0.955 + 0.002 * index
        learner.observe(evidence_from({
            "evidence_id": f"e{index}", "action_id": QUALIFIED_ACTION_ID,
            "workload_class": "single_short", "reference_stack": "A",
            "hardware_fingerprint": "fp", "gpu_architecture": "applegpu_g13s",
            "mlx": "0.32.0", "mlx_lm": "0.31.3", "model_id": "model",
            "model_identity_sha256": "sha", "model_revision": "rev",
            "ratio": ratio, "ci_low": ratio - 0.01, "ci_high": ratio + 0.01,
            "within_session_se": 0.005, "aa_median": 1.0, "aa_half_width": 0.005,
            "correctness_passed": True, "resource_gate_passed": True, "status": VALID,
            "measured_at": f"2026-09-11T{index:02d}:00:00+00:00"}))
    state = learner.save(tmp_path / "local_learning.json")
    requalification_path(state).write_text(json.dumps({"reason": "drift"}), encoding="utf-8")

    started = []
    monkeypatch.setattr(requalification_module, "run_comparison",
                        lambda *a, **k: started.append(True))
    monkeypatch.setattr(requalification_module, "resources_available",
                        lambda *a, **k: (True, "", {}))
    monkeypatch.setattr(requalification_module, "check_preconditions",
                        lambda *a, **k: requalification_module.Preconditions(
                            True, "allowed", {}))

    class _Tune:
        DEFAULT_MODEL = "model"

        @staticmethod
        def gpu_busy():
            return None

        @staticmethod
        def resolve_local_model(model_id):
            from types import SimpleNamespace

            return SimpleNamespace(identity=SimpleNamespace(
                identity_sha256="sha", revision="rev"))

    monkeypatch.setitem(__import__("sys").modules, "ironmule.tune", _Tune)
    not_ready = {"ready": False, "reason": "the A/A half width is 0.3 against a limit of 0.05"}
    record = requalification_module.requalify(
        "model", state_path=state, readiness=not_ready)
    assert record["outcome"] == "NOT_READY"
    assert record["applied"] is False
    assert record["state"] == "REQUALIFICATION_REQUIRED"
    assert not started, "no comparison may start"
    assert requalification_path(state).exists()


def test_an_already_measured_probe_is_not_paid_for_twice() -> None:
    signature = inspect.signature(requalification_module.requalify)
    assert "readiness" in signature.parameters
    assert signature.parameters["readiness"].default is True
    source = inspect.getsource(requalification_module.requalify)
    assert "isinstance(readiness, Mapping)" in source


def test_skipping_readiness_is_possible_and_is_not_the_default() -> None:
    import ironmule_cli

    source = inspect.getsource(ironmule_cli._run_requalify)
    assert "--skip-readiness" in source
    assert inspect.signature(
        requalification_module.requalify).parameters["readiness"].default is True


def test_not_ready_is_a_result_and_not_an_error() -> None:
    assert "NOT_READY" in requalification_module.OUTCOMES
    probe = Readiness(False, "too noisy", {"wall_seconds": 1.0})
    assert probe.as_dict()["ready"] is False
    assert probe.as_dict()["reason"] == "too noisy"
