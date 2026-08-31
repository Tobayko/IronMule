import ast
import sys
import types
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

# Import the two stdlib-only modules without executing ironmule/__init__.py,
# whose legacy public surface imports the MLX runtime.  Q3 itself must not do so.
_package = types.ModuleType("ironmule")
_package.__path__ = [str(Path(__file__).parents[1].joinpath("ironmule"))]
sys.modules.setdefault("ironmule", _package)

from ironmule.adaptive import (
    AdaptiveContext,
    AdaptiveObservation,
    AdaptiveOutcome,
    AdaptiveValidationError,
    EligibilityStatus,
    KnobAction,
    Method,
    ReplayDataset,
    ReplaySplit,
    RollbackStatus,
    OutcomeStatus,
    method_eligibility,
    next_missing_evidence,
)
from ironmule.evidence import ArtifactRef, EvidenceQuality


DIGEST = "a" * 64


def ref(name="raw/sample.json"):
    return ArtifactRef(name, DIGEST, EvidenceQuality.RAW_SAMPLES)


def context(**updates):
    values = {name + "_digest": DIGEST for name in ("study", "model", "hardware", "framework", "workload", "time")}
    values.update(updates)
    return AdaptiveContext(**values)


def outcome(**updates):
    values = {
        "raw_sample_refs": (ref(),),
        "raw_sample_count": 1,
        "total_ns": 100.0,
        "prefill_ns": 60.0,
        "decode_ns": 40.0,
        "token_identity": True,
        "stop_reason_identity": True,
        "token_count_identity": True,
        "state_identity": True,
        "deterministic": True,
        "mlx_active_memory_bytes": 10,
        "mlx_peak_memory_bytes": 20,
        "rss_peak_bytes": 30,
        "swap_delta_bytes": 0,
        "timeout": False,
        "crash": False,
        "fallbacks": 0,
        "hard_gates_passed": True,
        "status": OutcomeStatus.MEASURED,
    }
    values.update(updates)
    return AdaptiveOutcome(**values)


def observation(**updates):
    values = {
        "context": context(),
        "action": KnobAction.baseline(),
        "measurements": {"total_ns": 100.0},
        "uncertainty": {"total_ns": 1.0},
        "outcome": outcome(),
        "rollback": RollbackStatus.NOT_REQUIRED,
        "evidence": (ref(),),
        "split": ReplaySplit.TRAIN,
        "group_key": "",
    }
    values.update(updates)
    return AdaptiveObservation(**values)


def test_adaptive_module_has_only_stdlib_and_evidence_imports():
    tree = ast.parse(Path(__file__).parents[1].joinpath("ironmule/adaptive.py").read_text())
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    names = {
        (node.module.split(".")[0] if isinstance(node, ast.ImportFrom) and node.module else alias.name.split(".")[0])
        for node in imports
        for alias in node.names
    }
    assert names <= {"__future__", "dataclasses", "enum", "typing", "ironmule", "evidence"}
    assert not any("runtime" in ast.unparse(node) or "tune" in ast.unparse(node) for node in imports)


def test_action_is_closed_defaulted_immutable_and_canonical():
    action = KnobAction()
    assert len(action.to_dict()) == 12  # schema + ten knobs + derived ID
    assert action.key == KnobAction.from_dict(action.to_dict()).key
    assert action.action_id == KnobAction.from_dict(action.to_dict()).action_id
    with pytest.raises(FrozenInstanceError):
        action.readback_every = 2
    with pytest.raises(AdaptiveValidationError):
        KnobAction.from_dict({**action.to_dict(), "unknown": 1})
    with pytest.raises(AdaptiveValidationError):
        KnobAction(wired_fraction=1.1)


def test_records_roundtrip_and_duplicate_or_conflicting_ids_fail():
    item = observation()
    assert AdaptiveObservation.from_dict(item.to_dict()).observation_id == item.observation_id
    with pytest.raises(AdaptiveValidationError):
        ReplayDataset((item, item), action_pool=(item.action,))
    with pytest.raises(AdaptiveValidationError):
        AdaptiveContext.from_dict({**context().to_dict(), "context_id": "b" * 64})


def test_dataset_rejects_cross_split_group_leakage_and_retains_failures():
    failed = observation(
        outcome=outcome(status=OutcomeStatus.FAILED, total_ns=None, crash=True),
    )
    failed = replace(failed, context=context(model_digest="b" * 64), split=ReplaySplit.VALIDATION, group_key="", observation_id="")
    train = observation()
    with pytest.raises(AdaptiveValidationError):
        ReplayDataset((train, replace(train, split=ReplaySplit.VALIDATION, group_key="", observation_id="")), action_pool=(train.action,))
    dataset = ReplayDataset((train, failed), action_pool=(train.action,))
    report = dataset.coverage_report()
    assert report["observation_count"] == 2
    assert report["failures_or_invalids"] == 1
    assert report["splits"]["VALIDATION"] == 1


def test_method_eligibility_is_structural_and_rl_needs_a_horizon():
    report = method_eligibility(ReplayDataset((observation(),), action_pool=(KnobAction.baseline(),)))
    assert report["methods"][Method.BASELINE.value]["status"] == EligibilityStatus.STRUCTURALLY_ELIGIBLE.value
    for method in (Method.BO, Method.SURROGATE, Method.CONTEXTUAL_BANDIT):
        assert report["methods"][method.value]["status"] == EligibilityStatus.DATA_INSUFFICIENT.value
    assert report["methods"][Method.OFFLINE_RL.value]["status"] == EligibilityStatus.NOT_APPLICABLE.value
    assert report["coverage"]["no_invented_performance"] is True


def test_voi_is_a_non_executable_evidence_request():
    request = next_missing_evidence(ReplayDataset((observation(),), action_pool=(KnobAction.baseline(),)))
    assert request["needs"]["cheapest_action_panel"]["need"] == "complete_raw_counterfactual_action_panel"
    assert request["needs"]["cheapest_action_panel"]["executable"] is False
    assert request["no_performance_estimate"] is True


def test_unsafe_measured_outcome_is_not_complete_and_signed_swap_is_allowed():
    unsafe = observation(outcome=outcome(hard_gates_passed=False, swap_delta_bytes=-16))
    report = ReplayDataset((unsafe,), action_pool=(unsafe.action,)).coverage_report()
    assert report["complete_observation_count"] == 0
    assert unsafe.outcome.swap_delta_bytes == -16


@pytest.mark.parametrize("rollback", [RollbackStatus.FAILED, RollbackStatus.NOT_ATTEMPTED])
def test_failed_or_unattempted_rollback_is_never_complete(rollback):
    item = observation(rollback=rollback)
    report = ReplayDataset((item,), action_pool=(item.action,)).coverage_report()
    assert report["complete_observation_count"] == 0


def test_applied_rollback_remains_complete_when_all_other_gates_pass():
    item = observation(rollback=RollbackStatus.APPLIED)
    report = ReplayDataset((item,), action_pool=(item.action,)).coverage_report()
    assert report["complete_observation_count"] == 1


@pytest.mark.parametrize("field", ["raw_sample_refs", "evidence"])
def test_raw_sample_count_alone_cannot_make_observation_complete(field):
    item = observation()
    if field == "raw_sample_refs":
        item = observation(outcome=outcome(raw_sample_refs=()))
    else:
        item = observation(evidence=())
    report = ReplayDataset((item,), action_pool=(item.action,)).coverage_report()
    assert report["complete_observation_count"] == 0


def test_group_key_is_derived_and_duplicate_context_action_fails():
    item = observation()
    assert item.group_key == item.context.context_id
    with pytest.raises(AdaptiveValidationError):
        observation(group_key="forged-group")
    with pytest.raises(AdaptiveValidationError):
        ReplayDataset((item, observation(outcome=outcome(total_ns=101.0))), action_pool=(item.action,))


def test_action_panel_is_explicit_and_incomplete_panel_stays_insufficient():
    item = observation()
    candidate = KnobAction(compiled_fixed_cache=True)
    other = observation(action=candidate, group_key="")
    dataset = ReplayDataset((item, other), action_pool=(KnobAction.baseline(), candidate))
    assert [action.action_id for action in dataset.action_pool] == [KnobAction.baseline().action_id, candidate.action_id]
    assert method_eligibility(dataset)["methods"][Method.CURRENT_COORDINATE.value]["status"] == EligibilityStatus.STRUCTURALLY_ELIGIBLE.value
    incomplete = ReplayDataset((item,), action_pool=(KnobAction.baseline(), candidate))
    assert method_eligibility(incomplete)["methods"][Method.CURRENT_COORDINATE.value]["status"] == EligibilityStatus.DATA_INSUFFICIENT.value
    with pytest.raises(AdaptiveValidationError):
        ReplayDataset((item,), action_pool=())


def test_runtime_knob_schema_and_search_values_remain_the_source_of_truth():
    root = Path(__file__).parents[1]
    runtime = ast.parse(root.joinpath("ironmule/runtime.py").read_text())
    tune = ast.parse(root.joinpath("ironmule/tune.py").read_text())
    knob = next(node for node in runtime.body if isinstance(node, ast.ClassDef) and node.name == "Knobs")
    runtime_fields = [node.target.id for node in knob.body if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)]
    runtime_defaults = [ast.literal_eval(node.value) for node in knob.body if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)]
    search = next(node for node in tune.body if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "SEARCH")
    search_rows = ast.literal_eval(search.value)
    from ironmule.adaptive import KNOB_DEFAULTS, KNOB_NAMES, SEARCH_VALUES
    assert tuple(runtime_fields) == KNOB_NAMES
    assert tuple(zip(runtime_fields, runtime_defaults)) == KNOB_DEFAULTS
    assert SEARCH_VALUES == tuple((name, tuple(values)) for name, values in search_rows)


def test_bandit_requires_all_three_grouped_splits():
    base = observation()
    candidate = KnobAction(compiled_fixed_cache=True)
    rows = []
    for idx, split in enumerate((ReplaySplit.TRAIN, ReplaySplit.VALIDATION)):
        ctx = context(study_digest=chr(ord("a") + idx) * 64)
        rows.extend((observation(context=ctx, action=base.action, split=split, group_key="", observation_id=""), observation(context=ctx, action=candidate, split=split, group_key="", observation_id="")))
    dataset = ReplayDataset(tuple(rows), action_pool=(base.action, candidate))
    assert method_eligibility(dataset)["methods"][Method.CONTEXTUAL_BANDIT.value]["status"] == EligibilityStatus.DATA_INSUFFICIENT.value
