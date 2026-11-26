from pathlib import Path

from training.replay_qualified_restart import check


def test_archived_source_order_trained_dispatch_and_restarted_reference_agree():
    root = Path(__file__).resolve().parents[2]
    order = "d938522e9c93ef05d746bb030631675ccd3c76b3c8292be7f0ad6dba5b3c31b3"
    report = check(root / "lab/autonomous/results/a100-restart" / order)
    assert report["github_pr"] == 60
    assert report["measured_policy_effect"]["same_work_policy_changed_source"]
    assert report["explicit_rollback_preserved_correct_work"]
    assert not report["automatic_withdrawal"]["first_reference_attempt_served"]
    reference = report["automatic_withdrawal"]["fresh_process"]
    assert reference["arm"] == "reference"
    assert (reference["cold_performance_completions"], reference["warm_performance_completions"],
            reference["graded_correct"]) == (32, 32, 18)
