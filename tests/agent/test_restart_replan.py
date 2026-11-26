from training.restart_replan import rank


def test_complete_product_work_and_measured_history_bound_information_probe():
    data = {
        "http_cold_s": [13, 14],
        "http_mixed_warm_s": [3.2, 3.4],
        "cpu_decision_ms": [3, 4],
        "native_init_s": [11, 12],
        "http_request_to_worker_ready_s": [12, 13],
        "native_factual_worker_elapsed_s": [35, 36],
    }
    historic = {
        "measure_kg": [18, 19],
        "measure_pk2": [18, 18],
        "measure_passes": [3.6, 3.7],
        "measure_padding": [2, 2],
    }
    history = {
        "matched_microseconds": [84, 107],
        "prefill_history": {"optimistic_best_sum_ms": 24},
    }
    cost = {"quoted_compute_usd_per_h": 2, "disk_rate_estimate_usd_per_h": 0.02}
    eligible = rank(data, historic, history, cost)
    assert eligible["next"]["selected_phase"] == "measure_passes"
    assert eligible["next"]["book_additional_resources"] is False
    assert eligible["next"]["estimated_equivalent_lease_usd"] == 36 * 2.02 / 3600
    assert len(eligible["retired"]) == 4
    deferred = rank({**data, "http_cold_s": [50]}, historic, history, cost)
    assert deferred["next"]["selected_phase"] is None


def test_completed_source_attribution_retains_the_timing_exclusion(tmp_path):
    import json
    import shutil
    from pathlib import Path

    import pytest

    from training.restart_replan import checksum, completed_phase_probe

    results = Path(__file__).parents[2] / "lab/autonomous/results"
    folder = results / "a100-post-restart-native-60"
    verdict = json.loads((folder / "decision.json").read_text())
    released = results / "a100-restart" / verdict["parent_order"]
    receipt = json.loads((released / "control.json").read_text())
    head = verdict["weights_sha256"]
    pin = checksum(released / "control.json")
    proof = completed_phase_probe(results, receipt["order"], receipt, head, pin)
    assert proof["timing_eligible"] is proof["training_eligible"] is False
    assert proof["closed_sampling_experiment"]["regular_pairs"] == 8
    assert proof["closed_sampling_experiment"]["full_fallback"] == 1
    assert len(proof["observed_cells"]) == len(proof["observation_sha256"]) == 2
    copy = tmp_path / folder.name
    shutil.copytree(folder, copy)
    (copy / "product-nested-diagnostic.json").write_text("{}")
    with pytest.raises(ValueError, match="conflict"):
        completed_phase_probe(tmp_path, receipt["order"], receipt, head, pin)
