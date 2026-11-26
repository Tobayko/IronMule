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
    history = {"matched_microseconds": [84, 107], "prefill_history": {"optimistic_best_sum_ms": 24}}
    cost = {"quoted_compute_usd_per_h": 2, "disk_rate_estimate_usd_per_h": 0.02}
    eligible = rank(data, historic, history, cost)
    assert eligible["next"]["selected_phase"] == "measure_passes"
    assert eligible["next"]["book_additional_resources"] is False
    assert eligible["next"]["estimated_equivalent_lease_usd"] == 36 * 2.02 / 3600
    assert len(eligible["retired"]) == 4
    deferred = rank({**data, "http_cold_s": [50]}, historic, history, cost)
    assert deferred["next"]["selected_phase"] is None
