"""PROD10 metadata and genuine OS-cleanup tests, never model simulations."""

from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import product_open_validation as screen
from friday_evidence.open_observation import OpenObservation


def test_removed_hardware_gates_are_explicit_nulls():
    assert screen.POLICY["required_pause_s"] == 0
    assert screen.POLICY["host_readiness_gate"] is False
    for key in ("request_timeout_s", "startup_timeout_s", "work_budget_s",
                "continuous_budget_s", "duty_cycle_limit", "rss_stop_bytes", "swap_stop_bytes"):
        assert screen.POLICY[key] is None


def test_case_definition_stays_fixed_and_finite():
    cases = screen.cases()
    assert [(c["name"], c["max_tokens"]) for c in cases] == [("long_8", 8), ("short_32", 32), ("long_32", 32)]
    assert cases[0]["messages"] == cases[2]["messages"]
    assert cases[0]["messages"][0]["content"].endswith("END-OF-PUBLIC-ORCHARD-NOTE.")


def test_endurance_schedule_has_predeclared_periodic_four_client_bursts():
    assert screen.soak_batch_indices(0) == [0]
    assert screen.soak_batch_indices(11) == [11]
    assert screen.soak_batch_indices(12) == [12, 13, 14, 15]
    assert screen.soak_batch_indices(16) == [16]
    assert screen.soak_batch_indices(24) == [24, 25, 26, 27]


def test_output_digest_binds_tokens_text_finish_and_counts():
    done = {"prompt_tokens": 17, "completion_tokens": 2, "finish_reason": "length"}
    first = screen.output_metadata([1, 2], "public fixture", done)
    assert first["output_sha256"] != screen.output_metadata([1, 3], "public fixture", done)["output_sha256"]
    with pytest.raises(screen.ValidationFailure, match="output_count_invalid"):
        screen.output_metadata([1], "public fixture", done)


def test_reference_cleanup_reaps_real_child_even_after_control_pipe_closes():
    # A real harmless OS child exercises broken-pipe cleanup. It has no model,
    # fabricated device handshake or inference output.
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    reference = screen.ReferenceProcess.__new__(screen.ReferenceProcess)
    reference.process = process
    reference.observer = OpenObservation()
    process.stdin.close()
    try:
        with pytest.raises(screen.ValidationFailure, match="reference_shutdown_failed"):
            reference.close()
        assert process.poll() is not None
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
