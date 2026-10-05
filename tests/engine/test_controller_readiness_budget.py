"""Readiness admission deadline edges; fake clocks and existing scripted callbacks."""
from __future__ import annotations

import copy

import pytest

from tests.engine.test_online_controller_harness import readiness_setup
from tools import online_controller_eval as harness


@pytest.mark.parametrize("budget", ["readiness", "global"])
def test_final_snapshot_overrun_is_persisted_and_never_admitted(monkeypatch, tmp_path, budget):
    _, runtime, controller, spec, calls = readiness_setup(monkeypatch, tmp_path)
    clock = [0.0]
    monkeypatch.setattr(harness.time, "perf_counter", lambda: clock[0])
    original = controller.snapshot
    snapshots = 0
    def snapshot():
        nonlocal snapshots
        snapshots += 1
        if snapshots == 2:
            clock[0] = 181.0 if budget == "readiness" else 101.0
        return original()
    monkeypatch.setattr(controller, "snapshot", snapshot)
    def global_budget():
        if budget == "global" and clock[0] >= 100:
            raise TimeoutError("outer global deadline exceeded")
    record, persisted = {}, []
    with pytest.raises(TimeoutError, match="wall budget|global deadline"):
        harness.run_readiness(runtime, controller, spec, phase="preparation",
            startup_started=0.0, within_budget=global_budget, record=record,
            save=lambda: persisted.append(copy.deepcopy(record)))
    assert len(calls) == len(record["raw"]) == 22
    assert record["status"] == persisted[-1]["status"] == "failed"
    assert record["failure_type"] == "TimeoutError"
    assert record["no_learning"] is True
    assert all(row["status"] == "completed" for row in record["raw"])


def test_final_save_crossing_startup_budget_fails_admission(monkeypatch, tmp_path):
    _, runtime, controller, spec, calls = readiness_setup(monkeypatch, tmp_path)
    clock = [0.0]
    monkeypatch.setattr(harness.time, "perf_counter", lambda: clock[0])
    record, persisted = {}, []
    def save():
        if record["status"] == "ready":
            clock[0] = 180.25
        persisted.append(copy.deepcopy(record))
    with pytest.raises(TimeoutError, match="startup wall budget"):
        harness.run_readiness(runtime, controller, spec, phase="preparation",
            startup_started=0.0, within_budget=lambda: None, record=record, save=save)
    assert len(calls) == len(record["raw"]) == 22
    assert persisted[-1]["status"] == record["status"] == "failed"
    assert record["setup_elapsed_s"] == 180.25
    assert record["no_learning"] is True


def test_one_synchronous_call_soft_overrun_is_recorded_without_more_attempts(monkeypatch, tmp_path):
    _, runtime, controller, spec, calls = readiness_setup(monkeypatch, tmp_path)
    clock = [0.0]
    monkeypatch.setattr(harness.time, "perf_counter", lambda: clock[0])
    original = harness.execute_registered_profile
    def blocking(*args):
        rows, _, digest = original(*args)
        clock[0] = 181.0
        return rows, 181.0, digest
    monkeypatch.setattr(harness, "execute_registered_profile", blocking)
    record, persisted = {}, []
    with pytest.raises(TimeoutError, match="startup wall budget"):
        harness.run_readiness(runtime, controller, spec, phase="preparation",
            startup_started=0.0, within_budget=lambda: None, record=record,
            save=lambda: persisted.append(copy.deepcopy(record)))
    assert len(calls) == len(record["raw"]) == 1
    assert record["raw"][0]["status"] == "completed"
    assert record["raw"][0]["elapsed_s"] == 181.0
    assert record["execution_wall_s"] == 181.0
    assert record["status"] == persisted[-1]["status"] == "failed"
    assert record["no_learning"] is True
    assert "soft acceptance" in record["budget_enforcement"]
