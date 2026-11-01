"""AG1 S: the online-learning simulation harness runs the product's agent end to end (a smoke, not S itself)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import ag1_online  # noqa: E402


def test_the_harness_learns_the_machine_and_compares_every_window_with_the_reference(tmp_path, monkeypatch):
    monkeypatch.setattr(ag1_online.ag, "MIN_GROUPS", 4)
    monkeypatch.setattr(ag1_online.ag, "MIN_ARRIVALS", 16)
    result = ag1_online.run(traces=12, episodes=40, root=tmp_path, log=lambda line: None)
    assert len(result["windows"]) == 12
    assert all(w["agent_cost"] > 0 and w["reference_cost"] > 0 for w in result["windows"])
    fitted, truth = result["final"]["world_model"], ag1_online.MACHINES["t4"]
    for key in ("step_s", "step_s_per_session", "prefill_s_per_ktoken"):
        assert abs(fitted[key] / truth[key] - 1) < 0.25, key
    assert result["counters"]["groups"] > 12 and result["counters"]["plans"] >= 1
    summary = ag1_online.summarise(result)
    assert sum(phase["windows"] for phase in summary.values()) == 12
