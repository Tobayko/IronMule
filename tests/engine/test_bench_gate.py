"""R10/R11: the memory gate records why it stopped, and an aborted file is refused."""

from __future__ import annotations

import pytest

from ironmule import bench

GIB = 1024**3


def test_a_peak_above_the_ceiling_is_recorded_as_an_abort():
    gate = bench.MemoryGate(peak_ceiling=10 * GIB, read_swap=lambda: 0)
    assert gate.check(0, 4 * GIB) is None
    reason = gate.check(1, 17 * GIB)
    assert reason and "exceeds the backstop" in reason
    assert gate.record["aborted"] == {"block": 1, "reason": reason}
    assert [block["aborted_here"] for block in gate.record["blocks"]] == [False, True]


def test_swap_growth_stops_the_run_whatever_the_peak():
    swap = iter((0, 300 * 1024**2))
    gate = bench.MemoryGate(peak_ceiling=None, read_swap=lambda: next(swap),
                            read_installed=lambda: None)
    reason = gate.check(0, 1)
    assert reason and "swap grew" in reason
    assert gate.record["aborted"]["block"] == 0


def test_an_aborted_file_is_refused_unless_accepted():
    gate = bench.MemoryGate(peak_ceiling=1, read_swap=lambda: 0)
    gate.check(0, 2)
    payload = {"runs": [{}], "memory_gate": gate.record}
    with pytest.raises(ValueError, match="run aborted at block 0"):
        bench.refuse_aborted(payload)
    bench.refuse_aborted(payload, accept_abort=True)


def test_files_without_a_gate_record_and_complete_runs_pass():
    bench.refuse_aborted({"runs": []})
    gate = bench.MemoryGate(peak_ceiling=10 * GIB, read_swap=lambda: 0)
    gate.check(0, GIB)
    bench.refuse_aborted({"runs": [{}], "memory_gate": gate.record})
