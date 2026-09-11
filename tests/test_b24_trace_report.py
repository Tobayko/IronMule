from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from b24_trace_report import _measured_phase, _rows  # noqa: E402

XML = """<?xml version="1.0"?>
<trace-query-result>
<node xpath='//trace-toc[1]/run[1]/data[1]/table[1]'>
<schema name="metal-application-command-buffer-submissions">
<col><mnemonic>start</mnemonic></col>
<col><mnemonic>num-encoders</mnemonic></col>
<col><mnemonic>process</mnemonic></col>
</schema>
<row><start-time id="1">1000</start-time><uint32 id="2">7</uint32>
<process id="3" fmt="python (99)"><pid id="4">99</pid></process></row>
<row><start-time id="5">2000</start-time><uint32 ref="2"/><process ref="3"/></row>
</node>
</trace-query-result>
"""


def test_referenced_values_are_resolved_and_a_process_reads_as_its_pid() -> None:
    rows = _rows(XML)

    assert [row["start"] for row in rows] == ["1000", "2000"]
    # The second row repeats the encoder count by reference, not by value.
    assert [row["num-encoders"] for row in rows] == ["7", "7"]
    assert {row["process"] for row in rows} == {"99"}


def test_the_measured_phase_starts_after_the_last_quiet_gap() -> None:
    rows = [{"start": str(value)} for value in (0, 10, 20, 10_000, 10_010)]

    kept = _measured_phase(rows, gap_ns=1_000)

    assert [row["start"] for row in kept] == ["10000", "10010"]
