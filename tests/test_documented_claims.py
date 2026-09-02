"""Every headline number in the status documents must still exist in evidence.

This project's whole value is that a claim can be traced to a sealed
measurement. Documentation drift - a number edited, rounded differently, or
carried over from a superseded run - is therefore its worst defect class, and
nothing checked for it. This ledger does.

Each entry pins the German-formatted string as it appears in the document and
the raw value as it appears in the evidence. Both forms are written out on
purpose: a test that recomputed the formatting would pass while the document
said something else.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: (label, documented text, document, evidence file, raw value in that file)
JSON_CLAIMS = [
    ("persistent process ratio", "0,346968", "PROJECT_STATUS.md",
     "experiments/persistent_process/results.json", "0.34696789209993684"),
    ("persistent process effect", "65,3032", "PROJECT_STATUS.md",
     "experiments/persistent_process/results.json", "-65.30321079000632"),
    ("cycle 16 ratio", "0,9295921887", "PROJECT_STATUS.md",
     "experiments/matmul_compile_ab/results.json", "0.9295921887"),
    ("cycle 17 ratio", "0,9581074518", "PROJECT_STATUS.md",
     "experiments/batched_readback_compile/results.json", "0.9581074518"),
    ("cycle 21 ratio", "1,000510010", "PROJECT_STATUS.md",
     "experiments/fused_greedy_compile_v4/results.json", "1.000510009822041"),
]

#: Decisions must match the wording the status table reports.
DECISION_CLAIMS = [
    ("experiments/persistent_process/results.json", "engineering_gain_confirmed_exact_scope"),
    ("experiments/matmul_compile_ab/results.json", "runtime_compile_wins_exact_scope"),
    ("experiments/batched_readback_compile/results.json", "no_clear_speedup_baseline_retained"),
    ("experiments/fused_greedy_compile_v4/results.json", "fused_greedy_compile_inconclusive"),
]

#: The head-skip study lives in a sealed database rather than a JSON file.
HEAD_SKIP_DATABASE = ROOT / ".friday-data" / "head-skip-v1.sqlite3"
HEAD_SKIP_CLAIMS = [
    ("ratio", "0,846385", 0.8463845562069244),
    ("ci_low", "0,843147", 0.8431470041496976),
    ("ci_high", "0,851284", 0.8512844842159696),
]


def document(name: str) -> str:
    return (ROOT / name).read_text()


@pytest.mark.parametrize(
    "label,text,doc,evidence,raw",
    JSON_CLAIMS,
    ids=[entry[0].replace(" ", "-") for entry in JSON_CLAIMS],
)
def test_a_documented_number_is_still_in_its_evidence(label, text, doc, evidence, raw):
    assert text in document(doc), f"{label}: {text} is no longer in {doc}"
    body = (ROOT / evidence).read_text()
    assert raw in body, f"{label}: {raw} is no longer in {evidence}"


@pytest.mark.parametrize("evidence,decision", DECISION_CLAIMS, ids=[entry[1] for entry in DECISION_CLAIMS])
def test_each_study_still_records_the_decision_the_status_reports(evidence, decision):
    payload = json.loads((ROOT / evidence).read_text())
    assert payload.get("decision") == decision
    assert payload.get("formal_claim") is False


def test_head_skip_numbers_match_the_sealed_database():
    if not HEAD_SKIP_DATABASE.is_file():
        pytest.skip("sealed head-skip database is not present in this checkout")
    connection = sqlite3.connect(f"file:{HEAD_SKIP_DATABASE}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT payload_json FROM records WHERE kind='study_decision' ORDER BY seq"
        ).fetchall()
    finally:
        connection.close()
    assert rows, "the sealed study carries no decision record"
    intervals = json.loads(rows[0][0])["intervals"]["all"]
    status = document("PROJECT_STATUS.md")
    for field, text, expected in HEAD_SKIP_CLAIMS:
        assert intervals[field] == pytest.approx(expected, abs=1e-12), field
        assert text in status, f"{field}: {text} is no longer in PROJECT_STATUS.md"
    # Six confirmation sessions, all token-identical, is what the table claims.
    assert intervals["sessions"] == 6


def test_the_ledger_would_notice_a_changed_number():
    """A guard nobody has seen fail is not a guard."""

    body = (ROOT / "experiments/persistent_process/results.json").read_text()
    assert "0.34696789209993684" in body
    assert "0.34696789209993685" not in body
