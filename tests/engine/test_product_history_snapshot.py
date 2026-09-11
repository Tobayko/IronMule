"""Actual SQLite source tests for the PROD7 history adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from friday_evidence.events import EventJournal
from tools.product_history_snapshot import _model_label, snapshot


def _write_journal(path: Path, source: str, rows: list[dict]) -> None:
    with EventJournal(path) as journal:
        for row in rows:
            run_id = row["run_id"]
            model_id = row["model_id"]
            journal.append(run_id, "run_started", {
                "model_id": model_id, "revision": "revision",
                **({"readiness_policy": {"max_load_ratio": 0.2}} if source == "optimization" else {}),
            })
            for sample_index in row.get("passed_samples", ()):
                if source == "optimization":
                    journal.append(run_id, "validation", {"sample": {
                        "sample_index": sample_index, "status": "passed",
                    }})
            if source == "load":
                journal.append(run_id, "validation", {"state": "load_memory_sample", "observation": {
                    "started_unix_ns": row["started_unix_ns"] + 1_000_000,
                    "finished_unix_ns": row["started_unix_ns"] + 2_000_000,
                    "pid": 10, "rss_bytes": 100, "swap_used_bytes": row.get("swap_bytes", 0),
                    "swap_delta_bytes": row.get("swap_delta_bytes", 0), "errors": row.get("errors", []),
                }})
            journal.append(run_id, "run_finished", {
                "status": row["status"], "completed_calls": row.get("terminal_completed_calls"),
                "error_code": row.get("error_code"),
                "worker": row.get("worker", {}),
            })


def test_single_source_interface_and_dual_source_export_are_verified(tmp_path: Path):
    optimization = tmp_path / "optimization.sqlite3"
    load = tmp_path / "load-screen.sqlite3"
    _write_journal(optimization, "optimization", [{
        "run_id": "1" * 32, "model_id": "org/gemma-3-1b", "status": "finished",
        "passed_samples": [1, 1, 2], "terminal_completed_calls": 2,
        "started_unix_ns": 1_000_000_000,
    }])
    _write_journal(load, "load", [{
        "run_id": "2" * 32, "model_id": "org/gemma-3-12b", "status": "passed",
        "started_unix_ns": 2_000_000_000, "swap_delta_bytes": 257 * 1024 * 1024,
        "error_code": "swap_growth", "worker": {"returncode": -6, "exit_class": "forced_abort"},
    }])
    artifact, export = snapshot(optimization, load)
    assert export["schema"] == "ironmule.optimization_export.v2"
    assert len(export["journals"]) == 2
    assert all(len(item["chain_sha256"]) == 64 for item in export["journals"])
    assert all(item["as_of_seq"] == item["verified_events"] for item in export["journals"])
    runs = artifact["snapshot"]["datasets"]["runs"]
    assert len(runs) == 2
    assert sum(row["completed_requests"] for row in runs) == 2
    load_row = next(row for row in runs if row["kind"] == "load")
    assert load_row["status"] == "failed"
    assert load_row["raw_status"] == "passed"
    assert load_row["raw_returncode"] == -6
    assert load_row["raw_exit_class"] == "forced_abort"
    assert "swap_growth" in load_row["detail"]
    totals = artifact["snapshot"]["datasets"]["totals"][0]
    assert totals["attempts"] == 2
    assert totals["completed_requests"] == 2
    assert totals["failed_or_blocked_attempts"] == 1
    assert artifact["snapshot"]["datasets"]["load_observations"][0]["swap_delta_mib"] == pytest.approx(257)


def test_single_source_output_remains_compatible(tmp_path: Path):
    path = tmp_path / "optimization.sqlite3"
    _write_journal(path, "optimization", [{
        "run_id": "3" * 32, "model_id": "org/gemma-3-4b", "status": "deferred",
        "passed_samples": [], "terminal_completed_calls": 0, "started_unix_ns": 1,
    }])
    artifact, export = snapshot(path)
    assert export["schema"] == "ironmule.optimization_export.v1"
    assert isinstance(export["events"], list)
    assert artifact["snapshot"]["datasets"]["runs"][0]["model_label"] == "4B"
    assert _model_label("org/gemma-3-27b-rev1") == "gemma-3-27b-rev1"


def test_missing_explicit_load_database_fails_closed(tmp_path: Path):
    path = tmp_path / "optimization.sqlite3"
    _write_journal(path, "optimization", [{
        "run_id": "4" * 32, "model_id": "org/gemma-3-1b", "status": "finished",
        "started_unix_ns": 1, "terminal_completed_calls": 0,
    }])
    with pytest.raises(ValueError, match="event journal"):
        snapshot(path, tmp_path / "missing-load.sqlite3")


def test_manifest_has_three_metric_cards_and_compact_five_column_table(tmp_path: Path):
    path = tmp_path / "optimization.sqlite3"
    _write_journal(path, "optimization", [{
        "run_id": "5" * 32, "model_id": "org/gemma-3-1b", "status": "finished",
        "started_unix_ns": 1, "terminal_completed_calls": 0,
    }])
    artifact, _ = snapshot(path)
    blocks = artifact["manifest"]["blocks"]
    assert blocks[0]["type"] == "metric-strip"
    assert len(artifact["manifest"]["cards"]) == 3
    assert all(card["sourceId"] == "history" and card["metrics"] for card in artifact["manifest"]["cards"])
    assert all("title" not in card for card in artifact["manifest"]["cards"])
    assert len(artifact["manifest"]["tables"][0]["columns"]) == 5
    assert artifact["manifest"]["tables"][0]["defaultSort"] == {"field": "start_utc", "direction": "desc"}
    rendered = json.dumps(artifact, sort_keys=True)
    assert "prompt" not in rendered.lower()
    assert str(tmp_path) not in rendered


def test_reconciliation_mismatch_and_final_failed_sample_are_fail_closed(tmp_path: Path):
    path = tmp_path / "optimization.sqlite3"
    with EventJournal(path) as journal:
        run_id = "6" * 32
        journal.append(run_id, "run_started", {"model_id": "org/gemma-3-4b-rev1", "revision": "r"})
        journal.append(run_id, "validation", {"sample": {"sample_index": 1, "status": "passed"}})
        journal.append(run_id, "validation", {"sample": {"sample_index": 1, "status": "failed"}})
        journal.append(run_id, "run_finished", {"status": "failed", "completed_calls": 1, "worker": {}})
    with pytest.raises(ValueError, match="reconcile"):
        snapshot(path)


def test_incomplete_calibration_preserves_passed_request_count(tmp_path: Path):
    path = tmp_path / "optimization.sqlite3"
    with EventJournal(path) as journal:
        run_id = "7" * 32
        journal.append(run_id, "run_started", {"model_id": "org/gemma-3-4b", "revision": "r"})
        journal.append(run_id, "validation", {"sample": {"sample_index": 1, "status": "passed"}})
        journal.append(run_id, "validation", {"sample": {"sample_index": 2, "status": "passed"}})
    artifact, _ = snapshot(path)
    row = artifact["snapshot"]["datasets"]["runs"][0]
    assert row["status"] == "incomplete"
    assert row["completed_requests"] == 2
    assert row["request_reconciliation"] == "pending"
