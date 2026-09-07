"""Bounded PROD4 load-screen control tests; no model generation is allowed."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from friday_evidence.events import EventJournal
from ironmule_product.state import ProductStore
from ironmule_product.types import ModelSpec
from tools import product_load_screen


ROOT = Path(__file__).resolve().parents[2]


def _store(tmp_path: Path) -> ProductStore:
    store = ProductStore(tmp_path / "state")
    store.setup()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    store.register_model(ModelSpec("local/test", "revision", str(snapshot), 1))
    return store


def test_help_runs_from_source_without_execution():
    result = subprocess.run(
        [sys.executable, "-S", str(ROOT / "tools" / "product_load_screen.py"), "--help"],
        cwd=ROOT, env={"PYTHONPATH": str(ROOT)},
        capture_output=True, text=True, timeout=10, check=False,
    )
    assert result.returncode == 0
    assert "--execute" in result.stdout and "load-only" in result.stdout


@pytest.mark.parametrize(
    "returncode,forced_error,expected",
    [
        (0, None, "clean"),
        (-6, "rss_limit_exceeded", "forced_abort"),
        (-6, None, "crashed"),
        (-15, None, "expected_abort"),
    ],
)
def test_worker_exit_classifier_preserves_forced_guard_reason(returncode, forced_error, expected):
    controlled = "cancelled" if expected == "expected_abort" else None
    assert product_load_screen.classify_worker_exit(returncode, forced_error, controlled) == expected


def test_wrapped_memory_guard_error_is_classified_as_forced_abort():
    try:
        try:
            raise product_load_screen.MemoryGuardError("rss_limit_exceeded")
        except product_load_screen.MemoryGuardError as cause:
            raise product_load_screen.LoadScreenFailure(cause.code) from cause
    except product_load_screen.LoadScreenFailure as wrapped:
        forced = product_load_screen._memory_guard_code(wrapped)  # noqa: SLF001
        controlled = wrapped.code
    assert product_load_screen.classify_worker_exit(-6, forced, controlled) == "forced_abort"


def test_invalid_wait_and_existing_output_are_rejected_before_model_work(tmp_path: Path):
    store = _store(tmp_path)
    output = tmp_path / "result.json"
    output.write_text("existing", encoding="utf-8")
    with pytest.raises(product_load_screen.LoadScreenFailure, match="invalid_wait_ready"):
        product_load_screen.run(store.root, "local/test", output, float("nan"))
    with pytest.raises(product_load_screen.LoadScreenFailure, match="output_exists"):
        product_load_screen.run(store.root, "local/test", output, 1)


def test_source_package_failure_still_writes_one_terminal_journal_event(tmp_path: Path):
    store = _store(tmp_path)
    report = product_load_screen.run(store.root, "local/test", tmp_path / "result.json", 0)
    assert report["status"] == "failed"
    assert report["error_code"] == "source_tree_package_in_use"
    with EventJournal(store.root / product_load_screen.JOURNAL_NAME, read_only=True) as journal:
        events = journal.events()
    assert [event["kind"] for event in events] == ["run_started", "run_finished"]
    assert events[-1]["payload"]["status"] == "failed"
    assert "persist" not in (ROOT / "tools" / "product_load_screen.py").read_text()


def test_wait_ready_checks_cancellation_before_probe(tmp_path: Path):
    called = []

    def check():
        raise product_load_screen.LoadScreenFailure("cancelled")

    def unexpected_probe():
        called.append(True)
        raise AssertionError("probe must follow the pre-probe check")

    original = product_load_screen.probe
    product_load_screen.probe = unexpected_probe
    try:
        with pytest.raises(product_load_screen.LoadScreenFailure, match="cancelled"):
            product_load_screen._wait_ready(  # noqa: SLF001
                deadline=1e20, global_deadline=1e20, check=check,
                sleep=lambda _seconds: None, journal=None, run_id="a" * 32,
                phase="before_load", sample_sink=[],
            )
    finally:
        product_load_screen.probe = original
    assert called == []


def test_report_contains_no_absolute_paths_or_generation_claims(tmp_path: Path):
    store = _store(tmp_path)
    report = product_load_screen.run(store.root, "local/test", tmp_path / "result.json", 0)
    rendered = json.dumps(report, sort_keys=True)
    assert str(tmp_path) not in rendered
    assert report["performance_claim"] is False
    assert report["activation_allowed"] is False
