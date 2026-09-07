"""Real control-plane tests for calibration lifecycle and persistence."""

from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ironmule_product.calibration import CalibrationJob, _job_lease, history, status
from ironmule_product.state import ProductStore, _atomic_write
from ironmule_product.types import ModelSpec


ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable


def _store(tmp_path: Path) -> ProductStore:
    store = ProductStore(tmp_path / "state")
    store.setup()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    store.register_model(ModelSpec("local/test", "revision", str(snapshot), 1))
    return store


def test_zero_wait_readiness_only_defers_without_loading_worker(tmp_path: Path):
    store = _store(tmp_path)
    job = CalibrationJob(store, "local/test", max_wait_s=0, readiness_only=True)
    report = job.run()
    assert report["status"] == "deferred"
    assert report["error_code"] == "readiness_timeout"
    journal = history(store)
    assert journal["verified"] is True
    assert [event["kind"] for event in journal["events"]] == [
        "run_started", "readiness", "deferred", "run_finished",
    ]
    assert all(event["payload"].get("model_loaded") is not True for event in journal["events"])


def test_readiness_only_run_does_not_import_mlx(tmp_path: Path):
    state = tmp_path / "state"
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    script = """
import json, sys
class DenyInference:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mlx', 'mlx_lm'}:
            raise AssertionError('readiness-only control path imported inference: ' + fullname)
sys.meta_path.insert(0, DenyInference())
from ironmule_product.state import ProductStore
from ironmule_product.types import ModelSpec
from ironmule_product.calibration import CalibrationJob
store = ProductStore(sys.argv[1]); store.setup()
store.register_model(ModelSpec('local/test', 'revision', sys.argv[2], 1))
report = CalibrationJob(store, 'local/test', max_wait_s=0, readiness_only=True).run()
print(json.dumps({'status': report['status'], 'error_code': report.get('error_code')}))
"""
    result = subprocess.run(
        [PYTHON, "-S", "-c", script, str(state), str(snapshot)],
        cwd=ROOT, capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "deferred"


def _hold_lease(root: str, ready, release) -> None:
    store = ProductStore(Path(root))
    with _job_lease(store):
        ready.set()
        release.wait(10)


def test_real_process_lease_defers_second_job_and_status_reports_running(tmp_path: Path):
    store = _store(tmp_path)
    ready = multiprocessing.Event()
    release = multiprocessing.Event()
    process = multiprocessing.Process(target=_hold_lease, args=(str(store.root), ready, release))
    process.start()
    try:
        assert ready.wait(5)
        running = status(store)
        assert running["engine_started"] is True
        report = CalibrationJob(store, "local/test", max_wait_s=0, readiness_only=True).run()
        assert report["status"] == "deferred"
        assert report["error_code"] == "another_calibration_running"
    finally:
        release.set()
        process.join(5)
        if process.is_alive():
            process.terminate()
            process.join(5)
    assert process.exitcode == 0
    assert status(store)["engine_started"] is False


def test_stale_lease_file_alone_does_not_report_running(tmp_path: Path):
    store = _store(tmp_path)
    lease = store.root / ".optimization.lock"
    lease.touch(mode=0o600)
    os.chmod(lease, 0o600)
    assert status(store)["engine_started"] is False


def test_saved_active_stage_without_lease_is_interrupted(tmp_path: Path):
    store = _store(tmp_path)
    _atomic_write(store.root / "optimization-status.json", {
        "schema": 1, "stage": "measuring", "run_id": "a" * 32,
        "model_id": "local/test", "model_loaded": True,
        "updated_unix_ns": 1, "verdict": None,
    })
    current = status(store)
    assert current["stage"] == "interrupted"
    assert current["engine_started"] is False


def test_pause_resume_controls_calibration_status(tmp_path: Path):
    store = _store(tmp_path)
    assert status(store)["paused"] is False
    store.set_optimization_paused(True)
    assert status(store)["paused"] is True
    report = CalibrationJob(store, "local/test", max_wait_s=0, readiness_only=True).run()
    assert report["status"] == "deferred" and report["error_code"] == "paused"
    store.set_optimization_paused(False)
    assert status(store)["paused"] is False


def test_cli_optimize_help_works_without_site_packages():
    result = subprocess.run(
        [PYTHON, "-S", "-m", "ironmule_cli", "optimize", "--help"],
        cwd=ROOT, capture_output=True, text=True, timeout=10, check=False,
    )
    assert result.returncode == 0
    assert "optimize {run|status|history|pause|resume}" in result.stdout


def test_cli_history_without_state_does_not_create_database(tmp_path: Path):
    state = tmp_path / "absent-state"
    result = subprocess.run(
        [PYTHON, "-S", "-m", "ironmule_cli", "optimize", "history",
         "--state-dir", str(state)], cwd=ROOT, capture_output=True,
        text=True, timeout=10, check=False,
    )
    assert result.returncode != 0
    assert not state.exists()
    assert not (tmp_path / "optimization.sqlite3").exists()
