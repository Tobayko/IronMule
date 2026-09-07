"""Real control-plane tests for calibration lifecycle and persistence."""

from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from ironmule_product.calibration import CalibrationJob, _job_lease, history, status
from ironmule_product.state import ProductStore, _atomic_write
from ironmule_product.types import ModelSpec
from friday_evidence.events import EventJournal


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
    assert journal["events"][0]["payload"]["load_monitor"] == {
        "schema": "ironmule.load_monitor.v1",
        "poll_interval_seconds": 0.25,
        "swap_delta_limit_bytes": 256 * 1024 * 1024,
        "rss_limit_fraction": 0.60,
        "mlx_peak_limit_fraction": 0.60,
        "clean_shutdown_required": True,
    }
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


def test_check_polls_active_load_guard_without_recursing(tmp_path: Path):
    store = _store(tmp_path)
    job = CalibrationJob(store, "local/test", max_wait_s=0, readiness_only=True)
    job.deadline = time.monotonic() + 10
    calls = []

    class Guard:
        def __call__(self, pid, force=False):
            calls.append((pid, force))
            return {"pid": pid}

    job._active_load_guard = Guard()
    job._active_process = SimpleNamespace(pid=12345)
    job._check()
    assert calls == [(12345, False)]


def test_export_reconstructs_load_memory_rows(tmp_path: Path):
    database = tmp_path / "optimization.sqlite3"
    run_id = "a" * 32
    ready = {
        "startup_wall_seconds": 0.5,
        "process_peak_rss_bytes": 100,
        "mlx_active_bytes": 50,
        "mlx_peak_bytes": 80,
        "mlx_cache_bytes": 10,
        "recommended_working_set_bytes": 1000,
        "rss_limit_bytes": 500,
        "swap_delta_limit_bytes": 256 * 1024 * 1024,
    }
    load_monitor = {
        "schema": "ironmule.load_monitor.v1",
        "poll_interval_seconds": 0.25,
        "swap_delta_limit_bytes": 256 * 1024**2,
        "rss_limit_fraction": 0.60,
        "mlx_peak_limit_fraction": 0.60,
        "clean_shutdown_required": True,
    }
    with EventJournal(database) as journal:
        journal.append(run_id, "run_started", {"plan_id": "plan", "model_id": "local/test", "revision": "revision", "load_monitor": load_monitor})
        journal.append(run_id, "validation", {"state": "load_memory_sample", "worker_index": 0,
                                                "observation": {"pid": 12345, "rss_bytes": 100, "swap_delta_bytes": 0, "errors": []}})
        journal.append(run_id, "validation", {"state": "load_memory_ready", "worker_index": 0,
                                                "ready": ready, "observation": {"pid": 12345, "rss_bytes": 100, "swap_delta_bytes": 0, "errors": []}})
        journal.append(run_id, "validation", {"state": "worker_cleanup",
                                                "worker": {"worker_index": 0, "started": True, "closed": True, "pid": 12345},
                                                "worker_exit": {"worker_index": 0, "returncode": 0, "normal_shutdown": True}})
        journal.append(run_id, "run_finished", {"status": "failed", "report_status": "failed"})

    from tools.product_calibration_export import export_run

    exported = export_run(database, run_id)
    rows = exported["report"]["worker_load_memory"]
    assert rows == [{
        "worker_index": 0,
        "samples": [{"errors": [], "pid": 12345, "rss_bytes": 100, "swap_delta_bytes": 0}],
        "ready": ready,
        "ready_observation": {"errors": [], "pid": 12345, "rss_bytes": 100, "swap_delta_bytes": 0},
    }]
    assert exported["report"]["load_monitor"] == load_monitor
    assert exported["report"]["worker_exit_codes"] == [{"worker_index": 0, "returncode": 0, "normal_shutdown": True}]


def test_export_preserves_legacy_report_shape_without_load_monitor_events(tmp_path: Path):
    database = tmp_path / "optimization.sqlite3"
    run_id = "b" * 32
    with EventJournal(database) as journal:
        journal.append(run_id, "run_started", {"plan_id": "plan", "model_id": "local/test", "revision": "revision"})
        journal.append(run_id, "run_finished", {"status": "failed", "report_status": "failed"})

    from tools.product_calibration_export import export_run

    report = export_run(database, run_id)["report"]
    assert set(report) == {
        "schema", "run_id", "plan_id", "model_id", "revision", "status", "activation_allowed",
        "resource_valid", "error_code", "error_type", "error_stage", "error_detail", "evaluation",
        "budget", "samples", "resource_events", "workers", "worker_timings",
    }
    assert "load_monitor" not in report
    assert "worker_load_memory" not in report


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
