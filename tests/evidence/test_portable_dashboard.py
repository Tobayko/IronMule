"""Control-path tests for the local portable DATA1 status page."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from friday_evidence.portable import dashboard
from friday_evidence.events import EventJournal


def _dataset() -> dict:
    return {
        "counts": {"by_status": {"incorrect": 1, "censored": 2}, "records": 3},
        "holdout": {"hidden": True},
        "records": [
            {"backend": "cuda", "run_id": "a" * 32, "case_id": "case-1", "label": {"pair_count": 12, "median_ratio": 0.1}},
            {"backend": "mlx", "run_id": "b" * 32, "case_id": "case-2", "label": None},
            {"backend": "tpu", "run_id": "c" * 32, "case_id": "case-3", "label": {"pair_count": 12}},
        ],
    }


def test_snapshot_missing_state_is_readonly_uninitialized(tmp_path) -> None:
    state = tmp_path / "missing"
    value = dashboard.snapshot(state)
    assert value["read_only"] is True
    assert value["learning_claim"] is False
    assert value["quota"]["state"] == "uninitialized"
    assert value["corpus"]["state"] == "ready"
    assert not state.exists()


def test_snapshot_projects_safe_corpus_counts_without_labels_or_paths(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(dashboard, "build_dataset", lambda _root: _dataset())
    value = dashboard.snapshot(tmp_path)
    assert value["corpus"]["unique_cases"] == 3
    assert value["corpus"]["raw_samples"] == 24
    assert value["corpus"]["invalid"] == 1
    assert value["corpus"]["censored"] == 2
    encoded = json.dumps(value)
    assert "median_ratio" not in encoded
    assert "prompt" not in encoded
    assert str(tmp_path) not in encoded
    page = dashboard.render_dashboard(value)
    assert "lokaler Verlauf" in page
    assert "Holdout-Labels bleiben verborgen" in page


def _handler(tmp_path, path: str, origin: str | None = None):
    """Exercise handler dispatch in-process; the sandbox need not bind a port."""
    handler = object.__new__(dashboard._Handler)
    handler.path = path
    handler.headers = {} if origin is None else {"Origin": origin}
    handler.server = SimpleNamespace(server_address=("127.0.0.1", 8789), state_dir=tmp_path)
    captured: list[tuple[int, str, object]] = []
    handler._send = lambda status, kind, body: captured.append((status, kind, body))
    handler._json = lambda status, value: captured.append((status, "json", value))
    return handler, captured


def test_handler_serves_only_get_status_and_rejects_foreign_origin(tmp_path) -> None:
    handler, captured = _handler(tmp_path, "/api/status")
    handler.do_GET()
    assert captured[0][0] == 200
    assert captured[0][2]["read_only"] is True
    handler, captured = _handler(tmp_path, "/", "https://outside.example")
    handler.do_GET()
    assert captured == [(403, "json", {"error": "origin_forbidden"})]
    handler, captured = _handler(tmp_path, "/")
    handler.do_POST()
    assert captured == [(405, "json", {"error": "method_not_allowed"})]


def test_remote_host_and_non_get_routes_are_rejected(tmp_path) -> None:
    with pytest.raises(dashboard.DashboardError):
        dashboard.make_server(tmp_path, host="0.0.0.0", port=0)
    handler, captured = _handler(tmp_path, "/api/status?anything=1")
    handler.do_GET()
    assert captured == [(400, "json", {"error": "invalid_request"})]


def test_snapshot_reads_both_journals_and_uses_newest_paged_history(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(dashboard, "build_dataset", lambda _root: _dataset())
    run_id = "a" * 32
    for name, kind, payload in (("control.sqlite3", "run_finished", {"status": "failed", "error_code": "worker_timeout"}),
                                ("supervisor-events.sqlite3", "run_started", {"module": "portable"})):
        with EventJournal(tmp_path / name) as journal:
            journal.append(run_id, kind, payload)
    value = dashboard.snapshot(tmp_path)
    assert [row["kind"] for row in value["history"]] == ["run_finished", "run_started"]
    assert value["history"][0]["error_code"] == "worker_timeout"
    page = dashboard.render_dashboard(value)
    assert "Versuchshistorie" in page and "aaaaaaaa…" in page and "worker_timeout" in page


def test_snapshot_counts_only_verified_real_captures_and_raw_samples(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(dashboard, "build_dataset", lambda _root: {
        "counts": {"raw_samples": 37, "by_status": {}}, "records": [
            {"backend": "cuda", "run_id": "b" * 32, "case_id": "case", "sample_count": 12}],
        "holdout": {"hidden": True},
    })
    capture = tmp_path / "captures" / "one" / "data"
    capture.mkdir(parents=True)
    (capture / "capture.json").write_text(json.dumps({"status": "captured", "cases": [{"shape": [1, 2, 3]}]}))
    (capture.parent / "supervisor.json").write_text(json.dumps({"status": "finished", "code_unchanged": True}))
    bad = tmp_path / "captures" / "two" / "data"
    bad.mkdir(parents=True)
    (bad / "capture.json").write_text(json.dumps({"status": "captured", "cases": [{"shape": [9, 9, 9]}]}))
    (bad.parent / "supervisor.json").write_text(json.dumps({"status": "failed", "code_unchanged": True}))
    value = dashboard.snapshot(tmp_path)
    assert value["captures"] == {"real": 1, "dimensions": ["1×2×3"]}
    assert value["corpus"]["raw_samples"] == 37
    assert value["corpus"]["backends"]["GPU"]["raw_samples"] == 12
    assert "1×2×3" in dashboard.render_dashboard(value)


def test_handler_rejects_host_outside_loopback_allowlist(tmp_path) -> None:
    handler, captured = _handler(tmp_path, "/api/status")
    handler.headers = {"Host": "attacker.example:8789"}
    handler.do_GET()
    assert captured == [(403, "json", {"error": "host_forbidden"})]


def test_dashboard_projects_real_provider_smokes_without_performance_claim(tmp_path) -> None:
    passed = tmp_path / "provider-probes" / ("a" * 32)
    failed = tmp_path / "provider-probes" / ("b" * 32) / "recovered-output"
    passed.mkdir(parents=True)
    failed.mkdir(parents=True)
    (passed / "outcome.json").write_text(json.dumps({
        "schema":"ironmule.provider-smoke-outcome.v1", "report": {
            "schema":"ironmule.provider-smoke.v1", "run_id":"a"*32, "backend":"cuda",
            "status":"passed", "hardware_verified":True, "correctness_verified":True,
            "performance_claim":False, "hardware":{"accelerator":"Tesla T4",
            "framework":"torch", "framework_version":"2.10"}}}))
    (failed / "probe-result.json").write_text(json.dumps({
        "schema":"ironmule.provider-smoke.v1", "run_id":"b"*32, "backend":"cuda",
        "status":"failed", "hardware_verified":False, "correctness_verified":False,
        "performance_claim":False, "error_code":"cuda_unavailable"}))
    value = dashboard.snapshot(tmp_path)
    assert value["provider_probes"]["passed"] == 1
    assert value["provider_probes"]["failed"] == 1
    page = dashboard.render_dashboard(value)
    assert "Tesla T4" in page and "cuda_unavailable" in page
    assert "keine Performanceaussage" in page
