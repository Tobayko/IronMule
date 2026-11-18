"""Live monitor projection, privacy, and HTTP transport checks."""
import http.client
import json
import queue
import threading
from collections import deque
from contextlib import closing
from types import SimpleNamespace

import pytest

from ironmule_product.http_server import create_server
from ironmule_product.live_view import benchmark, progress_path, runtime
from ironmule_product.service import ProductService, _Session


def test_benchmark_progress_contains_metrics_without_raw_evidence(tmp_path):
    path = tmp_path / "progress.json"
    assert benchmark(path)["status"] == "idle"
    row = {"engine": "llama_cpp", "repeat": 0, "concurrency": 4, "tokens_per_second": 7.25, "ttft_median_ms": 54}
    report = {"schema": "ironmule.compare.v1", "status": "running", "stage": "quality",
              "model": "/private/model.gguf", "device": "GPU-1", "current": {"engine": "llama_cpp"},
              "updated_at": "2026-11-18T12:30:00Z",
              "arms": [{"identity": {"device": "GPU-1", "device_name": "GPU name", "args": ["private"]}}],
              "runs": [row | {"quality": {"passed": 4, "total": 5, "answers": {"private": False}}},
                       row | {"quality": None, "prompt": "private"}],
              "verdict": {"status": "pending", "reasons": ["private"]}, "error": "private"}
    path.write_text(json.dumps(report))
    data = benchmark(path)
    assert data["model"] == "model.gguf" and data["device"] == "GPU name"
    assert data["status"] == "running" and data["updated_at"] == "2026-11-18T12:30:00+00:00"
    assert data["run_count"] == 2 and data["runs"][-1]["quality"] is None
    assert data["runs"][0]["ttft_p95_ms"] is None
    assert data["runs"][0]["quality"] == {"passed": 4, "total": 5}
    assert data["runs"][0]["tokens_per_second"] == 7.25
    assert "private" not in json.dumps(data)
    report.update(status="error", error_code="launch_failed", updated_at="private",
                  verdict={"status": "incomplete", "reasons": ["missing_arm", "private"]})
    path.write_text(json.dumps(report))
    assert benchmark(path)["reasons"] == ["missing_arm"]
    assert benchmark(path)["error_code"] == "launch_failed"
    assert benchmark(path)["updated_at"] is None


@pytest.mark.parametrize("value", [b'{', b'{"schema": "other"}', b'0' * ((4 << 20) + 1)])
def test_unreadable_report_has_no_live_claim(tmp_path, value):
    path = tmp_path / "progress.json"
    path.write_bytes(value)
    assert benchmark(path) == {"status": "unavailable", "runs": [], "run_count": 0}


def test_runtime_projects_status_recent_requests_and_width_decisions():
    health = {"ready": True, "active_requests": 0, "queued_requests": 0, "completed_requests": 33,
              "ggml": {"gate": {"passes": True}, "agent": {"serving": "w4", "last_decisions": [
                  {"action": "switch", "to": "w4", "at": 15., "trace": "private"},
                  {"action": "private"}]}},
              "loaded_model": "model", "execution": "numeric@ggml", "device": "cpu"}
    make_samples = lambda start: [{"at": i, "completion_tokens": 2, "tokens_per_second": 3., "ttft_ms": 4,
                                    "request_id": "private"} for i in range(start, start + 32)]
    service = SimpleNamespace(health=lambda: health, shards=[SimpleNamespace(live_samples=lambda: make_samples(32)),
                                                             SimpleNamespace(live_samples=lambda: make_samples(0))])
    data = runtime(service)
    assert data["status"] == "idle" and data["completed"] == 33
    assert [s["at"] for s in data["samples"]] == list(range(32, 64))
    assert data["config"] == "w4" and data["gate"] is True
    assert data["decisions"] == [{"action": "switch", "at": 15., "to": "w4"}]
    assert "private" not in json.dumps(data)
    health["queued_requests"] = 1
    assert runtime(service)["status"] == "running"
    health["ready"] = False
    assert runtime(service)["status"] == "not_ready"
    assert runtime(None)["status"] == "detached"


def test_scheduler_retirement_records_only_successful_delivery():
    service = object.__new__(ProductService)
    service._lock, service._pending = threading.RLock(), queue.Queue()
    service._active, service._sessions, service._live_samples = set(), {}, deque(maxlen=32)
    for outcome, count in [("ok", 3)] * 33 + [("cancelled", 2), ("failed", 1), ("ok", 0)]:
        item = _Session(SimpleNamespace(request_id="private"), None)
        item.submitted_ns, item.first_ns, item.end_ns = 0, 250_000_000, 2_000_000_000
        item.outcome, item.tokens = outcome, count
        service._pending.put(item)
        service._retire(service._pending.get_nowait())
    samples = service.live_samples()
    assert len(samples) == 32 and service._pending.unfinished_tasks == 0
    assert all(set(s) == {"at", "completion_tokens", "tokens_per_second", "ttft_ms"} for s in samples)
    assert all(s["tokens_per_second"] == 1.5 and s["ttft_ms"] == 250 and s["completion_tokens"] == 3 for s in samples)
    samples[-1]["ttft_ms"] = 0
    assert service.live_samples()[-1]["ttft_ms"] == 250


def test_progress_file_precedence(tmp_path, monkeypatch):
    service = SimpleNamespace(store=SimpleNamespace(root=tmp_path))
    monkeypatch.delenv("IRONMULE_BENCH_PROGRESS", raising=False)
    assert progress_path(service) == tmp_path / "benchmark-progress.json"
    monkeypatch.setenv("IRONMULE_BENCH_PROGRESS", str(tmp_path / "env"))
    assert progress_path(service) == tmp_path / "env"
    assert progress_path(service, tmp_path / "override") == tmp_path / "override"


def test_standalone_exposes_only_authenticated_data_and_static_page(tmp_path):
    path = tmp_path / "report"
    with create_server(None, port=0, api_key="test-key", progress=path) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def get(method, route, auth=False):
            with closing(http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)) as conn:
                headers = {"Authorization": "Bearer test-key"} if auth else {}
                conn.request(method, route, headers=headers)
                response = conn.getresponse()
                return response, response.read()

        try:
            for route in ("/", "/live"):
                response, html = get("GET", route)
                assert response.status == 200 and b"Live monitor" in html
                assert "connect-src 'self'" in response.getheader("Content-Security-Policy")
            assert get("GET", "/v1/ironmule/live")[0].status == 401
            response, raw = get("GET", "/v1/ironmule/live", True)
            assert response.status == 200
            assert json.loads(raw)["runtime"]["status"] == "detached"
            assert get("GET", "/v1/models", True)[0].status == 404
            assert get("POST", "/v1/ironmule/learning", True)[0].status == 404
        finally:
            server.shutdown()
            thread.join(timeout=2)
