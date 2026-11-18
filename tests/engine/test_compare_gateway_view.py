"""Provider bootstrap serves a read-only projection with no installed IronMule runtime."""
import http.client
import json
import os
import secrets
import subprocess
import sys
from contextlib import closing

from lab.portable_gpu.watch_gateway import public_snapshot, render


def write_run(folder):
    (folder / "status.json").write_text(json.dumps({"state": "running", "stage": "compare_qwen3_06b", "raw": "private"}))
    doc = {"schema": "ironmule.compare.v1", "status": "running", "stage": "quality",
           "model": "/private/input.gguf", "device": "GPU-private",
           "current": {"engine": "vllm", "repeat": 0, "concurrency": 8},
           "arms": [{"identity": {"device": "GPU-private", "device_name": "NVIDIA A100-SXM4-80GB", "argv": "private"}}],
           "runs": [{"engine": "vllm", "tokens_per_second": 12.5, "ttft_median_ms": 30,
                     "quality": {"passed": 15, "total": 18, "answers": {"private": True}}}]}
    (folder / "qwen3-06b.json").write_text(json.dumps(doc))
    return doc


def test_projection_tracks_active_model_and_reports_setup_and_interruption(tmp_path):
    assert public_snapshot(tmp_path)["benchmark"]["status"] == "unavailable"
    doc = write_run(tmp_path)
    small = public_snapshot(tmp_path)["benchmark"]
    assert small["status"] == "running" and small["model"] == "Qwen3-0.6B"
    assert small["device"] == "NVIDIA A100-SXM4-80GB" and small["runs"][0]["quality"] == {"passed": 15, "total": 18}
    assert "private" not in json.dumps(small) and ".gguf" not in json.dumps(small)

    (tmp_path / "status.json").write_text('{"state":"running","stage":"fetch_qwen3_32b"}')
    large = public_snapshot(tmp_path)["benchmark"]
    assert large["model"] == "Qwen3-32B" and large["runs"] == [] and large["status"] == "preparing"
    for phase in ("readiness_qwen3-32b", "readiness_saved_qwen3-32b", "compare_qwen3-32b"):
        (tmp_path / "status.json").write_text(json.dumps({"state": "running", "stage": phase}))
        current = public_snapshot(tmp_path)["benchmark"]
        assert current["model"] == "Qwen3-32B" and current["runs"] == []
        assert current["status"] == "preparing" and current["provider"]["event"]
    (tmp_path / "qwen3-32b.json").write_text(json.dumps(doc))
    (tmp_path / "status.json").write_text('{"state":"error","stage":"compare_qwen3_32b"}')
    interrupted = public_snapshot(tmp_path)["benchmark"]
    assert interrupted["run_count"] == 1 and interrupted["status"] == "error"
    assert interrupted["verdict"] == "incomplete"


def test_standalone_bootstrap_has_only_two_unauthenticated_capability_routes(tmp_path):
    template = (
        "import http.server, os, pathlib\n"
        "ROOT=pathlib.Path(__file__).parent\n"
        "TOKEN=os.environ['RESULT_TOKEN']\n"
        "class Handler(http.server.BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        "        self.send_error(401)\n"
        "    def do_POST(self):\n"
        "        self.send_error(401)\n"
        "    def log_message(self,*args): pass\n"
        "server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)\n"
        "print(server.server_port,flush=True)\nserver.serve_forever()\n"
    )
    path = tmp_path / "gateway.py"
    path.write_text(render(template))
    watch, control = secrets.token_urlsafe(24), secrets.token_urlsafe(32)
    write_run(tmp_path)
    with subprocess.Popen([sys.executable, "-I", "-u", str(path)], cwd=tmp_path, stdout=subprocess.PIPE, text=True,
                          env={**os.environ, "PYTHONPATH": "", "WATCH_ID": watch, "RESULT_TOKEN": control}) as child:
        try:
            port = int(child.stdout.readline())
            url = f"/watch/{watch}/"

            def request(method, route):
                with closing(http.client.HTTPConnection("127.0.0.1", port, timeout=2)) as conn:
                    conn.request(method, route)
                    res = conn.getresponse()
                    return res.status, dict(res.getheaders()), res.read()

            code, headers, html = request("GET", url)
            assert code == 200 and b'data-watch="shared"' in html
            assert headers["Referrer-Policy"] == "no-referrer" and headers["Cache-Control"] == "private, no-store"
            code, _, body = request("GET", url + "data")
            assert code == 200 and json.loads(body)["benchmark"]["runs"][0]["tokens_per_second"] == 12.5
            assert all(value not in body for value in (b"private", control.encode(), watch.encode(), b"/tmp/", b".gguf"))
            for method, route in (("GET", "/"), ("GET", "/status.json"), ("GET", "/watch/unknown/data"),
                                  ("GET", url + "data?raw=1"), ("GET", url + "qwen3-06b.json"),
                                  ("GET", url + "../status.json"), ("POST", url), ("POST", url + "data"),
                                  ("POST", "/bundle")):
                assert request(method, route)[0] == 401
        finally:
            child.terminate()
            child.wait(timeout=5)
