"""Tiny token-authenticated job agent that runs inside a RunPod pod (stdlib only).

The pod's start command decodes this file from the AGENT_B64 environment variable and runs it on port 8000;
RunPod's HTTPS proxy exposes it as https://<pod>-8000.proxy.runpod.net. Every request needs the header
`X-Token: $AGENT_TOKEN`. Requests stay short (the proxy cuts at ~100 s): long commands run as background jobs.

  POST /put?path=P        body -> file P (parents created)
  GET  /get?path=P        file P
  POST /run               {"cmd": "...", "timeout": 60}  -> {"rc", "out"} (foreground, short)
  POST /job               {"cmd": "...", "name": "x"}    -> starts `bash -c cmd` with log /jobs/x.log
  GET  /job?name=x&tail=N -> {"running", "rc", "tail"}
"""
import hmac
import json
import os
import subprocess
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

TOKEN = os.environ["AGENT_TOKEN"]
JOBS = Path("/jobs")
JOBS.mkdir(exist_ok=True)
RUNNING = {}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, kind="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _auth(self):
        if not hmac.compare_digest(self.headers.get("X-Token", ""), TOKEN):
            self._send(403, {"error": "token"})
            return None
        url = urllib.parse.urlsplit(self.path)
        return url.path, dict(urllib.parse.parse_qsl(url.query))

    def _body(self):
        return self.rfile.read(int(self.headers.get("Content-Length", 0)))

    def do_GET(self):
        got = self._auth()
        if not got:
            return
        path, q = got
        if path == "/get":
            p = Path(q["path"])
            return self._send(200, p.read_bytes(), "application/octet-stream") if p.is_file() else self._send(404, {})
        if path == "/job":
            name = q["name"]
            proc = RUNNING.get(name)
            log = JOBS / f"{name}.log"
            tail = int(q.get("tail", 40))
            lines = log.read_text(errors="replace").splitlines()[-tail:] if log.exists() else []
            rc = None if proc is None else proc.poll()
            return self._send(200, {"running": proc is not None and rc is None, "rc": rc, "tail": "\n".join(lines)})
        if path == "/health":
            return self._send(200, {"ok": True})
        self._send(404, {})

    def do_POST(self):
        got = self._auth()
        if not got:
            return
        path, q = got
        if path == "/put":
            p = Path(q["path"])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(self._body())
            return self._send(200, {"bytes": p.stat().st_size})
        spec = json.loads(self._body() or b"{}")
        if path == "/run":
            try:
                done = subprocess.run(["bash", "-c", spec["cmd"]], capture_output=True, text=True,
                                      timeout=spec.get("timeout", 60))
                return self._send(200, {"rc": done.returncode, "out": (done.stdout + done.stderr)[-20000:]})
            except subprocess.TimeoutExpired:
                return self._send(200, {"rc": None, "out": "timeout"})
        if path == "/job":
            name = spec["name"]
            old = RUNNING.get(name)
            if old is not None and old.poll() is None:
                return self._send(409, {"error": "running"})
            log = (JOBS / f"{name}.log").open("w")
            RUNNING[name] = subprocess.Popen(["bash", "-c", spec["cmd"]], stdout=log, stderr=subprocess.STDOUT,
                                             start_new_session=True)
            return self._send(200, {"started": name, "pid": RUNNING[name].pid})
        if path == "/kill":
            proc = RUNNING.get(spec["name"])
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid, 15)
            return self._send(200, {"killed": spec["name"]})
        self._send(404, {})

    def log_message(self, *args):
        pass


ThreadingHTTPServer(("0.0.0.0", 8000), Handler).serve_forever()
