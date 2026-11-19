"""Pod-side command channel for a RunPod lab run (stdlib only, one token, background jobs).

Started by the pod's start command from the AGENT_B64 environment variable. Every request needs the header
X-Token equal to the TOKEN environment variable. Routes: PUT /file?path=P (write the body), GET /file?path=P,
POST /run {"name", "cmd"} (bash in the background, log at /work/logs/<name>.log), GET /job?name=N&tail=K.
"""
import json
import os
import subprocess
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

TOKEN = os.environ["TOKEN"]
LOGS = Path("/work/logs")
LOGS.mkdir(parents=True, exist_ok=True)
JOBS = {}


class Handler(BaseHTTPRequestHandler):
    def _reply(self, code, body, kind="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _ok(self):
        if self.headers.get("X-Token") != TOKEN:
            self._reply(403, {"error": "token"})
            return None
        url = urllib.parse.urlparse(self.path)
        return url.path, dict(urllib.parse.parse_qsl(url.query))

    def _body(self):
        return self.rfile.read(int(self.headers.get("Content-Length") or 0))

    def do_GET(self):
        got = self._ok()
        if got is None:
            return
        path, q = got
        if path == "/ping":
            return self._reply(200, {"ok": True})
        if path == "/file":
            p = Path(q["path"])
            return self._reply(200, p.read_bytes(), "application/octet-stream") if p.is_file() else self._reply(404, {})
        if path == "/job":
            name = q["name"]
            proc = JOBS.get(name)
            log = LOGS / f"{name}.log"
            text = log.read_bytes().decode(errors="replace") if log.exists() else ""
            tail = int(q.get("tail", 40))
            return self._reply(200, {"known": proc is not None, "rc": None if proc is None else proc.poll(),
                                     "tail": "\n".join(text.splitlines()[-tail:])})
        self._reply(404, {})

    def do_PUT(self):
        got = self._ok()
        if got is None:
            return
        path, q = got
        if path != "/file":
            return self._reply(404, {})
        p = Path(q["path"])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(self._body())
        self._reply(200, {"bytes": p.stat().st_size})

    def do_POST(self):
        got = self._ok()
        if got is None:
            return
        path, _ = got
        if path != "/run":
            return self._reply(404, {})
        spec = json.loads(self._body())
        name = spec["name"]
        if name in JOBS and JOBS[name].poll() is None:
            return self._reply(409, {"error": "running"})
        log = (LOGS / f"{name}.log").open("ab")
        JOBS[name] = subprocess.Popen(["bash", "-lc", spec["cmd"]], stdout=log, stderr=subprocess.STDOUT,
                                      cwd=spec.get("cwd", "/work"), start_new_session=True)
        self._reply(200, {"pid": JOBS[name].pid})

    def log_message(self, *args):
        pass


ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("AGENT_PORT", "8000"))), Handler).serve_forever()
