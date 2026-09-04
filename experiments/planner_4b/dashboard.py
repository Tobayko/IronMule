#!/usr/bin/env python3
"""Read-only loopback UI for the Gemma-4B planner study."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

RESULT_PATH = Path(__file__).with_name("results.json")
MAX_RESULT_BYTES = 2_000_000


class DashboardError(RuntimeError):
    """The stored result cannot be exposed as verified read-only history."""


def _load_result(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise DashboardError("result is unavailable")
    payload = path.read_bytes()
    if not payload or len(payload) > MAX_RESULT_BYTES:
        raise DashboardError("result size is invalid")
    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {item}")
            ),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise DashboardError("result is invalid JSON") from exc
    if (
        not isinstance(value, dict)
        or value.get("study_id") != "gemma-4b-evidence-planner-20260824-01"
        or value.get("formal_claim") is not False
        or not isinstance(value.get("decision"), str)
        or not isinstance(value.get("runs"), list)
        or len(value["runs"]) > 3
    ):
        raise DashboardError("result identity is invalid")
    return value


def snapshot(path: Path = RESULT_PATH) -> dict[str, Any]:
    """Return a small projection without raw model text or token arrays."""

    result = _load_result(path)
    rows: list[dict[str, Any]] = []
    for index, run in enumerate(result["runs"]):
        if not isinstance(run, dict):
            raise DashboardError("run history is invalid")
        rows.append(
            {
                "candidate_id": run.get("candidate_id"),
                "compute_seconds": (
                    run.get("compute_ns") / 1_000_000_000
                    if type(run.get("compute_ns")) is int
                    else None
                ),
                "finish_reason": run.get("finish_reason"),
                "process_wall_seconds": (
                    run.get("process_wall_ns") / 1_000_000_000
                    if type(run.get("process_wall_ns")) is int
                    else None
                ),
                "run": index + 1,
                "token_sha256": run.get("token_sha256"),
            }
        )
    return {
        "decision": result["decision"],
        "formal_claim": False,
        "gates": result.get("gates"),
        "metrics": result.get("metrics"),
        "read_only": True,
        "resources": result.get("resources"),
        "rows": rows,
        "run_id": result.get("run_id"),
        "study_id": result.get("study_id"),
    }


HTML = b"""<!doctype html><html lang='de'><meta charset='utf-8'>
<meta name='viewport' content='width=device-width'><title>Friday 4B-Planer</title>
<style>body{font:15px system-ui;max-width:1000px;margin:30px auto;padding:0 16px;background:#10151b;color:#e9eef5}
.card{background:#17202a;border:1px solid #314052;border-radius:10px;padding:16px;margin:14px 0}
table{border-collapse:collapse;width:100%}th,td{padding:8px;border-bottom:1px solid #314052;text-align:left}
.muted{color:#9fb0c3}code{color:#8bd5ff}</style>
<h1>Gemma-4B-Planertest</h1><p class='muted'>Gepruefte, nur lesbare lokale Ergebnisansicht</p>
<div class='card' id='summary'>Wird geladen...</div><div class='card'><table><thead><tr>
<th>Lauf</th><th>Auswahl</th><th>Abschluss</th><th>Rechenzeit</th><th>Gesamtzeit</th><th>Antwort-Fingerabdruck</th>
</tr></thead><tbody id='rows'></tbody></table></div>
<script>const summary=document.getElementById('summary');const rows=document.getElementById('rows');
const add=(tr,value)=>{const td=document.createElement('td');td.textContent=String(value ?? 'n/a');tr.appendChild(td)};
fetch('/api/snapshot').then(r=>r.json()).then(x=>{summary.textContent=`Ergebnis: ${x.decision} | Laeufe: ${x.metrics?.runs_completed ?? 0}`;
for(const item of Array.isArray(x.rows)?x.rows:[]){const tr=document.createElement('tr');
for(const value of [item.run,item.candidate_id,item.finish_reason,item.compute_seconds,item.process_wall_seconds,item.token_sha256])add(tr,value);rows.appendChild(tr)}})
.catch(e=>summary.textContent=`Nicht verfuegbar: ${String(e)}`)</script></html>"""


def _host_header_allowed(value: str | None, bound_host: str, bound_port: int) -> bool:
    if bound_host == "127.0.0.1":
        allowed = {"127.0.0.1", f"127.0.0.1:{bound_port}"}
    elif bound_host == "::1":
        allowed = {"[::1]", f"[::1]:{bound_port}"}
    else:
        return False
    return value in allowed


def _handler(result_path: Path) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _headers(self, status: int, content_type: str, length: int) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'unsafe-inline'")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            address = self.server.server_address
            if not _host_header_allowed(
                self.headers.get("Host"), str(address[0]), int(address[1])
            ):
                payload = b'{"error":"host not allowed"}'
                self._headers(421, "application/json; charset=utf-8", len(payload))
                self.wfile.write(payload)
                return
            path = urlsplit(self.path).path
            if path == "/":
                self._headers(200, "text/html; charset=utf-8", len(HTML))
                self.wfile.write(HTML)
                return
            if path == "/api/snapshot":
                try:
                    payload = json.dumps(
                        snapshot(result_path), allow_nan=False, sort_keys=True
                    ).encode("utf-8")
                    status = 200
                except DashboardError as exc:
                    payload = json.dumps({"error": str(exc)}, sort_keys=True).encode("utf-8")
                    status = 503
                self._headers(status, "application/json; charset=utf-8", len(payload))
                self.wfile.write(payload)
                return
            payload = b'{"error":"not found"}'
            self._headers(404, "application/json; charset=utf-8", len(payload))
            self.wfile.write(payload)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="planner_4b_dashboard", allow_abbrev=False)
    parser.add_argument("--host", default="127.0.0.1", choices=("127.0.0.1", "::1"))
    parser.add_argument("--port", type=int, default=8783)
    parser.add_argument("--result", type=Path, default=RESULT_PATH)
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535:
        parser.error("port must be between 1024 and 65535")
    server = ThreadingHTTPServer((args.host, args.port), _handler(args.result))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
