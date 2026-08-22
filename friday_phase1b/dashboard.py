"""Read-only loopback dashboard for Phase-1B history."""

from __future__ import annotations

import json
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .constants import (
    DEFAULT_DASHBOARD_PORT,
    EXPERIMENT_ID,
    MAX_DASHBOARD_BYTES,
    SCHEMA_VERSION,
)
from .history import History, HistoryError, snapshot_revision


class DashboardError(RuntimeError):
    """A Phase-1B dashboard request cannot be served read-only."""


class DashboardService:
    def __init__(self, database: str | Path) -> None:
        self.database = Path(database)

    def snapshot(self, limit: int = 2) -> dict[str, object]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 2:
            raise DashboardError("snapshot limit must be in 1..2")
        try:
            with History.open(self.database, read_only=True) as history:
                with history.read_transaction():
                    records = history.verified_records()
        except HistoryError as exc:
            raise DashboardError(str(exc)) from exc
        recent = []
        for row in reversed(records[-limit:]):
            report = row["report"]
            recent.append(
                {
                    "record_id": row["record_id"],
                    "previous_record_id": row["previous_record_id"],
                    "created_at_unix_ns": row["created_at_unix_ns"],
                    "run_id": report["run_id"],
                    "kind": report["kind"],
                    "status": report["status"],
                    "action": report["action"],
                    "formal_claim": False,
                    "scope": report["scope"],
                    "metrics": report["metrics"],
                }
            )
        return {
            "experiment_id": EXPERIMENT_ID,
            "schema_version": SCHEMA_VERSION,
            "database": "phase1b_rmsnorm",
            "read_only": True,
            "hash_chain_verified": True,
            "total": len(records),
            "by_kind": dict(Counter(row["report"]["kind"] for row in records)),
            "by_status": dict(Counter(row["report"]["status"] for row in records)),
            "revision": snapshot_revision(records),
            "recent": recent,
        }


def _html() -> bytes:
    return b"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Friday Phase 1B RMSNorm</title>
<style>body{font:14px system-ui;margin:2rem;background:#0b1020;color:#e8eefc}code{color:#8be9fd}
table{border-collapse:collapse;width:100%}th,td{padding:.55rem;border-bottom:1px solid #29324d;text-align:left}
.ok{color:#50fa7b}.muted{color:#9aa7c7}</style></head><body>
<h1>Friday Phase 1B &middot; residual add + RMSNorm</h1><p class="muted">Read-only history; no runtime activation</p>
<div id="summary">Loading verified history...</div><table><thead><tr><th>Kind</th><th>Status</th><th>Action</th><th>Run</th><th>Record</th></tr></thead><tbody id="rows"></tbody></table>
<script>const esc=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));fetch('/api/snapshot?limit=2').then(r=>r.json()).then(s=>{document.getElementById('summary').innerHTML=`<span class="ok">hash chain verified</span> &middot; ${Number(s.total)} records &middot; <code>${esc(s.revision)}</code>`;document.getElementById('rows').innerHTML=s.recent.map(x=>`<tr><td>${esc(x.kind)}</td><td>${esc(x.status)}</td><td>${esc(x.action)}</td><td>${esc(x.run_id)}</td><td><code>${esc(x.record_id.slice(0,16))}</code></td></tr>`).join('')}).catch(e=>{document.getElementById('summary').textContent=String(e)});</script>
</body></html>"""


def _snapshot_limit(target: str) -> int:
    parsed = urlsplit(target)
    if parsed.path != "/api/snapshot":
        raise DashboardError("unknown dashboard path")
    values = parse_qs(parsed.query, strict_parsing=False)
    if set(values) - {"limit"} or len(values.get("limit", ["2"])) != 1:
        raise DashboardError("invalid snapshot query")
    try:
        return int(values.get("limit", ["2"])[0])
    except ValueError as exc:
        raise DashboardError("snapshot limit must be an integer") from exc


def _trusted_host(value: str | None, port: int) -> bool:
    return value in {f"127.0.0.1:{port}", f"localhost:{port}"}


def serve(database: str | Path, *, port: int = DEFAULT_DASHBOARD_PORT) -> None:
    if isinstance(port, bool) or not isinstance(port, int) or not 1024 <= port <= 65535:
        raise DashboardError("dashboard port must be in 1024..65535")
    service = DashboardService(database)

    class Handler(BaseHTTPRequestHandler):
        server_version = "FridayPhase1B/1"

        def _send(self, status: int, content_type: str, body: bytes, *, head: bool = False) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; connect-src 'self'; style-src 'unsafe-inline'; "
                "script-src 'unsafe-inline'; base-uri 'none'; frame-ancestors 'none'",
            )
            self.end_headers()
            if not head:
                self.wfile.write(body)

        def _read(self, *, head: bool) -> None:
            try:
                if not _trusted_host(self.headers.get("Host"), server.server_port):
                    self._send(421, "application/json", b'{"error":"untrusted-host"}', head=head)
                    return
                parsed = urlsplit(self.path)
                if parsed.path == "/" and not parsed.query:
                    self._send(200, "text/html; charset=utf-8", _html(), head=head)
                    return
                snapshot = service.snapshot(_snapshot_limit(self.path))
                body = json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode("utf-8")
                if len(body) > MAX_DASHBOARD_BYTES:
                    raise DashboardError("dashboard response exceeds its byte limit")
                self._send(200, "application/json", body, head=head)
            except DashboardError as exc:
                body = json.dumps({"error": str(exc)}, sort_keys=True).encode("utf-8")
                self._send(400, "application/json", body, head=head)

        def do_GET(self) -> None:  # noqa: N802
            self._read(head=False)

        def do_HEAD(self) -> None:  # noqa: N802
            self._read(head=True)

        def do_POST(self) -> None:  # noqa: N802
            if not _trusted_host(self.headers.get("Host"), server.server_port):
                self._send(421, "application/json", b'{"error":"untrusted-host"}')
                return
            self._send(405, "application/json", b'{"error":"read-only"}')

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    try:
        server.serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
