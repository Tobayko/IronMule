"""Read-only loopback history UI for native and legacy H1/H2 evidence."""

from __future__ import annotations

import html
import json
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, urlsplit

from .canonical import canonical_sha256
from .registry import MAX_RECENT_ROWS, MAX_RESPONSE_BYTES, REGISTERED_TOOLS
from .storage import EvidenceStorage, StorageError

LOOPBACK_HOST = "127.0.0.1"
DEFAULT_PORT = 8767
DEFAULT_LIMIT = 50
MAX_TARGET_BYTES = 2048
_HEADERS = (
    ("Cache-Control", "no-store"),
    ("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; "
     "base-uri 'none'; frame-ancestors 'none'; form-action 'none'"),
    ("Cross-Origin-Opener-Policy", "same-origin"),
    ("Cross-Origin-Resource-Policy", "same-origin"),
    ("Referrer-Policy", "no-referrer"),
    ("X-Content-Type-Options", "nosniff"),
    ("X-Frame-Options", "DENY"),
)


class DashboardError(ValueError):
    """A request exceeds the closed read-only dashboard contract."""


def _metrics(report: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        key: report[key]
        for key in ("effect_percent", "gpu_work_seconds", "wall_seconds", "mde")
        if isinstance(report.get(key), (int, float)) and type(report.get(key)) is not bool
    }
    for container in ("aggregate", "confirmation"):
        nested = report.get(container)
        if isinstance(nested, Mapping):
            for key in ("ratio", "ci_low", "ci_high"):
                if isinstance(nested.get(key), (int, float)):
                    if type(nested[key]) is not bool:
                        result[key] = nested[key]
    return result


class DashboardService:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)

    def snapshot(self, limit: int = DEFAULT_LIMIT) -> dict[str, Any]:
        if type(limit) is not int or not 1 <= limit <= MAX_RECENT_ROWS:
            raise DashboardError("limit is outside the registered range")
        with EvidenceStorage.open(self.database_path, read_only=True) as storage:
            with storage.read_transaction():
                rows = storage.verified_rows()
        recent = [
            {
                "record_id": row["record_id"],
                "recorded_at_unix_ns": row["recorded_at_unix_ns"],
                "observed_at_unix_ns": row["observed_at_unix_ns"],
                "evidence_kind": row["evidence_kind"],
                "tool": row["tool"],
                "result_status": row["result_status"],
                "raw_measurements_available": bool(row["raw_measurements_available"]),
                "metrics": _metrics(row["report"]),
            }
            for row in rows[:limit]
        ]
        revision_material = [
            (row["record_id"], row["report_sha256"], row["provenance_sha256"])
            for row in rows
        ]
        return {
            "schema_version": 1,
            "database": "research",
            "read_only": True,
            "revision": canonical_sha256(revision_material),
            "total": len(rows),
            "native": sum(row["evidence_kind"] == "native" for row in rows),
            "legacy_summary": sum(row["evidence_kind"] == "legacy_summary" for row in rows),
            "with_raw_measurements": sum(bool(row["raw_measurements_available"]) for row in rows),
            "by_tool": {name: Counter(row["tool"] for row in rows)[name] for name in REGISTERED_TOOLS},
            "by_status": dict(sorted(Counter(row["result_status"] for row in rows).items())),
            "recent": recent,
        }

    def detail(self, record_id: str) -> dict[str, Any] | None:
        with EvidenceStorage.open(self.database_path, read_only=True) as storage:
            with storage.read_transaction():
                row = storage.get_verified(record_id)
        if row is None:
            return None
        return {
            "record_id": row["record_id"],
            "evidence_kind": row["evidence_kind"],
            "source_key": row["source_key"],
            "tool": row["tool"],
            "workload_key": row["workload_key"],
            "result_status": row["result_status"],
            "raw_measurements_available": bool(row["raw_measurements_available"]),
            "observed_at_unix_ns": row["observed_at_unix_ns"],
            "recorded_at_unix_ns": row["recorded_at_unix_ns"],
            "report_sha256": row["report_sha256"],
            "provenance_sha256": row["provenance_sha256"],
            "report": row["report"],
            "provenance": row["provenance"],
        }


def _target(value: str) -> tuple[str, dict[str, list[str]]]:
    if not isinstance(value, str) or len(value.encode("ascii", errors="strict")) > MAX_TARGET_BYTES:
        raise DashboardError("invalid request target")
    split = urlsplit(value)
    if split.scheme or split.netloc or split.fragment:
        raise DashboardError("absolute or fragmented targets are forbidden")
    try:
        query = parse_qs(
            split.query, keep_blank_values=True, strict_parsing=True, max_num_fields=2
        )
    except ValueError as exc:
        raise DashboardError("malformed query") from exc
    return split.path, query


def _html(snapshot: Mapping[str, Any]) -> bytes:
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(item['recorded_at_unix_ns']))}</td>"
        f"<td>{html.escape(item['tool'])}</td>"
        f"<td>{html.escape(item['evidence_kind'])}</td>"
        f"<td>{html.escape(item['result_status'])}</td>"
        f"<td>{'yes' if item['raw_measurements_available'] else 'no'}</td>"
        f"<td><code>{html.escape(item['record_id'][:12])}</code></td></tr>"
        for item in snapshot["recent"]
    ) or "<tr><td colspan='6'>No research evidence persisted.</td></tr>"
    document = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width'><title>Friday research evidence</title>
<style>body{{font:15px system-ui;margin:2rem;max-width:90rem;color:#17212b;background:#f6f8fa}}
.card{{background:#fff;border:1px solid #d8dee4;border-radius:10px;padding:1rem;margin:1rem 0}}
table{{width:100%;border-collapse:collapse}}th,td{{padding:.55rem;border-bottom:1px solid #ddd;text-align:left}}
code{{font-size:.85em}}.muted{{color:#57606a}}</style></head><body>
<h1>Friday research evidence</h1><div class='card'><strong>{snapshot['total']}</strong> records —
{snapshot['native']} native, {snapshot['legacy_summary']} legacy summaries, {snapshot['with_raw_measurements']} with raw measurements.
<div class='muted'>Append-only SQLite v1; this UI opens it read-only. Revision <code>{snapshot['revision']}</code>.</div></div>
<div class='card'><h2>History</h2><table><thead><tr><th>Recorded ns</th><th>Tool</th><th>Kind</th><th>Status</th><th>Raw</th><th>ID</th></tr></thead><tbody>{rows}</tbody></table></div>
</body></html>"""
    return document.encode("utf-8")


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "FridayEvidence/1"
    sys_version = ""

    @property
    def service(self) -> DashboardService:
        service = getattr(self.server, "dashboard_service", None)
        if not isinstance(service, DashboardService):
            raise RuntimeError("dashboard service missing")
        return service

    def _send(self, status: int, kind: str, payload: bytes, *, head: bool) -> None:
        if len(payload) > MAX_RESPONSE_BYTES:
            raise DashboardError("response exceeds byte limit")
        self.send_response(status)
        for key, value in _HEADERS:
            self.send_header(key, value)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if not head:
            self.wfile.write(payload)

    def _json(self, status: int, value: Mapping[str, Any], *, head: bool) -> None:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
        self._send(status, "application/json; charset=utf-8", payload, head=head)

    def _dispatch(self, *, head: bool) -> None:
        try:
            path, query = _target(self.path)
            if path == "/" and not query:
                self._send(200, "text/html; charset=utf-8", _html(self.service.snapshot()), head=head)
                return
            if path == "/api/snapshot":
                if set(query) - {"limit"} or any(len(values) != 1 for values in query.values()):
                    raise DashboardError("invalid snapshot query")
                raw = query.get("limit", [str(DEFAULT_LIMIT)])[0]
                if not raw.isascii() or not raw.isdecimal():
                    raise DashboardError("limit must be decimal")
                self._json(200, self.service.snapshot(int(raw)), head=head)
                return
            if path == "/api/detail" and set(query) == {"id"} and len(query["id"]) == 1:
                detail = self.service.detail(query["id"][0])
                self._json(200 if detail else 404, detail or {"error": "not_found"}, head=head)
                return
            self._json(404, {"error": "not_found"}, head=head)
        except (DashboardError, StorageError, UnicodeError):
            self._json(400, {"error": "invalid_request"}, head=head)

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch(head=False)

    def do_HEAD(self) -> None:  # noqa: N802
        self._dispatch(head=True)

    def _method_not_allowed(self) -> None:
        self._json(405, {"error": "method_not_allowed"}, head=False)

    def do_POST(self) -> None:  # noqa: N802
        self._method_not_allowed()

    def do_PUT(self) -> None:  # noqa: N802
        self._method_not_allowed()

    def do_PATCH(self) -> None:  # noqa: N802
        self._method_not_allowed()

    def do_DELETE(self) -> None:  # noqa: N802
        self._method_not_allowed()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._method_not_allowed()

    def log_message(self, _format: str, *_args: object) -> None:
        return


def serve(database_path: str | Path, *, port: int = DEFAULT_PORT) -> None:
    if type(port) is not int or not 1024 <= port <= 65535:
        raise DashboardError("port must be between 1024 and 65535")
    service = DashboardService(database_path)
    service.snapshot(limit=1)
    server = ThreadingHTTPServer((LOOPBACK_HOST, port), DashboardHandler)
    server.dashboard_service = service  # type: ignore[attr-defined]
    try:
        server.serve_forever()
    finally:
        server.server_close()


__all__ = ["DashboardError", "DashboardHandler", "DashboardService", "serve"]
