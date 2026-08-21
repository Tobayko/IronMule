"""Small read-only loopback UI for runtime policy and validation history."""

from __future__ import annotations

import html
import json
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, urlsplit

from friday_h1.canonical import canonical_sha256

from .constants import (
    DEFAULT_DASHBOARD_PORT,
    MAX_HISTORY_ROWS,
    MAX_RESPONSE_BYTES,
    MAX_TARGET_BYTES,
    RUNTIME_ID,
    SCHEMA_VERSION,
)
from .history import History, HistoryError

LOOPBACK_HOST = "127.0.0.1"
DEFAULT_LIMIT = 100
_HEADERS = (
    ("Cache-Control", "no-store"),
    (
        "Content-Security-Policy",
        "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; "
        "frame-ancestors 'none'; form-action 'none'",
    ),
    ("Cross-Origin-Opener-Policy", "same-origin"),
    ("Cross-Origin-Resource-Policy", "same-origin"),
    ("Referrer-Policy", "no-referrer"),
    ("X-Content-Type-Options", "nosniff"),
    ("X-Frame-Options", "DENY"),
)


class DashboardError(ValueError):
    """A request is outside the bounded read-only UI contract."""


def _metrics(report: Mapping[str, Any]) -> dict[str, int | float | bool]:
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        return {}
    names = (
        "baseline_median_ns",
        "candidate_median_ns",
        "policy_median_ns",
        "policy_p95_ns",
        "incremental_median_ns",
        "ratio",
        "effect_percent",
        "max_abs_error",
        "byte_identical",
        "gate_passed",
    )
    return {
        name: metrics[name]
        for name in names
        if type(metrics.get(name)) in {int, float, bool}
    }


class DashboardService:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)

    def snapshot(self, limit: int = DEFAULT_LIMIT) -> dict[str, Any]:
        if type(limit) is not int or not 1 <= limit <= MAX_HISTORY_ROWS:
            raise DashboardError("limit is outside the registered range")
        with History.open(self.database_path, read_only=True) as history:
            with history.read_transaction():
                rows = history.verified_records()
        recent = [
            {
                "record_id": row["record_id"],
                "previous_record_id": row["previous_record_id"],
                "created_at_unix_ns": row["created_at_unix_ns"],
                "kind": row["record_kind"],
                "status": row["status"],
                "run_id": row["report"]["run_id"],
                "metrics": _metrics(row["report"]),
                "policy": row["report"].get("policy"),
            }
            for row in reversed(rows[-limit:])
        ]
        revision = canonical_sha256(
            [
                [
                    row["record_id"],
                    row["previous_record_id"],
                    row["report_sha256"],
                    row["provenance_sha256"],
                ]
                for row in rows
            ]
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "runtime_id": RUNTIME_ID,
            "database": "runtime",
            "read_only": True,
            "hash_chain_verified": True,
            "revision": revision,
            "total": len(rows),
            "by_kind": dict(sorted(Counter(row["record_kind"] for row in rows).items())),
            "by_status": dict(sorted(Counter(row["status"] for row in rows).items())),
            "recent": recent,
        }

    def detail(self, record_id: str) -> dict[str, Any] | None:
        if (
            not isinstance(record_id, str)
            or len(record_id) != 64
            or any(character not in "0123456789abcdef" for character in record_id)
        ):
            raise DashboardError("record ID is invalid")
        with History.open(self.database_path, read_only=True) as history:
            with history.read_transaction():
                rows = history.verified_records()
        row = next((item for item in rows if item["record_id"] == record_id), None)
        if row is None:
            return None
        return row


def _target(value: str) -> tuple[str, dict[str, list[str]]]:
    try:
        encoded = value.encode("ascii", errors="strict")
    except (AttributeError, UnicodeEncodeError) as exc:
        raise DashboardError("request target is not bounded ASCII") from exc
    if len(encoded) > MAX_TARGET_BYTES:
        raise DashboardError("request target exceeds its byte limit")
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
    def metric_text(item: Mapping[str, Any]) -> str:
        metrics = item.get("metrics")
        if not isinstance(metrics, Mapping) or not metrics:
            return "—"
        return ", ".join(f"{name}={value}" for name, value in metrics.items())

    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(item['created_at_unix_ns']))}</td>"
        f"<td>{html.escape(item['kind'])}</td>"
        f"<td>{html.escape(item['status'])}</td>"
        f"<td>{html.escape(item['run_id'])}</td>"
        f"<td>{html.escape(metric_text(item))}</td>"
        f"<td><code>{html.escape(item['record_id'][:12])}</code></td></tr>"
        for item in snapshot["recent"]
    ) or "<tr><td colspan='6'>No runtime measurements persisted yet.</td></tr>"
    document = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width'><title>Friday runtime history</title>
<style>body{{font:15px system-ui;margin:2rem;max-width:100rem;color:#17212b;background:#f6f8fa}}
.card{{background:#fff;border:1px solid #d8dee4;border-radius:10px;padding:1rem;margin:1rem 0}}
table{{width:100%;border-collapse:collapse}}th,td{{padding:.55rem;border-bottom:1px solid #ddd;text-align:left}}
code{{font-size:.85em}}.muted{{color:#57606a}}</style></head><body>
<h1>Friday bounded runtime</h1><div class='card'><strong>{snapshot['total']}</strong> records.
<div class='muted'>Append-only, hash-chained SQLite; this UI opens it read-only.
Runtime <code>{html.escape(snapshot['runtime_id'])}</code>. Revision
<code>{snapshot['revision']}</code>.</div></div>
<div class='card'><h2>Measurement history</h2><table><thead><tr><th>Recorded ns</th>
<th>Kind</th><th>Status</th><th>Run</th><th>Metrics</th><th>ID</th></tr></thead>
<tbody>{rows}</tbody></table></div></body></html>"""
    payload = document.encode("utf-8")
    if len(payload) > MAX_RESPONSE_BYTES:
        raise DashboardError("HTML response exceeds its byte limit")
    return payload


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "FridayRuntime/1"
    sys_version = ""

    @property
    def service(self) -> DashboardService:
        service = getattr(self.server, "dashboard_service", None)
        if not isinstance(service, DashboardService):
            raise RuntimeError("dashboard service missing")
        return service

    def _send(self, status: int, kind: str, payload: bytes, *, head: bool) -> None:
        if len(payload) > MAX_RESPONSE_BYTES:
            raise DashboardError("response exceeds its byte limit")
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
            if path == "/":
                if query:
                    raise DashboardError("HTML endpoint takes no query")
                self._send(
                    200,
                    "text/html; charset=utf-8",
                    _html(self.service.snapshot()),
                    head=head,
                )
                return
            if path == "/api/snapshot":
                if set(query) - {"limit"} or len(query.get("limit", [])) > 1:
                    raise DashboardError("snapshot query is invalid")
                limit = int(query["limit"][0]) if "limit" in query else DEFAULT_LIMIT
                self._json(200, self.service.snapshot(limit), head=head)
                return
            if path == "/api/detail":
                if set(query) != {"id"} or len(query["id"]) != 1:
                    raise DashboardError("detail query is invalid")
                detail = self.service.detail(query["id"][0])
                self._json(200 if detail else 404, detail or {"error": "not_found"}, head=head)
                return
            self._json(404, {"error": "not_found"}, head=head)
        except (DashboardError, HistoryError, ValueError):
            self._json(400, {"error": "invalid_request"}, head=head)

    def do_GET(self) -> None:
        self._dispatch(head=False)

    def do_HEAD(self) -> None:
        self._dispatch(head=True)

    def _method_not_allowed(self) -> None:
        self._json(405, {"error": "method_not_allowed"}, head=False)

    do_POST = _method_not_allowed
    do_PUT = _method_not_allowed
    do_PATCH = _method_not_allowed
    do_DELETE = _method_not_allowed
    do_OPTIONS = _method_not_allowed

    def log_message(self, _format: str, *_args: object) -> None:
        return None


def serve(database_path: str | Path, *, port: int = DEFAULT_DASHBOARD_PORT) -> None:
    if type(port) is not int or not 1024 <= port <= 65535:
        raise DashboardError("port is outside the registered range")
    service = DashboardService(database_path)
    service.snapshot(limit=1)
    server = ThreadingHTTPServer((LOOPBACK_HOST, port), DashboardHandler)
    server.daemon_threads = True
    server.dashboard_service = service  # type: ignore[attr-defined]
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()


__all__ = [
    "DashboardError",
    "DashboardHandler",
    "DashboardService",
    "serve",
]
