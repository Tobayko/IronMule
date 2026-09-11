"""Read-only loopback history UI for the formal N10-v1 study."""

from __future__ import annotations

import html
import json
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, urlsplit

from .canonical import canonical_sha256
from .constants import MAX_HISTORY_ROWS, MAX_RESPONSE_BYTES, SCHEMA_VERSION, STUDY_ID
from .storage import Storage, StorageError

LOOPBACK_HOST = "127.0.0.1"
DEFAULT_PORT = 8770
DEFAULT_LIMIT = 50
MAX_TARGET_BYTES = 2048
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
    """A request exceeds the closed read-only dashboard contract."""


def _metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    metrics = payload.get("metrics")
    if isinstance(metrics, Mapping):
        for name in ("ratio", "effect_percent", "median_a_ns", "median_b_ns"):
            if type(metrics.get(name)) in {int, float}:
                result[name] = metrics[name]
    for name in ("mde", "effect_percent", "raw_mde"):
        if type(payload.get(name)) in {int, float}:
            result[name] = payload[name]
    aggregate = payload.get("aggregate")
    if isinstance(aggregate, Mapping):
        for name in ("ratio", "ci_low", "ci_high"):
            if type(aggregate.get(name)) in {int, float}:
                result[f"aggregate_{name}"] = aggregate[name]
    intervals = payload.get("intervals")
    if isinstance(intervals, Mapping) and isinstance(intervals.get("all"), Mapping):
        for name in ("ratio", "ci_low", "ci_high"):
            if type(intervals["all"].get(name)) in {int, float}:
                result[f"all_{name}"] = intervals["all"][name]
    return result


class DashboardService:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)

    def snapshot(self, limit: int = DEFAULT_LIMIT) -> dict[str, Any]:
        if type(limit) is not int or not 1 <= limit <= MAX_HISTORY_ROWS:
            raise DashboardError("limit is outside the registered range")
        with Storage.open(self.database_path, read_only=True) as storage:
            with storage.read_transaction():
                rows = storage.verified_records()
        recent = [
            {
                "record_id": row["record_id"],
                "created_at_unix_ns": row["created_at_unix_ns"],
                "kind": row["record_kind"],
                "stage": row["stage"],
                "session_id": row["session_id"] or None,
                "status": row["status"],
                "formal_claim": row["formal_claim"],
                "metrics": _metrics(row["payload"]),
            }
            for row in reversed(rows[-limit:])
        ]
        revision = canonical_sha256(
            [[row["record_id"], row["payload_sha256"], row["provenance_sha256"]] for row in rows]
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "study_id": STUDY_ID,
            "database": "n10-v1",
            "read_only": True,
            "revision": revision,
            "total": len(rows),
            "formal_claims": sum(bool(row["formal_claim"]) for row in rows),
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
        with Storage.open(self.database_path, read_only=True) as storage:
            with storage.read_transaction():
                rows = storage.verified_records()
        row = next((item for item in rows if item["record_id"] == record_id), None)
        if row is None:
            return None
        return {
            "record_id": row["record_id"],
            "entity_key": row["entity_key"],
            "kind": row["record_kind"],
            "stage": row["stage"],
            "session_id": row["session_id"] or None,
            "status": row["status"],
            "formal_claim": row["formal_claim"],
            "created_at_unix_ns": row["created_at_unix_ns"],
            "payload_sha256": row["payload_sha256"],
            "provenance_sha256": row["provenance_sha256"],
            "payload": row["payload"],
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
    def metric_text(item: Mapping[str, Any]) -> str:
        metrics = item.get("metrics")
        if not isinstance(metrics, Mapping) or not metrics:
            return "—"
        selected = []
        for name in (
            "ratio",
            "aggregate_ratio",
            "all_ratio",
            "effect_percent",
            "mde",
            "aggregate_ci_low",
            "aggregate_ci_high",
            "all_ci_low",
            "all_ci_high",
        ):
            if name in metrics:
                selected.append(f"{name}={metrics[name]}")
        return ", ".join(selected) or "—"

    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(item['created_at_unix_ns']))}</td>"
        f"<td>{html.escape(item['kind'])}</td>"
        f"<td>{html.escape(str(item['session_id'] or '—'))}</td>"
        f"<td>{html.escape(item['status'])}</td>"
        f"<td>{html.escape(metric_text(item))}</td>"
        f"<td>{'yes' if item['formal_claim'] else 'no'}</td>"
        f"<td><code>{html.escape(item['record_id'][:12])}</code></td></tr>"
        for item in snapshot["recent"]
    ) or "<tr><td colspan='7'>No formal N10-v1 evidence persisted.</td></tr>"
    document = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width'><title>Friday N10-v1 evidence</title>
<style>body{{font:15px system-ui;margin:2rem;max-width:90rem;color:#17212b;background:#f6f8fa}}
.card{{background:#fff;border:1px solid #d8dee4;border-radius:10px;padding:1rem;margin:1rem 0}}
table{{width:100%;border-collapse:collapse}}th,td{{padding:.55rem;border-bottom:1px solid #ddd;text-align:left}}
code{{font-size:.85em}}.muted{{color:#57606a}}</style></head><body>
<h1>Friday N10-v1 formal evidence</h1><div class='card'><strong>{snapshot['total']}</strong> records —
{snapshot['formal_claims']} formal terminal claims.
<div class='muted'>Append-only SQLite v1; this UI opens it read-only. Study
<code>{html.escape(snapshot['study_id'])}</code>. Revision <code>{snapshot['revision']}</code>.</div></div>
<div class='card'><h2>History</h2><table><thead><tr><th>Recorded ns</th><th>Kind</th>
<th>Session</th><th>Status</th><th>Metrics</th><th>Formal</th><th>ID</th></tr></thead><tbody>{rows}</tbody></table></div>
</body></html>"""
    return document.encode("utf-8")


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "FridayN10/1"
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

    do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = _method_not_allowed  # type: ignore[assignment]

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
