"""Read-only loopback dashboard for the separate H0.1 evidence database."""

from __future__ import annotations

import html
import json
from collections.abc import Mapping, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from .canonical import CanonicalError, canonical_sha256, nonnegative_int64, positive_int64
from .storage import (
    ENTITY_KINDS,
    ENTITY_STATUSES,
    MAX_DASHBOARD_ROWS,
    BundleError,
    Storage,
    StorageError,
    validate_entity_id,
)

DASHBOARD_SCHEMA_VERSION = 1
LOOPBACK_HOST = "127.0.0.1"
DEFAULT_RECENT_LIMIT = 50
MAX_PATH_BYTES = 2048
MAX_QUERY_BYTES = 1024
MAX_RESPONSE_BYTES = 1 * 1024 * 1024

_ALL_STATUSES = tuple(sorted(set().union(*ENTITY_STATUSES.values())))
_SESSION_MANIFEST_VIEW_FIELDS = (
    "schema_version",
    "phase",
    "study",
    "run_id",
    "session",
    "schedule",
    "budgets",
    "gates",
    "fixture",
    "study_spec_sha256",
    "code_sha256",
    "environment_sha256",
)
_SESSION_TRACE_VIEW_FIELDS = (
    "schema_version",
    "phase",
    "study",
    "run_id",
    "manifest_sha256",
    "session_id",
    "schedule_sha256",
    "fixture",
    "study_spec_sha256",
    "code_sha256",
    "environment_sha256",
    "cooldown",
    "telemetry",
)
_SESSION_RESULT_VIEW_FIELDS = (
    "schema_version",
    "phase",
    "study",
    "run_id",
    "manifest_sha256",
    "trace_sha256",
    "status",
    "conclusion",
    "action",
    "h0_reclassification",
    "promotion_applicable",
    "error",
    "sample_accounting",
    "metrics",
    "gates",
    "decision_sha256",
)
_STUDY_RESULT_VIEW_FIELDS = (
    "schema_version",
    "phase",
    "study",
    "study_id",
    "session_order",
    "session_count",
    "failed_gate_count",
    "status",
    "conclusion",
    "action",
    "h0_reclassification",
    "promotion_applicable",
    "error",
    "decision_sha256",
)
_PROVENANCE_VIEW_FIELDS = (
    "study_spec_sha256",
    "code_sha256",
    "environment_sha256",
    "fixture",
)
_LEGACY_MANIFEST_VIEW_FIELDS = (
    "schema_version",
    "entity_id",
    "source_phase",
    "source_mode",
    "source_status",
    "source_classification",
    "source_created_at_unix_ns",
    "observation_kind",
    "adapter",
    "registry_schema_version",
    "registry_sha256",
    "descriptor_sha256",
    "selector_sha256",
    "parser_id",
    "raw_warmup_sha256",
)
_LEGACY_RESULT_VIEW_FIELDS = (
    "schema_version",
    "status",
    "conclusion",
    "interpretation",
    "action",
    "stationarity_supported",
    "paced_gate_applicable",
    "h0_reclassification",
    "promotion_applicable",
)
_SECURITY_HEADERS = (
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
    """A bounded dashboard request is invalid."""


def _limit(value: Any, name: str = "limit") -> int:
    try:
        return positive_int64(value, name, maximum=MAX_DASHBOARD_ROWS)
    except CanonicalError as exc:
        raise DashboardError(str(exc)) from exc


def _bounded_list(value: Any) -> tuple[list[Any], bool]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return [], False
    rows = list(value[:MAX_DASHBOARD_ROWS])
    return rows, len(value) > MAX_DASHBOARD_ROWS


def _project(value: Any, fields: Sequence[str], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(field not in value for field in fields):
        raise DashboardError(f"verified {name} cannot be projected")
    return {field: value[field] for field in fields}


class DashboardService:
    """No-write query service; every call opens and closes a verified mode=ro handle."""

    def __init__(self, database_path: str | Path) -> None:
        self._database_path = database_path

    def snapshot(self, *, limit: Any = DEFAULT_RECENT_LIMIT) -> dict[str, Any]:
        checked_limit = _limit(limit)
        with Storage.open(self._database_path, read_only=True) as storage:
            with storage.read_transaction() as schema_snapshot:
                verified_rows = storage.verified_rows()
                total = len(verified_rows)
                by_kind = {name: 0 for name in sorted(ENTITY_KINDS)}
                by_status = {name: 0 for name in _ALL_STATUSES}
                bundles = [row["bundle"] for row in verified_rows]
                for bundle in bundles:
                    by_kind[bundle["entity_kind"]] += 1
                    by_status[bundle["status"]] += 1
                recent_bundles = sorted(
                    bundles,
                    key=lambda bundle: (
                        bundle["created_at_unix_ns"],
                        bundle["entity_id"],
                    ),
                    reverse=True,
                )[:checked_limit]
                recent = [
                    {
                        "entity_id": bundle["entity_id"],
                        "entity_kind": bundle["entity_kind"],
                        "status": bundle["status"],
                        "created_at_unix_ns": bundle["created_at_unix_ns"],
                        "bundle_sha256": bundle["bundle_sha256"],
                    }
                    for bundle in recent_bundles
                ]
                revision_material = storage.revision_material(
                    verified_rows, schema_snapshot
                )
        return {
            "schema_version": DASHBOARD_SCHEMA_VERSION,
            "database": "h01",
            "snapshot_scope": "single_h01_database_read_transaction",
            "cross_database_atomicity": False,
            "revision": canonical_sha256(revision_material),
            "total": total,
            "by_kind": by_kind,
            "by_status": by_status,
            "recent": recent,
        }

    def detail(self, entity_id: Any) -> dict[str, Any] | None:
        try:
            checked_id = validate_entity_id(entity_id)
        except BundleError as exc:
            raise DashboardError(str(exc)) from exc
        with Storage.open(self._database_path, read_only=True) as storage:
            with storage.read_transaction():
                bundle = storage.get_verified_bundle(checked_id)
        if bundle is None:
            return None

        kind = bundle["entity_kind"]
        manifest = bundle["manifest"]
        trace = bundle["trace"]
        result = bundle["result"]
        records: list[Any] = []
        records_truncated = False
        trace_points: list[Any] = []
        trace_points_truncated = False
        manifest_summary: Any
        trace_summary: Any
        result_summary: Any

        if kind == "paced_session":
            samples = trace.get("samples") if isinstance(trace, Mapping) else None
            trace_points, trace_points_truncated = _bounded_list(samples)
            manifest_summary = _project(
                manifest, _SESSION_MANIFEST_VIEW_FIELDS, "session manifest"
            )
            trace_summary = _project(trace, _SESSION_TRACE_VIEW_FIELDS, "session trace")
            result_summary = _project(
                result, _SESSION_RESULT_VIEW_FIELDS, "session result"
            )
        elif kind == "paced_study":
            source_records = manifest.get("session_records") if isinstance(manifest, Mapping) else None
            session_bindings = (
                result.get("session_bindings") if isinstance(result, Mapping) else None
            )
            records, records_truncated = _bounded_list(session_bindings)
            manifest_summary = {
                "session_record_count": len(source_records)
                if isinstance(source_records, Sequence)
                and not isinstance(source_records, (str, bytes, bytearray))
                else 0
            }
            trace_summary = {
                "session_binding_count": len(trace.get("session_bindings", []))
                if isinstance(trace, Mapping)
                and isinstance(trace.get("session_bindings"), Sequence)
                else 0
            }
            result_summary = _project(
                result, _STUDY_RESULT_VIEW_FIELDS, "study result"
            )
            shared_provenance = result.get("shared_provenance")
            result_summary["shared_provenance"] = _project(
                shared_provenance,
                _PROVENANCE_VIEW_FIELDS,
                "study shared provenance",
            )
        else:
            observation = trace.get("observation") if isinstance(trace, Mapping) else None
            if not isinstance(observation, Mapping):
                raise DashboardError("verified legacy observation cannot be projected")
            warmups = observation.get("warmup_ns")
            bounded_observations, records_truncated = _bounded_list(warmups)
            records = [
                {
                    "index": index,
                    "warmup_ns": value,
                    "observation_sha256": canonical_sha256(
                        {"index": index, "warmup_ns": value}
                    ),
                }
                for index, value in enumerate(bounded_observations)
            ]
            manifest_summary = _project(
                manifest, _LEGACY_MANIFEST_VIEW_FIELDS, "legacy manifest"
            )
            trace_summary = {
                "adapter": observation["adapter"],
                "source_status": observation["source_status"],
                "source_classification": observation["source_classification"],
                "source_error_code": observation["source_error_code"],
                "registry_schema_version": observation["registry_schema_version"],
                "registry_sha256": observation["registry_sha256"],
                "descriptor_sha256": observation["descriptor_sha256"],
                "selector_sha256": observation["selector_sha256"],
                "parser_id": observation["parser_id"],
                "raw_warmup_sha256": observation["raw_warmup_sha256"],
                "warmup_count": len(warmups),
                "statistics": observation["statistics"],
                "source_diagnostic_present": observation["source_diagnostic"] is not None,
            }
            result_summary = _project(
                result, _LEGACY_RESULT_VIEW_FIELDS, "legacy result"
            )

        return {
            "schema_version": DASHBOARD_SCHEMA_VERSION,
            "database": "h01",
            "snapshot_scope": "single_h01_database_read_transaction",
            "cross_database_atomicity": False,
            "entity_id": bundle["entity_id"],
            "entity_kind": kind,
            "status": bundle["status"],
            "action": bundle["action"],
            "created_at_unix_ns": bundle["created_at_unix_ns"],
            "hashes": {
                "manifest_sha256": bundle["manifest_sha256"],
                "trace_sha256": bundle["trace_sha256"],
                "result_sha256": bundle["result_sha256"],
                "lineage_sha256": bundle["lineage_sha256"],
                "bundle_sha256": bundle["bundle_sha256"],
            },
            "manifest": manifest_summary,
            "trace": trace_summary,
            "result": result_summary,
            "records": records,
            "records_truncated": records_truncated,
            "trace_points": trace_points,
            "trace_points_truncated": trace_points_truncated,
            "parent_h0_lineage": bundle["lineage"],
        }


def _parse_request_target(target: Any) -> tuple[str, dict[str, list[str]]]:
    if not isinstance(target, str):
        raise DashboardError("request target must be text")
    try:
        encoded = target.encode("ascii", errors="strict")
    except UnicodeError as exc:
        raise DashboardError("request target must be ASCII") from exc
    if len(encoded) > MAX_PATH_BYTES:
        raise DashboardError("request target exceeds its byte bound")
    split = urlsplit(target)
    if split.scheme or split.netloc or split.fragment:
        raise DashboardError("absolute or fragmented request targets are forbidden")
    if len(split.query.encode("ascii")) > MAX_QUERY_BYTES:
        raise DashboardError("request query exceeds its byte bound")
    try:
        query = parse_qs(
            split.query,
            keep_blank_values=True,
            strict_parsing=True,
            max_num_fields=4,
        )
    except ValueError as exc:
        raise DashboardError("request query is malformed") from exc
    return split.path, query


def _parse_limit_query(query: Mapping[str, list[str]]) -> int:
    if not query:
        return DEFAULT_RECENT_LIMIT
    if set(query) != {"limit"} or len(query["limit"]) != 1:
        raise DashboardError("snapshot query permits one limit field")
    raw = query["limit"][0]
    if not raw.isascii() or not raw.isdecimal() or len(raw) > 3:
        raise DashboardError("limit must be bounded decimal text")
    return _limit(int(raw))


def _parse_detail_query(query: Mapping[str, list[str]]) -> str:
    if set(query) != {"id"} or len(query["id"]) != 1:
        raise DashboardError("detail query requires exactly one id")
    try:
        return validate_entity_id(query["id"][0])
    except BundleError as exc:
        raise DashboardError(str(exc)) from exc


def _history_html(snapshot: Mapping[str, Any]) -> bytes:
    rows = []
    for item in snapshot["recent"]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item['created_at_unix_ns']))}</td>"
            f"<td>{html.escape(item['entity_kind'])}</td>"
            f"<td>{html.escape(item['status'])}</td>"
            f"<td>{html.escape(item['entity_id'])}</td>"
            "</tr>"
        )
    body = "".join(rows) or "<tr><td colspan='4'>No H0.1 evidence persisted.</td></tr>"
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Friday H0.1 evidence</title><style>
body{{font:15px system-ui;margin:2rem;max-width:76rem;color:#17212b;background:#f6f8fa}}
.card{{background:white;border:1px solid #d8dee4;border-radius:10px;padding:1rem;margin-bottom:1rem}}
table{{width:100%;border-collapse:collapse}}th,td{{padding:.55rem;border-bottom:1px solid #ddd;text-align:left}}
.muted{{color:#57606a}}code{{font-size:.85em}}
</style></head><body><h1>Friday H0.1 evidence</h1>
<div class="card"><strong>{snapshot['total']}</strong> append-only bundles
<div class="muted">Revision <code>{html.escape(snapshot['revision'])}</code>; H0 lineage remains separate.</div></div>
<div class="card"><h2>Local history</h2><table><thead><tr><th>Created ns</th><th>Kind</th><th>Status</th><th>ID</th></tr></thead>
<tbody>{body}</tbody></table></div></body></html>"""
    return document.encode("utf-8")


class DashboardRequestHandler(BaseHTTPRequestHandler):
    """GET/HEAD-only adapter.  The test suite invokes dispatch without a socket."""

    server_version = "FridayH01Dashboard/1"
    sys_version = ""

    @property
    def service(self) -> DashboardService:
        service = getattr(self.server, "dashboard_service", None)
        if not isinstance(service, DashboardService):
            raise RuntimeError("dashboard server has no registered read-only service")
        return service

    def _security_headers(self) -> None:
        for name, value in _SECURITY_HEADERS:
            self.send_header(name, value)

    def _send(self, status: int, content_type: str, payload: bytes, *, head: bool) -> None:
        if not isinstance(payload, bytes) or len(payload) > MAX_RESPONSE_BYTES:
            raise DashboardError("dashboard response exceeds its byte cap")
        self.send_response(status)
        self._security_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if not head:
            self.wfile.write(payload)

    def _json(self, status: int, value: Mapping[str, Any], *, head: bool) -> None:
        payload = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        self._send(status, "application/json; charset=utf-8", payload, head=head)

    def _dispatch(self, *, head: bool) -> None:
        try:
            path, query = _parse_request_target(self.path)
            if path == "/":
                if query:
                    raise DashboardError("root path accepts no query")
                self._send(
                    200,
                    "text/html; charset=utf-8",
                    _history_html(self.service.snapshot()),
                    head=head,
                )
                return
            if path == "/api/snapshot":
                self._json(200, self.service.snapshot(limit=_parse_limit_query(query)), head=head)
                return
            if path == "/api/detail":
                detail = self.service.detail(_parse_detail_query(query))
                if detail is None:
                    self._json(404, {"error": "not_found"}, head=head)
                else:
                    self._json(200, detail, head=head)
                return
            self._json(404, {"error": "not_found"}, head=head)
        except DashboardError as exc:
            self._json(400, {"error": "bad_request", "message": str(exc)}, head=head)
        except (BundleError, StorageError) as exc:
            self._json(500, {"error": "evidence_unavailable", "message": str(exc)}, head=head)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract.
        self._dispatch(head=False)

    def do_HEAD(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract.
        self._dispatch(head=True)

    def _method_not_allowed(self) -> None:
        payload = b'{"error":"method_not_allowed"}'
        self.send_response(405)
        self._security_headers()
        self.send_header("Allow", "GET, HEAD")
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    do_POST = _method_not_allowed
    do_PUT = _method_not_allowed
    do_PATCH = _method_not_allowed
    do_DELETE = _method_not_allowed
    do_OPTIONS = _method_not_allowed
    do_CONNECT = _method_not_allowed
    do_TRACE = _method_not_allowed

    def __getattr__(self, name: str) -> Any:
        if name.startswith("do_") and len(name) > 3:
            return self._method_not_allowed
        raise AttributeError(name)

    def log_message(self, _format: str, *_args: object) -> None:
        return


class LoopbackDashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, database_path: str | Path, port: Any) -> None:
        try:
            checked_port = nonnegative_int64(port, "dashboard port", maximum=65535)
        except CanonicalError as exc:
            raise DashboardError(str(exc)) from exc
        self.dashboard_service = DashboardService(database_path)
        # The host is intentionally not caller-configurable.
        super().__init__((LOOPBACK_HOST, checked_port), DashboardRequestHandler)


def serve(database_path: str | Path, *, port: Any = 8765) -> None:
    """Serve until interrupted; this is the only real-socket construction path."""

    with LoopbackDashboardServer(database_path, port) as server:
        server.serve_forever()


__all__ = [
    "DASHBOARD_SCHEMA_VERSION",
    "DEFAULT_RECENT_LIMIT",
    "DashboardError",
    "DashboardRequestHandler",
    "DashboardService",
    "LOOPBACK_HOST",
    "LoopbackDashboardServer",
    "MAX_RESPONSE_BYTES",
    "serve",
]
