"""Read-only loopback H0 history dashboard.

The process owns its database path at startup.  HTTP is intentionally limited
to a fixed, local, read-only surface; no request can select a file or SQL
object.  This module uses only the Python standard library plus the existing
SQLite-v1 Storage class.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import re
import sqlite3
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from .dashboard_assets import CSS, HTML, JS
from .storage import Storage, StorageError

HOST = "127.0.0.1"
MAX_REQUEST_LINE = 2048
MAX_HEADERS = 16 * 1024
MAX_SNAPSHOT = 256 * 1024
MAX_DETAIL = 128 * 1024
MAX_ASSET = 96 * 1024
MAX_TEXT = 4096
MAX_RUNS = 100
MAX_RAW_SAMPLES = 200
RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$", re.ASCII)
SOURCE = "SQLite-v1, read-only Snapshot"
CSP = "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"


def _text(value: Any, reason: str = "not_recorded") -> dict[str, Any]:
    if value is None:
        return {"value": None, "missing_reason": reason}
    value = str(value)
    return {"value": value[: MAX_TEXT - 1] + "…" if len(value) >= MAX_TEXT else value, "missing_reason": None}


def _number(value: Any, reason: str = "not_recorded") -> dict[str, Any]:
    if value is None:
        return {"value": None, "missing_reason": reason}
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return {"value": None, "missing_reason": "invalid_source_value"}
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return {"value": None, "missing_reason": "invalid_source_value"}
    if not math.isfinite(number):
        return {"value": None, "missing_reason": "non_finite_source_value"}
    return {"value": number, "missing_reason": None}


def _iso(ns: Any) -> str | None:
    try:
        return datetime.fromtimestamp(int(ns) / 1_000_000_000, timezone.utc).isoformat().replace("+00:00", "Z")
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _json(value: Any) -> dict[str, Any]:
    if not isinstance(value, str):
        return {}

    def reject_constant(_: str) -> None:
        raise ValueError("non-finite JSON constant")

    try:
        parsed = json.loads(value, parse_constant=reject_constant)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}


def _path(data: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(data, dict):
            return None
        data = data.get(key)
    return data


def _metric(metrics: dict[str, sqlite3.Row], *names: str) -> Any:
    for name in names:
        row = metrics.get(name)
        if row is not None and row["value"] is not None:
            return row["value"]
    return None


def _metric_reason(metrics: dict[str, sqlite3.Row], *names: str) -> str:
    for name in names:
        row = metrics.get(name)
        if row is not None:
            return str(row["missing_reason"] or "not_recorded")
    return "not_recorded"


def _status(payloads: list[dict[str, Any]], fallback: str = "not_recorded") -> str:
    if not payloads:
        return fallback
    for event in reversed(payloads):
        payload = event.get("payload")
        if isinstance(payload, dict) and "result" in payload:
            result = payload.get("result")
            if not isinstance(result, dict):
                return fallback
            raw = result.get("status")
            if not isinstance(raw, str) or not raw:
                return fallback
            value = raw
            return value[: MAX_TEXT - 1] + "…" if len(value) >= MAX_TEXT else value
        if isinstance(payload, dict) and "status" in payload:
            raw = payload.get("status")
            if not isinstance(raw, str) or not raw:
                return fallback
            value = raw
            return value[: MAX_TEXT - 1] + "…" if len(value) >= MAX_TEXT else value
        raw = event.get("status")
        if isinstance(raw, str) and raw:
            value = raw
            return value[: MAX_TEXT - 1] + "…" if len(value) >= MAX_TEXT else value
    return fallback


def _guardrail(events: list[dict[str, Any]], name: str) -> str:
    key = name.lower()
    for event in reversed(events):
        payload = event.get("payload") or {}
        values = payload.get("guardrails") if isinstance(payload, dict) else None
        if isinstance(values, dict) and key in values:
            value = values[key]
            result = str(value).upper() if value is not None else "UNKNOWN"
            return result[: MAX_TEXT - 1] + "…" if len(result) >= MAX_TEXT else result
        if key in payload:
            value = payload[key]
            result = str(value).upper() if value is not None else "UNKNOWN"
            return result[: MAX_TEXT - 1] + "…" if len(result) >= MAX_TEXT else result
    return "UNKNOWN"


def _run_record(row: sqlite3.Row, events: list[dict[str, Any]], samples: list[sqlite3.Row], metrics: dict[str, sqlite3.Row], artifacts: list[sqlite3.Row], *, detail: bool = False) -> dict[str, Any]:
    manifest = _json(row["manifest_json"])
    workload = manifest.get("workload") if isinstance(manifest.get("workload"), dict) else {}
    seeds = manifest.get("seeds") if isinstance(manifest.get("seeds"), dict) else {}
    ratio = _metric(metrics, "primary_ratio", "ratio", "candidate_baseline_ratio")
    ratio_reason = _metric_reason(metrics, "primary_ratio", "ratio", "candidate_baseline_ratio")
    ci_low = _metric(metrics, "bootstrap_ci_low", "ci_low", "primary_ci_low")
    ci_high = _metric(metrics, "bootstrap_ci_high", "ci_high", "primary_ci_high")
    ci_low_reason = _metric_reason(metrics, "bootstrap_ci_low", "ci_low", "primary_ci_low")
    ci_high_reason = _metric_reason(metrics, "bootstrap_ci_high", "ci_high", "primary_ci_high")
    ratio_number = _number(ratio, ratio_reason)
    ci_low_number = _number(ci_low, ci_low_reason)
    ci_high_number = _number(ci_high, ci_high_reason)
    ci = (
        None
        if ci_low_number["value"] is None or ci_high_number["value"] is None
        else {"low": ci_low_number["value"], "high": ci_high_number["value"]}
    )
    result: dict[str, Any] = {
        "run_id": _text(row["run_id"]), "created_at": _iso(row["created_at_unix_ns"]),
        "phase": _text(row["phase"]), "mode": _text(row["mode"]), "status": _status(events),
        "workload_family": _text("matmul" if workload else None),
        "shape": workload.get("a_shape") if isinstance(workload.get("a_shape"), list) else None,
        "dtype": _text(workload.get("dtype")), "layout": _text(workload.get("layout")),
        "baseline": _text(_path(manifest, "baseline", "id")), "candidate": _text(_path(manifest, "candidate", "id")),
        "search_strategy": _text(_path(manifest, "search", "strategy")), "trial_budget": _number(_path(manifest, "search", "trial_budget")),
        "time_budget": _number(_path(manifest, "search", "time_budget_s")), "primary_ratio": ratio_number, "ci_low": ci_low_number, "ci_high": ci_high_number, "ci": ci,
        "guardrails": {name: _guardrail(events, name) for name in ("correctness", "memory", "timeout", "crash", "rollback")},
        "timing": {name: _number(_metric(metrics, name, f"{name}_ns", f"{name}_s"), _metric_reason(metrics, name, f"{name}_ns", f"{name}_s")) for name in ("cold", "compile", "warm", "tuning", "break_even", "median", "mad", "iqr", "bootstrap")},
        "hashes": {"manifest": _text(row["manifest_hash"]), "code": _text(row["code_sha256"]), "spec": _text(row["spec_sha256"]), "environment": _text(row["environment_sha256"])},
        "seeds": {"fixture": _text(seeds.get("fixture"), "not_recorded"), "order": _text(seeds.get("order"), "not_recorded")},
        "revision": _text(row["revision"], row["revision_missing_reason"] or "not_recorded"),
        "model": _text(None, "not_applicable_h0"), "prompt": _text(None, "not_applicable_h0"),
        "raw_sample_count": len(samples), "raw_samples_truncated": False,
    }
    if detail:
        result["raw_samples"] = [{"session_id": _text(s["session_id"]), "kind": _text(s["sample_kind"]), "sample_index": int(s["sample_index"]), "block_index": int(s["block_index"]), "arm": _text(s["arm"]), "value": _number(s["value"]), "unit": _text(s["unit"]), "observed_at": _iso(s["observed_at_ns"])} for s in samples[:MAX_RAW_SAMPLES]]
        result["raw_samples_truncated"] = len(samples) > MAX_RAW_SAMPLES
        result["raw_sample_count"] = len(samples)
        result["artifacts"] = [{"name": _text(a["artifact_name"]), "kind": _text(a["artifact_kind"]), "sha256": _text(a["sha256"])} for a in artifacts]
        result["events"] = [{"kind": _text(e.get("event_kind")), "status": _text(e.get("status")), "recorded_at": _iso(e.get("recorded_at_ns"))} for e in events]
    return result


class DashboardService:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)

    def _open(self, deadline_s: float) -> tuple[Storage, float]:
        storage = Storage.open(self.db_path, read_only=True)
        try:
            storage.connection.execute("PRAGMA busy_timeout=75")
            deadline = time.monotonic() + deadline_s
            storage.connection.set_progress_handler(lambda: 1 if time.monotonic() >= deadline else 0, 1000)
            storage.connection.execute("BEGIN")
            return storage, deadline
        except BaseException:
            try:
                storage.close()
            except Exception:
                pass
            raise

    def snapshot(self) -> dict[str, Any]:
        storage, _ = self._open(0.250)
        try:
            conn = storage.connection
            total = int(conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0])
            rows = conn.execute("SELECT run_id,phase,mode,manifest_json,manifest_hash,code_sha256,spec_sha256,environment_sha256,revision,revision_missing_reason,created_at_unix_ns FROM runs ORDER BY created_at_unix_ns DESC, run_id DESC LIMIT 100").fetchall()
            records = [self._record(conn, row, False) for row in rows]
            latest = conn.execute("SELECT MAX(created_at_unix_ns) FROM runs").fetchone()[0]
            source_revision = self._source_revision(conn)
            observed = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            displayed_ids = ":".join(str(r["run_id"]) for r in rows)
            snapshot_id = hashlib.sha256((source_revision + ":" + displayed_ids).encode()).hexdigest()
            payload: dict[str, Any] = {"schema_version": 1, "source": SOURCE, "snapshot_id": snapshot_id, "source_revision": source_revision, "observed_at": observed, "source_last_updated_at": _iso(latest), "freshness_basis": "latest_run_created_at; evidence revision tracked without timestamps", "freshness_state": "snapshot", "data_state": "available" if total else "empty", "run_count": total, "available_count": total, "returned_count": len(records), "truncated": total > MAX_RUNS, "runs": records}
            return _bounded_payload(payload, MAX_SNAPSHOT)
        finally:
            storage.close()

    def detail(self, run_id: str) -> tuple[int, dict[str, Any]]:
        storage, _ = self._open(0.500)
        try:
            row = storage.connection.execute("SELECT run_id,phase,mode,manifest_json,manifest_hash,code_sha256,spec_sha256,environment_sha256,revision,revision_missing_reason,created_at_unix_ns FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                return 404, {"error": "run_not_found"}
            return 200, _bounded_payload(self._record(storage.connection, row, True), MAX_DETAIL)
        finally:
            storage.close()

    @staticmethod
    def _record(conn: sqlite3.Connection, row: sqlite3.Row, detail: bool) -> dict[str, Any]:
        events_rows = conn.execute("SELECT event_kind,status,substr(payload_json,1,65536) AS payload_json,recorded_at_ns FROM status_events WHERE run_id=? ORDER BY event_id", (row["run_id"],)).fetchall()
        events = [{"event_kind": e["event_kind"], "status": e["status"], "payload": _json(e["payload_json"]), "recorded_at_ns": e["recorded_at_ns"]} for e in events_rows]
        metric_rows = conn.execute("SELECT metric_name,value,missing_reason FROM scalar_metrics WHERE run_id=? ORDER BY metric_id", (row["run_id"],)).fetchall()
        metrics = {str(m["metric_name"]): m for m in metric_rows}
        samples = conn.execute("SELECT session_id,sample_kind,sample_index,block_index,arm,value,unit,observed_at_ns FROM raw_samples WHERE run_id=? ORDER BY sample_id LIMIT 201", (row["run_id"],)).fetchall()
        raw_sample_count = int(conn.execute("SELECT COUNT(*) FROM raw_samples WHERE run_id=?", (row["run_id"],)).fetchone()[0])
        artifacts = conn.execute("SELECT artifact_name,artifact_kind,sha256 FROM artifacts WHERE run_id=? ORDER BY artifact_id LIMIT 256", (row["run_id"],)).fetchall() if detail else []
        record = _run_record(row, events, samples, metrics, artifacts, detail=detail)
        record["raw_sample_count"] = raw_sample_count
        if detail:
            record["raw_samples_truncated"] = raw_sample_count > MAX_RAW_SAMPLES
        return record

    @staticmethod
    def _source_revision(conn: sqlite3.Connection) -> str:
        """Hash bounded append-only table aggregates inside the open snapshot."""

        runs = conn.execute(
            "SELECT COUNT(*), COALESCE(MAX(created_at_unix_ns), 0), COALESCE(MAX(run_id), '') FROM runs"
        ).fetchone()
        status_events = conn.execute(
            "SELECT COUNT(*), COALESCE(MAX(event_id), 0) FROM status_events"
        ).fetchone()
        raw_samples = conn.execute(
            "SELECT COUNT(*), COALESCE(MAX(sample_id), 0) FROM raw_samples"
        ).fetchone()
        scalar_metrics = conn.execute(
            "SELECT COUNT(*), COALESCE(MAX(metric_id), 0) FROM scalar_metrics"
        ).fetchone()
        correctness_metrics = conn.execute(
            "SELECT COUNT(*), COALESCE(MAX(metric_id), 0) FROM correctness_metrics"
        ).fetchone()
        artifacts = conn.execute(
            "SELECT COUNT(*), COALESCE(MAX(artifact_id), 0) FROM artifacts"
        ).fetchone()
        material = {
            "runs": [int(runs[0]), int(runs[1]), str(runs[2])],
            "status_events": [int(status_events[0]), int(status_events[1])],
            "raw_samples": [int(raw_samples[0]), int(raw_samples[1])],
            "scalar_metrics": [int(scalar_metrics[0]), int(scalar_metrics[1])],
            "correctness_metrics": [int(correctness_metrics[0]), int(correctness_metrics[1])],
            "artifacts": [int(artifacts[0]), int(artifacts[1])],
        }
        encoded = json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _bounded_payload(payload: dict[str, Any], cap: int) -> dict[str, Any]:
    try:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, OverflowError):
        return {"schema_version": 1, "source": SOURCE, "freshness_state": "snapshot", "data_state": "error", "error": "response_serialization_error"}
    if len(encoded) <= cap:
        return payload
    if "runs" in payload:
        runs = payload["runs"]
        while runs and len(encoded) > cap:
            runs = runs[:-1]
            payload = dict(payload, runs=runs, returned_count=len(runs), truncated=True)
            try:
                encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
            except (TypeError, ValueError, OverflowError):
                return {"schema_version": 1, "source": SOURCE, "freshness_state": "snapshot", "data_state": "error", "error": "response_serialization_error"}
        if len(encoded) <= cap:
            return payload
    return {"schema_version": 1, "source": SOURCE, "freshness_state": "snapshot", "data_state": "error", "error": "response_limit"}


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "FridayH0/1"
    protocol_version = "HTTP/1.1"

    def _headers(self, content_type: str, length: int) -> None:
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Content-Security-Policy", CSP)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Connection", "close")

    def _send(self, status: int, body: bytes, content_type: str = "application/json; charset=utf-8") -> None:
        self.send_response(status)
        self._headers(content_type, len(body))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def parse_request(self) -> bool:
        if len(self.raw_requestline) > MAX_REQUEST_LINE:
            self.requestline = self.raw_requestline[:MAX_REQUEST_LINE].decode("iso-8859-1", "replace")
            self.request_version = "HTTP/1.1"
            self.command = self.requestline.split(" ", 1)[0] or "GET"
            self._send(HTTPStatus.REQUEST_URI_TOO_LONG, b'{"error":"request_target_too_long"}')
            return False
        if not super().parse_request():
            return False
        header_bytes = sum(len(k) + len(v) for k, v in self.headers.items())
        if header_bytes > MAX_HEADERS:
            self._send(HTTPStatus.REQUEST_HEADER_FIELDS_TOO_LARGE, b'{"error":"headers_too_large"}')
            return False
        return True

    def do_GET(self) -> None:
        self._dispatch()

    def do_HEAD(self) -> None:
        self._dispatch()

    def do_POST(self) -> None:
        self._dispatch()

    def do_PUT(self) -> None:
        self._dispatch()

    def do_PATCH(self) -> None:
        self._dispatch()

    def do_DELETE(self) -> None:
        self._dispatch()

    def do_OPTIONS(self) -> None:
        self._dispatch()

    def _dispatch(self) -> None:
        if self.command not in {"GET", "HEAD"}:
            self.send_response(HTTPStatus.METHOD_NOT_ALLOWED)
            self.send_header("Allow", "GET, HEAD")
            self._headers("application/json; charset=utf-8", 0)
            self.end_headers()
            return
        if self.headers.get("Content-Length", "0") not in {"", "0"}:
            self._send(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, b'{"error":"request_body_not_allowed"}')
            return
        parsed = urlsplit(self.path)
        path = parsed.path
        if len(self.path) > MAX_REQUEST_LINE or "#" in self.path:
            self._send(414, b'{"error":"request_target_too_long"}')
            return
        try:
            if path == "/" and not parsed.query:
                body = HTML.encode()
                if len(body) > MAX_ASSET: raise ValueError("asset")
                self._send(200, body, "text/html; charset=utf-8"); return
            if path == "/assets/app.css" and not parsed.query:
                body = CSS.encode()
                if len(body) > MAX_ASSET: raise ValueError("asset")
                self._send(200, body, "text/css; charset=utf-8"); return
            if path == "/assets/app.js" and not parsed.query:
                body = JS.encode()
                if len(body) > MAX_ASSET: raise ValueError("asset")
                self._send(200, body, "application/javascript; charset=utf-8"); return
            service: DashboardService = self.server.service  # type: ignore[attr-defined]
            if path == "/api/snapshot" and not parsed.query:
                self._send_json(service.snapshot()); return
            if path == "/api/run":
                query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=False)
                if set(query) != {"id"} or len(query["id"]) != 1 or not RUN_ID_RE.fullmatch(query["id"][0]):
                    self._send(400, b'{"error":"invalid_run_id"}'); return
                status, payload = service.detail(query["id"][0]); self._send_json(payload, status, MAX_DETAIL); return
            self._send(404, b'{"error":"not_found"}')
        except StorageError as exc:
            state = "source_busy" if any(word in str(exc).lower() for word in ("busy", "locked")) else "invalid_source"
            self._send_json({"schema_version": 1, "source": SOURCE, "freshness_state": "snapshot", "data_state": state, "error": state}, 503)
        except sqlite3.OperationalError as exc:
            state = "source_busy" if any(word in str(exc).lower() for word in ("busy", "locked")) else "error"
            self._send_json({"schema_version": 1, "source": SOURCE, "freshness_state": "snapshot", "data_state": state, "error": state}, 503)
        except (sqlite3.Error, OSError, ValueError):
            self._send_json({"schema_version": 1, "source": SOURCE, "freshness_state": "snapshot", "data_state": "error", "error": "error"}, 503)
        except Exception:
            self._send(500, b'{"error":"error"}')

    def _send_json(self, payload: dict[str, Any], status: int = 200, cap: int = MAX_SNAPSHOT) -> None:
        try:
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        except (TypeError, ValueError, OverflowError):
            body = b'{"schema_version":1,"source":"SQLite-v1, read-only Snapshot","freshness_state":"snapshot","data_state":"error","error":"response_serialization_error"}'
            status = 500
        if len(body) > cap:
            body = b'{"error":"response_limit"}'
            status = 500
        self._send(status, body)

    def log_message(self, *_: Any) -> None:
        return


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, db_path: str | Path, port: int = 0) -> None:
        if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
            raise ValueError("port must be an integer in 0..65535")
        self.service = DashboardService(db_path)
        super().__init__((HOST, port), DashboardHandler)


def serve(db_path: str | Path, port: int = 0) -> DashboardServer:
    """Create a server; callers own its lifecycle and should call shutdown/close."""
    return DashboardServer(db_path, port)
