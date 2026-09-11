"""Read-only loopback status UI for the portable DATA1 collection boundary."""

from __future__ import annotations

import html
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import stat
from collections import deque
import re
from datetime import datetime, timezone
from typing import Any, Mapping
from urllib.parse import urlsplit

from friday_evidence.canonical import canonical_sha256
from friday_evidence.events import EventJournal, EventJournalError

from .corpus import CorpusError, build_dataset
from .quota import QuotaController, QuotaError


LOOPBACK_HOST = "127.0.0.1"
DEFAULT_PORT = 8789
_MAX_HISTORY = 80
_MAX_RESPONSE_BYTES = 512 * 1024
_BACKENDS = (("cuda", "GPU"), ("tpu", "TPU"), ("mlx", "MLX"))
_SAFE_RUN_STATUS = frozenset({"finished", "failed", "deferred", "captured", "imported", "complete", "completed", "passed", "valid", "censored", "smoke"})


class DashboardError(ValueError):
    """The portable dashboard cannot uphold its loopback/read-only contract."""


def _root(state_dir: Path) -> Path:
    root = Path(state_dir).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    if root.is_symlink() or any(parent.is_symlink() for parent in root.parents):
        raise DashboardError("unsicheres Statusverzeichnis")
    if not root.exists():
        return root
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise DashboardError("Statusverzeichnis ist kein Verzeichnis")
    return root


def _quota(root: Path) -> dict[str, Any]:
    # Both names are supported so an older local journal remains observable;
    # this inspection never creates either a lock or a journal.
    for name in ("quota-events.sqlite3", "quota.sqlite3"):
        path = root / name
        if not path.exists():
            continue
        try:
            state = QuotaController(path).status(read_only=True)
        except QuotaError:
            return {"state": "unavailable", "reserved_seconds": None, "consumed_seconds": None, "frozen": None}
        reservations = state.get("reservations", [])
        if not isinstance(reservations, list):
            return {"state": "unavailable", "reserved_seconds": None, "consumed_seconds": None, "frozen": None}
        reserved = sum(
            float(item.get("reserved_seconds", 0.0))
            for item in reservations if isinstance(item, Mapping) and isinstance(item.get("reserved_seconds"), (int, float))
        )
        consumed = sum(
            float(item.get("charged_seconds", 0.0))
            for item in reservations if isinstance(item, Mapping) and isinstance(item.get("charged_seconds"), (int, float))
        )
        return {
            "state": "ready",
            "reserved_seconds": round(reserved, 3),
            "consumed_seconds": round(consumed, 3),
            "frozen": bool(state.get("frozen")),
            "active_jobs": len(state.get("active_run_slugs", [])),
        }
    return {"state": "uninitialized", "reserved_seconds": None, "consumed_seconds": None, "frozen": None, "active_jobs": 0}


def _captures(root: Path) -> dict[str, Any]:
    """Count only locally supervised, finished real captures."""
    count = 0
    dimensions: set[str] = set()
    for manifest_path in sorted((root / "captures").glob("*/data/capture.json")):
        parent = manifest_path.parent.parent
        supervisor_path = parent / "supervisor.json"
        try:
            manifest = json.loads(manifest_path.read_bytes())
            supervisor = json.loads(supervisor_path.read_bytes())
        except (OSError, ValueError, TypeError):
            continue
        if (manifest.get("status") != "captured" or supervisor.get("status") != "finished"
                or supervisor.get("code_unchanged") is not True):
            continue
        count += 1
        for case in manifest.get("cases", []):
            shape = case.get("shape") if isinstance(case, Mapping) else None
            if isinstance(shape, list) and len(shape) == 3 and all(type(n) is int for n in shape):
                dimensions.add("×".join(str(n) for n in shape))
    return {"real": count, "dimensions": sorted(dimensions)}


def _provider_probes(root: Path) -> dict[str, Any]:
    """Project verified/failed cloud smokes without paths or timing claims."""
    rows: list[dict[str, Any]] = []
    for directory in sorted((root / "provider-probes").glob("*")):
        candidates = (directory / "outcome.json", directory / "recovered-output" / "probe-result.json")
        value = None
        for path in candidates:
            try:
                candidate = json.loads(path.read_bytes())
            except (OSError, ValueError, TypeError):
                continue
            if isinstance(candidate, dict):
                value = candidate
                break
        if value is None:
            continue
        report = value.get("report") if value.get("schema") == "ironmule.provider-smoke-outcome.v1" else value
        if not isinstance(report, Mapping) or report.get("schema") != "ironmule.provider-smoke.v1":
            continue
        if report.get("performance_claim") is not False or report.get("backend") not in {"cuda", "tpu"}:
            continue
        run_id = report.get("run_id")
        if not isinstance(run_id, str) or not re.fullmatch(r"[0-9a-f]{32}", run_id):
            continue
        hardware = report.get("hardware") if isinstance(report.get("hardware"), Mapping) else {}
        rows.append({"run_id": run_id, "backend": report["backend"],
            "status": "passed" if report.get("status") == "passed" else "failed",
            "hardware_verified": report.get("hardware_verified") is True,
            "correctness_verified": report.get("correctness_verified") is True,
            "accelerator": hardware.get("accelerator") if isinstance(hardware.get("accelerator"), str) else None,
            "framework": hardware.get("framework") if isinstance(hardware.get("framework"), str) else None,
            "framework_version": (hardware.get("framework_version")
                                  if isinstance(hardware.get("framework_version"), str) else None),
            "error_code": (report.get("error_code") if isinstance(report.get("error_code"), str)
                           and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,127}", report["error_code"])
                           else None)})
    return {"runs": rows, "passed": sum(row["status"] == "passed" for row in rows),
            "failed": sum(row["status"] == "failed" for row in rows)}


def _corpus(root: Path) -> dict[str, Any]:
    try:
        dataset = build_dataset(root)
    except (CorpusError, OSError, ValueError):
        return {
            "state": "uninitialized", "unique_cases": 0, "raw_samples": 0,
            "invalid": 0, "censored": 0, "runs": [],
            "backends": {label: {"unique_cases": 0, "raw_samples": 0} for _, label in _BACKENDS},
        }
    records = dataset.get("records", [])
    counts = dataset.get("counts", {})
    if not isinstance(records, list) or not isinstance(counts, Mapping):
        raise DashboardError("unzulässige Corpus-Zusammenfassung")
    by_backend: dict[str, dict[str, Any]] = {
        name: {"unique_cases": set(), "raw_samples": 0} for name, _ in _BACKENDS
    }
    cases: set[tuple[str, str]] = set()
    raw_samples = 0
    count_raw_samples = counts.get("raw_samples")
    if type(count_raw_samples) is int and count_raw_samples >= 0:
        raw_samples = count_raw_samples
    for record in records:
        if not isinstance(record, Mapping):
            continue
        backend = record.get("backend")
        run_id, case_id = record.get("run_id"), record.get("case_id")
        if not isinstance(backend, str) or not isinstance(run_id, str) or not isinstance(case_id, str):
            continue
        key = record.get("input_sha256") or case_id
        cases.add(key)
        row = by_backend.get(backend)
        if row is not None:
            row["unique_cases"].add(key)
        # The UI exposes only the cardinality, never a timing label or a
        # holdout record.  build_dataset already projects holdout labels away.
        label = record.get("label")
        pair_count = label.get("pair_count") if isinstance(label, Mapping) else None
        sample_count = record.get("sample_count")
        if type(count_raw_samples) is int and count_raw_samples >= 0:
            pair_count = sample_count if type(sample_count) is int and sample_count >= 0 else 0
            if row is not None:
                row["raw_samples"] += pair_count
        if type(pair_count) is int and pair_count >= 0 and not (type(count_raw_samples) is int and count_raw_samples >= 0):
            raw_samples += pair_count
            if row is not None:
                row["raw_samples"] += pair_count
    statuses = counts.get("by_status", {})
    if not isinstance(statuses, Mapping):
        statuses = {}
    rendered_backends = {
        label: {
            "unique_cases": len(by_backend[name]["unique_cases"]),
            "raw_samples": by_backend[name]["raw_samples"],
        }
        for name, label in _BACKENDS
    }
    reports = dataset.get("reports", [])
    runs: list[dict[str, Any]] = []
    if isinstance(reports, list):
        for report in reports[:_MAX_HISTORY]:
            if not isinstance(report, Mapping):
                continue
            run_id, backend, partition, status = (
                report.get("run_id"), report.get("backend"), report.get("partition"), report.get("status")
            )
            if (
                isinstance(run_id, str) and len(run_id) == 32 and run_id.isascii()
                and backend in {name for name, _ in _BACKENDS}
                and partition in {"train", "validation", "holdout"}
            ):
                runs.append({
                    "run_id": run_id,
                    "kind": "capture_collection",
                    "backend": backend,
                    "partition": partition,
                    "status": status if isinstance(status, str) and status in _SAFE_RUN_STATUS else None,
                })
    return {
        "state": "ready",
        "unique_cases": len(cases),
        "raw_samples": raw_samples,
        "invalid": sum(int(statuses.get(name, 0)) for name in ("incorrect", "unsupported", "failed")),
        "censored": int(statuses.get("censored", 0)),
        "runs": runs,
        "backends": rendered_backends,
        "holdout_labels_hidden": bool(dataset.get("holdout", {}).get("hidden", True)),
    }


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: deque[dict[str, Any]] = deque(maxlen=_MAX_HISTORY)
    try:
        with EventJournal(path, read_only=True) as journal:
            after = 0
            while True:
                page = journal.events(limit=100, after_seq=after)
                if not page:
                    break
                events.extend(page)
                after = page[-1]["seq"]
                if len(page) < 100:
                    break
    except EventJournalError:
        return []
    return list(events)


def _history(root: Path) -> list[dict[str, Any]]:
    events = _read_events(root / "control.sqlite3") + _read_events(root / "supervisor-events.sqlite3")
    result: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda item: (item.get("recorded_unix_ns", 0), item.get("seq", 0)))[-_MAX_HISTORY:]:
        payload = event.get("payload")
        status = payload.get("status") if isinstance(payload, Mapping) else None
        error = None
        if isinstance(payload, Mapping):
            candidate = payload.get("error_code") or payload.get("reason")
            if isinstance(candidate, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,127}", candidate):
                error = candidate
        result.append({
            "run_id": event["run_id"],
            "kind": event["kind"],
            "status": status if isinstance(status, str) and status in _SAFE_RUN_STATUS else None,
            "error_code": error,
            "recorded_unix_ns": event["recorded_unix_ns"],
        })
    return result


def snapshot(state_dir: Path) -> dict[str, Any]:
    """Return safe local summaries only; missing stores remain uninitialized."""
    root = _root(state_dir)
    corpus = _corpus(root)
    history = _history(root)
    value = {
        "schema": "ironmule.portable-dashboard.v1",
        "read_only": True,
        "learning_claim": False,
        "quota": _quota(root),
        "corpus": corpus,
        "captures": _captures(root),
        "provider_probes": _provider_probes(root),
        "history": history,
    }
    value["revision"] = canonical_sha256(value)
    return value


def _timestamp_label(value: Any) -> str:
    if type(value) is not int or value <= 0:
        return "—"
    try:
        return datetime.fromtimestamp(value / 1e9, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, OverflowError, OSError):
        return "—"


def render_dashboard(value: Mapping[str, Any]) -> str:
    """Render a compact responsive German dashboard from an already-safe snapshot."""
    corpus = value.get("corpus", {})
    quota = value.get("quota", {})
    captures = value.get("captures", {})
    provider = value.get("provider_probes", {})
    backends = corpus.get("backends", {}) if isinstance(corpus, Mapping) else {}
    rows = "".join(
        "<tr><th>" + html.escape(label) + "</th><td>" + html.escape(str(data.get("unique_cases", 0)))
        + "</td><td>" + html.escape(str(data.get("raw_samples", 0))) + "</td></tr>"
        for label, data in backends.items() if isinstance(data, Mapping)
    ) or "<tr><td colspan='3'>Keine lokalen Daten.</td></tr>"
    provider_rows = "".join(
        "<tr><td>" + html.escape(str(item.get("run_id", ""))[:8]) + "…</td><td>"
        + html.escape(str(item.get("backend", "—")).upper()) + "</td><td>"
        + html.escape(str(item.get("status", "—"))) + "</td><td>"
        + html.escape(str(item.get("accelerator") or "—")) + "</td><td>"
        + html.escape(str(item.get("framework") or "—")) + " "
        + html.escape(str(item.get("framework_version") or "")) + "</td><td>"
        + html.escape(str(item.get("error_code") or "")) + "</td></tr>"
        for item in provider.get("runs", []) if isinstance(item, Mapping)
    ) or "<tr><td colspan='6'>Keine Cloud-Smokes.</td></tr>"
    return f"""<!doctype html><html lang='de'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>
<title>Friday DATA1 – lokaler Status</title><style>
:root{{color-scheme:dark;--bg:#10151c;--card:#19222d;--line:#314254;--muted:#a9b7c6;--good:#8ee3b5;--warn:#ffd280}}
body{{margin:0;background:var(--bg);color:#edf5fc;font:16px system-ui,sans-serif}}main{{max-width:1080px;margin:auto;padding:24px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}}.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}}
.number{{font-size:1.8rem;font-weight:700}}.muted{{color:var(--muted)}}table{{width:100%;border-collapse:collapse}}th,td{{padding:9px;border-bottom:1px solid var(--line);text-align:left}}
@media(max-width:560px){{main{{padding:14px}}th,td{{padding:7px}}}}</style></head><body><main>
<h1>DATA1 · lokaler Verlauf</h1><p class='muted'>Nur lesend. Keine Hardware-, Cloud- oder Lernentscheidung.</p>
<section class='grid'><article class='card'><div class='muted'>Eindeutige Fälle</div><div class='number'>{html.escape(str(corpus.get('unique_cases', 0)))}</div></article>
<article class='card'><div class='muted'>Rohstichproben (Messpaare)</div><div class='number'>{html.escape(str(corpus.get('raw_samples', 0)))}</div></article>
<article class='card'><div class='muted'>Ungültig / zensiert</div><div class='number'>{html.escape(str(corpus.get('invalid', 0)))} / {html.escape(str(corpus.get('censored', 0)))}</div></article>
<article class='card'><div class='muted'>Quote (Sekunden) reserviert / verbraucht</div><div class='number'>{html.escape(str(quota.get('reserved_seconds') if quota.get('reserved_seconds') is not None else 'unbekannt'))} / {html.escape(str(quota.get('consumed_seconds') if quota.get('consumed_seconds') is not None else 'unbekannt'))}</div><div class='muted'>eingefroren: {html.escape(str(quota.get('frozen') if quota.get('frozen') is not None else 'unbekannt'))}</div></article>
<article class='card'><div class='muted'>Echte Captures</div><div class='number'>{html.escape(str(captures.get('real', 0)))}</div><div class='muted'>Dimensionen: {html.escape(', '.join(captures.get('dimensions', [])) or 'unbekannt')}</div></article></section>
<section class='card'><h2>Backends</h2><table><thead><tr><th>Backend</th><th>Fälle</th><th>Rohstichproben</th></tr></thead><tbody>{rows}</tbody></table></section>
<section class='card'><h2>Reale Cloud-Smokes</h2><p class='muted'>Bestanden / fehlgeschlagen: {html.escape(str(provider.get('passed', 0)))} / {html.escape(str(provider.get('failed', 0)))} · keine Performanceaussage</p><div style='overflow-x:auto'><table><thead><tr><th>Run</th><th>Backend</th><th>Status</th><th>Hardware</th><th>Framework</th><th>Fehler</th></tr></thead><tbody>{provider_rows}</tbody></table></div></section>
<section class='card'><h2>Versuchshistorie</h2><div style='overflow-x:auto'><table><thead><tr><th>Run</th><th>Stufe</th><th>Status</th><th>Zeit (UTC)</th><th>Fehler</th></tr></thead><tbody>{''.join("<tr><td>" + html.escape(str(item.get('run_id',''))[:8]) + "…</td><td>" + html.escape(str(item.get('kind') or '—')) + "</td><td>" + html.escape(str(item.get('status') or '—')) + "</td><td>" + html.escape(_timestamp_label(item.get('recorded_unix_ns'))) + "</td><td>" + html.escape(str(item.get('error_code') or '')) + "</td></tr>" for item in value.get('history', []) if isinstance(item, Mapping)) or "<tr><td colspan='5'>Keine lokale Historie.</td></tr>"}</tbody></table></div></section>
<p class='muted'>Holdout-Labels bleiben verborgen. Lernanspruch: nein.</p></main></body></html>"""


class _Handler(BaseHTTPRequestHandler):
    server_version = "FridayPortableDashboard/1"
    sys_version = ""

    def _send(self, status: int, kind: str, body: bytes) -> None:
        if len(body) > _MAX_RESPONSE_BYTES:
            raise DashboardError("Antwort zu groß")
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, value: Mapping[str, Any]) -> None:
        self._send(status, "application/json; charset=utf-8", json.dumps(value, ensure_ascii=False, sort_keys=True).encode())

    def do_GET(self) -> None:  # noqa: N802
        origin = self.headers.get("Origin")
        expected = f"http://{self.server.server_address[0]}:{self.server.server_address[1]}"
        host = self.headers.get("Host")
        allowed_hosts = {f"localhost:{self.server.server_address[1]}", f"127.0.0.1:{self.server.server_address[1]}"}
        if host is not None and host not in allowed_hosts:
            self._json(403, {"error": "host_forbidden"})
            return
        if origin is not None and origin != expected:
            self._json(403, {"error": "origin_forbidden"})
            return
        target = urlsplit(self.path)
        if target.query or target.fragment or target.scheme or target.netloc:
            self._json(400, {"error": "invalid_request"})
            return
        value = snapshot(self.server.state_dir)  # type: ignore[attr-defined]
        if target.path == "/":
            self._send(200, "text/html; charset=utf-8", render_dashboard(value).encode())
        elif target.path == "/api/status":
            self._json(200, value)
        else:
            self._json(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        self._json(405, {"error": "method_not_allowed"})

    def do_PUT(self) -> None:  # noqa: N802
        self.do_POST()

    def do_PATCH(self) -> None:  # noqa: N802
        self.do_POST()

    def do_DELETE(self) -> None:  # noqa: N802
        self.do_POST()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.do_POST()

    def log_message(self, _format: str, *_args: object) -> None:
        return


def make_server(state_dir: Path, *, host: str = LOOPBACK_HOST, port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    if host != LOOPBACK_HOST:
        raise DashboardError("nur Loopback-Host ist erlaubt")
    if type(port) is not int or not 0 <= port <= 65535:
        raise DashboardError("ungültiger Port")
    server = ThreadingHTTPServer((host, port), _Handler)
    server.state_dir = _root(state_dir)  # type: ignore[attr-defined]
    return server


def serve(state_dir: Path, host: str = LOOPBACK_HOST, port: int = DEFAULT_PORT) -> None:
    server = make_server(state_dir, host=host, port=port)
    try:
        server.serve_forever()
    finally:
        server.server_close()


__all__ = ["DashboardError", "make_server", "render_dashboard", "serve", "snapshot"]
