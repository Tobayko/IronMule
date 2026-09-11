"""Export verified local calibration and load-screen history for dashboards."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from friday_evidence.canonical import canonical_sha256
from friday_evidence.events import EventJournal, EventJournalError

MAX_EVENTS_PER_SOURCE = 2000
MAX_DATASET_ROWS = 2000
_QUERY = ("SELECT seq, recorded_unix_ns, run_id, kind, payload_json, prev_sha256, sha256 "
          "FROM event_journal ORDER BY seq ASC")


def iso(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1e9, timezone.utc).isoformat()


def _read_events(path: Path) -> tuple[list[dict[str, Any]], str]:
    try:
        with EventJournal(path, read_only=True) as journal:
            total = journal.verify()
            if total > MAX_EVENTS_PER_SOURCE:
                raise ValueError("history exceeds the 2000-event per-source bound")
            events: list[dict[str, Any]] = []
            after = 0
            while len(events) < total:
                page = journal.events(limit=min(1000, total - len(events)), after_seq=after)
                if not page:
                    break
                events.extend(page)
                after = page[-1]["seq"]
            if len(events) != total:
                raise ValueError("history changed while exporting")
            if journal.verify() != total:
                raise ValueError("history changed while exporting")
            return events, canonical_sha256(events)
    except EventJournalError as exc:
        raise ValueError("event journal is missing or failed integrity checks") from exc


def _model_label(model_id: str) -> str:
    lowered = model_id.lower()
    for label in ("12B", "4B", "1B"):
        if re.search(r"(?:^|[-_/])" + label.lower() + r"(?:$|[-_/])", lowered):
            return label
    return model_id.rsplit("/", 1)[-1][:32]


def _completed_samples(events: list[dict[str, Any]], run_id: str) -> set[int]:
    states: dict[int, str] = {}
    for event in events:
        if event["run_id"] != run_id or event["kind"] != "validation":
            continue
        payload = event["payload"]
        sample = payload.get("sample") if isinstance(payload, dict) else None
        if isinstance(sample, dict) and sample.get("status") in {"passed", "failed"}:
            index = sample.get("sample_index")
            if type(index) is int and index >= 0:
                states[index] = sample["status"]
    return {index for index, state in states.items() if state == "passed"}


def _runs(events: list[dict[str, Any]], source: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runs: dict[str, dict[str, Any]] = {}
    finished: set[str] = set()
    observations: list[dict[str, Any]] = []
    for event in events:
        run_id = event["run_id"]
        payload = event["payload"]
        if event["kind"] != "run_started" and run_id not in runs:
            raise ValueError("event journal contains an event for an unknown run")
        if run_id in finished and event["kind"] != "run_finished":
            raise ValueError("event journal contains events after run_finished")
        if event["kind"] == "run_started":
            if run_id in runs:
                raise ValueError("event journal contains duplicate run_started")
            policy = payload.get("readiness_policy") if isinstance(payload, dict) else {}
            runs[run_id] = {
                "run_id": run_id, "trial": run_id[:8],
                "started_unix_ns": event["recorded_unix_ns"],
                "start_utc": iso(event["recorded_unix_ns"]),
                "model_id": payload.get("model_id"),
                "model_label": _model_label(str(payload.get("model_id", "unknown"))),
                "kind": "load" if source == "load" else "calibration",
                "raw_status": "incomplete", "derived_status": "incomplete", "status": "incomplete",
                "raw_error_code": None, "raw_exit_class": None, "raw_returncode": None,
                "terminal_completed_calls": None, "completed_requests": 0,
                "request_reconciliation": "pending", "detail": "incomplete",
                "readiness_only": bool(payload.get("readiness_only", source == "load")),
                "load_gate": policy.get("max_load_ratio") if isinstance(policy, dict) else None,
            }
        elif event["kind"] == "readiness":
            observed = payload.get("observation") if isinstance(payload, dict) else None
            decision = payload.get("decision") if isinstance(payload, dict) else None
            if run_id in runs and isinstance(observed, dict) and isinstance(decision, dict):
                observations.append({
                    "run_id": run_id, "trial": runs[run_id]["trial"],
                    "model_label": runs[run_id]["model_label"], "kind": runs[run_id]["kind"],
                    "time": iso(observed["observed_unix_ns"]),
                    "relative_elapsed_sec": (event["recorded_unix_ns"] - runs[run_id]["started_unix_ns"]) / 1e9,
                    "load_ratio": observed.get("load_ratio"), "eligible": decision.get("eligible"),
                    "probe_seconds": observed.get("duration_seconds"), "swap_delta_mib": None,
                    "swap_limit_mib": 256,
                })
        elif event["kind"] == "validation" and run_id in runs:
            state = payload.get("state") if isinstance(payload, dict) else None
            if state == "load_memory_sample":
                sample = payload.get("observation")
                if isinstance(sample, dict) and runs[run_id]["model_label"] == "12B":
                    started = sample["started_unix_ns"]
                    observations.append({
                        "run_id": run_id, "trial": runs[run_id]["trial"], "model_label": "12B", "kind": "load",
                        "time": iso(started), "relative_elapsed_sec": (started - runs[run_id]["started_unix_ns"]) / 1e9,
                        "load_ratio": None, "eligible": not bool(sample.get("errors")),
                        "probe_seconds": (sample.get("finished_unix_ns", started) - started) / 1e9,
                        "swap_delta_mib": None if sample.get("swap_delta_bytes") is None else sample["swap_delta_bytes"] / (1024 * 1024),
                        "swap_limit_mib": 256,
                    })
        elif event["kind"] == "run_finished" and run_id in runs:
            if run_id in finished:
                raise ValueError("event journal contains duplicate run_finished")
            finished.add(run_id)
            worker = payload.get("worker") if isinstance(payload, dict) else None
            worker = worker if isinstance(worker, dict) else {}
            raw_status = payload.get("status", "failed")
            returncode = worker.get("returncode")
            exit_class = worker.get("exit_class")
            computed = len(_completed_samples(events, run_id)) if source == "optimization" else 0
            terminal_calls = payload.get("completed_calls")
            if source == "optimization" and terminal_calls is not None and (type(terminal_calls) is not int or terminal_calls != computed):
                raise ValueError("terminal completed_calls do not reconcile with passed sample validations")
            derived_status = raw_status
            detail = payload.get("error_code") or ("readiness_only" if runs[run_id]["readiness_only"] else "completed")
            if source == "load" and raw_status == "passed" and returncode != 0:
                derived_status, detail = "failed", "raw passed corrected: worker returncode was not 0"
            if source == "load" and exit_class == "forced_abort":
                detail = payload.get("error_code") or "forced_abort"
            runs[run_id].update(
                raw_status=raw_status, derived_status=derived_status, status=derived_status,
                raw_error_code=payload.get("error_code"), raw_exit_class=exit_class,
                raw_returncode=returncode, terminal_completed_calls=payload.get("completed_calls"),
                completed_requests=computed,
                request_reconciliation=("not_applicable" if terminal_calls is None else "match"),
                detail=detail,
            )
    for run_id, row in runs.items():
        if source == "optimization":
            row["completed_requests"] = len(_completed_samples(events, run_id))
        if run_id not in finished:
            row["status"] = row["derived_status"] = "incomplete"
            row["request_reconciliation"] = "pending"
            row["detail"] = "incomplete"
    return sorted(runs.values(), key=lambda row: row["started_unix_ns"], reverse=True), observations


def _source(source_id: str, path_label: str, chain_sha: str, as_of_seq: int, generated: str) -> dict[str, Any]:
    return {"id": source_id, "label": "Verifizierte lokale Ereignishistorie", "path": path_label,
            "query": {"engine": "sqlite", "language": "sql", "sql": _QUERY, "executed_at": generated,
                      "description": "Vollständiger Export nach SQLite-Schema- und SHA-256-Kettenprüfung.",
                      "tables_used": ["event_journal"], "filters": ["complete source, maximum 2000 events"],
                      "chain_sha256": chain_sha, "as_of_seq": as_of_seq,
                      "transform_file": "tools/product_history_snapshot.py"}}


def snapshot(path: Path, load_database: Path | None = None) -> tuple[dict, dict]:
    dual_source = load_database is not None
    sources = [("optimization", path, "optimization.sqlite3")]
    if load_database is not None:
        sources.append(("load", load_database, "load-screen.sqlite3"))
    if load_database is not None and path.resolve() == load_database.resolve():
        raise ValueError("optimization and load journals must be distinct sources")
    generated = datetime.now(timezone.utc).isoformat()
    all_events: list[dict[str, Any]] = []
    source_exports, run_rows, observations, journal_descriptors = [], [], [], []
    seen_runs: set[str] = set()
    for source_id, source_path, label in sources:
        events, chain_sha = _read_events(source_path)
        run_ids = {event["run_id"] for event in events}
        if seen_runs & run_ids:
            raise ValueError("journals contain overlapping run_id values")
        seen_runs.update(run_ids)
        all_events.extend([{**event, "source_id": source_id} for event in events])
        as_of_seq = events[-1]["seq"] if events else 0
        source_exports.append({"id": source_id, "events": events, "verified_events": len(events), "chain_sha256": chain_sha, "as_of_seq": as_of_seq})
        rows, points = _runs(events, source_id)
        run_rows.extend(rows); observations.extend(points); journal_descriptors.append(_source(source_id, label, chain_sha, as_of_seq, generated))
    if len(run_rows) + len(observations) > MAX_DATASET_ROWS:
        raise ValueError("dashboard datasets exceed the 2000-row bound")
    run_rows.sort(key=lambda row: row["started_unix_ns"], reverse=True)
    attempts = len(run_rows)
    requests = sum(row["completed_requests"] for row in run_rows)
    failed = sum(row["status"] in {"failed", "deferred", "interrupted"} for row in run_rows)
    scope_note = ("Nur verifizierte lokale PROD3-/PROD4-Aufträge aus den angegebenen Primärjournalen. "
                  "Readiness-only und Load-only sind keine Generierungs- oder Performancequalifikation. "
                  "Legacy-Korrektur: ein roher Load-Status passed mit Worker-Returncode ungleich 0 wird hier als failed dargestellt; Rohstatus und Exitfelder bleiben erhalten.")
    chart_rows = [row for row in observations if row["kind"] == "load" and row["model_label"] == "12B" and row["swap_delta_mib"] is not None]
    totals = [{"attempts": attempts, "completed_requests": requests, "failed_or_blocked_attempts": failed}]
    combined_source = {"id": "history", "label": "Verifizierte lokale Optimierungs- und Ladehistorie",
        "path": "optimization-events.json", "query": {"engine": "sqlite", "language": "sql", "sql": _QUERY,
        "executed_at": generated, "description": "Zwei vollständige, separat kettengeprüfte Primärjournale; terminale Status und Modellanfragen werden aus den Ereignissen abgeleitet.",
        "tables_used": ["event_journal"], "filters": ["complete optimization and optional load sources, maximum 2000 events each"],
        "transform_file": "tools/product_history_snapshot.py", "metric_definitions": {
            "attempts": "Anzahl eindeutiger run_started/run_finished-Aufträge.",
            "completed_requests": "Anzahl eindeutiger finaler passed sample_index-Validierungen; Load-only ist nicht anwendbar.",
            "failed_or_blocked_attempts": "Aufträge mit abgeleitetem failed, deferred oder interrupted Status.",
        }, "journal_provenance": [item["id"] for item in journal_descriptors]}}
    sources = [combined_source, *journal_descriptors]
    cards = [
        {"id": "card_attempts", "dataset": "totals", "sourceId": "history", "description": "Eindeutige Aufträge mit Start und Abschluss.", "metrics": [{"label": "Versuche", "field": "attempts", "format": "number"}]},
        {"id": "card_requests", "dataset": "totals", "sourceId": "history", "description": "Eindeutige finale passed sample validations; Load-only ist nicht anwendbar.", "metrics": [{"label": "Anfragen", "field": "completed_requests", "format": "number"}]},
        {"id": "card_failed", "dataset": "totals", "sourceId": "history", "description": "Aufträge mit failed, deferred oder interrupted Status.", "metrics": [{"label": "Aufträge", "field": "failed_or_blocked_attempts", "format": "number"}]},
    ]
    artifact = {"surface": "dashboard", "manifest": {"version": 1, "title": "IronMule · Kalibrierungs- und Ladeverlauf",
        "generatedAt": generated, "snapshotDate": generated, "sources": sources, "cards": cards, "blocks": [
            {"id": "summary", "type": "metric-strip", "cardIds": [card["id"] for card in cards], "sourceId": "history"},
            {"id": "title", "type": "markdown", "sourceId": "history", "body": "# IronMule · Kalibrierungs- und Ladeverlauf\nLokaler, verifizierter Snapshot ohne Nutzerinhalte oder Modellantworten."},
            {"id": "load", "type": "chart", "chartId": "load_history", "sourceId": "history"}, {"id": "runs", "type": "table", "tableId": "run_history", "sourceId": "history"},
            {"id": "scope", "type": "markdown", "sourceId": "history", "body": scope_note}],
        "charts": [{"id": "load_history", "title": "12B Lade-Swapdelta (MiB) vs. 256 MiB Grenze", "type": "line", "dataset": "load_observations", "sourceId": "history",
                     "encodings": {"x": {"field": "relative_elapsed_sec", "type": "quantitative"}, "y": {"fields": ["swap_delta_mib", "swap_limit_mib"], "type": "quantitative"}, "color": {"field": "trial", "type": "nominal", "legend": {"title": "Versuch"}}}}],
        "tables": [{"id": "run_history", "title": "Aufträge", "dataset": "runs", "sourceId": "history", "columns": [
            {"field": "start_utc", "label": "Versuch", "type": "string"}, {"field": "model_label", "label": "Modell", "type": "string"},
            {"field": "kind", "label": "Art", "type": "string"}, {"field": "status", "label": "Status", "type": "string"},
            {"field": "completed_requests", "label": "Anfragen", "type": "number"}],
            "defaultSort": {"field": "start_utc", "direction": "desc"}}]},
        "snapshot": {"version": 1, "status": "ready", "generatedAt": generated, "snapshotDate": generated,
                     "datasets": {"totals": totals, "runs": run_rows, "load_observations": chart_rows}},
        "sources": sources}
    exported_events = all_events if dual_source else source_exports[0]["events"]
    export = {"schema": "ironmule.optimization_export.v2", "generated_at": generated, "verified_events": len(exported_events),
              "events": exported_events, "journals": source_exports, "sha256": canonical_sha256(exported_events)}
    if not dual_source:
        export["schema"] = "ironmule.optimization_export.v1"
    return artifact, export


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--database", type=Path, required=True); parser.add_argument("--load-database", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True); args = parser.parse_args()
    artifact, export = snapshot(args.database, args.load_database)
    args.output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    for name, value in (("artifact.json", artifact), ("optimization-events.json", export)):
        with (args.output_dir / name).open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False); handle.write("\n")
    print(json.dumps({"events": export["verified_events"], "export_sha256": export["sha256"]}))
