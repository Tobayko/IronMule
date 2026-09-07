"""Export a verified local optimization history to a bounded dashboard snapshot.

This research/UI adapter does not run models, start a server or upload data.
The installed product remains independent of the optional dashboard renderer.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from friday_evidence.canonical import canonical_sha256
from friday_evidence.events import EventJournal


def iso(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1e9, timezone.utc).isoformat()


def snapshot(path: Path) -> tuple[dict, dict]:
    with EventJournal(path, read_only=True) as journal:
        total = journal.verify()
        if total > 2000:
            raise ValueError("history exceeds this complete-snapshot adapter's 2000-event bound")
        events = journal.events(limit=1000)
        if len(events) < total:
            events.extend(journal.events(after_seq=events[-1]["seq"], limit=1000))
    if len(events) != total:
        raise ValueError("history changed while exporting; retry as a separate snapshot")
    runs, observations = {}, []
    for event in events:
        payload = event["payload"]
        run_id = event["run_id"]
        if event["kind"] == "run_started":
            runs[run_id] = {"run_id": run_id, "started": iso(event["recorded_unix_ns"]),
                            "start_utc": datetime.fromtimestamp(event["recorded_unix_ns"] / 1e9, timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                            "model": payload["model_id"], "status": "incomplete",
                            "model_label": payload["model_id"].split("/")[-1],
                            "readiness_only": payload["readiness_only"], "model_calls": 0,
                            "load_gate": payload["readiness_policy"]["max_load_ratio"]}
        elif event["kind"] == "readiness":
            observed = payload["observation"]
            observations.append({"run_id": run_id, "time": iso(observed["observed_unix_ns"]),
                                 "probe": len(observations) + 1,
                                 "load_ratio": observed["load_ratio"], "gate": runs[run_id]["load_gate"],
                                 "eligible": payload["decision"]["eligible"],
                                 "model_loaded": payload["model_loaded"],
                                 "free_memory_percent": observed["memory_free_percent"],
                                 "probe_seconds": observed["duration_seconds"]})
        elif event["kind"] == "run_finished" and run_id in runs:
            evaluation = payload.get("evaluation") or {}
            runs[run_id].update(status=payload["status"], model_calls=payload["completed_calls"],
                               detail=payload.get("error_code") or evaluation.get("verdict") or "readiness_only")
    generated = datetime.now(timezone.utc).isoformat()
    source = {"id": "journal", "label": "Verifizierte lokale Optimierungshistorie",
              "path": "optimization-events.json",
              "query": {"engine": "sqlite", "language": "sql",
                        "sql": "SELECT seq, recorded_unix_ns, run_id, kind, payload_json, prev_sha256, sha256 FROM event_journal ORDER BY seq ASC;",
                        "description": "Vollständiger Export nach Schema- und SHA-256-Kettenprüfung; Aufträge aus run_started/run_finished, Sensorwerte aus readiness.",
                        "tables_used": ["event_journal"], "executed_at": generated,
                        "filters": ["Vollständige Historie bis maximal 2000 Ereignisse; keine Stichprobe"],
                        "metric_definitions": {"model_calls": "Summe completed_calls terminaler Aufträge; Readiness-only-Aufträge führen keine Modellgenerierung aus.",
                                               "load_ratio": "load_1m geteilt durch logische CPU-Kerne; keine prozentuale CPU-Auslastung."}}}
    rows = list(runs.values())
    deferred = sum(row["status"] == "deferred" for row in rows)
    failed = sum(row["status"] == "failed" for row in rows)
    model_calls = sum(row["model_calls"] for row in rows)
    artifact = {
        "surface": "dashboard",
        "manifest": {"version": 1, "title": "IronMule · Kalibrierungsverlauf",
                     "generatedAt": generated, "sources": [source],
                     "blocks": [
                         {"id": "title", "type": "markdown", "body": "# IronMule · Kalibrierungsverlauf\nLokaler, unveränderlicher Snapshot der verifizierten Ereignishistorie."},
                         {"id": "summary", "type": "markdown", "sourceId": "journal",
                          "body": f"**{len(rows)} Aufträge · {failed} fehlgeschlagen · {deferred} zurückgestellt · {model_calls} abgeschlossene Modellanfragen**\n\nReadiness-only ist kein Modelltest. Diese Ansicht begründet weder einen Speedup noch eine automatische Aktivierung."},
                         {"id": "load", "type": "chart", "chartId": "load_history"},
                         {"id": "runs", "type": "table", "tableId": "run_history"},
                         {"id": "scope", "type": "markdown", "sourceId": "journal",
                          "body": "## Geltungsbereich\nNur die hier protokollierten PROD3-Aufträge. Frühere Gemma-Tests sind separate Evidenz.\n\nLast pro Kern ist keine CPU-Prozentmessung. Low Power, Thermik, Speicher und zeitliche Stabilität sind zusätzliche Gates. Proben sind Einzelbeobachtungen; Zwischenräume bleiben unbeobachtet.\n\nKeine Prompts oder Modellantworten sind enthalten."}],
                     "charts": [{"id": "load_history", "title": "Last pro Kern und Readiness-Grenze",
                                 "type": "line", "dataset": "observations", "sourceId": "journal",
                                 "encodings": {"x": {"field": "probe", "type": "quantitative"},
                                               "y": {"fields": ["load_ratio", "gate"], "type": "quantitative"}}}],
                     "tables": [{"id": "run_history", "title": "Aufträge", "dataset": "runs", "sourceId": "journal",
                                 "columns": [{"field": "start_utc", "label": "Start (UTC)", "type": "string"},
                                             {"field": "model_label", "label": "Modell", "type": "string"},
                                             {"field": "status", "label": "Status", "type": "string"},
                                             {"field": "detail", "label": "Ergebnis / Grund", "type": "string"},
                                             {"field": "model_calls", "label": "Modellanfragen", "type": "number"}],
                                 "defaultSort": {"field": "start_utc", "direction": "desc"}}]},
        "snapshot": {"version": 1, "status": "ready", "generatedAt": generated,
                     "datasets": {"observations": observations, "runs": rows}},
        "sources": [source],
    }
    export = {"schema": "ironmule.optimization_export.v1", "generated_at": generated,
              "verified_events": total, "events": events, "sha256": canonical_sha256(events)}
    return artifact, export


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    artifact, export = snapshot(args.database)
    args.output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    for name, value in (("artifact.json", artifact), ("optimization-events.json", export)):
        with (args.output_dir / name).open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
    print(json.dumps({"events": export["verified_events"], "export_sha256": export["sha256"]}))
