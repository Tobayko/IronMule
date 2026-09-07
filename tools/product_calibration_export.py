"""Export/reconstruct one completed calibration from its verified event journal."""

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


def export_run(database: Path, run_id: str) -> dict:
    events = []
    with EventJournal(database, read_only=True) as journal:
        while True:
            rows = journal.events(run_id=run_id, after_seq=events[-1]["seq"] if events else 0, limit=1000)
            if not rows:
                break
            events.extend(rows)
            if len(events) > 10000:
                raise ValueError("run exceeds the export bound")
    if not events or events[-1]["kind"] != "run_finished":
        raise ValueError("only a terminal run may be exported")
    start = next((e["payload"] for e in events if e["kind"] == "run_started"), None)
    if start is None:
        raise ValueError("run has no starting specification")
    terminal = events[-1]["payload"]
    report = {"schema": "ironmule.calibration.v1", "run_id": run_id,
              "plan_id": start["plan_id"], "model_id": start["model_id"],
              "revision": start["revision"], "status": terminal.get("report_status", terminal["status"]),
              "activation_allowed": False, "resource_valid": False,
              "error_code": terminal.get("error_code"), "error_type": terminal.get("error_type"),
              "error_stage": terminal.get("error_stage"), "error_detail": terminal.get("error_detail"),
              "evaluation": terminal.get("evaluation"), "budget": terminal.get("budget")}
    if "load_monitor" in start:
        report["load_monitor"] = start["load_monitor"]
    samples, resources, workers, timings, load_memory, exits = {}, {}, {}, {}, {}, {}
    for event in events:
        payload, kind = event["payload"], event["kind"]
        if kind == "readiness" and payload["decision"].get("stable") and "memory_total_bytes" not in report:
            observed = payload["observation"]
            report.update(memory_total_bytes=observed["memory_total_bytes"], swap_baseline_bytes=observed["swap_used_bytes"])
        if kind == "sample":
            samples[payload["sample_index"]] = payload
        if kind == "worker_started":
            workers[payload["worker_index"]] = {key: payload[key] for key in ("worker_index", "started", "closed", "pid")}
        if kind != "validation":
            continue
        state = payload.get("state")
        if state in {"load_memory_sample", "load_memory_ready"}:
            worker_index = payload.get("worker_index")
            if type(worker_index) is int:
                row = load_memory.setdefault(worker_index, {"worker_index": worker_index, "samples": [], "ready": None, "ready_observation": None})
                if state == "load_memory_sample":
                    row["samples"].append(payload.get("observation"))
                else:
                    row["ready"] = payload.get("ready")
                    row["ready_observation"] = payload.get("observation")
        if state == "identity_bound":
            report["identity_before"] = payload["identity"]
        elif state == "evaluated":
            report["identity_after"] = payload["identity"]
            report["evaluation"] = payload["evaluation"]
        elif state == "worker_cleanup":
            worker = payload["worker"]
            workers[worker["worker_index"]] = worker
            if "timing" in payload:
                timings[worker["worker_index"]] = payload["timing"]
            if "worker_exit" in payload:
                exit_row = payload["worker_exit"]
                if isinstance(exit_row, dict) and type(exit_row.get("worker_index")) is int:
                    exits[exit_row["worker_index"]] = exit_row
        if "sample" in payload:
            samples[payload["sample"]["sample_index"]] = payload["sample"]
        elif "sample_index" in payload and payload.get("status") == "failed":
            samples[payload["sample_index"]] = payload
        if "resource_event" in payload:
            resource = payload["resource_event"]
            resources[resource["sample_index"]] = resource
    report.update(samples=[samples[k] for k in sorted(samples)],
                  resource_events=[resources[k] for k in sorted(resources)],
                  workers=[workers[k] for k in sorted(workers)],
                  worker_timings=[timings[k] for k in sorted(timings)])
    if load_memory:
        report["worker_load_memory"] = [load_memory[k] for k in sorted(load_memory)]
    if exits:
        report["worker_exit_codes"] = [exits[k] for k in sorted(exits)]
    report["resource_valid"] = terminal["status"] == "finished" and report["status"] == "measured"
    return {"schema": "ironmule.calibration_export.v1", "exported_at": datetime.now(timezone.utc).isoformat(),
            "events_sha256": canonical_sha256(events), "report_sha256": canonical_sha256(report),
            "report": report, "events": events}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export_run(args.database, args.run_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": result["report"]["status"], "samples": len(result["report"]["samples"]),
                      "report_sha256": result["report_sha256"]}))
