"""Read-only evidence audit for the completed PROD12 server-memory run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from friday_evidence.canonical import canonical_sha256
from friday_evidence.events import EventJournal

SCHEMA = "ironmule.prod12.server-memory.v1"
FORBIDDEN = {"prompt", "prompts", "messages", "content", "text", "tokens", "token_ids", "answers", "snapshot_path"}


def need(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def privacy(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        need(not (FORBIDDEN & set(value)), f"privacy forbidden field at {path}")
        for k, v in value.items():
            need(not k.startswith("/"), f"absolute path key at {path}")
            privacy(v, f"{path}.{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            privacy(v, f"{path}[{i}]")
    elif isinstance(value, str):
        need(not value.startswith("/") and "/Users/" not in value and "/private/" not in value,
             f"absolute/private path at {path}")


def events_for(database: Path, run_id: str) -> list[dict[str, Any]]:
    with EventJournal(database, read_only=True) as journal:
        result: list[dict[str, Any]] = []
        while True:
            part = journal.events(run_id=run_id, after_seq=result[-1]["seq"] if result else 0, limit=1000)
            if not part:
                return result
            result.extend(part)


def audit(report_path: Path, database: Path) -> dict[str, Any]:
    report_path, database = Path(report_path), Path(database)
    raw = report_path.read_bytes()
    report = json.loads(raw)
    privacy(report)
    need(report.get("schema") == SCHEMA and report.get("status") == "passed", "completed report")
    run_id = report.get("run_id")
    need(isinstance(run_id, str) and len(run_id) == 32, "run id")
    events = events_for(database, run_id)
    need(len(events) > 0 and events[-1]["kind"] == "run_finished", "terminal journal event")
    need(sum(e["kind"] == "run_started" for e in events) == 1 and sum(e["kind"] == "run_finished" for e in events) == 1,
         "run event uniqueness")
    terminal = events[-1]["payload"]
    need(terminal.get("status") == "passed" and terminal.get("report_sha256") == canonical_sha256(report),
         "terminal report digest")
    started = next(e["payload"] for e in events if e["kind"] == "run_started")
    need(started.get("model_id") == report.get("model_id") and started.get("server_pid") != started.get("model_pid"),
         "run identity")

    from product_server_memory import request_plan
    plan = request_plan()
    reference = report.get("reference", {})
    reference_path = ROOT / "research/raw/PROD10_12B_open_20260907_attempt2.json"
    reference_bytes = reference_path.read_bytes()
    need(reference.get("sha256") == hashlib.sha256(reference_bytes).hexdigest(), "bound stock reference file")
    stock = json.loads(reference_bytes)
    need(stock.get("status") == "passed" and stock.get("model_id") == report.get("model_id")
         and stock.get("revision") == report.get("revision"), "stock reference scope")
    for case, expected in reference.get("cases", {}).items():
        rows = [r for r in stock["samples"] if r.get("backend") == "stock" and r.get("case") == case]
        need(len(rows) == 4 and all(all(r.get(k) == v for k, v in expected.items()) for r in rows),
             "independent stock reference equality")
    samples = report.get("samples")
    need(isinstance(samples, list) and len(samples) == len(plan) == 16, "16 request samples")
    for i, (sample, expected) in enumerate(zip(samples, plan)):
        need(all(sample.get(k) == expected.get(k) for k in expected), f"request plan sample {i}")
        need(type(sample.get("prompt_tokens")) is int and type(sample.get("completion_tokens")) is int,
             f"request counts {i}")
        need(sample.get("finish_reason") in {"stop", "length"}, f"finish reason {i}")
        need(isinstance(sample.get("text_sha256"), str) and len(sample["text_sha256"]) == 64, f"HTTP text digest {i}")
        need("token_ids" not in sample and "tokens" not in sample, f"HTTP token-id claim {i}")
        expected_output = reference["cases"][sample["case"]]
        need(all(sample.get(k) == expected_output[k]
                 for k in ("text_sha256", "prompt_tokens", "completion_tokens", "finish_reason")),
             f"exact HTTP/stock equality {i}")

    request_events = [e["payload"]["sample"] for e in events if e["kind"] == "validation" and e["payload"].get("state") == "request_sample"]
    need(request_events == samples, "journal/report request samples")

    journal_obs = [e["payload"]["observation"] for e in events if e["kind"] == "validation" and e["payload"].get("state") == "resource_observation"]
    observations = report.get("observations")
    need(isinstance(observations, dict), "observations object")
    normalized = {"model_worker": [], "server": []}
    for row in journal_obs:
        label = row.get("label")
        if label == "http_server": label = "server"
        need(label in normalized, "observation label")
        normalized[label].append(row)
    need(normalized == observations, "journal/report observations")
    for label, rows in normalized.items():
        need(rows and all(isinstance(r.get("pid"), int) and not r.get("errors") for r in rows), f"{label} observations")

    processes = report.get("processes")
    need(isinstance(processes, list) and len(processes) == 2, "two processes")
    by_role = {p.get("role"): p for p in processes}
    need(set(by_role) == {"http_server", "model_worker"}, "process roles")
    need(all(p.get("closed") is True and p.get("returncode") == 0 for p in processes), "process exits")
    server, model = by_role["http_server"], by_role["model_worker"]
    need(server["pid"] != model["pid"] and server["protocol"].get("server_pid") == server["pid"]
         and server["protocol"].get("model_pid") == model["pid"]
         and server["protocol"].get("model_returncode") == 0, "server/model protocol")
    need(all(r["pid"] == server["pid"] for r in normalized["server"])
         and all(r["pid"] == model["pid"] for r in normalized["model_worker"]), "resource PID binding")
    health = report.get("health_final", {})
    need(health.get("ready") is True and health.get("completed_requests") == 16
         and all(health.get(k) == 0 for k in ("active_requests", "queued_requests", "failed_requests", "cancelled_requests")),
         "final server counters")

    summaries = report.get("observation_summary")
    need(isinstance(summaries, dict), "observation summaries")
    recomputed = {}
    for label, rows in normalized.items():
        maxima = {k: max(r[k] for r in rows if type(r.get(k)) is int)
                  for k in ("rss_bytes", "physical_footprint_bytes", "peak_footprint_bytes", "wired_bytes")}
        recomputed[label] = {"sample_count": len(rows), "memory_sample_count": sum(type(r.get("rss_bytes")) is int for r in rows),
                             "max_memory_bytes": maxima, "errors": {}}
    for label, summary in summaries.items():
        for key in ("sample_count", "memory_sample_count", "max_memory_bytes", "errors"):
            need(summary.get(key) == recomputed[label][key], f"summary {label}.{key}")

    for name in ("identity", "source", "provider", "installed", "metadata"):
        need(isinstance(report.get(name + "_before"), dict) and bool(report[name + "_before"])
             and report[name + "_before"] == report.get(name + "_after"), f"binding {name}")
    need(report.get("performance_claim") is False and report.get("activation_allowed") is False, "safety flags")
    return {"schema": "ironmule.prod12.server-memory-audit.v1", "run_id": run_id,
            "model_id": report["model_id"], "revision": report["revision"],
            "report_sha256": hashlib.sha256(raw).hexdigest(),
            "report_canonical_sha256": canonical_sha256(report), "events": len(events),
            "hash_chain_verified": True, "request_samples": 16,
            "processes": {"server_pid": server["pid"], "model_pid": model["pid"], "both_exit_0": True},
            "observations": {k: len(v) for k, v in normalized.items()},
            "max_memory_bytes": {k: v["max_memory_bytes"] for k, v in recomputed.items()},
            "bindings_verified": ["identity", "source", "provider", "installed", "metadata"],
            "performance_claim": False, "activation_allowed": False,
            "claims": ["correctness_and_memory_observation_only", "no_speedup_or_leak_claim"]}


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--database", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = audit(args.report, args.database)
    from product_load_screen import _exclusive_write
    _exclusive_write(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
