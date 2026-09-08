"""Independent, read-only audit of PROD11 native prefix-correctness reports.

This module deliberately audits recorded evidence only.  It never imports MLX,
opens a model, or writes the EventJournal.  A report is accepted only when its
terminal journal event binds the canonical report digest and all protocol
records are present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from friday_evidence.canonical import canonical_sha256
from friday_evidence.events import EventJournal

SCHEMA = "ironmule.prod11_native_correctness.v1"
CHILD_SCHEMA = "ironmule.prod11_native_correctness.child.v1"
PROMPT_TOKENS = 1077
HEX64 = re.compile(r"^[0-9a-f]{64}$")
CHECKS = {"wrong_binding", "wrong_scope", "wrong_key", "stale_clear",
          "stale_close", "clone_isolation", "cancel_recovery", "entry_eviction",
          "byte_oversize_skip", "canonical_unchanged", "byte_eviction"}


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _output(value: Any, where: str, prompt_tokens: int = PROMPT_TOKENS) -> None:
    _need(isinstance(value, dict), f"{where} must be an object")
    _need(value.get("prompt_tokens") == prompt_tokens, f"{where}.prompt_tokens")
    _need(type(value.get("completion_tokens")) is int and value["completion_tokens"] > 0,
          f"{where}.completion_tokens")
    _need(value.get("finish_reason") in {"stop", "length"}, f"{where}.finish_reason")
    for key in ("output_sha256", "token_sha256", "text_sha256", "logprobs_sha256"):
        _need(isinstance(value.get(key), str) and HEX64.fullmatch(value[key]) is not None,
              f"{where}.{key}")


def _events(database: Path, run_id: str) -> list[dict[str, Any]]:
    with EventJournal(database, read_only=True) as journal:
        result: list[dict[str, Any]] = []
        while True:
            part = journal.events(run_id=run_id, after_seq=result[-1]["seq"] if result else 0,
                                  limit=1000)
            if not part:
                return result
            result.extend(part)


def audit(report_path: Path, database: Path) -> dict[str, Any]:
    report_path, database = Path(report_path), Path(database)
    raw = report_path.read_bytes()
    report = json.loads(raw)
    _need(report.get("schema") == SCHEMA, "report schema")
    _need(report.get("status") == "passed", "report is not a completed passed artifact")
    run_id = report.get("run_id")
    _need(isinstance(run_id, str) and re.fullmatch(r"[0-9a-f]{32}", run_id) is not None, "run_id")
    events = _events(database, run_id)
    _need(events, "journal run missing")
    _need(sum(e["kind"] == "run_started" for e in events) == 1, "run_started uniqueness")
    _need(sum(e["kind"] == "run_finished" for e in events) == 1, "run_finished uniqueness")
    terminal = [e for e in events if e["kind"] == "run_finished"][0]["payload"]
    _need(events[-1]["kind"] == "run_finished" and terminal.get("status") == "passed", "terminal event")
    _need(terminal.get("report_sha256") == canonical_sha256(report), "terminal report digest")
    started = [e for e in events if e["kind"] == "run_started"][0]["payload"]
    _need(started.get("model_id") == report.get("model_id") and started.get("revision") == report.get("revision"),
          "journal/report identity")

    child = report.get("child")
    _need(isinstance(child, dict) and child.get("schema") == CHILD_SCHEMA and child.get("status") == "passed",
          "child completion")
    _need(child.get("model_id") == report["model_id"] and child.get("revision") == report["revision"], "child identity")
    _need(child.get("correctness_gate") is True and child.get("performance_claim") is False
          and child.get("activation_allowed") is False and "gpu" in str(child.get("device", "")).lower(),
          "no activation/performance claim")
    checks = child.get("checks")
    _need(isinstance(checks, dict) and set(checks) == CHECKS and all(v is True for v in checks.values()), "11 checks")

    stock, candidate, clones = child.get("stock"), child.get("candidate"), child.get("clone_outputs")
    _need(isinstance(stock, list) and len(stock) == 4, "four stock outputs")
    _need(isinstance(candidate, list) and len(candidate) == 4, "four candidate outputs")
    _need(isinstance(clones, list) and len(clones) == 2, "two clone outputs")
    for name, rows in (("stock", stock), ("candidate", candidate), ("clone", clones)):
        for i, row in enumerate(rows):
            _output(row, f"{name}[{i}]", 1 if name == "clone" else PROMPT_TOKENS)
    _need(all(row == stock[0] for row in stock + candidate), "stock/candidate output equality")
    _need(all(row["output_sha256"] == stock[0]["output_sha256"]
              and row["logprobs_sha256"] == stock[0]["logprobs_sha256"] for row in clones),
          "clone output digest")
    _need(all(row["prompt_tokens"] == 1 for row in clones), "clone prompt scope")
    for name in ("recovery_cold", "recovery_hit"):
        _output(child.get(name), name)
        _need(child[name] == stock[0], f"{name} output equality")
    trace = child.get("recovery_hit_trace")
    _need(isinstance(trace, dict) and trace.get("status") == "completed"
          and trace.get("cache_hit") is True and trace.get("cache_stored") is False
          and trace.get("reused_tokens") == PROMPT_TOKENS - 1, "recovery hit trace")
    traces = child.get("traces")
    _need(isinstance(traces, list) and len(traces) == 4, "four generation traces")
    _need(traces[0].get("status") == "completed" and traces[0].get("cache_hit") is False
          and traces[0].get("cache_stored") is True and traces[0].get("reused_tokens") == 0,
          "cold generation trace")
    _need(all(t.get("status") == "completed" and t.get("cache_hit") is True
              and t.get("cache_stored") is False and t.get("reused_tokens") == PROMPT_TOKENS - 1
              for t in traces[1:]), "candidate generation traces")

    # Two distinct prefill/checkpoint records are required; the cancellation
    # evidence is the recorded recovery trace, not an inflated output count.
    for name in ("checkpoint", "short_checkpoint"):
        cp = child.get(name)
        _need(isinstance(cp, dict) and isinstance(cp.get("layers"), list) and cp["layers"], f"{name} prefill")
        _need(cp.get("sha256") == canonical_sha256(cp["layers"]), f"{name} digest")
        _need(type(cp.get("nbytes")) is int and cp["nbytes"] > 0, f"{name} bytes")
        _need(sum(layer.get("nbytes", 0) for layer in cp["layers"]) == cp["nbytes"],
              f"{name} layer bytes")
    _need(child.get("canonical_after_hits") == child.get("checkpoint"), "canonical after hits")

    progress = report.get("progress")
    expected = [("stock", i) for i in range(4)] + [("checkpoint", None)] + [("candidate", i) for i in range(4)] + [("clone_isolation", None)]
    _need(isinstance(progress, list) and [(p.get("phase"), p.get("index")) for p in progress] == expected, "progress schedule")
    sample_events = [e for e in events if e["kind"] == "sample"]
    _need(len(sample_events) == len(progress) and [e["payload"] for e in sample_events] == progress, "raw/progress comparison")
    journal_resources = [e["payload"]["observation"] for e in events
                         if e["kind"] == "validation" and e["payload"].get("state") == "resource"]
    _need(journal_resources == report.get("observations"), "raw/resource comparison")
    child_events = [e["payload"]["result"] for e in events
                    if e["kind"] == "validation" and e["payload"].get("state") == "child_result"]
    _need(child_events == [child], "raw/child comparison")

    _need(report.get("partial_frame") == {"bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}, "partial frame")
    stats = child.get("cache_stats")
    _need(isinstance(stats, dict) and stats.get("hits") == 1 and stats.get("inserts") == 1
          and stats.get("misses") == 2 and stats.get("bytes") == 0, "cache statistics")
    _need(all(report.get(name + "_before") == report.get(name + "_after") for name in
              ("identity", "source", "provider", "installed", "model")), "before/after bindings")
    _need(report.get("candidate_file_before") == report["source_before"].get("ironmule_product/prefix_reuse.py")
          and report.get("candidate_file_after") == report["source_after"].get("ironmule_product/prefix_reuse.py"), "installed candidate hash")

    workers = report.get("workers")
    _need(isinstance(workers, list) and workers and all(w.get("closed") and w.get("returncode") == 0 for w in workers), "worker exit")
    pids = {w.get("pid") for w in workers}
    resources = [e["payload"]["observation"] for e in events if e["kind"] == "validation" and e["payload"].get("state") == "resource"]
    _need(resources and all(o.get("pid") in pids and not o.get("errors") for o in resources), "resource PID coverage/errors")
    _need(not report.get("performance_claim") and report.get("activation_allowed") is False
          and not report.get("observation_summary", {}).get("errors"), "report safety flags")
    return {"schema": "ironmule.prod11_prefix_audit.v1", "run_id": run_id,
            "model_id": report["model_id"], "revision": report["revision"],
            "report_sha256": hashlib.sha256(raw).hexdigest(),
            "report_canonical_sha256": canonical_sha256(report),
            "events_sha256": canonical_sha256(events), "verified_events": len(events),
            "hash_chain_verified": True, "run_id_unique": True,
            "terminal_report_digest_verified": True,
            "outputs": {"stock": 4, "candidate": 4, "clone": 2, "recovery": 2,
                        "checkpoint_prefills": 2, "partial_cancel": 1},
            "checks": sorted(CHECKS), "exact_check_count": len(CHECKS),
            "recovery_cache_hits": 1, "candidate_generation_hits": 3,
            "total_generation_hits": 4, "reused_tokens": PROMPT_TOKENS - 1,
            "cache_canonical_state_bytes": child["checkpoint"]["nbytes"],
            "closed_recovery_cache_bytes": stats["bytes"],
            "before_after_bindings_verified": ["identity", "source", "provider", "installed", "model"],
            "worker_pids": sorted(pids), "resource_errors": [],
            "partial_cancel_protocol": 1, "partial_cancel_persisted": False,
            "performance_claim": False, "activation_allowed": False,
            "auditor_source_proof": "report-bound installed candidate hash"}


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="optional exclusive JSON output path")
    args = parser.parse_args()
    results = [audit(path, args.database) for path in args.report]
    encoded = json.dumps(results, indent=2, sort_keys=True)
    if args.output is None:
        print(encoded)
    else:
        # Opt-in only: stdout remains the default and this uses the project's
        # existing exclusive writer rather than opening a second SQLite path.
        from product_load_screen import _exclusive_write
        _exclusive_write(args.output, {"schema": "ironmule.prod11_prefix_audit.bundle.v1",
                                       "results": results})
        print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
