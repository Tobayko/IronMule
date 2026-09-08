"""Read-only integrity audit for completed PROD14 native variant reports.

This is deliberately metadata-only: it imports no model, MLX, or product
worker code and never opens the journal for writing.  Counts are derived from
the recorded samples and EventJournal, rather than trusted from planned fields.
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

SCHEMA = "ironmule.prod14_variant_qualification.v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX32 = re.compile(r"^[0-9a-f]{32}$")
FULL_KEYS = ("output_sha256", "token_sha256", "text_sha256", "prompt_ids_sha256",
             "prompt_tokens", "completion_tokens", "finish_reason")
BOUNDINGS = ("identity", "source", "provider", "installed", "metadata")
ORACLES = {
    "mlx-community/gemma-3-1b-it-4bit": ROOT / "research/raw/PROD10_1B_open_20260907_attempt1.json",
    "mlx-community/gemma-3-4b-it-4bit": ROOT / "research/raw/PROD10_4B_open_20260907_attempt1.json",
    "mlx-community/gemma-3-12b-it-4bit": ROOT / "research/raw/PROD10_12B_open_20260907_attempt2.json",
}


def _need(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _events(database: Path, run_id: str) -> list[dict[str, Any]]:
    with EventJournal(database, read_only=True) as journal:
        result: list[dict[str, Any]] = []
        while True:
            part = journal.events(run_id=run_id,
                                  after_seq=result[-1]["seq"] if result else 0,
                                  limit=1000)
            if not part:
                return result
            result.extend(part)


def _full(row: dict[str, Any], where: str) -> None:
    _need(isinstance(row, dict), where)
    for key in FULL_KEYS:
        value = row.get(key)
        if key.endswith("_sha256"):
            _need(isinstance(value, str) and HEX64.fullmatch(value) is not None,
                  f"{where}.{key}")
        elif key in ("prompt_tokens", "completion_tokens"):
            _need(type(value) is int and value > 0, f"{where}.{key}")
    _need(row.get("finish_reason") == "length", f"{where}.finish_reason")


def _metadata_only_sample(sample: dict[str, Any], where: str) -> None:
    # Candidate records keep the actual rendered-prompt digest in metadata;
    # stock oracle records carry it at the sample boundary.
    value = dict(sample)
    if value.get("prompt_ids_sha256") is None:
        value["prompt_ids_sha256"] = (value.get("metadata") or {}).get("prompt_ids_sha256")
    _full(value, where)
    _need(sample.get("matches_reference") is True, f"{where}.matches_reference")


def audit(report_path: Path, database: Path) -> dict[str, Any]:
    report = json.loads(Path(report_path).read_text())
    _need(report.get("schema") == SCHEMA, "report schema")
    _need(report.get("status") == "passed", "report not passed")
    run_id = report.get("run_id")
    _need(isinstance(run_id, str) and HEX32.fullmatch(run_id) is not None, "run_id")
    events = _events(Path(database), run_id)
    _need(events, "journal run missing")
    _need(sum(e["kind"] == "run_started" for e in events) == 1, "run_started uniqueness")
    _need(sum(e["kind"] == "run_finished" for e in events) == 1, "run_finished uniqueness")
    _need(events[-1]["kind"] == "run_finished", "run_finished must be terminal")
    started = next(e["payload"] for e in events if e["kind"] == "run_started")
    terminal = events[-1]["payload"]
    _need(terminal.get("status") == "passed", "journal terminal status")
    _need(terminal.get("report_sha256") == canonical_sha256(report),
          "canonical report digest binding")
    _need(started.get("model_id") == report.get("model_id")
          and started.get("revision") == report.get("revision")
          and started.get("variant") == report.get("variant"), "journal/report identity")
    _need(report.get("performance_claim") is False
          and report.get("activation_allowed") is False, "safety flags")

    samples = report.get("samples")
    _need(isinstance(samples, list), "samples")
    sample_events = [e["payload"]["sample"] for e in events
                     if e["kind"] == "validation" and e["payload"].get("state") == "request_sample"]
    _need(sample_events == samples, "journal/report sample comparison")
    stock = [s for s in samples if s.get("backend") == "stock_oracle_reused"]
    direct = [s for s in samples if s.get("backend") == "candidate_direct"]
    http = [s for s in samples if s.get("backend") == "candidate_http"]
    cancel = [s for s in samples if s.get("backend") == "candidate_cancel"]
    recovery = [s for s in samples if s.get("backend") == "candidate_recovery"]
    _need(len(stock) == 4 and all(s.get("reused_record") is True
                                  and s.get("timing_reused") is False for s in stock),
          "four reused oracle records")
    oracle = stock[0]
    _need(all(all(s.get(k) == oracle.get(k) for k in FULL_KEYS + ("prompt_ids_sha256",))
               for s in stock), "oracle stability")
    _need(report.get("stock_oracle", {}).get("path") == ORACLES[report["model_id"]].relative_to(ROOT).as_posix(),
          "oracle path mapping")
    oracle_path = ORACLES[report["model_id"]]
    _need(hashlib.sha256(oracle_path.read_bytes()).hexdigest()
          == report.get("stock_oracle", {}).get("sha256"), "oracle file digest")
    frozen = json.loads(oracle_path.read_text())
    _need(frozen.get("status") == "passed" and frozen.get("model_id") == report["model_id"]
          and frozen.get("revision") == report["revision"], "frozen oracle identity")
    _need(all(report["identity_before"].get(k) == frozen["identity_before"].get(k)
              for k in ("model_sha256", "environment_sha256", "hardware_sha256"))
          and report.get("provider_before") == frozen.get("provider_before"),
          "oracle identity/provider binding")
    frozen_rows = [s for s in frozen.get("samples", [])
                   if s.get("backend") == "stock" and s.get("case") == "long_8"]
    _need(len(frozen_rows) == 4 and all(
        all(s.get(k) == frozen_rows[0].get(k) for k in FULL_KEYS + ("prompt_ids_sha256",))
        for s in frozen_rows), "frozen stock rows")
    _need(all(all(s.get(k) == frozen_rows[i].get(k) for k in FULL_KEYS + ("prompt_ids_sha256",))
               for i, s in enumerate(stock)), "report/oracle stock rows")
    _need(all("wall_seconds" not in s and "first_token_seconds" not in s
              and "timing_seconds" not in s and "metrics" not in s for s in stock),
          "reused stock timing exclusion")
    _need((report.get("output_accounting") or {}).get("stock_new_runs") == 0, "stock new runs")
    _need(len(direct) == 4 and len(http) == 3, "direct/http count")
    _need([s.get("repeat") for s in stock] == [0, 1, 2, 3]
          and [s.get("repeat") for s in direct] == [0, 1, 2, 3]
          and [s.get("repeat") for s in http] == [0, 1, 2], "repeat sequence")
    _need(all("repeat" not in s for s in cancel + recovery), "partial repeat fields")
    for i, row in enumerate(direct + recovery):
        _metadata_only_sample(row, f"candidate[{i}]")
        _need((row.get("prompt_ids_sha256") or (row.get("metadata") or {}).get("prompt_ids_sha256"))
              == oracle.get("prompt_ids_sha256"),
              "rendered prompt identity")
    for row in http:
        _need((row.get("prompt_ids_sha256") or (row.get("metadata") or {}).get("prompt_ids_sha256"))
              == oracle.get("prompt_ids_sha256"), "rendered prompt identity")
        _need(row.get("matches_reference") is True
              and row.get("text_sha256") == oracle.get("text_sha256")
              and row.get("prompt_tokens") == oracle.get("prompt_tokens")
              and row.get("completion_tokens") == oracle.get("completion_tokens")
              and row.get("finish_reason") == oracle.get("finish_reason"), "http exact gates")
    _need(all(row.get("output_sha256") == oracle.get("output_sha256")
              and row.get("token_sha256") == oracle.get("token_sha256")
              and row.get("text_sha256") == oracle.get("text_sha256")
              and row.get("prompt_tokens") == oracle.get("prompt_tokens")
              and row.get("completion_tokens") == oracle.get("completion_tokens")
              and row.get("finish_reason") == oracle.get("finish_reason")
              for row in direct + recovery), "direct/recovery exact gates")

    variant = report.get("variant")
    _need(variant in {"prefix_reuse", "current_engine"}, "variant")
    _need(len(recovery) == (1 if variant == "prefix_reuse" else 0)
          and len(cancel) == (1 if variant == "prefix_reuse" else 0), "variant sample shape")
    if variant == "prefix_reuse":
        for i, row in enumerate(direct):
            meta = row.get("metadata") or {}; prefix = meta.get("prefix_cache") or {}
            trace = prefix.get("trace") or {}; stats = prefix.get("stats") or {}
            hit = i > 0
            _need(prefix.get("status") == "accepted"
                  and prefix.get("commit_status") == ("no_pending_commit" if hit else "stored")
                  and trace.get("status") == "completed" and trace.get("cache_hit") is hit
                  and trace.get("cache_stored") is (not hit)
                  and trace.get("reused_tokens") == (1076 if hit else 0), "prefix lifecycle")
            _need(type(stats.get("entries")) is int and type(stats.get("bytes")) is int
                  and stats["entries"] >= (0 if hit else 1)
                  and stats["bytes"] >= (0 if hit else 1), "prefix cache stats")
        for i, row in enumerate(http):
            meta = row.get("metadata") or {}; prefix = meta.get("prefix_cache") or {}
            trace = prefix.get("trace") or {}; stats = prefix.get("stats") or {}
            hit = i > 0
            _need(prefix.get("status") == "accepted"
                  and prefix.get("commit_status") == ("no_pending_commit" if hit else "stored")
                  and trace.get("status") == "completed" and trace.get("cache_hit") is hit
                  and trace.get("cache_stored") is (not hit)
                  and trace.get("reused_tokens") == (1076 if hit else 0), "prefix lifecycle")
            _need(type(stats.get("entries")) is int and type(stats.get("bytes")) is int
                  and stats["entries"] >= (0 if hit else 1)
                  and stats["bytes"] >= (0 if hit else 1), "prefix cache stats")
        c = cancel[0]; pm = c.get("metadata") or {}; pc = pm.get("prefix_cache") or {}
        tr = pc.get("trace") or {}; st = pc.get("stats") or {}
        _need(c.get("actual_token_observed") is True and c.get("finish_reason") == "cancelled"
              and pc.get("status") == "discarded" and pc.get("commit_status") == "no_pending_commit"
              and tr.get("status") == "cancelled" and tr.get("cache_stored") is False
              and st.get("entries") == 0 and st.get("bytes") == 0, "cancel protocol")
        _need(report.get("cancellation_recovery", {}).get("same_worker_pid") == c.get("worker_pid"),
              "same worker recovery")
        rec = recovery[0]; rt = ((rec.get("metadata") or {}).get("prefix_cache") or {}).get("trace") or {}
        _need(rt.get("reused_tokens") == 0 and rt.get("cache_hit") is False
              and rt.get("cache_stored") is True
              and report.get("cancellation_recovery", {}).get("worker_ready") is True,
              "recovery lifecycle")
    else:
        ready_engine = report.get("engine_ready_metadata") or {}
        _need(ready_engine.get("configuration") == "current_profile"
              and ready_engine.get("profile_source") == "baseline_no_compatible_profile"
              and ready_engine.get("profile_available") is False
              and ready_engine.get("historical_configuration_candidate") is None,
              "engine baseline configuration")
        baseline_knobs = {"capacity_slack": 0, "compiled_fixed_cache": False,
                          "fuse_projections": False, "fused_argmax": False,
                          "head_skip_prefill": False, "prefill_into_fixed": False,
                          "readback_every": 1, "speculate_k": 0,
                          "speculate_ngram": 3, "wired_fraction": 0.0}
        _need(ready_engine.get("selected_knobs") == baseline_knobs, "engine baseline knobs")
        for row in direct + http:
            engine = (row.get("metadata") or {}).get("engine") or {}
            _need(engine.get("fallback_used") is False and engine.get("fallback_count") == 0,
                  "engine fallback")

    for name in BOUNDINGS:
        before, after = report.get(f"{name}_before"), report.get(f"{name}_after")
        _need(isinstance(before, dict) and bool(before) and isinstance(after, dict) and bool(after),
              f"{name} binding presence")
        _need(before == after, f"{name} binding")
    workers = report.get("workers")
    _need(isinstance(workers, list) and workers, "workers")
    _need(all(w.get("closed") is True and w.get("returncode") == 0 for w in workers), "worker reaped")
    pids = {w.get("pid") for w in workers}
    if variant == "prefix_reuse":
        _need(report.get("cancellation_recovery", {}).get("same_worker_pid") in pids,
              "recovery worker pid")
    resources = [e["payload"].get("observation") for e in events
                 if e["kind"] == "validation" and e["payload"].get("state") == "resource_observation"]
    _need(resources and all(o.get("pid") in pids and o.get("errors") == [] for o in resources),
          "resource observations")
    summary = report.get("observation_summary") or {}
    _need(isinstance(summary, dict) and isinstance(summary.get("errors"), dict), "observer summary")
    maxes = {key: max(o.get(key, 0) for o in resources)
             for key in ("peak_footprint_bytes", "physical_footprint_bytes", "rss_bytes", "wired_bytes")}
    _need(summary.get("max_memory_bytes") == maxes
          and summary.get("memory_sample_count") == len(resources)
          and summary.get("sample_count") == len(resources)
          and summary.get("swap_sample_count") == len(resources)
          and summary.get("latest_system_swap_delta_bytes") == resources[-1].get("system_swap_delta_bytes"),
          "observer summary aggregates")
    expected_full = len(direct) + len(http) + len(recovery)
    account = report.get("output_accounting") or {}
    _need(account.get("actual_new_full") == expected_full
          and account.get("actual_partial_cancelled") == len(cancel)
          and account.get("stock_reused_records") == len(stock), "derived accounting")
    return {"schema": "ironmule.prod14_variant_audit.v1", "run_id": run_id,
            "model_id": report["model_id"], "revision": report["revision"],
            "variant": variant, "report_sha256": hashlib.sha256(Path(report_path).read_bytes()).hexdigest(),
            "report_canonical_sha256": canonical_sha256(report),
            "events_sha256": canonical_sha256(events), "verified_events": len(events),
            "hash_chain_verified": True, "terminal_report_digest_verified": True,
            "counts": {"planned": account.get("planned_candidate_new_full"),
                       "new_full": expected_full, "reused_stock": len(stock),
                       "partial_cancelled": len(cancel), "direct": len(direct), "http": len(http)},
            "exact_output": True, "prompt_identity": True,
            "prefix_cancel_recovery": variant == "prefix_reuse",
            "engine_fallback": False if variant == "current_engine" else None,
            "before_after_bindings": list(BOUNDINGS), "worker_pids": sorted(pids),
            "resource_errors": [], "resource_swap": {
                "max_system_swap_delta_bytes": max(o.get("system_swap_delta_bytes", 0) for o in resources),
                "latest_system_swap_delta_bytes": resources[-1].get("system_swap_delta_bytes")},
            "engine_configuration": ({"configuration": "current_profile",
                "profile_source": "baseline_no_compatible_profile",
                "profile_available": False, "baseline_knobs": True}
                if variant == "current_engine" else None),
            "performance_claim": False, "activation_allowed": False}


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--report", action="append", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = [audit(path, args.database) for path in args.report]
    encoded = json.dumps(results, indent=2, sort_keys=True)
    if args.output:
        from product_load_screen import _exclusive_write
        _exclusive_write(args.output, {"schema": "ironmule.prod14_variant_audit.bundle.v1", "results": results})
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
