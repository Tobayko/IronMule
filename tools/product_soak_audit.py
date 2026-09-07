"""Model-free, read-only independent audit of the completed PROD10-S study."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from friday_evidence.canonical import canonical_sha256
from friday_evidence.events import EventJournal


def audit(report_path, database):
    raw = report_path.read_bytes()
    report = json.loads(raw)
    assert report["status"] == "passed"
    assert report["soak_seconds_requested"] == 3600
    assert report["soak"]["wall_seconds"] >= 3600
    run_id = report["run_id"]
    events = []
    with EventJournal(database, read_only=True) as journal:
        while True:
            part = journal.events(run_id=run_id, after_seq=events[-1]["seq"] if events else 0, limit=1000)
            if not part:
                break
            events.extend(part)
    assert sum(e["kind"] == "run_started" for e in events) == 1
    assert sum(e["kind"] == "run_finished" for e in events) == 1
    terminal = events[-1]["payload"]
    assert terminal["status"] == "passed"
    assert terminal["report_sha256"] == canonical_sha256(report)
    sample_events = [e for e in events if e["kind"] == "validation"
                     and e["payload"].get("state") == "request_sample"]
    samples = [e["payload"]["sample"] for e in sample_events]
    assert samples == report["samples"]
    soak = [s for s in samples if s["backend"] == "soak"]
    assert [s["index"] for s in soak] == list(range(len(soak)))
    assert len(soak) == report["soak"]["requests"]
    assert not any(s["backend"] == "soak_failure" for s in samples)
    stock = {s["case"]: s for s in samples if s["backend"] == "stock"}
    for s in soak:
        assert all(s[k] == stock[s["case"]][k] for k in
                   ("text_sha256", "prompt_tokens", "completion_tokens", "finish_reason"))
    groups = defaultdict(list)
    for s in soak:
        groups[s["batch_index"]].append(s)
    index = 0
    while index < len(soak):
        width = 4 if index > 0 and index % 12 == 0 else 1
        rows = groups[index]
        assert [s["index"] for s in rows] == list(range(index, index + width))
        assert all(s["concurrency"] == width for s in rows)
        index += width
    health = report["health_final"]
    assert health["ready"] and health["active_requests"] == health["queued_requests"] == health["failed_requests"] == 0
    assert health["cancelled_requests"] == 1 and health["completed_requests"] == 11 + len(soak)
    assert len(report["workers"]) == 2
    assert all(w["closed"] and w["returncode"] == 0 for w in report["workers"])
    for k in ("identity", "source", "provider", "installed", "metadata"):
        assert report[k + "_before"] == report[k + "_after"]
    assert not report["observation_summary"]["errors"]
    # EventJournal supplies original recorded timestamps. The trend window is
    # explicitly first-to-last completed soak request, not an invented start.
    import sqlite3
    with sqlite3.connect(database.absolute().as_uri() + "?mode=ro", uri=True) as db:
        stamped = [(t, json.loads(p)) for t, p in db.execute(
            "SELECT recorded_unix_ns,payload_json FROM event_journal WHERE run_id=? AND kind='validation' ORDER BY seq",
            (run_id,))]
    soak_stamps = [t for t, p in stamped if p.get("state") == "request_sample" and p["sample"]["backend"] == "soak"]
    start, end = soak_stamps[0], soak_stamps[-1]
    resource = [(t, p["observation"]) for t, p in stamped if p.get("state") == "resource_observation"]
    product_pid = next(w["pid"] for w in report["workers"] if w["backend"] == "product")
    product = [(t, o) for t, o in resource if start <= t <= end and o["label"] == "product"]
    assert product and {o["pid"] for _, o in product} == {product_pid}
    quarters = []
    for q in range(4):
        left, right = start + (end - start) * q // 4, start + (end - start) * (q + 1) // 4
        rows = [o for t, o in product if left <= t and (t < right or q == 3 and t == right)]
        assert rows and all(not o["errors"] for o in rows)
        quarters.append({"quarter": q + 1, "samples": len(rows),
            "current_footprint_median_bytes": statistics.median(o["physical_footprint_bytes"] for o in rows),
            "rss_median_bytes": statistics.median(o["rss_bytes"] for o in rows),
            "swap_delta_median_bytes": statistics.median(o["system_swap_delta_bytes"] for o in rows)})
    assert sum(q["samples"] for q in quarters) == len(product)
    delta = quarters[-1]["current_footprint_median_bytes"] - quarters[0]["current_footprint_median_bytes"]
    return {"schema": "ironmule.prod10s_audit.v1", "assessment": "share_with_caveats",
        "run_id": run_id, "report_sha256": hashlib.sha256(raw).hexdigest(),
        "events_sha256": canonical_sha256(events), "verified_events": len(events),
        "completed_samples": len(samples), "soak_requests": len(soak),
        "soak_wall_seconds": report["soak"]["wall_seconds"],
        "case_counts": dict(Counter(s["case"] for s in soak)),
        "concurrent_requests": sum(s["concurrency"] == 4 for s in soak),
        "batch_counts": {str(k): v for k, v in Counter(len(rows) for rows in groups.values()).items()},
        "last_request_indices": [{"index": s["index"], "concurrency": s["concurrency"]} for s in soak[-10:]],
        "memory_window": {"definition": "first_to_last_completed_soak_event",
            "start_utc": datetime.fromtimestamp(start / 1e9, timezone.utc).isoformat(),
            "end_utc": datetime.fromtimestamp(end / 1e9, timezone.utc).isoformat(),
            "duration_seconds": (end - start) / 1e9},
        "current_footprint_quarters": quarters,
        "q4_minus_q1_current_footprint_bytes": delta,
        "q4_minus_q1_current_footprint_fraction": delta / quarters[0]["current_footprint_median_bytes"],
        "max_system_swap_delta_bytes": max(o["system_swap_delta_bytes"] for _, o in resource),
        "final_system_swap_delta_bytes": resource[-1][1]["system_swap_delta_bytes"],
        "bindings_equal": True, "normal_worker_exits": True,
        "caveats": ["one_device_one_model_three_fixed_cases", "not_a_speedup_or_universal_leak_freedom_claim",
                    "http_compares_text_and_usage_not_unexposed_token_ids",
                    "server_controller_and_client_share_process_parent_memory_not_separately_measured",
                    "record_sample_medians_not_time_weighted", "system_swap_not_attributable_to_model_alone"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.report, args.database)
    from product_load_screen import _exclusive_write
    _exclusive_write(args.output, result)
    print(json.dumps(result, indent=2))
