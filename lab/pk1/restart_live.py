"""Bounded passive product listener. The independent client is a separately invoked process."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import secrets
import time

from ironmule_product import compare
from lab.pk1 import native_http as http, restart_http as startup
from training import scoped_release as release
from training.code_trial import read, save
from training.trial_release import require


def passive(args, receipt, evidence, withdrawal):
    decision = read(evidence / "decision.json")
    require(decision.get("passed") is True and startup.assess(evidence, receipt) == decision,
            "an ordinary listener needs passing, authenticated product measurements")
    gates = {arm: read(evidence / f"gate-{arm}.json") for arm in ("reference", "candidate")}
    window = min(args.deadline, time.monotonic() + args.lifetime)
    records = []
    completed = 0
    while time.monotonic() < window:
        nonce = secrets.token_hex(16)
        since, misses = time.time(), 0
        def capture(row):
            record = {"nonce": nonce, "selection": row, "source_sha256": receipt["qualification"][row["arm"] + "_sha256"]}
            save(evidence / "live-selection.json", record)
            records.append(record)
        args.on_startup_selection = capture
        with startup.cold_endpoint(args, receipt, gates, withdrawn=withdrawal.exists, port=args.port) as (address, outer, selection):
            ready = {"nonce": nonce, "model_id": address["model_id"], "port": outer.port, "pid": os.getpid(),
                     "receipt_sha256": args.receipt_sha256, "geometry_sha256": args.geometry_sha256,
                     "ready_ns": time.monotonic_ns(), "client_code_sha256": release.checksum(__file__)}
            save(evidence / "live-ready.json", ready)
            while time.monotonic() < window:
                service = outer.backend
                if outer.failure or (service and not service.ready()):
                    if withdrawal.exists():
                        raise RuntimeError("ordinary reference worker unavailable")
                    save(withdrawal, {"withdrawn": True, "reason": "ordinary product worker unavailable"})
                    break
                if selection.get("arm") == "candidate":
                    if withdrawal.exists() or release.selected(args.state, args.receipt_sha256) != "candidate":
                        break
                    for row in service.live_samples():
                        if row["at"] > since:
                            since = row["at"]
                            miss = http.missed(row, decision["reference_bounds"])
                            if miss is not None:
                                misses = misses + 1 if miss else 0
                    if misses >= 3:
                        save(withdrawal, {"withdrawn": True, "reason": "three matched product timing regressions"})
                        break
                time.sleep(min(.2, max(0., window - time.monotonic())))
            # The normal listener never sends any requests, even if no customers arrive.
            if outer.backend:
                completed += outer.backend.health()["completed_requests"]
    return {"arm": records[-1]["selection"]["arm"] if records else "idle", "selections": records,
            "requests_served": completed, "self_generated_requests": 0, "receipt_sha256": args.receipt_sha256}


def orders():
    """Independent held-out inventory updates. Nothing in the listener depends on these strings or expected answers."""
    tasks = []
    for i in range(6):
        tasks.extend([(f"Stock is {21 + i}. Add 3 units. How many now? Reply with digits only.", str(24 + i)),
                      (f"The recorded unit count for order XL-{91+i} is {174+i}; what is the recorded unit count? Reply with digits only.", str(174+i)),
                      (f"Our dispatch sheet lists package {48+i} at loading bay {81+i}. What is its bay number? Digits only.", str(81+i))])
    return tasks


def client(ready_path, expected_receipt, timeout):
    ready = read(ready_path)
    require(ready["receipt_sha256"] == expected_receipt and ready["client_code_sha256"] == release.checksum(__file__)
            and 0 < ready["port"] <= 65535 and os.getpid() != ready["pid"], "client and passive listener differ")
    address = {"base_url": "http://127.0.0.1:" + str(ready["port"]), "model_id": ready["model_id"]}
    started = time.monotonic_ns()
    tasks = orders()
    rows, wall = compare.batch("ironmule", address, [q for q, _ in tasks], len(tasks), 32,
                               time.monotonic() + timeout, request_id_prefix="verify-" + ready["nonce"], timeline=True)
    return {"nonce": ready["nonce"], "pid": os.getpid(), "listener_pid": ready["pid"], "started_ns": started,
            "finished_ns": time.monotonic_ns(), "receipt_sha256": expected_receipt, "work_sha256": http.digest(tasks),
            "server_ids": [r["server_id"] for r in rows], "wall_s": wall, "first_token_ms": [r["ttft_ms"] for r in rows], "responses": http.witness(rows)}


def verify(listener, witness):
    tasks = orders()
    first, = [r for r in listener["selections"] if r["nonce"] == witness["nonce"]]
    seen = first["selection"]
    ids = set(witness["server_ids"])
    responses = witness["responses"]
    require(witness["pid"] != witness["listener_pid"] and listener["self_generated_requests"] == 0
            and listener["receipt_sha256"] == witness["receipt_sha256"] and listener["requests_served"] >= len(tasks)
            and witness["work_sha256"] == http.digest(tasks) and len(ids) == len(tasks) and seen["first_request_id"] in ids
            and witness["started_ns"] <= seen["request_received_monotonic_ns"] < seen["worker_ready_monotonic_ns"] < witness["finished_ns"]
            and len(responses) == len(witness["first_token_ms"]) == len(tasks)
            and all(isinstance(row, list) and len(row) == 4 and row[1] in ("stop", "length") and row[2] > 0 and 0 < row[3] <= 32
                    for row in responses) and all(math.isfinite(x) and x > 0 for x in [witness["wall_s"], *witness["first_token_ms"]]),
            "external requests do not match the actual listener")
    graded = sum(row[0].strip().strip(".").strip() == answer for row, (_, answer) in zip(responses, tasks, strict=True))
    completion = {"arm": seen["arm"], "source_sha256": first["source_sha256"], "work_sha256": http.digest(responses),
                  "graded_correct": graded, "cold_to_18_correct_s": witness["wall_s"] if graded == 18 else None,
                  "correct_completions_per_s_including_startup": graded / witness["wall_s"], "first_token_ms": witness["first_token_ms"],
                  "startup_decision": seen["policy"], "prior_decision": seen["prior"], "cold_admission": seen}
    return {"arm": listener["arm"], "requests_served": listener["requests_served"], "completions": [completion],
            "listener_sha256": http.digest(listener), "external_client_sha256": http.digest(witness)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Send external inventory questions over HTTP to an already listening product")
    p.add_argument("--ready", type=Path, required=True)
    p.add_argument("--receipt-sha256", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--timeout", type=int, default=180)
    a = p.parse_args()
    require(0 < a.timeout <= 3600, "invalid external client timeout")
    save(a.out, client(a.ready, a.receipt_sha256, a.timeout))
