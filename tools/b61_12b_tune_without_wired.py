#!/usr/bin/env python3
"""A 12B tuning candidate screened without the knob that cannot survive confirmation.

`B60` reproduced why the 12B tune ended: its screening kept `wired_fraction=0.6`, and
`Engine.__init__` takes that branch into `ironmule.hw.static_facts()`, which reads every
fact through a `sysctl` subprocess. The Q3f guard blocks `subprocess.Popen` in a
confirmation child, so the child raises `GuardViolation` and exits with status `1` -- on
any model, on any machine, every time. Three arms on a `1B` model separated it from
`fuse_projections`, and the same failure was then recorded on the `12B` itself.

The knob is therefore removed from the search for this run, and only from this run. The
library is not changed, its default is not changed, and no earlier result is relabelled:
the aborted tune stays aborted and keeps its own record.

The parent owns the tune process, so this time the return code, the decoded wait status
and both streams are captured whatever happens. A resource probe runs every two seconds
and will terminate *this file's own child* -- nothing else -- if free memory stays under
the floor, because a `12B` screening parent and a `12B` confirmation child are resident at
the same time and that is the one part of the aborted run that was never bounded.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import resource
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

_spec = importlib.util.spec_from_file_location(
    "b60", Path(__file__).resolve().parent / "b60_12b_child_diagnosis.py")
b60 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b60)

MODEL = "mlx-community/gemma-3-12b-it-4bit"
DROPPED_KNOB = "wired_fraction"
MAX_TOKENS = 32                 # the autotuner's own default; the profile is keyed on it
SAMPLE_INTERVAL_S = 2.0
FREE_FLOOR_PERCENT = 8.0        # a protective bound on this run, not a gate on any result
FREE_FLOOR_CONSECUTIVE = 5      # ten seconds under the floor before this file stops itself
TUNE_TIMEOUT_S = 3600.0

PREREGISTRATION = {
    "experiment": "B61_12b_tune_without_wired_fraction",
    "question": (
        "does a 12B tuning candidate screened without wired_fraction survive the ordinary "
        "paired confirmation and produce a stored profile for this fingerprint"
    ),
    "why_this_is_a_new_study": (
        "B60 proved the earlier abort's cause and that it is avoidable by removing one "
        "knob from the search. The aborted tune is not reinterpreted, not relabelled and "
        "keeps its own record; this is a separate run with its own evidence"
    ),
    "model": MODEL,
    "search_change": (
        f"the single entry {DROPPED_KNOB!r} is removed from ironmule.tune.SEARCH in this "
        "process only. Every other entry keeps its order and its values. ironmule/tune.py "
        "is not edited and the shipped default is unchanged"
    ),
    "protocol": (
        "ironmule.tune.tune() unchanged: coordinate descent from BASELINE with token "
        "identity required at every step, then the standard paired confirmation across "
        "CONFIRM_PROCESSES=6 fresh processes and CONFIRM_REPEATS=7 repeats, and a profile "
        "stored only when that confirmation is accepted"
    ),
    "acceptance": (
        "unchanged and owned by ironmule.tune: token, count and stop-reason identity, "
        "determinism, and a 95 per cent interval on total_ns whose upper bound is below "
        "1.0. No threshold here is new and none is moved"
    ),
    "captured": [
        "tune process return code", "decoded wait status and signal name",
        "unbuffered stdout", "unbuffered stderr", "every screening trial",
        "the confirmation record", "free memory, swap, swapin/swapout/pageout and load "
        "every two seconds",
    ],
    "protective_bound": (
        f"if system free memory stays below {FREE_FLOOR_PERCENT} per cent for "
        f"{FREE_FLOOR_CONSECUTIVE} consecutive probes, this file terminates the tune "
        "process it started. No other process is ever signalled, nothing is purged, and "
        "no system setting is changed. A run stopped this way is reported as stopped, "
        "never as a result"
    ),
    "gates_before_start": (
        f"no other model process, 1-minute load below {b60.LOAD_CEILING}, system free "
        f"memory at or above {b60.MIN_FREE_PERCENT} per cent"
    ),
    "kill": (
        "a rejected confirmation leaves this machine without a 12B profile, stack C stays "
        "unqualified on 12B, and no number from the screening is quoted as a gain"
    ),
}

CHILD = f"""
import importlib, json, sys
sys.path.insert(0, {str(PROJECT_ROOT)!r})
# `ironmule.tune` the attribute is the re-exported function, not the module, so the
# module is fetched by name; patching it is what `tune()` reads its search from.
tune = importlib.import_module("ironmule.tune")
kept = [row for row in tune.SEARCH if row[0] != {DROPPED_KNOB!r}]
assert len(kept) == len(tune.SEARCH) - 1, "the dropped knob was not in the search"
tune.SEARCH = kept
print("@@search " + json.dumps([[name, list(values)] for name, values in kept]), flush=True)
profile = tune.tune(model_id={MODEL!r}, max_tokens={MAX_TOKENS})
print("@@profile " + json.dumps({{
    "knobs": profile["knobs"],
    "gain": profile["gain"],
    "confirmation": profile["confirmation"],
    "confirmation_candidate_knobs": profile["confirmation_candidate_knobs"],
    "trials": profile["trials"],
    "baseline_ns": profile["baseline_ns"],
    "tuned_ns": profile["tuned_ns"],
    "token_count": profile["token_count"],
    "model_identity_sha256": profile["model_identity"]["identity_sha256"],
}}, sort_keys=True, allow_nan=False), flush=True)
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preregister", type=Path)
    args = parser.parse_args(argv)

    from b52_automatic_selection import write_once

    if args.preregister:
        write_once(args.preregister, json.dumps(
            {"preregistration": PREREGISTRATION, "source_binding": b60.source_binding(),
             "written_at": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True))
        print(f"preregistration written to {args.preregister}")
        return 0

    ok, reason, probe = b60.gates_ok()
    if not ok:
        raise SystemExit(f"refusing to start: {reason}")

    samples = [b60.sample("before_start")]
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", CHILD],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(PROJECT_ROOT),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    stop = threading.Event()
    stopped_by_floor = {"hit": False, "consecutive": 0}

    def sampler() -> None:
        while not stop.wait(SAMPLE_INTERVAL_S):
            row = b60.sample("during", process.pid)
            samples.append(row)
            free = row.get("memory_free_percent")
            if free is not None and free < FREE_FLOOR_PERCENT:
                stopped_by_floor["consecutive"] += 1
                if stopped_by_floor["consecutive"] >= FREE_FLOOR_CONSECUTIVE:
                    stopped_by_floor["hit"] = True
                    process.terminate()      # this file's own child, and only it
                    return
            else:
                stopped_by_floor["consecutive"] = 0

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=TUNE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()
        stdout, stderr = process.communicate()
    finally:
        stop.set()
        thread.join(timeout=5)

    samples.append(b60.sample("after_end"))
    status = b60._decode_status(process.returncode)

    def marker(prefix: str):
        line = next((l for l in stdout.splitlines() if l.startswith(prefix)), None)
        return json.loads(line[len(prefix):]) if line else None

    profile = marker("@@profile ")
    confirmation = (profile or {}).get("confirmation") or {}
    record = {
        "experiment": PREREGISTRATION["experiment"],
        "preregistration": PREREGISTRATION,
        "source_binding": b60.source_binding(),
        "environment": {"platform": platform.platform(), "model": MODEL,
                        "python": sys.version.split()[0]},
        "started_at": started_at,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "duration_s": time.monotonic() - started,
        "process_end": status,
        "timed_out": timed_out,
        "stopped_by_free_memory_floor": stopped_by_floor["hit"],
        "search_used": marker("@@search "),
        "profile": profile,
        "stdout": stdout,
        "stderr": stderr,
        "samples": samples,
        "child_peak_rss_bytes_rusage": int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss),
        "verdict": ("STOPPED_BY_PROTECTIVE_BOUND" if stopped_by_floor["hit"]
                    else "TUNE_DID_NOT_COMPLETE" if status["returncode"] != 0
                    else "PROFILE_CONFIRMED" if confirmation.get("accepted")
                    else "CONFIRMATION_REJECTED"),
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True, default=str))
    print(json.dumps({"verdict": record["verdict"], "process_end": status,
                      "duration_s": round(record["duration_s"], 1),
                      "knobs": (profile or {}).get("knobs"),
                      "gain": (profile or {}).get("gain"),
                      "confirmation": {k: confirmation.get(k) for k in
                                       ("accepted", "rejection_reason", "token_identity",
                                        "deterministic")} if profile else None},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
