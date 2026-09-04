#!/usr/bin/env python3
"""Drive R2 campaign points inside a time window, resumably.

One point is one subprocess (``tools/run_r2_point.py``). That costs a model load
per point -- about two seconds against a hundred and seventy-nine -- and buys a
fresh ``BudgetGuard`` each time, which is what makes a long campaign possible at
all: the guard's limits are per process (``gpu_work_limit_s = 120`` against
``26.5 s`` of GPU work per point, ``wall_limit_s = 1200``), so roughly four
points exhaust one guard. A single long-lived process cannot run the campaign;
forty short ones can.

**The cursor is the corpus.** A point counts as done when its ``outcome:`` record
exists in Optimization Memory. There is no separate state file to fall out of
sync, and an interrupted night resumes by asking the same question again. The
draw is deterministic per index (``CampaignPlan.seed_for``), so a point
re-attempted later is the same point.

Points whose drawn action has no engine knob are skipped without a record, per
``docs/R2_VORREGISTRIERUNG.md`` Amendment A2. They cost no GPU time.

Run:  python tools/run_r2_campaign.py --execute --until 18:00
"""

from __future__ import annotations

import argparse
import datetime
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

POINT_RUNNER = PROJECT_ROOT / "tools" / "run_r2_point.py"
#: Measured on this machine, two pilot points: 179.3 s and 178.5 s. Used only to
#: decide whether the next point still fits in the window, never as evidence.
POINT_SECONDS = 185.0


def _deadline(until: str | None, minutes: float | None) -> float:
    """Monotonic deadline from a wall-clock time or a duration."""

    if minutes is not None:
        return time.monotonic() + minutes * 60.0
    now = datetime.datetime.now()
    hour, _, minute = (until or "").partition(":")
    target = now.replace(hour=int(hour), minute=int(minute), second=0, microsecond=0)
    if target <= now:  # a window that already closed today means tomorrow
        target += datetime.timedelta(days=1)
    return time.monotonic() + (target - now).total_seconds()


def _completed(memory_path: str, campaign_id: str) -> set[int]:
    """Indices whose label is already in the corpus. This is the whole cursor."""

    import sqlite3

    done: set[int] = set()
    prefix = f"outcome:{campaign_id}."
    try:
        connection = sqlite3.connect(f"file:{memory_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return done
    try:
        for (record_id,) in connection.execute(
            "SELECT record_id FROM optimization_records WHERE record_id LIKE ?",
            (prefix + "%",),
        ):
            try:
                done.add(int(record_id[len(prefix) :]))
            except ValueError:
                continue
    finally:
        connection.close()
    return done


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--until", default=None, help="Stop at this wall-clock time, HH:MM")
    parser.add_argument("--minutes", type=float, default=None, help="Stop after this many minutes")
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--max-points", type=int, default=None, help="Stop after this many measured points")
    parser.add_argument(
        "--memory", default=str(PROJECT_ROOT / ".friday-data" / "optimizer-v2.sqlite3")
    )
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"state": "not_released", "hint": "pass --execute"}))
        return 78
    if args.until is None and args.minutes is None:
        print(json.dumps({"state": "usage", "hint": "pass --until HH:MM or --minutes N"}))
        return 64

    import run_r2_point as point
    from friday_optimizer.campaign import CampaignPlan
    from friday_optimizer.decisions import SelectionPolicy

    deadline = _deadline(args.until, args.minutes)
    fingerprint = point._fingerprint()
    plan = CampaignPlan(
        campaign_id=point.CAMPAIGN_ID,
        policy=SelectionPolicy("r2-logging-v1", rule="epsilon_greedy", epsilon=point.EPSILON),
        seed_base=point.SEED_BASE,
        points=point.POINTS,
        hints=(point.HINT,),
    )
    drawn = [d.chosen for d in plan.decisions(fingerprint)]
    done = _completed(args.memory, point.CAMPAIGN_ID)
    measurable = [
        index
        for index, action in enumerate(drawn)
        if action in point.CANDIDATE_KNOBS and index not in done
    ]
    skipped_total = sum(1 for action in drawn if action not in point.CANDIDATE_KNOBS)

    print(
        json.dumps(
            {
                "state": "starting",
                "campaign_hash": plan.campaign_hash[:16],
                "already_done": len(done),
                "pending_measurable": len(measurable),
                "unmeasurable_by_design": skipped_total,
                "window_seconds": round(deadline - time.monotonic()),
                "fits_in_window": int((deadline - time.monotonic()) // POINT_SECONDS),
            }
        ),
        flush=True,
    )

    stopping = {"now": False}

    def _stop(_signum, _frame):
        stopping["now"] = True
        print(json.dumps({"state": "stop_requested", "note": "finishing the point in flight"}), flush=True)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    measured = 0
    failed = 0
    started = time.monotonic()
    for index in measurable:
        if stopping["now"]:
            break
        remaining = deadline - time.monotonic()
        if remaining < POINT_SECONDS:
            print(
                json.dumps({"state": "window_closed", "remaining_seconds": round(remaining)}),
                flush=True,
            )
            break
        if args.max_points is not None and measured >= args.max_points:
            break
        completed = subprocess.run(
            [
                sys.executable,
                str(POINT_RUNNER),
                "--execute",
                "--write",
                "--index",
                str(index),
                "--pairs",
                str(args.pairs),
                "--memory",
                args.memory,
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        line = (completed.stdout or "").strip().splitlines()
        summary = {}
        for raw in reversed(line):
            try:
                summary = json.loads(raw)
                break
            except json.JSONDecodeError:
                continue
        if completed.returncode != 0 or summary.get("state") != "measured":
            failed += 1
            print(
                json.dumps(
                    {
                        "state": "point_failed",
                        "index": index,
                        "action": drawn[index],
                        "returncode": completed.returncode,
                        "detail": (summary or {"stderr": (completed.stderr or "")[-400:]}),
                    }
                ),
                flush=True,
            )
            # A failing point is information, not a reason to abandon the night;
            # but three in a row means something structural, not a bad draw.
            if failed >= 3 and measured == 0:
                print(json.dumps({"state": "aborting", "reason": "three failures, nothing measured"}), flush=True)
                break
            continue
        measured += 1
        print(
            json.dumps(
                {
                    "state": "point",
                    "index": index,
                    "action": drawn[index],
                    "ratio_median": summary.get("reward_ratio_median"),
                    "gain_percent": summary.get("gain_percent"),
                    "breaks": summary.get("token_identity_breaks"),
                    "seconds": summary.get("wall_seconds"),
                    "measured_so_far": measured,
                }
            ),
            flush=True,
        )

    done_after = _completed(args.memory, point.CAMPAIGN_ID)
    print(
        json.dumps(
            {
                "state": "finished",
                "measured_this_run": measured,
                "failed_this_run": failed,
                "labelled_total": len(done_after),
                "measurable_total": point.POINTS - skipped_total,
                "remaining_measurable": point.POINTS - skipped_total - len(done_after),
                "elapsed_minutes": round((time.monotonic() - started) / 60, 1),
                "formal_claim": False,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
