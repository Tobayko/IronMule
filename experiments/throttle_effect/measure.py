"""Does stepping back actually hand the Mac back? The kill criterion.

A considerate mode that feels good and gives nothing back is decoration, and
this repository's central finding is that the noise floor swallows most real
effects. So the claim is tested the way every other claim here is: paired,
alternating AB/BA, median over repeats, with the spread reported next to it.

Design. A fixed, bandwidth-heavy foreign job runs while the model decodes
continuously. Only the *foreign* job's wall time is measured. Two arms per run:

* ``full``     -- the pause is zero, today's behaviour.
* ``--level``  -- ``gentle`` or ``minimal``, the step-back the mode applies when
  the Mac is busy. Each level is a separate run and writes its own report.

The detection logic is not under test here (that is offline in
``tests/test_throttle.py``); the level is forced, so this measures exactly one
thing: what standing aside is worth to whoever else is using the machine.

Measured 2026-09-04, six pairs per level, median with spread:
``gentle`` buys the foreign job ``4.0 %`` while the model keeps ``80.7 %`` of its
tokens; ``minimal`` buys ``7.4 %`` and keeps ``39.1 %``.

Run: ``python experiments/throttle_effect/measure.py --execute --level gentle``
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

STUDY_ID = "throttle-effect-20260904-01"
MODEL_ID = "mlx-community/gemma-3-4b-it-4bit"
PROMPT = "Explain why false sharing slows down multithreaded code, with an example."
#: Foreign job: a bandwidth-bound sum over ~256 MB, the kind of work that
#: competes with the GPU for unified memory rather than for a core. Sized to
#: about two seconds so it spans many decode bundles, and so the whole arm stays
#: under the 6 s continuous-GPU limit the budget policy enforces.
ARRAY_MB = 256
FOREIGN_PASSES = 180


def _foreign_job(array) -> float:
    """Fixed work, timed. Returns wall seconds."""

    started = time.perf_counter()
    total = 0.0
    for _ in range(FOREIGN_PASSES):
        total += float(array.sum())
    assert total != 0.0  # keep the sums from being optimised away
    return time.perf_counter() - started


def _run_arm(backend, throttle, level_name: str, array) -> dict:
    """Drive the model continuously, time the foreign job against it."""

    from friday_serve.throttle import FULL, GENTLE, MINIMAL

    forced = {"full": FULL, "gentle": GENTLE, "minimal": MINIMAL}[level_name]
    throttle._level = forced.because("forced")
    stop = threading.Event()
    decoded = {"tokens": 0}

    def decode_loop() -> None:
        token_ids = backend.encode(PROMPT)
        while not stop.is_set():
            for event in backend.stream_generate(token_ids, 64, {}):
                if stop.is_set():
                    break
                if event.get("type") == "token":
                    decoded["tokens"] += len(event.get("tokens", []))

    worker = threading.Thread(target=decode_loop, daemon=True)
    started = time.perf_counter()
    worker.start()
    time.sleep(0.5)  # let the model reach steady-state decoding first
    foreign_seconds = _foreign_job(array)
    stop.set()
    worker.join(timeout=30.0)
    gpu_seconds = time.perf_counter() - started
    return {
        "level": level_name,
        "foreign_seconds": foreign_seconds,
        "model_tokens": decoded["tokens"],
        "gpu_seconds": gpu_seconds,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--level", choices=("gentle", "minimal"), default="minimal")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"state": "not_released", "hint": "pass --execute"}))
        return 78

    import numpy as np
    from _bench import require_ac_power

    from friday_evidence.budget import BudgetGuard
    from friday_serve.ironmule_backend import IronMuleBackend
    from friday_serve.throttle import Throttle, set_global_throttle

    require_ac_power()
    guard = BudgetGuard()
    array = np.ones(ARRAY_MB * 1024 * 1024 // 4, dtype=np.float32)
    backend = IronMuleBackend.load(MODEL_ID)
    throttle = set_global_throttle(Throttle(enabled=True, max_width=1))

    # Warm up once: the first generation pays for Metal JIT and would land
    # entirely in whichever arm happens to go first.
    _run_arm(backend, throttle, "full", array)

    # Duty-cycle accounting exactly as friday_calibrate.runner does it: charge
    # the GPU seconds, carry the debt, and take the enforced break when it comes
    # due. The 60 s candidate cooldown is not used -- these arms are two settings
    # of the same path, not two promotion candidates.
    debt = 0.0

    def charge(seconds: float) -> None:
        nonlocal debt
        guard.record_gpu(seconds)
        # Repay against 20 %, not the 25 % ceiling, and round the break up. Two
        # earlier runs of this script tripped the rolling window: charging the
        # whole arm (including its warm-up and drain) as GPU time and then
        # repaying exactly at the limit leaves no headroom, and the shortfall
        # accumulates until the guard refuses a few pairs in.
        debt += seconds * (1 - 0.20) / 0.20
        while debt > 0.0:
            guard.required_break()
            debt -= 4.0

    pairs = []
    for index in range(args.pairs):
        order = ("full", args.level) if index % 2 == 0 else (args.level, "full")
        arms = {}
        for level_name in order:
            arms[level_name] = _run_arm(backend, throttle, level_name, array)
            charge(arms[level_name]["gpu_seconds"])
        pairs.append(
            {
                "pair": index,
                "order": list(order),
                "full": arms["full"],
                "stepped_back": arms[args.level],
                # Below 1.0 means the foreign job finished faster while the
                # model was stepping back -- the effect this exists to have.
                "ratio": arms[args.level]["foreign_seconds"] / arms["full"]["foreign_seconds"],
                "token_ratio": (
                    arms[args.level]["model_tokens"] / arms["full"]["model_tokens"]
                    if arms["full"]["model_tokens"]
                    else None
                ),
            }
        )
        print(
            json.dumps(
                {
                    "pair": index,
                    "order": list(order),
                    "ratio": round(pairs[-1]["ratio"], 4),
                    "token_ratio": round(pairs[-1]["token_ratio"] or 0.0, 4),
                    "foreign_seconds": [
                        round(arms["full"]["foreign_seconds"], 3),
                        round(arms[args.level]["foreign_seconds"], 3),
                    ],
                    "arm_seconds": round(arms["full"]["gpu_seconds"], 2),
                }
            )
        )

    ratios = [item["ratio"] for item in pairs]
    median = statistics.median(ratios)
    report = {
        "study_id": STUDY_ID,
        "model_id": MODEL_ID,
        "level": args.level,
        "pairs": args.pairs,
        "array_mb": ARRAY_MB,
        "foreign_time_ratio_median": median,
        "foreign_time_ratio_min": min(ratios),
        "foreign_time_ratio_max": max(ratios),
        "foreign_time_ratio_stdev": statistics.stdev(ratios) if len(ratios) > 1 else 0.0,
        "model_token_ratio_median": statistics.median(
            [item["token_ratio"] for item in pairs if item["token_ratio"] is not None]
        ),
        "detail": pairs,
        "budget": guard.summary(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "formal_claim": False,
    }
    out = Path(__file__).resolve().parent / f"report_{args.level}.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "state": "measured",
                "foreign_time_ratio_median": round(median, 4),
                "spread": [round(min(ratios), 4), round(max(ratios), 4)],
                "model_token_ratio_median": round(report["model_token_ratio_median"], 4),
                "report": str(out),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
