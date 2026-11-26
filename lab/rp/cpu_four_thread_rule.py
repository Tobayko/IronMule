"""Compare a static four-thread rule on the same qualified CPU workload, seeds and serving evaluator."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ironmule.pilot import Pilot
from training import serving_trial as st


def measure(args):
    active = False
    decisions = 0
    actual = set()
    wave = st.run_wave
    decide = Pilot.decide
    decode = st.ge.GgmlEngine._decode

    def run_wave(*a, **kw):
        nonlocal active
        active = True
        try:
            return wave(*a, **kw)
        finally:
            active = False

    def fixed(pilot, *a, **kw):
        nonlocal decisions
        result = decide(pilot, *a, **kw)
        if active:
            if pilot.gpu or pilot.cores != 4:
                raise ValueError("static challenger needs the same four CPU cores")
            decisions += 1
            result["threads"] = 4
        return result

    def check(engine, *a, **kw):
        if active:
            actual.add(engine.threads)
        return decode(engine, *a, **kw)

    st.run_wave, Pilot.decide, st.ge.GgmlEngine._decode = run_wave, fixed, check
    try:
        report = st.measure(args)
    finally:
        st.run_wave, Pilot.decide, st.ge.GgmlEngine._decode = wave, decide, decode
    if not decisions or actual != {4} or report["correct"] != report["completed"] or report["scratch_measured_decisions"]:
        raise ValueError("incomplete four-thread comparator or incorrect output")
    report["simple_thread_rule"] = {"name": "always_four_threads", "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                    "overrides": decisions, "actual_threads": sorted(actual)}
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gguf", type=Path, required=True)
    p.add_argument("--thread-candidate", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--require-provider", choices=["cpu"], required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--threads", type=int, default=2, choices=[2])
    p.add_argument("--clients", type=int, default=4, choices=[4])
    p.add_argument("--waves", type=int, default=4, choices=[4])
    p.add_argument("--tokens", type=int, default=3, choices=[3])
    p.set_defaults(profile_dir=None)
    args = p.parse_args()
    report = measure(args)
    args.out.write_text(json.dumps(report, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
