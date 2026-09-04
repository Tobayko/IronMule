"""Drive H1.0 across one model's answer lengths, one slice per process.

Each slice gets its own `BudgetGuard`, which is why the study fits at all: the
120 s GPU cap is not raised, the run is cut into pieces that fit under it. Per
length: one A/A slice (which also fixes that regime's pair count), then one
slice per draft width.

    python experiments/switch_point/sweep.py --execute --model 4b
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MEASURE = HERE / "measure.py"
PYTHON = Path(sys.executable)
LENGTHS = (32, 48, 64, 96, 128)
WIDTHS = (1, 2, 3)


def run_slice(argv: list[str]) -> int:
    print(f"\n=== {' '.join(argv)}", flush=True)
    started = time.time()
    completed = subprocess.run([str(PYTHON), str(MEASURE), "--execute", *argv], check=False)
    print(f"=== exit {completed.returncode} after {time.time() - started:.0f} s", flush=True)
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--model", choices=("4b", "1b"), default="4b")
    parser.add_argument("--lengths", default="")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"state": "not_released", "hint": "pass --execute"}))
        return 78

    lengths = tuple(int(t) for t in args.lengths.split(",") if t) or LENGTHS
    for tokens in lengths:
        aa_file = HERE / f"aa_{args.model}_{tokens}.json"
        if not aa_file.exists():
            code = run_slice(["--model", args.model, "--tokens", str(tokens), "--aa-only"])
            if code != 0:
                print(f"!!! A/A slice failed for {args.model}/{tokens}", flush=True)
                return code
        aa = json.loads(aa_file.read_text())
        spread, pairs = aa["aa_spread"], aa["derived_pairs"]
        print(f"--- {args.model}/{tokens}: A/A {spread:.4%}, {pairs} pairs", flush=True)
        for width in WIDTHS:
            out = HERE / f"switch_{args.model}_{tokens}_w{width}.json"
            if out.exists():
                continue
            code = run_slice([
                "--model", args.model, "--tokens", str(tokens), "--widths", str(width),
                "--mde", repr(spread), "--pairs", str(pairs),
            ])
            if code == 2:
                # Token identity broke — terminal for the whole study by
                # preregistration. A budget ceiling is not: that slice reports
                # `budget_exceeded` and the sweep moves on.
                print("!!! token identity broke, stopping the sweep", flush=True)
                return 2
            if code != 0:
                return code
    print("\n=== sweep complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
