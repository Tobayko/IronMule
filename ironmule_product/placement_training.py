"""Train a per-worker latency model and compare its frozen routing decisions on later hardware trials."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

from .placement_native import NativeModel
from .state import _atomic_write

ROUNDS = 5
FIT_ROUNDS = 3
CASES = 8
SCHEMA = "ironmule.placement.v1"


def signature(binding):
    return hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def workload(sizes, devices):
    """A small factorial design covers short/long prompts, two output budgets and asymmetric queues."""
    chars, limits = [96, 1024], [8, 64]
    if sizes:
        for index, (bounds, minimum, maximum) in enumerate(((chars, 32, 4096), (limits, 4, 128))):
            seen = sorted(row[index] for row in sizes)
            for i, fraction in enumerate((0.2, 0.8)):
                val = seen[int((len(seen) - 1) * fraction)]
                bounds[i] = max(minimum, min(maximum, val))
            if bounds[0] == bounds[1]:
                bounds[0] = max(minimum, bounds[1] // 2)
    cases = []
    for i, size in enumerate(chars):
        for j, limit in enumerate(limits):
            for loaded in (False, True):
                queues = [0] * devices
                if loaded:
                    queues[(2 * i + j) % devices] = 1
                cases.append([size, limit, queues])
    return cases


def valid_trial(row, spec):
    if not isinstance(row, dict) or row.get("case") != spec:
        return False
    readings = row.get("arms")
    if not isinstance(readings, list) or len(readings) != len(spec[2]):
        return False
    for item in readings:
        if not isinstance(item, list) or len(item) != 3:
            return False
        ms, first, tokens = item
        if not all(type(x) in (int, float) and math.isfinite(x) for x in item):
            return False
        if not (0 < first <= ms <= 86400000 and type(tokens) is int and 1 <= tokens <= spec[1]):
            return False
    return True


def check_trace(trace):
    binding, plan, rows = trace["binding"], trace["plan"], trace["trials"]
    if type(trace.get("started")) not in (int, float) or not math.isfinite(trace["started"]):
        raise ValueError("invalid campaign timestamp")
    devices = binding["devices"]
    if (trace.get("schema") != SCHEMA or trace.get("identity") != signature(binding)
            or not isinstance(devices, list) or not devices or len(devices) != len(set(devices))
            or not isinstance(plan, list) or len(plan) != CASES
            or not isinstance(rows, list) or len(rows) > CASES * ROUNDS):
        raise ValueError("invalid placement trace identity or dimensions")
    for case in plan:
        if not isinstance(case, list) or len(case) != 3:
            raise ValueError("invalid placement workload")
        chars, limit, queues = case
        if (type(chars) is not int or not 1 <= chars <= 4096 or type(limit) is not int or not 1 <= limit <= 128
                or not isinstance(queues, list) or len(queues) != len(devices)
                or any(type(q) is not int or q not in (0, 1) for q in queues)):
            raise ValueError("unsupported placement workload")
    for i, row in enumerate(rows):
        if (not valid_trial(row, plan[i % CASES]) or ("wall_s" in row and
                (type(row["wall_s"]) not in (int, float) or not math.isfinite(row["wall_s"]) or row["wall_s"] < 0))):
            raise ValueError("invalid hardware timing or incomplete paired trial")


def covered(plan, chars, limit, queues):
    matches = [case for case in plan if case[2] == queues]
    return bool(matches and min(x[0] for x in matches) <= chars <= max(x[0] for x in matches)
                and min(x[1] for x in matches) <= limit <= max(x[1] for x in matches))


def decision(native, weights, chars, limit, queues, future=None, weight=0.0):
    return native.choose(weights, chars, limit, queues, future or (chars, limit), weight)


def qualify(native, trace):
    check_trace(trace)
    rows = trace["trials"]
    if len(rows) != CASES * ROUNDS:
        raise ValueError(f"qualification needs {CASES * ROUNDS} completed paired trials, got {len(rows)}")
    if trace["binding"]["trainer"] != native.version:
        raise ValueError("this trace belongs to a different native trainer")
    fitted = []
    for i in range(len(trace["binding"]["devices"])):
        inputs = []
        for row in rows[:CASES * FIT_ROUNDS]:
            size, limit, queues = row["case"]
            inputs.append((size, limit, queues[i], row["arms"][i][0]))
        fitted.append(native.fit(inputs))
    # Each alternative imagines a second, long arrival. Pick its weight using completed training rounds only.
    future = [max(c[i] for c in trace["plan"]) for i in (0, 1)]
    def training_cost(weight):
        total = 0.0
        for row in rows[:CASES * FIT_ROUNDS]:
            choice = decision(native, fitted, *row["case"], future, weight)
            total += row["arms"][choice][0]
        return total
    weight = min((0.0, 0.25, 0.5), key=training_cost)
    rounds = []
    for r in range(FIT_ROUNDS, ROUNDS):
        totals = [0.0] * 4
        comparable = True
        changes = 0
        for row in rows[r * CASES:(r + 1) * CASES]:
            size, limit, queues = row["case"]
            chosen = decision(native, fitted, size, limit, queues, future, weight)
            reference = min(range(len(queues)), key=queues.__getitem__)
            base, actual = row["arms"][reference], row["arms"][chosen]
            comparable &= base[2] == actual[2]
            changes += chosen != reference
            for index, value in enumerate((actual[0], base[0], actual[1], base[1])):
                totals[index] += value
        rounds.append({"completion_ratio": totals[0] / totals[1], "first_token_ratio": totals[2] / totals[3],
                       "equal_output_counts": comparable, "changed_routes": changes})
    passed = all(r["completion_ratio"] <= 0.95 and r["first_token_ratio"] <= 1.0 and r["equal_output_counts"]
                 for r in rounds)
    return {"qualified": passed, "weights": fitted, "next_shape": future, "future_weight": weight,
            "holdout": rounds, "identity": trace["identity"]}


def wait_for_calibration(service, minutes):
    """Both calibration and serving run the same hardware gates and interruptible worker probes."""
    start = time.monotonic()
    previous = None
    while time.monotonic() - start < 60 * minutes:
        if service.hardware_placement is None:
            print(json.dumps({"status": "reference", "error": service.hardware_placement_error or
                              "the placement controller did not start"}), flush=True)
            return 1
        status = service.hardware_placement.status()
        progress = (status["paired_trials"], status["phase"], status["error"])
        if progress != previous:
            print(json.dumps({"placement": status, "calibration_seconds": round(time.monotonic() - start, 2)}),
                  flush=True)
            previous = progress
        if status["phase"] == "reference_after_failed_measurements":
            return 1
        if status["paired_trials"] == status["target_trials"]:
            return 0
        time.sleep(1)
    print(json.dumps({"status": "calibration budget expired", "resumable": True}), flush=True)
    return 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("calibrate", "train", "benchmark", "estimate"))
    parser.add_argument("--gguf", type=Path, help="local model for calibrate")
    parser.add_argument("--state-dir", type=Path, help="resumable product state for calibrate")
    parser.add_argument("--minutes", type=float, default=240, help="maximum idle calibration time after startup")
    parser.add_argument("--trace", type=Path, help="hardware trace shown by /health under placement.trace")
    parser.add_argument("--output", type=Path, help="optional offline comparison report")
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/ironmule/placement-native")
    parser.add_argument("--devices", type=int, default=2, help="number of candidate workers for a planning estimate")
    parser.add_argument("--seconds-per-arm", type=float, help="mean time for one worker in one case, including load")
    parser.add_argument("--startup-minutes", type=float, default=0.0)
    args = parser.parse_args(argv)
    if args.command == "calibrate":
        from .cli import serve
        from .state import ProductStore
        if not args.gguf or not args.gguf.is_file() or not math.isfinite(args.minutes) or args.minutes <= 0:
            parser.error("calibrate needs an existing --gguf and positive --minutes")
        store = ProductStore(args.state_dir)
        if not (store.root / "settings.json").exists():
            store.setup("server")
        return serve(["--gguf", str(args.gguf), "--placement", "auto", "--state-dir", str(store.root)],
                     calibration_minutes=args.minutes)
    if args.command == "estimate":
        if (args.devices < 1 or args.seconds_per_arm is None or not math.isfinite(args.seconds_per_arm)
                or args.seconds_per_arm <= 0 or not math.isfinite(args.startup_minutes) or args.startup_minutes < 0):
            parser.error("estimate needs positive devices, positive --seconds-per-arm and nonnegative startup time")
        runs = CASES * ROUNDS * args.devices
        from .placement import IDLE_SECONDS
        idle_minutes = (runs * args.seconds_per_arm + CASES * ROUNDS + IDLE_SECONDS) / 60
        print(json.dumps({"worker_measurements": runs, "estimated_idle_minutes": idle_minutes,
                          "estimated_total_minutes": args.startup_minutes + idle_minutes,
                          "excludes": "waiting for external load, interrupted trials, downloads and builds"}))
        return
    native = NativeModel(args.cache)
    if args.command == "benchmark":
        # Timing the real native fit on explicit synthetic inputs does not qualify a policy.
        inputs = [(32 + i % 2048, 1 + i % 128, i % 3, 2 + i % 937) for i in range(4096)]
        timings = []
        for _ in range(21):
            started = time.perf_counter()
            native.fit(inputs)
            timings.append((time.perf_counter() - started) * 1000)
        result = {"input": "synthetic, timing only", "observations": len(inputs),
                  "fits": len(timings), "median_fit_ms": statistics.median(timings)}
    else:
        if args.trace is None:
            parser.error("train needs --trace from a completed calibration")
        trace = json.loads(args.trace.read_text())
        started = time.perf_counter()
        result = qualify(native, trace)
        result["fit_and_holdout_ms"] = (time.perf_counter() - started) * 1000
    if args.output:
        if args.trace and args.output.resolve() == args.trace.resolve():
            parser.error("write the report to a different file from the source trace")
        _atomic_write(args.output, result)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    raise SystemExit(main())
