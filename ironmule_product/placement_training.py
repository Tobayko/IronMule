"""Train a per-worker latency model and compare its frozen routing decisions on later hardware trials."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import time
from collections import Counter
from pathlib import Path

from .placement_native import NativeModel
from .state import _atomic_write

ROUNDS = 5
FIT_ROUNDS = 3
CASES = 8
LEGACY_SCHEMA = "ironmule.placement.v1"
RESPONSE_SCHEMA = "ironmule.placement.v2"
SCHEMA = "ironmule.placement.v3"
MAX_BACKGROUND = 4


def signature(binding):
    return hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def workload(sizes, devices, backlogs=()):
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
    frequencies = Counter(q for q in backlogs if 0 < sum(q) <= MAX_BACKGROUND)
    patterns = sorted((q for q, n in frequencies.items() if n >= 4), key=lambda q: (-frequencies[q], q))[:4]
    cases = []
    for i, size in enumerate(chars):
        for j, limit in enumerate(limits):
            for loaded in (False, True):
                queues = [0] * devices
                if loaded:
                    slot = 2 * i + j
                    if patterns:
                        queues = list(patterns[slot % len(patterns)])
                    else:
                        queues[slot % devices] = 1
                cases.append([size, limit, queues])
    return cases


def valid_trial(row, spec, schema=SCHEMA):
    if not isinstance(row, dict) or row.get("case") != spec:
        return False
    readings = row.get("arms")
    if not isinstance(readings, list) or len(readings) != len(spec[2]):
        return False
    if schema != LEGACY_SCHEMA:
        outputs = row.get("outputs")
        if (not isinstance(outputs, list) or len(outputs) != len(readings)
                or any(not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) for digest in outputs)):
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
    if (trace.get("schema") not in (SCHEMA, RESPONSE_SCHEMA, LEGACY_SCHEMA)
            or binding.get("protocol", trace["schema"]) != trace["schema"] or trace.get("identity") != signature(binding)
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
                or any(type(q) is not int or q < 0 for q in queues)
                or (sum(queues) > MAX_BACKGROUND if trace["schema"] == SCHEMA else max(queues) > 1)):
            raise ValueError("unsupported placement workload")
    for i, row in enumerate(rows):
        if (not valid_trial(row, plan[i % CASES], trace["schema"]) or ("wall_s" in row and
                (type(row["wall_s"]) not in (int, float) or not math.isfinite(row["wall_s"]) or row["wall_s"] < 0))):
            raise ValueError("invalid hardware timing or incomplete paired trial")


def covered(plan, chars, limit, queues):
    matches = [case for case in plan if case[2] == queues]
    return bool(matches and min(x[0] for x in matches) <= chars <= max(x[0] for x in matches)
                and min(x[1] for x in matches) <= limit <= max(x[1] for x in matches))


def decision(native, weights, chars, limit, queues, future=None, weight=0.0):
    return native.choose(weights, chars, limit, queues, future or (chars, limit), weight)


def choose_route(native, policy, chars, limit, queues):
    if policy.get("method") == "reward_policy":
        return native.choose_rewards(policy["reward_weights"], chars, limit, queues)
    # The two-request lookahead models at most one request ahead on the next lane.
    future_weight = 0.0 if max(queues) > 1 else policy["future_weight"]
    return decision(native, policy["weights"], chars, limit, queues, policy["next_shape"], future_weight)


def _latency_weights(native, rows):
    fitted = []
    for i in range(len(rows[0]["arms"])):
        inputs = []
        for row in rows:
            size, limit, queues = row["case"]
            inputs.append((size, limit, queues[i], row["arms"][i][0]))
        fitted.append(native.fit(inputs))
    return fitted


def _compare_routes(native, policy, rows, schema):
    totals = [0.0] * 4
    counts, responses, changes = True, True if schema != LEGACY_SCHEMA else None, 0
    for row in rows:
        size, limit, queues = row["case"]
        chosen = choose_route(native, policy, size, limit, queues)
        ref = min(range(len(queues)), key=queues.__getitem__)
        base, actual = row["arms"][ref], row["arms"][chosen]
        counts &= base[2] == actual[2]
        if responses is not None:
            responses &= row["outputs"][ref] == row["outputs"][chosen]
        changes += chosen != ref
        for i, value in enumerate((actual[0], base[0], actual[1], base[1])):
            totals[i] += value
    return {"completion_ratio": totals[0] / totals[1], "first_token_ratio": totals[2] / totals[3],
            "equal_output_counts": counts, "matching_responses": responses, "changed_routes": changes}


def qualify(native, trace):
    check_trace(trace)
    rows = trace["trials"]
    if len(rows) != CASES * ROUNDS:
        raise ValueError(f"qualification needs {CASES * ROUNDS} completed paired trials, got {len(rows)}")
    binding, schema = trace["binding"], trace["schema"]
    if binding["trainer"] != native.version:
        raise ValueError("this trace belongs to a different native trainer")
    use_rewards = "reward_trainer" in binding
    if use_rewards and (schema == LEGACY_SCHEMA or binding["reward_trainer"] != native.reward_version):
        raise ValueError("incompatible response evidence or reward trainer")
    training = rows[:CASES * FIT_ROUNDS]
    # Traces before the reward learner priced their lookahead on the whole training partition.
    fitting = training[:-CASES] if use_rewards else training
    selection = training[-CASES:] if use_rewards else training
    future = [max(c[i] for c in trace["plan"]) for i in (0, 1)]
    prices = _latency_weights(native, fitting)
    candidates = [{"method": "latency_model", "weights": prices, "next_shape": future, "future_weight": w}
                  for w in (0.0, 0.25, 0.5)]
    if use_rewards and len(prices) > 1:
        candidates.append({**candidates[0], "method": "reward_policy", "reward_weights": native.fit_rewards(fitting)})
    scores = [_compare_routes(native, c, selection, schema) for c in candidates]
    def rank(i):
        row = scores[i]
        admissible = row["equal_output_counts"] and row["matching_responses"] and row["first_token_ratio"] <= 1
        return (use_rewards and not admissible, row["completion_ratio"])
    policy = candidates[min(range(len(scores)), key=rank)].copy()
    if use_rewards:
        policy["selection"] = [{"method": c["method"], "future_weight": c["future_weight"], **score}
                               for c, score in zip(candidates, scores)]
        policy["weights"] = _latency_weights(native, training)
        if policy["method"] == "reward_policy":
            policy["reward_weights"] = native.fit_rewards(training)
    rounds = [_compare_routes(native, policy, rows[r * CASES:(r + 1) * CASES], schema)
              for r in range(FIT_ROUNDS, ROUNDS)]
    passed = all(r["completion_ratio"] <= 0.95 and r["first_token_ratio"] <= 1 and r["equal_output_counts"]
                 and r["matching_responses"] is not False for r in rounds)
    return {**policy, "qualified": passed, "holdout": rounds, "identity": trace["identity"], "protocol": schema}


def remaining_idle_seconds(trace):
    """Estimate each remaining case from its own elapsed worker time once measured."""
    rows = trace["trials"]
    total = CASES * ROUNDS
    if len(rows) == total:
        return 0.0
    known = [r["wall_s"] for r in rows if "wall_s" in r]
    if not known:
        return None
    typical = statistics.mean(known)
    by_case = []
    for case in range(CASES):
        repeats = [r["wall_s"] for r in rows[case::CASES] if "wall_s" in r]
        by_case.append(statistics.median(repeats) if repeats else typical)
    return sum(1 + by_case[i % CASES] for i in range(len(rows), total))


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
        if status["paired_trials"] == status["target_trials"] and status["phase"] in ("qualified", "reference"):
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
    parser.add_argument("--devices", type=int, help="number of workers for an equal-cost estimate (default 2)")
    times = parser.add_mutually_exclusive_group()
    times.add_argument("--seconds-per-worker", nargs="+", type=float, metavar="SECONDS",
                       help="mean wall seconds for each worker in the same case, in device order")
    times.add_argument("--seconds-per-arm", type=float, help="mean time for one worker in one case, including load")
    parser.add_argument("--startup-minutes", type=float, default=0.0)
    args = parser.parse_args(argv)
    if args.command == "calibrate":
        from .cli import serve
        from .state import ProductStore
        if not args.gguf or not args.gguf.is_file() or not math.isfinite(args.minutes) or args.minutes <= 0:
            parser.error("calibrate needs an existing --gguf and positive --minutes")
        store = ProductStore(args.state_dir)
        return serve(["--gguf", str(args.gguf), "--placement", "auto", "--state-dir", str(store.root)],
                     calibration_minutes=args.minutes)
    if args.command == "estimate":
        if args.trace:
            if any(v is not None for v in (args.seconds_per_arm, args.seconds_per_worker, args.devices)) or args.startup_minutes:
                parser.error("--trace estimates remaining work from measurements and takes no assumed times")
            trace = json.loads(args.trace.read_text())
            check_trace(trace)
            print(json.dumps({"completed_cases": len(trace["trials"]), "target_cases": CASES * ROUNDS,
                              "estimated_remaining_idle_seconds": remaining_idle_seconds(trace),
                              "excludes": "foreground traffic, external contention and interrupted work"}))
            return
        if args.seconds_per_worker is not None:
            if args.devices is not None:
                parser.error("--seconds-per-worker already specifies the number of workers")
            durations = args.seconds_per_worker
        else:
            count = 2 if args.devices is None else args.devices
            if count < 1 or args.seconds_per_arm is None:
                parser.error("supply positive --devices and --seconds-per-arm, or --seconds-per-worker")
            durations = [args.seconds_per_arm]
        if (any(not math.isfinite(v) or v <= 0 for v in durations)
                or not math.isfinite(args.startup_minutes) or args.startup_minutes < 0):
            parser.error("worker seconds must be finite and positive; startup minutes must be finite and nonnegative")
        from .placement import IDLE_SECONDS
        seconds = sum(durations) if args.seconds_per_worker is not None else count * durations[0]
        workers = len(durations) if args.seconds_per_worker is not None else count
        idle_minutes = (CASES * ROUNDS * (seconds + 1) + IDLE_SECONDS) / 60
        print(json.dumps({"worker_measurements": CASES * ROUNDS * workers, "estimated_idle_minutes": idle_minutes,
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
