"""Turn warmed GGUF wave profiles into a bounded code experiment for an isolated evaluator.

Only an explicit measurement run installs the profiler. A dream estimates workload sensitivity
to time removed from an observed Python function. Its output is a development order.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import math
import re
import statistics
import subprocess
import time
from contextlib import contextmanager
from itertools import pairwise
from pathlib import Path
from typing import Any

from ironmule.ggml_agent import StepCost, simulate
from ironmule_product.state import _atomic_write

OBSERVATION = "ironmule.measured-wave.v1"
ORDER = "ironmule.code-experiment.v1"
EVALUATOR = "ironmule.serving_trial.v1"
TARGETS = ("ironmule/ggml_engine.py", "ironmule/gguf_tokenizer.py")
MAX_WAVES = 48
MAX_REQUESTS = 128
MAX_TOKENS = 4096


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


@contextmanager
def observe_wave(output: Path, workload: str, binding: dict, repo_root: Path, evaluator_id: str = EVALUATOR):
    """Capture caller-thread exclusive time. The caller adds [prompt, generated] token counts per request."""
    root = Path(repo_root).resolve()
    base = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *TARGETS], cwd=root, check=False).returncode:
        raise ValueError("profile the committed engine and tokenizer in a clean worktree")
    files = {str(root / name): name for name in TARGETS}
    observation = {"schema": OBSERVATION, "kind": "measured", "base_commit": base, "binding": binding,
                   "evaluator": evaluator_id, "workload": workload, "captured_ns": time.time_ns()}
    profiler = cProfile.Profile()
    start = time.perf_counter()
    with profiler:
        yield observation
    observation["wall_s"] = time.perf_counter() - start
    functions = []
    for entry in profiler.getstats():
        code = entry.code
        if isinstance(code, str) or code.co_filename not in files or code.co_name.startswith("<"):
            continue
        # ctypes wrappers can charge a whole native forward pass, kernel wait or join to
        # the Python frame. Do not count that time as headroom in the host scheduler.
        if {"llama_decode", "join", "wait", "sleep", "synchronize"}.intersection(code.co_names):
            continue
        functions.append({"path": files[code.co_filename], "function": code.co_qualname,
                          "line": code.co_firstlineno, "self_s": entry.inlinetime, "calls": entry.callcount})
    observation["functions"] = sorted(functions, key=lambda item: (item["path"], item["line"]))
    _validate(observation)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _atomic_write(output / f"wave-{digest(observation)}.json", observation)


def _validate(row):
    if (row["schema"] != OBSERVATION or row["kind"] != "measured" or row.get("correct") is not True
            or not row["binding"] or not row["workload"] or not row["evaluator"] or not row["base_commit"]):
        raise ValueError("expected a bound, successful measured wave")
    requests = row["requests"]
    if (not 0 < len(requests) <= MAX_REQUESTS or not 0 < row.get("width", len(requests)) <= MAX_REQUESTS
            or type(row.get("width", len(requests))) is not int
            or not all(len(pair) == 2 and all(type(n) is int and 0 < n <= MAX_TOKENS for n in pair)
                       for pair in requests) or sum(pair[1] for pair in requests) > 32768):
        raise ValueError("invalid or out-of-envelope request sizes")
    if not math.isfinite(row["wall_s"]) or row["wall_s"] <= 0:
        raise ValueError("invalid wall time")
    for item in row["functions"]:
        if (item["path"] not in TARGETS or not item["function"] or not isinstance(item["line"], int)
                or item["line"] < 1 or item["calls"] <= 0 or not math.isfinite(item["self_s"])
                or not 0 <= item["self_s"] <= row["wall_s"]):
            raise ValueError("invalid function measurement")


def _rollout(wave, cost, trace=None):
    arrivals = [(0.0, p, n) for p, n in wave["requests"]]
    return simulate(cost, wave.get("width", len(arrivals)), arrivals, trace=trace)


def _fit(rows):
    """Extend the GGML agent's per-step model with completed-wave timing observations.

    A wave supplies one regression row. Use the *average* simulated step features, since
    the measured target is elapsed seconds divided by the number of steps.
    """
    steps = []
    for wave in rows:
        trace = []
        _rollout(wave, StepCost([0.001, 0, 0, 0]), trace)
        steps.append((statistics.fmean(t[0] for t in trace), statistics.fmean(t[1] for t in trace),
                      statistics.fmean(t[2] for t in trace), wave["wall_s"] / len(trace)))
    model = StepCost([statistics.median(t[3] for t in steps), 0, 0, 0])
    for row in steps:
        model.update(*row)
    return model


def _next_waves(waves):
    """Replay what followed this workload last time, with all recent waves as a cold-start prior."""
    last = waves[-1]["workload"]
    transitions = [after for before, after in pairwise(waves) if before["workload"] == last]
    return transitions or waves


def _historical_notes(target, archive):
    """Fetch provenance-bearing references for the chosen source site, including prior failures."""
    from training.experience_archive import notebook, search

    root = Path(__file__).resolve().parents[1]
    parts = re.findall(r"[a-z0-9]+", target["function"].lower())
    terms = [part for part in parts if len(part) > 3 and part not in ("locals", "ggmlengine")]
    if target["path"].endswith("gguf_tokenizer.py"):
        terms += ["tokenizer", "tokenization"]
    if not terms:
        terms = [Path(target["path"]).name]
    if archive is not None:
        rows = (row for term in terms for row in search(archive, term, 100))
    else:
        blob = (root / "docs/LAB.md").read_bytes()
        rows = notebook("docs/LAB.md", hashlib.sha256(blob).hexdigest(), blob)
    matches = {}
    for row in rows:
        content = row.get("note") or row.get("reason") or ""
        haystack = (row.get("title", "") + " " + content).lower()
        hits = sum(term in haystack for term in terms)
        if hits == 0 or row["id"] in matches:
            continue
        position = min((content.lower().find(term) for term in terms if term in content.lower()), default=0)
        start = max(0, position - 150)
        excerpt = content[start:start + 650]
        negative = any(word in haystack for word in ("rejected", "confounded", "regressed", "slower", "no gain"))
        ref = {k: row.get(k) for k in ("id", "source", "source_sha256", "locator", "title", "provenance",
                                       "reason", "authorship", "uncertainty")}
        matches[row["id"]] = (hits, negative, {**ref, "excerpt": excerpt, "contains_negative_result": negative})
    ordered = sorted(matches.values(), key=lambda r: (-r[0], r[2]["id"]))
    selected = ordered[:3]
    if selected and not any(r[1] for r in selected):
        prior_failure = next((r for r in ordered if r[1]), None)
        if prior_failure is not None:
            selected[-1] = prior_failure
    return [r[2] for r in selected]


def plan(waves: list[dict], prior_orders=(), *, repetitions=4, archive=None) -> dict[str, Any]:
    if not 2 <= len(waves) <= MAX_WAVES or not 1 <= repetitions <= 32:
        raise ValueError("need two to 48 measured waves and a bounded repeat count")
    waves = sorted(waves, key=lambda w: w["captured_ns"])
    for row in waves:
        _validate(row)
    if len({digest(row) for row in waves}) != len(waves):
        raise ValueError("repeated wave record")
    context = {key: waves[0][key] for key in ("binding", "evaluator", "base_commit")}
    if any(any(row[k] != v for k, v in context.items()) for row in waves[1:]):
        raise ValueError("mixed machine, model, source or evaluator")
    model = _fit(waves)
    # Cost model error on the newest whole wave, predicted using only earlier observations.
    last = waves[-1]
    error = abs(_rollout(last, _fit(waves[:-1]))["seconds"] / last["wall_s"] - 1)
    forecast = _next_waves(waves)
    measured_cost_s = 2 * repetitions * statistics.fmean(w["wall_s"] for w in waves)
    seen = set()
    for order in prior_orders:
        if order.get("schema") == ORDER and all(order.get(k) == v for k, v in context.items()):
            target = order["target"]
            seen.add((target["path"], target["function"], target["line"]))
    observations = []
    for wave in waves:
        observations.append({(f["path"], f["function"], f["line"]): f["self_s"] for f in wave["functions"]})
    keys = set().union(*observations) - seen
    forecasts = [_rollout(wave, model) for wave in forecast]
    alternatives = []
    for key in sorted(keys):
        fractions = []
        shares = [functions.get(key, 0.0) / wave["wall_s"] for functions, wave in zip(observations, waves)]
        for wave in forecast:
            matched = [share for share, sample in zip(shares, waves) if sample["workload"] == wave["workload"]]
            fractions.append(statistics.median(matched))
        # A bounded sensitivity study: zero, quarter or half of measured self time removable.
        # These are assumptions about unpublished code, never labels or confidence intervals.
        # With one simultaneous burst, reducing every simulated step by the same
        # fraction scales its span linearly. The admission order and token counts stay fixed.
        exposed_s = statistics.fmean(base["seconds"] * fraction for base, fraction in zip(forecasts, fractions))
        expected_s = 0.25 * exposed_s
        if expected_s <= 0:
            continue
        alternatives.append({"path": key[0], "function": key[1], "line": key[2],
                             "measured_self_share_range": [min(shares), max(shares)],
                             "predicted_saved_s": expected_s, "sensitivity_saved_s": [0.0, 0.5 * exposed_s],
                             "score": expected_s / measured_cost_s})
    if not alternatives:
        raise ValueError("no remaining observed source function with positive modeled headroom")
    alternatives.sort(key=lambda item: -item["score"])
    best = alternatives[0]
    target = {k: best[k] for k in ("path", "function", "line")}
    order = {"schema": ORDER, **context, "target": target, "allowed_paths": [target["path"]],
             "hypothesis": f"Reduce equivalent per-request work in {target['function']} at {target['path']} "
                           f"line {target['line']}. Determine which cost is recoverable in the measured workload.",
             "workload": [w["workload"] for w in waves], "profile_sha256": [digest(w) for w in waves],
             "next_workloads": [w["workload"] for w in forecast], "dreams": alternatives,
             "historical_experience": _historical_notes(target, archive),
             "simulation": {"kind": "simulated", "step_model": model.to_json(),
                            "newest_wave_relative_prediction_error": error,
                            "removable_self_time_fractions": [0, 0.25, 0.5],
                            "paired_profiled_wave_cost_s": measured_cost_s,
                            "repetitions_estimate": repetitions,
                            "assumptions": "Synchronous burst arrival. Host cost scales across steps. Predicted "
                                           "test time excludes build, pod startup and billing. Profiling overhead "
                                           "and native calls can inflate attributed time."},
             "constraints": ["Preserve full request outputs, stop reasons and quality on fresh inputs.",
                             "Measure throughput and TTFT on matching real hardware with the profiler disabled.",
                             "Keep the evaluator, workload, reference, model weights and budget fixed."]}
    order["order_id"] = digest(order)
    return order


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prior-orders", type=Path)
    parser.add_argument("--archive", type=Path, help="experience JSONL; default searches the lab-notebook importer")
    args = parser.parse_args()
    files = sorted(args.observations.glob("wave-*.json"))
    if len(files) > MAX_WAVES:
        parser.error(f"select at most {MAX_WAVES} recent observations from the same binding")
    waves = [json.loads(file.read_text()) for file in files]
    prior = [] if args.prior_orders is None else [json.loads(file.read_text())
                                                  for file in sorted(args.prior_orders.glob("*.json"))]
    order = plan(waves, prior, archive=args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(args.output, order)
    print(json.dumps({"order_id": order["order_id"], "target": order["target"]}))


if __name__ == "__main__":
    main()
