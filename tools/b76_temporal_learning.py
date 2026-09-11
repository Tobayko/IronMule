#!/usr/bin/env python3
"""Does this machine's own history predict its next session, or only describe its last one?

`B75` found the `(4, 8)` geometry from a cold start and qualified it locally at `0.9624`
`[0.9563; 0.9630]`. `B69` had qualified the same intervention on the same machine at
`0.8469` `[0.7580; 0.9488]`. Both passed every gate they were measured under and their
intervals do not overlap. Sign agrees, magnitude does not, and neither run could have known
that because each was one confirmation.

`B76` asks the question that disagreement raises: over repeated independent sessions on one
M1 Max under natural state variation, is the *effect size* a property of this machine that a
small local learner can predict, or is it a property of the hour it was measured in?

**Nothing here is a cross-hardware claim.** One machine, one model, one intervention. `B73`
is the second-machine test and it has not run.

**The learner may only ever learn from `B76`'s own valid sessions.** No `B69` or `B75`
number enters training, and this file opens no historical record; the comparison against
them is `tools/b76_historical_context.py` and runs after the verdict is sealed.

**Every prediction is sealed before its ground truth exists.** Session `N + 1`'s predicted
ratio, interval, action, confidence and feature list are written `write_once` before the
session runs. A prediction that cannot be written cannot be scored.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import resource
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
TOOLS = PROJECT_ROOT / "tools"

MODEL = "mlx-community/gemma-3-12b-it-4bit"
GEOMETRY = (4, 8)
ARMS = ("reference", "candidate", "reference_aa")

#: One session is three blocks of three children. A child loads one model, runs one arm and
#: exits, which is `B69`'s fix for the swap growth that blocked its predecessor.
SESSION_BLOCKS = 3
REPEATS = 5
WARMUPS = 2
CHILD_TIMEOUT_S = 1800.0
DRIFT_LIMIT = 0.25

#: Measured before preregistration on this machine: 26.0 s for a single-class child, 56.8 s
#: for a three-class child, of which 9.6 s is the model load. Nine children per session.
CALIBRATED_CHILD_SECONDS = 56.8
CLASSES = (("single_short", 1, "strict", 32, "A"),
           ("single_long", 1, "strict", 128, "A"),
           ("session_warm", 3, "session_warm", 32, "B"))
PRIMARY_CLASS = "single_short"

TIME_BUDGET_S = 3 * 3600.0
MIN_SESSIONS = 10
MAX_SESSIONS = 14
#: Leaves the analysis, the sealing and one over-running session inside the three hours.
RESERVE_S = 600.0

#: The A/A arm's gate, kept from `B75` so the numbers stay comparable. Here it classifies a
#: session; it does not exclude one, and the reason is in the preregistration.
AA_MAX_OFFSET = 0.05
AA_MAX_HALF_WIDTH = 0.05

#: Model B's features. Fixed here, three of them, because ten rows do not support more.
LINEAR_FEATURES = ("load_1min", "memory_free_percent", "swap_used_gb")
RIDGE_ALPHA = 1.0
MIN_ROWS_FOR_LINEAR = 6
#: Model C's prior: no effect, with a spread wide enough to be moved by four sessions.
PRIOR_MEAN = 1.0
PRIOR_SD = 0.15

MEASURED_SOURCES = ("tools/b76_temporal_learning.py", "tools/b69_stack_proof.py",
                    "tools/b66_stack_proof.py", "tools/b57_stack_composition.py",
                    "ironmule/qmv_k3840.py", "ironmule/runtime.py", "ironmule/service.py",
                    "ironmule/plans.py")

PREREGISTRATION = {
    "experiment": "B76_temporal_local_learning",
    "question": ("over repeated independent sessions on this one M1 Max under natural state "
                 "variation, does IronMule's local evidence predict the next session's "
                 "effect size, and is its uncertainty calibrated"),
    "not_a_cross_hardware_test": ("one machine, one model, one intervention. B73 is the "
                                  "second-machine test and it has not run"),
    "not_an_optimisation": ("no new geometry, no search, no router or product dispatch. The "
                            "intervention is exactly the (4, 8) geometry B69 and B75 "
                            "measured, built from the same source"),
    "isolation": ("this file opens no B69, B75, B72 or silicon_profile record and no "
                  "historical number enters the learner. tools/b76_historical_context.py "
                  "places B69 and B75 beside the result and runs after the verdict is sealed"),
    "budget": {
        "wall_clock_seconds": TIME_BUDGET_S,
        "min_sessions_for_a_full_verdict": MIN_SESSIONS,
        "max_sessions": MAX_SESSIONS,
        "reserve_seconds": RESERVE_S,
        "stop_rule": ("before each session, stop and seal PARTIAL if elapsed plus the "
                      "estimated session duration plus the reserve exceeds the budget. The "
                      "estimate is the calibrated 56.8 s per child for session 1 and the "
                      "slowest completed session afterwards"),
        "no_extension": ("the budget is not raised because of a result, in either "
                         "direction, and no session is added to reach significance"),
    },
    "session": {
        "blocks": SESSION_BLOCKS, "children_per_block": 3, "repeats": REPEATS,
        "warmups": WARMUPS,
        "arms": ("A = the confirmed reference stack of the class, B = the same stack with "
                 "the (4, 8) geometry installed, plus the reference again under another "
                 "name as the A/A control"),
        "one_arm_per_process": ("each child loads one 12B image, runs one arm and exits, so "
                                "one model is resident at a time"),
        "arm_order": "rotated by (session index + block index) modulo 3, so AB and BA "
                     "alternate within a session and the phase shifts between sessions",
        "classes": [row[0] for row in CLASSES],
        "primary_class": PRIMARY_CLASS,
        "why_primary": ("B69 and B75 both qualified single_short, so it is the only class "
                        "where their disagreement is directly comparable. single_long and "
                        "session_warm run in every session, on the same fixed design, and "
                        "are reported as secondary. Nothing is selected after the fact"),
        "statistic": ("per-block ratio against the reference child of the same block, "
                      "median, 95 per cent bootstrap over 10000 resamples"),
    },
    "state_capture": {
        "when": "outside every timing window, immediately before the session's first child",
        "fields": ["load_1min", "load_5min", "load_15min", "memory_free_percent",
                   "memory_pressure_level", "swap_used_gb", "swap_occupancy",
                   "compressor_page_count", "wire_count", "swapouts", "pageouts",
                   "harness_rss_bytes", "seconds_since_previous_session",
                   "thermal_indicator"],
        "thermal": ("pmset -g therm is read and recorded. On this machine it reports no "
                    "thermal or performance warning level and no CPU power status, so there "
                    "is no reliable thermal or power indicator available without elevated "
                    "privileges and none is invented"),
        "no_interference": ("no artificial load, no process is stopped, no purge, no memory "
                            "or scheduler tuning. The machine's own state is the treatment "
                            "under study and interfering with it would remove the subject"),
    },
    "readiness": {
        "hard_gate": "gpu_busy() must find no other model process, or the session refuses to run",
        "load_average_is_recorded_not_gated": (
            "B69's 4.0 start gate is deliberately not applied. A study of natural state "
            "variation that only runs when the machine is quiet has removed its own "
            "independent variable. Load is a recorded feature; correctness, the B65 "
            "resource gate and the drift gate decide whether a session is usable"),
    },
    "session_validity": {
        "required_for_learning": [
            "three complete blocks with no child failure",
            "token ids, physical counts and stop reasons identical across all three arms",
            "zero fallbacks",
            "no block deviating more than 25 per cent from the session's median block wall",
            "B65 in full: macOS pressure normal at every probe, free memory at or above 10 "
            "per cent, peak child RSS at or below 60 per cent of installed, swap in use "
            "never above its value at session start",
        ],
        "aa_is_measured_not_gating": (
            "the A/A arm is recorded, reported and used as this session's own estimate of "
            "measurement error. It does not exclude a session from learning. Excluding "
            "noisy sessions would answer H2 by construction: the between-session spread "
            "under natural state variation is the quantity being measured"),
        "blocked_and_invalid_are_never_learned": True,
    },
    "learner": {
        "starts_empty": "session 1 runs with no performance knowledge of any kind",
        "updates_only_from_B76": ("after each valid session and from that session's own "
                                  "ground truth only. No B69 or B75 number is training data"),
        "sealed_before_ground_truth": (
            "before session N + 1 runs, every model writes predicted_ratio, "
            "prediction_interval, predicted_action, confidence, abstain reason and the "
            "state features it used, write_once, to its own file. Only then is the session "
            "measured"),
        "models": {
            "A_constant": ("the mean of prior valid B76 primary ratios with a normal "
                           "predictive interval. The trivial baseline H3 must beat"),
            "B_ridge": (f"ridge regression, alpha {RIDGE_ALPHA} fixed, on standardised "
                        f"{list(LINEAR_FEATURES)}; abstains below {MIN_ROWS_FOR_LINEAR} "
                        "rows. No hyperparameter sweep and no feature search"),
            "C_bayes": (f"normal-normal update from prior N({PRIOR_MEAN}, {PRIOR_SD}^2) "
                        "with a method-of-moments between-session variance, giving an "
                        "explicit posterior predictive interval"),
        },
        "all_three_are_sealed_every_session": (
            "so naming the best model at the end is a report of prospectively sealed "
            "predictions, not a selection made after seeing the answers"),
        "action_rule": ("CANDIDATE if the predictive interval lies entirely below 1.0, "
                        "REFERENCE if entirely above, ABSTAIN if it contains 1.0 or the "
                        "model has too little evidence to form one"),
        "primary_model_for_H3": "B_ridge against A_constant",
        "primary_model_for_H4": "C_bayes",
    },
    "hypotheses": {
        "H1": "the (4, 8) geometry keeps the same sign across valid sessions",
        "H2": ("effect size varies more between sessions than each session's own "
               "measurement error explains, tested as a method-of-moments between-session "
               "variance above zero against the pooled within-session variance"),
        "H3": ("observed machine state improves prediction over a constant model, tested as "
               "prequential mean absolute error of B_ridge against A_constant on sealed "
               "out-of-sample predictions"),
        "H4": ("uncertainty becomes better calibrated as local evidence accumulates, tested "
               "as C_bayes coverage and interval width in the second half against the first"),
        "no_causal_claim": ("a state feature that predicts is reported as predictive. "
                            "Nothing here randomises machine state, so no coefficient is "
                            "read as a cause"),
    },
    "verdicts": {
        "B76_TEMPORAL_LEARNING_CONFIRMED": ("future sessions are predicted better than the "
                                            "trivial baseline and uncertainty is usably "
                                            "calibrated"),
        "B76_STABLE_ACTION_VARIABLE_GAIN": ("the geometry wins reliably but its effect size "
                                            "is not dependably predictable"),
        "B76_SAFE_ABSTAIN": ("the data do not support a reliable forecast and the learner "
                             "correctly declines to make one"),
        "B76_LOCAL_LEARNING_FAIL": "confidently wrong or catastrophic decisions",
        "B76_INVALID": "a methodological or safety defect",
        "B76_PARTIAL": f"fewer than {MIN_SESSIONS} valid sessions inside the budget",
    },
    "nothing_is_activated": ("no threshold is changed, no profile written, no default moved, "
                             "no kernel released, nothing committed, pushed or activated"),
}

CHILD = r'''
import json, sys
sys.path.insert(0, {root!r})
sys.path.insert(0, {tools!r})
from b69_stack_proof import child_main
print("@@" + json.dumps(child_main(json.loads(sys.argv[1])), sort_keys=True, allow_nan=False), flush=True)
'''


# --------------------------------------------------------------------------- statistics


def bootstrap(ratios, resamples: int = 10000, seed: int = 20260910) -> dict:
    rng = random.Random(seed)
    if not ratios:
        return {"median": None, "ci_low": None, "ci_high": None, "n": 0, "ratios": []}
    draws = sorted(median([rng.choice(ratios) for _ in ratios]) for _ in range(resamples))
    return {"median": median(ratios), "ci_low": draws[int(0.025 * resamples)],
            "ci_high": draws[int(0.975 * resamples) - 1], "n": len(ratios),
            "ratios": list(ratios)}


def _sd(values) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


# --------------------------------------------------------------------------- state


def _thermal_indicator() -> dict:
    try:
        out = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True,
                             timeout=10.0).stdout.strip()
    except Exception as error:  # pragma: no cover - depends on the host
        return {"available": False, "reason": repr(error), "raw": None}
    available = "No thermal warning level" not in out
    return {"available": available, "raw": out,
            "reason": ("" if available else
                       "macOS records no thermal or performance warning level on this "
                       "machine, so there is no reliable indicator to read without "
                       "elevated privileges")}


def capture_state(previous_finished_at: float | None) -> dict:
    """Everything the learner is allowed to see, read outside any timing window."""
    from ironmule.hw import (installed_memory_bytes, memory_pressure_level, swap_used_bytes,
                             vm_counters)

    counters = vm_counters() or {}
    total = installed_memory_bytes() or 0
    total_pages = total // 16384
    free_pages = (counters.get("free_count", 0) + counters.get("speculative_count", 0)
                  + counters.get("inactive_count", 0))
    swap = swap_used_bytes()
    load = os.getloadavg()
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "load_1min": load[0], "load_5min": load[1], "load_15min": load[2],
        "memory_free_percent": (100.0 * free_pages / total_pages) if total_pages else None,
        "memory_pressure_level": memory_pressure_level(),
        "swap_used_gb": (swap / 1e9) if swap is not None else None,
        "swap_occupancy": (swap / total) if (swap is not None and total) else None,
        "compressor_page_count": counters.get("compressor_page_count"),
        "wire_count": counters.get("wire_count"),
        "swapouts": counters.get("swapouts"), "pageouts": counters.get("pageouts"),
        "harness_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "peak_child_rss_bytes": int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss),
        "seconds_since_previous_session": (None if previous_finished_at is None
                                           else time.time() - previous_finished_at),
        "thermal_indicator": _thermal_indicator(),
    }


# --------------------------------------------------------------------------- models


def _predictive(mean: float, sd: float) -> list[float]:
    return [mean - 1.96 * sd, mean + 1.96 * sd]


def _action(interval) -> str:
    if interval is None:
        return "ABSTAIN"
    low, high = interval
    if high < 1.0:
        return "CANDIDATE"
    if low > 1.0:
        return "REFERENCE"
    return "ABSTAIN"


def _abstained(model: str, reason: str, features) -> dict:
    return {"model": model, "predicted_ratio": None, "prediction_interval": None,
            "predicted_action": "ABSTAIN", "confidence": None, "abstain_reason": reason,
            "state_features_used": list(features)}


def model_a_constant(history: list[dict], state: dict) -> dict:
    """The trivial baseline: this machine has one number and no idea what moves it."""
    ratios = [row["ratio"] for row in history]
    if len(ratios) < 2:
        return _abstained("A_constant", "fewer than two prior valid sessions, so no spread "
                                        "can be formed and a point estimate is not a "
                                        "prediction", [])
    mean = statistics.fmean(ratios)
    spread = _sd(ratios) * math.sqrt(1.0 + 1.0 / len(ratios))
    interval = _predictive(mean, spread)
    return {"model": "A_constant", "predicted_ratio": mean, "prediction_interval": interval,
            "predicted_action": _action(interval),
            "confidence": {"half_width": (interval[1] - interval[0]) / 2,
                           "n_prior_sessions": len(ratios)},
            "abstain_reason": "", "state_features_used": []}


def model_b_ridge(history: list[dict], state: dict) -> dict:
    """Does the machine's own observed state carry any of the variation?"""
    import numpy as np

    usable = [row for row in history
              if all(row["state"].get(name) is not None for name in LINEAR_FEATURES)]
    if len(usable) < MIN_ROWS_FOR_LINEAR:
        return _abstained("B_ridge", f"{len(usable)} usable prior rows, below the "
                                     f"{MIN_ROWS_FOR_LINEAR} this model needs before three "
                                     f"features may be fitted", LINEAR_FEATURES)
    if any(state.get(name) is None for name in LINEAR_FEATURES):
        missing = [n for n in LINEAR_FEATURES if state.get(n) is None]
        return _abstained("B_ridge", f"state features unreadable this session: {missing}",
                          LINEAR_FEATURES)

    matrix = np.array([[row["state"][name] for name in LINEAR_FEATURES] for row in usable],
                      dtype=float)
    target = np.array([row["ratio"] for row in usable], dtype=float)
    centre, scale = matrix.mean(axis=0), matrix.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    design = (matrix - centre) / scale
    intercept = target.mean()
    gram = design.T @ design + RIDGE_ALPHA * np.eye(design.shape[1])
    coefficients = np.linalg.solve(gram, design.T @ (target - intercept))

    # Leave-one-out residuals, so the spread is not the in-sample fit congratulating itself.
    loo = []
    for index in range(len(usable)):
        keep = [i for i in range(len(usable)) if i != index]
        sub, sub_y = design[keep], target[keep]
        sub_intercept = sub_y.mean()
        sub_gram = sub.T @ sub + RIDGE_ALPHA * np.eye(sub.shape[1])
        sub_coefficients = np.linalg.solve(sub_gram, sub.T @ (sub_y - sub_intercept))
        loo.append(target[index] - (sub_intercept + design[index] @ sub_coefficients))
    residual_sd = float(np.sqrt(np.mean(np.square(loo)))) if loo else 0.0

    point = np.array([state[name] for name in LINEAR_FEATURES], dtype=float)
    predicted = float(intercept + ((point - centre) / scale) @ coefficients)
    interval = _predictive(predicted, residual_sd)
    return {"model": "B_ridge", "predicted_ratio": predicted, "prediction_interval": interval,
            "predicted_action": _action(interval),
            "confidence": {"half_width": (interval[1] - interval[0]) / 2,
                           "n_prior_sessions": len(usable),
                           "leave_one_out_residual_sd": residual_sd,
                           "standardised_coefficients": dict(
                               zip(LINEAR_FEATURES, [float(c) for c in coefficients]))},
            "abstain_reason": "", "state_features_used": list(LINEAR_FEATURES)}


def model_c_bayes(history: list[dict], state: dict) -> dict:
    """A posterior that carries its own uncertainty and says when it has none to spare."""
    rows = [row for row in history if row["within_session_se"] is not None]
    if len(rows) < 2:
        return _abstained("C_bayes", "fewer than two prior valid sessions with a usable "
                                     "within-session error estimate", [])
    ratios = [row["ratio"] for row in rows]
    within = statistics.fmean([row["within_session_se"] ** 2 for row in rows])
    between = max(0.0, statistics.variance(ratios) - within) if len(ratios) > 1 else 0.0

    precision = 1.0 / PRIOR_SD ** 2
    weighted = PRIOR_MEAN / PRIOR_SD ** 2
    for row in rows:
        variance = row["within_session_se"] ** 2 + between
        variance = max(variance, 1e-9)
        precision += 1.0 / variance
        weighted += row["ratio"] / variance
    posterior_mean = weighted / precision
    posterior_variance = 1.0 / precision
    predictive_sd = math.sqrt(posterior_variance + between + within)
    interval = _predictive(posterior_mean, predictive_sd)
    return {"model": "C_bayes", "predicted_ratio": posterior_mean,
            "prediction_interval": interval, "predicted_action": _action(interval),
            "confidence": {"half_width": (interval[1] - interval[0]) / 2,
                           "n_prior_sessions": len(rows),
                           "posterior_sd": math.sqrt(posterior_variance),
                           "between_session_variance": between,
                           "mean_within_session_variance": within},
            "abstain_reason": "", "state_features_used": []}


MODELS = {"A_constant": model_a_constant, "B_ridge": model_b_ridge, "C_bayes": model_c_bayes}
PRIMARY_MODEL = "C_bayes"


def predict_all(history: list[dict], state: dict) -> dict:
    return {name: builder(history, state) for name, builder in MODELS.items()}


# --------------------------------------------------------------------------- session


def run_session(session_index: int) -> dict:
    """Three blocks, nine children, one arm and one resident model per child."""
    from ironmule.hw import (installed_memory_bytes, memory_pressure_level, swap_used_bytes,
                             vm_counters)

    total = installed_memory_bytes() or 0
    total_pages = total // 16384

    def probe(label):
        counters = vm_counters() or {}
        free_pages = (counters.get("free_count", 0) + counters.get("speculative_count", 0)
                      + counters.get("inactive_count", 0))
        return {"label": label, "swap_used_bytes": swap_used_bytes(),
                "memory_pressure_level": memory_pressure_level(),
                "load_average": list(os.getloadavg()),
                "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss),
                "memory_free_percent": (100.0 * free_pages / total_pages) if total_pages else None}

    started = time.perf_counter()
    samples = [probe("start")]
    blocks, failures = [], []
    for block_index in range(SESSION_BLOCKS):
        shift = (session_index + block_index) % len(ARMS)
        order = ARMS[shift:] + ARMS[:shift]
        entry = {"block": block_index, "order": list(order), "children": {}}
        for arm in order:
            spec = {"model": MODEL, "arm": "reference" if arm == "reference_aa" else arm,
                    "geometry": list(GEOMETRY), "classes": [list(row) for row in CLASSES],
                    "repeats": REPEATS, "warmups": WARMUPS, "root": str(PROJECT_ROOT)}
            process = subprocess.Popen(
                [sys.executable, "-u", "-c",
                 CHILD.format(root=str(PROJECT_ROOT), tools=str(TOOLS)), json.dumps(spec)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                cwd=str(PROJECT_ROOT),
                env={**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"})
            try:
                stdout, stderr = process.communicate(timeout=CHILD_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
            code = process.returncode
            marker = next((l[2:] for l in stdout.splitlines() if l.startswith("@@")), None)
            if code != 0 or marker is None:
                failures.append({"block": block_index, "arm": arm, "returncode": code,
                                 "signal_name": (signal.Signals(-code).name
                                                 if code is not None and code < 0 else None),
                                 "stderr_tail": stderr[-3000:]})
                break
            entry["children"][arm] = json.loads(marker)
            print(f"  session {session_index} block {block_index} {arm} done", flush=True)
        blocks.append(entry)
        samples.append(probe(f"after_block_{block_index}"))
        if failures:
            break
    samples.append(probe("end"))

    complete = [b for b in blocks if set(b["children"]) == set(ARMS)]
    differences, comparisons, block_ratios = [], {}, {}
    for name, *_rest in CLASSES:
        base_tokens = None
        for entry in complete:
            base = entry["children"]["reference"]["classes"][name]
            base_tokens = base_tokens or base["tokens"]
            for arm in ARMS:
                row = entry["children"][arm]["classes"][name]
                if row["tokens"] != base_tokens or row["stop_reasons"] != base["stop_reasons"]:
                    differences.append({"block": entry["block"], "arm": arm, "class": name})
        block_ratios[name] = {
            arm: [entry["children"][arm]["classes"][name]["wall_ns"]
                  / entry["children"]["reference"]["classes"][name]["wall_ns"]
                  for entry in complete]
            for arm in ("candidate", "reference_aa")}
        comparisons[name] = {arm: bootstrap(values)
                             for arm, values in block_ratios[name].items()}

    walls = [sum(entry["children"][arm]["classes"][name]["wall_ns"] / 1e6
                 for arm in ARMS for name, *_r in CLASSES) for entry in complete]
    wall_median = median(walls) if walls else None
    disturbed = ([{"block": complete[i]["block"], "wall_ms": w,
                   "deviation": w / wall_median - 1.0}
                  for i, w in enumerate(walls) if abs(w / wall_median - 1.0) > DRIFT_LIMIT]
                 if wall_median else [])

    swaps = [s["swap_used_bytes"] for s in samples if s["swap_used_bytes"] is not None]
    frees = [s["memory_free_percent"] for s in samples if s["memory_free_percent"] is not None]
    levels = [s["memory_pressure_level"] for s in samples]
    peak_rss = max((s["peak_rss_bytes"] for s in samples), default=0)
    fallbacks = sum(entry["children"][arm]["classes"][name]["fallbacks"]
                    for entry in complete for arm in ARMS for name, *_r in CLASSES)
    gate = []
    if not swaps or max(swaps) > swaps[0]:
        gate.append("swap in use rose above its value at session start")
    if not frees or min(frees) < 10.0:
        gate.append("free memory below 10 per cent")
    if not all(level == 1 and level is not None for level in levels):
        gate.append("macOS memory pressure not normal at every probe, or unreadable")
    if total and peak_rss > total * 0.60:
        gate.append("peak child RSS above 60 per cent of installed memory")

    valid = (len(complete) == SESSION_BLOCKS and not failures and not differences
             and not fallbacks and not disturbed and not gate)
    primary = comparisons.get(PRIMARY_CLASS, {}).get("candidate")
    aa = comparisons.get(PRIMARY_CLASS, {}).get("reference_aa")
    aa_half = ((aa["ci_high"] - aa["ci_low"]) / 2
               if aa and aa["ci_high"] is not None else None)
    aa_ok = (aa_half is not None and aa_half <= AA_MAX_HALF_WIDTH
             and abs(aa["median"] - 1.0) <= AA_MAX_OFFSET)
    ratios = block_ratios.get(PRIMARY_CLASS, {}).get("candidate", [])
    within_se = (_sd(ratios) / math.sqrt(len(ratios))) if len(ratios) > 1 else None

    return {
        "session": session_index, "seconds": time.perf_counter() - started,
        "complete_blocks": len(complete), "child_failures": failures,
        "correctness_identical": not differences, "differences": differences,
        "fallbacks": fallbacks, "disturbed_blocks": disturbed,
        "resource_gate_passed": not gate, "resource_gate_reasons": gate,
        "resource_samples": samples, "peak_child_rss_bytes": peak_rss,
        "comparisons": comparisons, "block_ratios": block_ratios,
        "blocks": blocks,
        "valid_for_learning": bool(valid),
        "status": ("VALID" if valid else ("BLOCKED" if (failures or gate or disturbed)
                                          else "INVALID")),
        "primary": primary, "aa": aa, "aa_half_width": aa_half, "aa_gate_passed": bool(aa_ok),
        "within_session_se": within_se,
        "block_wall_ms": walls, "median_block_wall_ms": wall_median,
    }


# --------------------------------------------------------------------------- scoring


def score(history: list[dict], sealed: list[dict]) -> dict:
    """Prediction against ground truth, per model, on sealed out-of-sample predictions."""
    per_model = {}
    for name in MODELS:
        rows = []
        for record in sealed:
            prediction = record["predictions"][name]
            truth = record.get("ground_truth")
            if truth is None or truth.get("ratio") is None or not truth["valid_for_learning"]:
                continue
            actual = truth["ratio"]
            best_action = "CANDIDATE" if truth["ci_high"] < 1.0 else (
                "REFERENCE" if truth["ci_low"] > 1.0 else "AMBIGUOUS")
            action = prediction["predicted_action"]
            realised = actual if action == "CANDIDATE" else 1.0
            rows.append({
                "session": record["session"],
                "predicted_ratio": prediction["predicted_ratio"],
                "interval": prediction["prediction_interval"],
                "actual_ratio": actual,
                "absolute_error": (None if prediction["predicted_ratio"] is None
                                   else abs(prediction["predicted_ratio"] - actual)),
                "covered": (None if prediction["prediction_interval"] is None else
                            bool(prediction["prediction_interval"][0] <= actual
                                 <= prediction["prediction_interval"][1])),
                "predicted_action": action, "best_action": best_action,
                "action_correct": (None if best_action == "AMBIGUOUS"
                                   else bool(action == best_action)),
                "regret": realised - min(actual, 1.0),
                "abstained": action == "ABSTAIN",
                "confidently_wrong": bool(
                    prediction["prediction_interval"] is not None
                    and not (prediction["prediction_interval"][0] <= actual
                             <= prediction["prediction_interval"][1])
                    and (prediction["prediction_interval"][1]
                         - prediction["prediction_interval"][0]) / 2 < 0.02),
            })
        scored = [r for r in rows if r["absolute_error"] is not None]
        covered = [r for r in rows if r["covered"] is not None]
        judged = [r for r in rows if r["action_correct"] is not None]
        per_model[name] = {
            "n_scored": len(scored),
            "mean_absolute_error": statistics.fmean([r["absolute_error"] for r in scored]) if scored else None,
            "median_absolute_error": median([r["absolute_error"] for r in scored]) if scored else None,
            "coverage_95": (sum(r["covered"] for r in covered) / len(covered)) if covered else None,
            "mean_interval_half_width": (statistics.fmean(
                [(r["interval"][1] - r["interval"][0]) / 2 for r in covered]) if covered else None),
            "action_accuracy": (sum(r["action_correct"] for r in judged) / len(judged)) if judged else None,
            "total_regret": sum(r["regret"] for r in rows) if rows else None,
            "mean_regret": statistics.fmean([r["regret"] for r in rows]) if rows else None,
            "abstain_rate": (sum(r["abstained"] for r in rows) / len(rows)) if rows else None,
            "confidently_wrong": sum(r["confidently_wrong"] for r in rows),
            "rows": rows,
        }
        half = len(covered) // 2
        if half >= 2:
            first, second = covered[:half], covered[half:]
            per_model[name]["calibration_over_time"] = {
                "first_half_coverage": sum(r["covered"] for r in first) / len(first),
                "second_half_coverage": sum(r["covered"] for r in second) / len(second),
                "first_half_half_width": statistics.fmean(
                    [(r["interval"][1] - r["interval"][0]) / 2 for r in first]),
                "second_half_half_width": statistics.fmean(
                    [(r["interval"][1] - r["interval"][0]) / 2 for r in second]),
            }
    return per_model


def variance_decomposition(history: list[dict]) -> dict:
    """H2: is the spread between sessions bigger than each session's own error explains?"""
    ratios = [row["ratio"] for row in history]
    within = [row["within_session_se"] for row in history if row["within_session_se"] is not None]
    if len(ratios) < 2:
        return {"n": len(ratios), "between_session_variance_total": None}
    total = statistics.variance(ratios)
    mean_within = statistics.fmean([se ** 2 for se in within]) if within else None
    tau_squared = (max(0.0, total - mean_within) if mean_within is not None else None)
    return {
        "n": len(ratios),
        "session_ratios": ratios,
        "between_session_variance_total": total,
        "between_session_sd_total": math.sqrt(total),
        "mean_within_session_variance": mean_within,
        "mean_within_session_sd": math.sqrt(mean_within) if mean_within is not None else None,
        "tau_squared_method_of_moments": tau_squared,
        "tau_sd": math.sqrt(tau_squared) if tau_squared is not None else None,
        "variance_ratio_total_over_within": (total / mean_within
                                             if mean_within else None),
        "range": [min(ratios), max(ratios)],
    }


def source_binding() -> dict:
    digests, running = {}, hashlib.sha256()
    for name in MEASURED_SOURCES:
        data = (PROJECT_ROOT / name).read_bytes()
        digests[name] = hashlib.sha256(data).hexdigest()
        running.update(name.encode())
        running.update(data)
    return {"files": digests, "combined_sha256": running.hexdigest()}


# --------------------------------------------------------------------------- verdict


def decide_verdict(history, sessions, scored, decomposition) -> dict:
    reasons = []
    safety_defect = any(
        (not s["correctness_identical"]) or s["fallbacks"] for s in sessions)
    if safety_defect:
        return {"verdict": "B76_INVALID",
                "reason": "a session differed in tokens or stop reasons, or took a fallback"}
    if len(history) < MIN_SESSIONS:
        reasons.append(f"{len(history)} valid sessions, below the preregistered "
                       f"{MIN_SESSIONS}")
        return {"verdict": "B76_PARTIAL", "reason": "; ".join(reasons)}

    primary = scored[PRIMARY_MODEL]
    baseline = scored["A_constant"]
    linear = scored["B_ridge"]
    if primary["confidently_wrong"] or baseline["confidently_wrong"] or linear["confidently_wrong"]:
        return {"verdict": "B76_LOCAL_LEARNING_FAIL",
                "reason": "a model placed a narrow interval that the ground truth fell outside"}

    signs = [row["ci_high"] < 1.0 for row in
             [{"ci_high": s["primary"]["ci_high"]} for s in sessions if s["valid_for_learning"]]]
    h1 = all(signs)
    h3 = (linear["mean_absolute_error"] is not None
          and baseline["mean_absolute_error"] is not None
          and linear["mean_absolute_error"] < baseline["mean_absolute_error"])
    coverage = primary["coverage_95"]
    calibrated = coverage is not None and coverage >= 0.80

    abstain = primary["abstain_rate"]
    common = {"H1": h1, "H3": h3, "coverage_95": coverage, "abstain_rate": abstain}
    if h3 and calibrated and abstain is not None and abstain < 0.6:
        return {"verdict": "B76_TEMPORAL_LEARNING_CONFIRMED",
                "reason": ("state-aware prediction beat the constant baseline, the primary "
                           "model's intervals covered at the nominal rate, and it forecast "
                           "rather than abstained in most sessions"),
                **common}
    if h1:
        # A stable action is the more informative answer than a wide interval, so it is read
        # before the abstention rate: the learner declining to forecast a magnitude does not
        # erase the finding that the sign never moved.
        return {"verdict": "B76_STABLE_ACTION_VARIABLE_GAIN",
                "reason": ("the geometry kept its sign in every valid session, but the "
                           "effect size was not predicted better than a constant, or the "
                           "uncertainty was not usably calibrated"),
                **common}
    if abstain is not None and abstain >= 0.6:
        return {"verdict": "B76_SAFE_ABSTAIN",
                "reason": (f"the sign was not stable across sessions and the primary model "
                           f"declined to forecast in {abstain:.0%} of them rather than guess"),
                **common}
    return {"verdict": "B76_SAFE_ABSTAIN",
            "reason": "neither a stable action nor a usable forecast, and nothing was adopted",
            **common}


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seal-dir", type=Path)
    parser.add_argument("--preregister", type=Path)
    parser.add_argument("--budget-seconds", type=float, default=TIME_BUDGET_S)
    args = parser.parse_args(argv)

    from b52_automatic_selection import write_once

    if args.preregister:
        write_once(args.preregister, json.dumps(
            {"preregistration": PREREGISTRATION, "source_binding": source_binding(),
             "calibration_before_preregistration": {
                 "single_class_child_seconds": 26.0, "three_class_child_seconds": 56.8,
                 "model_load_seconds": 9.6,
                 "note": ("one reference child, measured before this preregistration was "
                          "written, to fix the session design against the budget. It is not "
                          "evidence and no learner ever sees it")},
             "written_at": datetime.now(timezone.utc).isoformat()},
            indent=2, sort_keys=True))
        print(f"preregistration written to {args.preregister}")
        return 0

    seal_dir = args.seal_dir or args.out.parent / (args.out.stem + "_sealed")
    seal_dir.mkdir(parents=True, exist_ok=True)

    import mlx.core as mx
    import mlx_lm
    from ironmule.hw import fingerprint, installed_memory_bytes, static_facts
    from ironmule.tune import gpu_busy

    busy = gpu_busy()
    if busy:
        raise SystemExit(f"another model process is running, refusing to measure ({busy})")

    wall_started = time.perf_counter()
    history: list[dict] = []
    sealed: list[dict] = []
    sessions: list[dict] = []
    previous_finished_at = None
    estimate = CALIBRATED_CHILD_SECONDS * SESSION_BLOCKS * len(ARMS)
    stopped_because = None

    for session_index in range(MAX_SESSIONS):
        elapsed = time.perf_counter() - wall_started
        if elapsed + estimate + RESERVE_S > args.budget_seconds:
            stopped_because = (f"the next session would not fit: {elapsed:.0f} s elapsed, "
                               f"{estimate:.0f} s estimated, {RESERVE_S:.0f} s reserved, "
                               f"budget {args.budget_seconds:.0f} s")
            break

        state = capture_state(previous_finished_at)
        predictions = predict_all(history, state)
        seal_path = seal_dir / f"session_{session_index:02d}_prediction.json"
        write_once(seal_path, json.dumps(
            {"experiment": PREREGISTRATION["experiment"], "session": session_index,
             "sealed_at": datetime.now(timezone.utc).isoformat(),
             "learner_state": {"n_prior_valid_sessions": len(history),
                               "prior_ratios": [r["ratio"] for r in history]},
             "state": state, "predictions": predictions,
             "primary_model": PRIMARY_MODEL,
             "note": ("written before the session ran. The ground truth for this session "
                      "did not exist when this file was created")},
            indent=2, sort_keys=True, default=str))
        print(f"session {session_index}: sealed prediction "
              f"{predictions[PRIMARY_MODEL]['predicted_action']} "
              f"{predictions[PRIMARY_MODEL]['predicted_ratio']}", flush=True)

        result = run_session(session_index)
        previous_finished_at = time.time()
        result["state_before"] = state
        result["sealed_prediction_path"] = str(seal_path)
        sessions.append(result)
        write_once(seal_dir / f"session_{session_index:02d}_result.json",
                   json.dumps(result, indent=2, sort_keys=True, default=str))

        primary = result["primary"]
        ground_truth = None
        if primary and primary["median"] is not None:
            ground_truth = {"ratio": primary["median"], "ci_low": primary["ci_low"],
                            "ci_high": primary["ci_high"],
                            "valid_for_learning": result["valid_for_learning"]}
        sealed.append({"session": session_index, "state": state, "predictions": predictions,
                       "ground_truth": ground_truth, "status": result["status"]})

        if result["valid_for_learning"] and ground_truth:
            history.append({"session": session_index, "ratio": ground_truth["ratio"],
                            "ci": [ground_truth["ci_low"], ground_truth["ci_high"]],
                            "within_session_se": result["within_session_se"],
                            "aa_median": result["aa"]["median"] if result["aa"] else None,
                            "aa_gate_passed": result["aa_gate_passed"],
                            "state": state})
        estimate = max(estimate, result["seconds"] * 1.05)
        print(f"session {session_index}: {result['status']} ratio="
              f"{ground_truth['ratio'] if ground_truth else None} "
              f"({result['seconds']:.0f} s)", flush=True)

    scored = score(history, sealed)
    decomposition = variance_decomposition(history)
    verdict = decide_verdict(history, sessions, scored, decomposition)

    aa_all = [s["aa"]["median"] for s in sessions if s["aa"] and s["aa"]["median"] is not None]
    secondary = {}
    for name, *_rest in CLASSES:
        if name == PRIMARY_CLASS:
            continue
        values = [s["comparisons"][name]["candidate"]["median"] for s in sessions
                  if s["valid_for_learning"] and name in s["comparisons"]
                  and s["comparisons"][name]["candidate"]["median"] is not None]
        secondary[name] = {"n": len(values), "ratios": values,
                           "median": median(values) if values else None,
                           "sd": _sd(values) if values else None,
                           "bootstrap": bootstrap(values) if values else None}

    record = {
        "experiment": PREREGISTRATION["experiment"],
        "preregistration": PREREGISTRATION,
        "source_binding": source_binding(),
        "environment": {"platform": platform.platform(), "mlx": mx.__version__,
                        "mlx_lm": mlx_lm.__version__, "fingerprint": fingerprint(),
                        "chip": static_facts().get("chip"),
                        "installed_memory_bytes": installed_memory_bytes()},
        "sealed_at": datetime.now(timezone.utc).isoformat(),
        "isolation": ("no B69, B75, B72 or silicon_profile record was opened. The historical "
                      "comparison is tools/b76_historical_context.py and runs after this"),
        "budget": {"seconds": args.budget_seconds,
                   "wall_seconds_used": time.perf_counter() - wall_started,
                   "stopped_because": stopped_because,
                   "sessions_run": len(sessions), "sessions_valid": len(history)},
        "sessions": sessions,
        "sealed_predictions": sealed,
        "learning_history": history,
        "primary_class": PRIMARY_CLASS,
        "secondary_classes": secondary,
        "aa_distribution": {"n": len(aa_all), "values": aa_all,
                            "median": median(aa_all) if aa_all else None,
                            "sd": _sd(aa_all) if aa_all else None,
                            "max_absolute_offset": max((abs(v - 1.0) for v in aa_all),
                                                       default=None)},
        "variance_decomposition": decomposition,
        "scoring": scored,
        "verdict": verdict,
        "time_to_calibrated_knowledge": _time_to_calibrated(sealed),
        "nothing_activated": PREREGISTRATION["nothing_is_activated"],
    }
    write_once(args.out, json.dumps(record, indent=2, sort_keys=True, default=str))
    print(json.dumps({"verdict": verdict, "sessions_valid": len(history),
                      "session_ratios": [r["ratio"] for r in history],
                      "scoring": {k: {m: v[m] for m in
                                      ("mean_absolute_error", "coverage_95",
                                       "action_accuracy", "mean_regret", "abstain_rate")}
                                  for k, v in scored.items()},
                      "variance": {k: decomposition.get(k) for k in
                                   ("between_session_sd_total", "mean_within_session_sd",
                                    "tau_sd", "range")}},
                     indent=2, default=str))
    return 0


def _time_to_calibrated(sealed: list[dict]) -> dict:
    """The first session whose sealed primary prediction both named an action and got it right."""
    for record in sealed:
        prediction = record["predictions"][PRIMARY_MODEL]
        truth = record.get("ground_truth")
        if not truth or not truth["valid_for_learning"]:
            continue
        if prediction["predicted_action"] == "ABSTAIN":
            continue
        interval = prediction["prediction_interval"]
        covered = interval[0] <= truth["ratio"] <= interval[1]
        best = "CANDIDATE" if truth["ci_high"] < 1.0 else "REFERENCE"
        if covered and prediction["predicted_action"] == best:
            return {"session": record["session"],
                    "prior_valid_sessions": record["predictions"][PRIMARY_MODEL]
                    ["confidence"]["n_prior_sessions"],
                    "note": ("the first sealed prediction that named an action, named the "
                             "right one, and contained the ground truth in its interval")}
    return {"session": None, "note": "no sealed prediction met all three conditions"}


if __name__ == "__main__":
    raise SystemExit(main())
