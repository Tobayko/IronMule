#!/usr/bin/env python3
"""What would this model cost on a different device?

Decode at batch one reads every weight per token, so the obvious answer is "scale
by memory bandwidth". That answer is wrong here, and the two local models show why:
the 4B carries 3.9x the weights of the 1B but takes only 1.94x as long. A pure
bandwidth model cannot produce that.

Two terms can. Each layer costs a fixed amount that has nothing to do with weight
size -- dispatch, launch, synchronisation -- and on top of that the weights have to
be read:

    ms_per_token  =  layers * per_layer_ms  +  weight_gigabytes / effective_bandwidth

Both terms are device properties, and they scale differently: bandwidth is a
published number that varies by an order of magnitude between a laptop and a phone,
while per-layer overhead is a property of the scheduler. Separating them is what
makes a projection to another device something better than a guess.

The two parameters are fitted from two configurations and then checked against two
that were held out, because a two-point fit of a two-parameter model is not a
measurement of anything -- it always fits exactly.

Run with --execute; without it nothing is imported or measured.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bench import (  # noqa: E402
    BudgetGuard,
    release_gate,
    require_ac_power,
    resolve_local_model_snapshot,
)

MODELS = {
    "4b": "mlx-community/gemma-3-4b-it-4bit",
    "1b": "mlx-community/gemma-3-1b-it-4bit",
}
CONTEXT_TOKENS = 256
REPETITIONS = 15
# Fit on the full models, hold out the truncated ones. Halving the layers changes
# both terms by different amounts, which is exactly what a wrong model gets wrong.
FIT_FRACTIONS = (1.0,)
HOLDOUT_FRACTIONS = (0.5, 0.25)

DUTY_TARGET = 0.15
BREAK_SECONDS = 4.0
PEAK_BANDWIDTH_GB_S = 400.0  # Published M1 Max figure; not measured here.


def fit_device_model(observations: list[dict]) -> dict[str, float]:
    """Least squares for per-layer cost and effective bandwidth.

    Each observation needs ``layers``, ``weight_gb`` and ``ms``. Two unknowns, so at
    least two observations that differ in more than a scale factor are required --
    otherwise the system is singular and any answer would be arbitrary.
    """

    if len(observations) < 2:
        raise ValueError("need at least two observations to separate two terms")
    xs = [(float(o["layers"]), float(o["weight_gb"]), float(o["ms"])) for o in observations]
    if any(l <= 0 or w <= 0 or m <= 0 for l, w, m in xs):
        raise ValueError("layers, weights and times must all be positive")

    # ms = a*layers + b*weight_gb, solved by normal equations.
    sll = sum(l * l for l, _, _ in xs)
    slw = sum(l * w for l, w, _ in xs)
    sww = sum(w * w for _, w, _ in xs)
    slm = sum(l * m for l, _, m in xs)
    swm = sum(w * m for _, w, m in xs)
    det = sll * sww - slw * slw
    if abs(det) < 1e-12:
        raise ValueError("observations are collinear; the two terms cannot be separated")
    a = (slm * sww - swm * slw) / det
    b = (swm * sll - slm * slw) / det
    if a <= 0 or b <= 0:
        raise ValueError("fit produced a non-physical negative cost")
    return {
        "per_layer_ms": a,
        "ms_per_gigabyte": b,
        "effective_bandwidth_gb_s": 1000.0 / b,
    }


def predict(layers: int, weight_gb: float, model: dict[str, float]) -> float:
    """Milliseconds per token for a given shape under a fitted device model."""

    if layers <= 0 or weight_gb <= 0:
        raise ValueError("layers and weights must be positive")
    return layers * model["per_layer_ms"] + weight_gb * model["ms_per_gigabyte"]


def project(
    layers: int, weight_gb: float, *, per_layer_ms: float, bandwidth_gb_s: float
) -> dict[str, float]:
    """Project to a device described by its own two parameters.

    Both must be supplied. Reusing this machine's per-layer cost with a phone's
    bandwidth would silently assume the schedulers are equally fast, which is the
    kind of assumption that makes a projection look like a measurement.
    """

    if bandwidth_gb_s <= 0 or per_layer_ms <= 0:
        raise ValueError("device parameters must be positive")
    fixed = layers * per_layer_ms
    stream = weight_gb / bandwidth_gb_s * 1000.0
    total = fixed + stream
    return {
        "ms_per_token": total,
        "tokens_per_second": 1000.0 / total,
        "fixed_share": fixed / total,
        "bandwidth_share": stream / total,
    }


def breaks_for(seconds: float, duty_target: float = DUTY_TARGET) -> int:
    if seconds < 0:
        raise ValueError("worked seconds must be non-negative")
    if not 0.0 < duty_target < 1.0:
        raise ValueError("duty target must lie strictly between 0 and 1")
    needed = seconds * (1.0 - duty_target) / duty_target
    return math.ceil(needed / BREAK_SECONDS - 1e-9)


def account(guard: BudgetGuard, seconds: float) -> None:
    guard.record_gpu(seconds)
    for _ in range(breaks_for(seconds)):
        guard.required_break()


def _self_check() -> int:
    checks = 0

    # A model built from known parameters must be recovered exactly.
    truth = {"per_layer_ms": 0.2, "ms_per_gigabyte": 4.0}
    obs = [
        {"layers": 34, "weight_gb": 2.0, "ms": 34 * 0.2 + 2.0 * 4.0},
        {"layers": 26, "weight_gb": 0.5, "ms": 26 * 0.2 + 0.5 * 4.0},
    ]
    fit = fit_device_model(obs)
    assert abs(fit["per_layer_ms"] - 0.2) < 1e-9, fit
    assert abs(fit["ms_per_gigabyte"] - 4.0) < 1e-9, fit
    assert abs(fit["effective_bandwidth_gb_s"] - 250.0) < 1e-6, fit
    checks += 1

    # And it must then predict a shape it never saw.
    assert abs(predict(17, 1.0, fit) - (17 * 0.2 + 4.0)) < 1e-9
    checks += 1

    # Collinear observations cannot separate the terms and must be refused rather
    # than answered: doubling both inputs carries no information about their ratio.
    try:
        fit_device_model([
            {"layers": 10, "weight_gb": 1.0, "ms": 6.0},
            {"layers": 20, "weight_gb": 2.0, "ms": 12.0},
        ])
    except ValueError:
        checks += 1
    else:  # pragma: no cover
        raise AssertionError("a singular system must be refused")

    try:
        fit_device_model([{"layers": 10, "weight_gb": 1.0, "ms": 6.0}])
    except ValueError:
        checks += 1
    else:  # pragma: no cover
        raise AssertionError("one observation cannot fit two parameters")

    # Halving the bandwidth must not halve the time: the fixed term does not move.
    fast = project(34, 2.0, per_layer_ms=0.2, bandwidth_gb_s=400.0)
    slow = project(34, 2.0, per_layer_ms=0.2, bandwidth_gb_s=50.0)
    assert slow["ms_per_token"] > fast["ms_per_token"]
    assert slow["ms_per_token"] < fast["ms_per_token"] * 8, "fixed cost must damp the ratio"
    checks += 1
    # A small model on a slow device becomes bandwidth-bound where it was not before.
    small_fast = project(26, 0.5, per_layer_ms=0.2, bandwidth_gb_s=400.0)
    small_slow = project(26, 0.5, per_layer_ms=0.2, bandwidth_gb_s=50.0)
    assert small_fast["bandwidth_share"] < 0.5 < small_slow["bandwidth_share"], (
        small_fast, small_slow
    )
    checks += 1

    for bad in ((0, 1.0), (10, 0.0)):
        try:
            predict(*bad, model=truth)
        except ValueError:
            checks += 1
        else:  # pragma: no cover
            raise AssertionError(f"invalid shape must be refused: {bad}")
    for bad in ({"per_layer_ms": 0.0, "bandwidth_gb_s": 100.0},
                {"per_layer_ms": 0.2, "bandwidth_gb_s": 0.0}):
        try:
            project(10, 1.0, **bad)
        except ValueError:
            checks += 1
        else:  # pragma: no cover
            raise AssertionError(f"invalid device must be refused: {bad}")

    assert breaks_for(3.0) == 5
    checks += 1

    print(json.dumps({"self_check": "pass", "checks": checks}))
    return 0


def observe(model_key: str, fraction: float, guard: BudgetGuard) -> dict[str, object]:
    """Time one decode step with a fraction of the layers actually in the stack."""

    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    snapshot = resolve_local_model_snapshot(MODELS[model_key])
    model, _ = load(str(snapshot.path))
    inner = model.language_model if hasattr(model, "language_model") else model
    body = inner.model

    full = list(body.layers)
    keep = max(1, int(round(len(full) * fraction)))
    body.layers = full[:keep]

    def weight_gb(module) -> float:
        return sum(p.size * p.dtype.size for _, p in tree_flatten(module.parameters())) / 1e9

    # Everything that runs regardless of depth, plus the layers that remain.
    resident = weight_gb(body)

    cache = make_prompt_cache(body)
    context = mx.array([[1] * CONTEXT_TOKENS])
    primed = body(context, cache=cache)
    mx.eval(primed)
    mx.synchronize()
    step = mx.array([[7]])
    for _ in range(3):
        mx.eval(body(step, cache=cache))
    mx.synchronize()

    samples = []
    started = time.perf_counter()
    for _ in range(REPETITIONS):
        at = time.perf_counter_ns()
        out = body(step, cache=cache)
        mx.eval(out)
        mx.synchronize()
        samples.append(time.perf_counter_ns() - at)
    account(guard, time.perf_counter() - started)

    body.layers = full
    result = {
        "model": model_key,
        "fraction": fraction,
        "layers": keep,
        "layers_total": len(full),
        "weight_gb": round(resident, 6),
        "ms": round(statistics.median(samples) / 1e6, 4),
    }
    del model
    mx.clear_cache()
    return result


def measure(guard: BudgetGuard) -> dict[str, object]:
    fit_points = [observe(m, f, guard) for m in ("4b", "1b") for f in FIT_FRACTIONS]
    held_out = [observe(m, f, guard) for m in ("4b", "1b") for f in HOLDOUT_FRACTIONS]

    model = fit_device_model(fit_points)
    for point in held_out:
        expected = predict(point["layers"], point["weight_gb"], model)
        point["predicted_ms"] = round(expected, 4)
        point["error_percent"] = round((expected - point["ms"]) / point["ms"] * 100.0, 2)

    worst = max(abs(p["error_percent"]) for p in held_out)
    return {
        "fit_points": fit_points,
        "held_out": held_out,
        "device_model": {
            "per_layer_ms": round(model["per_layer_ms"], 5),
            "ms_per_gigabyte": round(model["ms_per_gigabyte"], 5),
            "effective_bandwidth_gb_s": round(model["effective_bandwidth_gb_s"], 1),
            "share_of_peak_bandwidth": round(
                model["effective_bandwidth_gb_s"] / PEAK_BANDWIDTH_GB_S, 4
            ),
        },
        "worst_holdout_error_percent": worst,
        "peak_bandwidth_gb_s": PEAK_BANDWIDTH_GB_S,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    gate = release_gate(args, _self_check)
    if gate is not None:
        return gate

    power = require_ac_power()
    guard = BudgetGuard()
    report = measure(guard)
    report["power_source"] = power
    report["budget"] = guard.summary()
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
