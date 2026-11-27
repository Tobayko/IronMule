"""Additional protected checks for a source experiment reducing fresh samples in PR60 calibration."""

from __future__ import annotations

import math
from pathlib import Path
from statistics import median

from lab.pk1.native_trial import checksum
from training.code_author import write_once
from training.trial_release import require

SCHEMA = "ironmule.qwen-calibration-sampling-order.v1"
WORKER = Path(__file__).with_name("sampling_arm.py")
CONTRACT = WORKER.resolve().parents[2] / "training/calibration_sampling_contract.json"
PREFIX = "IRONMULE_CALIBRATION_SAMPLING_"
SCREEN_SEED = 97101
CASES = ("valid_control", "missing", "wrong_pin", "expired", "foreign", "jitter")


def validate_order(order):
    p = order["selected_plan"]
    require(
        order["sampling_contract_sha256"] == checksum(CONTRACT)
        and p["method"] == order["target"]["function"] == "measure_padding"
        and p["reference_warmups"] == 1
        and p["reference_samples"] == 5
        and p["warmups"] in range(4)
        and p["samples"] in range(2, 6)
        and p["warmups"] + p["samples"] < 6
        and p["uncertainty_slack"] == 0.01
        and len(order["blind_results"]) == 2
        and all(v["qualifies_for_experiment"] for v in order["blind_results"]),
        "unvalidated sampling operator or prospective scope",
    )


def profile(order, cache, cache_sha):
    validate_order(order)
    return {
        "schema": "ironmule.qwen-sampling-profile.v1",
        "order_id": order["order_id"],
        "source_sha256": order["source_sha256"],
        "method": order["target"]["function"],
        "selected_plan": order["selected_plan"],
        "parent_cache_sha256": cache_sha,
        "runtime": cache["binding"],
        "created_s": cache["created_s"],
        "expires_s": cache["created_s"] + 86400,
    }


def mounts(directory, data, env, *, negative):
    p = directory / "sampling-profile.json"
    write_once(p, data)
    path = "/kernel/sampling-profile.json"
    result = [(p, path), (WORKER, "/kernel/sampling_arm.py")]
    env.update({PREFIX + "PROFILE": path, PREFIX + "SHA256": checksum(p)})
    if negative:
        for name, obj in (
            (
                "expired",
                {
                    **data,
                    "created_s": data["created_s"] - 172800,
                    "expires_s": data["expires_s"] - 172800,
                },
            ),
            (
                "foreign",
                {
                    **data,
                    "runtime": {
                        **data["runtime"],
                        "threads": data["runtime"]["threads"] + 1,
                    },
                },
            ),
        ):
            p = directory / f"sampling-{name}.json"
            write_once(p, obj)
            path = "/kernel/" + p.name
            env[PREFIX + name.upper()] = path
            result.append((p, path))
    return result


def values(state, row):
    prices = state["prices"]
    n = row["binding"]["max_rows"]
    require(
        set(prices) == {str(i) for i in range(1, n + 1)}
        and all(
            type(v) in (float, int) and math.isfinite(v) and v > 0
            for v in prices.values()
        ),
        "incomplete fresh calibration price vector",
    )
    expected = {}
    for i in range(1, n + 1):
        best = min(range(i, n + 1), key=lambda j: (prices[str(j)], j))
        if best > i and prices[str(best)] < 0.95 * prices[str(i)]:
            expected[str(i)] = best
    require(
        expected == state["dispatch"],
        "deployed dispatch differs from the freshly measured prices",
    )
    return prices


def actual_samples(state, measured, *, adaptive):
    """Recompute the fresh per-width values from witnessed real decode calls."""
    raw = state["native_seconds"]
    prices = state["prices"]
    require(
        set(raw) == set(prices)
        and all(
            len(v) == state["calls"][k]
            and len(v) >= measured
            and all(math.isfinite(t) and t > 0 for t in v)
            for k, v in raw.items()
        ),
        "missing independent native samples",
    )
    intervals = {}
    for width, times in raw.items():
        tail = times[-measured:]
        require(
            abs(median(tail) / prices[width] - 1) <= 0.02,
            "telemetry differs from witnessed native sample median",
        )
        intervals[int(width)] = (0.99 * min(tail), 1.01 * max(tail))
    if adaptive:
        # Independently move every competing width to the adverse edge of its recorded interval.
        for k, (lo, hi) in intervals.items():
            picked = state["dispatch"].get(str(k))
            if picked is None:
                require(
                    all(
                        lower >= 0.95 * hi
                        for m, (lower, _) in intervals.items()
                        if m > k
                    ),
                    "uncertain width was admitted",
                )
            else:
                last = intervals[picked][1]
                require(
                    last < 0.95 * lo
                    and all(
                        last < lower or (last == lower and picked < other)
                        for other, (lower, _) in intervals.items()
                        if other >= k and other != picked
                    ),
                    "uncertain winner among larger widths",
                )


def validate(row, label, plan, profile_sha, *, screen=None):
    entry = row["sampling"]
    require(
        entry["worker_sha256"] == checksum(WORKER) and entry["screen"] == screen,
        "different protected sampling observer or mode",
    )
    actual = entry["observed"]
    state = actual["startup"]
    prices = values(state, row)
    require(
        state["method"] == plan["method"]
        and state["dispatch"] == row["startup_prices"]["padding"],
        "serving used another dispatch",
    )
    if label == "candidate":
        t = state["telemetry"]
        require(
            t
            and t.get("profile_sha256") == profile_sha
            and t.get("mode") in ("full", "adaptive")
            and t.get("reason")
            and t["prices"] == prices
            and set(t["counts"]) == set(prices),
            "missing actual new sampling telemetry",
        )
        adaptive = t["mode"] == "adaptive"
        if adaptive:
            require(
                all(
                    v == [plan["warmups"], plan["samples"]]
                    for v in t["counts"].values()
                ),
                "different fresh warmup/sample counts",
            )
        else:
            require(
                all(sum(v) >= 6 for v in t["counts"].values()),
                "incomplete sampling fallback",
            )
        if screen:
            require(
                {k: sum(v) for k, v in t["counts"].items()} == state["calls"],
                "source telemetry differs from actual calls",
            )
    else:
        require(
            state["telemetry"] is None,
            "unmodified source implemented an unqualified sample strategy",
        )
        if screen:
            require(
                state["calls"] == dict.fromkeys(prices, 6),
                "unmodified reference did not execute 1+5 calls per width",
            )
    if screen:
        short = label == "candidate" and state["telemetry"]["mode"] == "adaptive"
        actual_samples(state, plan["samples"] if short else 5, adaptive=short)
    else:
        require(
            state["native_seconds"] == {} and state["calls"] == {},
            "call profiler active during timed comparison",
        )
    if screen != "candidate":
        require(
            "negative" not in actual, "unexpected diagnostic calls in normal timing"
        )
        return
    require(set(actual["negative"]) == set(CASES), "missing negative sampling checks")
    for name, stage in actual["negative"].items():
        table = values(stage, row)
        t = stage["telemetry"]
        if name == "valid_control":
            require(
                t is not None
                and t["mode"] == "adaptive"
                and t["profile_sha256"] == profile_sha
                and t["prices"] == table
                and stage["calls"]
                == dict.fromkeys(table, plan["warmups"] + plan["samples"]),
                "no matched positive control before scope and jitter withdrawal",
            )
            actual_samples(stage, plan["samples"], adaptive=True)
            continue
        require(
            t is None
            or (
                t.get("mode") == "full" and t.get("reason") and t.get("prices") == table
            ),
            "foreign/unstable plan changed the standard calibration",
        )
        if name == "jitter":
            # An adaptive prefix may run before the fallback repeats the original samples.
            require(
                set(stage["calls"]) == set(table)
                and min(stage["calls"].values()) >= 6
                and t is not None
                and t["mode"] == "full",
                "observed ambiguity did not restore full sampling",
            )
        else:
            actual_samples(stage, 5, adaptive=False)
            require(
                stage["calls"] == dict.fromkeys(table, 6),
                "missing/expired/unbound profile altered original call counts",
            )


def compare(ref, cand):
    a, b = (r["sampling"]["observed"]["startup"] for r in (ref, cand))
    require(
        a["dispatch"] == b["dispatch"] and set(a["prices"]) == set(b["prices"]),
        "different complete calibrated decision",
    )
    error = max(abs(b["prices"][k] / v - 1) for k, v in a["prices"].items())
    require(
        error <= 0.02, "fresh per-width calibration differs by more than two percent"
    )
    return error
