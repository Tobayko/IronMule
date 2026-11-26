"""Score the reverse-order pool holdout and the discovery-selected intermediate budget."""

import argparse
import gzip
import json
from pathlib import Path

from lab.pk6.mixed_progress import grade, pool_score
from lab.pk6.mixed_progress.holdout_trial import SEED, predicted_caps
from lab.pk6.mixed_progress.pool_trial import ENGINE

BALANCED_ENGINE = "4b6b95dc1169d44489a1508fa3ea0365edd696ea860c32c6397c285f3da7c1f8"


def read(folder, name):
    return json.loads((folder / name).read_bytes())


def width_block(waves, width):
    return [wave for wave in waves if wave["repeat"] < 3 and wave["width"] == width]


def snapshot(folder, arm, board, reference, after, cap):
    waves = grade.lines(folder / f"waves-{arm}.jsonl")
    start, end = (read(folder, f"{name}-{arm}.json") for name in ("start", "end"))
    health = read(folder, f"ready-{arm}.json")["health"]
    if (
        start["hardware"] != board
        or end["hardware"] != board
        or not after <= start["unix"] < end["unix"]
        or health.get("kernel_generation", {}).get("status") != "disabled"
        or health["kernel_profile"] != reference["kernel_profile"]
        or health["sequences"]["sequences"] != reference["sequences"]["sequences"]
        or health["shared_pool"]["cap"] != reference["shared_pool"]["cap"]
        or health["gate"]["passes"] is not True
    ):
        raise ValueError("hardware, ordering or serving profile differs")
    expected = [(r, w) for r in range(4) for w in ((1, 8, 32) if r < 3 else (32,))]
    if [(w["repeat"], w["width"]) for w in waves] != expected:
        raise ValueError("incomplete holdout")
    for wave in waves:
        grade.measure(wave)
        if (
            wave["seed"] != SEED
            or wave["arm"] != arm
            or not start["unix"] <= wave["unix_start"] < wave["unix_end"] <= end["unix"]
        ):
            raise ValueError("foreign or out-of-window holdout wave")
    receipts = grade.bindings(folder, arm, waves)
    trace = folder / f"trace-{arm}.txt"
    raw = (
        trace.read_text()
        if trace.exists()
        else gzip.decompress(trace.with_suffix(".txt.gz").read_bytes()).decode()
    )
    marked = pool_score.markers(raw, waves)
    for line in raw.splitlines():
        if line.startswith("# shared_budget "):
            _, _, _, budget, queued, _ = line.split()
            if (
                not cap
                or int(budget.split("=")[1]) > cap
                or int(queued.split("=")[1]) < (12 if arm == "L1" else 4)
            ):
                raise ValueError("unexpected budget or queue depth")
    if cap and not all(
        c["changes"] > 0 for c in marked if c["width"] == 32 and c["repeat"] < 3
    ):
        raise ValueError("candidate never exercised in a counted high-load wave")
    facts = waves[-1]["requests"]
    stats = {
        "trace_sha256": receipts["trace_sha256"],
        "waves_sha256": receipts["waves_sha256"],
        "quality": [sum(r["correct"] is True for r in facts), len(facts)],
        "by_width": {
            str(w): grade.summarize([grade.measure(v) for v in width_block(waves, w)])
            for w in (1, 8, 32)
        },
        "changes": marked,
    }
    return waves, stats, end["unix"]


def score(root, *, balanced=False):
    primary = root / "l40s-mixed-prefill-progress"
    pool = root / "l40s-pool-admission"
    fresh = root / "l40s-pool-holdout"
    base = read(primary, "target.json")
    board = base["hardware"]
    reference = read(primary, "ready-R1.json")["health"]
    if (
        SEED in grade.SEEDS
        or read(fresh, "target.json")
        != {**base, "seed": SEED, "pool_engine_sha256": ENGINE, "order": ["H1", "B1"]}
        or read(fresh, "completed.json") != {"hardware": board, "order": ["H1", "B1"]}
    ):
        raise ValueError("foreign holdout or incomplete rollback")
    arms = [(fresh, "H1", 2048), (fresh, "B1", 0)]
    selection = None
    if balanced:
        extra = root / "l40s-pool-balanced"
        cap, forecasts = predicted_caps(primary, pool, root / "l40s-vllm-034")
        selection = read(extra, "target.json")
        if selection != {
            "hardware": board,
            "reference_seed": SEED,
            "engine_sha256": BALANCED_ENGINE,
            "cap": cap,
            "min_queue": 12,
            "simulated_gap_ratios": forecasts,
            "external": "0.34.0",
        } or read(extra, "completed.json") != {"hardware": board, "order": ["L1"]}:
            raise ValueError("changed discovery-only selection, engine or hardware")
        arms.append((extra, "L1", cap))
    waves, stats = {}, {}
    after = read(pool, "end-S2.json")["unix"]
    for folder, arm, cap in arms:
        waves[arm], stats[arm], after = snapshot(
            folder, arm, board, reference, after, cap
        )

    arm = "L1" if balanced else "H1"

    def factual_inputs(tag):
        wave = waves[tag][-1]
        return wave["input_digest"], [
            (r["case"], r["prompt_tokens"]) for r in wave["requests"]
        ]

    if factual_inputs(arm) != factual_inputs("B1"):
        raise ValueError("different graded questions")
    ratios = {
        str(w): grade.paired(width_block(waves["B1"], w), width_block(waves[arm], w))
        for w in (1, 8, 32)
    }
    return {
        "hardware": board,
        "seed": SEED,
        "selection": selection,
        "reference": "B1",
        "candidate": arm,
        "starts": {name: stats[name] for name in ("B1", arm)},
        "paired": ratios,
        "rollback_served_correctly": stats["B1"]["quality"] == [32, 32]
        and not any(c["changes"] for c in stats["B1"]["changes"]),
        "promotion": False,
        "scope": "One L40S and one fresh seed. Paired ratios compare the candidate with restored stock. All stream metrics count.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--balanced", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(score(args.root, balanced=args.balanced), indent=2, allow_nan=False)
    )
