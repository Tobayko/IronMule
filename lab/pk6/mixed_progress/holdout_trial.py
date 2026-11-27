"""Repeat the pool admission experiment on fresh requests, then screen an interpolated budget."""

import argparse
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time

from ironmule.ggml_profiles import FLAGS
from lab.pk6.mixed_progress import grade, trial
from lab.pk6.mixed_progress.pool_trial import ENGINE
from lab.pk6.replay_http import checksum
from lab.pk6.warm_repeat import save

SEED = 723449


def predicted_caps(primary, pool, current):
    """Use the original seeds only. The model interpolates a latency, not a serving benefit."""

    def pause(folder, arm, seed):
        values = [
            grade.measure(w)["gap_p95"]
            for w in grade.lines(folder / f"waves-{arm}.jsonl")
            if w["seed"] == seed and w["width"] == 32 and w["repeat"] < 3
        ]
        if len(values) != 3:
            raise ValueError("incomplete calibration block")
        return statistics.median(values)

    inputs = [
        (
            pause(primary, f"R{i}", seed),
            pause(pool, f"S{i}", seed),
            pause(current, "vllm", seed),
        )
        for i, seed in enumerate(grade.SEEDS, 1)
    ]
    forecasts = {
        str(cap): [
            (baseline + (candidate - baseline) * (cap - 128) / (2048 - 128)) / external
            for baseline, candidate, external in inputs
        ]
        for cap in (768, 1024, 1280, 1536)
    }
    feasible = [int(cap) for cap, ratios in forecasts.items() if max(ratios) <= 0.9]
    return max(feasible, default=None), forecasts


def balanced_engine(source, cap):
    replacements = {
        "and not PROMPT_BUDGET and len(pending) >= 4": "and not PROMPT_BUDGET and len(pending) >= 12",
        "self.batch_cap - len(active), 2048)": f"self.batch_cap - len(active), {cap})",
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError("unexpected engine source for the balanced variant")
        source = source.replace(old, new)
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "primary",
        "out",
        "models",
        "state",
        "pool",
        "current",
        "balanced",
        "checkout",
    ):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    for key, value in vars(args).items():
        setattr(args, key, value.resolve())
    args.out.mkdir(mode=0o700, exist_ok=False)
    base = json.loads((args.primary / "target.json").read_bytes())
    hw = trial.board()
    engine = Path("ironmule/ggml_engine.py")
    if hw != base["hardware"] or checksum(engine) != ENGINE:
        raise ValueError("holdout requires the original board and pool candidate")
    trial.SEEDS = (SEED,)
    order = ["H1", "B1"]
    save(
        args.out / "target.json",
        {**base, "seed": SEED, "pool_engine_sha256": ENGINE, "order": order},
    )
    deadline = time.monotonic() + float(os.environ["RP_DEADLINE"]) - time.time() - 40
    os.environ.update(
        IRONMULE_KG="0",
        IRONMULE_KG_LIB="",
        IRONMULE_GGML_PACK="1",
        IRONMULE_GGML_UBATCH="8192",
        IRONMULE_CUDA_GRAPHS_ROWS="32",
    )
    for arm in order:
        for flag in FLAGS:
            os.environ.pop(flag, None)
        os.environ["IRONMULE_LAB_POOL_BURST"] = "1" if arm == "H1" else "0"
        trial.measure(args, arm, hw, 0, deadline)
    save(args.out / "completed.json", {"hardware": hw, "order": order})
    if float(os.environ["RP_DEADLINE"]) - time.time() <= 190:
        return
    cap, estimates = predicted_caps(args.primary, args.pool, args.current)
    if cap is None:
        return
    args.balanced.mkdir(mode=0o700, exist_ok=False)
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(args.checkout), "HEAD"], check=True
    )
    for name in (str(engine), "ironmule/ggml_profiles.py", "ironmule_product/cli.py"):
        shutil.copyfile(name, args.checkout / name)
    engine = args.checkout / engine
    engine.write_text(balanced_engine(engine.read_text(), cap))
    save(
        args.balanced / "target.json",
        {
            "hardware": hw,
            "reference_seed": SEED,
            "cap": cap,
            "min_queue": 12,
            "engine_sha256": checksum(engine),
            "simulated_gap_ratios": estimates,
            "external": "0.34.0",
        },
    )
    args.out = args.balanced
    os.chdir(args.checkout)
    os.environ.update(PYTHONPATH=str(args.checkout), IRONMULE_LAB_POOL_BURST="1")
    for flag in FLAGS:
        os.environ.pop(flag, None)
    trial.measure(args, "L1", hw, 0, deadline)
    save(args.out / "completed.json", {"hardware": trial.board(), "order": ["L1"]})


if __name__ == "__main__":
    main()
