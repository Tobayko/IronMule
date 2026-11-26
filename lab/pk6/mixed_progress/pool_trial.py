"""Measure the shared-pool admission follow-up in two fresh product processes on the primary board."""

import argparse
import json
import os
from pathlib import Path
import time

from lab.pk6.mixed_progress.trial import board, measure
from lab.pk6.replay_http import checksum
from lab.pk6.warm_repeat import save

ENGINE = "cc7a2467f00c138d8cfde890a0e025d7bd421c8ad5e9dd3de98c08f4bd9f1b9d"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("primary", "models", "state", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    args = p.parse_args()
    target = json.loads((args.primary / "target.json").read_bytes())
    hw = board()
    if hw != target["hardware"] or checksum(Path("ironmule/ggml_engine.py")) != ENGINE:
        raise ValueError("different board or experimental code")
    for arm in ("R1", "R2"):
        ready = json.loads((args.primary / f"ready-{arm}.json").read_bytes())["health"]
        if ready.get("kernel_fallback", {}).get("serving") != "stock" or not ready.get(
            "shared_pool"
        ):
            raise ValueError(
                "the registered native reference does not serve shared stock"
            )
    # Reuse precisely the already qualified stock configuration, bypassing the observed doomed W4 attempt.
    os.environ.update(
        IRONMULE_KG="0",
        IRONMULE_KG_LIB="",
        IRONMULE_GGML_PACK="1",
        IRONMULE_GGML_UBATCH="8192",
        IRONMULE_CUDA_GRAPHS_ROWS="32",
        IRONMULE_LAB_POOL_BURST="1",
    )
    args.out.mkdir(mode=0o700, exist_ok=False)
    save(args.out / "target.json", {**target, "pool_engine_sha256": ENGINE})
    deadline = time.monotonic() + float(os.environ["RP_DEADLINE"]) - time.time() - 45
    from ironmule.ggml_profiles import FLAGS

    for arm in ("S1", "S2"):
        # The previous arm's client tokenizer can import raw defaults into the parent environment.
        # Restore the reference's own observed product profile before launching this arm.
        profile = json.loads((args.primary / f"ready-R{arm[1]}.json").read_bytes())[
            "health"
        ]["kernel_profile"]
        for key in FLAGS:
            os.environ.pop(key, None)
        if profile["profile"] == "explicit":
            os.environ.update(profile["flags"])
        elif profile["profile"] != "conservative":
            raise ValueError("unregistered reference profile")
        measure(args, arm, hw, 0, deadline)
    save(args.out / "completed.json", {"hardware": board(), "order": ["S1", "S2"]})


if __name__ == "__main__":
    main()
