"""Prioritize one remaining kernel from target measurements and run its complete model qualification."""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import median

from lab.pk6.mixed_progress.trial import board
from lab.pk6.replay_http import checksum
from lab.pk6.warm_repeat import GGUF, save

CANDIDATES = {
    "marlin-check": "M",
    "int8-check": "I",
    "cached-check": "J",
    "half-check": "Y",
    "kernel-check": "F",
}


def shortlist(evidence):
    operators = evidence / "operator-calibration"
    target = json.loads((operators / "completed.json").read_bytes())
    previously = json.loads((evidence / "comparison.json").read_bytes())["arms"]
    if target["gguf"] != GGUF:
        raise ValueError("foreign model")
    references = {}
    for file in operators.glob("shape-*.json"):
        observation = json.loads(file.read_bytes())
        for width, stats in observation["cells"].items():
            key = (observation["type"], observation["n"], observation["k"], int(width))
            references.setdefault(key, {})[observation["seed"]] = stats["stock"]["us"]
    result = []
    for folder, arm in CANDIDATES.items():
        check = evidence / folder / "result.json"
        if not check.exists() or any(run.startswith(arm) for run in previously):
            continue
        sampled = json.loads(check.read_bytes())
        if sampled["hardware"] != target["hardware"] or sampled["weights"] != GGUF:
            raise ValueError("the arithmetic test belongs to a different target")
        cells = []
        for c in sampled["cells"]:
            key = ({"Q4_K": 12, "Q6_K": 14}[c["type"]], c["n"], c["k"], c["b"])
            times = references.get(key, {})
            if (
                c["b"] not in (143, 448)
                or set(times) != set(target["seeds"])
                or c["rc"]
                or not c["finite"]
                or c["relative_rms"] >= 0.04
                or c["native_launches"] < 8
            ):
                continue
            ours = 1000 * c["median_ms"]
            if not 0 < ours < min(times.values()) or not math.isfinite(ours):
                continue
            cells.append(
                {
                    "shape": key,
                    "estimated_saved_us": min(times.values()) - ours,
                    "warm_device_us": ours,
                    "native_graph_us": median(times.values()),
                }
            )
        if cells:
            result.append(
                {
                    "arm": arm,
                    "family": folder,
                    "source_sha256": checksum(check),
                    "library_sha256": sampled["library_sha256"],
                    "cells": cells,
                    "estimated_saved_us": sum(c["estimated_saved_us"] for c in cells),
                }
            )
    result.sort(key=lambda candidate: candidate["estimated_saved_us"], reverse=True)
    return {
        "hardware": target["hardware"],
        "native": target["ggml_cuda"],
        "candidates": result,
        "interpretation": "Ranking hypothesis. Warm CUDA arithmetic and full GGML graph timing have different boundaries. Only matched full-server measurements determine benefit.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("evidence", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--library", action="append", default=[], metavar="FAMILY=FILE")
    p.add_argument("--max-seconds", type=int, default=1800)
    a = p.parse_args()
    started = time.time()
    plan = shortlist(a.evidence)
    a.out.mkdir(parents=True, exist_ok=False)
    save(a.out / "ranking.json", plan)
    if not a.library or not plan["candidates"]:
        return
    available = {k: Path(v).resolve() for k, v in (s.split("=", 1) for s in a.library)}
    trial = next((c for c in plan["candidates"] if c["family"] in available), None)
    if trial is None:
        raise ValueError("no screened candidate library supplied")
    lib = available[trial["family"]]
    if (
        board() != plan["hardware"]
        or checksum(lib) != trial["library_sha256"]
        or checksum(Path(os.environ["LLAMA_CPP_LIB_PATH"]) / "libggml-cuda.so")
        != plan["native"]
    ):
        raise ValueError("loaded target or kernel differs from its arithmetic receipt")
    if a.max_seconds <= 45:
        raise ValueError("insufficient measurement window")
    deadline = min(int(os.environ["RP_DEADLINE"]), int(started) + a.max_seconds)
    remaining = deadline - time.time()
    if remaining <= 45:
        raise ValueError("GPU reservation has expired")
    driver = Path(__file__).with_name("serve.sh")
    arms = [trial["arm"] + str(i) for i in (1, 2)]
    save(
        a.out / "selected.json",
        {"arm": trial["arm"], "library": trial["library_sha256"], "deadline": deadline},
    )
    subprocess.run(
        ["bash", driver, a.out.resolve() / "http", "--lib", lib, "--arms", *arms],
        env={**os.environ, "RP_DEADLINE": str(deadline)},
        check=True,
    )


if __name__ == "__main__":
    main()
