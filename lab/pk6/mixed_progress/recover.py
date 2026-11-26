"""Repeat the native comparison with explicitly bound stock fallback after the observed L40S rejection."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from lab.pk6.mixed_progress.trial import ARMS, board, measure
from lab.pk6.replay_http import checksum
from lab.pk6.warm_repeat import save

FROZEN = "7fa0e4d8974c190192f056f6bc5b30dc72b82571"
FIX = "f97a05f74561a9b5bf4dbebcc727d900c108df29"
REFERENCE = {
    "ironmule/ggml_profiles.py": "17d71f1b480cda75540906e3f3fd803db6a60e4d239040e549c7ae1003718c4f",
    "ironmule_product/cli.py": "cc4e8df49dfff0ff8f3d3ae45cb2b658e6754b2dc6f7ec3744edf0d31a739955",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("primary", "out", "models", "state", "vllm"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    read = lambda name: json.loads((args.primary / name).read_bytes())
    target = read("target.json")
    hw = board()
    if (
        target["source"] != FROZEN
        or hw != target["hardware"]
        or read("end-vllm.json")["hardware"] != hw
    ):
        raise ValueError("primary comparator belongs to a different trial")
    if any(checksum(Path(k)) != v for k, v in REFERENCE.items()):
        raise ValueError("incorrect fallback recovery source")
    if checksum(Path("ironmule/ggml_engine.py")) != target["engine_sha256"]:
        raise ValueError("scheduler changed")
    original_wave = args.primary / "waves-vllm.jsonl"
    if (
        checksum(original_wave)
        != "5543ab721ff0579d88319d9d249cb35f5b22b5c5a899115583a765845f818e43"
    ):
        raise ValueError("primary comparator changed")
    if (args.primary / "ready-R1.json").exists():
        raise ValueError(
            "original incumbent already came online; preserve its complete series"
        )
    logs = [
        json.loads(l)["ggml_gate"]
        for l in (args.primary / "serve-R1.log").read_text().splitlines()
        if l.startswith('{"ggml_gate":')
    ]
    if not any(
        v["kernel_mode"] == "generated" and v["diagnostic"]["code"] == "out_of_memory"
        for v in logs
    ):
        raise ValueError("unrelated primary interruption")
    args.out.mkdir(mode=0o700, exist_ok=False)
    for name in (
        "waves-vllm.jsonl",
        "start-vllm.json",
        "ready-vllm.json",
        "end-vllm.json",
        "device-load.csv",
    ):
        shutil.copyfile(args.primary / name, args.out / name)
    save(
        args.out / "target.json",
        {
            **target,
            "stock_retry_revision": FIX,
            "stock_retry_sources": REFERENCE,
            "external_waves_sha256": checksum(original_wave),
        },
    )
    deadline = time.monotonic() + float(os.environ["RP_DEADLINE"]) - time.time() - 35
    with (args.out / "device-load.csv").open("ab") as log:
        timer = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,utilization.gpu,memory.used,power.draw",
                "--format=csv,noheader,nounits",
                "--loop=2",
            ],
            stdout=log,
            env={**os.environ, "TZ": "UTC"},
        )
        try:
            for arm in ARMS[1:]:
                measure(args, arm, hw, target["rows"], deadline)
        finally:
            timer.terminate()
            timer.wait()
    save(args.out / "completed.json", {"hardware": board(), "order": ARMS})


if __name__ == "__main__":
    main()
