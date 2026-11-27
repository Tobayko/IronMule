"""Supplement the frozen PR 57 comparator with a separate current release on the same lease."""

import argparse
import json
import os
from pathlib import Path
import time

from lab.pk6.mixed_progress.trial import board, measure
from lab.pk6.warm_repeat import output, save

VERSION = "0.34.0"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("models", "out", "vllm", "primary"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    complete = json.loads((args.primary / "completed.json").read_bytes())
    target = json.loads((args.primary / "target.json").read_bytes())
    hw = board()
    if hw != target["hardware"] or complete["hardware"] != hw:
        raise ValueError("different hardware from the completed primary comparison")
    versions = json.loads(
        output(
            str(args.vllm / "bin/python"),
            "-c",
            "import importlib.metadata as m,json;print(json.dumps({k:m.version(k) for k in ('vllm','torch','transformers')}))",
        )
    )
    if versions["vllm"] != VERSION:
        raise ValueError("supplement requires the pinned current release")
    # The mounted checkpoint must still have the original verified files before this separate server loads it.
    from lab.portable_gpu.fetch_awq import verify

    if (
        verify(args.models / "qwen3-32b-awq")["manifest_sha256"]
        != target["awq_manifest_sha256"]
    ):
        raise ValueError("current competitor checkpoint changed")
    args.out.mkdir(mode=0o700, parents=True, exist_ok=False)
    save(
        args.out / "target.json",
        {**target, "vllm_version": VERSION, "versions": versions},
    )
    deadline = time.monotonic() + float(os.environ["RP_DEADLINE"]) - time.time() - 35
    measure(args, "vllm", hw, 0, deadline)
    save(args.out / "completed.json", {"hardware": hw, "version": VERSION})


if __name__ == "__main__":
    main()
