"""Run complete stock, candidate and comparator processes against fixed rendered requests."""

import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import time

from ironmule.ggml_profiles import FLAGS
from lab.pk6.mixed_progress import trial
from lab.pk6.replay_http import checksum
from lab.pk6.warm_repeat import GGUF, save
from lab.portable_gpu.fetch_awq import verify

SEEDS = (180341, 943819)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field, default in (
        ("models", "/models"),
        ("state", "/work/state"),
        ("vllm", "/opt/vllm034"),
    ):
        parser.add_argument("--" + field, type=Path, default=Path(default))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--lib", type=Path)
    parser.add_argument("--threshold", type=int, default=128)
    parser.add_argument("--seeds", type=int, nargs=2, default=SEEDS)
    args = parser.parse_args()
    args.out.mkdir(mode=0o700, parents=True, exist_ok=False)
    hw = trial.board()
    gguf = args.models / trial.GGUF_NAME
    if checksum(gguf) != GGUF:
        raise ValueError("unexpected GGUF")
    awq = verify(args.models / "qwen3-32b-awq")
    trial.SEEDS = tuple(args.seeds)
    version = subprocess.check_output(
        [
            str(args.vllm / "bin/python"),
            "-c",
            "import importlib.metadata as m; print(m.version('vllm'))",
        ],
        text=True,
    ).strip()
    if version != "0.34.0":
        raise ValueError("unexpected comparator")
    save(
        args.out / "target.json",
        {
            "hardware": hw,
            "gguf_sha256": GGUF,
            "awq_manifest_sha256": awq["manifest_sha256"],
            "seeds": trial.SEEDS,
            "order": args.arms,
            "vllm_version": version,
            "source": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "engine_sha256": checksum(Path("ironmule/ggml_engine.py")),
            "library_sha256": checksum(args.lib) if args.lib else None,
            "threshold": args.threshold,
        },
    )
    deadline = time.monotonic() + float(os.environ["RP_DEADLINE"]) - time.time() - 45
    for arm in args.arms:
        for flag in (
            *FLAGS,
            "IRONMULE_LAB_ACTIVITY",
            "IRONMULE_CUBLAS_MIN_ROWS",
            "IRONMULE_LAB_FP8",
            "IRONMULE_LAB_SHARED_PROFILE",
            "IRONMULE_LAB_STREAM_ALTERNATIVE",
            "IRONMULE_LAB_KERNEL_TRACE",
        ):
            os.environ.pop(flag, None)
        os.environ.update(
            IRONMULE_KG="0",
            IRONMULE_KG_LIB="",
            IRONMULE_GGML_PACK="1",
            IRONMULE_GGML_UBATCH="8192",
            IRONMULE_CUDA_GRAPHS_ROWS="32",
        )
        if arm.startswith(("N", "G")):
            os.environ["IRONMULE_LAB_STREAM_ALTERNATIVE"] = "plain"
            if arm.startswith("G"):
                os.environ.update(IRONMULE_PS1="1", IRONMULE_PK6="1")
        if arm.startswith(("U", "V")):
            os.environ["IRONMULE_LAB_SHARED_PROFILE"] = "wide"
        if arm.startswith("D"):
            os.environ["IRONMULE_LAB_ACTIVITY"] = str(args.out)
        elif arm.startswith("P"):
            os.environ["IRONMULE_CUBLAS_MIN_ROWS"] = str(args.threshold)
        elif arm.startswith(("F", "V")):
            if not args.lib:
                raise ValueError("missing measured kernel library")
            os.environ.update(
                IRONMULE_LAB_KERNEL_TRACE="1",
                IRONMULE_KG_LIB=str(args.lib.resolve()),
                IRONMULE_LAB_FP8=str(args.threshold),
            )
        elif arm != "vllm" and not arm.startswith(("R", "U", "N", "G")):
            raise ValueError("unregistered arm")
        trial.measure(args, arm, hw, 0, deadline)
    save(args.out / "completed.json", {"hardware": trial.board(), "order": args.arms})


if __name__ == "__main__":
    with Path(
        os.environ.get("IRONMULE_LAB_GPU_LOCK", "/work/native-dispatch.lock")
    ).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        main()
