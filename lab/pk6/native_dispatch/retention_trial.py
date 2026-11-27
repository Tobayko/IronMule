"""Compare the existing finished-prefix retention limit with stock in the product server."""

import argparse
import os
import threading
import time
from pathlib import Path

from ironmule.ggml_profiles import FLAGS
from ironmule_product.state import ProductStore
from lab.pk6.mixed_progress import trial
from lab.pk6.warm_repeat import save


def witness(root, arms, stop):
    """Save the retention fraction inherited by each counted production worker."""
    while not stop.is_set():
        for arm in arms:
            directory = root / arm
            receipt = directory / f"worker-{arm}.json"
            if (
                receipt.exists()
                or not (directory / f"ready-{arm}.json").exists()
                or (directory / f"end-{arm}.json").exists()
            ):
                continue
            trace = b"IRONMULE_GGML_TRACE=" + bytes(directory / f"trace-{arm}.txt")
            for proc in Path("/proc").glob("[0-9]*"):
                try:
                    argv = (proc / "cmdline").read_bytes().split(b"\0")
                    if not any(
                        x.endswith(b"/ironmule_product/worker.py") for x in argv
                    ):
                        continue
                    env = (proc / "environ").read_bytes().split(b"\0")
                    if trace not in env:
                        continue
                    keep = next(x for x in env if x.startswith(b"IRONMULE_GGML_KEEP="))
                    save(
                        receipt,
                        {
                            "process": "ironmule_product/worker.py",
                            "arm": arm,
                            "unix": time.time(),
                            "finished_fraction": float(keep.partition(b"=")[2]),
                        },
                    )
                    break
                except (FileNotFoundError, ProcessLookupError, PermissionError):
                    continue
        stop.wait(1)


def run(args):
    board = trial.board()
    ProductStore(args.state).setup("server")
    trial.SEEDS = (180341, 943819)
    root = args.out
    root.mkdir(parents=True, exist_ok=False)
    stop = threading.Event()
    monitor = threading.Thread(target=witness, args=(root, args.arms, stop), daemon=True)
    monitor.start()
    settings = {key: os.environ.get(key) for key in FLAGS}
    for arm in args.arms:
        # The benchmark client's native tokenizer may set PS1 and PK6 in this process.
        for key, value in settings.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        keep = "0.03125" if arm.startswith("H") else "1"
        args.out = root / arm
        args.out.mkdir()
        os.environ["IRONMULE_GGML_KEEP"] = keep
        save(
            args.out / "plan.json",
            {
                "hardware": board,
                "arm": arm,
                "finished_fraction": float(keep),
                "seed": trial.SEEDS[int(arm[1]) - 1],
            },
        )
        deadline = (
            time.monotonic() + float(os.environ["RP_DEADLINE"]) - time.time() - 45
        )
        trial.measure(args, arm, board, 0, deadline)
        print(arm, "finished", flush=True)
    stop.set()
    monitor.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("models", "state", "out"):
        parser.add_argument("--" + flag, required=True, type=Path)
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=("R1", "H1", "H2", "R2"),
        default=("R1", "H1", "H2", "R2"),
    )
    run(parser.parse_args())
