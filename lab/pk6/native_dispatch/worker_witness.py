"""Witness only the numeric override inherited by the counted B1 production worker, on Linux."""

import json
import time
from pathlib import Path

from lab.pk6.mixed_progress.trial import board
from lab.pk6.warm_repeat import save


def sample(folder):
    start = json.loads((folder / "start-B1.json").read_bytes())
    procs = Path("/proc")
    for candidate in procs.glob("[0-9]*"):
        try:
            argv = (candidate / "cmdline").read_bytes().split(b"\0")
            if not any(a.endswith(b"/ironmule_product/worker.py") for a in argv):
                continue
            env = (candidate / "environ").read_bytes().split(b"\0")
            trace = b"IRONMULE_GGML_TRACE=" + bytes(folder / "trace-B1.txt")
            if trace not in env:
                continue
            threshold = next(
                (v for v in env if v.startswith(b"IRONMULE_CUBLAS_MIN_ROWS=")), b""
            )
            return {
                "process": "ironmule_product/worker.py",
                "pid": int(candidate.name),
                "unix": time.time(),
                "start": start["unix"],
                "hardware": board(),
                "threshold": threshold.partition(b"=")[2].decode(),
                "empty_plugin": b"IRONMULE_KG_LIB=" in env,
            }
        except (FileNotFoundError, ProcessLookupError):
            continue
    return None


if __name__ == "__main__":
    import sys

    folder = Path(sys.argv[1])
    while not (folder / "end-B1.json").exists():
        if (folder / "ready-B1.json").exists():
            result = sample(folder)
            if result:
                save(folder / "worker-B1.json", result)
                break
        time.sleep(0.4)
