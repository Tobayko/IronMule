"""PORT1-F: why does the tune's confirmation child exit with status 1 after screening?

Usage: python tune_child_probe.py MODEL_ID OUT.json

BACKLOG3 ran the paired confirmation alone for 25 min without a failure, while both full
tunes (BACKLOG1, BACKLOG2) lost confirmation child 0 after screening in the same parent.
This runs the full `ironmule tune` in-process, records each failed child's stderr through
the hook `ab.run` calls on a failure, and samples the GPU's used memory every 10 s. A Kaggle
diagnostic only: the prompt is tune's built-in `DEFAULT_PROMPT`, so no user text reaches
the stderr it keeps.
"""
import importlib
import json
import subprocess
import sys
import threading
import time

ab = importlib.import_module("ironmule.ab")
tune = importlib.import_module("ironmule.tune")  # `ironmule.tune` is also a function

captured = []
original = ab._child_exception_type


def spy(stderr):
    captured.append({"at": time.time(), "stderr": stderr[-20000:]})
    return original(stderr)


ab._child_exception_type = spy
samples = []
stop = threading.Event()


def sample():
    while not stop.is_set():
        try:
            used = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            used = None
        samples.append({"at": time.time(), "memory_used_mib": used})
        stop.wait(10)


model_id, out = sys.argv[1:3]
started = time.time()
thread = threading.Thread(target=sample, daemon=True)
thread.start()
result = {"model_id": model_id, "started": started}
try:
    result["exit"] = tune.main(["--model", model_id])
except BaseException as exc:  # noqa: BLE001 - the failure is what is being recorded
    result["exit"] = f"{type(exc).__name__}: {exc}"
stop.set()
thread.join(timeout=15)
result.update(seconds=time.time() - started, child_stderr=captured, gpu_memory=samples)
with open(out, "w") as stream:
    json.dump(result, stream, indent=1)
print(json.dumps({"exit": result["exit"], "seconds": round(result["seconds"]), "failed_children": len(captured)}))
for entry in captured:
    print(entry["stderr"][-4000:])
