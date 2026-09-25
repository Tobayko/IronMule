"""PORT1-F: why does the tune's confirmation child exit with status 1 after screening?

Usage: python tune_child_probe.py MODEL_ID OUT.json

BACKLOG3 ran the paired confirmation alone for 25 min without a failure, while both full
tunes (BACKLOG1, BACKLOG2) lost confirmation child 0 after screening in the same parent.
This runs the full `ironmule tune` in-process, records each failed child's stderr through
the hook `ab.run` calls on a failure, and samples the GPU's used memory every 10 s. A Kaggle
diagnostic only: the prompt is tune's built-in `DEFAULT_PROMPT`, so no user text reaches
the stderr it keeps.

BACKLOG4 found the cause (CUDA out of memory while the child loaded the model next to the
parent's cache). For the fix's run it also records, when the confirmation starts, the GPU's
used memory and MLX's active and cached bytes in the parent, the candidate and the
confirmation's verdict.
"""
import importlib
import json
import subprocess
import sys
import threading
import time

import mlx.core as mx

ab = importlib.import_module("ironmule.ab")
tune = importlib.import_module("ironmule.tune")  # `ironmule.tune` is also a function

captured = []
original = ab._child_exception_type


def spy(stderr):
    captured.append({"at": time.time(), "stderr": stderr[-20000:]})
    return original(stderr)


ab._child_exception_type = spy
samples = []
confirmations = []
stop = threading.Event()


def used_mib():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None


def sample():
    while not stop.is_set():
        samples.append({"at": time.time(), "memory_used_mib": used_mib()})
        stop.wait(10)


original_confirm = tune.confirm


def confirm_spy(model_id, baseline, candidate, *args, **kwargs):
    entry = {"at": time.time(), "memory_used_mib": used_mib(), "mlx_active_bytes": mx.get_active_memory(),
             "mlx_cache_bytes": mx.get_cache_memory(), "candidate": candidate.as_dict()}
    confirmations.append(entry)
    value = original_confirm(model_id, baseline, candidate, *args, **kwargs)
    entry.update(finished=time.time(), ratio=value.get("ratios", {}).get("candidate/baseline", {}).get("total_ns"),
                 token_identity=value.get("token_identity"), deterministic=value.get("deterministic"))
    return value


tune.confirm = confirm_spy


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
result.update(seconds=time.time() - started, child_stderr=captured, confirmations=confirmations,
              gpu_memory=samples)
with open(out, "w") as stream:
    json.dump(result, stream, indent=1, default=str)
print(json.dumps({"exit": result["exit"], "seconds": round(result["seconds"]), "failed_children": len(captured)}))
for entry in captured:
    print(entry["stderr"][-4000:])
