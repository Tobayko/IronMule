"""PORT1-F: what still holds the parent's GPU memory after tune releases MLX's cache?

Usage: python pool_probe.py MODEL_ID OUT.json

BACKLOG5: after `_release_device_memory()` MLX reported 56 active and 0 cached bytes in the
parent, and nvidia-smi still counted 9553 MiB on its card, so the confirmation child ran out
of memory again. MLX 0.32.2 frees its buffers with `cudaFreeAsync` on a separate stream into
the device's default memory pool, and `mx.synchronize()` synchronizes only MLX's own stream.
This loads the model as tune does, generates as screening does, releases it as tune now does,
and then records the used memory, the pool's reserved and used bytes and MLX's counters after
each further step: a context-wide synchronize, then trimming the pool to zero. Last, a child
process loads the same model. The CUDA driver is called through ctypes, in the process MLX
already initialised; a diagnostic only, not product code.
"""
import ctypes
import importlib
import json
import subprocess
import sys
import time

import mlx.core as mx

import ironmule

tune = importlib.import_module("ironmule.tune")  # `ironmule.tune` is also a function

# CUmemPool_attribute values (CUDA 12 driver API). MLX links the CUDA runtime statically, so
# the probe uses the driver: the default memory pool and the primary context are driver
# objects, the same ones MLX's runtime uses.
RELEASE_THRESHOLD, RESERVED_CURRENT, USED_CURRENT = 4, 5, 7
cuda = ctypes.CDLL("libcuda.so.1")
device, context, pool = ctypes.c_int(), ctypes.c_void_p(), ctypes.c_void_p()


def check(code, name):
    if code:
        raise RuntimeError(f"{name} returned CUDA error {code}")


check(cuda.cuInit(0), "cuInit")
check(cuda.cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet")
check(cuda.cuDevicePrimaryCtxRetain(ctypes.byref(context), device), "cuDevicePrimaryCtxRetain")
check(cuda.cuCtxPushCurrent_v2(context), "cuCtxPushCurrent")
check(cuda.cuDeviceGetDefaultMemPool(ctypes.byref(pool), device), "cuDeviceGetDefaultMemPool")


def state(label):
    values = {}
    for name, attr in (("pool_release_threshold", RELEASE_THRESHOLD), ("pool_reserved_bytes", RESERVED_CURRENT),
                       ("pool_used_bytes", USED_CURRENT)):
        value = ctypes.c_uint64()
        check(cuda.cuMemPoolGetAttribute(pool, attr, ctypes.byref(value)), "cuMemPoolGetAttribute")
        values[name] = value.value
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    check(cuda.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total)), "cuMemGetInfo")
    smi = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=30).stdout.strip()
    row = {"step": label, "at": time.time(), "nvidia_smi_used_mib": smi, "device_free_bytes": free.value,
           "mlx_active_bytes": mx.get_active_memory(), "mlx_cache_bytes": mx.get_cache_memory(), **values}
    print(json.dumps(row), flush=True)
    return row


model_id, out = sys.argv[1:3]
rows = [state("start")]
engine, tokenizer = tune.load_engine(model_id, ironmule.BASELINE)
ids = tune.prompt_ids(tokenizer, tune.DEFAULT_PROMPT)
tune.measure(engine, ids, 32, tune._eos_ids(tokenizer), repeats=2, warmup=1)
rows.append(state("after_generate"))
tune._close_engine(engine)
del engine
tune._release_device_memory()
rows.append(state("after_release"))
check(cuda.cuCtxSynchronize(), "cuCtxSynchronize")
rows.append(state("after_context_synchronize"))
check(cuda.cuMemPoolTrimTo(pool, ctypes.c_size_t(0)), "cuMemPoolTrimTo")
rows.append(state("after_trim"))
child = subprocess.run([sys.executable, "-c", "import sys, mlx.core as mx; from ironmule.tune import load_engine; "
                        "import ironmule; load_engine(sys.argv[1], ironmule.BASELINE); "
                        "print(mx.get_active_memory())", model_id],
                       capture_output=True, text=True, timeout=900)
rows.append(state("after_child"))
report = {"model_id": model_id, "rows": rows, "child_exit": child.returncode,
          "child_tail": child.stderr.strip().splitlines()[-1:] if child.returncode else child.stdout.strip()[-200:],
          "performance_claim": False}
with open(out, "w") as stream:
    json.dump(report, stream, indent=1)
print(json.dumps({"child_exit": child.returncode}))
