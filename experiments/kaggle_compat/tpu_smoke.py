# PORT2/DATA1 TPU smoke: what a free Kaggle TPU VM v3-8 actually offers, measured.
# Private notebook, internet on. TPU quota, not GPU quota. No performance claim.
#
# IronMule cannot run here and this notebook does not pretend otherwise: MLX has exactly
# two device types, `cpu` and `gpu` (Metal or CUDA), so there is no backend to port. What
# a TPU can answer is the question DATA1 left open — whether the free TPU SKU is reachable
# at all, what it is, and whether the image carries a stack that could decode a real model
# later. Everything below is inventory and a correctness check; the microbenchmark is a
# diagnostic, deliberately reported next to the T4's own numbers and never against them as
# a speed-up.
import json
import os
import subprocess
import sys
import time

WORK = "/kaggle/working"
report = {"schema": "ironmule.tpu-smoke.v1", "performance_claim": False, "stages": {}}


def save():
    with open(f"{WORK}/tpu-smoke-result.json", "w") as stream:
        json.dump(report, stream, indent=1, default=str)


def stage(name, fn):
    started = time.time()
    try:
        report["stages"][name] = {"exit": 0, "value": fn()}
    except BaseException as exc:  # noqa: BLE001 — an absent capability is the result
        report["stages"][name] = {"exit": 1, "error": f"{type(exc).__name__}: {exc}"}
    report["stages"][name]["seconds"] = round(time.time() - started, 2)
    save()
    print(f"== {name}: {json.dumps(report['stages'][name], default=str)[:1200]}", flush=True)


def inventory():
    import jax

    devices = jax.devices()
    rows = []
    for device in devices:
        row = {"repr": str(device), "platform": device.platform,
               "device_kind": str(getattr(device, "device_kind", "unknown")),
               "id": getattr(device, "id", None)}
        try:
            stats = device.memory_stats() or {}
            row["hbm_limit_bytes"] = stats.get("bytes_limit")
            row["hbm_in_use_bytes"] = stats.get("bytes_in_use")
        except Exception as exc:  # noqa: BLE001
            row["memory_stats_error"] = repr(exc)
        rows.append(row)
    return {"jax": jax.__version__, "device_count": len(devices),
            "tpu_devices": sum(d.platform == "tpu" for d in devices), "devices": rows}


def stack():
    frozen = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                            text=True).stdout.splitlines()
    wanted = ("jax", "jaxlib", "libtpu", "torch", "torch-xla", "torch_xla", "flax",
              "transformers", "optax", "orbax", "keras", "tensorflow")
    return {"python": sys.version.split()[0],
            "packages": sorted(line for line in frozen
                               if line.split("==")[0].lower().startswith(wanted))}


def correctness():
    """The DATA1 contract: float32 inputs at the highest available precision, vs numpy."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    rng = np.random.default_rng(20260916)
    left = rng.standard_normal((512, 512), dtype=np.float32)
    right = rng.standard_normal((512, 512), dtype=np.float32)
    expected = left @ right
    device = next(d for d in jax.devices() if d.platform == "tpu")
    a, b = jax.device_put(left, device), jax.device_put(right, device)
    with jax.default_matmul_precision("float32"):
        observed = jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST)
    result = np.asarray(jax.device_get(observed.block_until_ready()), dtype=np.float32)
    return {"shape_exact": result.shape == expected.shape,
            "finite": bool(np.isfinite(result).all()),
            "allclose_rtol_2e-4": bool(np.allclose(result, expected, rtol=2e-4, atol=2e-4)),
            "max_abs_diff": float(np.abs(result - expected).max()),
            "device": str(device)}


def decode_shapes():
    """One decode step's matmuls at Qwen 3 8B's shapes, bf16 and float32, one chip.

    A TPU has no 4-bit path, so this is not comparable to the T4's `quantized_matmul`
    diagnostic and is not presented as if it were. It says what a dense decode step costs
    here, which is what a later real decode would be built out of.
    """
    import jax
    import jax.numpy as jnp

    device = next(d for d in jax.devices() if d.platform == "tpu")
    out = {}
    for dtype_name, dtype in (("bfloat16", jnp.bfloat16), ("float32", jnp.float32)):
        rows = {}
        for label, (m, k, n) in {"qkv_4096x6144": (1, 4096, 6144),
                                 "gate_up_4096x24576": (1, 4096, 24576),
                                 "down_12288x4096": (1, 12288, 4096)}.items():
            x = jnp.ones((m, k), dtype=dtype)
            w = jnp.ones((k, n), dtype=dtype)
            x, w = jax.device_put(x, device), jax.device_put(w, device)
            step = jax.jit(lambda a, b: a @ b)
            step(x, w).block_until_ready()  # compile
            began = time.perf_counter()
            for _ in range(200):
                result = step(x, w)
            result.block_until_ready()
            rows[label] = round((time.perf_counter() - began) / 200 * 1000, 4)
        out[dtype_name] = {"milliseconds_per_matmul": rows}
    return out


def real_decode_feasibility():
    """Can anything here decode a real causal LM, and with which weights?"""
    found = {}
    for module in ("torch_xla", "flax", "transformers", "jax.experimental.pallas"):
        try:
            __import__(module)
            found[module] = True
        except Exception as exc:  # noqa: BLE001
            found[module] = f"{type(exc).__name__}"
    try:
        import mlx.core as mx  # noqa: PLC0415
        found["mlx"] = {"version": mx.__version__,
                        "device_types": [name for name in dir(mx.DeviceType)
                                         if not name.startswith("_")]}
    except Exception as exc:  # noqa: BLE001
        found["mlx"] = f"absent: {type(exc).__name__}"
    return found


os.makedirs(WORK, exist_ok=True)
stage("inventory", inventory)
stage("stack", stack)
stage("correctness", correctness)
stage("decode_shapes", decode_shapes)
stage("real_decode_feasibility", real_decode_feasibility)
report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
