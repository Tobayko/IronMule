# PERF1 run 15: the MoE expert kernel through the product, not the harness. Private notebook,
# internet on. Free quota: one run, <= 25 min (the week stands at about 9.2 h; it stays under 10).
#
# Run 14 (`perf1-run14-0f10c1f8`) measured the gather kernel through `perf1.py`'s global patch
# with free float32 rounding: Qwen3.6 35B-A3B decode 1.97x its `kernel+p16` control over two
# cards, Gemma 4 26B-A4B 7.86x its `kernel` control on one. `ironmule/cuda_native.py` now swaps
# mlx-lm's `QuantizedSwitchLinear` too (the `native` plan, pinned rounding, probe-gated on the
# model's own first expert weight), carried as a patch against the pinned commit. That source
# has not run on CUDA. Rules fixed here:
#   * device check first: `_probe` on random expert weights at both models' shapes and
#     `gather_matmul` at 64 sorted rows and at a 2048-row prefill, each against float32, at most
#     1e-2, else the model stage is skipped; the model-free test file runs as well;
#   * `cross.py` on Gemma 4 26B-A4B (one card), stock against `ironmule_native`, one repetition,
#     one measured pass: the product answers and its wall ratio. Screening, no quality claim.
import json
import os
import signal
import subprocess
import sys
import time

COMMIT = "b6886a0823f2da14a1a02ea8a3176660786a3de8"  # origin/main
CROSS_SOURCE = __CROSS_SOURCE__  # experiments/kaggle_compat/cross.py
PATCH_SOURCE = __PATCH_SOURCE__  # the working tree's `ironmule/cuda_native.py` with expert routing, and its test
MODEL = ("mlx-community/gemma-4-26b-a4b-it-4bit", "0d77464eeb233a2da68ebf9d7dc4edaac7db956d")
STOCK = {"name": "stock", "env": {"MLX_MAX_OPS_PER_BUFFER": "20", "MLX_MAX_MB_PER_BUFFER": "100"},
         "knobs": {}, "mode": "interactive"}
CONFIGS = [STOCK, {"name": "ironmule_native", "env": {}, "knobs": {}, "mode": "interactive", "dtype": "native"}]
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable
DEADLINE = time.time() + 25 * 60
os.makedirs(f"{WORK}/logs", exist_ok=True)
with open("/tmp/cross.py", "w") as stream:
    stream.write(CROSS_SOURCE)
with open("/tmp/ironmule.patch", "w") as stream:
    stream.write(PATCH_SOURCE)
report = {"schema": "ironmule.perf1-kaggle.v15", "commit": COMMIT, "stages": {}, "performance_claim": False}
env = dict(os.environ, PATH=f"{VENV}/bin:" + os.environ["PATH"], PYTHONPATH="", PYTHONNOUSERSITE="1",
           HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONUNBUFFERED="1", IRONMULE_HOME="/tmp/ironmule-home",
           IRONMULE_API_KEY="perf1-local-only")


def save():
    with open(f"{WORK}/perf1-result.json", "w") as stream:
        json.dump(report, stream, indent=1, default=str)


def sh(name, cmd, timeout=900, cwd="/tmp"):
    left = DEADLINE - time.time()
    if left < 30:
        report["stages"][name] = {"exit": "skipped_deadline"}
        save()
        return None, ""
    started = time.time()
    proc = subprocess.Popen(cmd, shell=True, cwd=cwd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        out, _ = proc.communicate(timeout=min(timeout, left))
        code = proc.returncode
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out, _ = proc.communicate()
        code = "timeout"
    with open(f"{WORK}/logs/{name}.log", "w") as stream:
        stream.write(out)
    report["stages"][name] = {"exit": code, "seconds": round(time.time() - started, 1), "tail": out[-2500:]}
    save()
    print(f"== {name}: exit={code} {report['stages'][name]['seconds']}s", flush=True)
    return code, out


CHECK = r"""
import json, sys
import mlx.core as mx
from mlx_lm.models.switch_layers import QuantizedSwitchLinear
from ironmule import cuda_native as cn
out = {}
for name, (e, n, k) in {"qwen36_gate_up": (256, 512, 2048), "qwen36_down": (256, 2048, 512),
                        "gemma4_gate_up": (128, 704, 2816), "gemma4_down": (128, 2816, 704)}.items():
    layer = QuantizedSwitchLinear(k, n, e, bias=False)
    layer.set_dtype(mx.bfloat16)
    out[f"{name}_probe"] = cn._probe(layer)
    w, s, b = layer["weight"], layer["scales"], layer["biases"]
    for rows in (64, 2048):
        x = mx.random.normal((rows, 1, k)).astype(mx.bfloat16)
        idx = mx.sort(mx.random.randint(0, e, (rows,))).astype(mx.uint32)
        y = cn.gather_matmul(x, w, s, b, idx, sorted_indices=True).astype(mx.float32)
        ref = mx.gather_qmm(x.astype(mx.float32), w, s.astype(mx.float32), b.astype(mx.float32), rhs_indices=idx,
                            transpose=True, group_size=64, bits=4, sorted_indices=True)
        out[f"{name}_{rows}"] = float(mx.max(mx.abs(y - ref)) / mx.max(mx.abs(ref)))
print(json.dumps(out))
json.dump(out, open(sys.argv[1], "w"), indent=1)
"""
with open("/tmp/check.py", "w") as stream:
    stream.write(CHECK)

sh("env", "nvidia-smi --query-gpu=index,name,memory.total,clocks.max.sm --format=csv; nproc; free -g")
sh("clone", f"git clone -q https://github.com/Tobayko/IronMule {REPO} && git -C {REPO} checkout -q {COMMIT} "
            f"&& git -C {REPO} apply /tmp/ironmule.patch && git -C {REPO} status --short")
sh("venv", f"{SYS} -m pip install -q uv && {SYS} -m uv python install 3.12 "
           f"&& {SYS} -m uv venv --python-preference only-managed --python 3.12 {VENV}")
sh("install", f"{SYS} -m uv pip install --python {PY} -e '{REPO}[cuda]' 'mlx-lm==0.31.3' pytest", timeout=1200)
sh("freeze", f"{SYS} -m uv pip freeze --python {PY}")
sh("device_check", f"{PY} /tmp/check.py {WORK}/device-check.json", timeout=600)
sh("test_cuda_native", f"{PY} -m pytest -q -p no:cacheprovider -o addopts='' tests/engine/test_cuda_native.py", cwd=REPO)
try:
    with open(f"{WORK}/device-check.json") as stream:
        errors = json.load(stream)
except (OSError, ValueError):
    errors = {}
report["gate"] = {"device_check": len(errors) == 12 and all(v <= 1e-2 for v in errors.values())}
save()
if report["gate"]["device_check"]:
    model_id, revision = MODEL
    sh("download_gemma4-26b-a4b", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                                  f"print(s('{model_id}', revision='{revision}'))\"", timeout=1200)
    sh("cross_gemma4-26b-a4b", f"{PY} /tmp/cross.py {model_id} {revision} '{json.dumps(CONFIGS)}' "
                               f"{WORK}/cross-native-gemma4-26b-a4b.json 1 1", timeout=1500)

report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
