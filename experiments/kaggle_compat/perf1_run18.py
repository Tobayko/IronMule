# PERF1 run 18: validate every CUDA number the website shows, and measure its projected one.
# Private notebook, internet on. Quota: the user allowed about 1.5 h on 2026-09-24 (the week
# stood at 9.83 h); this run is capped at 75 min of stages.
#
# The website's CUDA view shows, per research/LEDGER.md: T4 exact speed-up 1.03x / 1.05x / 1.82x
# (Gemma 3 12B / 4B / 1B, PORT1 `cross.py`), Qwen 3 8B stock 6.3 -> 32.5 tok/s and TTFT 0.76 s
# under the native kernels (PERF1 run 2, `perf1.py e2e`), 2.49x native over the float32 plan on
# 12B (run 17), and "up to 5.52x", chained as 1 / (0.4905 x 0.3695) from two runs whose float32
# arms differ (PORT1: head_skip_prefill, throughput mode; run 17: no knobs, interactive).
# Rules fixed here, IronMule at `b6886a0` plus run 15's patch byte for byte, every number
# re-measured with the protocol it was published with, stock = IronMule off:
#   * Gemma 3 12B, `cross.py`, 2 interleaved repetitions, 1 measured pass (PORT1's protocol):
#     stock; exact (head_skip_prefill, throughput); float32 plan with the same knobs; float32
#     plan without knobs (run 17's reference); `native`; `native` + compiled_fixed_cache. The
#     last against stock replaces the projection with a measurement. Exact must stay 6/6
#     token-identical to stock.
#   * Qwen 3 8B, `perf1.py e2e` (run 2's protocol: 512-token prompt, 1 warm + 3 measured
#     generations of 128 tokens): stock and `kernel+p16`, plus `kernel+p16` with pinned rounding,
#     the arithmetic the product's `native` plan uses; MLX_MAX_OPS_PER_BUFFER=400 as run 2 set it.
#   * Gemma 3 1B (3 reps x 3 passes, tuned knobs) and 4B (3 reps x 2 passes, exact): stock vs
#     IronMule as PORT1 ran them; 6/6 identical required.
#   * A published ratio counts as reproduced if the new median lies within 5% of it; otherwise
#     the new measurement replaces it. Absolute tok/s differ between T4 cells by up to 2x
#     (PERF1 runs 8/9), so they are reported, not judged. Speed only; PERF1-Y stays open.
import json
import os
import signal
import subprocess
import sys
import time

COMMIT = "b6886a0823f2da14a1a02ea8a3176660786a3de8"  # origin/main
CROSS_SOURCE = __CROSS_SOURCE__  # experiments/kaggle_compat/cross.py
PATCH_SOURCE = __PATCH_SOURCE__  # the working tree's `ironmule/cuda_native.py` with expert routing, and its test
PERF1_SOURCE = __PERF1_SOURCE__  # experiments/kaggle_compat/perf1.py
STOCK = {"name": "stock", "env": {"MLX_MAX_OPS_PER_BUFFER": "20", "MLX_MAX_MB_PER_BUFFER": "100"},
         "knobs": {}, "mode": "interactive"}
HS = {"head_skip_prefill": True}
TUNED = {"compiled_fixed_cache": True, "fused_argmax": True, "head_skip_prefill": True, "readback_every": 2,
         "capacity_slack": 128, "fuse_projections": True}
# (key, model, revision, configs, reps, measure): each published number with its own protocol.
CROSS = [
    ("12b", "mlx-community/gemma-3-12b-it-4bit", "86cc6a8dedbc456dd0e4af01a9d09f396f77e558",
     [STOCK,
      {"name": "ironmule_exact", "env": {}, "knobs": HS, "mode": "throughput"},
      {"name": "ironmule_fp32", "env": {}, "knobs": HS, "mode": "throughput", "dtype": "float32"},
      {"name": "ironmule_fp32_plain", "env": {}, "knobs": {}, "mode": "interactive", "dtype": "float32"},
      {"name": "ironmule_native", "env": {}, "knobs": {}, "mode": "interactive", "dtype": "native"},
      {"name": "ironmule_native_compiled", "env": {}, "knobs": {"compiled_fixed_cache": True},
       "mode": "interactive", "dtype": "native"}], 2, 1),
    ("1b", "mlx-community/gemma-3-1b-it-4bit", "2d44e83dc9e80843d22fb941d3d699a0b1351aa6",
     [STOCK, {"name": "ironmule", "env": {}, "knobs": TUNED, "mode": "throughput"}], 3, 3),
    ("4b", "mlx-community/gemma-3-4b-it-4bit", "93724907d4ed1745d2fe50baadf3b0b01a65abf2",
     [STOCK, {"name": "ironmule_exact", "env": {}, "knobs": HS, "mode": "throughput"}], 3, 2),
]
QWEN = ("mlx-community/Qwen3-8B-4bit", "545dc4251c05440727734bcd94334791f6ab0192")
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable
DEADLINE = time.time() + 75 * 60
os.makedirs(f"{WORK}/logs", exist_ok=True)
with open("/tmp/cross.py", "w") as stream:
    stream.write(CROSS_SOURCE)
with open("/tmp/perf1.py", "w") as stream:
    stream.write(PERF1_SOURCE)
with open("/tmp/ironmule.patch", "w") as stream:
    stream.write(PATCH_SOURCE)
report = {"schema": "ironmule.perf1-kaggle.v18", "commit": COMMIT, "stages": {}, "performance_claim": False}
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


sh("env", "nvidia-smi --query-gpu=index,name,memory.total,clocks.max.sm --format=csv; nproc; free -g")
sh("clone", f"git clone -q https://github.com/Tobayko/IronMule {REPO} && git -C {REPO} checkout -q {COMMIT} "
            f"&& git -C {REPO} apply /tmp/ironmule.patch && git -C {REPO} status --short")
sh("venv", f"{SYS} -m pip install -q uv && {SYS} -m uv python install 3.12 "
           f"&& {SYS} -m uv venv --python-preference only-managed --python 3.12 {VENV}")
sh("install", f"{SYS} -m uv pip install --python {PY} -e '{REPO}[cuda]' 'mlx-lm==0.31.3'", timeout=1200)
sh("freeze", f"{SYS} -m uv pip freeze --python {PY}")


def download(key, model_id, revision):
    code, out = sh(f"download_{key}", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                                      f"print(s('{model_id}', revision='{revision}'))\"", timeout=1200)
    return out.strip().splitlines()[-1] if code == 0 else None


key, model_id, revision, configs, reps, measure = CROSS[0]
if download(key, model_id, revision):
    sh(f"cross_{key}", f"{PY} /tmp/cross.py {model_id} {revision} '{json.dumps(configs)}' "
                       f"{WORK}/cross-gemma3-{key}.json {reps} {measure}", timeout=2400)
path = download("qwen3-8b", *QWEN)
if path:
    for label, arm, env_prefix in (("stock", "stock", ""), ("kernel+p16", "kernel+p16", ""),
                                   ("kernel+p16-pinned", "kernel+p16", "PERF1_ARITH=pinned ")):
        sh(f"e2e_qwen3-8b_{label}", f"MLX_MAX_OPS_PER_BUFFER=400 {env_prefix}{PY} /tmp/perf1.py e2e {path} {arm} "
                                    f"{WORK}/e2e-qwen3-8b-{label}.json", timeout=900)
for key, model_id, revision, configs, reps, measure in CROSS[1:]:
    if download(key, model_id, revision):
        sh(f"cross_{key}", f"{PY} /tmp/cross.py {model_id} {revision} '{json.dumps(configs)}' "
                           f"{WORK}/cross-gemma3-{key}.json {reps} {measure}", timeout=1500)

report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
