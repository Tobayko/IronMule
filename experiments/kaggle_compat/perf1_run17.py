# PERF1 run 17: Gemma 3 12B under the `native` plan. Private notebook, internet on. Free quota:
# one run, <= 22 min (the week stands at 9.57 h; it stays under 10).
#
# User goal: at least +15% on Gemma 3 12B on NVIDIA over the best plan it has. That is the
# float32 plan, qualified on the T4 (PORT1: 0.4905 of stock, 2.04x). `native` (bf16 row kernel
# for decode, float16 GEMM for prefill) ran at 0.200 of stock on Qwen 3 8B (run 4) and was never
# measured on Gemma 3, whose float16 plan failed its gate (perplexity 102.5 -> 209.6 at 4B). On
# the Mac (diagnostic only, not T4 evidence), native's prefill arithmetic on Gemma 3 12B stayed
# far inside float16 (largest input 7264, output 2806) and its WikiText-2 NLL over 8 x 512 tokens
# was 2.66999 against 2.66993 for bf16. Rules fixed here: `cross.py`, 2 interleaved
# repetitions, the float32 plan as the reference; `native`, and `native` with
# `compiled_fixed_cache`. The goal counts as met on the median wall ratio <= 0.870 (1.15x) with
# coherent answers; token identity to float32 is not expected (a different plan). Screening:
# no quality gate here, so no recommendation follows from it. Patch: run 15's, byte for byte.
import json
import os
import signal
import subprocess
import sys
import time

COMMIT = "b6886a0823f2da14a1a02ea8a3176660786a3de8"  # origin/main
CROSS_SOURCE = __CROSS_SOURCE__  # experiments/kaggle_compat/cross.py
PATCH_SOURCE = __PATCH_SOURCE__  # the working tree's `ironmule/cuda_native.py` with expert routing, and its test
MODEL = ("mlx-community/gemma-3-12b-it-4bit", "86cc6a8dedbc456dd0e4af01a9d09f396f77e558")
CONFIGS = [{"name": "ironmule_fp32", "env": {}, "knobs": {}, "mode": "interactive", "dtype": "float32"},
           {"name": "ironmule_native", "env": {}, "knobs": {}, "mode": "interactive", "dtype": "native"},
           {"name": "ironmule_native_compiled", "env": {}, "knobs": {"compiled_fixed_cache": True},
            "mode": "interactive", "dtype": "native"}]
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable
DEADLINE = time.time() + 22 * 60
os.makedirs(f"{WORK}/logs", exist_ok=True)
with open("/tmp/cross.py", "w") as stream:
    stream.write(CROSS_SOURCE)
with open("/tmp/ironmule.patch", "w") as stream:
    stream.write(PATCH_SOURCE)
report = {"schema": "ironmule.perf1-kaggle.v17", "commit": COMMIT, "stages": {}, "performance_claim": False}
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
model_id, revision = MODEL
sh("download_gemma3-12b", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                          f"print(s('{model_id}', revision='{revision}'))\"", timeout=1200)
sh("cross_gemma3-12b", f"{PY} /tmp/cross.py {model_id} {revision} '{json.dumps(CONFIGS)}' "
                       f"{WORK}/cross-native-gemma3-12b.json 2 1", timeout=1800)

report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
