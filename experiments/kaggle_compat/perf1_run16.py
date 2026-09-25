# PERF1 run 16: run 15 again for the one stage it could not answer. Private notebook, internet
# on. Free quota: one run, <= 20 min (the week stands at about 9.45 h; it stays under 10).
#
# Run 15 (`perf1-run15-22fe0a42`) passed its device check on the product's expert kernel (12/12
# at most 3.8e-3) and the test file on the T4 (6 passed); stock Gemma 4 26B-A4B then ran, and
# `ironmule_native` died in its first prefill: `cudaMallocAsync ... out of memory` at
# `mx.eval(logits)`. Suspected cause, not the expert path: the plan's dense prefill dequantises
# a whole weight to float16, and Gemma's tied 262144 x 2816 head is 1.48 GB of it beside 14.2 GB
# of weights; PERF1-P's per-slice sync fixed that in `perf1.py` only. `head_skip_prefill`
# projects the last position alone, one row, which the row kernel takes without a copy.
# Rules fixed here: `cross.py` on Gemma 4 26B-A4B, both arms with `head_skip_prefill` (the
# reference is IronMule's bfloat16 path with the same knob), one repetition, one measured pass.
# The patch is run 15's, byte for byte. A native arm that runs is consistent with the head as
# the cause, not proof of it. Screening, no quality claim.
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
SKIP = {"head_skip_prefill": True}
CONFIGS = [{"name": "ironmule_head_skip", "env": {}, "knobs": SKIP, "mode": "interactive"},
           {"name": "ironmule_native_head_skip", "env": {}, "knobs": SKIP, "mode": "interactive", "dtype": "native"}]
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable
DEADLINE = time.time() + 20 * 60
os.makedirs(f"{WORK}/logs", exist_ok=True)
with open("/tmp/cross.py", "w") as stream:
    stream.write(CROSS_SOURCE)
with open("/tmp/ironmule.patch", "w") as stream:
    stream.write(PATCH_SOURCE)
report = {"schema": "ironmule.perf1-kaggle.v16", "commit": COMMIT, "stages": {}, "performance_claim": False}
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
sh("download_gemma4-26b-a4b", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                              f"print(s('{model_id}', revision='{revision}'))\"", timeout=1200)
sh("cross_gemma4-26b-a4b", f"{PY} /tmp/cross.py {model_id} {revision} '{json.dumps(CONFIGS)}' "
                           f"{WORK}/cross-native-gemma4-26b-a4b.json 1 1", timeout=1500)

report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
