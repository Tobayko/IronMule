# PERF1 run 4: the kernel inside IronMule's own runtime, and where a decode step's time goes.
# Private notebook, internet on. Free quota: one run, <= 45 min. Backlog: PERF1-E.
# Screening, no performance claim. Nothing from runs 1-3 is repeated: run 3's IronMule stage
# never measured anything (ModelIdentityError under huggingface_hub 1.32's hub-wide blob store),
# and the fix for that ships here as a patch against the pinned commit.
#
# Rules fixed here, before the run:
#   * workload: `ironmule benchmark`'s (6 strict requests x 48 tokens), one warm pass and two
#     measured, each arm a fresh process; the baseline is stock bf16 with no knobs, interactive.
#   * the combination counts only if the tuned knobs beat the kernel without them
#     (median wall ratio < 0.98), and only if every request's tokens equal the same plan
#     without knobs; kill otherwise.
#   * the chain probe is diagnostic: GPU time of one decode step's matmuls without a sync.
import json
import os
import signal
import subprocess
import sys
import time

COMMIT = "b339c817299de173f8a330089afb69973c519353"  # research/port2-model-families
PERF1_SOURCE = __PERF1_SOURCE__  # experiments/kaggle_compat/perf1.py
PATCH_SOURCE = __PATCH_SOURCE__  # git diff -- ironmule/model_identity.py (hub-wide HF blobs)
MODELS = [("qwen3-8b", "mlx-community/Qwen3-8B-4bit", "545dc4251c05440727734bcd94334791f6ab0192"),
          ("qwen3-14b", "mlx-community/Qwen3-14B-4bit", "a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4")]
TUNED = {"compiled_fixed_cache": True, "fused_argmax": True, "head_skip_prefill": True,
         "readback_every": 2, "capacity_slack": 128, "fuse_projections": True}
ARMS = {"qwen3-8b": [("stock", {}, "interactive"), ("kernel+p16", {}, "interactive"),
                     ("kernel+p16", TUNED, "interactive"), ("kernel+p16", TUNED, "throughput"),
                     ("fp32+k32+p16", {}, "interactive"), ("fp32+k32+p16", TUNED, "throughput")],
        "qwen3-14b": [("kernel+p16", {}, "interactive"), ("kernel+p16", TUNED, "throughput")]}
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable
DEADLINE = time.time() + 40 * 60
os.makedirs(f"{WORK}/logs", exist_ok=True)
with open("/tmp/perf1.py", "w") as stream:
    stream.write(PERF1_SOURCE)
with open("/tmp/ironmule.patch", "w") as stream:
    stream.write(PATCH_SOURCE)
report = {"schema": "ironmule.perf1-kaggle.v4", "commit": COMMIT, "stages": {}, "performance_claim": False}
env = dict(os.environ, PATH=f"{VENV}/bin:" + os.environ["PATH"], PYTHONPATH="", PYTHONNOUSERSITE="1",
           HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONUNBUFFERED="1", MLX_MAX_OPS_PER_BUFFER="400",
           IRONMULE_HOME="/tmp/ironmule-home")


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
sh("chain_probe", f"{PY} /tmp/perf1.py chain {WORK}/chain-probe.json", timeout=600)

for key, model_id, revision in MODELS:
    code, _ = sh(f"download_{key}", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                                    f"print(s('{model_id}', revision='{revision}'))\"", timeout=1200)
    if code != 0:
        continue
    for arm, knobs, mode in ARMS[key]:
        name = f"{arm}-{'tuned' if knobs else 'baseline'}-{mode}"
        sh(f"ironmule_{key}_{name}", f"{PY} /tmp/perf1.py ironmule {model_id} {revision} '{arm}' "
                                     f"'{json.dumps(knobs)}' {mode} {WORK}/ironmule-{key}-{name}.json", timeout=900)

report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
