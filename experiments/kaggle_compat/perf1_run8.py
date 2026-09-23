# PERF1 run 8: tensor cores for the server's batch. Private notebook, internet on. Free quota:
# one run, <= 45 min. Backlog: PERF1-I. Nothing from runs 1-7 is repeated except the in-run
# control every comparison needs (`kernel+p16` at widths 1 and 8).
#
# Run 6 found continuous batching barely paying: the row kernel's cost grows with every
# activation row, so a width-8 step cost ~8x a width-1 step. `mma` dequantises each weight
# once, in registers, and multiplies it with up to 16 rows in one `mma.sync.m16n8k8`.
# Rules fixed here:
#   * the probe must find `mma` correct (rel. error <= 1e-2 on every shape at M = 1, 8, 16),
#     else the server arms using it are skipped;
#   * server: 8 different chat requests, greedy, 128 tokens each, aggregate tok/s and TTFT, as
#     run 6. `mma` counts if width 8 beats the in-run control's width 8 by >= 1.2x; kill otherwise.
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
SERVER = {"qwen3-8b": [("kernel+mma+p16", "free", "1,4,8,16"), ("kernel+p16", "free", "1,8")],
          "qwen3-14b": [("kernel+mma+p16", "free", "1,8,16")]}
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable
DEADLINE = time.time() + 45 * 60
os.makedirs(f"{WORK}/logs", exist_ok=True)
with open("/tmp/perf1.py", "w") as stream:
    stream.write(PERF1_SOURCE)
with open("/tmp/ironmule.patch", "w") as stream:
    stream.write(PATCH_SOURCE)
report = {"schema": "ironmule.perf1-kaggle.v8", "commit": COMMIT, "stages": {}, "performance_claim": False}
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
sh("mma_probe", f"{PY} /tmp/perf1.py mma {WORK}/mma-probe.json", timeout=900)
try:
    with open(f"{WORK}/mma-probe.json") as stream:
        probe = json.load(stream)
    mma_ok = all(row.get(f"mma_m{m}_rel_err", 1.0) <= 1e-2 for row in probe["shapes"].values() for m in (1, 8, 16))
except (OSError, ValueError, KeyError):
    mma_ok = False
report["mma_ok"] = mma_ok
save()
if not mma_ok:
    SERVER = {key: [arm for arm in arms if "mma" not in arm[0]] for key, arms in SERVER.items()}
for key, model_id, revision in MODELS:
    code, out = sh(f"download_{key}", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                                      f"print(s('{model_id}', revision='{revision}'))\"", timeout=1200)
    if code != 0:
        continue
    path = out.strip().splitlines()[-1]
    for arm, arith, widths in SERVER[key]:
        sh(f"server_{key}_{arm}_{arith}", f"PERF1_ARITH={arith} {PY} /tmp/perf1.py server {path} '{arm}' {widths} "
                                          f"{WORK}/server-{key}-{arm}-{arith}.json", timeout=1200)

report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
