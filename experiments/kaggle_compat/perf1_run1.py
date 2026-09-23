# PERF1 run 1: two T4s against one, and a native 4-bit matvec against emulated bfloat16.
# Private notebook, internet on. Free quota: one run, <= 1.5 h of the 3 h PERF1 budget.
# Backlog: docs/PROJECT_FRIDAY_BACKLOG.md, PERF1. Screening, no performance claim.
#
# Every arm is a fresh process with MLX_MAX_OPS_PER_BUFFER=400 (IronMule's own setting on
# compute capability < 8), so the arms differ only in what they name. The kernel arm only
# runs end to end when the microbenchmark says the kernel is correct.
import json
import os
import signal
import subprocess
import sys
import time

COMMIT = "b339c817299de173f8a330089afb69973c519353"  # research/port2-model-families
PERF1_SOURCE = __PERF1_SOURCE__  # experiments/kaggle_compat/perf1.py
MODELS = [("qwen3-8b", "mlx-community/Qwen3-8B-4bit", "545dc4251c05440727734bcd94334791f6ab0192"),
          ("qwen3-14b", "mlx-community/Qwen3-14B-4bit", "a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4")]
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable
DEADLINE = time.time() + 80 * 60
os.makedirs(f"{WORK}/logs", exist_ok=True)
with open("/tmp/perf1.py", "w") as stream:
    stream.write(PERF1_SOURCE)
report = {"schema": "ironmule.perf1-kaggle.v1", "commit": COMMIT, "stages": {}, "performance_claim": False}
env = dict(os.environ, PATH=f"{VENV}/bin:" + os.environ["PATH"], PYTHONPATH="", PYTHONNOUSERSITE="1",
           HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONUNBUFFERED="1", MLX_MAX_OPS_PER_BUFFER="400")


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
            f"&& git -C {REPO} rev-parse HEAD")
sh("venv", f"{SYS} -m pip install -q uv && {SYS} -m uv python install 3.12 "
           f"&& {SYS} -m uv venv --python-preference only-managed --python 3.12 {VENV}")
sh("install", f"{SYS} -m uv pip install --python {PY} -e '{REPO}[cuda]' 'mlx-lm==0.31.3'", timeout=1200)
sh("freeze", f"{SYS} -m uv pip freeze --python {PY}")

# -- B first: it is the cheapest and it decides whether the kernel arms run at all ----------
sh("kernel_probe", f"{PY} /tmp/perf1.py kernel {WORK}/kernel-probe.json", timeout=600)
# Fixed before the run: among row variants correct on every shape (rel. error <= 1e-2), take
# the fastest estimated decode step; run the kernel arms only if it beats float32
# `quantized_matmul` at the same shapes (PERF1-B kill criterion).
try:
    with open(f"{WORK}/kernel-probe.json") as stream:
        probe = json.load(stream)
    steps = {r: probe[f"step_kernel_r{r}_ms"] for r in (1, 2, 4, 8)
             if f"step_kernel_r{r}_ms" in probe
             and all(s.get(f"kernel_r{r}_rel_err", 1.0) <= 1e-2 for s in probe["shapes"].values())}
    rows = min(steps, key=steps.get) if steps else None
    kernel_ok = rows is not None and steps[rows] < probe["step_qmm_fp32_ms"]
except (OSError, ValueError, KeyError):
    rows, kernel_ok = None, False
report["kernel_rows"], report["kernel_ok"] = rows, kernel_ok
save()
if kernel_ok:
    env["PERF1_ROWS"] = str(rows)
ARMS = ["stock", "fp32"] + (["kernel"] if kernel_ok else [])

for key, model_id, revision in MODELS:
    code, out = sh(f"download_{key}", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                                      f"print(s('{model_id}', revision='{revision}'))\"", timeout=1200)
    if code != 0:
        continue
    path = out.strip().splitlines()[-1]
    for arm in ARMS:
        sh(f"e2e_{key}_{arm}", f"{PY} /tmp/perf1.py e2e {path} {arm} {WORK}/e2e-{key}-{arm}.json",
           timeout=900)
        sh(f"tp_{key}_{arm}", f"{VENV}/bin/mlx.launch --hosts 127.0.0.1 -n 2 --backend ring --python {PY} "
                              f"/tmp/perf1.py tp {path} {arm} {WORK}/tp-{key}-{arm}", timeout=900)

report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
