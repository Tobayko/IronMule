# BACKLOG9: the engine suite on PERF1-T2's change, and the entry's own test. Private notebook,
# internet on. Quota: the user asked on 2026-09-25 to work through the backlog (budget: the 30 h
# Kaggle week) and left the open decisions to the agent. Rules, before the run:
#   * The engine suite as pytest.ini configures it, not integration, on this commit.
#   * PERF1-T2 (docs/PROJECT_FRIDAY_BACKLOG.md): Qwen 3.5 9B through `cross.py`, whose children
#     load with `load_engine`, IronMule interactive without knobs, 3 fresh processes. Pass: every
#     process reports MLX_USE_CUDA_GRAPHS=0 set by IronMule (the notebook does not set it) and
#     the three give one output digest. Kill: more than one digest. Diagnostic, not a gate: the
#     same arm with MLX_USE_CUDA_GRAPHS=1 set by the caller, 3 processes, which IronMule must
#     leave on.
import json
import os
import signal
import subprocess
import sys
import time

COMMIT = __COMMIT__
QWEN35 = ("mlx-community/Qwen3.5-9B-MLX-4bit", "938d8919941c6e7efd3c7150eff7fe9d12afa631")
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable
DEADLINE = time.time() + 60 * 60
os.makedirs(f"{WORK}/logs", exist_ok=True)
report = {"schema": "ironmule.backlog9-kaggle.v1", "commit": COMMIT, "stages": {}, "performance_claim": False}
env = dict(os.environ, PATH=f"{VENV}/bin:" + os.environ["PATH"], PYTHONPATH="", PYTHONNOUSERSITE="1",
           HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONUNBUFFERED="1", IRONMULE_HOME="/tmp/ironmule-home")


def save():
    with open(f"{WORK}/backlog9-result.json", "w") as stream:
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


sh("env", "nvidia-smi --query-gpu=index,name,driver_version,memory.total,clocks.max.sm --format=csv; nproc; free -g")
sh("clone", f"git clone -q https://github.com/Tobayko/IronMule {REPO} && git -C {REPO} checkout -q {COMMIT} "
            f"&& git -C {REPO} rev-parse HEAD && git -C {REPO} status --short")
sh("venv", f"{SYS} -m pip install -q uv && {SYS} -m uv python install 3.12 "
           f"&& {SYS} -m uv venv --python-preference only-managed --python 3.12 {VENV}")
sh("install", f"{SYS} -m uv pip install --python {PY} -e '{REPO}[cuda]' 'mlx-lm==0.31.3' pyarrow 'pytest>=8' 'pytest-xdist>=3' psutil scipy", timeout=1200)
sh("freeze", f"{SYS} -m uv pip freeze --python {PY}")
sh("pytest_engine", f"{PY} -m pytest tests/engine -m 'not integration' -rfEs -p no:cacheprovider "
                    f"--junitxml={WORK}/pytest_engine.xml", timeout=1800, cwd=REPO)


def download(key, model):
    code, out = sh(f"download_{key}", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                                      f"print(s('{model[0]}', revision='{model[1]}'))\"", timeout=1200)
    return out.strip().splitlines()[-1] if code == 0 else None


# PERF1-T2: graphs off by IronMule's own rule, then the caller's override as a diagnostic.
if download("qwen35-9b", QWEN35):
    CROSS = f"{REPO}/experiments/kaggle_compat/cross.py"
    for name, graph_env in (("ironmule", {}), ("ironmule_graphs_on", {"MLX_USE_CUDA_GRAPHS": "1"})):
        configs = json.dumps([{"name": name, "env": graph_env, "knobs": {}, "mode": "interactive"}])
        sh(f"cross_qwen35-9b_{name}", f"{PY} {CROSS} {QWEN35[0]} {QWEN35[1]} '{configs}' "
                                      f"{WORK}/cross-qwen35-9b-{name}.json 3 1", timeout=1500, cwd=REPO)
report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
