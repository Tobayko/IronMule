# TESTS1 run 2: the Qwen 3.5 integration gate alone, in a pytest process of its own. Run 1
# (`tests1-run1-c97fe1e7`) passed the engine suite, the claims modules, ruff and 12 integration
# tests, and failed this gate after the Gemma tests in the same process had already fixed CUDA
# graphs on. Private notebook, internet on; budget: the 30 h Kaggle week. Rules: Qwen 3.5 9B's
# snapshot in IRONMULE_QWEN_MODEL, no graph variable set by the notebook, the gate's own test
# file only. Pass: the file passes, which also shows PERF1-T2's automatic graphs-off on the
# library path. Fail: the cause is not the process order, and it is investigated.
import json
import os
import signal
import subprocess
import sys
import time

COMMIT = "afc6889c4d3913439ad857d4edfd40be009172b7"
QWEN35 = ("mlx-community/Qwen3.5-9B-MLX-4bit", "938d8919941c6e7efd3c7150eff7fe9d12afa631")
GEMMA4B = ("mlx-community/gemma-3-4b-it-4bit", "93724907d4ed1745d2fe50baadf3b0b01a65abf2")
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable
DEADLINE = time.time() + 40 * 60
os.makedirs(f"{WORK}/logs", exist_ok=True)
report = {"schema": "ironmule.tests1-kaggle.v1", "commit": COMMIT, "stages": {}, "performance_claim": False}
env = dict(os.environ, PATH=f"{VENV}/bin:" + os.environ["PATH"], PYTHONPATH="", PYTHONNOUSERSITE="1",
           HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONUNBUFFERED="1", IRONMULE_HOME="/tmp/ironmule-home")


def save():
    with open(f"{WORK}/tests1-result.json", "w") as stream:
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
sh("install", f"{SYS} -m uv pip install --python {PY} -e '{REPO}[cuda]' 'mlx-lm==0.31.3' pyarrow 'pytest>=8' 'pytest-xdist>=3' psutil scipy 'ruff==0.16.6'", timeout=1200)
sh("freeze", f"{SYS} -m uv pip freeze --python {PY}")
def download(key, model):
    code, out = sh(f"download_{key}", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                                      f"print(s('{model[0]}', revision='{model[1]}'))\"", timeout=1200)
    return out.strip().splitlines()[-1] if code == 0 else None


qwen = download("qwen35-9b", QWEN35)
if qwen:
    env["IRONMULE_QWEN_MODEL"] = qwen
    # Run 1 ran every integration test in one process and the Qwen gate failed after the Gemma
    # tests' kernels had fixed CUDA graphs on; this run gives the gate a process of its own.
    sh("pytest_qwen", f"{PY} -m pytest tests/engine/test_qwen_hybrid_integration.py -m integration -n 0 -rfEs "
                      f"-p no:cacheprovider --junitxml={WORK}/pytest_qwen.xml", timeout=2400, cwd=REPO)
report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
