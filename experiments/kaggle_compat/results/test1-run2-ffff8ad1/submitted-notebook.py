# TEST1, run 2: the opt-in Qwen hybrid-cache gate with its reference count corrected to the
# runtime contract, twice with CUDA graphs on and twice off. Kaggle T4, MLX CUDA backend.
# Private notebook, internet on (pip, git, Hugging Face). No performance claim.
import json
import os
import re
import signal
import subprocess
import sys
import time

COMMIT = "2151eb27f628ec2863ed44a732b52c7d3f1a6c8e"
MODEL = ("mlx-community/Qwen3.5-9B-MLX-4bit", "938d8919941c6e7efd3c7150eff7fe9d12afa631")
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable
DEADLINE = time.time() + 40 * 60
TEST = "tests/engine/test_qwen_hybrid_integration.py"
os.makedirs(f"{WORK}/logs", exist_ok=True)
report = {"schema": "ironmule.test1-qwen-kaggle.v1", "commit": COMMIT, "stages": {}}
env = dict(os.environ, PATH=f"{VENV}/bin:" + os.environ["PATH"], PYTHONPATH="", PYTHONNOUSERSITE="1",
           IRONMULE_HOME="/tmp/ironmule-home", HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONUNBUFFERED="1")


def save():
    with open(f"{WORK}/test1-qwen-result.json", "w") as stream:
        json.dump(report, stream, indent=1, default=str)


def sh(name, cmd, timeout=900, cwd="/tmp", extra=None):
    left = DEADLINE - time.time()
    if left < 30:
        report["stages"][name] = {"exit": "skipped_deadline"}
        save()
        return None, ""
    started = time.time()
    proc = subprocess.Popen(cmd, shell=True, cwd=cwd, env=dict(env, **(extra or {})), stdout=subprocess.PIPE,
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


sh("env", "nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv; nproc; free -g")
sh("clone", f"git clone -q https://github.com/Tobayko/IronMule {REPO} && git -C {REPO} checkout -q {COMMIT} "
            f"&& git -C {REPO} rev-parse HEAD && git -C {REPO} status --short")
sh("venv", f"{SYS} -m pip install -q uv && {SYS} -m uv python install 3.12 "
           f"&& {SYS} -m uv venv --python-preference only-managed --python 3.12 {VENV}")
sh("install", f"{SYS} -m uv pip install --python {PY} -e '{REPO}[cuda]' 'mlx-lm==0.31.3' "
              "'pytest>=8' 'pytest-xdist>=3' psutil scipy", timeout=1200)
sh("freeze", f"{SYS} -m uv pip freeze --python {PY}")
code, out = sh("download_qwen35_9b", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                                     f"print(s('{MODEL[0]}', revision='{MODEL[1]}'))\"", timeout=1200)
if code == 0:
    qwen = {"IRONMULE_QWEN_MODEL": out.strip().splitlines()[-1]}
    for label, extra in (("graphs1_a", {}), ("graphs0_a", {"MLX_USE_CUDA_GRAPHS": "0"}),
                         ("graphs1_b", {}), ("graphs0_b", {"MLX_USE_CUDA_GRAPHS": "0"})):
        name = f"pytest_qwen_{label}"
        code, out = sh(name, f"{PY} -m pytest {TEST} -m integration -n 0 -vv -rfE -p no:cacheprovider "
                             f"--junitxml={WORK}/{name}.xml", timeout=900, cwd=REPO, extra=dict(qwen, **extra))
        lines = [line for line in out.splitlines() if re.search(r"\d+ (passed|failed|error)", line)]
        report[name] = {"exit": code, "line": lines[-1].strip("= ") if lines else None}
        save()
report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
