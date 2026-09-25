# TESTS1: the reworked test setup on a Kaggle T4. The user asked on 2026-09-25 to rework the
# tests (four points). Private notebook, internet on; budget: the 30 h Kaggle week. Rules:
#   * The engine suite as pytest.ini configures it, not integration, on this commit (unused
#     imports removed, process-wide MLX variables restored after every test).
#   * The claims modules (`tests/test_numeric_plans.py`, `tests/test_documented_claims.py`),
#     now collected outside the target device; tests needing `.friday-data/` must skip, not fail.
#   * The real-model integration tests, serially, with Gemma 3 4B cached and IRONMULE_QWEN_MODEL
#     set to the Qwen 3.5 9B snapshot. No graph variable is set by the notebook: since PERF1-T2
#     `load_engine` switches CUDA graphs off for Qwen 3.5 itself.
#   * CI's extended `ruff check --select F`.
import json
import os
import signal
import subprocess
import sys
import time

COMMIT = "5518f97b68e74ed84d803e5e87e0f7fbb4506eae"
QWEN35 = ("mlx-community/Qwen3.5-9B-MLX-4bit", "938d8919941c6e7efd3c7150eff7fe9d12afa631")
GEMMA4B = ("mlx-community/gemma-3-4b-it-4bit", "93724907d4ed1745d2fe50baadf3b0b01a65abf2")
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable
DEADLINE = time.time() + 70 * 60
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
sh("pytest_engine", f"{PY} -m pytest tests/engine -m 'not integration' -rfEs -p no:cacheprovider "
                    f"--junitxml={WORK}/pytest_engine.xml", timeout=1800, cwd=REPO)
sh("pytest_claims", f"{PY} -m pytest tests/test_numeric_plans.py tests/test_documented_claims.py -n 0 -rfEs "
                    f"-p no:cacheprovider --junitxml={WORK}/pytest_claims.xml", timeout=900, cwd=REPO)
sh("ruff", f"{VENV}/bin/ruff check ironmule ironmule_product ironmule_inventory.py ironmule_cli.py "
           "friday_evidence/events.py friday_evidence/identity.py friday_evidence/statistics.py "
           "tools/product_bounded_bench.py tools/product_history_snapshot.py tests/engine tests/conftest.py "
           "tests/test_numeric_plans.py tests/test_documented_claims.py --select F", cwd=REPO)


def download(key, model):
    code, out = sh(f"download_{key}", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                                      f"print(s('{model[0]}', revision='{model[1]}'))\"", timeout=1200)
    return out.strip().splitlines()[-1] if code == 0 else None


if download("gemma3-4b", GEMMA4B):
    qwen = download("qwen35-9b", QWEN35)
    if qwen:
        env["IRONMULE_QWEN_MODEL"] = qwen
    sh("pytest_integration", f"{PY} -m pytest tests/engine -m integration -n 0 -rfEs -p no:cacheprovider "
                             f"--junitxml={WORK}/pytest_integration.xml", timeout=2400, cwd=REPO)
report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
