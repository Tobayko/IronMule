# PERF1-X: `native` on a card its weights nearly fill. Private notebook, internet on. Quota: the user
# delegated all decisions on 2026-09-26 and asked that tests run on Kaggle (30 h week).
# Rules, before the run (docs/PROJECT_FRIDAY_BACKLOG.md PERF1-X):
#   * The engine suite as pytest.ini configures it, not integration, on this commit (f70805a:
#     load_engine switches head_skip_prefill on under `native` only when the head's float16
#     prefill copy does not fit beside the weights in 90% of the card's memory).
#   * perf1x_check.py through the product engine with untuned knobs and `native`:
#     Gemma 4 26B-A4B (run 15 ran out of memory in its first prefill) must load, record the
#     forced head skip with its reason, and serve `ironmule benchmark`'s workload once;
#     Qwen 3 8B must load without a forced head skip and serve the same workload.
#   * Kill (the entry's): Gemma 4 26B-A4B still fails -> `native` on such cards stays unusable
#     and the check becomes a refusal; Qwen 3 8B forced into head skip -> the threshold is wrong.
import json
import os
import signal
import subprocess
import sys
import time

COMMIT = "f70805a06d278ceea026cf4251ee5482058cca25"
WIKITEXT = ("Salesforce/wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3",
            "wikitext-2-raw-v1/test-00000-of-00001.parquet")
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable
DEADLINE = time.time() + 60 * 60
os.makedirs(f"{WORK}/logs", exist_ok=True)
report = {"schema": "ironmule.perf1x-kaggle.v1", "commit": COMMIT, "stages": {}, "performance_claim": False}
env = dict(os.environ, PATH=f"{VENV}/bin:" + os.environ["PATH"], PYTHONPATH="", PYTHONNOUSERSITE="1",
           HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONUNBUFFERED="1", IRONMULE_HOME="/tmp/ironmule-home")


def save():
    with open(f"{WORK}/perf1x-result.json", "w") as stream:
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



MODELS = (("gemma4-26b-a4b", "mlx-community/gemma-4-26b-a4b-it-4bit", "0d77464eeb233a2da68ebf9d7dc4edaac7db956d"),
          ("qwen3-8b", "mlx-community/Qwen3-8B-4bit", "545dc4251c05440727734bcd94334791f6ab0192"))

sh("env", "nvidia-smi --query-gpu=index,name,driver_version,memory.total,clocks.max.sm --format=csv; nproc; free -g")
sh("clone", f"git clone -q https://github.com/Tobayko/IronMule {REPO} && git -C {REPO} checkout -q {COMMIT} "
            f"&& git -C {REPO} rev-parse HEAD && git -C {REPO} status --short")
sh("venv", f"{SYS} -m pip install -q uv && {SYS} -m uv python install 3.12 "
           f"&& {SYS} -m uv venv --python-preference only-managed --python 3.12 {VENV}")
sh("install", f"{SYS} -m uv pip install --python {PY} -e '{REPO}[cuda]' 'mlx-lm==0.31.3' 'pytest>=8' 'pytest-xdist>=3' psutil scipy", timeout=1200)
sh("freeze", f"{SYS} -m uv pip freeze --python {PY}")
sh("pytest_engine", f"{PY} -m pytest tests/engine -m 'not integration' -rfEs -p no:cacheprovider "
                    f"--junitxml={WORK}/pytest_engine.xml", timeout=1800, cwd=REPO)
for key, model_id, revision in MODELS:
    code, _ = sh(f"download_{key}", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                                    f"print(s('{model_id}', revision='{revision}'))\"", timeout=1500)
    if code == 0:
        sh(f"check_{key}", f"CUDA_VISIBLE_DEVICES=0 {PY} {REPO}/experiments/kaggle_compat/perf1x_check.py "
                           f"{model_id} {revision} {WORK}/perf1x-{key}.json", timeout=1500)
report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
