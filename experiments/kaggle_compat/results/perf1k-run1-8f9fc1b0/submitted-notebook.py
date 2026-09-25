# PERF1-K1: the opt-in tensor-batch server (`serve --batch-width`) on the engine suite, its
# quality gate and its throughput. Private notebook, internet on. The user asked on 2026-09-25
# to continue with PERF1-K (budget: the 30 h Kaggle week). Rules, fixed in the backlog entry
# before the run:
#   * The engine suite as pytest.ini configures it, not integration, on this commit.
#   * Gate: `batch_gate.py`, Qwen 3 8B under `native`, 16 x 512 WikiText-2, teacher-forced,
#     eight equal-length chunks per batch against one at a time, prefill and decode path, one
#     process; pass per path at an upper bound <= 1.005 with no non-finite value.
#   * Throughput: `batch_serve.py`, the real `ironmule serve` with `--compute-dtype native`,
#     reference and `--batch-width 8` started in the order ref b8 ref b8, one warm-up and three
#     measured rounds each of 8 concurrent `stream: false` requests, max_tokens 128. Pass: the
#     median aggregate tok/s of b8 over ref >= 1.2. Answers are compared per prompt (reported).
#   * Kill: either fails - the variant is removed again.
import json
import os
import signal
import subprocess
import sys
import time

COMMIT = "82c9f03e48f320be26527928de64efa858343df1"
QWEN8B = ("mlx-community/Qwen3-8B-4bit", "545dc4251c05440727734bcd94334791f6ab0192")
WIKITEXT = ("Salesforce/wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3",
            "wikitext-2-raw-v1/test-00000-of-00001.parquet")
GATE_ENV = "PERF1_NLL_CHUNKS=16 PERF1_NLL_TOKENS=512 "
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable
DEADLINE = time.time() + 75 * 60
os.makedirs(f"{WORK}/logs", exist_ok=True)
report = {"schema": "ironmule.perf1k-kaggle.v1", "commit": COMMIT, "stages": {}, "performance_claim": False}
env = dict(os.environ, PATH=f"{VENV}/bin:" + os.environ["PATH"], PYTHONPATH="", PYTHONNOUSERSITE="1",
           HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONUNBUFFERED="1", IRONMULE_HOME="/tmp/ironmule-home")


def save():
    with open(f"{WORK}/perf1k-result.json", "w") as stream:
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


def download(key, model):
    code, out = sh(f"download_{key}", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                                      f"print(s('{model[0]}', revision='{model[1]}'))\"", timeout=1200)
    return out.strip().splitlines()[-1] if code == 0 else None


sh("download_wikitext", f"{PY} -c \"from huggingface_hub import hf_hub_download as d; import pyarrow.parquet as pq; "
                        f"p = d('{WIKITEXT[0]}', '{WIKITEXT[2]}', repo_type='dataset', revision='{WIKITEXT[1]}'); "
                        "open('/tmp/wikitext.txt', 'w').write(''.join(pq.read_table(p).column('text').to_pylist())); print(p)\"")

sh("pytest_engine", f"{PY} -m pytest tests/engine -m 'not integration' -rfEs -p no:cacheprovider "
                    f"--junitxml={WORK}/pytest_engine.xml", timeout=1800, cwd=REPO)

# PERF1-K1: the gate, then the served throughput.
path = download("qwen3-8b", QWEN8B)
if path:
    KC = f"{REPO}/experiments/kaggle_compat"
    sh("gate_qwen3-8b_native", f"{PY} {KC}/batch_gate.py {QWEN8B[0]} {QWEN8B[1]} native /tmp/wikitext.txt "
                               f"{WORK} qwen3-8b-native", timeout=2400, cwd=REPO)
    sh("setup", "ironmule setup --mode desktop")
    sh("models_add", f"ironmule models add {QWEN8B[0]} --revision {QWEN8B[1]}")
    sh("serve_qwen3-8b_native", f"{PY} {KC}/batch_serve.py {QWEN8B[0]} {WORK}/serve-qwen3-8b-native.json "
                                f"native ref b8 ref b8", timeout=2400, cwd=REPO)
report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
