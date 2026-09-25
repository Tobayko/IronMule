# BACKLOG2: the engine suite on the two fixes, PORT1-F with its new diagnostic, and PERF1-M as
# fixed. Private notebook, internet on. Quota: the user asked on 2026-09-25 to work through the
# backlog (budget: the 30 h Kaggle week); stages are capped at 95 min. Rules, before the run:
#   * The engine suite as pytest.ini configures it, not integration, on this commit.
#   * PORT1-F (docs/PROJECT_FRIDAY_BACKLOG.md): `ironmule tune` on Gemma 3 4B in bf16, once;
#     a failed confirmation child now names its exception class (`309728e`).
#   * PERF1-M: Qwen 3 14B's decode-path gate with PERF1 run 5's protocol, set explicitly this
#     time (BACKLOG1 ran perf1.py's defaults): PERF1_NLL_CHUNKS=16, PERF1_NLL_TOKENS=512,
#     stock bf16 against `kernel` with pinned arithmetic. Pass: upper bound of the 10 000-sample
#     chunk bootstrap <= 1.005 and no non-finite value. Started only with 55 min left.
import json
import os
import signal
import subprocess
import sys
import time

COMMIT = __COMMIT__
QWEN14 = ("mlx-community/Qwen3-14B-4bit", "a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4")
GEMMA4B = ("mlx-community/gemma-3-4b-it-4bit", "93724907d4ed1745d2fe50baadf3b0b01a65abf2")
WIKITEXT = ("Salesforce/wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3",
            "wikitext-2-raw-v1/test-00000-of-00001.parquet")
GATE_ENV = "PERF1_NLL_CHUNKS=16 PERF1_NLL_TOKENS=512 "
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable
DEADLINE = time.time() + 95 * 60
os.makedirs(f"{WORK}/logs", exist_ok=True)
report = {"schema": "ironmule.backlog2-kaggle.v1", "commit": COMMIT, "stages": {}, "performance_claim": False}
env = dict(os.environ, PATH=f"{VENV}/bin:" + os.environ["PATH"], PYTHONPATH="", PYTHONNOUSERSITE="1",
           HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONUNBUFFERED="1", IRONMULE_HOME="/tmp/ironmule-home")


def save():
    with open(f"{WORK}/backlog2-result.json", "w") as stream:
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

# PORT1-F: the bf16 tune whose confirmation child died in BACKLOG1.
if download("gemma3-4b", GEMMA4B):
    sh("tune_gemma3-4b", f"ironmule tune --model {GEMMA4B[0]}", timeout=1800)

# PERF1-M: the stock reference first (about 40 min), then the kernel (about 9 min).
path = download("qwen3-14b", QWEN14) if DEADLINE - time.time() > 57 * 60 else None
if path:
    sh("download_wikitext", f"{PY} -c \"from huggingface_hub import hf_hub_download as d; import pyarrow.parquet as pq; "
                            f"p = d('{WIKITEXT[0]}', '{WIKITEXT[2]}', repo_type='dataset', revision='{WIKITEXT[1]}'); "
                            "open('/tmp/wikitext.txt', 'w').write(''.join(pq.read_table(p).column('text').to_pylist())); print(p)\"")
    PERF1 = f"{REPO}/experiments/kaggle_compat/perf1.py"
    for arm, prefix in (("stock", ""), ("kernel", "PERF1_ARITH=pinned ")):
        sh(f"gate_qwen3-14b_{arm}_decode", f"{GATE_ENV}{prefix}{PY} {PERF1} nll {path} {arm} decode "
                                          f"/tmp/wikitext.txt {WORK}/gate-qwen3-14b-{arm}-decode.json", timeout=3600)
report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
