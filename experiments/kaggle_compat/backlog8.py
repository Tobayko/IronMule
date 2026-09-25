# BACKLOG8: the engine suite on the merged tree (research/port2-model-families, including the
# MoE row kernel) and PERF1-Y, the quality gate for Gemma 3 12B under `native`. Private notebook,
# internet on. Quota: the user asked on 2026-09-25 to work through the backlog (budget: the 30 h
# Kaggle week) and left the open decisions to the agent. Rules, before the run:
#   * The engine suite as pytest.ini configures it, not integration, on this commit.
#   * PERF1-Y (docs/PROJECT_FRIDAY_BACKLOG.md): `perf1.py nll`, 16 chunks x 512 tokens of
#     WikiText-2 raw test, BOS on every chunk (perf1.py sets it since the port2 branch), the
#     native gates' arms: decode path stock bf16 against `kernel` with pinned arithmetic, prefill
#     path stock bf16 against `p16`. Pass per path: upper bound of the 10 000-sample chunk
#     bootstrap (seed 20260915) <= 1.005 and no non-finite value. Kill: an upper bound above
#     1.005 refuses `native` for `mlx_lm.models.gemma3_text` (a table row); float32 stays the plan.
#     Prefill first (minutes), then the decode path, stock (about 45 min) before the kernel.
import json
import os
import signal
import subprocess
import sys
import time

COMMIT = __COMMIT__
GEMMA12B = ("mlx-community/gemma-3-12b-it-4bit", "86cc6a8dedbc456dd0e4af01a9d09f396f77e558")
WIKITEXT = ("Salesforce/wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3",
            "wikitext-2-raw-v1/test-00000-of-00001.parquet")
GATE_ENV = "PERF1_NLL_CHUNKS=16 PERF1_NLL_TOKENS=512 "
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable
DEADLINE = time.time() + 110 * 60
os.makedirs(f"{WORK}/logs", exist_ok=True)
report = {"schema": "ironmule.backlog8-kaggle.v1", "commit": COMMIT, "stages": {}, "performance_claim": False}
env = dict(os.environ, PATH=f"{VENV}/bin:" + os.environ["PATH"], PYTHONPATH="", PYTHONNOUSERSITE="1",
           HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONUNBUFFERED="1", IRONMULE_HOME="/tmp/ironmule-home")


def save():
    with open(f"{WORK}/backlog8-result.json", "w") as stream:
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

# PERF1-Y: Gemma 3 12B under `native`, both paths the plan changes.
path = download("gemma3-12b", GEMMA12B)
if path:
    PERF1 = f"{REPO}/experiments/kaggle_compat/perf1.py"
    for path_mode, arms in (("prefill", (("stock", ""), ("p16", ""))),
                            ("decode", (("stock", ""), ("kernel", "PERF1_ARITH=pinned ")))):
        for arm, prefix in arms:
            out = f"{WORK}/gate-gemma3-12b-{arm}-{path_mode}.json"
            sh(f"gate_gemma3-12b_{arm}_{path_mode}", f"{GATE_ENV}{prefix}{PY} {PERF1} nll {path} {arm} "
                                                     f"{path_mode} /tmp/wikitext.txt {out}", timeout=4200)
report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
