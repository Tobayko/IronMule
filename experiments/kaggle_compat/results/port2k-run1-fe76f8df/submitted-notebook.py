# PORT2-K: the Gemma numeric-plan gates again, with BOS on every chunk, plus the engine suite on
# this commit. Private notebook, internet on. Quota: the user delegated all decisions on
# 2026-09-26 and asked that tests run on Kaggle; budget is the 30 h Kaggle week (0 h used).
# Rules, before the run (docs/PROJECT_FRIDAY_BACKLOG.md PORT2-K):
#   * The engine suite as pytest.ini configures it, not integration, on this commit.
#   * quality.py (BOS on every chunk since 3b830de), 16 chunks x 512 tokens of WikiText-2 raw
#     test, one precision per process (QUALITY_ONLY = bf16, float32, float16), for the two
#     models whose gates ran without BOS: Gemma 3 4B (float32 port2 run 2, float16 port2 run 6)
#     and Gemma 4 E2B (both plans, port2 run 9b). Same pinned revisions as those runs.
#   * port2k_summary.py judges each plan against bf16 with tests/test_numeric_plans.py's seed
#     (20260916) and 10 000 draws: passes at an upper bound <= 1.005 with no non-finite NLL,
#     refused when the lower bound lies above 1.005, otherwise inconclusive.
#   * Kill (the entry's): no verdict changes -> a footnote; a changed verdict -> a new row in
#     numeric_plans.py with this run as its evidence. Old verdicts stay recorded as they were.
import json
import os
import signal
import subprocess
import sys
import time

COMMIT = "3b830dec74c39a75247bd71c35d33aa0350ba90f"
MODELS = (("gemma3-4b", "mlx-community/gemma-3-4b-it-4bit", "93724907d4ed1745d2fe50baadf3b0b01a65abf2"),
          ("gemma4-e2b", "mlx-community/gemma-4-e2b-it-4bit", "238767527555cb75a05732a84dff5d6ba0dd6809"))
WIKITEXT = ("Salesforce/wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3",
            "wikitext-2-raw-v1/test-00000-of-00001.parquet")
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable
DEADLINE = time.time() + 110 * 60
os.makedirs(f"{WORK}/logs", exist_ok=True)
report = {"schema": "ironmule.port2k-kaggle.v1", "commit": COMMIT, "stages": {}, "performance_claim": False}
env = dict(os.environ, PATH=f"{VENV}/bin:" + os.environ["PATH"], PYTHONPATH="", PYTHONNOUSERSITE="1",
           HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONUNBUFFERED="1", IRONMULE_HOME="/tmp/ironmule-home")


def save():
    with open(f"{WORK}/port2k-result.json", "w") as stream:
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
sh("download_wikitext", f"{PY} -c \"from huggingface_hub import hf_hub_download as d; import pyarrow.parquet as pq; "
                        f"p = d('{WIKITEXT[0]}', '{WIKITEXT[2]}', repo_type='dataset', revision='{WIKITEXT[1]}'); "
                        "open('/tmp/wikitext.txt', 'w').write(''.join(pq.read_table(p).column('text').to_pylist())); print(p)\"")

sh("pytest_engine", f"{PY} -m pytest tests/engine -m 'not integration' -rfEs -p no:cacheprovider "
                    f"--junitxml={WORK}/pytest_engine.xml", timeout=1800, cwd=REPO)

QUALITY = f"{REPO}/experiments/kaggle_compat/quality.py"
for key, model_id, revision in MODELS:
    code, _ = sh(f"download_{key}", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                                    f"print(s('{model_id}', revision='{revision}'))\"", timeout=1200)
    if code != 0:
        continue
    for precision in ("bf16", "float32", "float16"):
        sh(f"quality_{key}_{precision}",
           f"QUALITY_ONLY={precision} {PY} {QUALITY} {model_id} {revision} /tmp/wikitext.txt "
           f"{WORK}/quality-{key}-{precision}.json 16 512", timeout=1800)

sh("summary", f"{PY} {REPO}/experiments/kaggle_compat/port2k_summary.py {WORK} {WORK}/port2k-summary.json")
report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
