# PORT2-K run 2: Gemma 3 4B `float16` with BOS on every chunk, measured past the loader that
# refuses it, plus the engine suite on this commit. Private notebook, internet on. Quota: the user
# delegated all decisions on 2026-09-26 and asked that tests run on Kaggle (30 h week).
# Rules, before the run (docs/PROJECT_FRIDAY_BACKLOG.md, PORT2-K Rest):
#   * quality.py at the commit below, BOS on every chunk, 16 x 512 tokens of WikiText-2 raw test,
#     QUALITY_LOADER=mlx_lm (mlx-lm's load plus `set_dtype`, as load_engine applies the plan,
#     without its refusal), one process each for bf16 and float16, Gemma 3 4B at the pinned
#     revision of run 1.
#   * Loader check: this run's bf16 per-chunk NLL must equal run 1's (load_engine) bf16 file to
#     1e-6 on every chunk, else the loaders differ and the float16 verdict is not recorded.
#   * port2k_summary.py judges float16 against this run's bf16 (seed 20260916, 10 000 draws).
#     Kill (the entry's): upper bound > 1.005 -> the refusal stays with a BOS gate as evidence;
#     otherwise the row becomes recommended.
#   * Diagnosis (PORT2-K Rest 2, no gate): Gemma 4 E2B's bf16 reference scores perplexity 355.7
#     with BOS at 4 bit. The same chunks through the 8-bit and the bf16 checkpoint of the same
#     model (pinned below) say whether the 4-bit quantisation explains it. Recorded, not judged.
import json
import os
import signal
import subprocess
import sys
import time

COMMIT = "2175a345e3bcea5ca768c3e96a6dedb83734270b"
MODEL = ("mlx-community/gemma-3-4b-it-4bit", "93724907d4ed1745d2fe50baadf3b0b01a65abf2")
DIAGNOSIS = (("gemma4-e2b-8bit", "mlx-community/gemma-4-e2b-it-8bit", "03dcf209f3f549b4075e7191e77cf69b3d48e1b2"),
             ("gemma4-e2b-bf16w", "mlx-community/gemma-4-e2b-it-bf16", "fb0b166bbb9a0eb4b37915bfc515a197c9122f39"))
RUN1_BF16 = "experiments/kaggle_compat/results/port2k-run1-fe76f8df/quality-gemma3-4b-bf16.json"
WIKITEXT = ("Salesforce/wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3",
            "wikitext-2-raw-v1/test-00000-of-00001.parquet")
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable
DEADLINE = time.time() + 90 * 60
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

code, _ = sh("download_gemma3-4b", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                                   f"print(s('{MODEL[0]}', revision='{MODEL[1]}'))\"", timeout=1200)
if code == 0:
    for precision in ("bf16", "float16"):
        sh(f"quality_gemma3-4b_{precision}",
           f"QUALITY_LOADER=mlx_lm QUALITY_ONLY={precision} {PY} {REPO}/experiments/kaggle_compat/quality.py "
           f"{MODEL[0]} {MODEL[1]} /tmp/wikitext.txt {WORK}/quality-gemma3-4b-{precision}.json 16 512", timeout=1800)
    try:
        mine = json.load(open(f"{WORK}/quality-gemma3-4b-bf16.json"))["rows"]
        run1 = json.load(open(f"{REPO}/{RUN1_BF16}"))["rows"]
        diffs = [abs(a["nll_bf16"] - b["nll_bf16"]) for a, b in zip(mine, run1)]
        report["loader_check"] = {"chunks": len(diffs), "max_abs_diff": max(diffs),
                                  "passes": len(diffs) == 16 and max(diffs) <= 1e-6}
    except (OSError, ValueError, KeyError) as exc:
        report["loader_check"] = {"passes": False, "error": type(exc).__name__}
    save()
    sh("summary", f"{PY} {REPO}/experiments/kaggle_compat/port2k_summary.py {WORK} {WORK}/port2k-summary.json")
for key, model_id, revision in DIAGNOSIS:
    code, _ = sh(f"download_{key}", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                                    f"print(s('{model_id}', revision='{revision}'))\"", timeout=1500)
    if code == 0:
        sh(f"quality_{key}_bf16",
           f"QUALITY_LOADER=mlx_lm QUALITY_ONLY=bf16 {PY} {REPO}/experiments/kaggle_compat/quality.py "
           f"{model_id} {revision} /tmp/wikitext.txt {WORK}/diagnosis-{key}.json 16 512", timeout=1800)
report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
print(json.dumps(report.get("loader_check"), indent=1))
