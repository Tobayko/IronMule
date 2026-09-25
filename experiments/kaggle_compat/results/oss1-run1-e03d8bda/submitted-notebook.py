# OSS1: is gpt-oss 20B's quality-gate spread the bf16 reference's expert routing?
# Private notebook, internet on. Quota: the user asked for this run on 2026-09-25 (budget: the
# 30 h Kaggle week); stages are capped at 45 min. Rules (docs/PROJECT_FRIDAY_BACKLOG.md, OSS1):
#   * `moe_routing.py` from the cloned commit, one fresh process per dtype, in this order:
#     bf16 (A), float32, bf16 (B, the A/A control), float16.
#   * WikiText-2 raw test at the pinned dataset revision, sliced exactly as `quality.py`
#     slices it; chunks 1, 8, 9 and 11, chosen before the run from PORT2 run 7: the smallest
#     and the three largest |dNLL| between bf16 and float32 there.
#   * A process starts only with its estimated duration left (bf16 12 min, the plans 6 min).
import json
import os
import signal
import subprocess
import sys
import time

COMMIT = "eeae3f1beb3bc2d2af5960c655a219df67e90045"
MODEL = ("mlx-community/gpt-oss-20b-MXFP4-Q4", "f356f2747216d7e98fee755df25987459fc19089")
WIKITEXT = ("Salesforce/wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3",
            "wikitext-2-raw-v1/test-00000-of-00001.parquet")
CHUNKS = "1 8 9 11"
ORDER = [("bf16_a", "bf16", 720), ("float32", "float32", 360), ("bf16_b", "bf16", 720),
         ("float16", "float16", 360)]
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable
DEADLINE = time.time() + 45 * 60
os.makedirs(f"{WORK}/logs", exist_ok=True)
report = {"schema": "ironmule.oss1-kaggle.v1", "commit": COMMIT, "stages": {}, "performance_claim": False}
env = dict(os.environ, PATH=f"{VENV}/bin:" + os.environ["PATH"], PYTHONPATH="", PYTHONNOUSERSITE="1",
           HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONUNBUFFERED="1", IRONMULE_HOME="/tmp/ironmule-home")


def save():
    with open(f"{WORK}/oss1-result.json", "w") as stream:
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
sh("install", f"{SYS} -m uv pip install --python {PY} -e '{REPO}[cuda]' 'mlx-lm==0.31.3' pyarrow", timeout=1200)
sh("freeze", f"{SYS} -m uv pip freeze --python {PY}")
sh("download_model", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                    f"print(s('{MODEL[0]}', revision='{MODEL[1]}'))\"", timeout=1200)
sh("download_wikitext", f"{PY} -c \"from huggingface_hub import hf_hub_download as d; import pyarrow.parquet as pq; "
                        f"p = d('{WIKITEXT[0]}', '{WIKITEXT[2]}', repo_type='dataset', revision='{WIKITEXT[1]}'); "
                        "open('/tmp/wikitext.txt', 'w').write(''.join(pq.read_table(p).column('text').to_pylist())); print(p)\"")
for name, dtype, estimate in ORDER:
    if DEADLINE - time.time() < estimate:
        report["stages"][f"routing_{name}"] = {"exit": "skipped_deadline"}
        save()
        continue
    sh(f"routing_{name}", f"{PY} {REPO}/experiments/kaggle_compat/moe_routing.py {MODEL[0]} {MODEL[1]} "
                          f"/tmp/wikitext.txt {WORK}/routing-{name}.json {dtype} {CHUNKS}", timeout=1500)
report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
