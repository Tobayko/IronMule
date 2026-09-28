# GATE-OSS: the numeric-plan quality gates for gpt-oss 20B again, with BOS on every chunk. Its
# PORT2 gates (port2 run 7) ran before the BOS fix: bf16 perplexity 144, chunks moving by up to
# 0.27 nats, float32 1.0037 [0.9400; 1.0647] and float16 1.0088 [0.9486; 1.0666], both
# inconclusive. Private notebook, internet on. Quota: the user asked on 2026-09-28 to qualify it.
# Rules, before the run (PORT2-K's, unchanged):
#   * quality.py at IronMule main 58cbb1d (BOS on every chunk), 16 chunks x 512 tokens of
#     WikiText-2 raw test, one precision per process (QUALITY_ONLY = bf16, float16, float32), the
#     pinned revision of every earlier gpt-oss run. bf16 and float16 run side by side, one card
#     each; float32 after them.
#   * port2k_summary.py judges each plan against bf16 (seed 20260916, 10 000 paired draws): passes
#     at an upper bound <= 1.005 with no non-finite NLL, refused when the lower bound lies above
#     1.005, otherwise inconclusive.
#   * Kill: inconclusive or refused leaves the row as it is, with this run as a footnote; a pass
#     becomes the row's quality evidence in numeric_plans.py. Speed rows stay PORT2's.
import json
import os
import signal
import subprocess
import sys
import time

COMMIT = "58cbb1dadac725c63632399379e9ef15222d16b8"
MODELS = (("gptoss-20b", "mlx-community/gpt-oss-20b-MXFP4-Q4", "f356f2747216d7e98fee755df25987459fc19089"),)
WIKITEXT = ("Salesforce/wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3",
            "wikitext-2-raw-v1/test-00000-of-00001.parquet")
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable
DEADLINE = time.time() + 80 * 60
os.makedirs(f"{WORK}/logs", exist_ok=True)
report = {"schema": "ironmule.gate-oss-kaggle.v1", "commit": COMMIT, "stages": {}, "performance_claim": False}
env = dict(os.environ, PATH=f"{VENV}/bin:" + os.environ["PATH"], PYTHONPATH="", PYTHONNOUSERSITE="1",
           HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONUNBUFFERED="1", IRONMULE_HOME="/tmp/ironmule-home")


def save():
    with open(f"{WORK}/gate-oss-result.json", "w") as stream:
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
sh("install", f"{SYS} -m uv pip install --python {PY} -e '{REPO}[cuda]' 'mlx-lm==0.31.3' pyarrow ", timeout=1200)
sh("freeze", f"{SYS} -m uv pip freeze --python {PY}")
sh("download_wikitext", f"{PY} -c \"from huggingface_hub import hf_hub_download as d; import pyarrow.parquet as pq; "
                        f"p = d('{WIKITEXT[0]}', '{WIKITEXT[2]}', repo_type='dataset', revision='{WIKITEXT[1]}'); "
                        "open('/tmp/wikitext.txt', 'w').write(''.join(pq.read_table(p).column('text').to_pylist())); print(p)\"")


QUALITY = f"{REPO}/experiments/kaggle_compat/quality.py"
for key, model_id, revision in MODELS:
    code, _ = sh(f"download_{key}", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                                    f"print(s('{model_id}', revision='{revision}'))\"", timeout=1200)
    if code != 0:
        continue
    def gate(precision, card):
        return (f"CUDA_VISIBLE_DEVICES={card} QUALITY_ONLY={precision} {PY} {QUALITY} {model_id} {revision} "
                f"/tmp/wikitext.txt {WORK}/quality-{key}-{precision}.json 16 512 > {WORK}/logs/quality_{key}_{precision}.log 2>&1")
    # 11.2 GB does not fit twice on one card: bf16 and float16 side by side, one card each.
    sh(f"quality_{key}_bf16+float16", f"({gate('bf16', 0)}) & ({gate('float16', 1)}); wait", timeout=3000)
    sh(f"quality_{key}_float32", gate("float32", 1), timeout=2400)

sh("summary", f"{PY} {REPO}/experiments/kaggle_compat/port2k_summary.py {WORK} {WORK}/gate-oss-summary.json")
report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
