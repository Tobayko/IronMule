# PERF1 run 5: the quality gate the kernel and the float16 prefill owe, and which IronMule knob
# breaks bf16 token identity on top of the kernel. Private notebook, internet on. Free quota:
# one run, <= 80 min. Backlog: PERF1-F. Nothing from runs 1-4 is repeated.
#
# Gate, fixed here and in PORT2's format: WikiText-2 raw test, 16 chunks x 512 tokens,
# per-chunk mean next-token NLL, perplexity ratio candidate / reference with a 10000-sample
# chunk bootstrap (computed offline from the per-chunk values). Pass: upper bound <= 1.005 and
# no non-finite value. Each part is compared on the path it changes, against stock bf16 on the
# same path:
#   * `p16` (prefill path) for Qwen 3 8B and 14B;
#   * `kernel` (decode path, teacher-forced through the cache) for Qwen 3 8B only — the stock
#     decode reference alone costs ~22 min at 6.3 tok/s; 14B would not fit the budget.
# Knob isolation: `kernel+p16` with only `fuse_projections`, and with only
# `compiled_fixed_cache`, on `ironmule benchmark`'s workload; tokens compared offline with run 4's
# `kernel+p16` baseline (same harness bytes for that arm, deterministic across processes).
import json
import os
import signal
import subprocess
import sys
import time

COMMIT = "b339c817299de173f8a330089afb69973c519353"  # research/port2-model-families
PERF1_SOURCE = __PERF1_SOURCE__  # experiments/kaggle_compat/perf1.py
PATCH_SOURCE = __PATCH_SOURCE__  # git diff -- ironmule/model_identity.py (hub-wide HF blobs)
MODELS = [("qwen3-8b", "mlx-community/Qwen3-8B-4bit", "545dc4251c05440727734bcd94334791f6ab0192"),
          ("qwen3-14b", "mlx-community/Qwen3-14B-4bit", "a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4")]
GATES = {"qwen3-8b": [("stock", "decode"), ("kernel", "decode"), ("stock", "prefill"), ("p16", "prefill")],
         "qwen3-14b": [("stock", "prefill"), ("p16", "prefill")]}
KNOBS = [{"fuse_projections": True}, {"compiled_fixed_cache": True}]
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable
DEADLINE = time.time() + 75 * 60
os.makedirs(f"{WORK}/logs", exist_ok=True)
with open("/tmp/perf1.py", "w") as stream:
    stream.write(PERF1_SOURCE)
with open("/tmp/ironmule.patch", "w") as stream:
    stream.write(PATCH_SOURCE)
report = {"schema": "ironmule.perf1-kaggle.v5", "commit": COMMIT, "stages": {}, "performance_claim": False}
env = dict(os.environ, PATH=f"{VENV}/bin:" + os.environ["PATH"], PYTHONPATH="", PYTHONNOUSERSITE="1",
           HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONUNBUFFERED="1", MLX_MAX_OPS_PER_BUFFER="400",
           IRONMULE_HOME="/tmp/ironmule-home", PERF1_NLL_CHUNKS="16", PERF1_NLL_TOKENS="512")


def save():
    with open(f"{WORK}/perf1-result.json", "w") as stream:
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


sh("env", "nvidia-smi --query-gpu=index,name,memory.total,clocks.max.sm --format=csv; nproc; free -g")
sh("clone", f"git clone -q https://github.com/Tobayko/IronMule {REPO} && git -C {REPO} checkout -q {COMMIT} "
            f"&& git -C {REPO} apply /tmp/ironmule.patch && git -C {REPO} status --short")
sh("venv", f"{SYS} -m pip install -q uv && {SYS} -m uv python install 3.12 "
           f"&& {SYS} -m uv venv --python-preference only-managed --python 3.12 {VENV}")
sh("install", f"{SYS} -m uv pip install --python {PY} -e '{REPO}[cuda]' 'mlx-lm==0.31.3' pyarrow", timeout=1200)
sh("freeze", f"{SYS} -m uv pip freeze --python {PY}")
sh("download_wikitext", f"{PY} -c \"from huggingface_hub import hf_hub_download as d; import pyarrow.parquet as pq; "
                        "p = d('Salesforce/wikitext', 'wikitext-2-raw-v1/test-00000-of-00001.parquet', repo_type='dataset'); "
                        "open('/tmp/wikitext.txt', 'w').write(''.join(pq.read_table(p).column('text').to_pylist())); print(p)\"")

for key, model_id, revision in MODELS:
    code, out = sh(f"download_{key}", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                                      f"print(s('{model_id}', revision='{revision}'))\"", timeout=1200)
    if code != 0:
        continue
    path = out.strip().splitlines()[-1]
    if key == "qwen3-8b":
        for knobs in KNOBS:
            name = "kernel+p16-" + "-".join(knobs) + "-interactive"
            sh(f"ironmule_{key}_{name}", f"{PY} /tmp/perf1.py ironmule {model_id} {revision} 'kernel+p16' "
                                         f"'{json.dumps(knobs)}' interactive {WORK}/ironmule-{key}-{name}.json",
               timeout=900)
    for arm, path_mode in GATES[key]:
        sh(f"gate_{key}_{arm}_{path_mode}", f"{PY} /tmp/perf1.py nll {path} '{arm}' {path_mode} /tmp/wikitext.txt "
                                            f"{WORK}/gate-{key}-{arm}-{path_mode}.json", timeout=2100)

report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
