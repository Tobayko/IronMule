# BACKLOG1: three open CUDA backlog entries in one Kaggle session. Private notebook, internet on.
# Quota: the user asked on 2026-09-25 to work through the backlog (budget: the 30 h Kaggle
# week); stages are capped at 100 min, in this order of value. Rules are the entries' own
# (docs/PROJECT_FRIDAY_BACKLOG.md), fixed before the run:
#   * PERF1-M: Qwen 3 14B's decode-path gate, run 5's protocol (`perf1.py nll ... decode`,
#     16 strided chunks x 512 tokens, teacher-forced through the cache), stock bf16 against
#     `kernel` with pinned arithmetic, the arithmetic the product's `native` plan uses. Pass:
#     upper bound of the 10 000-sample chunk bootstrap <= 1.005 and no non-finite value.
#   * PERF1-T: Qwen 3.5 9B with MLX_USE_CUDA_GRAPHS=0 in every process: stock, IronMule
#     interactive and IronMule throughput (`cross.py child`, `ironmule benchmark`'s workload),
#     3 repetitions, arm order rotated. Kill: not deterministic across processes without
#     graphs, then the throughput refusal stays.
#   * PORT1-F: `ironmule tune` on Gemma 3 4B in bf16, once. Kill: the crash reproduces.
import json
import os
import signal
import subprocess
import sys
import time

COMMIT = __COMMIT__
QWEN14 = ("mlx-community/Qwen3-14B-4bit", "a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4")
QWEN35 = ("mlx-community/Qwen3.5-9B-MLX-4bit", "938d8919941c6e7efd3c7150eff7fe9d12afa631")
GEMMA4B = ("mlx-community/gemma-3-4b-it-4bit", "93724907d4ed1745d2fe50baadf3b0b01a65abf2")
WIKITEXT = ("Salesforce/wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3",
            "wikitext-2-raw-v1/test-00000-of-00001.parquet")
GRAPHS_OFF = {"MLX_USE_CUDA_GRAPHS": "0"}
CONFIGS_T = [
    {"name": "stock", "env": {"MLX_MAX_OPS_PER_BUFFER": "20", "MLX_MAX_MB_PER_BUFFER": "100", **GRAPHS_OFF},
     "knobs": {}, "mode": "interactive"},
    {"name": "ironmule_interactive", "env": dict(GRAPHS_OFF), "knobs": {}, "mode": "interactive"},
    {"name": "ironmule_throughput", "env": dict(GRAPHS_OFF), "knobs": {}, "mode": "throughput"},
]
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable
DEADLINE = time.time() + 100 * 60
os.makedirs(f"{WORK}/logs", exist_ok=True)
report = {"schema": "ironmule.backlog1-kaggle.v1", "commit": COMMIT, "stages": {}, "performance_claim": False}
env = dict(os.environ, PATH=f"{VENV}/bin:" + os.environ["PATH"], PYTHONPATH="", PYTHONNOUSERSITE="1",
           HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONUNBUFFERED="1", IRONMULE_HOME="/tmp/ironmule-home")


def save():
    with open(f"{WORK}/backlog1-result.json", "w") as stream:
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


def download(key, model):
    code, out = sh(f"download_{key}", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                                      f"print(s('{model[0]}', revision='{model[1]}'))\"", timeout=1200)
    return out.strip().splitlines()[-1] if code == 0 else None


sh("download_wikitext", f"{PY} -c \"from huggingface_hub import hf_hub_download as d; import pyarrow.parquet as pq; "
                        f"p = d('{WIKITEXT[0]}', '{WIKITEXT[2]}', repo_type='dataset', revision='{WIKITEXT[1]}'); "
                        "open('/tmp/wikitext.txt', 'w').write(''.join(pq.read_table(p).column('text').to_pylist())); print(p)\"")
PERF1 = f"{REPO}/experiments/kaggle_compat/perf1.py"

# PERF1-M: the stock reference first (about 40 min), then the kernel (about 8 min).
path = download("qwen3-14b", QWEN14)
if path:
    for arm, prefix in (("stock", ""), ("kernel", "PERF1_ARITH=pinned ")):
        sh(f"gate_qwen3-14b_{arm}_decode", f"{prefix}{PY} {PERF1} nll {path} {arm} decode /tmp/wikitext.txt "
                                          f"{WORK}/gate-qwen3-14b-{arm}-decode.json", timeout=3600)

# PERF1-T: fresh processes, arm order rotated, stock first in the file as the reference.
CROSS = f"{REPO}/experiments/kaggle_compat/cross.py"
if DEADLINE - time.time() > 25 * 60 and download("qwen35-9b", QWEN35):
    runs = {c["name"]: [] for c in CONFIGS_T}
    for rep in range(3):
        if DEADLINE - time.time() < 9 * 60:
            report["stages"][f"cross_qwen35_rep{rep}"] = {"exit": "skipped_deadline"}
            break
        for config in CONFIGS_T[rep % 3:] + CONFIGS_T[:rep % 3]:
            child_env = {k: v for k, v in env.items() if not k.startswith("MLX_MAX_")}
            child_env.update(config["env"])
            try:
                proc = subprocess.run([PY, CROSS, "child", QWEN35[0], QWEN35[1], json.dumps(config), "1"],
                                      env=child_env, capture_output=True, text=True,
                                      timeout=max(60, min(900, DEADLINE - time.time())))
                row = {"rep": rep, "exit": proc.returncode, "stderr_tail": proc.stderr[-1500:]}
                if proc.returncode == 0:
                    row.update(json.loads(proc.stdout.strip().splitlines()[-1]))
            except subprocess.TimeoutExpired:
                row = {"rep": rep, "exit": "timeout"}
            runs[config["name"]].append(row)
            print("qwen35", config["name"], rep, row.get("median_ms"), row["exit"], flush=True)
            with open(f"{WORK}/cross-qwen35-9b-graphs0.json", "w") as stream:
                json.dump({"schema": "ironmule.port1-cross.v1", "model_id": QWEN35[0], "revision": QWEN35[1],
                           "configs": CONFIGS_T, "reps": rep + 1, "measure": 1, "runs": runs,
                           "performance_claim": False}, stream, indent=1)

# PORT1-F: the bf16 tune that crashed once in run 632b904f.
if DEADLINE - time.time() > 22 * 60 and download("gemma3-4b", GEMMA4B):
    sh("tune_gemma3-4b", f"ironmule tune --model {GEMMA4B[0]}", timeout=1500)
report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
