# PERF1 run 13: fix what runs 11-12 left broken. Private notebook, internet on. Free quota: one
# run, <= 70 min (week total stays under 10 h). Backlog: PERF1-P, PERF1-Q, PERF1-K.
#
# 1. `p16` ran out of memory for Mistral 3.2 24B on one card, and Qwen3.8 27B's `kernel+p16`
#    died in `cudaGraphInstantiate`. Run 12's peaks explain both: 8B and 32B `p16` sat about
#    3 GB above their weights, because every float16 copy of a prefill lives until its command
#    buffer ends, and a 131072- or 248320-row head is four to eight such copies. PERF1_P16_SYNC
#    evaluates each copy's product at once. On the Mac (Qwen3.8 27B, 512 tokens, diagnostic
#    only) that cut the peak from 18.90 to 16.42 GB with identical tokens.
# 2. Qwen3.8 27B's stock arm gave different tokens per repetition, and the old check threw the
#    evidence away. `measure` now records every repetition and whether the prefill logits were
#    finite. On CUDA mlx-lm runs Qwen 3.5's gated delta as a Python loop per token (its Metal
#    kernel is Mac-only), a very large graph; CUDA graphs are the first suspect.
# Rules fixed here:
#   * Mistral 24B, one card: `kernel` (in-run control) against `kernel+p16` with PERF1_P16_SYNC=1;
#     if that leaves no result, one more rung adds 8192-row slices and a zero cache limit. The
#     server, widths 1 and 8: `kernel+mma+p16` against `kernel+mma`, same settings;
#   * Qwen3.8 27B, two cards, 256 tokens: stock with CUDA graphs on and off. Graphs are the cause
#     if off is deterministic and on is not. `kernel+p16` with PERF1_P16_SYNC=1 then runs with
#     the setting that was deterministic (on, if both were);
#   * Qwen 3.5 9B, one card, stock, 256 tokens, graphs on, before the 27B: PERF1-Q's control.
#     Non-deterministic there too means upstream, not the pipeline.
import json
import os
import signal
import subprocess
import sys
import time

COMMIT = "69f9937370456a18740213555e9bcd752ce60dd6"  # origin/main
PERF1_SOURCE = __PERF1_SOURCE__  # experiments/kaggle_compat/perf1.py
MODELS = {"mistral-24b": ("mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit",
                          "2a1d5eabfc504747bdc24178394821a1efc0edde", 512),
          "qwen38-27b": ("mlx-community/Qwen3.8-27B-4bit", "10c35caafbb80f7dc6a7a432cdd11af10a6d4818", 256),
          "qwen35-9b": ("mlx-community/Qwen3.5-9B-MLX-4bit", "938d8919941c6e7efd3c7150eff7fe9d12afa631", 256)}
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable
PIPE = f"{VENV}/bin/mlx.launch --hosts 127.0.0.1 -n 2 --backend ring -- {PY} /tmp/perf1.py"
DEADLINE = time.time() + 70 * 60
os.makedirs(f"{WORK}/logs", exist_ok=True)
with open("/tmp/perf1.py", "w") as stream:
    stream.write(PERF1_SOURCE)
report = {"schema": "ironmule.perf1-kaggle.v13", "commit": COMMIT, "stages": {}, "performance_claim": False}
env = dict(os.environ, PATH=f"{VENV}/bin:" + os.environ["PATH"], PYTHONPATH="", PYTHONNOUSERSITE="1",
           HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONUNBUFFERED="1", MLX_MAX_OPS_PER_BUFFER="400",
           IRONMULE_HOME="/tmp/ironmule-home")


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


def run(name, cmd, out, timeout=900):
    """A stage with a result file; `mlx.launch` exits 0 even when its ranks fail (run 11)."""
    sh(name, cmd, timeout)
    if name in report["stages"]:
        report["stages"][name]["output"] = os.path.exists(out)
        save()
    return os.path.exists(out)


def tokens(path):
    try:
        with open(path) as stream:
            return json.load(stream)["tokens"]
    except (OSError, ValueError, KeyError):
        return None


def download(key):
    model_id, revision, prompt = MODELS[key]
    code, out = sh(f"download_{key}", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                                      f"print(s('{model_id}', revision='{revision}'))\"", timeout=1200)
    return (out.strip().splitlines()[-1], prompt) if code == 0 else (None, prompt)


sh("env", "nvidia-smi --query-gpu=index,name,memory.total,clocks.max.sm --format=csv; nproc; free -g; df -h / | tail -1")
sh("clone", f"git clone -q https://github.com/Tobayko/IronMule {REPO} && git -C {REPO} checkout -q {COMMIT} "
            f"&& git -C {REPO} status --short")
sh("venv", f"{SYS} -m pip install -q uv && {SYS} -m uv python install 3.12 "
           f"&& {SYS} -m uv venv --python-preference only-managed --python 3.12 {VENV}")
sh("install", f"{SYS} -m uv pip install --python {PY} -e '{REPO}[cuda]' 'mlx-lm==0.31.3'", timeout=1200)
sh("freeze", f"{SYS} -m uv pip freeze --python {PY}")



def result(path):
    try:
        with open(path) as stream:
            return json.load(stream)
    except (OSError, ValueError):
        return None


path, _ = download("mistral-24b")
if path:
    run("e2e_mistral-24b_kernel", f"{PY} /tmp/perf1.py e2e {path} kernel {WORK}/e2e-mistral-24b-kernel.json",
        f"{WORK}/e2e-mistral-24b-kernel.json", timeout=1200)
    low = "PERF1_P16_SYNC=1"
    out = f"{WORK}/e2e-mistral-24b-kernel+p16.json"
    if not run("e2e_mistral-24b_kernel+p16", f"{low} {PY} /tmp/perf1.py e2e {path} kernel+p16 {out}", out):
        low = "PERF1_P16_SYNC=1 PERF1_P16_ROWS=8192 PERF1_CACHE_LIMIT=0"
        out = f"{WORK}/e2e-mistral-24b-kernel+p16-rung2.json"
        run("e2e_mistral-24b_kernel+p16_rung2", f"{low} {PY} /tmp/perf1.py e2e {path} kernel+p16 {out}", out)
    report["mistral_p16_settings"] = low
    for arm in ("kernel+mma+p16", "kernel+mma"):
        out = f"{WORK}/server-mistral-24b-{arm}.json"
        run(f"server_mistral-24b_{arm}", f"{low} {PY} /tmp/perf1.py server {path} {arm} 1,8 {out}", out, timeout=1200)

path, prompt = download("qwen35-9b")
if path:
    out = f"{WORK}/e2e-qwen35-9b-stock.json"
    run("e2e_qwen35-9b_stock", f"PERF1_PROMPT_TOKENS={prompt} {PY} /tmp/perf1.py e2e {path} stock {out}", out)

path, prompt = download("qwen38-27b")
if path:
    for graphs in ("1", "0"):
        out = f"{WORK}/e2e-qwen38-27b-stock-graphs{graphs}.json"
        run(f"e2e_qwen38-27b_stock_graphs{graphs}", f"MLX_USE_CUDA_GRAPHS={graphs} PERF1_PROMPT_TOKENS={prompt} "
            f"{PIPE} e2e {path} stock {out}", out, timeout=1200)
    on, off = (result(f"{WORK}/e2e-qwen38-27b-stock-graphs{g}.json") for g in ("1", "0"))
    graphs = "0" if (off or {}).get("repetitions_identical") and not (on or {}).get("repetitions_identical") else "1"
    out = f"{WORK}/e2e-qwen38-27b-kernel+p16-graphs{graphs}.json"
    run("e2e_qwen38-27b_kernel+p16", f"MLX_USE_CUDA_GRAPHS={graphs} PERF1_P16_SYNC=1 PERF1_PROMPT_TOKENS={prompt} "
        f"{PIPE} e2e {path} kernel+p16 {out}", out, timeout=900)

report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
