# DEMO5: Qwen3.8 27B for a show video. It does not fit one T4 (PORT2), so it runs as PERF1-O's layer
# pipeline over the cell's two T4s (`perf1.py pipelined`, one card per `mlx.launch` rank), which
# IronMule's single-process Runtime cannot do; this is the research harness, not the product.
# Private notebook, internet on. Quota: the user asked for this model on 2026-09-28; capped at 40 min.
#   * Stock (the bf16 checkpoint as loaded) against `kernel+p16`, the native plan's kernels: the
#     4-bit row kernel for decode and the float16 tensor-core prefill, with PERF1_P16_SYNC=1. Both
#     with MLX_USE_CUDA_GRAPHS=0: graphs make this family non-deterministic on CUDA (run 13), where
#     this pair measured 1.911 -> 8.819 tok/s. Speed only; no quality gate has run on this model.
#   * One chat question in Qwen's direct mode (`enable_thinking=False`), up to 64 tokens or the end
#     of turn, one warm-up and three recorded answers per arm, every token timed on the host as it
#     arrives (a synchronous loop, the same for both arms). One question only: a grouped race needs
#     the Runtime, which cannot split a model over two cards.
#   * A demo, not a measurement: the ledger's paired runs are the evidence.
import json
import os
import signal
import subprocess
import sys
import time

COMMIT = "58cbb1dadac725c63632399379e9ef15222d16b8"
MODEL = ("mlx-community/Qwen3.8-27B-4bit", "10c35caafbb80f7dc6a7a432cdd11af10a6d4818")
PERF1_SOURCE = __PERF1_SOURCE__  # experiments/kaggle_compat/perf1.py, embedded at submit time
QUESTION, MAX_TOKENS, REPS = "Explain in two sentences why the sky is blue.", 64, 3
RECORDER = '''
import json, sys, time
sys.path.insert(0, "/tmp")
import perf1  # picks this rank's card before MLX starts
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

path, arm, question, max_tokens, reps, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
model, tokenizer, pipe = perf1.pipelined(path)
perf1.apply_arm(model, arm)
rendered = tokenizer.apply_chat_template([{"role": "user", "content": question}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
ids = list(tokenizer.encode(rendered, add_special_tokens=False))
eos = set(tokenizer.eos_token_ids)


def answer():
    cache = make_prompt_cache(model)
    began = time.perf_counter()
    y = mx.argmax(model(mx.array(ids)[None, :], cache=cache)[:, -1, :], axis=-1)
    tokens, times = [int(y.item())], [(time.perf_counter() - began) * 1000]
    while len(tokens) < max_tokens and tokens[-1] not in eos:
        y = mx.argmax(model(y.reshape(1, 1), cache=cache)[:, -1, :], axis=-1)
        tokens.append(int(y.item()))
        times.append((time.perf_counter() - began) * 1000)
    visible = [t for t in tokens if t not in eos]
    return {"prompt": question, "tokens": tokens, "first_ms": round(times[0], 1), "total_ms": round(times[-1], 1),
            "usage": {"completion_tokens": len(visible)}, "text": tokenizer.decode(visible), "cumulative": True,
            "chunks": [[round(times[k - 1], 1), tokenizer.decode(visible[:k])] for k in range(1, len(visible) + 1)]}


answer()  # warm-up
reps = [answer() for _ in range(reps)]
print(arm, pipe["rank"], [r["total_ms"] for r in reps], flush=True)
if pipe["rank"] == 0:
    json.dump({"arm": arm, "reps": reps, "prompt_tokens": len(ids), "pipeline": pipe,
               "peak_gb": mx.get_peak_memory() / 1e9, "routed": dict(perf1.routed)}, open(out, "w"))
'''
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable
PIPE = f"{VENV}/bin/mlx.launch --hosts 127.0.0.1 -n 2 --backend ring -- {PY} /tmp/race5.py"
DEADLINE = time.time() + 40 * 60
os.makedirs(f"{WORK}/logs", exist_ok=True)
with open("/tmp/perf1.py", "w") as stream:
    stream.write(PERF1_SOURCE)
with open("/tmp/race5.py", "w") as stream:
    stream.write(RECORDER)
report = {"schema": "ironmule.demo5-kaggle.v1", "commit": COMMIT, "model": MODEL, "label": "Qwen3.8 27B 4-bit",
          "hardware": "2 × NVIDIA TESLA T4", "plan_status": "unqualified", "performance_claim": False,
          "config": {"dtype": "native", "knobs": {}, "single": {"prompt": QUESTION, "max_tokens": MAX_TOKENS}},
          "sides": [["STOCK MLX", "layers split over two T4s"], ["IRONMULE ON", "native kernels, same split"]],
          "notes": ["Stock MLX and IronMule's native 4-bit kernels (its research harness), the model split by layers",
                    "over the two T4s of one Kaggle cell, CUDA graphs off. Speed only: no quality gate has run on",
                    "this model, and the kernels change the arithmetic"],
          "stages": {}}
env = dict(os.environ, PATH=f"{VENV}/bin:" + os.environ["PATH"], PYTHONPATH="", PYTHONNOUSERSITE="1",
           HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONUNBUFFERED="1", MLX_USE_CUDA_GRAPHS="0", PERF1_P16_SYNC="1")


def save():
    with open(f"{WORK}/demo5-result.json", "w") as stream:
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


sh("env", "nvidia-smi --query-gpu=index,name,memory.total,clocks.max.sm --format=csv; nproc; free -g; df -h / | tail -1")
sh("clone", f"git clone -q https://github.com/Tobayko/IronMule {REPO} && git -C {REPO} checkout -q {COMMIT} "
            f"&& git -C {REPO} rev-parse HEAD")
sh("venv", f"{SYS} -m pip install -q uv && {SYS} -m uv python install 3.12 "
           f"&& {SYS} -m uv venv --python-preference only-managed --python 3.12 {VENV}")
sh("install", f"{SYS} -m uv pip install --python {PY} -e '{REPO}[cuda]' 'mlx-lm==0.31.3'", timeout=1200)
sh("freeze", f"{SYS} -m uv pip freeze --python {PY}")
code, out = sh("download", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                           f"print(s('{MODEL[0]}', revision='{MODEL[1]}'))\"", timeout=1500)
if code == 0:
    path = out.strip().splitlines()[-1]
    # `mlx.launch` exits 0 even when its ranks fail (run 11): a stage counts by its result file.
    for side, arm in (("off", "stock"), ("on", "kernel+p16")):
        sh(f"race_{side}", f"{PIPE} {path} {arm} '{QUESTION}' {MAX_TOKENS} {REPS} {WORK}/race-{side}.json", timeout=1200)
        report["stages"][f"race_{side}"]["output"] = os.path.exists(f"{WORK}/race-{side}.json")
        save()
report["finished"] = True
save()
print(json.dumps({k: (v.get("exit"), v.get("output")) for k, v in report["stages"].items()}, indent=1))
