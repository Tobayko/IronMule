# PERF1 run 11: models over 16 GB, split by layers across the cell's two T4s. Private notebook,
# internet on. Free quota: one run, <= 115 min. Backlog: PERF1-O, and PERF1-K on a larger model.
# Nothing from runs 1-10 is repeated except the in-run controls named below.
#
# One card holds 15360 MiB and 16.05 GB does not load (PORT2). Tensor parallelism halves the
# weights per card but reduces twice per layer per token over TCP (0.34x, run 1), and
# `sharded_load` ran out of memory on Qwen3.8 27B because it materialises whole weights before
# slicing them. `perf1.py`'s layer pipeline gives each rank half the layers, reads only those
# from disk, and hands over once per token. Checked on a Mac before submission (ring over
# localhost, one Metal device): Qwen 3 0.6B and Qwen3.8 27B produce the same 128 tokens
# pipelined as in one process, on both ranks, and the batch server completes pipelined.
# Rules fixed here:
#   * gate: Qwen 3 8B `kernel+p16` pipelined must reproduce the single-card arm's 128 tokens;
#     otherwise every later stage is skipped (PERF1-O kill criterion);
#   * each model: stock against `kernel+p16`, 512-token prompt, 128 greedy tokens, one warm and
#     three measured generations (perf1.py `e2e`); Qwen 3.5-family models use a 256-token prompt,
#     because their bf16 path returns non-finite logits at 512 on CUDA (PORT2, upstream);
#   * Qwen 3 32B server, 8 requests, widths 1 and 8: `kernel+mma+p16` against the in-run control
#     `kernel+p16`; `mma` counts for PERF1-K if width 8 is >= 1.2x the control's width 8;
#   * the same four arms on one card for Mistral 3.2 24B (13.26 GB, the largest dense model
#     that fits one T4, PORT2), independent of the gate; if `kernel+p16` runs out of memory
#     (it dequantises the 131072-row head to float16 for the prompt), `kernel` alone runs.
import json
import os
import signal
import subprocess
import sys
import time

COMMIT = "69f9937370456a18740213555e9bcd752ce60dd6"  # origin/main
PERF1_SOURCE = __PERF1_SOURCE__  # experiments/kaggle_compat/perf1.py
MODELS = {"qwen3-8b": ("mlx-community/Qwen3-8B-4bit", "545dc4251c05440727734bcd94334791f6ab0192", 512),
          "qwen3-32b": ("mlx-community/Qwen3-32B-4bit", "bcaaf7f538adf166c1080a2befdb4f6019f66639", 512),
          "qwen38-27b": ("mlx-community/Qwen3.8-27B-4bit", "10c35caafbb80f7dc6a7a432cdd11af10a6d4818", 256),
          "qwen36-35b-a3b": ("mlx-community/Qwen3.6-35B-A3B-4bit", "38740b847e4cb78f352aba30aa41c76e08e6eb46", 256),
          "mistral-24b": ("mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit",
                          "2a1d5eabfc504747bdc24178394821a1efc0edde", 512)}
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable
PIPE = f"{VENV}/bin/mlx.launch --hosts 127.0.0.1 -n 2 --backend ring -- {PY} /tmp/perf1.py"
DEADLINE = time.time() + 115 * 60
os.makedirs(f"{WORK}/logs", exist_ok=True)
with open("/tmp/perf1.py", "w") as stream:
    stream.write(PERF1_SOURCE)
report = {"schema": "ironmule.perf1-kaggle.v11", "commit": COMMIT, "stages": {}, "performance_claim": False}
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

path, _ = download("qwen3-8b")
if path:
    sh("e2e_qwen3-8b_single", f"{PY} /tmp/perf1.py e2e {path} kernel+p16 {WORK}/e2e-qwen3-8b-single.json")
    sh("e2e_qwen3-8b_pipe", f"{PIPE} e2e {path} kernel+p16 {WORK}/e2e-qwen3-8b-pipe.json")
single = tokens(f"{WORK}/e2e-qwen3-8b-single.json")
report["gate"] = {"single_tokens": bool(single),
                  "rank0_identical": single is not None and tokens(f"{WORK}/e2e-qwen3-8b-pipe.json") == single,
                  "rank1_identical": single is not None and tokens(f"{WORK}/e2e-qwen3-8b-pipe-rank1.json") == single}
report["gate"]["pass"] = report["gate"]["rank0_identical"] and report["gate"]["rank1_identical"]
save()

server_path = None
if report["gate"]["pass"]:
    for key in ("qwen3-32b", "qwen38-27b", "qwen36-35b-a3b"):
        path, prompt = download(key)
        if not path:
            continue
        for arm in ("stock", "kernel+p16"):
            sh(f"e2e_{key}_{arm}", f"PERF1_PROMPT_TOKENS={prompt} {PIPE} e2e {path} {arm} {WORK}/e2e-{key}-{arm}.json",
               timeout=1500)
        if key == "qwen3-32b":
            server_path = path
    if server_path:
        for arm in ("kernel+p16", "kernel+mma+p16"):
            sh(f"server_qwen3-32b_{arm}", f"{PIPE} server {server_path} {arm} 1,8 {WORK}/server-qwen3-32b-{arm}.json",
               timeout=1200)

path, _ = download("mistral-24b")
if path:
    sh("e2e_mistral-24b_stock", f"{PY} /tmp/perf1.py e2e {path} stock {WORK}/e2e-mistral-24b-stock.json", timeout=1500)
    decode = "kernel+p16"
    code, _ = sh("e2e_mistral-24b_kernel+p16", f"{PY} /tmp/perf1.py e2e {path} kernel+p16 {WORK}/e2e-mistral-24b-kernel+p16.json")
    if code != 0:
        decode = "kernel"
        sh("e2e_mistral-24b_kernel", f"{PY} /tmp/perf1.py e2e {path} kernel {WORK}/e2e-mistral-24b-kernel.json")
    for arm in (decode, decode.replace("kernel", "kernel+mma")):
        sh(f"server_mistral-24b_{arm}", f"{PY} /tmp/perf1.py server {path} {arm} 1,8 {WORK}/server-mistral-24b-{arm}.json",
           timeout=1200)

report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
