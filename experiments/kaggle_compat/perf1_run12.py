# PERF1 run 12: run 11 again for what it could not answer, with two corrected methods. Private
# notebook, internet on. Free quota: one run, <= 100 min. Backlog: PERF1-O, PERF1-K.
#
# Run 11 (`perf1-run11-7b29bb97`) passed the pipeline gate (Qwen 3 8B, 128/128 tokens on both
# ranks) and then ran out of memory loading every model over 16 GB, on both ranks, at about
# 10 GB per rank. The ranks chose their card with `mx.Device(gpu, rank)`; loaded weights live
# in managed memory, and where they land is not pinned by that. `perf1.py` now sets
# CUDA_VISIBLE_DEVICES to the rank before MLX starts and records `nvidia-smi` after loading.
# `mlx.launch` also returned 0 although both ranks failed, so a stage counts only if its JSON
# exists. Mistral 3.2 24B's `kernel+p16` ran out of memory dequantising its 131072-row head to
# float16; `p16` now dequantises at most 32768 rows at a time (bit-identical on a Mac check,
# and no matrix below that size changes).
# Rules fixed here, the rest as run 11:
#   * gate again, because the card selection changed: Qwen 3 8B `kernel+p16` pipelined must
#     reproduce the single-card tokens on both ranks, else the pipeline stages are skipped;
#   * Qwen 3 32B, Qwen3.8 27B, Qwen3.6 35B-A3B pipelined, stock against `kernel+p16` (Qwen 3.5
#     family at 256 prompt tokens); Qwen 3 32B server widths 1 and 8, `kernel+mma+p16` against
#     `kernel+p16`, 1.2x at width 8 counts;
#   * Mistral 3.2 24B on one card: `kernel+p16` against the in-run control `kernel` (decode is
#     the same path, so the difference is prefill); server widths 1 and 8, `kernel+mma+p16`
#     against `kernel+mma`.
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
DEADLINE = time.time() + 100 * 60
os.makedirs(f"{WORK}/logs", exist_ok=True)
with open("/tmp/perf1.py", "w") as stream:
    stream.write(PERF1_SOURCE)
report = {"schema": "ironmule.perf1-kaggle.v12", "commit": COMMIT, "stages": {}, "performance_claim": False}
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

path, _ = download("qwen3-8b")
if path:
    run("e2e_qwen3-8b_single", f"{PY} /tmp/perf1.py e2e {path} kernel+p16 {WORK}/e2e-qwen3-8b-single.json",
        f"{WORK}/e2e-qwen3-8b-single.json")
    run("e2e_qwen3-8b_pipe", f"{PIPE} e2e {path} kernel+p16 {WORK}/e2e-qwen3-8b-pipe.json", f"{WORK}/e2e-qwen3-8b-pipe.json")
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
            out = f"{WORK}/e2e-{key}-{arm}.json"
            run(f"e2e_{key}_{arm}", f"PERF1_PROMPT_TOKENS={prompt} {PIPE} e2e {path} {arm} {out}", out, timeout=1500)
        if key == "qwen3-32b":
            server_path = path
    if server_path:
        for arm in ("kernel+p16", "kernel+mma+p16"):
            out = f"{WORK}/server-qwen3-32b-{arm}.json"
            run(f"server_qwen3-32b_{arm}", f"{PIPE} server {server_path} {arm} 1,8 {out}", out, timeout=1200)

path, _ = download("mistral-24b")
if path:
    for arm in ("kernel", "kernel+p16"):
        out = f"{WORK}/e2e-mistral-24b-{arm}.json"
        run(f"e2e_mistral-24b_{arm}", f"{PY} /tmp/perf1.py e2e {path} {arm} {out}", out, timeout=1200)
    for arm in ("kernel+mma+p16", "kernel+mma"):
        out = f"{WORK}/server-mistral-24b-{arm}.json"
        run(f"server_mistral-24b_{arm}", f"{PY} /tmp/perf1.py server {path} {arm} 1,8 {out}", out, timeout=1200)

report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
