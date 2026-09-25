# PERF1 run 14: PERF1-S, a native kernel for MoE experts. Private notebook, internet on. Free
# quota: one run, <= 60 min (the week stood at 8.6 h before it; the total stays under 10 h).
#
# Run 12 decoded Qwen3.6 35B-A3B over two cards only 11% faster with `kernel+p16` than stock,
# because its experts go through `gather_qmm`, which no kernel routed: still emulated bfloat16.
# `perf1.py` now routes `gather_qmm` too: "gather" sends up to 64 (token, expert) rows through
# the row kernel with an index input, each warp reading its expert's rows in place; "g16" runs
# the larger prefill calls as the same `gather_qmm` in float16. On the Mac the routing gave
# mlx-lm's SwitchGLU the same shapes and values within 1.3e-2 of stock, with the kernel replaced
# by a float32 reference; the CUDA source itself has never run.
# Rules fixed here:
#   * probe first: every shape's decode and width-8 error against the per-pair float32 reference
#     at most 1e-2 (IronMule's install probe tolerance), else no model stage runs; `g16` arms run
#     only if its prefill error is within the same bound;
#   * Qwen3.6 35B-A3B, two cards pipelined, 256 prompt tokens: `kernel+p16` (the in-run control),
#     `kernel+p16+gather`, `kernel+p16+gather+g16`. Kill (backlog): `gather` decode under 1.5x the
#     control. `gather` leaves prefill alone and `g16` leaves decode alone, so decode is judged on
#     `kernel+p16+gather` and TTFT on the `g16` arm against it;
#   * Gemma 4 26B-A4B, one card, 256 prompt tokens: `kernel` (control; `p16` may not fit beside
#     14.2 GB), `kernel+gather+g16`, then `kernel+gather` if time is left. Same 1.5x reading.
#   * tokens are compared to the control arm and recorded, not gated: the kernel accumulates in
#     float32 where stock rounds to bfloat16, so the answers may part; no quality claim.
import json
import os
import signal
import subprocess
import sys
import time

COMMIT = "69f9937370456a18740213555e9bcd752ce60dd6"  # origin/main
PERF1_SOURCE = __PERF1_SOURCE__  # experiments/kaggle_compat/perf1.py
MODELS = {"qwen36-35b-a3b": ("mlx-community/Qwen3.6-35B-A3B-4bit", "38740b847e4cb78f352aba30aa41c76e08e6eb46", 256),
          "gemma4-26b-a4b": ("mlx-community/gemma-4-26b-a4b-it-4bit", "0d77464eeb233a2da68ebf9d7dc4edaac7db956d", 256)}
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable
PIPE = f"{VENV}/bin/mlx.launch --hosts 127.0.0.1 -n 2 --backend ring -- {PY} /tmp/perf1.py"
DEADLINE = time.time() + 60 * 60
os.makedirs(f"{WORK}/logs", exist_ok=True)
with open("/tmp/perf1.py", "w") as stream:
    stream.write(PERF1_SOURCE)
report = {"schema": "ironmule.perf1-kaggle.v14", "commit": COMMIT, "stages": {}, "performance_claim": False}
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


run("probe_gather", f"{PY} /tmp/perf1.py gather {WORK}/gather-probe.json", f"{WORK}/gather-probe.json", timeout=600)
shapes = (result(f"{WORK}/gather-probe.json") or {}).get("shapes", {})
report["gate"] = {
    "gather": len(shapes) == 4 and all(r.get(f"{c}_rel_err", 1) <= 1e-2 for r in shapes.values() for c in ("decode", "width8")),
    "g16": len(shapes) == 4 and all(r.get("prefill_g16_rel_err", 1) <= 1e-2 for r in shapes.values())}
save()


def compare(key, control, arms):
    base = tokens(f"{WORK}/e2e-{key}-{control}.json")
    for arm in arms:
        got = tokens(f"{WORK}/e2e-{key}-{arm}.json")
        if base and got:
            report.setdefault("tokens_vs_control", {})[f"{key} {arm}"] = next(
                (i for i, (a, b) in enumerate(zip(base, got)) if a != b), len(base))
    save()


if report["gate"]["gather"]:
    arms = ["kernel+p16", "kernel+p16+gather"] + (["kernel+p16+gather+g16"] if report["gate"]["g16"] else [])
    path, prompt = download("qwen36-35b-a3b")
    if path:
        for arm in arms:
            out = f"{WORK}/e2e-qwen36-35b-a3b-{arm}.json"
            run(f"e2e_qwen36-35b-a3b_{arm}", f"PERF1_PROMPT_TOKENS={prompt} {PIPE} e2e {path} {arm} {out}", out, timeout=1200)
        compare("qwen36-35b-a3b", arms[0], arms[1:])
    arms = ["kernel"] + (["kernel+gather+g16"] if report["gate"]["g16"] else []) + ["kernel+gather"]
    path, prompt = download("gemma4-26b-a4b")
    if path:
        for arm in arms:
            out = f"{WORK}/e2e-gemma4-26b-a4b-{arm}.json"
            run(f"e2e_gemma4-26b-a4b_{arm}", f"PERF1_PROMPT_TOKENS={prompt} {PY} /tmp/perf1.py e2e {path} {arm} {out}", out, timeout=900)
        compare("gemma4-26b-a4b", arms[0], arms[1:])

report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
