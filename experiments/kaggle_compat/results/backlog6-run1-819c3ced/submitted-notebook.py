# BACKLOG6: DATA3-B's fix on the engine suite, what holds PORT1-F's parent memory, and PERF1-R's
# upper bound before anything is built. Private notebook, internet on, two T4. Quota: the user
# asked on 2026-09-25 to work through the backlog (budget: the 30 h Kaggle week). Rules:
#   * The engine suite as pytest.ini configures it, not integration, on this commit.
#   * PORT1-F: BACKLOG5's release left MLX at 56 active and 0 cached bytes while nvidia-smi still
#     counted 9553 MiB, and the child ran out of memory again. `pool_probe.py` loads Gemma 3 4B
#     as tune does and records the default memory pool's reserved and used bytes after the
#     release, after a context synchronize and after trimming the pool, then loads the model in
#     a child. A diagnostic: it decides between trimming the pool and screening in a child.
#   * PERF1-R (micro-batches in the layer pipeline): two micro-batches of four can at best keep
#     both cards busy with one width-4 stream each, so their aggregate is bounded by twice the
#     pipeline's own width-4 aggregate. Qwen 3 32B pipelined over both cards, `kernel+mma+p16`
#     (run 12's server stack and env), `perf1.py server` in four launches with widths `4,8`,
#     `8,4`, `4,8`, `8,4`, each warmed at both widths first (run 12 warmed only the widest).
#     Bound = 2 x median aggregate tok/s at width 4 / median at width 8. A bound below 1.2 (the
#     entry's kill) closes PERF1-R without an implementation; otherwise the implementation is
#     the next run. Screening numbers, no performance claim.
import json
import os
import signal
import subprocess
import sys
import time

COMMIT = "d4117989cb4e1d280d691c409e90f2b98d5e9811"
GEMMA4B = ("mlx-community/gemma-3-4b-it-4bit", "93724907d4ed1745d2fe50baadf3b0b01a65abf2")
QWEN32B = ("mlx-community/Qwen3-32B-4bit", "bcaaf7f538adf166c1080a2befdb4f6019f66639")
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable
DEADLINE = time.time() + 90 * 60
os.makedirs(f"{WORK}/logs", exist_ok=True)
report = {"schema": "ironmule.backlog6-kaggle.v1", "commit": COMMIT, "stages": {}, "performance_claim": False}
env = dict(os.environ, PATH=f"{VENV}/bin:" + os.environ["PATH"], PYTHONPATH="", PYTHONNOUSERSITE="1",
           HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONUNBUFFERED="1", IRONMULE_HOME="/tmp/ironmule-home")
PIPE = f"{VENV}/bin/mlx.launch --hosts 127.0.0.1 -n 2 --backend ring -- {PY} {REPO}/experiments/kaggle_compat/perf1.py"
RUN12_ENV = "MLX_MAX_OPS_PER_BUFFER=400 PERF1_SERVER_WARMUP=4,8"  # run 12's env; both widths warm


def save():
    with open(f"{WORK}/backlog6-result.json", "w") as stream:
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


sh("env", "nvidia-smi --query-gpu=index,name,driver_version,memory.total,clocks.max.sm --format=csv; nproc; free -g; df -h / | tail -1")
sh("clone", f"git clone -q https://github.com/Tobayko/IronMule {REPO} && git -C {REPO} checkout -q {COMMIT} "
            f"&& git -C {REPO} rev-parse HEAD && git -C {REPO} status --short")
sh("venv", f"{SYS} -m pip install -q uv && {SYS} -m uv python install 3.12 "
           f"&& {SYS} -m uv venv --python-preference only-managed --python 3.12 {VENV}")
sh("install", f"{SYS} -m uv pip install --python {PY} -e '{REPO}[cuda]' 'mlx-lm==0.31.3' pyarrow 'pytest>=8' 'pytest-xdist>=3' psutil scipy", timeout=1200)
sh("freeze", f"{SYS} -m uv pip freeze --python {PY}")
sh("pytest_engine", f"{PY} -m pytest tests/engine -m 'not integration' -rfEs -p no:cacheprovider "
                    f"--junitxml={WORK}/pytest_engine.xml", timeout=1800, cwd=REPO)


def download(key, model):
    code, out = sh(f"download_{key}", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                                      f"print(s('{model[0]}', revision='{model[1]}'))\"", timeout=1200)
    return out.strip().splitlines()[-1] if code == 0 else None


# PORT1-F: what still holds the parent's memory.
if download("gemma3-4b", GEMMA4B):
    sh("pool_probe_gemma3-4b", f"{PY} {REPO}/experiments/kaggle_compat/pool_probe.py {GEMMA4B[0]} "
                               f"{WORK}/pool-probe-gemma3-4b.json", timeout=1200, cwd=REPO)

# PERF1-R: the bound, in four launches of alternating width order.
path = download("qwen3-32b", QWEN32B)
report["perf1r"] = []
if path:
    for index, widths in enumerate(("4,8", "8,4", "4,8", "8,4")):
        out = f"{WORK}/server-qwen3-32b-kernel+mma+p16-{index}.json"
        sh(f"server_qwen3-32b_{index}", f"{RUN12_ENV} {PIPE} server {path} kernel+mma+p16 {widths} {out}", timeout=900)
        try:
            with open(out) as stream:
                rows = json.load(stream)["widths"]
            report["perf1r"].append({"launch": index, "order": widths,
                                     "tps": {w: rows[w]["aggregate_tps"] for w in rows}})
        except (OSError, ValueError, KeyError):  # `mlx.launch` exits 0 when its ranks fail (run 11)
            report["perf1r"].append({"launch": index, "order": widths, "tps": None})
        save()
report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
