# PORT2 run 3: close the three things run 2 left open, and nothing else.
# Private notebook, internet on (pip, git, Hugging Face). Free quota: one run, <= 1.5 h.
#
# Run 2 measured five families cleanly and left exactly three holes:
#   1. llama under projection fusion returns different tokens on two calls in one process
#      and raises `cudaGraphInstantiate … illegal memory access` under the six-request
#      workload, while the same bodies are token identical on Gemma 3 and Qwen 3. One arm
#      per candidate cause: a contiguity barrier where the QK norm would have been, and
#      CUDA graph capture off.
#   2. llama has no float32 number, because both its IronMule arms died of (1). Every
#      other family's float32 arm ran between 0.28 and 0.54 of stock, so the 1.54 run 1
#      reported for llama is the one datum that disagrees, and it was measured on the
#      unfused path. Re-measure unfused.
#   3. Qwen 3.5's exact arm was 3/6 token identical and NOT deterministic across
#      processes. Three repetitions decide whether that repeats.
# Plus Mistral 3 24B, which loads and decodes correctly at 13.26 GB but whose tuned
# profile ran out of memory: PORT1's 12B arm (`head_skip_prefill` alone) is what fits.
import json
import os
import signal
import subprocess
import sys
import time

COMMIT = "8064419f409366e08f8e651e747f483d9c0483df"  # origin/HEAD
PATCH_SOURCE = __PATCH_SOURCE__  # uncommitted IronMule changes under test, embedded at submit time
PROBE_SOURCE = __PROBE_SOURCE__  # experiments/kaggle_compat/fused_probe.py
CROSS_SOURCE = __CROSS_SOURCE__  # experiments/kaggle_compat/cross.py
LLAMA = ("mlx-community/Llama-3.1-8B-Instruct-4bit", "90215b22ec18e72f623dde2ea7af4097025160e2")
QWEN35 = ("mlx-community/Qwen3.5-9B-MLX-4bit", "938d8919941c6e7efd3c7150eff7fe9d12afa631")
MISTRAL = ("mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit",
           "2a1d5eabfc504747bdc24178394821a1efc0edde")
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable
DEADLINE = time.time() + 75 * 60
os.makedirs(f"{WORK}/logs", exist_ok=True)
with open("/tmp/ironmule.patch", "w") as stream:
    stream.write(PATCH_SOURCE)
for name, source in (("fused_probe", PROBE_SOURCE), ("cross", CROSS_SOURCE)):
    with open(f"/tmp/{name}.py", "w") as stream:
        stream.write(source)
report = {"schema": "ironmule.port2-kaggle.v3", "commit": COMMIT, "stages": {},
          "performance_claim": False}
env = dict(os.environ, PATH=f"{VENV}/bin:" + os.environ["PATH"], PYTHONPATH="", PYTHONNOUSERSITE="1",
           IRONMULE_HOME="/tmp/ironmule-home", IRONMULE_API_KEY="port2-local-only",
           HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONUNBUFFERED="1")


def save():
    with open(f"{WORK}/port2-result.json", "w") as stream:
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


sh("env", "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; nproc; free -g")
sh("clone", f"git clone -q https://github.com/Tobayko/IronMule {REPO} && git -C {REPO} checkout -q {COMMIT} "
            f"&& git -C {REPO} apply /tmp/ironmule.patch && git -C {REPO} diff --stat")
sh("venv", f"{SYS} -m pip install -q uv && {SYS} -m uv python install 3.12 "
           f"&& {SYS} -m uv venv --python-preference only-managed --python 3.12 {VENV}")
sh("install", f"{SYS} -m uv pip install --python {PY} -e '{REPO}[cuda]' 'mlx-lm==0.31.3' psutil scipy",
   timeout=1200)
sh("freeze", f"{SYS} -m uv pip freeze --python {PY}")
for key, (model_id, revision) in (("llama", LLAMA), ("qwen35", QWEN35), ("mistral", MISTRAL)):
    sh(f"download_{key}", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                          f"print(s('{model_id}', revision='{revision}'))\"", timeout=1200)

# -- 1. why fusion corrupts llama and not Qwen 3 --------------------------------------------
llama, rev_llama = LLAMA
sh("probe_llama_graphs_on", f"{PY} /tmp/fused_probe.py {llama} {rev_llama} "
                            f"{WORK}/fused-probe-graphs-on.json 32", timeout=900)
sh("probe_llama_graphs_off", f"MLX_USE_CUDA_GRAPHS=0 {PY} /tmp/fused_probe.py {llama} {rev_llama} "
                             f"{WORK}/fused-probe-graphs-off.json 32", timeout=900)

STOCK = {"name": "stock", "env": {"MLX_MAX_OPS_PER_BUFFER": "20", "MLX_MAX_MB_PER_BUFFER": "100"},
         "knobs": {}, "mode": "interactive"}
UNFUSED = {"compiled_fixed_cache": True, "fused_argmax": True, "head_skip_prefill": True,
           "readback_every": 2, "capacity_slack": 128, "fuse_projections": False}

# -- 2. llama's missing float32 cell, on the path that survives ------------------------------
llama_configs = [STOCK,
                 {"name": "ironmule_unfused", "env": {}, "knobs": UNFUSED, "mode": "throughput"},
                 {"name": "ironmule_fp32_unfused", "env": {}, "knobs": UNFUSED,
                  "mode": "throughput", "dtype": "float32"}]
sh("cross_llama_unfused", f"{PY} /tmp/cross.py {llama} {rev_llama} '{json.dumps(llama_configs)}' "
                          f"{WORK}/cross-llama-unfused.json 2 1", timeout=1800)

# -- 3. does Qwen 3.5's exact arm stay non-deterministic over three repetitions? --------------
qwen, rev_qwen = QWEN35
qwen_configs = [STOCK,
                {"name": "ironmule_exact", "env": {}, "knobs": UNFUSED, "mode": "throughput"}]
sh("cross_qwen35_determinism", f"{PY} /tmp/cross.py {qwen} {rev_qwen} '{json.dumps(qwen_configs)}' "
                               f"{WORK}/cross-qwen35-determinism.json 3 1", timeout=2100)

# -- the largest checkpoint that fits, on the profile that fits with it -----------------------
# PORT1 ran Gemma 3 12B with `head_skip_prefill` alone; at 13.26 GB of 15360 MiB the tuned
# profile's compiled graph is what ran out of memory in run 2, not the weights.
mistral, rev_mistral = MISTRAL
LEAN = {"head_skip_prefill": True}
mistral_configs = [STOCK,
                   {"name": "ironmule_lean", "env": {}, "knobs": LEAN, "mode": "throughput"},
                   {"name": "ironmule_lean_fp32", "env": {}, "knobs": LEAN, "mode": "throughput",
                    "dtype": "float32"}]
sh("cross_mistral_lean", f"{PY} /tmp/cross.py {mistral} {rev_mistral} '{json.dumps(mistral_configs)}' "
                         f"{WORK}/cross-mistral-lean.json 2 1", timeout=2400)
report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
