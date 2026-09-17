# PORT2 run 1: does the CUDA result generalise past Gemma 3, and how large a model fits?
# Private notebook, internet on (pip, git, Hugging Face). Free quota: one run, <= 2.5 h.
#
# Two passes, in this order on purpose. Pass 1 screens every candidate with `families.py`:
# cheap, and it answers the question the run exists for — what loads, what fits in 15 GB,
# which knobs engage on an unfamiliar block layout. Pass 2 spends the rest of the budget on
# `cross.py` wall ratios, smallest model first. If the deadline bites in pass 2 the
# compatibility and memory answer is already complete for every candidate.
import json
import os
import signal
import subprocess
import sys
import time

COMMIT = "8064419f409366e08f8e651e747f483d9c0483df"  # origin/HEAD: PORT1 compute_dtype shipped
FAMILIES_SOURCE = __FAMILIES_SOURCE__  # experiments/kaggle_compat/families.py
CROSS_SOURCE = __CROSS_SOURCE__  # experiments/kaggle_compat/cross.py
QUALITY_SOURCE = __QUALITY_SOURCE__  # experiments/kaggle_compat/quality.py
# Every candidate is a 4-bit mlx-community checkpoint pinned to an exact revision, chosen
# for architecture coverage and size, largest under the T4's 15360 MiB. `gb` is the
# safetensors total as Hugging Face reports it, recorded so the run can be read without
# the network. The last two exist to make the card sweat, not because they are safe.
#
# `cross` is False for one candidate only. Qwen3.5's gated-delta layers fall back to
# `gated_delta_ops` off Metal — mlx-lm returns no kernel when `mx.metal.is_available()` is
# false — so its wall ratio would mostly measure a missing kernel, not IronMule, at the
# highest cost per cell in the list. Screening still answers what that model is here for:
# whether a hybrid KV/Arrays cache loads and which knobs engage on it.
MODELS = [
    ("llama31-8b", "mlx-community/Llama-3.1-8B-Instruct-4bit",
     "90215b22ec18e72f623dde2ea7af4097025160e2", 4.52, "llama", True),
    ("qwen3-8b", "mlx-community/Qwen3-8B-4bit",
     "545dc4251c05440727734bcd94334791f6ab0192", 4.61, "qwen3", True),
    ("qwen35-9b", "mlx-community/Qwen3.5-9B-MLX-4bit",
     "938d8919941c6e7efd3c7150eff7fe9d12afa631", 5.95, "qwen3_5", False),
    ("qwen3-14b", "mlx-community/Qwen3-14B-4bit",
     "a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4", 8.31, "qwen3", True),
    ("gptoss-20b", "mlx-community/gpt-oss-20b-MXFP4-Q4",
     "f356f2747216d7e98fee755df25987459fc19089", 11.18, "gpt_oss", True),
    ("mistral-24b", "mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit",
     "2a1d5eabfc504747bdc24178394821a1efc0edde", 13.26, "mistral3", True),
]
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable  # Kaggle's interpreter carries pip; the venv does not
DEADLINE = time.time() + 160 * 60
os.makedirs(f"{WORK}/logs", exist_ok=True)
for name, source in (("families", FAMILIES_SOURCE), ("cross", CROSS_SOURCE), ("quality", QUALITY_SOURCE)):
    with open(f"/tmp/{name}.py", "w") as stream:
        stream.write(source)
report = {"schema": "ironmule.port2-kaggle.v1", "commit": COMMIT, "stages": {},
          "models": [{"key": k, "model_id": m, "revision": r, "gb": g, "model_type": t, "cross": c}
                     for k, m, r, g, t, c in MODELS],
          "screening": {}, "performance_claim": False}
# PYTHONNOUSERSITE: Kaggle's root user site holds the sitecustomize importing `wrapt`.
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
    # Own process group: a timeout must also stop the workers a command started, or they
    # keep the model lease and the next stage fails (PORT1 attempt 1).
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


def collect(path, key=None):
    try:
        with open(path) as stream:
            payload = json.load(stream)
    except (OSError, ValueError):
        return None
    if key:
        report["screening"][key] = {k: v for k, v in payload.items() if k != "arms"}
        report["screening"][key]["arms"] = [
            {k: v for k, v in arm.items() if k != "tokens"} for arm in payload.get("arms", [])]
        save()
    return payload


sh("env", "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; nproc; free -g; df -h / | tail -1")
sh("clone", f"git clone -q https://github.com/Tobayko/IronMule {REPO} && git -C {REPO} checkout -q {COMMIT} "
            f"&& git -C {REPO} rev-parse HEAD && git -C {REPO} status --short")
# only-managed: a venv resolving to /usr/lib/python3.12 inherits Debian's sitecustomize,
# which imports a missing `wrapt` and slows every child interpreter's start (PORT1).
sh("venv", f"{SYS} -m pip install -q uv && {SYS} -m uv python install 3.12 "
           f"&& {SYS} -m uv venv --python-preference only-managed --python 3.12 {VENV}")
sh("install", f"{SYS} -m uv pip install --python {PY} -e '{REPO}[cuda]' 'mlx-lm==0.31.3' psutil scipy pyarrow",
   timeout=1200)
sh("freeze", f"{SYS} -m uv pip freeze --python {PY}")

# -- pass 1: download and screen every candidate ------------------------------------------
survivors = []
for key, model_id, revision, gb, model_type, cross in MODELS:
    code, _ = sh(f"download_{key}", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                                    f"print(s('{model_id}', revision='{revision}'))\"", timeout=1200)
    if code != 0:
        continue
    sh(f"screen_{key}", f"{PY} /tmp/families.py {model_id} {revision} {WORK}/families-{key}.json 24",
       timeout=1500)
    screened = collect(f"{WORK}/families-{key}.json", key)
    if screened and screened.get("ok") and cross:
        survivors.append((key, model_id, revision, bool(screened.get("tuned_profile_reduced"))))
report["survivors"] = [s[0] for s in survivors]
save()

# -- pass 2: the like-for-like wall ratio, smallest first ----------------------------------
# "stock" pins MLX's own graph limits, baseline knobs and interactive mode; the IronMule
# arms use whatever IronMule applies by itself. The tuned profile is PORT1's confirmed 1B
# profile, unchanged: the question is whether a profile transfers across families.
TUNED = {"compiled_fixed_cache": True, "fused_argmax": True, "head_skip_prefill": True,
         "readback_every": 2, "capacity_slack": 128, "fuse_projections": True}
STOCK = {"name": "stock", "env": {"MLX_MAX_OPS_PER_BUFFER": "20", "MLX_MAX_MB_PER_BUFFER": "100"},
         "knobs": {}, "mode": "interactive"}
for key, model_id, revision, reduced in survivors:
    knobs = {**TUNED, "fuse_projections": False} if reduced else dict(TUNED)
    configs = [STOCK,
               {"name": "ironmule_exact", "env": {}, "knobs": knobs, "mode": "throughput"},
               {"name": "ironmule_fp32", "env": {}, "knobs": knobs, "mode": "throughput",
                "dtype": "float32"}]
    sh(f"cross_{key}", f"{PY} /tmp/cross.py {model_id} {revision} '{json.dumps(configs)}' "
                       f"{WORK}/cross-{key}.json 2 1", timeout=2700)

# -- the float32 arms carry no quality claim until this runs --------------------------------
sh("download_wikitext", f"{PY} -c \"from huggingface_hub import hf_hub_download as d; import pyarrow.parquet as pq; "
                        "p = d('Salesforce/wikitext', 'wikitext-2-raw-v1/test-00000-of-00001.parquet', repo_type='dataset'); "
                        "open('/tmp/wikitext.txt', 'w').write(''.join(pq.read_table(p).column('text').to_pylist())); print(p)\"")
for key, model_id, revision, _ in survivors:
    # Both precisions live side by side here, so only the models that fit twice are asked.
    gb = next(entry["gb"] for entry in report["models"] if entry["key"] == key)
    if gb > 6.0:
        continue
    sh(f"quality_{key}", f"{PY} /tmp/quality.py {model_id} {revision} /tmp/wikitext.txt "
                         f"{WORK}/quality-{key}.json 12 512", timeout=1500)
report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
