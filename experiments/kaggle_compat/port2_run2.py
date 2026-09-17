# PORT2 run 2: the same question as run 1, with run 1's three defects fixed.
# Private notebook, internet on (pip, git, Hugging Face). Free quota: one run, <= 3 h.
#
# Run 1 answered less than it cost, for reasons that were all ours:
#   * IronMule called `model.make_cache()`, which mlx-lm makes optional, so Qwen 3 and
#     Mistral 3 never loaded — three of six candidates, including the largest.
#   * `fast.py` applied Gemma 3's block body to every architecture. It crashed on llama
#     (`q_norm`) and would have computed GELU where llama and Qwen 3 want SwiGLU.
#   * `families.py` retried a failed arm inside the exception handler, whose traceback
#     still held the failed model, so gpt-oss met a 15 GB card holding 22 GB.
# All three are fixed in the patch this notebook applies. Gemma 3 4B is back in the list
# as a control: PORT1 measured its float32 win on mlx 0.32.0 and this stack is 0.32.2, so
# the reversal llama showed (float32 at 1.54 of stock) needs the old case re-measured on
# the new stack before it is attributed to the architecture rather than to the library.
import json
import os
import signal
import subprocess
import sys
import time

COMMIT = "8064419f409366e08f8e651e747f483d9c0483df"  # origin/HEAD
PATCH_SOURCE = __PATCH_SOURCE__  # uncommitted IronMule changes under test, embedded at submit time
FAMILIES_SOURCE = __FAMILIES_SOURCE__  # experiments/kaggle_compat/families.py
CROSS_SOURCE = __CROSS_SOURCE__  # experiments/kaggle_compat/cross.py
QUALITY_SOURCE = __QUALITY_SOURCE__  # experiments/kaggle_compat/quality.py
MODELS = [
    ("gemma3-4b", "mlx-community/gemma-3-4b-it-4bit",
     "93724907d4ed1745d2fe50baadf3b0b01a65abf2", 2.50, "gemma3"),
    ("llama31-8b", "mlx-community/Llama-3.1-8B-Instruct-4bit",
     "90215b22ec18e72f623dde2ea7af4097025160e2", 4.52, "llama"),
    ("qwen3-8b", "mlx-community/Qwen3-8B-4bit",
     "545dc4251c05440727734bcd94334791f6ab0192", 4.61, "qwen3"),
    ("qwen35-9b", "mlx-community/Qwen3.5-9B-MLX-4bit",
     "938d8919941c6e7efd3c7150eff7fe9d12afa631", 5.95, "qwen3_5"),
    ("qwen3-14b", "mlx-community/Qwen3-14B-4bit",
     "a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4", 8.31, "qwen3"),
    ("gptoss-20b", "mlx-community/gpt-oss-20b-MXFP4-Q4",
     "f356f2747216d7e98fee755df25987459fc19089", 11.18, "gpt_oss"),
    ("mistral-24b", "mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit",
     "2a1d5eabfc504747bdc24178394821a1efc0edde", 13.26, "mistral3"),
]
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable  # Kaggle's interpreter carries pip; the venv does not
DEADLINE = time.time() + 170 * 60
os.makedirs(f"{WORK}/logs", exist_ok=True)
with open("/tmp/ironmule.patch", "w") as stream:
    stream.write(PATCH_SOURCE)
for name, source in (("families", FAMILIES_SOURCE), ("cross", CROSS_SOURCE), ("quality", QUALITY_SOURCE)):
    with open(f"/tmp/{name}.py", "w") as stream:
        stream.write(source)
report = {"schema": "ironmule.port2-kaggle.v2", "commit": COMMIT, "stages": {},
          "models": [{"key": k, "model_id": m, "revision": r, "gb": g, "model_type": t}
                     for k, m, r, g, t in MODELS],
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


def collect(path, key):
    try:
        with open(path) as stream:
            payload = json.load(stream)
    except (OSError, ValueError):
        return None
    report["screening"][key] = {k: v for k, v in payload.items() if k != "arms"}
    report["screening"][key]["arms"] = [
        {k: v for k, v in arm.items() if k != "tokens"} for arm in payload.get("arms", [])]
    save()
    return payload


sh("env", "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; nproc; free -g; df -h / | tail -1")
sh("clone", f"git clone -q https://github.com/Tobayko/IronMule {REPO} && git -C {REPO} checkout -q {COMMIT} "
            f"&& git -C {REPO} apply /tmp/ironmule.patch && git -C {REPO} diff --stat")
# only-managed: a venv resolving to /usr/lib/python3.12 inherits Debian's sitecustomize,
# which imports a missing `wrapt` and slows every child interpreter's start (PORT1).
sh("venv", f"{SYS} -m pip install -q uv && {SYS} -m uv python install 3.12 "
           f"&& {SYS} -m uv venv --python-preference only-managed --python 3.12 {VENV}")
sh("install", f"{SYS} -m uv pip install --python {PY} -e '{REPO}[cuda]' 'mlx-lm==0.31.3' "
              "'pytest>=8' 'pytest-xdist>=3' psutil scipy pyarrow", timeout=1200)
sh("freeze", f"{SYS} -m uv pip freeze --python {PY}")
# The fused bodies are transcriptions; this is the check that they are still bit identical
# on this stack, and that an architecture without a transcription is refused.
sh("fusion_self_check", f"{PY} -m ironmule.fast", cwd=REPO)
sh("pytest_fusion", f"{PY} -m pytest -q -n0 -o addopts='' tests/engine/test_ironmule.py", cwd=REPO)

# -- pass 1: download and screen every candidate ------------------------------------------
survivors = []
for key, model_id, revision, gb, model_type in MODELS:
    code, _ = sh(f"download_{key}", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                                    f"print(s('{model_id}', revision='{revision}'))\"", timeout=1200)
    if code != 0:
        continue
    sh(f"screen_{key}", f"{PY} /tmp/families.py {model_id} {revision} {WORK}/families-{key}.json 16",
       timeout=1800)
    screened = collect(f"{WORK}/families-{key}.json", key)
    if screened and screened.get("ok"):
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
