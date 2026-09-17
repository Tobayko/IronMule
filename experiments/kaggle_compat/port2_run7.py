# PORT2 run 7: close the quality gates the speed numbers are standing on, and localise
# the two things run 5 and run 6 could only exclude.
# Private notebook, internet on (pip, git, Hugging Face). Free quota: one run, <= 2.5 h.
#
#   A. `float16` is a per-model plan, not a device plan: Qwen 3 8B passes the WikiText-2
#      bound at 0.9977 and Gemma 3 4B fails it at 2.04, same card, same code. Every other
#      model that has a float16 *speed* number has no float16 *quality* number, which means
#      none of them may keep it. gpt-oss 20B, Qwen 3 14B and Qwen 3.5 9B get theirs here.
#   B. Mistral 3 24B's float32 gate died of memory at 512-token chunks; 256 is the retry.
#   C. gpt-oss 20B's float32 interval is `[0.9187; 1.0650]` at 12 chunks, far too wide to
#      qualify. 24 chunks is the retry.
#   D. Why float32 costs llama what it pays everyone else. The matvec diagnostic excluded
#      `quantized_matmul`; `Llama3RoPE`'s own `freqs` and llama's untied 128256-row
#      `lm_head` are what is left. Timed per dtype on the real model, with Qwen 3 8B as the
#      control, because a number without a control is not an answer.
#   E. Where `sharded_load` spends its memory. Tensor parallelism works and halves per-rank
#      weights, but Qwen3.8 27B still ran out; this prints active memory around the call so
#      the next attempt knows whether the whole model is materialised before sharding.
import json
import os
import signal
import subprocess
import sys
import time

COMMIT = "8064419f409366e08f8e651e747f483d9c0483df"  # origin/HEAD
PATCH_SOURCE = __PATCH_SOURCE__
QUALITY_SOURCE = __QUALITY_SOURCE__  # experiments/kaggle_compat/quality.py
OPS_SOURCE = __OPS_SOURCE__  # experiments/kaggle_compat/dtype_ops.py
MODELS = {
    "llama31-8b": ("mlx-community/Llama-3.1-8B-Instruct-4bit",
                   "90215b22ec18e72f623dde2ea7af4097025160e2"),
    "qwen3-8b": ("mlx-community/Qwen3-8B-4bit", "545dc4251c05440727734bcd94334791f6ab0192"),
    "qwen35-9b": ("mlx-community/Qwen3.5-9B-MLX-4bit", "938d8919941c6e7efd3c7150eff7fe9d12afa631"),
    "qwen3-14b": ("mlx-community/Qwen3-14B-4bit", "a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4"),
    "gptoss-20b": ("mlx-community/gpt-oss-20b-MXFP4-Q4",
                   "f356f2747216d7e98fee755df25987459fc19089"),
    "mistral-24b": ("mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit",
                    "2a1d5eabfc504747bdc24178394821a1efc0edde"),
}
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable
DEADLINE = time.time() + 140 * 60
os.makedirs(f"{WORK}/logs", exist_ok=True)
with open("/tmp/ironmule.patch", "w") as stream:
    stream.write(PATCH_SOURCE)
for name, source in (("quality", QUALITY_SOURCE), ("dtype_ops", OPS_SOURCE)):
    with open(f"/tmp/{name}.py", "w") as stream:
        stream.write(source)
report = {"schema": "ironmule.port2-kaggle.v7", "commit": COMMIT, "stages": {},
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
    report["stages"][name] = {"exit": code, "seconds": round(time.time() - started, 1), "tail": out[-3000:]}
    save()
    print(f"== {name}: exit={code} {report['stages'][name]['seconds']}s", flush=True)
    return code, out


sh("env", "nvidia-smi --query-gpu=index,name,memory.total --format=csv; nproc; free -g")
sh("clone", f"git clone -q https://github.com/Tobayko/IronMule {REPO} && git -C {REPO} checkout -q {COMMIT} "
            f"&& git -C {REPO} apply /tmp/ironmule.patch && git -C {REPO} diff --stat")
sh("venv", f"{SYS} -m pip install -q uv && {SYS} -m uv python install 3.12 "
           f"&& {SYS} -m uv venv --python-preference only-managed --python 3.12 {VENV}")
sh("install", f"{SYS} -m uv pip install --python {PY} -e '{REPO}[cuda]' 'mlx-lm==0.31.3' psutil scipy pyarrow",
   timeout=1200)
sh("freeze", f"{SYS} -m uv pip freeze --python {PY}")
for key, (model_id, revision) in MODELS.items():
    sh(f"download_{key}", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                          f"print(s('{model_id}', revision='{revision}'))\"", timeout=1200)
sh("download_wikitext", f"{PY} -c \"from huggingface_hub import hf_hub_download as d; import pyarrow.parquet as pq; "
                        "p = d('Salesforce/wikitext', 'wikitext-2-raw-v1/test-00000-of-00001.parquet', repo_type='dataset'); "
                        "open('/tmp/wikitext.txt', 'w').write(''.join(pq.read_table(p).column('text').to_pylist())); print(p)\"")

# -- D and E first: they are cheap, and a deadline must not eat the diagnostics -------------
for key in ("llama31-8b", "qwen3-8b"):
    model_id, revision = MODELS[key]
    sh(f"dtype_ops_{key}", f"{PY} /tmp/dtype_ops.py {model_id} {revision} "
                           f"{WORK}/dtype-ops-{key}.json", timeout=900)
sh("shard_memory", f"""{PY} - <<'PYEOF'
import json
import mlx.core as mx
out = {{"before_bytes": int(mx.get_active_memory())}}
try:
    from mlx_lm.utils import load_model, _download
    path = _download("mlx-community/Qwen3.8-27B-4bit",
                     allow_patterns=["*.json", "*.py", "tokenizer.model", "*.txt"])
    model, config = load_model(path, lazy=True, strict=False)
    out["lazy_loaded_bytes"] = int(mx.get_active_memory())
    out["has_shard"] = hasattr(model, "shard")
    out["has_pipeline"] = hasattr(model, "model") and hasattr(model.model, "pipeline")
    out["model_type"] = config.get("model_type")
    out["layers"] = len(model.layers)
except Exception as exc:
    out["error"] = f"{{type(exc).__name__}}: {{exc}}"
print(json.dumps(out, indent=1))
PYEOF""", timeout=900)

# -- A, B, C: the gates every speed number is standing on -----------------------------------
# Each row is one process: (model, precision, chunks, chunk tokens). The pairs are
# (bf16, float16) for the float16 gate and (bf16, float32) where float32 is still open.
# Mistral runs 256-token chunks because 512 ran out of memory at 13.26 GB of weights.
GATES = [
    ("gptoss-20b", "bf16", 16, 512), ("gptoss-20b", "float16", 16, 512),
    ("qwen3-14b", "bf16", 16, 512), ("qwen3-14b", "float16", 16, 512),
    ("qwen35-9b", "bf16", 16, 512), ("qwen35-9b", "float16", 16, 512),
    ("mistral-24b", "bf16", 8, 256), ("mistral-24b", "float32", 8, 256),
    ("gptoss-20b", "float32", 24, 512), ("gptoss-20b", "bf16", 24, 512),
]
for key, precision, chunks, chunk_tokens in GATES:
    model_id, revision = MODELS[key]
    sh(f"quality_{key}-{precision}-{chunks}",
       f"QUALITY_ONLY={precision} {PY} /tmp/quality.py {model_id} {revision} /tmp/wikitext.txt "
       f"{WORK}/quality-{key}-{precision}-{chunks}.json {chunks} {chunk_tokens}", timeout=2400)
report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
