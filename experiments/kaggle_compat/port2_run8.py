# PORT2 run 8: the gates that are still open, and the one number that came back NaN.
# Private notebook, internet on. Free quota: one run, <= 1.5 h, sized against the reserve.
#
#   A. Qwen 3.5 9B's bfloat16 arm returned a non-finite loss on all sixteen 512-token
#      chunks while its float16 arm returned 1.95-2.60. That is not a quality result, it is
#      a defect, and it is upstream: `quality.py` calls the model forward directly, so this
#      is mlx-lm's own gated-delta path on CUDA. Bisect by chunk length and confirm against
#      a load that never touches IronMule.
#   B. gpt-oss 20B's intervals are too wide to qualify anything: float32
#      `1.003673 [0.939996; 1.064706]`, float16 `1.008752 [0.948634; 1.066555]`, both at 16
#      chunks on text whose perplexity is 144. More chunks is the only lever; bfloat16 is
#      the slow arm here because it is emulated, so it gets the clock.
#   C. Mistral 3 24B's float32 gate ran out of memory at 512 and again at 256 tokens.
#      128 is the last size worth trying before the gate is declared unreachable on one card.
#   D. run 7's `dtype_ops` control failed on a bug of mine: Qwen 3's head width was
#      reconstructed as `scale ** -2` and rounded to 127, which `mx.fast.rope` refuses.
#      Fixed. llama already showed its `lm_head` costing 5.65 ms against 0.027 for rope, and
#      gaining only 8.7 per cent from float32 where the projections gain 42 - the control is
#      what turns that into an answer.
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
DEADLINE = time.time() + 78 * 60
os.makedirs(f"{WORK}/logs", exist_ok=True)
with open("/tmp/ironmule.patch", "w") as stream:
    stream.write(PATCH_SOURCE)
for name, source in (("quality", QUALITY_SOURCE), ("dtype_ops", OPS_SOURCE)):
    with open(f"/tmp/{name}.py", "w") as stream:
        stream.write(source)
report = {"schema": "ironmule.port2-kaggle.v8", "commit": COMMIT, "stages": {},
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

# -- D. the control that run 7's bug cost, and llama beside it -------------------------------
for key in ("qwen3-8b", "llama31-8b"):
    model_id, revision = MODELS[key]
    sh(f"dtype_ops_{key}", f"{PY} /tmp/dtype_ops.py {model_id} {revision} "
                           f"{WORK}/dtype-ops-{key}.json", timeout=900)

# -- A. where Qwen 3.5's bfloat16 loss stops being finite -------------------------------------
qwen35, rev35 = MODELS["qwen35-9b"]
sh("qwen35_nan_bisect", f"""{PY} - <<'PYEOF'
import json
import mlx.core as mx
from mlx_lm import load
import sys
sys.path.insert(0, "/tmp/IronMule")
from ironmule.tune import resolve_local_model

# Straight through mlx-lm, no IronMule anywhere, so a non-finite loss lands where it belongs.
model, tokenizer = load(str(resolve_local_model("{qwen35}", "{rev35}").path))
ids = tokenizer.encode(open("/tmp/wikitext.txt", encoding="utf-8").read())
out = {{"model_id": "{qwen35}", "loader": "mlx_lm.load", "lengths": {{}}}}
for length in (64, 128, 256, 512):
    chunk = ids[: length + 1]
    inputs, targets = mx.array([chunk[:-1]]), mx.array(chunk[1:])
    logits = model(inputs)[0].astype(mx.float32)
    finite_logits = bool(mx.isfinite(logits).all().item())
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    nll = -mx.mean(lp[mx.arange(length), targets])
    mx.eval(nll)
    value = nll.item()
    out["lengths"][str(length)] = {{"nll": value, "finite": value == value,
                                    "logits_finite": finite_logits}}
    print(length, out["lengths"][str(length)], flush=True)
    mx.clear_cache()
with open("{WORK}/qwen35-nan-bisect.json", "w") as stream:
    json.dump(out, stream, indent=1)
print(json.dumps(out, indent=1))
PYEOF""", timeout=1200)

# -- B and C. the gates themselves ------------------------------------------------------------
GATES = [
    ("mistral-24b", "bf16", 8, 128), ("mistral-24b", "float32", 8, 128),
    ("gptoss-20b", "float16", 32, 512), ("gptoss-20b", "float32", 32, 512),
    ("gptoss-20b", "bf16", 32, 512),
]
for key, precision, chunks, chunk_tokens in GATES:
    model_id, revision = MODELS[key]
    sh(f"quality_{key}-{precision}-{chunks}",
       f"QUALITY_ONLY={precision} {PY} /tmp/quality.py {model_id} {revision} /tmp/wikitext.txt "
       f"{WORK}/quality-{key}-{precision}-{chunks}.json {chunks} {chunk_tokens}", timeout=2700)
report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
