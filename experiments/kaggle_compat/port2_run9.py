# PORT2 run 9: Gemma 4, which splits into three answers before a single measurement.
# Private notebook, internet on. Free quota: one run, <= 1.5 h, sized against the reserve.
#
# Checked before spending any of it, from the checkpoints' own `config.json`:
#   * `gemma4` is implemented in mlx-lm 0.31.3 as `gemma4_text`. E2B (3.55 GB), E4B
#     (5.15 GB) and E4B-qat (6.80 GB) fit the card and are what this measures.
#   * `gemma4_unified` is implemented nowhere. The 12B checkpoints — 6.74 GB and 10.99 GB,
#     both of which would fit comfortably — declare that model type, and 0.31.3 is the
#     latest mlx-lm on PyPI, so this is not a version pin, it is simply absent. One stage
#     records the exact failure rather than asserting it from the outside.
#   * 26B-A4B is 15.34 GB and 31B is 18.41 GB, both above the 13.26 GB that Mistral 3
#     proved loads. One load attempt on 26B-A4B gives the ceiling a second point instead
#     of leaving it interpolated between 13.26 and 16.05.
#
# Projection fusion will refuse Gemma 4 and that is the gate working, not a defect: its
# attention takes `shared_kv` and `offset`, returns a tuple, drops `k_proj`/`v_proj`
# entirely on KV-shared layers, may set values equal to keys, and its MLP is GeGLU. None of
# that is the Gemma 3 body, so the module-to-body table has no entry and the remaining
# knobs are measured without it.
import json
import os
import signal
import subprocess
import sys
import time

# The PORT2 fixes are pushed, so this checks out the commit that has them instead of
# carrying a patch. Run 9's first attempt (`207e4efc`) shipped an empty patch against the
# old commit for exactly that reason — the working tree was clean once the work was
# committed — and ran without `float16` in `COMPUTE_DTYPES`, which killed its fp16 arms.
COMMIT = "26d4b16e9bdf9baa7fd0288087c5b4a6e374403e"  # research/port2-model-families
FAMILIES_SOURCE = __FAMILIES_SOURCE__  # experiments/kaggle_compat/families.py
CROSS_SOURCE = __CROSS_SOURCE__  # experiments/kaggle_compat/cross.py
QUALITY_SOURCE = __QUALITY_SOURCE__  # experiments/kaggle_compat/quality.py
MODELS = [
    ("gemma4-e2b", "mlx-community/gemma-4-e2b-it-4bit",
     "238767527555cb75a05732a84dff5d6ba0dd6809", 3.55),
    ("gemma4-e4b", "mlx-community/gemma-4-e4b-it-4bit",
     "475b9088d29754a3379866cf5aeb6b41acd313c2", 5.15),
    ("gemma4-e4b-qat", "mlx-community/gemma-4-E4B-it-qat-4bit",
     "0f35c6f6d386f7f74e628bd7c6526ce531212300", 6.80),
]
UNSUPPORTED = ("mlx-community/gemma-4-12b-it-4bit", "73bcf09092aa277861d5a191b989b666f7f32e8f")
ABOVE_CEILING = ("mlx-community/gemma-4-26b-a4b-it-4bit",
                 "0d77464eeb233a2da68ebf9d7dc4edaac7db956d")
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable
DEADLINE = time.time() + 72 * 60
os.makedirs(f"{WORK}/logs", exist_ok=True)
for name, source in (("families", FAMILIES_SOURCE), ("cross", CROSS_SOURCE),
                     ("quality", QUALITY_SOURCE)):
    with open(f"/tmp/{name}.py", "w") as stream:
        stream.write(source)
report = {"schema": "ironmule.port2-kaggle.v9", "commit": COMMIT, "stages": {},
          "screening": {}, "performance_claim": False,
          "models": [{"key": k, "model_id": m, "revision": r, "gb": g} for k, m, r, g in MODELS]}
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
            f"&& git -C {REPO} rev-parse HEAD && git -C {REPO} status --short")
sh("venv", f"{SYS} -m pip install -q uv && {SYS} -m uv python install 3.12 "
           f"&& {SYS} -m uv venv --python-preference only-managed --python 3.12 {VENV}")
sh("install", f"{SYS} -m uv pip install --python {PY} -e '{REPO}[cuda]' 'mlx-lm==0.31.3' psutil scipy pyarrow",
   timeout=1200)
sh("freeze", f"{SYS} -m uv pip freeze --python {PY}")

# -- what the model type costs, recorded rather than asserted -------------------------------
unsupported, rev_unsupported = UNSUPPORTED
sh("download_gemma4_unified", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                              f"print(s('{unsupported}', revision='{rev_unsupported}'))\"", timeout=900)
sh("load_gemma4_unified", f"""{PY} - <<'PYEOF'
import json
import sys
sys.path.insert(0, "/tmp/IronMule")
from ironmule.tune import resolve_local_model
out = {{"model_id": "{unsupported}", "expected": "gemma4_unified is not implemented in mlx-lm 0.31.3"}}
try:
    from mlx_lm import load
    load(str(resolve_local_model("{unsupported}", "{rev_unsupported}").path))
    out["loaded"] = True
except Exception as exc:
    out["loaded"] = False
    out["error"] = f"{{type(exc).__name__}}: {{exc}}"
with open("{WORK}/gemma4-unified-load.json", "w") as stream:
    json.dump(out, stream, indent=1)
print(json.dumps(out, indent=1))
PYEOF""", timeout=600)

# -- pass 1: download and screen the three that fit ------------------------------------------
survivors = []
for key, model_id, revision, gb in MODELS:
    code, _ = sh(f"download_{key}", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                                    f"print(s('{model_id}', revision='{revision}'))\"", timeout=1200)
    if code != 0:
        continue
    sh(f"screen_{key}", f"{PY} /tmp/families.py {model_id} {revision} {WORK}/families-{key}.json 16",
       timeout=1200)
    try:
        with open(f"{WORK}/families-{key}.json") as stream:
            screened = json.load(stream)
    except (OSError, ValueError):
        continue
    report["screening"][key] = {k: v for k, v in screened.items() if k != "arms"}
    report["screening"][key]["arms"] = [
        {k: v for k, v in arm.items() if k != "tokens"} for arm in screened.get("arms", [])]
    save()
    if screened.get("ok"):
        survivors.append((key, model_id, revision, bool(screened.get("tuned_profile_reduced"))))
report["survivors"] = [entry[0] for entry in survivors]
save()

# -- pass 2: stock against exact, float32 and float16 -----------------------------------------
TUNED = {"compiled_fixed_cache": True, "fused_argmax": True, "head_skip_prefill": True,
         "readback_every": 2, "capacity_slack": 128, "fuse_projections": True}
STOCK = {"name": "stock", "env": {"MLX_MAX_OPS_PER_BUFFER": "20", "MLX_MAX_MB_PER_BUFFER": "100"},
         "knobs": {}, "mode": "interactive"}
for key, model_id, revision, reduced in survivors:
    knobs = {**TUNED, "fuse_projections": False} if reduced else dict(TUNED)
    configs = [STOCK,
               {"name": "ironmule_exact", "env": {}, "knobs": knobs, "mode": "throughput"},
               {"name": "ironmule_fp32", "env": {}, "knobs": knobs, "mode": "throughput",
                "dtype": "float32"},
               {"name": "ironmule_fp16", "env": {}, "knobs": knobs, "mode": "throughput",
                "dtype": "float16"}]
    sh(f"cross_{key}", f"{PY} /tmp/cross.py {model_id} {revision} '{json.dumps(configs)}' "
                       f"{WORK}/cross-{key}.json 2 1", timeout=2400)

# -- the gates, on the one small enough to hold two precisions at once ------------------------
sh("download_wikitext", f"{PY} -c \"from huggingface_hub import hf_hub_download as d; import pyarrow.parquet as pq; "
                        "p = d('Salesforce/wikitext', 'wikitext-2-raw-v1/test-00000-of-00001.parquet', repo_type='dataset'); "
                        "open('/tmp/wikitext.txt', 'w').write(''.join(pq.read_table(p).column('text').to_pylist())); print(p)\"")
e2b, rev_e2b = MODELS[0][1], MODELS[0][2]
for precision in ("bf16", "float32", "float16"):
    sh(f"quality_gemma4-e2b-{precision}",
       f"QUALITY_ONLY={precision} {PY} /tmp/quality.py {e2b} {rev_e2b} /tmp/wikitext.txt "
       f"{WORK}/quality-gemma4-e2b-{precision}.json 16 512", timeout=1500)

# -- the ceiling, one point above the 13.26 GB that is known to load --------------------------
big, rev_big = ABOVE_CEILING
sh("download_gemma4_26b", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                          f"print(s('{big}', revision='{rev_big}'))\"", timeout=1800)
sh("load_gemma4_26b", f"""{PY} - <<'PYEOF'
import json
import mlx.core as mx
import sys
sys.path.insert(0, "/tmp/IronMule")
import ironmule
from ironmule.tune import load_engine
out = {{"model_id": "{big}", "gb_on_disk": 15.34}}
try:
    engine, tokenizer = load_engine("{big}", ironmule.BASELINE, revision="{rev_big}")
    out["loaded"] = True
    out["weights_bytes"] = int(mx.get_active_memory())
    with ironmule.Runtime(engine, tokenizer, model_id="{big}") as runtime:
        result = runtime.generate("Explain in two sentences why the sky is blue.", max_tokens=16)
        out["text"] = result.text
    out["peak_memory_bytes"] = int(mx.get_peak_memory())
except Exception as exc:
    out["loaded"] = False
    out["error"] = f"{{type(exc).__name__}}: {{exc}}"
with open("{WORK}/gemma4-26b-load.json", "w") as stream:
    json.dump(out, stream, indent=1)
print(json.dumps(out, indent=1))
PYEOF""", timeout=1200)
report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
