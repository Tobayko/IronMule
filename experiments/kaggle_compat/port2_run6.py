# PORT2 run 6: the two things run 5 opened — a faster numeric plan, and a bigger model.
# Private notebook, internet on (pip, git, Hugging Face). Free quota: one run, <= 2.5 h.
#
#   A. float16. Run 5's 4-bit matvec diagnostic, at llama 3.1 8B's and Qwen 3 8B's own
#      shapes, put float16 at 0.32-0.33 of bf16 where float32 sits at 0.58. float32 is
#      already worth about 2x end to end on this card; if that ratio carries, float16 is
#      worth about 1.75x again. float16's exponent range is not bf16's, so this measures
#      speed and quality together and claims neither without the other.
#   B. The second card. Run 5 proved tensor parallelism works here: two ranks, one T4 each,
#      ring backend, tokens identical to the single-card reference (nccl failed with
#      `There is no Stream(gpu, 1) in current thread`). That lifts the weight ceiling from
#      about 13.3 GB to about 30 GB, so the checkpoints that did not fit now can. Qwen3.8
#      27B at 16.05 GB is the one to try: newest Qwen, `qwen3_5`, which implements `shard`.
#      Stock mlx-lm only — IronMule is single process and cannot join a group.
import json
import os
import signal
import subprocess
import sys
import time

COMMIT = "8064419f409366e08f8e651e747f483d9c0483df"  # origin/HEAD
PATCH_SOURCE = __PATCH_SOURCE__
SHARD_SOURCE = __SHARD_SOURCE__  # experiments/kaggle_compat/shard_probe.py
CROSS_SOURCE = __CROSS_SOURCE__  # experiments/kaggle_compat/cross.py
QUALITY_SOURCE = __QUALITY_SOURCE__  # experiments/kaggle_compat/quality.py
SMALL = {
    "gemma3-4b": ("mlx-community/gemma-3-4b-it-4bit", "93724907d4ed1745d2fe50baadf3b0b01a65abf2"),
    "qwen3-8b": ("mlx-community/Qwen3-8B-4bit", "545dc4251c05440727734bcd94334791f6ab0192"),
    "gptoss-20b": ("mlx-community/gpt-oss-20b-MXFP4-Q4",
                   "f356f2747216d7e98fee755df25987459fc19089"),
}
# Two cards, so the ceiling is no longer one card's 15360 MiB. Sharded arms only.
BIG = {
    "qwen38-27b": "mlx-community/Qwen3.8-27B-4bit",
    "qwen3-8b": "mlx-community/Qwen3-8B-4bit",  # the control that already has a reference
}
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable
DEADLINE = time.time() + 135 * 60
os.makedirs(f"{WORK}/logs", exist_ok=True)
with open("/tmp/ironmule.patch", "w") as stream:
    stream.write(PATCH_SOURCE)
for name, source in (("shard_probe", SHARD_SOURCE), ("cross", CROSS_SOURCE),
                     ("quality", QUALITY_SOURCE)):
    with open(f"/tmp/{name}.py", "w") as stream:
        stream.write(source)
report = {"schema": "ironmule.port2-kaggle.v6", "commit": COMMIT, "stages": {},
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


sh("env", "nvidia-smi --query-gpu=index,name,memory.total --format=csv; nproc; free -g; df -h / | tail -1")
sh("clone", f"git clone -q https://github.com/Tobayko/IronMule {REPO} && git -C {REPO} checkout -q {COMMIT} "
            f"&& git -C {REPO} apply /tmp/ironmule.patch && git -C {REPO} diff --stat")
sh("venv", f"{SYS} -m pip install -q uv && {SYS} -m uv python install 3.12 "
           f"&& {SYS} -m uv venv --python-preference only-managed --python 3.12 {VENV}")
sh("install", f"{SYS} -m uv pip install --python {PY} -e '{REPO}[cuda]' 'mlx-lm==0.31.3' psutil scipy pyarrow",
   timeout=1200)
sh("freeze", f"{SYS} -m uv pip freeze --python {PY}")
for key, (model_id, revision) in SMALL.items():
    sh(f"download_{key}", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                          f"print(s('{model_id}', revision='{revision}'))\"", timeout=1200)

# -- A. is float16 the plan float32 should have been? ----------------------------------------
STOCK = {"name": "stock", "env": {"MLX_MAX_OPS_PER_BUFFER": "20", "MLX_MAX_MB_PER_BUFFER": "100"},
         "knobs": {}, "mode": "interactive"}
TUNED = {"compiled_fixed_cache": True, "fused_argmax": True, "head_skip_prefill": True,
         "readback_every": 2, "capacity_slack": 128, "fuse_projections": True}
LEAN = {"head_skip_prefill": True}  # gpt-oss has biased projections; fusion refuses them
for key, (model_id, revision) in SMALL.items():
    knobs = LEAN if key == "gptoss-20b" else TUNED
    configs = [STOCK,
               {"name": "ironmule_fp32", "env": {}, "knobs": knobs, "mode": "throughput",
                "dtype": "float32"},
               {"name": "ironmule_fp16", "env": {}, "knobs": knobs, "mode": "throughput",
                "dtype": "float16"}]
    sh(f"cross_fp16_{key}", f"{PY} /tmp/cross.py {model_id} {revision} '{json.dumps(configs)}' "
                            f"{WORK}/cross-fp16-{key}.json 2 1", timeout=2700)
sh("download_wikitext", f"{PY} -c \"from huggingface_hub import hf_hub_download as d; import pyarrow.parquet as pq; "
                        "p = d('Salesforce/wikitext', 'wikitext-2-raw-v1/test-00000-of-00001.parquet', repo_type='dataset'); "
                        "open('/tmp/wikitext.txt', 'w').write(''.join(pq.read_table(p).column('text').to_pylist())); print(p)\"")
# A float16 speed-up means nothing without this: float16 can overflow where bfloat16 cannot.
for key in ("gemma3-4b", "qwen3-8b"):
    model_id, revision = SMALL[key]
    for precision in ("bf16", "float16"):
        sh(f"quality16_{key}_{precision}",
           f"QUALITY_ONLY={precision} {PY} /tmp/quality.py {model_id} {revision} /tmp/wikitext.txt "
           f"{WORK}/quality16-{key}-{precision}.json 16 512", timeout=1800)

# -- B. the checkpoints that never fit on one card -------------------------------------------
for key, repo in BIG.items():
    sh(f"download_shard_{key}", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                                f"print(s('{repo}'))\"", timeout=2400)
    sh(f"shard_{key}", f"{VENV}/bin/mlx.launch --hosts 127.0.0.1 -n 2 --backend ring --python {PY} "
                       f"/tmp/shard_probe.py {repo} {WORK}/shard-{key} 48", timeout=2400)
report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
