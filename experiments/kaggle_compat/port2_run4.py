# PORT2 run 4: the fusion fix on real weights, the hybrid non-determinism, and the two
# quality gates that no single process could hold.
# Private notebook, internet on (pip, git, Hugging Face). Free quota: one run, <= 3 h.
#
# Run 3 found the cause of the corrupted fused llama and the patch this notebook carries
# fixes it: every split half is copied out of the fused buffer with `mx.contiguous` before
# `rope`, the cache or the attention kernel sees it. Three things follow, in this order.
#   A. Does the fix hold on real weights, and what do the copies cost? llama is the target;
#      Gemma 3 4B and Qwen 3 8B/14B are the control, because they were already token
#      identical in run 2 and must stay so at no worse a ratio. The llama probe reruns with
#      CUDA graph capture ON, which is what aborted the process in run 3.
#   B. Qwen 3.5's exact arm was 4/6 token identical over three repetitions and not
#      deterministic across processes. One arm per knob, plus an arm with no knobs at all in
#      throughput mode, so the answer names either a knob or the grouping itself.
#   C. gpt-oss 20B and Mistral 24B have no float32 quality gate, because 11.2 and 13.3 GB
#      do not fit twice on a 15 GB card. `quality.py` now decodes one precision per process,
#      so the pair is compared across two processes instead of inside one.
import json
import math
import os
import signal
import statistics as st
import subprocess
import sys
import time

COMMIT = "8064419f409366e08f8e651e747f483d9c0483df"  # origin/HEAD
PATCH_SOURCE = __PATCH_SOURCE__  # uncommitted IronMule changes under test, embedded at submit time
PROBE_SOURCE = __PROBE_SOURCE__  # experiments/kaggle_compat/fused_probe.py
CROSS_SOURCE = __CROSS_SOURCE__  # experiments/kaggle_compat/cross.py
QUALITY_SOURCE = __QUALITY_SOURCE__  # experiments/kaggle_compat/quality.py
MODELS = {
    "gemma3-4b": ("mlx-community/gemma-3-4b-it-4bit", "93724907d4ed1745d2fe50baadf3b0b01a65abf2"),
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
FUSED = ("gemma3-4b", "llama31-8b", "qwen3-8b", "qwen3-14b")
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable
DEADLINE = time.time() + 150 * 60
os.makedirs(f"{WORK}/logs", exist_ok=True)
with open("/tmp/ironmule.patch", "w") as stream:
    stream.write(PATCH_SOURCE)
for name, source in (("fused_probe", PROBE_SOURCE), ("cross", CROSS_SOURCE),
                     ("quality", QUALITY_SOURCE)):
    with open(f"/tmp/{name}.py", "w") as stream:
        stream.write(source)
report = {"schema": "ironmule.port2-kaggle.v4", "commit": COMMIT, "stages": {},
          "quality_pairs": {}, "performance_claim": False}
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
sh("install", f"{SYS} -m uv pip install --python {PY} -e '{REPO}[cuda]' 'mlx-lm==0.31.3' "
              "'pytest>=8' psutil scipy pyarrow", timeout=1200)
sh("freeze", f"{SYS} -m uv pip freeze --python {PY}")
sh("fusion_self_check", f"{PY} -m ironmule.fast", cwd=REPO)
sh("pytest_fusion", f"{PY} -m pytest -q -n0 -o addopts='' tests/engine/test_ironmule.py", cwd=REPO)
for key, (model_id, revision) in MODELS.items():
    sh(f"download_{key}", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                          f"print(s('{model_id}', revision='{revision}'))\"", timeout=1200)

# -- A. the fusion fix on real weights ------------------------------------------------------
llama, rev_llama = MODELS["llama31-8b"]
sh("probe_llama_graphs_on", f"{PY} /tmp/fused_probe.py {llama} {rev_llama} "
                            f"{WORK}/fused-probe-graphs-on.json 32", timeout=900)
STOCK = {"name": "stock", "env": {"MLX_MAX_OPS_PER_BUFFER": "20", "MLX_MAX_MB_PER_BUFFER": "100"},
         "knobs": {}, "mode": "interactive"}
TUNED = {"compiled_fixed_cache": True, "fused_argmax": True, "head_skip_prefill": True,
         "readback_every": 2, "capacity_slack": 128, "fuse_projections": True}
for key in FUSED:
    model_id, revision = MODELS[key]
    configs = [STOCK,
               {"name": "ironmule_fused", "env": {}, "knobs": TUNED, "mode": "throughput"},
               {"name": "ironmule_fused_fp32", "env": {}, "knobs": TUNED, "mode": "throughput",
                "dtype": "float32"}]
    sh(f"cross_fused_{key}", f"{PY} /tmp/cross.py {model_id} {revision} '{json.dumps(configs)}' "
                             f"{WORK}/cross-fused-{key}.json 2 1", timeout=2700)

# -- B. which knob, or the grouping itself, costs Qwen 3.5 its determinism? -------------------
qwen, rev_qwen = MODELS["qwen35-9b"]
bisect = [STOCK, {"name": "no_knobs_throughput", "env": {}, "knobs": {}, "mode": "throughput"}]
for knob, value in (("compiled_fixed_cache", True), ("fused_argmax", True),
                    ("head_skip_prefill", True), ("readback_every", 2)):
    bisect.append({"name": f"only_{knob}", "env": {}, "knobs": {knob: value}, "mode": "throughput"})
sh("cross_qwen35_bisect", f"{PY} /tmp/cross.py {qwen} {rev_qwen} '{json.dumps(bisect)}' "
                          f"{WORK}/cross-qwen35-bisect.json 2 1", timeout=3000)

# -- C. the two float32 quality gates that need one process per precision ---------------------
sh("download_wikitext", f"{PY} -c \"from huggingface_hub import hf_hub_download as d; import pyarrow.parquet as pq; "
                        "p = d('Salesforce/wikitext', 'wikitext-2-raw-v1/test-00000-of-00001.parquet', repo_type='dataset'); "
                        "open('/tmp/wikitext.txt', 'w').write(''.join(pq.read_table(p).column('text').to_pylist())); print(p)\"")
for key in ("gptoss-20b", "mistral-24b"):
    model_id, revision = MODELS[key]
    for precision in ("bf16", "float32"):
        sh(f"quality_{key}_{precision}",
           f"QUALITY_ONLY={precision} {PY} /tmp/quality.py {model_id} {revision} /tmp/wikitext.txt "
           f"{WORK}/quality-{key}-{precision}.json 12 512", timeout=2100)
    try:
        halves = {p: json.load(open(f"{WORK}/quality-{key}-{p}.json")) for p in ("bf16", "float32")}
    except (OSError, ValueError) as exc:
        report["quality_pairs"][key] = {"error": repr(exc)}
        save()
        continue
    # Paired across processes: the same chunks of the same text, one precision each. The
    # ratio is the same quantity `quality.py` bootstraps in-process, without the KL and
    # top-1 terms, which need both sets of logits at once and so cannot be had here.
    pairs = list(zip((row["nll_bf16"] for row in halves["bf16"]["rows"]),
                     (row["nll_fp32"] for row in halves["float32"]["rows"])))
    report["quality_pairs"][key] = {
        "chunks": len(pairs),
        "ppl_bf16": math.exp(st.mean(a for a, _ in pairs)),
        "ppl_fp32": math.exp(st.mean(b for _, b in pairs)),
        "ppl_ratio_fp32_over_bf16": math.exp(st.mean(b for _, b in pairs) - st.mean(a for a, _ in pairs)),
        "max_abs_delta_nll": max(abs(b - a) for a, b in pairs),
        "note": "paired across two processes; KL and top-1 need both logit sets in one process",
    }
    save()
report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
