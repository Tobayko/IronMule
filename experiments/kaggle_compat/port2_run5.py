# PORT2 run 5: the second T4, the hybrid guard on real weights, and the quality gate
# run 4 ran out of clock for.
# Private notebook, internet on (pip, git, Hugging Face). Free quota: one run, <= 2.5 h.
#
#   A. The cell has two T4s and every PORT2 number so far used one, because one MLX process
#      uses one device. That set the ceiling: 13.26 GB runs, 16.05 GB does not fit. mlx-lm
#      implements tensor parallelism for llama, qwen2, qwen3, qwen3_5, gpt_oss and
#      ministral3 — every family measured here. Prerequisite first, on a model small enough
#      to have a single-card reference: do two ranks each take a card, does the load
#      survive on Turing, do the tokens still match. IronMule is single process and cannot
#      join a group yet; this decides whether opening that entry is worth it.
#   B. Run 4 bisected Qwen 3.5's non-determinism to the grouped path itself — an arm with no
#      knobs was already 3/6 and produced two digests. The patch refuses throughput mode on
#      a recurrent cache. Verify on the real model that it refuses, and that interactive
#      still works.
#   C. Mistral 24B's float32 quality gate: run 4's bf16 half hit the deadline at 848 s.
#      Fewer chunks, more clock, and it runs first among the quality work.
#   D. Why float32 loses on llama and wins everywhere else. Same 4-bit matvec diagnostic
#      PORT1 used, at llama's shapes next to Qwen 3 8B's, which are its attention twin.
import json
import os
import signal
import subprocess
import sys
import time

COMMIT = "8064419f409366e08f8e651e747f483d9c0483df"  # origin/HEAD
PATCH_SOURCE = __PATCH_SOURCE__
SHARD_SOURCE = __SHARD_SOURCE__  # experiments/kaggle_compat/shard_probe.py
QMM_SOURCE = __QMM_SOURCE__  # experiments/kaggle_compat/bench_qmm.py
QUALITY_SOURCE = __QUALITY_SOURCE__  # experiments/kaggle_compat/quality.py
QWEN8 = ("mlx-community/Qwen3-8B-4bit", "545dc4251c05440727734bcd94334791f6ab0192")
QWEN35 = ("mlx-community/Qwen3.5-9B-MLX-4bit", "938d8919941c6e7efd3c7150eff7fe9d12afa631")
MISTRAL = ("mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit",
           "2a1d5eabfc504747bdc24178394821a1efc0edde")
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable
DEADLINE = time.time() + 130 * 60
os.makedirs(f"{WORK}/logs", exist_ok=True)
with open("/tmp/ironmule.patch", "w") as stream:
    stream.write(PATCH_SOURCE)
for name, source in (("shard_probe", SHARD_SOURCE), ("bench_qmm", QMM_SOURCE),
                     ("quality", QUALITY_SOURCE)):
    with open(f"/tmp/{name}.py", "w") as stream:
        stream.write(source)
report = {"schema": "ironmule.port2-kaggle.v5", "commit": COMMIT, "stages": {},
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
for key, (model_id, revision) in (("qwen3-8b", QWEN8), ("qwen35-9b", QWEN35), ("mistral-24b", MISTRAL)):
    sh(f"download_{key}", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                          f"print(s('{model_id}', revision='{revision}'))\"", timeout=1200)

# -- A. two ranks, two cards -----------------------------------------------------------------
qwen8, rev8 = QWEN8
sh("devices", f"""{PY} - <<'PYEOF'
import mlx.core as mx
for index in (0, 1):
    device = mx.Device(mx.gpu, index)
    mx.set_default_device(device)
    a = mx.random.normal((2048, 2048)); b = a @ a; mx.eval(b)
    print(index, device, "matmul ok", float(b.sum().item()) == float(b.sum().item()),
          "active", int(mx.get_active_memory()))
print("mlx", mx.__version__, "distributed available", mx.distributed.is_available())
PYEOF""")
sh("shard_reference", f"{PY} /tmp/shard_probe.py --reference {qwen8} {rev8} "
                      f"{WORK}/shard-reference-qwen3-8b.json 32", timeout=900)
for backend in ("ring", "nccl"):
    sh(f"shard_qwen3_8b_{backend}",
       f"{VENV}/bin/mlx.launch --hosts 127.0.0.1 -n 2 --backend {backend} --python {PY} "
       f"/tmp/shard_probe.py {qwen8} {WORK}/shard-qwen3-8b-{backend} 32", timeout=1200)

# -- B. the hybrid guard, on the model that produced the measurement -------------------------
qwen35, rev35 = QWEN35
sh("hybrid_guard", f"""{PY} - <<'PYEOF'
import json
import ironmule
from ironmule.tune import load_engine
engine, tokenizer = load_engine("{qwen35}", ironmule.BASELINE, revision="{rev35}")
out = {{}}
try:
    ironmule.Runtime(engine, tokenizer, model_id="{qwen35}", mode=ironmule.ThroughputMode()).close()
    out["throughput_refused"] = False
except ValueError as exc:
    out["throughput_refused"] = str(exc)
engine, tokenizer = load_engine("{qwen35}", ironmule.BASELINE, revision="{rev35}")
with ironmule.Runtime(engine, tokenizer, model_id="{qwen35}") as runtime:
    result = runtime.generate("Explain in two sentences why the sky is blue.", max_tokens=16)
    out["interactive_text"] = result.text
    out["interactive_tokens"] = [int(t) for t in result.tokens]
print(json.dumps(out, indent=1))
PYEOF""", timeout=900)

# -- C. Mistral 24B's float32 quality gate, one precision per process -------------------------
sh("download_wikitext", f"{PY} -c \"from huggingface_hub import hf_hub_download as d; import pyarrow.parquet as pq; "
                        "p = d('Salesforce/wikitext', 'wikitext-2-raw-v1/test-00000-of-00001.parquet', repo_type='dataset'); "
                        "open('/tmp/wikitext.txt', 'w').write(''.join(pq.read_table(p).column('text').to_pylist())); print(p)\"")
mistral, rev_mistral = MISTRAL
for precision in ("bf16", "float32"):
    sh(f"quality_mistral_{precision}",
       f"QUALITY_ONLY={precision} {PY} /tmp/quality.py {mistral} {rev_mistral} /tmp/wikitext.txt "
       f"{WORK}/quality-mistral-24b-{precision}.json 8 512", timeout=2400)

# -- D. where float32 wins and where it does not, at the shapes themselves --------------------
# llama 3.1 8B and Qwen 3 8B share their attention shapes exactly; their MLP widths differ.
LLAMA_SHAPES = json.dumps([[6144, 4096], [28672, 4096], [4096, 14336], [4096, 4096]])
QWEN_SHAPES = json.dumps([[6144, 4096], [24576, 4096], [4096, 12288], [4096, 4096]])
sh("qmm_llama_shapes", f"QMM_SHAPES='{LLAMA_SHAPES}' {PY} /tmp/bench_qmm.py", timeout=900)
sh("qmm_qwen_shapes", f"QMM_SHAPES='{QWEN_SHAPES}' {PY} /tmp/bench_qmm.py", timeout=900)
report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
