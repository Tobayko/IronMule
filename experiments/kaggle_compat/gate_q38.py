# GATE-Q38: the quality gate for Qwen3.8 27B under `native`, which no gate has covered (DEMO5 ran it
# for speed only: 36.13 -> 8.41 s over two T4s). Private notebook, internet on. Quota: the user asked
# on 2026-09-28 to qualify it. Rules, before the run (PERF1-Y's as BACKLOG8 ran them, three changes):
#   * `perf1.py nll` at IronMule main 58cbb1d, BOS on every chunk where the tokenizer has one (Qwen
#     has none), WikiText-2 raw test; the native gates' arms: prefill path stock bf16 against `p16`,
#     decode path stock bf16 against `kernel` with pinned arithmetic.
#   * Change 1: 32 chunks x 256 tokens instead of 16 x 512, the same 8192 tokens: this family's bf16
#     path went non-finite at 512 tokens on CUDA (PORT2), and every earlier Qwen 3.5 run used 256.
#   * Change 2: the model does not fit one card, so every arm runs PERF1-O's layer pipeline over the
#     cell's two T4s (`mlx.launch`, one card per rank); rank 0 writes the gate file.
#   * Change 3: MLX_USE_CUDA_GRAPHS=0 on every arm (graphs break this family's determinism, run 13)
#     and PERF1_P16_SYNC=1.
#   * Pass per path: gate_summary.py, upper bound of the 10 000-sample paired chunk bootstrap (seed
#     20260915) <= 1.005 and no non-finite value. Kill: a path above the bound refuses `native` for
#     `mlx_lm.models.qwen3_5` (a table row); a pass on both paths qualifies it for this checkpoint
#     on pre-Ampere CUDA, where the product itself still needs a card that holds 15 GB.
#   * Order: prefill (minutes), then decode, the kernel before stock (about 70 min).
import json
import os
import signal
import subprocess
import sys
import time

COMMIT = "58cbb1dadac725c63632399379e9ef15222d16b8"
QWEN38 = ("mlx-community/Qwen3.8-27B-4bit", "10c35caafbb80f7dc6a7a432cdd11af10a6d4818")
WIKITEXT = ("Salesforce/wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3",
            "wikitext-2-raw-v1/test-00000-of-00001.parquet")
GATE_ENV = "PERF1_NLL_CHUNKS=32 PERF1_NLL_TOKENS=256 MLX_USE_CUDA_GRAPHS=0 PERF1_P16_SYNC=1 "
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable
DEADLINE = time.time() + 170 * 60
os.makedirs(f"{WORK}/logs", exist_ok=True)
report = {"schema": "ironmule.gate-q38-kaggle.v1", "commit": COMMIT, "stages": {}, "performance_claim": False}
env = dict(os.environ, PATH=f"{VENV}/bin:" + os.environ["PATH"], PYTHONPATH="", PYTHONNOUSERSITE="1",
           HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONUNBUFFERED="1", IRONMULE_HOME="/tmp/ironmule-home")


def save():
    with open(f"{WORK}/gate-q38-result.json", "w") as stream:
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


sh("env", "nvidia-smi --query-gpu=index,name,driver_version,memory.total,clocks.max.sm --format=csv; nproc; free -g")
sh("clone", f"git clone -q https://github.com/Tobayko/IronMule {REPO} && git -C {REPO} checkout -q {COMMIT} "
            f"&& git -C {REPO} rev-parse HEAD && git -C {REPO} status --short")
sh("venv", f"{SYS} -m pip install -q uv && {SYS} -m uv python install 3.12 "
           f"&& {SYS} -m uv venv --python-preference only-managed --python 3.12 {VENV}")
sh("install", f"{SYS} -m uv pip install --python {PY} -e '{REPO}[cuda]' 'mlx-lm==0.31.3' pyarrow 'pytest>=8' 'pytest-xdist>=3' psutil scipy", timeout=1200)
sh("freeze", f"{SYS} -m uv pip freeze --python {PY}")


def download(key, model):
    code, out = sh(f"download_{key}", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                                      f"print(s('{model[0]}', revision='{model[1]}'))\"", timeout=1200)
    return out.strip().splitlines()[-1] if code == 0 else None


sh("download_wikitext", f"{PY} -c \"from huggingface_hub import hf_hub_download as d; import pyarrow.parquet as pq; "
                        f"p = d('{WIKITEXT[0]}', '{WIKITEXT[2]}', repo_type='dataset', revision='{WIKITEXT[1]}'); "
                        "open('/tmp/wikitext.txt', 'w').write(''.join(pq.read_table(p).column('text').to_pylist())); print(p)\"")

path = download("qwen38-27b", QWEN38)
if path:
    PERF1 = f"{REPO}/experiments/kaggle_compat/perf1.py"
    PIPE = f"{VENV}/bin/mlx.launch --hosts 127.0.0.1 -n 2 --backend ring -- {PY} {PERF1}"
    for path_mode, arms in (("prefill", (("stock", ""), ("p16", ""))),
                            ("decode", (("kernel", "PERF1_ARITH=pinned "), ("stock", "")))):
        for arm, prefix in arms:
            out = f"{WORK}/gate-qwen38-27b-{arm}-{path_mode}.json"
            # `mlx.launch` exits 0 when its ranks fail (run 11); rank 1's copy goes to logs so the
            # summary pairs rank 0's files only.
            sh(f"gate_qwen38-27b_{arm}_{path_mode}", f"{GATE_ENV}{prefix}{PIPE} nll {path} {arm} {path_mode} "
                                                     f"/tmp/wikitext.txt {out}; mv -f {out[:-5]}-rank1.json "
                                                     f"{WORK}/logs/ 2>/dev/null; test -s {out}", timeout=6000)
    sh("summary", f"{PY} {REPO}/experiments/kaggle_compat/gate_summary.py {WORK} qwen38-27b "
                  f"{WORK}/gate-summary-qwen38-27b.json")
report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
