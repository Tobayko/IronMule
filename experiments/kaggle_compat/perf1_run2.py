# PERF1 run 2: the native kernel in both numeric plans, tensor-core prefill, and a quality check.
# Private notebook, internet on. Free quota: one run, <= 1.5 h; PERF1 budget 3 h, run 1 used 1.05 h.
# Backlog: docs/PROJECT_FRIDAY_BACKLOG.md, PERF1-C. Screening, no performance claim.
#
# Run 1 (`69dbc7af`) settled tensor parallelism (0.34x, killed) and showed the bf16 kernel at
# 5.5x decode with prefill untouched. This run asks three things, with rules fixed here:
#   * does the kernel also carry the qualified float32 plan ("k32"), and does float16
#     tensor-core prefill ("p16") cut TTFT, each in both plans;
#   * a part runs end to end only if the probe finds it correct (rel. error <= 1e-2 on every
#     shape) and faster than what it replaces (k32/kernel: float32 `quantized_matmul`;
#     p16: bf16 `quantized_matmul` at 512 rows);
#   * quality: mean next-token NLL on WikiText-2, 4 x 256 tokens, through the path each part
#     changes — decode (teacher-forced) for kernel/k32, prefill for p16 — against the same path
#     without it. Diagnostic only: 4 chunks cannot qualify a plan.
import json
import os
import signal
import subprocess
import sys
import time

COMMIT = "b339c817299de173f8a330089afb69973c519353"  # research/port2-model-families
PERF1_SOURCE = __PERF1_SOURCE__  # experiments/kaggle_compat/perf1.py
MODELS = [("qwen3-8b", "mlx-community/Qwen3-8B-4bit", "545dc4251c05440727734bcd94334791f6ab0192"),
          ("qwen3-14b", "mlx-community/Qwen3-14B-4bit", "a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4")]
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable
DEADLINE = time.time() + 75 * 60
os.makedirs(f"{WORK}/logs", exist_ok=True)
with open("/tmp/perf1.py", "w") as stream:
    stream.write(PERF1_SOURCE)
report = {"schema": "ironmule.perf1-kaggle.v2", "commit": COMMIT, "stages": {}, "performance_claim": False}
env = dict(os.environ, PATH=f"{VENV}/bin:" + os.environ["PATH"], PYTHONPATH="", PYTHONNOUSERSITE="1",
           HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONUNBUFFERED="1", MLX_MAX_OPS_PER_BUFFER="400")


def save():
    with open(f"{WORK}/perf1-result.json", "w") as stream:
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


sh("env", "nvidia-smi --query-gpu=index,name,memory.total,clocks.max.sm --format=csv; nproc; free -g")
sh("clone", f"git clone -q https://github.com/Tobayko/IronMule {REPO} && git -C {REPO} checkout -q {COMMIT} "
            f"&& git -C {REPO} rev-parse HEAD")
sh("venv", f"{SYS} -m pip install -q uv && {SYS} -m uv python install 3.12 "
           f"&& {SYS} -m uv venv --python-preference only-managed --python 3.12 {VENV}")
sh("install", f"{SYS} -m uv pip install --python {PY} -e '{REPO}[cuda]' 'mlx-lm==0.31.3' pyarrow", timeout=1200)
sh("freeze", f"{SYS} -m uv pip freeze --python {PY}")

sh("kernel_probe", f"{PY} /tmp/perf1.py kernel {WORK}/kernel-probe.json", timeout=900)
ok = {"kernel": False, "k32": False, "p16": False}
try:
    with open(f"{WORK}/kernel-probe.json") as stream:
        probe = json.load(stream)
    shapes = probe["shapes"].values()
    correct = lambda key: all(s.get(key, 1.0) <= 1e-2 for s in shapes)  # noqa: E731
    ok["kernel"] = correct("kernel_bf16_rel_err") and probe["step_kernel_bf16_ms"] < probe["step_qmm_fp32_ms"]
    ok["k32"] = correct("kernel_f32_rel_err") and probe["step_kernel_f32_ms"] < probe["step_qmm_fp32_ms"]
    ok["p16"] = correct("p16_rel_err") and probe["step_p16_512_ms"] < probe["step_qmm_bf16_512_ms"]
except (OSError, ValueError, KeyError):
    pass
report["parts_ok"] = ok
save()


def allowed(arm):
    return all(ok.get(part, True) for part in arm.split("+"))


E2E = [a for a in ("stock", "fp32", "kernel", "kernel+p16", "fp32+k32", "fp32+k32+p16") if allowed(a)]
NLL = [(a, m) for a, m in (("stock", "decode"), ("kernel", "decode"), ("fp32", "decode"), ("fp32+k32", "decode"),
                           ("stock", "prefill"), ("p16", "prefill"), ("fp32", "prefill"), ("fp32+p16", "prefill"))
       if allowed(a)]
report["e2e_arms"], report["nll_arms"] = E2E, NLL
save()

sh("download_wikitext", f"{PY} -c \"from huggingface_hub import hf_hub_download as d; import pyarrow.parquet as pq; "
                        "p = d('Salesforce/wikitext', 'wikitext-2-raw-v1/test-00000-of-00001.parquet', repo_type='dataset'); "
                        "open('/tmp/wikitext.txt', 'w').write(''.join(pq.read_table(p).column('text').to_pylist())); print(p)\"")

for index, (key, model_id, revision) in enumerate(MODELS):
    code, out = sh(f"download_{key}", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                                      f"print(s('{model_id}', revision='{revision}'))\"", timeout=1200)
    if code != 0:
        continue
    path = out.strip().splitlines()[-1]
    for arm in E2E:
        sh(f"e2e_{key}_{arm}", f"{PY} /tmp/perf1.py e2e {path} '{arm}' {WORK}/e2e-{key}-{arm}.json", timeout=900)
    if index == 0:  # quality on the 8B only; the 14B is speed
        for arm, path_mode in NLL:
            sh(f"nll_{key}_{arm}_{path_mode}", f"{PY} /tmp/perf1.py nll {path} '{arm}' {path_mode} /tmp/wikitext.txt "
                                               f"{WORK}/nll-{key}-{arm}-{path_mode}.json", timeout=900)

report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
