# PERF1 run 3: a cheaper nibble conversion, speculative decoding, and IronMule's own runtime.
# Private notebook, internet on. Free quota: one run, <= 1.25 h. Backlog: PERF1-D, B13.
# Screening, no performance claim. Nothing measured in runs 1-2 is repeated except the one
# in-run control every comparison needs (`kernel+p16`, the run-2 winner).
#
# Rules fixed here, before the run:
#   * nibble conversion: "magic" replaces "cvt" everywhere below if the probe finds it correct
#     (rel. error <= 1e-2) and faster at M = 1 on the estimated decode step; else "cvt" stays.
#   * the verify widths M = 2..5 must be bit-identical to M single rows, or speculative
#     decoding is reported as not exact.
#   * speculative decoding (B13, reopened: new hardware and a new verify kernel): Qwen 3 0.6B
#     drafts for 8B and 14B, k = 2, 3, 4. B13's kill: acceptance < 0.65 at k = 3; and here
#     also: median decode rate not above plain decode of the same arm in the same process.
#   * IronMule combination: its benchmark workload with the PORT2 tuned knobs, with and without
#     the kernel routes. Kill: the knobs add nothing on top of the kernel (ratio >= 0.98).
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
DRAFT = ("mlx-community/Qwen3-0.6B-4bit", "73e3e38d981303bc594367cd910ea6eb48349da8")
TUNED = {"compiled_fixed_cache": True, "fused_argmax": True, "head_skip_prefill": True,
         "readback_every": 2, "capacity_slack": 128, "fuse_projections": True}
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable
DEADLINE = time.time() + 65 * 60
os.makedirs(f"{WORK}/logs", exist_ok=True)
with open("/tmp/perf1.py", "w") as stream:
    stream.write(PERF1_SOURCE)
report = {"schema": "ironmule.perf1-kaggle.v3", "commit": COMMIT, "stages": {}, "performance_claim": False}
env = dict(os.environ, PATH=f"{VENV}/bin:" + os.environ["PATH"], PYTHONPATH="", PYTHONNOUSERSITE="1",
           HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONUNBUFFERED="1", MLX_MAX_OPS_PER_BUFFER="400",
           IRONMULE_HOME="/tmp/ironmule-home")


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


def download(key, model_id, revision):
    code, out = sh(f"download_{key}", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                                      f"print(s('{model_id}', revision='{revision}'))\"", timeout=1200)
    return out.strip().splitlines()[-1] if code == 0 else None


sh("env", "nvidia-smi --query-gpu=index,name,memory.total,clocks.max.sm --format=csv; nproc; free -g")
sh("clone", f"git clone -q https://github.com/Tobayko/IronMule {REPO} && git -C {REPO} checkout -q {COMMIT} "
            f"&& git -C {REPO} rev-parse HEAD")
sh("venv", f"{SYS} -m pip install -q uv && {SYS} -m uv python install 3.12 "
           f"&& {SYS} -m uv venv --python-preference only-managed --python 3.12 {VENV}")
sh("install", f"{SYS} -m uv pip install --python {PY} -e '{REPO}[cuda]' 'mlx-lm==0.31.3'", timeout=1200)
sh("freeze", f"{SYS} -m uv pip freeze --python {PY}")

sh("kernel_probe", f"{PY} /tmp/perf1.py kernel {WORK}/kernel-probe.json", timeout=900)
qf, verify_exact = "cvt", False
try:
    with open(f"{WORK}/kernel-probe.json") as stream:
        probe = json.load(stream)
    shapes = probe["shapes"].values()
    if (all(s.get("kernel_bf16_magic_rel_err", 1.0) <= 1e-2 for s in shapes)
            and probe["step_kernel_bf16_magic_ms"] < probe["step_kernel_bf16_cvt_ms"]):
        qf = "magic"
    verify_exact = all(s.get(f"kernel_bf16_{qf}_m{m}_equal_to_m1") is True for s in shapes for m in (2, 3, 4, 5))
except (OSError, ValueError, KeyError):
    pass
env["PERF1_QF"] = qf
report["qf"], report["verify_exact"] = qf, verify_exact
save()

draft_path = download("qwen3-0.6b", *DRAFT)
for index, (key, model_id, revision) in enumerate(MODELS):
    path = download(key, model_id, revision)
    if path is None:
        continue
    if index == 0:  # the in-run control, and the conversion it is compared with
        for arm, conversion in (("kernel+p16", "cvt"), ("kernel+p16", "magic")):
            sh(f"e2e_{key}_{arm}_{conversion}", f"PERF1_QF={conversion} {PY} /tmp/perf1.py e2e {path} '{arm}' "
                                                f"{WORK}/e2e-{key}-{arm}-{conversion}.json", timeout=900)
    if draft_path:
        spec_arms = ["kernel+p16"] + (["fp32+k32+p16"] if index == 0 else [])
        for arm in spec_arms:
            sh(f"spec_{key}_{arm}", f"{PY} /tmp/perf1.py spec {path} '{arm}' {draft_path} 2,3,4 "
                                    f"{WORK}/spec-{key}-{arm}.json", timeout=1200)
    if index == 0:
        for arm, knobs, mode in (("stock", {}, "interactive"), ("stock", TUNED, "throughput"),
                                 ("kernel+p16", {}, "interactive"), ("kernel+p16", TUNED, "throughput")):
            name = f"{arm}-{'tuned' if knobs else 'baseline'}-{mode}"
            sh(f"ironmule_{key}_{name}", f"{PY} /tmp/perf1.py ironmule {model_id} {revision} '{arm}' "
                                         f"'{json.dumps(knobs)}' {mode} {WORK}/ironmule-{key}-{name}.json",
               timeout=900)

report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
