# TEST2-G: gpt-oss 20B's float32 and float16 ratios with more than two processes; TEST2 ran
# out of time before them. Private notebook, internet on. Quota: the user asked for this run on
# 2026-09-24 (TEST2 estimated ~35 min); stages are capped at 45 min. Rules
# (docs/PROJECT_FRIDAY_BACKLOG.md, TEST2-G), the same as TEST2's:
#   * `cross.py child` from the cloned commit, the arms as PORT2 run 6 published them, each in
#     a fresh process, arm order rotated by repetition as `cross.py main` rotates it; the file
#     is rewritten after every process, and its summary uses `cross.py`'s formulas.
#   * gpt-oss 20B: stock, float32 (0.2818), float16 (0.1991); 3 repetitions. A repetition
#     starts only with its full estimated duration left (TEST2's failure): 750 s from PORT2
#     run 6 for the first, the measured one after that, times 1.15.
#   * A new median more than 5% from the published ratio replaces it. Stock's output digest
#     across processes is the model's own determinism control.
import json
import os
import statistics as st
import signal
import subprocess
import sys
import time

COMMIT = __COMMIT__
STOCK = {"name": "stock", "env": {"MLX_MAX_OPS_PER_BUFFER": "20", "MLX_MAX_MB_PER_BUFFER": "100"},
         "knobs": {}, "mode": "interactive"}
MODELS = [
    ("gptoss-20b", "mlx-community/gpt-oss-20b-MXFP4-Q4", "f356f2747216d7e98fee755df25987459fc19089", 3,
     [STOCK,
      {"name": "ironmule_fp32", "env": {}, "knobs": {"head_skip_prefill": True}, "mode": "throughput",
       "dtype": "float32"},
      {"name": "ironmule_fp16", "env": {}, "knobs": {"head_skip_prefill": True}, "mode": "throughput",
       "dtype": "float16"}]),
]
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable
DEADLINE = time.time() + 45 * 60
FIRST_REP_ESTIMATE = 750.0  # seconds; PORT2 run 6: 1495 s for two repetitions
os.makedirs(f"{WORK}/logs", exist_ok=True)
report = {"schema": "ironmule.test2g-kaggle.v1", "commit": COMMIT, "stages": {}, "performance_claim": False}
env = dict(os.environ, PATH=f"{VENV}/bin:" + os.environ["PATH"], PYTHONPATH="", PYTHONNOUSERSITE="1",
           HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONUNBUFFERED="1", IRONMULE_HOME="/tmp/ironmule-home")


def save():
    with open(f"{WORK}/test2g-result.json", "w") as stream:
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
sh("install", f"{SYS} -m uv pip install --python {PY} -e '{REPO}[cuda]' 'mlx-lm==0.31.3'", timeout=1200)
sh("freeze", f"{SYS} -m uv pip freeze --python {PY}")
CROSS = f"{REPO}/experiments/kaggle_compat/cross.py"


def summarise(configs, runs):
    """`cross.py main`'s summary, with ratios paired by repetition number."""
    base = configs[0]["name"]
    reference = next((r["tokens"] for r in runs[base] if r.get("tokens")), None)
    stock = {r["rep"]: r for r in runs[base] if r.get("median_ms")}
    summary = {}
    for config in configs:
        name = config["name"]
        pairs = [r["median_ms"] / stock[r["rep"]]["median_ms"] for r in runs[name]
                 if r.get("median_ms") and r["rep"] in stock]
        tokens = next((r["tokens"] for r in runs[name] if r.get("tokens")), None)
        summary[name] = {
            "ratio_vs_reference_per_rep": pairs,
            "median_ratio": st.median(pairs) if pairs else None,
            "max_ratio": max(pairs) if pairs else None,
            "identical_requests": (sum(a == b for a, b in zip(tokens, reference))
                                   if tokens and reference else None),
            "deterministic_across_processes": len({r.get("outputs_sha256") for r in runs[name]}) == 1,
            "failed_processes": sum(r["exit"] != 0 for r in runs[name]),
        }
    return summary


for key, model_id, revision, reps, configs in MODELS:
    if DEADLINE - time.time() < 1.15 * FIRST_REP_ESTIMATE + 120:
        report["stages"][f"download_{key}"] = {"exit": "skipped_deadline"}
        break
    code, _ = sh(f"download_{key}", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                                    f"print(s('{model_id}', revision='{revision}'))\"", timeout=1200)
    if code != 0:
        continue
    runs = {c["name"]: [] for c in configs}
    result = {"schema": "ironmule.port1-cross.v1", "model_id": model_id, "revision": revision,
              "configs": configs, "reps": 0, "measure": 1, "runs": runs, "performance_claim": False}
    started, last = time.time(), None
    for rep in range(reps):
        # A repetition that would not finish is not started, the first one included.
        if DEADLINE - time.time() < 1.15 * (last if last is not None else FIRST_REP_ESTIMATE):
            report["stages"][f"cross_{key}_rep{rep}"] = {"exit": "skipped_deadline"}
            break
        rep_started = time.time()
        for config in configs[rep % len(configs):] + configs[:rep % len(configs)]:
            child_env = {k: v for k, v in env.items() if not k.startswith("MLX_MAX_")}
            child_env.update(config.get("env", {}))
            try:
                proc = subprocess.run([PY, CROSS, "child", model_id, revision, json.dumps(config), "1"],
                                      env=child_env, capture_output=True, text=True,
                                      timeout=max(60, min(1800, DEADLINE - time.time())))
                row = {"rep": rep, "exit": proc.returncode, "stderr_tail": proc.stderr[-1500:]}
                if proc.returncode == 0:
                    row.update(json.loads(proc.stdout.strip().splitlines()[-1]))
            except subprocess.TimeoutExpired:
                row = {"rep": rep, "exit": "timeout"}
            runs[config["name"]].append(row)
            print(key, config["name"], rep, row.get("median_ms"), row["exit"], flush=True)
            result.update(reps=rep + 1, summary=summarise(configs, runs), seconds=time.time() - started)
            with open(f"{WORK}/cross-{key}.json", "w") as stream:
                json.dump(result, stream, indent=1)
        last = time.time() - rep_started
        report["stages"][f"cross_{key}_rep{rep}"] = {"exit": 0, "seconds": round(last, 1)}
        save()
    report[f"summary_{key}"] = result.get("summary")
    save()
report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
