# PERF1-V: 8-bit router matvecs through the row kernel. Private notebook, internet on. Quota: the
# user delegated all decisions on 2026-09-26 and asked that tests run on Kaggle (30 h week).
# Rules, before the run (docs/PROJECT_FRIDAY_BACKLOG.md PERF1-V):
#   * perf1.py at the commit below; every arm with PERF1_ARITH=pinned (the product's rounding).
#   * Probe first: `perf1.py r8probe`, the 8-bit kernel against a float32 reference on router
#     shapes, M = 1, 2, 8; every relative error <= 1e-2 (the install probe's tolerance), else no
#     model stage runs.
#   * Gemma 4 26B-A4B (one card, 256 prompt tokens, 128 new tokens, 1 warm + 3 measured
#     generations per process), fresh process per arm, order control, r8, control, r8:
#     control `kernel+gather` (run 14's decode arm), candidate `kernel+gather+r8`.
#   * Kill (the entry's): median decode tok/s of the candidate's six measured generations under
#     1.05x the control's six -> rejected. Tokens are compared with the control and recorded,
#     not gated: `r8` changes the router's arithmetic, so expert choices may change (PERF1-U).
#   * Qwen3.6 35B-A3B over two cards, the same pair on `kernel+p16+gather`, only if time is left.
import json
import os
import signal
import statistics as st
import subprocess
import sys
import time

COMMIT = "3888dce7d7fd4ed91761293e8de66693eb0c2091"
MODELS = {"gemma4-26b-a4b": ("mlx-community/gemma-4-26b-a4b-it-4bit", "0d77464eeb233a2da68ebf9d7dc4edaac7db956d"),
          "qwen36-35b-a3b": ("mlx-community/Qwen3.6-35B-A3B-4bit", "38740b847e4cb78f352aba30aa41c76e08e6eb46")}
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable
PERF1 = f"{REPO}/experiments/kaggle_compat/perf1.py"
PIPE = f"{VENV}/bin/mlx.launch --hosts 127.0.0.1 -n 2 --backend ring -- {PY} {PERF1}"
DEADLINE = time.time() + 105 * 60
os.makedirs(f"{WORK}/logs", exist_ok=True)
report = {"schema": "ironmule.perf1v-kaggle.v1", "commit": COMMIT, "stages": {}, "verdicts": {},
          "performance_claim": False}
env = dict(os.environ, PATH=f"{VENV}/bin:" + os.environ["PATH"], PYTHONPATH="", PYTHONNOUSERSITE="1",
           HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONUNBUFFERED="1", IRONMULE_HOME="/tmp/ironmule-home",
           PERF1_ARITH="pinned", PERF1_PROMPT_TOKENS="256")


def save():
    with open(f"{WORK}/perf1v-result.json", "w") as stream:
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


def load(path):
    try:
        with open(path) as stream:
            return json.load(stream)
    except (OSError, ValueError):
        return None


def judge(key, control, candidate, runs):
    rows = {arm: [load(f"{WORK}/e2e-{key}-{arm}-{i}.json") for i in (1, 2)] for arm in (control, candidate)}
    if not all(rows[control]) or not all(rows[candidate]):
        return {"missing": {arm: [r is None for r in rs] for arm, rs in rows.items()}}
    tps = {arm: [v for r in rs for v in r["decode_tps"]] for arm, rs in rows.items()}
    ratio = st.median(tps[candidate]) / st.median(tps[control])
    return {"control": control, "candidate": candidate, "decode_tps": tps, "ratio": ratio,
            "verdict": "passes" if ratio >= 1.05 else "rejected",
            "tokens_equal_to_control": [r["tokens"] == rows[control][0]["tokens"] for r in rows[candidate]],
            "routed": {arm: rs[0]["routed"] for arm, rs in rows.items()},
            "ttft_ms_median": {arm: [r["ttft_ms_median"] for r in rs] for arm, rs in rows.items()}}


sh("env", "nvidia-smi --query-gpu=index,name,driver_version,memory.total,clocks.max.sm --format=csv; nproc; free -g")
sh("clone", f"git clone -q https://github.com/Tobayko/IronMule {REPO} && git -C {REPO} checkout -q {COMMIT} "
            f"&& git -C {REPO} rev-parse HEAD && git -C {REPO} status --short")
sh("venv", f"{SYS} -m pip install -q uv && {SYS} -m uv python install 3.12 "
           f"&& {SYS} -m uv venv --python-preference only-managed --python 3.12 {VENV}")
sh("install", f"{SYS} -m uv pip install --python {PY} -e '{REPO}[cuda]' 'mlx-lm==0.31.3' psutil", timeout=1200)
sh("freeze", f"{SYS} -m uv pip freeze --python {PY}")

sh("r8probe", f"{PY} {PERF1} r8probe {WORK}/r8-probe.json", timeout=600)
probe = load(f"{WORK}/r8-probe.json")
report["probe_ok"] = bool(probe and probe.get("ok"))
save()

if report["probe_ok"]:
    for key, launcher, control in (("gemma4-26b-a4b", f"{PY} {PERF1}", "kernel+gather"),
                                   ("qwen36-35b-a3b", PIPE, "kernel+p16+gather")):
        model_id, revision = MODELS[key]
        code, out = sh(f"download_{key}", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                                          f"print(s('{model_id}', revision='{revision}'))\"", timeout=1800)
        if code != 0:
            continue
        path = out.strip().splitlines()[-1]
        candidate = control + "+r8"
        for i, arm in ((1, control), (1, candidate), (2, control), (2, candidate)):
            out_file = f"{WORK}/e2e-{key}-{arm}-{i}.json"
            sh(f"e2e_{key}_{arm}_{i}", f"{launcher} e2e {path} {arm} {out_file}", timeout=1500)
        report["verdicts"][key] = judge(key, control, candidate, 2)
        save()
        print(json.dumps({key: {k: v for k, v in report["verdicts"][key].items() if k != "decode_tps"}}, indent=1))

report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
