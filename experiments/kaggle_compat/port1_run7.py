# PORT1 run 7: the opt-in float32 plan through the product (tune, benchmark, serve) on 4B; 12B stock vs IronMule.
# Private notebook, internet on (pip, git, Hugging Face). Free quota: one run, <= 60 min.
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

COMMIT = "5fde53f940852c92f30d5f30fe6377d88e17c9a4"
PATCH_SOURCE = __PATCH_SOURCE__  # uncommitted IronMule changes under test, embedded at submit time
ABCD_SOURCE = __ABCD_SOURCE__  # experiments/kaggle_compat/abcd.py
GRAPHS_SOURCE = __GRAPHS_SOURCE__  # experiments/kaggle_compat/graphs.py
PRECISION_SOURCE = __PRECISION_SOURCE__  # experiments/kaggle_compat/precision.py
CROSS_SOURCE = __CROSS_SOURCE__  # experiments/kaggle_compat/cross.py
QUALITY_SOURCE = __QUALITY_SOURCE__  # experiments/kaggle_compat/quality.py
PROBE_SOURCE = __PROBE_SOURCE__  # experiments/kaggle_compat/probe_ab.py
# The profile `ironmule tune` stored on the T4 in PORT1 attempt 2 (1B): confirmed 0.7028
# [0.6614; 0.7394], tokens identical; `revalidate` still_valid.
TUNED = ('{"compiled_fixed_cache":true,"fused_argmax":true,"head_skip_prefill":true,'
         '"readback_every":2,"capacity_slack":128,"fuse_projections":true}')
MODELS = {
    "1b": ("mlx-community/gemma-3-1b-it-4bit", "2d44e83dc9e80843d22fb941d3d699a0b1351aa6"),
    "4b": ("mlx-community/gemma-3-4b-it-4bit", "93724907d4ed1745d2fe50baadf3b0b01a65abf2"),
    "12b": ("mlx-community/gemma-3-12b-it-4bit", "86cc6a8dedbc456dd0e4af01a9d09f396f77e558"),
}
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable  # Kaggle's interpreter carries pip; the venv does not
DEADLINE = time.time() + 58 * 60
os.makedirs(f"{WORK}/logs", exist_ok=True)
with open("/tmp/ironmule.patch", "w") as stream:
    stream.write(PATCH_SOURCE)
for name, source in (("abcd", ABCD_SOURCE), ("graphs", GRAPHS_SOURCE), ("precision", PRECISION_SOURCE), ("cross", CROSS_SOURCE), ("quality", QUALITY_SOURCE), ("probe_ab", PROBE_SOURCE)):
    with open(f"/tmp/{name}.py", "w") as stream:
        stream.write(source)
report = {"schema": "ironmule.port1-kaggle.v1", "commit": COMMIT, "stages": {}}
# PYTHONNOUSERSITE: Kaggle's root user site holds the sitecustomize importing `wrapt` (attempt 1).
env = dict(os.environ, PATH=f"{VENV}/bin:" + os.environ["PATH"], PYTHONPATH="", PYTHONNOUSERSITE="1",
           IRONMULE_HOME="/tmp/ironmule-home", IRONMULE_API_KEY="port1-local-only",
           HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONUNBUFFERED="1")


def save():
    with open(f"{WORK}/port1-result.json", "w") as stream:
        json.dump(report, stream, indent=1, default=str)


def sh(name, cmd, timeout=900, cwd="/tmp"):
    left = DEADLINE - time.time()
    if left < 30:
        report["stages"][name] = {"exit": "skipped_deadline"}
        save()
        return None, ""
    started = time.time()
    # Own process group: a timeout must also stop the workers a command started, or they
    # keep the model lease and the next stage fails (attempt 1: optimize, then serve).
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
# only-managed: attempt 2's venv still resolved to /usr/lib/python3.12, whose Debian
# sitecustomize imports a missing `wrapt` and slows every child interpreter's start.
sh("venv", f"{SYS} -m pip install -q uv && {SYS} -m uv python install 3.12 "
           f"&& {SYS} -m uv venv --python-preference only-managed --python 3.12 {VENV} && {PY} -c 'import sys; print(sys.prefix, sys.base_prefix)'")
sh("install", f"{SYS} -m uv pip install --python {PY} -e '{REPO}[cuda]' 'mlx-lm==0.31.3' "
              "'pytest>=8' 'pytest-xdist>=3' psutil scipy pyarrow", timeout=1200)
sh("freeze", f"{SYS} -m uv pip freeze --python {PY}")
for key in ("4b", "12b"):
    model_id, revision = MODELS[key]
    sh(f"download_{key}", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                          f"print(s('{model_id}', revision='{revision}'))\"")
sh("download_wikitext", f"{PY} -c \"from huggingface_hub import hf_hub_download as d; import pyarrow.parquet as pq; "
                        "p = d('Salesforce/wikitext', 'wikitext-2-raw-v1/test-00000-of-00001.parquet', repo_type='dataset'); "
                        "open('/tmp/wikitext.txt', 'w').write(''.join(pq.read_table(p).column('text').to_pylist())); print(p)\"")
(four, rev4), (twelve, rev12) = MODELS["4b"], MODELS["12b"]
F32 = "--compute-dtype float32"
sh("cli_doctor", "ironmule doctor")
sh("cli_tune_4b_float32", f"ironmule tune --model {four} {F32}", timeout=1800)
sh("cli_tune_show_4b_float32", f"ironmule tune --show --model {four} {F32}")
sh("benchmark_4b_float32", f"ironmule benchmark --model {four} {F32} --json {WORK}/benchmark-4b-float32.json",
   timeout=1200)
state = "--state-dir /tmp/ironmule-product"
sh("serve_setup", f"ironmule setup {state} --mode desktop && ironmule models add {state} {four}")
server = subprocess.Popen(f"exec ironmule serve {state} --model {four} --port 8080 {F32}", shell=True, env=env,
                          cwd="/tmp", stdout=open(f"{WORK}/logs/serve.log", "w"), stderr=subprocess.STDOUT)
auth = {"Authorization": "Bearer port1-local-only"}
http = {"health": None}
for _ in range(200):
    try:
        with urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/health", headers=auth),
                                    timeout=2) as response:
            http["health"] = json.loads(response.read())
            break
    except Exception as exc:  # noqa: BLE001
        http["health_error"] = repr(exc)
        if server.poll() is not None:
            break
        time.sleep(2)
if http["health"] is not None:
    body = {"model": four, "max_tokens": 16, "messages": [{"role": "user", "content": "Say hello in five words."}]}
    try:
        for kind, payload in (("completion", body), ("stream", {**body, "stream": True})):
            request = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",
                                             data=json.dumps(payload).encode(),
                                             headers={"Content-Type": "application/json", **auth})
            with urllib.request.urlopen(request, timeout=600) as response:
                raw = response.read().decode()
            http[kind] = json.loads(raw) if kind == "completion" else raw[-1200:]
    except Exception as exc:  # noqa: BLE001
        http["request_error"] = repr(exc)
server.terminate()
try:
    server.wait(timeout=20)
except subprocess.TimeoutExpired:
    server.kill()
http["server_exit"] = server.poll()
report["serve_http"] = http
save()
STOCK = {"name": "stock", "env": {"MLX_MAX_OPS_PER_BUFFER": "20", "MLX_MAX_MB_PER_BUFFER": "100"},
         "knobs": {}, "mode": "interactive"}
cross_12b = [STOCK,
             {"name": "ironmule_exact", "env": {}, "knobs": {"head_skip_prefill": True}, "mode": "throughput"},
             {"name": "ironmule_fp32", "env": {}, "knobs": {"head_skip_prefill": True}, "mode": "throughput",
              "dtype": "float32"}]
sh("cross_12b", f"{PY} /tmp/cross.py {twelve} {rev12} '{json.dumps(cross_12b)}' {WORK}/cross-12b.json 2 1",
   timeout=2700)
sh("quality_12b_float32", f"QUALITY_ONLY=float32 {PY} /tmp/quality.py {twelve} {rev12} /tmp/wikitext.txt "
                          f"{WORK}/quality-12b-float32.json 16 512", timeout=1500)
report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
