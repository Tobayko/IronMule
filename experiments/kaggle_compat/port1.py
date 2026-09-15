# PORT1: IronMule on NVIDIA CUDA (Kaggle T4) with the portability changes applied.
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
# Screening winner of `ironmule tune` on the T4 in PORT1 attempt 1 (1B): 0.7387 of baseline.
TUNED = ('{"compiled_fixed_cache":true,"head_skip_prefill":true,"prefill_into_fixed":true,'
         '"readback_every":8,"capacity_slack":128}')
MODELS = {
    "1b": ("mlx-community/gemma-3-1b-it-4bit", "2d44e83dc9e80843d22fb941d3d699a0b1351aa6"),
    "4b": ("mlx-community/gemma-3-4b-it-4bit", "93724907d4ed1745d2fe50baadf3b0b01a65abf2"),
}
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable  # Kaggle's interpreter carries pip; the venv does not
DEADLINE = time.time() + 57 * 60
os.makedirs(f"{WORK}/logs", exist_ok=True)
with open("/tmp/ironmule.patch", "w") as stream:
    stream.write(PATCH_SOURCE)
with open("/tmp/abcd.py", "w") as stream:
    stream.write(ABCD_SOURCE)
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
# A uv-managed CPython: Kaggle's own interpreter lacks ensurepip and loads a sitecustomize
# that imports a missing `wrapt`, which broke two stderr-exact tests in DATA3.
sh("venv", f"{SYS} -m pip install -q uv && {SYS} -m uv venv --managed-python --python 3.12 {VENV}")
sh("install", f"{SYS} -m uv pip install --python {PY} -e '{REPO}[cuda]' 'mlx-lm==0.31.3' "
              "'pytest>=8' 'pytest-xdist>=3' psutil scipy", timeout=1200)
sh("freeze", f"{SYS} -m uv pip freeze --python {PY}")
sh("cli_doctor", "ironmule doctor --json")
for key, (model_id, revision) in MODELS.items():
    sh(f"download_{key}", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                          f"print(s('{model_id}', revision='{revision}'))\"")
one, four = MODELS["1b"][0], MODELS["4b"][0]
sh("cli_models", "ironmule models")
sh("cli_status", f"ironmule status --model {one}")
# Full suite in parallel, then the failures alone and serially: attempt 1's four failures were
# two stderr checks hit by Kaggle's sitecustomize and two timing tests under -n 4 load.
sh("pytest", f"{PY} -m pytest -q -n 4 -o addopts='' tests", timeout=1200, cwd=REPO)
sh("pytest_lastfailed", f"{PY} -m pytest -q -n0 -o addopts='' --lf tests", timeout=600, cwd=REPO)
sh("cli_tune_1b", f"ironmule tune --model {one}", timeout=900)
sh("cli_tune_show_1b", f"ironmule tune --show --model {one}")
sh("cli_revalidate_1b", f"ironmule revalidate --model {one}", timeout=600)
sh("abcd_1b", f"{PY} /tmp/abcd.py {one} {MODELS['1b'][1]} '{TUNED}' {WORK}/abcd-1b.json 2 6", timeout=900)

# Product path: settings, registration, readiness, calibration, HTTP (JSON and SSE).
state = "--state-dir /tmp/ironmule-product"
sh("serve_setup", f"ironmule setup {state} --mode desktop && ironmule models add {state} {one}")
sh("optimize_readiness", f"ironmule optimize run {state} --model {one} --readiness-only --wait-ready 120 --json",
   timeout=300)
server = subprocess.Popen(f"exec ironmule serve {state} --model {one} --port 8080", shell=True, env=env,
                          cwd="/tmp", stdout=open(f"{WORK}/logs/serve.log", "w"), stderr=subprocess.STDOUT)
auth = {"Authorization": "Bearer port1-local-only"}
http = {"health": None}
for _ in range(150):
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
    body = {"model": one, "max_tokens": 16, "messages": [{"role": "user", "content": "Say hello in five words."}]}
    try:
        for kind, payload in (("completion", body), ("stream", {**body, "stream": True})):
            request = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",
                                             data=json.dumps(payload).encode(),
                                             headers={"Content-Type": "application/json", **auth})
            with urllib.request.urlopen(request, timeout=300) as response:
                raw = response.read().decode()
            http[kind] = json.loads(raw) if kind == "completion" else raw[-1500:]
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
sh("abcd_4b", f"{PY} /tmp/abcd.py {four} {MODELS['4b'][1]} '{TUNED}' {WORK}/abcd-4b.json 2 6", timeout=1500)
# Last: the calibration paces itself (25 % duty cycle, required breaks) and uses what time is left.
sh("optimize_run", f"ironmule optimize run {state} --model {one} --wait-ready 120 --json", timeout=3600)
report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
