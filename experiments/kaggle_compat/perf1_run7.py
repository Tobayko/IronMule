# PERF1 run 7: the `native` plan through the product, not the harness. Private notebook,
# internet on. Free quota: one run, <= 50 min. Backlog: PERF1-H.
#
# Runs 1-6 measured the kernel through `perf1.py`'s global patch. This runs IronMule itself
# with the working tree's `ironmule/cuda_native.py` (`compute_dtype="native"`: installed per
# model after fusion, probe-gated, pinned arithmetic), carried as a patch against the pinned
# commit. Rules fixed here:
#   * `cross.py`, the harness every numeric-plan cell comes from, 2 interleaved repetitions:
#     stock against native with no knobs, with `compiled_fixed_cache`, and with PORT2's tuned
#     knobs in throughput mode. The knobbed arms must reproduce the knob-free native tokens
#     (6/6), else that combination is reported as not exact under the plan.
#   * product smoke: `ironmule doctor`, `ironmule benchmark --compute-dtype native`, and
#     `ironmule serve --compute-dtype native` answering JSON, SSE and four concurrent requests,
#     with `health` reporting the plan.
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request

COMMIT = "b339c817299de173f8a330089afb69973c519353"  # research/port2-model-families
CROSS_SOURCE = __CROSS_SOURCE__  # experiments/kaggle_compat/cross.py
PATCH_SOURCE = __PATCH_SOURCE__  # the working tree's IronMule changes (identity fix, native plan)
MODELS = [("qwen3-8b", "mlx-community/Qwen3-8B-4bit", "545dc4251c05440727734bcd94334791f6ab0192"),
          ("qwen3-14b", "mlx-community/Qwen3-14B-4bit", "a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4")]
TUNED = {"compiled_fixed_cache": True, "fused_argmax": True, "head_skip_prefill": True,
         "readback_every": 2, "capacity_slack": 128, "fuse_projections": True}
STOCK = {"name": "stock", "env": {"MLX_MAX_OPS_PER_BUFFER": "20", "MLX_MAX_MB_PER_BUFFER": "100"},
         "knobs": {}, "mode": "interactive"}
CONFIGS = [STOCK,
           {"name": "ironmule_native", "env": {}, "knobs": {}, "mode": "interactive", "dtype": "native"},
           {"name": "ironmule_native_compiled", "env": {}, "knobs": {"compiled_fixed_cache": True},
            "mode": "interactive", "dtype": "native"},
           {"name": "ironmule_native_tuned", "env": {}, "knobs": TUNED, "mode": "throughput", "dtype": "native"}]
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable
DEADLINE = time.time() + 50 * 60
os.makedirs(f"{WORK}/logs", exist_ok=True)
with open("/tmp/cross.py", "w") as stream:
    stream.write(CROSS_SOURCE)
with open("/tmp/ironmule.patch", "w") as stream:
    stream.write(PATCH_SOURCE)
report = {"schema": "ironmule.perf1-kaggle.v7", "commit": COMMIT, "stages": {}, "performance_claim": False}
env = dict(os.environ, PATH=f"{VENV}/bin:" + os.environ["PATH"], PYTHONPATH="", PYTHONNOUSERSITE="1",
           HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONUNBUFFERED="1", IRONMULE_HOME="/tmp/ironmule-home",
           IRONMULE_API_KEY="perf1-local-only")


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


def post(path, payload, auth, timeout=900):
    request = urllib.request.Request(f"http://127.0.0.1:8080{path}", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json", **auth})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode()


sh("env", "nvidia-smi --query-gpu=index,name,memory.total,clocks.max.sm --format=csv; nproc; free -g")
sh("clone", f"git clone -q https://github.com/Tobayko/IronMule {REPO} && git -C {REPO} checkout -q {COMMIT} "
            f"&& git -C {REPO} apply /tmp/ironmule.patch && git -C {REPO} status --short")
sh("venv", f"{SYS} -m pip install -q uv && {SYS} -m uv python install 3.12 "
           f"&& {SYS} -m uv venv --python-preference only-managed --python 3.12 {VENV}")
sh("install", f"{SYS} -m uv pip install --python {PY} -e '{REPO}[cuda]' 'mlx-lm==0.31.3'", timeout=1200)
sh("freeze", f"{SYS} -m uv pip freeze --python {PY}")
sh("cli_doctor", "ironmule doctor")

(eight, rev8), (fourteen, rev14) = [(m, r) for _, m, r in MODELS]
for key, model_id, revision in MODELS:
    sh(f"download_{key}", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                          f"print(s('{model_id}', revision='{revision}'))\"", timeout=1200)

sh("benchmark_8b_native", f"ironmule benchmark --model {eight} --compute-dtype native "
                          f"--json {WORK}/benchmark-8b-native.json", timeout=1200)

state = "--state-dir /tmp/ironmule-product"
sh("serve_setup", f"ironmule setup {state} --mode desktop && ironmule models add {state} {eight}")
server = subprocess.Popen(f"exec ironmule serve {state} --model {eight} --port 8080 --compute-dtype native",
                          shell=True, env=env, cwd="/tmp", stdout=open(f"{WORK}/logs/serve.log", "w"),
                          stderr=subprocess.STDOUT)
auth = {"Authorization": "Bearer perf1-local-only"}
http = {"health": None}
for _ in range(300):
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
    body = {"model": eight, "max_tokens": 32, "messages": [{"role": "user", "content": "Say hello in five words."}]}
    try:
        http["completion"] = json.loads(post("/v1/chat/completions", body, auth))
        http["stream"] = post("/v1/chat/completions", {**body, "stream": True}, auth)[-1200:]
        questions = ["Name three rivers in Europe.", "What is 17 times 23?", "Define entropy in one sentence.",
                     "Give a synonym for happy."]
        answers, began = {}, time.time()

        def ask(i, q):
            answers[i] = json.loads(post("/v1/chat/completions", {**body, "max_tokens": 64,
                                         "messages": [{"role": "user", "content": q}]}, auth))
        threads = [threading.Thread(target=ask, args=(i, q)) for i, q in enumerate(questions)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        http["concurrent"] = {"requests": len(questions), "wall_s": round(time.time() - began, 2),
                              "answers": [answers[i]["choices"][0]["message"]["content"] for i in sorted(answers)],
                              "usage": [answers[i].get("usage") for i in sorted(answers)]}
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

for key, model_id, revision in MODELS:
    sh(f"cross_{key}", f"{PY} /tmp/cross.py {model_id} {revision} '{json.dumps(CONFIGS)}' "
                       f"{WORK}/cross-native-{key}.json 2 1", timeout=2400)

report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
