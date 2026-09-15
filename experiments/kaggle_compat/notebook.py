# DATA3: does IronMule run on a Kaggle T4 through MLX's CUDA backend?
# Private notebook, internet on (pip, git, Hugging Face). No performance claim.
import json
import os
import subprocess
import sys
import time
import urllib.request

COMMIT = "5fde53f940852c92f30d5f30fe6377d88e17c9a4"
MODELS = [
    ("mlx-community/gemma-3-1b-it-4bit", "2d44e83dc9e80843d22fb941d3d699a0b1351aa6"),
    ("mlx-community/gemma-3-4b-it-4bit", "93724907d4ed1745d2fe50baadf3b0b01a65abf2"),
    ("mlx-community/gemma-3-12b-it-4bit", "86cc6a8dedbc456dd0e4af01a9d09f396f77e558"),
]
MODE = __MODE__  # "full", or "serve": 1B HTTP path only (free-plan quota, attempt 4)
if MODE == "serve":
    MODELS = MODELS[:1]
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable  # Kaggle's interpreter; the venv has no pip
DEADLINE = time.time() + 55 * 60
INFER = "/tmp/infer.py"
INFER_SOURCE = __INFER_SOURCE__  # experiments/kaggle_compat/infer.py, embedded at submit time
PATCH_SOURCE = __PATCH_SOURCE__  # uncommitted IronMule changes under test, embedded at submit time
os.makedirs(f"{WORK}/logs", exist_ok=True)
# Kaggle's stdlib sitecustomize imports a missing `wrapt` and writes to stderr in every
# child interpreter; an empty module earlier on sys.path shadows it (attempt 2 finding).
os.makedirs("/tmp/nosite", exist_ok=True)
open("/tmp/nosite/sitecustomize.py", "w").close()
with open("/tmp/ironmule.patch", "w") as stream:
    stream.write(PATCH_SOURCE)
with open(INFER, "w") as stream:
    stream.write(INFER_SOURCE)
report = {"schema": "ironmule.data3-kaggle.v1", "commit": COMMIT, "stages": {}}
env = dict(os.environ, PATH=f"{VENV}/bin:" + os.environ["PATH"], PYTHONPATH="/tmp/nosite", IRONMULE_API_KEY="data3-local-only", IRONMULE_HOME="/tmp/ironmule-home",
           HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONUNBUFFERED="1")


def save():
    with open(f"{WORK}/data3-result.json", "w") as stream:
        json.dump(report, stream, indent=1, default=str)


def sh(name, cmd, timeout=900, cwd=None):
    left = DEADLINE - time.time()
    if left < 30:
        report["stages"][name] = {"exit": "skipped_deadline"}
        save()
        return None, ""
    started = time.time()
    try:
        proc = subprocess.run(cmd, shell=True, cwd=cwd, env=env, capture_output=True, text=True,
                              timeout=min(timeout, left))
        code, out = proc.returncode, proc.stdout + proc.stderr
    except subprocess.TimeoutExpired as exc:
        raw = (exc.stdout or b"") + (exc.stderr or b"")
        code, out = "timeout", raw.decode(errors="replace") if isinstance(raw, bytes) else str(raw)
    with open(f"{WORK}/logs/{name}.log", "w") as stream:
        stream.write(out)
    report["stages"][name] = {"exit": code, "seconds": round(time.time() - started, 1),
                              "tail": out[-2500:]}
    save()
    print(f"== {name}: exit={code} {report['stages'][name]['seconds']}s", flush=True)
    return code, out


sh("env", "nvidia-smi; python3 -V; ldd --version | head -1; nproc; free -g; df -h /tmp | tail -1")
sh("clone", f"git clone -q https://github.com/Tobayko/IronMule {REPO} && git -C {REPO} checkout -q {COMMIT} "
            f"&& git -C {REPO} rev-parse HEAD && (test ! -s /tmp/ironmule.patch || git -C {REPO} apply /tmp/ironmule.patch) "
            f"&& git -C {REPO} diff --stat")
# Kaggle's Python lacks ensurepip, so `python -m venv` fails; uv needs no pip in the venv.
sh("venv", f"{SYS} -m pip install -q uv && {SYS} -m uv venv --python 3.12 {VENV}")
sh("install", f"{SYS} -m uv pip install --python {PY} 'mlx[cuda12]==0.32.0' 'mlx-lm==0.31.3' -e {REPO} "
              "'pytest>=8' 'pytest-xdist>=3' psutil scipy", timeout=1200)
sh("freeze", f"{SYS} -m uv pip freeze --python {PY}")
sh("mlx_gpu_smoke", f"""{PY} - <<'EOF'
import mlx.core as mx
print("mlx", mx.__version__, "device", mx.default_device(), "cuda", mx.cuda.is_available())
a = mx.random.normal((1024, 1024)); b = a @ a; mx.eval(b); print("matmul ok", b.shape)
w = mx.random.normal((4096, 1024)); q, s, z = mx.quantize(w, group_size=64, bits=4)
x = mx.random.normal((1, 1024)); y = mx.quantized_matmul(x, q, s, z, transpose=True, group_size=64, bits=4)
ref = x @ mx.dequantize(q, s, z, group_size=64, bits=4).T
mx.eval(y, ref); print("qmm ok max_abs_diff", mx.abs(y - ref).max().item())
EOF""")

for model_id, revision in MODELS:
    short = model_id.split("/")[1]
    code, _ = sh(f"download_{short}", f"{PY} -c \"from huggingface_hub import snapshot_download as s; print(s('{model_id}', revision='{revision}'))\"", timeout=900)
    if code != 0:
        continue
    if MODE == "full":
        sh(f"infer_{short}", f"{PY} {INFER} {model_id} {revision} "
                         f"{WORK}/infer-{short}.json", timeout=900, cwd="/tmp")
    try:
        with open(f"{WORK}/infer-{short}.json") as stream:
            report[f"infer_{short}"] = json.load(stream)
        save()
    except OSError:
        pass
    if short == "gemma-3-1b-it-4bit":
        # IRONMULE_HOME is also the runtime store (0755); the product root must be 0700 (attempt 2).
        state = "--state-dir /tmp/ironmule-product"
        sh("serve_setup", f"ironmule setup {state} --mode desktop && ironmule models add {state} {model_id}",
           timeout=300, cwd="/tmp")
        server = subprocess.Popen(f"exec ironmule serve {state} --model {model_id} --port 8080", shell=True,
                                  env=env, cwd="/tmp", stdout=open(f"{WORK}/logs/serve.log", "w"),
                                  stderr=subprocess.STDOUT)
        http = {"health": None, "completion": None}
        auth = {"Authorization": "Bearer data3-local-only"}
        for _ in range(120):
            try:
                probe = urllib.request.Request("http://127.0.0.1:8080/health", headers=auth)
                with urllib.request.urlopen(probe, timeout=2) as response:
                    http["health"] = json.loads(response.read())
                    break
            except Exception as exc:  # noqa: BLE001
                http["health_error"] = repr(exc)
                if server.poll() is not None:
                    break
                time.sleep(2)
        if http["health"] is not None:
            body = json.dumps({"model": model_id, "max_tokens": 16,
                               "messages": [{"role": "user", "content": "Say hello in five words."}]})
            try:
                request = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",
                                                 data=body.encode(),
                                                 headers={"Content-Type": "application/json",
                                                          "Authorization": "Bearer data3-local-only"})
                with urllib.request.urlopen(request, timeout=300) as response:
                    http["completion"] = json.loads(response.read())
                stream = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",
                    data=json.dumps({**json.loads(body), "stream": True}).encode(),
                    headers={"Content-Type": "application/json", **auth})
                with urllib.request.urlopen(stream, timeout=300) as response:
                    http["stream_tail"] = response.read().decode()[-1500:]
            except Exception as exc:  # noqa: BLE001
                http["completion_error"] = repr(exc)
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()
        http["server_exit"] = server.poll()
        with open(f"{WORK}/logs/serve.log") as stream:
            http["log_tail"] = stream.read()[-2500:]
        report["serve_http"] = http
        save()

if MODE == "full":
  sh("snapshot_files", "find /root/.cache/huggingface/hub -path '*snapshots*' \\( -type f -o -type l \\) | sort")
# Attempt 2 failures, serially and without the sitecustomize noise, plus the new device-gate check.
  sh("pytest_rerun", f"{PY} -m pytest -q -n0 -p no:cacheprovider -o addopts='' "
   "tests/engine/test_product_worker.py tests/engine/test_product_http.py::test_handler_saturation_returns_429_then_recovers "
   "tests/engine/test_q3d_stability_gate.py::test_real_macos_process_identity_and_cleanup_reap "
   "tests/engine/test_q3f_child_guard.py tests/engine/test_ironmule_runtime_integration.py", timeout=900, cwd=REPO)
report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
