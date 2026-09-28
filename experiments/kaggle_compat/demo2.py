# DEMO2 run 2 (run 1: stock's warm-up hit the product's default 120 s request timeout and got
# HTTP 503; this run sets the demo state's timeout to 900 s, nothing else changed): a short race on a Kaggle Tesla T4 for a show video. Gemma 3 12B 4-bit behind
# `ironmule serve`, first stock MLX (IronMule off), then `--compute-dtype native` (IronMule on,
# opt-in, quality-gated for this model; changes output), one after the other on GPU 0.
# Private notebook, internet on. Quota: the user asked for short runs on 2026-09-28; capped at 25 min.
#   * IronMule main 58cbb1d (DEMO1's 306b469 is gone since the history rewrite), no patch.
#   * One short question, at most 64 tokens, so the stock answer fits a video (DEMO1's
#     143 tokens took stock 87 s). Each server is warmed up with the question itself, so the
#     recording shows the steady state: stock compiles kernels once per new prompt length,
#     40-60 s in DEMO1's TTFT check, which the video leaves out.
#   * Three streamed answers per arm; every chunk keeps its arrival time from the moment the
#     request was sent. The video replays one of them in that timing.
#   * A demo, not a measurement: the ledger's paired runs are the evidence.
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

COMMIT = "58cbb1dadac725c63632399379e9ef15222d16b8"
MODEL = ("mlx-community/gemma-3-12b-it-4bit", "86cc6a8dedbc456dd0e4af01a9d09f396f77e558")
PROMPT = "Explain in two sentences why the sky is blue."
MAX_TOKENS, REPS = 64, 3
RECORDER = '''
import json, sys, time, urllib.request
port, model, prompt, max_tokens, reps, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]


def ask():
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "stream": True}).encode()
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                                     {"Content-Type": "application/json"})
    t0, chunks, usage = time.perf_counter(), [], None
    with urllib.request.urlopen(request, timeout=900) as response:
        for line in response:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            usage = chunk.get("usage") or usage
            delta = (chunk.get("choices") or [{}])[0].get("delta", {}).get("content")
            if delta:
                chunks.append([round((time.perf_counter() - t0) * 1000, 1), delta])
    total = (time.perf_counter() - t0) * 1000
    row = {"total_ms": round(total, 1), "first_ms": chunks[0][0] if chunks else None, "usage": usage,
           "text": "".join(d for _, d in chunks), "chunks": chunks}
    print({k: v for k, v in row.items() if k != "chunks"}, flush=True)
    return row


warmup = ask()
json.dump({"warmup": warmup, "reps": [ask() for _ in range(reps)]}, open(out, "w"), indent=1)
'''
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable
DEADLINE = time.time() + 25 * 60
os.makedirs(f"{WORK}/logs", exist_ok=True)
with open("/tmp/record.py", "w") as stream:
    stream.write(RECORDER)
report = {"schema": "ironmule.demo2-kaggle.v1", "commit": COMMIT, "model": MODEL, "prompt": PROMPT,
          "max_tokens": MAX_TOKENS, "stages": {}, "health": {}, "performance_claim": False}
env = dict(os.environ, PATH=f"{VENV}/bin:" + os.environ["PATH"], PYTHONPATH="", PYTHONNOUSERSITE="1",
           HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONUNBUFFERED="1", IRONMULE_HOME="/tmp/ironmule-home",
           CUDA_VISIBLE_DEVICES="0")
env.pop("IRONMULE_API_KEY", None)


def save():
    with open(f"{WORK}/demo2-result.json", "w") as stream:
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


def get(path, port):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as response:
        return json.loads(response.read())


sh("env", "nvidia-smi --query-gpu=index,name,driver_version,memory.total,clocks.max.sm --format=csv; nproc; free -g")
sh("clone", f"git clone -q https://github.com/Tobayko/IronMule {REPO} && git -C {REPO} checkout -q {COMMIT} "
            f"&& git -C {REPO} rev-parse HEAD && git -C {REPO} status --short")
sh("venv", f"{SYS} -m pip install -q uv && {SYS} -m uv python install 3.12 "
           f"&& {SYS} -m uv venv --python-preference only-managed --python 3.12 {VENV}")
sh("install", f"{SYS} -m uv pip install --python {PY} -e '{REPO}[cuda]' 'mlx-lm==0.31.3'", timeout=1200)
sh("freeze", f"{SYS} -m uv pip freeze --python {PY}")
code, _ = sh("download", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                         f"print(s('{MODEL[0]}', revision='{MODEL[1]}'))\"", timeout=1200)
state = "--state-dir /tmp/ironmule-product"
sh("serve_setup", f"ironmule setup {state} --mode desktop && ironmule models add {state} {MODEL[0]} --revision {MODEL[1]} "
                  f"&& {PY} -c \"from ironmule_product.state import ProductStore as S; "
                  f"print(S('/tmp/ironmule-product').set_request_timeout(900))\"")
for label, port, extra in (("stock", 8080, ""), ("ironmule-native", 8081, "--compute-dtype native")):
    if code != 0 or DEADLINE - time.time() < 240:
        break
    server = subprocess.Popen(f"exec ironmule serve {state} --model {MODEL[0]} --port {port} {extra}", shell=True,
                              env=env, cwd="/tmp", stdout=open(f"{WORK}/logs/serve-{label}.log", "w"),
                              stderr=subprocess.STDOUT)
    try:
        for _ in range(300):
            try:
                if get("/ready", port).get("ready"):
                    break
            except Exception:  # noqa: BLE001 - not up yet
                if server.poll() is not None:
                    break
            time.sleep(2)
        report["health"][label] = get("/health", port)
        save()
        sh(f"race_{label}", f"{PY} /tmp/record.py {port} {MODEL[0]} '{PROMPT}' {MAX_TOKENS} {REPS} "
                            f"{WORK}/race-{label}.json", timeout=600)
    except Exception as exc:  # noqa: BLE001 - recorded, not raised
        report["health"][label] = {"error": repr(exc)}
        save()
    finally:
        server.send_signal(signal.SIGTERM)
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()
report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
