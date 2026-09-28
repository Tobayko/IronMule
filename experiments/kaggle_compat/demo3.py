# DEMO3: DEMO2 with a larger model and a second scene. Qwen 3 14B 4-bit behind `ironmule serve`,
# first stock MLX (IronMule off), then `--compute-dtype native` (IronMule on; the plan is
# qualified for this checkpoint on pre-Ampere CUDA, `ironmule/numeric_plans.py`), one after the
# other on GPU 0 of a Kaggle T4 cell. Private notebook, internet on. Quota: the user asked for
# short runs on 2026-09-28; capped at 35 min.
#   * IronMule main 58cbb1d, no patch; request timeout 900 s for the demo state (DEMO2 run 1).
#   * Qwen 3's direct mode through its `/no_think` switch, since the HTTP API passes no template
#     options: the answer starts with an empty think block.
#   * Scene 1: one question, at most 64 tokens, three streamed answers per arm.
#   * Scene 2: four different questions sent at the same moment, at most 48 tokens each. On this
#     plan `serve` answers one request at a time (batch capacity 1), on both arms, so the later
#     ones wait in its queue; every chunk keeps its time from the moment the four were sent.
#   * Every question is asked once before anything is recorded: stock compiles kernels per new
#     prompt length (40-80 s on DEMO1/DEMO2), which the video leaves out.
#   * A demo, not a measurement: the ledger's paired runs are the evidence.
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

COMMIT = "58cbb1dadac725c63632399379e9ef15222d16b8"
MODEL = ("mlx-community/Qwen3-14B-4bit", "a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4")
CONFIG = {"suffix": " /no_think", "reps": 3,
          "single": {"prompt": "Explain in two sentences why the sky is blue.", "max_tokens": 64},
          "multi": {"max_tokens": 48, "prompts": ["Name three rivers in Europe.",
                                                  "Write a haiku about morning coffee.",
                                                  "What is the boiling point of water at sea level?",
                                                  "Give two tips for a good night's sleep."]}}
RECORDER = '''
import json, sys, threading, time, urllib.request
port, model, config, out = sys.argv[1], sys.argv[2], json.load(open(sys.argv[3])), sys.argv[4]


def ask(prompt, max_tokens, t0=None):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt + config["suffix"]}],
                       "max_tokens": max_tokens, "stream": True}).encode()
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                                     {"Content-Type": "application/json"})
    t0 = time.perf_counter() if t0 is None else t0
    sent, chunks, usage = round((time.perf_counter() - t0) * 1000, 1), [], None
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
    row = {"prompt": prompt, "sent_ms": sent, "total_ms": round(total, 1), "first_ms": chunks[0][0] if chunks else None,
           "usage": usage, "text": "".join(d for _, d in chunks), "chunks": chunks}
    print({k: v for k, v in row.items() if k != "chunks"}, flush=True)
    return row


single, multi = config["single"], config["multi"]
record = {"warmup": [ask(single["prompt"], single["max_tokens"])] + [ask(p, multi["max_tokens"]) for p in multi["prompts"]]}
record["reps"] = [ask(single["prompt"], single["max_tokens"]) for _ in range(config["reps"])]
rows, t0 = [None] * len(multi["prompts"]), time.perf_counter()


def one(i):
    rows[i] = ask(multi["prompts"][i], multi["max_tokens"], t0)


threads = [threading.Thread(target=one, args=(i,)) for i in range(len(rows))]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
record["multi"] = rows
json.dump(record, open(out, "w"), indent=1)
'''
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable
DEADLINE = time.time() + 35 * 60
os.makedirs(f"{WORK}/logs", exist_ok=True)
with open("/tmp/record.py", "w") as stream:
    stream.write(RECORDER)
with open("/tmp/demo3.json", "w") as stream:
    json.dump(CONFIG, stream)
report = {"schema": "ironmule.demo3-kaggle.v1", "commit": COMMIT, "model": MODEL, "config": CONFIG,
          "stages": {}, "health": {}, "performance_claim": False}
env = dict(os.environ, PATH=f"{VENV}/bin:" + os.environ["PATH"], PYTHONPATH="", PYTHONNOUSERSITE="1",
           HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONUNBUFFERED="1", IRONMULE_HOME="/tmp/ironmule-home",
           CUDA_VISIBLE_DEVICES="0")
env.pop("IRONMULE_API_KEY", None)


def save():
    with open(f"{WORK}/demo3-result.json", "w") as stream:
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
        sh(f"race_{label}", f"{PY} /tmp/record.py {port} {MODEL[0]} /tmp/demo3.json {WORK}/race-{label}.json",
           timeout=1500)
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
