# SHOT1: screenshots of IronMule's chat page, served by `ironmule serve` from `main` with a real
# model on a Kaggle T4. The user asked on 2026-09-25 to see the application. Private notebook,
# internet on; a demonstration, no measurement and no performance claim. Rules:
#   * `ironmule setup`, `models add` at the pinned Gemma 3 1B revision, `serve` on 127.0.0.1:8080;
#     wait for /ready.
#   * Headless Chromium (Playwright) opens `/`, sends one fixed question through the page's own
#     form and waits until the answer has finished streaming; one screenshot in the light and
#     one in the dark colour scheme, each with its own fresh question, and the answer text.
import json
import os
import signal
import subprocess
import sys
import time

COMMIT = "70a83db3061431494b9b92e61a58d2dfd6fa84af"
GEMMA1B = ("mlx-community/gemma-3-1b-it-4bit", "2d44e83dc9e80843d22fb941d3d699a0b1351aa6")
WORK = "/kaggle/working"
REPO = "/tmp/IronMule"
VENV = "/tmp/im"
PY = f"{VENV}/bin/python"
SYS = sys.executable
DEADLINE = time.time() + 40 * 60
os.makedirs(f"{WORK}/logs", exist_ok=True)
report = {"schema": "ironmule.shot1-kaggle.v1", "commit": COMMIT, "stages": {}, "performance_claim": False}
env = dict(os.environ, PATH=f"{VENV}/bin:" + os.environ["PATH"], PYTHONPATH="", PYTHONNOUSERSITE="1",
           HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONUNBUFFERED="1", IRONMULE_HOME="/tmp/ironmule-home")


def save():
    with open(f"{WORK}/shot1-result.json", "w") as stream:
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
sh("install", f"{SYS} -m uv pip install --python {PY} -e '{REPO}[cuda]' 'mlx-lm==0.31.3' 'playwright==1.55.0'", timeout=1200)
sh("chromium", f"{PY} -m playwright install --with-deps chromium", timeout=900)
sh("freeze", f"{SYS} -m uv pip freeze --python {PY}")


def download(key, model):
    code, out = sh(f"download_{key}", f"{PY} -c \"from huggingface_hub import snapshot_download as s; "
                                      f"print(s('{model[0]}', revision='{model[1]}'))\"", timeout=1200)
    return out.strip().splitlines()[-1] if code == 0 else None


SHOT = r"""
import json, sys, time
from playwright.sync_api import sync_playwright
QUESTIONS = {"light": "Explain in two short sentences why a local language model keeps your data private.",
             "dark": "Write a four-line poem about a GPU that works at night."}
result = {}
with sync_playwright() as p:
    browser = p.chromium.launch()
    for scheme, question in QUESTIONS.items():
        page = browser.new_page(viewport={"width": 1100, "height": 720}, color_scheme=scheme,
                                device_scale_factor=2)
        page.goto("http://127.0.0.1:8080/")
        page.wait_for_function("document.getElementById('model').textContent !== 'connecting'", timeout=60000)
        started = time.time()
        page.fill("#input", question)
        page.press("#input", "Enter")
        page.wait_for_selector(".msg.assistant", timeout=60000)
        page.wait_for_function("!document.getElementById('send').disabled && "
                               "document.querySelector('.msg.assistant').textContent.length > 0", timeout=300000)
        page.screenshot(path=f"/kaggle/working/ironmule-chat-{scheme}.png")
        result[scheme] = {"question": question, "model_label": page.inner_text("#model"),
                          "answer": page.inner_text(".msg.assistant"), "seconds": round(time.time() - started, 1)}
        page.close()
    browser.close()
json.dump(result, open("/kaggle/working/shot1-answers.json", "w"), indent=1)
print(json.dumps(result, indent=1))
"""
with open("/tmp/shot.py", "w") as stream:
    stream.write(SHOT)

if download("gemma3-1b", GEMMA1B):
    sh("setup", "ironmule setup --mode desktop")
    sh("models_add", f"ironmule models add {GEMMA1B[0]} --revision {GEMMA1B[1]}")
    server = subprocess.Popen(f"ironmule serve --model {GEMMA1B[0]}", shell=True, env=env, cwd="/tmp",
                              stdout=open(f"{WORK}/logs/serve.log", "w"), stderr=subprocess.STDOUT,
                              start_new_session=True)
    sh("wait_ready", "for i in $(seq 1 120); do curl -sf http://127.0.0.1:8080/ready && exit 0; sleep 5; done; exit 1",
       timeout=660)
    sh("doctor", "ironmule doctor")
    sh("screenshot", f"{PY} /tmp/shot.py", timeout=900)
    os.killpg(server.pid, signal.SIGTERM)
    try:
        server.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)
    report["stages"]["serve"] = {"exit": server.wait()}
report["finished"] = True
save()
print(json.dumps({k: v.get("exit") for k, v in report["stages"].items()}, indent=1))
