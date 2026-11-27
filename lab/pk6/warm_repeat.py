"""Run one pre-registered K1/vLLM/K2 series on a prepared single-A100 host."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.request import urlopen

SOURCE = "8cf0a3106f09a3338eb53c28047af7e36fc7b340"
GGUF = "efd971561896866f0e910cce52761ca77b1b138090c7f15fe284676d57d1f689"


def save(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    temp.replace(path)


def output(*command):
    return subprocess.check_output(command, text=True).strip()


def board():
    raw = output("nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader")
    name, uid, driver, memory = raw.split(", ")
    if "\n" in raw or name != "NVIDIA A100-SXM4-80GB":
        raise ValueError("need one physical A100 SXM4 80 GB")
    return dict(name=name, device_sha256=hashlib.sha256(uid.encode()).hexdigest(), driver=driver, memory=memory)


def fetch(url):
    with urlopen(url, timeout=3) as response:
        return json.load(response)


def wait_ready(proc, url, model):
    deadline = time.monotonic() + 1500
    while proc.poll() is None and time.monotonic() < deadline:
        try:
            models = fetch(url + "/v1/models")["data"]
            if any(m["id"] == model for m in models):
                if model == "compare":
                    return {"model": model}
                health = fetch(url + "/health")
                if health["ready"] and health["ggml"]["gate"]["passes"]:
                    return {"model": model, "ggml": health["ggml"], "device": health.get("device")}
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(2)
    raise RuntimeError(f"{model} failed to become ready, process status {proc.poll()}")


def stop(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def arm(args, label, binding, *, source=SOURCE, seed=20261119):
    env = dict(os.environ)
    vllm = label == "vllm"
    model = "compare" if vllm else "gguf/Qwen3-32B-Q4_K_M"
    port = "8001" if vllm else "8002"
    url = "http://127.0.0.1:" + port
    target = args.output
    if board() != binding or (target / f"start-{label}.json").exists():
        raise ValueError("hardware changed or arm already started")
    if vllm:
        command = [str(args.vllm / "bin/vllm"), "serve", str(args.models / "qwen3-32b-awq"),
                   "--served-model-name", model, "--tokenizer", str(args.models / "tokenizers/qwen3-32b"),
                   "--dtype", "half", "--quantization", "awq_marlin", "--generation-config", "vllm",
                   "--max-model-len", "2048", "--max-num-seqs", "32", "--tensor-parallel-size", "1",
                   "--gpu-memory-utilization", "0.88", "--host", "127.0.0.1", "--port", port]
    else:
        env.pop("IRONMULE_COMPARE_LOCAL_KEY", None)
        env.update(IRONMULE_GGML_SEQ_CTX="2048", IRONMULE_ADMISSION_LEARN="0", IRONMULE_PS1="1",
                   IRONMULE_PK6="1", IRONMULE_SV5="0", IRONMULE_GGML_DH2="0",
                   IRONMULE_GGML_TRACE=str(target / f"trace-{label}.txt"))
        command = [sys.executable, "-m", "ironmule_cli", "serve", "--gguf", str(args.models / "Qwen3-32B-Q4_K_M.gguf"),
                   "--state-dir", str(args.state), "--agent", "on", "--no-thinking", "--host", "127.0.0.1",
                   "--port", port, "--api-key-env", "IRONMULE_COMPARE_LOCAL_KEY"]
    start = time.time()
    save(target / f"start-{label}.json", {"time": start, "hardware": binding, "source_commit": source})
    with (target / f"serve-{label}.log").open("xb") as log:
        proc = subprocess.Popen(command, cwd=args.checkout, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            ready = wait_ready(proc, url, model)
            save(target / f"ready-{label}.json", {**ready, "time": time.time(), "startup_s": time.time() - start})
            for width, count, suffix in ((32, 3, ""), (8, 2, "q")):
                name = label + suffix
                cmd = [sys.executable, "lab/pk1/http_bench.py", "--url", url, "--engine", "vllm" if vllm else "ironmule",
                       "--model", model, "--label", name, "--widths", str(width), "--repeats", str(count),
                       "--seed", str(seed), "--max-tokens", "32"]
                cmd += ["--timeline", str(target / f"timeline-{label}.jsonl")] if width == 32 else ["--quality"]
                with (target / f"http_{name}_{width}.jsonl").open("xb") as rows, (target / f"benchmark-{name}.log").open("xb") as err:
                    subprocess.run(cmd, cwd=args.checkout, env=env, stdout=rows, stderr=err, check=True, timeout=420)
        finally:
            stop(proc)
    save(target / f"end-{label}.json", {"time": time.time(), "hardware": board()})
    time.sleep(10)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkout", "models", "vllm", "output", "state"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(mode=0o700, parents=True, exist_ok=True)
    if output("git", "-C", str(args.checkout), "rev-parse", "HEAD") != SOURCE:
        raise ValueError("unexpected code revision")
    binding = board()
    gguf = args.models / "Qwen3-32B-Q4_K_M.gguf"
    if output("sha256sum", str(gguf)).split()[0] != GGUF:
        raise ValueError("wrong GGUF")
    awq = json.loads(output(sys.executable, str(args.checkout / "lab/portable_gpu/fetch_awq.py"), str(args.models / "qwen3-32b-awq"), "--verify-only"))
    versions = json.loads(output(str(args.vllm / "bin/python"), "-c",
        "import importlib.metadata as m,json;print(json.dumps({n:m.version(n) for n in ('vllm','torch','transformers','xformers')}))"))
    if versions["vllm"] != "0.11.0" or versions["transformers"] != "4.57.1":
        raise ValueError("unregistered competitor version")
    save(args.output / "target.json", {"source_commit": SOURCE, "gguf_sha256": GGUF, "awq": awq, "hardware": binding, "versions": versions})
    for label in ("K1", "vllm", "K2"):
        arm(args, label, binding)
    save(args.output / "completed.json", {"hardware": board(), "process_order": ["K1", "vllm", "K2"], "time": time.time()})


if __name__ == "__main__":
    main()
