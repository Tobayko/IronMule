"""Collect the new warm HTTP series incrementally on an already reserved RunPod lease."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import time
from pathlib import Path
from urllib.parse import urlencode

from lab.pk6 import warm_repeat
from lab.rp import native_campaign as jobs
from lab.rp import rp

TARGET = "/work/warm-results"
FILES = ["target.json", "completed.json"]
for label in ("K1", "vllm", "K2"):
    FILES += [f"{prefix}-{label}.json" for prefix in ("start", "ready", "end")]
    FILES += [f"http_{label}_32.jsonl", f"http_{label}q_8.jsonl", f"timeline-{label}.jsonl",
              f"benchmark-{label}.log", f"benchmark-{label}q.log", f"serve-{label}.log"]


def collect(pod, state, *, target=TARGET, files=FILES):
    paths = [target + "/" + name for name in files]
    sizes = jobs.shell(pod, 'for path in ' + shlex.join(paths) +
                       '; do if test -f "$path"; then stat -c "%n %s" "$path"; fi; done')
    for line in sizes.splitlines():
        remote, size = line.rsplit(" ", 1)
        name = Path(remote).name
        if remote not in paths:
            raise ValueError("unexpected target file")
        dest = state / name
        if dest.is_file() and dest.stat().st_size == int(size):
            continue
        raw = jobs.call(pod, "GET", jobs.endpoint("get", remote), raw=True)
        if len(raw) < int(size) or (dest.exists() and not raw.startswith(dest.read_bytes())):
            raise ValueError("target record was truncated or replaced")
        temp = dest.with_suffix(".incoming")
        with temp.open("wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(dest)


def status(pod, name):
    return jobs.call(pod, "GET", "/job?" + urlencode({"name": name, "tail": 8}))


def run(state, pod_id, deadline):
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    script = Path(warm_repeat.__file__)
    remote = "/work/warm_repeat_runner.py"
    name = "registered-warm-repeat"
    while time.monotonic() < deadline:
        pod = jobs.lease(pod_id)
        setup = status(pod, "warm-setup")
        if setup.get("running"):
            time.sleep(8)
            continue
        if setup.get("rc") != 0:
            raise RuntimeError("target setup incomplete: " + str(setup)[-600:])
        run_state = status(pod, name)
        if not run_state.get("running") and run_state.get("rc") is None:
            jobs.upload(pod, remote, script.read_bytes())
            cmd = """set -eu
cd /work/IronMule
export IRONMULE_GGML_BINDINGS="$HOME/.cache/ironmule/ggml/ironmule/cuda/llama_cpp_python-0.3.16"
export LLAMA_CPP_LIB_PATH="$IRONMULE_GGML_BINDINGS/vendor/llama.cpp/build/bin"
umask 077
exec python /work/warm_repeat_runner.py --checkout /work/IronMule --models /models --vllm /opt/vllm --state /work/state --output /work/warm-results"""
            jobs.job(pod, name, cmd)
        collect(pod, state)
        if run_state.get("rc") is not None and not run_state.get("running"):
            return run_state
        time.sleep(8)
    return {"running": True, "detail": "polling window elapsed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--seconds", required=True, type=int, choices=range(1, 10801), metavar="1..10800")
    args = parser.parse_args()
    print(json.dumps(run(args.state, args.pod_id, time.monotonic() + args.seconds)), flush=True)


if __name__ == "__main__":
    main()
