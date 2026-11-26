"""Start and continuously collect one mixed-progress trial on an already budgeted lease."""

import argparse
import json
from pathlib import Path
import re
import shlex
import time
from urllib.error import URLError

from lab.pk6.mixed_progress.trial import ARMS
from lab.rp import native_campaign as jobs
from lab.rp import warm_repeat as sync

JOB = "mixed-progress-trial"
TARGET = "/work/mixed-progress"


def collect(pod_id, state, commit):
    if not re.fullmatch("[a-f0-9]{40}", commit):
        raise ValueError("expected a full experiment revision")
    launch = f"""set -eu
mkdir -p /work
git clone --quiet https://github.com/Tobayko/IronMule.git /work/IronMule
cd /work/IronMule
git checkout --detach {shlex.quote(commit)}
git apply --check lab/pk6/mixed_progress/engine.patch
git apply lab/pk6/mixed_progress/engine.patch
env JOBS=16 PIP_NO_CACHE_DIR=1 VLLM=1 bash lab/rp/setup.sh 89
export IRONMULE_GGML_BINDINGS="$HOME/.cache/ironmule/ggml/ironmule/cuda/llama_cpp_python-0.3.16"
export LLAMA_CPP_LIB_PATH="$IRONMULE_GGML_BINDINGS/vendor/llama.cpp/build/bin"
export PYTHONPATH="$PWD"
exec python -m lab.pk6.mixed_progress.trial --out {TARGET} --state /work/state --vllm /opt/vllm --models /models
"""
    files = ["target.json", "completed.json", "device-load.csv"]
    for arm in ARMS:
        files.extend(
            [
                f"serve-{arm}.log",
                f"waves-{arm}.jsonl",
                *(f"{s}-{arm}.json" for s in ("start", "ready", "end")),
            ]
        )
        if arm != "vllm":
            files.append(f"trace-{arm}.txt")
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    while True:
        pod = jobs.lease(pod_id)
        try:
            if jobs.call(pod, "GET", "/health").get("ok"):
                status = sync.status(pod, JOB)
                if not status.get("running") and status.get("rc") is None:
                    jobs.job(pod, JOB, launch)
                sync.collect(pod, state, target=TARGET, files=files)
                if status.get("rc") is not None and not status.get("running"):
                    (state / "control-end.json").write_text(json.dumps(status))
                    return status["rc"]
        except (URLError, OSError, json.JSONDecodeError):
            pass  # the run and the two lease watchdogs survive a transient bridge outage
        time.sleep(6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    raise SystemExit(collect(args.pod_id, args.state, args.commit))
