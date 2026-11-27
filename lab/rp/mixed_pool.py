"""Run the bounded follow-up against the qualified shared-pool reference using an isolated checkout."""

import json
from pathlib import Path
import time

from lab.pk6.mixed_progress import pool_trial
from lab.rp import native_campaign as jobs, warm_repeat as sync


def collect(pod_id, primary, out):
    pod = jobs.lease(pod_id)
    if pod["deadline"] - time.time() < 8 * 60:
        return "not enough time in existing reservation"
    for arm in ("R1", "R2"):
        health = json.loads((primary / f"ready-{arm}.json").read_bytes())["health"]
        if (
            not health.get("shared_pool")
            or health.get("kernel_fallback", {}).get("serving") != "stock"
        ):
            return "reference not shared stock"
    if sync.status(pod, "current-vllm").get("running"):
        raise RuntimeError("competitor still active")
    out.mkdir(mode=0o700, exist_ok=True)
    root = Path(pool_trial.__file__).parent
    jobs.upload(
        pod, "/work/pool-budget.patch", (root / "pool-budget.patch").read_bytes()
    )
    jobs.upload(pod, "/work/pool_runner.py", (root / "pool_trial.py").read_bytes())
    cmd = """set -eu
cd /work/IronMule
git worktree add --detach /work/pool-source HEAD
cd /work/pool-source
git restore --source f97a05f74561a9b5bf4dbebcc727d900c108df29 -- ironmule_product/cli.py ironmule/ggml_profiles.py
git apply lab/pk6/mixed_progress/engine.patch
git apply /work/pool-budget.patch
export IRONMULE_GGML_BINDINGS="$HOME/.cache/ironmule/ggml/ironmule/cuda/llama_cpp_python-0.3.16"
export LLAMA_CPP_LIB_PATH="$IRONMULE_GGML_BINDINGS/vendor/llama.cpp/build/bin"
export PYTHONPATH="$PWD"
exec python /work/pool_runner.py --primary /work/mixed-recovered --out /work/mixed-pool --models /models --state /work/state"""
    jobs.job(pod, "shared-pool-budget", cmd)
    files = ["target.json", "completed.json"]
    for a in ("S1", "S2"):
        files.extend([f"{s}-{a}.json" for s in ("start", "ready", "end")])
        files.extend([f"waves-{a}.jsonl", f"trace-{a}.txt", f"serve-{a}.log"])
    while time.time() < pod["deadline"] - 35:
        try:
            stat = sync.status(pod, "shared-pool-budget")
            sync.collect(pod, out, target="/work/mixed-pool", files=files)
            if not stat.get("running") and stat.get("rc") is not None:
                (out / "control-end.json").write_text(json.dumps(stat))
                return "complete" if (out / "completed.json").exists() else "incomplete"
        except (OSError, json.JSONDecodeError):
            pass
        time.sleep(5)
    return "deadline"
