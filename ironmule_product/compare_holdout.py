"""Re-create the selected comparison worker's launch environment before initializing GGML.

This module can also run as a standalone script with the original benchmark's installed
product Python. The original release then owns both the trial and its qualification rules.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import time


def verified_device(report, binding, provider):
    device = report.get("device")
    if provider == "cuda":
        arms = [a["identity"] for a in report["arms"] if a["engine"] == "ironmule"
                and a["identity"].get("ggml") == binding]
        if not isinstance(device, str) or not re.fullmatch(r"GPU-(?:[a-fA-F0-9]{8}-)(?:[a-fA-F0-9]{4}-){3}[a-fA-F0-9]{12}", device):
            raise ValueError("comparison lacks a full CUDA device UUID")
        if not arms or not all(a.get("device") == device and a.get("provider") == "cuda"
                               and a.get("placement_verified") is True and a.get("cuda_visible_devices") == device
                               for a in arms):
            raise ValueError("comparison lacks an identical verified CUDA selector")
    return device


def worker_environment(device, binding, provider):
    env = os.environ.copy()
    if provider == "cuda":
        from ironmule_product.compare_servers import _device

        # Start with the caller's current CUDA visibility. The same resolver used to launch the
        # original arm must select that physical board again before narrowing the child mask.
        selected, _ = _device(env, "ironmule")
        if selected != device or env.get("CUDA_VISIBLE_DEVICES") != device:
            raise ValueError("current visible CUDA device differs from measured incumbent")
    if "seqs" in binding:
        env["IRONMULE_GGML_SEQS"] = str(binding["seqs"])
    env["IRONMULE_GGML_SEQ_CTX"] = str(binding["seq_ctx"])
    if not binding.get("kg_sha256"):
        env["IRONMULE_KG_LIB"] = ""
    return env


def replay(argv=None):
    from ironmule.compare_learning import implementation, respond, select
    from ironmule_product.state import _atomic_write

    parser = argparse.ArgumentParser(description="Remeasure one completed comparison on its original worker hardware.")
    for key in ("report", "gguf", "state", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--budget-seconds", type=int, default=240)
    args = parser.parse_args(argv)
    if not 1 <= args.budget_seconds <= 600 or args.output.exists() or args.output.resolve() == args.report.resolve():
        parser.error("choose a new output path and a trial budget of 1 to 600 seconds")
    report = json.loads(args.report.read_text())
    if report.get("status") != "complete":
        parser.error("a completed comparison is required")
    deadline = time.monotonic() + args.budget_seconds
    selection = select(report, required_implementation=implementation())
    provider = None
    if selection:
        bindings = selection["binding"]
        arms = [a["identity"] for a in report["arms"] if a["engine"] == "ironmule"
                and a["identity"].get("ggml") == bindings]
        provider = arms[0]["provider"]
        for arm in arms:
            command = arm.get("argv", [])
            for flag, target in (("--state-dir", args.state), ("--gguf", args.gguf)):
                if command.count(flag) != 1 or Path(command[command.index(flag) + 1]).resolve() != target.resolve():
                    parser.error("model or state directory differs from the original serving command")
        os.environ.update(worker_environment(verified_device(report, bindings, provider), bindings, provider))
    result = respond(report, gguf=args.gguf, state=args.state, provider=provider, deadline=deadline)
    _atomic_write(args.output, result)
    print(json.dumps({"status": result["status"], "reasons": result.get("reasons", []),
                      "reason": result.get("reason"), "pairs": len(result.get("holdout", []))}))


if __name__ == "__main__":
    replay()
