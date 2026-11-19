"""Run both HTTP arms locally after tools/build_ggml.py. Its printed exports must be set.

python lab/ag10/run_pair.py /path/to/model.gguf /new/output-directory
Outputs include server logs, product state directories, per-wave results and a paired summary.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from urllib.error import URLError

if __package__:
    from .http_transition import PHASES, health, measure
else:
    from http_transition import PHASES, health, measure


def run_arm(arm: str, args: argparse.Namespace) -> dict:
    state = args.out / (arm + "-state")
    command = [sys.executable, "-m", "ironmule_cli"]
    subprocess.run([*command, "setup", "--state-dir", str(state), "--mode", "server"], check=True,
                   stdout=subprocess.DEVNULL)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    env = {**os.environ, "IRONMULE_PILOT": "0", "IRONMULE_GGML_SHARED": "0", "IRONMULE_GGML_GPU_LAYERS": "0",
           "IRONMULE_GGML_AGENT": str(int(arm == "agent"))}
    # A manual width suppresses SEQ1 startup calibration and disables the AG10 agent.
    env.pop("IRONMULE_GGML_SEQS", None)
    with (args.out / (arm + "-server.log")).open("w") as log:
        process = subprocess.Popen([*command, "serve", "--gguf", str(args.gguf.resolve()), "--state-dir",
                                    str(state), "--port", str(port), "--speculate", "off"], env=env,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + args.startup_timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"{arm} server stopped during startup; see {log.name}")
                try:
                    if health(url)["ready"]:
                        break
                except (URLError, OSError):
                    pass
                time.sleep(1)
            else:
                raise TimeoutError(f"{arm} startup deadline exceeded; see {log.name}")
            result = measure(argparse.Namespace(**vars(args), arm=arm, url=url))
            (args.out / (arm + ".json")).write_text(json.dumps(result, indent=2) + "\n")
            return result
        finally:
            # The calibration and worker subprocesses inherit this dedicated process group.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def compare(agent: dict, fixed: dict) -> dict:
    if (agent["arm"], fixed["arm"]) != ("agent", "fixed") or agent["model"] != fixed["model"]:
        raise ValueError("expected agent and fixed arms on the same model")
    if (agent["protocol"] != fixed["protocol"]
            or agent["starting_ggml"]["sequences"] != fixed["starting_ggml"]["sequences"]
            or agent["starting_ggml"]["gate"] != fixed["starting_ggml"]["gate"]):
        raise ValueError("protocol, startup measurement or quality gate differ")
    summary = {}
    for phase in PHASES:
        a, f = agent["phases"][phase], fixed["phases"][phase]
        summary[phase] = {"throughput_ratio": a["completion_tok_s_active"] / f["completion_tok_s_active"],
                          "p95_ttft_ratio": a["ttft_p95_s"] / f["ttft_p95_s"],
                          "same_answers": sum(x == y for x, y in zip(a["outputs"], f["outputs"], strict=True)),
                          "requests": a["requests"], "agent_correct": a["correct"], "fixed_correct": f["correct"]}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--lines", type=int, default=96)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--idle", type=float, default=5)
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--first", choices=("agent", "fixed"), default="agent")
    parser.add_argument("--gate-cache", type=Path, help="optional ggml-gate directory from a previous run on this machine")
    args = parser.parse_args()
    if not args.gguf.is_file() or min(args.requests, args.width, args.lines, args.max_tokens, args.startup_timeout) < 1:
        parser.error("provide a GGUF file and positive sizes")
    if not 0 <= args.idle < float("inf"):
        parser.error("--idle must be finite and nonnegative")
    args.out.mkdir(parents=True, exist_ok=False)
    first, second = args.first, "fixed" if args.first == "agent" else "agent"
    first_cache = args.out / (first + "-state") / "ggml-gate"
    if args.gate_cache:
        shutil.copytree(args.gate_cache, first_cache)
    results = {first: run_arm(first, args)}
    # Use the same completed startup calibrations but never copy a learned policy or past traffic.
    shutil.copytree(first_cache, args.out / (second + "-state") / "ggml-gate")
    results[second] = run_arm(second, args)
    summary = compare(results["agent"], results["fixed"])
    (args.out / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
