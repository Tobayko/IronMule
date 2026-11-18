"""Prequential workload forecasts from actual SmolLM2 inference. Protocol in docs/BACKLOG.md."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ironmule import ggml_agent as ag, ggml_engine as ge
from lab.portable_gpu.bench import TARGETS, output_text, save

PHASES = ("short", "document", "followup")
SEED = 728103
FLOORS = (1, 1, 1, 0.001)
CONTEXT = 192


def digest(path):
    with path.open("rb") as handle:
        checksum = hashlib.sha256()
        for part in iter(lambda: handle.read(2**20), b""):
            checksum.update(part)
    return checksum.hexdigest()


def forecast(memory, matched, seed):
    profiles = []
    for window in ag.windows(memory, random.Random(seed), size=32, matched=matched):
        previous = 0.0
        rows = []
        for arrived, prompt, answer, computed in window:
            rows.append((arrived - previous, prompt, answer, computed))
            previous = arrived
        profiles.append(ag.workload_profile(rows))
    return [statistics.fmean(values) for values in zip(*profiles)]


def prepare_wave(engine, role, phase, block, wave):
    jobs, targets = [], []
    length = 32 if phase == "short" else 128
    rng = random.Random(SEED + block * 32 + wave * 8)
    for lane in range(8):
        target = TARGETS[role][(wave + block + lane) % len(TARGETS[role])]
        if phase == "followup":
            prefix = f"Session {lane % 4}. Tool result: " + " city records show clear weather. " * 40
        else:
            prefix = f"{rng.getrandbits(60):x} " + " Notes for the copy task. " * 40
        tail = engine.tokenize(f"\nColor: red\nCopy: red\nColor: blue\nCopy: blue\nColor: {target}\nCopy:")
        prefix = engine.tokenize(prefix)
        if len(tail) >= length or len(prefix) < length - len(tail):
            raise ValueError("copy fixture cannot fit")
        jobs.append(ge.Job(prefix[:length - len(tail)] + tail, 3))
        targets.append(target)
    return jobs, targets


def collect(args):
    if args.output.exists() or args.state_dir.exists():
        raise ValueError("use a fresh state directory and output file")
    if ge.THREADS != 2 or not ge.PREFIX_REUSE or not ge.gpu_layers(ge.bindings()) == 0:
        raise ValueError("CPU, two threads and prefix reuse required")
    args.state_dir.mkdir(parents=True)
    started = time.monotonic()
    calibration = ge.measure_widths(str(args.gguf), {}, CONTEXT, [4, 8], steps=3, repeats=2, long_ctx=96,
                                    experience={"prompt_tokens": 64, "concurrency": 8, "widths": [4, 8]})
    configs = ag.configs_from_measurement([calibration], CONTEXT)
    outputs = calibration.get("output_digests", {})
    qualified = bool(outputs.get("4") and outputs["4"] == outputs.get("8"))
    if not qualified:
        del configs["w8"]
    binding = {"gguf_sha256": digest(args.gguf), "provider": "cpu",
               "machine": ge.machine_key(), "threads": 2, "study": "ag10-replay-v1"}
    agent = ag.GgmlAgent(args.state_dir, binding, configs, "w4", seed=SEED)
    engine = ge.GgmlEngine(str(args.gguf), seqs=4, seq_ctx=CONTEXT, threads=2, speculate=False)
    engine.observer = agent.observe_step
    source = {str(path.relative_to(ROOT)): digest(path) for path in
              (Path(__file__).resolve(), ROOT / "ironmule/ggml_agent.py", ROOT / "ironmule/ggml_engine.py")}
    report = {"schema": "ironmule.ag10.replay-study.v1", "source_sha256": source,
              "gguf_sha256": binding["gguf_sha256"], "cpu_arch": platform.machine(), "threads": 2, "seed": SEED,
              "native_llama_sha256": digest(Path(os.environ["LLAMA_CPP_LIB_PATH"]) / "libllama.so"),
              "context": CONTEXT, "max_new_tokens": 3, "width4_width8_calibration_exact": qualified,
              "calibration_output_digests": outputs, "configs": sorted(configs),
              "features": ["prompt_tokens", "computed_prompt_tokens", "generated_tokens", "arrival_gap_s"],
              "blocks": []}
    try:
        for cycle, role in enumerate(("calibration", "holdout", "replay")):
            for phase in PHASES:
                for _ in range(3):
                    index = len(report["blocks"])
                    memory = list(agent.memory)
                    matched = ag.matching_windows(memory, 32)
                    forecasts = {}
                    if cycle:
                        forecasts = {kind: forecast(memory, starts, SEED + index)
                                     for kind, starts in (("recent", []), ("conditioned", matched))}
                    row = {"block": index, "role": role, "phase": phase, "serving": agent.current,
                           "matched_prior_blocks": len(matched)}
                    correct, emitted = 0, 0
                    for wave in range(4):
                        jobs, targets = prepare_wave(engine, role, phase, index, wave)
                        arrived = time.monotonic_ns()
                        engine.run(jobs)
                        for job, target in zip(jobs, targets):
                            if not (job.reason in ("eos", "length") and arrived <= job.first_ns <= job.finished_ns):
                                raise ValueError("incomplete native request")
                            agent.observe_request(arrived / 1e9, len(job.ids), len(job.tokens), len(job.ids) - job.reused,
                                                  first=job.first_ns / 1e9, finished=job.finished_ns / 1e9)
                            first = re.search(r"\w+", output_text(engine, job.tokens).casefold())
                            correct += int(bool(first and first.group() == target))
                            emitted += len(job.tokens)
                    actual = ag.workload_profile(list(agent.memory)[-32:])
                    row.update(requests=32, correct=correct, generated=emitted, actual=actual)
                    if cycle:
                        row["forecast"] = forecasts
                        row["error"] = {k: statistics.fmean(abs(math.log2((pred + f) / (val + f)))
                                                for pred, val, f in zip(profile, actual, FLOORS))
                                        for k, profile in forecasts.items()}
                    target_config = agent.idle()
                    if agent.history and agent.history[-1]["requests"] == agent.requests:
                        row["decision"] = agent.history[-1]["action"]
                    if target_config:
                        before = time.monotonic()
                        engine.reconfigure(configs[target_config]["seqs"], configs[target_config]["settings"])
                        agent.switched(target_config, seconds=time.monotonic() - before)
                        row["switched_to"] = target_config
                    report["blocks"].append(row)
                    print(f"block {index} {role} {phase} {correct}/32 {agent.current}", flush=True)
    finally:
        engine.close()
    heldout = [row for row in report["blocks"] if row["role"] != "calibration"]
    report["holdout"] = {"blocks": len(heldout), "requests": sum(row["requests"] for row in heldout),
                         "correct": sum(row["correct"] for row in heldout),
                         "conditioned_blocks": sum(bool(row["matched_prior_blocks"]) for row in heldout),
                         "mean_absolute_log2_error": {kind: statistics.fmean(row["error"][kind] for row in heldout)
                                                      for kind in ("recent", "conditioned")}}
    report["live_steps"] = agent.live
    report["world_models"] = agent.status()["world_model"]
    report["elapsed_s"] = time.monotonic() - started
    save(args.output, report)
    print(json.dumps(report["holdout"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    collect(parser.parse_args())
