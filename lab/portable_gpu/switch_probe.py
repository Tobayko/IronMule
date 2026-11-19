"""Bounded GGML controller replay with real width-switch costs on one device."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import bench

SOURCE = ("lab/portable_gpu/switch_probe.py", "lab/portable_gpu/bench.py", "ironmule/ggml_engine.py",
          "ironmule/ggml_agent.py", "ironmule/ggml_binary.py", "ironmule_devices.py",
          "ironmule/pilot.py", "ironmule/speculation.py")


def serve(engine, tasks, limit, arrived_ns):
    from ironmule.ggml_engine import Job

    engine._clear()
    jobs = [Job(ids, limit) for ids, _ in tasks]
    engine.run(jobs)
    last = max(job.finished_ns for job in jobs)
    rows = []
    for (_, expected), job in zip(tasks, jobs):
        if job.reason not in ("length", "eos") or not (arrived_ns <= job.first_ns <= job.finished_ns):
            raise RuntimeError("invalid completion")
        answer = bench.output_text(engine, job.tokens)
        first_word = re.search(r"\w+", answer.casefold())
        rows.append({"expected": expected, "answer": answer, "correct": bool(first_word and first_word.group() == expected),
                     "tokens": job.tokens, "stop": job.reason, "ttft_ms": (job.first_ns - arrived_ns) / 1e6})
    observations = [(arrived_ns / 1e9, len(job.ids), len(job.tokens), len(job.ids) - job.reused,
                     job.first_ns / 1e9, job.finished_ns / 1e9) for job in jobs]
    return {"requests": rows, "elapsed_s": (last - arrived_ns) / 1e9}, observations


def record(agent, arrivals, shift=0.0):
    for arrived, prompt, answer, computed, first, end in arrivals:
        agent.observe_request(arrived + shift, prompt, answer, computed, first=first + shift, finished=end + shift)


def replay(engine, configs, binding, calibration, fixtures, args, label, seed):
    from ironmule.ggml_agent import GgmlAgent

    # Both arms observe local costs; only the diagnostic arm applies them to its dream.
    agent = GgmlAgent(None, binding, configs, "w1", seed=seed, price_switches=label == "cost_aware")
    if engine.seqs != 1:
        start = time.monotonic()
        engine.reconfigure(1, configs["w1"]["settings"])
        reset_s = time.monotonic() - start
    else:
        reset_s = 0.0
    observations, steps = calibration
    for step in steps:
        agent.observe_step(*step)
    # Align the same measured traffic history with each independent replay start.
    record(agent, observations, time.monotonic() - max(obs[5] for obs in observations))
    if agent.idle() is not None:
        raise RuntimeError("unexpected switch before replay")
    engine.observer = agent.observe_step
    waves = []
    for label_wave, tasks in fixtures:
        arrived = time.monotonic_ns()
        prior = agent.current
        target = agent.idle() if waves else None
        decision_s = (time.monotonic_ns() - arrived) / 1e9
        switched_s = None
        if target is not None and target != agent.current:
            started = time.monotonic()
            config = configs[target]
            engine.reconfigure(config["seqs"], config["settings"])
            switched_s = time.monotonic() - started
            agent.switched(target, seconds=switched_s)
        if engine.seqs != configs[agent.current]["seqs"]:
            raise RuntimeError("selected and serving widths diverged")
        result, completions = serve(engine, tasks, args.tokens, arrived)
        record(agent, completions)
        waves.append({**result, "wave": label_wave, "config_before": prior, "config_served": agent.current,
                      "idle_invoked": len(waves) > 0, "decision_s": decision_s, "physical_switch_s": switched_s,
                      "decision": agent.history[-1] if agent.history else None})
    engine.observer = None
    duration = sum(w["elapsed_s"] for w in waves)
    requests = [r for w in waves for r in w["requests"]]
    return {"policy": label, "price_switches": agent.price_switches, "seed": seed, "between_replays_reset_s": reset_s, "waves": waves,
            "elapsed_s": duration, "output_tokens": sum(len(r["tokens"]) for r in requests),
            "tokens_per_second": sum(len(r["tokens"]) for r in requests) / duration,
            "median_ttft_ms": statistics.median(r["ttft_ms"] for r in requests),
            "mean_ttft_ms": statistics.mean(r["ttft_ms"] for r in requests),
            "copy_correct": sum(r["correct"] for r in requests), "request_count": len(requests),
            "switch_count": sum(w["physical_switch_s"] is not None for w in waves)}


def worker(args, report):
    os.environ["IRONMULE_GGML_THREADS"] = str(args.threads)
    bench.configure(args.provider)
    from ironmule.ggml_agent import MIN_ARRIVALS, configs_from_measurement
    from ironmule.ggml_binary import identity
    from ironmule.ggml_engine import (
        GgmlEngine,
        file_sha256,
        machine_key,
        measure_widths,
    )
    from ironmule_devices import bindings, ggml_devices, verify_ggml

    devices = verify_ggml(args.provider)
    if args.provider != "cpu" and (len(devices) != 1 or len(ggml_devices()) != 1):
        raise RuntimeError("probe requires one visible, verified GPU")
    report.update(devices=devices, machine=machine_key(), native=identity(), model_sha256=file_sha256(str(args.gguf)))
    layers = []
    L = bindings()

    @ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
    def logger(level, message, data):
        if match := re.search(rb"offloaded (\d+)/(\d+) layers to GPU", message or b""):
            layers.append([int(n) for n in match.groups()])
        if message and level >= 3:
            os.write(2, message)

    L.llama_log_set(logger, None)
    try:
        seq_ctx = max(512, args.long_prompt + 256)
        measured = measure_widths(str(args.gguf), {}, seq_ctx, (1, 4), steps=4, repeats=2, long_ctx=args.long_prompt)
        configs = configs_from_measurement([measured], seq_ctx)
        if any(not configs[k].get(field) for k in ("w1", "w4") for field in ("context_build_s", "context_release_s")):
            raise RuntimeError("incomplete native width calibration")
        report.update(configs=configs, width_calibration=measured, seq_ctx=seq_ctx)
        bench.save(args.output, report)
        layers.clear()
        engine = GgmlEngine(str(args.gguf), seqs=1, seq_ctx=seq_ctx, speculate=False, threads=args.threads)
        try:
            report["serving_offloaded_layers"] = layers.copy()
            if args.provider != "cpu" and not any(n == total and n > 0 for n, total in layers):
                raise RuntimeError("serving model did not confirm full GPU offload")
            binding = {"model": report["model_sha256"], "machine": report["machine"],
                       "gpu_layers": engine.gpu_layers}
            report["cost_aware_gpu_branch_active"] = bool(binding["gpu_layers"])
            tasks = bench.prompts(engine, "calibration", 32, 4)
            observations, steps = [], []
            engine.observer = lambda *step: steps.append(step)
            while len(observations) < MIN_ARRIVALS - 1:
                batch = tasks[:min(4, MIN_ARRIVALS - 1 - len(observations))]
                _, arrivals = serve(engine, batch, args.tokens, time.monotonic_ns())
                observations.extend(arrivals)
            engine.observer = None
            report["history_calibration"] = {"requests": len(observations), "steps": len(steps),
                                             "targets": bench.TARGETS["calibration"]}
            short = bench.prompts(engine, "replay", 32, 4)
            docs = bench.prompts(engine, "replay", args.long_prompt, 4)
            fixtures = [("short", short[:1]), ("four_documents", docs), ("short_after_documents", short[1:2]),
                        ("sustained", short * 8), ("after_sustained", short * 4)]
            for rep in range(args.repeats):
                order = ("unpriced", "cost_aware") if rep % 2 == 0 else ("cost_aware", "unpriced")
                for label in order:
                    result = replay(engine, configs, binding, (observations, steps), fixtures, args, label, rep)
                    report["replays"].append(result)
                    bench.save(args.output, report)
        finally:
            engine.close()
    finally:
        L.llama_log_set(None, None)
    def outputs(replay_run):
        return [[(r["tokens"], r["stop"]) for r in w["requests"]] for w in replay_run["waves"]]
    runs = report["replays"]
    report["all_output_ids_and_stops_match"] = all(outputs(r) == outputs(runs[0]) for r in runs)
    report["all_copy_correct"] = all(r["copy_correct"] == r["request_count"] for r in runs)
    report["status"] = "complete"
    bench.save(args.output, report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gguf", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--provider", required=True, choices=("cpu", "cuda", "rocm"))
    p.add_argument("--budget-s", default=110, type=int)
    p.add_argument("--threads", default=2, type=int)
    p.add_argument("--long-prompt", choices=(64, 128, 256, 512, 1024), type=int, default=256)
    p.add_argument("--tokens", default=3, type=int)
    p.add_argument("--repeats", default=2, choices=(1, 2), type=int)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if not args.gguf.is_file() or args.gguf.resolve() == args.output.resolve():
        p.error("provide an existing GGUF and a separate output path")
    if args.budget_s <= 0 or not 1 <= args.tokens <= 12 or not 1 <= args.threads <= 128:
        p.error("budget must be positive, tokens 1..12 and threads 1..128")
    if args.worker:
        worker(args, json.loads(args.output.read_text()))
        return 0
    report = {"schema": "ironmule.ggml-switch-probe.v1", "status": "running", "provider": args.provider,
              "replays": [], "long_prompt": args.long_prompt, "max_output_tokens": args.tokens, "threads": args.threads,
              "repeats": args.repeats, "budget_s": args.budget_s,
              "comparison": "Previous unpriced controller and revised cost-aware controller. Both start at width 1.",
              "arrivals": "Closed-loop waves. After each batch completes, the next batch arrives together.",
              "idle_trigger": "Harness invokes idle() between waves, charging decision and any actual rebuild to incoming requests. Product waits for 3 s of quiet.",
              "between_replays_reset": "An administrative reset restores width 1 before another independent policy replay. Its duration is reported outside that policy's request times.",
              "source_sha256": {f: hashlib.sha256((bench.ROOT / f).read_bytes()).hexdigest() for f in SOURCE}}
    bench.save(args.output, report)
    try:
        child = subprocess.run([sys.executable, __file__, *sys.argv[1:], "--worker"], timeout=args.budget_s, check=False)
        if child.returncode == 0:
            return 0
        reason = f"worker exit {child.returncode}"
    except subprocess.TimeoutExpired:
        reason = f"exceeded {args.budget_s}s wall clock"
    report = json.loads(args.output.read_text())
    report.update(status="incomplete", failure=reason)
    bench.save(args.output, report)
    return 1


if __name__ == "__main__":
    sys.exit(main())
