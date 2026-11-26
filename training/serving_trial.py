"""Time fixed copy requests through the product GGUF tokenizer, Pilot and native engine on Linux.

Point PYTHONPATH at the implementation being measured and run this file from the trusted
evaluator checkout. Warm-up and profiling are reported separately from serving measurements.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import random
import re
import resource
import subprocess
import time
from contextlib import closing, nullcontext
from pathlib import Path

from ironmule import ggml_engine as ge
from ironmule.ggml_binary import identity
from ironmule.gguf_tokenizer import GgufTokenizer
from ironmule_devices import ggml_devices, verify_ggml
from lab.portable_gpu.bench import save

EVALUATOR = "ironmule.serving_trial.v1"
REPO = Path(ge.__file__).resolve().parents[1]
COLORS = ("orange", "green", "black", "yellow", "white", "pink", "brown", "purple")


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_digest(path):
    checksum = hashlib.sha256()
    with open(path, "rb") as src:
        for block in iter(lambda: src.read(1 << 20), b""):
            checksum.update(block)
    return checksum.hexdigest()


def fixture(seed, clients, workload):
    rng = random.Random(seed)
    fixtures = []
    for lane in range(clients):
        target = COLORS[(seed + lane) % len(COLORS)]
        prompt = (f"Record {rng.randrange(1_000_000)}. " + "Notes for the copy task. " * (1 if workload == "short" else 12)
                  + f"\nColor: red\nCopy: red\nColor: blue\nCopy: blue\nColor: {target}\nCopy:")
        fixtures.append((prompt, target))
    return fixtures


def run_wave(engine, tokenizer, fixtures, limit, profile):
    """Time tokenization, scheduling, all native work and the visible output callback for each stream."""
    jobs, streams = [], []
    arrived = time.monotonic_ns()
    for prompt, _ in fixtures:
        stream = {"decoder": tokenizer.detokenizer, "pieces": [], "first": None}
        streams.append(stream)

        def emit(token, stream=stream):
            stream["decoder"].add_token(token)
            piece = stream["decoder"].last_segment
            if piece:
                if stream["first"] is None:
                    stream["first"] = time.monotonic_ns()
                stream["pieces"].append(piece)

        jobs.append(ge.Job(tokenizer.encode(prompt), limit, emit))
    engine.run(jobs)
    for stream in streams:
        stream["decoder"].finalize()
        stream["pieces"].append(stream["decoder"].last_segment)
    elapsed = (time.monotonic_ns() - arrived) / 1e9
    outputs, ttfts = [], []
    for job, stream, (_, target) in zip(jobs, streams, fixtures):
        if stream["first"] is None or job.reason not in ("length", "eos") or job.tokens != stream["decoder"].tokens:
            raise RuntimeError("incomplete generation or delivery")
        answer = "".join(stream["pieces"])
        word = re.search(r"\w+", answer.casefold())
        outputs.append({"prompt_digest": digest(job.ids), "output_ids": job.tokens, "answer": answer,
                        "stop": job.reason, "target": target, "correct": bool(word and word.group() == target)})
        ttfts.append((stream["first"] - arrived) / 1e9)
    correct = sum(r["correct"] for r in outputs)
    profile.update(requests=[[len(j.ids), len(j.tokens)] for j in jobs], width=len(jobs), correct=correct == len(jobs))
    return {"wall_s": elapsed, "ttfts_s": ttfts, "output_sha256": digest(outputs), "completed": len(outputs),
            "tokens": sum(len(j.tokens) for j in jobs), "correct": correct, "requests": profile["requests"],
            "reused_prompt_tokens": sum(j.reused for j in jobs), "answers": outputs}


def measure(args):
    started = time.monotonic()
    L = ge.bindings()
    L.llama_backend_init()
    devices = ggml_devices(L)
    provider = args.require_provider or (devices[0]["provider"] if ge.gpu_layers(L) and devices else "cpu")
    if provider != "cpu" and len(devices) != 1:
        raise ValueError("select exactly one visible GPU to measure this workload")
    verified = verify_ggml(provider)
    placement = []

    @ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
    def logger(level, message, _):
        if match := re.search(rb"offloaded (\d+)/(\d+) layers to GPU", message or b""):
            placement.append(list(map(int, match.groups())))
        if message and (level >= 3 or b"offloaded" in message):
            os.write(2, message)

    L.llama_log_set(logger, None)
    try:
        engine = ge.GgmlEngine(str(args.gguf), threads=args.threads, seqs=args.clients, seq_ctx=512)
    finally:
        L.llama_log_set(None, None)
    with closing(engine), closing(GgufTokenizer(str(args.gguf), L)) as tokenizer:
        if provider != "cpu" and not any(n == total and n > 0 for n, total in placement):
            raise RuntimeError("GGUF loader did not confirm full GPU offload")
        if provider == "cpu" and engine.gpu_layers:
            raise RuntimeError("CPU workload requested but engine configured for GPU offload")
        if engine.pilot is None:
            raise RuntimeError(f"Pilot inactive: {engine.pilot_refused}")
        observed = [{k: v for k, v in row.items() if k != "free_memory"} for row in verified]
        native = identity()
        if not native.get("sha256"):
            raise RuntimeError("could not bind the loaded native runtime")
        binding = {"provider": provider, "model_sha256": file_digest(args.gguf), "model_name": args.gguf.name,
                   "model_bytes": args.gguf.stat().st_size, "machine": ge.machine_key(), "devices": observed,
                   "native": native, "python": platform.python_version(), "gpu_layers": engine.gpu_layers,
                   "offloaded_layers": placement, "affinity": sorted(os.sched_getaffinity(0)),
                   "pilot_weights_sha256": file_digest(engine.pilot.weights_path),
                   "cores": engine.pilot.cores, "threads": args.threads, "engine_settings": engine.settings,
                   "prefix_reuse": engine.prefix_reuse, "speculate": engine.speculate,
                   "env_sha256": digest({k: v for k, v in os.environ.items() if k.startswith(
                       ("IRONMULE_", "GGML_", "OPENBLAS_", "OMP_", "CUDA_"))})}
        workload = {"seed": args.seed, "clients": args.clients, "waves": args.waves, "max_tokens": args.tokens,
                    "warmup_waves": 1, "context": 512, "phases": ["short", "document"]}
        initialization = time.monotonic() - started
        warmup = run_wave(engine, tokenizer, fixture(args.seed ^ 0xbeef, args.clients, "short"), args.tokens, {})
        prior_decisions = engine.pilot.decisions
        waves = []
        free_bytes = []
        for num in range(args.waves):
            phase = workload["phases"][num % 2]
            cases = fixture(args.seed + num, args.clients, phase)
            monitor = nullcontext({})
            if args.profile_dir:
                # This module lives beside the trusted evaluator when invoked as a script.
                try:
                    from .experiment_planner import observe_wave
                except ImportError:
                    from experiment_planner import observe_wave
                monitor = observe_wave(args.profile_dir, phase, binding, REPO, evaluator_id=EVALUATOR)
            with monitor as profile:
                wave = run_wave(engine, tokenizer, cases, args.tokens, profile)
            waves.append({"workload": phase, **wave})
            if provider != "cpu":
                free_bytes.append(ggml_devices(L)[0]["free_memory"])
        decisions = engine.pilot.decisions - prior_decisions
        if decisions <= 0:
            raise RuntimeError("Pilot did not make an in-range decision in the measured run")
        wall_s = sum(row["wall_s"] for row in waves)
        ttfs = sorted(v for row in waves for v in row["ttfts_s"])
        return {"schema": EVALUATOR, "profiled": bool(args.profile_dir), "binding": binding, "workload": workload,
                "commit": args.revision or subprocess.check_output(
                    ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
                "runtime_source_sha256": {name: file_digest(REPO / "ironmule" / name) for name in
                                          ("ggml_engine.py", "gguf_tokenizer.py", "pilot.py")},
                "evaluator_sha256": file_digest(__file__), "initialization_s": initialization,
                "warmup_s": warmup["wall_s"], "wall_s": wall_s, "ttft_s": ttfs[math.ceil(.95 * len(ttfs)) - 1],
                "output_sha256": digest([w["output_sha256"] for w in waves]), "waves": waves,
                "completed": sum(w["completed"] for w in waves), "correct": sum(w["correct"] for w in waves),
                "tokens": sum(w["tokens"] for w in waves), "pilot_decisions": decisions, "pilot": engine.pilot.status(),
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "device_free_bytes_after_waves": free_bytes, "total_elapsed_s": time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--require-provider", choices=("cpu", "cuda"))
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--waves", type=int, default=4, choices=range(1, 65), metavar="1..64")
    parser.add_argument("--threads", type=int, default=2, choices=range(1, 257), metavar="1..256")
    parser.add_argument("--clients", type=int, default=4, choices=range(1, 33), metavar="1..32")
    parser.add_argument("--tokens", type=int, default=3, choices=range(1, 65), metavar="1..64")
    parser.add_argument("--seed", type=int, default=1241)
    parser.add_argument("--revision", help="40-character base Git SHA supplied by the outer worktree runner")
    args = parser.parse_args()
    if args.revision is not None and not re.fullmatch(r"[a-f0-9]{40}", args.revision):
        parser.error("--revision requires a full Git SHA")
    report = measure(args)
    save(args.out, report)
    if report["correct"] != report["completed"]:
        raise SystemExit(f"copy quality failed: {report['correct']}/{report['completed']} correct")


if __name__ == "__main__":
    main()
