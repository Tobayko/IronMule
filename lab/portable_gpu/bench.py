"""Bounded local GGUF width measurements through IronMule's native serving engine."""
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
import tempfile
import time
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
WIDTHS = (1, 4)
# Fixed before measurement. Each holdout uses different copy targets from calibration.
TARGETS = {"smoke": ("blue", "red", "gray", "pink"),
           "calibration": ("white", "gold", "green", "black"),
           "holdout": ("brown", "orange", "purple", "yellow"),
           "replay": ("silver", "violet", "cyan", "copper")}


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        try:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.close()
            os.replace(handle.name, path)
        finally:
            Path(handle.name).unlink(missing_ok=True)


def configure(provider):
    # The two arms differ only in n_seq_max and the KV space needed for that width.
    os.environ.update(IRONMULE_GGML_GPU_LAYERS="0" if provider == "cpu" else "999",
                      IRONMULE_GGML_SPLIT="layer", IRONMULE_PILOT="0", IRONMULE_KG5="0",
                      IRONMULE_KG_LIB="", IRONMULE_GGML_PREFIX_REUSE="0", IRONMULE_GGML_DEFRAG="0",
                      IRONMULE_GGML_PAD="0", IRONMULE_GGML_PROMPT_BUDGET="0")
    sys.path.insert(0, str(ROOT))
    # As in lab/kg5, this worker uses only the GGML engine and its direct dependencies.
    package = ModuleType("ironmule")
    package.__path__ = [str(ROOT / "ironmule")]
    sys.modules["ironmule"] = package


def prompts(engine, role, length, count):
    prefix = engine.tokenize(" Notes for the copy task. " * length)
    rows = []
    for target in TARGETS[role][:count]:
        tail = engine.tokenize(f"\nColor: red\nCopy: red\nColor: blue\nCopy: blue\nColor: {target}\nCopy:")
        if len(tail) > length:
            raise ValueError(f"tokenizer needs {len(tail)} tokens for a {length}-token task")
        rows.append((prefix[:length - len(tail)] + tail, target))
    return rows


def output_text(engine, tokens):
    ids = (engine.L.llama_token * len(tokens))(*tokens)
    buf = ctypes.create_string_buffer(256)
    n = engine.L.llama_detokenize(engine.vocab, ids, len(ids), buf, len(buf), True, False)
    if n < 0:
        buf = ctypes.create_string_buffer(-n)
        n = engine.L.llama_detokenize(engine.vocab, ids, len(ids), buf, len(buf), True, False)
    if n < 0:
        raise ValueError("failed to decode output tokens")
    return buf.raw[:n].decode("utf-8", errors="replace")


def sample(engine, tasks, limit, arrived_ns=None):
    from ironmule.ggml_engine import Job

    engine._clear()
    jobs = [Job(ids, limit) for ids, _ in tasks]
    started = time.monotonic_ns() if arrived_ns is None else arrived_ns
    engine.run(jobs)
    elapsed = (time.monotonic_ns() - started) / 1e9
    if not all(job.reason in ("eos", "length") and job.first_ns > started for job in jobs):
        raise RuntimeError("incomplete response")
    ttfts = [(job.first_ns - started) / 1e6 for job in jobs]
    tokens = [job.tokens for job in jobs]
    answers = [output_text(engine, ids) for ids in tokens]
    expected = [target for _, target in tasks]
    first_words = [re.search(r"\w+", answer.casefold()) for answer in answers]
    correct = [bool(word and word.group() == target) for word, target in zip(first_words, expected)]
    return {"tokens_per_second": sum(map(len, tokens)) / elapsed, "ttft_ms": statistics.median(ttfts),
            "request_ttft_ms": ttfts, "elapsed_s": elapsed, "output_tokens": tokens,
            "answers": answers, "correct": correct, "stop_reasons": [j.reason for j in jobs]}


def arm(width, samples):
    return {"seqs": width, **{key: [s[key] for s in samples] for key in samples[0]}}


def pair(role, length, count, measurements):
    arms = [arm(w, measurements[w]) for w in WIDTHS]
    baseline = arms[0]["output_tokens"][0]
    exact = all(output == baseline for a in arms for output in a["output_tokens"])
    correct = all(all(scores) for a in arms for scores in a["correct"])
    return {"role": role, "prompt_tokens": length, "concurrency": count, "quality_matched": exact and correct,
            "exact_output_match": exact, "all_copy_tasks_correct": correct, "expected": TARGETS[role][:count],
            "arms": arms}


def preferred_width(row):
    narrow, wide = row["arms"]
    if (row["quality_matched"]
            and statistics.median(wide["tokens_per_second"]) >= 1.02 * statistics.median(narrow["tokens_per_second"])
            and statistics.median(wide["ttft_ms"]) <= statistics.median(narrow["ttft_ms"])):
        return wide["seqs"]
    return narrow["seqs"]


def transition(engine, choices, args):
    # Each new wave arrives only when the preceding one completes. Charge all
    # context rebuilds and warmups to the wave that requested the switch.
    pattern = ((32, 1), (args.long_prompt, 4), (32, 1))
    fixtures = [prompts(engine, "replay", length, count) for length, count in pattern]
    settings = dict(engine.settings)
    arms = {"fixed": [], "selected": []}
    for rep in range(2):
        for label in (("fixed", "selected") if rep == 0 else ("selected", "fixed")):
            engine.reconfigure(1, settings)
            waves = []
            for (length, count), tasks in zip(pattern, fixtures):
                width = choices[length, count] if label == "selected" else 1
                previous = engine.seqs
                arrived = time.monotonic_ns()
                if width != previous:
                    engine.reconfigure(width, settings)
                row = sample(engine, tasks, args.tokens, arrived_ns=arrived)
                waves.append({**row, "prompt_tokens": length, "concurrency": count, "seqs": width,
                              "reconfigured": previous != width})
            arms[label].append(waves)
    passed = all(a["output_tokens"] == b["output_tokens"] and all(a["correct"]) and all(b["correct"])
                 for f, s in zip(arms["fixed"], arms["selected"]) for a, b in zip(f, s))
    return {"arms": arms, "quality_matched": passed,
            "switches": sum(w["reconfigured"] for trial in arms["selected"] for w in trial)}


def worker(args, report):
    configure(args.provider)
    from ironmule.ggml_binary import identity
    from ironmule.ggml_engine import GgmlEngine, file_sha256, machine_key
    from ironmule_devices import bindings, ggml_devices, verify_ggml

    devices = verify_ggml(args.provider)
    if args.provider != "cpu" and (len(devices) != 1 or len(ggml_devices()) != 1):
        raise ValueError("select one visible GPU for single-device qualification")
    report.update(devices=devices, machine=machine_key(), native=identity(), model_sha256=file_sha256(str(args.gguf)),
                  model_bytes=args.gguf.stat().st_size, model_name=args.gguf.name)
    L = bindings()
    offloaded = []

    @ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
    def logger(level, message, data):
        if message:
            match = re.search(rb"offloaded (\d+)/(\d+) layers to GPU", message)
            if match:
                offloaded.append([int(n) for n in match.groups()])
            if level >= 3:
                os.write(2, message)

    L.llama_log_set(logger, None)
    start = time.monotonic()
    first = None
    try:
        first = GgmlEngine(str(args.gguf), seqs=1, seq_ctx=report["seq_ctx"], speculate=False, threads=args.threads)
        report["cold_model_context_warmup_s"] = time.monotonic() - start
        report["offloaded_layers"] = offloaded
        if args.provider != "cpu" and not any(n == total and n > 0 for n, total in offloaded):
            raise RuntimeError("model loader did not confirm complete GPU layer offload")
        tasks = prompts(first, "smoke", 32, 1)
        report["first_request"] = sample(first, tasks, args.tokens)
        report["warm_request"] = sample(first, tasks, args.tokens)
        a, b = report["first_request"], report["warm_request"]
        report["smoke_passed"] = a["output_tokens"] == b["output_tokens"] and all(a["correct"]) and all(b["correct"])
        save(args.output, report)
        if args.stage == "compare":
            start = time.monotonic()
            wide = GgmlEngine(str(args.gguf), seqs=4, seq_ctx=report["seq_ctx"], speculate=False,
                              threads=args.threads, model=first.model)
            report["wide_context_warmup_s"] = time.monotonic() - start
            engines = {1: first, 4: wide}
            choices = {}
            try:
                for role in ("calibration", "holdout"):
                    workloads = ((32, 1), (32, 4), (args.long_prompt, 1), (args.long_prompt, 4))
                    for index, (length, count) in enumerate(workloads):
                        tasks = prompts(first, role, length, count)
                        measurements = {w: [] for w in WIDTHS}
                        for rep in range(args.repeats):
                            order = WIDTHS if (index + rep) % 2 == 0 else WIDTHS[::-1]
                            for width in order:
                                measurements[width].append(sample(engines[width], tasks, args.tokens))
                        row = pair(role, length, count, measurements)
                        if role == "calibration":
                            choices[length, count] = preferred_width(row)
                        row["selected_seqs_from_calibration"] = choices[length, count]
                        report["runs"].append(row)
                        save(args.output, report)
                        print(f"{role} {length} tokens, {count} requests: exact={row['exact_output_match']}, "
                              f"correct={row['all_copy_tasks_correct']}", flush=True)
            finally:
                wide.close()
            if args.transition:
                report["transition"] = transition(first, choices, args)
    finally:
        if first:
            first.close()
        L.llama_log_set(None, None)
    report["status"] = "complete"
    save(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--provider", choices=("cpu", "cuda", "rocm"), required=True)
    parser.add_argument("--stage", choices=("smoke", "compare"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--transition", action="store_true",
                        help="after comparison, replay short/burst/short fixed and selected widths including switch time")
    parser.add_argument("--long-prompt", default=256, choices=(256, 512, 1024, 2048), type=int)
    parser.add_argument("--threads", default=2, type=int)
    parser.add_argument("--tokens", default=6, type=int)
    parser.add_argument("--repeats", default=2, type=int)
    parser.add_argument("--budget-s", default=240, type=int)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.gguf.is_file() or args.gguf.resolve() == args.output.resolve():
        parser.error("provide an existing input GGUF and a separate output path")
    if not (1 <= args.threads <= 128 and 1 <= args.tokens <= 32 and 2 <= args.repeats <= 8 and args.budget_s > 0):
        parser.error("threads must be 1..128, tokens 1..32, repeats 2..8 and budget positive")
    if args.transition and args.stage != "compare":
        parser.error("--transition requires --stage compare")
    if args.worker:
        worker(args, json.loads(args.output.read_text()))
        return 0
    sources = ["lab/portable_gpu/bench.py", "ironmule_devices.py", *[f"ironmule/{name}.py" for name in (
        "ggml_engine", "ggml_binary", "pilot", "speculation")]]
    report = {"schema": "ironmule.gpu-width-bench.v1", "status": "running", "stage": args.stage,
              "provider": args.provider, "runs": [], "threads": args.threads, "max_output_tokens": args.tokens,
              "source_sha256": {s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() for s in sources},
              "seq_ctx": max(512, args.long_prompt + 256), "reference_width": 1, "repeats": args.repeats}
    save(args.output, report)
    try:
        child = subprocess.run([sys.executable, __file__, *sys.argv[1:], "--worker"], timeout=args.budget_s, check=False)
        if child.returncode == 0:
            return 0
        failure = f"worker exit {child.returncode}"
    except subprocess.TimeoutExpired:
        failure = f"exceeded {args.budget_s}s wall clock"
    report = json.loads(args.output.read_text())
    report.update(status="incomplete", failure=failure)
    save(args.output, report)
    print(failure, file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
