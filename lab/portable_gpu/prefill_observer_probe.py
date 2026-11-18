"""Compare live GGUF observer coverage in two checkouts against their real CPU _decode calls.

Set IRONMULE_GGML_BINDINGS and LLAMA_CPP_LIB_PATH as for a normal local GGUF worker.
Run with --before and --after pointing to checkouts and --gguf to the same local model.
The optional --revisions BEFORE AFTER validates each measured engine file against Git.
Per-call elapsed times include host scheduling. The assertions compare coverage and token IDs.
"""

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

SEED = 20261118
WIDTHS = (8, 32)
WAVES = ("short", "long", "short")


def digest(data):
    return hashlib.sha256(json.dumps(data, separators=(",", ":")).encode()).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as stream:
        sha = hashlib.sha256()
        for block in iter(lambda: stream.read(1 << 20), b""):
            sha.update(block)
        return sha.hexdigest()


def run_wave(engine, module, width, wave):
    rng = random.Random(SEED + width * 3 + wave)
    words = ["red", "blue", "silver", "amber", "orange", "black", "yellow", "green", "purple", "white", "violet", "gold"]
    prompts = []
    for i in range(width):
        # A stable per-request prefix makes the third wave exercise real prefix reuse.
        size = 68 if WAVES[wave] == "long" else 12
        tail = " ".join(rng.choices(words, k=size + i % 4))
        prompts.append(engine.tokenize(f"Record {i}, round {wave}: {tail}. Next word:"))
    jobs = [module.Job(prompt, 3) for prompt in prompts]

    real_decode, real_prefill = engine._decode, engine._prefill
    in_prefill = False
    calls = []
    origin = time.perf_counter_ns()

    def prefill(*args, **kwargs):
        nonlocal in_prefill
        in_prefill = True
        try:
            return real_prefill(*args, **kwargs)
        finally:
            in_prefill = False

    def decode(rows, **kwargs):
        positions = [row[1] for row in rows]
        call = {"standalone": in_prefill, "rows": len(rows), "pos": min(positions), "end": max(positions),
                "observations": []}
        calls.append(call)
        start = time.perf_counter_ns()
        call["age_ns"] = start - origin
        result = real_decode(rows, **kwargs)
        call["native_ns"] = time.perf_counter_ns() - start
        return result

    def observe(decode_rows, prompt_rows, kv, seconds):
        calls[-1]["observations"].append((decode_rows, prompt_rows, kv, seconds))

    engine._prefill, engine._decode, engine.observer = prefill, decode, observe
    try:
        engine.run(jobs)
    finally:
        engine._prefill, engine._decode, engine.observer = real_prefill, real_decode, None

    if any(j.reason not in {"length", "eos"} or not j.tokens for j in jobs):
        raise ValueError("one or more requests did not complete normally")
    for call in calls:
        observations = call["observations"]
        if len(observations) > 1:
            raise ValueError("duplicate observation for one native call")
        for d, p, kv, duration in observations:
            if d + p != call["rows"] or duration <= 0:
                raise ValueError("observed row count or time is invalid")
            if call["standalone"] and (d, p, kv) != (0, call["rows"], call["pos"]):
                raise ValueError("standalone prompt rows or prior KV position are misclassified")
    standalone = [c for c in calls if c["standalone"]]
    standalone_observed = [o for c in standalone for o in c["observations"]]
    observed = [o for c in calls for o in c["observations"]]
    if any(not c["observations"] for c in calls if not c["standalone"]):
        raise ValueError("missing main-batch observation")
    return {
        "width": width, "wave": wave + 1, "prompt_profile": WAVES[wave],
        "prompt_tokens_range": [min(map(len, prompts)), max(map(len, prompts))],
        "prompt_sha256": digest(prompts), "token_ids_and_stop_reasons_sha256": digest([(j.tokens, j.reason) for j in jobs]),
        "output_tokens": sum(len(j.tokens) for j in jobs), "stop_reasons": dict(Counter(j.reason for j in jobs)),
        "call_shapes_sha256": digest([(c["standalone"], c["rows"], c["pos"], c["end"]) for c in calls]),
        "native_calls": len(calls), "native_rows": sum(c["rows"] for c in calls),
        "standalone_calls": len(standalone), "standalone_rows": sum(c["rows"] for c in standalone),
        "standalone_kv_before_range": [min(c["pos"] for c in standalone), max(c["pos"] for c in standalone)] if standalone else [],
        "observations": {
            "calls": len(observed), "rows": sum(d + p for d, p, _, _ in observed),
            "standalone_calls": len(standalone_observed), "standalone_prompt_rows": sum(o[1] for o in standalone_observed),
            "standalone_native_ms": round(sum(c["native_ns"] for c in standalone) / 1e6, 3),
            "standalone_reported_ms": round(sum(o[3] for o in standalone_observed) * 1e3, 3),
            "standalone_first_last_start_ms_from_wave": [round(c["age_ns"] / 1e6, 3) for c in (standalone[0], standalone[-1])]
            if standalone else [],
        },
    }


def measure(args):
    sys.path.insert(0, str(args.source))
    from ironmule import ggml_engine

    source = Path(ggml_engine.__file__).resolve()
    if source != args.source / "ironmule/ggml_engine.py":
        raise ValueError("wrong source checkout imported")
    rows = []
    for width in WIDTHS:
        engine = ggml_engine.GgmlEngine(str(args.gguf), threads=2, seqs=width, seq_ctx=128, shared=True,
                                        cap=width, speculate=False)
        try:
            if engine.gpu_layers != 0 or not engine.settings["kv_unified"]:
                raise ValueError("unexpected execution device or cache")
            for wave in range(len(WAVES)):
                rows.append(run_wave(engine, ggml_engine, width, wave))
            lib = Path(engine.L.llama_cpp._lib._name).resolve()
            # Record the compute backend actually mapped alongside the ctypes entry point on Linux.
            maps = Path("/proc/self/maps")
            cpu_dsos = {}
            if maps.is_file():
                for line in maps.read_text().splitlines():
                    fields = line.split(maxsplit=5)
                    if len(fields) == 6 and Path(fields[5]).name.startswith("libggml-cpu.so"):
                        backend = Path(fields[5]).resolve()
                        if backend.name not in cpu_dsos:
                            cpu_dsos[backend.name] = file_sha(backend)
        finally:
            engine.close()
    return {"source_sha256": file_sha(source), "libllama_sha256": file_sha(lib), "mapped_cpu_libraries": cpu_dsos, "waves": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gguf", type=Path, required=True)
    p.add_argument("--before", type=Path)
    p.add_argument("--after", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--revisions", nargs=2, metavar=("BEFORE", "AFTER"))
    p.add_argument("--source", type=lambda s: Path(s).resolve(), help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.source:
        print(json.dumps(measure(args)))
        return
    if not (args.before and args.after and args.output):
        p.error("--before, --after and --output are required")
    env = {k: v for k, v in os.environ.items() if not k.startswith("IRONMULE_") or k == "IRONMULE_GGML_BINDINGS"}
    env.update(IRONMULE_GGML_GPU_LAYERS="0", IRONMULE_GGML_THREADS="2", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
    phases = [json.loads(subprocess.check_output(
        [sys.executable, __file__, "--source", str(tree), "--gguf", str(args.gguf)], env=env,
    )) for tree in (args.before, args.after)]
    before, after = phases
    if any(before[k] != after[k] for k in ("libllama_sha256", "mapped_cpu_libraries")):
        raise ValueError("native library changed across the two runs")
    waves = []
    for old, new in zip(before["waves"], after["waves"], strict=True):
        left, right = old.pop("observations"), new.pop("observations")
        if old != new:
            raise ValueError("output token IDs, stops, request IDs, or native call shapes changed")
        if left["calls"] != old["native_calls"] - old["standalone_calls"] or left["standalone_calls"] != 0:
            raise ValueError("baseline does not exhibit the original omission")
        if (right["calls"], right["rows"], right["standalone_prompt_rows"]) != (
                new["native_calls"], new["native_rows"], new["standalone_rows"]):
            raise ValueError("observer still omits native work")
        waves.append({**old, "before": left, "after": right})
    if any(not any(r["standalone_calls"] for r in waves if r["width"] == w) for w in WIDTHS):
        raise ValueError("workload failed to exercise standalone prefill at each width")
    commits = {}
    if args.revisions:
        repo = Path(__file__).resolve().parents[2]
        for side, phase, rev in zip(("before", "after"), phases, args.revisions, strict=True):
            commit = subprocess.check_output(["git", "-C", repo, "rev-parse", "--verify", f"{rev}^{{commit}}"],
                                             text=True).strip()
            contents = subprocess.check_output(["git", "-C", repo, "show", f"{commit}:ironmule/ggml_engine.py"])
            if hashlib.sha256(contents).hexdigest() != phase["source_sha256"]:
                raise ValueError(f"{side} measured source differs from claimed Git revision")
            commits[side] = commit
    result = {"schema": "ironmule.prefill-observer-proof.v1", "model": args.gguf.name,
              "gguf_sha256": file_sha(args.gguf), "libllama_sha256": before["libllama_sha256"],
              "engine_source_sha256": {side: phase["source_sha256"] for side, phase in zip(("before", "after"), phases)},
              "mapped_cpu_libraries": before["mapped_cpu_libraries"], "source_revisions": commits,
              "protocol": {"seed": SEED, "device": "cpu", "threads": 2, "sequence_context": 128,
                           "shared_kv": True, "speculation": False, "max_output_tokens": 3,
                           "fresh_engine_per_width_and_revision": True, "initialization_excluded": True},
              "waves": waves}
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
