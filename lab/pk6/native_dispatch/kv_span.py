"""Measure the physical KV span of a mixed step on one L40S with the lab-only shim."""

import argparse
import ctypes
import hashlib
import json
import os
from contextlib import closing
from pathlib import Path
from time import perf_counter

from ironmule.ggml_engine import GgmlEngine, Job, file_sha256
from ironmule.gguf_tokenizer import GgufTokenizer
from ironmule_product import compare
from lab.pk1.native_order import activity
from lab.portable_gpu.fetch import MODELS


def write(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, allow_nan=False, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def prompts(tok, seed, repeat, width):
    group = compare.performance_prompts(seed + repeat * 1009 + width, width, max(width, 8))
    texts = [group[repeat % 2]] if width == 1 else group
    return [tok.apply_chat_template(compare.conversation(text), enable_thinking=False) for text in texts]


def run(args):
    import subprocess

    import torch
    from torch.profiler import ProfilerActivity, profile

    target = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader,nounits"],
        text=True,
    ).strip().split(", ")
    if len(target) != 4 or target[0] != "NVIDIA L40S" or file_sha256(str(args.model)) != MODELS["qwen3-32b"][-1]:
        raise ValueError("wrong target or model")
    if not os.environ.get("IRONMULE_GGML_TRACE") or os.environ.get("IRONMULE_GGML_PROMPT_BUDGET") != "128":
        raise ValueError("expected the native trace and 128-row admission profile")
    bins = Path(os.environ["LLAMA_CPP_LIB_PATH"])
    lib = ctypes.CDLL(str(bins / "libllama.so"))
    lib.ironmule_lab_kv_floor.argtypes = [ctypes.c_uint32]
    lib.ironmule_lab_kv_floor.restype = None
    for fn in (lib.ironmule_lab_kv_natural, lib.ironmule_lab_kv_effective):
        fn.argtypes, fn.restype = [], ctypes.c_uint32
    seeds = (180341, 943819)
    args.out.mkdir(parents=True, exist_ok=False)
    write(args.out / "target.json", {
        "gpu": target[0], "device_sha256": hashlib.sha256(target[1].encode()).hexdigest(),
        "driver": target[2], "memory_mib": int(target[3]), "gguf": MODELS["qwen3-32b"][-1],
        "libraries": {n: file_sha256(str(bins / n)) for n in ("libllama.so", "libggml-cuda.so")},
        "script_sha256": file_sha256(__file__), "torch": torch.__version__, "threads": 4,
        "environment": {k: v for k, v in os.environ.items() if k.startswith("IRONMULE_")
                        and k not in ("IRONMULE_GGML_BINDINGS", "IRONMULE_GGML_TRACE")},
    })
    for seed in seeds:
        with closing(GgmlEngine(str(args.model), threads=4, seqs=32, seq_ctx=2048, shared=True, cap=32,
                                speculate=False, flash_attn=True, kernel_mask=False)) as engine, \
                closing(GgufTokenizer(str(args.model), engine.L)) as tok:
            if engine.gpu_layers != 999 or not engine.settings["flash_attn"] or not engine.settings["kv_unified"]:
                raise ValueError("wrong worker configuration")
            # Natural observation uses the unchanged span and the original request tokens.
            lib.ironmule_lab_kv_floor(0)
            engine.run([Job(tok.apply_chat_template(compare.conversation(compare.WARMUP), enable_thinking=False), 8)])
            decode = engine._decode
            calls = []

            def measured(rows, **kwargs):
                jobs = engine._tl_jobs
                p = sum(s in jobs and pos < len(jobs[s].ids) for _, pos, s, _ in rows)
                start = perf_counter()
                out = decode(rows, **kwargs)
                elapsed = perf_counter() - start
                n = lib.ironmule_lab_kv_natural()
                if not n or lib.ironmule_lab_kv_effective() != n:
                    raise ValueError("unexpected natural span")
                calls.append({"prompt_rows": p, "decode_rows": len(rows) - p, "kv": n, "seconds": elapsed})
                return out

            engine._decode = measured
            for repeat in range(3):
                for width in (1, 8, 32):
                    ids = prompts(tok, seed, repeat, width)
                    fingerprint = hashlib.sha256(b"".join(bytes.fromhex(digest(x)) for x in ids)).hexdigest()
                    jobs = [Job(x, 32, tag=f"{repeat}-{width}-{i}") for i, x in enumerate(ids)]
                    calls.clear()
                    start = perf_counter()
                    engine.run(jobs)
                    seconds = perf_counter() - start
                    write(args.out / f"wave-{seed}-{repeat}-{width}.json", {
                        "seed": seed, "repeat": repeat, "width": width, "input_digest": fingerprint,
                        "seconds": seconds, "output_counts": [len(j.tokens) for j in jobs],
                        "output_digests": [digest(j.tokens) for j in jobs], "calls": calls,
                    })
            engine._decode = decode
            # Use real short prompts and a 444-token document. One active row per short
            # sequence and the uncached tail of the document form the 449-row intervention.
            inputs = prompts(tok, seed, 1, 32)
            short = [x for x in inputs if len(x) == 33]
            doc = next(x for x in inputs if len(x) == 444)
            if len(short) != 16:
                raise ValueError("the benchmark prompt lengths changed")
            prefix = [(t, p, s, p == len(x) - 1) for s, x in enumerate(short) for p, t in enumerate(x)]
            prefix += [(t, p, 16, False) for p, t in enumerate(doc[:11])]
            # The prefix exceeds the 512-row physical batch. Decode all short logits
            # together, then the 11-token document prefix as its own preload.
            short_rows, doc_rows = prefix[:-11], prefix[-11:]
            floors = (0, 8192, 16384, 32768)
            observed = []
            for repeat in range(5):
                # Four counterbalanced timing rounds followed by a separate GPU profile.
                order = floors[repeat % 4:] + floors[:repeat % 4]
                for floor in order:
                    lib.ironmule_lab_kv_floor(0)
                    engine._clear()
                    next_ids = decode(short_rows)
                    decode(doc_rows)
                    rows = [(t, 33, s, True) for s, t in enumerate(next_ids)]
                    rows += [(t, p, 16, p == len(doc) - 1) for p, t in enumerate(doc) if p >= 11]
                    if len(rows) != 449:
                        raise ValueError("expected 433 prompt rows and 16 decode rows")
                    lib.ironmule_lab_kv_floor(floor)
                    torch.cuda.synchronize()
                    if repeat == 4:
                        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                            output = decode(rows)
                            torch.cuda.synchronize()
                        kernels = activity(prof)
                        if not kernels:
                            raise RuntimeError("no CUDA activity")
                        write(args.out / f"cuda-{seed}-{floor}.json", kernels)
                        elapsed = None
                    else:
                        start = perf_counter()
                        output = decode(rows)
                        torch.cuda.synchronize()
                        elapsed = perf_counter() - start
                    natural, effective = lib.ironmule_lab_kv_natural(), lib.ironmule_lab_kv_effective()
                    if not natural or effective != max(natural, floor) or len(output) != 17:
                        raise ValueError("incomplete intervention witness")
                    observed.append({"repeat": repeat, "floor": floor, "natural": natural, "effective": effective,
                                     "seconds": elapsed, "rows_sha256": digest(rows), "result_sha256": digest(output)})
                    write(args.out / f"intervention-{seed}.json", observed)
            lib.ironmule_lab_kv_floor(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("out", type=Path)
    run(parser.parse_args())
