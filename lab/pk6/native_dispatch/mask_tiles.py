"""Count visible and fully masked tiles on original benchmark inputs in an isolated worker."""

import argparse
import hashlib
import os
from pathlib import Path
from contextlib import closing

from lab.pk6.native_dispatch.kv_span import (
    GgmlEngine,
    GgufTokenizer,
    Job,
    compare,
    digest,
    prompts,
    write,
)
from lab.pk6.mixed_progress.trial import board
from lab.portable_gpu.fetch import MODELS
from ironmule.ggml_engine import file_sha256


def prepare(tree):
    source = tree / "src/llama-kv-cache-unified.cpp"
    text = source.read_text()
    signature = "\n}\n\nvoid llama_kv_cache_unified::set_input_pos_bucket("
    if text.count(signature) != 1 or "IRONMULE_LAB_MASK_TILES" in text:
        raise ValueError("expected a fresh native mask implementation")
    body = Path(__file__).with_suffix(".cpp.inc").read_text()
    source.write_text(text.replace(signature, "\n" + body + signature))


def observe(model, out):
    gguf = file_sha256(str(model))
    hw = board()
    if gguf != MODELS["qwen3-32b"][-1]:
        raise ValueError("wrong weights for the mask replay")
    out.mkdir(parents=True, exist_ok=False)
    waves = []
    capacities = []
    for seed in (180341, 943819):
        with (
            closing(
                GgmlEngine(
                    str(model),
                    threads=4,
                    seqs=32,
                    seq_ctx=2048,
                    shared=True,
                    cap=32,
                    speculate=False,
                    flash_attn=True,
                    kernel_mask=False,
                )
            ) as engine,
            closing(GgufTokenizer(str(model), engine.L)) as tok,
        ):
            capacities.append(
                {
                    "cells": engine.pool.cells,
                    "finished_cell_limit": engine.pool.keep_cells,
                }
            )
            if engine.gpu_layers != 999:
                raise ValueError("expected GPU offload")
            engine.run(
                [
                    Job(
                        tok.apply_chat_template(
                            compare.conversation(compare.WARMUP), enable_thinking=False
                        ),
                        8,
                    )
                ]
            )
            for repeat in range(3):
                for width in (1, 8, 32):
                    ids = prompts(tok, seed, repeat, width)
                    name = f"{seed}-{repeat}-{width}"
                    os.environ["IRONMULE_LAB_MASK_TILES"] = str(out / (name + ".csv"))
                    jobs = engine.run([Job(x, 32) for x in ids])
                    if any(len(j.tokens) != 32 for j in jobs):
                        raise ValueError("unequal work")
                    waves.append(
                        {
                            "name": name,
                            "input_digest": hashlib.sha256(
                                b"".join(bytes.fromhex(digest(x)) for x in ids)
                            ).hexdigest(),
                            "output_digests": [digest(j.tokens) for j in jobs],
                        }
                    )
                    del os.environ["IRONMULE_LAB_MASK_TILES"]
    lib = Path(os.environ["LLAMA_CPP_LIB_PATH"]) / "libllama.so"
    write(
        out / "complete.json",
        {
            "hardware": hw,
            "capacities": capacities,
            "gguf": gguf,
            "waves": waves,
            "libllama_sha256": file_sha256(str(lib)),
            "query_rows": [440, 485],
            "query_tile": 8,
            "key_tile": 256,
        },
    )


if __name__ == "__main__":
    arg = argparse.ArgumentParser(description=__doc__)
    sub = arg.add_subparsers(dest="action", required=True)
    sub.add_parser("prepare").add_argument("tree", type=Path)
    parser = sub.add_parser("observe")
    parser.add_argument("model", type=Path)
    parser.add_argument("out", type=Path)
    args = arg.parse_args()
    if args.action == "prepare":
        prepare(args.tree)
    else:
        observe(args.model, args.out)
