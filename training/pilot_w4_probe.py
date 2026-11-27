"""Trusted experimental worker for independent-stream, matched A100 context observations."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import random
import time
from contextlib import closing
from pathlib import Path

CONTEXTS = ("32", "96", "224", "512", "serving_mix")
WIDTHS = (1, 8, 16, 32)


def main():
    # restricted_launch stages these immutable scripts together in the worker's read-only input directory.
    import native_trial as trial

    from ironmule import ggml_engine as ge
    from ironmule.ggml_binary import identity
    from ironmule.kernel_gen import fingerprint
    from ironmule_product.compare import performance_prompts
    from lab.portable_gpu.bench import save

    p = argparse.ArgumentParser()
    for field in ("gguf", "input", "output"):
        p.add_argument("--" + field, type=Path, required=True)
    a = p.parse_args()
    spec = json.loads(a.input.read_text())
    lib = Path(os.environ["IRONMULE_KG_LIB"])
    if (trial.checksum(lib) != spec["reference"] or trial.checksum(os.environ["IRONMULE_W4_TABLE"]) != spec["table"]
            or trial.checksum(a.gguf) != spec["gguf"] or identity() != spec["native"]
            or fingerprint()["devices"] != spec["devices"]
            or any(os.environ.get("IRONMULE_" + k) != v for k, v in trial.FLAGS.items())):
        raise ValueError("probe target changed")
    profile = spec["profile"]
    plugin = ctypes.CDLL(str(lib))
    getters = []
    for suffix, bits in zip(("", "6"), profile["mask_bits"], strict=True):
        setter = getattr(plugin, "ironmule_kg_set_mask" + suffix)
        setter.argtypes = [ctypes.c_uint32]
        setter(bits)
        getter = getattr(plugin, "ironmule_kg_get_mask" + suffix)
        getter.restype = ctypes.c_uint32
        getters.append(getter)
    cls = trial.fixture_engine(ge, profile)
    cells = []
    with closing(cls(str(a.gguf), seqs=32, seq_ctx=2048, flash_attn=True, speculate=True, kernel_mask=False)) as e:
        if e.gpu_layers != 999 or e.pilot is None or [g() for g in getters] != profile["mask_bits"]:
            raise RuntimeError("probe did not bind the local GPU configuration")
        tokens = e.tokenize(" ".join(f"Site {x} recorded {x % 91} units." for x in range(1000)))
        regimes = [(c, n) for c in CONTEXTS for n in WIDTHS]
        random.Random(7151 + spec["round"]).shuffle(regimes)
        for context, width in regimes:
            e._clear()
            if context == "serving_mix":
                prompts = [e.tokenize(trial.TEMPLATE.format(p)) for p in performance_prompts(92741, width, width)]
            else:
                prompts = [[tokens[(311 * s + j) % len(tokens)] for j in range(32 if s % 2 == 0 else int(context))]
                           for s in range(width)]
            lengths = [len(p) for p in prompts]
            rows = [(t, j, s, j + 1 == len(p)) for s, p in enumerate(prompts) for j, t in enumerate(p)]
            first = []
            for offset in range(0, len(rows), e.batch_cap):
                first.extend(e._decode(rows[offset:offset + e.batch_cap]))
            if len(first) != width:
                raise ValueError("missing prefill logits")
            # One warm step then eight actual native steps at one row per independent sequence.
            next_tokens = e._decode([(token, lengths[s], s, True) for s, token in enumerate(first)])
            warm = list(next_tokens)
            predicted = []
            start = time.perf_counter_ns()
            for offset in range(1, 9):
                next_tokens = e._decode([(tok, lengths[s] + offset, s, True) for s, tok in enumerate(next_tokens)])
                if len(next_tokens) != width:
                    raise ValueError("incomplete decode")
                predicted.append(list(next_tokens))
            seconds = (time.perf_counter_ns() - start) / 1e9
            cells.append({"context": context, "width": width, "prompt_lengths": lengths, "warm_tokens": [first, warm],
                          "steps": 8, "seconds": seconds, "tokens": predicted})
    if [g() for g in getters] != profile["mask_bits"]:
        raise ValueError("native masks changed")
    save(a.output, {"schema": "ironmule.w4-context-probe.v1", "source_sha256": trial.checksum(__file__),
                    "spec": spec, "cells": cells})


if __name__ == "__main__":
    main()
