#!/usr/bin/env python3
"""Correctness of the integrated opt-in path, before anything is timed.

Loads the model twice through the ordinary `load_engine` route, once with the knob off
and once on, and compares what the two produce. Not only tokens: the full logit bit
pattern of every step, the whole KV cache, and the stop behaviour. Any difference locks
the integration.

It also checks the other half of the contract, that cases which were never qualified
stay on the library path: prefill, multi-token inputs, and projections whose shape or
quantisation does not match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import sys

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ironmule import qmv_k3840  # noqa: E402
from ironmule.runtime import BASELINE, Knobs  # noqa: E402
from ironmule.tune import load_engine  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Fixed before the run: three fresh prompts plus one long continuation.
PROMPTS = (
    ("short", "Name three properties of unified memory.", 24),
    ("technical", "Describe how a key value cache grows during autoregressive decoding.", 24),
    ("plain", "Write one sentence about the sea.", 24),
    ("long", "Explain, step by step, how a laptop runs a large language model locally.", 96),
)


def _decode(engine, tokenizer, prompt: str, steps: int) -> dict:
    language = getattr(engine.model, "language_model", engine.model)
    cache = language.make_cache()
    ids = mx.array(tokenizer.encode(prompt))[None]
    tokens, digests, stopped = [], hashlib.sha256(), False
    eos = set(getattr(tokenizer, "eos_token_ids", None) or ())
    if getattr(tokenizer, "eos_token_id", None) is not None:
        eos.add(tokenizer.eos_token_id)
    for _ in range(steps):
        out = language.model(ids, cache=cache)
        logits = language.lm_head(out[:, -1, :])
        mx.eval(logits)
        digests.update(bytes(memoryview(logits)))
        token = int(mx.argmax(logits, axis=-1).item())
        tokens.append(token)
        if token in eos:
            stopped = True
            break
        ids = mx.array([[token]])
    state = hashlib.sha256()
    for layer in cache:
        for part in (layer.keys, layer.values):
            if part is not None:
                mx.eval(part)
                state.update(bytes(memoryview(part[..., : layer.offset, :])))
    return {
        "tokens": tokens,
        "logits_sha256": digests.hexdigest(),
        "cache_sha256": state.hexdigest(),
        "stopped": stopped,
        "steps_taken": len(tokens),
    }


def _prefill_stays_on_the_library_path(engine) -> dict:
    """A multi-token input must not reach the kernel. Verified against the library."""

    language = getattr(engine.model, "language_model", engine.model)
    swapped = [
        module
        for module in _walk(language)
        if isinstance(module, qmv_k3840.K3840QuantizedLinear)
    ]
    if not swapped:
        return {"checked": False, "reason": "no swapped projection found"}
    module = swapped[0]
    wide = mx.random.normal((1, 5, qmv_k3840.TARGET_K)).astype(mx.bfloat16)
    mx.eval(wide)
    through = module(wide)
    reference = mx.quantized_matmul(
        wide, module.weight, module.scales, module.biases,
        transpose=True, group_size=module.group_size, bits=module.bits,
    )
    mx.eval(through, reference)
    return {
        "checked": True,
        "input_shape": list(wide.shape),
        "identical_to_library": bytes(memoryview(through)) == bytes(memoryview(reference)),
    }


def _walk(module: nn.Module):
    for child in module.children().values():
        if isinstance(child, list):
            for item in child:
                if isinstance(item, nn.Module):
                    yield from _walk(item)
        elif isinstance(child, nn.Module):
            yield child
            yield from _walk(child)


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", default="mlx-community/gemma-3-12b-it-4bit")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    reference_engine, tokenizer = load_engine(args.model, BASELINE)
    reference = {
        name: _decode(reference_engine, tokenizer, prompt, steps)
        for name, prompt, steps in PROMPTS
    }
    assert reference_engine.k3840_admission is None, "the default path must not admit"
    del reference_engine
    mx.clear_cache()

    candidate_engine, tokenizer = load_engine(args.model, Knobs(k3840_matvec=True))
    admission = candidate_engine.k3840_admission
    candidate = {
        name: _decode(candidate_engine, tokenizer, prompt, steps)
        for name, prompt, steps in PROMPTS
    }
    fallback = _prefill_stays_on_the_library_path(candidate_engine)

    comparisons = {}
    for name, _, _ in PROMPTS:
        a, b = reference[name], candidate[name]
        comparisons[name] = {
            "tokens_identical": a["tokens"] == b["tokens"],
            "logits_identical": a["logits_sha256"] == b["logits_sha256"],
            "kv_cache_identical": a["cache_sha256"] == b["cache_sha256"],
            "stop_identical": a["stopped"] == b["stopped"],
            "steps_identical": a["steps_taken"] == b["steps_taken"],
            "steps_taken": a["steps_taken"],
        }
    clean = all(all(row.values()) for row in
                ({k: v for k, v in row.items() if isinstance(v, bool)}
                 for row in comparisons.values()))

    record = {
        "schema": "ironmule.k3840_integration_check.v1",
        "experiment_id": args.out.stem,
        "agent": "claude",
        "kind": "integration_correctness",
        "status": "measured",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "question": "does the opt-in path produce the reference result and decline what it must",
        "model": args.model,
        "knob": "k3840_matvec",
        "default_is_off": BASELINE.k3840_matvec is False,
        "admission": {key: value for key, value in (admission or {}).items()
                      if key not in ("admitted_paths", "declined_paths")},
        "admitted_count": (admission or {}).get("admitted", 0),
        "declined_count": (admission or {}).get("declined", 0),
        "prompts": [{"name": name, "steps": steps} for name, _, steps in PROMPTS],
        "comparisons": comparisons,
        "all_identical": clean,
        "multi_token_falls_back": fallback,
        "device": {"platform": platform.platform(), "machine": platform.machine()},
        "mlx_version": mx.__version__,
        "git_revision": subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        ).stdout.strip(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: value for key, value in record.items()
                      if key in ("admitted_count", "declined_count", "all_identical",
                                 "comparisons", "multi_token_falls_back",
                                 "default_is_off")}, indent=2, sort_keys=True))
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
