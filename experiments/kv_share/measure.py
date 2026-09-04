"""H2.0 — how much of a decode step is actually KV traffic, on this device?

The master plan claimed FP8-KV "doubles effective decode bandwidth". That
cannot be right by inspection: the 4B snapshot streams gigabytes of weights per
token and Gemma 3 caps most layers with a sliding window. But inspection is not
evidence, so this study measures two things on the running model and states the
ceiling FP8-KV could have:

* **the cache the engine really allocates** — the key and value arrays of every
  layer after a real prefill, their shapes, dtype and byte counts. Note that
  `FixedKVCache` is capacity shaped: a decode step reads the *whole* capacity of
  every layer, not the occupied prefix, so capacity is the honest denominator;
* **the effect that traffic has** — decode rate against context length. If KV
  traffic were a large share of the bytes moved per token, the decode rate would
  fall visibly as the context grows. If it does not, the ceiling is small no
  matter what the arithmetic says.

Both are measurements. The one derived number — KV share of bytes per step — is
computed from measured array sizes and the measured weight files, and is labelled
as the ceiling of an FP8 study, not as its expected gain.

Run: ``python experiments/kv_share/measure.py --execute --model 4b``
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

STUDY_ID = "kv-share-20260902-01"
OUTPUT = Path(__file__).resolve().parent
IRONMULE = ROOT / ".worktrees" / "friday-optimizer-ironmule"

#: The serving default, so the rates are the rates the product would see.
COMBINED = {"head_skip_prefill": True, "compiled_fixed_cache": True, "readback_every": 8}
TOKENS = 32
REPEATS = 3
#: Filler repetitions -> roughly 112, 449, 897 (the sealed prompt) and 1793 tokens.
CONTEXTS = (5, 20, 40, 80)
#: Frozen: below this share of the bytes moved per step, halving the KV cache
#: cannot pay for a quality gate, and the FP8 study is not worth its cost.
SHARE_THRESHOLD = 0.10
#: Frozen: and if the decode rate does not fall by more than this between the
#: smallest and largest context, the traffic is not where the time goes.
RATE_DROP_THRESHOLD = 0.10


def self_check() -> int:
    print(json.dumps({
        "state": "self_check", "study_id": STUDY_ID, "arm": COMBINED,
        "contexts_filler_repeats": list(CONTEXTS), "tokens": TOKENS, "repeats": REPEATS,
        "share_threshold": SHARE_THRESHOLD, "rate_drop_threshold": RATE_DROP_THRESHOLD,
        "measures": ["allocated KV bytes per layer after a real prefill",
                     "decode rate against context length"],
        "formal_claim": False, "no_activation": True,
    }, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--self-check", action="store_true", dest="self_check")
    parser.add_argument("--model", choices=("4b", "1b"), default="4b")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    from _bench import (
        BudgetGuard, enforce_offline, release_gate, require_ac_power,
        resolve_local_model_snapshot, study_provenance,
    )

    gate = release_gate(args, self_check)
    if gate is not None:
        return gate

    from friday_calibrate.runner import MODELS

    require_ac_power()
    guard = BudgetGuard()
    enforce_offline()
    sys.path.insert(0, str(IRONMULE))
    from ironmule import Engine, Knobs  # noqa: E402
    from mlx_lm import load  # noqa: E402

    model_id = MODELS[args.model]
    snapshot = resolve_local_model_snapshot(model_id)
    model, tokenizer = load(str(snapshot.path))
    weight_bytes = sum(
        path.stat().st_size for path in Path(snapshot.path).glob("*.safetensors")
    )
    engine = Engine(model, tokenizer, Knobs(**COMBINED))
    eos_ids = tuple(sorted({
        int(value) for value in (
            getattr(tokenizer, "eos_token_id", None),
            *(getattr(tokenizer, "eos_token_ids", None) or ()),
        ) if isinstance(value, int)
    }))

    def prompt_ids(repeats: int) -> list[int]:
        filler = (
            "You are a careful engineering assistant working in a Python repository. "
            "Follow the existing style and explain your reasoning briefly. "
        ) * repeats
        templated = tokenizer.apply_chat_template(
            [{"role": "user", "content": filler + "\n\n" + "Why is false sharing slow?"}],
            add_generation_prompt=True,
        )
        return list(templated if isinstance(templated, list) else tokenizer.encode(templated))

    debt = 0.0

    def charge(seconds: float) -> None:
        nonlocal debt
        guard.record_gpu(seconds)
        debt += seconds * (1 - 0.15) / 0.15
        while debt >= 4.0:
            guard.required_break()
            debt -= 4.0

    started = time.time()
    rows = []
    layers_report = None
    for repeats in CONTEXTS:
        ids = prompt_ids(repeats)
        capacity = engine._capacity(len(ids), TOKENS)
        at = time.perf_counter()
        engine.generate(ids, TOKENS, eos_ids)  # warmup: compiles this capacity
        charge(time.perf_counter() - at)
        rates = []
        for _ in range(REPEATS):
            at = time.perf_counter()
            out = engine.generate(ids, TOKENS, eos_ids)
            charge(time.perf_counter() - at)
            decode_seconds = out["decode_ns"] / 1e9
            tokens = len(out["logical_tokens"])
            rates.append((tokens - 1) / decode_seconds if tokens > 1 and decode_seconds else 0.0)
            ttft = out["prefill_ns"] / 1e9
        # The cache the engine really allocated, read off the live state.
        at = time.perf_counter()
        state, _ = engine._prefill(ids, capacity)
        charge(time.perf_counter() - at)
        layers = state["layers"]
        per_layer = [
            {"keys": [list(layer["keys"].shape), str(layer["keys"].dtype),
                      layer["keys"].nbytes],
             "values": [list(layer["values"].shape), str(layer["values"].dtype),
                        layer["values"].nbytes]}
            for layer in layers
        ]
        kv_bytes = sum(item["keys"][2] + item["values"][2] for item in per_layer)
        if layers_report is None:
            layers_report = {"layer_count": len(layers), "first_layer": per_layer[0],
                             "distinct_shapes": sorted({str(item["keys"][0]) for item in per_layer})}
        rows.append({
            "filler_repeats": repeats, "prompt_tokens": len(ids), "capacity": capacity,
            "median_decode_tps": statistics.median(rates), "decode_tps": rates,
            "ttft_seconds": ttft,
            "kv_bytes_allocated": kv_bytes,
            "kv_share_of_step_bytes": kv_bytes / (kv_bytes + weight_bytes),
        })

    largest, smallest = rows[-1], rows[0]
    rate_drop = 1.0 - largest["median_decode_tps"] / smallest["median_decode_tps"]
    share = largest["kv_share_of_step_bytes"]
    verdict = (
        "fp8_kv_not_worth_studying"
        if share < SHARE_THRESHOLD and rate_drop < RATE_DROP_THRESHOLD
        else "fp8_kv_ceiling_material"
    )
    report = {
        "study_id": STUDY_ID, "state": "measured", "model": args.model,
        "model_id": model_id, "model_revision": snapshot.revision,
        "arm": COMBINED, "tokens": TOKENS, "repeats": REPEATS,
        "weight_bytes": weight_bytes, "layers": layers_report, "rows": rows,
        "kv_share_at_largest_context": share,
        "decode_rate_drop_smallest_to_largest": rate_drop,
        "share_threshold": SHARE_THRESHOLD, "rate_drop_threshold": RATE_DROP_THRESHOLD,
        "verdict": verdict,
        "ceiling_note": (
            "kv_share is the ceiling an FP8 KV cache could address, halved at best; "
            "it is not a projected gain"
        ),
        "wall_seconds": round(time.time() - started, 1),
        "budget": guard.summary(),
        "provenance": study_provenance(
            [Path(__file__)],
            extra={"model_id": model_id, "model_revision": snapshot.revision},
        ),
        "formal_claim": False, "no_activation": True,
    }
    destination = args.out or OUTPUT / f"kv_share_{args.model}.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in report.items() if k != "provenance"},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
