"""PORT2 screening: does IronMule load and engage on a model family it has never seen?

Usage: python families.py MODEL_ID REVISION OUT.json [MAX_TOKENS]

`cross.py` measures wall ratios but cannot say *why* a ratio is 1.00: a mechanism that
silently did not apply looks exactly like a mechanism that applied and did not pay. This
runs before it and records the mechanism state directly — which cache kinds the
architecture asks for, whether projection fusion rewrote the blocks, what the weights and
the peak allocation cost — plus a baseline/tuned token comparison inside one process.

Baseline and tuned are loaded one after the other, not side by side: a 13 GB checkpoint
does not fit twice on a 15 GB T4. Exit code 0 with `"ok": false` is a normal screening
result, not a harness failure; the caller decides whether the model reaches `cross.py`.
"""
import json
import sys
import time

import mlx.core as mx

import ironmule
from ironmule.runtime import _cache_kinds, _new_cache, _trunk
from ironmule.tune import load_engine

MODEL_ID, REVISION, OUT = sys.argv[1:4]
MAX_TOKENS = int(sys.argv[4]) if len(sys.argv) > 4 else 24
PROMPT = "Explain in two sentences why the sky is blue."
# The 1B profile `ironmule tune` confirmed on the T4 in PORT1 (0.7028 [0.6614; 0.7394]).
# Carried over unchanged on purpose: PORT2 asks whether a profile searched on one family
# transfers to another, so searching a fresh one per model would answer a different question.
TUNED = {"compiled_fixed_cache": True, "fused_argmax": True, "head_skip_prefill": True,
         "readback_every": 2, "capacity_slack": 128, "fuse_projections": True}


def stock(report):
    """A greedy reference decode that never touches IronMule's engine.

    Written out rather than called through `mlx_lm.generate` so the plan is visible: one
    prefill of the whole prompt, then argmax, with the model's own cache. If IronMule's
    tokens differ from these, the difference is IronMule's — run 1 left that unattributed
    for the hybrid model, whose output had nothing to do with the prompt.
    """
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    from ironmule.tune import resolve_local_model

    model, tokenizer = load(str(resolve_local_model(MODEL_ID, REVISION).path))
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": PROMPT}],
                                             tokenize=False, add_generation_prompt=True)
    ids = list(tokenizer.encode(rendered, add_special_tokens=False))
    cache = make_prompt_cache(model)
    logits = model(mx.array(ids)[None, :], cache=cache)
    tokens = []
    for _ in range(MAX_TOKENS):
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)
        tokens.append(int(token.item()))
        logits = model(token.reshape(1, 1), cache=cache)
    row = {"knobs": "stock", "tokens": tokens, "text": tokenizer.decode(tokens),
           "cache_kinds": sorted(set(_cache_kinds(cache)))}
    report["arms"].append(row)
    return row


def arm(knobs, report):
    """Load once, record what the mechanisms did, generate, then give the memory back."""
    started = time.time()
    mx.reset_peak_memory()
    engine, tokenizer = load_engine(MODEL_ID, ironmule.Knobs(**knobs), revision=REVISION)
    loaded = time.time() - started
    trunk = _trunk(engine.model)
    block = trunk.layers[0]
    attn, mlp = getattr(block, "self_attn", None), getattr(block, "mlp", None)
    row = {
        "knobs": knobs,
        "load_seconds": round(loaded, 1),
        "n_layers": len(trunk.layers),
        "cache_kinds": sorted(set(_cache_kinds(_new_cache(engine.model)))),
        "fused_qkv": hasattr(attn, "qkv_proj"),
        "fused_gate_up": hasattr(mlp, "gate_up_proj"),
        "weights_bytes": int(mx.get_active_memory()),
    }
    with ironmule.Runtime(engine, tokenizer, model_id=MODEL_ID) as runtime:
        # Run 1 reported first-call latency as a decode rate and got 0.72 tokens/s for an
        # 8B model. That number was compilation and lazy weight paging, not decoding. One
        # warmup pays both off; only the second call is a rate.
        warmup = runtime.generate(PROMPT, max_tokens=MAX_TOKENS)
        began = time.time()
        result = runtime.generate(PROMPT, max_tokens=MAX_TOKENS)
        elapsed = time.time() - began
        row.update({
            "tokens": [int(t) for t in result.tokens],
            "text": result.text,
            "stop_reason": result.stop_reason,
            "metrics": result.metrics,
            "warmup_seconds": round(warmup.metrics.get("total_ms", 0) / 1000, 3)
            if isinstance(warmup.metrics, dict) else None,
            "generate_seconds": round(elapsed, 3),
            "decode_tokens_per_second": round(len(result.tokens) / elapsed, 2) if elapsed else None,
            "warmup_tokens_match": [int(t) for t in warmup.tokens] == [int(t) for t in result.tokens],
            "peak_memory_bytes": int(mx.get_peak_memory()),
        })
    report["arms"].append(row)
    return row


report = {"schema": "ironmule.port2-families.v1", "model_id": MODEL_ID, "revision": REVISION,
          "max_tokens": MAX_TOKENS, "arms": [], "ok": False, "performance_claim": False}
try:
    import mlx_lm
    report["mlx"], report["mlx_lm"] = mx.__version__, mlx_lm.__version__
    reference = stock(report)
    mx.clear_cache()
    base = arm({}, report)
    report["baseline_matches_stock"] = base["tokens"] == reference["tokens"]
    mx.clear_cache()
    try:
        tuned = arm(TUNED, report)
    except Exception as exc:  # noqa: BLE001
        # The retry must happen *after* this block, not inside it. While the exception is
        # being handled its traceback holds `arm`'s frame, and that frame holds the failed
        # engine — so loading the second copy inside the handler keeps both models
        # resident. That is what made gpt-oss run out of memory in run 1 at 11.2 GB on a
        # 15 GB card, and what doubled every reported `weights_bytes`.
        report["tuned_full_error"] = f"{type(exc).__name__}: {exc}"
        tuned = None
    if tuned is None:
        # Fusion is the one knob that rewrites the graph, so it is the one an unfamiliar
        # block layout refuses. Keep the remaining knobs measurable rather than lose the
        # model, and record what was dropped and why.
        mx.clear_cache()
        tuned = arm({**TUNED, "fuse_projections": False}, report)
        report["tuned_profile_reduced"] = True
    report["tokens_identical"] = base["tokens"] == tuned["tokens"]
    report["tuned_matches_stock"] = tuned["tokens"] == reference["tokens"]
    report["tuned_over_baseline_generate"] = (
        round(tuned["generate_seconds"] / base["generate_seconds"], 4)
        if base["generate_seconds"] else None)
    report["ok"] = True
except Exception as exc:  # noqa: BLE001 — a screening failure is a result, not a crash
    report["error"] = f"{type(exc).__name__}: {exc}"
    report["failed_after_arms"] = len(report["arms"])
report["device"] = str(mx.default_device())
with open(OUT, "w") as stream:
    json.dump(report, stream, indent=1, default=str)
print(json.dumps({k: v for k, v in report.items() if k != "arms"}, indent=1, default=str))
for row in report["arms"]:
    print({k: v for k, v in row.items() if k not in {"tokens", "text", "metrics"}})
