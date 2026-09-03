#!/usr/bin/env python3
"""Investigate Quantized KV-Cache (INT8 Turbo-Cache) on Apple Silicon M1 Max.

Tests whether quantizing Key and Value states to 8-bit inside the KV cache:
1. Preserves 100% Token Identity against full FP16/BF16 baseline.
2. Provides memory bandwidth savings or introduces dequantization compute overhead.
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IRONMULE = PROJECT_ROOT / ".worktrees" / "friday-optimizer-ironmule"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))
sys.path.insert(0, str(IRONMULE))

import mlx.core as mx
from mlx_lm import load
from _bench import enforce_offline, resolve_local_model_snapshot
from ironmule.runtime import Knobs, Engine, FixedKVCache

MODEL_ID = "mlx-community/gemma-3-4b-it-4bit"

TEST_PROMPTS = [
    {
        "name": "QA Concept",
        "prompt": "Explain why Apple Silicon unified memory reduces memory copy overhead in AI workloads.",
        "max_tokens": 32,
    },
    {
        "name": "Math Reasoning",
        "prompt": "Solve this step-by-step: A shop offers 20% discount on $60, then 10% tax. What is final cost?",
        "max_tokens": 32,
    },
    {
        "name": "Coding Sliding Window",
        "prompt": "Write a Python function to compute the moving average of a list using a sliding window.",
        "max_tokens": 32,
    },
]


class QuantizedKVCache:
    """8-bit Symmetric Per-Head Quantized KV Cache."""

    def __init__(self, state: dict[str, Any], position: dict[str, Any], capacity: int):
        self._state, self._position, self._capacity = state, position, capacity

    @property
    def offset(self):
        return self._position["offset"]

    def update_and_fetch(self, keys, values):
        # 1. Compute per-head scales
        # keys shape: (batch, heads, seq, dim)
        k_max = mx.maximum(mx.abs(keys).max(axis=-1, keepdims=True), 1e-4)
        v_max = mx.maximum(mx.abs(values).max(axis=-1, keepdims=True), 1e-4)
        k_scale = k_max / 127.0
        v_scale = v_max / 127.0

        k_quant = mx.clip(mx.round(keys / k_scale), -128, 127).astype(mx.int8)
        v_quant = mx.clip(mx.round(values / v_scale), -128, 127).astype(mx.int8)

        zero = mx.array(0, dtype=self._position["offset"].dtype)
        starts = mx.stack((zero, zero, self._position["offset"], zero))

        self._state["keys"] = mx.slice_update(self._state["keys"], k_quant, start_indices=starts, axes=(0, 1, 2, 3))
        self._state["values"] = mx.slice_update(self._state["values"], v_quant, start_indices=starts, axes=(0, 1, 2, 3))

        # Store scale
        starts_scale = mx.stack((zero, zero, self._position["offset"], zero))
        self._state["k_scale"] = mx.slice_update(self._state["k_scale"], k_scale, start_indices=starts_scale, axes=(0, 1, 2, 3))
        self._state["v_scale"] = mx.slice_update(self._state["v_scale"], v_scale, start_indices=starts_scale, axes=(0, 1, 2, 3))

        # Dequantize cached slice for attention computation
        # (Only up to capacity)
        dequant_keys = self._state["keys"].astype(keys.dtype) * self._state["k_scale"]
        dequant_values = self._state["values"].astype(values.dtype) * self._state["v_scale"]
        return dequant_keys, dequant_values

    def make_mask(self, n_tokens: int, *, window_size: int | None = None, return_array: bool = False):
        del return_array
        dtype = self._position["offset"].dtype
        slots = mx.arange(self._capacity, dtype=dtype)
        queries = self._position["offset"] + mx.arange(n_tokens, dtype=dtype)
        mask = (slots[None, :] <= queries[:, None]) & (slots[None, :] < self._position["offset"] + n_tokens)
        if window_size is not None:
            mask = mask & (slots[None, :] >= queries[:, None] - window_size + 1)
        return mask[None, None, :, :]


def main():
    enforce_offline()
    print("================================================================================")
    print(f"🚀 REAL HARDWARE INVESTIGATION: QUANTIZED KV-CACHE (INT8) ({MODEL_ID})")
    print("================================================================================")

    snapshot = resolve_local_model_snapshot(MODEL_ID)
    model, tokenizer = load(str(snapshot.path))
    eos_ids = tuple(sorted({int(getattr(tokenizer, "eos_token_id", 1))}))

    # 1. Baseline FP16 Engine
    print("--- [ARM 1] Measuring Full-Precision FP16 KV-Cache Baseline ---")
    base_engine = Engine(model, tokenizer, Knobs(head_skip_prefill=True, compiled_fixed_cache=True, readback_every=8))

    baseline_runs = {}
    for task in TEST_PROMPTS:
        p_ids = tokenizer.encode(task["prompt"])
        _ = base_engine.generate(p_ids, 8, eos_ids)
        mx.eval()
        mx.synchronize()

        t0 = time.perf_counter_ns()
        res = base_engine.generate(p_ids, task["max_tokens"], eos_ids)
        mx.eval()
        mx.synchronize()
        wall_ms = (time.perf_counter_ns() - t0) / 1e6
        dec_ms = res["decode_ns"] / 1e6
        cnt = len(res["physical_tokens"])
        tps = cnt / (dec_ms / 1000.0)

        baseline_runs[task["name"]] = {
            "tokens": res["physical_tokens"],
            "wall_ms": wall_ms,
            "tps": tps,
        }
        print(f"  ✓ {task['name']:<22}: Wall={wall_ms:6.1f} ms | TPS={tps:5.1f} tok/s | Tokens: {cnt}")

    # 2. Test INT8 Quantization impact on precision and token identity
    print("\n--- [ARM 2] Testing INT8 Quantized KV-Cache Simulation ---")
    # We test precision degradation of INT8 KV state on model layers
    # We patch Engine to construct state with int8 keys/values and scales
    def make_quant_state(engine, capacity):
        layers = []
        num_layers = len(engine.model.layers)
        num_kv_heads = engine.model.layers[0].self_attn.num_key_value_heads
        head_dim = engine.model.layers[0].self_attn.head_dim
        for _ in range(num_layers):
            layers.append({
                "keys": mx.zeros((1, num_kv_heads, capacity, head_dim), dtype=mx.int8),
                "values": mx.zeros((1, num_kv_heads, capacity, head_dim), dtype=mx.int8),
                "k_scale": mx.ones((1, num_kv_heads, capacity, 1), dtype=mx.bfloat16),
                "v_scale": mx.ones((1, num_kv_heads, capacity, 1), dtype=mx.bfloat16),
            })
        return {
            "position": {"offset": mx.array(0, dtype=mx.int32)},
            "layers": layers,
        }

    # Test numerical delta on single step
    prompt_ids = tokenizer.encode(TEST_PROMPTS[0]["prompt"])
    capacity = 128
    state_fp16, tok_fp16 = base_engine._prefill(prompt_ids, capacity)
    mx.eval(tok_fp16)
    mx.synchronize()

    # Compare first decode step
    body_fp16 = base_engine._body(capacity, 1)
    out_fp16 = body_fp16(tok_fp16, state_fp16)
    mx.eval(out_fp16[0])
    mx.synchronize()
    top_fp16 = mx.argmax(out_fp16[0][:, -1, :]).item()

    # Quantize state_fp16 keys/values to int8
    quant_layers = []
    for layer in state_fp16["layers"]:
        k = layer["keys"]
        v = layer["values"]
        k_scale = mx.maximum(mx.abs(k).max(axis=-1, keepdims=True), 1e-4) / 127.0
        v_scale = mx.maximum(mx.abs(v).max(axis=-1, keepdims=True), 1e-4) / 127.0
        k_q = mx.clip(mx.round(k / k_scale), -128, 127).astype(mx.int8)
        v_q = mx.clip(mx.round(v / v_scale), -128, 127).astype(mx.int8)
        # Dequantized back
        k_deq = k_q.astype(k.dtype) * k_scale
        v_deq = v_q.astype(v.dtype) * v_scale
        quant_layers.append({"keys": k_deq, "values": v_deq})

    state_int8 = {
        "position": state_fp16["position"],
        "layers": quant_layers,
    }

    out_int8 = body_fp16(tok_fp16, state_int8)
    mx.eval(out_int8[0])
    mx.synchronize()
    top_int8 = mx.argmax(out_int8[0][:, -1, :]).item()

    max_logit_diff = mx.abs(out_fp16[0] - out_int8[0]).max().item()

    print(f"FP16 Next Token:        {top_fp16}")
    print(f"INT8 Dequant Next Token:{top_int8}")
    print(f"Max Logit Abs Delta:    {max_logit_diff:.5f}")
    print(f"Token Match at Step 1:  {'MATCH ✅' if top_fp16 == top_int8 else 'DIFF ❌'}")

    # Now let's test auto-regressive generation over 32 steps with INT8 KV dequantization
    print("\nEvaluating 32-token generation under INT8 KV-Cache...")
    sim_tokens = [int(tok_fp16.reshape((-1,)).item())]
    curr_state = state_int8
    curr_tok = tok_fp16

    for step in range(31):
        out = body_fp16(curr_tok, curr_state)
        next_tok = mx.argmax(out[0][:, -1, :], keepdims=True)
        # Re-quantize and dequantize
        re_quant_layers = []
        for l in out[1]["layers"]:
            k = l["keys"]
            v = l["values"]
            k_s = mx.maximum(mx.abs(k).max(axis=-1, keepdims=True), 1e-4) / 127.0
            v_s = mx.maximum(mx.abs(v).max(axis=-1, keepdims=True), 1e-4) / 127.0
            k_q = mx.clip(mx.round(k / k_s), -128, 127).astype(mx.int8)
            v_q = mx.clip(mx.round(v / v_s), -128, 127).astype(mx.int8)
            re_quant_layers.append({"keys": k_q.astype(k.dtype) * k_s, "values": v_q.astype(v.dtype) * v_s})
        curr_state = {"position": out[1]["position"], "layers": re_quant_layers}
        curr_tok = next_tok
        mx.eval(curr_tok)
        mx.synchronize()
        tok_id = int(curr_tok.item())
        sim_tokens.append(tok_id)
        if tok_id in eos_ids:
            break

    base_toks = baseline_runs[TEST_PROMPTS[0]["name"]]["tokens"]
    is_match_32 = (sim_tokens == base_toks)
    print(f"Tokens match across 32 steps: {'MATCH ✅ (100% Identical)' if is_match_32 else 'DIFF ❌ (Diverged)'}")
    if not is_match_32:
        diff_idx = next(i for i, (a, b) in enumerate(zip(sim_tokens, base_toks)) if a != b)
        print(f"Divergence detected at token {diff_idx}: FP16={base_toks[diff_idx]} vs INT8={sim_tokens[diff_idx]}")


if __name__ == "__main__":
    main()
