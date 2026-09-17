"""PORT2: why does projection fusion corrupt llama on CUDA when Qwen 3 and Gemma 3 survive?

Usage: python fused_probe.py MODEL_ID REVISION OUT.json [MAX_TOKENS]

Run 2 found the fused llama path returning different tokens on two calls in one process
(`' interest>plom widthListenerRequestrite …'`) and, under the six-request workload,
`cudaGraphInstantiate … an illegal memory access was encountered`. The same fused bodies
are bit identical on CPU and token identical on Gemma 3 4B and Qwen 3 8B/14B on this card.

The candidate mechanism is what the bodies do *differently*, not what the shapes are:
llama's attention shapes are identical to Qwen 3 8B's. Gemma 3 and Qwen 3 push the split
halves of the fused matmul through `q_norm`/`k_norm` before anything else, which
materialises a fresh array. llama has no QK norm, so its split views reach `rope`,
`update_and_fetch` and the attention kernel as views into one larger buffer. This probe
puts an explicit `mx.contiguous` where the norm would have been and, separately, turns
CUDA graph capture off — one arm per candidate cause, so a pass attributes and a failure
excludes.
"""
import json
import os
import sys
import time

import mlx.core as mx

import ironmule
from ironmule import fast
from ironmule.tune import load_engine, resolve_local_model

MODEL_ID, REVISION, OUT = sys.argv[1:4]
MAX_TOKENS = int(sys.argv[4]) if len(sys.argv) > 4 else 32
PROMPT = "Explain in two sentences why the sky is blue."
TUNED = {"compiled_fixed_cache": True, "fused_argmax": True, "head_skip_prefill": True,
         "readback_every": 2, "capacity_slack": 128}


def _contiguous_attention(self, x, mask=None, cache=None):
    """`_llama_attention` with the split halves copied out of the fused buffer."""
    from mlx_lm.models.base import scaled_dot_product_attention

    B, L, _ = x.shape
    queries, keys, values = mx.split(self.qkv_proj(x), self.qkv_splits, axis=-1)
    queries = mx.contiguous(queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3))
    keys = mx.contiguous(keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3))
    values = mx.contiguous(values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3))

    if cache is not None:
        queries = self.rope(queries, offset=cache.offset)
        keys = self.rope(keys, offset=cache.offset)
        keys, values = cache.update_and_fetch(keys, values)
    else:
        queries = self.rope(queries)
        keys = self.rope(keys)

    output = scaled_dot_product_attention(
        queries, keys, values, cache=cache, scale=self.scale, mask=mask
    )
    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return self.o_proj(output)


def _contiguous_mlp(self, x):
    """`_swiglu_mlp` with the gate and up halves copied out of the fused buffer."""
    from mlx_lm.models.activations import swiglu

    gate, up = mx.split(self.gate_up_proj(x), self.gate_up_split, axis=-1)
    return self.down_proj(swiglu(mx.contiguous(gate), mx.contiguous(up)))


def reference():
    """Greedy decode straight through mlx-lm, no IronMule engine and no fusion."""
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    model, tokenizer = load(str(resolve_local_model(MODEL_ID, REVISION).path))
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": PROMPT}],
                                             tokenize=False, add_generation_prompt=True)
    cache = make_prompt_cache(model)
    logits = model(mx.array(list(tokenizer.encode(rendered, add_special_tokens=False)))[None, :],
                   cache=cache)
    tokens = []
    for _ in range(MAX_TOKENS):
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)
        tokens.append(int(token.item()))
        logits = model(token.reshape(1, 1), cache=cache)
    return tokens, tokenizer.decode(tokens)


def arm(name, *, fuse, contiguous):
    """Two decodes in one process: a single fused arm must agree with itself first."""
    if contiguous:
        fast._ATTENTION_BODIES["mlx_lm.models.llama"] = _contiguous_attention
        fast._MLP_BODIES["mlx_lm.models.llama"] = _contiguous_mlp
    else:
        fast._ATTENTION_BODIES["mlx_lm.models.llama"] = fast._llama_attention
        fast._MLP_BODIES["mlx_lm.models.llama"] = fast._swiglu_mlp
    started = time.time()
    engine, tokenizer = load_engine(MODEL_ID, ironmule.Knobs(**TUNED, fuse_projections=fuse),
                                    revision=REVISION)
    with ironmule.Runtime(engine, tokenizer, model_id=MODEL_ID) as runtime:
        first = runtime.generate(PROMPT, max_tokens=MAX_TOKENS)
        second = runtime.generate(PROMPT, max_tokens=MAX_TOKENS)
    row = {"arm": name, "fuse_projections": fuse, "contiguous_barrier": contiguous,
           "cuda_graphs": os.environ.get("MLX_USE_CUDA_GRAPHS", "on"),
           "tokens": [int(t) for t in first.tokens], "text": first.text,
           "self_consistent": [int(t) for t in first.tokens] == [int(t) for t in second.tokens],
           "seconds": round(time.time() - started, 1)}
    mx.clear_cache()
    return row


report = {"schema": "ironmule.port2-fused-probe.v1", "model_id": MODEL_ID, "revision": REVISION,
          "max_tokens": MAX_TOKENS, "arms": [], "performance_claim": False,
          "cuda_graphs_env": os.environ.get("MLX_USE_CUDA_GRAPHS", "on")}
try:
    tokens, text = reference()
    report["reference"] = {"tokens": tokens, "text": text}
    mx.clear_cache()
    for name, fuse, contiguous in (("unfused", False, False),
                                   ("fused", True, False),
                                   ("fused_contiguous", True, True)):
        try:
            row = arm(name, fuse=fuse, contiguous=contiguous)
        except Exception as exc:  # noqa: BLE001 — a corrupted arm is the result
            row = {"arm": name, "error": f"{type(exc).__name__}: {exc}"}
            mx.clear_cache()
        row["matches_reference"] = row.get("tokens") == tokens
        report["arms"].append(row)
except Exception as exc:  # noqa: BLE001
    report["error"] = f"{type(exc).__name__}: {exc}"
report["device"] = str(mx.default_device())
with open(OUT, "w") as stream:
    json.dump(report, stream, indent=1, default=str)
for row in report["arms"]:
    print({k: v for k, v in row.items() if k != "tokens"})
