"""CS1's internal check: how much perplexity costs keeping only part of the feed-forward neurons per request (BACKLOG CS1).

Per layer the kept neurons are GRIFFIN's: the largest norm, over the prompt's positions, of each position's
unit-normalised feed-forward activation. The prompt (the first half of a gate chunk) is computed in full; the second
half is scored with only the kept neurons, against the full model.

    python tools/cs1_probe.py --snapshot DIR --out cs1.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEEPS = (1.0, 0.9, 0.75, 0.5)
PROMPT = 256


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load

    from ironmule.numeric_choice import CHUNKS, CHUNK_TOKENS, _text_ids, gate_bos, gate_lead, ratio_interval
    from ironmule.numeric_plans import dequantize_model

    mx.set_default_device(mx.cpu)
    model, tokenizer = load(args.snapshot)
    dequantize_model(model)
    state = {"keep": 1.0, "prompt": PROMPT}

    class Kept(nn.Module):
        def __init__(self, mlp):
            super().__init__()
            self.mlp = mlp

        def __call__(self, x):
            z = nn.silu(self.mlp.gate_proj(x)) * self.mlp.up_proj(x)  # (1, L, F)
            if state["keep"] < 1.0:
                cut = state["prompt"]
                head = z[:, :cut]
                unit = head / (mx.linalg.norm(head, axis=-1, keepdims=True) + 1e-8)
                score = mx.linalg.norm(unit, axis=1)  # (1, F)
                keep = max(1, int(round(state["keep"] * score.shape[-1])))
                threshold = mx.sort(score, axis=-1)[:, score.shape[-1] - keep]
                mask = (score >= threshold[:, None]).astype(z.dtype)
                z = mx.concatenate([head, z[:, cut:] * mask[:, None, :]], axis=1)
            return self.mlp.down_proj(z)

    for layer in model.model.layers:
        layer.mlp = Kept(layer.mlp)
    ids, bos = _text_ids(tokenizer), gate_bos(tokenizer)
    rows = {keep: [] for keep in KEEPS}
    for index in range(CHUNKS):
        chunk = ids[index * CHUNK_TOKENS:(index + 1) * CHUNK_TOKENS + 1]
        if len(chunk) < CHUNK_TOKENS + 1:
            break
        lead = gate_lead(chunk, bos)
        for keep in KEEPS:
            state["keep"], state["prompt"] = keep, len(lead) + PROMPT
            logits = model(mx.array([lead + chunk[:-1]]))[0, len(lead) + PROMPT:].astype(mx.float32)
            log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            targets = mx.array(chunk[PROMPT + 1:])
            nll = -mx.mean(log_probs[mx.arange(targets.shape[0]), targets])
            mx.eval(nll)
            rows[keep].append(float(nll.item()))
    out = {"protocol": "cs1-check", "chunks": len(rows[1.0]), "prompt_tokens": PROMPT,
           "perplexity_full": math.exp(sum(rows[1.0]) / len(rows[1.0])),
           "keep": {str(keep): ratio_interval(rows[1.0], rows[keep]) for keep in KEEPS[1:]}}
    if args.out:
        args.out.write_text(json.dumps({**out, "nll": {str(k): v for k, v in rows.items()}}))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
