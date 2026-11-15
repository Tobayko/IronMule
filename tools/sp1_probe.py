"""SP1-0: speculation from the context on the CPU plan, exactness and gain for one request (BACKLOG SP1-0).

Drafts are copied from the request's own prompt and answer (prompt lookup: the longest suffix of 3 down to 1
tokens that occurred before, its most recent occurrence, the tokens after it). The last accepted token and `k`
drafts are verified as `k + 1` rows of one slot at consecutive positions in one step of the CPU stacked plan,
whose products are padded to a fixed width (rows invariant, CPU6) and whose attention runs per row after the step's
keys are written, so each row sees exactly the positions before it. The longest prefix of drafts equal to the
plan's argmax is accepted, plus the argmax after it. Each prompt is decoded sequentially, with k = 2, 4, 7, and
sequentially again; `--analyse REPORT` applies the rule fixed in the entry before the data.

    python tools/sp1_probe.py --dolly DIR --out sp1.json
    python -I tools/sp1_probe.py --analyse sp1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
MODEL, REVISION = "mlx-community/SmolLM2-135M-Instruct-8bit", "0f0d9b8218915bc34d401e1a340b8c049d300d5e"
KS = (2, 4, 7)
ARMS = ("seq", "k2", "k4", "k7", "seq2")
PROMPTS, FIRST_PROMPT, LIMIT, MAX_NGRAM = 48, 12_000, 128, 3
RESAMPLES, GAIN_BOUND, PER_REQUEST_BOUND = 2000, 1.3, 1.05


def draft(context: Sequence[int], k: int, max_ngram: int = MAX_NGRAM) -> list[int]:
    """Up to `k` tokens after the most recent earlier occurrence of the context's longest repeated suffix."""
    if k < 1:
        return []
    n_ctx = len(context)
    for n in range(min(max_ngram, n_ctx - 1), 0, -1):
        suffix = list(context[n_ctx - n:])
        for start in range(n_ctx - n - 1, -1, -1):
            if list(context[start:start + n]) == suffix:
                return list(context[start + n:start + n + k])
    return []


def accept(drafts: Sequence[int], argmax: Sequence[int]) -> list[int]:
    """The drafts the model agrees with, then its own next token."""
    n = 0
    while n < len(drafts) and drafts[n] == argmax[n]:
        n += 1
    return list(drafts[:n]) + [int(argmax[n])]


def spec_plan_class():
    """The CPU plan with a verify window: rows may share a slot at distinct positions (the plan itself refuses
    that today, so this is the probe's own copy of its forward with only that check relaxed)."""
    import mlx.core as mx
    import mlx.nn as nn

    from ironmule.dense_cpu import CpuDensePlan

    class SpecPlan(CpuDensePlan):
        def forward(self, tokens: Any, slots: Any, positions: Any) -> Any:
            rows = tokens.shape[0]
            host_slots, host_positions = slots.tolist(), positions.tolist()
            if len(set(zip(host_slots, host_positions))) != rows:
                raise ValueError("each stacked row needs its own slot position")
            if max(host_positions) >= self.capacity:
                raise ValueError("a row's position is past its slot")
            h = self.trunk.embed_tokens(tokens.astype(mx.uint32)).astype(mx.float32)
            written = self._rows(slots, positions)
            for i, layer in enumerate(self.trunk.layers):
                attn, mlp = layer.self_attn, layer.mlp
                x = mx.fast.rms_norm(h, layer.input_layernorm.weight, self.eps)
                q = self._proj(x, attn.q_proj).reshape(rows, self.heads, self.dim)
                k = self._proj(x, attn.k_proj).reshape(rows, self.kv_heads, self.dim)
                v = self._proj(x, attn.v_proj).reshape(rows, self.kv_heads, self.dim)
                if self.qk_norm:
                    q, k = attn.q_norm(q), attn.k_norm(k)
                q, k = self._rope(attn, q, positions), self._rope(attn, k, positions)
                self.keys[i][written] = k.reshape(-1, self.dim)
                self.values[i][written] = v.reshape(-1, self.dim)
                o = mx.stack([self._attend(q[r], i, host_slots[r], host_positions[r] + 1, attn.scale)
                              for r in range(rows)])
                h = h + self._proj(o.reshape(rows, self.heads * self.dim), attn.o_proj)
                x = mx.fast.rms_norm(h, layer.post_attention_layernorm.weight, self.eps)
                h = h + self._proj(nn.silu(self._proj(x, mlp.gate_proj)) * self._proj(x, mlp.up_proj), mlp.down_proj)
            h = mx.fast.rms_norm(h, self.trunk.norm.weight, self.eps)
            return self.linear(h, self.head)

    return SpecPlan


def prefill(plan: Any, ids: list[int]) -> int:
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(plan.model)
    logits = plan.prefill(mx.array(ids)[None], cache)
    mx.eval(logits, [c.state for c in cache])
    plan.write_prefill(0, [c.state for c in cache])
    return int(mx.argmax(logits[0]).item())


def decode(plan: Any, ids: list[int], first: int, eos: set[int], limit: int, k: int = 0) -> dict[str, Any]:
    """One answer from its first token: one row per step (k = 0), or verify windows of up to k drafts."""
    import mlx.core as mx

    out, drafted, accepted, steps = [first], 0, 0, 0
    started = time.perf_counter()
    while len(out) < limit and out[-1] not in eos:
        drafts = draft(ids + out, min(k, limit - len(out) - 1)) if k else []
        rows = [out[-1]] + drafts
        position = len(ids) + len(out) - 1
        logits = plan.forward(mx.array(rows, mx.uint32), mx.zeros((len(rows),), mx.int32),
                              mx.arange(position, position + len(rows), dtype=mx.int32))
        new = accept(drafts, mx.argmax(logits, axis=-1).tolist())
        steps, drafted, accepted = steps + 1, drafted + len(drafts), accepted + len(new) - 1
        for token in new:
            out.append(token)
            if token in eos:
                break
    seconds = time.perf_counter() - started
    return {"tokens": out, "decode_s": seconds, "steps": steps, "drafted": drafted, "accepted": accepted,
            "tokens_per_s": (len(out) - 1) / seconds if seconds > 0 and len(out) > 1 else None}


def _interval(values: list[float], seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(RESAMPLES))
    return [sum(values) / len(values), means[int(0.025 * RESAMPLES)], means[int(0.975 * RESAMPLES) - 1]]


def analyse(report: dict[str, Any]) -> dict[str, Any]:
    """The rule fixed in BACKLOG SP1-0 before the data."""
    rows = [r for r in report["prompts"] if all(r["arms"][a]["tokens_per_s"] for a in ARMS)]
    differing = [{"prompt": r["index"], "arm": a,
                  "position": next((j for j, (x, y) in enumerate(zip(r["arms"][a]["tokens"], r["arms"]["seq"]["tokens"]))
                                    if x != y), min(len(r["arms"][a]["tokens"]), len(r["arms"]["seq"]["tokens"])))}
                 for r in report["prompts"] for a in ARMS[1:] if r["arms"][a]["tokens"] != r["arms"]["seq"]["tokens"]]
    out: dict[str, Any] = {"prompts": len(report["prompts"]), "timed": len(rows), "exact": not differing,
                           "differing": differing[:10]}
    if not rows:
        return {**out, "decision": "invalid: no prompt with a timed answer"}
    base = {r["index"]: (r["arms"]["seq"]["tokens_per_s"] + r["arms"]["seq2"]["tokens_per_s"]) / 2 for r in rows}
    out["aa_noise"] = sum(abs(r["arms"]["seq2"]["tokens_per_s"] / r["arms"]["seq"]["tokens_per_s"] - 1)
                          for r in rows) / len(rows)
    gains = {}
    for k in KS:
        arm = f"k{k}"
        ratios = [r["arms"][arm]["tokens_per_s"] / base[r["index"]] for r in rows]
        drafted = sum(r["arms"][arm]["drafted"] for r in rows)
        per_prompt = [r["arms"][arm]["accepted"] / r["arms"][arm]["drafted"] for r in rows if r["arms"][arm]["drafted"]]
        gains[arm] = {"ratio": _interval(ratios), "acceptance": (sum(r["arms"][arm]["accepted"] for r in rows) / drafted
                                                                 if drafted else None),
                      "acceptance_spread": [min(per_prompt), max(per_prompt)] if per_prompt else None}
    out["gains"] = gains
    best = max(gains, key=lambda arm: gains[arm]["ratio"][0])
    out["best_k"] = best
    per_request = [max([1.0] + [r["arms"][f"k{k}"]["tokens_per_s"] / base[r["index"]] for k in KS])
                   / (r["arms"][best]["tokens_per_s"] / base[r["index"]]) for r in rows]
    out["per_request_over_best_k"] = _interval(per_request)
    if not out["exact"]:
        decision = "not exact: speculation does not ship on this plan (Tier 0, scoped to the CPU plan)"
    elif gains[best]["ratio"][1] >= GAIN_BOUND:
        decision = f"go: {best} at {gains[best]['ratio'][0]:.2f}x (lower bound {gains[best]['ratio'][1]:.2f}); SP1 to the product"
    else:
        decision = f"kill: best {best} lower bound {gains[best]['ratio'][1]:.2f} < {GAIN_BOUND} (Tier 0, this plan and model)"
    out["decision"] = decision
    out["window_learned_per_request"] = out["per_request_over_best_k"][1] > PER_REQUEST_BOUND
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dolly", type=Path, help="a directory holding (or to receive) dolly.jsonl")
    parser.add_argument("--prompts", type=int, default=PROMPTS)
    parser.add_argument("--limit", type=int, default=LIMIT)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--analyse", type=Path)
    args = parser.parse_args(argv)
    if args.analyse:
        print(json.dumps(analyse(json.loads(args.analyse.read_text())), indent=1))
        return 0
    sys.path.insert(0, str(ROOT))
    from ironmule import cpu_blas

    if "IRONMULE_CPU_BLAS_LIBRARY" not in os.environ:  # the shim must be preloaded at process start
        variables, record = cpu_blas.environment()
        if not variables:
            raise SystemExit(f"SP1-0 runs on OpenBLAS; the shim is unavailable: {record.get('reason')}")
        os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])],
                  {**os.environ, **variables, "HF_HUB_OFFLINE": "1"})
    if cpu_blas.active().get("serving") != "openblas":
        raise SystemExit("SP1-0: the shim was preloaded but OpenBLAS does not serve")
    sys.path.insert(0, str(ROOT / "training"))
    import mlx.core as mx
    from mlx_lm import load

    import rl2_headroom
    from ironmule.dense_cpu import to_float32
    from ironmule.tune import _eos_ids

    texts, digest = rl2_headroom.dolly_prompts(args.dolly)
    snapshot = Path(os.path.expanduser(f"~/.cache/huggingface/hub/models--{MODEL.replace('/', '--')}/snapshots/{REVISION}"))
    model, tokenizer = load(str(snapshot))
    to_float32(model)
    plan = spec_plan_class()(model, slots=8, capacity=2048)
    eos = set(_eos_ids(tokenizer))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    report: dict[str, Any] = {"protocol": "sp1-0", "model": MODEL, "revision": REVISION, "commit": commit,
                              "dolly_sha256": digest, "limit": args.limit, "ks": list(KS), "arms": list(ARMS),
                              "blas": cpu_blas.active(),
                              "machine": {"machine": platform.machine(), "cpus": os.cpu_count(), "mlx": mx.__version__},
                              "prompts": []}

    def ids_of(text: str) -> list[int]:
        return list(tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                                  tokenize=True))

    warm = ids_of(texts[FIRST_PROMPT - 1])  # not recorded
    for k in (0, *KS):
        decode(plan, warm, prefill(plan, warm), eos, 16, k)
    for index in range(FIRST_PROMPT, FIRST_PROMPT + args.prompts):
        ids = ids_of(texts[index])
        if len(ids) + args.limit + 8 > plan.capacity:
            continue
        arms = {}
        for arm in ARMS:
            k = int(arm[1:]) if arm.startswith("k") else 0
            arms[arm] = decode(plan, ids, prefill(plan, ids), eos, args.limit, k)
        report["prompts"].append({"index": index, "prompt_tokens": len(ids),
                                  "text_sha256": hashlib.sha256(texts[index].encode()).hexdigest()[:16], "arms": arms})
        line = {a: (round(arms[a]["tokens_per_s"] or 0, 1), arms[a]["accepted"]) for a in ARMS}
        print(json.dumps({"prompt": index, "tokens": len(arms["seq"]["tokens"]), "arms": line,
                          "exact": all(arms[a]["tokens"] == arms["seq"]["tokens"] for a in ARMS)}), flush=True)
        if args.out:
            args.out.write_text(json.dumps(report))
    report["analysis"] = analyse(report)
    if args.out:
        args.out.write_text(json.dumps(report))
    print(json.dumps(report["analysis"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
