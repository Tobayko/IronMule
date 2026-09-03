"""Draft-Model Speculative Decoding Engine on Apple Silicon Unified Memory.

Orchestrates Gemma 1B (Draft) proposing tokens for Gemma 12B (Target) with:
1. Exact Greedy Mathematical Equivalence: Target model verifies all logits.
2. Zero-Copy KV Cache Rollback: Unaccepted draft tokens rolled back in O(1).
3. Saturated UMA Bandwidth: Verification runs in a single multi-token forward pass.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

import mlx.core as mx
from ironmule.runtime import Engine, FixedKVCache, Knobs


class SpeculativeDraftEngine:
    """Speculative decoding coordinator between a fast Draft model and a large Target model."""

    def __init__(
        self,
        target_model: Any,
        draft_model: Any,
        tokenizer: Any,
        target_knobs: Knobs | None = None,
        draft_knobs: Knobs | None = None,
        k: int = 3,
    ) -> None:
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.k = max(1, k)
        self.eos_ids = tuple(sorted({int(getattr(tokenizer, "eos_token_id", 1))}))

        t_knobs = target_knobs or Knobs(head_skip_prefill=True, compiled_fixed_cache=False, readback_every=1)
        d_knobs = draft_knobs or Knobs(head_skip_prefill=True, compiled_fixed_cache=False, readback_every=1)

        self.target_engine = Engine(target_model, tokenizer, t_knobs)
        self.draft_engine = Engine(draft_model, tokenizer, d_knobs)

    def _leaves(self, tree: Any) -> list[Any]:
        flat = []
        if isinstance(tree, mx.array):
            return [tree]
        if isinstance(tree, dict):
            for v in tree.values():
                flat.extend(self._leaves(v))
        elif isinstance(tree, (list, tuple)):
            for v in tree:
                flat.extend(self._leaves(v))
        return flat

    def generate(
        self, prompt_ids: Sequence[int], max_tokens: int
    ) -> dict[str, Any]:
        """Execute speculative decoding with exact target verification."""
        p_ids = [int(t) for t in prompt_ids]
        capacity = ((len(p_ids) + max_tokens + self.k + 32 + 63) // 64) * 64

        t0_prefill = time.perf_counter_ns()
        # 1. Prefill both models
        target_state, target_token_arr = self.target_engine._prefill(p_ids, capacity)
        draft_state, _ = self.draft_engine._prefill(p_ids, capacity)

        mx.eval(target_token_arr, *self._leaves(target_state), *self._leaves(draft_state))
        mx.synchronize()
        prefill_ns = time.perf_counter_ns() - t0_prefill

        first_tok = int(target_token_arr.reshape((-1,)).item())
        logical_tokens = [first_tok]

        if first_tok in self.eos_ids or max_tokens <= 1:
            return {
                "logical_tokens": logical_tokens,
                "prefill_ns": prefill_ns,
                "decode_ns": 0,
                "total_ns": prefill_ns,
                "drafted_tokens": 0,
                "accepted_tokens": 0,
                "acceptance_rate": 0.0,
                "steps": 1,
            }

        # Setup bodies
        draft_body = self.draft_engine._body(capacity, 1)
        target_body = self.target_engine._body(capacity, self.k + 1)
        target_single_body = self.target_engine._body(capacity, 1)

        t0_decode = time.perf_counter_ns()
        curr_tok = first_tok
        draft_curr_tok = mx.array([[first_tok]])
        total_drafted = 0
        total_accepted = 0
        spec_steps = 0

        target_offset = len(p_ids) + 1
        draft_offset = len(p_ids) + 1

        while len(logical_tokens) < max_tokens:
            spec_steps += 1
            remaining = max_tokens - len(logical_tokens)
            k_step = min(self.k, remaining)

            # A. DRAFT PHASE: Gemma 1B drafts k tokens
            draft_tokens = []
            d_tok_arr = draft_curr_tok
            for _ in range(k_step):
                d_out = draft_body(d_tok_arr, draft_state)
                d_picks = self.draft_engine._picks(d_out)
                d_tok_arr, draft_state = d_picks[:, -1:], d_out[1]
                mx.eval(d_tok_arr, *self._leaves(draft_state))
                mx.synchronize()
                t_val = int(d_tok_arr.reshape((-1,)).item())
                draft_tokens.append(t_val)
                if t_val in self.eos_ids:
                    break

            total_drafted += len(draft_tokens)

            # B. TARGET VERIFICATION PHASE: Gemma 12B verifies in a single forward pass
            # Verify input: [curr_tok] + draft_tokens
            verify_ids = [curr_tok] + draft_tokens
            verify_width = len(verify_ids)
            v_body = self.target_engine._body(capacity, verify_width)

            v_out = v_body(mx.array([verify_ids]), target_state)
            logits, target_state = v_out[0], v_out[1]
            picks = mx.argmax(logits, axis=-1)
            mx.eval(picks, *self._leaves(target_state))
            mx.synchronize()

            predicted_tokens = picks[0].tolist()

            # C. ACCEPTANCE MATCHING
            # predicted_tokens[0] is target's greedy choice given curr_tok -> must match draft_tokens[0]
            # predicted_tokens[i] is target's greedy choice given draft_tokens[i-1] -> checks draft_tokens[i]
            accepted_chunk = []
            hit_eos = False
            m = 0

            for i in range(len(draft_tokens)):
                target_choice = predicted_tokens[i]
                draft_choice = draft_tokens[i]
                if draft_choice == target_choice:
                    # Accepted!
                    accepted_chunk.append(draft_choice)
                    m += 1
                    total_accepted += 1
                    if draft_choice in self.eos_ids:
                        hit_eos = True
                        break
                else:
                    # Diverged! Target choice is the authoritative token
                    accepted_chunk.append(target_choice)
                    if target_choice in self.eos_ids:
                        hit_eos = True
                    break

            # If all draft tokens matched and no EOS, take the bonus prediction from the end!
            if m == len(draft_tokens) and not hit_eos and len(logical_tokens) + len(accepted_chunk) < max_tokens:
                bonus_token = predicted_tokens[-1]
                accepted_chunk.append(bonus_token)
                total_accepted += 1
                if bonus_token in self.eos_ids:
                    hit_eos = True

            # Rollback target state offset to exact accepted boundary
            actual_advance = len(accepted_chunk)
            target_offset += actual_advance
            target_state["position"]["offset"] = mx.array(target_offset, dtype=mx.int32)

            logical_tokens.extend(accepted_chunk)
            curr_tok = accepted_chunk[-1]
            draft_curr_tok = mx.array([[curr_tok]])

            # Resync draft state offset
            draft_offset = target_offset
            draft_state["position"]["offset"] = mx.array(draft_offset, dtype=mx.int32)

            if hit_eos:
                break

        decode_ns = time.perf_counter_ns() - t0_decode
        acc_rate = (total_accepted / total_drafted) if total_drafted > 0 else 0.0

        return {
            "logical_tokens": logical_tokens[:max_tokens],
            "prefill_ns": prefill_ns,
            "decode_ns": decode_ns,
            "total_ns": prefill_ns + decode_ns,
            "drafted_tokens": total_drafted,
            "accepted_tokens": total_accepted,
            "acceptance_rate": round(acc_rate, 4),
            "steps": spec_steps,
        }
