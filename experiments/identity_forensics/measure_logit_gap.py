"""P2 — is the recorded token-identity divergence a greedy tie?

Ten of eleven recorded divergences sit at generated position 10. This run
records the top-1 minus top-2 logit distance per generated position for the
reference prefill, then reproduces the two chunkings that diverged and asks
whether the flip happens exactly where that distance collapses.

The run does not touch the token-identity gate. A flipped argmax remains an
identity failure; this measurement only decides whether the mechanism or the
workload is responsible, and therefore whether the prefill lever class can be
reopened with a different prompt family.

Gated like every model run: AC power, budget guard, offline snapshot, and it
refuses to start without --execute.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gap_analysis import classify, summarise  # noqa: E402

STUDY_ID = "identity-gap-20260902-01"
MODEL_ID = "mlx-community/gemma-3-4b-it-4bit"
GEN = 16
TARGET_TOKENS = 680
EXPECTED_PROMPT_TOKENS = 677
#: Both diverged at position 10 in experiments/chunk_identity/results.json.
VARIANT_CHUNKS = (128, 512)
OUTPUT = Path(__file__).resolve().parent / "logit_gap.json"


def build_prompt(tokenizer) -> list[int]:
    """Rebuild the exact prompt of the chunk-identity study.

    Copied verbatim from experiments/chunk_identity/measure_chunk_identity.py.
    A different prompt would have a different sensitive position and would
    answer nothing, so the token count is asserted below.
    """

    unit = ("You are a careful engineering assistant working in a Python repository. "
            "Follow the existing style and explain your reasoning briefly. ")
    reps = max(1, TARGET_TOKENS // 22)
    body = unit * reps + "\n\nWhy is false sharing slow?"
    text = tokenizer.apply_chat_template([{"role": "user", "content": body}],
                                         add_generation_prompt=True)
    return list(text if isinstance(text, list) else tokenizer.encode(text))


def self_check() -> int:
    """Offline structure check; never loads a model or touches the device."""

    report = {
        "study_id": STUDY_ID, "state": "self_check", "model_id": MODEL_ID,
        "generate_tokens": GEN, "variant_chunks": list(VARIANT_CHUNKS),
        "expected_prompt_tokens": EXPECTED_PROMPT_TOKENS,
        "gate_unchanged": True, "formal_claim": False,
    }
    print(json.dumps(report, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--self-check", action="store_true", dest="self_check")
    args = parser.parse_args(argv)

    from _bench import (BudgetGuard, enforce_offline, release_gate, require_ac_power,
                        resolve_local_model_snapshot, study_provenance)

    gate = release_gate(args, self_check)
    if gate is not None:
        return gate

    require_ac_power()
    guard = BudgetGuard()
    # Before the model library is imported: the Hugging Face client reads
    # these once at import time, so setting them later changes nothing.
    offline = enforce_offline()

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    debt = 0.0

    def charge(seconds: float) -> None:
        nonlocal debt
        guard.record_gpu(seconds)
        debt += seconds * (1 - 0.15) / 0.15
        while debt >= 4.0:
            guard.required_break()
            debt -= 4.0

    snapshot = resolve_local_model_snapshot(MODEL_ID)
    model, tokenizer = load(str(snapshot.path))
    ids = build_prompt(tokenizer)
    if len(ids) != EXPECTED_PROMPT_TOKENS:
        raise SystemExit(
            f"prompt is {len(ids)} tokens, expected {EXPECTED_PROMPT_TOKENS}; "
            "the sensitive position would not be comparable"
        )

    def run(chunk: int | None) -> tuple[list[int], list[dict]]:
        """Prefill in blocks of *chunk*, then GEN greedy tokens with gaps."""

        cache = make_prompt_cache(model)
        size = len(ids) if chunk is None else chunk
        logits = None
        for start in range(0, len(ids), size):
            piece = ids[start:start + size]
            at = time.perf_counter()
            logits = model(mx.array([piece]), cache=cache)
            mx.eval(logits)
            mx.synchronize()
            charge(time.perf_counter() - at)

        tokens: list[int] = []
        gaps: list[dict] = []
        at = time.perf_counter()
        for position in range(GEN):
            if position > 0:
                logits = model(mx.array([[tokens[-1]]]), cache=cache)
            row = logits[:, -1, :].astype(mx.float32)
            order = mx.argsort(row, axis=-1)
            top1 = int(order[0, -1])
            top2 = int(order[0, -2])
            first = float(row[0, top1])
            second = float(row[0, top2])
            mx.eval(row)
            tokens.append(top1)
            gaps.append({
                "position": position, "top1_id": top1, "top2_id": top2,
                "top1_logit": first, "top2_logit": second, "gap": first - second,
            })
        mx.synchronize()
        charge(time.perf_counter() - at)
        return tokens, gaps

    reference_tokens, reference_gaps = run(None)
    variants = []
    for chunk in VARIANT_CHUNKS:
        tokens, _ = run(chunk)
        first_diff = next(
            (index for index, (left, right) in enumerate(zip(reference_tokens, tokens)) if left != right),
            None,
        )
        verdict = classify(reference_gaps, first_diff)
        variants.append({
            "chunk": chunk, "blocks": -(-len(ids) // chunk),
            "identical": tokens == reference_tokens, "first_diff": first_diff,
            "tokens": tokens, **verdict.as_dict(),
        })
        print(json.dumps({k: v for k, v in variants[-1].items() if k != "tokens"}), flush=True)

    result = {
        "study_id": STUDY_ID, "formal_claim": False, "gate_unchanged": True,
        "provenance": study_provenance(
            [Path(__file__), Path(__file__).with_name("gap_analysis.py")],
            preregistration=Path(__file__).with_name("PREREGISTRATION.md"),
            extra={"model_snapshot": snapshot.revision, "model_id": MODEL_ID,
                   "offline_environment": offline},
        ),
        "model_id": MODEL_ID, "snapshot_revision": snapshot.revision,
        "prompt_tokens": len(ids), "generate_tokens": GEN,
        "reference_tokens": reference_tokens, "reference_gaps": reference_gaps,
        "variants": variants, "summary": summarise(variants),
        "peak_gb": round(mx.get_peak_memory() / 1e9, 3),
        "budget": {k: v for k, v in guard.summary().items() if "limit" not in k},
    }
    OUTPUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
