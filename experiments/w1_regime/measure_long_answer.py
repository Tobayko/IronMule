"""W1 — does the decode rate hold over a realistic answer length?

The whole regime argument (Journal, 2026-09-02) rests on one assumption that
nobody has measured for this model at this prompt length: that decode
throughput stays roughly constant while the KV cache grows. Every optimisation
study in this project generated 32 tokens or fewer, so the assumption has
never been tested where it matters.

One process, one prompt, two generations: a 32-token control that reproduces
the known rate, and a 256-token run. Per-token timings are recorded so the
rate curve is visible, not just its mean.

Gated like every model run: AC power, budget guard, offline snapshot, refuses
to start without --execute.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from regime_analysis import classify  # noqa: E402

STUDY_ID = "w1-regime-20260902-01"
MODEL_ID = "mlx-community/gemma-3-4b-it-4bit"
CONTROL_TOKENS = 32
LONG_TOKENS = 256
#: The prompt of the sealed persistent-process study, so ttft is comparable.
PROMPT_TOKENS = 897
#: Median warm baseline of that study, the number this run checks against.
REFERENCE_TPS = 70.99
OUTPUT = Path(__file__).resolve().parent / "long_answer.json"


def build_prompt(tokenizer) -> list[int]:
    """A prompt of PROMPT_TOKENS tokens, built the same way as elsewhere."""

    unit = ("You are a careful engineering assistant working in a Python repository. "
            "Follow the existing style and explain your reasoning briefly. ")
    body = unit * 40 + "\n\nExplain why false sharing is slow, in detail."
    text = tokenizer.apply_chat_template([{"role": "user", "content": body}],
                                         add_generation_prompt=True)
    ids = list(text if isinstance(text, list) else tokenizer.encode(text))
    return ids


def self_check() -> int:
    report = {
        "study_id": STUDY_ID, "state": "self_check", "model_id": MODEL_ID,
        "control_tokens": CONTROL_TOKENS, "long_tokens": LONG_TOKENS,
        "reference_tps": REFERENCE_TPS, "formal_claim": False,
    }
    print(json.dumps(report, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--self-check", action="store_true", dest="self_check")
    args = parser.parse_args(argv)

    from _bench import (BudgetGuard, release_gate, require_ac_power,
                        resolve_local_model_snapshot, study_provenance)

    gate = release_gate(args, self_check)
    if gate is not None:
        return gate

    require_ac_power()
    guard = BudgetGuard()

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

    def run(count: int) -> dict:
        """One request: chunked prefill at 256, then *count* greedy tokens."""

        cache = make_prompt_cache(model)
        logits = None
        started = time.perf_counter()
        for start in range(0, len(ids), 256):
            logits = model(mx.array([ids[start:start + 256]]), cache=cache)
            mx.eval(logits)
        mx.synchronize()
        ttft = time.perf_counter() - started
        charge(ttft)

        steps: list[float] = []
        tokens: list[int] = []
        decode_started = time.perf_counter()
        for index in range(count):
            if index > 0:
                logits = model(mx.array([[tokens[-1]]]), cache=cache)
            at = time.perf_counter()
            token = int(mx.argmax(logits[:, -1, :], axis=-1)[0])
            mx.eval(logits)
            tokens.append(token)
            steps.append(time.perf_counter() - at)
        mx.synchronize()
        decode_seconds = time.perf_counter() - decode_started
        charge(decode_seconds)
        # The first token arrives with the prefill, so only the rest is decode.
        rate = (count - 1) / decode_seconds if count > 1 and decode_seconds > 0 else 0.0
        quarter = max(1, len(steps) // 4)
        return {
            "tokens_generated": count, "ttft_seconds": ttft,
            "decode_seconds": decode_seconds, "decode_tps": rate,
            "first_quarter_tps": quarter / max(sum(steps[:quarter]), 1e-9),
            "last_quarter_tps": quarter / max(sum(steps[-quarter:]), 1e-9),
            "step_median_ms": statistics.median(steps) * 1e3,
            "token_ids": tokens,
        }

    # A warm-up request so neither measured run pays first-call allocation.
    run(4)
    control = run(CONTROL_TOKENS)
    long_run = run(LONG_TOKENS)
    verdict = classify(ttft=long_run["ttft_seconds"], control_tps=control["decode_tps"],
                       long_tps=long_run["decode_tps"], tokens=LONG_TOKENS)

    result = {
        "study_id": STUDY_ID, "formal_claim": False, "model_id": MODEL_ID,
        "provenance": study_provenance(
            [Path(__file__), Path(__file__).with_name("regime_analysis.py")],
            preregistration=Path(__file__).with_name("PREREGISTRATION.md"),
            extra={"model_snapshot": snapshot.revision, "model_id": MODEL_ID},
        ),
        "snapshot_revision": snapshot.revision, "prompt_tokens": len(ids),
        "reference_tps": REFERENCE_TPS,
        "control": {k: v for k, v in control.items() if k != "token_ids"},
        "long": {k: v for k, v in long_run.items() if k != "token_ids"},
        "prefix_identical": long_run["token_ids"][:CONTROL_TOKENS] == control["token_ids"],
        "verdict": verdict.as_dict(),
        "peak_gb": round(mx.get_peak_memory() / 1e9, 3),
        "budget": {k: v for k, v in guard.summary().items() if "limit" not in k},
    }
    OUTPUT.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ("budget",)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
