"""Does the batcher's group width change the tokens a request gets back?

The step-back mode narrows ``ContinuousBatcher``'s decode group while the Mac is
busy. That lever is only allowed if the width does not decide the output, and
this repository has one measured reason to doubt it: at bf16 a forward of width
3 instead of 1 flips a one-ULP argmax (``GEMINI_SELF_LEARNING_SYSTEM`` E02, on
the *sequence* dimension). The batch dimension has never been checked, and the
batcher already varies it today, so the question is open with or without the
throttle.

Method: the same prompt, greedy, decoded once alone and once beside three other
sessions in the same ``mx.eval``. Token ids compared exactly. Real model, real
Metal -- no mocks; a simulation could not answer a numerics question.

Run: ``python experiments/batch_width_identity/measure.py --execute``
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

STUDY_ID = "batch-width-identity-20260904-01"
MODEL_ID = "mlx-community/gemma-3-4b-it-4bit"
MAX_TOKENS = 48

#: One subject prompt plus three companions. The companions only exist to widen
#: the group; different lengths on purpose, because a real batch is ragged.
SUBJECT = "Explain why false sharing slows down multithreaded code."
COMPANIONS = (
    "Summarise what a memory barrier does.",
    "What is the difference between a mutex and a spinlock in one paragraph?",
    "Describe cache line alignment briefly.",
)


def _tokens(batcher, backend, prompt: str, companions: tuple[str, ...]) -> list[int]:
    """Decode ``prompt`` with ``len(companions)`` others in flight; return its ids."""

    subject = batcher.submit(backend.encode(prompt), MAX_TOKENS, {})
    others = [batcher.submit(backend.encode(text), MAX_TOKENS, {}) for text in companions]
    tokens: list[int] = []
    for event in subject.stream(timeout=180.0):
        if event.get("type") == "done":
            tokens = list(event.get("logical_tokens", []))
            break
        if event.get("type") == "error":
            raise RuntimeError(event.get("error", "batcher error"))
    for session in others:
        for event in session.stream(timeout=180.0):
            if event.get("type") in ("done", "error"):
                break
    return tokens


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"state": "not_released", "hint": "pass --execute"}))
        return 78

    from _bench import require_ac_power

    from friday_serve.batcher import ContinuousBatcher
    from friday_serve.ironmule_backend import IronMuleBackend

    require_ac_power()
    backend = IronMuleBackend.load(MODEL_ID)

    results = []
    for repeat in range(args.repeats):
        # Alternate the order so a warm-up drift cannot masquerade as a width
        # effect, the same reason the calibration runs AB/BA.
        widths = (1, 4) if repeat % 2 == 0 else (4, 1)
        by_width: dict[int, list[int]] = {}
        for width in widths:
            batcher = ContinuousBatcher(backend, max_concurrency=8, max_width=width)
            try:
                companions = COMPANIONS if width > 1 else ()
                by_width[width] = _tokens(batcher, backend, SUBJECT, companions)
            finally:
                batcher.stop()
        identical = by_width[1] == by_width[4]
        first_divergence = next(
            (
                index
                for index, (left, right) in enumerate(zip(by_width[1], by_width[4]))
                if left != right
            ),
            None,
        )
        results.append(
            {
                "repeat": repeat,
                "order": list(widths),
                "identical": identical,
                "tokens_width_1": len(by_width[1]),
                "tokens_width_4": len(by_width[4]),
                "first_divergence_index": first_divergence,
            }
        )
        print(json.dumps(results[-1]))

    report = {
        "study_id": STUDY_ID,
        "model_id": MODEL_ID,
        "max_tokens": MAX_TOKENS,
        "repeats": args.repeats,
        "all_identical": all(item["identical"] for item in results),
        "results": results,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "formal_claim": False,
    }
    out = Path(__file__).resolve().parent / "report.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"state": "measured", "all_identical": report["all_identical"], "report": str(out)}))
    return 0 if report["all_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
