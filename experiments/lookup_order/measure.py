"""Is the prompt-lookup sweep measuring speculation, or measuring warm-up?

`tools/measure_prompt_lookup.py:196-201` runs each draft length **once**, in
ascending order, in one process, after a single 8-token warm-up — and it takes
the *first* arm as the baseline. Every later arm therefore runs on a warmer
machine than the reference it is divided by.

W1 established that this matters here: within a single 256-token run the decode
rate rose from `68.34` to `77.23` tok/s, and the project concluded that warm-up,
not context growth, dominates a short run. A monotone-looking sweep is exactly
what that drift produces on its own.

The test is cheap and decisive: run the same sweep **descending**, so the
baseline is measured last and warmest. If speculation is real, the ranking
survives. If the sweep was measuring warm-up, the speedups collapse.

Same tool, same prompt, same model, same guard — only the order changes.

Run: ``python experiments/lookup_order/measure.py --execute``
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

STUDY_ID = "lookup-order-20260902-01"
MODEL_ID = "mlx-community/gemma-3-4b-it-4bit"
DRAFT_LENGTHS = (0, 1, 2, 3, 4)
GENERATE_TOKENS = 96
REPEATS = 3
OUTPUT = Path(__file__).resolve().parent


def self_check() -> int:
    print(json.dumps({
        "state": "self_check", "study_id": STUDY_ID, "model_id": MODEL_ID,
        "draft_lengths": list(DRAFT_LENGTHS), "generate_tokens": GENERATE_TOKENS,
        "repeats": REPEATS,
        "hypothesis": "ascending-order single-shot sweeps confound warm-up with draft length",
        "formal_claim": False,
    }, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--self-check", action="store_true", dest="self_check")
    parser.add_argument("--prompt-file", type=Path,
                        default=ROOT / "experiments" / "prompt_lookup" / "real" / "journal.txt")
    parser.add_argument("--tokens", type=int, default=GENERATE_TOKENS,
                        help="generated tokens; real/results.json used 64, the sweep uses 96")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    from _bench import (BudgetGuard, release_gate, require_ac_power,
                        resolve_local_model_snapshot, study_provenance)

    gate = release_gate(args, self_check)
    if gate is not None:
        return gate

    power = require_ac_power()
    guard = BudgetGuard()

    from mlx_lm import load
    from mlx_lm.sample_utils import make_sampler

    from friday_hardware import speculative_generate

    def account(seconds: float) -> None:
        guard.record_gpu(seconds)
        for _ in range(int(-(-(seconds * (1 - 0.15) / 0.15) // 4))):
            guard.required_break()

    snapshot = resolve_local_model_snapshot(MODEL_ID)
    model, tokenizer = load(str(snapshot.path))
    sampler = make_sampler(temp=0.0)
    prompt = args.prompt_file.read_text(encoding="utf-8")
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True
    )
    ids = list(text if isinstance(text, list) else tokenizer.encode(text))

    def run(draft_length: int, max_tokens: int | None = None):
        max_tokens = args.tokens if max_tokens is None else max_tokens
        result = speculative_generate(
            model, sampler, ids, max_tokens=max_tokens, ngram=3, draft_length=draft_length
        )
        account(result.seconds)
        return result

    # One warm-up, exactly as the original tool does, so the comparison is fair.
    run(0, max_tokens=8)

    orders = {
        "ascending": list(DRAFT_LENGTHS),
        "descending": list(reversed(DRAFT_LENGTHS)),
    }
    seconds: dict[str, dict[int, list[float]]] = {name: {} for name in orders}
    tokens: dict[int, list[int]] = {}
    for repeat in range(REPEATS):
        for name, order in orders.items():
            for draft_length in order:
                result = run(draft_length)
                seconds[name].setdefault(draft_length, []).append(result.seconds)
                tokens.setdefault(draft_length, []).append(tuple(result.tokens))

    identical = all(set(values) == set(tokens[0]) for values in tokens.values())

    report_orders = {}
    for name, per_arm in seconds.items():
        medians = {length: statistics.median(values) for length, values in per_arm.items()}
        base = medians[0]
        report_orders[name] = {
            "median_seconds": {str(k): round(v, 4) for k, v in medians.items()},
            "speedup": {str(k): round(base / v, 4) for k, v in medians.items()},
            "best_draft_length": max(medians, key=lambda k: base / medians[k]),
        }

    # The single-shot reading the original tool would have produced: first
    # repeat, ascending, baseline taken from the first arm.
    first = {length: values[0] for length, values in seconds["ascending"].items()}
    single_shot = {str(k): round(first[0] / v, 4) for k, v in first.items()}

    report = {
        "study_id": STUDY_ID,
        "state": "measured",
        "prompt_file": args.prompt_file.name,
        "prompt_tokens": len(ids),
        "generate_tokens": args.tokens,
        "repeats": REPEATS,
        "all_identical_to_greedy": identical,
        "orders": report_orders,
        "single_shot_ascending_as_the_tool_reports_it": single_shot,
        "order_sensitivity": {
            str(length): round(
                report_orders["ascending"]["speedup"][str(length)]
                - report_orders["descending"]["speedup"][str(length)],
                4,
            )
            for length in DRAFT_LENGTHS
        },
        "power_source": power,
        "budget": guard.summary(),
        "provenance": study_provenance(
            [Path(__file__), ROOT / "friday_hardware" / "speculate.py"],
            extra={"model_id": MODEL_ID, "model_revision": snapshot.revision},
        ),
        "formal_claim": False,
    }
    destination = args.out or OUTPUT.joinpath("order.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in report.items() if k != "provenance"},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
