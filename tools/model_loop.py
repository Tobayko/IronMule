#!/usr/bin/env python3
"""H2: a local language model proposes execution plans, the harness judges them.

The difference to ``optimization_loop``: there, the search space and the
refinement rule are mine.  Here a model reads the measurements taken so far and
proposes what to try next.  It sees real device facts -- the ~340 ms disturbance
timescale, the cooldown penalty, the paired-vs-unpaired spread -- and answers
with candidates.  The harness then measures them the same way it measures
anything else, and rejects whatever fails to clear the frozen threshold.

**The model proposes parameters, never code.** Executing model-generated code on
the GPU is a separate safety problem requiring sandboxing; it is deliberately not
part of this tool.  Every proposal is parsed as a plain integer batch size and
discarded unless it falls inside the registered range.  A model that answers with
prose, with a shell command, or with 900 gets nothing executed.

The loop runs several rounds and feeds each round's measurements back, so the
model can react to what its own suggestions produced.

Run with --execute; without it nothing is imported or measured.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bench import (  # noqa: E402
    BudgetGuard,
    release_gate,
    require_ac_power,
    resolve_local_model_snapshot,
    run_persisted,
)

import importlib.util  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "optimization_loop", Path(__file__).resolve().parent / "optimization_loop.py"
)
_LOOP = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_LOOP)

MODEL_ID = "mlx-community/gemma-3-4b-it-4bit"
MIN_BATCH = 2
MAX_BATCH = 16
MAX_PROPOSALS = 3
MAX_MODEL_TOKENS = 80
DEFAULT_ROUNDS = 3
MDE = _LOOP.MDE
EXPLORE_BLOCKS = 20
CONFIRM_BLOCKS = 25
CONFIRM_REPLICATES = 3
BOOTSTRAP_SEED = 0x20092_2026

# Measured on this device; handed to the model as context rather than hidden.
DEVICE_FACTS = """Device facts measured on this machine:
  - a disturbance process with ~340 ms timescale affects all measurements
  - the first operation after an idle pause is up to 4x slower
  - between-run spread is ~20% unpaired but ~1.3% paired"""


def parse_proposals(text: str, *, already_tried: set[int]) -> list[int]:
    """Extract batch sizes from a model answer, discarding anything unusable.

    Deliberately strict and total: the model is an untrusted source of integers.
    Prose, shell commands, floats and out-of-range values all reduce to an empty
    list rather than to an error, so a bad answer costs a round and nothing else.
    """

    match = re.search(r"\[[^\]]*\]", text)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
    except (ValueError, TypeError):
        return []
    if not isinstance(parsed, list):
        return []
    accepted: list[int] = []
    for value in parsed:
        # bool is an int subclass; True must not become batch size 1.
        if type(value) is not int:
            continue
        if not MIN_BATCH <= value <= MAX_BATCH:
            continue
        if value in already_tried or value in accepted:
            continue
        accepted.append(value)
    return accepted[:MAX_PROPOSALS]


def build_prompt(evidence: list[dict], already_tried: set[int]) -> str:
    """Show the model what has been measured, then ask for untested candidates."""

    lines = [
        "You tune GPU execution plans on an Apple M1 Max (32-core GPU, 32 GB unified memory).",
        "",
        "Workload: 2048x2048 FP16 matrix multiplications, run N at a time.",
        "Two execution plans exist:",
        "  serial  - dispatch one matmul, wait for the GPU, repeat N times",
        "  batched - dispatch all N matmuls, wait once at the end",
        "",
        "Measured ratios (batched/serial, lower is better):",
    ]
    if evidence:
        for entry in sorted(evidence, key=lambda e: e["batch_size"]):
            verdict = "beats threshold" if entry["clears_mde"] else "not beyond noise"
            lines.append(f"  N={entry['batch_size']:<3} -> {entry['ratio']:.3f}  ({verdict})")
    else:
        lines.append("  (nothing measured yet)")
    lines += [
        "",
        DEVICE_FACTS,
        "",
        f"Already measured: {sorted(already_tried) if already_tried else 'none'}",
        f"Propose the {MAX_PROPOSALS} most promising UNTESTED values of N "
        f"(integers {MIN_BATCH}..{MAX_BATCH}) to measure next.",
        "Answer with ONLY a JSON array of integers, nothing else. Example: [3, 5, 7]",
    ]
    return "\n".join(lines)


def _self_check() -> int:
    """Offline checks of the parsing rules; no GPU, no model."""

    tried = {2, 4}
    assert parse_proposals("[3, 5, 7]", already_tried=tried) == [3, 5, 7]
    assert parse_proposals("```json\n[3, 5]\n```", already_tried=tried) == [3, 5]
    # Already-measured values are dropped, not re-run.
    assert parse_proposals("[2, 4, 6]", already_tried=tried) == [6]
    # Out-of-range, floats, bools and strings are all discarded.
    assert parse_proposals("[900, 0, -3]", already_tried=tried) == []
    assert parse_proposals("[3.5, 5]", already_tried=tried) == [5]
    assert parse_proposals("[true, 5]", already_tried=tried) == [5]
    assert parse_proposals('["7", 5]', already_tried=tried) == [5]
    # A model that answers with prose or a command gets nothing executed.
    assert parse_proposals("I suggest trying 5 and 7.", already_tried=tried) == []
    assert parse_proposals("[$(rm -rf /)]", already_tried=tried) == []
    assert parse_proposals("", already_tried=tried) == []
    # Duplicates collapse, and no more than MAX_PROPOSALS survive.
    assert parse_proposals("[5, 5, 6, 7, 9]", already_tried=tried) == [5, 6, 7]

    prompt = build_prompt([{"batch_size": 4, "ratio": 0.83, "clears_mde": True}], {4})
    assert "N=4" in prompt and "0.830" in prompt
    assert "[3, 5, 7]" in prompt  # the example format
    print(json.dumps({"self_check": "pass", "checks": 13}))
    return 0


def run(rounds: int) -> dict[str, object]:
    guard = BudgetGuard()
    from mlx_lm import generate, load

    snapshot = resolve_local_model_snapshot(MODEL_ID)
    model, tokenizer = load(str(snapshot.path))
    harness = _LOOP.Harness(guard)
    evidence: list[dict] = []
    tried: set[int] = set()
    history: list[dict] = []

    def measure(batch_size: int) -> dict:
        guard.before_candidate()
        try:
            if not harness.correctness_holds(batch_size):
                return {"batch_size": batch_size, "rejected": "correctness",
                        "clears_mde": False, "ratio": 1.0, "ci_high": 1.0,
                        "log_ratios": []}
            summary = _LOOP.summarize(harness.measure(batch_size, EXPLORE_BLOCKS))
            summary["batch_size"] = batch_size
            summary["rejected"] = None
            return summary
        finally:
            guard.finish_candidate()

    for round_index in range(rounds):
        guard.required_break()
        prompt = build_prompt(evidence, tried)
        gpu_started = time.perf_counter()
        answer = generate(
            model,
            tokenizer,
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True
            ),
            max_tokens=MAX_MODEL_TOKENS,
            verbose=False,
        )
        guard.record_gpu(time.perf_counter() - gpu_started)
        guard.required_break()
        proposals = parse_proposals(answer, already_tried=tried)
        measured = [measure(size) for size in proposals]
        tried.update(proposals)
        evidence.extend(measured)
        history.append(
            {
                "round": round_index,
                "raw_answer": answer.strip()[:200],
                "proposed": proposals,
                "usable_proposals": len(proposals),
                "results": [
                    {k: entry[k] for k in ("batch_size", "ratio", "ci_high", "clears_mde")}
                    for entry in measured
                ],
            }
        )
        if not proposals:
            # An unusable answer costs one round; it never stops the loop.
            continue

    ranked = _LOOP.rank([e for e in evidence if e.get("rejected") is None])
    leader = ranked[0] if ranked and ranked[0]["clears_mde"] else None

    confirmation = None
    if leader is not None:
        guard.before_candidate()
        try:
            replicates = []
            for replicate in range(CONFIRM_REPLICATES):
                if replicate:
                    guard.required_break()
                replicates.append(
                    harness.measure(leader["batch_size"], CONFIRM_BLOCKS)
                )
        finally:
            guard.finish_candidate()
        confirmation = _LOOP.hierarchical_bootstrap(replicates, seed=BOOTSTRAP_SEED)
        confirmation["clears_mde"] = _LOOP.clears_threshold(confirmation, MDE)
        confirmation["replicate_ratios"] = [
            round(math.exp(statistics.median(logs)), 4) for logs in replicates
        ]
        confirmation["replicate_log_ratios"] = replicates

    accepted = bool(confirmation and confirmation["clears_mde"])
    budget = guard.summary()
    return {
        "model": MODEL_ID,
        **snapshot.report_identity(),
        "rounds": history,
        "candidate_measurements": evidence,
        "candidates_measured": len(evidence),
        "proposed_by_model": sorted(tried),
        "selected_batch_size": leader["batch_size"] if leader else None,
        "confirmation": confirmation,
        "verdict": "optimization_confirmed" if accepted else "no_confirmed_optimization",
        "effect_percent": (
            round(100.0 * (confirmation["ratio"] - 1.0), 2) if accepted else None
        ),
        "mde": MDE,
        "gpu_work_seconds": budget["gpu_work_seconds"],
        "wall_seconds": budget["wall_seconds"],
        "budget": budget,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="model_loop", allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    args = parser.parse_args(argv)

    gated = release_gate(args, _self_check)
    if gated is not None:
        return gated
    if not 1 <= args.rounds <= 10:
        raise SystemExit("rounds must be between 1 and 10")

    power = require_ac_power()

    def operation() -> dict[str, object]:
        report = run(args.rounds)
        report["power_source"] = power
        return report

    report = run_persisted("model-loop", operation)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=1))
    return 0 if report["verdict"] == "optimization_confirmed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
