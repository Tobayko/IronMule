#!/usr/bin/env python3
"""H2 full: a local model writes execution plans, the harness runs and judges them.

This is the step past ``model_loop``.  There the model picked a number from a
space I designed; here it writes the plan itself -- the actual Python that
dispatches the work -- and the harness decides whether it may run, whether it is
correct, and whether it is faster.

Three gates stand between a generated string and a reported result:

  1. **Static validation** (``plan_sandbox``): an AST allowlist that rejects
     imports, dunder access, string literals, unknown names and every MLX call
     outside a small fixed set.  Nothing executes until this passes.
  2. **Process isolation**: what passes runs in a fresh subprocess with a
     wall-clock timeout, a CPU-time ceiling and a scrubbed environment. The
     plan language bounds iteration and matmul allocation; MLX's memory setting is
     additional best-effort defense, not a hard memory limit.
  3. **Correctness**: the plan must produce one result per operand, each
     byte-identical to the reference.  A faster plan that changes a single bit is
     discarded, not reported.

Only what survives all three gets timed, and only a paired measurement that
clears the frozen threshold counts as a finding.

Run with --execute; without it nothing is imported or measured.
"""

from __future__ import annotations

import argparse
import importlib.util
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
from plan_sandbox import (  # noqa: E402
    ALLOWED_MX_ATTRS,
    PlanRejected,
    run_plan_isolated,
    validate_plan_source,
)

_SPEC = importlib.util.spec_from_file_location(
    "measure_dispatch_plan", Path(__file__).resolve().parent / "measure_dispatch_plan.py"
)
_PLAN = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PLAN)

MODEL_ID = "mlx-community/gemma-3-4b-it-4bit"
FIXTURE_SEED = 4051312678
POOL_SEED = 20260821
OPERANDS = 8
EXPLORE_BLOCKS = 12
CONFIRM_BLOCKS = 20
CONFIRM_REPLICATES = 3
MAX_MODEL_TOKENS = 320
DEFAULT_ROUNDS = 4
MDE = _PLAN.MDE
BOOTSTRAP_SEED = 0x30092_2026

BASELINE_DESCRIPTION = """The baseline it must beat, for reference:

def plan(mx, a, operands):
    out = []
    for b in operands:
        x = mx.matmul(a, b)
        mx.eval(x)
        mx.synchronize()
        out.append(x)
    return out"""



BASELINE_SOURCE = """def plan(mx, a, operands):
    out = []
    for b in operands:
        x = mx.matmul(a, b)
        mx.eval(x)
        mx.synchronize()
        out.append(x)
    return out"""


def _normalized(source: str) -> str:
    """Whitespace-insensitive form, for comparing a plan against the baseline."""

    return " ".join(source.split())


def is_baseline_copy(source: str) -> bool:
    """True when the model simply echoed the baseline back.

    Worth catching explicitly: measuring the baseline against itself yields a
    ratio near 1.0, which looks like an honest null result and hides the fact
    that the model never proposed anything.
    """

    return _normalized(source) == _normalized(BASELINE_SOURCE)


def extract_plan(text: str) -> str | None:
    """Pull the plan function out of a model answer.

    Accepts a fenced block or a bare definition; returns None when there is no
    plausible candidate, so an unusable answer costs a round and nothing else.
    """

    if not isinstance(text, str):
        return None
    fenced = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    body = fenced.group(1) if fenced else text
    start = body.find("def plan")
    if start < 0:
        return None
    return body[start:].strip()


def build_prompt(history: list[dict]) -> str:
    operations = ", ".join(f"mx.{name}" for name in sorted(ALLOWED_MX_ATTRS))
    lines = [
        "You optimize GPU execution plans on an Apple M1 Max (32-core GPU, unified memory).",
        "",
        f"Write a Python function that computes mx.matmul(a, b) for each of the {OPERANDS}",
        "arrays in `operands` and returns the results in order, as a list.",
        "",
        "Hard requirements:",
        "  - exactly: def plan(mx, a, operands):",
        f"  - return a list of {OPERANDS} results, in the same order as operands",
        "  - results must be bit-identical to mx.matmul(a, operands[i])",
        f"  - you may only call: {operations}, and range/len/enumerate/list",
        "  - no imports, no string literals, no attribute access except mx.<op>",
        "",
        "Device facts measured on this machine:",
        "  - mx.eval accepts a LIST of results and evaluates them together",
        "  - one synchronize at the end costs far less than one per operation",
        "  - the baseline below is slow precisely because it synchronizes 8 times",
        "  - a disturbance process with ~340 ms timescale affects all measurements",
        "  - the first operation after an idle pause is up to 4x slower",
        "",
        BASELINE_DESCRIPTION,
        "",
    ]
    if history:
        lines.append("What you tried before:")
        for entry in history:
            if entry["outcome"] == "measured":
                lines.append(
                    f"  - {entry['label']}: ratio {entry['ratio']:.3f} "
                    f"({'beats baseline' if entry['clears_mde'] else 'not beyond noise'})"
                )
            else:
                lines.append(f"  - {entry['label']}: rejected ({entry['reason']})")
        lines.append("")
    lines += [
        "Write ONE new plan that is faster than the baseline.",
        "Do NOT copy the baseline. It is shown only so you can beat it.",
        "Answer with ONLY the function inside a ```python fence. No explanation.",
    ]
    return "\n".join(lines)


def _self_check() -> int:
    """Offline checks of extraction and prompting; no GPU, no model."""

    fenced = "Here you go:\n```python\ndef plan(mx, a, operands):\n    return operands\n```"
    assert extract_plan(fenced).startswith("def plan")
    bare = "def plan(mx, a, operands):\n    return operands\n"
    assert extract_plan(bare).startswith("def plan")
    # Leading prose before a bare definition is tolerated.
    assert extract_plan("Sure!\ndef plan(mx, a, operands):\n    return operands").startswith("def plan")
    assert extract_plan("I cannot help with that.") is None
    assert extract_plan("") is None
    assert extract_plan(None) is None

    # Whatever is extracted still has to survive validation; extraction is not
    # permission.  A hostile answer must be caught one layer down.
    hostile = extract_plan("```python\ndef plan(mx, a, operands):\n    import os\n    return operands\n```")
    assert hostile is not None
    try:
        validate_plan_source(hostile)
    except PlanRejected:
        pass
    else:  # pragma: no cover
        raise AssertionError("import must not pass validation")

    assert is_baseline_copy(BASELINE_SOURCE)
    assert is_baseline_copy(BASELINE_SOURCE.replace("\n", "\n  "))
    assert not is_baseline_copy("def plan(mx, a, operands):\n    return operands\n")

    prompt = build_prompt([{"outcome": "measured", "label": "chunked", "ratio": 0.83,
                            "clears_mde": True}])
    assert "chunked" in prompt and "0.830" in prompt
    prompt = build_prompt([{"outcome": "rejected", "label": "bad", "reason": "import os"}])
    assert "rejected" in prompt
    assert "def plan(mx, a, operands)" in build_prompt([])
    print(json.dumps({"self_check": "pass", "checks": 14}))
    return 0


def run(rounds: int) -> dict[str, object]:
    guard = BudgetGuard()
    from mlx_lm import generate, load

    snapshot = resolve_local_model_snapshot(MODEL_ID)
    model, tokenizer = load(str(snapshot.path))
    history: list[dict] = []
    attempts: list[dict] = []
    survivors: list[dict] = []

    for index in range(rounds):
        guard.required_break()
        gpu_started = time.perf_counter()
        answer = generate(
            model,
            tokenizer,
            tokenizer.apply_chat_template(
                [{"role": "user", "content": build_prompt(history)}],
                add_generation_prompt=True,
            ),
            max_tokens=MAX_MODEL_TOKENS,
            verbose=False,
        )
        guard.record_gpu(time.perf_counter() - gpu_started)
        label = f"plan_{index + 1}"
        source = extract_plan(answer)
        record: dict[str, object] = {"round": index, "label": label,
                                     "answer_head": (answer or "").strip()[:160]}

        if source is None:
            record.update(outcome="no_plan", reason="no function in answer")
            attempts.append(record)
            history.append({"outcome": "rejected", "label": label, "reason": "no function"})
            continue
        if is_baseline_copy(source):
            record.update(outcome="rejected", reason="copied the baseline verbatim")
            attempts.append(record)
            history.append({"outcome": "rejected", "label": label,
                            "reason": "you copied the baseline; write something different"})
            continue
        try:
            validate_plan_source(source)
        except PlanRejected as exc:
            record.update(outcome="rejected", reason=str(exc), source=source[:400])
            attempts.append(record)
            history.append({"outcome": "rejected", "label": label, "reason": str(exc)})
            continue

        guard.required_break()
        guard.before_candidate()
        try:
            outcome = run_plan_isolated(
                source, n=OPERANDS, blocks=EXPLORE_BLOCKS,
                fixture_seed=FIXTURE_SEED, pool_seed=POOL_SEED,
            )
            if isinstance(outcome.get("gpu_work_seconds"), (int, float)):
                guard.record_gpu(float(outcome["gpu_work_seconds"]))
        finally:
            guard.finish_candidate()
        if outcome.get("fatal"):
            raise SystemExit("isolated worker containment failure; measurement aborted")
        if not outcome.get("ok"):
            record.update(outcome="failed", reason=outcome.get("reason"), source=source[:400])
            attempts.append(record)
            history.append({"outcome": "rejected", "label": label,
                            "reason": str(outcome.get("reason"))})
            continue

        summary = _PLAN.paired_ratio(outcome["log_ratios"])
        summary["clears_mde"] = _PLAN.clears_threshold(summary, MDE)
        summary["log_ratios"] = outcome["log_ratios"]
        summary["gpu_work_seconds"] = outcome.get("gpu_work_seconds")
        record.update(outcome="measured", source=source, **summary)
        attempts.append(record)
        history.append({"outcome": "measured", "label": label,
                        "ratio": summary["ratio"], "clears_mde": summary["clears_mde"]})
        if summary["clears_mde"]:
            survivors.append(record)

    # Rank by upper confidence bound, not point estimate: the best of several
    # noisy candidates is optimistic by construction.
    leader = min(survivors, key=lambda e: e["ci_high"]) if survivors else None

    confirmation = None
    if leader is not None:
        replicates = []
        guard.before_candidate()
        try:
            for replicate in range(CONFIRM_REPLICATES):
                if replicate:
                    guard.required_break()
                outcome = run_plan_isolated(
                    leader["source"], n=OPERANDS, blocks=CONFIRM_BLOCKS,
                    fixture_seed=FIXTURE_SEED, pool_seed=POOL_SEED,
                )
                if isinstance(outcome.get("gpu_work_seconds"), (int, float)):
                    guard.record_gpu(float(outcome["gpu_work_seconds"]))
                if outcome.get("fatal"):
                    raise SystemExit(
                        "isolated worker containment failure; measurement aborted"
                    )
                if outcome.get("ok"):
                    replicates.append(outcome["log_ratios"])
        finally:
            guard.finish_candidate()
        if len(replicates) >= 2:
            confirmation = _PLAN.hierarchical_bootstrap(replicates, seed=BOOTSTRAP_SEED)
            confirmation["clears_mde"] = _PLAN.clears_threshold(confirmation, MDE)
            confirmation["replicate_ratios"] = [
                round(math.exp(statistics.median(logs)), 4) for logs in replicates
            ]
            confirmation["replicate_log_ratios"] = replicates

    accepted = bool(confirmation and confirmation["clears_mde"])
    budget = guard.summary()
    return {
        "model": MODEL_ID,
        **snapshot.report_identity(),
        "operands": OPERANDS,
        "attempts": attempts,
        "written": len(attempts),
        "passed_validation": sum(1 for a in attempts if a["outcome"] in {"measured", "failed"}),
        "measured": sum(1 for a in attempts if a["outcome"] == "measured"),
        "beat_threshold": len(survivors),
        "selected_plan": leader["source"] if leader else None,
        "confirmation": confirmation,
        "verdict": "optimization_confirmed" if accepted else "no_confirmed_optimization",
        "effect_percent": (
            round(100.0 * (confirmation["ratio"] - 1.0), 2) if accepted else None
        ),
        "mde": MDE,
        "wall_seconds": budget["wall_seconds"],
        "gpu_work_seconds": budget["gpu_work_seconds"],
        "budget": budget,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="codegen_loop", allow_abbrev=False)
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

    report = run_persisted("codegen", operation)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=1))
    return 0 if report["verdict"] == "optimization_confirmed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
