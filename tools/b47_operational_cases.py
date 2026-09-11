#!/usr/bin/env python3
"""The operational cases the paired mode still owed, on real model computation.

Five things had to be shown, not assumed:

* one request reaches a genuine EOS while the other keeps going,
* a request is cancelled while the pair is mid-flight,
* different context lengths and stop points return cleanly to the single path,
* a new compatible partner joins only at a step boundary,
* an error inside the shared step leaves no unclear state.

A full stop is not an EOS: the EOS case uses a prompt that produced a real end token in
an earlier run, and the check reads the stop reason rather than the text. Forced stops
and the injected fault exercise the control flow and are labelled as such; they do not
stand in for the real EOS evidence.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ironmule import paired_research as pr  # noqa: E402
from ironmule.plans import StrictOneShotPlan  # noqa: E402
from ironmule.service import (  # noqa: E402
    PairedThroughputMode,
    Request,
    Runtime,
    ThroughputMode,
    paired_status,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# This prompt stopped on a real end token after 23 steps in the B44 integration check.
EOS_PROMPT = "Write one sentence about the sea."
LONG_PROMPT = "Explain, step by step, how a laptop runs a large language model locally."
SHORT_PROMPT = "Describe how a key value cache grows during autoregressive decoding."


def _serve(runtime, mode, requests):
    runtime.mode = mode
    results = runtime.serve(requests)
    return [{"tokens": list(r.tokens), "stop_reason": r.stop_reason,
             "generated": r.metrics["generated_tokens"],
             "fell_back": r.metrics["fell_back"]} for r in results]


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", default="mlx-community/gemma-3-12b-it-4bit")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    runtime = Runtime.load(args.model, mode=ThroughputMode(), use_tuned_profile=False)
    cases: dict = {}
    try:
        opt_in = PairedThroughputMode()
        reference = ThroughputMode()

        def pair(a_prompt, a_tokens, b_prompt, b_tokens, arrival=0.0):
            return [
                Request(prompt_ids=runtime.encode(a_prompt), max_tokens=a_tokens,
                        plan=StrictOneShotPlan()),
                Request(prompt_ids=runtime.encode(b_prompt), max_tokens=b_tokens,
                        plan=StrictOneShotPlan(), arrival_ms=arrival),
            ]

        # 1. A genuine EOS on one request while the other continues.
        requests = pair(EOS_PROMPT, 64, LONG_PROMPT, 64)
        truth = _serve(runtime, reference, requests)
        got = _serve(runtime, opt_in, requests)
        natural = [row["stop_reason"] == "eos" for row in truth]
        cases["genuine_eos_one_side"] = {
            "kind": "real model computation",
            "reference_stop_reasons": [r["stop_reason"] for r in truth],
            "candidate_stop_reasons": [r["stop_reason"] for r in got],
            "reference_lengths": [r["generated"] for r in truth],
            "candidate_lengths": [r["generated"] for r in got],
            "a_real_eos_occurred": any(natural),
            "one_side_stopped_first": len({r["generated"] for r in truth}) > 1,
            "tokens_identical": [a["tokens"] == b["tokens"] for a, b in zip(truth, got)],
            "stop_identical": [a["stop_reason"] == b["stop_reason"]
                               for a, b in zip(truth, got)],
            "no_fallback": not any(r["fell_back"] for r in got),
        }

        # 2. Different context lengths and stop points.
        requests = pair(SHORT_PROMPT, 6, LONG_PROMPT, 20)
        truth = _serve(runtime, reference, requests)
        got = _serve(runtime, opt_in, requests)
        cases["unequal_lengths_return_to_single_path"] = {
            "kind": "real model computation",
            "lengths": [r["generated"] for r in got],
            "tokens_identical": [a["tokens"] == b["tokens"] for a, b in zip(truth, got)],
            "stop_identical": [a["stop_reason"] == b["stop_reason"]
                               for a, b in zip(truth, got)],
            "paired_then_solo": paired_status(opt_in)["solo_steps_no_partner"] > 0,
            "status_after": paired_status(opt_in),
        }

        # 3. A partner arriving later joins only at a step boundary.
        requests = pair(SHORT_PROMPT, 16, LONG_PROMPT, 16, arrival=60.0)
        truth = _serve(runtime, reference, requests)
        got = _serve(runtime, opt_in, requests)
        status = paired_status(opt_in)
        cases["late_partner_joins_at_a_boundary"] = {
            "kind": "real model computation",
            "arrival_ms": [0.0, 60.0],
            "tokens_identical": [a["tokens"] == b["tokens"] for a, b in zip(truth, got)],
            "solo_steps_before_pairing": status["solo_steps_no_partner"],
            "paired_steps": status["paired_steps"],
            "both_occurred": status["solo_steps_no_partner"] > 0 and status["paired_steps"] > 0,
        }

        # 4. A single request never waits for a partner.
        single = [Request(prompt_ids=runtime.encode(SHORT_PROMPT), max_tokens=8,
                          plan=StrictOneShotPlan())]
        truth = _serve(runtime, reference, single)
        got = _serve(runtime, opt_in, single)
        status = paired_status(opt_in)
        cases["single_request_never_waits"] = {
            "kind": "real model computation",
            "tokens_identical": truth[0]["tokens"] == got[0]["tokens"],
            "paired_steps": status["paired_steps"],
            "solo_steps_no_partner": status["solo_steps_no_partner"],
            "never_paired": status["paired_steps"] == 0,
        }

        # 5. A fault inside the shared step. Labelled: this is an injected fault that
        #    exercises the existing fallback, not a naturally occurring error.
        broken = PairedThroughputMode()
        runtime.mode = broken
        requests = pair(SHORT_PROMPT, 8, LONG_PROMPT, 8)
        broken.executor(runtime.backend, runtime.telemetry)  # force admission first
        original = pr.PairedBackend.step_pair
        calls = {"n": 0}

        def failing(self, states, tokens, capacity):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("injected fault inside the shared step")
            return original(self, states, tokens, capacity)

        pr.PairedBackend.step_pair = failing
        try:
            got = _serve(runtime, broken, requests)
        finally:
            pr.PairedBackend.step_pair = original
        truth = _serve(runtime, reference, requests)
        cases["injected_fault_in_shared_step"] = {
            "kind": "injected fault, control-flow test only",
            "fault_raised": calls["n"] >= 2,
            "requests_completed": len(got),
            "tokens_identical_to_reference": [a["tokens"] == b["tokens"]
                                              for a, b in zip(truth, got)],
            "lengths": [r["generated"] for r in got],
            "no_duplicate_tokens": all(r["generated"] == 8 for r in got),
            "fell_back": [r["fell_back"] for r in got],
        }

        # 6. Cancellation. The shipped Request/Result surface has no cancel handle, so
        #    this is reported as a gap rather than claimed as passed.
        cases["cancellation_mid_flight"] = {
            "kind": "not testable through the shipped surface",
            "gap": (
                "ironmule.service.Request carries prompt_ids, max_tokens, plan and "
                "arrival_ms, and serve() runs to completion; there is no cancel handle "
                "or cooperative stop token on the shipped API, so a mid-flight "
                "cancellation cannot be exercised without building one"
            ),
            "claimed_passed": False,
        }
    finally:
        runtime.close()

    real_cases = {name: row for name, row in cases.items()
                  if row.get("kind") == "real model computation"}
    identity_clean = all(
        all(value) if isinstance(value, list) else value
        for row in real_cases.values()
        for key, value in row.items()
        if key.endswith("_identical")
    )
    record = {
        "schema": "ironmule.paired_operational_cases.v1",
        "experiment_id": args.out.stem,
        "agent": "claude",
        "kind": "operational_cases",
        "status": "measured",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "question": "do the open operational cases hold on the opt-in paired path",
        "model": args.model,
        "cases": cases,
        "identity_clean_on_real_cases": identity_clean,
        "open_gaps": [name for name, row in cases.items() if row.get("claimed_passed") is False],
        "limits": [
            "a full stop is not an EOS; the EOS case reads the stop reason, not the text",
            "the injected fault tests control flow and is not a naturally occurring error",
            "these cases extend correctness evidence only, not the performance claim",
        ],
        "mlx_version": mx.__version__,
        "device": {"platform": platform.platform(), "machine": platform.machine()},
        "git_revision": subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False).stdout.strip(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"identity_clean_on_real_cases": identity_clean,
                      "open_gaps": record["open_gaps"],
                      "cases": {name: {k: v for k, v in row.items() if k != "status_after"}
                                for name, row in cases.items()}},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
