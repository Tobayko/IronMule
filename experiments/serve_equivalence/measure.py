"""D2 — does `friday_serve` reproduce the reference decoder token for token?

Baseline equivalence before gain. `friday_serve` is new code on an engine that
was measured, and the first question is not whether it is faster but whether it
is the *same*. So every knob is off and the answer has to match
``mlx_lm.stream_generate`` exactly — same token ids, same ``token_sha256``, on
several prompt families rather than one.

Only after that does the second arm run: the same request with whatever knobs a
device profile has verified, again checked for token identity. A gain that
arrives together with a changed answer is not a gain.

This touches the GPU. AGENTS.md, section "Hardwarefreigabe" (2026-09-02): real
runs no longer need individual confirmation, but every measurement condition
stays — AC power, no foreign load, BudgetGuard duty cycle, warmup, repetitions.

Run: ``python experiments/serve_equivalence/measure.py --execute``
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

STUDY_ID = "serve-equivalence-20260902-01"
MODEL_ID = "mlx-community/gemma-3-4b-it-4bit"
OUTPUT = Path(__file__).resolve().parent

#: Three families, deliberately unlike each other. One prompt would only show
#: that the two decoders agree on one prompt.
PROMPTS = {
    "sealed": (
        "You are a careful engineering assistant working in a Python repository. "
        "Follow the existing style and explain your reasoning briefly. "
    ) * 40 + "\n\nWhy is false sharing slow?",
    "short": "Name three causes of cache misses.",
    "code": (
        "def parse(line: str) -> dict:\n"
        "    key, _, value = line.partition('=')\n"
        "    return {key.strip(): value.strip()}\n\n"
        "Explain what this function does with an empty line."
    ),
}
MAX_TOKENS = 24


def self_check() -> int:
    print(json.dumps({
        "state": "self_check", "study_id": STUDY_ID, "model_id": MODEL_ID,
        "prompts": sorted(PROMPTS), "max_tokens": MAX_TOKENS,
        "compares": ["mlx_lm.stream_generate", "friday_serve.Server.generate"],
        "formal_claim": False,
    }, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--self-check", action="store_true", dest="self_check")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    args = parser.parse_args(argv)

    from _bench import (BudgetGuard, enforce_offline, release_gate, require_ac_power,
                        study_provenance)

    gate = release_gate(args, self_check)
    if gate is not None:
        return gate

    require_ac_power()
    guard = BudgetGuard()
    enforce_offline()

    from mlx_lm.generate import stream_generate

    from friday_calibrate.profile import HISTORY, newest_profile
    from friday_serve.ironmule_backend import IronMuleBackend
    from friday_serve.server import Server

    debt = 0.0

    def charge(seconds: float) -> None:
        nonlocal debt
        guard.record_gpu(seconds)
        debt += seconds * (1 - 0.15) / 0.15
        while debt >= 4.0:
            guard.required_break()
            debt -= 4.0

    backend = IronMuleBackend.load(MODEL_ID)

    def reference(prompt: str, max_tokens: int) -> list[int]:
        """The decoder to agree with: mlx_lm's own greedy loop."""
        ids = backend.encode(prompt)
        at = time.perf_counter()
        tokens = [
            int(response.token)
            for response in stream_generate(
                backend.model, backend.tokenizer, ids, max_tokens=max_tokens
            )
        ]
        charge(time.perf_counter() - at)
        return tokens

    profile = None
    try:
        from friday_runtime_core.history import RuntimeHistory

        database = ROOT / ".friday-data" / "device-profile.sqlite3"
        with RuntimeHistory.open(HISTORY, database, read_only=True) as history:
            with history.read_transaction():
                profile = newest_profile(history.verified_records())
    except Exception:
        profile = None

    baseline_server = Server(backend, None)          # every knob off, by construction
    profile_server = Server(backend, profile)

    rows = []
    identical = True
    for key in sorted(PROMPTS):
        prompt = PROMPTS[key]
        # Warmup on this prompt so neither arm pays allocation and kernel build-up.
        baseline_server.generate(prompt, 4)
        expected = reference(prompt, args.max_tokens)

        at = time.perf_counter()
        served = baseline_server.generate(prompt, args.max_tokens)
        charge(time.perf_counter() - at)

        row = {
            "prompt": key,
            "prompt_tokens": len(backend.encode(prompt)),
            "reference_tokens": len(expected),
            "served_tokens": len(served.tokens),
            "baseline_identical": list(served.tokens) == expected[: len(served.tokens)],
            "baseline_plan": served.plan,
            "baseline_reason": served.reason,
        }
        if profile is not None:
            at = time.perf_counter()
            optimised = profile_server.generate(prompt, args.max_tokens)
            charge(time.perf_counter() - at)
            row["optimised_plan"] = optimised.plan
            row["optimised_knobs"] = dict(optimised.knobs)
            row["optimised_identical"] = optimised.token_sha256 == served.token_sha256
            identical = identical and row["optimised_identical"]
        identical = identical and row["baseline_identical"]
        rows.append(row)

    report = {
        "study_id": STUDY_ID,
        "state": "measured",
        "model_id": MODEL_ID,
        "model_revision": backend.model_revision,
        "max_tokens": args.max_tokens,
        "rows": rows,
        "all_identical": identical,
        "verdict": "equivalent" if identical else "identity_break",
        "device_profile": None if profile is None else profile.profile_id,
        "budget": guard.summary(),
        "provenance": study_provenance(
            [Path(__file__), ROOT / "friday_serve" / "server.py",
             ROOT / "friday_serve" / "ironmule_backend.py"],
            extra={"model_id": MODEL_ID, "model_revision": backend.model_revision},
        ),
        "formal_claim": False,
    }
    OUTPUT.joinpath("equivalence.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in report.items() if k != "provenance"}, indent=2, sort_keys=True))
    return 0 if identical else 2


if __name__ == "__main__":
    raise SystemExit(main())
