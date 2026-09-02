"""F1 — measure the confirmed gains together on one real request path.

Warm arm: one process, one loaded model, two engines. The baseline engine runs
IronMule's BASELINE knobs; the candidate additionally carries
``compiled_fixed_cache`` and ``head_skip_prefill``. Pairs alternate AB and BA so
order cannot favour either arm.

The workload is the sealed one: the persistent-process prompt family, 897
tokens, 32 generated. Reusing it exactly is the point - two of the three
confirmed gains were measured on it.

This worker produces evidence, not a verdict. The verdict comes from

    python -m friday_optimizer integrate --result <file> --arm warm \
        --min-gain 0.10 --mde 0.05

Gated like every model run: AC power, budget guard, offline snapshot, and it
refuses to start without --execute. Token identity is checked per pair and a
single mismatch ends the run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IRONMULE = ROOT / ".worktrees" / "friday-optimizer-ironmule"
sys.path.insert(0, str(ROOT / "tools"))

STUDY_ID = "f1-integration-warm-20260902-01"
MODEL_ID = "mlx-community/gemma-3-4b-it-4bit"
#: The bound IronMule checkout; the search contract names the same commit.
EXPECTED_IRONMULE_HEAD = "03e884cb28a05d090d20844460fc3afc8e738a91"

OUTPUT_TOKENS = 32
PROMPT_TOKENS = 897
PREFILL_CHUNK = 256

#: Verbatim from experiments/persistent_process/worker.py. A different prompt
#: would not be the workload two of the three gains were confirmed on.
FILLER = (
    "You are a careful engineering assistant working in a Python repository. "
    "Follow the existing style and explain your reasoning briefly. "
) * 40
QUESTIONS = {
    "P": "Why is false sharing slow?",
    "Q": "What are TLB misses?",
    "R": "When does store forwarding fail?",
    "S": "Why can branch prediction fail?",
}

MODES = ("aa", "ab")
OUTPUT = Path(__file__).resolve().parent


def candidate_knobs(knobs_class):
    """The two knobs F1 composes, on top of the baseline."""

    return knobs_class(compiled_fixed_cache=True, head_skip_prefill=True)


def self_check() -> int:
    report = {
        "study_id": STUDY_ID, "state": "self_check", "model_id": MODEL_ID,
        "prompt_tokens": PROMPT_TOKENS, "output_tokens": OUTPUT_TOKENS,
        "modes": list(MODES), "prompt_keys": sorted(QUESTIONS),
        "expected_ironmule_head": EXPECTED_IRONMULE_HEAD,
        "candidate_knobs": ["compiled_fixed_cache", "head_skip_prefill"],
        "formal_claim": False,
    }
    print(json.dumps(report, indent=2))
    return 0


def _ironmule_head() -> str:
    import subprocess

    completed = subprocess.run(
        ["/usr/bin/git", "-C", str(IRONMULE), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False, timeout=10,
    )
    if completed.returncode != 0:
        raise SystemExit("bound IronMule checkout is unreadable")
    return completed.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--mode", choices=list(MODES), default="ab",
                        help="aa compares baseline against itself; ab against the candidate")
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--self-check", action="store_true", dest="self_check")
    args = parser.parse_args(argv)

    from _bench import (BudgetGuard, check_prompt_length, enforce_offline, release_gate,
                        require_ac_power, resolve_local_model_snapshot, study_provenance)

    gate = release_gate(args, self_check)
    if gate is not None:
        return gate
    if not 1 <= args.pairs <= 64:
        raise SystemExit("pairs must be between 1 and 64")

    head = _ironmule_head()
    if head != EXPECTED_IRONMULE_HEAD:
        raise SystemExit(f"IronMule checkout is at {head}, expected {EXPECTED_IRONMULE_HEAD}")

    require_ac_power()
    guard = BudgetGuard()
    offline = enforce_offline()

    sys.path.insert(0, str(IRONMULE))
    import mlx.core as mx  # noqa: F401  (imported for the peak-memory reading)
    from mlx_lm import load
    from ironmule import BASELINE, Engine, Knobs

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

    def prompt_ids(key: str) -> list[int]:
        templated = tokenizer.apply_chat_template(
            # The separator is part of the sealed prompt: without it the count
            # is 895, not 897, and it is a different workload.
            [{"role": "user", "content": FILLER + "\n\n" + QUESTIONS[key]}],
            add_generation_prompt=True,
        )
        ids = list(templated if isinstance(templated, list) else tokenizer.encode(templated))
        check_prompt_length(ids, PROMPT_TOKENS)
        return ids

    prompts = {key: prompt_ids(key) for key in sorted(QUESTIONS)}
    eos_ids = tuple(sorted({int(value) for value in (
        getattr(tokenizer, "eos_token_id", None),
        *(getattr(tokenizer, "eos_token_ids", None) or ()),
    ) if isinstance(value, int)}))
    if not eos_ids:
        raise SystemExit("tokenizer exposes no end-of-sequence id")

    baseline_engine = Engine(model, tokenizer, BASELINE)
    # In an A/A run both arms are the baseline: the study is measuring its own
    # noise, and any difference it reports there is measurement, not effect.
    other_knobs = BASELINE if args.mode == "aa" else candidate_knobs(Knobs)
    other_engine = Engine(model, tokenizer, other_knobs)

    def one(engine, ids: list[int]) -> dict:
        at = time.perf_counter()
        out = engine.generate(ids, OUTPUT_TOKENS, eos_ids)
        charge(time.perf_counter() - at)
        tokens = list(out["logical_tokens"])
        decode_seconds = out["decode_ns"] / 1e9
        return {
            "ttft_seconds": out["prefill_ns"] / 1e9,
            "decode_seconds": decode_seconds,
            # The first token arrives with the prefill, so only the rest is decode.
            "decode_tps": (len(tokens) - 1) / decode_seconds if len(tokens) > 1 and decode_seconds > 0 else 0.0,
            "tokens": len(tokens),
            "token_sha256": hashlib.sha256(
                json.dumps(tokens, separators=(",", ":")).encode()
            ).hexdigest(),
            "knobs": dict(out["knobs"]) if isinstance(out.get("knobs"), dict) else str(out.get("knobs")),
        }

    baseline_samples, candidate_samples, pairs = [], [], []
    keys = sorted(QUESTIONS)
    for index in range(args.pairs):
        key = keys[index % len(keys)]
        ids = prompts[key]
        order = "AB" if index % 2 == 0 else "BA"
        # Alternating order so neither arm systematically runs on a warmer cache.
        if order == "AB":
            left, right = one(baseline_engine, ids), one(other_engine, ids)
        else:
            right, left = one(other_engine, ids), one(baseline_engine, ids)
        if left["token_sha256"] != right["token_sha256"]:
            OUTPUT.joinpath(f"f1_{args.mode}_identity_break.json").write_text(json.dumps({
                "study_id": STUDY_ID, "mode": args.mode, "pair": index, "prompt_key": key,
                "status": "token_identity_broken", "baseline": left, "candidate": right,
                "formal_claim": False,
            }, indent=2))
            raise SystemExit(
                f"token identity broke on pair {index} ({key}); the study ends here"
            )
        shared = {"session_id": f"{args.mode}-s{index}", "pair_id": f"p{index}",
                  "order": order, "fingerprint": snapshot.revision, "workload": key}
        baseline_samples.append({**shared, "arm": "baseline", "status": "ok", "error": "",
                                 **{k: left[k] for k in ("ttft_seconds", "decode_tps", "tokens")}})
        candidate_samples.append({**shared, "arm": "candidate", "status": "ok", "error": "",
                                  **{k: right[k] for k in ("ttft_seconds", "decode_tps", "tokens")}})
        pairs.append({"pair": index, "prompt_key": key, "order": order,
                      "token_sha256": left["token_sha256"],
                      "baseline": left, "candidate": right})
        print(json.dumps({"pair": index, "key": key, "order": order,
                          "baseline_ttft": round(left["ttft_seconds"], 4),
                          "candidate_ttft": round(right["ttft_seconds"], 4),
                          "baseline_tps": round(left["decode_tps"], 2),
                          "candidate_tps": round(right["decode_tps"], 2)}), flush=True)

    result = {
        "study_id": STUDY_ID, "mode": args.mode, "arm": "warm", "formal_claim": False,
        "model_id": MODEL_ID, "snapshot_revision": snapshot.revision,
        "ironmule_head": head, "prompt_tokens": PROMPT_TOKENS,
        "output_tokens": OUTPUT_TOKENS, "eos_ids": list(eos_ids),
        "candidate_knobs": {} if args.mode == "aa" else {
            "compiled_fixed_cache": True, "head_skip_prefill": True},
        "token_identity": True, "pairs": pairs,
        "provenance": study_provenance(
            [Path(__file__)],
            preregistration=ROOT / "docs" / "F1_INTEGRATION_VORREGISTRIERUNG.md",
            extra={"model_snapshot": snapshot.revision, "model_id": MODEL_ID,
                   "ironmule_head": head, "offline_environment": offline},
        ),
        "peak_gb": round(mx.get_peak_memory() / 1e9, 3),
        "budget": {k: v for k, v in guard.summary().items() if "limit" not in k},
        # The wire shape friday_optimizer's integrate command reads.
        "payload": {"schema": "friday.ironmule.result.v1", "stage": "test",
                    "baseline_samples": baseline_samples,
                    "candidate_samples": candidate_samples,
                    "pair_count": len(pairs), "token_identity": True},
    }
    target = OUTPUT / f"f1_warm_{args.mode}.json"
    target.write_text(json.dumps(result, indent=2))
    print(json.dumps({"written": target.name, "pairs": len(pairs),
                      "token_identity": True, "mode": args.mode,
                      "budget": result["budget"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
