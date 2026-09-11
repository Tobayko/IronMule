#!/usr/bin/env python3
"""Close the identity gap the load matrix left open, on the eight stored load cases.

`B50` compared tokens, stop reasons and text. That is weaker than equal logits. This
walks each load case's prompts and token budgets through the decode path twice, once on
the shipped body and once on the paired step, and compares the full logit bit pattern of
every step and the whole KV state of every request.

Arrival times are not varied here: they change which requests get grouped, not the
arithmetic each request performs, and grouping variants are covered at service level by
`B46` and `B50`. What is new is bit-level evidence on these prompts and lengths.

Two real end-token cases are added, chosen so one request stops early while the other
runs on, and so the two stop at different points.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from b50_workload_matrix import MATRIX  # noqa: E402

from ironmule import paired_research as pr  # noqa: E402
from ironmule.plans import StrictOneShotPlan  # noqa: E402
from ironmule.service import Request, Runtime, ThroughputMode  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Known to end on a real end token in earlier runs, at clearly different lengths.
EOS_SHORT = "Write one sentence about the sea."
EOS_LONG = "Explain, step by step, how a laptop runs a large language model locally."


def _walk(runtime, prompts, budgets, *, paired: bool) -> dict:
    """Decode every request for its budget, collecting per-step logits and final KV."""

    backend = runtime.backend
    engine = runtime.engine
    ids = [runtime.encode(p) for p in prompts]
    capacity = backend.capacity_for([len(i) for i in ids], max(budgets))
    states, tokens, digests, produced, stopped = [], [], [], [], []
    eos = set(backend.eos_ids)
    for prompt_ids in ids:
        state, token = backend.prefill(prompt_ids, StrictOneShotPlan(), capacity)
        states.append(state)
        tokens.append(token)
        digests.append(hashlib.sha256())
        produced.append([token])
        stopped.append(token in eos)

    paired_backend = pr.PairedBackend(backend, share=True) if paired else None
    for step in range(max(budgets) - 1):
        live = [i for i in range(len(prompts))
                if not stopped[i] and len(produced[i]) < budgets[i]]
        if not live:
            break
        if paired and len(live) >= 2:
            group = live[:2]
            handles = paired_backend.step_pair([states[i] for i in group],
                                               [tokens[i] for i in group], capacity)
            rest = live[2:]
        else:
            group, handles, rest = [], [], live
        for index, ((out, new_state), pick) in zip(group, handles):
            mx.eval(out, pick)
            digests[index].update(bytes(memoryview(out[:, -1, :])))
            token = int(pick.item())
            states[index], tokens[index] = new_state, token
            produced[index].append(token)
            stopped[index] = token in eos
        for index in rest:
            body = engine._body(capacity, 1)
            out, new_state = body(mx.array([[tokens[index]]]), states[index])
            mx.eval(out)
            digests[index].update(bytes(memoryview(out[:, -1, :])))
            token = int(mx.argmax(out[:, -1, :].astype(mx.float32), axis=-1).item())
            states[index], tokens[index] = new_state, token
            produced[index].append(token)
            stopped[index] = token in eos
    return {
        "tokens": produced,
        "logits_sha256": [d.hexdigest() for d in digests],
        "kv_sha256": [backend.kv_hash(s, int(s["position"]["offset"].item())) for s in states],
        "stopped_on_eos": stopped,
        "steps": [len(p) for p in produced],
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", default="mlx-community/gemma-3-12b-it-4bit")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    runtime = Runtime.load(args.model, mode=ThroughputMode(), use_tuned_profile=False)
    try:
        cases = {name: (prompts, budgets) for name, (prompts, budgets, _) in MATRIX.items()}
        cases["eos_one_stops_early"] = ([EOS_SHORT, EOS_LONG], [64, 64])
        cases["eos_both_stop_differently"] = ([EOS_SHORT, "Name one colour of the sky."],
                                              [64, 64])

        results = {}
        for name, (prompts, budgets) in cases.items():
            reference = _walk(runtime, prompts, budgets, paired=False)
            candidate = _walk(runtime, prompts, budgets, paired=True)
            results[name] = {
                "requests": len(prompts),
                "logits_identical": [a == b for a, b in
                                     zip(reference["logits_sha256"], candidate["logits_sha256"])],
                "kv_identical": [a == b for a, b in
                                 zip(reference["kv_sha256"], candidate["kv_sha256"])],
                "tokens_identical": [a == b for a, b in
                                     zip(reference["tokens"], candidate["tokens"])],
                "steps_identical": reference["steps"] == candidate["steps"],
                "stop_identical": reference["stopped_on_eos"] == candidate["stopped_on_eos"],
                "steps": reference["steps"],
                "stopped_on_eos": reference["stopped_on_eos"],
            }

        clean = all(
            all(row[key]) if isinstance(row[key], list) else row[key]
            for row in results.values()
            for key in ("logits_identical", "kv_identical", "tokens_identical",
                        "steps_identical", "stop_identical")
        )
        eos_rows = {n: r for n, r in results.items() if n.startswith("eos_")}
        real_eos = any(any(r["stopped_on_eos"]) for r in eos_rows.values())
        one_early = any(any(r["stopped_on_eos"]) and not all(r["stopped_on_eos"])
                        for r in eos_rows.values())
        different_lengths = any(len(set(r["steps"])) > 1 for r in eos_rows.values())

        record = {
            "schema": "ironmule.paired_identity_gap.v1",
            "experiment_id": args.out.stem,
            "agent": "claude",
            "kind": "integration_correctness",
            "status": "measured",
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "question": "are the eight load cases bit-identical in logits and KV state",
            "model": args.model,
            "cases": results,
            "all_identical": clean,
            "real_eos_observed": real_eos,
            "one_request_stopped_while_another_continued": one_early,
            "requests_stopped_at_different_lengths": different_lengths,
            "method": (
                "each request is decoded for its own budget; pairs go through the paired "
                "step and a lone survivor falls to the single path, which also exercises "
                "the pair to single transition and a later partner"
            ),
            "not_covered": [
                "arrival times: they change grouping, not the arithmetic per request, and "
                "grouping variants are covered at service level by B46 and B50",
                "cancellation mid-flight: ironmule.service.Request carries no cancel "
                "handle and serve() runs to completion; still an interface limit, not a pass",
            ],
            "mlx_version": mx.__version__,
            "device": {"platform": platform.platform(), "machine": platform.machine()},
            "git_revision": subprocess.run(
                ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=False).stdout.strip(),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({"all_identical": clean, "real_eos_observed": real_eos,
                          "one_stopped_while_other_continued": one_early,
                          "different_stop_lengths": different_lengths,
                          "cases": {n: {"steps": r["steps"], "eos": r["stopped_on_eos"]}
                                    for n, r in results.items()}},
                         indent=2, sort_keys=True))
        return 0 if clean else 1
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
