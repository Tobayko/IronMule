"""S3 — does the identity break reproduce, and where does it sit?

H1.0 broke token identity at 4B/128 tokens/draft width 2 and kept nothing but
the message: `friday_calibrate.runner.Sample` carried a digest, and the two
sequences died with the process. This study repeats that one cell with the dump
in place.

It measures an **event**, not a gain: no A/A, no threshold, no verdict about
speed. Preregistration: `docs/S3_VORREGISTRIERUNG.md`.

The replay below is the part that costs nothing. `_lookup_draft` is a pure
function of the sequence so far, and the engine's gate accepts position `i` only
when `draft[i-1] == chosen[i-1]` — and `chosen[i-1]` is the token it already
emitted. Prompt plus emitted sequence therefore reconstruct the draft, the
accepted count and the iteration boundaries of every step, which is the
acceptance bookkeeping that would otherwise need the engine instrumented.

Run: ``python experiments/identity_break/measure.py --execute``
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
IRONMULE = ROOT / ".worktrees" / "friday-optimizer-ironmule"

STUDY_ID = "identity-break-20260902-01"
OUTPUT = Path(__file__).resolve().parent
PREREGISTRATION = ROOT / "docs" / "S3_VORREGISTRIERUNG.md"

COMBINED = {"head_skip_prefill": True, "compiled_fixed_cache": True, "readback_every": 8}
TOKENS = 128
WIDTH = 2
NGRAM = 3
PAIRS = 10


def replay(prompt_ids, emitted, *, ngram: int = NGRAM, k: int = WIDTH):
    """Reconstruct each emitted token's iteration and its index inside it.

    Returns one entry per emitted token: ``{"index", "iteration", "j", "draft"}``.
    ``j == 0`` is the free token of its iteration — no draft gated it. ``j > 0``
    is a gated output: the gate let the step before it through.

    The structure is *derived* from the emitted tokens, not observed. That makes
    it exact where the gate is deterministic, and it is why the check in
    :func:`well_formed` is necessary but not sufficient (see the preregistration).
    """

    sys.path.insert(0, str(IRONMULE))
    from ironmule.runtime import _lookup_draft  # noqa: E402

    width = k + 1
    entries = [{"index": 0, "iteration": None, "j": None, "draft": []}]
    sequence = list(prompt_ids) + [emitted[0]]
    position, iteration = 1, 0
    while position < len(emitted):
        draft = _lookup_draft(sequence, ngram, k)
        accepted = 1
        while (
            accepted < width
            and accepted - 1 < len(draft)
            and position + accepted - 1 < len(emitted)
            and draft[accepted - 1] == emitted[position + accepted - 1]
            and position + accepted < len(emitted)
        ):
            accepted += 1
        for offset in range(accepted):
            if position + offset >= len(emitted):
                break
            entries.append({"index": position + offset, "iteration": iteration,
                            "j": offset, "draft": list(draft)})
        sequence.extend(emitted[position:position + accepted])
        position += accepted
        iteration += 1
    return entries


def well_formed(entries, *, k: int = WIDTH) -> bool:
    """Every iteration holds `1..k+1` tokens, indices are contiguous from zero."""

    if not entries or entries[0]["iteration"] is not None:
        return False
    sizes: dict[int, int] = {}
    for entry in entries[1:]:
        sizes[entry["iteration"]] = sizes.get(entry["iteration"], 0) + 1
    return (
        [entry["index"] for entry in entries] == list(range(len(entries)))
        and all(1 <= size <= k + 1 for size in sizes.values())
        and sorted(sizes) == list(range(len(sizes)))
    )


def first_divergence(left, right) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def self_check() -> int:
    print(json.dumps({
        "state": "self_check", "study_id": STUDY_ID, "cell": "4b/128/width_2",
        "baseline_arm": COMBINED, "pairs": PAIRS, "ngram": NGRAM,
        "measures": "whether the identity break reproduces, and where it sits",
        "no_timing": True, "formal_claim": False, "no_activation": True,
    }, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--self-check", action="store_true", dest="self_check")
    parser.add_argument("--pairs", type=int, default=PAIRS)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    from _bench import release_gate, study_provenance

    gate = release_gate(args, self_check)
    if gate is not None:
        return gate

    from friday_evidence.budget import BudgetError

    from friday_calibrate.runner import MODELS, build_runner, paired_arms

    model_id = MODELS["4b"]
    runner, identity, guard = build_runner(
        args.pairs, model_id=model_id, output_tokens=TOKENS, prompt_tokens=897,
    )
    knobs = dict(COMBINED, speculate_k=WIDTH, speculate_ngram=NGRAM)
    started = time.time()

    # Warmup, and the replayer's healthy case in the same two calls.
    try:
        base_warm = runner(COMBINED)
        cand_warm = runner(knobs)
    except BudgetError as error:
        report = {"study_id": STUDY_ID, "state": "not_measured",
                  "verdict": "budget_error_in_warmup", "reason": str(error),
                  "budget": guard.summary()}
        (args.out or OUTPUT / "identity_break.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return 4

    prompt_len = identity["prompt_tokens"]
    warm_identical = base_warm.token_sha256 == cand_warm.token_sha256
    healthy = replay(range(prompt_len), list(cand_warm.token_ids))
    healthy_ok = well_formed(healthy)

    captured: dict = {}

    def on_break(index, left, right) -> None:
        captured.update(pair=index, baseline_tokens=list(left.token_ids),
                        candidate_tokens=list(right.token_ids))

    try:
        baseline, candidate, breaks = paired_arms(
            runner, knobs, pairs=args.pairs, workload=f"sealed-{prompt_len}-{TOKENS}",
            baseline_knobs=COMBINED, on_break=on_break,
        )
    except BudgetError as error:
        report = {"study_id": STUDY_ID, "state": "not_measured",
                  "verdict": "budget_error_in_pairs", "reason": str(error),
                  "pairs_completed": len(baseline) if "baseline" in dir() else 0,
                  "budget": guard.summary()}
        (args.out or OUTPUT / "identity_break.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return 4

    divergence = None
    if captured:
        index = first_divergence(captured["baseline_tokens"], captured["candidate_tokens"])
        entries = replay(range(prompt_len), captured["candidate_tokens"])
        entry = next((item for item in entries if item["index"] == index), None)
        divergence = {
            "first_divergent_index": index,
            "baseline_token": captured["baseline_tokens"][index] if index is not None else None,
            "candidate_token": captured["candidate_tokens"][index] if index is not None else None,
            "iteration": None if entry is None else entry["iteration"],
            "j_in_iteration": None if entry is None else entry["j"],
            "draft_at_iteration": None if entry is None else entry["draft"],
            # j == 0 rules branch 2 out for this position; j > 0 only burdens it.
            "reading": (
                "unknown" if entry is None
                else "free_token_points_to_numerics" if entry["j"] == 0
                else "gated_output_burdens_acceptance_logic"
            ),
            "replay_trustworthy_up_to_index": index,
        }

    report = {
        "study_id": STUDY_ID, "state": "measured", **identity,
        "cell": "4b/128/width_2", "baseline_arm": COMBINED, "candidate_knobs": knobs,
        "pairs_requested": args.pairs, "pairs_completed": len(baseline),
        "break_reproduced": bool(captured),
        "break_at_pair": captured.get("pair"),
        "divergence": divergence,
        "sequences": {
            "baseline": captured.get("baseline_tokens"),
            "candidate": captured.get("candidate_tokens"),
        },
        "replayer_healthy_case": {
            "warmup_arms_identical": warm_identical,
            "well_formed": healthy_ok,
            "iterations": 0 if not healthy else max(
                (item["iteration"] for item in healthy if item["iteration"] is not None),
                default=-1) + 1,
        },
        "verdict": (
            "break_reproduced" if captured
            else "not_reproduced_in_pairs"
        ),
        "no_timing": True,
        "wall_seconds": round(time.time() - started, 1),
        "budget": guard.summary(),
        "provenance": study_provenance(
            [Path(__file__), ROOT / "friday_calibrate" / "runner.py",
             IRONMULE / "ironmule" / "runtime.py"],
            preregistration=PREREGISTRATION,
            extra={"model_id": model_id, "model_revision": identity["model_revision"]},
        ),
        "formal_claim": False, "no_activation": True,
    }
    destination = args.out or OUTPUT / "identity_break.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True))
    printable = {k: v for k, v in report.items() if k not in ("provenance", "sequences")}
    print(json.dumps(printable, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
