"""Amendment to H1.0 — is the loss the method, or one implementation of it?

H1.0 measures speculation losing at every width and length against the serving
path. S1 measures it winning at the same lengths. The two studies do not share
an implementation: S1 runs `friday_hardware/speculate.py`, which evaluates only
the logits and synchronises once at the end; H1.0 runs
`ironmule/runtime.py:_decode_speculative`, which evaluates the whole KV state
and synchronises **every iteration**.

This study measures one cell on the repaired path. It does not change H1.0's
rules, thresholds or results, and it reuses that regime's measured A/A noise
because the baseline arm is unchanged.

Preregistration: `docs/H10_AMENDMENT_SPEKULATIONSPFAD.md`.

The patch lives in the worktree, not in a commit, so `EXPECTED_IRONMULE_HEAD`
would still pass while the code differs. The report therefore records the
worktree's dirty state and the hash of the patched file; without that the
deviation would be invisible in the evidence.

Run: ``python experiments/spec_path/measure.py --execute --mde 0.005226605385932226``
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

STUDY_ID = "spec-path-amendment-20260902-01"
OUTPUT = Path(__file__).resolve().parent
PREREGISTRATION = ROOT / "docs" / "H10_AMENDMENT_SPEKULATIONSPFAD.md"
#: Attempts survive the process, because each retry is a fresh one.
ATTEMPTS_FILE = OUTPUT / "amendment_4b_96_attempts.json"
IRONMULE = ROOT / ".worktrees" / "friday-optimizer-ironmule"
PATCHED_FILE = IRONMULE / "ironmule" / "runtime.py"

_spec = importlib.util.spec_from_file_location(
    "switch_point_measure", ROOT / "experiments" / "switch_point" / "measure.py"
)
h10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h10)

#: 96, not 64. H1.0's A/A over the four lengths reads `1.131 %`, `1.040 %`,
#: `2.217 %`, `0.523 %`: 64 is the noisiest point of the series and 96 the
#: quietest, by a factor of four. The defect this patch removes — materialising
#: the whole KV cache once per iteration — also scales with the iteration count,
#: so 96 is where it would show most. And S1's own headline file measures 96
#: tokens, which puts both implementations side by side at one length.
#: The choice was made before any number from the repaired path existed: a
#: measurement regime picked for its noise, not a result picked for its sign.
TOKENS = 96
WIDTHS = (1, 2)
PAIRS = 6
#: Retry rule, frozen before the run. A `BudgetError` in the *warmup* — before a
#: single pair exists — may be retried, because the first run of a shape pays for
#: compilation and the on-disk Metal shader cache makes that a first-run effect
#: rather than a property of the code. A finished run is never retried, whatever
#: its number: that boundary is the whole point of the rule. Three warmups into
#: the same error is a finding about the patched path, not an artefact.
MAX_WARMUP_ATTEMPTS = 3

#: The two-line patch is a worktree change, so `EXPECTED_IRONMULE_HEAD` cannot
#: see it. Logging the hash is not enough: an unpatched run would measure the
#: unchanged path, write `ironmule_worktree_dirty: false` into a file nobody
#: rereads, and hand back a clean-looking "loses too" — the exact evidence H1.0's
#: kill would then rest on. So the hash is pinned and the run fails closed.
#: A substring check would not do: `mx.synchronize()` and `*_leaves(state)`
#: appear five more times in the same file, in `_prefill` and `_decode`.
UNPATCHED_SHA256 = "9d30965eb7073771f2620fae7cb0cd42d799ca047bb59e4661f82d71b98a9f3b"
PATCHED_SHA256 = "1252f53891800dfa4efecb3cf135523452cf1dedc41296cecb14361302062a9d"


def attempt_number(attempts: list[dict], sha256: str, width: int) -> int:
    """Which attempt this is for *this* file and *this* width.

    Counting the whole log would be wrong in one direction only: a reworked
    patch, or an abort at another width, would spend the three attempts of a
    cell that never ran, and "not measured" points the same way as the kill.
    """

    prior = [item for item in attempts
             if item.get("runtime_py_sha256") == sha256 and item.get("width") == width]
    return len(prior) + 1


def worktree_state() -> dict:
    """What actually ran: the head, whether the tree is dirty, and the file hash."""

    status = subprocess.run(
        ["/usr/bin/git", "-C", str(IRONMULE), "status", "--porcelain"],
        capture_output=True, text=True, check=False, timeout=10,
    ).stdout.strip()
    return {
        "ironmule_worktree_dirty": bool(status),
        "ironmule_worktree_status": status.splitlines(),
        "runtime_py_sha256": hashlib.sha256(PATCHED_FILE.read_bytes()).hexdigest(),
    }


def self_check() -> int:
    state = worktree_state()
    print(json.dumps({
        "state": "self_check", "study_id": STUDY_ID, "tokens": TOKENS,
        "widths": list(WIDTHS), "pairs": PAIRS, "baseline_arm": h10.COMBINED,
        "aa_reused_from": "experiments/switch_point/aa_4b_96.json",
        "terminal_gate": "exact token identity per pair",
        **state, "formal_claim": False, "no_activation": True,
    }, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--self-check", action="store_true", dest="self_check")
    parser.add_argument("--mde", type=float, required=False,
                        help="the 96-token regime's measured A/A spread")
    parser.add_argument("--widths", default="")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    from _bench import release_gate, study_provenance

    gate = release_gate(args, self_check)
    if gate is not None:
        return gate
    if args.mde is None or not 0.0 < args.mde < 1.0:
        raise SystemExit("--mde must be the measured A/A spread of the 96-token regime")
    # Fail closed before the first GPU call, in both directions: an unpatched
    # run cannot masquerade as the amendment, and a rerun after the worktree
    # reset cannot either.
    state = worktree_state()
    if state["runtime_py_sha256"] != PATCHED_SHA256:
        which = ("the unpatched file" if state["runtime_py_sha256"] == UNPATCHED_SHA256
                 else "an unknown version")
        raise SystemExit(
            f"refusing to run: ironmule/runtime.py is {which} "
            f"({state['runtime_py_sha256']}), expected the patched "
            f"{PATCHED_SHA256}. Apply the two-line patch from "
            "docs/H10_AMENDMENT_SPEKULATIONSPFAD.md first."
        )

    from friday_evidence.budget import BudgetError
    from friday_optimizer.integration import evaluate_integration

    from friday_calibrate.runner import MODELS, build_runner, paired_arms

    attempts = json.loads(ATTEMPTS_FILE.read_text()) if ATTEMPTS_FILE.exists() else []

    model_id = MODELS["4b"]
    runner, identity, guard = build_runner(
        PAIRS, model_id=model_id, output_tokens=TOKENS, prompt_tokens=897,
    )
    started = time.time()
    widths = tuple(int(w) for w in args.widths.split(",") if w) or WIDTHS
    workload = f"sealed-{identity['prompt_tokens']}-{TOKENS}"

    rows = []
    for width in widths:
        knobs = dict(h10.COMBINED, speculate_k=width, speculate_ngram=h10.NGRAM)
        attempt = attempt_number(attempts, state["runtime_py_sha256"], width)
        try:
            # The warmup pays for `mx.compile`, and the patched loop is a shape
            # the on-disk Metal shader cache has never seen. A `BudgetError`
            # *here* — before a single pair exists — is the one case the
            # preregistered retry rule covers.
            runner(h10.COMBINED)
            runner(knobs)
        except BudgetError as error:
            attempts.append({"attempt": attempt, "width": width,
                             "runtime_py_sha256": state["runtime_py_sha256"],
                             "outcome": "budget_error_in_warmup",
                             "reason": str(error), "budget": guard.summary()})
            ATTEMPTS_FILE.write_text(json.dumps(attempts, indent=2, sort_keys=True))
            if attempt >= MAX_WARMUP_ATTEMPTS:
                print(json.dumps({
                    "study_id": STUDY_ID, "state": "not_measured",
                    "verdict": "warmup_budget_error_persistent",
                    "width": width, "attempts": attempt, "attempt_log": attempts,
                    "note": ("three warmups in the same BudgetError is not a compile "
                             "artefact but a finding about the patched path"),
                }, indent=2, sort_keys=True))
                return 4
            print(json.dumps({"state": "warmup_budget_error", "width": width,
                              "attempt": attempt,
                              "hint": "rerun; the retry rule covers this case only"}))
            return 3
        try:
            baseline, candidate, breaks = paired_arms(
                runner, knobs, pairs=PAIRS, workload=workload,
                baseline_knobs=h10.COMBINED,
            )
        except BudgetError as error:
            # Past the warmup a budget stop is terminal. Retrying here would be
            # retrying a measurement, which is exactly what the rule forbids.
            rows.append({"width": width, "verdict": "budget_exceeded",
                         "reason": str(error), "budget": guard.summary()})
            break
        except (MemoryError, RuntimeError, ValueError) as error:
            # Dropping the state evaluation can let the graph grow. A failure
            # here is a result, not a crash to be retried away.
            rows.append({"width": width, "verdict": "run_failed",
                         "reason": f"{type(error).__name__}: {error}"})
            break
        if breaks:
            rows.append({"width": width, "verdict": "identity_break",
                         "reason": breaks[0], "pairs": len(baseline)})
            break
        result = evaluate_integration(
            baseline, candidate, arm="warm", min_gain=args.mde, mde=args.mde,
            min_pairs=max(2, PAIRS // 2),
        )
        low, high = result.ci or (None, None)
        rows.append({
            "width": width, "attempt": attempt,
            "verdict": h10.verdict_for(result.status),
            "status": result.status, "pairs": result.pairs,
            "ratio_median": result.ratio_median, "gain_percent": result.gain_percent,
            "ci": None if result.ci is None else [low, high],
            "token_identical": True, "reasons": list(result.reasons),
        })

    winners = [row["width"] for row in rows if row["verdict"] == "wins"]
    losers = [row["width"] for row in rows if row["verdict"] == "loses"]
    report = {
        "study_id": STUDY_ID, "state": "measured", "model": "4b", **identity,
        "baseline_arm": h10.COMBINED, "workload": workload,
        "aa_spread": args.mde, "mde": args.mde, "pairs": PAIRS, "rows": rows,
        "winning_widths": winners, "losing_widths": losers,
        "attempt_log": attempts,
        "verdict": (
            "repaired_path_wins" if winners
            else "repaired_path_loses" if len(losers) == len(rows) and rows
            else "inconclusive"
        ),
        **worktree_state(),
        "wall_seconds": round(time.time() - started, 1),
        "budget": guard.summary(),
        "provenance": study_provenance(
            [Path(__file__), ROOT / "experiments" / "switch_point" / "measure.py",
             ROOT / "friday_calibrate" / "runner.py", PATCHED_FILE],
            preregistration=PREREGISTRATION,
            extra={"model_id": model_id, "model_revision": identity["model_revision"]},
        ),
        "formal_claim": False, "no_activation": True,
    }
    destination = args.out or OUTPUT / "amendment_4b_96.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in report.items() if k != "provenance"},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
