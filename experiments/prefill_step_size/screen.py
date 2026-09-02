"""Offline pre-screen for candidate 5: does this prompt family have a degenerate position?

Candidates 1 and 2 did not fail on their mechanism. P2 established that on
2026-09-02 (`tie_hypothesis_supported`): both chunkings flipped at position
`10`, where the top-2 logit gap is `0.500` — the smallest of the whole sequence,
median `4.0` — while chunking perturbs the logits by `2.25`-`2.50`. A gap that
small cannot survive a perturbation that large, and that is a property of the
*workload*, not of chunking.

So the prerequisite for measuring again is a prompt family without such a
position, and that is checkable before any measurement: run the reference
prefill once, read the top-2 gap at every generated position, and compare the
smallest one against the perturbation chunking is known to introduce. Filter
first, measure second.

This module is pure analysis — no model, no device, no file — so it is tested
offline and reviewable before a gated run. It reuses the decision rule from
``experiments/identity_forensics/gap_analysis.py`` rather than restating it.
"""

from __future__ import annotations

import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "identity_forensics"))

from gap_analysis import TIE_MARGIN, GapError  # noqa: E402

SCREEN_SCHEMA = "friday.prefill-step-size.screen.v1"

#: The perturbation chunked prefill introduces on this model, measured in
#: experiments/divergence_source and confirmed by P2. Not a guess and not a
#: tolerance: it is what the reordering of the same arithmetic actually costs.
MEASURED_PERTURBATION = 2.50

#: A position is degenerate when its top-2 gap is not comfortably larger than
#: the perturbation. `TIE_MARGIN` is the same slack the tie rule uses, so a
#: position this screen admits is one the tie rule would have called structural
#: had it flipped.
def degenerate_threshold(perturbation: float = MEASURED_PERTURBATION) -> float:
    return perturbation + TIE_MARGIN


VERDICTS = ("admissible", "degenerate", "insufficient_evidence")


@dataclass(frozen=True)
class ScreenResult:
    verdict: str
    positions: int
    min_gap: float | None
    median_gap: float | None
    threshold: float
    degenerate_positions: tuple[int, ...]
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCREEN_SCHEMA,
            "verdict": self.verdict,
            "positions": self.positions,
            "min_gap": self.min_gap,
            "median_gap": self.median_gap,
            "threshold": self.threshold,
            "degenerate_positions": list(self.degenerate_positions),
            "reason": self.reason,
            "admissible": self.verdict == "admissible",
            "formal_claim": False,
        }


def _gap(entry: Mapping[str, Any], index: int) -> float:
    if not isinstance(entry, Mapping):
        raise GapError(f"gap entry {index} is not an object")
    value = entry.get("gap")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GapError(f"gap entry {index} has no numeric gap")
    gap = float(value)
    if gap < 0.0:
        raise GapError(f"gap entry {index} is negative")
    return gap


def screen(
    reference_gaps: Sequence[Mapping[str, Any]],
    *,
    perturbation: float = MEASURED_PERTURBATION,
    minimum_positions: int = 8,
) -> ScreenResult:
    """Decide whether a prompt family may enter a step-size study.

    Refusing is the cheap outcome. Admitting a family with a degenerate position
    costs a gated hardware run that ends in an identity break which says nothing
    about chunking — which is precisely what happened to candidates 1 and 2.
    """

    threshold = degenerate_threshold(perturbation)
    if not isinstance(reference_gaps, Sequence) or isinstance(reference_gaps, (str, bytes)):
        raise GapError("reference_gaps must be a sequence")
    gaps = [_gap(entry, index) for index, entry in enumerate(reference_gaps)]
    if len(gaps) < minimum_positions:
        return ScreenResult(
            verdict="insufficient_evidence",
            positions=len(gaps),
            min_gap=min(gaps) if gaps else None,
            median_gap=statistics.median(gaps) if gaps else None,
            threshold=threshold,
            degenerate_positions=(),
            reason=f"fewer than {minimum_positions} positions measured",
        )
    degenerate = tuple(
        int(reference_gaps[index].get("position", index))
        for index, gap in enumerate(gaps)
        if gap < threshold
    )
    return ScreenResult(
        verdict="degenerate" if degenerate else "admissible",
        positions=len(gaps),
        min_gap=min(gaps),
        median_gap=statistics.median(gaps),
        threshold=threshold,
        degenerate_positions=degenerate,
        reason=(
            f"{len(degenerate)} position(s) with a top-2 gap below {threshold}"
            if degenerate
            else ""
        ),
    )


def _self_check() -> int:
    """Replay P2's own prompt: the screen must reject the family that broke."""

    import json

    path = Path(__file__).resolve().parents[1] / "identity_forensics" / "logit_gap.json"
    payload = json.loads(path.read_text())
    result = screen(payload["reference_gaps"])
    assert result.verdict == "degenerate", result
    assert 10 in result.degenerate_positions, result
    assert result.min_gap == 0.5, result
    print(json.dumps({"state": "self_check", "p2_prompt": result.as_dict()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_check())
