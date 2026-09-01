"""Decision rule for P2: is a token-identity divergence a tie or a defect?

Pure analysis. It imports no model framework, touches no device and reads no
file, so it can be tested offline and reviewed before any gated run.

The thresholds here classify a *hypothesis about a measurement*. They are not
part of the token-identity gate and cannot relax it: a flipped argmax stays an
identity failure whatever this module concludes.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

ANALYSIS_SCHEMA = "friday.identity.gap-analysis.v1"

#: Preregistered before the measurement, deliberately in two independent
#: forms so neither scale nor outliers alone can decide.
TIE_ABSOLUTE = 1e-2      # logit units at the divergence position
TIE_RATIO = 20.0         # median gap over all positions / gap at divergence

#: A divergence this early cannot be a late tie: the state is already wrong.
STRUCTURAL_POSITION = 1

VERDICTS = ("tie", "structural", "inconclusive", "no_divergence")


class GapError(ValueError):
    """Malformed gap evidence."""


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise GapError(f"{field} must be a finite number")
    return float(value)


@dataclass(frozen=True, slots=True)
class GapVerdict:
    """The classification, with every number that produced it."""

    verdict: str
    first_diff: int | None
    divergence_gap: float | None
    median_gap: float | None
    minimum_gap: float | None
    ratio: float | None
    is_minimum: bool
    positions: int
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise GapError("unknown verdict")

    @property
    def supports_tie_hypothesis(self) -> bool:
        return self.verdict == "tie"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": ANALYSIS_SCHEMA,
            "verdict": self.verdict,
            "first_diff": self.first_diff,
            "divergence_gap": self.divergence_gap,
            "median_gap": self.median_gap,
            "minimum_gap": self.minimum_gap,
            "ratio": self.ratio,
            "is_minimum": self.is_minimum,
            "positions": self.positions,
            "reasons": list(self.reasons),
            "tie_absolute": TIE_ABSOLUTE,
            "tie_ratio": TIE_RATIO,
            "gate_unchanged": True,
        }


def classify(
    gaps: Sequence[Mapping[str, Any]],
    first_diff: int | None,
    *,
    tie_absolute: float = TIE_ABSOLUTE,
    tie_ratio: float = TIE_RATIO,
) -> GapVerdict:
    """Classify one divergence against the preregistered thresholds.

    ``gaps`` is the per-position top-1 minus top-2 logit distance of the
    reference run. ``first_diff`` is the position at which the variant's tokens
    first differ, or ``None`` when the runs were identical.
    """

    values = [_finite(row.get("gap"), f"gap[{index}]") for index, row in enumerate(gaps)]
    if not values:
        raise GapError("gap evidence is empty")
    if any(value < 0.0 for value in values):
        raise GapError("a top-1 minus top-2 distance cannot be negative")
    median = statistics.median(values)
    minimum = min(values)
    if first_diff is None:
        return GapVerdict("no_divergence", None, None, median, minimum, None, False, len(values),
                          ("runs_were_identical",))
    if isinstance(first_diff, bool) or not isinstance(first_diff, int) or not 0 <= first_diff < len(values):
        raise GapError("first_diff is outside the recorded positions")
    divergence = values[first_diff]
    ratio = (median / divergence) if divergence > 0.0 else math.inf
    is_minimum = divergence <= minimum + 0.0

    reasons: list[str] = []
    if first_diff <= STRUCTURAL_POSITION:
        reasons.append("diverges_at_or_before_position_1")
        return GapVerdict("structural", first_diff, divergence, median, minimum, ratio,
                          is_minimum, len(values), tuple(reasons))
    absolute_ok = divergence <= tie_absolute
    ratio_ok = ratio >= tie_ratio
    if not absolute_ok:
        reasons.append("divergence_gap_above_absolute_threshold")
    if not ratio_ok:
        reasons.append("divergence_gap_not_small_against_the_median")
    if absolute_ok and ratio_ok:
        return GapVerdict("tie", first_diff, divergence, median, minimum, ratio, is_minimum,
                          len(values), ())
    if not absolute_ok and not ratio_ok:
        # Large in both forms: the model genuinely changed its mind, so the
        # mechanism is what broke, not the workload.
        return GapVerdict("structural", first_diff, divergence, median, minimum, ratio,
                          is_minimum, len(values), tuple(reasons))
    return GapVerdict("inconclusive", first_diff, divergence, median, minimum, ratio,
                      is_minimum, len(values), tuple(reasons))


def summarise(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate several variant verdicts into one campaign answer."""

    verdicts = []
    for entry in results:
        verdict = entry.get("verdict")
        if verdict not in VERDICTS:
            raise GapError("result carries no valid verdict")
        verdicts.append(verdict)
    diverged = [value for value in verdicts if value != "no_divergence"]
    counts = {name: verdicts.count(name) for name in VERDICTS if verdicts.count(name)}
    if not diverged:
        answer = "no_divergence_reproduced"
    elif all(value == "tie" for value in diverged):
        answer = "tie_hypothesis_supported"
    elif any(value == "structural" for value in diverged):
        answer = "tie_hypothesis_rejected"
    else:
        answer = "inconclusive"
    return {
        "schema": ANALYSIS_SCHEMA,
        "answer": answer,
        "counts": counts,
        "variants": len(verdicts),
        "diverged": len(diverged),
        "gate_unchanged": True,
        "formal_claim": False,
    }


__all__ = [
    "ANALYSIS_SCHEMA",
    "STRUCTURAL_POSITION",
    "TIE_ABSOLUTE",
    "TIE_RATIO",
    "VERDICTS",
    "GapError",
    "GapVerdict",
    "classify",
    "summarise",
]
