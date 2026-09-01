"""W1 decision rule: does the decode rate hold over a realistic answer?

Pure analysis. No model framework, no device, no file access, so it can be
tested offline and reviewed before the gated run.

The whole regime argument rests on one unmeasured assumption: that decode
throughput stays roughly constant as the KV cache grows. If it degrades, the
prefill share falls faster than the model predicts and the candidate ordering
flips earlier.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

ANALYSIS_SCHEMA = "friday.regime.w1.v1"

#: Preregistered before the measurement. A decode rate within this band of the
#: short-answer control counts as constant for the purposes of the model.
RATE_TOLERANCE = 0.10

VERDICTS = ("rate_stable", "rate_degrades", "rate_improves", "inconclusive")


class RegimeError(ValueError):
    """Malformed regime evidence."""


def _positive(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0.0:
        raise RegimeError(f"{field} must be a positive finite number")
    return float(value)


def request_ratio(*, ttft: float, tokens: int, decode_tps: float,
                  ttft_ratio: float = 1.0, decode_tps_ratio: float = 1.0) -> float:
    """Request-time ratio; the same composition the integration module uses."""

    ttft = _positive(ttft, "ttft")
    decode_tps = _positive(decode_tps, "decode_tps")
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
        raise RegimeError("tokens must be a positive integer")
    baseline = ttft + tokens / decode_tps
    candidate = ttft * _positive(ttft_ratio, "ttft_ratio") + tokens / (decode_tps * _positive(decode_tps_ratio, "decode_tps_ratio"))
    return candidate / baseline


@dataclass(frozen=True, slots=True)
class RegimeVerdict:
    """What the measured long-answer rate says about the model."""

    verdict: str
    control_tps: float
    long_tps: float
    relative_change: float
    predicted_long_tps: float
    tolerance: float
    leader: str
    head_skip_gain: float
    fixed_compiled_gain: float
    combined_gain: float
    tokens: int

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise RegimeError("unknown verdict")

    @property
    def model_holds(self) -> bool:
        return self.verdict in ("rate_stable", "rate_improves")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": ANALYSIS_SCHEMA, "verdict": self.verdict,
            "control_tps": self.control_tps, "long_tps": self.long_tps,
            "relative_change": self.relative_change,
            "predicted_long_tps": self.predicted_long_tps,
            "tolerance": self.tolerance, "tokens": self.tokens,
            "leader": self.leader, "head_skip_gain": self.head_skip_gain,
            "fixed_compiled_gain": self.fixed_compiled_gain,
            "combined_gain": self.combined_gain,
            "model_holds": self.model_holds, "formal_claim": False,
        }


def classify(
    *,
    ttft: float,
    control_tps: float,
    long_tps: float,
    tokens: int,
    head_skip_ttft_ratio: float = 0.846385,
    fixed_compiled_decode_ratio: float = 0.9295921887,
    tolerance: float = RATE_TOLERANCE,
) -> RegimeVerdict:
    """Classify the measured long-answer decode rate and rank the candidates."""

    control = _positive(control_tps, "control_tps")
    long = _positive(long_tps, "long_tps")
    change = (long - control) / control
    if abs(change) <= tolerance:
        verdict = "rate_stable"
    elif change < 0:
        verdict = "rate_degrades"
    else:
        verdict = "rate_improves"
    shared = {"ttft": ttft, "tokens": tokens, "decode_tps": long}
    head_skip = 1.0 - request_ratio(**shared, ttft_ratio=head_skip_ttft_ratio)
    fixed = 1.0 - request_ratio(**shared, decode_tps_ratio=1.0 / fixed_compiled_decode_ratio)
    combined = 1.0 - request_ratio(**shared, ttft_ratio=head_skip_ttft_ratio,
                                   decode_tps_ratio=1.0 / fixed_compiled_decode_ratio)
    return RegimeVerdict(
        verdict=verdict, control_tps=control, long_tps=long, relative_change=change,
        predicted_long_tps=control, tolerance=tolerance,
        leader="head_skip_prefill" if head_skip > fixed else "fixed_compiled_cache",
        head_skip_gain=head_skip, fixed_compiled_gain=fixed, combined_gain=combined,
        tokens=tokens,
    )


__all__ = ["ANALYSIS_SCHEMA", "RATE_TOLERANCE", "VERDICTS", "RegimeError",
           "RegimeVerdict", "classify", "request_ratio"]
