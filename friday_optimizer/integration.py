"""F1 — preregistered end-to-end composition of already confirmed gains.

The project has confirmed three gains separately and integrated none of them.
This module is the missing analysis half: it turns paired per-phase samples
into **one request-level number** and decides it against a threshold that is
fixed before any data exists.

It is analysis only.  It starts no model, probes no hardware, activates no
profile, and writes nothing.  Pairing validation, bootstrap and percentile
come from :mod:`friday_optimizer.evaluator`; nothing statistical is
reimplemented here (four divergent ``statistics.py`` copies are exactly the
history this package is not repeating).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .canonical import canonical_bytes, sha256_hex
from .evaluator import (
    EvaluationError,
    MetricSample,
    _bootstrap,
    _pair_ratios_checked,
    _percentile,
)


INTEGRATION_SCHEMA = "friday.optimizer.integration-f1.v1"

#: The two scenarios F1 measures.  They differ only in whether the model
#: process is reused; every other knob is identical.
ARMS = ("cold", "warm")

STATUSES = ("qualified", "below_threshold", "inconclusive", "rejected")

#: Published per-phase findings this study composes.  Each is a ratio of
#: candidate over baseline, taken verbatim from its terminal study record.
#: They are inputs to a projection, never evidence for the integrated claim.
CONFIRMED_RATIOS = MappingProxyType({
    "head_skip_prefill": 0.846385,        # prefill/TTFT time, formal
    "fixed_compiled_cache": 0.9295921887,  # decode time, engineering
    "persistent_process": 0.346968,        # whole request, engineering, cold path
})

MIN_PAIRS = 6
DEFAULT_RESAMPLES = 2_000


class IntegrationError(ValueError):
    """Malformed integration input."""


def _finite_positive(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        return None
    return number


def request_seconds(sample: MetricSample) -> float | None:
    """Return one request's wall time, or ``None`` when evidence is missing.

    ``ttft + tokens / decode_tps`` is the whole request: the prefill the user
    waits for plus the decode they read.  A sample missing any part is not
    silently completed — it drops out of the paired set instead.
    """

    if not isinstance(sample, MetricSample) or sample.status != "ok":
        return None
    ttft = _finite_positive(sample.ttft_seconds)
    decode = _finite_positive(sample.decode_tps)
    tokens = sample.tokens
    if ttft is None or decode is None:
        return None
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
        return None
    return ttft + tokens / decode


def prefill_share(*, ttft_seconds: float, tokens: int, decode_tps: float) -> float:
    """Fraction of request time spent in prefill.

    This is the lever that decides whether a prefill gain matters at all: at a
    5 % share, a 15 % prefill gain is worth 0.8 % end to end.
    """

    ttft = _finite_positive(ttft_seconds)
    decode = _finite_positive(decode_tps)
    if ttft is None or decode is None or isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
        raise IntegrationError("prefill share needs a positive ttft, token count and decode rate")
    return ttft / (ttft + tokens / decode)


def project_request_ratio(
    *,
    ttft_seconds: float,
    tokens: int,
    decode_tps: float,
    ttft_ratio: float = 1.0,
    decode_tps_ratio: float = 1.0,
) -> float:
    """Predict the request-time ratio from per-phase ratios.

    Phase gains do **not** multiply: a prefill gain and a decode gain act on
    different parts of the same request, so the composed effect is the
    time-weighted mean, not the product.  This projection is what F1 exists to
    falsify; it is never itself a claim.
    """

    ttft = _finite_positive(ttft_seconds)
    decode = _finite_positive(decode_tps)
    prefill_ratio = _finite_positive(ttft_ratio)
    throughput_ratio = _finite_positive(decode_tps_ratio)
    if None in (ttft, decode, prefill_ratio, throughput_ratio):
        raise IntegrationError("projection needs positive finite inputs")
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
        raise IntegrationError("projection needs a positive token count")
    baseline = ttft + tokens / decode
    candidate = ttft * prefill_ratio + tokens / (decode * throughput_ratio)
    return candidate / baseline


def paired_request_ratios(
    baseline: Sequence[MetricSample], candidate: Sequence[MetricSample]
) -> tuple[list[float], tuple[str, ...]]:
    """Return per-pair request-time ratios under the evaluator's pairing rules.

    Pairing, arm identity, order balance and duplicate detection are delegated
    to the evaluator so a pair that the A/B gate rejects can never survive here.
    """

    _, _, reasons = _pair_ratios_checked(baseline, candidate)
    if reasons:
        return [], reasons
    left = {(sample.pair_id, sample.session_id): sample for sample in baseline}
    right = {(sample.pair_id, sample.session_id): sample for sample in candidate}
    ratios: list[float] = []
    for key in sorted(left):
        base = request_seconds(left[key])
        cand = request_seconds(right[key])
        if base is None or cand is None:
            continue
        ratios.append(cand / base)
    if not ratios:
        return [], ("request_evidence_incomplete",)
    return ratios, ()


@dataclass(frozen=True, slots=True)
class IntegrationResult:
    """One arm's end-to-end verdict against its preregistered threshold."""

    arm: str
    status: str
    pairs: int
    ratio_median: float | None = None
    ci: tuple[float, float] | None = None
    min_gain: float = 0.0
    mde: float = 0.0
    reasons: tuple[str, ...] = ()
    projection: float | None = None
    evidence_hash: str = ""

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise IntegrationError("unknown integration arm")
        if self.status not in STATUSES:
            raise IntegrationError("unknown integration status")

    @property
    def qualified(self) -> bool:
        return self.status == "qualified"

    @property
    def gain_percent(self) -> float | None:
        """Measured end-to-end speedup in percent; negative means slower."""

        return None if self.ratio_median is None else (1.0 - self.ratio_median) * 100.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": INTEGRATION_SCHEMA,
            "arm": self.arm,
            "status": self.status,
            "pairs": self.pairs,
            "ratio_median": self.ratio_median,
            "ci": None if self.ci is None else list(self.ci),
            "gain_percent": self.gain_percent,
            "min_gain": self.min_gain,
            "mde": self.mde,
            "reasons": list(self.reasons),
            "projection": self.projection,
            "qualified": self.qualified,
            "no_activation": True,
            "formal_claim": False,
            "evidence_hash": self.evidence_hash,
        }


def evaluate_integration(
    baseline: Sequence[MetricSample],
    candidate: Sequence[MetricSample],
    *,
    arm: str,
    min_gain: float,
    mde: float,
    projection: float | None = None,
    min_pairs: int = MIN_PAIRS,
    seed: int = 11,
    resamples: int = DEFAULT_RESAMPLES,
) -> IntegrationResult:
    """Decide one arm against its preregistered end-to-end threshold.

    Fail-closed in the same direction as the A/B evaluator: a confirmed
    regression is ``rejected``, a threshold that is merely not reached is
    ``below_threshold`` and keeps the baseline, and thin or broken evidence is
    ``inconclusive``.  Only a confidence interval wholly beyond the threshold
    qualifies.
    """

    if arm not in ARMS:
        raise IntegrationError("unknown integration arm")
    for name, value in (("min_gain", min_gain), ("mde", mde)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0.0 <= float(value) < 1.0:
            raise IntegrationError(f"{name} must be a finite fraction within [0, 1)")
    if isinstance(min_pairs, bool) or not isinstance(min_pairs, int) or min_pairs < 1:
        raise IntegrationError("min_pairs must be a positive integer")
    ratios, reasons = paired_request_ratios(baseline, candidate)
    evidence = {
        "schema": INTEGRATION_SCHEMA, "arm": arm, "min_gain": float(min_gain), "mde": float(mde),
        "baseline": [sample.as_dict() if isinstance(sample, MetricSample) else repr(sample) for sample in baseline],
        "candidate": [sample.as_dict() if isinstance(sample, MetricSample) else repr(sample) for sample in candidate],
    }
    evidence_hash = sha256_hex(canonical_bytes(evidence, max_bytes=64 * 1024 * 1024, max_items=2_000_000))
    if reasons:
        hard = any(reason.startswith("ab_") for reason in reasons)
        return IntegrationResult(arm, "rejected" if hard else "inconclusive", 0, min_gain=float(min_gain),
                                 mde=float(mde), reasons=reasons, projection=projection,
                                 evidence_hash=evidence_hash)
    if len(ratios) < min_pairs:
        return IntegrationResult(arm, "inconclusive", len(ratios), min_gain=float(min_gain), mde=float(mde),
                                 reasons=("integration_requires_more_paired_sessions",), projection=projection,
                                 evidence_hash=evidence_hash)
    try:
        low, high = _bootstrap(ratios, seed=seed, resamples=resamples)
    except EvaluationError as exc:
        raise IntegrationError(str(exc)) from exc
    median = _percentile(ratios, 0.5)
    if low > 1.0 + mde:
        status, verdict = "rejected", ("statistically_confirmed_request_regression",)
    elif high < 1.0 - min_gain:
        status, verdict = "qualified", ()
    elif high > 1.0 + mde:
        status, verdict = "inconclusive", ("confidence_interval_crosses_adverse_gate",)
    else:
        status, verdict = "below_threshold", ("request_gain_below_preregistered_threshold",)
    return IntegrationResult(arm, status, len(ratios), median, (low, high), float(min_gain), float(mde),
                             verdict, projection, evidence_hash)


__all__ = [
    "ARMS",
    "CONFIRMED_RATIOS",
    "DEFAULT_RESAMPLES",
    "INTEGRATION_SCHEMA",
    "MIN_PAIRS",
    "STATUSES",
    "IntegrationError",
    "IntegrationResult",
    "evaluate_integration",
    "paired_request_ratios",
    "prefill_share",
    "project_request_ratio",
    "request_seconds",
]
