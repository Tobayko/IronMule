"""Deterministic, fail-closed offline A/A and A/B evaluator.

The evaluator is deliberately a statistics-and-gates component, not a tuner.
It consumes immutable raw observations, keeps failed observations in its
evidence hash, and can only return a shadow decision.  It never activates a
profile or executes a model.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from .fingerprint import ExactFingerprint
from .candidates import CANDIDATE_IDS, CandidateError, CandidateRegistry


class EvaluationError(ValueError):
    """Malformed evaluator input."""


MAX_RAW_SAMPLES = 10_000


def _finite(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise EvaluationError(f"{field_name} must be a finite number")
    return float(value)


def _optional_finite(value: Any, field_name: str) -> float | None:
    return None if value is None else _finite(value, field_name)


def _positive(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and value > 0


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    return value


def _canonical(value: Any) -> bytes:
    try:
        result = json.dumps(_thaw(value), sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise EvaluationError("evidence is not canonical") from exc
    if len(result) > 4 * 1024 * 1024:
        raise EvaluationError("evidence exceeds the size bound")
    return result


@dataclass(frozen=True, slots=True, init=False)
class MetricSample:
    """One raw measurement.  Negative finite values are retained as invalid evidence."""

    session_id: str = ""
    pair_id: str = ""
    arm: str = ""
    order: str = ""
    fingerprint: str = ""
    workload: str = ""
    ttft_seconds: float | None = None
    decode_tps: float | None = None
    tokens: int | None = None
    status: str = "ok"
    error: str = ""

    def __init__(
        self,
        session_id: str = "",
        pair_id: str = "",
        arm: str = "",
        order: str = "",
        fingerprint: str = "",
        workload: str = "",
        ttft_seconds: float | None = None,
        decode_tps: float | None = None,
        tokens: int | None = None,
        status: str = "ok",
        error: str = "",
        *,
        ttft: float | None = None,
        ttft_ms: float | None = None,
        tokens_per_second: float | None = None,
    ) -> None:
        aliases = [value is not None for value in (ttft, ttft_ms)]
        if sum(aliases) > 1:
            raise EvaluationError("provide only one TTFT unit")
        if ttft is not None:
            if ttft_seconds is not None and float(ttft_seconds) != float(ttft):
                raise EvaluationError("ttft aliases disagree")
            ttft_seconds = ttft
        if ttft_ms is not None:
            if ttft_seconds is not None and float(ttft_seconds) != float(ttft_ms) / 1000:
                raise EvaluationError("ttft aliases disagree")
            ttft_seconds = float(ttft_ms) / 1000
        if tokens_per_second is not None:
            if decode_tps is not None and float(decode_tps) != float(tokens_per_second):
                raise EvaluationError("decode aliases disagree")
            decode_tps = tokens_per_second
        for name, value in {
            "session_id": session_id, "pair_id": pair_id, "arm": arm, "order": order,
            "fingerprint": fingerprint, "workload": workload,
            "ttft_seconds": ttft_seconds, "decode_tps": decode_tps, "tokens": tokens,
            "status": status, "error": error,
        }.items():
            object.__setattr__(self, name, value)
        self.__post_init__()

    def __post_init__(self) -> None:
        for field_name in ("session_id", "pair_id", "arm", "order", "fingerprint", "workload", "error"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or len(value) > 256:
                raise EvaluationError(f"{field_name} is invalid")
        if self.arm not in ("", "A", "B", "baseline", "candidate", "aa_left", "aa_right", "control"):
            raise EvaluationError("arm is invalid")
        if self.order not in ("", "AB", "BA"):
            raise EvaluationError("order is invalid")
        object.__setattr__(self, "ttft_seconds", _optional_finite(self.ttft_seconds, "ttft_seconds"))
        object.__setattr__(self, "decode_tps", _optional_finite(self.decode_tps, "decode_tps"))
        if self.tokens is not None and (isinstance(self.tokens, bool) or not isinstance(self.tokens, int) or self.tokens < 0):
            raise EvaluationError("tokens must be a non-negative integer")
        if not isinstance(self.status, str) or self.status not in {"ok", "invalid", "error", "timeout", "crash", "resource"}:
            raise EvaluationError("unknown sample status")

    @property
    def ttft(self) -> float | None:
        return self.ttft_seconds

    @property
    def decode_tokens_per_second(self) -> float | None:
        return self.decode_tps

    @property
    def valid_metrics(self) -> bool:
        return self.status == "ok" and _positive(self.ttft_seconds) and _positive(self.decode_tps)

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id, "pair_id": self.pair_id, "arm": self.arm, "order": self.order,
            "fingerprint": self.fingerprint, "workload": self.workload,
            "ttft_seconds": self.ttft_seconds, "decode_tps": self.decode_tps,
            "tokens": self.tokens, "status": self.status, "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class CorrectnessResult:
    """Exact output contract used before any speed comparison."""

    token_ids: tuple[int, ...] = ()
    text: str = ""
    stop_reason: str = ""
    physical_tokens: int = 0
    visible_tokens: int = 0
    response_hash: str = ""
    passed: bool = True
    error: str = ""

    def __post_init__(self) -> None:
        ids = tuple(self.token_ids)
        if any(isinstance(v, bool) or not isinstance(v, int) for v in ids) or len(ids) > 1_000_000:
            raise EvaluationError("token_ids are invalid")
        object.__setattr__(self, "token_ids", ids)
        for name in ("text", "stop_reason", "response_hash", "error"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) > 4 * 1024 * 1024:
                raise EvaluationError(f"{name} is invalid")
        for name in ("physical_tokens", "visible_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise EvaluationError(f"{name} is invalid")
        if not isinstance(self.passed, bool):
            raise EvaluationError("passed must be bool")
        if not self.response_hash:
            digest_body = {
                "token_ids": list(self.token_ids), "text": self.text, "stop_reason": self.stop_reason,
                "physical_tokens": self.physical_tokens, "visible_tokens": self.visible_tokens,
            }
            object.__setattr__(self, "response_hash", hashlib.sha256(_canonical(digest_body)).hexdigest())

    def as_dict(self) -> dict[str, Any]:
        return {
            "token_ids": list(self.token_ids), "text": self.text, "stop_reason": self.stop_reason,
            "physical_tokens": self.physical_tokens, "visible_tokens": self.visible_tokens,
            "response_hash": self.response_hash, "passed": self.passed, "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class ResourceResult:
    """Raw safety/resource result; failures are evidence, never silently dropped."""

    peak_memory_bytes: int | None = None
    peak_rss_bytes: int | None = None
    swap_delta_bytes: int | None = None
    timed_out: bool = False
    crashed: bool = False
    foreign_load: bool = False
    status: str = "unknown"
    error: str = ""

    def __post_init__(self) -> None:
        for name in ("peak_memory_bytes", "peak_rss_bytes", "swap_delta_bytes"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                raise EvaluationError(f"{name} must be an integer")
        for name in ("timed_out", "crashed", "foreign_load"):
            if not isinstance(getattr(self, name), bool):
                raise EvaluationError(f"{name} must be bool")
        if self.status not in {"unknown", "ok", "invalid", "error", "timeout", "crash", "resource"}:
            raise EvaluationError("unknown resource status")
        if not isinstance(self.error, str) or len(self.error) > 4096:
            raise EvaluationError("resource error is invalid")

    @property
    def complete(self) -> bool:
        """Whether all mandatory resource facts were actually observed."""
        return (
            self.status != "unknown"
            and self.peak_memory_bytes is not None
            and self.peak_rss_bytes is not None
            and self.swap_delta_bytes is not None
            and self.peak_memory_bytes >= 0
            and self.peak_rss_bytes >= 0
            and self.swap_delta_bytes >= 0
        )

    @property
    def invalid_values(self) -> bool:
        return any(value is not None and value < 0 for value in (self.peak_memory_bytes, self.peak_rss_bytes, self.swap_delta_bytes))

    @property
    def passed(self) -> bool:
        return (
            self.complete and self.status == "ok" and not self.timed_out and not self.crashed and not self.foreign_load
            and self.swap_delta_bytes == 0
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "peak_memory_bytes": self.peak_memory_bytes, "peak_rss_bytes": self.peak_rss_bytes,
            "swap_delta_bytes": self.swap_delta_bytes, "timed_out": self.timed_out,
            "crashed": self.crashed, "foreign_load": self.foreign_load,
            "status": self.status, "error": self.error,
        }


# Names used by callers that distinguish a sample from a run-level result.
MetricResult = MetricSample
Correctness = CorrectnessResult
ResourceSample = ResourceResult


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise EvaluationError("percentile needs observations")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _bootstrap(values: Sequence[float], *, seed: int, resamples: int) -> tuple[float, float]:
    if not values:
        raise EvaluationError("bootstrap needs observations")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise EvaluationError("bootstrap seed must be integer")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or not 100 <= resamples <= 10_000:
        raise EvaluationError("bootstrap resamples must be between 100 and 10000")
    rng = random.Random(seed)
    draws: list[float] = []
    count = len(values)
    for _ in range(resamples):
        draws.append(_percentile([values[rng.randrange(count)] for _ in range(count)], 0.5))
    return _percentile(draws, 0.025), _percentile(draws, 0.975)


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """Conservative A/A noise/MDE estimate."""

    pair_count: int
    ttft_noise_floor: float | None
    decode_noise_floor: float | None
    mde: float | None
    adequate: bool
    confidence_intervals: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()
    evidence_hash: str = ""
    pair_keys: tuple[tuple[str, str], ...] = ()
    order_schedule: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.pair_count, bool) or not isinstance(self.pair_count, int) or self.pair_count < 0:
            raise EvaluationError("pair_count invalid")
        for name in ("ttft_noise_floor", "decode_noise_floor", "mde"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not math.isfinite(float(value)) or value < 0):
                raise EvaluationError(f"{name} invalid")
        if not isinstance(self.adequate, bool):
            raise EvaluationError("adequate must be bool")
        object.__setattr__(self, "confidence_intervals", MappingProxyType(dict(self.confidence_intervals)))
        object.__setattr__(self, "reasons", tuple(str(reason) for reason in self.reasons))
        object.__setattr__(self, "pair_keys", tuple(tuple(key) for key in self.pair_keys))
        object.__setattr__(self, "order_schedule", tuple(self.order_schedule))


def _validate_aa_pairs(
    left_samples: Sequence[MetricSample],
    right_samples: Sequence[MetricSample],
) -> tuple[tuple[MetricSample, MetricSample], ...]:
    """Bind A/A arms by identity, never by list position."""
    if not left_samples or not right_samples:
        raise EvaluationError("A/A pairs must be non-empty")
    if len(left_samples) > MAX_RAW_SAMPLES or len(right_samples) > MAX_RAW_SAMPLES:
        raise EvaluationError("A/A sample bound exceeded")
    left: dict[tuple[str, str], MetricSample] = {}
    right: dict[tuple[str, str], MetricSample] = {}
    for sample in left_samples:
        if not isinstance(sample, MetricSample):
            raise EvaluationError("A/A values must be MetricSample")
        if not sample.pair_id or not sample.session_id or not sample.fingerprint or not sample.workload:
            raise EvaluationError("A/A identity is incomplete")
        if sample.arm not in {"aa_left", "baseline", "A"}:
            raise EvaluationError("A/A left arm is invalid")
        if sample.order not in {"AB", "BA"}:
            raise EvaluationError("A/A order is missing")
        key = (sample.pair_id, sample.session_id)
        if key in left:
            raise EvaluationError("duplicate A/A left pair identity")
        left[key] = sample
    for sample in right_samples:
        if not isinstance(sample, MetricSample):
            raise EvaluationError("A/A values must be MetricSample")
        if not sample.pair_id or not sample.session_id or not sample.fingerprint or not sample.workload:
            raise EvaluationError("A/A identity is incomplete")
        if sample.arm not in {"aa_right", "control", "B"}:
            raise EvaluationError("A/A right arm is invalid")
        if sample.order not in {"AB", "BA"}:
            raise EvaluationError("A/A order is missing")
        key = (sample.pair_id, sample.session_id)
        if key in right:
            raise EvaluationError("duplicate A/A right pair identity")
        right[key] = sample
    if set(left) != set(right):
        raise EvaluationError("A/A pair identities do not match")
    pairs: list[tuple[MetricSample, MetricSample]] = []
    identities: set[tuple[str, str]] = set()
    for key in sorted(left):
        one, two = left[key], right[key]
        if one.order != two.order:
            raise EvaluationError("A/A pair order mismatch")
        if one.fingerprint != two.fingerprint or one.workload != two.workload:
            raise EvaluationError("A/A fingerprint or workload mismatch")
        identities.add((one.fingerprint, one.workload))
        pairs.append((one, two))
    if len(identities) != 1:
        raise EvaluationError("A/A pairs span multiple fingerprints or workloads")
    orders = [one.order for one, _ in pairs]
    if set(orders) != {"AB", "BA"} or orders.count("AB") != orders.count("BA"):
        raise EvaluationError("A/A order schedule must be balanced AB/BA")
    return tuple(pairs)


def _aa_evidence(pairs: Sequence[tuple[MetricSample, MetricSample]], *, seed: int, resamples: int) -> dict[str, Any]:
    return {
        "pairs": [
            {"pair_id": left.pair_id, "session_id": left.session_id, "order": left.order,
             "left": left.as_dict(), "right": right.as_dict()}
            for left, right in pairs
        ],
        "seed": seed,
        "resamples": resamples,
    }


def calibrate_aa(
    *,
    aa_left: Sequence[MetricSample] | None = None,
    aa_right: Sequence[MetricSample] | None = None,
    baseline: Sequence[MetricSample] | None = None,
    control: Sequence[MetricSample] | None = None,
    preregistered_mde: float = 0.05,
    bootstrap_resamples: int = 1_000,
    seed: int = 20260830,
) -> CalibrationReport:
    """Estimate noise from paired A/A blocks; one block is never sufficient."""

    if aa_left is not None or aa_right is not None:
        if aa_left is None or aa_right is None or baseline is not None or control is not None:
            raise EvaluationError("provide exactly aa_left and aa_right")
        left_samples, right_samples = aa_left, aa_right
    else:
        if baseline is None or control is None:
            raise EvaluationError("A/A calibration requires keyword-only arms")
        left_samples, right_samples = baseline, control
    if not isinstance(preregistered_mde, (int, float)) or isinstance(preregistered_mde, bool) or not math.isfinite(float(preregistered_mde)) or not 0 < preregistered_mde <= 1:
        raise EvaluationError("preregistered_mde must be in (0, 1]")
    pairs = _validate_aa_pairs(left_samples, right_samples)
    ttft_logs: list[float] = []
    decode_logs: list[float] = []
    for left, right in pairs:
        if left.status != "ok" or right.status != "ok" or not _positive(left.ttft_seconds) or not _positive(right.ttft_seconds) or not _positive(left.decode_tps) or not _positive(right.decode_tps):
            continue
        ttft_logs.append(math.log(right.ttft_seconds / left.ttft_seconds))
        decode_logs.append(math.log(right.decode_tps / left.decode_tps))
    if len(ttft_logs) < 3 or len(decode_logs) < 3:
        return CalibrationReport(
            len(ttft_logs), None, None, None, False,
            reasons=("aa_requires_at_least_three_valid_paired_blocks",),
            evidence_hash=_evidence_hash(_aa_evidence(pairs, seed=seed, resamples=bootstrap_resamples)),
            pair_keys=tuple((left.pair_id, left.session_id) for left, _ in pairs),
            order_schedule=tuple(left.order for left, _ in pairs),
        )
    ttft_ci_log = _bootstrap(ttft_logs, seed=seed, resamples=bootstrap_resamples)
    decode_ci_log = _bootstrap(decode_logs, seed=seed + 1, resamples=bootstrap_resamples)
    ttft_ci = (math.exp(ttft_ci_log[0]), math.exp(ttft_ci_log[1]))
    decode_ci = (math.exp(decode_ci_log[0]), math.exp(decode_ci_log[1]))
    ttft_noise = max(abs(ttft_ci[0] - 1), abs(ttft_ci[1] - 1))
    decode_noise = max(abs(decode_ci[0] - 1), abs(decode_ci[1] - 1))
    mde = max(float(preregistered_mde), ttft_noise, decode_noise)
    evidence = _aa_evidence(pairs, seed=seed, resamples=bootstrap_resamples)
    return CalibrationReport(
        len(ttft_logs), ttft_noise, decode_noise, mde, True,
        {"ttft": ttft_ci, "decode_tps": decode_ci}, (), _evidence_hash(evidence),
        tuple((left.pair_id, left.session_id) for left, _ in pairs),
        tuple(left.order for left, _ in pairs),
    )


def _evidence_hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _correctness_reasons(baseline: CorrectnessResult | None, candidate: CorrectnessResult | None) -> list[str]:
    if baseline is None or candidate is None:
        return ["correctness_evidence_missing"]
    if not baseline.passed or not candidate.passed:
        return ["correctness_gate_failed"]
    reasons: list[str] = []
    for field_name in ("token_ids", "text", "stop_reason", "physical_tokens", "visible_tokens", "response_hash"):
        if getattr(baseline, field_name) != getattr(candidate, field_name):
            reasons.append(f"correctness_mismatch:{field_name}")
    return reasons


def _pair_ratios_checked(baseline: Sequence[MetricSample], candidate: Sequence[MetricSample]) -> tuple[list[float], list[float], tuple[str, ...]]:
    if not baseline or not candidate:
        return [], [], ("ab_pairs_empty",)
    if len(baseline) != len(candidate):
        return [], [], ("ab_pair_count_mismatch",)
    reasons: list[str] = []
    left_keys: dict[tuple[str, str], MetricSample] = {}
    right_keys: dict[tuple[str, str], MetricSample] = {}
    for sample, target, side in ((item, left_keys, "baseline") for item in baseline):
        if not isinstance(sample, MetricSample) or not sample.pair_id or not sample.session_id:
            reasons.append("ab_identity_missing")
            continue
        if sample.arm not in ("A", "baseline"):
            reasons.append("ab_baseline_arm_mismatch")
        if sample.order not in ("AB", "BA"):
            reasons.append("ab_order_missing")
        key = (sample.pair_id, sample.session_id)
        if key in target:
            reasons.append("ab_duplicate_baseline_pair")
        target[key] = sample
    for sample, target, side in ((item, right_keys, "candidate") for item in candidate):
        if not isinstance(sample, MetricSample) or not sample.pair_id or not sample.session_id:
            reasons.append("ab_identity_missing")
            continue
        if sample.arm not in ("B", "candidate"):
            reasons.append("ab_candidate_arm_mismatch")
        if sample.order not in ("AB", "BA"):
            reasons.append("ab_order_missing")
        key = (sample.pair_id, sample.session_id)
        if key in target:
            reasons.append("ab_duplicate_candidate_pair")
        target[key] = sample
    if set(left_keys) != set(right_keys):
        reasons.append("ab_pair_identity_mismatch")
    for key in set(left_keys) & set(right_keys):
        left, right = left_keys[key], right_keys[key]
        if left.order != right.order:
            reasons.append("ab_pair_order_mismatch")
        if left.arm in ("A", "B") or right.arm in ("A", "B"):
            expected = ("A", "B") if left.order == "AB" else ("B", "A")
            if (left.arm, right.arm) != expected:
                reasons.append("ab_pair_arm_order_mismatch")
    orders = [sample.order for sample in tuple(left_keys.values()) + tuple(right_keys.values()) if sample.order]
    if not orders or set(orders) != {"AB", "BA"} or orders.count("AB") != orders.count("BA"):
        reasons.append("ab_order_unbalanced")
    if reasons:
        return [], [], tuple(dict.fromkeys(reasons))
    ttft: list[float] = []
    decode: list[float] = []
    for key in sorted(left_keys):
        left, right = left_keys[key], right_keys[key]
        if left.status == "ok" and right.status == "ok" and _positive(left.ttft_seconds) and _positive(right.ttft_seconds) and _positive(left.decode_tps) and _positive(right.decode_tps):
            ttft.append(right.ttft_seconds / left.ttft_seconds)
            decode.append(right.decode_tps / left.decode_tps)
    return ttft, decode, ()


def _normalise_aa_inputs(
    baseline: Sequence[MetricSample] | None,
    control: Sequence[MetricSample] | None,
    pairs: Sequence[tuple[MetricSample, MetricSample]] | None,
) -> tuple[Sequence[MetricSample] | None, Sequence[MetricSample] | None]:
    if pairs is not None:
        if baseline is not None or control is not None:
            raise EvaluationError("provide A/A pairs or A/A arms, not both")
        left: list[MetricSample] = []
        right: list[MetricSample] = []
        for pair in pairs:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise EvaluationError("each A/A pair must contain two MetricSample values")
            left.append(pair[0])
            right.append(pair[1])
        return tuple(left), tuple(right)
    if baseline is None and control is None:
        return None, None
    if baseline is None or control is None:
        raise EvaluationError("both A/A arms are required")
    return baseline, control


@dataclass(frozen=True, slots=True)
class ShadowDecision:
    """A recommendation record.  ``no_activation`` is always true by design."""

    fingerprint: str
    candidate_id: str
    status: str
    baseline_ratios: Mapping[str, float] = field(default_factory=dict)
    confidence_intervals: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    qualified: bool = False
    reasons: tuple[str, ...] = ()
    evidence_hash: str = ""
    no_activation: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.fingerprint, str) or len(self.fingerprint) > 256:
            raise EvaluationError("fingerprint invalid")
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise EvaluationError("candidate id invalid")
        if self.status not in {"qualified", "rejected", "inconclusive", "ood"}:
            raise EvaluationError("decision status invalid")
        if not isinstance(self.qualified, bool) or not self.no_activation:
            raise EvaluationError("shadow decision cannot activate")
        object.__setattr__(self, "baseline_ratios", MappingProxyType(dict(self.baseline_ratios)))
        object.__setattr__(self, "confidence_intervals", MappingProxyType(dict(self.confidence_intervals)))
        object.__setattr__(self, "reasons", tuple(self.reasons))

    @property
    def candidate(self) -> str:
        return self.candidate_id

    @property
    def ratios(self) -> Mapping[str, float]:
        return self.baseline_ratios

    @property
    def cis(self) -> Mapping[str, tuple[float, float]]:
        return self.confidence_intervals

    @property
    def no_recommendation(self) -> bool:
        return not self.qualified


class Evaluator:
    """Evaluate exact AB/BA paired blocks under fixed statistical gates."""

    def __init__(self, *, min_pairs: int = 3, bootstrap_resamples: int = 1_000,
                 seed: int = 20260830, preregistered_mde: float = 0.05,
                 registry: CandidateRegistry | None = None) -> None:
        if isinstance(min_pairs, bool) or not isinstance(min_pairs, int) or not 3 <= min_pairs <= 100:
            raise EvaluationError("min_pairs must be between 3 and 100")
        if isinstance(bootstrap_resamples, bool) or not isinstance(bootstrap_resamples, int) or not 100 <= bootstrap_resamples <= 10_000:
            raise EvaluationError("bootstrap_resamples must be between 100 and 10000")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise EvaluationError("seed must be integer")
        if isinstance(preregistered_mde, bool) or not isinstance(preregistered_mde, (int, float)) or not 0 < float(preregistered_mde) <= 1 or not math.isfinite(float(preregistered_mde)):
            raise EvaluationError("preregistered_mde invalid")
        self.min_pairs = min_pairs
        self.bootstrap_resamples = bootstrap_resamples
        self.seed = seed
        self.preregistered_mde = float(preregistered_mde)
        if registry is not None and not isinstance(registry, CandidateRegistry):
            raise EvaluationError("registry must be CandidateRegistry")
        # The orchestrator owns the one registry instance and injects it here;
        # direct callers still receive the immutable production registry.
        self.registry = registry or CandidateRegistry()

    def calibrate_aa(self, *, aa_left: Sequence[MetricSample] | None = None,
                     aa_right: Sequence[MetricSample] | None = None,
                     baseline: Sequence[MetricSample] | None = None,
                     control: Sequence[MetricSample] | None = None) -> CalibrationReport:
        return calibrate_aa(
            aa_left=aa_left, aa_right=aa_right, baseline=baseline, control=control,
            preregistered_mde=self.preregistered_mde,
            bootstrap_resamples=self.bootstrap_resamples, seed=self.seed,
        )

    def evaluate(
        self,
        fingerprint: ExactFingerprint,
        candidate_id: str,
        baseline_samples: Sequence[MetricSample],
        candidate_samples: Sequence[MetricSample],
        *,
        calibration: CalibrationReport | None = None,
        aa: CalibrationReport | None = None,
        aa_baseline_samples: Sequence[MetricSample] | None = None,
        aa_control_samples: Sequence[MetricSample] | None = None,
        aa_pairs: Sequence[tuple[MetricSample, MetricSample]] | None = None,
        parameters: Mapping[str, Any] | None = None,
        qualified: Iterable[str] = (),
        baseline_correctness: CorrectnessResult | None = None,
        candidate_correctness: CorrectnessResult | None = None,
        correctness: tuple[CorrectnessResult, CorrectnessResult] | None = None,
        resources: Sequence[ResourceResult] = (),
    ) -> ShadowDecision:
        if not isinstance(fingerprint, ExactFingerprint):
            raise EvaluationError("evaluate requires a complete ExactFingerprint object")
        if len(baseline_samples) > MAX_RAW_SAMPLES or len(candidate_samples) > MAX_RAW_SAMPLES:
            raise EvaluationError("A/B sample bound exceeded")
        fingerprint_hash = fingerprint.fingerprint_hash
        if not fingerprint.recommendation_allowed:
            return ShadowDecision(fingerprint_hash, str(candidate_id), "ood", reasons=(fingerprint.ood_reason or "fingerprint_ood",), evidence_hash=_evidence_hash({"candidate": str(candidate_id)}))
        try:
            registered = self.registry.get(candidate_id)
            effective_parameters = dict(registered.parameters) if parameters is None else parameters
            self.registry.validate(candidate_id, fingerprint=fingerprint, parameters=effective_parameters, qualified=qualified)
        except CandidateError as exc:
            status = "ood" if "outside its measured scope" in str(exc) else "rejected"
            return ShadowDecision(fingerprint_hash, str(candidate_id), status, reasons=(f"candidate_scope:{exc}",), evidence_hash=_evidence_hash({"candidate": str(candidate_id), "parameters": parameters}))
        if correctness is not None:
            if len(correctness) != 2:
                raise EvaluationError("correctness must contain baseline and candidate")
            baseline_correctness, candidate_correctness = correctness
        # A report supplied without its raw A/A observations is not trusted.
        # When raw pairs are supplied, recompute and bind the report hash so a
        # caller cannot forge a favorable noise/MDE estimate.
        aa_baseline, aa_control = _normalise_aa_inputs(aa_baseline_samples, aa_control_samples, aa_pairs)
        all_ab_samples = tuple(baseline_samples) + tuple(candidate_samples)
        if any(not isinstance(sample, MetricSample) or not sample.fingerprint or sample.fingerprint != fingerprint_hash for sample in all_ab_samples):
            return ShadowDecision(
                fingerprint_hash, candidate_id, "rejected",
                reasons=("ab_fingerprint_binding_missing_or_mismatched",),
                evidence_hash=_evidence_hash({"candidate": candidate_id, "baseline": repr(baseline_samples), "candidate_samples": repr(candidate_samples)}),
            )
        if aa_baseline is not None and any(sample.fingerprint != fingerprint_hash for sample in tuple(aa_baseline) + tuple(aa_control or ())):
            return ShadowDecision(
                fingerprint_hash, candidate_id, "rejected",
                reasons=("aa_fingerprint_binding_missing_or_mismatched",),
                evidence_hash=_evidence_hash({"candidate": candidate_id, "aa_baseline": repr(aa_baseline), "aa_control": repr(aa_control)}),
            )
        if aa_baseline is not None:
            try:
                recomputed = self.calibrate_aa(baseline=aa_baseline, control=aa_control)
            except EvaluationError as exc:
                return ShadowDecision(
                    fingerprint_hash, candidate_id, "rejected",
                    reasons=(f"aa_pair_validation:{exc}",),
                    evidence_hash=_evidence_hash({"candidate": candidate_id, "aa_baseline": repr(aa_baseline), "aa_control": repr(aa_control)}),
                )
            if calibration is not None and calibration != recomputed:
                calibration = None
                calibration_reason = "aa_report_does_not_match_raw_evidence"
            else:
                calibration = recomputed
                calibration_reason = ""
        else:
            calibration = None
            calibration_reason = "raw_aa_evidence_required"
        evidence = {
            "fingerprint": fingerprint_hash, "candidate": candidate_id,
            "baseline": [s.as_dict() if isinstance(s, MetricSample) else repr(s) for s in baseline_samples],
            "candidate_samples": [s.as_dict() if isinstance(s, MetricSample) else repr(s) for s in candidate_samples],
            "resources": [r.as_dict() if isinstance(r, ResourceResult) else repr(r) for r in resources],
            "aa_baseline": [] if aa_baseline is None else [s.as_dict() for s in aa_baseline],
            "aa_control": [] if aa_control is None else [s.as_dict() for s in aa_control or ()],
            "calibration": None if calibration is None else calibration.__dict__ if hasattr(calibration, "__dict__") else repr(calibration),
            "calibration_reason": calibration_reason,
            "registry_hash": self.registry.registry_hash,
        }
        evidence_hash = _evidence_hash(evidence)
        if calibration is None or not calibration.adequate or calibration.mde is None:
            reason = calibration_reason or "aa_calibration_missing_or_inadequate"
            return ShadowDecision(fingerprint_hash, candidate_id, "inconclusive", reasons=(reason,), evidence_hash=evidence_hash)
        reasons: list[str] = []
        if not resources:
            reasons.append("resource_evidence_missing")
        elif any(not isinstance(item, ResourceResult) or item.invalid_values for item in resources):
            reasons.append("resource_or_safety_gate_failed")
        elif any(not item.complete for item in resources):
            reasons.append("resource_evidence_incomplete_or_unknown")
        elif any(not item.passed for item in resources):
            reasons.append("resource_or_safety_gate_failed")
        reasons.extend(_correctness_reasons(baseline_correctness, candidate_correctness))
        ttft_ratios, decode_ratios, pair_reasons = _pair_ratios_checked(baseline_samples, candidate_samples)
        if pair_reasons:
            reasons.extend(pair_reasons)
            return ShadowDecision(fingerprint_hash, candidate_id, "rejected", reasons=tuple(dict.fromkeys(reasons)), evidence_hash=evidence_hash)
        if any(sample.status != "ok" for sample in tuple(baseline_samples) + tuple(candidate_samples) if isinstance(sample, MetricSample)):
            reasons.append("ab_run_failed_or_invalid")
        if len(ttft_ratios) < self.min_pairs or len(decode_ratios) < self.min_pairs:
            reasons.append("ab_requires_multiple_valid_paired_sessions")
            hard_reasons = {"resource_or_safety_gate_failed", "ab_run_failed_or_invalid"}
            hard = any(reason in hard_reasons or reason.startswith("correctness_mismatch") or reason == "correctness_gate_failed" for reason in reasons)
            return ShadowDecision(fingerprint_hash, candidate_id, "rejected" if hard else "inconclusive", reasons=tuple(dict.fromkeys(reasons)), evidence_hash=evidence_hash)
        if reasons:
            hard_reasons = {"resource_or_safety_gate_failed", "ab_run_failed_or_invalid"}
            hard = any(reason in hard_reasons or reason.startswith("correctness_mismatch") or reason == "correctness_gate_failed" for reason in reasons)
            return ShadowDecision(fingerprint_hash, candidate_id, "rejected" if hard else "inconclusive", reasons=tuple(dict.fromkeys(reasons)), evidence_hash=evidence_hash)
        ttft_ci = _bootstrap(ttft_ratios, seed=self.seed, resamples=self.bootstrap_resamples)
        decode_ci = _bootstrap(decode_ratios, seed=self.seed + 1, resamples=self.bootstrap_resamples)
        ratios = {"ttft": _percentile(ttft_ratios, 0.5), "decode_tps": _percentile(decode_ratios, 0.5)}
        cis = {"ttft": ttft_ci, "decode_tps": decode_ci}
        mde = calibration.mde
        # Ratios are candidate/baseline. TTFT is better below one; decode is
        # better above one.  A confidence interval crossing the adverse MDE is
        # uncertainty, while one wholly beyond it is a confirmed regression.
        ttft_regress = ttft_ci[0] > 1 + mde
        decode_regress = decode_ci[1] < 1 - mde
        ttft_uncertain = ttft_ci[1] > 1 + mde
        decode_uncertain = decode_ci[0] < 1 - mde
        ttft_improved = ttft_ci[1] < 1 - mde
        decode_improved = decode_ci[0] > 1 + mde
        if ttft_regress or decode_regress:
            reasons.append("statistically_confirmed_metric_regression")
        elif ttft_uncertain or decode_uncertain:
            reasons.append("confidence_interval_crosses_adverse_gate")
        if not (ttft_improved or decode_improved):
            reasons.append("no_metric_exceeds_noise_floor_or_mde")
        if reasons:
            status = "rejected" if "statistically_confirmed_metric_regression" in reasons or "resource_or_safety_gate_failed" in reasons or any(r.startswith("correctness_") for r in reasons) else "inconclusive"
            return ShadowDecision(fingerprint_hash, candidate_id, status, ratios, cis, False, tuple(dict.fromkeys(reasons)), evidence_hash)
        return ShadowDecision(fingerprint_hash, candidate_id, "qualified", ratios, cis, True, (), evidence_hash)

    def rank(self, decisions: Iterable[ShadowDecision]) -> tuple[ShadowDecision, ...]:
        """Conservative deterministic ranking; unqualified decisions trail baseline."""

        values = tuple(decisions)
        def key(decision: ShadowDecision) -> tuple[int, float, str]:
            if not decision.qualified:
                return (0, float("-inf"), decision.candidate_id)
            ttft_gain = 1 - decision.confidence_intervals.get("ttft", (float("inf"), float("inf")))[1]
            decode_gain = decision.confidence_intervals.get("decode_tps", (float("-inf"), float("-inf")))[0] - 1
            return (1, min(ttft_gain, decode_gain), decision.candidate_id)
        return tuple(sorted(values, key=key, reverse=True))


def evaluate_candidate(*args: Any, **kwargs: Any) -> ShadowDecision:
    """Convenience wrapper for the default preregistered evaluator."""

    return Evaluator().evaluate(*args, **kwargs)


__all__ = [
    "EvaluationError", "MetricSample", "MetricResult", "CorrectnessResult", "Correctness",
    "ResourceResult", "ResourceSample", "CalibrationReport", "ShadowDecision", "calibrate_aa",
    "Evaluator", "evaluate_candidate",
]
