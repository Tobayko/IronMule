"""Fail-closed evidence contracts for execution strategies.

This module is deliberately stdlib-only and has no execution surface.  It describes
existing IronMule paths, their exact validity domains, evaluator-owned evidence and
trusted profiles.  It does not import MLX or any IronMule runtime module, persist
records, select a strategy, or execute a model.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import InitVar, dataclass, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, ClassVar, Iterable, Mapping


_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")
_UNKNOWN = {"unknown", "missing", "none", "null", "unavailable"}
_PROFILE_CONSTRUCTION_TOKEN = object()


class EvidenceValidationError(ValueError):
    """An evidence record is incomplete, ambiguous, or outside the D1 contract."""


class EvidenceStatus(str, Enum):
    HYPOTHESIS = "HYPOTHESIS"
    QUALIFIED = "QUALIFIED"
    REJECTED = "REJECTED"
    INCONCLUSIVE = "INCONCLUSIVE"
    INVALIDATED = "INVALIDATED"
    REVALIDATION_REQUIRED = "REVALIDATION_REQUIRED"


class ActorRole(str, Enum):
    RESEARCHER = "RESEARCHER"
    REVIEWER = "REVIEWER"
    EVALUATOR = "EVALUATOR"
    RUNTIME = "RUNTIME"


class EvidenceQuality(str, Enum):
    RAW_SAMPLES = "RAW_SAMPLES"
    SUMMARY_ONLY = "SUMMARY_ONLY"
    PARTIAL = "PARTIAL"


class MetricSource(str, Enum):
    MEASURED = "MEASURED"
    DERIVED = "DERIVED"
    HYPOTHESIS = "HYPOTHESIS"


class DomainMatchStatus(str, Enum):
    MATCH = "MATCH"
    REVALIDATION_REQUIRED = "REVALIDATION_REQUIRED"


def _text(name: str, value: Any, *, known: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceValidationError(f"{name} must be a non-empty string")
    normalized = value.strip()
    if known and normalized.lower() in _UNKNOWN:
        raise EvidenceValidationError(f"{name} must be known, not {normalized!r}")
    return normalized


def _digest(name: str, value: Any, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    normalized = _text(name, value)
    if not _DIGEST.fullmatch(normalized):
        raise EvidenceValidationError(f"{name} must be a lowercase SHA-256 digest")
    return normalized


def _integer(name: str, value: Any, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise EvidenceValidationError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(name: str, value: Any, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceValidationError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or minimum is not None and result < minimum:
        qualifier = "finite" if minimum is None else f"finite and >= {minimum}"
        raise EvidenceValidationError(f"{name} must be {qualifier}")
    return result


def _optional_finite(name: str, value: Any, *, minimum: float = 0.0) -> float | None:
    return None if value is None else _finite(name, value, minimum=minimum)


def _enum(enum_type: type[Enum], name: str, value: Any) -> Enum:
    try:
        return value if isinstance(value, enum_type) else enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise EvidenceValidationError(f"{name} must be one of: {allowed}") from exc


def _utc(name: str, value: Any) -> str:
    normalized = _text(name, value)
    candidate = normalized[:-1] + "+00:00" if normalized.endswith("Z") else normalized
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise EvidenceValidationError(f"{name} must be an ISO-8601 UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise EvidenceValidationError(f"{name} must include the UTC offset")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _artifact_id(name: str, value: Any) -> str:
    normalized = _text(name, value)
    lowered = normalized.lower()
    if (normalized.startswith(("/", "\\")) or _WINDOWS_ABSOLUTE.match(normalized)
            or "file://" in lowered or "/users/" in lowered or "\\users\\" in lowered):
        raise EvidenceValidationError(f"{name} must not contain an absolute local path")
    return normalized


def _strict(data: Mapping[str, Any], expected: set[str], name: str) -> None:
    if not isinstance(data, Mapping):
        raise EvidenceValidationError(f"{name} must be an object")
    actual = set(data)
    missing = sorted(expected - actual)
    unknown = sorted(repr(key) for key in actual - expected)
    if missing or unknown:
        raise EvidenceValidationError(
            f"{name} fields differ; missing={missing!r}, unknown={unknown!r}"
        )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        keys = list(value)
        if any(not isinstance(key, str) for key in keys):
            raise EvidenceValidationError("canonical JSON object keys must be strings")
        result = {}
        for key in sorted(keys):
            result[key] = _jsonable(value[key])
        return result
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EvidenceValidationError("canonical JSON forbids NaN and Infinity")
        return value
    raise EvidenceValidationError(f"value is not canonical JSON data: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Return deterministic UTF-8 JSON while rejecting non-finite/non-JSON values."""
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ClosedInterval:
    lower: int
    upper: int

    def __post_init__(self) -> None:
        _integer("interval.lower", self.lower)
        _integer("interval.upper", self.upper)
        if self.lower > self.upper:
            raise EvidenceValidationError("interval.lower must be <= interval.upper")

    def contains(self, other: "ClosedInterval") -> bool:
        return self.lower <= other.lower and other.upper <= self.upper

    def to_dict(self) -> dict[str, int]:
        return {"lower": self.lower, "upper": self.upper}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ClosedInterval":
        _strict(data, {"lower", "upper"}, "ClosedInterval")
        return cls(lower=data["lower"], upper=data["upper"])


@dataclass(frozen=True, slots=True)
class DomainMatch:
    status: DomainMatchStatus
    mismatches: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _enum(DomainMatchStatus, "status", self.status))
        object.__setattr__(self, "mismatches", tuple(_text("mismatch", item) for item in self.mismatches))
        if self.status is DomainMatchStatus.MATCH and self.mismatches:
            raise EvidenceValidationError("MATCH cannot contain mismatches")
        if self.status is DomainMatchStatus.REVALIDATION_REQUIRED and not self.mismatches:
            raise EvidenceValidationError("REVALIDATION_REQUIRED needs at least one mismatch")


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    artifact_id: str
    sha256: str
    quality: EvidenceQuality

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", _artifact_id("artifact_id", self.artifact_id))
        object.__setattr__(self, "sha256", _digest("artifact.sha256", self.sha256))
        object.__setattr__(self, "quality", _enum(EvidenceQuality, "artifact.quality", self.quality))

    def to_dict(self) -> dict[str, Any]:
        return {"artifact_id": self.artifact_id, "sha256": self.sha256,
                "quality": self.quality.value}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactRef":
        _strict(data, {"artifact_id", "sha256", "quality"}, "ArtifactRef")
        return cls(artifact_id=data["artifact_id"], sha256=data["sha256"], quality=data["quality"])


@dataclass(frozen=True, slots=True)
class MetricEvidence:
    name: str
    unit: str
    source: MetricSource
    samples: tuple[float, ...]
    median: float | None
    p95: float | None
    ci_low: float | None
    ci_high: float | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text("metric.name", self.name))
        object.__setattr__(self, "unit", _text("metric.unit", self.unit))
        object.__setattr__(self, "source", _enum(MetricSource, "metric.source", self.source))
        object.__setattr__(self, "samples", tuple(
            _finite("metric.sample", sample, minimum=0.0) for sample in self.samples
        ))
        for field_name in ("median", "p95", "ci_low", "ci_high"):
            object.__setattr__(self, field_name, _optional_finite(
                f"metric.{field_name}", getattr(self, field_name)
            ))
        if (self.ci_low is None) != (self.ci_high is None):
            raise EvidenceValidationError("metric CI needs both low and high values")
        if self.ci_low is not None and self.ci_low > self.ci_high:
            raise EvidenceValidationError("metric.ci_low must be <= metric.ci_high")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "unit": self.unit, "source": self.source.value,
            "samples": list(self.samples), "median": self.median, "p95": self.p95,
            "ci_low": self.ci_low, "ci_high": self.ci_high,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MetricEvidence":
        expected = {"name", "unit", "source", "samples", "median", "p95", "ci_low", "ci_high"}
        _strict(data, expected, "MetricEvidence")
        if not isinstance(data["samples"], list):
            raise EvidenceValidationError("MetricEvidence.samples must be an array")
        return cls(
            name=data["name"], unit=data["unit"], source=data["source"],
            samples=tuple(data["samples"]), median=data["median"], p95=data["p95"],
            ci_low=data["ci_low"], ci_high=data["ci_high"],
        )


@dataclass(frozen=True, slots=True)
class CorrectnessEvidence:
    comparison_performed: bool
    token_identity: bool
    stop_reason_identity: bool
    token_count_identity: bool
    deterministic: bool
    state_identity: bool
    quality_class: str

    def __post_init__(self) -> None:
        for name in ("comparison_performed", "token_identity", "stop_reason_identity",
                     "token_count_identity", "deterministic", "state_identity"):
            if not isinstance(getattr(self, name), bool):
                raise EvidenceValidationError(f"correctness.{name} must be boolean")
        object.__setattr__(self, "quality_class", _text("correctness.quality_class", self.quality_class))
        identities = (self.token_identity, self.stop_reason_identity, self.token_count_identity,
                      self.deterministic, self.state_identity)
        if not self.comparison_performed and any(identities):
            raise EvidenceValidationError("unperformed correctness comparison cannot claim identity")

    @property
    def qualifies_exact(self) -> bool:
        return all((self.comparison_performed, self.token_identity,
                    self.stop_reason_identity, self.token_count_identity,
                    self.deterministic, self.state_identity))

    def to_dict(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CorrectnessEvidence":
        expected = {field.name for field in fields(cls)}
        _strict(data, expected, "CorrectnessEvidence")
        return cls(**{name: data[name] for name in expected})


@dataclass(frozen=True, slots=True)
class ResourceEvidence:
    mlx_active_memory_bytes: int | None
    mlx_peak_memory_bytes: int | None
    rss_peak_bytes: int | None
    swap_before_bytes: int | None
    swap_after_bytes: int | None
    timeout: bool
    crash_free: bool
    fallbacks: int
    worker_status: str
    gates_passed: bool

    def __post_init__(self) -> None:
        for name in ("mlx_active_memory_bytes", "mlx_peak_memory_bytes", "rss_peak_bytes",
                     "swap_before_bytes", "swap_after_bytes"):
            value = getattr(self, name)
            if value is not None:
                _integer(f"resources.{name}", value)
        if not isinstance(self.timeout, bool) or not isinstance(self.crash_free, bool):
            raise EvidenceValidationError("resource timeout/crash fields must be boolean")
        if not isinstance(self.gates_passed, bool):
            raise EvidenceValidationError("resources.gates_passed must be boolean")
        _integer("resources.fallbacks", self.fallbacks)
        object.__setattr__(self, "worker_status", _text("resources.worker_status", self.worker_status))

    @property
    def swap_delta_bytes(self) -> int | None:
        if self.swap_before_bytes is None or self.swap_after_bytes is None:
            return None
        return self.swap_after_bytes - self.swap_before_bytes

    @property
    def qualifies(self) -> bool:
        return all(value is not None for value in (
            self.mlx_active_memory_bytes, self.mlx_peak_memory_bytes, self.rss_peak_bytes,
            self.swap_before_bytes, self.swap_after_bytes,
        )) and self.gates_passed and self.crash_free and not self.timeout and self.fallbacks == 0

    def to_dict(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ResourceEvidence":
        expected = {field.name for field in fields(cls)}
        _strict(data, expected, "ResourceEvidence")
        return cls(**{name: data[name] for name in expected})


@dataclass(frozen=True, slots=True)
class StatisticsEvidence:
    warmup_repeats: int
    measured_repeats: int
    raw_sample_count: int
    paired: bool
    arm_order: tuple[str, ...]
    estimator: str
    uncertainty_method: str
    confidence_level: float | None

    def __post_init__(self) -> None:
        _integer("statistics.warmup_repeats", self.warmup_repeats)
        _integer("statistics.measured_repeats", self.measured_repeats)
        _integer("statistics.raw_sample_count", self.raw_sample_count)
        if not isinstance(self.paired, bool):
            raise EvidenceValidationError("statistics.paired must be boolean")
        object.__setattr__(self, "arm_order", tuple(
            _text("statistics.arm_order", item) for item in self.arm_order
        ))
        object.__setattr__(self, "estimator", _text("statistics.estimator", self.estimator))
        object.__setattr__(self, "uncertainty_method", _text(
            "statistics.uncertainty_method", self.uncertainty_method
        ))
        if self.confidence_level is not None:
            level = _finite("statistics.confidence_level", self.confidence_level, minimum=0.0)
            if not 0.0 < level < 1.0:
                raise EvidenceValidationError("confidence level must be strictly between 0 and 1")
            object.__setattr__(self, "confidence_level", level)

    @property
    def qualifies(self) -> bool:
        return (self.measured_repeats >= 2
                and self.raw_sample_count >= self.measured_repeats
                and self.confidence_level is not None
                and self.uncertainty_method.lower() not in _UNKNOWN | {"none"})

    def to_dict(self) -> dict[str, Any]:
        return {
            "warmup_repeats": self.warmup_repeats,
            "measured_repeats": self.measured_repeats,
            "raw_sample_count": self.raw_sample_count,
            "paired": self.paired,
            "arm_order": list(self.arm_order),
            "estimator": self.estimator,
            "uncertainty_method": self.uncertainty_method,
            "confidence_level": self.confidence_level,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StatisticsEvidence":
        expected = {"warmup_repeats", "measured_repeats", "raw_sample_count", "paired",
                    "arm_order", "estimator", "uncertainty_method", "confidence_level"}
        _strict(data, expected, "StatisticsEvidence")
        if not isinstance(data["arm_order"], list):
            raise EvidenceValidationError("StatisticsEvidence.arm_order must be an array")
        return cls(
            warmup_repeats=data["warmup_repeats"], measured_repeats=data["measured_repeats"],
            raw_sample_count=data["raw_sample_count"], paired=data["paired"],
            arm_order=tuple(data["arm_order"]), estimator=data["estimator"],
            uncertainty_method=data["uncertainty_method"],
            confidence_level=data["confidence_level"],
        )


@dataclass(frozen=True, slots=True)
class ExecutionStrategy:
    SCHEMA: ClassVar[str] = "ironmule.execution_strategy.v1"

    semantic_class: str
    prefill_policy: str
    decode_policy: str
    cache_policy: str
    scheduling_policy: str
    grouping_policy: str
    grouping_width: int
    synchronization_policy: str
    memory_policy: str
    compile_graph_policy: str
    prefix_reuse_policy: str
    plan_kind: str
    service_mode: str
    knobs_key: str
    implementation_revision: str
    strategy_id: str = ""

    def __post_init__(self) -> None:
        for name in ("semantic_class", "prefill_policy", "decode_policy", "cache_policy",
                     "scheduling_policy", "grouping_policy", "synchronization_policy",
                     "memory_policy", "compile_graph_policy", "prefix_reuse_policy",
                     "plan_kind", "service_mode", "knobs_key"):
            object.__setattr__(self, name, _text(f"strategy.{name}", getattr(self, name)))
        if self.semantic_class != "exact":
            raise EvidenceValidationError("D1 admits only semantic_class='exact'")
        _integer("strategy.grouping_width", self.grouping_width, minimum=1)
        if self.grouping_width > 4:
            raise EvidenceValidationError("D1 existing grouped paths have width <= 4")
        object.__setattr__(self, "implementation_revision", _digest(
            "strategy.implementation_revision", self.implementation_revision
        ))
        computed = canonical_sha256(self._semantic_dict())
        if self.strategy_id and self.strategy_id != computed:
            raise EvidenceValidationError("strategy_id does not match canonical content")
        object.__setattr__(self, "strategy_id", computed)

    def _semantic_dict(self) -> dict[str, Any]:
        return {"schema": self.SCHEMA, **{
            field.name: getattr(self, field.name)
            for field in fields(self) if field.name != "strategy_id"
        }}

    def to_dict(self) -> dict[str, Any]:
        return {**self._semantic_dict(), "strategy_id": self.strategy_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutionStrategy":
        expected = {"schema", "strategy_id", *(field.name for field in fields(cls)
                                                if field.name != "strategy_id")}
        _strict(data, expected, "ExecutionStrategy")
        if data["schema"] != cls.SCHEMA:
            raise EvidenceValidationError("unsupported ExecutionStrategy schema")
        kwargs = {field.name: data[field.name] for field in fields(cls)
                  if field.name != "strategy_id"}
        return cls(**kwargs, strategy_id=data["strategy_id"])


@dataclass(frozen=True, slots=True)
class ValidityDomain:
    SCHEMA: ClassVar[str] = "ironmule.validity_domain.v1"

    apple_chip: str
    machine: str
    ram_bytes: int
    gpu_cores: int
    gpu_configuration: str
    macos_build: str
    python_version: str
    mlx_version: str
    mlx_lm_version: str
    runtime_version: str
    model_id: str
    model_revision: str
    model_manifest_sha256: str
    model_architecture: str
    tokenizer_sha256: str
    quantization_bits: int
    quantization_group_size: int
    quantization_format: str
    cache_family: str
    cache_layer_pattern: str
    capacity_policy: str
    plan_kind: str
    prompt_bucket: ClosedInterval
    context_bucket: ClosedInterval
    output_bucket: ClosedInterval
    concurrency_bucket: ClosedInterval
    arrival_pattern: str
    workload_class: str
    power_source: str
    low_power_mode: bool
    thermal_state: str
    swap_class: str
    domain_id: str = ""

    _EXACT_FIELDS: ClassVar[tuple[str, ...]] = (
        "apple_chip", "machine", "ram_bytes", "gpu_cores", "gpu_configuration",
        "macos_build", "python_version", "mlx_version", "mlx_lm_version",
        "runtime_version", "model_id", "model_revision", "model_manifest_sha256",
        "model_architecture", "tokenizer_sha256", "quantization_bits",
        "quantization_group_size", "quantization_format", "cache_family",
        "cache_layer_pattern", "capacity_policy", "plan_kind", "arrival_pattern",
        "workload_class", "power_source", "low_power_mode", "thermal_state", "swap_class",
    )
    _BUCKET_FIELDS: ClassVar[tuple[str, ...]] = (
        "prompt_bucket", "context_bucket", "output_bucket", "concurrency_bucket",
    )

    def __post_init__(self) -> None:
        for name in ("apple_chip", "machine", "gpu_configuration", "macos_build",
                     "python_version", "mlx_version", "mlx_lm_version", "runtime_version",
                     "model_id", "model_revision", "model_architecture",
                     "quantization_format", "cache_family", "cache_layer_pattern",
                     "capacity_policy", "plan_kind", "arrival_pattern", "workload_class",
                     "power_source", "thermal_state", "swap_class"):
            object.__setattr__(self, name, _text(f"domain.{name}", getattr(self, name), known=True))
        _integer("domain.ram_bytes", self.ram_bytes, minimum=1)
        _integer("domain.gpu_cores", self.gpu_cores, minimum=1)
        _integer("domain.quantization_bits", self.quantization_bits, minimum=1)
        _integer("domain.quantization_group_size", self.quantization_group_size, minimum=1)
        if not isinstance(self.low_power_mode, bool):
            raise EvidenceValidationError("domain.low_power_mode must be boolean")
        object.__setattr__(self, "model_manifest_sha256", _digest(
            "domain.model_manifest_sha256", self.model_manifest_sha256
        ))
        object.__setattr__(self, "tokenizer_sha256", _digest(
            "domain.tokenizer_sha256", self.tokenizer_sha256
        ))
        for name in self._BUCKET_FIELDS:
            if not isinstance(getattr(self, name), ClosedInterval):
                raise EvidenceValidationError(f"domain.{name} must be ClosedInterval")
        computed = canonical_sha256(self._semantic_dict())
        if self.domain_id and self.domain_id != computed:
            raise EvidenceValidationError("domain_id does not match canonical content")
        object.__setattr__(self, "domain_id", computed)

    def _semantic_dict(self) -> dict[str, Any]:
        return {"schema": self.SCHEMA, **{
            field.name: _jsonable(getattr(self, field.name))
            for field in fields(self) if field.name != "domain_id"
        }}

    def to_dict(self) -> dict[str, Any]:
        return {**self._semantic_dict(), "domain_id": self.domain_id}

    def match(self, current: "ValidityDomain") -> DomainMatch:
        if not isinstance(current, ValidityDomain):
            raise EvidenceValidationError("current domain must be ValidityDomain")
        mismatches = [name for name in self._EXACT_FIELDS
                      if getattr(self, name) != getattr(current, name)]
        mismatches.extend(name for name in self._BUCKET_FIELDS
                          if not getattr(self, name).contains(getattr(current, name)))
        if mismatches:
            return DomainMatch(DomainMatchStatus.REVALIDATION_REQUIRED, tuple(mismatches))
        return DomainMatch(DomainMatchStatus.MATCH)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ValidityDomain":
        expected = {"schema", "domain_id", *(field.name for field in fields(cls)
                                              if field.name != "domain_id")}
        _strict(data, expected, "ValidityDomain")
        if data["schema"] != cls.SCHEMA:
            raise EvidenceValidationError("unsupported ValidityDomain schema")
        kwargs = {}
        for field in fields(cls):
            if field.name == "domain_id":
                continue
            value = data[field.name]
            kwargs[field.name] = ClosedInterval.from_dict(value) if field.name in cls._BUCKET_FIELDS else value
        return cls(**kwargs, domain_id=data["domain_id"])


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    SCHEMA: ClassVar[str] = "ironmule.evidence_record.v1"

    status: EvidenceStatus
    status_owner: ActorRole
    researcher_id: str
    reviewer_id: str
    evaluator_id: str
    diagnostic_verdict: str
    baseline_strategy_id: str
    candidate_strategy_id: str
    validity_domain_id: str
    experiment_id: str
    study_id: str
    preregistration_sha256: str
    reviewer_record_sha256: str | None
    code_sha256: str
    model_sha256: str
    environment_sha256: str
    workload_sha256: str
    raw_artifacts: tuple[ArtifactRef, ...]
    metrics: tuple[MetricEvidence, ...]
    correctness: CorrectnessEvidence
    resources: ResourceEvidence
    statistics: StatisticsEvidence
    evidence_quality: EvidenceQuality
    recorded_at: str
    supersedes: tuple[str, ...] = ()
    revalidates: tuple[str, ...] = ()
    evidence_id: str = ""

    _EVALUATOR_STATUSES: ClassVar[frozenset[EvidenceStatus]] = frozenset({
        EvidenceStatus.QUALIFIED, EvidenceStatus.REJECTED, EvidenceStatus.INCONCLUSIVE,
        EvidenceStatus.INVALIDATED, EvidenceStatus.REVALIDATION_REQUIRED,
    })

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _enum(EvidenceStatus, "evidence.status", self.status))
        object.__setattr__(self, "status_owner", _enum(ActorRole, "evidence.status_owner", self.status_owner))
        for name in ("diagnostic_verdict", "experiment_id", "study_id"):
            object.__setattr__(self, name, _text(f"evidence.{name}", getattr(self, name)))
        for name in ("baseline_strategy_id", "candidate_strategy_id", "validity_domain_id",
                     "preregistration_sha256", "code_sha256", "model_sha256",
                     "environment_sha256", "workload_sha256"):
            object.__setattr__(self, name, _digest(f"evidence.{name}", getattr(self, name)))
        object.__setattr__(self, "reviewer_record_sha256", _digest(
            "evidence.reviewer_record_sha256", self.reviewer_record_sha256, optional=True
        ))
        for name in ("researcher_id", "reviewer_id", "evaluator_id"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise EvidenceValidationError(f"evidence.{name} must be a string")
            object.__setattr__(self, name, value.strip())
        object.__setattr__(self, "researcher_id", _text(
            "evidence.researcher_id", self.researcher_id
        ))
        object.__setattr__(self, "raw_artifacts", tuple(self.raw_artifacts))
        object.__setattr__(self, "metrics", tuple(self.metrics))
        object.__setattr__(self, "evidence_quality", _enum(
            EvidenceQuality, "evidence.evidence_quality", self.evidence_quality
        ))
        object.__setattr__(self, "recorded_at", _utc("evidence.recorded_at", self.recorded_at))
        object.__setattr__(self, "supersedes", tuple(
            _digest("evidence.supersedes", item) for item in self.supersedes
        ))
        object.__setattr__(self, "revalidates", tuple(
            _digest("evidence.revalidates", item) for item in self.revalidates
        ))
        if not isinstance(self.correctness, CorrectnessEvidence):
            raise EvidenceValidationError("evidence.correctness must be CorrectnessEvidence")
        if not isinstance(self.resources, ResourceEvidence):
            raise EvidenceValidationError("evidence.resources must be ResourceEvidence")
        if not isinstance(self.statistics, StatisticsEvidence):
            raise EvidenceValidationError("evidence.statistics must be StatisticsEvidence")
        if any(not isinstance(item, ArtifactRef) for item in self.raw_artifacts):
            raise EvidenceValidationError("evidence.raw_artifacts must contain ArtifactRef")
        if any(not isinstance(item, MetricEvidence) for item in self.metrics):
            raise EvidenceValidationError("evidence.metrics must contain MetricEvidence")
        if len({item.name for item in self.metrics}) != len(self.metrics):
            raise EvidenceValidationError("evidence metric names must be unique")
        if self.status is EvidenceStatus.HYPOTHESIS:
            if (self.status_owner is not ActorRole.RESEARCHER
                    or self.reviewer_id or self.evaluator_id):
                raise EvidenceValidationError(
                    "HYPOTHESIS must be researcher-owned without reviewer/evaluator"
                )
        elif self.status in self._EVALUATOR_STATUSES:
            if self.status_owner is not ActorRole.EVALUATOR:
                raise EvidenceValidationError("terminal evidence status must be evaluator-owned")
            object.__setattr__(self, "evaluator_id", _text("evidence.evaluator_id", self.evaluator_id))
            object.__setattr__(self, "reviewer_id", _text("evidence.reviewer_id", self.reviewer_id))
            if len({self.researcher_id, self.reviewer_id, self.evaluator_id}) != 3:
                raise EvidenceValidationError(
                    "researcher, reviewer and evaluator identities must be distinct"
                )
            if self.reviewer_record_sha256 is None:
                raise EvidenceValidationError("terminal evidence requires a reviewer record digest")
        else:  # exhaustive guard for future enum changes
            raise EvidenceValidationError("unsupported evidence status ownership")
        if self.status is EvidenceStatus.QUALIFIED:
            self._validate_qualified()
        computed = canonical_sha256(self._semantic_dict())
        if self.evidence_id and self.evidence_id != computed:
            raise EvidenceValidationError("evidence_id does not match canonical content")
        object.__setattr__(self, "evidence_id", computed)

    def _validate_qualified(self) -> None:
        if self.evidence_quality is not EvidenceQuality.RAW_SAMPLES:
            raise EvidenceValidationError("QUALIFIED evidence requires RAW_SAMPLES quality")
        if not self.raw_artifacts or any(
            item.quality is not EvidenceQuality.RAW_SAMPLES for item in self.raw_artifacts
        ):
            raise EvidenceValidationError("QUALIFIED evidence requires raw artifact references")
        names = {metric.name for metric in self.metrics}
        required = {"outer_wall_ms", "physical_tokens_per_second"}
        if not required.issubset(names):
            raise EvidenceValidationError(f"QUALIFIED evidence is missing metrics {sorted(required - names)!r}")
        primary = [metric for metric in self.metrics if metric.name in required]
        if any(
            not metric.samples or metric.source is not MetricSource.MEASURED
            or metric.median is None or metric.p95 is None
            or metric.ci_low is None or metric.ci_high is None
            for metric in primary
        ):
            raise EvidenceValidationError(
                "QUALIFIED primary metrics require measured samples, median, p95 and interval"
            )
        if not self.correctness.qualifies_exact:
            raise EvidenceValidationError("QUALIFIED exact evidence failed correctness identity")
        if not self.resources.qualifies:
            raise EvidenceValidationError("QUALIFIED evidence failed resource gates")
        if not self.statistics.qualifies:
            raise EvidenceValidationError("QUALIFIED evidence lacks repeated uncertainty evidence")

    def _semantic_dict(self) -> dict[str, Any]:
        return {"schema": self.SCHEMA, **{
            field.name: _jsonable(getattr(self, field.name))
            for field in fields(self) if field.name != "evidence_id"
        }}

    def to_dict(self) -> dict[str, Any]:
        return {**self._semantic_dict(), "evidence_id": self.evidence_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceRecord":
        expected = {"schema", "evidence_id", *(field.name for field in fields(cls)
                                                if field.name != "evidence_id")}
        _strict(data, expected, "EvidenceRecord")
        if data["schema"] != cls.SCHEMA:
            raise EvidenceValidationError("unsupported EvidenceRecord schema")
        for name in ("raw_artifacts", "metrics", "supersedes", "revalidates"):
            if not isinstance(data[name], list):
                raise EvidenceValidationError(f"EvidenceRecord.{name} must be an array")
        kwargs = {field.name: data[field.name] for field in fields(cls)
                  if field.name not in {"evidence_id", "raw_artifacts", "metrics",
                                        "correctness", "resources", "statistics",
                                        "supersedes", "revalidates"}}
        kwargs.update({
            "raw_artifacts": tuple(ArtifactRef.from_dict(item) for item in data["raw_artifacts"]),
            "metrics": tuple(MetricEvidence.from_dict(item) for item in data["metrics"]),
            "correctness": CorrectnessEvidence.from_dict(data["correctness"]),
            "resources": ResourceEvidence.from_dict(data["resources"]),
            "statistics": StatisticsEvidence.from_dict(data["statistics"]),
            "supersedes": tuple(data["supersedes"]),
            "revalidates": tuple(data["revalidates"]),
        })
        return cls(**kwargs, evidence_id=data["evidence_id"])


@dataclass(frozen=True, slots=True)
class TrustedExecutionProfile:
    SCHEMA: ClassVar[str] = "ironmule.trusted_execution_profile.v1"

    strategy: ExecutionStrategy
    validity_domain: ValidityDomain
    evidence_ids: tuple[str, ...]
    protected_baseline_evidence_id: str
    creation_evidence_id: str
    last_revalidation_evidence_id: str
    status: EvidenceStatus
    profile_id: str = ""
    _construction_token: InitVar[object] = None

    def __post_init__(self, _construction_token: object) -> None:
        if _construction_token is not _PROFILE_CONSTRUCTION_TOKEN:
            raise EvidenceValidationError(
                "TrustedExecutionProfile must be built from verified qualified evidence"
            )
        if not isinstance(self.strategy, ExecutionStrategy):
            raise EvidenceValidationError("profile.strategy must be ExecutionStrategy")
        if not isinstance(self.validity_domain, ValidityDomain):
            raise EvidenceValidationError("profile.validity_domain must be ValidityDomain")
        evidence_ids = tuple(sorted({_digest("profile.evidence_id", item) for item in self.evidence_ids}))
        if not evidence_ids:
            raise EvidenceValidationError("profile needs at least one evidence ID")
        object.__setattr__(self, "evidence_ids", evidence_ids)
        for name in ("protected_baseline_evidence_id", "creation_evidence_id",
                     "last_revalidation_evidence_id"):
            object.__setattr__(self, name, _digest(f"profile.{name}", getattr(self, name)))
        object.__setattr__(self, "status", _enum(EvidenceStatus, "profile.status", self.status))
        if self.status is not EvidenceStatus.QUALIFIED:
            raise EvidenceValidationError("D1 constructs only QUALIFIED trusted profiles")
        if self.creation_evidence_id not in evidence_ids or self.last_revalidation_evidence_id not in evidence_ids:
            raise EvidenceValidationError("profile creation/revalidation evidence must be in evidence_ids")
        computed = canonical_sha256(self._semantic_dict())
        if self.profile_id and self.profile_id != computed:
            raise EvidenceValidationError("profile_id does not match canonical content")
        object.__setattr__(self, "profile_id", computed)

    @classmethod
    def from_qualified(
        cls,
        strategy: ExecutionStrategy,
        validity_domain: ValidityDomain,
        evidence: Iterable[EvidenceRecord],
        *,
        protected_baseline_evidence_id: str,
    ) -> "TrustedExecutionProfile":
        records = tuple(evidence)
        if not records:
            raise EvidenceValidationError("trusted profile needs qualified evidence")
        for record in records:
            if record.status is not EvidenceStatus.QUALIFIED:
                raise EvidenceValidationError("trusted profile rejects non-QUALIFIED evidence")
            if record.status_owner is not ActorRole.EVALUATOR:
                raise EvidenceValidationError("trusted profile requires evaluator-owned evidence")
            if record.candidate_strategy_id != strategy.strategy_id:
                raise EvidenceValidationError("trusted profile strategy/evidence mismatch")
            if record.validity_domain_id != validity_domain.domain_id:
                raise EvidenceValidationError("trusted profile domain/evidence mismatch")
        evidence_ids = tuple(record.evidence_id for record in records)
        return cls(
            strategy=strategy,
            validity_domain=validity_domain,
            evidence_ids=evidence_ids,
            protected_baseline_evidence_id=protected_baseline_evidence_id,
            creation_evidence_id=evidence_ids[0],
            last_revalidation_evidence_id=evidence_ids[-1],
            status=EvidenceStatus.QUALIFIED,
            _construction_token=_PROFILE_CONSTRUCTION_TOKEN,
        )

    def match(self, current_domain: ValidityDomain) -> DomainMatch:
        return self.validity_domain.match(current_domain)

    def _semantic_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "strategy": self.strategy.to_dict(),
            "validity_domain": self.validity_domain.to_dict(),
            "evidence_ids": list(self.evidence_ids),
            "protected_baseline_evidence_id": self.protected_baseline_evidence_id,
            "creation_evidence_id": self.creation_evidence_id,
            "last_revalidation_evidence_id": self.last_revalidation_evidence_id,
            "status": self.status.value,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._semantic_dict(), "profile_id": self.profile_id}

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], *, evidence: Iterable[EvidenceRecord]
    ) -> "TrustedExecutionProfile":
        expected = {"schema", "profile_id", "strategy", "validity_domain", "evidence_ids",
                    "protected_baseline_evidence_id", "creation_evidence_id",
                    "last_revalidation_evidence_id", "status"}
        _strict(data, expected, "TrustedExecutionProfile")
        if data["schema"] != cls.SCHEMA:
            raise EvidenceValidationError("unsupported TrustedExecutionProfile schema")
        if not isinstance(data["evidence_ids"], list):
            raise EvidenceValidationError("profile.evidence_ids must be an array")
        built = cls.from_qualified(
            ExecutionStrategy.from_dict(data["strategy"]),
            ValidityDomain.from_dict(data["validity_domain"]),
            tuple(evidence),
            protected_baseline_evidence_id=data["protected_baseline_evidence_id"],
        )
        if built.to_dict() != dict(data):
            raise EvidenceValidationError(
                "serialized trusted profile does not match supplied qualified evidence"
            )
        return built


def strategy_from_existing(
    *,
    plan_kind: str,
    service_mode: str,
    knobs_key: str,
    implementation_revision: str,
    prefill_policy: str,
    decode_policy: str,
    cache_policy: str,
    scheduling_policy: str,
    grouping_policy: str,
    grouping_width: int,
    synchronization_policy: str,
    memory_policy: str,
    compile_graph_policy: str,
    prefix_reuse_policy: str,
) -> ExecutionStrategy:
    """Describe existing caller-supplied path IDs without importing their owners."""
    return ExecutionStrategy(
        semantic_class="exact", prefill_policy=prefill_policy,
        decode_policy=decode_policy, cache_policy=cache_policy,
        scheduling_policy=scheduling_policy, grouping_policy=grouping_policy,
        grouping_width=grouping_width, synchronization_policy=synchronization_policy,
        memory_policy=memory_policy, compile_graph_policy=compile_graph_policy,
        prefix_reuse_policy=prefix_reuse_policy, plan_kind=plan_kind,
        service_mode=service_mode, knobs_key=knobs_key,
        implementation_revision=implementation_revision,
    )


def domain_from_fingerprint(
    fingerprint: Mapping[str, Any],
    *,
    model_revision: str,
    model_manifest_sha256: str,
    model_architecture: str,
    tokenizer_sha256: str,
    python_version: str,
    gpu_configuration: str,
    quantization_format: str,
    cache_family: str,
    cache_layer_pattern: str,
    capacity_policy: str,
    prompt_bucket: ClosedInterval,
    context_bucket: ClosedInterval,
    output_bucket: ClosedInterval,
    concurrency_bucket: ClosedInterval,
    arrival_pattern: str,
    workload_class: str,
    low_power_mode: bool,
    thermal_state: str,
    swap_class: str,
) -> ValidityDomain:
    """Complete the current fingerprint with identity D1 refuses to infer."""
    required = {"chip", "machine", "memory_bytes", "gpu_cores", "os", "mlx", "mlx_lm",
                "runtime_version", "model_id", "quantisation", "execution_plan",
                "power_source"}
    missing = sorted(required - set(fingerprint))
    if missing:
        raise EvidenceValidationError(f"fingerprint is missing {missing!r}")
    quantization = fingerprint["quantisation"]
    if not isinstance(quantization, Mapping):
        raise EvidenceValidationError("fingerprint.quantisation must be an object")
    _strict(quantization, {"bits", "group_size"}, "fingerprint.quantisation")
    os_value = _text("fingerprint.os", fingerprint["os"], known=True)
    return ValidityDomain(
        apple_chip=fingerprint["chip"], machine=fingerprint["machine"],
        ram_bytes=fingerprint["memory_bytes"], gpu_cores=fingerprint["gpu_cores"],
        gpu_configuration=gpu_configuration,
        macos_build=os_value.removeprefix("Darwin "),
        python_version=python_version,
        mlx_version=fingerprint["mlx"], mlx_lm_version=fingerprint["mlx_lm"],
        runtime_version=fingerprint["runtime_version"], model_id=fingerprint["model_id"],
        model_revision=model_revision, model_manifest_sha256=model_manifest_sha256,
        model_architecture=model_architecture, tokenizer_sha256=tokenizer_sha256,
        quantization_bits=quantization["bits"],
        quantization_group_size=quantization["group_size"],
        quantization_format=quantization_format,
        cache_family=cache_family, cache_layer_pattern=cache_layer_pattern,
        capacity_policy=capacity_policy, plan_kind=fingerprint["execution_plan"],
        prompt_bucket=prompt_bucket, context_bucket=context_bucket,
        output_bucket=output_bucket, concurrency_bucket=concurrency_bucket,
        arrival_pattern=arrival_pattern, workload_class=workload_class,
        power_source=fingerprint["power_source"], low_power_mode=low_power_mode,
        thermal_state=thermal_state, swap_class=swap_class,
    )


def evidence_from_b27_public_cell(
    cell: Mapping[str, Any],
    *,
    baseline_strategy_id: str,
    candidate_strategy_id: str,
    validity_domain_id: str,
    experiment_id: str,
    study_id: str,
    preregistration_sha256: str,
    reviewer_record_sha256: str,
    code_sha256: str,
    model_sha256: str,
    environment_sha256: str,
    workload_sha256: str,
    evaluator_id: str,
    researcher_id: str,
    reviewer_id: str,
    recorded_at: str,
    warmup_repeats: int,
    measured_repeats: int,
) -> EvidenceRecord:
    """Import a path-free B27 summary as explicitly non-qualifying summary evidence."""
    required = {"model", "raw_sha256", "comparison", "interactive", "throughput", "resources"}
    missing = sorted(required - set(cell))
    if missing:
        raise EvidenceValidationError(f"B27 cell is missing {missing!r}")
    comparison = cell["comparison"]
    wall = comparison["wall_ratio_throughput_over_interactive"]
    rate = comparison["physical_rate_ratio_throughput_over_interactive"]
    resources = cell["resources"]
    return EvidenceRecord(
        status=EvidenceStatus.INCONCLUSIVE,
        status_owner=ActorRole.EVALUATOR,
        researcher_id=researcher_id,
        reviewer_id=reviewer_id,
        evaluator_id=evaluator_id,
        diagnostic_verdict="ENGINEERING_BASELINE_CAPTURED_NOT_QUALIFIED",
        baseline_strategy_id=baseline_strategy_id,
        candidate_strategy_id=candidate_strategy_id,
        validity_domain_id=validity_domain_id,
        experiment_id=experiment_id,
        study_id=study_id,
        preregistration_sha256=preregistration_sha256,
        reviewer_record_sha256=reviewer_record_sha256,
        code_sha256=code_sha256,
        model_sha256=model_sha256,
        environment_sha256=environment_sha256,
        workload_sha256=workload_sha256,
        raw_artifacts=(ArtifactRef(
            artifact_id=f"{experiment_id}:{cell['model']['id']}",
            sha256=cell["raw_sha256"], quality=EvidenceQuality.SUMMARY_ONLY,
        ),),
        metrics=(
            MetricEvidence("outer_wall_ms", "ratio", MetricSource.MEASURED, (),
                           wall["median"], None, wall["ci_low"], wall["ci_high"]),
            MetricEvidence("physical_tokens_per_second", "ratio", MetricSource.MEASURED, (),
                           rate["median"], None, rate["ci_low"], rate["ci_high"]),
        ),
        correctness=CorrectnessEvidence(
            comparison_performed=True,
            token_identity=bool(comparison["token_identity"]),
            stop_reason_identity=bool(comparison["token_identity"]),
            token_count_identity=bool(comparison["token_identity"]),
            deterministic=bool(comparison["token_identity"]),
            state_identity=False,
            quality_class="engineering_exact_output_only",
        ),
        resources=ResourceEvidence(
            mlx_active_memory_bytes=None,
            mlx_peak_memory_bytes=resources["mlx_peak_memory_bytes"],
            rss_peak_bytes=None,
            swap_before_bytes=None,
            swap_after_bytes=None,
            timeout=False,
            crash_free=False,
            fallbacks=resources["fallbacks"],
            worker_status="summary_only_unverified",
            gates_passed=False,
        ),
        statistics=StatisticsEvidence(
            warmup_repeats=warmup_repeats,
            measured_repeats=measured_repeats,
            raw_sample_count=0,
            paired=True,
            arm_order=("AB", "BA"),
            estimator="paired_median_ratio",
            uncertainty_method="bootstrap_from_public_summary",
            confidence_level=0.95,
        ),
        evidence_quality=EvidenceQuality.SUMMARY_ONLY,
        recorded_at=recorded_at,
    )


__all__ = [
    "ActorRole", "ArtifactRef", "ClosedInterval", "CorrectnessEvidence", "DomainMatch",
    "DomainMatchStatus", "EvidenceQuality", "EvidenceRecord", "EvidenceStatus",
    "EvidenceValidationError", "ExecutionStrategy", "MetricEvidence", "MetricSource",
    "ResourceEvidence", "StatisticsEvidence", "TrustedExecutionProfile",
    "ValidityDomain", "canonical_json", "canonical_sha256", "domain_from_fingerprint",
    "evidence_from_b27_public_cell", "strategy_from_existing",
]
