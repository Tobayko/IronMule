"""Offline, fail-closed schemas for Q3 adaptive-method replay.

This module is intentionally a data contract only.  It has no runtime, model,
MLX, persistence, profile-selection, or execution surface.  Safety gates are
recorded as evidence and are deliberately not converted into optimizer policy.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, ClassVar, Iterable, Mapping

from .evidence import ArtifactRef, canonical_sha256


_DIGESTS = ("study", "model", "hardware", "framework", "workload", "time")
_KNOB_NAMES = (
    "fuse_projections",
    "compiled_fixed_cache",
    "fused_argmax",
    "head_skip_prefill",
    "prefill_into_fixed",
    "readback_every",
    "speculate_k",
    "speculate_ngram",
    "capacity_slack",
    "wired_fraction",
)
KNOB_NAMES = _KNOB_NAMES
KNOB_DEFAULTS = (
    ("fuse_projections", False), ("compiled_fixed_cache", False),
    ("fused_argmax", False), ("head_skip_prefill", False),
    ("prefill_into_fixed", False), ("readback_every", 1),
    ("speculate_k", 0), ("speculate_ngram", 3),
    ("capacity_slack", 0), ("wired_fraction", 0.0),
)
# This is a declarative mirror for replay validation, never an execution plan.
SEARCH_VALUES = (
    ("compiled_fixed_cache", (True,)), ("fused_argmax", (True,)),
    ("head_skip_prefill", (True,)), ("prefill_into_fixed", (True,)),
    ("readback_every", (2, 4, 8)), ("speculate_k", (4,)),
    ("capacity_slack", (128,)), ("wired_fraction", (0.6,)),
    ("fuse_projections", (True,)),
)
_BOOL_KNOBS = frozenset(_KNOB_NAMES[:5])
_INT_KNOBS = frozenset({"readback_every", "speculate_k", "speculate_ngram", "capacity_slack"})
_FLOAT_KNOBS = frozenset({"wired_fraction"})
_UNKNOWN = {"unknown", "missing", "none", "null", "unavailable"}


class AdaptiveValidationError(ValueError):
    """Raised when an adaptive record is ambiguous, incomplete, or non-canonical."""


class OutcomeStatus(str, Enum):
    MEASURED = "MEASURED"
    FAILED = "FAILED"
    INCONCLUSIVE = "INCONCLUSIVE"
    REJECTED = "REJECTED"
    INVALIDATED = "INVALIDATED"
    DATA_INSUFFICIENT = "DATA_INSUFFICIENT"


class RollbackStatus(str, Enum):
    NOT_REQUIRED = "NOT_REQUIRED"
    APPLIED = "APPLIED"
    FAILED = "FAILED"
    NOT_ATTEMPTED = "NOT_ATTEMPTED"


class ReplaySplit(str, Enum):
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    SEALED_HOLDOUT = "SEALED_HOLDOUT"


class Method(str, Enum):
    BASELINE = "BASELINE"
    CURRENT_COORDINATE = "CURRENT_COORDINATE"
    SEEDED_RANDOM = "SEEDED_RANDOM"
    BO = "BO"
    SURROGATE = "SURROGATE"
    CONTEXTUAL_BANDIT = "CONTEXTUAL_BANDIT"
    OFFLINE_RL = "OFFLINE_RL"


class EligibilityStatus(str, Enum):
    STRUCTURALLY_ELIGIBLE = "STRUCTURALLY_ELIGIBLE"
    DATA_INSUFFICIENT = "DATA_INSUFFICIENT"
    NOT_APPLICABLE = "NOT_APPLICABLE"


# Descriptive aliases keep callers from mistaking these lifecycle enums for
# evidence.py's evaluator-owned status vocabulary.
AdaptiveStatus = OutcomeStatus
AdaptiveMethod = Method
Split = ReplaySplit


def _text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or value.strip().lower() in _UNKNOWN:
        raise AdaptiveValidationError(f"{name} must be a known non-empty string")
    return value.strip()


def _digest(name: str, value: Any) -> str:
    value = _text(name, value)
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise AdaptiveValidationError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _int(name: str, value: Any, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AdaptiveValidationError(f"{name} must be an integer >= {minimum}")
    return value


def _signed_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AdaptiveValidationError(f"{name} must be an integer")
    return value


def _finite(name: str, value: Any, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdaptiveValidationError(f"{name} must be numeric")
    result = float(value)
    if result != result or result in (float("inf"), float("-inf")):
        raise AdaptiveValidationError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise AdaptiveValidationError(f"{name} must be >= {minimum}")
    return result


def _strict(data: Mapping[str, Any], expected: set[str], name: str) -> None:
    if not isinstance(data, Mapping):
        raise AdaptiveValidationError(f"{name} must be an object")
    actual = set(data)
    missing = sorted(expected - actual)
    unknown = sorted(repr(k) for k in actual - expected)
    if missing or unknown:
        raise AdaptiveValidationError(f"{name} fields differ; missing={missing!r}, unknown={unknown!r}")


def _pairs(name: str, value: Mapping[str, Any] | Iterable[tuple[str, Any]]) -> tuple[tuple[str, Any], ...]:
    if isinstance(value, Mapping):
        items = value.items()
    else:
        items = value
    result = []
    seen = set()
    for key, item in items:
        key = _text(f"{name}.key", key)
        if key in seen:
            raise AdaptiveValidationError(f"{name} keys must be unique")
        seen.add(key)
        result.append((key, _finite(f"{name}.{key}", item)))
    return tuple(sorted(result))


def _refs(name: str, value: Iterable[ArtifactRef]) -> tuple[ArtifactRef, ...]:
    result = tuple(value)
    if any(not isinstance(item, ArtifactRef) for item in result):
        raise AdaptiveValidationError(f"{name} must contain ArtifactRef values")
    ids = [item.artifact_id for item in result]
    if len(ids) != len(set(ids)):
        raise AdaptiveValidationError(f"{name} contains duplicate artifact IDs")
    return result


@dataclass(frozen=True, slots=True)
class AdaptiveContext:
    """Exact state identity used to group replay observations."""

    SCHEMA: ClassVar[str] = "ironmule.adaptive_context.v1"

    study_digest: str
    model_digest: str
    hardware_digest: str
    framework_digest: str
    workload_digest: str
    time_digest: str
    context_id: str = ""

    def __post_init__(self) -> None:
        for name in _DIGESTS:
            object.__setattr__(self, f"{name}_digest", _digest(f"context.{name}_digest", getattr(self, f"{name}_digest")))
        computed = canonical_sha256(self._semantic_dict())
        if self.context_id and self.context_id != computed:
            raise AdaptiveValidationError("context_id does not match canonical content")
        object.__setattr__(self, "context_id", computed)

    def _semantic_dict(self) -> dict[str, Any]:
        return {"schema": self.SCHEMA, **{field.name: getattr(self, field.name) for field in fields(self) if field.name != "context_id"}}

    def to_dict(self) -> dict[str, Any]:
        return {**self._semantic_dict(), "context_id": self.context_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AdaptiveContext":
        expected = {"schema", "context_id", *(f"{name}_digest" for name in _DIGESTS)}
        _strict(data, expected, "AdaptiveContext")
        if data["schema"] != cls.SCHEMA:
            raise AdaptiveValidationError("unsupported AdaptiveContext schema")
        return cls(*(data[f"{name}_digest"] for name in _DIGESTS), context_id=data["context_id"])


@dataclass(frozen=True, slots=True)
class KnobAction:
    """The closed ten-knob action space; no callable or executable payload."""

    SCHEMA: ClassVar[str] = "ironmule.knob_action.v1"

    fuse_projections: bool = False
    compiled_fixed_cache: bool = False
    fused_argmax: bool = False
    head_skip_prefill: bool = False
    prefill_into_fixed: bool = False
    readback_every: int = 1
    speculate_k: int = 0
    speculate_ngram: int = 3
    capacity_slack: int = 0
    wired_fraction: float = 0.0
    action_id: str = ""

    def __post_init__(self) -> None:
        for name in _BOOL_KNOBS:
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise AdaptiveValidationError(f"action.{name} must be boolean")
        for name in _INT_KNOBS:
            minimum = 1 if name == "readback_every" else 0
            object.__setattr__(self, name, _int(f"action.{name}", getattr(self, name), minimum))
        object.__setattr__(self, "wired_fraction", _finite("action.wired_fraction", self.wired_fraction, 0.0))
        if self.wired_fraction > 1.0:
            raise AdaptiveValidationError("action.wired_fraction must be <= 1")
        computed = canonical_sha256(self._semantic_dict())
        if self.action_id and self.action_id != computed:
            raise AdaptiveValidationError("action_id does not match canonical content")
        object.__setattr__(self, "action_id", computed)

    @classmethod
    def baseline(cls) -> "KnobAction":
        return cls()

    @property
    def key(self) -> str:
        return "|".join(f"{name}={getattr(self, name)}" for name in sorted(_KNOB_NAMES))

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in _KNOB_NAMES}

    def _semantic_dict(self) -> dict[str, Any]:
        return {"schema": self.SCHEMA, **{name: getattr(self, name) for name in _KNOB_NAMES}}

    def to_dict(self) -> dict[str, Any]:
        return {**self._semantic_dict(), "action_id": self.action_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "KnobAction":
        expected = {"schema", "action_id", *_KNOB_NAMES}
        _strict(data, expected, "KnobAction")
        if data["schema"] != cls.SCHEMA:
            raise AdaptiveValidationError("unsupported KnobAction schema")
        return cls(**{name: data[name] for name in _KNOB_NAMES}, action_id=data["action_id"])


@dataclass(frozen=True, slots=True)
class AdaptiveOutcome:
    """Measured or failed outcome; missing data remains missing and is never imputed."""

    SCHEMA: ClassVar[str] = "ironmule.adaptive_outcome.v1"

    raw_sample_refs: tuple[ArtifactRef, ...]
    raw_sample_count: int
    total_ns: float | None
    prefill_ns: float | None
    decode_ns: float | None
    token_identity: bool | None
    stop_reason_identity: bool | None
    token_count_identity: bool | None
    state_identity: bool | None
    deterministic: bool | None
    mlx_active_memory_bytes: int | None
    mlx_peak_memory_bytes: int | None
    rss_peak_bytes: int | None
    swap_delta_bytes: int | None
    timeout: bool
    crash: bool
    fallbacks: int
    hard_gates_passed: bool
    status: OutcomeStatus
    outcome_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_sample_refs", _refs("outcome.raw_sample_refs", self.raw_sample_refs))
        object.__setattr__(self, "raw_sample_count", _int("outcome.raw_sample_count", self.raw_sample_count))
        raw_refs = sum(item.quality.value == "RAW_SAMPLES" for item in self.raw_sample_refs)
        if self.raw_sample_count < raw_refs:
            raise AdaptiveValidationError("raw_sample_count cannot be below referenced samples")
        for name in ("total_ns", "prefill_ns", "decode_ns"):
            value = getattr(self, name)
            object.__setattr__(self, name, None if value is None else _finite(f"outcome.{name}", value, 0.0))
        for name in ("mlx_active_memory_bytes", "mlx_peak_memory_bytes", "rss_peak_bytes"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _int(f"outcome.{name}", value))
        if self.swap_delta_bytes is not None:
            object.__setattr__(self, "swap_delta_bytes", _signed_int("outcome.swap_delta_bytes", self.swap_delta_bytes))
        for name in ("token_identity", "stop_reason_identity", "token_count_identity", "state_identity", "deterministic", "timeout", "crash"):
            if not isinstance(getattr(self, name), bool) and getattr(self, name) is not None:
                raise AdaptiveValidationError(f"outcome.{name} must be boolean or null")
        object.__setattr__(self, "fallbacks", _int("outcome.fallbacks", self.fallbacks))
        if not isinstance(self.hard_gates_passed, bool):
            raise AdaptiveValidationError("outcome.hard_gates_passed must be boolean")
        try:
            object.__setattr__(self, "status", self.status if isinstance(self.status, OutcomeStatus) else OutcomeStatus(self.status))
        except (TypeError, ValueError) as exc:
            raise AdaptiveValidationError("outcome.status is not a closed OutcomeStatus") from exc
        computed = canonical_sha256(self._semantic_dict())
        if self.outcome_id and self.outcome_id != computed:
            raise AdaptiveValidationError("outcome_id does not match canonical content")
        object.__setattr__(self, "outcome_id", computed)

    def _semantic_dict(self) -> dict[str, Any]:
        return {"schema": self.SCHEMA, **{field.name: getattr(self, field.name) for field in fields(self) if field.name != "outcome_id"}}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "raw_sample_refs": [item.to_dict() for item in self.raw_sample_refs],
            "raw_sample_count": self.raw_sample_count,
            "total_ns": self.total_ns,
            "prefill_ns": self.prefill_ns,
            "decode_ns": self.decode_ns,
            "token_identity": self.token_identity,
            "stop_reason_identity": self.stop_reason_identity,
            "token_count_identity": self.token_count_identity,
            "state_identity": self.state_identity,
            "deterministic": self.deterministic,
            "mlx_active_memory_bytes": self.mlx_active_memory_bytes,
            "mlx_peak_memory_bytes": self.mlx_peak_memory_bytes,
            "rss_peak_bytes": self.rss_peak_bytes,
            "swap_delta_bytes": self.swap_delta_bytes,
            "timeout": self.timeout,
            "crash": self.crash,
            "fallbacks": self.fallbacks,
            "hard_gates_passed": self.hard_gates_passed,
            "status": self.status.value,
            "outcome_id": self.outcome_id,
        }

    @property
    def measurements_complete(self) -> bool:
        return all(getattr(self, name) is not None for name in ("total_ns", "prefill_ns", "decode_ns"))

    @property
    def correctness_complete(self) -> bool:
        return all(getattr(self, name) is True for name in ("token_identity", "stop_reason_identity", "token_count_identity", "state_identity", "deterministic"))

    @property
    def resources_complete(self) -> bool:
        return all(getattr(self, name) is not None for name in ("mlx_active_memory_bytes", "mlx_peak_memory_bytes", "rss_peak_bytes", "swap_delta_bytes"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AdaptiveOutcome":
        expected = {"schema", "outcome_id", *(field.name for field in fields(cls) if field.name not in {"SCHEMA", "outcome_id"})}
        _strict(data, expected, "AdaptiveOutcome")
        if data["schema"] != cls.SCHEMA:
            raise AdaptiveValidationError("unsupported AdaptiveOutcome schema")
        values = {field.name: data[field.name] for field in fields(cls) if field.name not in {"SCHEMA", "outcome_id"}}
        values["raw_sample_refs"] = tuple(ArtifactRef.from_dict(item) for item in data["raw_sample_refs"])
        return cls(**values, outcome_id=data["outcome_id"])


@dataclass(frozen=True, slots=True)
class AdaptiveObservation:
    """One immutable state -> action -> outcome transition."""

    SCHEMA: ClassVar[str] = "ironmule.adaptive_observation.v1"

    context: AdaptiveContext
    action: KnobAction
    measurements: tuple[tuple[str, float], ...]
    uncertainty: tuple[tuple[str, float], ...]
    outcome: AdaptiveOutcome
    rollback: RollbackStatus
    evidence: tuple[ArtifactRef, ...]
    split: ReplaySplit = ReplaySplit.TRAIN
    group_key: str = ""
    observation_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.context, AdaptiveContext) or not isinstance(self.action, KnobAction) or not isinstance(self.outcome, AdaptiveOutcome):
            raise AdaptiveValidationError("observation context/action/outcome types are strict")
        object.__setattr__(self, "measurements", _pairs("observation.measurements", self.measurements))
        object.__setattr__(self, "uncertainty", _pairs("observation.uncertainty", self.uncertainty))
        object.__setattr__(self, "evidence", _refs("observation.evidence", self.evidence))
        object.__setattr__(self, "rollback", self.rollback if isinstance(self.rollback, RollbackStatus) else RollbackStatus(self.rollback))
        object.__setattr__(self, "split", self.split if isinstance(self.split, ReplaySplit) else ReplaySplit(self.split))
        group = self.context.context_id
        if self.group_key and self.group_key != group:
            raise AdaptiveValidationError("observation.group_key must equal context.context_id")
        object.__setattr__(self, "group_key", group)
        computed = canonical_sha256(self._semantic_dict())
        if self.observation_id and self.observation_id != computed:
            raise AdaptiveValidationError("observation_id does not match canonical content")
        object.__setattr__(self, "observation_id", computed)

    def _semantic_dict(self) -> dict[str, Any]:
        return {"schema": self.SCHEMA, "context": self.context, "action": self.action, "measurements": self.measurements, "uncertainty": self.uncertainty, "outcome": self.outcome, "rollback": self.rollback, "evidence": self.evidence, "split": self.split, "group_key": self.group_key}

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.SCHEMA, "context": self.context.to_dict(), "action": self.action.to_dict(), "measurements": dict(self.measurements), "uncertainty": dict(self.uncertainty), "outcome": self.outcome.to_dict(), "rollback": self.rollback.value, "evidence": [item.to_dict() for item in self.evidence], "split": self.split.value, "group_key": self.group_key, "observation_id": self.observation_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AdaptiveObservation":
        expected = {"schema", "context", "action", "measurements", "uncertainty", "outcome", "rollback", "evidence", "split", "group_key", "observation_id"}
        _strict(data, expected, "AdaptiveObservation")
        if data["schema"] != cls.SCHEMA:
            raise AdaptiveValidationError("unsupported AdaptiveObservation schema")
        return cls(context=AdaptiveContext.from_dict(data["context"]), action=KnobAction.from_dict(data["action"]), measurements=data["measurements"], uncertainty=data["uncertainty"], outcome=AdaptiveOutcome.from_dict(data["outcome"]), rollback=data["rollback"], evidence=tuple(ArtifactRef.from_dict(item) for item in data["evidence"]), split=data["split"], group_key=data["group_key"], observation_id=data["observation_id"])


def _is_complete(observation: AdaptiveObservation) -> bool:
    outcome = observation.outcome
    return bool(
        outcome.status is OutcomeStatus.MEASURED
        and outcome.hard_gates_passed
        and not outcome.timeout and not outcome.crash and outcome.fallbacks == 0
        and outcome.raw_sample_refs and observation.evidence
        and outcome.raw_sample_count and outcome.measurements_complete
        and outcome.correctness_complete and outcome.resources_complete
        and observation.rollback in {RollbackStatus.NOT_REQUIRED, RollbackStatus.APPLIED}
        and observation.uncertainty
    )


@dataclass(frozen=True, slots=True)
class ReplayDataset:
    """In-memory replay corpus with leakage-safe group validation."""

    SCHEMA: ClassVar[str] = "ironmule.replay_dataset.v1"

    observations: tuple[AdaptiveObservation, ...]
    action_pool: tuple[KnobAction, ...]
    dataset_id: str = ""

    def __post_init__(self) -> None:
        observations = tuple(self.observations)
        if any(not isinstance(item, AdaptiveObservation) for item in observations):
            raise AdaptiveValidationError("dataset observations must be AdaptiveObservation values")
        ids = [item.observation_id for item in observations]
        if len(ids) != len(set(ids)):
            raise AdaptiveValidationError("duplicate observation IDs")
        action_pool = tuple(self.action_pool)
        if not action_pool:
            raise AdaptiveValidationError("dataset action_pool must be explicitly declared and non-empty")
        if any(not isinstance(item, KnobAction) for item in action_pool):
            raise AdaptiveValidationError("dataset action_pool must contain KnobAction values")
        if len({item.action_id for item in action_pool}) != len(action_pool):
            raise AdaptiveValidationError("duplicate action IDs in action_pool")
        pool_ids = {item.action_id for item in action_pool}
        if any(item.action.action_id not in pool_ids for item in observations):
            raise AdaptiveValidationError("observation action is absent from action_pool")
        group_splits: dict[str, ReplaySplit] = {}
        context_splits: dict[str, ReplaySplit] = {}
        context_actions: set[tuple[str, str]] = set()
        for item in observations:
            previous = group_splits.setdefault(item.group_key, item.split)
            if previous is not item.split:
                raise AdaptiveValidationError("one group cannot cross replay splits")
            previous_context = context_splits.setdefault(item.context.context_id, item.split)
            if previous_context is not item.split:
                raise AdaptiveValidationError("one context cannot cross replay splits")
            context_action = (item.context.context_id, item.action.action_id)
            if context_action in context_actions:
                raise AdaptiveValidationError("duplicate context/action observation")
            context_actions.add(context_action)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "action_pool", action_pool)
        computed = canonical_sha256(self._semantic_dict())
        if self.dataset_id and self.dataset_id != computed:
            raise AdaptiveValidationError("dataset_id does not match canonical content")
        object.__setattr__(self, "dataset_id", computed)

    def _semantic_dict(self) -> dict[str, Any]:
        return {"schema": self.SCHEMA, "action_pool": self.action_pool, "observations": self.observations}

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.SCHEMA, "action_pool": [item.to_dict() for item in self.action_pool], "observations": [item.to_dict() for item in self.observations], "dataset_id": self.dataset_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReplayDataset":
        _strict(data, {"schema", "action_pool", "observations", "dataset_id"}, "ReplayDataset")
        if data["schema"] != cls.SCHEMA or not isinstance(data["observations"], list):
            raise AdaptiveValidationError("unsupported ReplayDataset schema or observations")
        return cls(tuple(AdaptiveObservation.from_dict(item) for item in data["observations"]), action_pool=tuple(KnobAction.from_dict(item) for item in data["action_pool"]), dataset_id=data["dataset_id"])

    def coverage_report(self) -> dict[str, Any]:
        complete = sum(_is_complete(item) for item in self.observations)
        failures = sum(item.outcome.status is not OutcomeStatus.MEASURED or not item.outcome.hard_gates_passed or item.outcome.crash or item.outcome.timeout or item.outcome.fallbacks > 0 for item in self.observations)
        by_split = {split.value: sum(item.split is split for item in self.observations) for split in ReplaySplit}
        return {"schema": "ironmule.adaptive_coverage.v1", "observation_count": len(self.observations), "complete_observation_count": complete, "unique_contexts": len({item.context.context_id for item in self.observations}), "unique_actions": len({item.action.action_id for item in self.observations}), "action_pool_size": len(self.action_pool), "action_pool_hash": canonical_sha256(self.action_pool), "unique_groups": len({item.group_key for item in self.observations}), "failures_or_invalids": failures, "splits": by_split, "sequential_horizons": 0, "raw_sample_count": sum(item.outcome.raw_sample_count for item in self.observations), "no_invented_performance": True}

    def method_eligibility(self) -> dict[str, Any]:
        coverage = self.coverage_report()
        panels = self._complete_panels()
        complete_splits = {split for _, split in panels}
        comparable_contexts = len({context for context, _ in panels})
        result = {}
        for method in Method:
            if method is Method.OFFLINE_RL:
                status = EligibilityStatus.NOT_APPLICABLE
                reason = "v1 contains static knob actions and no measured sequential horizon"
            elif method in {Method.BO, Method.SURROGATE, Method.CONTEXTUAL_BANDIT} and complete_splits != set(ReplaySplit):
                status = EligibilityStatus.DATA_INSUFFICIENT
                reason = "requires a declared action_pool and complete counterfactual panels in separate TRAIN/VALIDATION/SEALED_HOLDOUT groups"
            elif method is Method.CONTEXTUAL_BANDIT and comparable_contexts < 2:
                status = EligibilityStatus.DATA_INSUFFICIENT
                reason = "requires multiple independent contexts with comparable complete declared-action panels"
            elif method in {Method.CURRENT_COORDINATE, Method.SEEDED_RANDOM} and not panels:
                status = EligibilityStatus.DATA_INSUFFICIENT
                reason = "requires one complete safe panel for the declared action_pool"
            elif method is Method.BASELINE:
                baseline = KnobAction.baseline().action_id
                status = EligibilityStatus.STRUCTURALLY_ELIGIBLE if any(_is_complete(item) and item.action.action_id == baseline for item in self.observations) else EligibilityStatus.DATA_INSUFFICIENT
                reason = "complete safe baseline observation required"
            else:
                status = EligibilityStatus.STRUCTURALLY_ELIGIBLE if panels else EligibilityStatus.DATA_INSUFFICIENT
                reason = "complete safe declared-action panel" if panels else "requires one complete safe panel for the declared action_pool"
            result[method.value] = {"status": status.value, "reason": reason, "safety_gates_external": True, "structural_minimum_is_not_statistical_qualification": True}
        return {"schema": "ironmule.adaptive_method_eligibility.v1", "coverage": coverage, "methods": result}

    def _complete_panels(self) -> tuple[tuple[str, ReplaySplit], ...]:
        panels = []
        expected = {item.action_id for item in self.action_pool}
        if len(expected) < 2:
            return ()
        for context_id in {item.context.context_id for item in self.observations}:
            rows = [item for item in self.observations if item.context.context_id == context_id]
            if {item.action.action_id for item in rows} == expected and all(_is_complete(item) for item in rows):
                panels.append((context_id, rows[0].split))
        return tuple(panels)

    def next_missing_evidence(self) -> dict[str, Any]:
        return {
            "schema": "ironmule.adaptive_voi.v1",
            "needs": {
                "cheapest_action_panel": {
                    "need": "complete_raw_counterfactual_action_panel",
                    "request": "complete raw counterfactual action panel in isolated fresh processes",
                    "executable": False,
                    "decisive_for": [Method.CURRENT_COORDINATE.value, Method.SEEDED_RANDOM.value, Method.BO.value, Method.SURROGATE.value],
                },
                "independent_contexts": {
                    "need": "independent_grouped_contexts_with_comparable_action_panels",
                    "request": "independent grouped contexts with comparable actions and sealed splits",
                    "executable": False,
                    "decisive_for": [Method.CONTEXTUAL_BANDIT.value, Method.SURROGATE.value],
                },
                "sequential_horizon": {
                    "need": "measured_sequential_horizon",
                    "request": "a preregistered sequential state/action/outcome trajectory",
                    "executable": False,
                    "decisive_for": [Method.OFFLINE_RL.value],
                    "status": EligibilityStatus.NOT_APPLICABLE.value,
                },
            },
            "no_performance_estimate": True,
        }


def coverage_report(dataset: ReplayDataset) -> dict[str, Any]:
    """Return structural coverage without dropping failures or inventing metrics."""
    return dataset.coverage_report()


def method_eligibility(dataset: ReplayDataset) -> dict[str, Any]:
    return dataset.method_eligibility()


def next_missing_evidence(dataset: ReplayDataset) -> dict[str, Any]:
    return dataset.next_missing_evidence()


__all__ = [
    "AdaptiveContext", "KnobAction", "AdaptiveOutcome", "AdaptiveObservation", "ReplayDataset",
    "OutcomeStatus", "RollbackStatus", "ReplaySplit", "Method", "EligibilityStatus",
    "AdaptiveStatus", "AdaptiveMethod", "Split",
    "AdaptiveValidationError", "coverage_report", "method_eligibility", "next_missing_evidence",
]
