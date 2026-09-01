"""Strict, offline contracts for the Q4 evidence-bound RL pilot.

This module is intentionally boring.  It contains immutable records and
canonical serializers only; importing it must never import a runtime, model,
MLX, process executor, tuner, network client, or persistence layer.  The
historical corpus builder uses these records as a validation boundary, while
the later replay/RL implementation can depend on the same boundary.

The Q4 preregistration is the source of truth for the vocabulary and gates.
In particular, missing values are not imputed, S11/S12 are risk probes, and
historical Q3 evidence never becomes a Q4 split merely by being imported.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, ClassVar, Iterable, Mapping, Sequence

# These two modules are existing offline contracts.  They are deliberately the
# only project imports here; no package import below this boundary can reach
# the runtime when this module is loaded under the normal offline loader.
try:  # pragma: no cover - direct source loading has no package parent
    from .evidence import ArtifactRef, EvidenceQuality
except ImportError:  # pragma: no cover
    ArtifactRef = Any  # type: ignore[misc,assignment]
    EvidenceQuality = Any  # type: ignore[misc,assignment]


SCHEMA_PREFIX = "ironmule.q4_"
HORIZON = 17
KNOB_COUNT = 12
STRATEGY_COUNT = 5
ACTION_COUNT = 12
REQUIRED_CONTEXTS = {"Q4_TRAIN": 12, "Q4_VALIDATION": 6, "Q4_SEALED_HOLDOUT": 6}
TRAJECTORIES_PER_CONTEXT = 3
REQUIRED_COMPLETE_TRAJECTORIES = 72
REQUIRED_COMPLETE_TRANSITIONS = 1224

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")
_UNKNOWN = frozenset({"", "unknown", "missing", "none", "null", "unavailable"})
_EXECUTABLE_KEYS = frozenset({
    "callable", "callback", "command", "cmd", "argv", "exec", "executable",
    "program", "script", "shell", "subprocess", "payload", "code", "source_code",
    "python", "module", "import", "entrypoint", "url", "uri",
})


class Q4ValidationError(ValueError):
    """Raised when a Q4 record is ambiguous, malformed, or unsafe."""


class ObjectiveClass(str, Enum):
    LATENCY = "LATENCY"
    THROUGHPUT = "THROUGHPUT"


class WorkloadStratum(str, Enum):
    HOMOGENEOUS = "homogeneous"
    HETEROGENEOUS = "heterogeneous"
    STAGGERED = "staggered"
    TERSE = "terse"


class PlanKind(str, Enum):
    STRICT = "StrictOneShotPlan"
    REUSABLE = "ReusableSessionPlan"


class Stage(str, Enum):
    KNOB_DELTA = "KNOB_DELTA"
    STRATEGY_SELECT = "STRATEGY_SELECT"
    REVALIDATE = "REVALIDATE"


class ActionSpace(str, Enum):
    KNOB_DELTA = "KNOB_DELTA"
    STRATEGY_SELECT = "STRATEGY_SELECT"
    REVALIDATE = "REVALIDATE"


class Q4Split(str, Enum):
    TRAIN = "Q4_TRAIN"
    VALIDATION = "Q4_VALIDATION"
    SEALED_HOLDOUT = "Q4_SEALED_HOLDOUT"


class TrajectoryStatus(str, Enum):
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    ABORTED = "ABORTED"


class FailureState(str, Enum):
    PREFLIGHT = "PREFLIGHT"
    KNOB_DELTA = "KNOB_DELTA"
    STRATEGY_SELECT = "STRATEGY_SELECT"
    REVALIDATE = "REVALIDATE"
    CLEANUP = "CLEANUP"
    UNKNOWN = "UNKNOWN"


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


class SemanticClass(str, Enum):
    EXACT = "exact"
    RISK_PROBE = "risk_probe"


class StrategyClass(str, Enum):
    EXISTING_EXECUTION_STRATEGY = "EXISTING_EXECUTION_STRATEGY"
    RISK_PROBE = "RISK_PROBE"


class HistoricalRole(str, Enum):
    Q3_VALIDATION = "Q3_VALIDATION"
    Q3_SEALED_HOLDOUT = "Q3_SEALED_HOLDOUT"
    LEDGER_ONLY = "LEDGER_ONLY"
    PRIOR_ONLY = "PRIOR_ONLY"
    CENSORED_FAILURE = "CENSORED_FAILURE"


def _canonical(value: Any) -> Any:
    """Convert values to canonical JSON while rejecting unsafe values."""
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _canonical(value.to_dict())
    if is_dataclass(value):
        return {
            field.name: _canonical(getattr(value, field.name))
            for field in fields(value)
            if not isinstance(getattr(type(value), field.name, None), ClassVar)
        }
    if isinstance(value, Mapping):
        keys = list(value)
        if any(not isinstance(key, str) for key in keys):
            raise Q4ValidationError("canonical object keys must be strings")
        result = {}
        for key in sorted(keys):
            _safe_key(key)
            result[key] = _canonical(value[key])
        return result
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise Q4ValidationError("canonical JSON forbids NaN and Infinity")
        return value
    raise Q4ValidationError(f"unsupported canonical value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Return deterministic UTF-8 JSON with no executable or non-finite data."""
    return json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _safe_key(key: str) -> None:
    if key.strip().lower() in _EXECUTABLE_KEYS:
        raise Q4ValidationError(f"executable field is not permitted: {key!r}")


def _text(name: str, value: Any, *, known: bool = True) -> str:
    if type(value) is not str:
        raise Q4ValidationError(f"{name} must be a string")
    value = value.strip()
    if known and value.lower() in _UNKNOWN:
        raise Q4ValidationError(f"{name} must be known and non-empty")
    if (value.startswith(("/", "\\")) or _WINDOWS_ABSOLUTE.match(value)
            or value.lower().startswith(("file:", "http:", "https:"))):
        raise Q4ValidationError(f"{name} must not contain a path or URL")
    return value


def _digest(name: str, value: Any, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    value = _text(name, value)
    if not _DIGEST.fullmatch(value):
        raise Q4ValidationError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _integer(name: str, value: Any, minimum: int = 0, maximum: int | None = None) -> int:
    if type(value) is not int or value < minimum or maximum is not None and value > maximum:
        bound = f"{minimum}..{maximum}" if maximum is not None else f">={minimum}"
        raise Q4ValidationError(f"{name} must be an integer in {bound}")
    return value


def _signed_integer(name: str, value: Any) -> int:
    if type(value) is not int:
        raise Q4ValidationError(f"{name} must be an integer")
    return value


def _number(name: str, value: Any, minimum: float | None = None,
            maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Q4ValidationError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise Q4ValidationError(f"{name} must be finite")
    if minimum is not None and result < minimum or maximum is not None and result > maximum:
        raise Q4ValidationError(f"{name} is outside [{minimum}, {maximum}]")
    return result


def _enum(enum_type: type[Enum], name: str, value: Any) -> Enum:
    try:
        return value if isinstance(value, enum_type) else enum_type(value)
    except (TypeError, ValueError) as exc:
        raise Q4ValidationError(f"{name} is not in the closed vocabulary") from exc


def _strict(data: Mapping[str, Any], expected: set[str], name: str) -> None:
    if not isinstance(data, Mapping):
        raise Q4ValidationError(f"{name} must be an object")
    actual = set(data)
    if any(not isinstance(key, str) for key in actual):
        raise Q4ValidationError(f"{name} keys must be strings")
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise Q4ValidationError(f"{name} fields differ; missing={missing!r}, unknown={unknown!r}")
    for key in actual:
        _safe_key(key)


def _string_tuple(name: str, value: Iterable[Any], *, unique: bool = True, allow_empty: bool = False) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise Q4ValidationError(f"{name} must be an array")
    result = tuple(_text(f"{name}[]", item) for item in value)
    if not result and not allow_empty:
        raise Q4ValidationError(f"{name} must contain at least one value")
    if unique and len(result) != len(set(result)):
        raise Q4ValidationError(f"{name} contains duplicates")
    return result


def _bool_map(name: str, value: Mapping[str, Any]) -> tuple[tuple[str, bool], ...]:
    if not isinstance(value, Mapping) or not value:
        raise Q4ValidationError(f"{name} must be a non-empty object")
    result = []
    for key, item in value.items():
        key = _text(f"{name}.key", key)
        if type(item) is not bool:
            raise Q4ValidationError(f"{name}.{key} must be boolean")
        result.append((key, item))
    return tuple(sorted(result))


def _utc(name: str, value: Any) -> str:
    value = _text(name, value)
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise Q4ValidationError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise Q4ValidationError(f"{name} must carry UTC offset")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _ref_tuple(name: str, refs: Iterable[Any]) -> tuple[Any, ...]:
    if isinstance(refs, (str, bytes)):
        raise Q4ValidationError(f"{name} must be an array")
    refs = tuple(refs)
    if not refs:
        raise Q4ValidationError(f"{name} must contain at least one reference")
    for ref in refs:
        if not isinstance(ref, ArtifactRef):
            raise Q4ValidationError(f"{name} must contain ArtifactRef records")
    ids = [ref.artifact_id for ref in refs]
    if len(ids) != len(set(ids)):
        raise Q4ValidationError(f"{name} contains duplicate artifact IDs")
    return refs


# ---------------------------------------------------------------------------
# Knob and strategy spaces

KNOB_NAMES = (
    "fuse_projections", "compiled_fixed_cache", "fused_argmax", "head_skip_prefill",
    "prefill_into_fixed", "readback_every", "speculate_k", "speculate_ngram",
    "capacity_slack", "wired_fraction",
)
_BOOL_KNOBS = frozenset(KNOB_NAMES[:5])
_INT_KNOBS = frozenset({"readback_every", "speculate_k", "speculate_ngram", "capacity_slack"})
_SEARCH_VALUES = (
    ("compiled_fixed_cache", (True,)), ("fused_argmax", (True,)),
    ("head_skip_prefill", (True,)), ("prefill_into_fixed", (True,)),
    ("readback_every", (2, 4, 8)), ("speculate_k", (4,)),
    ("capacity_slack", (128,)), ("wired_fraction", (0.6,)),
    ("fuse_projections", (True,)),
)
_BASE_KNOB_VALUES = {
    "fuse_projections": (False, True), "compiled_fixed_cache": (False, True),
    "fused_argmax": (False, True), "head_skip_prefill": (False, True),
    "prefill_into_fixed": (False, True), "readback_every": (1, 2, 4, 8),
    "speculate_k": (0, 4), "speculate_ngram": (3,), "capacity_slack": (0, 128),
    "wired_fraction": (0.0, 0.6),
}


@dataclass(frozen=True, slots=True)
class KnobAction:
    """A complete legal, non-executable knob state.

    ``KNOB_ACTIONS`` is the frozen twelve-slot measurement panel; this class
    also represents the finite declared state space used by trajectory deltas
    and the Q2-current interaction anchor.
    """

    SCHEMA: ClassVar[str] = "ironmule.q4_knob_action.v1"

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
            if type(getattr(self, name)) is not bool:
                raise Q4ValidationError(f"knob.{name} must be boolean")
        for name in _INT_KNOBS:
            minimum = 1 if name == "readback_every" else 0
            object.__setattr__(self, name, _integer(f"knob.{name}", getattr(self, name), minimum))
        object.__setattr__(self, "wired_fraction", _number("knob.wired_fraction", self.wired_fraction, 0, 1))
        for name in KNOB_NAMES:
            if getattr(self, name) not in _BASE_KNOB_VALUES[name]:
                raise Q4ValidationError(f"knob.{name} is outside the frozen declared values")
        computed = canonical_sha256(self._semantic_dict())
        if self.action_id and self.action_id != computed:
            raise Q4ValidationError("knob.action_id does not match canonical content")
        object.__setattr__(self, "action_id", computed)

    @classmethod
    def baseline(cls) -> "KnobAction":
        return cls()

    @classmethod
    def q2_current(cls) -> "KnobAction":
        return cls(compiled_fixed_cache=True, head_skip_prefill=True, readback_every=2)

    def _semantic_dict(self) -> dict[str, Any]:
        return {"schema": self.SCHEMA, **{name: getattr(self, name) for name in KNOB_NAMES}}

    def to_dict(self) -> dict[str, Any]:
        return {**self._semantic_dict(), "action_id": self.action_id}

    as_dict = lambda self: {name: getattr(self, name) for name in KNOB_NAMES}

    @property
    def key(self) -> str:
        return "|".join(f"{name}={getattr(self, name)}" for name in sorted(KNOB_NAMES))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "KnobAction":
        _strict(data, {"schema", "action_id", *KNOB_NAMES}, cls.__name__)
        if data["schema"] != cls.SCHEMA:
            raise Q4ValidationError("unsupported q4 knob schema")
        return cls(**{name: data[name] for name in KNOB_NAMES}, action_id=data["action_id"])


def knob_catalogue() -> tuple[KnobAction, ...]:
    # The preregistered panel is BASE plus one independent candidate for each
    # declared search slot.  It is *not* a replay of an acceptance path: a
    # rejected candidate must not accidentally become the source of the next
    # candidate.  Dynamic trajectory states are validated separately below.
    baseline = KnobAction.baseline()
    result = [baseline]
    for name, values in _SEARCH_VALUES:
        for value in values:
            values_for_action = baseline.as_dict()
            values_for_action[name] = value
            # The twelve-anchor interaction panel reserves the readback=2
            # slot for the exact Q2-current anchor.  CandidateSpecs remain
            # independent one-field slots; this replacement is only for the
            # interaction cross-product identity.
            if name == "readback_every" and value == 2:
                values_for_action.update({
                    "compiled_fixed_cache": True,
                    "head_skip_prefill": True,
                })
            result.append(KnobAction(**values_for_action))
    if len(result) != KNOB_COUNT or len({item.action_id for item in result}) != KNOB_COUNT:
        raise Q4ValidationError("frozen knob catalogue is not twelve unique actions")
    return tuple(result)


KNOB_ACTIONS = knob_catalogue()
KNOB_ACTION_BY_ID = {item.action_id: item for item in KNOB_ACTIONS}
Q2_CURRENT_ACTION = KnobAction.q2_current()
if Q2_CURRENT_ACTION.action_id not in {item.action_id for item in KNOB_ACTIONS}:
    raise Q4ValidationError("Q2-current anchor is absent from the twelve interaction anchors")
INTERACTION_KNOB_ANCHORS = KNOB_ACTIONS


@dataclass(frozen=True, slots=True)
class KnobCandidateSpec:
    """One fixed panel slot; this is metadata, never an executable delta."""

    SCHEMA: ClassVar[str] = "ironmule.q4_knob_candidate.v1"

    changed_field: str
    target_value: bool | int | float
    candidate_id: str = ""

    def __post_init__(self) -> None:
        if self.changed_field not in KNOB_NAMES:
            raise Q4ValidationError("candidate changed_field is not a knob")
        baseline = KnobAction.baseline()
        values = baseline.as_dict()
        values[self.changed_field] = self.target_value
        target = KnobAction(**values)
        if getattr(target, self.changed_field) != self.target_value:
            raise Q4ValidationError("candidate target_value does not match canonical knob type")
        computed = canonical_sha256(self._semantic_dict())
        if self.candidate_id and self.candidate_id != computed:
            raise Q4ValidationError("candidate_id does not match canonical content")
        object.__setattr__(self, "candidate_id", computed)

    def _semantic_dict(self) -> dict[str, Any]:
        return {"schema": self.SCHEMA, "changed_field": self.changed_field, "target_value": self.target_value}

    def to_dict(self) -> dict[str, Any]:
        return {**self._semantic_dict(), "candidate_id": self.candidate_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "KnobCandidateSpec":
        _strict(data, {"schema", "changed_field", "target_value", "candidate_id"}, cls.__name__)
        if data["schema"] != cls.SCHEMA:
            raise Q4ValidationError("unsupported q4 knob candidate schema")
        return cls(data["changed_field"], data["target_value"], data["candidate_id"])


KNOB_CANDIDATES = tuple(KnobCandidateSpec(name, value) for name, values in _SEARCH_VALUES for value in values)
KNOB_CANDIDATE_BY_ID = {item.candidate_id: item for item in KNOB_CANDIDATES}


def _all_declared_knob_actions() -> dict[str, KnobAction]:
    """Build the finite legal complete-state vocabulary for delta validation."""
    choices: dict[str, tuple[Any, ...]] = {name: (getattr(KnobAction.baseline(), name),) for name in KNOB_NAMES}
    for name, values in _SEARCH_VALUES:
        choices[name] = tuple(dict.fromkeys((choices[name][0], *values)))
    result: dict[str, KnobAction] = {}
    # Keep this deterministic and stdlib-only.  The state space is small
    # (1024 states) and is built once at import, not by any runtime path.
    import itertools
    for values in itertools.product(*(choices[name] for name in KNOB_NAMES)):
        action = KnobAction(**dict(zip(KNOB_NAMES, values)))
        result[action.action_id] = action
    return result


ALL_DECLARED_KNOB_ACTIONS = _all_declared_knob_actions()
LEGAL_KNOB_ACTIONS = tuple(sorted(ALL_DECLARED_KNOB_ACTIONS.values(), key=lambda item: item.action_id))


_STRATEGY_FIELDS = (
    "stage", "semantic_class", "plan_kind", "service_mode", "prefill_policy", "decode_policy",
    "scheduling_policy", "grouping_policy", "grouping_width", "synchronization_policy",
    "cache_policy", "prefix_reuse_policy", "memory_policy", "compile_graph_policy",
    "strategy_class",
)

_FROZEN_STRATEGY_IDS = frozenset({
    "712a6d6ea2cf1bb588fcd74a509f52dac5015b08f3b4bc5ae067232592c3a56a",
    "a97214109ad4b9f3c74ee0d3cc69a9925ae63b53d44e23ab4ca4801905e0d7ce",
    "9e9dc84b74669691fbcf8e4d4e617fc0bf09a0ebd310a276985cbfe46a75cc5f",
    "f27fcef617ab253adab4e86cc4a1a09f6a65cee9ef77ecbcb35ed9bff98ac25f",
    "b72cbd0476ec1014c3ac2dcd81163f543a0963a2c34f223c8746edd4bcfd754a",
    "5167fcbde4d9bf61ed89b711f2fbd366536e098502721762b1eed48677a3804c",
    "37b54ee14dd36c11cf31b49eea0471f32b90069b694858d6ecfe3240dedb8a90",
    "b100e7dde084b673fddfb15677021fcdb14639b99d5b6d1ab07999deee6ec4c2",
    "9ce45f487d9c9ca362532b61c30227a039261b3e0c360ad4d277bc20e0f250db",
    "3c61c473394070a7bc77bf41518ad463a9a3cf7b4d688f7613368a802d44a123",
    "8a9b3914099c51543335468f3bec9fc901ad8a047be26f02c74818b26c1c5608",
    "3c1fd4624bfbff81a87782ce87673f39f1fcaf8f806c8f1a02123596ef20758d",
})


def _strategy_values(label: str) -> dict[str, Any]:
    plan = PlanKind.STRICT.value if label in {"S01", "S02", "S03", "S04", "S05", "S11"} else PlanKind.REUSABLE.value
    reusable = plan == PlanKind.REUSABLE.value
    interactive = label in {"S01", "S06"}
    risk = label in {"S11", "S12"}
    return {
        "stage": Stage.STRATEGY_SELECT.value,
        "semantic_class": SemanticClass.RISK_PROBE.value if risk else SemanticClass.EXACT.value,
        "plan_kind": plan,
        "service_mode": "InteractiveMode" if interactive is True else "ThroughputMode",
        "prefill_policy": "reusable_session" if reusable else "strict_one_shot",
        "decode_policy": "greedy",
        "scheduling_policy": "true_batch_risk_probe" if risk else ("sequential" if interactive else "async_grouped_b1"),
        "grouping_policy": "true_batch_risk_probe" if risk else ("none" if interactive else "grouped_batch1"),
        "grouping_width": 4 if risk else (1 if interactive else {
            "S02": 1, "S03": 2, "S04": 3, "S05": 4,
            "S07": 1, "S08": 2, "S09": 3, "S10": 4,
        }[label]),
        "synchronization_policy": "group_barrier" if not interactive else "per_request",
        "cache_policy": "fixed_shape" if reusable else "standard",
        "prefix_reuse_policy": "exact_reuse" if reusable else "disabled",
        "memory_policy": "existing",
        "compile_graph_policy": "existing",
        "strategy_class": StrategyClass.RISK_PROBE.value if risk else StrategyClass.EXISTING_EXECUTION_STRATEGY.value,
    }


@dataclass(frozen=True, slots=True)
class ScheduleAction:
    """A concrete S01--S12 path descriptor with no executable payload."""

    SCHEMA: ClassVar[str] = "ironmule.q4_execution_strategy_action.v1"

    stage: str
    semantic_class: str
    plan_kind: str
    service_mode: str
    prefill_policy: str
    decode_policy: str
    scheduling_policy: str
    grouping_policy: str
    grouping_width: int
    synchronization_policy: str
    cache_policy: str
    prefix_reuse_policy: str
    memory_policy: str
    compile_graph_policy: str
    strategy_class: str
    action_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", _enum(Stage, "strategy.stage", self.stage).value)
        object.__setattr__(self, "semantic_class", _enum(SemanticClass, "strategy.semantic_class", self.semantic_class).value)
        object.__setattr__(self, "plan_kind", _enum(PlanKind, "strategy.plan_kind", self.plan_kind).value)
        for name in _STRATEGY_FIELDS:
            if name not in {"stage", "semantic_class", "plan_kind", "grouping_width", "strategy_class"}:
                # ``none`` is a frozen policy value for grouping; all other
                # free-text identity fields still use the known-value guard.
                object.__setattr__(self, name, _text(f"strategy.{name}", getattr(self, name), known=False))
        object.__setattr__(self, "grouping_width", _integer("strategy.grouping_width", self.grouping_width, 1, 4))
        object.__setattr__(self, "strategy_class", _enum(StrategyClass, "strategy.strategy_class", self.strategy_class).value)
        if self.semantic_class == SemanticClass.EXACT.value and self.strategy_class != StrategyClass.EXISTING_EXECUTION_STRATEGY.value:
            raise Q4ValidationError("exact strategy must be an existing execution strategy")
        if self.semantic_class == SemanticClass.RISK_PROBE.value and self.strategy_class != StrategyClass.RISK_PROBE.value:
            raise Q4ValidationError("risk strategy must be a risk probe")
        computed = canonical_sha256(self._semantic_dict())
        if computed not in _FROZEN_STRATEGY_IDS:
            raise Q4ValidationError("strategy descriptor is not one of frozen S01-S12 actions")
        if self.action_id and self.action_id != computed:
            raise Q4ValidationError("strategy.action_id does not match canonical content")
        object.__setattr__(self, "action_id", computed)

    def _semantic_dict(self) -> dict[str, Any]:
        return {"schema": self.SCHEMA, **{name: getattr(self, name) for name in _STRATEGY_FIELDS}}

    def to_dict(self) -> dict[str, Any]:
        return {**self._semantic_dict(), "action_id": self.action_id}

    @property
    def label(self) -> str:
        return STRATEGY_LABEL_BY_ID.get(self.action_id, "")

    @property
    def is_risk_probe(self) -> bool:
        return self.semantic_class == SemanticClass.RISK_PROBE.value

    @classmethod
    def from_label(cls, label: str) -> "ScheduleAction":
        if label not in {f"S{i:02d}" for i in range(1, 13)}:
            raise Q4ValidationError("strategy label must be S01..S12")
        return cls(**_strategy_values(label))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScheduleAction":
        _strict(data, {"schema", "action_id", *_STRATEGY_FIELDS}, cls.__name__)
        if data["schema"] != cls.SCHEMA:
            raise Q4ValidationError("unsupported q4 strategy schema")
        return cls(**{name: data[name] for name in _STRATEGY_FIELDS}, action_id=data["action_id"])

    @classmethod
    def safe_pool(cls, plan_kind: str | PlanKind) -> tuple["ScheduleAction", ...]:
        plan = _enum(PlanKind, "plan_kind", plan_kind).value
        labels = ("S01", "S02", "S03", "S04", "S05") if plan == PlanKind.STRICT.value else ("S06", "S07", "S08", "S09", "S10")
        return tuple(cls.from_label(label) for label in labels)

    @classmethod
    def risk_pool(cls, plan_kind: str | PlanKind | None = None) -> tuple["ScheduleAction", ...]:
        if plan_kind is None:
            return (cls.from_label("S11"), cls.from_label("S12"))
        plan = _enum(PlanKind, "plan_kind", plan_kind).value
        return (cls.from_label("S11" if plan == PlanKind.STRICT.value else "S12"),)


SCHEDULE_ACTIONS = tuple(ScheduleAction.from_label(f"S{i:02d}") for i in range(1, 13))
SCHEDULE_ACTION_BY_ID = {item.action_id: item for item in SCHEDULE_ACTIONS}
STRATEGY_LABEL_BY_ID = {item.action_id: f"S{i:02d}" for i, item in enumerate(SCHEDULE_ACTIONS, 1)}


@dataclass(frozen=True, slots=True)
class KnobDelta:
    SCHEMA: ClassVar[str] = "ironmule.q4_knob_delta.v1"

    stage: str
    source_action_id: str
    target_action_id: str
    changed_field: str
    target_value: bool | int | float
    action_id: str = ""

    def __post_init__(self) -> None:
        if _enum(Stage, "delta.stage", self.stage) is not Stage.KNOB_DELTA:
            raise Q4ValidationError("knob delta stage must be KNOB_DELTA")
        source = ALL_DECLARED_KNOB_ACTIONS.get(_digest("delta.source_action_id", self.source_action_id) or "")
        target = ALL_DECLARED_KNOB_ACTIONS.get(_digest("delta.target_action_id", self.target_action_id) or "")
        if source is None or target is None:
            raise Q4ValidationError("knob delta source/target must be frozen complete actions")
        if self.changed_field not in KNOB_NAMES:
            raise Q4ValidationError("knob delta changed_field is not a knob")
        if sum(getattr(source, name) != getattr(target, name) for name in KNOB_NAMES) != 1:
            raise Q4ValidationError("knob delta must change exactly one field")
        if getattr(target, self.changed_field) != self.target_value:
            raise Q4ValidationError("knob delta target_value does not match target action")
        object.__setattr__(self, "source_action_id", source.action_id)
        object.__setattr__(self, "target_action_id", target.action_id)
        computed = canonical_sha256(self._semantic_dict())
        if self.action_id and self.action_id != computed:
            raise Q4ValidationError("delta.action_id does not match canonical content")
        object.__setattr__(self, "action_id", computed)

    def _semantic_dict(self) -> dict[str, Any]:
        return {"schema": self.SCHEMA, "stage": self.stage, "source_action_id": self.source_action_id,
                "target_action_id": self.target_action_id, "changed_field": self.changed_field,
                "target_value": self.target_value}

    def to_dict(self) -> dict[str, Any]:
        return {**self._semantic_dict(), "action_id": self.action_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "KnobDelta":
        _strict(data, {"schema", "stage", "source_action_id", "target_action_id", "changed_field", "target_value", "action_id"}, cls.__name__)
        if data["schema"] != cls.SCHEMA:
            raise Q4ValidationError("unsupported q4 knob delta schema")
        return cls(**{name: data[name] for name in ("stage", "source_action_id", "target_action_id", "changed_field", "target_value", "action_id")})


def _legal_knob_delta_ids() -> frozenset[str]:
    ids: set[str] = set()
    for source in ALL_DECLARED_KNOB_ACTIONS.values():
        for target in ALL_DECLARED_KNOB_ACTIONS.values():
            changed = [name for name in KNOB_NAMES if getattr(source, name) != getattr(target, name)]
            if len(changed) != 1:
                continue
            name = changed[0]
            ids.add(canonical_sha256({
                "schema": KnobDelta.SCHEMA,
                "stage": Stage.KNOB_DELTA.value,
                "source_action_id": source.action_id,
                "target_action_id": target.action_id,
                "changed_field": name,
                "target_value": getattr(target, name),
            }))
    return frozenset(ids)


KNOB_DELTA_IDS = _legal_knob_delta_ids()


def _knob_delta_semantics() -> dict[str, tuple[str, str, str, bool | int | float]]:
    """Map every legal delta ID back to source/target/field/value."""
    result: dict[str, tuple[str, str, str, bool | int | float]] = {}
    for source in ALL_DECLARED_KNOB_ACTIONS.values():
        for target in ALL_DECLARED_KNOB_ACTIONS.values():
            changed = [name for name in KNOB_NAMES if getattr(source, name) != getattr(target, name)]
            if len(changed) != 1:
                continue
            field = changed[0]
            semantic = {
                "schema": KnobDelta.SCHEMA,
                "stage": Stage.KNOB_DELTA.value,
                "source_action_id": source.action_id,
                "target_action_id": target.action_id,
                "changed_field": field,
                "target_value": getattr(target, field),
            }
            result[canonical_sha256(semantic)] = (source.action_id, target.action_id, field, getattr(target, field))
    return result


KNOB_DELTA_BY_ID = _knob_delta_semantics()


@dataclass(frozen=True, slots=True)
class HybridAction:
    SCHEMA: ClassVar[str] = "ironmule.q4_hybrid_action.v1"

    knob_action_id: str
    strategy_action_id: str
    stage_order: tuple[str, ...] = (Stage.KNOB_DELTA.value, Stage.STRATEGY_SELECT.value, Stage.REVALIDATE.value)
    action_id: str = ""

    def __post_init__(self) -> None:
        _digest("hybrid.knob_action_id", self.knob_action_id)
        strategy_id = _digest("hybrid.strategy_action_id", self.strategy_action_id)
        if self.knob_action_id not in ALL_DECLARED_KNOB_ACTIONS:
            raise Q4ValidationError("hybrid knob action is outside the frozen declared state space")
        strategy = SCHEDULE_ACTION_BY_ID.get(strategy_id or "")
        if strategy is None or strategy.is_risk_probe:
            raise Q4ValidationError("hybrid strategy must be a safe strategy action")
        order = tuple(self.stage_order)
        if order != (Stage.KNOB_DELTA.value, Stage.STRATEGY_SELECT.value, Stage.REVALIDATE.value):
            raise Q4ValidationError("hybrid stage_order is frozen")
        object.__setattr__(self, "knob_action_id", self.knob_action_id)
        object.__setattr__(self, "strategy_action_id", strategy.action_id)
        object.__setattr__(self, "stage_order", order)
        computed = canonical_sha256(self._semantic_dict())
        if self.action_id and self.action_id != computed:
            raise Q4ValidationError("hybrid.action_id does not match canonical content")
        object.__setattr__(self, "action_id", computed)

    def _semantic_dict(self) -> dict[str, Any]:
        return {"schema": self.SCHEMA, "knob_action_id": self.knob_action_id,
                "strategy_action_id": self.strategy_action_id, "stage_order": self.stage_order}

    def to_dict(self) -> dict[str, Any]:
        return {**self._semantic_dict(), "action_id": self.action_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HybridAction":
        _strict(data, {"schema", "knob_action_id", "strategy_action_id", "stage_order", "action_id"}, cls.__name__)
        if data["schema"] != cls.SCHEMA or not isinstance(data["stage_order"], list):
            raise Q4ValidationError("unsupported q4 hybrid schema or stage_order")
        return cls(knob_action_id=data["knob_action_id"], strategy_action_id=data["strategy_action_id"], stage_order=tuple(data["stage_order"]), action_id=data["action_id"])


# ---------------------------------------------------------------------------
# Context, state, outcomes, transitions and trajectories

_CONTEXT_DIGEST_FIELDS = ("study_digest", "model_digest", "model_manifest_digest", "workload_digest", "hardware_digest", "runtime_digest", "time_digest")


@dataclass(frozen=True, slots=True)
class Q4Context:
    SCHEMA: ClassVar[str] = "ironmule.q4_context.v1"

    study_digest: str
    model_digest: str
    model_manifest_digest: str
    workload_digest: str
    hardware_digest: str
    runtime_digest: str
    time_digest: str
    objective_class: str
    workload_stratum: str
    arrival_pattern: str
    context_id: str = ""

    def __post_init__(self) -> None:
        for name in _CONTEXT_DIGEST_FIELDS:
            object.__setattr__(self, name, _digest(f"context.{name}", getattr(self, name)))
        object.__setattr__(self, "objective_class", _enum(ObjectiveClass, "context.objective_class", self.objective_class).value)
        object.__setattr__(self, "workload_stratum", _enum(WorkloadStratum, "context.workload_stratum", self.workload_stratum).value)
        object.__setattr__(self, "arrival_pattern", _enum(WorkloadStratum, "context.arrival_pattern", self.arrival_pattern).value)
        computed = canonical_sha256(self._semantic_dict())
        if self.context_id and self.context_id != computed:
            raise Q4ValidationError("context_id does not match canonical content")
        object.__setattr__(self, "context_id", computed)

    def _semantic_dict(self) -> dict[str, Any]:
        return {"schema": self.SCHEMA, **{name: getattr(self, name) for name in (*_CONTEXT_DIGEST_FIELDS, "objective_class", "workload_stratum", "arrival_pattern")}}

    @property
    def group_key(self) -> tuple[str, ...]:
        return tuple(getattr(self, name) for name in _CONTEXT_DIGEST_FIELDS)

    @property
    def group_key_digest(self) -> str:
        return canonical_sha256({"schema": "ironmule.q4_group_key.v1", "group_key": self.group_key})

    def to_dict(self) -> dict[str, Any]:
        return {**self._semantic_dict(), "context_id": self.context_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Q4Context":
        _strict(data, {"schema", "context_id", *_CONTEXT_DIGEST_FIELDS, "objective_class", "workload_stratum", "arrival_pattern"}, cls.__name__)
        if data["schema"] != cls.SCHEMA:
            raise Q4ValidationError("unsupported q4 context schema")
        return cls(**{name: data[name] for name in (*_CONTEXT_DIGEST_FIELDS, "objective_class", "workload_stratum", "arrival_pattern", "context_id")})


_BUCKETS = ("small", "medium", "large")
_MODEL_SIZES = ("1B", "4B", "12B")
_CATEGORIES = tuple(item.value for item in WorkloadStratum)
FEATURE_VECTOR_ORDER = (
    "intercept", "model_size[1B]", "model_size[4B]", "model_size[12B]",
    "memory_bucket[small]", "memory_bucket[medium]", "memory_bucket[large]",
    "gpu_core_bucket[small]", "gpu_core_bucket[medium]", "gpu_core_bucket[large]",
    "prompt_bucket[small]", "prompt_bucket[medium]", "prompt_bucket[large]",
    "output_bucket[small]", "output_bucket[medium]", "output_bucket[large]",
    "concurrency_bucket[small]", "concurrency_bucket[medium]", "concurrency_bucket[large]",
    "objective[LATENCY]", "objective[THROUGHPUT]", "plan[StrictOneShotPlan]",
    "plan[ReusableSessionPlan]", *(f"workload_stratum[{item}]" for item in _CATEGORIES),
    *(f"arrival_pattern[{item}]" for item in _CATEGORIES),
    *(f"current_action[{item.action_id}]" for item in LEGAL_KNOB_ACTIONS), "remaining_budget",
)


@dataclass(frozen=True, slots=True)
class Q4State:
    SCHEMA: ClassVar[str] = "ironmule.q4_state.v1"

    context_id: str
    stage: str
    step_index: int
    model_size: str
    memory_bucket: str
    gpu_core_bucket: str
    prompt_bucket: str
    output_bucket: str
    concurrency_bucket: str
    objective_class: str
    plan_kind: str
    workload_stratum: str
    arrival_pattern: str
    knob_action_id: str
    strategy_candidate_index: int | None
    state_digest: str = ""
    evaluated_candidate_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "context_id", _digest("state.context_id", self.context_id))
        object.__setattr__(self, "stage", _enum(Stage, "state.stage", self.stage).value)
        object.__setattr__(self, "step_index", _integer("state.step_index", self.step_index, 0, HORIZON - 1))
        object.__setattr__(self, "model_size", _text("state.model_size", self.model_size))
        for name in ("memory_bucket", "gpu_core_bucket", "prompt_bucket", "output_bucket", "concurrency_bucket"):
            object.__setattr__(self, name, _text(f"state.{name}", getattr(self, name)))
        object.__setattr__(self, "objective_class", _enum(ObjectiveClass, "state.objective_class", self.objective_class).value)
        object.__setattr__(self, "plan_kind", _enum(PlanKind, "state.plan_kind", self.plan_kind).value)
        object.__setattr__(self, "workload_stratum", _enum(WorkloadStratum, "state.workload_stratum", self.workload_stratum).value)
        object.__setattr__(self, "arrival_pattern", _enum(WorkloadStratum, "state.arrival_pattern", self.arrival_pattern).value)
        object.__setattr__(self, "knob_action_id", _digest("state.knob_action_id", self.knob_action_id))
        if self.knob_action_id not in ALL_DECLARED_KNOB_ACTIONS:
            raise Q4ValidationError("state knob action is outside the frozen declared state space")
        if isinstance(self.evaluated_candidate_ids, (str, bytes)):
            raise Q4ValidationError("state evaluated_candidate_ids must be an array")
        evaluated = tuple(_digest("state.evaluated_candidate_id", item) for item in self.evaluated_candidate_ids)
        if len(evaluated) != len(set(evaluated)) or any(item not in KNOB_CANDIDATE_BY_ID for item in evaluated):
            raise Q4ValidationError("state evaluated_candidate_ids must be unique declared candidate IDs")
        if self.step_index == 0 and evaluated:
            raise Q4ValidationError("step 0 must have no evaluated candidate IDs")
        # Dataset-bound Q4 states must carry the full persisted history; an
        # empty value is tolerated here for legacy synthetic state fixtures
        # that never enter a Dataset chain.
        if evaluated and self.step_index <= 10 and len(evaluated) != self.step_index:
            raise Q4ValidationError("knob state evaluated_candidate_ids length must equal step index")
        if evaluated and self.step_index >= 11 and (len(evaluated) != len(KNOB_CANDIDATES) or set(evaluated) != set(KNOB_CANDIDATE_BY_ID)):
            raise Q4ValidationError("strategy/revalidation state must contain all eleven evaluated candidate IDs")
        object.__setattr__(self, "evaluated_candidate_ids", evaluated)
        if 11 <= self.step_index <= 15:
            if type(self.strategy_candidate_index) is not int or not 0 <= self.strategy_candidate_index <= 4:
                raise Q4ValidationError("strategy candidate index must be 0..4 at strategy steps")
        elif self.strategy_candidate_index is not None:
            raise Q4ValidationError("strategy candidate index must be null outside strategy steps")
        expected_stage = Stage.KNOB_DELTA if self.step_index <= 10 else Stage.STRATEGY_SELECT if self.step_index <= 15 else Stage.REVALIDATE
        if self.stage != expected_stage.value:
            raise Q4ValidationError("state stage does not match step index")
        computed = canonical_sha256(self._semantic_dict())
        if self.state_digest and self.state_digest != computed:
            raise Q4ValidationError("state_digest does not match canonical content")
        object.__setattr__(self, "state_digest", computed)

    @property
    def in_domain(self) -> bool:
        return (self.model_size in _MODEL_SIZES and all(bucket in _BUCKETS for bucket in (
            self.memory_bucket, self.gpu_core_bucket, self.prompt_bucket,
            self.output_bucket, self.concurrency_bucket)))

    def feature_vector(self) -> tuple[float, ...]:
        """Exact frozen one-hot order; OOD categories yield an all-zero block."""
        blocks: list[float] = [1.0]
        blocks.extend(float(self.model_size == value) for value in _MODEL_SIZES)
        for value in (self.memory_bucket, self.gpu_core_bucket, self.prompt_bucket, self.output_bucket, self.concurrency_bucket):
            blocks.extend(float(value == category) for category in _BUCKETS)
        blocks.extend(float(self.objective_class == value) for value in (ObjectiveClass.LATENCY.value, ObjectiveClass.THROUGHPUT.value))
        blocks.extend(float(self.plan_kind == value) for value in (PlanKind.STRICT.value, PlanKind.REUSABLE.value))
        blocks.extend(float(self.workload_stratum == value) for value in _CATEGORIES)
        blocks.extend(float(self.arrival_pattern == value) for value in _CATEGORIES)
        blocks.extend(float(self.knob_action_id == item.action_id) for item in LEGAL_KNOB_ACTIONS)
        blocks.append(max(0.0, min(1.0, (HORIZON - 1 - self.step_index) / (HORIZON - 1))))
        return tuple(blocks)

    def _semantic_dict(self) -> dict[str, Any]:
        return {"schema": self.SCHEMA, **{field.name: getattr(self, field.name) for field in fields(self) if field.name not in {"state_digest", "SCHEMA"}}}

    def to_dict(self) -> dict[str, Any]:
        return {**self._semantic_dict(), "state_digest": self.state_digest}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Q4State":
        expected = {"schema", "state_digest", *(field.name for field in fields(cls) if field.name not in {"SCHEMA", "state_digest"})}
        _strict(data, expected, cls.__name__)
        if data["schema"] != cls.SCHEMA:
            raise Q4ValidationError("unsupported q4 state schema")
        return cls(**{field.name: data[field.name] for field in fields(cls) if field.name not in {"SCHEMA"}})


def _optional_nonnegative(name: str, value: Any) -> float | None:
    return None if value is None else _number(name, value, 0)


@dataclass(frozen=True, slots=True)
class Outcome:
    """Evaluator-owned outcome; incomplete or failed evidence remains visible."""

    SCHEMA: ClassVar[str] = "ironmule.q4_outcome.v1"

    raw_sample_refs: tuple[Any, ...]
    raw_sample_count: int
    total_ns: float | None
    prefill_ns: float | None
    decode_ns: float | None
    objective_class: str
    request_sample_count: int
    p95_full_response_ms: float | None
    p95_full_response_sample_count: int
    physical_tokens_per_second: float | None
    p95_physical_tokens_per_second: float | None
    p95_physical_tokens_per_second_sample_count: int
    knob_action_id: str
    strategy_action_id: str | None
    plan_kind: str
    samples: tuple[tuple[str, tuple[float, ...]], ...]
    uncertainty: tuple[tuple[str, float], ...]
    logical_token_identity: bool | None
    physical_token_identity: bool | None
    visible_token_identity: bool | None
    token_count_identity: bool | None
    stop_reason_identity: bool | None
    state_identity: bool | None
    capacity_identity: bool | None
    deterministic: bool | None
    mlx_active_memory_bytes: int | None
    mlx_peak_memory_bytes: int | None
    rss_peak_bytes: int | None
    swap_before_bytes: int | None
    swap_after_bytes: int | None
    swap_delta_bytes: int | None
    timeout: bool
    crash: bool
    fallbacks: int
    worker_status: str
    worker_reaped: bool
    hard_gates_passed: bool
    rollback: RollbackStatus | str
    status: OutcomeStatus | str
    preregistration_sha256: str
    code_digest: str
    model_digest: str
    model_manifest_digest: str
    environment_digest: str
    workload_digest: str
    researcher_id: str
    reviewer_id: str
    evaluator_id: str
    outcome_id: str = ""
    # Legacy evaluator fixtures may omit these fields when no strategy reward
    # is requested.  Strict ``from_dict`` records still carry them, and any
    # reward join fails closed if context_id remains empty.
    full_response_ms_samples: tuple[float, ...] = ()
    physical_tokens_per_second_samples: tuple[float, ...] = ()
    context_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_sample_refs", _ref_tuple("outcome.raw_sample_refs", self.raw_sample_refs))
        object.__setattr__(self, "raw_sample_count", _integer("outcome.raw_sample_count", self.raw_sample_count))
        if self.raw_sample_count < sum(ref.quality.value == "RAW_SAMPLES" for ref in self.raw_sample_refs):
            raise Q4ValidationError("raw_sample_count is below referenced raw samples")
        for name in ("total_ns", "prefill_ns", "decode_ns"):
            object.__setattr__(self, name, _optional_nonnegative(f"outcome.{name}", getattr(self, name)))
        object.__setattr__(self, "objective_class", _enum(ObjectiveClass, "outcome.objective_class", self.objective_class).value)
        object.__setattr__(self, "request_sample_count", _integer("outcome.request_sample_count", self.request_sample_count))
        object.__setattr__(self, "p95_full_response_ms", _optional_nonnegative("outcome.p95_full_response_ms", self.p95_full_response_ms))
        object.__setattr__(self, "p95_full_response_sample_count", _integer("outcome.p95_full_response_sample_count", self.p95_full_response_sample_count))
        object.__setattr__(self, "physical_tokens_per_second", _optional_nonnegative("outcome.physical_tokens_per_second", self.physical_tokens_per_second))
        object.__setattr__(self, "p95_physical_tokens_per_second", _optional_nonnegative("outcome.p95_physical_tokens_per_second", self.p95_physical_tokens_per_second))
        object.__setattr__(self, "p95_physical_tokens_per_second_sample_count", _integer("outcome.p95_physical_tokens_per_second_sample_count", self.p95_physical_tokens_per_second_sample_count))
        for name in ("full_response_ms_samples", "physical_tokens_per_second_samples"):
            values = getattr(self, name)
            if isinstance(values, (str, bytes)):
                raise Q4ValidationError(f"outcome.{name} must be an array")
            object.__setattr__(self, name, tuple(_number(f"outcome.{name}[]", value, 0) for value in values))
        if self.context_id:
            object.__setattr__(self, "context_id", _digest("outcome.context_id", self.context_id))
        object.__setattr__(self, "knob_action_id", _digest("outcome.knob_action_id", self.knob_action_id))
        if self.strategy_action_id is not None:
            object.__setattr__(self, "strategy_action_id", _digest("outcome.strategy_action_id", self.strategy_action_id))
            if self.strategy_action_id not in SCHEDULE_ACTION_BY_ID:
                raise Q4ValidationError("outcome.strategy_action_id is outside frozen strategies")
        object.__setattr__(self, "plan_kind", _enum(PlanKind, "outcome.plan_kind", self.plan_kind).value)
        sample_pairs = []
        if isinstance(self.samples, Mapping):
            iterable = self.samples.items()
        else:
            iterable = self.samples
        seen = set()
        for name, values in iterable:
            name = _text("outcome.samples.key", name)
            if name in seen:
                raise Q4ValidationError("outcome.samples keys must be unique")
            seen.add(name)
            if isinstance(values, (str, bytes)):
                raise Q4ValidationError("outcome.samples values must be arrays")
            sample_pairs.append((name, tuple(_number(f"outcome.samples.{name}[]", value, 0) for value in values)))
        object.__setattr__(self, "samples", tuple(sorted(sample_pairs)))
        if isinstance(self.uncertainty, Mapping):
            uncertainty = self.uncertainty.items()
        else:
            uncertainty = self.uncertainty
        object.__setattr__(self, "uncertainty", tuple(sorted((_text("outcome.uncertainty.key", key), _number("outcome.uncertainty", value, 0)) for key, value in uncertainty)))
        for name in ("logical_token_identity", "physical_token_identity", "visible_token_identity", "token_count_identity", "stop_reason_identity", "state_identity", "capacity_identity", "deterministic"):
            value = getattr(self, name)
            if value is not None and type(value) is not bool:
                raise Q4ValidationError(f"outcome.{name} must be boolean or null")
        for name in ("mlx_active_memory_bytes", "mlx_peak_memory_bytes", "rss_peak_bytes", "swap_before_bytes", "swap_after_bytes"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _integer(f"outcome.{name}", value))
        if self.swap_delta_bytes is not None:
            object.__setattr__(self, "swap_delta_bytes", _signed_integer("outcome.swap_delta_bytes", self.swap_delta_bytes))
        if self.swap_before_bytes is not None and self.swap_after_bytes is not None and self.swap_delta_bytes != self.swap_after_bytes - self.swap_before_bytes:
            raise Q4ValidationError("outcome.swap_delta_bytes does not match before/after")
        for name in ("timeout", "crash", "worker_reaped", "hard_gates_passed"):
            if type(getattr(self, name)) is not bool:
                raise Q4ValidationError(f"outcome.{name} must be boolean")
        object.__setattr__(self, "fallbacks", _integer("outcome.fallbacks", self.fallbacks))
        object.__setattr__(self, "worker_status", _text("outcome.worker_status", self.worker_status))
        object.__setattr__(self, "rollback", _enum(RollbackStatus, "outcome.rollback", self.rollback).value)
        object.__setattr__(self, "status", _enum(OutcomeStatus, "outcome.status", self.status).value)
        for name in ("preregistration_sha256", "code_digest", "model_digest", "model_manifest_digest", "environment_digest", "workload_digest"):
            object.__setattr__(self, name, _digest(f"outcome.{name}", getattr(self, name)))
        for name in ("researcher_id", "reviewer_id", "evaluator_id"):
            object.__setattr__(self, name, _text(f"outcome.{name}", getattr(self, name)))
        if len({self.researcher_id, self.reviewer_id, self.evaluator_id}) != 3:
            raise Q4ValidationError("researcher, reviewer and evaluator IDs must be distinct")
        if self.status == OutcomeStatus.MEASURED.value:
            primary = "total_ns" if self.objective_class == ObjectiveClass.LATENCY.value else "physical_tokens_per_second"
            sample_map = dict(self.samples)
            if self.raw_sample_count <= 0 or any(name not in sample_map or len(sample_map[name]) != self.raw_sample_count for name in ("total_ns", "prefill_ns", "decode_ns")):
                raise Q4ValidationError("measured outcome requires non-empty matching timing sample arrays")
            for name in ("total_ns", "prefill_ns", "decode_ns"):
                values = sorted(sample_map[name])
                middle = len(values) // 2
                median = values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2.0
                aggregate = getattr(self, name)
                if aggregate is None or not math.isclose(aggregate, median, rel_tol=0.0, abs_tol=1e-9):
                    raise Q4ValidationError(f"measured outcome {name} does not match sample median")
            if self.objective_class == ObjectiveClass.THROUGHPUT.value:
                if not self.physical_tokens_per_second_samples or len(self.physical_tokens_per_second_samples) != self.raw_sample_count:
                    raise Q4ValidationError("throughput aggregate requires matching raw throughput samples")
                values = sorted(self.physical_tokens_per_second_samples)
                middle = len(values) // 2
                median = values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2.0
                if self.physical_tokens_per_second is None or not math.isclose(self.physical_tokens_per_second, median, rel_tol=0.0, abs_tol=1e-9):
                    raise Q4ValidationError("physical_tokens_per_second does not match raw sample median")
            if not any(key == primary or key.startswith(primary + "_") for key, _ in self.uncertainty):
                raise Q4ValidationError("measured outcome requires uncertainty for its primary metric")
            if self.strategy_action_id is not None:
                if self.p95_full_response_ms is None or self.p95_full_response_sample_count < 20 or self.request_sample_count < 20:
                    raise Q4ValidationError("measured strategy outcomes require p95 full-response data with >=20 request samples")
                if self.p95_full_response_sample_count != self.request_sample_count:
                    raise Q4ValidationError("p95 full-response count must equal request sample count")
                if len(self.full_response_ms_samples) != self.request_sample_count:
                    raise Q4ValidationError("full-response request samples must match request_sample_count")
                rank = max(1, math.ceil(0.95 * len(self.full_response_ms_samples)))
                if not math.isclose(self.p95_full_response_ms, sorted(self.full_response_ms_samples)[rank - 1], rel_tol=0.0, abs_tol=1e-9):
                    raise Q4ValidationError("p95_full_response_ms does not match nearest-rank request samples")
            if self.objective_class == ObjectiveClass.THROUGHPUT.value and (self.physical_tokens_per_second is None or self.physical_tokens_per_second <= 0):
                raise Q4ValidationError("measured strategy throughput requires positive physical throughput")
        computed = canonical_sha256(self._semantic_dict())
        if self.outcome_id and self.outcome_id != computed:
            raise Q4ValidationError("outcome_id does not match canonical content")
        object.__setattr__(self, "outcome_id", computed)

    @property
    def raw_artifact_refs(self) -> tuple[Any, ...]:
        return self.raw_sample_refs

    @property
    def measurements_complete(self) -> bool:
        return all(getattr(self, name) is not None for name in ("total_ns", "prefill_ns", "decode_ns"))

    @property
    def correctness_complete(self) -> bool:
        return all(getattr(self, name) is True for name in ("logical_token_identity", "physical_token_identity", "visible_token_identity", "token_count_identity", "stop_reason_identity", "state_identity", "capacity_identity", "deterministic"))

    @property
    def resources_complete(self) -> bool:
        return all(getattr(self, name) is not None for name in ("mlx_active_memory_bytes", "mlx_peak_memory_bytes", "rss_peak_bytes", "swap_before_bytes", "swap_after_bytes", "swap_delta_bytes"))

    @property
    def complete_safe(self) -> bool:
        sample_map = dict(self.samples)
        matching_samples = self.raw_sample_count > 0 and all(name in sample_map and len(sample_map[name]) == self.raw_sample_count for name in ("total_ns", "prefill_ns", "decode_ns"))
        primary = "total_ns" if self.objective_class == ObjectiveClass.LATENCY.value else "physical_tokens_per_second"
        uncertain = any(key == primary or key.startswith(primary + "_") for key, _ in self.uncertainty)
        strategy_metrics = self.strategy_action_id is None or (self.request_sample_count >= 20 and self.p95_full_response_ms is not None and self.p95_full_response_sample_count >= 20)
        raw_only = bool(self.raw_sample_refs) and all(ref.quality.value == "RAW_SAMPLES" for ref in self.raw_sample_refs)
        objective_metric = ((self.objective_class == ObjectiveClass.LATENCY.value and self.total_ns is not None and self.total_ns > 0)
                            or (self.objective_class == ObjectiveClass.THROUGHPUT.value and self.physical_tokens_per_second is not None and self.physical_tokens_per_second > 0 and len(self.physical_tokens_per_second_samples) == self.raw_sample_count))
        return (self.status == OutcomeStatus.MEASURED.value and self.hard_gates_passed and self.measurements_complete
                and matching_samples and uncertain and raw_only
                and strategy_metrics and objective_metric
                and self.correctness_complete and self.resources_complete and self.raw_sample_count > 0
                and not self.timeout and not self.crash and self.fallbacks == 0
                and self.worker_reaped and self.rollback in {RollbackStatus.NOT_REQUIRED.value, RollbackStatus.APPLIED.value}
                and bool(self.uncertainty))

    def metric_value(self, transition_stage: str) -> float:
        """Return the frozen objective cost, or fail closed if unavailable."""
        if transition_stage in {Stage.STRATEGY_SELECT.value, Stage.REVALIDATE.value}:
            if self.objective_class == ObjectiveClass.LATENCY.value:
                if self.p95_full_response_ms is None or self.p95_full_response_sample_count < 20 or self.request_sample_count < 20:
                    raise Q4ValidationError("latency strategy reward requires p95 full-response data with >=20 samples")
                return self.p95_full_response_ms
            if self.physical_tokens_per_second is None or self.physical_tokens_per_second <= 0 or self.p95_full_response_ms is None or self.p95_full_response_sample_count < 20 or self.request_sample_count < 20:
                raise Q4ValidationError("throughput strategy reward requires physical throughput and p95 guard data")
            return 1.0 / self.physical_tokens_per_second
        # Knob steps always use wall total_ns, independent of objective class.
        if self.total_ns is not None and self.total_ns > 0:
            return self.total_ns
        raise Q4ValidationError("knob reward requires positive total_ns")

    def _semantic_dict(self) -> dict[str, Any]:
        return {"schema": self.SCHEMA, **{field.name: getattr(self, field.name) for field in fields(self) if field.name not in {"SCHEMA", "outcome_id"}}}

    def to_dict(self) -> dict[str, Any]:
        value = self._semantic_dict()
        value["raw_sample_refs"] = [ref.to_dict() for ref in self.raw_sample_refs]
        value["samples"] = {name: list(samples) for name, samples in self.samples}
        value["uncertainty"] = dict(self.uncertainty)
        value["outcome_id"] = self.outcome_id
        return value

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Outcome":
        expected = {"schema", "outcome_id", *(field.name for field in fields(cls) if field.name not in {"SCHEMA", "outcome_id"})}
        _strict(data, expected, cls.__name__)
        if data["schema"] != cls.SCHEMA or not isinstance(data["raw_sample_refs"], list):
            raise Q4ValidationError("unsupported q4 outcome schema or raw references")
        values = {field.name: data[field.name] for field in fields(cls) if field.name not in {"SCHEMA", "outcome_id"}}
        values["raw_sample_refs"] = tuple(ArtifactRef.from_dict(item) for item in data["raw_sample_refs"])
        values["outcome_id"] = data["outcome_id"]
        return cls(**values)


@dataclass(frozen=True, slots=True)
class PartialAbort:
    SCHEMA: ClassVar[str] = "ironmule.q4_partial_abort.v1"

    status: str
    failure_state: str
    terminal: bool
    terminal_step_index: int
    failure_reason: str
    fallback: str
    cleanup_verified: bool
    raw_artifact_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.status != "PARTIAL_ABORT" or type(self.terminal) is not bool or not self.terminal:
            raise Q4ValidationError("partial abort status/terminal marker is frozen")
        object.__setattr__(self, "failure_state", _enum(FailureState, "partial_abort.failure_state", self.failure_state).value)
        object.__setattr__(self, "terminal_step_index", _integer("partial_abort.terminal_step_index", self.terminal_step_index, 0, 16))
        object.__setattr__(self, "failure_reason", _text("partial_abort.failure_reason", self.failure_reason))
        if self.fallback != "BASE":
            raise Q4ValidationError("partial abort fallback must be BASE")
        if type(self.cleanup_verified) is not bool:
            raise Q4ValidationError("partial_abort.cleanup_verified must be boolean")
        object.__setattr__(self, "raw_artifact_ids", _string_tuple("partial_abort.raw_artifact_ids", self.raw_artifact_ids, allow_empty=True))

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.SCHEMA, "status": self.status, "failure_state": self.failure_state,
                "terminal": self.terminal, "terminal_step_index": self.terminal_step_index,
                "failure_reason": self.failure_reason, "fallback": self.fallback,
                "cleanup_verified": self.cleanup_verified, "raw_artifact_ids": list(self.raw_artifact_ids)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PartialAbort":
        _strict(data, {"schema", "status", "failure_state", "terminal", "terminal_step_index", "failure_reason", "fallback", "cleanup_verified", "raw_artifact_ids"}, cls.__name__)
        if data["schema"] != cls.SCHEMA or not isinstance(data["raw_artifact_ids"], list):
            raise Q4ValidationError("unsupported q4 partial-abort schema")
        return cls(status=data["status"], failure_state=data["failure_state"], terminal=data["terminal"], terminal_step_index=data["terminal_step_index"], failure_reason=data["failure_reason"], fallback=data["fallback"], cleanup_verified=data["cleanup_verified"], raw_artifact_ids=tuple(data["raw_artifact_ids"]))


@dataclass(frozen=True, slots=True)
class Transition:
    SCHEMA: ClassVar[str] = "ironmule.q4_transition.v1"

    trajectory_id: str
    context: Q4Context
    stage: str
    step_index: int
    horizon: int
    state_digest: str
    action_space: str
    action_id: str
    previous_action_id: str
    outcome_id: str
    next_state_digest: str
    terminal: bool
    split: str | Q4Split
    evidence_ids: tuple[str, ...]
    behaviour_propensity: float
    behaviour_policy_digest: str
    seed: str
    decision_budget_index: int
    strategy_candidate_index: int | None
    partial_abort: PartialAbort | None = None
    transition_id: str = ""
    # This is a policy-slot identity, not the dynamic delta identity in
    # ``action_id``.  It is optional only for constructor compatibility; the
    # validator requires it at every Stage-1 step and requires null later.
    candidate_id: str | None = None
    reference_outcome_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "trajectory_id", _digest("transition.trajectory_id", self.trajectory_id))
        if not isinstance(self.context, Q4Context):
            raise Q4ValidationError("transition.context must be Q4Context")
        object.__setattr__(self, "stage", _enum(Stage, "transition.stage", self.stage).value)
        object.__setattr__(self, "step_index", _integer("transition.step_index", self.step_index, 0, 16))
        if self.horizon != HORIZON:
            raise Q4ValidationError("Q4 transition horizon must be 17")
        for name in ("state_digest", "previous_action_id", "outcome_id", "next_state_digest", "behaviour_policy_digest"):
            object.__setattr__(self, name, _digest(f"transition.{name}", getattr(self, name)))
        if self.reference_outcome_id is None:
            raise Q4ValidationError("transition.reference_outcome_id is required for deterministic reward joins")
        object.__setattr__(self, "reference_outcome_id", _digest("transition.reference_outcome_id", self.reference_outcome_id))
        object.__setattr__(self, "action_space", _enum(ActionSpace, "transition.action_space", self.action_space).value)
        object.__setattr__(self, "split", _enum(Q4Split, "transition.split", self.split).value)
        object.__setattr__(self, "action_id", _digest("transition.action_id", self.action_id))
        if self.candidate_id is not None:
            object.__setattr__(self, "candidate_id", _digest("transition.candidate_id", self.candidate_id))
        object.__setattr__(self, "evidence_ids", _string_tuple("transition.evidence_ids", self.evidence_ids))
        object.__setattr__(self, "behaviour_propensity", _number("transition.behaviour_propensity", self.behaviour_propensity, 0, 1))
        if self.behaviour_propensity <= 0:
            raise Q4ValidationError("behaviour_propensity must be > 0")
        object.__setattr__(self, "seed", _text("transition.seed", self.seed))
        object.__setattr__(self, "decision_budget_index", _integer("transition.decision_budget_index", self.decision_budget_index, 0, 16))
        if self.decision_budget_index != self.step_index:
            raise Q4ValidationError("decision_budget_index must equal step_index")
        if type(self.terminal) is not bool:
            raise Q4ValidationError("transition.terminal must be boolean")
        expected_stage = Stage.KNOB_DELTA if self.step_index <= 10 else Stage.STRATEGY_SELECT if self.step_index <= 15 else Stage.REVALIDATE
        if self.stage != expected_stage.value or self.action_space != expected_stage.value:
            raise Q4ValidationError("transition stage/action_space does not match step")
        if self.step_index <= 10:
            if self.candidate_id not in KNOB_CANDIDATE_BY_ID:
                raise Q4ValidationError("knob transition candidate_id must be a declared policy slot")
            delta = KNOB_DELTA_BY_ID.get(self.action_id)
            candidate = KNOB_CANDIDATE_BY_ID[self.candidate_id]
            if delta is None or delta[2] != candidate.changed_field or delta[3] != candidate.target_value:
                raise Q4ValidationError("candidate_id does not match the KnobDelta field/value")
            if self.step_index == 0 and self.previous_action_id != KnobAction.baseline().action_id:
                raise Q4ValidationError("step 0 previous action must be BASE")
            if self.action_id not in KNOB_DELTA_IDS:
                # A supplied action may be any legal source/target pair, but
                # must be a delta ID rather than an absolute knob ID.
                raise Q4ValidationError("knob transition action must be a legal KnobDelta")
            if self.strategy_candidate_index is not None:
                raise Q4ValidationError("strategy_candidate_index must be null for knob steps")
        elif self.step_index <= 15:
            if self.candidate_id is not None:
                raise Q4ValidationError("candidate_id must be null for strategy steps")
            if self.previous_action_id not in ALL_DECLARED_KNOB_ACTIONS:
                raise Q4ValidationError("strategy transition previous action must be a complete knob state")
            action = SCHEDULE_ACTION_BY_ID.get(self.action_id)
            if action is None or action.is_risk_probe or action not in ScheduleAction.safe_pool(PlanKind.STRICT if self.context.workload_stratum in {WorkloadStratum.HOMOGENEOUS.value, WorkloadStratum.HETEROGENEOUS.value} else PlanKind.REUSABLE):
                raise Q4ValidationError("strategy transition action is outside the matching safe pool")
            if type(self.strategy_candidate_index) is not int or not 0 <= self.strategy_candidate_index <= 4:
                raise Q4ValidationError("strategy candidate index must be 0..4")
        else:
            if self.candidate_id is not None:
                raise Q4ValidationError("candidate_id must be null for revalidation")
            if self.previous_action_id not in SCHEDULE_ACTION_BY_ID or SCHEDULE_ACTION_BY_ID[self.previous_action_id].is_risk_probe:
                raise Q4ValidationError("revalidation previous action must be a safe strategy marker")
            if self.action_id not in SCHEDULE_ACTION_BY_ID or SCHEDULE_ACTION_BY_ID[self.action_id].is_risk_probe:
                raise Q4ValidationError("revalidation action must be a safe strategy marker")
            if self.strategy_candidate_index is not None:
                raise Q4ValidationError("strategy_candidate_index must be null for revalidation")
        if self.terminal and self.step_index != 16 and self.partial_abort is None:
            raise Q4ValidationError("normal terminal transition is allowed only at step 16")
        if self.partial_abort is not None:
            if not isinstance(self.partial_abort, PartialAbort) or self.partial_abort.terminal_step_index != self.step_index:
                raise Q4ValidationError("partial abort must terminate at this transition step")
        if self.step_index == 16 and not self.terminal and self.partial_abort is None:
            raise Q4ValidationError("step 16 must be terminal or an explicit partial abort")
        computed = canonical_sha256(self._semantic_dict())
        if self.transition_id and self.transition_id != computed:
            raise Q4ValidationError("transition_id does not match canonical content")
        object.__setattr__(self, "transition_id", computed)

    def _semantic_dict(self) -> dict[str, Any]:
        # The owning trajectory ID is intentionally excluded from the
        # transition digest: Trajectory hashes its ordered transition IDs.
        # Including both would create an unresolvable circular ID dependency.
        return {"schema": self.SCHEMA, **{field.name: getattr(self, field.name) for field in fields(self) if field.name not in {"SCHEMA", "transition_id", "trajectory_id"}}}

    def to_dict(self) -> dict[str, Any]:
        value = self._semantic_dict()
        value["trajectory_id"] = self.trajectory_id
        value["context"] = self.context.to_dict()
        value["evidence_ids"] = list(self.evidence_ids)
        value["partial_abort"] = None if self.partial_abort is None else self.partial_abort.to_dict()
        value["transition_id"] = self.transition_id
        return value

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Transition":
        expected = {"schema", "transition_id", *(field.name for field in fields(cls) if field.name not in {"SCHEMA", "transition_id"})}
        _strict(data, expected, cls.__name__)
        if data["schema"] != cls.SCHEMA or not isinstance(data["evidence_ids"], list):
            raise Q4ValidationError("unsupported q4 transition schema")
        values = {field.name: data[field.name] for field in fields(cls) if field.name not in {"SCHEMA", "transition_id"}}
        values["context"] = Q4Context.from_dict(data["context"])
        values["evidence_ids"] = tuple(data["evidence_ids"])
        values["partial_abort"] = None if data["partial_abort"] is None else PartialAbort.from_dict(data["partial_abort"])
        values["transition_id"] = data["transition_id"]
        return cls(**values)


@dataclass(frozen=True, slots=True)
class Trajectory:
    SCHEMA: ClassVar[str] = "ironmule.q4_trajectory.v1"

    context_id: str
    split: str | Q4Split
    horizon: int
    trajectory_status: str | TrajectoryStatus
    transition_ids: tuple[str, ...]
    terminal_step_index: int | None
    partial_abort: PartialAbort | None
    trajectory_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "context_id", _digest("trajectory.context_id", self.context_id))
        object.__setattr__(self, "split", _enum(Q4Split, "trajectory.split", self.split).value)
        if self.horizon != HORIZON:
            raise Q4ValidationError("trajectory horizon must be 17")
        object.__setattr__(self, "trajectory_status", _enum(TrajectoryStatus, "trajectory.trajectory_status", self.trajectory_status).value)
        object.__setattr__(self, "transition_ids", tuple(_digest("trajectory.transition_id", item) for item in self.transition_ids))
        if self.trajectory_status == TrajectoryStatus.COMPLETE.value:
            if len(self.transition_ids) != HORIZON or self.terminal_step_index != 16 or self.partial_abort is not None:
                raise Q4ValidationError("complete trajectory requires exactly 17 transitions and terminal step 16")
        elif self.trajectory_status == TrajectoryStatus.ABORTED.value:
            if self.partial_abort is None or self.terminal_step_index != self.partial_abort.terminal_step_index:
                raise Q4ValidationError("aborted trajectory requires matching partial abort marker")
        elif self.partial_abort is not None or len(self.transition_ids) > HORIZON:
            raise Q4ValidationError("running trajectory cannot carry an abort or exceed horizon")
        computed = canonical_sha256(self._semantic_dict())
        if self.trajectory_id and self.trajectory_id != computed:
            raise Q4ValidationError("trajectory_id does not match canonical content")
        object.__setattr__(self, "trajectory_id", computed)

    def _semantic_dict(self) -> dict[str, Any]:
        return {"schema": self.SCHEMA, **{field.name: getattr(self, field.name) for field in fields(self) if field.name not in {"SCHEMA", "trajectory_id"}}}

    def to_dict(self) -> dict[str, Any]:
        value = self._semantic_dict()
        value["transition_ids"] = list(self.transition_ids)
        value["partial_abort"] = None if self.partial_abort is None else self.partial_abort.to_dict()
        value["trajectory_id"] = self.trajectory_id
        return value

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Trajectory":
        _strict(data, {"schema", "trajectory_id", *(field.name for field in fields(cls) if field.name not in {"SCHEMA", "trajectory_id"})}, cls.__name__)
        if data["schema"] != cls.SCHEMA or not isinstance(data["transition_ids"], list):
            raise Q4ValidationError("unsupported q4 trajectory schema")
        return cls(context_id=data["context_id"], split=data["split"], horizon=data["horizon"], trajectory_status=data["trajectory_status"], transition_ids=tuple(data["transition_ids"]), terminal_step_index=data["terminal_step_index"], partial_abort=None if data["partial_abort"] is None else PartialAbort.from_dict(data["partial_abort"]), trajectory_id=data["trajectory_id"])


@dataclass(frozen=True, slots=True)
class RiskObservation:
    SCHEMA: ClassVar[str] = "ironmule.q4_risk_observation.v1"

    state_digest: str
    action: ScheduleAction
    risk_probe_id: str
    failure_code: str | None
    evaluator_gates: tuple[tuple[str, bool], ...]
    evidence_ids: tuple[str, ...]
    recorded_at: str
    risk_observation_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_digest", _digest("risk.state_digest", self.state_digest))
        if not isinstance(self.action, ScheduleAction) or not self.action.is_risk_probe:
            raise Q4ValidationError("risk observation action must be S11 or S12")
        if self.risk_probe_id != self.action.label or self.risk_probe_id not in {"S11", "S12"}:
            raise Q4ValidationError("risk_probe_id must match S11/S12 action")
        if self.failure_code is not None:
            object.__setattr__(self, "failure_code", _text("risk.failure_code", self.failure_code))
        gates = _bool_map("risk.evaluator_gates", dict(self.evaluator_gates))
        object.__setattr__(self, "evaluator_gates", gates)
        if self.failure_code is None and not all(value for _, value in gates):
            raise Q4ValidationError("failed risk gates require a failure_code")
        object.__setattr__(self, "evidence_ids", _string_tuple("risk.evidence_ids", self.evidence_ids))
        object.__setattr__(self, "recorded_at", _utc("risk.recorded_at", self.recorded_at))
        computed = canonical_sha256(self._semantic_dict())
        if self.risk_observation_id and self.risk_observation_id != computed:
            raise Q4ValidationError("risk_observation_id does not match canonical content")
        object.__setattr__(self, "risk_observation_id", computed)

    def _semantic_dict(self) -> dict[str, Any]:
        return {"schema": self.SCHEMA, "state_digest": self.state_digest, "action": self.action,
                "risk_probe_id": self.risk_probe_id, "failure_code": self.failure_code,
                "evaluator_gates": dict(self.evaluator_gates), "evidence_ids": self.evidence_ids,
                "recorded_at": self.recorded_at}

    def to_dict(self) -> dict[str, Any]:
        return {**self._semantic_dict(), "action": self.action.to_dict(), "evidence_ids": list(self.evidence_ids), "risk_observation_id": self.risk_observation_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RiskObservation":
        _strict(data, {"schema", "risk_observation_id", "state_digest", "action", "risk_probe_id", "failure_code", "evaluator_gates", "evidence_ids", "recorded_at"}, cls.__name__)
        if data["schema"] != cls.SCHEMA or not isinstance(data["evidence_ids"], list):
            raise Q4ValidationError("unsupported q4 risk schema")
        return cls(state_digest=data["state_digest"], action=ScheduleAction.from_dict(data["action"]), risk_probe_id=data["risk_probe_id"], failure_code=data["failure_code"], evaluator_gates=tuple(data["evaluator_gates"].items()) if isinstance(data["evaluator_gates"], dict) else data["evaluator_gates"], evidence_ids=tuple(data["evidence_ids"]), recorded_at=data["recorded_at"], risk_observation_id=data["risk_observation_id"])


# ---------------------------------------------------------------------------
# Provenance, split manifests, datasets, and foreign metadata

@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    SCHEMA: ClassVar[str] = "ironmule.q4_artifact_record.v1"

    artifact_id: str
    sha256: str
    quality: str
    source_name: str
    role: str
    status: str
    source_alias: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", _text("artifact.artifact_id", self.artifact_id))
        object.__setattr__(self, "sha256", _digest("artifact.sha256", self.sha256))
        try:
            quality = self.quality.value if isinstance(self.quality, Enum) else self.quality
            if quality not in {"RAW_SAMPLES", "SUMMARY_ONLY", "PARTIAL"}:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise Q4ValidationError("artifact quality is closed") from exc
        object.__setattr__(self, "quality", quality)
        for name in ("source_name", "role", "status"):
            object.__setattr__(self, name, _text(f"artifact.{name}", getattr(self, name)))
        if self.source_alias:
            object.__setattr__(self, "source_alias", _text("artifact.source_alias", self.source_alias))

    def as_ref(self) -> Any:
        return ArtifactRef(self.artifact_id, self.sha256, EvidenceQuality(self.quality))

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.SCHEMA, "artifact_id": self.artifact_id, "sha256": self.sha256, "quality": self.quality, "source_name": self.source_name, "role": self.role, "status": self.status, "source_alias": self.source_alias}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactRecord":
        _strict(data, {"schema", "artifact_id", "sha256", "quality", "source_name", "role", "status", "source_alias"}, cls.__name__)
        if data["schema"] != cls.SCHEMA:
            raise Q4ValidationError("unsupported q4 artifact record schema")
        return cls(**{name: data[name] for name in ("artifact_id", "sha256", "quality", "source_name", "role", "status", "source_alias")})


@dataclass(frozen=True, slots=True)
class PriorRecord:
    SCHEMA: ClassVar[str] = "ironmule.q4_prior_record.v1"

    prior_id: str
    source_artifact_ids: tuple[str, ...]
    source_name: str
    role: str
    usable_for_reward: bool
    allowed_uses: tuple[str, ...]
    forbidden_uses: tuple[str, ...]
    notes: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_artifact_ids", _string_tuple("prior.source_artifact_ids", self.source_artifact_ids))
        object.__setattr__(self, "source_name", _text("prior.source_name", self.source_name))
        object.__setattr__(self, "role", _text("prior.role", self.role))
        if type(self.usable_for_reward) is not bool:
            raise Q4ValidationError("prior.usable_for_reward must be boolean")
        object.__setattr__(self, "allowed_uses", _string_tuple("prior.allowed_uses", self.allowed_uses))
        object.__setattr__(self, "forbidden_uses", _string_tuple("prior.forbidden_uses", self.forbidden_uses))
        object.__setattr__(self, "notes", _text("prior.notes", self.notes))
        computed = canonical_sha256(self._semantic_dict())
        if self.prior_id and self.prior_id != computed:
            raise Q4ValidationError("prior_id does not match canonical content")
        object.__setattr__(self, "prior_id", computed)

    def _semantic_dict(self) -> dict[str, Any]:
        return {"schema": self.SCHEMA, "source_artifact_ids": self.source_artifact_ids, "source_name": self.source_name, "role": self.role, "usable_for_reward": self.usable_for_reward, "allowed_uses": self.allowed_uses, "forbidden_uses": self.forbidden_uses, "notes": self.notes}

    def to_dict(self) -> dict[str, Any]:
        return {**self._semantic_dict(), "prior_id": self.prior_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PriorRecord":
        _strict(data, {"schema", "prior_id", "source_artifact_ids", "source_name", "role", "usable_for_reward", "allowed_uses", "forbidden_uses", "notes"}, cls.__name__)
        if data["schema"] != cls.SCHEMA:
            raise Q4ValidationError("unsupported q4 prior schema")
        return cls(prior_id=data["prior_id"], source_artifact_ids=tuple(data["source_artifact_ids"]), source_name=data["source_name"], role=data["role"], usable_for_reward=data["usable_for_reward"], allowed_uses=tuple(data["allowed_uses"]), forbidden_uses=tuple(data["forbidden_uses"]), notes=data["notes"])


@dataclass(frozen=True, slots=True)
class SplitManifest:
    SCHEMA: ClassVar[str] = "ironmule.q4_split_manifest.v1"

    split_contexts: tuple[tuple[str, tuple[str, ...]], ...]
    model_sizes: tuple[tuple[str, tuple[str, ...]], ...]
    stratum_by_context: tuple[tuple[str, str], ...]
    seed: str
    manifest_id: str = ""
    model_size_by_context: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        split_map = {str(key): tuple(_digest("manifest.context_id", item) for item in values) for key, values in (self.split_contexts.items() if isinstance(self.split_contexts, Mapping) else self.split_contexts)}
        if set(split_map) != {item.value for item in Q4Split}:
            raise Q4ValidationError("split manifest must declare all three Q4 splits")
        all_contexts = [item for values in split_map.values() for item in values]
        if len(all_contexts) != len(set(all_contexts)):
            raise Q4ValidationError("context appears in more than one split")
        object.__setattr__(self, "split_contexts", tuple((key, split_map[key]) for key in sorted(split_map)))
        model_map = {str(key): tuple(_text("manifest.model_size", item) for item in values) for key, values in (self.model_sizes.items() if isinstance(self.model_sizes, Mapping) else self.model_sizes)}
        if set(model_map) != set(split_map):
            raise Q4ValidationError("split manifest model_sizes must cover all splits")
        for split, values in model_map.items():
            if values and set(values) != set(_MODEL_SIZES):
                raise Q4ValidationError(f"{split} must cover 1B, 4B and 12B")
        object.__setattr__(self, "model_sizes", tuple((key, model_map[key]) for key in sorted(model_map)))
        strata = self.stratum_by_context.items() if isinstance(self.stratum_by_context, Mapping) else self.stratum_by_context
        object.__setattr__(self, "stratum_by_context", tuple(sorted((_digest("manifest.context_id", key) or "", _enum(WorkloadStratum, "manifest.stratum", value).value) for key, value in strata)))
        model_contexts = self.model_size_by_context.items() if isinstance(self.model_size_by_context, Mapping) else self.model_size_by_context
        normalized_models = []
        for key, value in model_contexts:
            normalized_models.append((_digest("manifest.context_id", key) or "", _text("manifest.model_size", value)))
        if any(value not in _MODEL_SIZES for _, value in normalized_models):
            raise Q4ValidationError("manifest model_size_by_context contains an unsupported model")
        object.__setattr__(self, "model_size_by_context", tuple(sorted(normalized_models)))
        object.__setattr__(self, "seed", _text("manifest.seed", self.seed))
        computed = canonical_sha256(self._semantic_dict())
        if self.manifest_id and self.manifest_id != computed:
            raise Q4ValidationError("manifest_id does not match canonical content")
        object.__setattr__(self, "manifest_id", computed)

    def _semantic_dict(self) -> dict[str, Any]:
        return {"schema": self.SCHEMA, "split_contexts": dict(self.split_contexts), "model_sizes": dict(self.model_sizes), "stratum_by_context": dict(self.stratum_by_context), "seed": self.seed, "model_size_by_context": dict(self.model_size_by_context)}

    def to_dict(self) -> dict[str, Any]:
        return {**self._semantic_dict(), "manifest_id": self.manifest_id}

    @classmethod
    def empty(cls) -> "SplitManifest":
        return cls(split_contexts={split.value: () for split in Q4Split}, model_sizes={split.value: () for split in Q4Split}, stratum_by_context={}, seed="Q4-import", model_size_by_context={})

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SplitManifest":
        _strict(data, {"schema", "manifest_id", "split_contexts", "model_sizes", "stratum_by_context", "seed", "model_size_by_context"}, cls.__name__)
        if data["schema"] != cls.SCHEMA or not all(isinstance(data[name], dict) for name in ("split_contexts", "model_sizes", "stratum_by_context", "model_size_by_context")):
            raise Q4ValidationError("unsupported q4 split manifest schema")
        return cls(split_contexts=data["split_contexts"], model_sizes=data["model_sizes"], stratum_by_context=data["stratum_by_context"], seed=data["seed"], manifest_id=data["manifest_id"], model_size_by_context=data["model_size_by_context"])


def _default_action_pools() -> dict[str, tuple[Any, ...]]:
    return {"knob": KNOB_ACTIONS, "strict_safe": ScheduleAction.safe_pool(PlanKind.STRICT), "reusable_safe": ScheduleAction.safe_pool(PlanKind.REUSABLE), "risk": ScheduleAction.risk_pool()}


@dataclass(frozen=True, slots=True)
class RewardRecord:
    """Deterministic reward join; no missing/unsafe metric is imputed."""

    SCHEMA: ClassVar[str] = "ironmule.q4_reward.v1"

    transition_id: str
    candidate_outcome_id: str
    reference_outcome_id: str
    objective_class: str
    candidate_cost: float
    reference_cost: float
    reward: float
    reward_id: str = ""

    def __post_init__(self) -> None:
        for name in ("transition_id", "candidate_outcome_id", "reference_outcome_id"):
            object.__setattr__(self, name, _digest(f"reward.{name}", getattr(self, name)))
        object.__setattr__(self, "objective_class", _enum(ObjectiveClass, "reward.objective_class", self.objective_class).value)
        object.__setattr__(self, "candidate_cost", _number("reward.candidate_cost", self.candidate_cost, 0.0))
        object.__setattr__(self, "reference_cost", _number("reward.reference_cost", self.reference_cost, 0.0))
        if self.candidate_cost <= 0 or self.reference_cost <= 0:
            raise Q4ValidationError("reward costs must be strictly positive")
        object.__setattr__(self, "reward", _number("reward.reward", self.reward))
        computed = canonical_sha256(self._semantic_dict())
        if self.reward_id and self.reward_id != computed:
            raise Q4ValidationError("reward_id does not match canonical content")
        object.__setattr__(self, "reward_id", computed)

    def _semantic_dict(self) -> dict[str, Any]:
        return {"schema": self.SCHEMA, **{field.name: getattr(self, field.name) for field in fields(self) if field.name not in {"SCHEMA", "reward_id"}}}

    def to_dict(self) -> dict[str, Any]:
        return {**self._semantic_dict(), "reward_id": self.reward_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RewardRecord":
        _strict(data, {"schema", "reward_id", "transition_id", "candidate_outcome_id", "reference_outcome_id", "objective_class", "candidate_cost", "reference_cost", "reward"}, cls.__name__)
        if data["schema"] != cls.SCHEMA:
            raise Q4ValidationError("unsupported q4 reward schema")
        return cls(transition_id=data["transition_id"], candidate_outcome_id=data["candidate_outcome_id"], reference_outcome_id=data["reference_outcome_id"], objective_class=data["objective_class"], candidate_cost=data["candidate_cost"], reference_cost=data["reference_cost"], reward=data["reward"], reward_id=data["reward_id"])


@dataclass(frozen=True, slots=True)
class PanelCell:
    """One evaluator-owned absolute knob×strategy panel cell."""

    SCHEMA: ClassVar[str] = "ironmule.q4_panel_cell.v1"

    context_id: str
    knob_action_id: str
    strategy_action_id: str
    outcome_id: str
    reference_outcome_id: str
    cell_id: str = ""

    def __post_init__(self) -> None:
        for name in ("context_id", "knob_action_id", "strategy_action_id", "outcome_id", "reference_outcome_id"):
            object.__setattr__(self, name, _digest(f"panel.{name}", getattr(self, name)))
        if self.knob_action_id not in KNOB_ACTION_BY_ID:
            raise Q4ValidationError("panel knob action must be one of twelve absolute anchors")
        strategy = SCHEDULE_ACTION_BY_ID.get(self.strategy_action_id)
        if strategy is None or strategy.is_risk_probe:
            raise Q4ValidationError("panel strategy must be a safe strategy")
        if self.outcome_id == self.reference_outcome_id:
            raise Q4ValidationError("panel candidate/reference outcomes must differ")
        computed = canonical_sha256(self._semantic_dict())
        if self.cell_id and self.cell_id != computed:
            raise Q4ValidationError("panel.cell_id does not match canonical content")
        object.__setattr__(self, "cell_id", computed)

    def _semantic_dict(self) -> dict[str, Any]:
        return {"schema": self.SCHEMA, "context_id": self.context_id, "knob_action_id": self.knob_action_id, "strategy_action_id": self.strategy_action_id, "outcome_id": self.outcome_id, "reference_outcome_id": self.reference_outcome_id}

    def to_dict(self) -> dict[str, Any]:
        return {**self._semantic_dict(), "cell_id": self.cell_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PanelCell":
        _strict(data, {"schema", "context_id", "knob_action_id", "strategy_action_id", "outcome_id", "reference_outcome_id", "cell_id"}, cls.__name__)
        if data["schema"] != cls.SCHEMA:
            raise Q4ValidationError("unsupported q4 panel cell schema")
        return cls(context_id=data["context_id"], knob_action_id=data["knob_action_id"], strategy_action_id=data["strategy_action_id"], outcome_id=data["outcome_id"], reference_outcome_id=data["reference_outcome_id"], cell_id=data["cell_id"])


@dataclass(frozen=True, slots=True)
class RewardDerivationRecord:
    """Per-transition outcome, including explicit ineligibility."""

    SCHEMA: ClassVar[str] = "ironmule.q4_reward_derivation.v1"

    transition_id: str
    status: str
    reward_id: str | None
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "transition_id", _digest("derivation.transition_id", self.transition_id))
        if self.status not in {"DERIVED", "INELIGIBLE"}:
            raise Q4ValidationError("reward derivation status is closed")
        if self.reward_id is not None:
            object.__setattr__(self, "reward_id", _digest("derivation.reward_id", self.reward_id))
        object.__setattr__(self, "reason", _text("derivation.reason", self.reason))

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.SCHEMA, "transition_id": self.transition_id, "status": self.status, "reward_id": self.reward_id, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class RewardDerivationResult:
    rewards: Mapping[str, RewardRecord]
    records: Mapping[str, RewardDerivationRecord]

    @property
    def excluded(self) -> dict[str, str]:
        return {key: value.reason for key, value in self.records.items() if value.status == "INELIGIBLE"}

    def values(self):
        return self.rewards.values()

    def __getitem__(self, key: str) -> RewardRecord:
        return self.rewards[key]

    def __iter__(self):
        return iter(self.rewards)

    def __len__(self) -> int:
        return len(self.rewards)


@dataclass(frozen=True, slots=True)
class Dataset:
    SCHEMA: ClassVar[str] = "ironmule.q4_dataset.v1"

    preregistration_sha256: str
    source_artifacts: tuple[ArtifactRecord, ...]
    action_pools: Mapping[str, Sequence[Any]]
    contexts: tuple[Q4Context, ...]
    transitions: tuple[Transition, ...]
    outcomes: tuple[Outcome, ...]
    # States are stored explicitly rather than treated as a lossy digest-only
    # cache.  This keeps the frozen feature encoder replayable from a dataset
    # without consulting runtime state or reconstructing a state heuristically.
    states: tuple[Q4State, ...]
    trajectories: tuple[Trajectory, ...]
    split_manifest: SplitManifest
    seed_manifest: Mapping[str, str]
    no_invented_performance: bool
    dataset_id: str = ""
    risk_observations: tuple[RiskObservation, ...] = ()
    panel_cells: tuple[PanelCell, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "preregistration_sha256", _digest("dataset.preregistration_sha256", self.preregistration_sha256))
        artifacts = tuple(self.source_artifacts)
        if any(not isinstance(item, ArtifactRecord) for item in artifacts):
            raise Q4ValidationError("dataset source_artifacts must contain ArtifactRecord values")
        if len({item.artifact_id for item in artifacts}) != len(artifacts):
            raise Q4ValidationError("dataset source artifacts contain duplicate IDs")
        object.__setattr__(self, "source_artifacts", artifacts)
        pools = self.action_pools if isinstance(self.action_pools, Mapping) else dict(self.action_pools)
        expected_pools = {"knob", "strict_safe", "reusable_safe", "risk"}
        if set(pools) != expected_pools:
            raise Q4ValidationError("dataset action_pools must declare knob, strict_safe, reusable_safe and risk")
        for name, values in pools.items():
            values = tuple(values)
            if any(not isinstance(item, (KnobAction, ScheduleAction)) for item in values):
                raise Q4ValidationError(f"action pool {name} contains an unsupported action")
            if len({item.action_id for item in values}) != len(values):
                raise Q4ValidationError(f"action pool {name} contains duplicate IDs")
            if name == "knob" and tuple(item.action_id for item in values) != tuple(item.action_id for item in KNOB_ACTIONS):
                raise Q4ValidationError("knob action pool is not the frozen twelve-action panel")
            if name == "risk" and any(not item.is_risk_probe for item in values):
                raise Q4ValidationError("risk pool may contain only S11/S12")
            if name in {"strict_safe", "reusable_safe"} and any(item.is_risk_probe for item in values):
                raise Q4ValidationError("safe pools may not contain risk probes")
        object.__setattr__(self, "action_pools", {key: tuple(values) for key, values in sorted(pools.items())})
        contexts = tuple(self.contexts)
        if any(not isinstance(item, Q4Context) for item in contexts):
            raise Q4ValidationError("dataset contexts must contain Q4Context values")
        if len({item.context_id for item in contexts}) != len(contexts):
            raise Q4ValidationError("dataset contexts contain duplicates")
        context_groups: dict[tuple[str, ...], str] = {}
        for context in contexts:
            previous = context_groups.setdefault(context.group_key, context.context_id)
            if previous != context.context_id:
                raise Q4ValidationError("one seven-part group key cannot identify multiple contexts")
        object.__setattr__(self, "contexts", contexts)
        states = tuple(self.states)
        if any(not isinstance(item, Q4State) for item in states):
            raise Q4ValidationError("dataset states must contain Q4State values")
        if len({item.state_digest for item in states}) != len(states):
            raise Q4ValidationError("dataset states contain duplicate digests")
        context_ids = {item.context_id for item in contexts}
        if any(item.context_id not in context_ids for item in states):
            raise Q4ValidationError("state context is absent from dataset contexts")
        context_by_id = {item.context_id: item for item in contexts}
        manifest_strata = dict(self.split_manifest.stratum_by_context) if isinstance(self.split_manifest, SplitManifest) else {}
        manifest_models = dict(self.split_manifest.model_size_by_context) if isinstance(self.split_manifest, SplitManifest) else {}
        for state in states:
            context = context_by_id[state.context_id]
            if (state.objective_class != context.objective_class
                    or state.workload_stratum != context.workload_stratum
                    or state.arrival_pattern != context.arrival_pattern):
                raise Q4ValidationError("state categorical context fields do not match Q4Context")
            expected_plan = PlanKind.STRICT.value if state.workload_stratum in {WorkloadStratum.HOMOGENEOUS.value, WorkloadStratum.HETEROGENEOUS.value} else PlanKind.REUSABLE.value
            if state.plan_kind != expected_plan:
                raise Q4ValidationError("state plan_kind does not match frozen workload stratum")
            if state.context_id in manifest_strata and manifest_strata[state.context_id] != state.workload_stratum:
                raise Q4ValidationError("state workload stratum disagrees with split manifest")
            if state.context_id in manifest_models and manifest_models[state.context_id] != state.model_size:
                raise Q4ValidationError("state model_size disagrees with split manifest")
        object.__setattr__(self, "states", states)
        trajectories = tuple(self.trajectories)
        if any(not isinstance(item, Trajectory) for item in trajectories):
            raise Q4ValidationError("dataset trajectories must contain Trajectory values")
        if len({item.trajectory_id for item in trajectories}) != len(trajectories):
            raise Q4ValidationError("dataset trajectories contain duplicates")
        if any(item.context_id not in context_ids for item in trajectories):
            raise Q4ValidationError("trajectory context is absent from dataset contexts")
        object.__setattr__(self, "trajectories", trajectories)
        risks = tuple(self.risk_observations)
        if any(not isinstance(item, RiskObservation) for item in risks):
            raise Q4ValidationError("dataset risk_observations must contain RiskObservation values")
        if len({item.risk_observation_id for item in risks}) != len(risks):
            raise Q4ValidationError("dataset risk observations contain duplicates")
        state_ids = {item.state_digest for item in states}
        if any(item.state_digest not in state_ids for item in risks):
            raise Q4ValidationError("risk observation state is absent from dataset states")
        if any(evidence_id not in {artifact.artifact_id for artifact in artifacts} for item in risks for evidence_id in item.evidence_ids):
            raise Q4ValidationError("risk observation evidence is absent from source artifacts")
        object.__setattr__(self, "risk_observations", risks)
        panels = tuple(self.panel_cells)
        if any(not isinstance(item, PanelCell) for item in panels):
            raise Q4ValidationError("dataset panel_cells must contain PanelCell values")
        if len({item.cell_id for item in panels}) != len(panels):
            raise Q4ValidationError("dataset panel cells contain duplicates")
        if any(item.context_id not in context_ids for item in panels):
            raise Q4ValidationError("panel cell context is absent from dataset contexts")
        object.__setattr__(self, "panel_cells", panels)
        outcomes = tuple(self.outcomes)
        if any(not isinstance(item, Outcome) for item in outcomes):
            raise Q4ValidationError("dataset outcomes must contain Outcome values")
        if len({item.outcome_id for item in outcomes}) != len(outcomes):
            raise Q4ValidationError("dataset outcomes contain duplicates")
        object.__setattr__(self, "outcomes", outcomes)
        transitions = tuple(self.transitions)
        if any(not isinstance(item, Transition) for item in transitions):
            raise Q4ValidationError("dataset transitions must contain Transition values")
        if len({item.transition_id for item in transitions}) != len(transitions):
            raise Q4ValidationError("dataset transitions contain duplicates")
        state_ids = {item.state_digest for item in states}
        if transitions and any(item.state_digest not in state_ids or item.next_state_digest not in state_ids for item in transitions):
            raise Q4ValidationError("every transition current/next state must be present in dataset states")
        transition_by_id = {item.transition_id: item for item in transitions}
        trajectory_by_id = {item.trajectory_id: item for item in trajectories}
        if any(item.trajectory_id not in trajectory_by_id or item.transition_id not in trajectory_by_id[item.trajectory_id].transition_ids for item in transitions):
            raise Q4ValidationError("transition is not bound to its owning trajectory")
        for trajectory in trajectories:
            if any(item not in transition_by_id for item in trajectory.transition_ids):
                raise Q4ValidationError("trajectory references an absent transition")
            if any(transition_by_id[item].context.context_id != trajectory.context_id or transition_by_id[item].split != trajectory.split for item in trajectory.transition_ids):
                raise Q4ValidationError("trajectory transition context/split mismatch")
            if trajectory.trajectory_status == TrajectoryStatus.COMPLETE.value:
                linked = tuple(transition_by_id[item] for item in trajectory.transition_ids)
                if tuple(item.step_index for item in linked) != tuple(range(HORIZON)):
                    raise Q4ValidationError("complete trajectory steps must be ordered 0..16")
                if linked[-1].stage != Stage.REVALIDATE.value or not linked[-1].terminal:
                    raise Q4ValidationError("complete trajectory must end in terminal REVALIDATE")
                states_by_id = {item.state_digest: item for item in states}
                for index, transition in enumerate(linked):
                    if transition.state_digest not in states_by_id or transition.next_state_digest not in states_by_id:
                        raise Q4ValidationError("trajectory transition state is absent")
                    current_state = states_by_id[transition.state_digest]
                    next_state = states_by_id[transition.next_state_digest]
                    if current_state.context_id != trajectory.context_id or next_state.context_id != trajectory.context_id:
                        raise Q4ValidationError("trajectory state context mismatch")
                    if index < HORIZON - 1 and transition.next_state_digest != linked[index + 1].state_digest:
                        raise Q4ValidationError("trajectory next_state does not chain to next transition")
                    expected_stage = Stage.KNOB_DELTA if index <= 10 else Stage.STRATEGY_SELECT if index <= 15 else Stage.REVALIDATE
                    if transition.stage != expected_stage.value or current_state.step_index != index:
                        raise Q4ValidationError("trajectory stage/state step mismatch")
                    if index <= 10:
                        delta = KNOB_DELTA_BY_ID[transition.action_id]
                        if delta[0] != transition.previous_action_id or next_state.knob_action_id != delta[1]:
                            raise Q4ValidationError("knob transition does not update current state by its delta")
                        if transition.candidate_id is None or transition.candidate_id in current_state.evaluated_candidate_ids or next_state.evaluated_candidate_ids != current_state.evaluated_candidate_ids + (transition.candidate_id,):
                            raise Q4ValidationError("complete trajectory repeats or loses a knob candidate slot")
                    elif index <= 15:
                        if transition.previous_action_id != current_state.knob_action_id or next_state.knob_action_id != current_state.knob_action_id:
                            raise Q4ValidationError("strategy transition changed the final knob")
                        if transition.strategy_candidate_index != index - 11:
                            raise Q4ValidationError("strategy candidate index is not ordered 0..4")
                        if set(current_state.evaluated_candidate_ids) != set(KNOB_CANDIDATE_BY_ID) or next_state.evaluated_candidate_ids != current_state.evaluated_candidate_ids:
                            raise Q4ValidationError("strategy state does not preserve all evaluated knob candidate slots")
                    else:
                        if transition.previous_action_id != transition.action_id:
                            raise Q4ValidationError("revalidation must carry the final selected strategy marker")
                        if set(current_state.evaluated_candidate_ids) != set(KNOB_CANDIDATE_BY_ID):
                            raise Q4ValidationError("revalidation state does not preserve all evaluated knob candidate slots")
        by_context = {item.context.context_id: item.split for item in transitions}
        by_group: dict[tuple[str, ...], str] = {}
        context_by_id = {item.context_id: item for item in contexts}
        artifact_ids = {item.artifact_id for item in artifacts}
        outcome_by_id = {item.outcome_id: item for item in outcomes}
        candidate_ids_seen: set[str] = set()
        candidate_raw_seen: set[tuple[str, str, str]] = set()
        reference_scope: dict[str, tuple[str, str]] = {}
        for panel in panels:
            if panel.outcome_id not in outcome_by_id or panel.reference_outcome_id not in outcome_by_id:
                raise Q4ValidationError("panel cell outcome is absent from dataset outcomes")
            if not outcome_by_id[panel.outcome_id].complete_safe or not outcome_by_id[panel.reference_outcome_id].complete_safe:
                raise Q4ValidationError("panel cell requires complete-safe candidate/reference outcomes")
            candidate = outcome_by_id[panel.outcome_id]
            reference = outcome_by_id[panel.reference_outcome_id]
            context = next((item for item in contexts if item.context_id == panel.context_id), None)
            if context is None:
                raise Q4ValidationError("panel cell context is absent from dataset contexts")
            plan = PlanKind.STRICT if context.workload_stratum in {WorkloadStratum.HOMOGENEOUS.value, WorkloadStratum.HETEROGENEOUS.value} else PlanKind.REUSABLE
            base_strategy = ScheduleAction.safe_pool(plan)[0].action_id
            if any(item.context_id != panel.context_id for item in (candidate, reference)):
                raise Q4ValidationError("panel outcomes must belong to the panel context")
            if any(item.objective_class != context.objective_class or item.plan_kind != plan.value or item.model_digest != context.model_digest or item.model_manifest_digest != context.model_manifest_digest or item.workload_digest != context.workload_digest for item in (candidate, reference)):
                raise Q4ValidationError("panel outcome identity does not match canonical context")
            if (candidate.knob_action_id != panel.knob_action_id
                    or reference.knob_action_id != panel.knob_action_id
                    or candidate.strategy_action_id != panel.strategy_action_id
                    or reference.strategy_action_id != base_strategy):
                raise Q4ValidationError("panel cell candidate identity does not match its knob/strategy cell")
            if panel.outcome_id in candidate_ids_seen:
                raise Q4ValidationError("panel candidate outcome is reused across distinct cells")
            candidate_ids_seen.add(panel.outcome_id)
            for ref in candidate.raw_sample_refs:
                raw_key = (ref.artifact_id, ref.sha256, ref.quality.value)
                if raw_key in candidate_raw_seen:
                    raise Q4ValidationError("panel candidate raw artifact is reused across distinct cells")
                candidate_raw_seen.add(raw_key)
            previous_scope = reference_scope.setdefault(panel.reference_outcome_id, (panel.context_id, panel.knob_action_id))
            if previous_scope != (panel.context_id, panel.knob_action_id):
                raise Q4ValidationError("panel BASE reference is reused outside its exact context/knob")
        for item in transitions:
            if item.context.context_id not in {ctx.context_id for ctx in contexts}:
                raise Q4ValidationError("transition context is absent from dataset contexts")
            if context_by_id[item.context.context_id] != item.context:
                raise Q4ValidationError("transition context is not the canonical dataset context")
            if by_context[item.context.context_id] != item.split:
                raise Q4ValidationError("context/group crosses Q4 splits")
            previous_group = by_group.setdefault(item.context.group_key, item.split)
            if previous_group != item.split:
                raise Q4ValidationError("seven-part group key crosses Q4 splits")
            if item.outcome_id not in outcome_by_id:
                raise Q4ValidationError("transition outcome is absent from dataset outcomes")
            if item.reference_outcome_id not in outcome_by_id:
                raise Q4ValidationError("transition reference outcome is absent from dataset outcomes")
            if any(evidence_id not in artifact_ids for evidence_id in item.evidence_ids):
                raise Q4ValidationError("transition evidence reference is absent from source artifacts")
        for outcome in outcomes:
            for ref in outcome.raw_sample_refs:
                artifact = next((item for item in artifacts if item.artifact_id == ref.artifact_id), None)
                if artifact is None:
                    raise Q4ValidationError("outcome raw reference is absent from source artifacts")
                if artifact.sha256 != ref.sha256 or artifact.quality != ref.quality.value:
                    raise Q4ValidationError("outcome raw reference does not match source artifact hash/quality")
        # A COMPLETE trajectory may not hide an earlier censoring/failure or
        # partial abort.  All 17 candidate and reference outcomes are safe;
        # extra aborted trajectories remain retained outside this gate.
        for trajectory in trajectories:
            if trajectory.trajectory_status != TrajectoryStatus.COMPLETE.value:
                continue
            linked = tuple(transition_by_id[item] for item in trajectory.transition_ids)
            if any(item.partial_abort is not None or (item.terminal != (index == 16)) for index, item in enumerate(linked)):
                raise Q4ValidationError("complete trajectory contains an abort or non-terminal marker")
            if any(not outcome_by_id[item.outcome_id].complete_safe or not outcome_by_id[item.reference_outcome_id].complete_safe for item in linked):
                raise Q4ValidationError("complete trajectory contains an unsafe candidate/reference outcome")
        object.__setattr__(self, "transitions", transitions)
        if not isinstance(self.split_manifest, SplitManifest):
            raise Q4ValidationError("dataset split_manifest must be SplitManifest")
        if not isinstance(self.seed_manifest, Mapping) or any(type(k) is not str or type(v) is not str for k, v in self.seed_manifest.items()):
            raise Q4ValidationError("dataset seed_manifest must be a string map")
        object.__setattr__(self, "seed_manifest", dict(sorted(self.seed_manifest.items())))
        if self.no_invented_performance is not True:
            raise Q4ValidationError("dataset.no_invented_performance must be true")
        computed = canonical_sha256(self._semantic_dict())
        if self.dataset_id and self.dataset_id != computed:
            raise Q4ValidationError("dataset_id does not match canonical content")
        object.__setattr__(self, "dataset_id", computed)

    def _semantic_dict(self) -> dict[str, Any]:
        return {"schema": self.SCHEMA, "preregistration_sha256": self.preregistration_sha256,
                "source_artifacts": self.source_artifacts, "action_pools": self.action_pools,
                "contexts": self.contexts, "states": self.states, "trajectories": self.trajectories, "transitions": self.transitions, "outcomes": self.outcomes,
                "risk_observations": self.risk_observations,
                "panel_cells": self.panel_cells,
                "split_manifest": self.split_manifest, "seed_manifest": self.seed_manifest,
                "no_invented_performance": self.no_invented_performance}

    def to_dict(self) -> dict[str, Any]:
        return {**self._semantic_dict(), "source_artifacts": [item.to_dict() for item in self.source_artifacts],
                "action_pools": {key: [item.to_dict() for item in values] for key, values in self.action_pools.items()},
                "contexts": [item.to_dict() for item in self.contexts], "states": [item.to_dict() for item in self.states], "trajectories": [item.to_dict() for item in self.trajectories], "transitions": [item.to_dict() for item in self.transitions],
                "outcomes": [item.to_dict() for item in self.outcomes], "risk_observations": [item.to_dict() for item in self.risk_observations], "panel_cells": [item.to_dict() for item in self.panel_cells], "split_manifest": self.split_manifest.to_dict(), "dataset_id": self.dataset_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Dataset":
        _strict(data, {"schema", "preregistration_sha256", "dataset_id", "source_artifacts", "action_pools", "contexts", "states", "trajectories", "transitions", "outcomes", "risk_observations", "panel_cells", "split_manifest", "seed_manifest", "no_invented_performance"}, cls.__name__)
        if data["schema"] != cls.SCHEMA:
            raise Q4ValidationError("unsupported q4 dataset schema")
        pools = {}
        for name, values in data["action_pools"].items():
            if not isinstance(values, list):
                raise Q4ValidationError("dataset action pool must be an array")
            pools[name] = tuple(KnobAction.from_dict(item) if name == "knob" else ScheduleAction.from_dict(item) for item in values)
        return cls(preregistration_sha256=data["preregistration_sha256"], source_artifacts=tuple(ArtifactRecord.from_dict(item) for item in data["source_artifacts"]), action_pools=pools, contexts=tuple(Q4Context.from_dict(item) for item in data["contexts"]), states=tuple(Q4State.from_dict(item) for item in data["states"]), trajectories=tuple(Trajectory.from_dict(item) for item in data["trajectories"]), transitions=tuple(Transition.from_dict(item) for item in data["transitions"]), outcomes=tuple(Outcome.from_dict(item) for item in data["outcomes"]), risk_observations=tuple(RiskObservation.from_dict(item) for item in data["risk_observations"]), panel_cells=tuple(PanelCell.from_dict(item) for item in data["panel_cells"]), split_manifest=SplitManifest.from_dict(data["split_manifest"]), seed_manifest=data["seed_manifest"], no_invented_performance=data["no_invented_performance"], dataset_id=data["dataset_id"])

    def coverage_report(self) -> dict[str, Any]:
        by_trajectory: dict[str, set[int]] = {}
        terminal: set[str] = set()
        for transition in self.transitions:
            by_trajectory.setdefault(transition.trajectory_id, set()).add(transition.step_index)
            if transition.terminal and transition.step_index == 16:
                terminal.add(transition.trajectory_id)
        complete_h17 = sum(item.trajectory_status == TrajectoryStatus.COMPLETE.value for item in self.trajectories)
        readiness = self._rl_readiness()
        # Model size is intentionally not inferred from a model digest.  The
        # collection manifest owns that mapping; imported historical contexts
        # therefore remain ``unknown`` until a new Q4 context declares it.
        return {"schema": "ironmule.q4_coverage.v1", "dataset_id": self.dataset_id,
                "contexts": len(self.contexts), "transitions": len(self.transitions),
                "outcomes": len(self.outcomes), "complete_h17": complete_h17,
                "splits": {split.value: sum(item.split == split.value for item in self.transitions) for split in Q4Split},
                "model_cells": {size: 0 for size in _MODEL_SIZES},
                "required_contexts": dict(REQUIRED_CONTEXTS),
                "required_complete_trajectories": REQUIRED_COMPLETE_TRAJECTORIES,
                "required_complete_transitions": REQUIRED_COMPLETE_TRANSITIONS,
                "rl_structurally_eligible": readiness[0],
                "rl_eligibility_reasons": readiness[1],
                "no_invented_performance": self.no_invented_performance}

    def _rl_readiness(self) -> tuple[bool, tuple[str, ...]]:
        """Apply the complete preregistered Q4 structural gate.

        This deliberately returns false for partial/imported corpora.  It is
        not a statistical claim and never promotes a historical observation.
        """
        reasons: list[str] = []
        split_map = dict(self.split_manifest.split_contexts)
        context_ids = {item.context_id for item in self.contexts}
        if {key: len(value) for key, value in split_map.items()} != REQUIRED_CONTEXTS:
            reasons.append("split context counts are not 12/6/6")
        if set(item for values in split_map.values() for item in values) != context_ids:
            reasons.append("split manifest does not cover exactly dataset contexts")
        model_by_context = dict(self.split_manifest.model_size_by_context)
        context_by_id_for_manifest = {ctx.context_id: ctx for ctx in self.contexts}
        declared_strata = dict(self.split_manifest.stratum_by_context)
        if declared_strata != {context_id: context.workload_stratum for context_id, context in context_by_id_for_manifest.items()}:
            reasons.append("split manifest does not declare exact workload strata for every context")
        if set(model_by_context) != context_ids:
            reasons.append("split manifest does not declare model size for every context")
        for split, required_count in REQUIRED_CONTEXTS.items():
            for model in _MODEL_SIZES:
                count = sum(model_by_context.get(context_id) == model for context_id in split_map.get(split, ()))
                expected = required_count // 3
                if count != expected:
                    reasons.append(f"{split} lacks exact {expected} contexts for {model}")
        if len(self.contexts) != sum(REQUIRED_CONTEXTS.values()):
            reasons.append("dataset does not contain exactly 24 contexts")
        complete_trajectories = tuple(item for item in self.trajectories if item.trajectory_status == TrajectoryStatus.COMPLETE.value)
        if len(complete_trajectories) < REQUIRED_COMPLETE_TRAJECTORIES:
            reasons.append("dataset has fewer than 72 complete H17 trajectories")
        # Retained ABORTED trajectories are valuable failure/recovery evidence,
        # but are explicitly outside the safe RL training denominator.
        complete_ids = {item.trajectory_id for item in complete_trajectories}
        complete_transitions = tuple(item for item in self.transitions if item.trajectory_id in complete_ids)
        if len(complete_transitions) < REQUIRED_COMPLETE_TRANSITIONS:
            reasons.append("complete trajectories contribute fewer than 1224 transitions")
        transition_by_context: dict[str, list[Transition]] = {}
        complete_outcome_ids: set[str] = set()
        for transition in complete_transitions:
            transition_by_context.setdefault(transition.context.context_id, []).append(transition)
            complete_outcome_ids.add(transition.outcome_id)
        outcomes_by_id = {item.outcome_id: item for item in self.outcomes}
        if any(item.outcome_id not in outcomes_by_id or not outcomes_by_id[item.outcome_id].complete_safe for item in complete_transitions):
            reasons.append("a complete H17 transition lacks a complete-safe outcome")
        if len(complete_trajectories) >= REQUIRED_COMPLETE_TRAJECTORIES and any(len(item.transition_ids) != HORIZON for item in complete_trajectories):
            reasons.append("a complete trajectory does not contain exactly 17 transitions")
        if len(transition_by_context) != sum(REQUIRED_CONTEXTS.values()):
            reasons.append("complete transitions do not cover all 24 contexts")
        context_by_id = {ctx.context_id: ctx for ctx in self.contexts}
        panel_by_context: dict[str, list[PanelCell]] = {}
        for panel in getattr(self, "panel_cells", ()):
            panel_by_context.setdefault(panel.context_id, []).append(panel)
        for context_id, rows in transition_by_context.items():
            if len(rows) != HORIZON * TRAJECTORIES_PER_CONTEXT:
                reasons.append(f"{context_id} does not have 3 complete H17 trajectories")
                continue
            if sum(item.stage == Stage.KNOB_DELTA.value for item in rows) != 33:
                reasons.append(f"{context_id} does not have 33 knob transitions")
            if sum(item.stage == Stage.STRATEGY_SELECT.value for item in rows) != 15:
                reasons.append(f"{context_id} does not have 15 strategy transitions")
            if sum(item.stage == Stage.REVALIDATE.value for item in rows) != 3:
                reasons.append(f"{context_id} does not have 3 revalidation transitions")
            if context_id not in context_by_id:
                reasons.append(f"{context_id} context is absent")
                continue
            plan = PlanKind.STRICT if context_by_id[context_id].workload_stratum in {WorkloadStratum.HOMOGENEOUS.value, WorkloadStratum.HETEROGENEOUS.value} else PlanKind.REUSABLE
            strategy_ids = {item.action_id for item in rows if item.stage == Stage.STRATEGY_SELECT.value}
            expected_pool = {item.action_id for item in ScheduleAction.safe_pool(plan)}
            if strategy_ids != expected_pool:
                reasons.append(f"{context_id} does not cover all five plan-matching strategies")
            panel_rows = panel_by_context.get(context_id, [])
            anchor_ids = {item.knob_action_id for item in panel_rows}
            pair_ids = {(item.knob_action_id, item.strategy_action_id) for item in panel_rows}
            if anchor_ids != {item.action_id for item in KNOB_ACTIONS}:
                reasons.append(f"{context_id} lacks the complete twelve-anchor knob panel")
            if len(panel_rows) != 60 or len(pair_ids) != 60 or pair_ids != {(knob.action_id, strategy.action_id) for knob in KNOB_ACTIONS for strategy in ScheduleAction.safe_pool(plan)}:
                reasons.append(f"{context_id} lacks the exact sixty-cell knob-by-strategy panel")
        if not getattr(self, "risk_observations", ()) and not any(item.partial_abort is not None or item.outcome_id in outcomes_by_id and not outcomes_by_id[item.outcome_id].complete_safe for item in self.transitions):
            reasons.append("risk/failure support is empty")
        return not reasons, tuple(dict.fromkeys(reasons))

    def derive_rewards(self, *, strict: bool = False) -> RewardDerivationResult:
        """Join candidate/reference outcomes under the frozen reward contract.

        Expected failed/aborted rows are retained as explicit INELIGIBLE
        derivation records.  ``strict=True`` is available to callers that want
        the old all-or-nothing gate; neither mode imputes a reward.
        """
        rewards: dict[str, RewardRecord] = {}
        records: dict[str, RewardDerivationRecord] = {}
        for transition in self.transitions:
            try:
                reward = _derive_one_reward(self, transition)
            except Q4ValidationError as exc:
                if strict:
                    raise
                records[transition.transition_id] = RewardDerivationRecord(transition.transition_id, "INELIGIBLE", None, str(exc))
                continue
            rewards[transition.transition_id] = reward
            records[transition.transition_id] = RewardDerivationRecord(transition.transition_id, "DERIVED", reward.reward_id, "derived from complete-safe candidate/reference outcomes")
        return RewardDerivationResult(dict(sorted(rewards.items())), dict(sorted(records.items())))


def _derive_one_reward(dataset: Dataset, transition: Transition) -> RewardRecord:
    outcomes = {item.outcome_id: item for item in dataset.outcomes}
    contexts = {item.context_id: item for item in dataset.contexts}
    if transition.outcome_id == transition.reference_outcome_id:
        raise Q4ValidationError("candidate and reference outcome IDs must differ")
    candidate = outcomes.get(transition.outcome_id)
    reference = outcomes.get(transition.reference_outcome_id)
    if candidate is None or reference is None:
        raise Q4ValidationError("reward join references a missing outcome")
    if not candidate.complete_safe or not reference.complete_safe:
        raise Q4ValidationError("reward cannot be derived from unsafe or censored outcomes")
    context = contexts.get(transition.context.context_id)
    if context is None or context != transition.context:
        raise Q4ValidationError("reward transition context is not canonical")
    for outcome in (candidate, reference):
        if (outcome.objective_class != context.objective_class
                or outcome.context_id != context.context_id
                or outcome.model_digest != context.model_digest
                or outcome.model_manifest_digest != context.model_manifest_digest
                or outcome.workload_digest != context.workload_digest):
            raise Q4ValidationError("outcome identity does not match transition context")
    if transition.stage == Stage.KNOB_DELTA.value:
        delta = KNOB_DELTA_BY_ID.get(transition.action_id)
        if delta is None or candidate.knob_action_id != delta[1] or reference.knob_action_id != delta[0]:
            raise Q4ValidationError("knob candidate/reference identity does not match delta")
        if candidate.strategy_action_id is not None or reference.strategy_action_id is not None:
            raise Q4ValidationError("knob reward outcomes must not carry strategy identity")
    elif transition.stage == Stage.STRATEGY_SELECT.value:
        selected = SCHEDULE_ACTION_BY_ID.get(transition.action_id)
        if selected is None or selected.is_risk_probe:
            raise Q4ValidationError("strategy reward action is not safe")
        if candidate.knob_action_id != transition.previous_action_id or reference.knob_action_id != transition.previous_action_id:
            raise Q4ValidationError("strategy outcomes must share the final knob identity")
        if candidate.strategy_action_id != selected.action_id:
            raise Q4ValidationError("candidate strategy identity does not match transition")
        plan = PlanKind.STRICT if context.workload_stratum in {WorkloadStratum.HOMOGENEOUS.value, WorkloadStratum.HETEROGENEOUS.value} else PlanKind.REUSABLE
        if reference.strategy_action_id != ScheduleAction.safe_pool(plan)[0].action_id:
            raise Q4ValidationError("strategy reference must be the matching plan BASE strategy")
    else:
        selected = SCHEDULE_ACTION_BY_ID.get(transition.action_id)
        if selected is None or selected.is_risk_probe or transition.action_id != transition.previous_action_id:
            raise Q4ValidationError("revalidation must use the final selected strategy marker")
        terminal_state = next((state for state in dataset.states if state.state_digest == transition.state_digest), None)
        if terminal_state is None or candidate.knob_action_id != terminal_state.knob_action_id:
            raise Q4ValidationError("revalidation candidate must retain the terminal final knob")
        if candidate.strategy_action_id != transition.previous_action_id:
            raise Q4ValidationError("revalidation candidate strategy does not match marker")
        plan = PlanKind.STRICT if context.workload_stratum in {WorkloadStratum.HOMOGENEOUS.value, WorkloadStratum.HETEROGENEOUS.value} else PlanKind.REUSABLE
        if reference.knob_action_id != KnobAction.baseline().action_id or reference.strategy_action_id != ScheduleAction.safe_pool(plan)[0].action_id:
            raise Q4ValidationError("revalidation reference must be frozen BASE")
    candidate_cost = candidate.metric_value(transition.stage)
    reference_cost = reference.metric_value(transition.stage)
    if (context.objective_class == ObjectiveClass.THROUGHPUT.value
            and transition.stage in {Stage.STRATEGY_SELECT.value, Stage.REVALIDATE.value}
            and candidate.p95_full_response_ms > reference.p95_full_response_ms * 1.10):
        raise Q4ValidationError("throughput strategy p95 full-response inflation exceeds 10% guard")
    return RewardRecord(transition.transition_id, candidate.outcome_id, reference.outcome_id, context.objective_class, candidate_cost, reference_cost, math.log(reference_cost / candidate_cost))


@dataclass(frozen=True, slots=True)
class ForeignBundleMetadata:
    SCHEMA: ClassVar[str] = "ironmule.q4_foreign_bundle.v1"

    bundle_id: str
    exporter_id: str
    host_class: str
    hardware_digest: str
    model_digest: str
    model_manifest_digest: str
    runtime_digest: str
    code_digest: str
    workload_digest: str
    preregistration_sha256: str
    raw_artifacts: tuple[Any, ...]
    reviewer_record_sha256: str
    signature_algorithm: str
    signer_key_fingerprint: str
    signature: str
    exported_at_utc: str
    public_key_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "bundle_id", _digest("foreign.bundle_id", self.bundle_id))
        for name in ("exporter_id", "host_class", "public_key_id"):
            object.__setattr__(self, name, _text(f"foreign.{name}", getattr(self, name)))
        for name in ("hardware_digest", "model_digest", "model_manifest_digest", "runtime_digest", "code_digest", "workload_digest", "preregistration_sha256", "reviewer_record_sha256"):
            object.__setattr__(self, name, _digest(f"foreign.{name}", getattr(self, name)))
        object.__setattr__(self, "raw_artifacts", _ref_tuple("foreign.raw_artifacts", self.raw_artifacts))
        if self.signature_algorithm != "Ed25519":
            raise Q4ValidationError("foreign signature_algorithm must be Ed25519")
        for name in ("signer_key_fingerprint", "signature"):
            object.__setattr__(self, name, _text(f"foreign.{name}", getattr(self, name)))
        try:
            base64.b64decode(self.signature, validate=True)
        except Exception as exc:
            raise Q4ValidationError("foreign signature must be base64") from exc
        object.__setattr__(self, "exported_at_utc", _utc("foreign.exported_at_utc", self.exported_at_utc))

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.SCHEMA, "bundle_id": self.bundle_id, "exporter_id": self.exporter_id, "host_class": self.host_class, "hardware_digest": self.hardware_digest, "model_digest": self.model_digest, "model_manifest_digest": self.model_manifest_digest, "runtime_digest": self.runtime_digest, "code_digest": self.code_digest, "workload_digest": self.workload_digest, "preregistration_sha256": self.preregistration_sha256, "raw_artifacts": [item.to_dict() for item in self.raw_artifacts], "reviewer_record_sha256": self.reviewer_record_sha256, "signature_algorithm": self.signature_algorithm, "signer_key_fingerprint": self.signer_key_fingerprint, "signature": self.signature, "exported_at_utc": self.exported_at_utc, "public_key_id": self.public_key_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ForeignBundleMetadata":
        _strict(data, {"schema", "bundle_id", "exporter_id", "host_class", "hardware_digest", "model_digest", "model_manifest_digest", "runtime_digest", "code_digest", "workload_digest", "preregistration_sha256", "raw_artifacts", "reviewer_record_sha256", "signature_algorithm", "signer_key_fingerprint", "signature", "exported_at_utc", "public_key_id"}, cls.__name__)
        if data["schema"] != cls.SCHEMA or not isinstance(data["raw_artifacts"], list):
            raise Q4ValidationError("unsupported q4 foreign bundle schema")
        return cls(bundle_id=data["bundle_id"], exporter_id=data["exporter_id"], host_class=data["host_class"], hardware_digest=data["hardware_digest"], model_digest=data["model_digest"], model_manifest_digest=data["model_manifest_digest"], runtime_digest=data["runtime_digest"], code_digest=data["code_digest"], workload_digest=data["workload_digest"], preregistration_sha256=data["preregistration_sha256"], raw_artifacts=tuple(ArtifactRef.from_dict(item) for item in data["raw_artifacts"]), reviewer_record_sha256=data["reviewer_record_sha256"], signature_algorithm=data["signature_algorithm"], signer_key_fingerprint=data["signer_key_fingerprint"], signature=data["signature"], exported_at_utc=data["exported_at_utc"], public_key_id=data["public_key_id"])


# Names used by early Q4 design notes and callers.
Q4Dataset = Dataset
ExecutionStrategyAction = ScheduleAction
ForeignBundle = ForeignBundleMetadata
PartialAbortRecord = PartialAbort
Split = Q4Split


__all__ = [
    "Q4ValidationError", "canonical_json", "canonical_sha256", "HORIZON", "KNOB_NAMES",
    "REQUIRED_CONTEXTS", "TRAJECTORIES_PER_CONTEXT", "REQUIRED_COMPLETE_TRAJECTORIES",
    "REQUIRED_COMPLETE_TRANSITIONS",
    "ObjectiveClass", "WorkloadStratum", "PlanKind", "Stage", "ActionSpace", "Q4Split",
    "TrajectoryStatus", "FailureState", "OutcomeStatus", "RollbackStatus", "SemanticClass",
    "StrategyClass", "HistoricalRole", "ArtifactRef", "EvidenceQuality", "KnobAction",
    "KNOB_ACTIONS", "KNOB_CANDIDATES", "KnobCandidateSpec", "Q2_CURRENT_ACTION",
    "INTERACTION_KNOB_ANCHORS", "ALL_DECLARED_KNOB_ACTIONS", "LEGAL_KNOB_ACTIONS", "KNOB_CANDIDATE_BY_ID", "KNOB_DELTA_IDS", "KNOB_DELTA_BY_ID", "knob_catalogue", "ScheduleAction", "ExecutionStrategyAction",
    "SCHEDULE_ACTIONS", "KnobDelta", "HybridAction", "Q4Context", "Q4State",
    "FEATURE_VECTOR_ORDER", "Outcome", "PartialAbort", "PartialAbortRecord", "Transition",
    "Trajectory", "RiskObservation", "ArtifactRecord", "PriorRecord", "SplitManifest",
    "RewardRecord", "PanelCell", "RewardDerivationRecord", "RewardDerivationResult", "Dataset", "Q4Dataset", "ForeignBundleMetadata", "ForeignBundle", "Split",
]
