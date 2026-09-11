"""Closed candidate registry for the Friday Gemma optimizer.

There is intentionally no API for arbitrary flags, source, kernels, or search
parameters.  A historical result can influence ordering, but only this module
can say that a candidate is executable for a fingerprint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType
from typing import Any, Iterable, Mapping

#: An immutable empty mapping, shared. Spelled as a factory because a
#: dataclass default must be hashable on Python 3.11 and ``mappingproxy``
#: is not; 3.12 relaxed that check to list/dict/set only (gh-96151), which
#: is why the bare default worked here and nowhere else.
_EMPTY_MAPPING: Mapping[str, Any] = MappingProxyType({})

from .fingerprint import ExactFingerprint


CANDIDATE_IDS = (
    "baseline",
    "persistent_process",
    "fixed_compiled_cache",
    "head_skip_prefill",
    "readback_every_2",
    "combined_core_profile",
    "throughput_width_2",
    "throughput_width_3",
    "throughput_width_4",
)

_EXPECTED_PARAMETERS = {
    "baseline": {}, "persistent_process": {}, "fixed_compiled_cache": {},
    "head_skip_prefill": {}, "readback_every_2": {}, "combined_core_profile": {},
    "throughput_width_2": {"width": 2}, "throughput_width_3": {"width": 3},
    "throughput_width_4": {"width": 4},
}


class CandidateError(ValueError):
    """An unknown candidate or an attempt to extend the closed action space."""


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(v) for v in value)
    return value


def _str_tuple(values: Iterable[str], field: str) -> tuple[str, ...]:
    result = tuple(values)
    if any(not isinstance(v, str) or not v or len(v) > 128 for v in result):
        raise CandidateError(f"{field} contains an invalid identifier")
    return result


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    """One immutable, preregistered action.

    ``parameters`` is a closed declarative descriptor.  It is not forwarded to
    a shell or interpreted as code; changing a value creates no valid spec.
    """

    candidate_id: str
    allowed_model_ids: tuple[str, ...] = ()
    allowed_model_families: tuple[str, ...] = ()
    allowed_chip_families: tuple[str, ...] = ()
    allowed_mlx_versions: tuple[str, ...] = ()
    allowed_workload_modes: tuple[str, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=lambda: _EMPTY_MAPPING)
    prerequisites: tuple[str, ...] = ()
    requires_exact_fingerprint: bool = True
    requires_greedy: bool | None = None
    requires_prompt_logprobs: bool | None = None
    throughput_only: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or self.candidate_id not in CANDIDATE_IDS:
            raise CandidateError("candidate_id is not in the closed registry")
        if not isinstance(self.requires_exact_fingerprint, bool):
            raise TypeError("requires_exact_fingerprint must be bool")
        for name in ("requires_greedy", "requires_prompt_logprobs", "throughput_only"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{name} must be bool or None")
        object.__setattr__(self, "allowed_model_ids", _str_tuple(self.allowed_model_ids, "allowed_model_ids"))
        object.__setattr__(self, "allowed_model_families", _str_tuple(self.allowed_model_families, "allowed_model_families"))
        object.__setattr__(self, "allowed_chip_families", _str_tuple(self.allowed_chip_families, "allowed_chip_families"))
        object.__setattr__(self, "allowed_mlx_versions", _str_tuple(self.allowed_mlx_versions, "allowed_mlx_versions"))
        object.__setattr__(self, "allowed_workload_modes", _str_tuple(self.allowed_workload_modes, "allowed_workload_modes"))
        object.__setattr__(self, "prerequisites", _str_tuple(self.prerequisites, "prerequisites"))
        if any(item not in CANDIDATE_IDS for item in self.prerequisites):
            raise CandidateError("candidate prerequisite is not registered")
        if not isinstance(self.parameters, Mapping) or isinstance(self.parameters, (str, bytes, bytearray)):
            raise TypeError("parameters must be a mapping")
        object.__setattr__(self, "parameters", _freeze(dict(self.parameters)))
        if dict(self.parameters) != _EXPECTED_PARAMETERS[self.candidate_id]:
            raise CandidateError("candidate parameters do not match the preregistered spec")
        if not isinstance(self.notes, str) or len(self.notes) > 512:
            raise CandidateError("candidate notes are unbounded")

    @property
    def id(self) -> str:
        return self.candidate_id

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "allowed_model_ids": list(self.allowed_model_ids),
            "allowed_model_families": list(self.allowed_model_families),
            "allowed_chip_families": list(self.allowed_chip_families),
            "allowed_mlx_versions": list(self.allowed_mlx_versions),
            "allowed_workload_modes": list(self.allowed_workload_modes),
            "parameters": dict(self.parameters),
            "prerequisites": list(self.prerequisites),
            "requires_exact_fingerprint": self.requires_exact_fingerprint,
            "requires_greedy": self.requires_greedy,
            "requires_prompt_logprobs": self.requires_prompt_logprobs,
            "throughput_only": self.throughput_only,
        }

    def accepts(self, fingerprint: ExactFingerprint | None, *, qualified: Iterable[str] = ()) -> bool:
        """Return whether this exact spec can be offered for *fingerprint*."""

        if self.candidate_id == "baseline":
            return True
        if fingerprint is None or not isinstance(fingerprint, ExactFingerprint):
            return False
        if self.requires_exact_fingerprint and not fingerprint.recommendation_allowed:
            return False
        model = fingerprint.model
        workload = fingerprint.workload
        if self.allowed_model_ids and model.model_id not in self.allowed_model_ids:
            return False
        if self.allowed_model_families:
            model_id = model.model_id or ""
            if not any(family.lower() in model_id.lower() for family in self.allowed_model_families):
                return False
        if self.allowed_chip_families:
            chip = fingerprint.environment.chip or ""
            if not any(family.lower() in chip.lower() for family in self.allowed_chip_families):
                return False
        if self.allowed_mlx_versions and fingerprint.environment.mlx not in self.allowed_mlx_versions:
            return False
        if self.allowed_workload_modes and workload.mode not in self.allowed_workload_modes:
            return False
        if self.requires_greedy is not None and workload.greedy != self.requires_greedy:
            return False
        if self.requires_prompt_logprobs is not None and workload.prompt_logprobs != self.requires_prompt_logprobs:
            return False
        if self.throughput_only and workload.mode != "throughput":
            return False
        available = set(qualified)
        return all(prerequisite in available for prerequisite in self.prerequisites)


def _default_specs() -> tuple[CandidateSpec, ...]:
    return (
        CandidateSpec("baseline", requires_exact_fingerprint=False, notes="Unchanged reference path."),
        CandidateSpec("persistent_process", allowed_workload_modes=("interactive", "throughput")),
        CandidateSpec("fixed_compiled_cache", allowed_workload_modes=("interactive", "throughput")),
        CandidateSpec(
            "head_skip_prefill", allowed_workload_modes=("interactive",), requires_greedy=True,
            requires_prompt_logprobs=False,
        ),
        CandidateSpec(
            "readback_every_2", allowed_model_families=("gemma-3-4b", "gemma3-4b", "gemma-4b"),
            allowed_chip_families=("m1 max", "m1max"),
            allowed_mlx_versions=("0.32.0",),
            allowed_workload_modes=("interactive",), requires_greedy=True, requires_prompt_logprobs=False,
            notes="Q2 4B/M1 Max/MLX scope; this is not a general readback recommendation.",
        ),
        CandidateSpec(
            "combined_core_profile", allowed_workload_modes=("interactive",),
            requires_greedy=True, requires_prompt_logprobs=False,
            prerequisites=("fixed_compiled_cache", "head_skip_prefill"),
        ),
        CandidateSpec("throughput_width_2", allowed_workload_modes=("throughput",), throughput_only=True, parameters={"width": 2}),
        CandidateSpec("throughput_width_3", allowed_workload_modes=("throughput",), throughput_only=True, parameters={"width": 3}),
        CandidateSpec("throughput_width_4", allowed_workload_modes=("throughput",), throughput_only=True, parameters={"width": 4}),
    )


class CandidateRegistry:
    """Read-only registry and deterministic allowlist matcher."""

    __slots__ = ("_specs", "_registry_hash", "_sealed")

    def __init__(self, specs: Iterable[CandidateSpec] | None = None) -> None:
        if specs is not None:
            raise CandidateError("production registry has immutable preregistered specs")
        values = _default_specs()
        self._install(values)

    def _install(self, values: tuple[CandidateSpec, ...]) -> None:
        if getattr(self, "_sealed", False):
            raise CandidateError("candidate registry is sealed")
        if len(values) != len(set(spec.candidate_id for spec in values)):
            raise CandidateError("duplicate candidate id")
        if {spec.candidate_id for spec in values} != set(CANDIDATE_IDS):
            raise CandidateError("registry must contain exactly the preregistered candidates")
        if values[0].candidate_id != "baseline":
            raise CandidateError("baseline must be first")
        object.__setattr__(self, "_specs", MappingProxyType({spec.candidate_id: spec for spec in values}))
        payload = {
            "schema_version": 1,
            "candidates": [spec.as_dict() for spec in values],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        object.__setattr__(self, "_registry_hash", hashlib.sha256(encoded).hexdigest())
        object.__setattr__(self, "_sealed", True)

    @classmethod
    def for_testing(cls, specs: Iterable[CandidateSpec]) -> "CandidateRegistry":
        """Build a registry only for isolated unit tests.

        The production constructor intentionally cannot be supplied specs.
        """
        instance = object.__new__(cls)
        instance._install(tuple(specs))
        return instance

    @property
    def registry_hash(self) -> str:
        return self._registry_hash

    @property
    def specs(self) -> tuple[CandidateSpec, ...]:
        return tuple(self._specs.values())

    def __iter__(self):
        return iter(self._specs.values())

    def __len__(self) -> int:
        return len(self._specs)

    def get(self, candidate_id: str) -> CandidateSpec:
        if not isinstance(candidate_id, str) or candidate_id not in self._specs:
            raise CandidateError(f"unknown candidate: {candidate_id!r}")
        return self._specs[candidate_id]

    def resolve(self, candidate_id: str, parameters: Mapping[str, Any] | None = None) -> CandidateSpec:
        spec = self.get(candidate_id)
        if parameters is not None:
            if not isinstance(parameters, Mapping) or isinstance(parameters, (str, bytes, bytearray)):
                raise CandidateError("candidate parameters must be a mapping")
            if dict(parameters) != dict(spec.parameters):
                raise CandidateError("unknown or mismatched candidate parameter")
        return spec

    def validate(self, candidate_id: str, *, fingerprint: ExactFingerprint | None = None,
                 parameters: Mapping[str, Any] | None = None, qualified: Iterable[str] = ()) -> CandidateSpec:
        spec = self.resolve(candidate_id, parameters)
        if not spec.accepts(fingerprint, qualified=qualified):
            raise CandidateError(f"candidate {candidate_id!r} is outside its measured scope")
        return spec

    def is_allowed(self, candidate_id: str, *, fingerprint: ExactFingerprint | None = None,
                   parameters: Mapping[str, Any] | None = None, qualified: Iterable[str] = ()) -> bool:
        try:
            self.validate(candidate_id, fingerprint=fingerprint, parameters=parameters, qualified=qualified)
        except (CandidateError, TypeError, ValueError):
            return False
        return True

    def available(self, fingerprint: ExactFingerprint | None, *, qualified: Iterable[str] = (),
                  historical_hints: Iterable[str] = ()) -> tuple[CandidateSpec, ...]:
        """Return eligible specs; hints only reorder already eligible specs."""

        eligible = [spec for spec in self._specs.values() if spec.accepts(fingerprint, qualified=qualified)]
        if not eligible:
            return (self._specs["baseline"],)
        baseline = self._specs["baseline"]
        rest = [spec for spec in eligible if spec.candidate_id != "baseline"]
        hint_order = {candidate_id: index for index, candidate_id in enumerate(historical_hints)}
        rest.sort(key=lambda spec: (hint_order.get(spec.candidate_id, len(hint_order)), CANDIDATE_IDS.index(spec.candidate_id)))
        return tuple([baseline] + rest)

    def ordered_ids(self, fingerprint: ExactFingerprint | None, *, qualified: Iterable[str] = (),
                    historical_hints: Iterable[str] = ()) -> tuple[str, ...]:
        return tuple(spec.candidate_id for spec in self.available(fingerprint, qualified=qualified, historical_hints=historical_hints))


__all__ = ["CANDIDATE_IDS", "CandidateError", "CandidateSpec", "CandidateRegistry"]
