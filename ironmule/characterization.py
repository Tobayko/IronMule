"""What a machine can say about itself before anyone has tuned anything on it.

`B69` confirmed one hardware parameter on this M1 Max, and `B70` made it readable without
acting on it. Both are bound to a fingerprint, which is exact and says nothing: a fingerprint
tells you whether you have seen a machine before, never what a machine like it will do. A new
Mac starts from nothing and has to earn its own profile from scratch.

This is the type that could change that. It holds what a machine measured about itself,
separated into what was read off the system, what was actually timed, and under which
conditions -- so that two machines can be compared on the quantities that make an
optimisation pay rather than on their model names.

**Missing stays missing.** Every measured field defaults to `None`. Nothing here imputes,
interpolates or substitutes a plausible number, and a probe that did not run leaves a hole
that is visible in the serialised form. `B70` already met the cost of the opposite habit: a
field can be structurally valid and factually wrong, and the only defence is refusing to
write one you did not measure.

**One machine proves nothing.** The hypothesis this type exists to test -- that a compact
vector describes optimisation response better than a chip name -- cannot be tested where
there is one chip. On this machine the honest ceiling is that the vector is implemented,
self-consistent and traceable. What a second Mac would have to show is written into the
preregistration and not into this docstring.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

SCHEMA = "ironmule.hardware_characterization_vector.v1"

#: A measured quantity keeps its own provenance. A number without one is a rumour.
@dataclass(frozen=True)
class Measurement:
    value: float
    unit: str
    evidence_id: str
    #: Spread of the samples the value was taken from, in the same unit. `None` means the
    #: probe reported a single figure and its spread is genuinely unknown.
    spread: float | None = None
    samples: int | None = None
    #: Anything a later reader needs to know before comparing this to another machine's.
    context: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"value": self.value, "unit": self.unit, "evidence_id": self.evidence_id,
                "spread": self.spread, "samples": self.samples, "context": dict(self.context)}


@dataclass(frozen=True)
class StaticFacts:
    """Read off the system. Cheap, exact, and on its own not predictive of anything."""

    hardware_fingerprint: str = ""
    chip: str = ""
    gpu_architecture: str = ""
    gpu_cores: int | None = None
    cpu_logical: int | None = None
    unified_memory_bytes: int | None = None
    os_release: str = ""
    mlx: str = ""
    mlx_lm: str = ""


@dataclass(frozen=True)
class MeasuredResponses:
    """What the machine did when asked. Every field is a `Measurement` or `None`."""

    #: Achieved bytes per nanosecond on the largest working set the probe used.
    dram_bandwidth: Measurement | None = None
    #: Achieved bandwidth against matrix size, smallest first. Empty means not probed.
    bandwidth_by_working_set: tuple[Measurement, ...] = ()
    #: A working set inside the last-level cache over one clearly outside it.
    cache_residency_ratio: Measurement | None = None
    #: Cost of the unaligned K against an aligned one at the same width.
    k_alignment_ratio: Measurement | None = None
    #: Cost per row at each grouped width, keyed in `context` by the width.
    width_response: tuple[Measurement, ...] = ()
    #: Fixed nanoseconds of one submission boundary, and the marginal cost of a kernel in it.
    eval_fixed_cost: Measurement | None = None
    eval_marginal_cost: Measurement | None = None
    #: A candidate SIMD geometry over the library call on the same buffers.
    geometry_response: tuple[Measurement, ...] = ()


@dataclass(frozen=True)
class Conditions:
    """Whether the machine was in a state where its answers mean anything."""

    measured_at: str = ""
    probe_wall_seconds: float | None = None
    load_average_at_start: tuple[float, ...] = ()
    load_average_at_end: tuple[float, ...] = ()
    memory_pressure_normal_throughout: bool | None = None
    swap_grew: bool | None = None
    min_memory_free_percent: float | None = None
    resource_gate_passed: bool | None = None
    resource_gate_reasons: tuple[str, ...] = ()
    code_binding_sha256: str = ""
    probe_set_id: str = ""
    correctness_checked: bool | None = None
    notes: str = ""


@dataclass(frozen=True)
class HardwareCharacterizationVector:
    """One machine, described by what it did rather than by what it is called."""

    schema: str
    static: StaticFacts
    measured: MeasuredResponses
    conditions: Conditions
    #: Dimensionless relations, so two machines of different absolute speed compare. Each
    #: is derived from measurements in this same vector and names them.
    relations: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        def convert(value):
            if isinstance(value, Measurement):
                return value.as_dict()
            if isinstance(value, tuple):
                return [convert(item) for item in value]
            if isinstance(value, Mapping):
                return {k: convert(v) for k, v in value.items()}
            return value
        return {"schema": self.schema,
                "static": asdict(self.static),
                "measured": {name: convert(getattr(self.measured, name))
                             for name in MEASURED_FIELDS},
                "conditions": {**asdict(self.conditions),
                               "load_average_at_start": list(self.conditions.load_average_at_start),
                               "load_average_at_end": list(self.conditions.load_average_at_end),
                               "resource_gate_reasons": list(self.conditions.resource_gate_reasons)},
                "relations": convert(dict(self.relations)),
                "missing": list(self.missing())}

    def missing(self) -> tuple[str, ...]:
        """Which measured fields were not filled. Visible on purpose."""
        out = []
        for name in MEASURED_FIELDS:
            value = getattr(self.measured, name)
            if value is None or value == ():
                out.append(name)
        return tuple(out)

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True, allow_nan=False)


MEASURED_FIELDS = ("dram_bandwidth", "bandwidth_by_working_set", "cache_residency_ratio",
                   "k_alignment_ratio", "width_response", "eval_fixed_cost",
                   "eval_marginal_cost", "geometry_response")
STATIC_FIELDS = tuple(StaticFacts.__dataclass_fields__)
CONDITION_FIELDS = tuple(Conditions.__dataclass_fields__)
MEASUREMENT_FIELDS = frozenset({"value", "unit", "evidence_id", "spread", "samples", "context"})


class CharacterizationError(ValueError):
    """A vector that cannot be read back exactly as it was written."""


def _measurement(raw: Any, where: str) -> Measurement:
    if not isinstance(raw, Mapping):
        raise CharacterizationError(f"{where} is not a mapping")
    found = set(raw)
    if found != MEASUREMENT_FIELDS:
        raise CharacterizationError(
            f"{where} has {sorted(found ^ MEASUREMENT_FIELDS)} out of place")
    for name in ("value",):
        if isinstance(raw[name], bool) or not isinstance(raw[name], (int, float)):
            raise CharacterizationError(f"{where}.{name} must be a number")
    for name in ("unit", "evidence_id"):
        if not isinstance(raw[name], str) or not raw[name]:
            raise CharacterizationError(f"{where}.{name} must be a non-empty string")
    if not isinstance(raw["context"], Mapping):
        raise CharacterizationError(f"{where}.context must be a mapping")
    return Measurement(value=float(raw["value"]), unit=raw["unit"],
                       evidence_id=raw["evidence_id"], spread=raw["spread"],
                       samples=raw["samples"], context=dict(raw["context"]))


def load(raw: Any) -> HardwareCharacterizationVector:
    """Read a vector back, or raise. Round-tripping is the only way it is trusted."""
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, Mapping):
        raise CharacterizationError("the vector is not a mapping")
    expected = {"schema", "static", "measured", "conditions", "relations", "missing"}
    if set(raw) != expected:
        raise CharacterizationError(f"top level has {sorted(set(raw) ^ expected)} out of place")
    if raw["schema"] != SCHEMA:
        raise CharacterizationError(f"schema must be {SCHEMA!r}")
    for name, fields in (("static", STATIC_FIELDS), ("conditions", CONDITION_FIELDS)):
        section = raw[name]
        if not isinstance(section, Mapping) or set(section) != set(fields):
            raise CharacterizationError(f"{name} has fields out of place")
    measured = raw["measured"]
    if not isinstance(measured, Mapping) or set(measured) != set(MEASURED_FIELDS):
        raise CharacterizationError("measured has fields out of place")
    values: dict[str, Any] = {}
    for name in MEASURED_FIELDS:
        entry = measured[name]
        if entry is None:
            values[name] = None
        elif isinstance(entry, list):
            values[name] = tuple(_measurement(item, f"measured.{name}") for item in entry)
        else:
            values[name] = _measurement(entry, f"measured.{name}")
    static = StaticFacts(**raw["static"])
    conditions = Conditions(**{
        **raw["conditions"],
        "load_average_at_start": tuple(raw["conditions"]["load_average_at_start"]),
        "load_average_at_end": tuple(raw["conditions"]["load_average_at_end"]),
        "resource_gate_reasons": tuple(raw["conditions"]["resource_gate_reasons"])})
    vector = HardwareCharacterizationVector(
        schema=raw["schema"], static=static, measured=MeasuredResponses(**values),
        conditions=conditions, relations=dict(raw["relations"]))
    if list(vector.missing()) != list(raw["missing"]):
        raise CharacterizationError("the missing list does not agree with the vector")
    return vector


def relations_for(measured: MeasuredResponses) -> dict[str, Any]:
    """The dimensionless relations, each naming what it came from.

    No relation is invented where its inputs are missing, and none of them is combined into
    a single score: a machine that is fast at one thing and slow at another is exactly the
    case a single number destroys.
    """
    out: dict[str, Any] = {}

    def relation(name, value, sources, note):
        out[name] = {"value": value, "from": list(sources), "note": note}

    if measured.cache_residency_ratio is not None:
        relation("cache_to_dram_ratio", measured.cache_residency_ratio.value,
                 [measured.cache_residency_ratio.evidence_id],
                 "a working set inside the last-level cache over one clearly outside it, "
                 "with dispatch and eval count held equal. Below 1.0 means the cache helps")
    if measured.k_alignment_ratio is not None:
        relation("k_unaligned_to_aligned_ratio", measured.k_alignment_ratio.value,
                 [measured.k_alignment_ratio.evidence_id],
                 "cost of the model's own K against an aligned K at the same output width "
                 "and the same weight bytes per row. Above 1.0 means the shape costs")
    widths = {int(m.context.get("width", 0)): m for m in measured.width_response}
    if 1 in widths and 16 in widths and widths[1].value:
        relation("m16_cost_per_row_vs_m1", widths[16].value / widths[1].value,
                 [widths[16].evidence_id],
                 "cost per row at width 16 over width 1. Well below 1.0 means grouping pays")
    if 4 in widths and 8 in widths and widths[4].value:
        relation("m8_cost_per_row_vs_m4", widths[8].value / widths[4].value,
                 [widths[8].evidence_id],
                 "cost per row at width 8 over width 4. Near 1.0 means the widths between "
                 "the two tile boundaries buy nothing")
    for entry in measured.geometry_response:
        name = entry.context.get("geometry")
        if name:
            relation(f"geometry_{name}_ratio", entry.value, [entry.evidence_id],
                     "a candidate SIMD geometry over the library call on the same buffers, "
                     "byte identical. Below 1.0 means the geometry is worth measuring "
                     "against a stack, and nothing more")
    if (measured.eval_fixed_cost is not None and measured.eval_marginal_cost is not None
            and measured.eval_marginal_cost.value):
        relation("eval_fixed_over_one_kernel",
                 measured.eval_fixed_cost.value / measured.eval_marginal_cost.value,
                 [measured.eval_fixed_cost.evidence_id, measured.eval_marginal_cost.evidence_id],
                 "the fixed cost of a submission boundary in units of one kernel in it. "
                 "Large means the machine rewards putting more work between boundaries")
    return out


def _self_check() -> None:
    empty = HardwareCharacterizationVector(SCHEMA, StaticFacts(), MeasuredResponses(),
                                           Conditions())
    assert len(empty.missing()) == len(MEASURED_FIELDS)
    assert load(empty.to_json()).missing() == empty.missing()
    assert relations_for(MeasuredResponses()) == {}

    measured = MeasuredResponses(
        cache_residency_ratio=Measurement(0.95, "ratio", "probe", spread=0.01, samples=12),
        width_response=(Measurement(1.0, "ns_per_row", "probe", context={"width": 1}),
                        Measurement(0.48, "ns_per_row", "probe", context={"width": 16})),
        eval_fixed_cost=Measurement(250000.0, "ns", "probe"),
        eval_marginal_cost=Measurement(112000.0, "ns", "probe"))
    relations = relations_for(measured)
    assert relations["cache_to_dram_ratio"]["value"] == 0.95
    assert abs(relations["m16_cost_per_row_vs_m1"]["value"] - 0.48) < 1e-9
    assert "m8_cost_per_row_vs_m4" not in relations, "a relation needs its inputs"

    vector = HardwareCharacterizationVector(SCHEMA, StaticFacts(chip="Apple M1 Max"),
                                            measured, Conditions(), relations)
    assert load(vector.to_json()).relations == relations
    assert "dram_bandwidth" in load(vector.to_json()).missing()

    broken = json.loads(vector.to_json())
    broken["missing"] = []
    try:
        load(broken)
    except CharacterizationError:
        pass
    else:
        raise AssertionError("a missing list that disagrees must not load")
    print("characterization vector self-check ok")


if __name__ == "__main__":
    _self_check()
