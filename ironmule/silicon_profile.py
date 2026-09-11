"""Hardware parameters this machine has actually earned, read without acting on them.

`B69` confirmed one: the `(4, 8)` threadgroup geometry for the `K = 3840` quantised matvec,
against the confirmed complete stack of three `12B` workload classes. It is bound to a
fingerprint, a GPU generation, a library build, a model identity and revision, a shape, a
workload class and the code digest of the run that measured it. None of that is activated,
and nothing in this module activates anything.

**Shadow only.** The router may load a profile, match it and report what it found. It may
not route differently because of it. `ExecutionRouter.decide` builds its route from the same
facts it always did, and the match travels beside the decision as diagnosis. That separation
is the point: a parameter has to be visible before anyone can argue about switching it on,
and it must be possible to be wrong about the matching without being wrong about the answer.

**Fail closed, like `load_profile`.** An unknown field, a missing field, a wrong schema, a
digest that does not recompute, or any mismatch in hardware, library, model or shape yields
no match at all. There are no silent defaults: a profile that cannot be fully understood is
not partially believed.

**Nothing here touches the hot path.** Matching is a pure function of facts the caller
already has. It starts no process, hashes no model, reads no file, calls no GPU and measures
nothing. `B55` is the standard it is held to: two subprocess probes once cost `17.4 ms` of a
`21 ms` wrapper and made a routed request `3.7%` slower than naming its mode by hand.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "ironmule.silicon_profile.v1"
PROFILE_FIELDS = frozenset({"schema", "generated_from", "parameters", "digest"})
PARAMETER_FIELDS = frozenset({
    "parameter_id", "kind", "value", "hardware_fingerprint", "gpu_architecture",
    "mlx", "mlx_lm", "model_id", "model_identity_sha256", "model_revision",
    "quantization", "shape", "workload_class", "objective", "reference_stack",
    "effect", "evidence_status", "evidence_id", "code_binding_sha256",
    "correctness_contract", "activation",
})
QUANTIZATION_FIELDS = frozenset({"bits", "group_size"})
SHAPE_FIELDS = frozenset({"k", "m", "admitted_n"})
EFFECT_FIELDS = frozenset({"median", "ci_low", "ci_high"})
CONFIRMED = "CONFIRMED"
#: A parameter that names anything else is loaded and reported, and never matched.
MATCHABLE_STATUSES = frozenset({CONFIRMED})


class SiliconProfileError(ValueError):
    """A profile that cannot be fully understood. It is never partially believed."""


@dataclass(frozen=True)
class SiliconParameter:
    """One measured parameter and every condition it was measured under."""

    parameter_id: str
    kind: str
    value: Mapping[str, Any]
    hardware_fingerprint: str
    gpu_architecture: str
    mlx: str
    mlx_lm: str
    model_id: str
    model_identity_sha256: str
    model_revision: str
    quantization: Mapping[str, int]
    shape: Mapping[str, Any]
    workload_class: str
    objective: str
    reference_stack: str
    effect: Mapping[str, float]
    evidence_status: str
    evidence_id: str
    code_binding_sha256: str
    correctness_contract: str
    activation: str


@dataclass(frozen=True)
class SiliconProfile:
    """A loaded profile. Immutable, digest-checked, and inert until someone reads it."""

    schema: str
    generated_from: str
    digest: str
    parameters: tuple[SiliconParameter, ...]


@dataclass(frozen=True)
class RuntimeContext:
    """Only facts the caller already holds. Nothing here is computed per request.

    `k` and the quantisation come from the loaded model and are settled once. `n` is the
    projection width a parameter names, so a context that does not know its shapes leaves
    `admitted_n` empty and matches nothing.
    """

    hardware_fingerprint: str = ""
    gpu_architecture: str = ""
    mlx: str = ""
    mlx_lm: str = ""
    model_id: str = ""
    model_identity_sha256: str = ""
    model_revision: str = ""
    quantization_bits: int = 0
    quantization_group_size: int = 0
    hidden_size: int = 0
    projection_widths: tuple[int, ...] = ()
    workload_class: str = ""
    objective: str = ""
    decode_width: int = 0

    def for_dispatch(self, workload_class: str, objective: str,
                     decode_width: int) -> "RuntimeContext":
        """The same machine and model, told what this dispatch is. Constructed, not copied.

        `dataclasses.replace` rebuilds through `__init__` with every field re-passed and was
        measurable here; this is the same result written out.
        """
        return RuntimeContext(
            self.hardware_fingerprint, self.gpu_architecture, self.mlx, self.mlx_lm,
            self.model_id, self.model_identity_sha256, self.model_revision,
            self.quantization_bits, self.quantization_group_size, self.hidden_size,
            self.projection_widths, workload_class, objective, decode_width)


@dataclass(frozen=True)
class ShadowMatch:
    """What the router found, and why. Diagnostic: nothing acts on it."""

    silicon_match: bool
    eligible: bool
    reason: str
    silicon_parameter_id: str | None = None
    kind: str | None = None
    candidate_value: Mapping[str, Any] | None = None
    evidence_id: str | None = None
    evidence_status: str | None = None
    activation: str = "none"

    def as_dict(self) -> dict[str, Any]:
        # Built by hand rather than with `asdict`, which deep-copies. `B70` measured the
        # difference: the first shadow implementation cost 78 per cent of `decide()`.
        return {"silicon_match": self.silicon_match, "eligible": self.eligible,
                "reason": self.reason, "silicon_parameter_id": self.silicon_parameter_id,
                "kind": self.kind, "candidate_value": self.candidate_value,
                "evidence_id": self.evidence_id, "evidence_status": self.evidence_status,
                "activation": self.activation,
                "shadow_only": True, "affects_dispatch": False}


NO_PROFILE = ShadowMatch(False, False, "no silicon profile is loaded")


# --------------------------------------------------------------------------- loading


def canonical_digest(parameters: Sequence[Mapping[str, Any]]) -> str:
    """The digest a profile has to carry, over its parameters and nothing else."""
    payload = json.dumps(list(parameters), sort_keys=True, separators=(",", ":"),
                         allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _exact_keys(value: Any, expected: frozenset, what: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SiliconProfileError(f"{what} is not a mapping")
    found = set(value)
    missing = expected - found
    unknown = found - expected
    if missing:
        raise SiliconProfileError(f"{what} is missing {sorted(missing)}")
    if unknown:
        raise SiliconProfileError(f"{what} carries unknown {sorted(unknown)}")
    return value


def _parameter(raw: Any) -> SiliconParameter:
    row = _exact_keys(raw, PARAMETER_FIELDS, "a parameter")
    for name in ("parameter_id", "kind", "hardware_fingerprint", "gpu_architecture",
                 "mlx", "mlx_lm", "model_id", "model_identity_sha256", "model_revision",
                 "workload_class", "objective", "reference_stack", "evidence_status",
                 "evidence_id", "code_binding_sha256", "correctness_contract", "activation"):
        if not isinstance(row[name], str) or not row[name]:
            raise SiliconProfileError(f"{name} must be a non-empty string")
    quantization = _exact_keys(row["quantization"], QUANTIZATION_FIELDS, "quantization")
    for name in QUANTIZATION_FIELDS:
        if isinstance(quantization[name], bool) or not isinstance(quantization[name], int):
            raise SiliconProfileError(f"quantization.{name} must be an integer")
    shape = _exact_keys(row["shape"], SHAPE_FIELDS, "shape")
    for name in ("k", "m"):
        if isinstance(shape[name], bool) or not isinstance(shape[name], int) or shape[name] < 1:
            raise SiliconProfileError(f"shape.{name} must be a positive integer")
    if (not isinstance(shape["admitted_n"], list) or not shape["admitted_n"]
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 1
                   for v in shape["admitted_n"])):
        raise SiliconProfileError("shape.admitted_n must be a non-empty list of positive integers")
    effect = _exact_keys(row["effect"], EFFECT_FIELDS, "effect")
    for name in EFFECT_FIELDS:
        if isinstance(effect[name], bool) or not isinstance(effect[name], (int, float)):
            raise SiliconProfileError(f"effect.{name} must be a number")
    if not isinstance(row["value"], Mapping) or not row["value"]:
        raise SiliconProfileError("value must be a non-empty mapping")
    if row["activation"] != "none":
        raise SiliconProfileError(
            "activation must be 'none': this schema records evidence, and a profile that "
            "claims to activate something is refused rather than obeyed")
    return SiliconParameter(**row)


def load(source: Any) -> SiliconProfile:
    """Read a profile, or raise. There is no partially valid profile.

    `source` is a path or an already-parsed mapping. Nothing is defaulted, nothing is
    coerced, and the digest must recompute over the parameters exactly as written.
    """
    if isinstance(source, (str, Path)):
        try:
            raw = json.loads(Path(source).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SiliconProfileError(f"profile unreadable: {type(exc).__name__}") from exc
    else:
        raw = source
    body = _exact_keys(raw, PROFILE_FIELDS, "the profile")
    if body["schema"] != SCHEMA:
        raise SiliconProfileError(f"schema must be {SCHEMA!r}, found {body['schema']!r}")
    if not isinstance(body["generated_from"], str) or not body["generated_from"]:
        raise SiliconProfileError("generated_from must be a non-empty string")
    if not isinstance(body["parameters"], list):
        raise SiliconProfileError("parameters must be a list")
    digest = canonical_digest(body["parameters"])
    if body["digest"] != digest:
        raise SiliconProfileError("digest does not recompute over the parameters as written")
    parameters = tuple(_parameter(row) for row in body["parameters"])
    identifiers = [p.parameter_id for p in parameters]
    if len(set(identifiers)) != len(identifiers):
        raise SiliconProfileError("parameter_id must be unique within a profile")
    return SiliconProfile(schema=body["schema"], generated_from=body["generated_from"],
                          digest=digest, parameters=parameters)


def load_or_none(source: Any) -> SiliconProfile | None:
    """The router's way in. An unusable profile leaves the router exactly as it was."""
    try:
        return load(source)
    except SiliconProfileError:
        return None


# --------------------------------------------------------------------------- matching


#: How a dispatch is named as a workload class. Fixed here, deterministic, and derived only
#: from facts the router already has. A dispatch that fits none of these is unnamed and
#: matches nothing, which is the correct answer rather than a nearest guess.
WORKLOAD_CLASS_RULE = (
    "single_short: one request, no session plan, at most 32 new tokens. "
    "single_long: one request, no session plan, more than 32 new tokens. "
    "session_warm: more than one request, every one carrying a session plan that matches "
    "its own prompt. Anything else is unnamed"
)


def workload_class_for(*, requests: int, session_plan_matches: int, max_new_tokens: int,
                       plan_kinds: Sequence[str]) -> str:
    """Name the dispatch, or return "" and match nothing."""
    session = all(kind != "strict_one_shot" for kind in plan_kinds) if plan_kinds else False
    if requests == 1 and not session:
        return "single_short" if max_new_tokens <= 32 else "single_long"
    if requests > 1 and session and session_plan_matches == requests:
        return "session_warm"
    return ""


def match_silicon_parameter(context: RuntimeContext,
                            profile: SiliconProfile | None) -> ShadowMatch:
    """Which confirmed parameter, if any, this dispatch is inside. Pure, and inert.

    Every condition the parameter was measured under has to hold. The first failing one is
    named, because a match that cannot say why it failed is not diagnosis.
    """
    if profile is None:
        return NO_PROFILE
    if not context.hardware_fingerprint or not context.model_identity_sha256:
        return ShadowMatch(False, False,
                           "the runtime context does not identify this machine and model")
    reasons: list[str] = []
    widths = context.projection_widths
    for parameter in profile.parameters:
        # Ordered cheapest and most selective first, and short-circuited: on a machine that
        # does not match, this is one string comparison per parameter.
        failed = ""
        if parameter.workload_class != context.workload_class or not context.workload_class:
            failed = "workload class"
        elif parameter.hardware_fingerprint != context.hardware_fingerprint:
            failed = "hardware fingerprint"
        elif parameter.model_identity_sha256 != context.model_identity_sha256:
            failed = "model identity"
        elif parameter.model_revision != context.model_revision:
            failed = "model revision"
        elif parameter.model_id != context.model_id:
            failed = "model id"
        elif parameter.gpu_architecture != context.gpu_architecture:
            failed = "GPU architecture"
        elif parameter.mlx != context.mlx:
            failed = "mlx version"
        elif parameter.mlx_lm != context.mlx_lm:
            failed = "mlx_lm version"
        elif parameter.objective != context.objective:
            failed = "objective"
        elif parameter.evidence_status not in MATCHABLE_STATUSES:
            failed = "evidence status"
        elif parameter.quantization["bits"] != context.quantization_bits:
            failed = "quantisation bits"
        elif parameter.quantization["group_size"] != context.quantization_group_size:
            failed = "quantisation group size"
        elif parameter.shape["k"] != context.hidden_size:
            failed = "K"
        elif parameter.shape["m"] != context.decode_width:
            failed = "decode width"
        elif not widths or not all(n in widths for n in parameter.shape["admitted_n"]):
            failed = "N"
        if failed:
            reasons.append(f"{parameter.parameter_id}: {failed} does not match")
            continue
        return ShadowMatch(
            silicon_match=True, eligible=True,
            reason=("every condition this parameter was measured under holds for this "
                    "dispatch. Reported only: the route and the dispatch are unchanged"),
            silicon_parameter_id=parameter.parameter_id, kind=parameter.kind,
            candidate_value=dict(parameter.value), evidence_id=parameter.evidence_id,
            evidence_status=parameter.evidence_status, activation=parameter.activation)
    return ShadowMatch(False, False,
                       "; ".join(reasons) if reasons else "the profile carries no parameters")


# --------------------------------------------------------------------------- B71 contract


@dataclass(frozen=True)
class HardwareObservation:
    """The neutral shape a later characterisation would fill. Empty on purpose.

    `B71` is meant to build a characterisation vector out of observations like these. This
    is the data contract for that and nothing else: every field defaults to `None`, no value
    is invented here, and nothing in this module measures anything to fill one in. A field
    that has not been measured stays `None` rather than becoming a plausible number.
    """

    hardware_fingerprint: str = ""
    gpu_architecture: str = ""
    gpu_cores: int | None = None
    unified_memory_bytes: int | None = None
    #: Achieved bytes per second against working-set size, as `E4` and `B66` measured it.
    measured_bandwidth: tuple[tuple[int, float], ...] = ()
    #: How much a working set inside the system level cache is worth, as a ratio.
    cache_residency_ratio: float | None = None
    #: Cost per row against matrix shape and alignment, keyed by (K, N).
    shape_response: tuple[tuple[tuple[int, int], float], ...] = ()
    #: Fixed and marginal nanoseconds of one submission boundary.
    submission_fixed_ns: float | None = None
    submission_marginal_ns: float | None = None
    #: Cost per row against grouped width, keyed by width.
    width_response: tuple[tuple[int, float], ...] = ()
    #: Effect against kernel geometry, keyed by (num_simdgroups, results_per_simdgroup).
    geometry_response: tuple[tuple[tuple[int, int], float], ...] = ()
    #: Which run produced each of the above. An observation without provenance is a rumour.
    evidence_ids: tuple[str, ...] = ()

    def measured_fields(self) -> tuple[str, ...]:
        """What has actually been filled in. Everything else is unmeasured, not zero."""
        return tuple(name for name, value in asdict(self).items()
                     if value not in (None, "", ()))


def _self_check() -> None:
    parameters = [{
        "parameter_id": "demo", "kind": "kernel_geometry",
        "value": {"num_simdgroups": 4, "results_per_simdgroup": 8},
        "hardware_fingerprint": "fp", "gpu_architecture": "applegpu_g13s",
        "mlx": "0.32.0", "mlx_lm": "0.31.3", "model_id": "m",
        "model_identity_sha256": "id", "model_revision": "rev",
        "quantization": {"bits": 4, "group_size": 64},
        "shape": {"k": 3840, "m": 1, "admitted_n": [15360]},
        "workload_class": "single_short", "objective": "latency", "reference_stack": "A",
        "effect": {"median": 0.85, "ci_low": 0.75, "ci_high": 0.95},
        "evidence_status": "CONFIRMED", "evidence_id": "B69", "code_binding_sha256": "c",
        "correctness_contract": "byte identical", "activation": "none",
    }]
    body = {"schema": SCHEMA, "generated_from": "self check", "parameters": parameters,
            "digest": canonical_digest(parameters)}
    profile = load(body)
    assert len(profile.parameters) == 1

    context = RuntimeContext(
        hardware_fingerprint="fp", gpu_architecture="applegpu_g13s", mlx="0.32.0",
        mlx_lm="0.31.3", model_id="m", model_identity_sha256="id", model_revision="rev",
        quantization_bits=4, quantization_group_size=64, hidden_size=3840,
        projection_widths=(15360, 2048), workload_class="single_short",
        objective="latency", decode_width=1)
    assert match_silicon_parameter(context, profile).silicon_match

    from dataclasses import replace
    for changed in (replace(context, hardware_fingerprint="other"),
                    replace(context, mlx="0.33.0"),
                    replace(context, model_revision="other"),
                    replace(context, hidden_size=2560),
                    replace(context, workload_class="pair_short"),
                    replace(context, decode_width=4)):
        assert not match_silicon_parameter(changed, profile).silicon_match

    broken = dict(body, digest="0" * 64)
    assert load_or_none(broken) is None
    assert load_or_none(dict(body, extra=1)) is None
    assert match_silicon_parameter(context, None) is NO_PROFILE

    assert workload_class_for(requests=1, session_plan_matches=0, max_new_tokens=32,
                              plan_kinds=("strict_one_shot",)) == "single_short"
    assert workload_class_for(requests=1, session_plan_matches=0, max_new_tokens=128,
                              plan_kinds=("strict_one_shot",)) == "single_long"
    assert workload_class_for(requests=3, session_plan_matches=3, max_new_tokens=32,
                              plan_kinds=("reusable_session",) * 3) == "session_warm"
    assert workload_class_for(requests=2, session_plan_matches=0, max_new_tokens=32,
                              plan_kinds=("strict_one_shot",) * 2) == ""

    assert HardwareObservation().measured_fields() == ()
    print("silicon profile self-check ok")


if __name__ == "__main__":
    _self_check()
