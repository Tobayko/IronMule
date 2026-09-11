"""A silicon profile may be read and reported. It may never change a route.

`B70`. Every negative case here has to leave the router's answer byte for byte what it was
without a profile, because the whole claim of shadow mode is that being wrong about the
matching cannot make the answer wrong.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from ironmule import silicon_profile as sp
from ironmule.plans import ReusableSessionPlan, StrictOneShotPlan
from ironmule.router import ExecutionRouter
from ironmule.service import Request

ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = ROOT / "research" / "raw" / "silicon_profile_v1_20260910_corrected.json"


def _body():
    return json.loads(PROFILE_PATH.read_text())


def _profile():
    return sp.load(_body())


def _context(**overrides):
    row = _body()["parameters"][0]
    base = sp.RuntimeContext(
        hardware_fingerprint=row["hardware_fingerprint"],
        gpu_architecture=row["gpu_architecture"], mlx=row["mlx"], mlx_lm=row["mlx_lm"],
        model_id=row["model_id"], model_identity_sha256=row["model_identity_sha256"],
        model_revision=row["model_revision"],
        quantization_bits=row["quantization"]["bits"],
        quantization_group_size=row["quantization"]["group_size"],
        hidden_size=row["shape"]["k"], projection_widths=tuple(row["shape"]["admitted_n"]),
        workload_class="single_short", objective="latency", decode_width=1)
    return replace(base, **overrides)


# -- the profile B69 produced ------------------------------------------------


def test_the_b69_parameter_matches_its_own_conditions():
    match = sp.match_silicon_parameter(_context(), _profile())
    assert match.silicon_match and match.eligible
    assert match.silicon_parameter_id == "k3840_geometry_4_8@single_short"
    assert match.candidate_value == {"num_simdgroups": 4, "results_per_simdgroup": 8}
    assert match.evidence_id == "B69_stack_proof_20260910"
    assert match.evidence_status == "CONFIRMED"
    assert match.activation == "none"
    assert match.as_dict()["affects_dispatch"] is False


@pytest.mark.parametrize("field, value", [
    ("hardware_fingerprint", "0000000000000000"),
    ("gpu_architecture", "applegpu_g14s"),
    ("mlx", "0.33.0"),
    ("mlx_lm", "0.32.0"),
    ("model_identity_sha256", "f" * 64),
    ("model_revision", "0" * 40),
    ("quantization_bits", 8),
    ("quantization_group_size", 32),
    ("hidden_size", 2560),
    ("decode_width", 4),
    ("workload_class", "pair_short"),
    ("workload_class", ""),
    ("objective", "throughput"),
])
def test_every_mismatched_condition_refuses(field, value):
    match = sp.match_silicon_parameter(_context(**{field: value}), _profile())
    assert not match.silicon_match and not match.eligible
    assert match.reason


def test_a_missing_projection_width_refuses():
    """The parameter names four widths; a model that lacks one of them is a different model."""
    assert not sp.match_silicon_parameter(
        _context(projection_widths=(15360, 2048)), _profile()).silicon_match


def test_an_unqualified_evidence_status_never_matches():
    body = _body()
    for row in body["parameters"]:
        row["evidence_status"] = "EXPLORATORY_GAIN_UNDER_LOAD"
    body["digest"] = sp.canonical_digest(body["parameters"])
    assert not sp.match_silicon_parameter(_context(), sp.load(body)).silicon_match


def test_only_the_class_that_was_confirmed_matches_that_class():
    """Three parameters share every condition but the class. Each answers only its own."""
    profile = _profile()
    for name in ("single_short", "single_long", "session_warm"):
        match = sp.match_silicon_parameter(_context(workload_class=name), profile)
        assert match.silicon_match
        assert match.silicon_parameter_id == f"k3840_geometry_4_8@{name}"


# -- the loader refuses rather than guesses ----------------------------------


def test_a_missing_field_is_refused():
    body = _body()
    del body["parameters"][0]["effect"]
    body["digest"] = sp.canonical_digest(body["parameters"])
    with pytest.raises(sp.SiliconProfileError, match="missing"):
        sp.load(body)


def test_an_unknown_field_is_refused():
    body = _body()
    body["parameters"][0]["speedup_hint"] = 1.5
    body["digest"] = sp.canonical_digest(body["parameters"])
    with pytest.raises(sp.SiliconProfileError, match="unknown"):
        sp.load(body)


def test_an_unknown_top_level_field_is_refused():
    with pytest.raises(sp.SiliconProfileError, match="unknown"):
        sp.load(dict(_body(), notes="harmless"))


def test_a_corrupt_digest_is_refused():
    with pytest.raises(sp.SiliconProfileError, match="digest"):
        sp.load(dict(_body(), digest="0" * 64))


def test_a_tampered_parameter_breaks_the_digest():
    """Editing a value without re-deriving the digest must not be readable."""
    body = _body()
    body["parameters"][0]["value"]["num_simdgroups"] = 8
    with pytest.raises(sp.SiliconProfileError, match="digest"):
        sp.load(body)


def test_a_wrong_schema_is_refused():
    body = _body()
    with pytest.raises(sp.SiliconProfileError, match="schema"):
        sp.load(dict(body, schema="ironmule.silicon_profile.v2"))


def test_a_profile_claiming_to_activate_is_refused():
    body = _body()
    body["parameters"][0]["activation"] = "enable"
    body["digest"] = sp.canonical_digest(body["parameters"])
    with pytest.raises(sp.SiliconProfileError, match="activation"):
        sp.load(body)


def test_an_empty_profile_loads_and_matches_nothing():
    body = {"schema": sp.SCHEMA, "generated_from": "empty", "parameters": [],
            "digest": sp.canonical_digest([])}
    profile = sp.load(body)
    assert profile.parameters == ()
    assert not sp.match_silicon_parameter(_context(), profile).silicon_match


def test_an_unreadable_profile_yields_none_rather_than_raising(tmp_path):
    assert sp.load_or_none(tmp_path / "absent.json") is None
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    assert sp.load_or_none(broken) is None
    assert sp.load_or_none(dict(_body(), digest="bad")) is None


def test_no_profile_at_all():
    assert sp.match_silicon_parameter(_context(), None) is sp.NO_PROFILE


# -- the router's answer is the same with and without a profile --------------


def _router(**extra):
    return ExecutionRouter({"knobs": {}}, identity_sha256="id", fingerprint="fp",
                           mlx="0.32.0", mlx_lm="0.31.3", **extra)


def _requests(count, session=False, max_tokens=32):
    prefix = [1, 2, 3]
    out = []
    for index in range(count):
        plan = ReusableSessionPlan(prefix, name="s") if session else StrictOneShotPlan()
        ids = prefix + [10 + index] if session else [10 + index] * 8
        out.append(Request(prompt_ids=ids, max_tokens=max_tokens, plan=plan,
                           objective="latency"))
    return out


def _route_only(decision):
    """Everything the caller acts on. `silicon` and `workload_class` are not in it."""
    record = decision.as_dict()
    record.pop("silicon", None)
    record.pop("workload_class", None)
    return record


@pytest.mark.parametrize("count, session", [(1, False), (2, False), (4, False), (3, True)])
def test_the_decision_is_identical_with_and_without_a_profile(count, session):
    requests = _requests(count, session)
    router = _router(silicon_profile=_profile(), silicon_context=_context())
    plain = _router().decide(requests, "latency")
    shadowed = router.decide(requests, "latency")
    # Identical without any exclusion: `decide()` does not compute the diagnostic at all.
    assert plain == shadowed
    # And still identical once annotated, apart from the two diagnostic fields.
    assert _route_only(plain) == _route_only(router.annotate(shadowed))


@pytest.mark.parametrize("profile_source", ["invalid", "empty", "mismatched"])
def test_a_useless_profile_leaves_the_router_exactly_as_it_was(profile_source):
    if profile_source == "invalid":
        profile = sp.load_or_none({"schema": "wrong"})
    elif profile_source == "empty":
        profile = sp.load({"schema": sp.SCHEMA, "generated_from": "e", "parameters": [],
                           "digest": sp.canonical_digest([])})
    else:
        profile = _profile()
    context = _context(hardware_fingerprint="somewhere else")
    requests = _requests(2)
    router = _router(silicon_profile=profile, silicon_context=context)
    plain = _router().decide(requests, "throughput")
    shadowed = router.decide(requests, "throughput")
    assert _route_only(plain) == _route_only(shadowed)
    annotated = router.annotate(shadowed)
    assert _route_only(plain) == _route_only(annotated)
    if profile is not None:
        assert annotated.silicon is None or annotated.silicon["silicon_match"] is False


def test_the_shadow_field_is_reported_when_it_matches():
    router = ExecutionRouter({"knobs": {}}, identity_sha256="id", fingerprint="fp",
                             mlx="0.32.0", mlx_lm="0.31.3",
                             silicon_profile=_profile(), silicon_context=_context())
    raw = router.decide(_requests(1), "latency")
    # `decide()` is byte for byte what it was: the diagnostic is not in it.
    assert raw.silicon is None and raw.workload_class == ""
    decision = router.annotate(raw)
    assert decision.workload_class == "single_short"
    assert decision.silicon["silicon_match"] is True
    assert decision.silicon["affects_dispatch"] is False
    assert decision.route == raw.route == "interactive"
    assert decision.reason == raw.reason


def test_the_paired_class_can_never_match():
    """C is the confirmed stack there and the source keeps the kernel apart from it."""
    router = ExecutionRouter({"knobs": {}}, identity_sha256="id", fingerprint="fp",
                             mlx="0.32.0", mlx_lm="0.31.3",
                             silicon_profile=_profile(), silicon_context=_context())
    decision = router.annotate(router.decide(_requests(2), "throughput"))
    assert decision.route == "throughput"
    assert decision.workload_class == ""
    assert decision.silicon["silicon_match"] is False


# -- the workload class naming rule ------------------------------------------


@pytest.mark.parametrize("requests, matches, tokens, kinds, expected", [
    (1, 0, 32, ("strict_one_shot",), "single_short"),
    (1, 0, 33, ("strict_one_shot",), "single_long"),
    (1, 0, 128, ("strict_one_shot",), "single_long"),
    (3, 3, 32, ("reusable_session",) * 3, "session_warm"),
    (3, 2, 32, ("reusable_session",) * 3, ""),
    (2, 0, 32, ("strict_one_shot",) * 2, ""),
    (0, 0, 0, (), ""),
])
def test_the_class_rule_is_deterministic(requests, matches, tokens, kinds, expected):
    assert sp.workload_class_for(requests=requests, session_plan_matches=matches,
                                 max_new_tokens=tokens, plan_kinds=kinds) == expected


# -- the B71 contract is empty on purpose ------------------------------------


def test_a_hardware_observation_invents_nothing():
    observation = sp.HardwareObservation()
    assert observation.measured_fields() == ()
    assert observation.submission_fixed_ns is None
    assert observation.width_response == ()
    assert observation.evidence_ids == ()
