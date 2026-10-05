"""Metadata-only tests for reuse of the existing effective profile catalog."""

import copy

import pytest

from ironmule_product.engine_bridge import (
    _historical_configuration_metadata, serving_profile_contract,
)


def reference():
    knobs = _historical_configuration_metadata("baseline_interactive")["knobs"]
    knobs["k3840_matvec"] = False
    return knobs


def test_catalog_projects_existing_templates_preserving_every_other_knob():
    knobs = reference()
    knobs.update(fused_argmax=True, readback_every=8, capacity_slack=128)
    before = copy.deepcopy(knobs)
    contract = serving_profile_contract(knobs)
    assert knobs == before
    assert contract["profiles"] == ["current.sequential.v1", "current.grouped4.v1",
                                    "core.sequential.v1", "core.grouped4.v1"]
    for profile in contract["definitions"]:
        for key, value in before.items():
            if key not in ("compiled_fixed_cache", "head_skip_prefill"):
                assert profile["knobs"][key] == value
    contract["definitions"][0]["knobs"]["readback_every"] = 1
    assert knobs == before
    assert contract["definitions"][1]["knobs"]["readback_every"] == 8


def test_catalog_deduplicates_caller_core_and_uses_original_baseline_template():
    knobs = reference()
    knobs.update(compiled_fixed_cache=True, head_skip_prefill=True)
    contract = serving_profile_contract(knobs)
    assert contract["profiles"] == ["current.sequential.v1", "current.grouped4.v1",
                                    "baseline.sequential.v1", "baseline.grouped4.v1"]
    assert len({(profile["mode"], tuple(sorted(profile["knobs"].items())))
                for profile in contract["definitions"]}) == 4


def test_catalog_omits_unsupported_grouping_and_caps_mixed_caller_options():
    knobs = reference()
    assert len(serving_profile_contract(knobs, grouping_supported=False)["profiles"]) == 2
    knobs["compiled_fixed_cache"] = True
    assert len(serving_profile_contract(knobs)["profiles"]) == 4
    sequential = serving_profile_contract(knobs, grouping_supported=False)
    assert len(sequential["profiles"]) == 3
    assert all(profile["mode"] == "interactive" for profile in sequential["definitions"])


@pytest.mark.parametrize("mutation", ["missing", "bool", "grouping"])
def test_catalog_rejects_noncanonical_toggle_metadata(mutation):
    knobs = reference()
    if mutation == "missing":
        del knobs["head_skip_prefill"]
    elif mutation == "bool":
        knobs["head_skip_prefill"] = 1
    with pytest.raises(ValueError):
        serving_profile_contract(knobs, grouping_supported="yes" if mutation == "grouping" else True)


def test_speculative_catalog_adds_one_sequential_speculation_profile():
    knobs = reference()
    knobs.update(compiled_fixed_cache=True, head_skip_prefill=True, readback_every=4)
    contract = serving_profile_contract(knobs, speculative=True)
    assert contract["schema"] == "ironmule.execution_profiles.v3"
    assert contract["profiles"] == ["current.sequential.v1", "current.grouped4.v1",
                                    "speculative.sequential.v1", "baseline.sequential.v1"]
    speculative = contract["definitions"][2]
    assert speculative["mode"] == "interactive" and speculative["knobs"] == dict(knobs, speculate_k=4)
    assert serving_profile_contract(knobs)["schema"] == "ironmule.execution_profiles.v2"
    knobs["speculate_k"] = 4  # already speculating: nothing to add
    assert "speculative.sequential.v1" not in serving_profile_contract(knobs, speculative=True)["profiles"]
