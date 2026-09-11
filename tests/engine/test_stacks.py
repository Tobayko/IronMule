"""The stack model: what may be composed, and why each refusal is a fact not a taste."""

from __future__ import annotations

import pytest

from ironmule.stacks import (ACTIVATION, CANDIDATES, OPTIMISATIONS, PHASES,
                             STACK_MODEL_VERSION, Context, Optimisation, Stack,
                             internal_conflict, reduce)


def full_context(**overrides):
    base = dict(hardware_fingerprint="hw", model_identity_sha256="id",
                architecture="gemma3", mlx="0.32.0", mlx_lm="0.31.3",
                profile_present=True, kernel_admits=True, paired_admits=True)
    base.update(overrides)
    return Context(**base)


class TestTableIsWellFormed:
    def test_every_optimisation_declares_a_known_phase_and_objective(self):
        for row in OPTIMISATIONS.values():
            assert row.phase in PHASES and row.activation in ACTIVATION

    def test_every_optimisation_names_its_evidence(self):
        for row in OPTIMISATIONS.values():
            assert row.evidence_ids, f"{row.id} has no evidence"

    def test_an_optimisation_without_evidence_is_refused(self):
        with pytest.raises(ValueError, match="wish"):
            Optimisation(id="x", phase="decode", summary="", preconditions=(),
                         fingerprint_scope=(), objective="any", incompatible_with=(),
                         correctness_contract="c", evidence_ids=())

    def test_incompatibility_is_symmetric(self):
        for row in OPTIMISATIONS.values():
            for other in row.incompatible_with:
                assert row.id in OPTIMISATIONS[other].incompatible_with

    def test_every_candidate_names_known_members(self):
        for stack in CANDIDATES:
            assert all(name in OPTIMISATIONS for name in stack.members)

    def test_the_model_names_its_version(self):
        assert STACK_MODEL_VERSION.startswith("ironmule.stack_model.")


class TestExclusionsAreFacts:
    def test_the_kernel_and_the_paired_path_cannot_share_a_stack(self):
        # `ironmule/qmv_k3840.py` hands back anything but shape (1, 1, K), and a paired
        # step goes through the shared kernel instead. Not a preference.
        conflict = internal_conflict(next(s for s in CANDIDATES if s.id == "D"))
        assert conflict is not None and "k3840_matvec" in conflict

    def test_the_kernel_and_grouped_execution_cannot_share_a_stack(self):
        clash = internal_conflict(Stack("x", "", ("k3840_matvec", "throughput_mode"), "any"))
        assert clash is not None

    def test_interactive_and_throughput_cannot_share_a_stack(self):
        clash = internal_conflict(Stack("x", "", ("interactive_mode", "throughput_mode"), "any"))
        assert clash is not None

    def test_releasing_the_kernel_does_not_resolve_the_conflict(self):
        admitted, _ = reduce(CANDIDATES, full_context(released=("k3840_matvec",)))
        assert "D" not in {s.id for s in admitted}


class TestActivationHolds:
    def test_a_held_optimisation_never_composes_on_its_own(self):
        admitted, _ = reduce(CANDIDATES, full_context())
        assert "D_single" not in {s.id for s in admitted}

    def test_an_explicit_release_lets_it_compose(self):
        admitted, _ = reduce(CANDIDATES, full_context(released=("k3840_matvec",)))
        assert "D_single" in {s.id for s in admitted}

    def test_a_refused_optimisation_cannot_be_released_by_naming_it(self):
        refused = Stack("p", "", ("own_dispatch",), "latency")
        admitted, excluded = reduce([refused], full_context(released=("own_dispatch",)))
        assert not admitted and "refused" in excluded[0].detail

    def test_a_pending_optimisation_is_excluded_while_it_measures(self):
        from dataclasses import replace
        row = replace(OPTIMISATIONS["own_dispatch"], activation="pending")
        assert not Context(profile_present=True).allows(row)[0]


class TestFingerprintReduction:
    def test_no_profile_admits_nothing(self):
        admitted, excluded = reduce(CANDIDATES, Context())
        assert not admitted and len(excluded) == len(CANDIDATES)

    def test_a_model_that_does_not_admit_the_paired_path_loses_c(self):
        admitted, _ = reduce(CANDIDATES, full_context(paired_admits=False))
        ids = {s.id for s in admitted}
        assert "C" not in ids and {"A", "B", "B_t", "E"} <= ids

    def test_a_model_that_does_not_admit_the_kernel_loses_it_even_when_released(self):
        admitted, _ = reduce(CANDIDATES, full_context(kernel_admits=False,
                                                      released=("k3840_matvec",)))
        assert "D_single" not in {s.id for s in admitted}

    def test_every_exclusion_carries_a_reason(self):
        _, excluded = reduce(CANDIDATES, Context())
        assert all(row.reason and row.detail for row in excluded)

    def test_reduction_is_deterministic(self):
        context = full_context()
        first = [s.id for s in reduce(CANDIDATES, context)[0]]
        for _ in range(5):
            assert [s.id for s in reduce(CANDIDATES, context)[0]] == first


class TestReferencesAreDeclared:
    def test_every_non_reference_stack_names_what_it_must_beat(self):
        for stack in CANDIDATES:
            if stack.id in ("A", "A_t", "E"):
                continue
            assert stack.reference, f"{stack.id} must name its reference"

    def test_named_references_exist(self):
        ids = {stack.id for stack in CANDIDATES}
        for stack in CANDIDATES:
            assert stack.reference is None or stack.reference in ids


def test_module_self_check_runs():
    from ironmule.stacks import _self_check
    _self_check()
