"""What each qualified optimisation is, and which of them can stand in one stack.

`B55` and `B56` gave the runtime a router that picks between service modes. This is the
table that router and the composition study read: one record per optimisation, saying
which phase it acts in, what it needs, how far its evidence reaches, what it may stand
beside, and what it must keep bit-exact. Nothing here runs a model, stores a profile or
selects anything on its own -- it is the input to a decision, never the decision.

Two rules make the table worth having.

**A combination is its own candidate.** Prefix reuse, the tuned knob set and the paired
path were each measured against their own reference. Their percentages describe different
denominators and adding them is meaningless: `B55` already measured a case where a wrapper
that added nothing to the work still moved the number by `3.7%`. A stack counts only when
it has been executed and measured as one thing.

**An exclusion is a fact about the code, not a preference.** Two optimisations are marked
incompatible here only where the source makes them so, and the record says where.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

STACK_MODEL_VERSION = "ironmule.stack_model.v1"

PHASES = ("prefill", "decode", "scheduling", "memory")
OBJECTIVES = ("latency", "throughput", "any")
#: `available` may be composed now. `held` is qualified but deliberately not activated,
#: and needs an explicit release decision naming this state. `pending` is still measuring.
#: `refused` was measured and did not earn a place; the project rule "rejected is not
#: forbidden" lets it be reopened, but only with a new mechanism or new hardware evidence.
ACTIVATION = ("available", "held", "pending", "refused")


@dataclass(frozen=True)
class Optimisation:
    id: str
    phase: str
    summary: str
    preconditions: tuple[str, ...]
    fingerprint_scope: tuple[str, ...]
    objective: str
    incompatible_with: tuple[str, ...]
    correctness_contract: str
    evidence_ids: tuple[str, ...]
    activation: str = "available"
    activation_note: str = ""

    def __post_init__(self):
        if self.phase not in PHASES:
            raise ValueError(f"{self.id}: unknown phase {self.phase!r}")
        if self.objective not in OBJECTIVES:
            raise ValueError(f"{self.id}: unknown objective {self.objective!r}")
        if self.activation not in ACTIVATION:
            raise ValueError(f"{self.id}: unknown activation {self.activation!r}")
        if not self.evidence_ids:
            raise ValueError(f"{self.id}: an optimisation without evidence is a wish")


OPTIMISATIONS: dict[str, Optimisation] = {row.id: row for row in (
    Optimisation(
        id="tuned_knobs",
        phase="decode",
        summary="the knob set this machine confirmed against BASELINE, applied at load",
        preconditions=("a stored profile whose confirmation reads back as accepted",),
        fingerprint_scope=("hardware", "model_identity", "mlx", "mlx_lm"),
        objective="any",
        incompatible_with=(),
        correctness_contract=(
            "token identity against BASELINE over the confirmation prompt, plus "
            "determinism, checked before the profile is stored"
        ),
        evidence_ids=("autotuner_confirmation",),
    ),
    Optimisation(
        id="prefix_cache",
        phase="prefill",
        summary="ReusableSessionPlan: a declared shared prefix is prefilled once and reused",
        preconditions=("the caller declares the shared prefix",
                       "every request in the session carries the same plan"),
        fingerprint_scope=("model_identity",),
        objective="any",
        incompatible_with=(),
        correctness_contract=(
            "bit exact WITHIN the plan (E9: max|delta| = 0 over 12 requests and every "
            "step; E12: 756 requests across the sliding-window boundary). It does NOT "
            "agree with StrictOneShotPlan -- E9 measured up to 4.31 logits apart -- so "
            "the plan is a caller decision and the router never substitutes it"
        ),
        evidence_ids=("E9", "E12", "E10"),
    ),
    Optimisation(
        id="interactive_mode",
        phase="scheduling",
        summary="sequential batch-1; lowest latency for one caller",
        preconditions=(),
        fingerprint_scope=(),
        objective="latency",
        incompatible_with=("throughput_mode", "paired_throughput"),
        correctness_contract="the reference schedule; every other mode is compared to it",
        evidence_ids=("E15", "E16", "B55"),
    ),
    Optimisation(
        id="throughput_mode",
        phase="scheduling",
        summary="grouped batch-1 at width <= 4; aggregate throughput, at median latency",
        preconditions=("at least two requests ready at dispatch",),
        fingerprint_scope=("hardware", "model_identity"),
        objective="throughput",
        incompatible_with=("interactive_mode", "k3840_matvec"),
        correctness_contract="token, count and stop-reason identity against the sequential path",
        evidence_ids=("E15", "E16", "B55", "B56"),
    ),
    Optimisation(
        id="paired_throughput",
        phase="scheduling",
        summary="two ready requests share one weight sweep",
        preconditions=("exactly two ready compatible requests at a step boundary",
                       "a service-strategy record admitting this machine and model"),
        fingerprint_scope=("hardware", "model_identity", "mlx", "mlx_lm", "architecture"),
        objective="throughput",
        incompatible_with=("interactive_mode", "k3840_matvec"),
        correctness_contract=(
            "per request, logit bit patterns per step, the whole KV state, tokens and "
            "stop reason identical to the unpaired path"
        ),
        evidence_ids=("B45", "B46", "B47", "B50", "B51", "B52"),
    ),
    Optimisation(
        id="k3840_matvec",
        phase="decode",
        summary="the admitted K=3840 quantised matvec kernel on single-token decode",
        preconditions=("hidden size 3840", "4-bit weights, group size 64, bfloat16",
                       "single-token ungrouped decode: input shape exactly (1, 1, K)"),
        fingerprint_scope=("hardware", "model_identity", "mlx", "mlx_lm", "architecture"),
        objective="any",
        # `ironmule/qmv_k3840.py` refuses anything but shape (1, 1, K) and hands it back
        # to the library, and a paired step goes through the shared kernel instead. The
        # exclusion is in the source, not a judgement about which is better.
        incompatible_with=("throughput_mode", "paired_throughput"),
        correctness_contract=(
            "bit identical to the library call on the same buffers: tokens, the logit "
            "bit pattern of every step, the whole KV cache, stop behaviour and step count"
        ),
        evidence_ids=("B42", "B43", "B44", "B53", "B54"),
        activation="held",
        activation_note=(
            "B44 is INTEGRATION GO and left the knob False: 'nothing activates it'. B53 "
            "cleared the kernel of the defect it was suspected of, and cleared nothing "
            "else. No entry since has lifted the activation hold, so it may not enter a "
            "stack until a decision that names this state releases it"
        ),
    ),
    Optimisation(
        id="objective_router",
        phase="scheduling",
        summary="per-request objective decides which qualified schedule serves it",
        preconditions=("a qualified profile for this machine and model",),
        fingerprint_scope=("hardware", "model_identity", "mlx", "mlx_lm"),
        objective="any",
        incompatible_with=(),
        correctness_contract=(
            "the router selects, it never computes: token and stop-reason identity "
            "against every schedule it can select"
        ),
        evidence_ids=("B55", "B56"),
    ),
    Optimisation(
        id="own_dispatch",
        phase="scheduling",
        summary="a late latency request submitted from its own process",
        preconditions=("a second resident model",),
        fingerprint_scope=("hardware", "model_identity"),
        objective="latency",
        incompatible_with=(),
        correctness_contract="token and stop-reason identity against a solo reference",
        evidence_ids=("B56b",),
        activation="refused",
        activation_note=(
            "B56b measured it in two confirmation sessions and it did not protect the "
            "request. The submission really was concurrent -- 0.03 ms queue, 630 ms of "
            "overlap inside the cohort's window -- and the device simply shared itself: "
            "the request's decode rate fell from 75.1 to 50.0 tokens per second and its "
            "latency rose to 1.50x solo, against a 1.30x limit fixed before the run. It "
            "halves the wait (0.486x) but that is not protection, and the cohort pays "
            "1.22x for it. Two resident models cost 7.53 GB combined. Reopening needs a "
            "new mechanism, not another run of this one"
        ),
    ),
)}


@dataclass(frozen=True)
class Stack:
    """One execution candidate: a set of optimisations that must be measured as one."""

    id: str
    label: str
    members: tuple[str, ...]
    objective: str
    reference: str | None = None
    notes: str = ""

    def optimisations(self) -> tuple[Optimisation, ...]:
        return tuple(OPTIMISATIONS[name] for name in self.members)


@dataclass(frozen=True)
class Exclusion:
    stack_id: str
    reason: str
    detail: str


@dataclass
class Context:
    """What is known at composition time. No prediction, no response length."""

    hardware_fingerprint: str = ""
    model_identity_sha256: str = ""
    architecture: str = ""
    mlx: str = ""
    mlx_lm: str = ""
    profile_present: bool = False
    kernel_admits: bool = False
    paired_admits: bool = False
    released: tuple[str, ...] = field(default_factory=tuple)

    def allows(self, optimisation: Optimisation) -> tuple[bool, str]:
        if optimisation.activation == "pending":
            return False, f"{optimisation.id} is still being measured"
        if optimisation.activation == "refused":
            return False, f"{optimisation.id} was measured and refused: {optimisation.activation_note}"
        if optimisation.activation == "held" and optimisation.id not in self.released:
            return False, f"{optimisation.id} is qualified but held: {optimisation.activation_note}"
        if "model_identity" in optimisation.fingerprint_scope and not self.model_identity_sha256:
            return False, f"{optimisation.id} needs an exact model identity"
        if optimisation.id == "k3840_matvec" and not self.kernel_admits:
            return False, "the loaded model does not admit the K=3840 kernel"
        if optimisation.id == "paired_throughput" and not self.paired_admits:
            return False, "the loaded model does not admit the paired path"
        if optimisation.id == "tuned_knobs" and not self.profile_present:
            return False, "no confirmed profile for this machine and model"
        return True, ""


def internal_conflict(stack: Stack) -> str | None:
    """Two members the source itself keeps apart. Checked before anything is measured."""
    members = set(stack.members)
    for optimisation in stack.optimisations():
        clash = members & set(optimisation.incompatible_with)
        if clash:
            return (f"{optimisation.id} cannot run beside {sorted(clash)}: "
                    f"{optimisation.summary}")
    return None


def reduce(stacks: Sequence[Stack], context: Context
           ) -> tuple[tuple[Stack, ...], tuple[Exclusion, ...]]:
    """Which candidates are worth executing here, and why each of the others is not.

    This is the whole point of the table: the combinatorial set is never measured, and
    every candidate that is dropped is dropped for a stated, checkable reason.
    """
    admitted, excluded = [], []
    for stack in stacks:
        conflict = internal_conflict(stack)
        if conflict is not None:
            excluded.append(Exclusion(stack.id, "internally contradictory", conflict))
            continue
        blocked = [reason for ok, reason in
                   (context.allows(row) for row in stack.optimisations()) if not ok]
        if blocked:
            excluded.append(Exclusion(stack.id, "outside its fingerprint or held",
                                      "; ".join(blocked)))
            continue
        admitted.append(stack)
    return tuple(admitted), tuple(excluded)


#: The preregistered candidate set. `D` is stated as asked for and is expected to be
#: refused by `internal_conflict`; that refusal is the finding, not a workaround.
CANDIDATES: tuple[Stack, ...] = (
    Stack("A", "established reference", ("tuned_knobs", "interactive_mode"), "latency",
          notes="the reference every other stack is measured against"),
    Stack("A_t", "established reference, grouped",
          ("tuned_knobs", "throughput_mode"), "throughput",
          notes="the throughput-side reference"),
    Stack("B", "prefix reuse plus the tuned knob set",
          ("tuned_knobs", "prefix_cache", "interactive_mode"), "latency", reference="A"),
    Stack("B_t", "prefix reuse plus the tuned knob set, grouped",
          ("tuned_knobs", "prefix_cache", "throughput_mode"), "throughput",
          reference="A_t"),
    Stack("C", "B plus the paired path at two ready requests",
          ("tuned_knobs", "prefix_cache", "paired_throughput"), "throughput",
          reference="B_t"),
    Stack("D", "C plus the admitted K=3840 kernel",
          ("tuned_knobs", "prefix_cache", "paired_throughput", "k3840_matvec"),
          "throughput", reference="C",
          notes="as specified; the source keeps its last two members apart"),
    Stack("D_single", "B plus the admitted K=3840 kernel, ungrouped decode",
          ("tuned_knobs", "prefix_cache", "k3840_matvec"), "latency", reference="B",
          notes="the only shape in which the kernel can actually run beside the rest"),
    Stack("E", "the router choosing between the confirmed stacks",
          ("tuned_knobs", "prefix_cache", "objective_router"), "any",
          notes="measured only over stacks that were confirmed on their own"),
)


def _self_check() -> None:
    assert set(OPTIMISATIONS) == {row.id for row in OPTIMISATIONS.values()}
    for stack in CANDIDATES:
        for name in stack.members:
            assert name in OPTIMISATIONS, f"{stack.id} names an unknown optimisation {name}"

    # Incompatibility must be symmetric, or a stack would pass depending on member order.
    for row in OPTIMISATIONS.values():
        for other in row.incompatible_with:
            assert row.id in OPTIMISATIONS[other].incompatible_with, \
                f"{row.id}/{other} incompatibility is one-sided"

    assert internal_conflict(CANDIDATES[0]) is None
    d = next(s for s in CANDIDATES if s.id == "D")
    assert internal_conflict(d) is not None, \
        "D pairs the kernel with the paired path; the source keeps them apart"

    full = Context(hardware_fingerprint="hw", model_identity_sha256="id",
                   architecture="gemma3", mlx="0.32.0", mlx_lm="0.31.3",
                   profile_present=True, kernel_admits=True, paired_admits=True)
    admitted, excluded = reduce(CANDIDATES, full)
    ids = {s.id for s in admitted}
    assert "D" not in ids and "D_single" not in ids, \
        "the kernel is held; nothing composes it until a decision releases it"
    assert {"A", "A_t", "B", "B_t", "C", "E"} <= ids

    released = Context(**{**full.__dict__, "released": ("k3840_matvec",)})
    admitted, _ = reduce(CANDIDATES, released)
    assert "D_single" in {s.id for s in admitted}, "an explicit release must let it compose"
    assert "D" not in {s.id for s in admitted}, "releasing it does not resolve the conflict"

    bare = Context(profile_present=False)
    admitted, excluded = reduce(CANDIDATES, bare)
    assert not admitted and len(excluded) == len(CANDIDATES)

    no_paired = Context(**{**full.__dict__, "paired_admits": False})
    admitted, _ = reduce(CANDIDATES, no_paired)
    assert "C" not in {s.id for s in admitted} and "B_t" in {s.id for s in admitted}
    print("stack model self-check ok")


if __name__ == "__main__":
    _self_check()
