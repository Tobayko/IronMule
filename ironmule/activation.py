"""The one place a locally learned preference is allowed to change what actually runs.

`B78` built a controller that reaches `CANDIDATE_QUALIFIED` from this machine's own evidence
and cannot touch a dispatch. `B79` is the step that lets it, and every line here exists to
make that step small, explicit and reversible.

**Off by default, and the default is the product.** `AppleRuntime.load` takes
`enable_local_learned_dispatch=False`. A profile cannot switch it on, a silicon parameter
cannot, an environment variable cannot. Only a caller passing `True` *and* a controller
reporting `CANDIDATE_QUALIFIED` for exactly this context gets anything other than the
reference.

**The router is not touched.** `ExecutionRouter` plans as it always did. This layer runs
after that plan exists and may make exactly one substitution, `reference action -> locally
qualified candidate`, for the workload classes the controller itself qualified. It can never
choose a third thing, and a failure can only ever go one way.

**No magnitude decides anything.** The activation reads the controller's *state* and its
evidence count. It never reads the estimated ratio: `B76` measured a stable sign with a
magnitude that moves, and a decision made on a moving number would be a decision made on noise.

**One direction out of trouble.** Any anomaly -- a correctness difference, a fallback in the
candidate path, a resource gate, an exception, a controller whose state no longer qualifies --
shuts the gate for every request that has not started, restores the library path, and writes a
kill record that survives the process. There is no second experimental action to try.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .local_learner import CANDIDATE_QUALIFIED, LocalLearner, UNKNOWN
from .monitoring import DriftMonitor, REQUALIFICATION_REQUIRED
from .qmv_variant import (CORRECTNESS_CONTRACT, QUALIFIED_ACTION_ID, QUALIFIED_GEOMETRY,
                          VariantGate, VariantUnsupported, install, uninstall)

SCHEMA = "ironmule.local_activation.v1"
REFERENCE = "reference"
CANDIDATE = "candidate"
#: The variant is qualified for single-token decode on the sequential path. Nothing else.
ELIGIBLE_ROUTES = ("interactive",)
#: `K` the kernel is built for, and the width rule its geometry imposes.
TARGET_K = 3840
ROWS_PER_GROUP = QUALIFIED_GEOMETRY[0] * QUALIFIED_GEOMETRY[1]


@dataclass(frozen=True)
class ActivationContext:
    """Everything the admission has to agree on before anything is installed."""

    hardware_fingerprint: str = ""
    gpu_architecture: str = ""
    model_id: str = ""
    model_identity_sha256: str = ""
    model_revision: str = ""
    quantization_bits: int = 0
    quantization_group_size: int = 0
    hidden_size: int = 0
    projection_widths: tuple[int, ...] = ()
    mlx: str = ""
    mlx_lm: str = ""


@dataclass(frozen=True)
class Admission:
    """Whether the candidate may be installed at all, and which axis decided it."""

    admitted: bool
    reason: str
    action_id: str = ""
    controller_digest: str = ""
    workload_classes: tuple[str, ...] = ()
    checked: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"admitted": self.admitted, "reason": self.reason,
                "action_id": self.action_id, "controller_digest": self.controller_digest,
                "workload_classes": list(self.workload_classes),
                "checked": dict(self.checked)}


@dataclass(frozen=True)
class ActivationRecord:
    """What one dispatch was allowed to do, and on which state."""

    base_action: str
    effective_action: str
    local_learning_state: str
    evidence_count: int
    controller_digest: str
    activation_reason: str
    workload_class: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"base_action": self.base_action, "effective_action": self.effective_action,
                "local_learning_state": self.local_learning_state,
                "evidence_count": self.evidence_count,
                "controller_digest": self.controller_digest,
                "activation_reason": self.activation_reason,
                "workload_class": self.workload_class,
                "opt_in_required": True, "default_enabled": False}


OFF = ActivationRecord(REFERENCE, REFERENCE, UNKNOWN, 0, "",
                       "local learned dispatch is not enabled; the default is off")


def kill_path(state_path: Path) -> Path:
    return Path(state_path).with_name(Path(state_path).name + ".kill")


def read_kill(state_path: Path) -> dict[str, Any] | None:
    """A kill that survived the process. Unreadable counts as killed, never as absent."""
    path = kill_path(state_path)
    if not path.exists():
        return None
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"schema": SCHEMA, "reason": "the kill record is unreadable, which is "
                                            "treated as killed rather than as absent"}
    return stored if isinstance(stored, Mapping) else {
        "schema": SCHEMA, "reason": "the kill record is not an object"}


def write_kill(state_path: Path, reason: str, detail: Mapping[str, Any] | None = None) -> Path:
    """Persist the kill atomically. It is never removed by this module."""
    path = kill_path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"schema": SCHEMA, "reason": reason,
                          "detail": dict(detail or {}),
                          "effect": "local learned dispatch stays off on this machine until "
                                    "a person removes this file and understands why it exists"},
                         indent=2, sort_keys=True)
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".partial")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(payload)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return path


class LearnedDispatchActivation:
    """Small on purpose: one substitution, one direction out, one switch that stays off."""

    def __init__(self, learner: LocalLearner | None, context: ActivationContext, *,
                 enabled: bool = False, state_path: Path | None = None,
                 action_id: str = QUALIFIED_ACTION_ID,
                 monitor: DriftMonitor | None = None):
        self.learner = learner
        # `B80`, passive. It may take the candidate away and can never hand it back: only an
        # explicit requalification producing comparative evidence does that.
        self.monitor = monitor
        self.context = context
        self.action_id = action_id
        self.state_path = Path(state_path) if state_path else None
        self.gate: VariantGate | None = None
        self.installed: dict[str, Any] | None = None
        self.events: list[dict[str, Any]] = []
        self._model: Any | None = None
        #: One record per (workload class, route), built the first time that pair is seen
        #: and reused after. `B70`'s lesson: a per-dispatch path builds no strings.
        self._records: dict[tuple[str, str], ActivationRecord] = {}
        self._digest: str = ""

        killed = read_kill(self.state_path) if self.state_path else None
        self.killed_at_start = killed
        # A persisted kill wins over the caller's flag. That is the point of persisting it.
        self.enabled = bool(enabled) and killed is None
        self.disabled_reason = ""
        if not enabled:
            self.disabled_reason = "not enabled by the caller; the default is off"
        elif killed is not None:
            self.disabled_reason = (
                f"a persisted kill record is present: {killed.get('reason', 'no reason given')}")

    # -- admission ------------------------------------------------------------
    def admit(self) -> Admission:
        """Every axis, checked before anything is installed. The first failure is the reason."""
        checked: dict[str, Any] = {"opt_in": self.enabled}
        if not self.enabled:
            return Admission(False, self.disabled_reason, checked=checked)
        if self.learner is None:
            return Admission(False, "no local controller", checked=checked)

        stored = self.learner.as_dict()
        digest = stored["digest"]
        checked["controller_digest"] = digest
        stored_context = stored["context"]
        for name in ("hardware_fingerprint", "model_identity_sha256", "model_revision",
                     "mlx", "mlx_lm"):
            expected = getattr(self.context, name)
            checked[name] = expected
            if not expected or stored_context.get(name) != expected:
                return Admission(False, f"controller state does not match this runtime on "
                                        f"{name}", controller_digest=digest, checked=checked)
        if self.context.gpu_architecture and \
                stored_context.get("gpu_architecture") not in ("", self.context.gpu_architecture):
            return Admission(False, "gpu architecture", controller_digest=digest,
                             checked=checked)
        checked["gpu_architecture"] = self.context.gpu_architecture

        if self.context.quantization_bits != 4 or self.context.quantization_group_size != 64:
            return Admission(False, "quantisation: the kernel is built for 4 bits at group "
                                    "size 64", controller_digest=digest, checked=checked)
        checked["quantization"] = {"bits": self.context.quantization_bits,
                                   "group_size": self.context.quantization_group_size}
        if self.context.hidden_size and self.context.hidden_size != TARGET_K:
            return Admission(False, f"K is {self.context.hidden_size}, the kernel is built "
                                    f"for {TARGET_K}", controller_digest=digest,
                             checked=checked)
        widths = tuple(self.context.projection_widths)
        if widths and not any(width % ROWS_PER_GROUP == 0 for width in widths):
            return Admission(False, f"no projection width is a whole number of "
                                    f"{ROWS_PER_GROUP} rows", controller_digest=digest,
                             checked=checked)
        checked["k"] = self.context.hidden_size
        checked["admitted_n"] = [w for w in widths if w % ROWS_PER_GROUP == 0]
        checked["action_id"] = self.action_id
        checked["correctness_contract"] = CORRECTNESS_CONTRACT

        # Only the classes the controller itself qualified. Nothing is carried across to
        # paired, throughput or any class it has not measured.
        qualified = tuple(sorted(
            workload_class for (action, workload_class) in self.learner._recommendations
            if action == self.action_id
            and self.learner.recommendation(action, workload_class).local_learning_state
            == CANDIDATE_QUALIFIED))
        checked["local_preference"] = CANDIDATE_QUALIFIED if qualified else "not qualified"
        if not qualified:
            return Admission(False, "the local controller qualifies no workload class for "
                                    "this action", controller_digest=digest, checked=checked)
        return Admission(True, "every admission axis agrees and the local controller "
                               "qualifies at least one workload class",
                         action_id=self.action_id, controller_digest=digest,
                         workload_classes=qualified, checked=checked)

    # -- installation ---------------------------------------------------------
    def install_on(self, model: Any, identity: Any) -> Admission:
        """Install the variant if and only if admission passes. Shut, always."""
        admission = self.admit()
        if not admission.admitted:
            return admission
        try:
            self.gate, self.installed = install(model, identity)
        except (VariantUnsupported, RuntimeError) as error:
            self.kill(f"the variant could not be installed: {type(error).__name__}: {error}")
            return Admission(False, f"installation refused: {error}",
                             controller_digest=admission.controller_digest,
                             checked=dict(admission.checked))
        self._model = model
        self.admission = admission
        self._digest = admission.controller_digest
        return admission

    def prepare(self) -> Admission:
        """Admit without a model: the decision path, with nothing installed to run.

        This is what a preflight needs and what a runtime that has not loaded a model has.
        The gate it produces is shut and stays shut unless a dispatch opens it, and with no
        installed projection an open gate changes nothing.
        """
        admission = self.admit()
        if admission.admitted:
            self.gate = VariantGate()
            self.admission = admission
            self._digest = admission.controller_digest
        return admission

    # -- per dispatch ---------------------------------------------------------
    def for_dispatch(self, workload_class: str, route: str) -> ActivationRecord:
        """Open or shut the gate for the dispatch about to run, and say why.

        Reads the controller's state and evidence count. It never reads an estimated ratio:
        a magnitude that moves between sessions is not a thing to decide on.

        After the first dispatch of a given shape this is a string compare, a dictionary
        lookup and at most two attribute writes. Nothing is formatted and nothing is
        allocated, because a dispatch path that builds its own explanation is the mistake
        `B70` already paid for once.
        """
        if not self.enabled or self.gate is None or self.learner is None:
            return ActivationRecord(REFERENCE, REFERENCE, UNKNOWN, 0, "",
                                    self.disabled_reason or "no variant is installed",
                                    workload_class)
        if self.gate.killed:
            return self._killed_record(workload_class)
        # One attribute read. The monitor watches ordinary dispatches and, when the segment
        # it built from them has moved away from its own baseline, the reference serves.
        if self.monitor is not None and self.monitor.requalification_required:
            if self.gate.active:
                self.gate.close("requalification is required")
            return ActivationRecord(REFERENCE, REFERENCE, REQUALIFICATION_REQUIRED, 0, "",
                                    "passive monitoring requires a requalification before "
                                    "this action is used again", workload_class)

        digest = self.learner.digest
        if digest != self._digest:
            # The controller changed under us. Re-read it rather than trust a stale pass,
            # and throw away every record built from the old state.
            self._records.clear()
            recommendation = self.learner.recommendation(self.action_id, workload_class)
            if recommendation.local_learning_state != CANDIDATE_QUALIFIED                     and self._digest:
                self.kill("the controller state changed and no longer qualifies this action")
                return self._killed_record(workload_class)
            self._digest = digest
            self.admission = Admission(True, "admitted on the current controller state",
                                       action_id=self.action_id, controller_digest=digest)

        key = (workload_class, route)
        record = self._records.get(key)
        if record is None:
            record = self._build_record(workload_class, route, digest)
            self._records[key] = record
        if record.effective_action == CANDIDATE:
            if not self.gate.active:
                self.gate.open(record.activation_reason)
        elif self.gate.active:
            self.gate.close(record.activation_reason)
        return record

    def _killed_record(self, workload_class: str) -> ActivationRecord:
        return ActivationRecord(REFERENCE, REFERENCE, UNKNOWN, 0, "",
                                f"killed: {self.gate.reason if self.gate else 'unknown'}",
                                workload_class)

    def _build_record(self, workload_class: str, route: str,
                      digest: str) -> ActivationRecord:
        """Built once per shape, off the dispatch path after that."""
        recommendation = self.learner.recommendation(self.action_id, workload_class)
        if route not in ELIGIBLE_ROUTES:
            return ActivationRecord(REFERENCE, REFERENCE,
                                    recommendation.local_learning_state,
                                    recommendation.evidence_count, digest,
                                    f"route {route!r} is outside {list(ELIGIBLE_ROUTES)}",
                                    workload_class)
        if recommendation.local_learning_state != CANDIDATE_QUALIFIED:
            return ActivationRecord(REFERENCE, REFERENCE,
                                    recommendation.local_learning_state,
                                    recommendation.evidence_count, digest,
                                    f"workload class {workload_class!r} is "
                                    f"{recommendation.local_learning_state}", workload_class)
        return ActivationRecord(REFERENCE, CANDIDATE, recommendation.local_learning_state,
                                recommendation.evidence_count, digest,
                                f"the local controller qualifies {workload_class!r} on "
                                f"{recommendation.evidence_count} accepted sessions",
                                workload_class)

    def after_dispatch(self, record: ActivationRecord,
                       telemetry: Any) -> dict[str, Any] | None:
        """The only thing that may follow a candidate dispatch: nothing, or a kill."""
        if record.effective_action != CANDIDATE:
            return None
        reasons = []
        if getattr(telemetry, "fallbacks", 0):
            reasons.append(f"{telemetry.fallbacks} fallbacks in the candidate path")
        if getattr(telemetry, "correctness_errors", 0):
            reasons.append(f"{telemetry.correctness_errors} correctness errors")
        if not reasons:
            return None
        return self.kill("; ".join(reasons))

    def kill(self, reason: str, detail: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Shut the gate, restore the library path, and make it survive the process."""
        if self.gate is not None:
            self.gate.kill(reason)
        self._records.clear()
        restored = uninstall(self._model) if self._model is not None else 0
        self.enabled = False
        self.disabled_reason = f"killed: {reason}"
        event = {"killed": True, "reason": reason, "projections_restored": restored,
                 "detail": dict(detail or {}),
                 "persisted_to": (str(write_kill(self.state_path, reason, detail))
                                  if self.state_path else None)}
        self.events.append(event)
        return event

    def status(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "enabled": self.enabled,
                "default_enabled": False,
                "disabled_reason": self.disabled_reason,
                "action_id": self.action_id,
                "installed": self.installed,
                "gate": ({"active": self.gate.active, "killed": self.gate.killed,
                          "reason": self.gate.reason} if self.gate else None),
                "killed_at_start": self.killed_at_start,
                "requalification_required": (self.monitor.requalification_required
                                             if self.monitor is not None else None),
                "events": list(self.events)}


def _self_check() -> None:
    """Default off, a persisted kill wins, and nothing qualifies without a controller."""
    context = ActivationContext("fp", "applegpu_g13s", "m", "sha", "rev", 4, 64, 3840,
                                (15360,), "0.32.0", "0.31.3")
    off = LearnedDispatchActivation(None, context)
    assert off.enabled is False
    assert off.admit().admitted is False
    assert off.for_dispatch("single_short", "interactive").effective_action == REFERENCE

    asked = LearnedDispatchActivation(None, context, enabled=True)
    assert asked.enabled is True and asked.admit().admitted is False, \
        "opting in without a controller qualifies nothing"

    with tempfile.TemporaryDirectory() as directory:
        state = Path(directory) / "local_learning.json"
        write_kill(state, "a fallback was recorded")
        revived = LearnedDispatchActivation(None, context, enabled=True, state_path=state)
        assert revived.enabled is False, "a persisted kill outranks the caller's flag"
        assert "persisted kill" in revived.disabled_reason
    print("activation self-check passed")


if __name__ == "__main__":
    _self_check()
