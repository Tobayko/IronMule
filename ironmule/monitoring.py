"""Watching a qualified action during ordinary use, without ever making a user an experiment.

`B79` let a locally qualified preference change a real dispatch. The thing it cannot do is
notice that the preference has gone stale. A machine gets a library update, a thermal problem,
a different background load, a model revision -- and the qualification that was true in
`B76` quietly stops being true. Something has to watch.

**Watching is not learning, and the line is structural.** Two kinds of record exist here and
they never mix. A *comparative* record is what `LocalLearner` already takes: a ratio measured
against a reference arm under gates, from a study. An *observation* is one real dispatch with
no counterfactual at all -- it says how long something took, not whether anything else would
have been faster. This module produces and consumes observations only. It holds no reference
to `LocalLearner.observe`, cannot construct an `Evidence` row, and a test asserts both. A
thousand fast candidate dispatches cannot qualify anything, because none of them measured the
alternative.

**What an observation may do is raise doubt.** When the window's own distribution moves away
from the baseline this segment started with, the answer is not to go and test the candidate
against the reference on a user's request. The answer is to stop using the candidate and say
so: `REQUALIFICATION_REQUIRED`, and the reference from then on. An active requalification is a
separate, explicit thing that a person starts. Ordinary use is never experiment material.

**Evidence does not rot on a calendar.** A segment is keyed by everything that could make old
numbers describe a different machine -- fingerprint, GPU, model identity and revision,
quantisation, libraries, workload class, action, and the digest of the action's own code. Any
of those changing starts a new segment and the old one is kept and never consulted. Age alone
widens uncertainty and deletes nothing: a number from last week is not wrong, it is less
certain.

**One slow request decides nothing.** The rule works on a window median against a baseline
median through a robust scale, and the window has to be full. That is fixed here, before any
of it ran, and is not moved afterwards.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA = "ironmule.monitoring.v1"

#: The monitor's own states. They are not the controller's: `LocalLearner` describes what the
#: evidence supports, this describes whether it still seems to hold.
WARMING_UP = "WARMING_UP"
MONITORING = "MONITORING"
REQUALIFICATION_REQUIRED = "REQUALIFICATION_REQUIRED"
STATES = (WARMING_UP, MONITORING, REQUALIFICATION_REQUIRED)

#: Observations needed before a segment has a baseline at all. Below this the monitor watches
#: and says nothing, which is the correct answer rather than a nervous one.
BASELINE_MIN = 15
#: The window whose median is compared. A window is only read when it is full, so no single
#: request can move anything.
WINDOW = 8
#: How far the window median may sit from the baseline, in robust standard errors. Five is
#: deliberately loose: a false requalification costs a user the whole gain, and the thing being
#: caught is a shift big enough to matter, not a fluctuation.
ROBUST_Z_LIMIT = 5.0
#: How much wider the window's own spread may get. A machine that became erratic is drifting
#: even when its median has not moved.
SPREAD_RATIO_LIMIT = 3.0
#: The consistency constant that makes a median absolute deviation comparable to a standard
#: deviation for normal data.
MAD_TO_SD = 1.4826
#: A floor, so a segment whose baseline happened to be extremely tight cannot make every later
#: observation look like a catastrophe. Expressed as a fraction of the baseline median.
MIN_RELATIVE_SCALE = 0.005
#: How far the median must actually move before a statistically clear shift is worth acting
#: on. Not a tuned number: the caller derives it from the gain the action was qualified for,
#: because a slowdown smaller than what the action buys leaves the action still winning and
#: there is nothing to requalify. Zero means the statistical rule alone decides.
DEFAULT_MIN_MATERIAL_SHIFT = 0.0
MATERIAL_SHIFT_RULE = (
    "a shift must be both statistically clear and larger than the gain the action was "
    "qualified for. The second half is what stops a very stable machine from requalifying "
    "over a move too small to change which action wins. The caller sets it from the "
    "controller's own qualified interval, conservatively: one minus that interval's upper "
    "bound, so the smallest gain the evidence supports is the bar")

#: Everything that, if it changes, means the old numbers described a different thing.
#: `effective_action` is part of the key and not a field beside it. A reference dispatch and
#: a candidate dispatch of the same workload take different times by construction, so a
#: baseline that mixed them would drift every time the gate opened or shut.
SEGMENT_FIELDS = ("hardware_fingerprint", "gpu_architecture", "model_identity_sha256",
                  "model_revision", "quantization_bits", "quantization_group_size",
                  "mlx", "mlx_lm", "workload_class", "action_id", "action_code_digest",
                  "effective_action")

OBSERVATION_FIELDS = frozenset({
    *SEGMENT_FIELDS, "observed_at", "end_to_end_ms",
    "service_ttft_ms", "tokens_per_second", "new_tokens", "prompt_tokens",
    "fallbacks", "correctness_errors", "memory_pressure_level", "swap_used_bytes",
    "controller_digest",
})

AGING_RULE = (
    "a segment is invalidated by change, never by the calendar. Any of "
    f"{list(SEGMENT_FIELDS)} differing starts a new segment; the old one is retained and "
    "never consulted again. Age alone is recorded and widens the reported uncertainty of a "
    "baseline, and deletes nothing: an old measurement is less certain, not wrong")

SEPARATION_RULE = (
    "observational evidence is one dispatch with no counterfactual and may only raise doubt. "
    "It can move a segment to REQUALIFICATION_REQUIRED and can never create or strengthen a "
    "candidate qualification. Only comparative evidence, measured against a reference arm "
    "under gates, moves the preference boundary, and this module cannot produce any")


class MonitoringError(ValueError):
    """A record that cannot be fully understood. It is never partially believed."""


def action_code_digest(*paths: Path) -> str:
    """The digest of the action's own code, so a changed kernel starts a new segment."""
    running = hashlib.sha256()
    for path in sorted(Path(p) for p in paths):
        running.update(path.name.encode())
        running.update(path.read_bytes())
    return running.hexdigest()


def default_action_code_digest() -> str:
    here = Path(__file__).resolve().parent
    return action_code_digest(here / "qmv_variant.py", here / "qmv_k3840.py")


# --------------------------------------------------------------------------- observations


@dataclass(frozen=True)
class Observation:
    """One real dispatch. It knows what happened, and nothing about what else might have."""

    observed_at: str
    hardware_fingerprint: str
    gpu_architecture: str
    model_identity_sha256: str
    model_revision: str
    quantization_bits: int
    quantization_group_size: int
    mlx: str
    mlx_lm: str
    workload_class: str
    action_id: str
    action_code_digest: str
    effective_action: str
    end_to_end_ms: float
    service_ttft_ms: float | None
    tokens_per_second: float | None
    new_tokens: int
    prompt_tokens: int
    fallbacks: int
    correctness_errors: int
    memory_pressure_level: int | None
    swap_used_bytes: int | None
    controller_digest: str

    @property
    def segment(self) -> tuple:
        return tuple(getattr(self, name) for name in SEGMENT_FIELDS)

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in sorted(OBSERVATION_FIELDS)}


def observation_from(raw: Mapping[str, Any]) -> Observation:
    """Parse one record, or refuse it. Nothing is defaulted and nothing is coerced."""
    if not isinstance(raw, Mapping):
        raise MonitoringError("an observation is not a mapping")
    found = set(raw)
    missing = OBSERVATION_FIELDS - found
    unknown = found - OBSERVATION_FIELDS
    if missing:
        raise MonitoringError(f"observation is missing {sorted(missing)}")
    if unknown:
        raise MonitoringError(f"observation carries unknown {sorted(unknown)}")
    for name in ("observed_at", "hardware_fingerprint", "model_identity_sha256",
                 "model_revision", "mlx", "mlx_lm", "workload_class", "action_id",
                 "action_code_digest", "effective_action", "controller_digest"):
        if not isinstance(raw[name], str) or not raw[name]:
            raise MonitoringError(f"{name} must be a non-empty string")
    value = raw["end_to_end_ms"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(float(value)) or float(value) <= 0.0:
        raise MonitoringError("end_to_end_ms must be a finite positive number")
    for name in ("new_tokens", "prompt_tokens", "fallbacks", "correctness_errors"):
        if isinstance(raw[name], bool) or not isinstance(raw[name], int) or raw[name] < 0:
            raise MonitoringError(f"{name} must be a non-negative integer")
    return Observation(**raw)


# --------------------------------------------------------------------------- the rule


def _scale(values: Sequence[float], centre: float) -> float:
    """A robust spread, floored so a freakishly tight baseline cannot cry wolf forever."""
    deviation = statistics.median([abs(v - centre) for v in values]) * MAD_TO_SD
    return max(deviation, abs(centre) * MIN_RELATIVE_SCALE)


@dataclass(frozen=True)
class Baseline:
    """What a segment looked like when it was healthy, frozen once and never refitted."""

    n: int
    median: float
    scale: float
    first_observed_at: str
    last_observed_at: str

    def as_dict(self) -> dict[str, Any]:
        return {"n": self.n, "median": self.median, "scale": self.scale,
                "first_observed_at": self.first_observed_at,
                "last_observed_at": self.last_observed_at}


@dataclass(frozen=True)
class DriftVerdict:
    """Whether the window has moved, by how much, and on which rule."""

    drifted: bool
    reason: str
    robust_z: float | None = None
    window_median: float | None = None
    spread_ratio: float | None = None
    relative_shift: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"drifted": self.drifted, "reason": self.reason, "robust_z": self.robust_z,
                "window_median": self.window_median, "spread_ratio": self.spread_ratio,
                "relative_shift": self.relative_shift}


def evaluate_window(baseline: Baseline, window: Sequence[float],
                    min_material_shift: float = DEFAULT_MIN_MATERIAL_SHIFT) -> DriftVerdict:
    """The whole drift rule, as one pure function. Fixed before it ran and not moved.

    A window is only read when it is full, and what is compared is its median against the
    baseline's, through the baseline's own robust scale and the window's size. A single slow
    request moves a median of eight by nothing worth reporting.
    """
    if len(window) < WINDOW:
        return DriftVerdict(False, f"the window holds {len(window)} of {WINDOW} observations")
    centre = statistics.median(window)
    standard_error = baseline.scale / math.sqrt(len(window))
    robust_z = (centre - baseline.median) / standard_error if standard_error else 0.0
    spread_ratio = _scale(window, centre) / baseline.scale if baseline.scale else 1.0
    relative = abs(centre - baseline.median) / baseline.median if baseline.median else 0.0
    if abs(robust_z) > ROBUST_Z_LIMIT and relative >= min_material_shift:
        return DriftVerdict(True, f"the window median sits {robust_z:+.1f} robust standard "
                                  f"errors from the baseline, past {ROBUST_Z_LIMIT}, and "
                                  f"{relative:.1%} away from it, past the "
                                  f"{min_material_shift:.1%} the action's own gain sets",
                            robust_z, centre, spread_ratio, relative)
    if abs(robust_z) > ROBUST_Z_LIMIT:
        return DriftVerdict(False, f"the window median sits {robust_z:+.1f} robust standard "
                                   f"errors from the baseline but only {relative:.1%} away "
                                   f"from it, inside the {min_material_shift:.1%} the "
                                   f"action's gain would still cover",
                            robust_z, centre, spread_ratio, relative)
    if spread_ratio > SPREAD_RATIO_LIMIT:
        return DriftVerdict(True, f"the window's spread is {spread_ratio:.1f} times the "
                                  f"baseline's, past {SPREAD_RATIO_LIMIT}",
                            robust_z, centre, spread_ratio, relative)
    return DriftVerdict(False, f"the window sits {robust_z:+.1f} robust standard errors from "
                               f"the baseline with {spread_ratio:.1f} times its spread",
                        robust_z, centre, spread_ratio, relative)


# --------------------------------------------------------------------------- the monitor


@dataclass
class Segment:
    """One (machine, model, libraries, workload, action, code) combination, watched."""

    key: tuple
    fields: dict[str, Any]
    observations: int = 0
    baseline: Baseline | None = None
    warmup: list[float] = None
    window: list[float] = None
    state: str = WARMING_UP
    reason: str = "no baseline yet"
    verdict: DriftVerdict | None = None
    first_observed_at: str = ""
    last_observed_at: str = ""
    superseded: bool = False

    def __post_init__(self) -> None:
        if self.warmup is None:
            self.warmup = []
        if self.window is None:
            self.window = []

    def as_dict(self) -> dict[str, Any]:
        return {"fields": dict(self.fields), "observations": self.observations,
                "baseline": self.baseline.as_dict() if self.baseline else None,
                "state": self.state, "reason": self.reason,
                "verdict": self.verdict.as_dict() if self.verdict else None,
                "window": list(self.window),
                "first_observed_at": self.first_observed_at,
                "last_observed_at": self.last_observed_at,
                "superseded": self.superseded}


class DriftMonitor:
    """Passive. It records what happened and, at most, says the preference needs rechecking.

    It cannot qualify anything: it has no way to construct comparative evidence and no
    reference to the controller's intake.
    """

    def __init__(self, state_path: Path | None = None,
                 min_material_shift: float = DEFAULT_MIN_MATERIAL_SHIFT):
        self.state_path = Path(state_path) if state_path else None
        self.min_material_shift = float(min_material_shift)
        self.segments: dict[tuple, Segment] = {}
        self.transitions: list[dict[str, Any]] = []
        self.rejected: list[dict[str, Any]] = []
        self._required = self._read_required()
        #: A plain attribute, not a property. The activation layer reads it on every
        #: dispatch, and a property call is measurable against a decision this small.
        self.requalification_required = self._required is not None

    # -- intake ---------------------------------------------------------------
    def observe(self, observation: Observation) -> dict[str, Any]:
        """One real dispatch. Returns what this did to the segment, if anything."""
        key = observation.segment
        segment = self.segments.get(key)
        if segment is None:
            # A new key means the old numbers described something else. Old segments are
            # kept and marked, never deleted and never consulted again.
            for other in self.segments.values():
                # Only a segment describing the same thing under superseded conditions is
                # retired. A different action is a sibling, not a replacement.
                if (other.fields["workload_class"] == observation.workload_class
                        and other.fields["action_id"] == observation.action_id
                        and other.fields["effective_action"] == observation.effective_action):
                    other.superseded = True
            segment = Segment(key=key,
                              fields=dict(zip(SEGMENT_FIELDS, key)),
                              first_observed_at=observation.observed_at)
            self.segments[key] = segment
        segment.observations += 1
        segment.last_observed_at = observation.observed_at

        if segment.state == REQUALIFICATION_REQUIRED:
            return {"segment": segment.fields, "state": segment.state,
                    "reason": segment.reason, "changed": False}

        if segment.baseline is None:
            segment.warmup.append(observation.end_to_end_ms)
            if len(segment.warmup) >= BASELINE_MIN:
                centre = statistics.median(segment.warmup)
                segment.baseline = Baseline(
                    n=len(segment.warmup), median=centre,
                    scale=_scale(segment.warmup, centre),
                    first_observed_at=segment.first_observed_at,
                    last_observed_at=observation.observed_at)
                segment.state, segment.reason = MONITORING, "baseline established"
                self._transition(segment, WARMING_UP, MONITORING, segment.reason)
            return {"segment": segment.fields, "state": segment.state,
                    "reason": segment.reason, "changed": segment.baseline is not None}

        segment.window.append(observation.end_to_end_ms)
        if len(segment.window) > WINDOW:
            del segment.window[0]
        verdict = evaluate_window(segment.baseline, segment.window,
                                  self.min_material_shift)
        segment.verdict = verdict
        if verdict.drifted:
            self.require_requalification(segment, verdict.reason)
            return {"segment": segment.fields, "state": segment.state,
                    "reason": segment.reason, "changed": True,
                    "verdict": verdict.as_dict()}
        return {"segment": segment.fields, "state": segment.state,
                "reason": verdict.reason, "changed": False, "verdict": verdict.as_dict()}

    def observe_raw(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        try:
            return self.observe(observation_from(raw))
        except MonitoringError as error:
            record = {"accepted": False, "reason": str(error)}
            self.rejected.append(record)
            return record

    # -- state ----------------------------------------------------------------
    def require_requalification(self, segment: Segment, reason: str) -> None:
        """The only thing observation may cause, and it only ever points one way."""
        previous = segment.state
        segment.state, segment.reason = REQUALIFICATION_REQUIRED, reason
        self._transition(segment, previous, REQUALIFICATION_REQUIRED, reason)
        self._required = {"schema": SCHEMA, "reason": reason,
                          "segment": dict(segment.fields)}
        self.requalification_required = True
        if self.state_path:
            _write_atomic(requalification_path(self.state_path),
                          {"schema": SCHEMA, "reason": reason,
                           "segment": dict(segment.fields),
                           "effect": "the reference serves until an explicit requalification "
                                     "run produces new comparative evidence. Ordinary use "
                                     "never produces it"})

    def state_for(self, workload_class: str, action_id: str) -> str:
        if self._required is not None:
            return REQUALIFICATION_REQUIRED
        states = [segment.state for segment in self.segments.values()
                  if not segment.superseded
                  and segment.fields["workload_class"] == workload_class
                  and segment.fields["action_id"] == action_id
                  and segment.fields["effective_action"] == "candidate"]
        if REQUALIFICATION_REQUIRED in states:
            return REQUALIFICATION_REQUIRED
        return states[0] if states else WARMING_UP

    def _transition(self, segment: Segment, before: str, after: str, reason: str) -> None:
        self.transitions.append({"workload_class": segment.fields["workload_class"],
                                 "action_id": segment.fields["action_id"],
                                 "from": before, "to": after, "reason": reason,
                                 "observations": segment.observations,
                                 "at": segment.last_observed_at})

    def _read_required(self) -> dict[str, Any] | None:
        if not self.state_path:
            return None
        path = requalification_path(self.state_path)
        if not path.exists():
            return None
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"schema": SCHEMA, "reason": "the requalification record is unreadable, "
                                                "which is treated as required"}
        return stored if isinstance(stored, Mapping) else {
            "schema": SCHEMA, "reason": "the requalification record is not an object"}

    # -- persistence ----------------------------------------------------------
    def as_dict(self) -> dict[str, Any]:
        body = {"schema": SCHEMA,
                "segments": [segment.as_dict() for segment in self.segments.values()],
                "transitions": list(self.transitions),
                "requalification_required": self.requalification_required,
                "aging_rule": AGING_RULE,
                "separation_rule": SEPARATION_RULE,
                "rule": {"baseline_min": BASELINE_MIN, "window": WINDOW,
                         "robust_z_limit": ROBUST_Z_LIMIT,
                         "spread_ratio_limit": SPREAD_RATIO_LIMIT,
                         "min_relative_scale": MIN_RELATIVE_SCALE,
                         "min_material_shift": self.min_material_shift,
                         "material_shift_rule": MATERIAL_SHIFT_RULE},
                "monitoring_only": True, "activation": "none"}
        return {**body, "digest": hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":"),
                       allow_nan=False).encode("utf-8")).hexdigest()}

    def save(self, path: Path) -> Path:
        return _write_atomic(Path(path), self.as_dict())

    @classmethod
    def restore(cls, path: Path, state_path: Path | None = None,
                min_material_shift: float = DEFAULT_MIN_MATERIAL_SHIFT) -> "DriftMonitor":
        """Rebuilt or empty. A file that cannot be understood leaves a monitor that watches."""
        monitor = cls(state_path, min_material_shift)
        try:
            stored = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return monitor
        if not isinstance(stored, Mapping) or stored.get("schema") != SCHEMA:
            return monitor
        digest = stored.get("digest")
        body = {name: value for name, value in stored.items() if name != "digest"}
        recomputed = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                               allow_nan=False).encode("utf-8")).hexdigest()
        if digest != recomputed:
            return monitor
        for row in stored.get("segments") or ():
            fields = row["fields"]
            key = tuple(fields[name] for name in SEGMENT_FIELDS)
            baseline = row.get("baseline")
            monitor.segments[key] = Segment(
                key=key, fields=dict(fields), observations=row["observations"],
                baseline=Baseline(**baseline) if baseline else None,
                window=list(row.get("window") or []),
                state=row["state"], reason=row["reason"],
                first_observed_at=row.get("first_observed_at", ""),
                last_observed_at=row.get("last_observed_at", ""),
                superseded=bool(row.get("superseded")))
        monitor.transitions = list(stored.get("transitions") or ())
        if stored.get("requalification_required") and monitor._required is None:
            monitor._required = {"schema": SCHEMA,
                                 "reason": "restored from a state that required it"}
            monitor.requalification_required = True
        return monitor


def requalification_path(state_path: Path) -> Path:
    return Path(state_path).with_name(Path(state_path).name + ".requalify")


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".partial")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return path


def _self_check() -> None:
    """Stable use changes nothing; a real shift is caught; one slow request is not a shift."""
    def row(index: int, milliseconds: float, **overrides):
        base = {"observed_at": f"2026-09-11T00:{index:02d}:00+00:00",
                "hardware_fingerprint": "fp", "gpu_architecture": "arch",
                "model_identity_sha256": "sha", "model_revision": "rev",
                "quantization_bits": 4, "quantization_group_size": 64,
                "mlx": "0.32.0", "mlx_lm": "0.31.3", "workload_class": "single_short",
                "action_id": "a", "action_code_digest": "deadbeef",
                "effective_action": "candidate", "end_to_end_ms": milliseconds,
                "service_ttft_ms": 20.0, "tokens_per_second": 33.0, "new_tokens": 32,
                "prompt_tokens": 27, "fallbacks": 0, "correctness_errors": 0,
                "memory_pressure_level": 1, "swap_used_bytes": 0,
                "controller_digest": "cafe"}
        return observation_from({**base, **overrides})

    monitor = DriftMonitor()
    for index in range(BASELINE_MIN):
        monitor.observe(row(index, 920.0 + (index % 3)))
    assert monitor.state_for("single_short", "a") == MONITORING
    for index in range(40):
        monitor.observe(row(100 + index, 920.0 + (index % 5)))
    assert not monitor.requalification_required, "stable use must not requalify"

    monitor.observe(row(200, 3000.0))
    assert not monitor.requalification_required, "one slow request is not a drift"

    for index in range(WINDOW):
        monitor.observe(row(300 + index, 1400.0 + index))
    assert monitor.requalification_required, "a sustained shift must be caught"
    assert monitor.state_for("single_short", "a") == REQUALIFICATION_REQUIRED

    fresh = DriftMonitor()
    for index in range(BASELINE_MIN):
        fresh.observe(row(index, 920.0))
    fresh.observe(row(500, 920.0, mlx="0.33.0"))
    assert len(fresh.segments) == 2, "a library change starts a new segment"
    assert any(segment.superseded for segment in fresh.segments.values())
    print("monitoring self-check passed")


if __name__ == "__main__":
    _self_check()
