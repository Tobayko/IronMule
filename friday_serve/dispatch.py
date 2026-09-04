"""Phase-aware dispatch: which knobs this device earned, per phase, per request.

Prefill is compute-bound (`45.5 %` of peak here) and decode is bandwidth-bound
(`78 %` of the weight-stream limit); a knob verified for one says nothing about
the other, so the profile carries a phase per knob and dispatch reads it per
phase rather than once per process.

Only a knob with verdict ``verified`` in the device profile is ever turned on.
There is no "probably fine" path: an unverified knob is off, and a profile that
verified nothing serves the baseline, correctly and slightly slower.
"""

from __future__ import annotations

from typing import Any, Mapping

from friday_calibrate.profile import KNOB_PHASE, DeviceProfile

#: How a verified knob is expressed to the engine. Same table as calibration
#: uses, because a knob must be turned on exactly the way it was verified.
KNOB_TO_ENGINE: Mapping[str, Mapping[str, Any]] = {
    "head_skip": {"head_skip_prefill": True},
    "fixed_compiled": {"compiled_fixed_cache": True},
    "bundled_readback": {"readback_every": 8},
    "fuse_projections": {"fuse_projections": True},
}

PHASES = ("prefill", "decode")


def knobs_for(profile: DeviceProfile | None, *, phases: tuple[str, ...] = PHASES) -> dict[str, Any]:
    """Engine knob overrides this device is authorised to use. Empty is valid."""

    if profile is None:
        return {}
    overrides: dict[str, Any] = {}
    for knob, engine_knobs in KNOB_TO_ENGINE.items():
        if KNOB_PHASE[knob] not in phases:
            continue
        if profile.is_verified(knob):
            overrides.update(engine_knobs)
    return overrides


def explain(profile: DeviceProfile | None) -> dict[str, Any]:
    """Why each knob is on or off, for the dashboard and for a human."""

    if profile is None:
        return {"profile": None, "knobs": {}, "reason": "no_device_profile"}
    knobs = {}
    for knob in KNOB_TO_ENGINE:
        verdict = profile.verdict_for(knob)
        knobs[knob] = {
            "phase": KNOB_PHASE[knob],
            "active": profile.is_verified(knob),
            "verdict": None if verdict is None else verdict.verdict,
            "ratio": None if verdict is None else verdict.ratio,
            "reason": "never_calibrated" if verdict is None else verdict.reason,
        }
    return {
        "profile": profile.profile_id,
        "model_id": profile.model_id,
        "model_revision": profile.model_revision,
        "mde": profile.mde,
        "knobs": knobs,
    }


__all__ = ["KNOB_TO_ENGINE", "PHASES", "explain", "knobs_for"]
