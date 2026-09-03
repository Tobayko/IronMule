"""The device profile: what this machine verified, replacing what was frozen elsewhere.

The sealed runtimes decide whether an optimised path may run by comparing the
host against constants written on one machine at one moment
(``friday_runtime_n10/policy.py:186-199``). That is not a weak check, it is the
wrong check: it asks "are you Tobias' M1 Max as it was in August", and the
answer went to ``False`` on that very machine when macOS updated.

A device profile asks the question the evidence actually supports: *was this
knob verified as token-identical on this device, against this model snapshot?*
It is produced by a gated calibration run, stored in an append-only hash chain,
and it is the only thing that authorises a knob at serving time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from friday_evidence.canonical import canonical_sha256
from friday_runtime_core.history import HistorySpec

SCHEMA_VERSION = 1
RUNTIME_ID = "friday-device-profile-v1"
PROFILE_KIND = "device_profile"
FAILURE_KIND = "runtime_failure"

HISTORY = HistorySpec(
    runtime_id=RUNTIME_ID,
    kinds=frozenset({PROFILE_KIND, FAILURE_KIND}),
)

#: The knobs a calibration run may verify. Closed on purpose: a knob that is not
#: in this tuple cannot be turned on by a profile, whatever a profile claims.
CALIBRATED_KNOBS = ("head_skip", "fixed_compiled", "prefill_step_size", "bundled_readback", "fuse_projections")

#: A knob is either shown to preserve tokens on this device, shown not to, or
#: was not applicable here. There is no fourth, softer verdict.
VERDICTS = ("verified", "failed", "not_applicable")

#: Which phase each knob acts on. Dispatch is per phase, not per process:
#: prefill is compute-bound, decode is bandwidth-bound, and a knob that helps
#: one says nothing about the other.
KNOB_PHASE = {
    "head_skip": "prefill",
    "prefill_step_size": "prefill",
    "fixed_compiled": "decode",
    "bundled_readback": "decode",
    "fuse_projections": "decode",
}


class ProfileError(ValueError):
    """A device profile is malformed or claims something it did not measure."""


#: General promotion bar: a knob must show at least 5 % statistically confirmed gain
#: (interval wholly below 0.95) to qualify for general promotion.
PROMOTION_MAX_CI_HIGH: float = 0.95

#: Knobs authorized under the serving bar (interval wholly below 1.0 + exact token identity)
#: rather than the 5 % end-to-end promotion threshold, because decode knobs on short prompts
#: act only on the ~20 % decode share and cannot mathematically achieve 5 % end-to-end
#: despite being verified in their own phase.
#:
#: ``fuse_projections`` was removed 2026-09-03: it breaks token identity on bf16/4-bit
#: (candidate_correctness_failed) so it can never be ``verified`` under any bar, and no
#: user decision ever authorised it. ``fixed_compiled`` stays for now (Cycle 16 measured
#: a 7 % decode gain under this bar) but whether the weaker bar generalises past
#: ``bundled_readback`` (user decision D4, 2026-09-02) is open — see BACKLOG D4b.
SERVING_ONLY_KNOBS: frozenset[str] = frozenset({"bundled_readback", "fixed_compiled"})


@dataclass(frozen=True)
class KnobVerdict:
    """One knob, one verdict, and the evidence that produced it.

    Under D4b, the bar for ``verified`` requires reaching the promotion threshold
    (``ci_high < 0.95``), EXCEPT for named knobs in ``SERVING_ONLY_KNOBS``
    (such as ``bundled_readback``, user decision D4 from 2026-09-02) which are
    authorized under the serving bar (``ci_high < 1.0``).
    """

    knob: str
    verdict: str
    pairs: int = 0
    ratio: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    token_identical: bool = False
    reason: str = ""

    def __post_init__(self) -> None:
        if self.knob not in CALIBRATED_KNOBS:
            raise ProfileError(f"unregistered knob: {self.knob}")
        if self.verdict not in VERDICTS:
            raise ProfileError(f"unregistered verdict: {self.verdict}")
        if isinstance(self.pairs, bool) or not isinstance(self.pairs, int) or self.pairs < 0:
            raise ProfileError("pairs must be a non-negative integer")
        for name in ("ratio", "ci_low", "ci_high"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ProfileError(f"{name} must be a number or None")
            if not 0.0 < float(value) < 10.0:
                raise ProfileError(f"{name} is outside the plausible ratio range")
        if type(self.token_identical) is not bool:
            raise ProfileError("token_identical must be a bool")
        if self.verdict == "verified":
            if not self.token_identical:
                raise ProfileError(f"{self.knob}: verified requires token identity")
            if self.ratio is None or self.ci_high is None:
                raise ProfileError(f"{self.knob}: verified requires a ratio and an interval")
            threshold = 1.0 if self.knob in SERVING_ONLY_KNOBS else PROMOTION_MAX_CI_HIGH
            if self.ci_high >= threshold:
                raise ProfileError(
                    f"{self.knob}: verified requires an interval wholly below {threshold}"
                )
            if self.pairs < 1:
                raise ProfileError(f"{self.knob}: verified requires at least one pair")

    @property
    def phase(self) -> str:
        return KNOB_PHASE[self.knob]

    def as_dict(self) -> dict[str, Any]:
        return {
            "knob": self.knob,
            "phase": self.phase,
            "verdict": self.verdict,
            "pairs": self.pairs,
            "ratio": self.ratio,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "token_identical": self.token_identical,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "KnobVerdict":
        if not isinstance(value, Mapping):
            raise ProfileError("knob verdict must be an object")
        return cls(
            knob=value.get("knob", ""),
            verdict=value.get("verdict", ""),
            pairs=value.get("pairs", 0),
            ratio=value.get("ratio"),
            ci_low=value.get("ci_low"),
            ci_high=value.get("ci_high"),
            token_identical=bool(value.get("token_identical", False)),
            reason=str(value.get("reason", "")),
        )


@dataclass(frozen=True)
class DeviceProfile:
    """One device, one model snapshot, one calibration run."""

    profile_id: str
    model_id: str
    model_revision: str
    hardware_sha256: str
    environment_sha256: str
    mde: float
    knobs: tuple[KnobVerdict, ...]
    width_curve: Mapping[int, float] = field(default_factory=dict)
    roofline: Mapping[str, Any] = field(default_factory=dict)
    aa_noise: float | None = None
    #: Digest over the stable host identity (CPU/model/memory/arch, not macOS).
    #: ``None`` on profiles recorded before this field existed; serving skips the
    #: host check for those.
    machine_sha256: str | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("profile_id", "model_id", "model_revision"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ProfileError(f"{name} must be a non-empty string")
        for name in ("hardware_sha256", "environment_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise ProfileError(f"{name} must be a sha256 digest")
        if self.machine_sha256 is not None and (
            not isinstance(self.machine_sha256, str) or len(self.machine_sha256) != 64
        ):
            raise ProfileError("machine_sha256 must be a sha256 digest or None")
        if isinstance(self.mde, bool) or not isinstance(self.mde, (int, float)):
            raise ProfileError("mde must be a number")
        if not 0.0 <= float(self.mde) < 1.0:
            raise ProfileError("mde must be a fraction within [0, 1)")
        seen = [verdict.knob for verdict in self.knobs]
        if len(seen) != len(set(seen)):
            raise ProfileError("a knob may carry at most one verdict")
        if any(not isinstance(verdict, KnobVerdict) for verdict in self.knobs):
            raise ProfileError("knobs must be KnobVerdict values")
        for width, value in self.width_curve.items():
            # Width 0 is the "speculation off" action and a real point on the
            # curve — often the winning one, which is the whole reason the
            # bandit exists.
            if isinstance(width, bool) or not isinstance(width, int) or width < 0:
                raise ProfileError("width curve keys must be non-negative integers")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ProfileError("width curve values must be positive numbers")

    # -- queries --------------------------------------------------------------
    def verdict_for(self, knob: str) -> KnobVerdict | None:
        return next((item for item in self.knobs if item.knob == knob), None)

    def is_verified(self, knob: str) -> bool:
        """The single question serving asks. Anything but ``verified`` is off."""

        verdict = self.verdict_for(knob)
        return verdict is not None and verdict.verdict == "verified"

    def verified_knobs(self, phase: str | None = None) -> tuple[str, ...]:
        return tuple(
            item.knob
            for item in self.knobs
            if item.verdict == "verified" and (phase is None or item.phase == phase)
        )

    def unverified(self) -> tuple[str, ...]:
        return tuple(knob for knob in CALIBRATED_KNOBS if not self.is_verified(knob))

    # -- serialisation --------------------------------------------------------
    def as_report(self, run_id: str) -> dict[str, Any]:
        body = {
            "schema_version": self.schema_version,
            "runtime_id": RUNTIME_ID,
            "kind": PROFILE_KIND,
            "run_id": run_id,
            "status": "device_profile_recorded",
            "profile_id": self.profile_id,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "hardware_sha256": self.hardware_sha256,
            "environment_sha256": self.environment_sha256,
            "mde": float(self.mde),
            "aa_noise": None if self.aa_noise is None else float(self.aa_noise),
            "knobs": [verdict.as_dict() for verdict in self.knobs],
            "width_curve": {str(width): float(value) for width, value in self.width_curve.items()},
            "roofline": dict(self.roofline),
            "formal_claim": False,
        }
        if self.machine_sha256 is not None:
            body["machine_sha256"] = self.machine_sha256
        body["profile_sha256"] = canonical_sha256(body)
        return body

    @classmethod
    def from_report(cls, report: Mapping[str, Any]) -> "DeviceProfile":
        if not isinstance(report, Mapping) or report.get("kind") != PROFILE_KIND:
            raise ProfileError("report is not a device profile")
        body = {key: value for key, value in report.items() if key != "profile_sha256"}
        if report.get("profile_sha256") != canonical_sha256(body):
            raise ProfileError("device profile digest does not replay")
        knobs = report.get("knobs")
        if not isinstance(knobs, Sequence) or isinstance(knobs, (str, bytes)):
            raise ProfileError("device profile knobs must be a list")
        curve = report.get("width_curve") or {}
        if not isinstance(curve, Mapping):
            raise ProfileError("width curve must be an object")
        try:
            width_curve = {int(width): float(value) for width, value in curve.items()}
        except (TypeError, ValueError) as exc:
            raise ProfileError("width curve is malformed") from exc
        return cls(
            profile_id=report.get("profile_id", ""),
            model_id=report.get("model_id", ""),
            model_revision=report.get("model_revision", ""),
            hardware_sha256=report.get("hardware_sha256", ""),
            environment_sha256=report.get("environment_sha256", ""),
            mde=report.get("mde", 0.0),
            knobs=tuple(KnobVerdict.from_dict(item) for item in knobs),
            width_curve=width_curve,
            roofline=report.get("roofline") or {},
            aa_noise=report.get("aa_noise"),
            machine_sha256=report.get("machine_sha256"),
            schema_version=report.get("schema_version", SCHEMA_VERSION),
        )


def newest_profile(records: Iterable[Mapping[str, Any]]) -> DeviceProfile | None:
    """The newest *serviceable* profile in a verified chain, or ``None``.

    A row is skipped, not raised on, when it fails to replay
    (:class:`ProfileError`) or when it claims a verified knob without an A/A
    noise measurement. The real calibrator (``friday_calibrate.runner``) always
    records ``aa_noise`` because ``noise_mde`` runs first and raises rather than
    returning ``None``; a verified knob with ``aa_noise`` absent is the mark of
    a single-shot fabrication, and serving must fall back past it to the last
    row that was actually measured the way ``DEVICE_PROFILE_SPEC`` requires.
    """

    for row in reversed(list(records)):
        if row.get("record_kind") != PROFILE_KIND:
            continue
        try:
            profile = DeviceProfile.from_report(row["report"])
        except ProfileError:
            continue
        if profile.aa_noise is None and profile.verified_knobs():
            continue
        return profile
    return None


__all__ = [
    "CALIBRATED_KNOBS",
    "FAILURE_KIND",
    "HISTORY",
    "KNOB_PHASE",
    "PROFILE_KIND",
    "RUNTIME_ID",
    "SCHEMA_VERSION",
    "VERDICTS",
    "DeviceProfile",
    "KnobVerdict",
    "ProfileError",
    "newest_profile",
]
