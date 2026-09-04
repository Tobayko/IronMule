"""profile_for: pick the profile measured on *this* machine for *this* model.

The chain is append-only and holds every calibration a host ever ran, so
"newest row" and "the row that applies here" are different questions. These
tests are pure record selection -- no GPU, no model, no timing claim.
"""

from __future__ import annotations

import unittest

from friday_calibrate.profile import (
    PROFILE_KIND,
    DeviceProfile,
    KnobVerdict,
    newest_profile,
    profile_for,
)

MACHINE_A = "a" * 64
MACHINE_B = "b" * 64


def _profile(profile_id: str, *, model_id: str, machine: str | None, aa_noise=0.003):
    return DeviceProfile(
        profile_id=profile_id,
        model_id=model_id,
        model_revision="rev-" + model_id,
        hardware_sha256="c" * 64,
        environment_sha256="d" * 64,
        mde=0.006,
        aa_noise=aa_noise,
        machine_sha256=machine,
        knobs=(KnobVerdict("head_skip", "verified", 6, 0.877, 0.86, 0.89, True),),
    )


def _rows(*profiles):
    """The shape ``RuntimeHistory.verified_records`` hands back, oldest first."""

    return [
        {"record_kind": PROFILE_KIND, "report": profile.as_report(f"cal-{index}")}
        for index, profile in enumerate(profiles)
    ]


class ProfileLookupTest(unittest.TestCase):
    def test_the_newest_row_for_another_model_is_not_served(self) -> None:
        rows = _rows(
            _profile("p-12b", model_id="12b", machine=MACHINE_A),
            _profile("p-4b", model_id="4b", machine=MACHINE_A),
        )
        # What serving used to get, and why a 12B request fell back per request.
        self.assertEqual(newest_profile(rows).profile_id, "p-4b")
        self.assertEqual(
            profile_for(rows, machine_sha256=MACHINE_A, model_id="12b").profile_id,
            "p-12b",
        )

    def test_another_machines_profile_is_never_returned(self) -> None:
        rows = _rows(_profile("p-foreign", model_id="4b", machine=MACHINE_B))
        self.assertIsNone(profile_for(rows, machine_sha256=MACHINE_A, model_id="4b"))

    def test_an_uncalibrated_pair_is_a_clean_miss(self) -> None:
        rows = _rows(_profile("p-4b", model_id="4b", machine=MACHINE_A))
        self.assertIsNone(profile_for(rows, machine_sha256=MACHINE_A, model_id="27b"))

    def test_a_profile_recorded_before_the_field_existed_still_serves(self) -> None:
        # ``machine_sha256=None`` is how every row written before the field looks;
        # scope.in_calibrated_scope accepts those, so the lookup must too.
        rows = _rows(_profile("p-legacy", model_id="4b", machine=None))
        self.assertEqual(
            profile_for(rows, machine_sha256=MACHINE_A, model_id="4b").profile_id,
            "p-legacy",
        )

    def test_a_verified_knob_without_a_noise_measurement_is_skipped(self) -> None:
        # The fabrication guard newest_profile already had; it has to survive the
        # extra filtering, or a single-shot row becomes reachable via the model.
        rows = _rows(
            _profile("p-measured", model_id="4b", machine=MACHINE_A),
            _profile("p-fabricated", model_id="4b", machine=MACHINE_A, aa_noise=None),
        )
        self.assertEqual(
            profile_for(rows, machine_sha256=MACHINE_A, model_id="4b").profile_id,
            "p-measured",
        )

    def test_without_a_model_the_newest_row_for_this_machine_wins(self) -> None:
        rows = _rows(
            _profile("p-4b", model_id="4b", machine=MACHINE_A),
            _profile("p-other-host", model_id="12b", machine=MACHINE_B),
        )
        self.assertEqual(
            profile_for(rows, machine_sha256=MACHINE_A).profile_id, "p-4b"
        )


if __name__ == "__main__":
    unittest.main()
