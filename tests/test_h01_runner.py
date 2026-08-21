"""Offline verification of the controlled H0.1 execution path.

The whole path — parent binding, pacing, measurement, trace, analysis, bundle
persistence and the six-session study — runs here without MLX, Metal, a GPU or
any real waiting.  A virtual clock advances only when the simulated run sleeps
or measures, so a six-session study that takes about ten minutes on the target
device completes in well under a second.
"""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from typing import Any

from friday_h01 import runner
from friday_h01.constants import (
    BURN_IN_SAMPLES,
    COOLDOWN_NS,
    LONG_GAP_NS,
    MAX_GAP_OVERSHOOT_NS,
    SESSION_ORDER,
    SHORT_GAP_NS,
    SHORT_LABEL,
    TOTAL_SAMPLES,
)
from friday_h01.storage import Storage

H0_DB = Path(".friday-data/h0.sqlite3")

# Run22 measured single evaluations in the low-millisecond band; the simulation
# stays in that band so the offline path exercises realistic integer magnitudes.
BASE_DURATION_NS = 2_050_000
JITTER_SPAN_NS = 60_000
OVERSHOOT_SPAN_NS = 900_000


def _deterministic(label: str, index: int, span: int) -> int:
    """Uncorrelated, reproducible pseudo-noise; no library RNG, no seed drift."""

    digest = hashlib.sha256(f"{label}:{index}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") % span


class VirtualClock:
    """A monotonic nanosecond clock that only moves when the run makes it move."""

    def __init__(self) -> None:
        self.now = 1_000_000_000

    def __call__(self) -> int:
        return self.now

    def advance(self, delta_ns: int) -> None:
        if delta_ns < 0:
            raise AssertionError("virtual clock may never move backwards")
        self.now += delta_ns


class SimulatedBackend:
    """Stationary stand-in for the MLX matmul; charges its cost to the clock."""

    def __init__(self, clock: VirtualClock, *, label: str) -> None:
        self.clock = clock
        self.label = label
        self.evaluations = 0
        self.synchronizations = 0

    def from_host(self, value: Any) -> Any:
        return value

    def matmul(self, a: Any, b: Any) -> Any:
        return (a, b)

    def eval(self, value: Any) -> Any:
        duration = BASE_DURATION_NS + _deterministic(
            f"{self.label}.duration", self.evaluations, JITTER_SPAN_NS
        )
        self.evaluations += 1
        self.clock.advance(duration)
        return value

    def synchronize(self) -> None:
        self.synchronizations += 1


class SimulatedSleeper:
    """Sleeps by advancing the virtual clock, with a bounded scheduling overshoot."""

    def __init__(self, clock: VirtualClock, *, label: str) -> None:
        self.clock = clock
        self.label = label
        self.calls = 0

    def __call__(self, seconds: float) -> None:
        requested = int(round(seconds * 1_000_000_000))
        overshoot = _deterministic(f"{self.label}.gap", self.calls, OVERSHOOT_SPAN_NS)
        self.calls += 1
        self.clock.advance(requested + overshoot)


def _simulated_fixture(_seed: int) -> tuple[Any, Any]:
    """Skip the 2048^2 host materialization; identity is asserted separately."""

    return ("fixture-a", "fixture-b")


class PacingTest(unittest.TestCase):
    def test_pace_never_undershoots_the_registered_gap(self) -> None:
        clock = VirtualClock()
        sleeper = SimulatedSleeper(clock, label="pace")
        for target in (SHORT_GAP_NS, LONG_GAP_NS, COOLDOWN_NS):
            with self.subTest(target=target):
                elapsed = runner._pace(target, sleeper, clock)
                self.assertGreaterEqual(elapsed, target)
                self.assertLessEqual(elapsed - target, MAX_GAP_OVERSHOOT_NS)

    def test_pace_returns_immediately_when_the_gap_already_elapsed(self) -> None:
        clock = VirtualClock()

        def eager_sleeper(seconds: float) -> None:
            raise AssertionError("an already elapsed gap must not sleep")

        self.assertEqual(runner._pace(0, eager_sleeper, clock), 0)

    def test_measure_once_rejects_a_stalled_clock(self) -> None:
        clock = VirtualClock()
        backend = SimulatedBackend(clock, label="stalled")
        backend.eval = lambda value: value  # charges nothing to the clock
        with self.assertRaises(runner.RunnerError):
            runner._measure_once(backend, "a", "b", clock)


class ParentBindingTest(unittest.TestCase):
    def test_parent_binds_a_verified_h0_eager_baseline(self) -> None:
        if not H0_DB.exists():
            self.skipTest("H0 evidence database is not present")
        parent = runner.select_h0_parent(H0_DB)
        self.assertEqual(parent.source["parent_phase"], "H0")
        self.assertTrue(parent.source["parent_run_id"].startswith("h0-eager_baseline-"))
        self.assertEqual(set(parent.fixture), {"a_sha256", "b_sha256", "metadata_sha256", "fixture_sha256"})
        for value in parent.fixture.values():
            self.assertRegex(value, r"\A[0-9a-f]{64}\Z")

    def test_unknown_parent_run_is_rejected(self) -> None:
        if not H0_DB.exists():
            self.skipTest("H0 evidence database is not present")
        with self.assertRaises(runner.RunnerError):
            runner.select_h0_parent(H0_DB, run_id="h0-eager_baseline-does-not-exist")

    def test_fixture_components_are_inherited_from_the_trusted_h0_registry(self) -> None:
        if not H0_DB.exists():
            self.skipTest("H0 evidence database is not present")
        from friday_h0.correctness_contract import trusted_performance_fixture_identity

        from friday_h01.canonical import canonical_sha256

        parent = runner.select_h0_parent(H0_DB)
        trusted = trusted_performance_fixture_identity(
            a_shape=[2048, 2048],
            b_shape=[2048, 2048],
            dtype="float16",
            layout="C-contiguous",
            fixture_seed=parent.fixture_seed,
        )
        components = ("a_sha256", "b_sha256", "metadata_sha256")
        for key in components:
            with self.subTest(key=key):
                self.assertEqual(parent.fixture[key], trusted[key])
        # H0.1 aggregates those components in its own canonical form, so its
        # fixture_sha256 is deliberately not H0's raw-byte fixture digest.
        self.assertEqual(
            parent.fixture["fixture_sha256"],
            canonical_sha256({key: parent.fixture[key] for key in components}),
        )
        self.assertNotEqual(parent.fixture["fixture_sha256"], trusted["fixture_sha256"])


class TelemetryTest(unittest.TestCase):
    def test_telemetry_is_a_closed_value_or_reason_pair(self) -> None:
        telemetry = runner.collect_telemetry()
        self.assertEqual(set(telemetry), {"thermal_state", "power_source"})
        for name, entry in telemetry.items():
            with self.subTest(name=name):
                self.assertEqual(set(entry), {"value", "missing_reason"})
                # Exactly one side of the pair is populated; 0 is never a stand-in.
                self.assertNotEqual(entry["value"] is None, entry["missing_reason"] is None)
                if entry["missing_reason"] is not None:
                    self.assertIn(
                        entry["missing_reason"],
                        {"not_collected", "api_unavailable", "not_applicable"},
                    )


class SessionExecutionTest(unittest.TestCase):
    def setUp(self) -> None:
        if not H0_DB.exists():
            self.skipTest("H0 evidence database is not present")
        self._patched_loader = runner.load_fixture_arrays
        runner.load_fixture_arrays = _simulated_fixture

    def tearDown(self) -> None:
        runner.load_fixture_arrays = self._patched_loader

    def _run_session(self, session_id: str, database: Path) -> runner.SessionOutcome:
        clock = VirtualClock()
        return runner.run_session(
            session_id,
            h0_database=H0_DB,
            h01_database=database,
            backend_factory=lambda: SimulatedBackend(clock, label=session_id),
            sleeper=SimulatedSleeper(clock, label=session_id),
            clock=clock,
        )

    def test_measure_session_records_every_registered_sample(self) -> None:
        clock = VirtualClock()
        parent = runner.select_h0_parent(H0_DB)
        from friday_h01.protocol import build_manifest
        from friday_h01.provenance import collect_provenance

        provenance = collect_provenance()
        manifest = build_manifest(
            "C0",
            fixture=parent.fixture,
            study_spec_sha256=provenance.study_spec_sha256,
            code_sha256=provenance.code_sha256,
            environment_sha256=provenance.environment_sha256,
            source=parent.source,
        )
        recorded = runner.measure_session(
            "C0",
            manifest,
            backend=SimulatedBackend(clock, label="C0"),
            a="a",
            b="b",
            sleeper=SimulatedSleeper(clock, label="C0"),
            clock=clock,
        )
        self.assertEqual(len(recorded["durations_ns"]), TOTAL_SAMPLES)
        self.assertEqual(len(recorded["gap_overshoots_ns"]), TOTAL_SAMPLES)
        self.assertTrue(all(value > 0 for value in recorded["durations_ns"]))
        self.assertTrue(
            all(0 <= value <= MAX_GAP_OVERSHOOT_NS for value in recorded["gap_overshoots_ns"])
        )
        self.assertGreaterEqual(recorded["observed_cooldown_ns"], COOLDOWN_NS)

    def test_session_persists_a_complete_replayable_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "h01.sqlite3"
            outcome = self._run_session("C0", database)
            self.assertEqual(outcome.status, "h01_session_complete")
            self.assertEqual(outcome.persistence_state, "inserted")
            self.assertEqual(outcome.session_id, "C0")
            self.assertRegex(outcome.bundle_sha256, r"\A[0-9a-f]{64}\Z")
            with Storage.open(database, read_only=True) as storage:
                self.assertEqual(storage.counts_by("entity_kind"), {"paced_session": 1})

    def test_remeasuring_a_recorded_session_is_refused(self) -> None:
        from friday_h01.storage import StorageConflict

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "h01.sqlite3"
            self._run_session("C0", database)
            # A session's run_id is fixed by its provenance, but a second
            # measurement carries different durations and a different creation
            # time.  The append-only store must refuse it rather than silently
            # replace the recorded evidence: a retry is a new study, not a patch.
            with self.assertRaises(StorageConflict):
                self._run_session("C0", database)
            with Storage.open(database, read_only=True) as storage:
                self.assertEqual(storage.count(), 1)

    def test_unregistered_session_id_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(runner.RunnerError):
                self._run_session("C9", Path(directory) / "h01.sqlite3")

    def test_study_requires_all_six_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "h01.sqlite3"
            for session_id in SESSION_ORDER[:5]:
                self._run_session(session_id, database)
            with self.assertRaises(runner.RunnerError) as caught:
                runner.run_study(h01_database=database)
            self.assertIn("V2", str(caught.exception))

    def test_six_sessions_produce_one_terminal_study(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "h01.sqlite3"
            for session_id in SESSION_ORDER:
                outcome = self._run_session(session_id, database)
                self.assertEqual(outcome.status, "h01_session_complete")
            study = runner.run_study(h01_database=database)
            self.assertIn(
                study.status, {"h01_stationarity_supported", "h01_complete_unresolved"}
            )
            self.assertTrue(study.study_id.startswith("h01-study-"))
            self.assertEqual(study.persistence_state, "inserted")
            with Storage.open(database, read_only=True) as storage:
                self.assertEqual(
                    storage.counts_by("entity_kind"),
                    {"paced_session": 6, "paced_study": 1},
                )

    def test_study_replays_the_persisted_sessions_independently(self) -> None:
        from friday_h01.study import analyze_study, validate_study_result

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "h01.sqlite3"
            for session_id in SESSION_ORDER:
                self._run_session(session_id, database)
            runner.run_study(h01_database=database)
            with Storage.open(database, read_only=True) as storage:
                with storage.read_transaction():
                    rows = storage.verified_rows()
            studies = [
                row["bundle"] for row in rows if row["bundle"]["entity_kind"] == "paced_study"
            ]
            self.assertEqual(len(studies), 1)
            records = studies[0]["manifest"]["session_records"]
            # The stored decision must survive an independent recomputation.
            self.assertEqual(
                validate_study_result(studies[0]["result"], records),
                analyze_study(records),
            )


class CliGateTest(unittest.TestCase):
    def test_every_command_is_locked_without_the_release_flag(self) -> None:
        from friday_h01 import cli

        for argv in (
            ["preflight"],
            ["session", "--id", "C0"],
            ["study"],
            ["run-all"],
        ):
            with self.subTest(argv=argv):
                self.assertEqual(cli.main(argv), cli.EXIT_MLX_LOCKED)

    def test_unregistered_session_id_is_a_usage_error(self) -> None:
        from friday_h01 import cli

        self.assertEqual(cli.main(["session", "--id", "C9", "--execute"]), cli.EXIT_USAGE)


if __name__ == "__main__":
    unittest.main()
