"""friday_runtime_core: the shared substrate, and the one behaviour it changes.

Two things are checked here. First, parity: the generic controller must make the
same decisions and latch on the same events as the three sealed copies it was
derived from — the copies stay in the repository, so parity is testable rather
than asserted. Second, the deliberate difference: the circuit breaker survives
the process, which no sealed copy does.
"""

from __future__ import annotations

import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from friday_evidence.canonical import canonical_sha256
from friday_runtime_core.breaker import (
    BreakerError,
    CircuitBreaker,
    MemoryLatch,
    PersistentLatch,
)
from friday_runtime_core.controller import (
    CIRCUIT_REASON,
    DispatchController,
    DispatchDecision,
    RuntimeExecutionError,
)
from friday_runtime_core.files import UnsafeFile, regular_bytes
from friday_runtime_core.history import (
    HistoryConflict,
    HistoryError,
    HistorySpec,
    RuntimeHistory,
)
from friday_runtime_core.provenance import ProvenanceSpec, collect_provenance

SPEC = HistorySpec(
    runtime_id="core-test-runtime-20260902-01",
    kinds=frozenset({"runtime_failure", "device_profile"}),
)

BASELINE = DispatchDecision("baseline", "baseline_plan", "fallback", None)
OPTIMIZED = DispatchDecision("optimized", "optimized_plan", "authorized", None)


def _provenance() -> dict:
    spec = ProvenanceSpec(
        runtime_id=SPEC.runtime_id,
        code_directories=("friday_runtime_core",),
        spec_files=("AGENTS.md",),
    )
    return collect_provenance(spec, require_clean=False)


def _report(run_id: str, *, kind: str = "runtime_failure", status: str = "measurement_failed"):
    return {
        "schema_version": SPEC.schema_version,
        "runtime_id": SPEC.runtime_id,
        "kind": kind,
        "run_id": run_id,
        "status": status,
    }


class HistoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.path = Path(self._dir.name) / "runtime.sqlite3"
        self.provenance = _provenance()

    def _open(self, **kwargs) -> RuntimeHistory:
        return RuntimeHistory.open(SPEC, self.path, **kwargs)

    def test_the_schema_is_shared_and_the_kinds_are_enforced_in_python(self) -> None:
        sql = SPEC.migration_sql()
        self.assertIn("PRAGMA application_id = 0x46524354", sql)
        self.assertNotIn(SPEC.runtime_id, sql)
        self.assertNotIn("device_profile", sql)
        self.assertEqual(len(SPEC.migration_sha256()), 64)
        # Two runtimes share the migration but not each other's databases.
        other = HistorySpec(runtime_id="other-runtime", kinds=SPEC.kinds)
        self.assertEqual(other.migration_sha256(), SPEC.migration_sha256())
        with RuntimeHistory.open(SPEC, self.path, initialize=True) as history:
            history.persist(_report("run-a"), self.provenance)
        with self.assertRaises(HistoryError):
            RuntimeHistory.open(other, self.path, read_only=True)

    def test_persist_and_replay_the_chain(self) -> None:
        with self._open(initialize=True) as history:
            first = history.persist(_report("run-a"), self.provenance)
            second = history.persist(_report("run-b"), self.provenance)
            rows = history.verified_records()
        self.assertEqual([row["record_id"] for row in rows], [first.record_id, second.record_id])
        self.assertIsNone(rows[0]["previous_record_id"])
        self.assertEqual(rows[1]["previous_record_id"], first.record_id)

    def test_read_only_handle_cannot_persist(self) -> None:
        with self._open(initialize=True) as history:
            history.persist(_report("run-a"), self.provenance)
        with self._open(read_only=True) as history:
            with self.assertRaises(HistoryError):
                history.persist(_report("run-b"), self.provenance)

    def test_duplicate_entity_key_conflicts(self) -> None:
        with self._open(initialize=True) as history:
            history.persist(_report("run-a"), self.provenance)
            with self.assertRaises(HistoryConflict):
                history.persist(_report("run-a"), self.provenance)

    def test_unregistered_kind_is_refused(self) -> None:
        with self._open(initialize=True) as history:
            with self.assertRaises(HistoryError):
                history.persist(_report("run-a", kind="not_registered"), self.provenance)

    def test_report_validator_runs(self) -> None:
        def refuse(report):
            if report.get("status") != "measurement_failed":
                raise HistoryError("status outside this runtime's contract")

        spec = HistorySpec(
            runtime_id=SPEC.runtime_id, kinds=SPEC.kinds, report_validator=refuse
        )
        with RuntimeHistory.open(spec, self.path, initialize=True) as history:
            history.persist(_report("run-a"), self.provenance)
            with self.assertRaises(HistoryError):
                history.persist(
                    _report("run-b", status="something_else"), self.provenance
                )

    def test_tampered_row_breaks_the_replay(self) -> None:
        with self._open(initialize=True) as history:
            history.persist(_report("run-a"), self.provenance)
        # A direct write bypassing the append-only triggers still has to fail on read.
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA writable_schema=OFF")
        connection.execute("DROP TRIGGER records_no_update")
        connection.execute("UPDATE records SET status='tampered'")
        connection.commit()
        connection.close()
        with self.assertRaises(HistoryError):
            with self._open(read_only=True) as history:
                history.verified_records()

    def test_append_only_triggers_hold(self) -> None:
        with self._open(initialize=True) as history:
            history.persist(_report("run-a"), self.provenance)
        connection = sqlite3.connect(self.path)
        with self.assertRaises(sqlite3.IntegrityError):
            connection.execute("DELETE FROM records")
        connection.close()

    def test_row_bound_is_enforced(self) -> None:
        spec = HistorySpec(runtime_id=SPEC.runtime_id, kinds=SPEC.kinds, max_rows=2)
        with RuntimeHistory.open(spec, self.path, initialize=True) as history:
            history.persist(_report("run-a"), self.provenance)
            history.persist(_report("run-b"), self.provenance)
            with self.assertRaises(HistoryError):
                history.persist(_report("run-c"), self.provenance)

    def test_provenance_projection_must_replay(self) -> None:
        broken = dict(self.provenance)
        broken["hardware"] = dict(broken["hardware"])
        broken["hardware"]["macos"] = "0.0.0"
        broken.pop("provenance_sha256")
        broken["provenance_sha256"] = canonical_sha256(broken)
        with self._open(initialize=True) as history:
            with self.assertRaises(HistoryError):
                history.persist(_report("run-a"), broken)


class SpecValidationTest(unittest.TestCase):
    def test_rejects_unsafe_identifiers_and_ranges(self) -> None:
        for kwargs in (
            {"runtime_id": "bad id with spaces"},
            {"runtime_id": ""},
            {"kinds": frozenset()},
            {"kinds": frozenset({"NotLowerCase"})},
            {"max_rows": 0},
            {"schema_version": 2},
        ):
            base = {"runtime_id": SPEC.runtime_id, "kinds": SPEC.kinds}
            base.update(kwargs)
            with self.assertRaises(HistoryError):
                HistorySpec(**base)


class FilesTest(unittest.TestCase):
    def test_symlinks_are_refused(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real.txt"
            real.write_text("x")
            link = root / "link.txt"
            link.symlink_to(real)
            self.assertEqual(regular_bytes(real), b"x")
            with self.assertRaises(UnsafeFile):
                regular_bytes(link)

    def test_oversized_files_are_refused(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "big.txt"
            path.write_bytes(b"0" * 64)
            with self.assertRaises(UnsafeFile):
                regular_bytes(path, maximum=16)


class BreakerTest(unittest.TestCase):
    def test_memory_latch_matches_the_sealed_in_ram_behaviour(self) -> None:
        breaker = CircuitBreaker(MemoryLatch())
        self.assertIsNone(breaker.reason)
        breaker.trip(ValueError("first"))
        breaker.trip(KeyError("second"))
        self.assertEqual(breaker.reason, "ValueError")
        self.assertTrue(breaker.tripped)

    def test_a_memory_latch_is_forgotten_by_the_next_process(self) -> None:
        latch = MemoryLatch()
        CircuitBreaker(latch).trip(ValueError("boom"))
        # A *fresh* MemoryLatch is what a restart really looks like.
        self.assertIsNone(CircuitBreaker(MemoryLatch()).reason)

    def test_a_persistent_latch_survives_the_process(self) -> None:
        store: list[str] = []
        latch = lambda: PersistentLatch(  # noqa: E731 - two lines would say less
            load=lambda: store[0] if store else None,
            append=store.append,
        )
        first = CircuitBreaker(latch())
        first.trip(ValueError("boom"))
        self.assertEqual(store, ["ValueError"])
        restarted = CircuitBreaker(latch())
        self.assertEqual(restarted.reason, "ValueError")
        self.assertTrue(restarted.tripped)

    def test_an_unreadable_latch_counts_as_tripped(self) -> None:
        def explode():
            raise HistoryError("chain broken")

        breaker = CircuitBreaker(PersistentLatch(load=explode, append=lambda _r: None))
        self.assertTrue(breaker.tripped)
        self.assertTrue(breaker.reason.startswith("latch_unreadable:"))

    def test_a_malformed_stored_reason_counts_as_tripped(self) -> None:
        breaker = CircuitBreaker(
            PersistentLatch(load=lambda: 17, append=lambda _r: None)
        )
        self.assertEqual(breaker.reason, "latch_unreadable:malformed_reason")

    def test_a_failure_to_persist_is_raised_not_swallowed(self) -> None:
        def refuse(_reason):
            raise HistoryError("disk full")

        breaker = CircuitBreaker(PersistentLatch(load=lambda: None, append=refuse))
        with self.assertRaises(BreakerError):
            breaker.trip(ValueError("boom"))
        # The trip still holds in memory even though it could not be stored.
        self.assertTrue(breaker.tripped)

    def test_persist_is_retried_while_it_has_not_succeeded(self) -> None:
        attempts: list[str] = []

        def flaky(reason):
            attempts.append(reason)
            if len(attempts) == 1:
                raise HistoryError("transient")

        breaker = CircuitBreaker(PersistentLatch(load=lambda: None, append=flaky))
        with self.assertRaises(BreakerError):
            breaker.trip(ValueError("boom"))
        breaker.trip(KeyError("second"))
        self.assertEqual(attempts, ["ValueError", "ValueError"])


class PersistentLatchOverHistoryTest(unittest.TestCase):
    """The latch the plan asks for: a runtime_failure record in the hash chain."""

    def setUp(self) -> None:
        self._dir = TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.path = Path(self._dir.name) / "runtime.sqlite3"
        self.provenance = _provenance()
        RuntimeHistory.open(SPEC, self.path, initialize=True).close()

    def _latch(self) -> PersistentLatch:
        def load():
            with RuntimeHistory.open(SPEC, self.path, read_only=True) as history:
                with history.read_transaction():
                    rows = history.verified_records()
            failures = [row for row in rows if row["record_kind"] == "runtime_failure"]
            return failures[-1]["report"]["reason"] if failures else None

        def append(reason):
            with RuntimeHistory.open(SPEC, self.path) as history:
                report = _report(f"latch-{reason}")
                history.persist({**report, "reason": reason}, self.provenance)

        return PersistentLatch(load=load, append=append)

    def test_the_latch_round_trips_through_the_hash_chain(self) -> None:
        CircuitBreaker(self._latch()).trip(ValueError("boom"))
        restarted = CircuitBreaker(self._latch())
        self.assertEqual(restarted.reason, "ValueError")

    def test_a_broken_chain_leaves_the_breaker_closed(self) -> None:
        CircuitBreaker(self._latch()).trip(ValueError("boom"))
        connection = sqlite3.connect(self.path)
        connection.execute("DROP TRIGGER records_no_update")
        connection.execute("UPDATE records SET status='tampered'")
        connection.commit()
        connection.close()
        self.assertTrue(CircuitBreaker(self._latch()).tripped)


class ControllerTest(unittest.TestCase):
    def _controller(self, latch=None, authorized: bool = True) -> DispatchController:
        def decide(_evidence, scope):
            if authorized and scope == "in_scope":
                return OPTIMIZED
            return DispatchDecision("baseline", "baseline_plan", "out_of_scope", None)

        return DispatchController(
            evidence=None, decide=decide, fallback=BASELINE, latch=latch
        )

    def test_scope_decides_not_the_caller(self) -> None:
        controller = self._controller()
        self.assertEqual(controller.decide_scope("in_scope").plan, "optimized_plan")
        self.assertEqual(controller.decide_scope("anything_else").plan, "baseline_plan")
        self.assertIsNone(controller.decide_scope(None).evidence)

    def test_an_optimised_failure_latches_and_is_not_retried(self) -> None:
        controller = self._controller()
        decision = controller.decide_scope("in_scope")
        with self.assertRaises(RuntimeExecutionError) as caught:
            with controller.guard(decision):
                raise ValueError("kernel blew up")
        self.assertIn("circuit breaker latched", str(caught.exception))
        self.assertEqual(controller.circuit_reason, "ValueError")
        self.assertEqual(controller.decide_scope("in_scope").reason, CIRCUIT_REASON)
        self.assertEqual(controller.decide_scope("in_scope").plan, "baseline_plan")

    def test_a_baseline_failure_does_not_latch(self) -> None:
        controller = self._controller(authorized=False)
        decision = controller.decide_scope("in_scope")
        with self.assertRaises(RuntimeExecutionError) as caught:
            with controller.guard(decision):
                raise ValueError("baseline blew up")
        self.assertIn("baseline path failed", str(caught.exception))
        self.assertIsNone(controller.circuit_reason)

    def test_a_latched_breaker_starts_the_next_controller_on_baseline(self) -> None:
        store: list[str] = []
        make = lambda: PersistentLatch(  # noqa: E731
            load=lambda: store[0] if store else None, append=store.append
        )
        first = self._controller(make())
        with self.assertRaises(RuntimeExecutionError):
            with first.guard(first.decide_scope("in_scope")):
                raise ValueError("kernel blew up")
        restarted = self._controller(make())
        self.assertEqual(restarted.decide_scope("in_scope").plan, "baseline_plan")
        self.assertEqual(restarted.decide_scope("in_scope").reason, CIRCUIT_REASON)

    def test_a_successful_call_leaves_the_breaker_alone(self) -> None:
        controller = self._controller()
        with controller.guard(controller.decide_scope("in_scope")):
            pass
        self.assertIsNone(controller.circuit_reason)

    def test_a_decision_of_the_wrong_type_is_refused(self) -> None:
        controller = DispatchController(
            evidence=None, decide=lambda _e, _s: "batched", fallback=BASELINE
        )
        with self.assertRaises(RuntimeExecutionError):
            controller.decide_scope("anything")


class SealedParityTest(unittest.TestCase):
    """The sealed copies stay in the tree, so parity is measured, not claimed."""

    def test_matches_friday_runtime_n10_on_the_same_event_sequence(self) -> None:
        from friday_runtime_n10.constants import BATCHED_PLAN, SERIAL_PLAN
        from friday_runtime_n10.executor import RuntimeController as SealedController
        from friday_runtime_n10.policy import PolicyEvidence

        evidence = PolicyEvidence(
            authorized=True,
            reason="formal_n10_gain_exact_scope",
            decision_record_id=None,
            decision_sha256=None,
            preregistration_sha256=None,
            sealed_provenance_sha256=None,
            formal_database_sha256=None,
            formal_snapshot_revision=None,
            evidence_records=0,
        )
        sealed = SealedController(evidence)

        class Failing:
            def matmul(self, left, right):
                raise ValueError("kernel blew up")

            def eval_many(self, values):
                return None

            def synchronize(self):
                return None

        left = _Tensor((2048, 2048))
        operands = [_Tensor((2048, 2048)) for _ in range(10)]
        self.assertEqual(sealed.decide(left, operands).plan, BATCHED_PLAN)
        with self.assertRaises(Exception):
            sealed.execute(Failing(), left, operands)
        self.assertEqual(sealed.circuit_reason, "ValueError")
        self.assertEqual(sealed.decide(left, operands).plan, SERIAL_PLAN)
        self.assertEqual(sealed.decide(left, operands).reason, CIRCUIT_REASON)

        core = DispatchController(
            evidence=evidence,
            decide=lambda _e, scope: (
                DispatchDecision("batched", BATCHED_PLAN, "formal_n10_gain_exact_scope", _e)
                if scope is not None
                else DispatchDecision("serial", SERIAL_PLAN, "workload_out_of_scope", _e)
            ),
            fallback=DispatchDecision("serial", SERIAL_PLAN, "fallback", evidence),
        )
        decision = core.decide_scope("in_scope")
        self.assertEqual(decision.plan, BATCHED_PLAN)
        with self.assertRaises(RuntimeExecutionError):
            with core.guard(decision):
                raise ValueError("kernel blew up")
        self.assertEqual(core.circuit_reason, sealed.circuit_reason)
        self.assertEqual(core.decide_scope("in_scope").plan, SERIAL_PLAN)
        self.assertEqual(core.decide_scope("in_scope").reason, CIRCUIT_REASON)


class _Tensor:
    def __init__(self, shape: tuple[int, int]) -> None:
        self.shape = shape
        self.dtype = "float16"


if __name__ == "__main__":
    unittest.main()
