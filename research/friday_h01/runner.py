"""Controlled H0.1 live execution path for the preregistered six-session study.

This module is the only place in H0.1 that touches hardware.  It closes the gap
between the frozen contract (schedule, protocol, analysis, study) and a real
paced measurement:

    parent binding -> preflight -> paced session -> trace -> analysis -> bundle

Every threshold, seed, sample count and gate lives in :mod:`friday_h01.constants`
and is never read, adjusted or re-derived here.  The runner only executes the
already registered plan and records what actually happened.

The backend is injected, so the whole path is exercisable offline without MLX,
Metal, or a GPU.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from friday_h0.protocol import PRODUCTION_MANIFEST_BYTES, ProtocolError as H0ProtocolError, close_manifest
from friday_h0.storage import (
    PERSISTENCE_MAX_BUNDLE_BYTES,
    Storage as H0Storage,
    StorageError as H0StorageError,
)

from .analysis import analyze_trace
from .canonical import canonical_sha256
from .constants import (
    BURN_IN_SAMPLES,
    COOLDOWN_NS,
    MAX_GAP_OVERSHOOT_NS,
    SESSION_COMPLETE_STATUS,
    SESSION_ORDER,
    TOTAL_SAMPLES,
)
from .import_h0 import _parse_stored_json, _mapping, _array
from .protocol import build_manifest, build_trace
from .provenance import collect_provenance
from .storage import DEFAULT_H01_DB_PATH, Storage, StorageError
from .study import analyze_study

DEFAULT_H0_DB_PATH = Path(".friday-data/h0.sqlite3")

# A single evaluated 2048^2 FP16 matmul is a low-millisecond operation on the
# target device.  These caps only catch a hung or wildly degraded device; they
# are not performance gates and never enter the analysis.
SAMPLE_TIMEOUT_NS = 10_000_000_000
PREFLIGHT_PROBE_SAMPLES = 3


class RunnerError(RuntimeError):
    """Raised when the controlled execution path must fail closed."""


class Backend(Protocol):
    """Minimal injectable measurement backend."""

    def from_host(self, value: Any) -> Any: ...

    def matmul(self, a: Any, b: Any) -> Any: ...

    def eval(self, value: Any) -> Any: ...

    def synchronize(self) -> None: ...


@dataclass(frozen=True)
class ParentBinding:
    """The verified H0 parent a paced session inherits its workload from."""

    source: dict[str, str]
    fixture: dict[str, str]
    fixture_seed: int


@dataclass(frozen=True)
class SessionOutcome:
    """One completed paced session and where its evidence landed."""

    session_id: str
    run_id: str
    status: str
    failed_gates: list[str]
    bundle_sha256: str
    persistence_state: str
    wall_seconds: float


@dataclass(frozen=True)
class StudyOutcome:
    """The terminal study decision over exactly six paced sessions."""

    study_id: str | None
    status: str
    failed_gate_count: int | None
    bundle_sha256: str
    persistence_state: str


# ---------------------------------------------------------------------------
# H0 parent binding
# ---------------------------------------------------------------------------


def _verified_h0_rows(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    return connection.execute(
        "SELECT r.run_id,r.created_at_unix_ns,r.manifest_json,r.manifest_hash,"
        "e.payload_json,e.payload_hash "
        "FROM runs AS r JOIN status_events AS e ON e.run_id=r.run_id "
        "WHERE r.mode=? AND e.event_kind='common_result' "
        "ORDER BY r.created_at_unix_ns,r.run_id",
        ("eager_baseline",),
    ).fetchall()


def select_h0_parent(
    database: os.PathLike[str] | str = DEFAULT_H0_DB_PATH,
    *,
    run_id: str | None = None,
) -> ParentBinding:
    """Bind one fully replayed H0 eager-baseline run as the paced-study parent.

    The H0 database is opened read-only and every candidate is replayed through
    the public H0 bundle verifier before it can become a parent.  Without an
    explicit ``run_id`` the most recent verified eager baseline is selected;
    because the choice is written into all six manifests, a later parent change
    invalidates an in-flight study rather than silently mixing provenance.
    """

    from friday_h0.correctness_contract import (
        CorrectnessContractError,
        trusted_performance_fixture_identity,
    )

    path = Path(database)
    storage: H0Storage | None = None
    try:
        storage = H0Storage.open(path, read_only=True)
        connection = storage.connection
        if connection.execute("PRAGMA query_only").fetchone()[0] != 1:
            raise RunnerError("H0 source database is not query_only")
        rows = _verified_h0_rows(connection)
        if not rows:
            raise RunnerError("H0 source database holds no eager-baseline common result")
        selected = None
        for row in rows:
            if run_id is None or row["run_id"] == run_id:
                selected = row
        if selected is None:
            raise RunnerError("requested H0 parent run is not present")

        parent_run_id = selected["run_id"]
        manifest_value = _parse_stored_json(
            selected["manifest_json"],
            name=f"{parent_run_id} manifest",
            limit=PRODUCTION_MANIFEST_BYTES,
        )
        manifest = _mapping(manifest_value, f"{parent_run_id} manifest")
        try:
            closed = close_manifest(manifest)
        except (H0ProtocolError, TypeError, ValueError) as exc:
            raise RunnerError("H0 parent manifest replay failed") from exc
        if closed.run_id != parent_run_id or closed.mode != "eager_baseline":
            raise RunnerError("H0 parent mirrors do not match its manifest")

        payload = _mapping(
            _parse_stored_json(
                selected["payload_json"],
                name=f"{parent_run_id} common_result payload",
                limit=PERSISTENCE_MAX_BUNDLE_BYTES,
            ),
            f"{parent_run_id} common_result payload",
        )
        bundle = _mapping(payload.get("bundle"), f"{parent_run_id} evidence bundle")
        result = _mapping(bundle.get("result"), f"{parent_run_id} result")
        try:
            verified = storage.verify_common_result_bundle(
                closed,
                result,
                raw_samples=_array(bundle.get("raw_samples"), "raw_samples"),
                scalar_metrics=_array(bundle.get("scalar_metrics"), "scalar_metrics"),
                correctness_metrics=_array(bundle.get("correctness_metrics"), "correctness_metrics"),
                artifacts=_array(bundle.get("artifacts"), "artifacts"),
            )
        except (H0StorageError, H0ProtocolError, TypeError, ValueError, OverflowError) as exc:
            raise RunnerError("H0 parent bundle replay failed") from exc
        if verified != "verified":
            raise RunnerError("H0 parent verifier returned an unknown state")

        workload = _mapping(manifest.get("workload"), f"{parent_run_id} workload")
        seeds = _mapping(manifest.get("seeds"), f"{parent_run_id} seeds")
        fixture_seed = seeds.get("fixture")
        if type(fixture_seed) is not int:
            raise RunnerError("H0 parent fixture seed is not an integer")
        try:
            identity = trusted_performance_fixture_identity(
                a_shape=workload.get("a_shape"),
                b_shape=workload.get("b_shape"),
                dtype=workload.get("dtype"),
                layout=workload.get("layout"),
                fixture_seed=fixture_seed,
            )
        except CorrectnessContractError as exc:
            raise RunnerError("H0 parent fixture identity is not registered") from exc
        if workload.get("operation") != "matmul":
            raise RunnerError("H0 parent workload is not the registered matmul")

        # The three component digests are inherited unchanged from H0, but the
        # aggregate is H0.1's own: its manifest defines fixture_sha256 as the
        # canonical hash over those components, not as H0's raw-byte digest.
        components = {
            "a_sha256": identity["a_sha256"],
            "b_sha256": identity["b_sha256"],
            "metadata_sha256": identity["metadata_sha256"],
        }
        return ParentBinding(
            source={
                "parent_phase": "H0",
                "parent_run_id": parent_run_id,
                "parent_manifest_sha256": selected["manifest_hash"],
                "parent_result_sha256": payload["result_sha256"],
                "parent_bundle_sha256": payload["bundle_sha256"],
            },
            fixture={**components, "fixture_sha256": canonical_sha256(components)},
            fixture_seed=fixture_seed,
        )
    finally:
        if storage is not None:
            storage.close()


# ---------------------------------------------------------------------------
# Telemetry and preflight
# ---------------------------------------------------------------------------


def _power_source() -> dict[str, str | None]:
    """Read the macOS power source once; it is descriptive, never analytical.

    Battery operation changes the GPU power budget, so this is the first thing
    worth knowing when reading a trajectory afterwards.  It is sampled outside
    the paced window so the subprocess cannot perturb a measured sample.
    """

    try:
        completed = subprocess.run(
            ["/usr/bin/pmset", "-g", "ps"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"value": None, "missing_reason": "api_unavailable"}
    if completed.returncode != 0:
        return {"value": None, "missing_reason": "api_unavailable"}
    text = completed.stdout.decode("utf-8", errors="replace")
    if "AC Power" in text:
        return {"value": "ac_power", "missing_reason": None}
    if "Battery Power" in text:
        return {"value": "battery_power", "missing_reason": None}
    return {"value": None, "missing_reason": "api_unavailable"}


def collect_telemetry() -> dict[str, dict[str, str | None]]:
    """Closed descriptive telemetry pair; a missing value is never a zero."""

    return {
        # ProcessInfo.thermalState has no stdlib binding; a coarse guess would
        # be worse than an explicit, registered absence.
        "thermal_state": {"value": None, "missing_reason": "api_unavailable"},
        "power_source": _power_source(),
    }


class MlxBackend:
    """The only real-hardware backend; imported lazily behind the release gate."""

    def __init__(self) -> None:
        try:
            import mlx.core as mx
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RunnerError(f"MLX import unavailable: {type(exc).__name__}") from exc
        self.mx = mx

    def from_host(self, value: Any) -> Any:
        return self.mx.array(value)

    def matmul(self, a: Any, b: Any) -> Any:
        return self.mx.matmul(a, b)

    def eval(self, value: Any) -> Any:
        self.mx.eval(value)
        return value

    def synchronize(self) -> None:
        self.mx.synchronize()


def load_fixture_arrays(fixture_seed: int) -> tuple[Any, Any]:
    """Materialize the exact H0 fixture pair for this seed.

    The H0 generator is reused rather than reimplemented: it is the function
    that enforces the trusted fixture identity, and a second implementation
    could drift from it without any test noticing.
    """

    import importlib

    numpy_module = importlib.import_module("numpy")
    from friday_h0.benchmark import BenchmarkError, _generate_fixture

    try:
        fixture = _generate_fixture(numpy_module, fixture_seed)
    except BenchmarkError as exc:
        raise RunnerError(f"fixture materialization failed: {exc}") from exc
    return fixture.a, fixture.b


def preflight(
    *,
    h0_database: os.PathLike[str] | str = DEFAULT_H0_DB_PATH,
    h01_database: os.PathLike[str] | str = DEFAULT_H01_DB_PATH,
    parent_run_id: str | None = None,
    backend_factory: Callable[[], Backend] | None = None,
) -> dict[str, Any]:
    """Verify every precondition of a paced session without recording one."""

    provenance = collect_provenance()
    parent = select_h0_parent(h0_database, run_id=parent_run_id)

    existing: dict[str, int] = {}
    with Storage.open(h01_database, read_only=True) as storage:
        existing = storage.counts_by("entity_kind")

    probe: dict[str, Any] = {"state": "skipped", "samples": []}
    if backend_factory is not None:
        backend = backend_factory()
        a_host, b_host = load_fixture_arrays(parent.fixture_seed)
        a = backend.from_host(a_host)
        b = backend.from_host(b_host)
        durations = [_measure_once(backend, a, b) for _ in range(PREFLIGHT_PROBE_SAMPLES)]
        probe = {"state": "measured", "samples": durations}

    return {
        "phase": "H0.1",
        "code_sha256": provenance.code_sha256,
        "study_spec_sha256": provenance.study_spec_sha256,
        "environment_sha256": provenance.environment_sha256,
        "parent_run_id": parent.source["parent_run_id"],
        "parent_bundle_sha256": parent.source["parent_bundle_sha256"],
        "fixture_sha256": parent.fixture["fixture_sha256"],
        "fixture_seed": parent.fixture_seed,
        "existing_entities": existing,
        "paced_sessions": existing.get("paced_session", 0),
        "paced_studies": existing.get("paced_study", 0),
        "telemetry": collect_telemetry(),
        "probe": probe,
    }


# ---------------------------------------------------------------------------
# Paced measurement
# ---------------------------------------------------------------------------


def _measure_once(
    backend: Backend, a: Any, b: Any, clock: Callable[[], int] = time.perf_counter_ns
) -> int:
    """Time one fully evaluated and synchronized matmul in nanoseconds."""

    start = clock()
    output = backend.matmul(a, b)
    backend.eval(output)
    backend.synchronize()
    finished = clock()
    duration = finished - start
    if duration <= 0:
        raise RunnerError("measurement clock did not advance")
    if duration > SAMPLE_TIMEOUT_NS:
        raise RunnerError("single sample exceeded its controlled time limit")
    del output
    return duration


def _pace(target_ns: int, sleeper: Callable[[float], None], clock: Callable[[], int]) -> int:
    """Block until at least ``target_ns`` has elapsed and return the real wait.

    Sleeping short and re-checking keeps the wait from ever undershooting the
    registered gap, which the trace contract forbids outright.
    """

    start = clock()
    while True:
        elapsed = clock() - start
        remaining = target_ns - elapsed
        if remaining <= 0:
            return elapsed
        sleeper(remaining / 1_000_000_000.0)


def measure_session(
    session_id: str,
    manifest: Mapping[str, Any],
    *,
    backend: Backend,
    a: Any,
    b: Any,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, Any]:
    """Execute one paced session exactly as its materialized schedule dictates."""

    entries = manifest["schedule"]["entries"]
    if len(entries) != TOTAL_SAMPLES:
        raise RunnerError("schedule does not carry the registered sample count")

    durations: list[int] = []
    overshoots: list[int] = []
    observed_cooldown = COOLDOWN_NS
    for index, entry in enumerate(entries):
        if index == BURN_IN_SAMPLES:
            observed_cooldown = _pace(COOLDOWN_NS, sleeper, clock)
        actual_gap = _pace(entry["requested_gap_ns"], sleeper, clock)
        overshoot = actual_gap - entry["requested_gap_ns"]
        if overshoot < 0:
            raise RunnerError(f"sample {index} paced shorter than its registered gap")
        if overshoot > MAX_GAP_OVERSHOOT_NS:
            raise RunnerError(
                f"sample {index} exceeded the registered {MAX_GAP_OVERSHOOT_NS} ns overshoot bound"
            )
        overshoots.append(overshoot)
        durations.append(_measure_once(backend, a, b, clock))

    return {
        "durations_ns": durations,
        "gap_overshoots_ns": overshoots,
        "observed_cooldown_ns": observed_cooldown,
        # Binding the axis origin to the first real gap keeps every recorded
        # instant derived from measurement rather than from a synthetic offset.
        "first_start_ns": overshoots[0] + entries[0]["requested_gap_ns"],
    }


# ---------------------------------------------------------------------------
# Session and study orchestration
# ---------------------------------------------------------------------------


def run_session(
    session_id: str,
    *,
    h0_database: os.PathLike[str] | str = DEFAULT_H0_DB_PATH,
    h01_database: os.PathLike[str] | str = DEFAULT_H01_DB_PATH,
    parent_run_id: str | None = None,
    backend_factory: Callable[[], Backend] = MlxBackend,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], int] = time.perf_counter_ns,
) -> SessionOutcome:
    """Measure, analyze and persist exactly one paced session."""

    if session_id not in SESSION_ORDER:
        raise RunnerError("session id is not registered")
    provenance = collect_provenance()
    parent = select_h0_parent(h0_database, run_id=parent_run_id)
    manifest = build_manifest(
        session_id,
        fixture=parent.fixture,
        study_spec_sha256=provenance.study_spec_sha256,
        code_sha256=provenance.code_sha256,
        environment_sha256=provenance.environment_sha256,
        source=parent.source,
    )

    backend = backend_factory()
    a_host, b_host = load_fixture_arrays(parent.fixture_seed)
    a = backend.from_host(a_host)
    b = backend.from_host(b_host)
    telemetry = collect_telemetry()

    started = time.perf_counter_ns()
    recorded = measure_session(
        session_id, manifest, backend=backend, a=a, b=b, sleeper=sleeper, clock=clock
    )
    wall_seconds = (time.perf_counter_ns() - started) / 1_000_000_000.0

    trace = build_trace(
        manifest,
        recorded["durations_ns"],
        first_start_ns=recorded["first_start_ns"],
        observed_cooldown_ns=recorded["observed_cooldown_ns"],
        gap_overshoots_ns=recorded["gap_overshoots_ns"],
        telemetry=telemetry,
    )
    result = analyze_trace(manifest, trace)

    with Storage.open(h01_database) as storage:
        outcome = storage.persist_bundle(
            entity_id=manifest["run_id"],
            entity_kind="paced_session",
            status=result["status"],
            created_at_unix_ns=time.time_ns(),
            manifest=manifest,
            trace=trace,
            result=result,
            lineage=parent.source,
        )

    gates = result["gates"] or {}
    return SessionOutcome(
        session_id=session_id,
        run_id=manifest["run_id"],
        status=result["status"],
        failed_gates=[name for name, gate in gates.items() if gate["status"] == "fail"],
        bundle_sha256=outcome.bundle_sha256,
        persistence_state=outcome.state,
        wall_seconds=wall_seconds,
    )


def _session_records(storage: Storage) -> list[dict[str, Any]]:
    """Collect the six paced sessions in registered order, rejecting duplicates."""

    with storage.read_transaction():
        rows = storage.verified_rows()
    by_session: dict[str, dict[str, Any]] = {}
    for row in rows:
        bundle = row["bundle"]
        if bundle["entity_kind"] != "paced_session":
            continue
        session_id = bundle["manifest"]["session"]["id"]
        if session_id in by_session:
            raise RunnerError(f"session {session_id} is recorded more than once")
        by_session[session_id] = {
            "manifest": bundle["manifest"],
            "trace": bundle["trace"],
            "result": bundle["result"],
        }
    missing = [name for name in SESSION_ORDER if name not in by_session]
    if missing:
        raise RunnerError(f"study requires all six sessions; missing {','.join(missing)}")
    return [by_session[name] for name in SESSION_ORDER]


def run_study(
    *,
    h01_database: os.PathLike[str] | str = DEFAULT_H01_DB_PATH,
) -> StudyOutcome:
    """Replay all six persisted sessions and persist one terminal study decision."""

    # Reading the six sessions needs a verified read-only snapshot; persisting
    # the study needs a writable handle.  They are deliberately separate.
    with Storage.open(h01_database, read_only=True) as reader:
        records = _session_records(reader)
    result = analyze_study(records)
    if result["status"] == "h01_invalid":
        raise RunnerError(
            f"study replay failed: {(result['error'] or {}).get('message', 'unknown')}"
        )
    shared = result["shared_provenance"]
    with Storage.open(h01_database) as storage:
        outcome = storage.persist_bundle(
            entity_id=result["study_id"],
            entity_kind="paced_study",
            status=result["status"],
            created_at_unix_ns=time.time_ns(),
            manifest={"session_records": records},
            trace={"session_bindings": result["session_bindings"]},
            result=result,
            lineage=shared["source"],
        )
    return StudyOutcome(
        study_id=result["study_id"],
        status=result["status"],
        failed_gate_count=result["failed_gate_count"],
        bundle_sha256=outcome.bundle_sha256,
        persistence_state=outcome.state,
    )


__all__ = [
    "Backend",
    "DEFAULT_H0_DB_PATH",
    "MlxBackend",
    "ParentBinding",
    "RunnerError",
    "SessionOutcome",
    "StudyOutcome",
    "collect_telemetry",
    "load_fixture_arrays",
    "measure_session",
    "preflight",
    "run_session",
    "run_study",
    "select_h0_parent",
]
