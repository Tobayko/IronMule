"""Controlled live execution and persistence for the sealed H1-v2 study."""

from __future__ import annotations

import hashlib
import os
import resource
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from friday_evidence.budget import BudgetGuard

from .constants import (
    CALIBRATION,
    CONFIRMATION,
    DEFAULT_DATABASE_PATH,
    FIXTURE_SEED,
    N_MATMULS,
    OPERAND_SEED,
    PROJECT_ROOT,
    SESSION_ORDER,
    SHAPE,
)
from .protocol import (
    ProtocolError,
    build_calibration_summary,
    build_confirmation_seal,
    build_preregistration,
    build_session_failure,
    build_session_payload,
    build_study_decision,
    orders_for,
)
from .provenance import collect_provenance
from .storage import PersistenceOutcome, Storage, StorageError


class RunnerError(RuntimeError):
    """The formal live path cannot continue without weakening the study."""


class Backend(Protocol):
    def from_host(self, value: Any) -> Any: ...

    def matmul(self, left: Any, right: Any) -> Any: ...

    def eval_many(self, values: Sequence[Any]) -> None: ...

    def synchronize(self) -> None: ...

    def to_host(self, value: Any) -> Any: ...

    def memory_snapshot(self) -> dict[str, int | None]: ...


class MlxBackend:
    """The only real-hardware backend; MLX is imported only after release gates."""

    def __init__(self) -> None:
        import mlx.core as mx

        self.mx = mx

    def from_host(self, value: Any) -> Any:
        return self.mx.array(value)

    def matmul(self, left: Any, right: Any) -> Any:
        return self.mx.matmul(left, right)

    def eval_many(self, values: Sequence[Any]) -> None:
        self.mx.eval(*values)

    def synchronize(self) -> None:
        self.mx.synchronize()

    def to_host(self, value: Any) -> Any:
        import numpy as np

        return np.array(value, copy=False)

    def memory_snapshot(self) -> dict[str, int | None]:
        result: dict[str, int | None] = {}
        for key, name in (
            ("mlx_active_memory_bytes", "get_active_memory"),
            ("mlx_peak_memory_bytes", "get_peak_memory"),
            ("mlx_cache_memory_bytes", "get_cache_memory"),
        ):
            function = getattr(self.mx, name, None)
            try:
                value = function() if callable(function) else None
            except Exception:
                value = None
            result[key] = value if type(value) is int and value >= 0 else None
        return result


def read_power_source() -> str:
    try:
        completed = subprocess.run(
            ["/usr/bin/pmset", "-g", "ps"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    if completed.returncode != 0:
        return "unknown"
    text = completed.stdout.decode("utf-8", errors="replace")
    if "AC Power" in text:
        return "ac_power"
    if "Battery Power" in text:
        return "battery_power"
    return "unknown"


def require_ac_power() -> str:
    source = read_power_source()
    if source != "ac_power":
        raise RunnerError(f"mains power is required; observed {source}")
    return source


def _preregistration(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    matches = [row["payload"] for row in records if row["record_kind"] == "preregistration"]
    if len(matches) != 1:
        raise RunnerError("exactly one sealed preregistration is required")
    return matches[0]


def _same_provenance(records: Sequence[Mapping[str, Any]], provenance: Mapping[str, Any]) -> None:
    prereg = _preregistration(records)
    if prereg["provenance_sha256"] != provenance.get("provenance_sha256"):
        raise RunnerError("current code or environment differs from the sealed preregistration")


def seal_preregistration(
    database_path: os.PathLike[str] | str = DEFAULT_DATABASE_PATH,
) -> PersistenceOutcome:
    provenance = collect_provenance(require_clean=True)
    with Storage.open(database_path, initialize=True) as storage:
        records = storage.verified_records()
        if records:
            _same_provenance(records, provenance)
            row = records[0]
            return PersistenceOutcome(row["record_id"], row["entity_key"], "already_sealed")
        payload = build_preregistration(provenance["provenance_sha256"])
        return storage.persist(payload, provenance)


def _stage_records(
    records: Sequence[Mapping[str, Any]], stage: str
) -> list[dict[str, Any]]:
    kind = "calibration_session" if stage == CALIBRATION else "confirmation_session"
    return [row["payload"] for row in records if row["record_kind"] == kind]


def _single_payload(records: Sequence[Mapping[str, Any]], kind: str) -> dict[str, Any] | None:
    matches = [row["payload"] for row in records if row["record_kind"] == kind]
    if len(matches) > 1:
        raise RunnerError(f"formal history has duplicate {kind}")
    return matches[0] if matches else None


def preflight_session(
    stage: str,
    session_id: str,
    *,
    database_path: os.PathLike[str] | str = DEFAULT_DATABASE_PATH,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if stage not in {CALIBRATION, CONFIRMATION} or session_id not in SESSION_ORDER:
        raise RunnerError("unknown H1-v2 stage or session")
    current = dict(provenance or collect_provenance(require_clean=True))
    with Storage.open(database_path, read_only=True) as storage:
        with storage.read_transaction():
            records = storage.verified_records()
    _same_provenance(records, current)
    if any(row["record_kind"] == "session_failure" for row in records):
        raise RunnerError("a terminal failed attempt already closed this study")
    sessions = _stage_records(records, stage)
    if len(sessions) >= len(SESSION_ORDER) or SESSION_ORDER[len(sessions)] != session_id:
        raise RunnerError("session is not the next preregistered session")
    calibration_summary = _single_payload(records, "calibration_summary")
    seal = _single_payload(records, "confirmation_seal")
    if stage == CALIBRATION:
        if calibration_summary is not None or seal is not None:
            raise RunnerError("calibration stage is already closed")
    elif seal is None:
        raise RunnerError("confirmation is locked until calibration is sealed")
    prereg = _preregistration(records)
    return {
        "preregistration_sha256": prereg["preregistration_sha256"],
        "confirmation_seal_sha256": None if seal is None else seal["confirmation_seal_sha256"],
        "provenance_sha256": current["provenance_sha256"],
        "next_session": session_id,
        "stage": stage,
    }


def _time_plan(
    plan: Callable[[], list[Any]],
    *,
    guard: BudgetGuard,
    clock_ns: Callable[[], int],
) -> tuple[list[Any], int]:
    started = clock_ns()
    produced = plan()
    ended = clock_ns()
    if isinstance(started, bool) or isinstance(ended, bool) or ended <= started:
        raise RunnerError("measurement clock did not advance")
    duration = ended - started
    guard.record_gpu(duration / 1e9)
    return produced, duration


def _output_digest(values: Sequence[Any], backend: Backend) -> tuple[str, list[Any]]:
    digest = hashlib.sha256(b"friday-h1-output-v1\0")
    host_values: list[Any] = []
    for value in values:
        array = backend.to_host(value)
        host_values.append(array)
        shape = tuple(int(item) for item in array.shape)
        descriptor = f"{array.dtype}:{shape}".encode("ascii")
        payload = array.tobytes(order="C")
        digest.update(len(descriptor).to_bytes(4, "big"))
        digest.update(descriptor)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest(), host_values


def _rss_peak_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _fixed_budget(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {"passed": True, **dict(summary)}


def measure_prepared_session(
    stage: str,
    session_id: str,
    *,
    binding: Mapping[str, Any],
    backend: Backend,
    left: Any,
    operands: Sequence[Any],
    np_module: Any,
    power_source: str,
    guard: BudgetGuard | None = None,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    process_clock_ns: Callable[[], int] = time.process_time_ns,
    process_started_ns: int | None = None,
) -> dict[str, Any]:
    if len(operands) != N_MATMULS:
        raise RunnerError("prepared workload has the wrong operand count")
    active_guard = guard or BudgetGuard()
    cpu_started = process_clock_ns() if process_started_ns is None else process_started_ns
    active_guard.before_candidate()

    def serial() -> list[Any]:
        produced: list[Any] = []
        for operand in operands:
            value = backend.matmul(left, operand)
            backend.eval_many([value])
            backend.synchronize()
            produced.append(value)
        return produced

    def serial_a() -> list[Any]:
        return serial()

    def serial_b() -> list[Any]:
        return serial()

    def batched() -> list[Any]:
        produced = [backend.matmul(left, operand) for operand in operands]
        backend.eval_many(produced)
        backend.synchronize()
        return produced

    arm_a = serial_a if stage == CALIBRATION else serial
    arm_b = serial_b if stage == CALIBRATION else batched

    reference, _ = _time_plan(serial, guard=active_guard, clock_ns=clock_ns)
    produced_a, _ = _time_plan(arm_a, guard=active_guard, clock_ns=clock_ns)
    produced_b, _ = _time_plan(arm_b, guard=active_guard, clock_ns=clock_ns)
    reference_digest, reference_host = _output_digest(reference, backend)
    a_digest, a_host = _output_digest(produced_a, backend)
    b_digest, b_host = _output_digest(produced_b, backend)
    max_abs_error = max(
        float(np_module.max(np_module.abs(value.astype(np_module.float32) - reference_value.astype(np_module.float32))))
        for values in (a_host, b_host)
        for value, reference_value in zip(values, reference_host, strict=True)
    )
    if max_abs_error != 0.0 or len({reference_digest, a_digest, b_digest}) != 1:
        raise RunnerError("dispatch plans are not byte-identical")

    def measured_pair(order: str) -> tuple[int, int]:
        if order == "ab":
            _, a_ns = _time_plan(arm_a, guard=active_guard, clock_ns=clock_ns)
            _, b_ns = _time_plan(arm_b, guard=active_guard, clock_ns=clock_ns)
        elif order == "ba":
            _, b_ns = _time_plan(arm_b, guard=active_guard, clock_ns=clock_ns)
            _, a_ns = _time_plan(arm_a, guard=active_guard, clock_ns=clock_ns)
        else:
            raise RunnerError("unknown sealed arm order")
        return a_ns, b_ns

    warmups: list[dict[str, Any]] = []
    for index, order in enumerate(orders_for(stage, session_id, warmup=True)):
        a_ns, b_ns = measured_pair(order)
        warmups.append({"block_index": index, "order": order, "a_ns": a_ns, "b_ns": b_ns})
    measurements: list[dict[str, Any]] = []
    for index, order in enumerate(orders_for(stage, session_id, warmup=False)):
        a_ns, b_ns = measured_pair(order)
        measurements.append(
            {"block_index": index, "order": order, "a_ns": a_ns, "b_ns": b_ns}
        )

    active_guard.finish_candidate()
    budget = _fixed_budget(active_guard.summary())
    resources = {
        "cpu_process_ns": process_clock_ns() - cpu_started,
        "rss_peak_bytes": _rss_peak_bytes(),
        **backend.memory_snapshot(),
    }
    return build_session_payload(
        stage=stage,
        session_id=session_id,
        preregistration_sha256=binding["preregistration_sha256"],
        confirmation_seal_sha256=binding["confirmation_seal_sha256"],
        provenance_sha256=binding["provenance_sha256"],
        power_source=power_source,
        warmups=warmups,
        measurements=measurements,
        correctness={
            "status": "byte_identical",
            "max_abs_error": max_abs_error,
            "reference_sha256": reference_digest,
            "a_output_sha256": a_digest,
            "b_output_sha256": b_digest,
        },
        budget=budget,
        resources=resources,
    )


def _load_real_workload(
    backend: Backend, *, guard: BudgetGuard
) -> tuple[Any, list[Any], Any]:
    import numpy as np

    from friday_h0.benchmark import _generate_fixture

    fixture = _generate_fixture(np, FIXTURE_SEED, shape=SHAPE)
    rng = np.random.Generator(np.random.PCG64(OPERAND_SEED))
    host_operands = [
        rng.uniform(-1.0, 1.0, (SHAPE, SHAPE)).astype(np.float16)
        for _ in range(N_MATMULS)
    ]
    gpu_started = time.perf_counter()
    left = backend.from_host(fixture.a)
    operands = [backend.from_host(value) for value in host_operands]
    backend.eval_many([left, *operands])
    backend.synchronize()
    guard.record_gpu(time.perf_counter() - gpu_started)
    return left, operands, np


def execute_session(
    stage: str,
    session_id: str,
    *,
    database_path: os.PathLike[str] | str = DEFAULT_DATABASE_PATH,
    backend_factory: Callable[[], Backend] = MlxBackend,
) -> PersistenceOutcome:
    provenance = collect_provenance(require_clean=True)
    binding = preflight_session(
        stage, session_id, database_path=database_path, provenance=provenance
    )
    power = require_ac_power()
    try:
        process_started_ns = time.process_time_ns()
        guard = BudgetGuard()
        guard.before_candidate()
        backend = backend_factory()
        left, operands, np_module = _load_real_workload(backend, guard=guard)
        payload = measure_prepared_session(
            stage,
            session_id,
            binding=binding,
            backend=backend,
            left=left,
            operands=operands,
            np_module=np_module,
            power_source=power,
            guard=guard,
            process_started_ns=process_started_ns,
        )
    except (Exception, SystemExit) as exc:
        current = collect_provenance(require_clean=True)
        if current["provenance_sha256"] == provenance["provenance_sha256"]:
            failure = build_session_failure(
                stage=stage,
                session_id=session_id,
                preregistration_sha256=binding["preregistration_sha256"],
                confirmation_seal_sha256=binding["confirmation_seal_sha256"],
                provenance_sha256=provenance["provenance_sha256"],
                failure_type=type(exc).__name__,
            )
            with Storage.open(database_path) as storage:
                storage.persist(failure, provenance)
        raise
    current = collect_provenance(require_clean=True)
    if current["provenance_sha256"] != provenance["provenance_sha256"]:
        raise RunnerError("source or environment changed during measurement")
    with Storage.open(database_path) as storage:
        return storage.persist(payload, provenance)


def summarize_calibration(
    database_path: os.PathLike[str] | str = DEFAULT_DATABASE_PATH,
) -> PersistenceOutcome:
    provenance = collect_provenance(require_clean=True)
    with Storage.open(database_path) as storage:
        records = storage.verified_records()
        _same_provenance(records, provenance)
        sessions = _stage_records(records, CALIBRATION)
        payload = build_calibration_summary(sessions)
        return storage.persist(payload, provenance)


def seal_confirmation(
    database_path: os.PathLike[str] | str = DEFAULT_DATABASE_PATH,
) -> PersistenceOutcome:
    provenance = collect_provenance(require_clean=True)
    with Storage.open(database_path) as storage:
        records = storage.verified_records()
        _same_provenance(records, provenance)
        sessions = _stage_records(records, CALIBRATION)
        calibration = _single_payload(records, "calibration_summary")
        if calibration is None:
            raise RunnerError("calibration summary is missing")
        payload = build_confirmation_seal(calibration, sessions)
        return storage.persist(payload, provenance)


def decide_study(
    database_path: os.PathLike[str] | str = DEFAULT_DATABASE_PATH,
) -> PersistenceOutcome:
    provenance = collect_provenance(require_clean=True)
    with Storage.open(database_path) as storage:
        records = storage.verified_records()
        _same_provenance(records, provenance)
        sessions = _stage_records(records, CONFIRMATION)
        seal = _single_payload(records, "confirmation_seal")
        if seal is None:
            raise RunnerError("confirmation seal is missing")
        payload = build_study_decision(sessions, seal)
        return storage.persist(payload, provenance)


def current_record(
    kind: str,
    database_path: os.PathLike[str] | str = DEFAULT_DATABASE_PATH,
) -> dict[str, Any] | None:
    with Storage.open(database_path, read_only=True) as storage:
        with storage.read_transaction():
            records = storage.verified_records()
    return _single_payload(records, kind)


__all__ = [
    "Backend",
    "MlxBackend",
    "RunnerError",
    "current_record",
    "decide_study",
    "execute_session",
    "measure_prepared_session",
    "preflight_session",
    "read_power_source",
    "require_ac_power",
    "seal_confirmation",
    "seal_preregistration",
    "summarize_calibration",
]
