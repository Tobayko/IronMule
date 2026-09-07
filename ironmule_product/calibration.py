"""Bounded automatic calibration jobs, separate from serving and evaluation.

The controller chooses no thresholds. It executes the closed plan, waits for
real host readiness and records failures as well as complete measurements.
Only the separate evaluator assigns a verdict; deployment is not enabled here.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import http.client
import json
import math
import os
import secrets
import sys
import threading
import time
from typing import Any
import uuid

from friday_evidence.budget import BudgetGuard
from friday_evidence.canonical import canonical_sha256
from friday_evidence.events import EventJournal
from friday_evidence.identity import runtime_identity

from .calibration_plan import LIMIT_ORDERS, POLICY, PROMPT, plan_id, schedule, specification
from .errors import StateError
from .readiness import ReadinessPolicy, ReadinessWindow, evaluate, hardware_identity, probe
from .state import _atomic_write, _open_state_file, _read_json, _read_locked, _regular_or_missing


JOURNAL_NAME = "optimization.sqlite3"
_LEASE_NAME = ".optimization.lock"
_MODEL_LEASE_NAME = ".model.lock"
_STATUS_NAME = "optimization-status.json"
_ACTIVE_STAGES = {"waiting", "fingerprinting", "loading", "measuring", "cooldown", "evaluating"}
_STAGES = _ACTIVE_STAGES | {"idle", "deferred", "failed", "cancelled", "finished"}
_SAMPLE_TIMEOUT_S = 5.0


def _load_monitor_spec() -> dict[str, Any]:
    from .memory import POLL_INTERVAL_SECONDS, RSS_LIMIT_FRACTION, SWAP_DELTA_LIMIT_BYTES

    return {
        "schema": "ironmule.load_monitor.v1",
        "poll_interval_seconds": POLL_INTERVAL_SECONDS,
        "swap_delta_limit_bytes": SWAP_DELTA_LIMIT_BYTES,
        "rss_limit_fraction": RSS_LIMIT_FRACTION,
        "mlx_peak_limit_fraction": POLICY.peak_memory_fraction,
        "clean_shutdown_required": True,
    }


class CalibrationFailure(RuntimeError):
    def __init__(self, code: str, *, status: str = "failed"):
        super().__init__(code)
        self.code = code
        self.status = status


@contextmanager
def _exclusive_lease(store, name: str, busy_code: str):
    store.settings()  # Setup owns root creation and permissions.
    path = store.root / name
    _regular_or_missing(path)
    fd = _open_state_file(path, os.O_RDWR | os.O_CREAT)
    locked = False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except BlockingIOError as exc:
            raise CalibrationFailure(busy_code, status="deferred") from exc
        yield
    finally:
        if locked:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _job_lease(store):
    return _exclusive_lease(store, _LEASE_NAME, "another_calibration_running")


def model_lease(store):
    """Serialize serving/model calibration for one configured product state."""
    return _exclusive_lease(store, _MODEL_LEASE_NAME, "model_resource_busy")


def _lease_held(store) -> bool:
    path = store.root / _LEASE_NAME
    _regular_or_missing(path)
    if not path.exists():
        return False
    fd = _open_state_file(path, os.O_RDONLY)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def status(store) -> dict[str, Any]:
    settings = store.settings()
    with _read_locked(store.root):
        saved = _read_json(store.root / _STATUS_NAME)
        held = _lease_held(store)
    if saved is not None and (
        not isinstance(saved, dict) or type(saved.get("schema")) is not int or saved.get("schema") != 1
        or not isinstance(saved.get("stage"), str) or saved.get("stage") not in _STAGES
    ):
        raise StateError("optimization status is invalid")
    saved = saved or {"stage": "idle", "model_loaded": False}
    stage = saved["stage"]
    if stage in _ACTIVE_STAGES and not held:
        stage = "interrupted"
    return {"schema": 1, "paused": settings["optimization_paused"],
            "optimization_paused": settings["optimization_paused"],
            "configuration_only": False, "engine_started": held,
            "stage": stage, "run_id": saved.get("run_id"),
            "model_id": saved.get("model_id"),
            "model_loaded": bool(held and saved.get("model_loaded") is True),
            "activation_allowed": False, "updated_unix_ns": saved.get("updated_unix_ns"),
            "last_verdict": saved.get("verdict")}


def history(store, *, limit: int = 100, after_seq: int = 0, run_id: str | None = None) -> dict:
    store.settings()
    if type(limit) is not int or not 1 <= limit <= 1000 or type(after_seq) is not int or after_seq < 0:
        raise StateError("history pagination is invalid")
    path = store.root / JOURNAL_NAME
    _regular_or_missing(path)
    if not path.exists():
        return {"schema": 1, "events": [], "next_after_seq": after_seq, "verified": True}
    with EventJournal(path, read_only=True) as journal:
        rows = journal.events(run_id=run_id, limit=limit, after_seq=after_seq)
    return {"schema": 1, "events": rows, "next_after_seq": rows[-1]["seq"] if rows else after_seq,
            "verified": True}


class _VariantLease:
    """Forward real inference into one worker without owning its lifetime."""

    def __init__(self, client, variant: str):
        self.client, self.variant = client, variant
        self.trace = False
        self.last_events: list[dict] = []

    @property
    def ready(self):
        return self.client.ready

    def stream(self, request, cancel=None, timeout=120):
        self.last_events = []
        stream = self.client.stream(request, cancel=cancel, timeout=min(timeout, _SAMPLE_TIMEOUT_S),
                                    variant=self.variant, trace_forwards=self.trace)
        try:
            for event in stream:
                self.last_events.append(dict(event))
                yield event
        finally:
            stream.close()

    def close(self):
        pass  # The calibration job owns the single shared worker.


@contextmanager
def _http_paths(store, spec, client):
    from .http_server import create_server
    from .service import ProductService

    paths, servers, started_servers, threads, services = {}, [], [], [], []
    key = secrets.token_urlsafe(32)
    try:
        for variant in ("reference", "bounded_prefetch"):
            lease = _VariantLease(client, variant)
            service = ProductService(store, backend=lease, spec=spec)
            services.append(service)
            server = create_server(service, host="127.0.0.1", port=0, api_key=key)
            servers.append(server)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            threads.append(thread)
            thread.start()
            started_servers.append(server)
            paths[variant] = (server.server_address[1], lease)
        yield paths, key
    finally:
        for server in started_servers:
            server.shutdown()
        for server in servers:
            server.server_close()
        for thread in threads:
            if thread.ident is not None:
                thread.join(timeout=5)
        for service in services:
            service.close()


def _http_sample(paths, key: str, spec, descriptor: dict) -> tuple[dict, dict]:
    variant = descriptor["variant"]
    port, lease = paths[variant]
    lease.trace = descriptor["trace_forwards"]
    payload = {"model": spec.model_id, "messages": [{"role": "user", "content": PROMPT}],
               "max_tokens": descriptor["limit"], "temperature": 0}
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    started = time.monotonic()
    try:
        connection.request("POST", "/v1/chat/completions", json.dumps(payload),
                           {"Content-Type": "application/json", "Authorization": "Bearer " + key})
        response = connection.getresponse()
        raw = response.read()
        wall = time.monotonic() - started
        if response.status != 200:
            raise CalibrationFailure(f"http_status_{response.status}")
        completion = json.loads(raw)
    finally:
        connection.close()
    events = lease.last_events
    if not events or events[-1].get("type") != "done":
        raise CalibrationFailure("missing_worker_completion")
    done = events[-1]
    emitted = [event for event in events if event.get("type") == "token"]
    identity = {"token_ids": [event["token_id"] for event in emitted],
                "text": "".join(event["text"] for event in emitted),
                "finish_reason": done["finish_reason"],
                "prompt_tokens": done["prompt_tokens"], "completion_tokens": done["completion_tokens"]}
    choice = completion["choices"][0]
    usage = completion["usage"]
    if (choice["message"]["content"] != identity["text"]
            or choice["finish_reason"] != identity["finish_reason"]
            or usage["prompt_tokens"] != identity["prompt_tokens"]
            or usage["completion_tokens"] != identity["completion_tokens"]
            or usage["total_tokens"] != identity["prompt_tokens"] + identity["completion_tokens"]):
        raise CalibrationFailure("http_worker_output_mismatch")
    peak = done.get("metrics", {}).get("peak_memory")
    if (isinstance(peak, bool) or not isinstance(peak, (int, float))
            or not math.isfinite(peak) or peak < 0):
        raise CalibrationFailure("missing_memory_measurement")
    sample = {**descriptor, "status": "measured", "http_wall_seconds": wall,
              "peak_memory_bytes": int(peak * 1_000_000_000),
              "prompt_tokens": identity["prompt_tokens"], "completion_tokens": identity["completion_tokens"],
              "finish_reason": identity["finish_reason"],
              "token_sha256": canonical_sha256(identity["token_ids"]),
              "output_sha256": canonical_sha256(identity),
              "forward_count": done.get("model_forward_invocations")}
    return sample, identity


class CalibrationJob:
    """Execute one registered job with real readiness and no hidden retries."""

    def __init__(self, store, model_id: str, *, max_wait_s: float = 300,
                 readiness_only: bool = False, cancel: threading.Event | None = None,
                 on_event=None):
        if (type(max_wait_s) not in (int, float) or not math.isfinite(max_wait_s)
                or not 0 <= max_wait_s <= 900):
            raise ValueError("max_wait_s must be finite and between 0 and 900")
        self.store, self.spec = store, store.model(model_id)
        self.max_wait_s, self.readiness_only = float(max_wait_s), readiness_only
        self.cancel = cancel or threading.Event()
        self.on_event = on_event
        self.run_id = uuid.uuid4().hex
        self.journal = None
        self.guard = None
        self.owned = False
        self.stage = "waiting"
        self.model_loaded = False
        self.deadline = 0.0
        self.readiness_policy = ReadinessPolicy()
        self._active_load_guard = None
        self._active_process = None
        self.report = {"schema": "ironmule.calibration.v1", "run_id": self.run_id,
                       "plan_id": plan_id(), "status": "started", "model_id": self.spec.model_id,
                       "revision": self.spec.revision, "samples": [], "workers": [],
                       "resource_events": [], "worker_timings": [], "worker_load_memory": [],
                       "worker_exit_codes": [],
                       "load_monitor": _load_monitor_spec(),
                       "resource_valid": False, "activation_allowed": False}

    def _emit(self, kind: str, payload: dict) -> None:
        assert self.journal is not None
        event = self.journal.append(self.run_id, kind, payload)
        if self.owned:
            _atomic_write(self.store.root / _STATUS_NAME,
                          {"schema": 1, "stage": self.stage, "run_id": self.run_id,
                           "model_id": self.spec.model_id, "model_loaded": self.model_loaded,
                           "updated_unix_ns": event["recorded_unix_ns"],
                           "verdict": self.report.get("evaluation", {}).get("verdict")})
        if self.on_event is not None:
            self.on_event(event)

    def _check(self) -> None:
        if self.cancel.is_set():
            raise CalibrationFailure("cancelled", status="cancelled")
        if self.store.settings()["optimization_paused"]:
            raise CalibrationFailure("paused", status="deferred")
        if time.monotonic() > self.deadline:
            raise CalibrationFailure("job_deadline", status="deferred")
        if self.guard is not None:
            self.guard.check_wall()
        if self._active_load_guard is not None and self._active_process is not None:
            from .memory import MemoryGuardError
            try:
                self._active_load_guard(self._active_process.pid)
            except MemoryGuardError as exc:
                raise CalibrationFailure(exc.code) from exc

    def _sleep(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self._check()
            self.cancel.wait(min(1.0, max(0, end - time.monotonic())))

    def _wait_ready(self) -> dict:
        self.stage = "waiting"
        window = ReadinessWindow(self.readiness_policy)
        end = min(self.deadline, time.monotonic() + self.max_wait_s)
        while True:
            self._check()
            snapshot = probe()
            decision = window.observe(snapshot)
            self._emit("readiness", {"observation": snapshot, "decision": decision,
                                     "model_loaded": self.model_loaded})
            if decision["stable"]:
                return snapshot
            if time.monotonic() >= end:
                raise CalibrationFailure("readiness_timeout", status="deferred")
            self._sleep(min(self.readiness_policy.sample_interval_s, end - time.monotonic()))

    def _checkpoint(self) -> dict:
        self._check()
        snapshot = probe()
        decision = evaluate(snapshot, self.readiness_policy)
        self._emit("readiness", {"observation": snapshot, "decision": decision,
                                 "model_loaded": self.model_loaded})
        if not decision["eligible"]:
            raise CalibrationFailure("measurement_readiness_lost")
        if type(snapshot.get("swap_used_bytes")) is not int:
            raise CalibrationFailure("swap_telemetry_missing")
        if snapshot["swap_used_bytes"] - self.report["swap_baseline_bytes"] > POLICY.swap_delta_limit_bytes:
            raise CalibrationFailure("swap_growth")
        return snapshot

    def _pace(self) -> None:
        assert self.guard is not None
        while True:
            self._check()
            now = time.monotonic()
            cutoff = now - self.guard.policy.duty_window_s
            used = sum(max(0.0, end - max(start, cutoff)) for start, end in self.guard._gpu_events)
            if used + self.guard.policy.continuous_gpu_limit_s <= self.guard.policy.duty_window_s * self.guard.policy.duty_cycle_limit:
                return
            self.guard.required_break()

    def _book_work(self, seconds: float, sample_index: int) -> None:
        assert self.guard is not None
        self.guard.record_gpu(seconds)
        start, end = self.guard._gpu_events[-1]
        self.report["resource_events"].append(
            {"sample_index": sample_index, "start_monotonic": start,
             "end_monotonic": end, "upper_bound_seconds": seconds})

    def _measure(self) -> None:
        initial = self._wait_ready()
        with model_lease(self.store):
            self._measure_reserved(initial)

    def _measure_reserved(self, initial: dict) -> None:
        from .backend import MLXWorkerClient
        from .evaluation import evaluate_report

        hardware = hardware_identity()
        initial.update(hardware)
        self.report["memory_total_bytes"] = initial["memory_total_bytes"]
        self.report["swap_baseline_bytes"] = initial["swap_used_bytes"]
        self.stage = "fingerprinting"
        self._emit("validation", {"state": "binding_identity"})
        self.report["identity_before"] = runtime_identity(self.spec.as_dict(), initial)
        self._emit("validation", {"state": "identity_bound", "identity": self.report["identity_before"]})
        self._wait_ready()  # Hashing must not create its own measurement conditions.
        self.guard = BudgetGuard(sleeper=self._sleep)
        expected = {}
        descriptors = schedule()
        for worker_index, limits in enumerate(LIMIT_ORDERS):
            self.stage = "cooldown"
            self._emit("validation", {"state": "cooldown", "worker_index": worker_index})
            self.guard.before_candidate()
            self._wait_ready()
            self._checkpoint()
            self.stage = "loading"
            self._emit("validation", {"state": "worker_loading", "worker_index": worker_index})
            client = MLXWorkerClient(self.spec)
            process = None
            worker = {"worker_index": worker_index, "started": False, "closed": False, "pid": None}
            self.report["workers"].append(worker)
            load_memory_row: dict[str, Any] = {"worker_index": worker_index, "samples": [], "ready": None, "ready_observation": None}
            self.report["worker_load_memory"].append(load_memory_row)
            timing = {"worker_index": worker_index, "load_started_monotonic": time.monotonic(),
                      "ready_monotonic": None, "closed_monotonic": None}
            self.report["worker_timings"].append(timing)
            exit_row = {"worker_index": worker_index, "returncode": None, "normal_shutdown": False}
            self.report["worker_exit_codes"].append(exit_row)
            from .memory import LoadMemoryGuard, MemoryGuardError

            def on_load_memory_sample(observation: Any) -> None:
                if not isinstance(observation, dict):
                    raise CalibrationFailure("load_memory_observation_invalid")
                saved = dict(observation)
                load_memory_row["samples"].append(saved)
                self._emit("validation", {"state": "load_memory_sample", "worker_index": worker_index,
                                           "observation": saved})

            try:
                self._active_load_guard = LoadMemoryGuard(
                    swap_baseline_bytes=self.report["swap_baseline_bytes"],
                    memory_total_bytes=self.report["memory_total_bytes"],
                    on_sample=on_load_memory_sample,
                )
            except MemoryGuardError as exc:
                raise CalibrationFailure(getattr(exc, "code", "load_memory_guard_invalid")) from exc

            def startup_guard(pid: int) -> None:
                nonlocal process
                process = client._process
                if process is None or process.pid != pid:
                    raise CalibrationFailure("worker_start_handle_missing")
                self._active_process = process
                worker["pid"] = pid
                self._check()

            try:
                started = time.monotonic()
                ready_payload = client.start(startup_guard=startup_guard)
                if process is None or self._active_load_guard is None:
                    raise CalibrationFailure("worker_start_handle_missing")
                ready_keys = (
                    "startup_wall_seconds", "process_peak_rss_bytes", "mlx_active_bytes",
                    "mlx_peak_bytes", "mlx_cache_bytes", "recommended_working_set_bytes",
                )
                ready_metrics = {key: ready_payload.get(key) for key in ready_keys} if isinstance(ready_payload, dict) else {}
                # Persist the raw bounded telemetry before validation so a
                # failed peak/RSS check remains auditable.
                self._emit("validation", {"state": "load_memory_ready", "worker_index": worker_index,
                                           "ready": ready_metrics})
                try:
                    ready_observation = self._active_load_guard(process.pid, force=True)
                    validated_ready = self._active_load_guard.validate_ready(ready_payload)
                except MemoryGuardError as exc:
                    raise CalibrationFailure(getattr(exc, "code", "load_memory_ready_invalid")) from exc
                load_memory_row["ready"] = dict(validated_ready)
                load_memory_row["ready_observation"] = dict(ready_observation) if isinstance(ready_observation, dict) else ready_observation
                self._emit("validation", {"state": "load_memory_ready", "worker_index": worker_index,
                                           "ready": load_memory_row["ready"],
                                           "observation": load_memory_row["ready_observation"]})
                if process is None:
                    raise CalibrationFailure("worker_start_missing_handle")
                worker.update(started=True, pid=process.pid)
                timing["ready_monotonic"] = time.monotonic()
                self.model_loaded = True
                self._emit("worker_started", {**worker, "load_wall_seconds": time.monotonic() - started})
                self.guard.required_break()
                with _http_paths(self.store, self.spec, client) as (paths, key):
                    for limit in limits:
                        self._wait_ready()
                        self._checkpoint()
                        self.stage = "measuring"
                        forward_counts = {}
                        cell = [d for d in descriptors if d["worker_index"] == worker_index and d["limit"] == limit]
                        for descriptor in cell:
                            self._pace()
                            attempted = time.monotonic()
                            sample = None
                            request_attempted = False
                            measured_started = attempted
                            try:
                                if self._active_load_guard is not None and self._active_process is not None:
                                    try:
                                        self._active_load_guard(self._active_process.pid, force=True)
                                    except MemoryGuardError as exc:
                                        raise CalibrationFailure(getattr(exc, "code", "load_memory_guard_failed")) from exc
                                measured_started = time.monotonic()
                                request_attempted = True
                                sample, actual = _http_sample(paths, key, self.spec, descriptor)
                                sample["identity_sha256"] = self.report["identity_before"]["identity_sha256"]
                                self.report["samples"].append(sample)
                                # Persist measured facts before any correctness/resource verdict.
                                self._emit("sample", dict(sample))
                                # Include journal overhead in the conservative
                                # resource interval, but exclude pre/post guard
                                # probes from the hot request interval.
                                self._book_work(time.monotonic() - measured_started, descriptor["sample_index"])
                                if self._active_load_guard is not None and self._active_process is not None:
                                    try:
                                        self._active_load_guard(self._active_process.pid, force=True)
                                    except MemoryGuardError as exc:
                                        raise CalibrationFailure(getattr(exc, "code", "load_memory_guard_failed")) from exc
                                before = expected.setdefault(limit, actual)
                                sample["correctness"] = actual == before
                                sample["tokens_match"] = actual["token_ids"] == before["token_ids"]
                                sample["text_match"] = actual["text"] == before["text"]
                                if not sample["correctness"]:
                                    raise CalibrationFailure("exact_output_mismatch")
                                if descriptor["trace_forwards"]:
                                    count = sample["forward_count"]
                                    if type(count) is not int or count < 0:
                                        raise CalibrationFailure("forward_count_missing")
                                    forward_counts[descriptor["variant"]] = count
                                    if len(forward_counts) == 2:
                                        delta = 1 if sample["completion_tokens"] == limit else 0
                                        if forward_counts["reference"] - forward_counts["bounded_prefetch"] != delta:
                                            raise CalibrationFailure("forward_mechanism_mismatch")
                                memory = probe()
                                sample["swap_used_bytes"] = memory["swap_used_bytes"]
                                if sample["swap_used_bytes"] is None:
                                    raise CalibrationFailure("swap_telemetry_missing")
                                if sample["swap_used_bytes"] - self.report["swap_baseline_bytes"] > POLICY.swap_delta_limit_bytes:
                                    raise CalibrationFailure("swap_growth")
                                if sample["peak_memory_bytes"] > POLICY.peak_memory_fraction * self.report["memory_total_bytes"]:
                                    raise CalibrationFailure("peak_memory_limit")
                                sample["status"] = "passed"
                                self._emit("validation", {"sample_index": descriptor["sample_index"], "sample": dict(sample),
                                                           "resource_event": dict(self.report["resource_events"][-1])})
                                self.guard.required_break()
                            except Exception as exc:
                                fault = {**descriptor, "status": "failed", "error_type": type(exc).__name__,
                                         "error_code": getattr(exc, "code", "measurement_failed"),
                                         "attempt_wall_seconds": time.monotonic() - attempted}
                                if sample is not None:
                                    sample.update(status="failed", error_code=fault["error_code"])
                                    fault["sample"] = dict(sample)
                                else:
                                    self.report["samples"].append(dict(fault))
                                    if request_attempted:
                                        try:
                                            self._book_work(time.monotonic() - measured_started, descriptor["sample_index"])
                                        except Exception as budget_error:
                                            fault["budget_error_type"] = type(budget_error).__name__
                                self._emit("validation", fault)
                                raise
                        self._checkpoint()
            finally:
                self._active_load_guard = None
                self._active_process = None
                close_error = None
                active_exception = sys.exc_info()[0] is not None
                try:
                    client.close()
                except BaseException as exc:
                    close_error = exc
                self.model_loaded = False
                returncode = process.poll() if process is not None else None
                exit_row["returncode"] = returncode
                exit_row["normal_shutdown"] = bool(worker["started"] and returncode == 0 and close_error is None)
                worker["closed"] = process is None or returncode is not None
                timing["closed_monotonic"] = time.monotonic()
                cleanup_payload = {"state": "worker_cleanup", "worker": dict(worker), "timing": dict(timing),
                                   "worker_exit": dict(exit_row)}
                if close_error is not None:
                    cleanup_payload["close_error_type"] = type(close_error).__name__
                self._emit("validation", cleanup_payload)
                if not active_exception and close_error is not None:
                    raise CalibrationFailure("worker_close_failed") from close_error
                if not active_exception and worker["started"] and returncode != 0:
                    raise CalibrationFailure("worker_exit_nonzero")
                if not worker["closed"]:
                    raise CalibrationFailure("worker_cleanup_failed")
            self._checkpoint()
            self.guard.finish_candidate()
        self.stage = "evaluating"
        final = self._checkpoint()
        final.update(hardware)
        self.report["identity_after"] = runtime_identity(self.spec.as_dict(), final)
        self._check()
        self.report["resource_valid"] = True
        self.report["status"] = "measured"
        self.report["budget"] = self.guard.summary()
        self.report["evaluation"] = evaluate_report(self.report)
        self._check()
        self._emit("validation", {"state": "evaluated", "identity": self.report["identity_after"],
                                   "evaluation": self.report["evaluation"], "budget": self.report["budget"]})
        if self.report["evaluation"]["verdict"] == "invalid":
            raise CalibrationFailure("evaluation_invalid")

    def run(self) -> dict:
        self.deadline = time.monotonic() + self.max_wait_s + 20 * 60
        with EventJournal(self.store.root / JOURNAL_NAME) as journal:
            self.journal = journal
            try:
                with _job_lease(self.store):
                    self.owned = True
                    self._emit("run_started", {"model_id": self.spec.model_id, "revision": self.spec.revision,
                                               "specification": specification(), "plan_id": plan_id(),
                                               "readiness_policy": asdict(self.readiness_policy),
                                               "load_monitor": dict(self.report["load_monitor"]),
                                               "max_wait_s": self.max_wait_s, "readiness_only": self.readiness_only})
                    try:
                        if self.readiness_only:
                            self._wait_ready()
                            self.report["status"] = "ready"
                        else:
                            self._measure()
                        self.stage = "finished"
                    except (KeyboardInterrupt, Exception) as exc:
                        terminal = getattr(exc, "status", "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed")
                        error_stage = self.stage
                        error_type = type(exc).__name__
                        self.stage = terminal if terminal in _STAGES else "failed"
                        error_code = getattr(exc, "code", {"IdentityError": "identity_validation_failed",
                                                           "BudgetError": "resource_budget_exceeded",
                                                           "EventJournalError": "history_failure",
                                                           "BackendUnavailable": "backend_unavailable"}.get(error_type, "calibration_failed"))
                        self.report.update(status=self.stage, error_type=error_type, error_stage=error_stage,
                                           error_code=error_code, resource_valid=False)
                        # Only our own closed validation errors may contribute
                        # details; third-party exceptions can contain user text.
                        if error_type in {"IdentityError", "BudgetError", "EventJournalError", "CalibrationFailure"}:
                            self.report["error_detail"] = str(exc)[:512]
                        if self.stage == "deferred":
                            self._emit("deferred", {"reason": self.report["error_code"], "model_loaded": False,
                                                    "completed_calls": len(self.report["samples"])})
                    finally:
                        if self.guard is not None:
                            self.report["budget"] = self.guard.summary()
                        self._emit("run_finished", {"status": self.stage,
                                                    "report_status": self.report["status"],
                                                    "readiness_only": self.readiness_only,
                                                    "attempted_calls": len(self.report["samples"]),
                                                    "completed_calls": sum(s.get("status") == "passed" for s in self.report["samples"]),
                                                    "error_code": self.report.get("error_code"),
                                                    "error_type": self.report.get("error_type"),
                                                    "error_stage": self.report.get("error_stage"),
                                                    "error_detail": self.report.get("error_detail"),
                                                    "evaluation": self.report.get("evaluation"),
                                                    "budget": self.report.get("budget"),
                                                    "activation_allowed": False})
            except CalibrationFailure as exc:
                if self.owned:
                    raise
                self.report.update(status=exc.status, error_code=exc.code)
                self._emit("deferred", {"reason": exc.code, "model_loaded": False, "completed_calls": 0})
            finally:
                self.owned = False
                self.journal = None
        return self.report


__all__ = ["CalibrationFailure", "CalibrationJob", "JOURNAL_NAME", "history", "model_lease", "status"]
