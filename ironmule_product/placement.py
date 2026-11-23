"""Interruptible, paired on-device probes for opt-in GGUF request placement."""
from __future__ import annotations

import os
import subprocess
import threading
import time
from collections import deque
from copy import deepcopy
from pathlib import Path

from ironmule.pilot import usable_cores

from .errors import BackendUnavailable, StateError
from .placement_native import NativeModel
from .placement_training import (
    CASES,
    ROUNDS,
    SCHEMA,
    check_trace,
    covered,
    decision,
    qualify,
    signature,
    valid_trial,
    workload,
)
from .state import _atomic_write, _read_json
from .types import GenerationRequest

IDLE_SECONDS = 15
REFRESH_SECONDS = 6 * 3600
FIXTURE = ("Count upwards in English, writing one number per line. Continue the sequence after this example. "
           "One, two, three, four, five. ")


def room_for_probes(cuda):
    """Use Linux pressure from the job's cgroup and device-specific NVIDIA utilization when available."""
    cpu_pressure_known = False
    for name, threshold in (("cpu", 25), ("memory", 5)):
        try:
            row = (Path("/sys/fs/cgroup") / f"{name}.pressure").read_text().splitlines()[0]
            avg = float(next(key.split("=")[1] for key in row.split() if key.startswith("avg10=")))
            cpu_pressure_known |= name == "cpu"
            if avg > threshold:
                return False
        except (OSError, ValueError, IndexError, StopIteration):
            pass
    if not cpu_pressure_known and os.getloadavg()[0] > 0.85 * usable_cores():
        return False
    if cuda:
        try:
            used = subprocess.run(["nvidia-smi", "-i", ",".join(cuda), "--query-gpu=utilization.gpu",
                                   "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3,
                                  check=True)
            values = [int(v) for v in used.stdout.splitlines()]
            return len(values) == len(cuda) and max(values) <= 25
        except (OSError, ValueError, subprocess.SubprocessError):
            return False
    return True


def binding_for(pool, options, native):
    gates = []
    for device in pool.devices:
        opts = options[device]
        gate = opts["ggml_gate"]
        if gate.get("passes") is not True or not gate.get("key"):
            raise ValueError("placement needs each worker's hardware-bound GGUF quality gate")
        gates.append({"key": gate["key"], "settings": gate.get("settings"), "sequences": opts.get("ggml_sequences"),
                      "environment": opts.get("ggml_worker_env", {}), "draft": gate.get("draft"),
                      "speculate": opts.get("speculate", False)})
    tuning = {k: v for k, v in os.environ.items() if k.startswith(("IRONMULE_GGML_", "IRONMULE_W4",
               "IRONMULE_KG", "IRONMULE_PILOT", "GGML_CUDA_", "IRONMULE_CUDA_"))}
    return {"configuration": signature(tuning), "cpu_budget": usable_cores(), "model": pool.spec.model_id, "revision": pool.spec.revision, "devices": pool.devices, "gates": gates,
            "trainer": native.version, "protocol": SCHEMA, "affinity": sorted(os.sched_getaffinity(0)),
            "thinking": [s.backend.thinking for s in pool.shards]}


class Interrupted(Exception):
    """Foreground work won the race to the device."""


class Placement:
    def __init__(self, pool, options, root, *, idle_seconds=IDLE_SECONDS):
        self.pool = pool
        directory = Path(root) / "placement"
        self.native = NativeModel(directory / "native")
        self.binding = binding_for(pool, options, self.native)
        self.path = directory / (signature(self.binding) + ".json")
        self.idle_seconds = idle_seconds
        self.lock = threading.Lock()
        self.disk_lock = threading.Lock()
        self.stop = threading.Event()
        self.inflight = {}
        self.sizes = deque(maxlen=128)
        self.serial = 0
        self.last_arrival = time.monotonic()
        self.policy = None
        self.misses = 0
        self.status_text = "waiting_for_idle"
        self.problem = None
        self.paused = False
        self.last_trial_s = None
        self.interruptions = 0
        self.last_decision = None
        self.trace = self._fresh()
        if self.path.exists():
            try:
                stored = _read_json(self.path)
                check_trace(stored)
                if stored["binding"] == self.binding:
                    self.paused = stored.get("paused") is True
                    self.trace = stored
                    if stored.get("failure"):
                        self.status_text = "reference_after_failed_measurements"
                    elif (not stored.get("withdrawn") and 0 <= time.time() - stored["started"] < REFRESH_SECONDS
                          and len(stored["trials"]) == CASES * ROUNDS):
                        self._grade(stored)
            except (OSError, StateError, ValueError, KeyError, TypeError, OverflowError) as exc:
                self.trace, self.policy = self._fresh(), None
                self.problem = f"prior measurements could not be restored: {exc}"[:250]
        self.thread = threading.Thread(target=self._run, daemon=True, name="placement-dream")
        self.thread.start()

    def _fresh(self, previous=None, reason=None):
        history = []
        if previous:
            history = list(previous.get("history", []))[-7:]
            history.append({"started": previous["started"], "trials": len(previous["trials"]), "ended": time.time(),
                            "reason": reason, "holdout": (previous.get("comparison") or {}).get("holdout"),
                            "failure": previous.get("failure")})
        return {"history": history, "schema": SCHEMA, "binding": self.binding, "identity": signature(self.binding),
                "started": time.time(), "plan": workload(self.sizes, len(self.pool.devices)), "trials": []}

    def _grade(self, trace):
        report = qualify(self.native, trace)
        self.policy = report if report["qualified"] else None
        self.status_text = "qualified" if self.policy else "reference"
        trace["comparison"] = report

    def _cancel_probes(self):
        for rid, shard in self.inflight.items():
            shard.cancel(rid)

    def arrival(self, request):
        shape = (sum(len(s) for _, s in request.messages), request.max_tokens)
        with self.lock:
            self.last_arrival = time.monotonic()
            self.sizes.append(shape)
            self.serial += 1
            if self.inflight:
                self.interruptions += 1
                self._cancel_probes()
        return shape

    def order(self, order, shape, queues):
        with self.lock:
            if self.policy and not 0 <= time.time() - self.trace["started"] < REFRESH_SECONDS:
                self.policy = None
                self.status_text = "waiting_for_refresh"
            policy = self.policy
            if (self.paused or policy is None or len(order) != len(queues) or not covered(self.trace["plan"], *shape, queues)):
                return order, None
            choice = decision(self.native, policy["weights"], *shape, queues, policy["next_shape"], policy["future_weight"])
            expected = self.native.predict(policy["weights"][choice], *shape, queues[choice])
            return [choice, *(i for i in order if i != choice)], (choice, expected, policy)

    def admitted(self, index, prediction):
        with self.lock:
            self.last_decision = {"worker": self.pool.devices[index], "reason": "shortest_queue"}
            if prediction and prediction[0] == index:
                _, ms, policy = prediction
                self.last_decision.update(reason="latency_model", predicted_completion_ms=round(ms, 2),
                                          future_weight=policy["future_weight"])
            elif prediction:
                self.last_decision["reason"] = "preferred_worker_full"

    def _withdraw(self, policy, reason):
        with self.lock:
            if policy is not self.policy or policy is None:
                return
            self.policy = None
            self.trace["withdrawn"] = True
            self.status_text = reason
        self._save()

    def watch(self, events, prediction):
        try:
            for event in events:
                if event.get("type") == "done" and prediction:
                    _, expected, policy = prediction
                    ms = event.get("service_ns", 0) / 1e6
                    if ms > 0:
                        with self.lock:
                            if policy is self.policy:
                                self.misses = self.misses + 1 if ms > 2 * expected else 0
                            drift = policy is self.policy and self.misses >= 3
                        if drift:
                            self._withdraw(policy, "withdrawn_after_latency_drift")
                yield event
        except BackendUnavailable:
            if prediction:
                self._withdraw(prediction[2], "withdrawn_after_worker_failure")
            raise
        finally:
            close = getattr(events, "close", None)
            if close:
                close()

    def _save(self):
        # Serialize checkpoints without holding the admission lock across file I/O or fsync.
        with self.disk_lock:
            with self.lock:
                snapshot = deepcopy(self.trace)
            try:
                _atomic_write(self.path, snapshot)
            except (OSError, StateError) as exc:
                with self.lock:
                    self.problem = f"could not save placement state: {exc}"[:250]

    def _submit(self, shard, request, serial, queue):
        with self.lock:
            if self.stop.is_set() or self.paused or serial != self.serial or shard.outstanding() != queue:
                raise Interrupted()
            events = shard.stream(request)
            self.inflight[request.request_id] = shard
            return events

    def _probe(self, device, case, serial, nonce):
        size, limit, queues = case
        shard = self.pool.shards[device]
        # Distinct deterministic prefixes avoid a cache hit from the immediately preceding repetition.
        prefix = f"Exercise {nonce:06d}. "
        content = (prefix + FIXTURE * (1 + size // len(FIXTURE)))[:size]
        def request(text, cap):
            return GenerationRequest(self.pool.spec.model_id, (("user", text),), cap, hardware_probe=True)
        target = request(content, limit)
        loaded = request("Write the integers one through two hundred, one per line.", 128)
        opened = []
        try:
            if queues[device]:
                opened.append((loaded, self._submit(shard, loaded, serial, 0)))
            events = self._submit(shard, target, serial, queues[device])
            opened.append((target, events))
            for event in events:
                if event.get("type") == "done":
                    return [event["service_ns"] / 1e6, (event.get("ttft_ns") or 0) / 1e6, event["completion_tokens"]]
            raise ValueError("lane ended without a completed measurement")
        except Exception as exc:
            with self.lock:
                if serial != self.serial or self.stop.is_set():
                    raise Interrupted() from exc
            raise
        finally:
            with self.lock:
                for req, events in opened:
                    self.inflight.pop(req.request_id, None)
                    shard.cancel(req.request_id)
                    close = getattr(events, "close", None)
                    if close:
                        close()

    def step(self):
        with self.lock:
            if self.stop.is_set() or self.paused or time.monotonic() - self.last_arrival < self.idle_seconds:
                return
            done = len(self.trace["trials"]) == CASES * ROUNDS or bool(self.trace.get("failure"))
            shifted = done and self.serial >= 32 and workload(self.sizes, len(self.pool.devices)) != self.trace["plan"]
            age = time.time() - (self.trace["started"] if done else self.trace.get("updated", self.trace["started"]))
            expired = not 0 <= age < (REFRESH_SECONDS if done else 7 * 86400)
            reset = self.trace.get("withdrawn") or shifted or expired
            if reset:
                self.policy, self.misses, self.serial = None, 0, 0
                reason = "withdrawn" if self.trace.get("withdrawn") else "workload_shift" if shifted else "expired"
                self.trace = self._fresh(self.trace, reason)
            elif done:
                return
            serial = self.serial
            index = len(self.trace["trials"])
            case = self.trace["plan"][index % CASES]
        if reset:
            self._save()
        if any(not shard.accepting() or shard.outstanding() for shard in self.pool.shards):
            return
        cuda = [d for d in self.pool.devices if d not in ("cpu", "local")]
        if "local" in self.pool.devices:
            from ironmule_devices import ggml_devices

            from .device_pool import cuda_devices
            providers = {r["provider"] for r in ggml_devices()}
            if providers == {"cuda"}:
                cuda = cuda_devices()
            elif providers:
                sensors = list(Path("/sys/class/drm").glob("card[0-9]*/device/gpu_busy_percent"))
                if not sensors or any(int(p.read_text()) > 25 for p in sensors):
                    with self.lock:
                        self.status_text = "waiting_for_spare_capacity"
                    return
        if not room_for_probes(cuda):
            with self.lock:
                self.status_text = "waiting_for_spare_capacity"
            return
        started = time.monotonic()
        n = len(self.pool.shards)
        arms = [None] * n
        for i in range(n):
            # Alternate which worker goes first across repetitions of the same workload.
            arm = (i + index // CASES + index % CASES) % n
            arms[arm] = self._probe(arm, case, serial, index)
        row = {"case": case, "arms": arms}
        if not valid_trial(row, case):
            raise ValueError("incomplete timing or token count")
        with self.lock:
            if self.stop.is_set() or self.serial != serial:
                return
            self.trace.pop("failed_attempts", None)
            self.trace["trials"].append(row)
            self.trace["updated"] = time.time()
            self.last_trial_s = time.monotonic() - started
            row["wall_s"] = self.last_trial_s
            self.status_text, self.problem = "measuring", None
            if len(self.trace["trials"]) == CASES * ROUNDS:
                self._grade(self.trace)
        self._save()

    def _run(self):
        delay = 1
        while not self.stop.wait(delay):
            try:
                self.step()
                delay = 1
            except Interrupted:
                delay = 1
            except Exception as exc:  # noqa: BLE001 - a trainer failure leaves the reference serving
                with self.lock:
                    self.problem = f"{type(exc).__name__}: {exc}"[:250]
                    self.trace["failed_attempts"] = self.trace.get("failed_attempts", 0) + 1
                    if self.trace["failed_attempts"] >= 3 or len(self.trace["trials"]) == CASES * ROUNDS:
                        self.trace["failure"] = self.problem
                        self.policy = None
                        self.status_text = "reference_after_failed_measurements"
                self._save()
                delay = min(60, 2 * delay)

    def status(self):
        with self.lock:
            rows = self.trace["trials"]
            times = [r["wall_s"] for r in rows if "wall_s" in r]
            remaining = (CASES * ROUNDS - len(rows)) * (1 + sum(times) / len(times)) if times else None
            return {"phase": "paused" if self.paused else self.status_text,
                    "active": self.policy is not None and not self.paused, "probing": bool(self.inflight), "paired_trials": len(rows),
                    "target_trials": CASES * ROUNDS, "trace": str(self.path), "interruptions": self.interruptions,
                    "remaining_idle_seconds": remaining, "holdout": (self.trace.get("comparison") or {}).get("holdout"),
                    "last_decision": self.last_decision, "error": self.problem or self.trace.get("failure")}

    def control(self, action):
        if action not in ("pause", "resume"):
            raise ValueError("placement supports pause and resume")
        with self.lock:
            self.paused = action == "pause"
            self.serial += 1
            self._cancel_probes()
            self.trace["paused"] = self.paused
        self._save()

    def close(self):
        with self.lock:
            self.stop.set()
            self._cancel_probes()
        self.thread.join(timeout=5)
