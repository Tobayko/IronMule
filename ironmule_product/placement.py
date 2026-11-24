"""Interruptible, paired on-device probes for opt-in GGUF request placement."""
from __future__ import annotations

import hashlib
import os
import subprocess
import threading
import time
from collections import deque
from copy import deepcopy
from pathlib import Path

from ironmule.pilot import usable_cores

from .errors import BackendUnavailable, StateError
from .placement_devices import accelerators_quiet
from .placement_native import NativeModel
from .placement_training import (
    CASES,
    ROUNDS,
    SCHEMA,
    check_trace,
    covered,
    choose_route,
    qualify,
    remaining_idle_seconds,
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
            return len(values) == len(cuda) and all(0 <= value <= 25 for value in values)
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
            "trainer": native.version, "reward_trainer": native.reward_version, "protocol": SCHEMA, "affinity": sorted(os.sched_getaffinity(0)),
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
        self.backlogs = deque(maxlen=128)
        self.serial = 0
        self.idle_since = time.monotonic()
        self.idle_gaps = deque(maxlen=32)
        self.standard_wait = False
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
                "started": time.time(), "plan": workload(self.sizes, len(self.pool.devices), self.backlogs), "trials": []}

    def _grade(self, trace):
        report = qualify(self.native, trace)
        with self.lock:
            if trace is self.trace and not self.stop.is_set():
                self.policy = report if report["qualified"] else None
                self.status_text = "qualified" if self.policy else "reference"
                trace["comparison"] = report

    def _cancel_probes(self):
        for rid, shard in self.inflight.items():
            shard.cancel(rid)

    def arrival(self, request):
        shape = (sum(len(s) for _, s in request.messages), request.max_tokens)
        with self.lock:
            now = time.monotonic()
            if self.serial and self.idle_since is not None:
                self.idle_gaps.append(now - self.idle_since)
            self.idle_since = None
            self.sizes.append(shape)
            self.serial += 1
            if self.inflight:
                self.interruptions += 1
                self.standard_wait = True
                self._cancel_probes()
        return shape

    def order(self, order, shape, queues):
        with self.lock:
            if len(order) == len(queues):
                self.backlogs.append(tuple(queues))
            if self.policy and not 0 <= time.time() - self.trace["started"] < REFRESH_SECONDS:
                self.policy = None
                self.status_text = "waiting_for_refresh"
            policy = self.policy
            if (self.paused or policy is None or len(order) != len(queues) or not covered(self.trace["plan"], *shape, queues)):
                return order, None
            choice = choose_route(self.native, policy, *shape, queues)
            expected = self.native.predict(policy["weights"][choice], *shape, queues[choice])
            return [choice, *(i for i in order if i != choice)], (choice, expected, policy)

    def admitted(self, index, prediction):
        with self.lock:
            self.last_decision = {"worker": self.pool.devices[index], "reason": "shortest_queue"}
            if prediction and prediction[0] == index:
                _, ms, policy = prediction
                self.last_decision.update(reason=policy["method"], predicted_completion_ms=round(ms, 2),
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
        opened = []
        try:
            # Recreate the entire measured queue, including contention on other workers.
            for lane, depth in enumerate(queues):
                peer = self.pool.shards[lane]
                for q in range(depth):
                    load = request("Write the integers one through two hundred, one per line.", 128)
                    events = self._submit(peer, load, serial, q)
                    opened.append((peer, load, events))
            if any(peer.outstanding() != depth for peer, depth in zip(self.pool.shards, queues)):
                raise Interrupted()
            events = self._submit(shard, target, serial, queues[device])
            opened.append((shard, target, events))
            text_hash = hashlib.sha256()
            visible = False
            for event in events:
                if event.get("type") == "token":
                    piece = event["text"]
                    text_hash.update(piece.encode("utf-8"))
                    visible |= bool(piece.strip())
                elif event.get("type") == "done":
                    stop = event.get("finish_reason")
                    if not visible or stop not in ("length", "stop"):
                        raise ValueError("fixture produced no visible answer or valid stop")
                    timing = [event["service_ns"] / 1e6, (event.get("ttft_ns") or 0) / 1e6, event["completion_tokens"]]
                    return timing, signature([text_hash.hexdigest(), stop])
            raise ValueError("lane ended without a completed measurement")
        except Exception as exc:
            with self.lock:
                if serial != self.serial or self.stop.is_set():
                    raise Interrupted() from exc
            raise
        finally:
            with self.lock:
                for peer, req, events in opened:
                    self.inflight.pop(req.request_id, None)
                    peer.cancel(req.request_id)
                    close = getattr(events, "close", None)
                    if close:
                        close()
            # Cancellation is asynchronous. The next arm starts with a clean baseline.
            deadline = time.monotonic() + 5
            while any(peer.holds(req.request_id) for peer, req, _ in opened):
                if self.stop.wait(0.01) or serial != self.serial or time.monotonic() >= deadline:
                    raise Interrupted()

    def _idle_delay(self):
        # A whole case must fit in a typical gap. After preemption, wait for a long lull
        # before trying that same case again. No foreground request waits for the estimate.
        if self.standard_wait or len(self.idle_gaps) < 8 or self.idle_seconds <= 2:
            return self.idle_seconds
        gaps = sorted(self.idle_gaps)
        lower_quartile = gaps[(len(gaps) - 1) // 4]
        work = self.last_trial_s if self.last_trial_s is not None else 2.0
        if 2 + work <= lower_quartile < self.idle_seconds:
            return 2.0
        return self.idle_seconds

    def step(self):
        with self.lock:
            if self.stop.is_set() or self.paused:
                return
            serial = self.serial
        available = all(shard.accepting() and not shard.outstanding() for shard in self.pool.shards)
        with self.lock:
            if self.stop.is_set() or self.paused or serial != self.serial:
                return
            if not available:
                self.idle_since = None
                return
            now = time.monotonic()
            if self.idle_since is None:
                self.idle_since = now
            if now - self.idle_since < self._idle_delay():
                return
            done = len(self.trace["trials"]) == CASES * ROUNDS or bool(self.trace.get("failure"))
            shifted = done and self.serial >= 32 and workload(self.sizes, len(self.pool.devices), self.backlogs) != self.trace["plan"]
            age = time.time() - (self.trace["started"] if done else self.trace.get("updated", self.trace["started"]))
            expired = not 0 <= age < (REFRESH_SECONDS if done else 7 * 86400)
            reset = self.trace.get("withdrawn") or shifted or expired
            if reset:
                self.policy, self.misses, self.serial = None, 0, 0
                reason = "withdrawn" if self.trace.get("withdrawn") else "workload_shift" if shifted else "expired"
                self.trace = self._fresh(self.trace, reason)
                self.status_text, self.problem = "recalibrating", None
            elif done:
                return
            serial = self.serial
            index = len(self.trace["trials"])
            case = self.trace["plan"][index % CASES]
        if reset:
            self._save()
        cuda = accelerators_quiet(self.pool.devices)
        if cuda is None or not room_for_probes(cuda):
            with self.lock:
                self.status_text = "waiting_for_spare_capacity"
            return
        started = time.monotonic()
        n = len(self.pool.shards)
        arms = [None] * n
        outputs = [None] * n
        for i in range(n):
            # Alternate which worker goes first across repetitions of the same workload.
            arm = (i + index // CASES + index % CASES) % n
            arms[arm], outputs[arm] = self._probe(arm, case, serial, index)
        row = {"case": case, "arms": arms, "outputs": outputs}
        if not valid_trial(row, case):
            raise ValueError("incomplete timing or token count")
        with self.lock:
            if self.stop.is_set() or self.serial != serial:
                return
            self.trace.pop("failed_attempts", None)
            self.trace["trials"].append(row)
            self.trace["updated"] = time.time()
            self.standard_wait = False
            self.last_trial_s = time.monotonic() - started
            row["wall_s"] = self.last_trial_s
            self.status_text, self.problem = "measuring", None
            complete = len(self.trace["trials"]) == CASES * ROUNDS
            trace = self.trace
            if complete:
                self.status_text = "training"
        if complete:
            self._grade(trace)
        self._save()

    def _run(self):
        delay = 1
        while not self.stop.wait(delay):
            try:
                self.step()
                delay = 1
            except Interrupted:
                with self.lock:
                    self.standard_wait = True
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
            return {"phase": "paused" if self.paused else self.status_text,
                    "method": self.policy["method"] if self.policy and not self.paused else "shortest_queue",
                    "active": self.policy is not None and not self.paused, "probing": bool(self.inflight), "paired_trials": len(rows),
                    "target_trials": CASES * ROUNDS, "trace": str(self.path), "interruptions": self.interruptions,
                    "idle_delay_seconds": self._idle_delay(), "observed_idle_gaps": len(self.idle_gaps),
                    "remaining_idle_seconds": remaining_idle_seconds(self.trace), "holdout": (self.trace.get("comparison") or {}).get("holdout"),
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
