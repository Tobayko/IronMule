"""HW1: one model worker per GPU, behind join-the-shortest-queue dispatch.

Every device gets its own `ProductService`: its own worker process, pinned to that device
with `CUDA_VISIBLE_DEVICES`, its own scheduler, grouping, deadlines and STAB1 recovery, all
unchanged. A request is placed on exactly one of them when it is admitted and never moves,
so its tokens are what that one worker returns on its own.

Placement is join-the-shortest-queue: the device with the fewest admitted requests that are
not yet retired, ties to the lower index. For identical servers with exponential service
it is the optimal assignment (Winston 1977, J. Appl. Prob. 14, 181-189; Weber 1978 extends it
to non-decreasing hazard rates); grouped decode is neither, so it is a heuristic here, and
BACKLOG HW1 holds the measurement it still needs.

Live learning and idle tuning model one worker and are not run under a pool; each device
writes its own serving log, so the single-worker dreamer never mixes their traffic.
"""

from __future__ import annotations

import os
import subprocess
import threading
from typing import Any, Callable, Iterator

from .errors import BackendUnavailable, InvalidRequest, Overloaded
from .service import STABILITY_LOG
from .serving_policy import ServingLog
from .types import GenerationRequest

_SUMMED = ("queued_requests", "active_requests", "completed_requests", "failed_requests", "cancelled_requests")


def cuda_devices(environ: Any = os.environ, run: Callable[..., Any] = subprocess.run) -> list[str]:
    """The CUDA devices a worker may be pinned to: the inherited `CUDA_VISIBLE_DEVICES` entries
    when it is set (an empty value hides every device), else the indices `nvidia-smi` lists."""
    visible = environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return [entry.strip() for entry in visible.split(",") if entry.strip()]
    try:
        listed = run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                     capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return []
    if listed.returncode != 0:
        return []
    return [line.strip() for line in listed.stdout.splitlines() if line.strip().isdigit()]


def gpu_name(devices: list[str], run: Callable[..., Any] = subprocess.run) -> str | None:
    """LD1: the name of the listed GPUs as `nvidia-smi` reports it (one name, or names joined by `+`
    when they differ), which learned-dispatch evidence binds; None when it cannot be read."""
    try:
        listed = run(["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
                     capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if listed.returncode != 0:
        return None
    names = {}
    for line in listed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3:
            names[parts[0]], names[parts[1]] = parts[2], parts[2]
    found = [names.get(device) for device in devices]
    if not found or any(name is None for name in found):
        return None
    return "+".join(sorted(set(found)))


def select_devices(value: str, available: list[str]) -> list[str]:
    """`all`, or a comma-separated list of devices from `available`, each at most once."""
    if value == "all":
        if not available:
            raise InvalidRequest("--gpus all found no CUDA device (nvidia-smi or CUDA_VISIBLE_DEVICES)")
        return list(available)
    chosen = [entry.strip() for entry in value.split(",")]
    if not chosen or any(not entry for entry in chosen):
        raise InvalidRequest("--gpus takes `all` or device indices such as 0,1")
    if len(set(chosen)) != len(chosen):
        raise InvalidRequest("--gpus names a device twice: two workers would share one card")
    missing = [entry for entry in chosen if entry not in available]
    if missing:
        raise InvalidRequest(f"--gpus names devices this machine does not show: {', '.join(missing)}")
    return chosen


class _DeviceLog:
    """One shared bounded log, whose own lock serialises appends and rotation across devices;
    every record carries the device it came from."""

    def __init__(self, log: ServingLog, device: str):
        self._log, self._device = log, device

    def append(self, record: dict[str, Any]) -> None:
        self._log.append({**record, "device": self._device})


class DevicePool:
    """The `ProductService` surface the HTTP server uses, over one service per device."""

    learning = None
    dispatch = None  # LD1: an evidence-bound learned_dispatch.LearnedDispatch; attach() also hands it to each device

    def __init__(self, shards: list[Any], devices: list[str]):
        if not shards or len(shards) != len(devices):
            raise ValueError("a device pool needs one service per device")
        self.shards, self.devices = list(shards), list(devices)
        self.spec = shards[0].spec
        self.store = shards[0].store
        self._lock = threading.Lock()  # placement only; each device's service keeps its own
        root = getattr(self.store, "root", None)
        if root is not None:
            shared = ServingLog(root, 4 << 20, STABILITY_LOG)
            for shard, device in zip(self.shards, self.devices):
                shard.stability_log = _DeviceLog(shared, device)
                shard.serving_log = ServingLog(root, name=f"serving-log.cuda-{device}.jsonl")

    def attach(self, dispatch: Any) -> None:
        """LD1: place by the learned policy, and let each device order and size its groups by it."""
        self.dispatch = dispatch
        for shard in self.shards:
            shard.dispatch = dispatch

    def _order(self, request: GenerationRequest | None = None) -> list[int]:
        """Accepting devices, least outstanding work first, ties to the lower index; an active learned
        dispatch (LD1) moves its choice for `request` to the front."""
        accepting = [index for index, shard in enumerate(self.shards) if shard.accepting()]
        order = sorted(accepting, key=lambda index: (self.shards[index].outstanding(), index))
        dispatch = self.dispatch
        if request is not None and dispatch is not None and dispatch.active and len(order) > 1:
            card = dispatch.card(request, [shard.admitted_requests() for shard in self.shards])
            if card in order:
                order.remove(card)
                order.insert(0, card)
        return order

    def stream(self, request: GenerationRequest) -> Iterator[dict[str, Any]]:
        # Under one lock, so two arrivals never both see the same device as the shortest. Only
        # placement adds a session to a device; its own scheduler only ever retires them.
        with self._lock:
            if any(shard.holds(request.request_id) for shard in self.shards):
                raise InvalidRequest("duplicate request id")
            order = self._order(request)
            if not order:
                raise BackendUnavailable("no model worker is ready")
            overloaded: Overloaded | None = None
            for index in order:
                try:
                    return self.shards[index].stream(request)
                except Overloaded as exc:  # this device's queue is full; a longer one may not be
                    overloaded = exc
            raise overloaded

    def cancel(self, request_id: str) -> bool:
        return any(shard.cancel(request_id) for shard in self.shards)

    def health(self) -> dict[str, Any]:
        healths = [shard.health() for shard in self.shards]
        result = dict(healths[0])
        result["ready"] = any(health["ready"] for health in healths)
        for key in _SUMMED:
            result[key] = sum(health.get(key, 0) for health in healths)
        for key in ("last_batch_size", "batch_count", "worker", "in_flight", "groups_started",
                    "oldest_queued_s", "learning"):
            result.pop(key, None)
        result["devices"] = [{"cuda_device": device, "ready": health["ready"], "device": health.get("device"),
                              "worker": health.get("worker"), "in_flight": health.get("in_flight"),
                              "queued_requests": health.get("queued_requests"),
                              "active_requests": health.get("active_requests"),
                              "completed_requests": health.get("completed_requests")}
                             for device, health in zip(self.devices, healths)]
        dispatch = self.dispatch
        result["dispatch"] = "learned" if dispatch is not None and dispatch.active else "join_shortest_queue"
        if dispatch is not None:
            result["learned_dispatch"] = dispatch.status()
        return result

    def learning_status(self) -> dict[str, Any]:
        health = self.health()
        serving = {"ready": health["ready"], "device": health.get("device"), "backend": health["backend"],
                   "loaded_model": health["loaded_model"], "queued_requests": health["queued_requests"],
                   "replay_phase": None}
        reason = "live learning models one worker; this server spreads requests over several GPUs (HW1)"
        return {"schema": "ironmule.live_learning.v1", "mode": "off", "eligible": False, "reason": reason,
                "strategy": {"active": "reference", "version": None, "reason": reason},
                "candidate": None, "serving": serving}

    def learning_control(self, action: str, arm: str | None = None,
                         evidence: dict[str, Any] | None = None) -> dict[str, Any]:
        raise InvalidRequest("learning is off: it models one worker, and this server runs one per GPU (HW1)")

    def models(self) -> list[dict[str, Any]]:
        ready = self.health()["ready"]
        return [{"id": spec.model_id, "object": "model", "owned_by": "local",
                 "revision": spec.revision, "loaded": self.spec == spec and ready}
                for spec in self.store.models()]

    def close(self) -> None:
        for shard in self.shards:
            shard.close()


def start_pool(store: Any, spec: Any, devices: list[str], make_backend: Callable[[str], Any],
               service_type: Callable[..., Any]) -> DevicePool:
    """Start one worker per device at once (each loads its own copy of the model) and wrap them.

    All or nothing: a device that was asked for and does not come up stops the server, because
    serving on fewer cards than requested would hide the failure behind slower answers."""
    backends = [make_backend(device) for device in devices]
    failures: list[BaseException | None] = [None] * len(backends)

    def start(index: int) -> None:
        try:
            backends[index].start()
        except BaseException as exc:  # noqa: BLE001 - reported below, after every start has ended
            failures[index] = exc

    threads = [threading.Thread(target=start, args=(index,), daemon=True) for index in range(len(backends))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    failed = [(device, exc) for device, exc in zip(devices, failures) if exc is not None]
    if failed:
        for backend in backends:
            backend.close()
        device, exc = failed[0]
        raise BackendUnavailable(f"the worker for CUDA device {device} did not start") from exc
    shards = []
    for device, backend in zip(devices, backends):
        shard = service_type(store, backend=backend, spec=spec)
        shard.make_backend = (lambda device=device: make_backend(device))  # STAB1, on the same card
        shards.append(shard)
    return DevicePool(shards, devices)
