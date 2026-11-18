"""Bounded idle admission experiments on shapes observed in this GGUF worker."""
from __future__ import annotations

from collections import deque
import math
import os
from pathlib import Path
import time

from ironmule import compare_learning as core
from .state import _atomic_write, _read_json, StateError

TRIAL_SECONDS = 60
COOLDOWN_SECONDS = 300
MAX_COHORTS = 64


def live_path(root: Path, pid: int) -> Path:
    return root / f"admission-live-{pid}.json"


def live_status(root: str, pid: int) -> dict | None:
    try:
        value = _read_json(live_path(Path(root), pid))
        if not isinstance(value, dict) or value.get("pid") != pid:
            return None
        value.pop("pid")
        value["remaining_cooldown_s"] = max(0, math.ceil(value.pop("retry_at") - time.monotonic()))
        return value
    except (OSError, ValueError, TypeError, KeyError, StateError):
        return None


class AdmissionLearning:
    def __init__(self, root: Path, gguf: str, engine, binding: dict, *, enabled: bool):
        self.root, self.gguf, self.engine, self.binding = root, gguf, engine, binding
        self.enabled = enabled
        self.samples = deque(maxlen=MAX_COHORTS)
        self.checked = None
        self.refresh_rules = False
        self.retry_at = 0.0
        self.status = "waiting" if enabled else "disabled"
        self.last = None
        self.path = live_path(root, os.getpid())
        self._last_view = None
        self.engine.admission_budgets = core.read_rules(root, binding)
        self.engine.admission_budget_hits = 0
        self.engine.admission_observer = self.observe if enabled else None
        self._publish()

    def _publish(self):
        view = {"pid": os.getpid(), "enabled": self.enabled, "status": self.status,
                "observations": len(self.samples), "rules": len(self.engine.admission_budgets),
                "retry_at": self.retry_at, "last": self.last}
        if view != self._last_view:
            try:
                _atomic_write(self.path, view)
                self._last_view = view
            except (StateError, OSError):
                pass  # optional telemetry must not interrupt serving

    def observe(self, started: int, lengths: tuple[int, ...], limit: int, rate: float, ttft: float) -> None:
        """Only numeric measurements cross this boundary. Append never waits on the idle worker."""
        self.samples.append(core.Observation(started, lengths, limit, rate, ttft))

    def _room_for_child(self) -> bool:
        from ironmule.ggml_engine import gpu_memory_bytes, memory_bytes
        model = os.stat(self.gguf).st_size
        kv = self.engine.seqs * self.engine.seq_ctx * self.engine.kv_cell_bytes()
        host, total = memory_bytes()
        if host < model + (0 if self.engine.gpu_layers else kv) + max(512 * 2**20, total // 20):
            return False
        if self.engine.gpu_layers:
            device = gpu_memory_bytes()
            return bool(device and device[0] >= model + kv + max(512 * 2**20, device[1] // 20))
        return True

    def idle(self, interrupted) -> None:
        self._publish()
        if not self.enabled or interrupted():
            return
        if self.refresh_rules:
            self.engine.admission_budgets = core.read_rules(self.root, self.binding)
            self.refresh_rules = False
            self._publish()
        if time.monotonic() < self.retry_at:
            return
        observations = self.samples.copy()
        if not observations or observations[-1].sample_id == self.checked:
            return
        self.checked = observations[-1].sample_id
        choice = core.select(list(observations), binding=self.binding)
        if choice is None:
            return
        self.retry_at = time.monotonic() + COOLDOWN_SECONDS
        if not self._room_for_child():
            self.status, self.last = "deferred", {"reason": "insufficient memory for isolated holdout"}
            self._publish()
            return
        # The native trial runs in a killable child with its own model/context. Never take the
        # serving worker's execution lock here. New foreground work latches cancellation.
        selection = {k: choice[k] for k in ("concurrency", "max_tokens", "samples", "focus") if k in choice}
        self.status, self.last = "testing", {"selection": selection}
        self._publish()
        result = core.respond(list(observations), binding=self.binding, gguf=self.gguf, state=self.root,
                              deadline=time.monotonic() + TRIAL_SECONDS, interrupted=interrupted)
        self.status = result["status"]
        self.last = {k: result[k] for k in ("status", "reason", "reasons", "workload_key", "candidate_budget",
                                             "incumbent_budget") if k in result}
        self.last["selection"] = selection
        # A dict swap is safe while a foreground job reads the rule for its next admission.
        if self.status == "qualified":
            if interrupted():
                self.refresh_rules = True
            else:
                self.engine.admission_budgets = core.read_rules(self.root, self.binding)
        self._publish()
