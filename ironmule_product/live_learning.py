"""RSI3-LIVE: the serving scheduler's opt-in learner (cost model, sequential RL, dreaming).

One decision owner. When ``ironmule serve --learning active|paused`` attaches this object, every
group the scheduler starts is decided here: a group width limit and a gather wait from a fixed
catalogue (action 0 is the service's own reference schedule). Nothing else changes: the model,
numeric plan, outputs, isolation and caches follow the existing grouped batch-1 contract, and an
execution path that cannot group requests keeps the reference with a stated reason.

``active``: the reference (or a qualified policy) serves; a bounded share of decisions explores
another catalogue action unless a recent decision failed, was slow, or the queue is near its limit.
Each finished group closes one real transition (state before the decision, the action actually
executed, outcomes, the state after). It updates the native cost model and two Q-learners fed the
same transitions: the main learner, which also dreams (bounded rollouts through the model) while
nothing waits, and a twin that never dreams (the "RL without dreaming" comparison arm).

After enough real updates the main learner's greedy policy is frozen as a versioned candidate. It
is qualified while the server is idle: A/B/AA windows replay one held-out trace (arrival times,
prompt sizes and output limits recorded after the freeze; prompts are a fixed filler, never user
text) through the real backend, judged by the optimizer's independent rule on measured cost, with
identical outputs required. New demand cancels a window, which is discarded. Only a qualified
candidate serves; failed requests under it or an operator rollback withdraw it. ``compare`` runs
reference, best fixed, a queue rule, RL and RL with dreaming on further held-out traces.
``paused`` records, explores, dreams and qualifies nothing and keeps whatever already serves.
Prompts and responses are never stored; experience is metadata bound to hardware, model, plan,
catalogue, reward contract and learner library.
"""

from __future__ import annotations

import base64
from collections import deque
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import threading
import time
from typing import Any, Callable, NamedTuple

from friday_evidence.canonical import canonical_sha256

from .code_identity import sources_digest
from .serving_policy import ServingLog

SCHEMA = "ironmule.live_learning.v1"
MAX_ACTIONS = 12
STATES = 30
MODES = ("active", "paused")
ARMS = ("reference", "best_fixed", "queue_rule", "rl", "rl_dream")
CONTRACT = {
    "schema": "ironmule.live_learning.contract.v3",
    # Reward per decision interval, lower cost is better: request-seconds in the system, plus
    # ttft_weight x summed time to first content, plus a penalty per failed, cancelled, timed-out
    # or rejected request. The penalty exceeds (1 + ttft_weight) x the request deadline, so
    # dropping work is never cheaper than serving it.
    "reward": {"ttft_weight": 1.0, "penalty": "(1 + ttft_weight) x request_timeout_s + 1", "scale_s": 10.0,
               "discount": "exp(-interval_s / tau_s)", "tau_s": 30.0},
    "actions": {"widths": [1, 2, 4, 8], "gather_ms": [0, 50, 150], "reference": "service policy",
                "join": "off; learning is unavailable while join is on"},
    # The native state bins: queue ranges x arrival rate (<0.1, <1, >=1 per s) x prompt (<=2000 chars).
    "state_bins": {"queue": [[1, 1], [2, 2], [3, 4], [5, 8], [9, 4096]], "rate": 3, "prompt": 2},
    # The contextual bandit (P3 comparison arm) is the same native learner with a vanishing time
    # constant: it values an action by its immediate cost alone, no bootstrapped future.
    "bandit": {"tau_s": 1e-9},
    "learner": {"min_support": 3, "horizon": 6, "alpha_min": 0.05, "synthetic_weight": 0.5,
                "max_rel_std": 0.5, "ring": 4096},
    "exploration": {"share": 0.25, "cooldown_decisions": 3, "queue_guard": 0.75, "ttft_slo_s": 10.0},
    # Hourly caps unchanged from P1. Planning is coupled to new information: a real transition
    # the model did not support, or missed by more than one predictive standard deviation, adds
    # 50 steps to an allowance; one it forecast within that adds 10; one without first content
    # adds nothing (cap 2,000). The hour is also paced: steps so far may not exceed the hourly
    # cap times the elapsed share of the hour plus the allowance cap. Counters survive pause
    # and restart.
    "dreaming": {"slice_ms": 10, "rollouts_per_slice": 32, "idle_call_ms": 50,
                 "cpu_s_per_hour": 60, "rollout_steps_per_hour": 20000,
                 "grant_new_information": 50, "grant_expected": 10, "allowance_cap": 2000},
    # First content = known waiting (ages of the waiting members, gather) + expected members x the
    # model's dispatch-to-first-content time. Forecasts are scored before their update, per
    # segment of 100 transitions. Criterion kept from RSI3-LIVE (b): mean relative error of the
    # summed first content (denominator max(|actual|, 1 ms)) below 1.0 on the latest segment.
    "model": {"segment": 100, "criterion": {"metric": "ttft_sum mean relative error", "below": 1.0},
              "diagnostics": "absolute error, relative error with a 50 ms floor, two-sigma coverage, by group size"},
    # RSI3-LIVE (d): a real request must not wait for replay. Extra wait = when the scheduler takes it
    # up minus its arrival, when it arrived during a replay window (the schedule's own gather wait
    # after take-up is reported apart); above 100 ms at the 95th percentile of the
    # last 64 such requests, replay qualification and comparison are suspended for this identity.
    "preemption": {"p95_limit_s": 0.1, "samples": 64},
    "qualification": {"protocol": 2, "pairs": 8, "min_gain": 0.03, "max_regression": 0.10,
                      "family_alpha": 0.05, "max_trials": 4, "candidate_every": 40, "min_coverage": 0.2,
                      "candidate": "greedy supported policy; actions that behave like the reference over a "
                                   "state's whole queue range count as the reference",
                      "order": "A,B,AA then B,A,AA; AA last", "window": "one replayed held-out trace",
                      "trace_requests": 16, "max_gap_s": 2.0, "max_tokens_cap": 128,
                      "holdout": "arrivals recorded after the candidate froze",
                      "cost": "mean per request of e2e + ttft_weight x ttft + penalty x failed",
                      "correct": "identical outputs per request in A, B and AA; no failed request in B",
                      "replay_s_per_hour": 600, "start_after_idle_s": 5.0},
    "compare": {"holdout_traces": 4, "selection": "one window per fixed catalogue action on the last trace "
                "recorded before the request", "queue_rule": "width 1 alone, else min(4, capacity); no gather",
                "release": "rl_dream beats the better simple arm by min_gain on every holdout trace with "
                "identical outputs"},
    "withdrawal": {"failed_windows_in_a_row": 2, "window_decisions": 4, "window_requests": 24},
    "retention": {"days": 14, "experience_log_bytes": 2 * 1024 * 1024, "checkpoint_bytes": 8 * 1024 * 1024,  # three learners with full rings fit
                  "arrivals": 2048},
    "save_every_s": 30,
}
VERDICT_REASONS = {0: None, 1: "correctness", 2: "regression", 3: "noisy or unstable reference windows",
                   4: "drift between reference windows", 5: "no qualified gain", 6: "invalid measurement"}
FILLER = ("Describe how a local language model server can schedule many chat requests fairly while keeping "
          "the first visible answer quick and the total throughput high. ")


class LearningUnavailable(RuntimeError):
    """The learner cannot run here; the reference schedule serves and the reason is shown."""


class Decision(NamedTuple):
    width: int
    gather_ms: int
    action: int
    source: str


WAITING = 8  # waiting requests a decision state describes (the widest catalogue action)


class _State(C.Structure):
    _fields_ = [("queue", C.c_double), ("arrival_rate", C.c_double), ("prompt_chars", C.c_double),
                ("max_tokens", C.c_double), ("age_cum", C.c_double * WAITING), ("chars_cum", C.c_double * WAITING)]


class _Config(C.Structure):
    _fields_ = [("abi_version", C.c_uint32), ("action_count", C.c_uint32),
                ("widths", C.c_uint32 * MAX_ACTIONS), ("gather_s", C.c_double * MAX_ACTIONS),
                ("max_pending", C.c_uint32), ("min_support", C.c_uint32), ("ring_capacity", C.c_uint32),
                ("horizon", C.c_uint32), ("tau_s", C.c_double), ("alpha_min", C.c_double), ("epsilon", C.c_double),
                ("synthetic_weight", C.c_double), ("ttft_weight", C.c_double), ("failure_penalty_s", C.c_double),
                ("reward_scale_s", C.c_double), ("max_rel_std", C.c_double), ("seed", C.c_uint64)]


class _Transition(C.Structure):
    _fields_ = [("state", _State), ("next", _State), ("action", C.c_uint32), ("served", C.c_uint32),
                ("ok", C.c_uint32), ("bad", C.c_uint32), ("first_known", C.c_uint32), ("duration_s", C.c_double), ("service_s", C.c_double),
                ("ttft_sum_s", C.c_double), ("first_exec_s", C.c_double), ("area_s", C.c_double),
                ("unix_s", C.c_double)]


class _Decision(C.Structure):
    _fields_ = [("action", C.c_uint32), ("explored", C.c_uint32), ("state_bin", C.c_uint32),
                ("supported_mask", C.c_uint32), ("q", C.c_double * MAX_ACTIONS),
                ("predicted_service_s", C.c_double), ("predicted_service_std", C.c_double),
                ("predicted_first_s", C.c_double), ("predicted_first_std", C.c_double),
                ("expected_members", C.c_double), ("known_wait_s", C.c_double),
                ("predicted_ttft_sum_s", C.c_double), ("predicted_ttft_std", C.c_double), ("supported", C.c_uint32)]


class _Dream(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in ("rollouts", "steps", "stopped_unsupported", "stopped_uncertain",
                                                 "stopped_idle", "stopped_horizon", "budget_exhausted")] + [
        ("elapsed_ns", C.c_uint64)]


class _Status(C.Structure):
    _fields_ = [(name, C.c_uint64) for name in (
        "ring_len", "real_updates", "synthetic_updates", "rollouts", "rollout_steps", "stopped_unsupported",
        "stopped_uncertain", "decisions", "explorations", "policy_changes", "drift_resets", "pruned",
        "model_observations")] + [("mean_abs_rel_error", C.c_double * 2), ("within_2sigma", C.c_double),
                                  ("mean_reward", C.c_double), ("last_td_error", C.c_double),
                                  ("state_bytes", C.c_uint64), ("depth_hist", C.c_uint64 * 8)]


class _Verdict(C.Structure):
    _fields_ = [("status", C.c_uint32), ("reason", C.c_uint32), ("mean_ratio", C.c_double),
                ("mean_noise", C.c_double), ("upper_bound", C.c_double)]


def default_library_path() -> Path:
    suffix = ".dylib" if sys.platform == "darwin" else ".dll" if sys.platform == "win32" else ".so"
    name = ("" if sys.platform == "win32" else "lib") + "ironmule_serving_learner" + suffix
    return Path(__file__).resolve().parents[1] / "native" / "serving_learner" / "target" / "release" / name


def _load(path: Path):
    lib = C.PyDLL(str(path))  # short CPU calls; keeping the GIL as the existing controller binding does
    p = C.c_void_p
    signatures = {
        "isl_create": ([C.POINTER(_Config)], p), "isl_free": ([p], None),
        "isl_decide": ([p, C.POINTER(_State), C.c_uint32, C.c_uint32, C.POINTER(_Decision)], C.c_int32),
        "isl_observe": ([p, C.POINTER(_Transition)], C.c_int32),
        "isl_predict": ([p, C.POINTER(_State), C.c_uint32, C.POINTER(_Decision)], C.c_int32),
        "isl_dream": ([p, C.c_uint32, C.c_uint64, C.POINTER(_Dream)], C.c_int32),
        "isl_policy": ([p, C.POINTER(C.c_uint32)], C.c_int32),
        "isl_bin": ([p, C.POINTER(_State)], C.c_uint32),
        "isl_status": ([p, C.POINTER(_Status)], C.c_int32),
        "isl_prune": ([p, C.c_double], C.c_uint32),
        "isl_checkpoint": ([p, C.POINTER(C.c_uint8), C.c_size_t, C.POINTER(C.c_size_t)], C.c_int32),
        "isl_restore": ([C.POINTER(_Config), C.POINTER(C.c_uint8), C.c_size_t], p),
        "isl_judge": ([C.c_uint32, C.c_double, C.c_double, C.c_uint32, C.c_double, C.c_uint32,
                       C.POINTER(C.c_double), C.POINTER(C.c_double), C.POINTER(C.c_double),
                       C.POINTER(C.c_uint8), C.c_uint32, C.POINTER(_Verdict)], C.c_int32),
    }
    for name, (args, result) in signatures.items():
        function = getattr(lib, name)
        function.argtypes, function.restype = args, result
    return lib


def catalogue(capacity: int, reference_width: int, reference_gather_ms: int) -> list[tuple[int, int]]:
    """Action 0 is the reference; the rest are distinct width/gather pairs the backend can run.
    Width one never waits for company. Empty below a capacity of two: nothing to choose."""
    if capacity < 2:
        return []
    out = [(min(reference_width, capacity), reference_gather_ms)]
    for width in CONTRACT["actions"]["widths"]:
        for gather in CONTRACT["actions"]["gather_ms"]:
            pair = (width, gather)
            if width <= capacity and not (width == 1 and gather) and pair not in out:
                out.append(pair)
    return out[:MAX_ACTIONS]


def judge(lib, triplets: list[dict[str, Any]]) -> dict[str, Any]:
    """The optimizer's protocol over measured window costs; predictions never enter here."""
    q = CONTRACT["qualification"]
    n = len(triplets)
    arrays = [(C.c_double * max(n, 1))(*[t[key] for t in triplets]) for key in ("a", "b", "aa")]
    correct = (C.c_uint8 * max(n, 1))(*[1 if t["correct"] else 0 for t in triplets])
    out = _Verdict()
    code = lib.isl_judge(q["protocol"], q["min_gain"], q["max_regression"], q["pairs"], q["family_alpha"],
                         q["max_trials"], *arrays, correct, n, C.byref(out))
    if code != 0:
        raise ValueError("invalid qualification input")
    status = {0: "incomplete", 1: "qualified", 2: "rejected"}[out.status]
    finite = lambda v: v if math.isfinite(v) else None  # noqa: E731
    return {"status": status, "reason": VERDICT_REASONS.get(out.reason), "mean_ratio": finite(out.mean_ratio),
            "mean_noise": finite(out.mean_noise), "upper_bound": finite(out.upper_bound), "triplets": n}


def filler(chars: int) -> str:
    """A fixed synthetic prompt of the recorded length; replay never carries user text."""
    chars = max(1, int(chars))
    return (FILLER * (chars // len(FILLER) + 1))[:chars]


class ModelEvaluation:
    """Forecast quality, scored before each transition is learned (prequential), overall, by
    actual group size and per segment of transitions. Unsupported forecasts are counted apart:
    they are not usable and are not scored."""

    TARGETS = ("service_s", "first_s", "ttft_sum_s")

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self.data = data if isinstance(data, dict) and "overall" in data else {
            "overall": {}, "by_members": {}, "current": {}, "current_n": 0, "segments": []}

    @staticmethod
    def _bucket(members: int) -> str:
        return "1" if members <= 1 else "2" if members == 2 else "3-4" if members <= 4 else "5-8" if members <= 8 else "9+"

    @staticmethod
    def _add(cell: dict, key: str, predicted: float, std: float, actual: float, supported: bool) -> None:
        c = cell.setdefault(key, {"n": 0, "abs": 0.0, "rel": 0.0, "rel_floor": 0.0, "small": 0, "inside": 0,
                                  "unsupported": 0})
        if not supported:
            c["unsupported"] += 1
            return
        error = abs(predicted - actual)
        c["n"] += 1
        c["abs"] += error
        c["rel"] += error / max(abs(actual), 1e-3)
        c["rel_floor"] += error / max(abs(actual), 0.05)
        c["small"] += abs(actual) < 0.05
        c["inside"] += error <= 2.0 * std

    def add(self, forecast: Any, *, service: float, first: float | None, ttft_sum: float, members: int,
            clean: bool) -> None:
        supported = bool(forecast.supported)
        values = [("service_s", forecast.predicted_service_s, forecast.predicted_service_std, service)]
        if first is not None:
            values.append(("first_s", forecast.predicted_first_s, forecast.predicted_first_std, first))
        if clean:  # a failed or rejected request has no first content to compare
            values.append(("ttft_sum_s", forecast.predicted_ttft_sum_s, forecast.predicted_ttft_std, ttft_sum))
        cells = (self.data["overall"], self.data["by_members"].setdefault(self._bucket(members), {}),
                 self.data["current"])
        for key, predicted, std, actual in values:
            for cell in cells:
                self._add(cell, key, predicted, std, actual, supported)
        self.data["current_n"] += 1
        if self.data["current_n"] >= CONTRACT["model"]["segment"]:
            self.data["segments"] = (self.data["segments"] + [self._summarize(self.data["current"])])[-20:]
            self.data["current"], self.data["current_n"] = {}, 0

    @staticmethod
    def _summarize(cell: dict) -> dict[str, Any]:
        out = {}
        for key, c in cell.items():
            n = c["n"]
            out[key] = {"n": n, "unsupported": c["unsupported"],
                        "mean_abs_s": c["abs"] / n if n else None, "mean_rel": c["rel"] / n if n else None,
                        "mean_rel_floor_50ms": c["rel_floor"] / n if n else None,
                        "share_below_50ms": c["small"] / n if n else None,
                        "coverage_2sigma": c["inside"] / n if n else None}
        return out

    def summary(self) -> dict[str, Any]:
        return {"overall": self._summarize(self.data["overall"]),
                "by_members": {k: self._summarize(v) for k, v in sorted(self.data["by_members"].items())},
                "segments": self.data["segments"][-5:], "segments_total": len(self.data["segments"])}

    def criterion(self) -> dict[str, Any]:
        rule = CONTRACT["model"]["criterion"]
        last = self.data["segments"][-1].get("ttft_sum_s") if self.data["segments"] else None
        value = last["mean_rel"] if last and last["n"] else None
        return {**rule, "value": value, "met": None if value is None else value < rule["below"],
                "basis": "latest complete segment, supported forecasts only"}


class LiveLearning:
    """The learner one service owns. All native calls run under one lock; arrival bookkeeping
    has its own lock so admission never waits for a dream slice or a replay window."""

    def __init__(self, root: Path, binding: dict[str, Any], *, capacity: int, reference_width: int,
                 reference_gather_ms: int, join: bool, max_pending: int, request_timeout_s: float | None,
                 mode: str = "active", dreaming: bool = True, library_path: Path | None = None,
                 replay: bool = True,
                 clock_ns: Callable[[], int] = time.monotonic_ns, wall: Callable[[], float] = time.time) -> None:
        if mode not in MODES:
            raise ValueError("learning mode must be active or paused")
        self.mode = mode
        self.dreaming = bool(dreaming)
        self.replay_enabled = bool(replay)  # off: qualification is left to an external harness (P3)
        self.evaluating: str | None = None  # P3: a frozen arm serves, nothing is learned
        self.frozen: dict[str, Any] | None = None
        self.clock_ns, self.wall = clock_ns, wall
        self._lock = threading.RLock()
        self._track = threading.Lock()
        self._live: dict[str, list[int | None]] = {}
        self._rejections: list[int] = []
        self._arrivals: deque[tuple[int, int, int]] = deque(maxlen=CONTRACT["retention"]["arrivals"])
        self._last_arrival: int | None = None
        self._gap_s: float | None = None
        self.max_pending = max_pending
        self.capacity = capacity
        self.actions = catalogue(capacity, reference_width, reference_gather_ms)
        self.reason: str | None = None
        if join:
            self.reason = "joining running groups is on; learning covers group width and gather wait only"
        elif not self.actions:
            self.reason = ("this execution path cannot group requests (capacity 1); the reference schedule "
                           "serves. Grouping needs `--engine tuned`")
        elif binding.get("worker_sources_sha256") is None:
            self.reason = "the model worker did not report its source identity; the reference schedule serves"
        timeout = float(request_timeout_s) if request_timeout_s else 120.0
        self.penalty_s = (1.0 + CONTRACT["reward"]["ttft_weight"]) * timeout + 1.0
        self.binding = {**binding, "schema": SCHEMA, "actions": [list(a) for a in self.actions],
                        "capacity": capacity, "max_pending": max_pending, "penalty_s": self.penalty_s,
                        "contract_sha256": canonical_sha256(CONTRACT),
                        # Every source file the server and its worker actually loaded (RSI3-LIVE (f)):
                        # any change starts fresh, never with old experience or qualifications.
                        "code": {"server_sources_sha256": sources_digest(),
                                 "worker_sources_sha256": binding.get("worker_sources_sha256")}}
        self.root = Path(root) / "live-learning"
        self.lib = None
        self.handle = self.twin = self.bandit = None
        self.binding_sha256 = canonical_sha256(self.binding)
        self.quarantined: str | None = None
        self.restored = False
        self.last_save_unix: float | None = None
        self._dirty = False
        self._last_save_ns = clock_ns()
        self._pending: dict[str, Any] | None = None
        self._save_lock = threading.Lock()
        self._save_job: tuple | None = None
        self._writer: threading.Thread | None = None
        self.save_error: str | None = None
        self._last_activity_ns = clock_ns()
        self._recent_bins: deque[int] = deque(maxlen=512)
        self._cooldown = 0
        self.counters = {"transitions": 0, "dropped": 0, "overridden": 0, "rejections": 0, "explorations": 0,
                         "dream_cpu_ns_hour": 0, "dream_steps_hour": 0, "replay_ns_hour": 0,
                         "hour_start_unix": wall(), "learning_cpu_ns": 0, "aborted_windows": 0, "withdrawals": 0,
                         "replay_windows": 0}
        self.candidate: dict[str, Any] | None = None
        self.active: dict[str, Any] | None = None
        self.compare: dict[str, Any] | None = None
        self.history: list[dict[str, Any]] = []
        self.trials_used = 0
        self._active_failed_windows = 0
        self._active_window: dict[str, Any] | None = None
        self.last_dream: dict[str, Any] | None = None
        self.dream_state = "not run since start"  # set by the first idle call; never a guess
        self._dream_allowance = 0
        self.evaluation = ModelEvaluation()
        self.preemptions: list[dict[str, Any]] = []
        self.replay_suspended: str | None = None
        self.replay_blocked: str | None = None  # set by the service while old replay work is not settled
        self.log: ServingLog | None = None
        if self.reason is not None:
            return
        path = Path(library_path or default_library_path())
        if not path.is_file():
            self.reason = ("the native serving learner is not built; run `cargo build --offline --release "
                           "--manifest-path native/serving_learner/Cargo.toml`")
            return
        try:
            self.binding["library_sha256"] = canonical_sha256(path.read_bytes())
            self.binding_sha256 = canonical_sha256(self.binding)
            self.lib = _load(path)
            self.config = self._config()
            self.bandit_config = self._config(CONTRACT["bandit"]["tau_s"])
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.path = self.root / f"state-{self.binding_sha256[:16]}.json"
            self.log = ServingLog(self.root, max_bytes=CONTRACT["retention"]["experience_log_bytes"],
                                  name=f"experience-{self.binding_sha256[:16]}.jsonl")
            self._restore()
        except (OSError, AttributeError, LearningUnavailable, ValueError) as exc:
            # A broken library or state never stops serving: the reference schedule serves.
            self.handle = self.twin = self.bandit = None
            self.reason = f"the serving learner could not start ({type(exc).__name__}); the reference schedule serves"

    # ---- native state ----

    def _config(self, tau_s: float | None = None) -> _Config:
        cfg = _Config()
        cfg.abi_version = 3
        cfg.action_count = len(self.actions)
        for index, (width, gather) in enumerate(self.actions):
            cfg.widths[index] = width
            cfg.gather_s[index] = gather / 1000.0
        learner, reward = CONTRACT["learner"], CONTRACT["reward"]
        cfg.max_pending = self.max_pending
        cfg.min_support = learner["min_support"]
        cfg.ring_capacity = learner["ring"]
        cfg.horizon = learner["horizon"]
        cfg.tau_s = reward["tau_s"] if tau_s is None else tau_s
        cfg.alpha_min = learner["alpha_min"]
        cfg.epsilon = CONTRACT["exploration"]["share"]
        cfg.synthetic_weight = learner["synthetic_weight"]
        cfg.ttft_weight = reward["ttft_weight"]
        cfg.failure_penalty_s = self.penalty_s
        cfg.reward_scale_s = reward["scale_s"]
        cfg.max_rel_std = learner["max_rel_std"]
        cfg.seed = int(self.binding_sha256[:16], 16)
        return cfg

    def _native(self, encoded: str | None, config: _Config | None = None):
        config = self.config if config is None else config
        if encoded is None:
            handle = self.lib.isl_create(C.byref(config))
        else:
            payload = base64.b64decode(encoded, validate=True)
            buffer = (C.c_uint8 * len(payload)).from_buffer_copy(payload)
            handle = self.lib.isl_restore(C.byref(config), buffer, len(payload))
        if not handle:
            raise ValueError("native learner state is corrupt")
        return handle

    def _restore(self) -> None:
        if self.path.exists():
            try:
                envelope = json.loads(self.path.read_text())
                expected = envelope.pop("sha256")
                if expected != canonical_sha256(envelope):
                    raise ValueError("checkpoint digest mismatch")
                if envelope.get("schema") != SCHEMA or envelope.get("binding") != self.binding:
                    raise ValueError("checkpoint belongs to another hardware, model, plan or learner")
                self.handle = self._native(envelope["payload"])
                self.twin = self._native(envelope["twin_payload"])
                self.bandit = self._native(envelope["bandit_payload"], self.bandit_config)
                self.frozen = envelope.get("frozen")
                self.candidate = envelope.get("candidate")
                if self.candidate is not None and self.candidate.get("state") == "qualifying":
                    # A comparison interrupted by a restart is discarded, never resumed as evidence;
                    # every arrival seen from now on was recorded after the freeze.
                    if self.candidate.get("current"):
                        self.counters["aborted_windows"] += 1
                    self.candidate["current"] = None
                    self.candidate["frozen_ns"] = self.clock_ns()
                self.active = envelope.get("active")
                compare = envelope.get("compare")
                self.compare = compare if compare is None or compare.get("state") == "done" else None
                self.history = list(envelope.get("history", []))[-16:]
                self.trials_used = int(envelope.get("trials_used", 0))
                self.counters.update({k: v for k, v in envelope.get("counters", {}).items() if k in self.counters})
                self.evaluation = ModelEvaluation(envelope.get("evaluation"))
                self.preemptions = list(envelope.get("preemptions", []))
                self.replay_suspended = envelope.get("replay_suspended")
                self._dream_allowance = int(envelope.get("dream_allowance", 0))
                self.restored = True
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                for handle in (self.handle, self.twin, self.bandit):
                    if handle:
                        self.lib.isl_free(handle)
                self.handle = self.twin = self.bandit = None
                quarantine = self.path.with_name(self.path.name + f".rejected-{int(self.wall())}")
                try:
                    self.path.replace(quarantine)
                except OSError:
                    pass
                self.quarantined = str(exc) if isinstance(exc, ValueError) else "unreadable checkpoint"
                self.candidate = self.active = self.compare = None
                self.history, self.trials_used = [], 0
        if not self.handle:
            try:
                self.handle, self.twin = self._native(None), self._native(None)
                self.bandit = self._native(None, self.bandit_config)
            except ValueError as exc:
                raise LearningUnavailable("the serving learner refused its configuration") from exc
        cutoff = self.wall() - CONTRACT["retention"]["days"] * 86400.0
        for handle in (self.handle, self.twin, self.bandit):
            self.lib.isl_prune(handle, cutoff)

    def _checkpoint(self, handle) -> bytes:
        needed = C.c_size_t(0)
        self.lib.isl_checkpoint(handle, None, 0, C.byref(needed))
        buffer = (C.c_uint8 * needed.value)()
        if self.lib.isl_checkpoint(handle, buffer, needed.value, C.byref(needed)) != 0:
            raise LearningUnavailable("checkpoint failed")
        return bytes(buffer)

    def save(self, *, wait: bool = True) -> None:
        """Persist atomically. Under the lock only a consistent snapshot is taken (native state
        bytes, a JSON copy of the rest); encoding, digest, write and fsync run in a writer thread
        unless ``wait`` (shutdown, explicit control), so serving never waits for the disk."""
        with self._lock:
            if self.handle is None:
                return
            main, twin, bandit = (self._checkpoint(self.handle), self._checkpoint(self.twin),
                                  self._checkpoint(self.bandit))
            if len(main) + len(twin) + len(bandit) > CONTRACT["retention"]["checkpoint_bytes"]:
                raise LearningUnavailable("learner state exceeds its storage budget")
            rest = json.loads(json.dumps({
                "candidate": self.candidate, "active": self.active, "compare": self.compare,
                "history": self.history[-16:], "trials_used": self.trials_used, "counters": self.counters,
                "evaluation": self.evaluation.data, "preemptions": self.preemptions[-CONTRACT["preemption"]["samples"]:],
                "replay_suspended": self.replay_suspended, "dream_allowance": self._dream_allowance,
                "frozen": self.frozen, "bandit_payload": base64.b64encode(bandit).decode("ascii")}))
            saved_unix = self.wall()
            self._dirty = False
            self._last_save_ns = self.clock_ns()
        job = (main, twin, rest, saved_unix)
        with self._save_lock:
            self._save_job = job  # a newer snapshot replaces one not yet written
            writer = self._writer
            if writer is None or not writer.is_alive():
                writer = self._writer = threading.Thread(target=self._write_loop, name="ironmule-learning-save",
                                                         daemon=True)
                writer.start()
        if wait:
            writer.join()

    def _write_loop(self) -> None:
        while True:
            with self._save_lock:
                job, self._save_job = self._save_job, None
                if job is None:
                    self._writer = None
                    return
            main, twin, rest, saved_unix = job
            envelope = {"schema": SCHEMA, "binding": self.binding, "saved_unix": saved_unix,
                        "payload": base64.b64encode(main).decode("ascii"),
                        "twin_payload": base64.b64encode(twin).decode("ascii"), **rest}
            envelope["sha256"] = canonical_sha256(envelope)
            tmp = self.path.with_suffix(".tmp")
            try:
                descriptor = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(descriptor, "w") as stream:
                    stream.write(json.dumps(envelope, sort_keys=True))
                    stream.flush()
                    os.fsync(stream.fileno())
                tmp.replace(self.path)
                self.last_save_unix, self.save_error = saved_unix, None
            except OSError:
                self.save_error = "the learning state could not be written"
                self._dirty = True  # retried at the next save point

    def _native_status(self, handle) -> _Status:
        out = _Status()
        self.lib.isl_status(handle, C.byref(out))
        return out

    def _policy(self, handle) -> list[int]:
        out = (C.c_uint32 * STATES)()
        self.lib.isl_policy(handle, out)
        return list(out)

    # ---- request bookkeeping (any thread) ----

    def arrived(self, request_id: str, t_ns: int, request: Any = None) -> None:
        if not self.eligible:
            return  # nothing would ever close these entries
        with self._track:
            self._live[request_id] = [t_ns, None]
            if request is not None:  # metadata only: when, how long a prompt, how many tokens allowed
                self._arrivals.append((t_ns, sum(len(content) for _, content in request.messages),
                                       int(request.max_tokens)))
            if self._last_arrival is not None:
                gap = max((t_ns - self._last_arrival) / 1e9, 1e-3)
                self._gap_s = gap if self._gap_s is None else 0.8 * self._gap_s + 0.2 * gap
            self._last_arrival = t_ns
            self._last_activity_ns = max(self._last_activity_ns, t_ns)

    def rejected(self, t_ns: int) -> None:
        if not self.eligible:
            return
        with self._track:
            self._rejections.append(t_ns)
            self.counters["rejections"] += 1

    def ended(self, request_id: str, t_ns: int) -> None:
        with self._track:
            entry = self._live.get(request_id)
            if entry is not None and entry[1] is None:
                entry[1] = t_ns

    def _rate(self, now_ns: int) -> float:
        with self._track:
            if self._gap_s is None or self._last_arrival is None:
                return 0.0
            idle = max((now_ns - self._last_arrival) / 1e9, 0.0)
            return 1.0 / max(self._gap_s, idle, 1e-3)

    @staticmethod
    def _state(queue: int, waiting: list[tuple[int, Any]], rate: float, now_ns: int) -> _State:
        """Decision-time metadata: queue size, arrival rate, and for the oldest waiting requests
        their ages and prompt sizes (cumulative, so a width k reads the first k). Never text."""
        head = waiting[0][1] if waiting else None
        prompt = sum(len(content) for _, content in head.messages) if head is not None else 0
        state = _State(float(queue), rate, float(prompt), float(head.max_tokens if head is not None else 0))
        age = chars = 0.0
        for k in range(WAITING):
            if k < len(waiting):
                submitted, request = waiting[k]
                age += max(0.0, (now_ns - submitted) / 1e9)
                chars += sum(len(content) for _, content in request.messages)
            state.age_cum[k], state.chars_cum[k] = age, chars
        return state

    # ---- decisions (scheduler thread) ----

    @property
    def eligible(self) -> bool:
        return self.reason is None and self.handle is not None

    def _table_action(self, table: list[int] | None, state: _State) -> int:
        if table is None:
            return 0
        index = self.lib.isl_bin(self.handle, C.byref(state))
        return int(table[index]) if index < STATES and int(table[index]) < len(self.actions) else 0

    def decide(self, queue: int, waiting: list[tuple[int, Any]], explore: bool = True) -> Decision | None:
        """The group to start now, or None when the reference path is not learnable here.
        ``waiting``: (submitted_ns, request) of the oldest waiting requests, head first."""
        if not self.eligible:
            return None
        now = self.clock_ns()
        with self._lock:
            state = self._state(queue, waiting, self._rate(now), now)
            if self.evaluating is not None:  # P3: a frozen arm serves; nothing is learned or explored
                self._pending = None
                arm = self.evaluating
                action = self._arm_action(arm, queue, state)
                width, gather = self.actions[action]
                return Decision(width, gather, action, f"evaluate:{arm}")
            action = self._table_action(self.active["policy"] if self.active else None, state)
            source = "learned" if self.active else "reference"
            guard = CONTRACT["exploration"]["queue_guard"] * self.max_pending
            if explore and self.mode == "active" and self._cooldown == 0 and queue < guard:
                out = _Decision()
                started = time.perf_counter_ns()
                mask = (1 << len(self.actions)) - 1
                if self.lib.isl_decide(self.handle, C.byref(state), mask, 1, C.byref(out)) == 0 and out.explored:
                    action, source = int(out.action), "explore"
                    self.counters["explorations"] += 1
                self.counters["learning_cpu_ns"] += time.perf_counter_ns() - started
            self._pending = ({"t0": now, "state": state, "action": action, "source": source}
                             if self.mode == "active" else None)
            width, gather = self.actions[action]
            return Decision(width, gather, action, source)

    def complete(self, sessions: list[Any], *, refused: bool, gather_waited_ns: int, queue_after: int,
                 next_waiting: list[tuple[int, Any]]) -> None:
        """Close the interval opened by the last decision with what actually happened."""
        if not self.eligible:
            return
        now = self.clock_ns()
        self._last_activity_ns = max(self._last_activity_ns, now)
        with self._lock:
            pending, self._pending = self._pending, None
            if pending is None or self.mode != "active":
                self._forget(now)
                self._watch_active(sessions)
                return
            action = pending["action"]
            if refused and self.actions[action][0] > 1:
                # The backend's gate ran this group alone: credit what ran, never what was proposed.
                self.counters["overridden"] += 1
                single = (1, 0) if gather_waited_ns == 0 else None
                action = self.actions.index(single) if single in self.actions else None
            t0 = pending["t0"]
            rows = [self._row(item, now) for item in sessions]
            ok = [r for r in rows if r["outcome"] == "ok"]
            failed = sum(r["outcome"] == "failed" for r in rows)
            with self._track:
                rejected = sum(1 for t in self._rejections if t <= now)  # since the last closed interval
                area = sum(max(0, min(end if end is not None else now, now) - max(arrive, t0)) / 1e9
                           for arrive, end in self._live.values() if arrive < now)
            self._forget(now)
            bad = len(rows) - len(ok) + rejected
            ttft_sum = sum(r["ttft_s"] or 0.0 for r in ok)
            dispatched = [item.dispatched_ns for item in sessions if item.dispatched_ns is not None]
            ends = [item.end_ns for item in sessions if item.end_ns is not None]
            service = max(0.0, (max(ends) - min(dispatched)) / 1e9) if dispatched and ends else 0.0
            successes = [item for item in sessions if (item.outcome or "failed") == "ok"]
            # RSI3-LIVE (f): known only if every successful member has one; missing is never zero.
            first_known = bool(successes) and bool(dispatched) and all(item.first_ns is not None for item in successes)
            firsts = [(item.first_ns - min(dispatched)) / 1e9 for item in successes] if first_known else []
            first_exec = max(0.0, sum(firsts) / len(firsts)) if firsts else 0.0
            cost = area + CONTRACT["reward"]["ttft_weight"] * ttft_sum + self.penalty_s * bad
            record = {"unix": self.wall(), "source": pending["source"], "proposed": pending["action"],
                      "executed": action, "queue": pending["state"].queue,
                      "arrival_rate": pending["state"].arrival_rate, "next_queue": queue_after,
                      "interval_s": (now - t0) / 1e9, "gather_waited_s": gather_waited_ns / 1e9,
                      "service_s": service, "area_s": area, "ttft_sum_s": ttft_sum, "rejected": rejected,
                      "failed": failed, "cost": cost, "requests": rows}
            if action is None:
                self.counters["dropped"] += 1
            else:
                nxt = self._state(queue_after, next_waiting, self._rate(now), now)
                transition = _Transition(pending["state"], nxt, action, len(rows), len(ok), bad, int(first_known),
                                         (now - t0) / 1e9, service, ttft_sum, first_exec, area, self.wall())
                started = time.perf_counter_ns()
                # Predicted before this transition is learned: an out-of-sample forecast.
                forecast = _Decision()
                if self.lib.isl_predict(self.handle, C.byref(pending["state"]), action, C.byref(forecast)) == 0:
                    self.evaluation.add(forecast, service=service, first=first_exec if firsts else None,
                                        ttft_sum=ttft_sum, members=len(rows), clean=not bad)
                    record["forecast"] = {"service_s": forecast.predicted_service_s,
                                          "first_s": forecast.predicted_first_s,
                                          "known_wait_s": forecast.known_wait_s,
                                          "ttft_sum_s": forecast.predicted_ttft_sum_s,
                                          "ttft_sum_std": forecast.predicted_ttft_std,
                                          "supported": bool(forecast.supported)}
                    surprised = (not forecast.supported or abs(service - forecast.predicted_service_s)
                                 > forecast.predicted_service_std)
                    grant = (0 if not firsts else CONTRACT["dreaming"]["grant_new_information"] if surprised
                             else CONTRACT["dreaming"]["grant_expected"])
                    self._dream_allowance = min(self._dream_allowance + grant, CONTRACT["dreaming"]["allowance_cap"])
                accepted = [self.lib.isl_observe(h, C.byref(transition)) == 0
                            for h in (self.handle, self.twin, self.bandit)]
                self._recent_bins.append(self.lib.isl_bin(self.handle, C.byref(pending["state"])))
                self.counters["learning_cpu_ns"] += time.perf_counter_ns() - started
                if all(accepted):
                    self.counters["transitions"] += 1
                    self._dirty = True
                else:
                    self.counters["dropped"] += 1
            slow = any((r["ttft_s"] or 0.0) > CONTRACT["exploration"]["ttft_slo_s"] for r in ok)
            if bad or slow:
                self._cooldown = CONTRACT["exploration"]["cooldown_decisions"]
            elif self._cooldown:
                self._cooldown -= 1
            if self.log is not None:
                self.log.append(record)
            self._watch_active(sessions)
            if self._dirty and now - self._last_save_ns >= CONTRACT["save_every_s"] * 1e9:
                self.save(wait=False)  # busy servers persist too; the disk work is off this thread

    @staticmethod
    def _row(item: Any, now: int) -> dict[str, Any]:
        end = item.end_ns or now
        first, tokens = item.first_ns, int(getattr(item, "tokens", 0) or 0)
        return {"queue_s": ((item.dispatched_ns or end) - item.submitted_ns) / 1e9,
                "ttft_s": None if first is None else (first - item.submitted_ns) / 1e9,
                "itl_s": None if first is None or tokens < 2 else (end - first) / 1e9 / (tokens - 1),
                "e2e_s": (end - item.submitted_ns) / 1e9, "tokens": tokens, "outcome": item.outcome or "failed"}

    def _forget(self, now: int) -> None:
        with self._track:
            for key in [k for k, (_, end) in self._live.items() if end is not None and end <= now]:
                del self._live[key]
            self._rejections = [t for t in self._rejections if t > now]

    # ---- withdrawal of a qualified policy: ordinary observations may only take it away ----

    def _watch_active(self, sessions: list[Any]) -> None:
        if self.active is None:
            return
        rule = CONTRACT["withdrawal"]
        window = self._active_window = self._active_window or {"failed": 0, "decisions": 0, "requests": 0}
        window["failed"] += sum((item.outcome or "failed") == "failed" for item in sessions)
        window["decisions"] += 1
        window["requests"] += len(sessions)
        if window["decisions"] >= rule["window_decisions"] and window["requests"] >= rule["window_requests"]:
            self._active_failed_windows = self._active_failed_windows + 1 if window["failed"] else 0
            self._active_window = None
            if self._active_failed_windows >= rule["failed_windows_in_a_row"]:
                self._withdraw("failed requests under the learned policy in "
                               f"{self._active_failed_windows} windows in a row")

    def _withdraw(self, reason: str) -> None:
        if self.active is None:
            return
        self.history.append({"version": self.active["version"], "state": "withdrawn", "reason": reason,
                             "decided_unix": self.wall()})
        self.counters["withdrawals"] += 1
        self.active = None
        self._active_window = None
        self._active_failed_windows = 0
        self._dirty = True

    # ---- candidates, replay qualification and comparison (idle, scheduler thread) ----

    def _normalized(self, policy: list[int]) -> list[int]:
        """Map an action to the reference where it behaves the same over the bin's whole queue range."""
        bins = CONTRACT["state_bins"]
        per_queue = bins["rate"] * bins["prompt"]
        ref_width, ref_gather = self.actions[0]
        out = []
        for index, action in enumerate(policy):
            low, high = bins["queue"][index // per_queue]
            width, gather = self.actions[action]
            same = gather == ref_gather and all(min(q, width) == min(q, ref_width) for q in range(low, min(high, 64) + 1))
            out.append(0 if same else action)
        return out

    def _maybe_candidate(self) -> None:
        q = CONTRACT["qualification"]
        if (self.trials_used >= q["max_trials"]
                or (self.candidate is not None and self.candidate.get("state") == "qualifying")):
            return
        native = self._native_status(self.handle)
        since = self.candidate["training"]["real_updates"] if self.candidate else 0
        if native.real_updates < since + q["candidate_every"]:
            return
        policy = self._normalized(self._policy(self.handle))
        known = {tuple(h["policy"]) for h in self.history if "policy" in h}
        if self.active:
            known.add(tuple(self.active["policy"]))
        coverage = (sum(1 for b in self._recent_bins if b < STATES and policy[b]) / len(self._recent_bins)
                    if self._recent_bins else 0.0)
        if coverage < q["min_coverage"] or tuple(policy) in known:
            return  # it would act like the reference on most real decisions: no trial spent on it
        checkpoint = canonical_sha256(self._checkpoint(self.handle))
        self.candidate = {
            "version": canonical_sha256({"policy": policy, "checkpoint_sha256": checkpoint,
                                         "binding_sha256": self.binding_sha256})[:16],
            "policy": policy, "coverage": coverage, "state": "qualifying", "triplets": [], "current": None,
            "verdict": None,
            "created_unix": self.wall(), "frozen_ns": self.clock_ns(), "waiting": None,
            "training": {"real_updates": native.real_updates, "experience": native.ring_len,
                         "checkpoint_sha256": checkpoint},
            "simulator": {"model_observations": native.model_observations, "drift_resets": native.drift_resets,
                          "synthetic_updates": native.synthetic_updates, "rollouts": native.rollouts,
                          "dreaming": self.dreaming},
        }
        self.history.append({"version": self.candidate["version"], "state": "created", "policy": policy,
                             "decided_unix": self.wall()})
        self._dirty = True

    def _traces(self, *, after_ns: int | None = None, before_ns: int | None = None) -> list[list[tuple]]:
        """Consecutive recorded arrivals as replay traces: offsets with gaps capped, prompt sizes and
        output limits capped, never text."""
        q = CONTRACT["qualification"]
        with self._track:
            rows = [r for r in self._arrivals
                    if (after_ns is None or r[0] > after_ns) and (before_ns is None or r[0] < before_ns)]
        size, out = q["trace_requests"], []
        for start in range(0, len(rows) - size + 1, size):
            chunk, offset, trace = rows[start:start + size], 0.0, []
            for index, (t, chars, max_tokens) in enumerate(chunk):
                if index:
                    offset += min((t - chunk[index - 1][0]) / 1e9, q["max_gap_s"])
                trace.append((offset, chars, min(max_tokens, q["max_tokens_cap"])))
            out.append(trace)
        return out

    ARMS_P3 = ("reference", "queue_rule", "bandit", "rl", "rl_dream")

    def _arm_action(self, arm: str, queue: int, state: _State) -> int:
        """The catalogue action a named P3 arm takes: reference, queue_rule, fixed:<index>, or a
        frozen learned table (bandit, rl, rl_dream)."""
        if arm == "reference":
            return 0
        if arm == "queue_rule":
            pair = (1, 0) if queue <= 1 else (min(4, self.capacity), 0)
            return self.actions.index(pair) if pair in self.actions else 0
        if arm.startswith("fixed:"):
            return int(arm.split(":", 1)[1])
        return self._table_action(self.frozen["arms"][arm]["policy"], state)

    def _valid_arm(self, arm: str) -> bool:
        if arm in ("reference", "queue_rule"):
            return True
        if arm.startswith("fixed:"):
            index = arm.split(":", 1)[1]
            return index.isdigit() and int(index) < len(self.actions)
        return self.frozen is not None and arm in self.frozen["arms"]

    def _chooser(self, arm: str | list[int] | int) -> Callable[..., tuple[int, int]]:
        """(queue, waiting [(submitted_ns, request)], arrival rate, now_ns) -> (width, gather_ms)."""
        def choose(queue: int, waiting: list[tuple[int, Any]], rate: float, now_ns: int) -> tuple[int, int]:
            if arm == "reference":
                return self.actions[0]
            if arm == "queue_rule":
                return (1, 0) if queue <= 1 else (min(4, self.capacity), 0)
            if isinstance(arm, int):
                return self.actions[arm]
            with self._lock:
                return self.actions[self._table_action(arm, self._state(queue, waiting, rate, now_ns))]
        return choose

    def _window(self, result: dict[str, Any]) -> dict[str, Any]:
        rows = result["rows"]
        weight = CONTRACT["reward"]["ttft_weight"]
        costs = [r["e2e_s"] + weight * (r["ttft_s"] or 0.0) + (self.penalty_s if r["outcome"] != "ok" else 0.0)
                 for r in rows]
        ttfts = sorted(r["ttft_s"] for r in rows if r["ttft_s"] is not None)
        itls = [r["itl_s"] for r in rows if r["itl_s"] is not None]
        span = max(result["span_s"], 1e-9)
        return {"cost": sum(costs) / max(len(costs), 1), "requests": len(rows),
                "failed": sum(r["outcome"] != "ok" for r in rows),
                "p95_ttft_s": ttfts[min(len(ttfts) - 1, int(round(0.95 * (len(ttfts) - 1))))] if ttfts else None,
                "mean_itl_s": statistics.fmean(itls) if itls else None,
                "tok_s": sum(r["tokens"] for r in rows) / span,
                "outputs_sha256": hashlib.sha256(json.dumps(result["texts"]).encode()).hexdigest()}

    def _replay_budget_left(self) -> bool:
        return (self.counters["replay_ns_hour"] < CONTRACT["qualification"]["replay_s_per_hour"] * 1e9
                and self.replay_suspended is None and self.replay_blocked is None)

    def preempted(self, record: dict[str, Any]) -> None:
        """A real request that arrived during a replay window (RSI3-LIVE (d)): its timeline in
        seconds from arrival. Above the preemption limit at the 95th percentile, replay stops."""
        with self._lock:
            self.preemptions = (self.preemptions + [record])[-CONTRACT["preemption"]["samples"]:]
            waits = sorted(r["extra_wait_s"] for r in self.preemptions)
            p95 = waits[min(len(waits) - 1, math.ceil(0.95 * len(waits)) - 1)]
            limit = CONTRACT["preemption"]["p95_limit_s"]
            if p95 > limit and self.replay_suspended is None:
                self.replay_suspended = (f"real requests waited {p95 * 1000:.0f} ms (p95 of {len(waits)}) for replay "
                                         f"to yield, above {limit * 1000:.0f} ms; replay qualification is suspended")
                self.history.append({"version": "replay", "state": "suspended", "reason": self.replay_suspended,
                                     "decided_unix": self.wall()})
            self._dirty = True

    def preemption_summary(self) -> dict[str, Any]:
        records = self.preemptions
        def p95(values):
            values = sorted(values)
            return values[min(len(values) - 1, math.ceil(0.95 * len(values)) - 1)] if values else None
        out = {"samples": len(records), "p95_extra_wait_s": p95([r["extra_wait_s"] for r in records]),
               "limit_s": CONTRACT["preemption"]["p95_limit_s"], "suspended": self.replay_suspended,
               "blocked": self.replay_blocked, "by_phase": {}}
        for phase in ("prestart", "prefill", "decode", "drain"):
            rows = [r for r in records if r["phase"] == phase]
            if rows:
                out["by_phase"][phase] = {
                    "samples": len(rows), "p95_extra_wait_s": p95([r["extra_wait_s"] for r in rows]),
                    "max_extra_wait_s": max(r["extra_wait_s"] for r in rows),
                    "median_detect_s": statistics.median(r["detect_s"] for r in rows),
                    "median_drain_s": statistics.median(r["drain_s"] for r in rows),
                    "median_handoff_s": statistics.median(r["handoff_s"] for r in rows),
                    "max_schedule_to_dispatch_s": max(r.get("schedule_to_dispatch_s", 0.0) for r in rows),
                    "median_first_after_dispatch_s": statistics.median(
                        r["first_after_dispatch_s"] for r in rows if r["first_after_dispatch_s"] is not None)
                    if any(r["first_after_dispatch_s"] is not None for r in rows) else None}
        return out

    def _run_window(self, replay: Callable, trace: list[tuple], arm: Any, preempted: Callable[[], bool]):
        started = time.monotonic_ns()
        self._lock.release()  # admission and status stay responsive while the backend runs the window
        try:
            result = replay(trace, self._chooser(arm), preempted)
        finally:
            self._lock.acquire()
        self.counters["replay_ns_hour"] += time.monotonic_ns() - started
        if self.handle is None:
            return None  # closed while the window ran: nothing native may be touched any more
        if result is None:
            self.counters["aborted_windows"] += 1
            return None
        self.counters["replay_windows"] += 1
        return self._window(result)

    def _qualify_step(self, replay: Callable, preempted: Callable[[], bool]) -> None:
        candidate = self.candidate
        traces = self._traces(after_ns=candidate["frozen_ns"])
        if not traces:
            candidate["waiting"] = "held-out traffic recorded after the freeze"
            return
        candidate["waiting"] = None
        current = candidate["current"]
        if current is None:
            # The triplet's trace is frozen here: A, B and AA replay exactly the same arrivals.
            order = ["A", "B", "AA"] if len(candidate["triplets"]) % 2 == 0 else ["B", "A", "AA"]
            current = candidate["current"] = {"order": order, "windows": {},
                                              "trace": traces[len(candidate["triplets"]) % len(traces)]}
        arm = current["order"][len(current["windows"])]
        window = self._run_window(replay, [tuple(row) for row in current["trace"]],
                                  candidate["policy"] if arm == "B" else "reference", preempted)
        if self.candidate is not candidate or candidate.get("current") is not current:
            return  # paused or rolled over while the window ran
        if window is None:
            candidate["current"] = None  # an interrupted triplet is discarded
            return
        current["windows"][arm] = window
        self._dirty = True
        if len(current["windows"]) == 3:
            candidate["current"] = None
            self._close_triplet(current["windows"])

    def _close_triplet(self, windows: dict[str, dict[str, Any]]) -> None:
        digests = {w["outputs_sha256"] for w in windows.values()}
        correct = len(digests) == 1 and windows["B"]["failed"] == 0
        self.candidate["triplets"].append({"a": windows["A"]["cost"], "b": windows["B"]["cost"],
                                           "aa": windows["AA"]["cost"], "correct": correct,
                                           "windows": windows})
        verdict = judge(self.lib, self.candidate["triplets"])
        self.candidate["verdict"] = verdict
        if verdict["status"] == "incomplete":
            return
        self.trials_used += 1
        self.candidate["state"] = verdict["status"]
        self.candidate["decided_unix"] = self.wall()
        self.history.append({k: self.candidate[k] for k in ("version", "state", "verdict", "decided_unix")})
        if verdict["status"] == "qualified":
            self.active = {"version": self.candidate["version"], "policy": self.candidate["policy"],
                           "verdict": verdict, "promoted_unix": self.wall()}
            self._active_failed_windows = 0

    def _compare_step(self, replay: Callable, preempted: Callable[[], bool]) -> None:
        job = self.compare
        if job["selection_windows"] is None:
            traces = self._traces(before_ns=job["frozen_ns"])
            if not traces:
                job.update(state="failed", reason="no recorded traffic before the request to select the best fixed arm")
                return
            job["selection_windows"], job["selection_trace"] = {}, traces[-1]
        if len(job["selection_windows"]) < len(self.actions):
            index = len(job["selection_windows"])
            window = self._run_window(replay, [tuple(r) for r in job["selection_trace"]], index, preempted)
            if window is not None:
                job["selection_windows"][str(index)] = window
                if len(job["selection_windows"]) == len(self.actions):
                    job["best_fixed"] = min(range(len(self.actions)),
                                            key=lambda i: job["selection_windows"][str(i)]["cost"])
            return
        needed = CONTRACT["compare"]["holdout_traces"]
        if job.get("holdout") is None:
            traces = self._traces(after_ns=job["frozen_ns"])
            if len(traces) < needed:
                job.update(state="waiting", reason=f"held-out traffic: {len(traces)} of {needed} traces recorded")
                return
            job["holdout"] = traces[:needed]  # frozen: every arm replays the same arrivals
        traces = [[tuple(r) for r in trace] for trace in job["holdout"]]
        job.update(state="running", reason=None)
        k = len(job["results"])
        if k == needed:
            return
        rotation = ARMS[k % len(ARMS):] + ARMS[:k % len(ARMS)]
        done = job.setdefault("current", {})
        arm = rotation[len(done)]
        frozen = {"best_fixed": job["best_fixed"], "rl": job["policies"]["rl"], "rl_dream": job["policies"]["rl_dream"]}
        window = self._run_window(replay, traces[k], frozen.get(arm, arm), preempted)
        if window is None:
            job["current"] = {}  # an interrupted trace is repeated from its first arm
            return
        done[arm] = window
        if len(done) == len(ARMS):
            job["results"].append(done)
            job["current"] = {}
            if len(job["results"]) == needed:
                job.update(state="done", report=self._report(job["results"]), finished_unix=self.wall())
                self.history.append({"version": "compare", "state": "done", "decided_unix": self.wall()})
        self._dirty = True

    def _report(self, results: list[dict[str, dict[str, Any]]]) -> dict[str, Any]:
        gain = CONTRACT["qualification"]["min_gain"]
        out: dict[str, Any] = {"traces": len(results), "arms": {}}
        for arm in ARMS:
            costs = [r[arm]["cost"] for r in results]
            out["arms"][arm] = {
                "mean_cost": statistics.fmean(costs),
                "cost_ratio_vs_reference": statistics.fmean(r[arm]["cost"] / r["reference"]["cost"] for r in results),
                "p95_ttft_s": statistics.fmean(r[arm]["p95_ttft_s"] or 0.0 for r in results),
                "tok_s": statistics.fmean(r[arm]["tok_s"] for r in results),
                "failed": sum(r[arm]["failed"] for r in results),
                "outputs_match_reference": all(r[arm]["outputs_sha256"] == r["reference"]["outputs_sha256"]
                                               for r in results)}
        simple = [min(r["best_fixed"]["cost"], r["queue_rule"]["cost"]) for r in results]
        for arm in ("rl", "rl_dream"):
            out["arms"][arm]["beats_best_simple"] = sum(r[arm]["cost"] < (1 - gain) * s for r, s in zip(results, simple))
        out["dreaming_vs_rl"] = statistics.fmean(r["rl_dream"]["cost"] / r["rl"]["cost"] for r in results)
        out["release"] = (out["arms"]["rl_dream"]["beats_best_simple"] == len(results)
                          and out["arms"]["rl_dream"]["outputs_match_reference"]
                          and out["arms"]["rl_dream"]["failed"] == 0)
        return out

    # ---- idle work (scheduler thread, nothing waiting) ----

    def idle(self, preempted: Callable[[], bool], replay: Callable | None = None) -> None:
        """Dream in bounded slices, freeze candidates, run one replay window, save. Demand preempts."""
        if not self.eligible or self.mode != "active" or self.evaluating is not None:
            return  # an evaluated arm serves alone: no dreams, replay or saves compete with it
        outer = preempted
        preempted = lambda: outer() or self.mode != "active"  # noqa: E731 - a pause also stops idle work
        dreaming = CONTRACT["dreaming"]
        with self._lock:
            now_unix = self.wall()
            if now_unix - self.counters["hour_start_unix"] >= 3600:
                self.counters.update(hour_start_unix=now_unix, dream_cpu_ns_hour=0, dream_steps_hour=0,
                                     replay_ns_hour=0)
            deadline = time.perf_counter_ns() + dreaming["idle_call_ms"] * 1_000_000
            total = {"rollouts": 0, "steps": 0, "stopped_unsupported": 0, "stopped_uncertain": 0}
            state = ("dreaming disabled" if not self.dreaming
                     else "no experience yet" if self._native_status(self.handle).ring_len == 0 else None)
            blocked = state is not None
            while not blocked and time.perf_counter_ns() < deadline:
                if preempted():
                    state = "paused for demand"
                    break
                if (self.counters["dream_cpu_ns_hour"] >= dreaming["cpu_s_per_hour"] * 1e9
                        or self.counters["dream_steps_hour"] >= dreaming["rollout_steps_per_hour"]):
                    state = "hourly budget exhausted"
                    break
                if self._dream_allowance <= 0:
                    state = "waiting for new experience"
                    break
                elapsed = min(1.0, max(0.0, (now_unix - self.counters["hour_start_unix"]) / 3600))
                if (self.counters["dream_steps_hour"]
                        >= dreaming["rollout_steps_per_hour"] * elapsed + dreaming["allowance_cap"]):
                    state = "paced over the hour"
                    break
                out = _Dream()
                code = self.lib.isl_dream(self.handle, min(dreaming["rollouts_per_slice"], self._dream_allowance),
                                          dreaming["slice_ms"] * 1_000_000, C.byref(out))
                self.counters["dream_cpu_ns_hour"] += out.elapsed_ns
                self.counters["dream_steps_hour"] += out.steps  # a rollout stopped before its first step is free
                self._dream_allowance = max(0, self._dream_allowance - out.steps)
                for key in total:
                    total[key] += getattr(out, key)
                if code == 3:
                    state = "no experience yet"
                    break
                if out.steps == 0:
                    state = ("model too uncertain" if out.stopped_uncertain >= out.stopped_unsupported
                             else "no state with enough real support")
                    break
                state = "dreaming"
                self._dirty = True
            if state is not None:
                self.dream_state = state
            if total["rollouts"]:
                self.last_dream = {**total, "unix": now_unix, "state": self.dream_state}
            if preempted():
                return
            if self.replay_enabled:  # without replay nothing could qualify a candidate here
                self._maybe_candidate()
            quiet = (self.clock_ns() - self._last_activity_ns) / 1e9 >= CONTRACT["qualification"]["start_after_idle_s"]
            if replay is not None and self.replay_enabled and quiet and self._replay_budget_left() and not preempted():
                if self.compare is not None and self.compare.get("state") not in ("done", "failed"):
                    self._compare_step(replay, preempted)
                elif self.candidate is not None and self.candidate.get("state") == "qualifying":
                    self._qualify_step(replay, preempted)
            if preempted():
                return  # a real request waits: nothing more on this thread now
            if self._dirty and self.clock_ns() - self._last_save_ns >= CONTRACT["save_every_s"] * 1e9:
                self.save(wait=False)

    # ---- control and status (any thread) ----

    def _freeze(self) -> dict[str, Any]:
        """P3: freeze the bandit, RL and RL-with-dreaming tables with their versions and the
        simulator's use, bound to this identity. Frozen arms never learn."""
        arms = {}
        for name, handle in (("bandit", self.bandit), ("rl", self.twin), ("rl_dream", self.handle)):
            native = self._native_status(handle)
            policy = self._normalized(self._policy(handle))
            checkpoint = canonical_sha256(self._checkpoint(handle))
            arms[name] = {"policy": policy, "checkpoint_sha256": checkpoint,
                          "version": canonical_sha256({"policy": policy, "checkpoint_sha256": checkpoint,
                                                       "binding_sha256": self.binding_sha256})[:16],
                          "real_updates": native.real_updates, "synthetic_updates": native.synthetic_updates,
                          "rollouts": native.rollouts, "rollout_steps": native.rollout_steps,
                          "depth_hist": list(native.depth_hist),
                          "stopped_outside_support": native.stopped_unsupported,
                          "stopped_uncertain": native.stopped_uncertain}
        return {"arms": arms, "frozen_unix": self.wall(), "binding_sha256": self.binding_sha256}

    def control(self, action: str, arm: str | None = None, evidence: dict[str, Any] | None = None) -> dict[str, Any]:
        if action not in ("pause", "resume", "rollback", "compare", "freeze", "evaluate", "activate"):
            raise ValueError("action must be pause, resume, rollback, compare, freeze, evaluate or activate")
        with self._lock:
            if action == "freeze":
                self.frozen = self._freeze()
            elif action == "evaluate":
                if arm is not None and not self._valid_arm(arm):
                    raise ValueError("unknown or unfrozen arm")
                self.evaluating = arm
                self._pending = None
            elif action == "activate":
                # The existing activation path, fed by an external qualification (P3): only a frozen
                # learned arm of this identity, named by the version the evidence qualified.
                if (arm not in ("bandit", "rl", "rl_dream") or self.frozen is None or not isinstance(evidence, dict)
                        or evidence.get("binding_sha256") != self.binding_sha256
                        or evidence.get("version") != self.frozen["arms"][arm]["version"]
                        or evidence.get("verdict", {}).get("status") != "qualified"):
                    raise ValueError("activation needs a qualified verdict for a frozen arm of this identity")
                self.active = {"version": evidence["version"], "policy": self.frozen["arms"][arm]["policy"],
                               "verdict": evidence["verdict"], "arm": arm, "promoted_unix": self.wall(),
                               "evidence_sha256": canonical_sha256(evidence)}
                self._active_failed_windows = 0
                self.history.append({"version": evidence["version"], "state": "activated", "arm": arm,
                                     "decided_unix": self.wall()})
            elif action == "rollback":
                self._withdraw("operator rollback")
            elif action == "compare":
                if self.compare is not None and self.compare.get("state") not in ("done", "failed"):
                    raise ValueError("a comparison is already running")
                self.compare = {"state": "waiting", "reason": "held-out traffic after the request",
                                "requested_unix": self.wall(), "frozen_ns": self.clock_ns(),
                                "policies": {"rl": self._policy(self.twin), "rl_dream": self._policy(self.handle)},
                                "selection_windows": None, "best_fixed": None, "results": [], "current": {},
                                "report": None}
            elif action == "pause":
                self.mode = "paused"
                self._pending = None
                if self.candidate is not None and self.candidate.get("current"):
                    self.candidate["current"] = None  # an interrupted comparison is discarded, not resumed
                    self.counters["aborted_windows"] += 1
            else:
                self.mode = "active"
            self._dirty = True
            if self.handle is not None:
                self.save()
        return self.status()

    def summary(self) -> dict[str, Any]:
        """Lock-free on purpose: readiness checks before admission never wait for a dream slice."""
        candidate, active = self.candidate, self.active
        return {"mode": self.mode if self.eligible else "unavailable",
                "strategy": "learned" if active else "reference",
                "candidate": None if candidate is None else candidate.get("state")}

    def status(self) -> dict[str, Any]:
        with self._lock:
            out: dict[str, Any] = {"schema": SCHEMA, "mode": self.mode if self.eligible else "unavailable",
                                   "eligible": self.eligible, "reason": self.reason,
                                   "actions": [{"width": w, "gather_ms": g} for w, g in self.actions]}
            if self.active:
                out["strategy"] = {"active": "learned", "version": self.active["version"],
                                   "verdict": self.active["verdict"], "reason": None}
            else:
                out["strategy"] = {"active": "reference", "version": None, "reason": self._reference_reason()}
            c = self.candidate
            out["candidate"] = None if c is None else {
                "version": c["version"], "state": c["state"], "verdict": c.get("verdict"), "waiting": c.get("waiting"),
                "created_unix": c["created_unix"], "training": c["training"], "simulator": c["simulator"],
                "triplets": len(c["triplets"]), "triplets_needed": CONTRACT["qualification"]["pairs"]}
            out["compare"] = None if self.compare is None else {
                k: self.compare.get(k) for k in ("state", "reason", "best_fixed", "report", "requested_unix")} | {
                "holdout_traces_done": len(self.compare.get("results", []))}
            out["history"] = [{k: v for k, v in h.items() if k != "policy"} for h in self.history[-8:]]
            if self.handle is None:
                return out
            native, twin = self._native_status(self.handle), self._native_status(self.twin)
            out["learning"] = {"real_transitions": native.ring_len, "real_updates": native.real_updates,
                               "policy_changes": native.policy_changes, "explorations": self.counters["explorations"],
                               "dropped": self.counters["dropped"], "overridden": self.counters["overridden"],
                               "rejections_seen": self.counters["rejections"], "mean_reward": native.mean_reward,
                               "last_td_error": native.last_td_error, "pruned": native.pruned,
                               "learning_cpu_ms": self.counters["learning_cpu_ns"] / 1e6,
                               "twin_without_dreaming": {"real_updates": twin.real_updates,
                                                         "policy_changes": twin.policy_changes}}
            out["model"] = {"observations": native.model_observations,
                            # native running errors of the two model targets (not summed first content,
                            # which `forecasts` and `criterion` report)
                            "mean_abs_rel_error": {"service": native.mean_abs_rel_error[0],
                                                   "dispatch_to_first": native.mean_abs_rel_error[1]},
                            "within_2sigma": native.within_2sigma, "drift_resets": native.drift_resets}
            out["model"]["forecasts"] = self.evaluation.summary()
            out["model"]["criterion"] = self.evaluation.criterion()
            out["replay"] = self.preemption_summary()
            out["dreaming"] = {"enabled": self.dreaming, "state": self.dream_state if self.mode == "active" else
                               "learning paused", "allowance_steps": self._dream_allowance,
                               "synthetic_updates": native.synthetic_updates,
                               "rollouts": native.rollouts, "rollout_steps": native.rollout_steps,
                               "stopped_outside_support": native.stopped_unsupported,
                               "stopped_uncertain": native.stopped_uncertain, "last": self.last_dream}
            d, q = CONTRACT["dreaming"], CONTRACT["qualification"]
            out["budgets"] = {"dream_cpu_s_hour": [self.counters["dream_cpu_ns_hour"] / 1e9, d["cpu_s_per_hour"]],
                              "dream_steps_hour": [self.counters["dream_steps_hour"], d["rollout_steps_per_hour"]],
                              "replay_s_hour": [self.counters["replay_ns_hour"] / 1e9, q["replay_s_per_hour"]],
                              "qualification_trials": [self.trials_used, q["max_trials"]],
                              "state_bytes": [native.state_bytes + twin.state_bytes,
                                              CONTRACT["retention"]["checkpoint_bytes"]],
                              "experience": [native.ring_len, CONTRACT["learner"]["ring"]]}
            bandit = self._native_status(self.bandit)
            out["learning"]["bandit"] = {"real_updates": bandit.real_updates, "policy_changes": bandit.policy_changes}
            out["dreaming"]["depth_hist"] = list(native.depth_hist)
            out["evaluating"] = self.evaluating
            out["replay_enabled"] = self.replay_enabled
            out["frozen"] = None if self.frozen is None else {
                "frozen_unix": self.frozen["frozen_unix"],
                "arms": {k: {kk: vv for kk, vv in v.items() if kk != "policy"} for k, v in self.frozen["arms"].items()}}
            out["persistence"] = {"restored": self.restored, "quarantined": self.quarantined,
                                  "last_save_unix": self.last_save_unix, "save_error": self.save_error, "identity": self.binding_sha256[:16], "binding_sha256": self.binding_sha256,
                                  "binding": {k: self.binding.get(k) for k in (
                                      "model", "device", "plan", "runtime", "capacity", "library_sha256",
                                      "contract_sha256", "code")},
                                  "hour_start_unix": self.counters["hour_start_unix"],
                                  "replay_windows": self.counters["replay_windows"],
                                  "aborted_windows": self.counters["aborted_windows"]}
            return out

    def _reference_reason(self) -> str:
        if self.reason:
            return self.reason
        if self.replay_suspended:
            return self.replay_suspended
        c = self.candidate
        if c is not None and c.get("state") == "qualifying":
            return "a candidate is being qualified against the reference" + (
                f" (waiting for {c['waiting']})" if c.get("waiting") else "")
        if self.history and self.history[-1].get("state") == "withdrawn":
            return "the learned policy was withdrawn: " + self.history[-1].get("reason", "")
        if c is not None and c.get("state") == "rejected":
            reason = (c.get("verdict") or {}).get("reason") or "no qualified gain"
            return f"the last candidate was rejected ({reason})"
        if self.trials_used >= CONTRACT["qualification"]["max_trials"]:
            return "the qualification budget is spent"
        return "no candidate has qualified yet; learning gathers experience"

    def close(self) -> None:
        with self._lock:
            if self.handle is not None:
                try:
                    self.save()
                finally:
                    for handle in (self.handle, self.twin, self.bandit):
                        self.lib.isl_free(handle)
                    self.handle = self.twin = self.bandit = None
