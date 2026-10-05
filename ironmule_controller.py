"""Opt-in CPU policy learning; no model, network, or GPU imports.

The native core owns numerical state. This adapter owns bounded asynchronous
updates, complete service accounting, isolated comparisons and atomic persistence.
See docs/ONLINE_CONTROLLER.md for the deliberately narrow execution contract.
"""

from __future__ import annotations

import base64
import ctypes as C
import json
import math
import os
import queue
import sys
import tempfile
import threading
import time
from collections import deque
from contextlib import contextmanager
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from friday_evidence.canonical import canonical_json_bytes, canonical_sha256

SCHEMA = "ironmule.online_controller.v1"
ACTION_CONTRACT = {
    "schema": "ironmule.execution_profiles.v1",
    "profiles": ["sequential.v1", "grouped4.v1"],
    "plan": "strict_one_shot", "transition": "idle_complete_group",
    "max_width": 4, "precision_change": False,
}
FEATURE_CONTRACT = {
    "schema": "ironmule.workload_cells.v1", "request_count_bins": [1, 2, 3],
    "requested_max_tokens_boundary": 32, "missing": "reference_only",
}
_MAX_CHECKPOINT = 1024 * 1024
_U8_32 = C.c_uint8 * 32


def _action_contract(value: Mapping | None) -> dict:
    """Validate the single registered action space before it reaches the ABI."""
    contract = json.loads(canonical_json_bytes(value if value is not None else ACTION_CONTRACT))
    if contract == ACTION_CONTRACT:
        return contract
    if (not isinstance(contract, dict)
            or set(contract) != set(ACTION_CONTRACT) | {"definitions"}
            or contract.get("schema") not in ("ironmule.execution_profiles.v2", "ironmule.execution_profiles.v3")
            or contract.get("plan") != "strict_one_shot"
            or contract.get("transition") != "idle_complete_group"
            or contract.get("max_width") != 4
            or contract.get("precision_change") is not False):
        raise ValueError("unsupported action contract")
    ids, definitions = contract.get("profiles"), contract.get("definitions")
    if (not isinstance(ids, list) or not 2 <= len(ids) <= 4
            or any(not isinstance(item, str) or not item or len(item) > 100 for item in ids)
            or len(set(ids)) != len(ids)
            or not isinstance(definitions, list) or len(definitions) != len(ids)):
        raise ValueError("action profiles must be two to four distinct definitions")
    reference = None
    unique = set()
    # v3 (CTRL23) may also vary draft-gated speculation; every other knob stays load-bound.
    varying = {"compiled_fixed_cache", "head_skip_prefill"} | (
        {"speculate_k"} if contract["schema"] == "ironmule.execution_profiles.v3" else set())
    for index, definition in enumerate(definitions):
        if (not isinstance(definition, dict)
                or set(definition) != {"profile_id", "mode", "max_width", "knobs"}
                or definition["profile_id"] != ids[index]
                or definition["mode"] not in ("interactive", "throughput")
                or type(definition["max_width"]) is not int
                or definition["max_width"] != (1 if definition["mode"] == "interactive" else 4)
                or not isinstance(definition["knobs"], dict)):
            raise ValueError("invalid complete execution profile")
        knobs = definition["knobs"]
        if any(type(knobs.get(name)) is not bool for name in ("compiled_fixed_cache", "head_skip_prefill")):
            raise ValueError("served cache and head settings must be explicit booleans")
        if index == 0:
            if definition["mode"] != "interactive":
                raise ValueError("action zero must preserve sequential reference execution")
            reference = knobs
        if "speculate_k" in varying and (type(knobs.get("speculate_k")) is not int
                                         or not 0 <= knobs["speculate_k"] <= 8):
            raise ValueError("speculation width must be an integer from 0 to 8")
        if (set(knobs) != set(reference) or any(knobs[name] != reference[name] for name in reference
                if name not in varying)):
            raise ValueError("profile changes a load-bound setting or an unsupported serving knob")
        key = canonical_sha256({name: definition[name] for name in ("mode", "max_width", "knobs")})
        if key in unique:
            raise ValueError("duplicate effective execution profiles")
        unique.add(key)
    return contract


class _Config(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in ("abi_version", "enabled")] + [
        (n, _U8_32) for n in ("identity", "action_digest", "schema_digest", "objective_digest")
    ] + [(n, C.c_uint32) for n in (
        "action_count", "min_train", "freeze_every", "min_pairs", "max_trials",
        "exploration_ppm", "trial_budget",
    )] + [(n, C.c_double) for n in (
        "min_gain", "max_regression", "learning_rate", "family_alpha",
    )] + [("seed", C.c_uint64)]


class _Context(C.Structure):
    _fields_ = [("abi_version", C.c_uint32), ("feature_valid", C.c_uint32),
                ("identity", _U8_32), ("workload_digest", _U8_32)] + [
        (n, C.c_uint32) for n in ("eligible_mask", "request_count", "max_tokens", "phase")
    ] + [("now_ns", C.c_uint64)]


class _Decision(C.Structure):
    _fields_ = [(n, C.c_uint64) for n in ("ticket", "policy_version", "candidate_version")] + [
        (n, C.c_uint32) for n in ("action", "kind", "reason", "cell")
    ] + [("propensity", C.c_double)]


class _Completion(C.Structure):
    _fields_ = [("ticket", C.c_uint64)] + [(n, C.c_uint32) for n in (
        "actual_action", "status", "correctness", "resource_ok",
    )] + [("cost", C.c_double), ("completed_ns", C.c_uint64)]


class _TrainingStatus(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in ("granted", "used", "remaining", "active")]


class _Status(C.Structure):
    _fields_ = [(n, C.c_uint64) for n in (
        "active_version", "fallback_version", "candidate_version", "frozen_after",
        "decisions", "observations", "updates", "dropped", "overridden", "abandoned",
        "promoted", "rejected", "invalid_trials", "last_pair_id",
    )] + [(n, C.c_uint32) for n in (
        "enabled", "killed", "fault_mask", "candidate_cell", "candidate_action",
        "pairs_completed", "min_pairs", "trials_started", "credits", "pending",
        "last_outcome", "next_order",
    )] + [(n, C.c_double) for n in ("mean_ratio", "aa_noise", "upper_bound")] + [
        ("active_actions", C.c_uint32 * 6), ("fallback_actions", C.c_uint32 * 6),
        ("candidate_actions", C.c_uint32 * 6), ("working_counts", C.c_uint64 * 24),
        ("working_means", C.c_double * 24), ("last_update_ns", C.c_uint64),
        ("model_bytes", C.c_uint64), ("state_bytes", C.c_uint64),
        ("runtime_bytes", C.c_uint64),
    ]


# Native protocol ids; each is immutable per handle and bound into checkpoint identity.
PROTOCOLS = {"strict_v1": 1, "adverse_control_v2": 2, "sequential_v3": 3, "sequential_noise_v4": 4}


@dataclass(frozen=True)
class ControllerConfig:
    enabled: bool = True
    frozen: bool = False
    verify_exploration: bool = False
    min_train: int = 8
    freeze_every: int = 32
    min_pairs: int = 64
    max_trials: int = 8
    exploration_ppm: int = 20_000
    trial_budget: int = 6
    min_gain: float = 0.05
    max_regression: float = 0.05
    learning_rate: float = 0.125
    family_alpha: float = 0.05
    seed: int = 1
    queue_capacity: int = 256
    checkpoint_every: int = 64
    checkpoint_interval_s: float = 5.0
    ttft_limit_s: float = 2.0
    latency_limit_s: float = 30.0
    memory_limit_bytes: int | None = None
    qualification_protocol: str = "strict_v1"
    # Inside an explicit offline grant only: collect missing labels deterministically (CTRL13).
    directed_training: bool = False
    # CTRL17: repeat each comparison arm until one lasts about this long (0 = run once).
    comparison_floor_s: float = 0.0
    # CTRL18: "delivered" compares what a buffered caller receives (each arm's group
    # completion) in the relative gates; "per_request" keeps internal per-request times.
    latency_view: str = "per_request"
    # CTRL23: offer a speculative serving profile, and judge strict-plan arms by their
    # terminal offset instead of terminal cache bits (speculation changes low bits).
    speculative_profiles: bool = False
    state_signature: str = "hash"

    def __post_init__(self):
        if (type(self.qualification_protocol) is not str
                or self.qualification_protocol not in PROTOCOLS):
            raise ValueError("unsupported qualification protocol")
        for name, low, high in (
            ("min_train", 1, 1_000_000), ("freeze_every", 1, 1_000_000),
            ("min_pairs", 8, 256), ("max_trials", 1, 64),
            ("exploration_ppm", 0, 50_000), ("trial_budget", 2, 256),
            ("queue_capacity", 1, 256), ("checkpoint_every", 1, 1_000_000),
            ("seed", 1, (1 << 64) - 1),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}]")
        for name, lower, upper in (
            ("min_gain", 0, 1), ("max_regression", -1e-20, 1),
            ("learning_rate", 0, 1), ("family_alpha", 0, 0.1),
            ("ttft_limit_s", 0, 3600), ("latency_limit_s", 0, 86400),
            ("checkpoint_interval_s", 0, 3600),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lower < value <= upper:
                raise ValueError(f"invalid {name}")
        if self.latency_view not in ("per_request", "delivered"):
            raise ValueError("latency_view must be per_request or delivered")
        if self.state_signature not in ("hash", "offset") or type(self.speculative_profiles) is not bool:
            raise ValueError("state_signature must be hash or offset; speculative_profiles a boolean")
        if (isinstance(self.comparison_floor_s, bool) or not isinstance(self.comparison_floor_s, (int, float))
                or not 0 <= self.comparison_floor_s <= 10):
            raise ValueError("invalid comparison_floor_s")
        if self.min_gain >= 1 or self.ttft_limit_s > self.latency_limit_s:
            raise ValueError("invalid gain or latency limits")
        if any(type(getattr(self, name)) is not bool for name in ("enabled", "frozen", "verify_exploration",
                                                                  "directed_training")):
            raise ValueError("enabled, frozen, verify_exploration and directed_training must be booleans")
        if self.memory_limit_bytes is not None and (
            type(self.memory_limit_bytes) is not int or self.memory_limit_bytes <= 0
        ):
            raise ValueError("memory limit must be positive or unavailable")

    def objective(self) -> dict:
        objective = {
            "schema": "ironmule.complete_group_objective.v1",
            "cost": "elapsed_seconds / completed_requests; hard TTFT/latency gates",
            "ttft_limit_s": self.ttft_limit_s, "latency_limit_s": self.latency_limit_s,
            "memory_limit_bytes": self.memory_limit_bytes,
            "min_gain": self.min_gain, "max_regression": self.max_regression,
        }
        if self.latency_view != "per_request":
            objective["relative_gate_view"] = "delivered: each arm's slowest request, as a buffered caller receives it"
        if self.state_signature != "hash":
            objective["state_signature"] = "strict-plan terminal offset instead of terminal cache bits"
        if self.qualification_protocol != "strict_v1":
            version = PROTOCOLS[self.qualification_protocol]
            objective.update(schema=f"ironmule.complete_group_objective.v{version}",
                             qualification=self.qualification_contract())
        return objective

    def qualification_contract(self) -> dict:
        """Explain native scores; this metadata never computes or grants qualification."""
        adverse = self.qualification_protocol != "strict_v1"
        contract = {"protocol": self.qualification_protocol, "fixed_samples": self.min_pairs,
                "unstable_aa_threshold": self.max_regression,
                "ratio_score": ("actual ratio; when raw A/A noise exceeds threshold, 1 + max_regression"
                                if adverse else "actual ratio; unstable A/A terminates the trial"),
                "noise_score": "full raw abs(A - AA) / mean(A, AA); never clipped",
                "unstable_ratio_score": 1 + self.max_regression if adverse else None,
                "sample_accounting": ("all windows count; no filtering or retries" if adverse
                                      else "unstable window terminates trial; no filtering or retry within trial"),
                "hard_gates": "actual ratio, relative TTFT/latency, correctness, resource and caller SLO unchanged",
                "family_alpha": self.family_alpha, "max_trials": self.max_trials,
                "uncertainty_scope": "nominal alpha under independence assumptions; not a mean confidence interval or universal safety"}
        if self.qualification_protocol == "sequential_noise_v4":
            contract.update(
                ratio_score="actual ratio for every window; noise is added to the win criterion and the mean gate",
                unstable_ratio_score=None,
                sample_accounting="all windows count; a noisy control is penalised by its noise, not forced to a loss")
        if self.qualification_protocol in ("sequential_v3", "sequential_noise_v4"):
            contract.update(
                fixed_samples=None, max_samples=self.min_pairs, min_samples=8,
                sequential_test=("mixture e-process mean_l prod(1 + l (win - 1/2)), l in 0.5/1/1.5/1.9, "
                                 "may promote after any window at E >= 2 max_trials / family_alpha; "
                                 "else the exact binomial test at max_samples with half the alpha"),
                early_stop="futility when neither test can still pass; never improves a score")
        return contract


@dataclass(frozen=True)
class ControllerContext:
    identity: Mapping
    request_count: int
    max_tokens: int
    workload: Mapping
    eligible_mask: int = 3


@dataclass(frozen=True)
class ExecutionOutcome:
    value: Any
    signature: Any
    elapsed_s: float
    ttft_s: tuple[float, ...]
    latencies_s: tuple[float, ...]
    generated_tokens: int
    fallback_count: int = 0
    resource_ok: bool = True
    status: str = "ok"
    memory_bytes: int | None = None


def _digest(value: Any) -> bytes:
    return bytes.fromhex(canonical_sha256(value))


def default_library_path() -> Path:
    suffix = ".dylib" if sys.platform == "darwin" else ".dll" if sys.platform == "win32" else ".so"
    filename = ("" if sys.platform == "win32" else "lib") + "ironmule_online_controller" + suffix
    return Path(__file__).resolve().parent / "native" / "online_controller" / "target" / "release" / filename


def _load_library(path: Path):
    # Calls do bounded CPU work without Python callbacks. Retaining the GIL
    # avoids handing a tiny state-lock critical section to another Python thread.
    # This improves learning progress under burst load; it is not a promise of
    # lower full-system latency. The binding comparison is recorded in CTRL1.
    lib = C.PyDLL(str(path))
    signatures = {
        "create": ([C.POINTER(_Config)], C.c_void_p),
        "free": ([C.c_void_p], None),
        "decide": ([C.c_void_p, C.POINTER(_Context), C.POINTER(_Decision)], C.c_int32),
        "predict": ([C.c_void_p, C.POINTER(_Context), C.POINTER(_Decision)], C.c_int32),
        "complete": ([C.c_void_p, C.POINTER(_Completion)], C.c_int32),
        "status": ([C.c_void_p, C.POINTER(_Status)], C.c_int32),
        "fault": ([C.c_void_p, C.c_uint32, C.c_uint32], C.c_int32),
        "begin_pair": ([C.c_void_p, C.POINTER(_Context), C.c_uint64,
                        C.POINTER(_Decision), C.POINTER(_Decision), C.POINTER(_Decision), C.POINTER(C.c_uint32)], C.c_int32),
        "checkpoint": ([C.c_void_p, C.POINTER(C.c_uint8), C.c_size_t, C.POINTER(C.c_size_t)], C.c_int32),
        "restore": ([C.POINTER(_Config), C.POINTER(C.c_uint8), C.c_size_t], C.c_void_p),
    }
    for name, (args, result) in signatures.items():
        function = getattr(lib, "imc_" + name)
        function.argtypes, function.restype = args, result
    optional = {
        "training_begin": ([C.c_void_p, C.c_uint32], C.c_int32),
        "training_end": ([C.c_void_p], C.c_int32),
        "training_status": ([C.c_void_p, C.POINTER(_TrainingStatus)], C.c_int32),
        "begin_training_pair": signatures["begin_pair"],
    }
    if all(hasattr(lib, "imc_" + name) for name in optional):
        for name, (args, result) in optional.items():
            function = getattr(lib, "imc_" + name)
            function.argtypes, function.restype = args, result
    if hasattr(lib, "imc_decide_training"):
        lib.imc_decide_training.argtypes, lib.imc_decide_training.restype = signatures["decide"]
    protocol = {
        "create_with_protocol": ([C.POINTER(_Config), C.c_uint32], C.c_void_p),
        "restore_with_protocol": ([C.POINTER(_Config), C.c_uint32, C.POINTER(C.c_uint8), C.c_size_t], C.c_void_p),
    }
    if all(hasattr(lib, "imc_" + name) for name in protocol):
        for name, (args, result) in protocol.items():
            function = getattr(lib, "imc_" + name)
            function.argtypes, function.restype = args, result
    return lib


class OnlineController:
    """One long-lived Rust instance with a bounded, sleeping learning worker.

    Explicit construction/attachment is the opt-in. A rejected checkpoint stays
    quarantined and uses the reference; it is never silently overwritten.
    """

    def __init__(self, identity: Mapping, *, library_path: Path | None = None,
                 checkpoint_path: Path | None = None, config: ControllerConfig | None = None,
                 start_worker: bool = True, action_contract: Mapping | None = None,
                 hardware_record: Mapping | None = None):
        self.config = config or ControllerConfig()
        self._actions = _action_contract(action_contract)
        self._action_count = len(self._actions["profiles"])
        if self._actions != ACTION_CONTRACT and not self.config.verify_exploration:
            raise ValueError("extended profiles require checked exploration")
        if not isinstance(identity, Mapping) or not identity:
            raise ValueError("a complete backend/model/hardware identity is required")
        self._identity = json.loads(canonical_json_bytes(dict(identity)))
        self._identity_digest = _digest(self._identity)
        self._hardware_record = json.loads(canonical_json_bytes(dict(hardware_record))) if hardware_record is not None else None
        if self._hardware_record is not None:
            binding = self._hardware_record.get("binding")
            if not isinstance(binding, dict) or not binding or self._identity.get("hardware_probe_binding") != binding:
                raise ValueError("hardware observation differs from the bound runtime")
        self.library_path = Path(library_path or default_library_path()).resolve()
        self.library_digest = canonical_sha256(self.library_path.read_bytes())
        self.library_bytes = self.library_path.stat().st_size
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self._lib = _load_library(self.library_path)
        self._protocol_id = PROTOCOLS[self.config.qualification_protocol]
        if self._protocol_id != 1 and not all(hasattr(self._lib, "imc_" + name) for name in (
                "create_with_protocol", "restore_with_protocol")):
            raise RuntimeError(f"native library does not support {self.config.qualification_protocol}")
        self._config = _Config(1, int(self.config.enabled),
            _U8_32.from_buffer_copy(self._identity_digest),
            _U8_32.from_buffer_copy(_digest(self._actions)),
            _U8_32.from_buffer_copy(_digest(FEATURE_CONTRACT)),
            _U8_32.from_buffer_copy(_digest(self.config.objective())),
            self._action_count, self.config.min_train, self.config.freeze_every, self.config.min_pairs,
            self.config.max_trials, self.config.exploration_ppm, self.config.trial_budget,
            self.config.min_gain, self.config.max_regression, self.config.learning_rate,
            self.config.family_alpha, self.config.seed)
        self._lock = threading.Lock()
        self._run_lock = threading.Lock()
        self._save_lock = threading.Lock()
        self._queue = queue.Queue(maxsize=self.config.queue_capacity)
        self._lost: dict[int, _Completion] = {}
        self._lost_lock = threading.Lock()
        self._stop = threading.Event()
        self._closed = False
        self._quarantined = False
        self._latched_kill = False
        self._offline_owner = None
        self._comparison_funding = None
        self._training_api = all(hasattr(self._lib, "imc_" + name) for name in (
            "training_begin", "training_end", "training_status", "begin_training_pair"))
        if self.config.directed_training and not (self._training_api and hasattr(self._lib, "imc_decide_training")):
            raise RuntimeError("native library does not support directed_training")
        self.last_error: str | None = None
        self.last_decision: dict = {}
        self._events = deque(maxlen=128)
        self._service = dict(groups=0, requests=0, completed_requests=0, errors=0,
                             failed_requests=0, timeouts=0, cancelled=0,
                             deadline_violations=0,
                             comparisons=0, comparison_errors=0, mismatches=0,
                             exploration_checks=0, exploration_errors=0,
                             learning_dropped=0, lock_fallbacks=0, overrides=0,
                             wall_ns=0, execution_ns=0, decision_ns=0,
                             learning_ns=0, checkpoint_ns=0, checkpoints=0,
                             generated_tokens=0, pending_groups=0, audit_overwritten=0)
        self._handle = (self._lib.imc_create_with_protocol(C.byref(self._config), self._protocol_id)
                        if self._protocol_id != 1 else self._lib.imc_create(C.byref(self._config)))
        if not self._handle:
            raise ValueError("native controller rejected configuration")
        if self.checkpoint_path is not None and self.checkpoint_path.exists():
            self._restore()
        self._worker = threading.Thread(target=self._learn, name="ironmule-learning", daemon=True)
        self._worker_started = start_worker and not self.config.frozen
        if self._worker_started:
            self._worker.start()

    @property
    def identity(self) -> dict:
        return json.loads(canonical_json_bytes(self._identity))

    @property
    def action_contract(self) -> dict:
        return json.loads(canonical_json_bytes(self._actions))

    def _native_context(self, context: ControllerContext) -> _Context:
        valid = (type(context.request_count) is int and 0 < context.request_count <= 65536
                 and type(context.max_tokens) is int and 0 < context.max_tokens <= 8192
                 and type(context.eligible_mask) is int and context.eligible_mask & 1 == 1
                 and context.eligible_mask >= 1 and context.eligible_mask < 1 << self._action_count)
        identity = _digest(context.identity)
        return _Context(1, 3 if valid else 0, _U8_32.from_buffer_copy(identity),
                        _U8_32.from_buffer_copy(_digest(context.workload)),
                        context.eligible_mask if valid else 1,
                        context.request_count if valid else 0,
                        context.max_tokens if valid else 0, 1, time.monotonic_ns())

    def _status_unlocked(self) -> dict:
        status = _Status()
        if self._lib.imc_status(self._handle, C.byref(status)):
            raise RuntimeError("native status failed")
        return {name: list(value) if isinstance(value := getattr(status, name), C.Array) else value
                for name, _ in status._fields_}

    def snapshot(self) -> dict:
        with self._lock:
            state = self._status_unlocked() if self._handle else {}
            training = self._training_status_unlocked() if self._handle else None
        return {**state, "service": dict(self._service), "config": asdict(self.config),
                "last_decision": dict(self.last_decision), "last_error": self.last_error,
                "checkpoint_rejected": self._quarantined, "binary_bytes": self.library_bytes,
                "learning_queue": self._queue.qsize(), "events": list(self._events),
                "offline_training": training, "qualification_protocol": self.config.qualification_protocol,
                "qualification_score_semantics": self.config.qualification_contract()}

    def _training_status_unlocked(self) -> dict | None:
        if not self._training_api:
            return None
        status = _TrainingStatus()
        if self._lib.imc_training_status(self._handle, C.byref(status)):
            raise RuntimeError("native offline training status failed")
        return {name: getattr(status, name) for name, _ in status._fields_}

    def _end_training_unlocked(self) -> None:
        if self._offline_owner is not None:
            try:
                if self._handle and self._lib.imc_training_end(self._handle):
                    raise RuntimeError("native offline training revocation failed")
            finally:
                self._offline_owner = None

    @contextmanager
    def offline_training(self, *, max_extra_comparison_executions: int):
        """Grant one ephemeral budget to this thread; used means reserved, not completed."""
        budget = max_extra_comparison_executions
        if type(budget) is not int or budget % 2 or not 2 <= budget <= 32768:
            raise ValueError("offline comparison budget must be an even integer in 2..32768")
        with self._run_lock, self._lock:
            if (self._closed or self._quarantined or self._latched_kill or self.config.frozen
                    or not self.config.enabled):
                raise RuntimeError("offline training requires a live, enabled, unquarantined controller")
            if not self._training_api:
                raise RuntimeError("native library does not support explicit offline training")
            if self._offline_owner is not None or self._status_unlocked()["killed"]:
                raise RuntimeError("offline training is already active or the controller is killed")
            if self._lib.imc_training_begin(self._handle, budget):
                raise RuntimeError("native offline training grant rejected")
            self._offline_owner = threading.get_ident()
        try:
            yield self
        finally:
            with self._run_lock, self._lock:
                self._end_training_unlocked()

    def hardware_knowledge(self) -> dict:
        """Describe existing observations; this view cannot qualify an action."""
        state = self.snapshot()
        cells = []
        counts, means = state.get("working_counts", []), state.get("working_means", [])
        for cell in range(6):
            actions = []
            for action, profile in enumerate(self._actions["profiles"]):
                index = cell * 4 + action
                count = counts[index] if index < len(counts) else 0
                actions.append({"action": action, "profile_id": profile,
                                "observations": count,
                                "ew_cost_seconds_per_request": means[index] if count else None})
            selected = state.get("active_actions", [0] * 6)[cell]
            cells.append({"cell": cell, "request_count": ("1", "2", "3+")[cell // 2],
                          "requested_tokens": "at_most_32" if cell % 2 == 0 else "over_32",
                          "active_action": selected, "actions": actions})
        return {"schema": "ironmule.hardware_knowledge.v1", "identity": self.identity,
                "hardware_observation": (json.loads(canonical_json_bytes(self._hardware_record))
                                         if self._hardware_record is not None else None),
                "profiles": self.action_contract, "cells": cells,
                "source": "completed runtime actions; unobserved costs remain unavailable",
                "diagnostics_authorize_activation": False,
                "qualification_protocol": self.config.qualification_protocol,
                "qualification_score_semantics": self.config.qualification_contract(),
                "qualification": {key: state.get(key) for key in (
                    "active_version", "promoted", "rejected", "fault_mask", "killed")}}

    def _completion(self, decision: _Decision, outcome: ExecutionOutcome | None,
                    context: ControllerContext, *, checked: bool = False,
                    matched: bool = True) -> _Completion:
        if not decision.ticket:
            return _Completion()
        if outcome is None:
            return _Completion(decision.ticket, decision.action, 1, 1, 1,
                               self.config.latency_limit_s, time.monotonic_ns())
        n = context.request_count
        valid = self._valid_outcome(outcome, context)
        status = {"ok": 0, "error": 1, "timeout": 2, "cancelled": 3, "interrupted": 4}.get(outcome.status, 1)
        resource_ok = self._within_limits(outcome, context)
        actual_action = decision.action
        if outcome.fallback_count:
            self._service["overrides"] += 1
            self._withdraw_failed_profile(decision)
            # Recovery can mix execution profiles. It has no pure-action label.
            status = 4
        cost = max(outcome.elapsed_s / max(n, 1), 1e-12) if valid else self.config.latency_limit_s
        if status or not resource_ok or not matched:
            cost = max(cost, self.config.latency_limit_s)
        # Only comparative tickets can qualify; ordinary success asserts no
        # detected failure, not independent correctness verification.
        correctness = int(matched if checked else True)
        if (checked and not matched) or not valid:
            status = 4
        return _Completion(decision.ticket, actual_action, status,
                           correctness, int(resource_ok), cost, time.monotonic_ns())

    @staticmethod
    def _valid_outcome(outcome: ExecutionOutcome, context: ControllerContext) -> bool:
        numbers = (outcome.elapsed_s, *outcome.ttft_s, *outcome.latencies_s)
        finite = all(isinstance(x, (int, float)) and not isinstance(x, bool)
                     and math.isfinite(x) and x >= 0 for x in numbers)
        return (finite and outcome.elapsed_s > 0
                and len(outcome.ttft_s) == context.request_count
                and len(outcome.latencies_s) == context.request_count
                and type(outcome.generated_tokens) is int and outcome.generated_tokens >= 0
                and type(outcome.fallback_count) is int and outcome.fallback_count >= 0
                and type(outcome.resource_ok) is bool)

    def _within_limits(self, outcome: ExecutionOutcome, context: ControllerContext) -> bool:
        limits = (self._valid_outcome(outcome, context)
                  and max(outcome.ttft_s, default=math.inf) <= self.config.ttft_limit_s
                  and max(outcome.latencies_s, default=math.inf) <= self.config.latency_limit_s)
        memory_ok = self.config.memory_limit_bytes is None or (
            type(outcome.memory_bytes) is int and 0 <= outcome.memory_bytes <= self.config.memory_limit_bytes)
        return bool(outcome.resource_ok and limits and memory_ok)

    def _withdraw_failed_profile(self, decision: _Decision) -> None:
        if decision.action == 0:
            return
        withdrawn = False
        with self._lock:
            state = self._status_unlocked()
            if (state["active_version"] == decision.policy_version
                    and state["active_actions"][decision.cell] == decision.action):
                self._lib.imc_fault(self._handle, 1, decision.action)
                withdrawn = True
        if withdrawn:
            self._service["profile_withdrawals"] = self._service.get("profile_withdrawals", 0) + 1
            self.save()

    def _record_event(self, event: dict) -> None:
        if len(self._events) == self._events.maxlen:
            self._service["audit_overwritten"] += 1
        self._events.append(event)

    def _enqueue(self, completions: tuple[_Completion, ...]) -> None:
        completions = tuple(item for item in completions if item.ticket)
        if not completions:
            return
        try:
            self._queue.put_nowait(completions)
        except queue.Full:
            self._service["learning_dropped"] += len(completions)
            # At most the core's 256 outstanding tickets can be present here.
            with self._lost_lock:
                for item in completions:
                    self._lost[item.ticket] = _Completion(item.ticket, item.actual_action, 4, 0, 0,
                                                         self.config.latency_limit_s, item.completed_ns)

    def _choose(self, context: _Context):
        self._comparison_funding = None
        if self._latched_kill or self._quarantined or self._closed:
            return _Decision(0, 0, 0, 0, 0, 2, 0, 1.0), None
        if not self._lock.acquire(blocking=False):
            self._service["lock_fallbacks"] += 1
            return _Decision(0, 0, 0, 0, 0, 6, 0, 1.0), None
        try:
            if not self.config.frozen:
                status = self._status_unlocked()
                offline = self._offline_owner == threading.get_ident()
                if status["candidate_version"] and (offline or status["credits"] >= 2):
                    a, aa, b, order = _Decision(), _Decision(), _Decision(), C.c_uint32()
                    operation = self._lib.imc_begin_training_pair if offline else self._lib.imc_begin_pair
                    code = operation(self._handle, C.byref(context), status["last_pair_id"] + 1,
                                                    C.byref(a), C.byref(aa), C.byref(b), C.byref(order))
                    if code == 0:
                        self._comparison_funding = "offline_training" if offline else "live_credits"
                        return a, ((a, aa, b), order.value)
            decision = _Decision()
            operation = self._lib.imc_predict if self.config.frozen else self._lib.imc_decide
            directed = (not self.config.frozen and self.config.directed_training
                        and self._offline_owner == threading.get_ident())
            if directed:
                operation = self._lib.imc_decide_training
            if operation(self._handle, C.byref(context), C.byref(decision)):
                return _Decision(0, 0, 0, 0, 0, 3, 0, 1.0), None
            if directed and decision.kind == 2 and decision.propensity == 1.0:
                # Only a directed label is exactly deterministic; random exploration is not.
                self._comparison_funding = "offline_training_directed_label"
            return decision, None
        finally:
            self._lock.release()

    def run(self, context: ControllerContext,
            execute: Callable[[int], ExecutionOutcome]) -> Any:
        """Execute a real profile, and occasionally an isolated A/A/B comparison.

        The callback must fully complete device work, start from equivalent strict
        state each time, and have no externally delivered side effects. Streaming
        and reusable caches are intentionally outside this initial adapter.
        """
        started = time.perf_counter_ns()
        with self._run_lock:
            if self._closed:
                raise RuntimeError("controller is closed")
            request_count = context.request_count if type(context.request_count) is int and context.request_count >= 0 else 0
            self._service["groups"] += 1
            self._service["requests"] += request_count
            self._service["pending_groups"] += 1
            try:
                decision_start = time.perf_counter_ns()
                native_context = None
                try:
                    native_context = self._native_context(context)
                    decision, pair = self._choose(native_context)
                except (TypeError, ValueError, OverflowError):
                    decision, pair = _Decision(0, 0, 0, 0, 0, 3, 0, 1.0), None
                self._service["decision_ns"] += time.perf_counter_ns() - decision_start
                self.last_decision = {name: getattr(decision, name) for name, _ in decision._fields_}
                self.last_decision.update(eligible_mask=context.eligible_mask,
                                          proposed_action=decision.action,
                                          request_count=context.request_count,
                                          requested_max_tokens=context.max_tokens,
                                          workload_digest=(bytes(native_context.workload_digest).hex()
                                                           if native_context is not None else None),
                                          comparison=pair is not None, correctness="unchecked",
                                          qualification_protocol=self.config.qualification_protocol,
                                          comparison_funding=self._comparison_funding,
                                          started_ns=started)
                if pair is not None:
                    outcome = self._compare(context, execute, pair, started)
                elif self.config.verify_exploration and decision.kind == 2:
                    outcome = self._checked_exploration(context, execute, decision, started)
                else:
                    recovered = False
                    try:
                        outcome = self._execute(execute, decision.action)
                    except Exception as exc:
                        if decision.action == 0:
                            if not self.config.frozen:
                                self._enqueue((self._completion(decision, None, context),))
                            raise
                        self._withdraw_failed_profile(decision)
                        if not self.config.frozen:
                            self._enqueue((self._abandon(decision),))
                        outcome = self._execute(execute, 0)
                        recovered = True
                        self._service["reference_recoveries"] = self._service.get("reference_recoveries", 0) + 1
                        self.last_decision.update(returned_action=0, recovery_reason="optimized_execution_failed",
                                                  execution_error=type(exc).__name__,
                                                  proposal_propensity=decision.propensity, propensity=1.0)
                    except BaseException:
                        if not self.config.frozen:
                            self._enqueue((self._abandon(decision),))
                        raise
                    wall_s = (time.perf_counter_ns() - started) / 1e9
                    outcome = replace(outcome, elapsed_s=max(outcome.elapsed_s, wall_s),
                                      resource_ok=outcome.resource_ok and wall_s <= min(
                                          self.config.ttft_limit_s, self.config.latency_limit_s))
                    if not self.config.frozen and not recovered:
                        self._enqueue((self._completion(decision, outcome, context),))
                if (time.perf_counter_ns() - started) / 1e9 > min(
                    self.config.ttft_limit_s, self.config.latency_limit_s
                ):
                    self._service["deadline_violations"] += request_count
                self.last_decision["actual_action"] = ("mixed_recovery" if outcome.fallback_count else
                                                       self.last_decision.get("returned_action", self.last_decision["action"]))
                self.last_decision["overridden"] = bool(outcome.fallback_count or self.last_decision.get("returned_action", decision.action) != decision.action)
                if outcome.fallback_count:
                    self.last_decision["proposal_propensity"] = self.last_decision["propensity"]
                    self.last_decision["propensity"] = None
                self.last_decision.update(status=outcome.status, completed_ns=time.perf_counter_ns())
                if outcome.status == "ok":
                    self._service["completed_requests"] += request_count
                    self._service["generated_tokens"] += outcome.generated_tokens
                else:
                    self._service["failed_requests"] += request_count
                    if outcome.status in ("timeout", "cancelled"):
                        self._service["timeouts" if outcome.status == "timeout" else "cancelled"] += request_count
                self._record_event(dict(self.last_decision))
                return outcome.value
            except BaseException:
                self._service["errors"] += 1
                self._service["failed_requests"] += request_count
                raise
            finally:
                self._service["pending_groups"] -= 1
                self._service["wall_ns"] += time.perf_counter_ns() - started

    def _execute(self, execute, action) -> ExecutionOutcome:
        started = time.perf_counter_ns()
        try:
            outcome = execute(action)
            if not isinstance(outcome, ExecutionOutcome):
                raise TypeError("executor must return ExecutionOutcome")
            return outcome
        finally:
            self._service["execution_ns"] += time.perf_counter_ns() - started

    @staticmethod
    def _signature(outcome: ExecutionOutcome | None) -> str | None:
        if outcome is None or outcome.signature is None:
            return None
        try:
            return canonical_sha256(outcome.signature)
        except (TypeError, ValueError):
            return None

    def _abandon(self, decision: _Decision) -> _Completion:
        return _Completion(decision.ticket, decision.action, 4, 0, 0,
                           self.config.latency_limit_s, time.monotonic_ns())

    def _quarantine_alternatives(self) -> None:
        with self._lock:
            for action in range(1, self._action_count):
                self._lib.imc_fault(self._handle, 2, action)
        self.save()

    def _checked_exploration(self, context, execute, decision, started_ns) -> ExecutionOutcome:
        """Measure an alternative, but deliver only the exact caller reference.

        The existing exploration credit pays for its one extra execution. Only
        an actually completed, matched alternative receives a training label.
        """
        self._service["exploration_checks"] += 1
        self.last_decision.update(verification=True, returned_action=0,
                                  proposal_propensity=decision.propensity, propensity=1.0,
                                  executed_actions=[])
        try:
            reference = self._execute(execute, 0)
            self.last_decision["executed_actions"].append(0)
        except BaseException:
            self._enqueue((self._abandon(decision),))
            raise
        candidate = None
        try:
            candidate = self._execute(execute, decision.action)
            self.last_decision["executed_actions"].append(decision.action)
        except Exception as exc:
            self._service["exploration_errors"] += 1
            self.last_decision["execution_error"] = type(exc).__name__
        except BaseException:
            self._enqueue((self._abandon(decision),))
            raise
        ref_digest, cand_digest = self._signature(reference), self._signature(candidate)
        checked = (candidate is not None and ref_digest is not None and cand_digest is not None
                   and reference.status == candidate.status == "ok"
                   and reference.fallback_count == candidate.fallback_count == 0
                   and self._valid_outcome(reference, context) and self._valid_outcome(candidate, context))
        matched = checked and ref_digest == cand_digest
        usable = matched and self._within_limits(reference, context) and self._within_limits(candidate, context)
        if checked and not matched:
            self._service["mismatches"] += 1
            self._quarantine_alternatives()
        if candidate is not None and usable:
            if (time.perf_counter_ns() - started_ns) / 1e9 > min(self.config.ttft_limit_s, self.config.latency_limit_s):
                candidate = replace(candidate, resource_ok=False)
            completion = self._completion(decision, candidate, context, checked=True, matched=True)
        else:
            completion = self._abandon(decision)
        self._enqueue((completion,))
        self.last_decision.update(correctness="matched" if matched else "failed_or_unavailable",
                                  observation_action=decision.action if usable else None,
                                  output_digest=ref_digest, alternative_digest=cand_digest)
        return reference

    def _comparison_audit(self, decisions, sequence, raw, final, completions, *,
                          checked=False, matched=False, relative=None, caller_s=None) -> None:
        """Bounded prepared-label diagnostics, independent of async native acceptance."""
        def number(value):
            return value if isinstance(value, (int, float)) and math.isfinite(value) else None

        arms = []
        for index, (decision, completion) in enumerate(zip(decisions, completions)):
            item, gated = raw.get(index), final.get(index)
            arm = {"role": ("A", "AA", "B")[index], "ticket": decision.ticket,
                   "action": decision.action, "elapsed_s": number(item.elapsed_s) if item else None,
                   "status": item.status if item else "error_or_not_executed",
                   "signature_digest": self._signature(item),
                   "signature_matched": bool(matched) if checked else None,
                   "resource_ok_before_gates": item.resource_ok if item else None,
                   "resource_ok_after_gates": gated.resource_ok if gated else None,
                   "resource_ok_for_completion": bool(completion.resource_ok) if completion.ticket else None,
                   "prepared_for_learning": bool(completion.ticket),
                   "prepared_completion": {
                       name: number(getattr(completion, name)) for name, _ in completion._fields_
                   } if completion.ticket else None}
            for name in ("ttft_s", "latencies_s"):
                values = getattr(item, name) if item else None
                arm[name] = [number(value) for value in values[:32]] if values is not None else None
                arm[name + "_count"] = len(values) if values is not None else None
                arm[name + "_truncated"] = len(values) > 32 if values is not None else None
            arms.append(arm)
        self.last_decision.update(
            qualification_protocol=self.config.qualification_protocol,
            comparison_score_semantics=self.config.qualification_contract(),
            comparison_order=[("A", "AA", "B")[index] for index in sequence],
            comparison_arms=arms,
            comparison_relative_ttft_ok=relative[0] if relative is not None else None,
            comparison_relative_latency_ok=relative[1] if relative is not None else None,
            comparison_caller_slo_check_s=caller_s,
            comparison_caller_slo_limit_s=min(self.config.ttft_limit_s, self.config.latency_limit_s),
            comparison_caller_slo_ok=caller_s <= min(self.config.ttft_limit_s, self.config.latency_limit_s)
            if caller_s is not None else None,
        )

    def _repetitions(self, decision, context) -> int:
        """Arm repetitions so one arm lasts about comparison_floor_s, from the learned reference cost."""
        if not self.config.comparison_floor_s:
            return 1
        state = self.snapshot()
        index = decision.cell * 4 + decision.action
        counts, means = state.get("working_counts", []), state.get("working_means", [])
        if index >= len(means) or not counts[index] or means[index] <= 0:
            return 1
        group_s = means[index] * max(context.request_count, 1)
        return max(1, min(8, math.ceil(self.config.comparison_floor_s / group_s)))

    def _execute_repeated(self, execute, action, repetitions) -> ExecutionOutcome:
        """Back-to-back repetitions of one arm: summed time, averaged per-request times.

        Every repetition must reproduce the same output; a disagreement makes the arm's
        signature differ from the others, which the comparison treats as a mismatch.
        """
        runs = [self._execute(execute, action) for _ in range(repetitions)]
        if repetitions == 1:
            return runs[0]
        first, n = runs[0], len(runs)
        signatures = [self._signature(run) for run in runs]
        memory = [run.memory_bytes for run in runs if run.memory_bytes is not None]
        return replace(
            first,
            signature=first.signature if len(set(signatures)) == 1 else {"inconsistent_repetitions": signatures},
            elapsed_s=sum(run.elapsed_s for run in runs),
            ttft_s=tuple(sum(values) / n for values in zip(*(run.ttft_s for run in runs))),
            latencies_s=tuple(sum(values) / n for values in zip(*(run.latencies_s for run in runs))),
            fallback_count=sum(run.fallback_count for run in runs),
            resource_ok=all(run.resource_ok for run in runs),
            status=first.status if all(run.status == first.status for run in runs) else "error",
            memory_bytes=max(memory) if len(memory) == n else None)

    def _compare(self, context, execute, pair, started_ns) -> ExecutionOutcome:
        decisions, order = pair
        repetitions = self._repetitions(decisions[0], context)
        self.last_decision["comparison_repetitions"] = repetitions
        outcomes: dict[int, ExecutionOutcome] = {}
        failure: BaseException | None = None
        failed_action: int | None = None
        sequence = (0, 1, 2) if order == 0 else (2, 1, 0)
        attempted = []
        def abandon():
            completions = tuple(self._abandon(item) for item in decisions)
            self._comparison_audit(decisions, attempted, outcomes, outcomes, completions)
            self._enqueue(completions)

        self._service["comparisons"] += 1
        for index in sequence:
            attempted.append(index)
            try:
                outcomes[index] = self._execute_repeated(execute, decisions[index].action, repetitions)
            except Exception as exc:
                failure = exc
                failed_action = decisions[index].action
                self._service["comparison_errors"] += 1
                # Always obtain the unchanged reference after a candidate failure.
                if index in (0, 1):
                    break
            except BaseException:
                abandon()
                raise
        for index, decision in enumerate(decisions):
            item = outcomes.get(index)
            self._record_event({
                **{name: getattr(decision, name) for name, _ in decision._fields_},
                "comparison": True, "executed": item is not None,
                "status": item.status if item else "error_or_not_executed",
                "cost_s": item.elapsed_s if item else None,
                "output_digest": self._signature(item),
                "workload_digest": canonical_sha256(context.workload),
            })
        reference = outcomes.get(0) or outcomes.get(1)
        if reference is None and failure is not None and failed_action != 0:
            self.fault(failed_action, kind=1)
            try:
                reference = self._execute(execute, 0)
            except BaseException:
                abandon()
                raise
            self.last_decision.update(returned_action=0, recovery_reason="optimized_comparison_failed",
                                      proposal_propensity=self.last_decision["propensity"], propensity=1.0)
            self._service["reference_recoveries"] = self._service.get("reference_recoveries", 0) + 1
        complete = len(outcomes) == 3 and failure is None
        digests = [self._signature(outcomes.get(index)) for index in range(3)]
        checked = complete and all(value is not None for value in digests)
        matched = checked and len(set(digests)) == 1
        if checked and not matched:
            self._service["mismatches"] += 1
            # A mismatch does not identify which optimized arm was at fault.
            # Quarantine every nonreference action and return only action zero.
            self._quarantine_alternatives()
            reference = next((outcomes[i] for i in outcomes if decisions[i].action == 0), None)
            if reference is None:
                # More than two actions can compare two optimized profiles.
                # Neither is a trustworthy fallback after a disagreement.
                try:
                    reference = self._execute(execute, 0)
                except BaseException:
                    abandon()
                    raise
                self._service["reference_recoveries"] = self._service.get("reference_recoveries", 0) + 1
            self.last_decision["action"] = 0
            self.last_decision["returned_action"] = 0
            self.last_decision["proposal_propensity"] = self.last_decision["propensity"]
            self.last_decision["propensity"] = 1.0
        if reference is None:
            completions = tuple(self._completion(d, outcomes.get(i), context, checked=True, matched=False)
                                for i, d in enumerate(decisions))
            self._comparison_audit(decisions, attempted, outcomes, outcomes, completions,
                                   checked=checked, matched=matched)
            self._enqueue(completions)
            if failure is not None:
                raise failure
            raise RuntimeError("comparison has no completed reference")
        raw_outcomes = dict(outcomes)
        relative = None
        if matched:
            # Each held-out comparison also enforces per-request noninferiority. A buffered
            # caller receives every result at group completion, so "delivered" compares that.
            candidate, baseline = outcomes[2], outcomes[0]
            relative = []
            pairs = ((candidate.ttft_s, baseline.ttft_s), (candidate.latencies_s, baseline.latencies_s))
            if self.config.latency_view == "delivered":
                delivered = ((max(candidate.latencies_s, default=math.inf),),
                             (max(baseline.latencies_s, default=0.0),))
                pairs = (delivered, delivered)
            for cand_values, base_values in pairs:
                failed = len(cand_values) != len(base_values) or any(
                    c > b * (1 + self.config.max_regression) for c, b in zip(cand_values, base_values)
                )
                relative.append(not failed)
                if failed:
                    outcomes[2] = replace(candidate, resource_ok=False)
        # This adapter returns a buffered result. The first externally available
        # result follows all comparison work, not an arm's internal first token.
        caller_s = (time.perf_counter_ns() - started_ns) / 1e9
        if caller_s > min(
            self.config.ttft_limit_s, self.config.latency_limit_s
        ):
            outcomes = {index: replace(item, resource_ok=False) for index, item in outcomes.items()}
        completions = tuple(self._completion(d, outcomes.get(i), context,
                                              checked=True, matched=bool(matched))
                            for i, d in enumerate(decisions))
        self._comparison_audit(decisions, attempted, raw_outcomes, outcomes, completions,
                               checked=checked, matched=matched, relative=relative, caller_s=caller_s)
        self._enqueue(completions)
        self.last_decision["correctness"] = "matched" if matched else "failed_or_unavailable"
        return reference

    def _learn(self):
        processed = 0
        next_checkpoint = time.monotonic() + self.config.checkpoint_interval_s
        while not self._stop.is_set() or not self._queue.empty() or self._lost:
            try:
                items = self._queue.get(timeout=0.05)
            except queue.Empty:
                items = ()
            started = time.perf_counter_ns()
            try:
                with self._lost_lock:
                    lost, self._lost = tuple(self._lost.values()), {}
                with self._lock:
                    for completion in (*lost, *items):
                        code = self._lib.imc_complete(self._handle, C.byref(completion))
                        if code not in (0, 2):
                            self.last_error = f"native completion rejected: {code}"
                    processed += len(items)
                if processed >= self.config.checkpoint_every and time.monotonic() >= next_checkpoint:
                    self.save()
                    processed = 0
                    next_checkpoint = time.monotonic() + self.config.checkpoint_interval_s
            except Exception as exc:
                self.last_error = f"learning failure: {type(exc).__name__}"
                self._latched_kill = True
            finally:
                self._service["learning_ns"] += time.perf_counter_ns() - started
                if items:
                    self._queue.task_done()

    def flush(self, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while self._queue.unfinished_tasks or self._lost:
            if not self._worker_started or time.monotonic() >= deadline:
                return False
            self._stop.wait(0.001)
        return True

    def fault(self, action: int, *, kind: int = 2) -> None:
        if type(action) is not int or action not in range(self._action_count) or kind not in (1, 2, 3, 4):
            raise ValueError("unknown action or fault")
        with self._lock:
            self._lib.imc_fault(self._handle, kind, action)
        self.save()

    def kill(self) -> None:
        self._latched_kill = True
        self.fault(0, kind=4)

    def _envelope_binding(self) -> dict:
        config = asdict(self.config)
        # Freezing evaluation changes observation behavior, not the stored model.
        config.pop("frozen")
        if not config["verify_exploration"]:
            config.pop("verify_exploration")
        if config["qualification_protocol"] == "strict_v1":
            config.pop("qualification_protocol")
        if not config["directed_training"]:
            config.pop("directed_training")
        if not config["comparison_floor_s"]:
            config.pop("comparison_floor_s")
        if config["latency_view"] == "per_request":
            config.pop("latency_view")
        if not config["speculative_profiles"]:
            config.pop("speculative_profiles")
        if config["state_signature"] == "hash":
            config.pop("state_signature")
        binding = {"schema": SCHEMA, "identity": self._identity,
                "config": config, "library_sha256": self.library_digest,
                "actions": self._actions, "features": FEATURE_CONTRACT}
        if self._protocol_id != 1:
            binding.update(schema=f"ironmule.online_controller.v{self._protocol_id}", objective=self.config.objective())
        return binding

    def _restore(self) -> None:
        try:
            if self.checkpoint_path.stat().st_size > _MAX_CHECKPOINT:
                raise ValueError("oversized checkpoint")
            envelope = json.loads(self.checkpoint_path.read_bytes())
            binding = self._envelope_binding()
            if not isinstance(envelope, dict) or set(envelope) != set(binding) | {"sha256", "payload"}:
                raise ValueError("checkpoint envelope schema mismatch")
            if any(envelope.get(k) != v for k, v in binding.items()):
                raise ValueError("checkpoint identity/configuration mismatch")
            expected = envelope.pop("sha256")
            if expected != canonical_sha256(envelope):
                raise ValueError("checkpoint digest mismatch")
            payload = base64.b64decode(envelope["payload"], validate=True)
            buffer = (C.c_uint8 * len(payload)).from_buffer_copy(payload)
            handle = (self._lib.imc_restore_with_protocol(C.byref(self._config), self._protocol_id, buffer, len(payload))
                      if self._protocol_id != 1 else self._lib.imc_restore(C.byref(self._config), buffer, len(payload)))
            if not handle:
                raise ValueError("native checkpoint rejected")
            self._lib.imc_free(self._handle)
            self._handle = handle
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self._quarantined = True
            self._latched_kill = True
            self.last_error = f"checkpoint rejected: {type(exc).__name__}"
            self._lib.imc_fault(self._handle, 4, 0)

    def save(self) -> Path | None:
        if self.checkpoint_path is None or self._quarantined or self.config.frozen or self._closed:
            return None
        started = time.perf_counter_ns()
        with self._save_lock:
            with self._lock:
                needed = C.c_size_t()
                code = self._lib.imc_checkpoint(self._handle, None, 0, C.byref(needed))
                if code not in (0, 5) or not 0 < needed.value <= _MAX_CHECKPOINT:
                    raise ValueError("native checkpoint size rejected")
                payload = (C.c_uint8 * needed.value)()
                if self._lib.imc_checkpoint(self._handle, payload, len(payload), C.byref(needed)):
                    raise RuntimeError("native checkpoint failed")
            envelope = self._envelope_binding()
            envelope["payload"] = base64.b64encode(bytes(payload)).decode("ascii")
            envelope["sha256"] = canonical_sha256(envelope)
            content = canonical_json_bytes(envelope)
            if len(content) > _MAX_CHECKPOINT:
                raise ValueError("checkpoint exceeds bounded envelope")
            path = self.checkpoint_path
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd, temporary = tempfile.mkstemp(prefix=".controller-", dir=path.parent)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
                if hasattr(os, "O_DIRECTORY"):
                    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(directory)
                    finally:
                        os.close(directory)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            self._service["checkpoints"] += 1
            self._service["checkpoint_ns"] += time.perf_counter_ns() - started
            return path

    def close(self):
        with self._run_lock:
            if self._closed:
                return
            with self._lock:
                self._end_training_unlocked()
            self.flush()
            self._stop.set()
            if self._worker_started:
                self._worker.join(timeout=5)
                if self._worker.is_alive():
                    self.last_error = "learning worker did not stop; native state retained"
                    raise RuntimeError(self.last_error)
            self.save()
            self._closed = True
            with self._lock:
                self._lib.imc_free(self._handle)
                self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
