#!/usr/bin/env python3
"""Fail-closed parent harness for the Cycle-17 batched-readback study.

The worker's pure Python protocol may be imported for preflight and self-checks;
MLX and the model remain unreachable until every execution gate passes.  The
default and ``--show`` modes do not import the worker.  The child owns
BudgetGuard charging and duty pauses; the parent aggregates that evidence and
enforces the study wall clock.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import random
import signal
import stat
import statistics
import subprocess
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from _bench import require_ac_power, resolve_local_model_snapshot  # noqa: E402
from friday_evidence.budget import BudgetGuard  # noqa: E402
from friday_evidence.registry import BudgetPolicy  # noqa: E402

STUDY_ID = "fixed-compiled-batched-readback-20260824-01"
RUN_ID = "fixed-compiled-batched-readback-validation-20260824-01"
CANDIDATE_ID = "fixed_compiled_batched_readback_n8_v1"
MODEL_KEY = "4b"
MODEL_ID = "mlx-community/gemma-3-4b-it-4bit"
MODEL_REVISION = "93724907d4ed1745d2fe50baadf3b0b01a65abf2"
EXPECTED_SNAPSHOT_SHA256 = "e6edcd46c52b4cf5580f095185a94858565896df7f31c23522294e8f73b3edae"
EXPECTED_WEIGHT_SHA256 = "94d3d701367d78584a9334ca00672b1c86e4aefa6a94167556c0485381e74af3"
EXPECTED_MACHINE = "arm64"
EXPECTED_CPU_BRAND = "Apple M1 Max"
EXPECTED_MEMORY_BYTES = 32 * 1024**3
REQUIRED_PACKAGES = {"mlx": "0.32.0", "mlx-lm": "0.31.3"}
FROZEN_PREREGISTRATION_SHA256 = "74f63c36ddd141c4b4666d9f15d7b17d3ac9294e2d63cb29f6d9e35a80db21b1"
EXPECTED_PROTOCOL_SHA256 = "a58b6298a22e676b9213cc0e4b8fc22ecdc7e0adb25eb07a58f663d268164c30"
PREREGISTRATION = Path(__file__).with_name("PREREGISTRATION.md")
WORKER = Path(__file__).with_name("worker.py")
RESULT_PATH = Path(__file__).with_name("results.json")
ATTEMPT_DIR = PROJECT_ROOT / ".friday-data" / "batched-readback-compile"
ATTEMPT_PATH = ATTEMPT_DIR / "attempt.json"
MAX_EVENT_BYTES = 1_000_000
MAX_STDERR_BYTES = 64_000
WORKER_TIMEOUT_SECONDS = 300.0
FINALIZATION_RESERVE_SECONDS = 15.0
MAX_WALL_SECONDS = 1200.0
MAX_MODEL_WORK_SECONDS = 120.0
CONTINUOUS_MODEL_LIMIT_SECONDS = 6.0
MAX_RSS_BYTES = 5 * 1024**3
MAX_MLX_BYTES = 5 * 1024**3
PAIR_COUNT = 6
ARM_COUNT = 2
OUTPUT_TOKEN_LIMIT = 32
WARMUP_FORWARDS = 8
CAPACITY = 512
EXPECTED_PROMPT_SHA256 = "c746eca8644a18fc75673acb9b3dbdf03825cbfba6c76faede5d909cf3d2ea0b"
EXPECTED_PROMPT_TOKEN_SHA256 = "80ecf700cf0dfdc82616c73f1b6a5fccc137b68e9bb9586ca376c3f2adb260ad"
EXPECTED_RENDERED_PROMPT_SHA256 = "9e18d10b7b101bda3d28593190e622544d474655872aed826c9cbc44211a2cca"
BOOTSTRAP_SEED = 20260824
BOOTSTRAP_RESAMPLES = 10_000
ARM_NAMES = ("fixed_compiled_readback_1", "fixed_compiled_readback_8")
ARM_INTERVALS = {ARM_NAMES[0]: 1, ARM_NAMES[1]: 8}
PAIR_SCHEDULE = tuple(
    (ARM_NAMES[0], ARM_NAMES[1]) if index % 2 == 0 else (ARM_NAMES[1], ARM_NAMES[0])
    for index in range(PAIR_COUNT)
)
WORKER_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1", "PYTHONNOUSERSITE": "1",
}
UNSAFE_ENVIRONMENT = ("PYTHONHOME", "PYTHONINSPECT", "PYTHONPATH", "PYTHONSTARTUP")
POLICY = BudgetPolicy(
    gpu_work_limit_s=120.0, continuous_gpu_limit_s=6.0,
    required_break_s=4.0, duty_window_s=60.0, duty_cycle_limit=0.15,
    wall_limit_s=1200.0, candidate_cooldown_s=0.0,
)


class StudyError(RuntimeError):
    pass


class ResourceError(StudyError):
    pass


class WorkerError(StudyError):
    pass


class WorkerEventError(WorkerError):
    def __init__(self, message: str, event: dict[str, Any]):
        super().__init__(message)
        self.event = event


@lru_cache(maxsize=1)
def _worker_module() -> Any:
    """Load the worker's offline contract; hardware imports remain guarded."""
    if not WORKER.is_file() or WORKER.is_symlink():
        raise StudyError("worker is unavailable")
    spec = importlib.util.spec_from_file_location("cycle17_batched_readback_worker", WORKER)
    if spec is None or spec.loader is None:
        raise StudyError("worker import specification is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, allow_nan=False,
                      separators=(",", ":"), sort_keys=True).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _evidence_state() -> dict[str, Any]:
    state: dict[str, Any] = {}
    for name, path in (("result", RESULT_PATH), ("marker", ATTEMPT_PATH)):
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            state[name] = {"exists": False, "regular": False, "symlink": False,
                           "mode": None, "sha256": None}
            continue
        symlink = stat.S_ISLNK(metadata.st_mode)
        regular = stat.S_ISREG(metadata.st_mode)
        if symlink or not regular:
            raise StudyError(f"{name} evidence path is not a regular file")
        state[name] = {"exists": True, "regular": True, "symlink": False,
                       "mode": stat.S_IMODE(metadata.st_mode), "sha256": _sha256(path)}
    return state


def _validate_evidence_state(state: dict[str, Any]) -> None:
    for name, value in state.items():
        if value["exists"] != value["regular"] or value["symlink"]:
            raise StudyError(f"invalid {name} evidence lifecycle state")
    if state["marker"]["exists"] and state["marker"]["mode"] != 0o600:
        raise StudyError("attempt marker must be mode 0600")
    if state["result"]["exists"] and state["result"]["mode"] != 0o644:
        raise StudyError("result must be mode 0644")


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=PROJECT_ROOT, check=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
    return result.stdout.decode("utf-8", errors="strict").strip()


def _sysctl(name: str) -> str | None:
    try:
        result = subprocess.run(["/usr/sbin/sysctl", "-n", name], check=True,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.decode("utf-8", errors="replace").strip() or None


def _swap_used_bytes() -> int | None:
    try:
        import psutil
        used = psutil.swap_memory().used
    except Exception:
        return None
    return used if type(used) is int and used >= 0 else None


def _private_dir(path: Path) -> None:
    metadata = path.lstat()
    if (not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o700):
        raise StudyError("marker directory is not private")


def _exclusive_json(path: Path, value: dict[str, Any], mode: int) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            os.fchmod(stream.fileno(), mode)
            stream.write(_canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _atomic_result(value: dict[str, Any], *, replace: bool = False) -> None:
    """Create or atomically checkpoint one fail-safe result document."""

    if RESULT_PATH.is_symlink():
        raise StudyError("result path is a symlink")
    if RESULT_PATH.exists():
        metadata = RESULT_PATH.lstat()
        if not replace or not stat.S_ISREG(metadata.st_mode):
            raise StudyError("result already exists or is not regular")
    temp = RESULT_PATH.with_name(f".{RESULT_PATH.name}.tmp-{os.getpid()}")
    if temp.exists() or temp.is_symlink():
        raise StudyError("result temporary path already exists")
    _exclusive_json(temp, value, 0o600)
    os.chmod(temp, 0o644)
    os.replace(temp, RESULT_PATH)


def _snapshot_identity(snapshot: Any) -> dict[str, Any]:
    root = Path(snapshot.path).resolve(strict=True)
    repository = root.parent.parent.resolve(strict=True)
    try:
        root.relative_to(repository)
    except ValueError as exc:
        raise StudyError("snapshot escapes local repository") from exc
    if root.parent.name != "snapshots":
        raise StudyError("unexpected snapshot layout")
    # The sealed snapshot digest is the project-wide resolver-compatible core
    # manifest.  generation_config.json is additionally bound in the execution
    # manifest because it supplies the exact EOS IDs, without silently changing
    # the preregistered snapshot digest definition.
    required = ["config.json", "tokenizer_config.json"]
    for name in ("tokenizer.json", "tokenizer.model"):
        if (root / name).is_file() or (root / name).is_symlink():
            required.append(name)
            break
    required.extend(snapshot.weight_files)
    core_required = list(dict.fromkeys(required))
    required.append("generation_config.json")
    files: dict[str, str] = {}
    manifest: dict[str, dict[str, Any]] = {}
    for relative in dict.fromkeys(required):
        path = (root / relative).resolve(strict=True)
        try:
            path.relative_to(repository)
        except ValueError as exc:
            raise StudyError(f"snapshot path escaped repository: {relative}") from exc
        if not path.is_file():
            raise StudyError(f"snapshot file is not regular: {relative}")
        files[relative] = _sha256(path)
        metadata = path.stat()
        manifest[relative] = {"dev": int(metadata.st_dev), "inode": int(metadata.st_ino),
                              "mtime_ns": int(metadata.st_mtime_ns), "size": int(metadata.st_size),
                              "path": str(path)}
    weights = {name: files.get(name) for name in snapshot.weight_files}
    if any(value is None for value in weights.values()):
        raise StudyError("weight hash missing")
    identity = dict(snapshot.report_identity())
    core_files = {name: files[name] for name in core_required}
    identity.update({"snapshot_path": str(root), "snapshot_files_sha256": core_files,
                     "execution_files_sha256": files,
                     "snapshot_sha256": _sha256_bytes(_canonical(core_files)),
                     "execution_stat_manifest": manifest, "weight_sha256": weights})
    return identity


def _clean_worktree() -> tuple[str, str]:
    revision = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=all", "--", ".",
                  ":(exclude)ProjectAtlas")
    if status:
        raise StudyError("project worktree is dirty")
    return revision, status


def _require_target() -> None:
    if platform.machine() != EXPECTED_MACHINE:
        raise ResourceError("architecture is not arm64")
    if _sysctl("machdep.cpu.brand_string") != EXPECTED_CPU_BRAND:
        raise ResourceError("CPU is not Apple M1 Max")
    if _sysctl("hw.memsize") != str(EXPECTED_MEMORY_BYTES):
        raise ResourceError("memory is not 32 GiB")
    for package, expected in REQUIRED_PACKAGES.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ResourceError(f"package unavailable: {package}") from exc
        if actual != expected:
            raise ResourceError(f"unexpected package version: {package}")
    import mlx.core as mx
    if str(mx.default_device()) != "Device(gpu, 0)":
        raise ResourceError("MLX default device is not GPU")


def _current_code_fingerprints() -> dict[str, str]:
    return {
        path.relative_to(PROJECT_ROOT).as_posix(): _sha256(path)
        for path in (WORKER, Path(__file__), PREREGISTRATION)
    }


def _validate_protocol_contract() -> dict[str, Any]:
    """Bind the offline worker protocol before any attempt marker is written."""

    worker = _worker_module()
    expected = {
        "PROTOCOL_VERSION": 1,
        "PROTOCOL_SHA256": EXPECTED_PROTOCOL_SHA256,
        "AUTH_ENV_PREFIX": "FRIDAY_BRB_",
        "AUTH_NONCE": "cycle17-fixed-compiled-batched-readback-v1",
        "STUDY_ID": STUDY_ID,
        "RUN_ID": RUN_ID,
        "CANDIDATE_ID": CANDIDATE_ID,
        "MODEL_KEY": MODEL_KEY,
        "MODEL_ID": MODEL_ID,
        "MODEL_REVISION": MODEL_REVISION,
        "FROZEN_PREREGISTRATION_SHA256": FROZEN_PREREGISTRATION_SHA256,
    }
    for name, value in expected.items():
        if getattr(worker, name, None) != value:
            raise StudyError(f"worker protocol binding changed: {name}")
    if (
        tuple(worker.ARM_NAMES) != ARM_NAMES
        or dict(worker.ARM_INTERVALS) != ARM_INTERVALS
        or tuple(tuple(order) for order in worker.ARM_ORDERS)
        != tuple(dict.fromkeys(PAIR_SCHEDULE))
        or worker.PROTOCOL_SHA256
        != _sha256_bytes(_canonical(worker.protocol_contract()))
        or worker.PROMPT_SHA256 != EXPECTED_PROMPT_SHA256
        or worker.EXPECTED_PROMPT_TOKEN_SHA256 != EXPECTED_PROMPT_TOKEN_SHA256
        or worker.EXPECTED_RENDERED_PROMPT_SHA256 != EXPECTED_RENDERED_PROMPT_SHA256
        or worker.CACHE_CAPACITY != CAPACITY
        or worker.MAX_PHYSICAL_TOKENS != OUTPUT_TOKEN_LIMIT
        or worker.WARMUP_FORWARDS != WARMUP_FORWARDS
        or dict(worker.OFFLINE_ENVIRONMENT) != WORKER_ENVIRONMENT
        or tuple(worker.UNSAFE_ENVIRONMENT) != UNSAFE_ENVIRONMENT
    ):
        raise StudyError("worker protocol contents changed")
    code_fingerprints = _current_code_fingerprints()
    if worker.code_fingerprints() != code_fingerprints:
        raise StudyError("parent/worker code fingerprints differ")
    environment_sha256 = _environment_fingerprint()
    if worker.environment_fingerprint() != environment_sha256:
        raise StudyError("parent/worker environment fingerprint differs")
    return {
        "protocol_version": worker.PROTOCOL_VERSION,
        "protocol_sha256": worker.PROTOCOL_SHA256,
        "auth_required_env_names": frozenset(worker.AUTH_REQUIRED_ENV_NAMES),
        "event_required_fields": frozenset(worker.EVENT_REQUIRED_FIELDS),
        "arm_required_fields": frozenset(worker.ARM_REQUIRED_FIELDS),
        "boundary_required_fields": frozenset(worker.BOUNDARY_REQUIRED_FIELDS),
        "resource_required_fields": frozenset(worker.RESOURCE_REQUIRED_FIELDS),
        "arm_budget_required_fields": frozenset(worker.ARM_BUDGET_REQUIRED_FIELDS),
        "budget_required_fields": frozenset(worker.BUDGET_REQUIRED_FIELDS),
        "correctness_required_fields": frozenset(worker.CORRECTNESS_REQUIRED_FIELDS),
        "code_fingerprints": code_fingerprints,
        "code_sha256": _sha256_bytes(_canonical(code_fingerprints)),
        "environment_sha256": environment_sha256,
    }


def _preflight(run_id: str) -> tuple[str, str, str, dict[str, Any], int, dict[str, Any]]:
    if run_id != RUN_ID:
        raise StudyError("run ID is not preregistered")
    expected_python = (PROJECT_ROOT / ".venv" / "bin" / "python").resolve(strict=True)
    if Path(sys.executable).resolve() != expected_python:
        raise StudyError("study must use the project virtualenv")
    if _sha256(PREREGISTRATION) != FROZEN_PREREGISTRATION_SHA256:
        raise StudyError("preregistration hash is not sealed")
    if not WORKER.is_file() or WORKER.is_symlink():
        raise StudyError("worker is missing")
    bindings = _validate_protocol_contract()
    if RESULT_PATH.exists() or RESULT_PATH.is_symlink() or ATTEMPT_PATH.exists() or ATTEMPT_PATH.is_symlink():
        raise StudyError("existing result or attempt marker blocks retry")
    revision, dirty = _clean_worktree()
    _require_target()
    try:
        power = require_ac_power()
    except SystemExit as exc:
        raise ResourceError(str(exc)) from exc
    if power != "ac_power":
        raise ResourceError("AC power is required")
    snapshot = resolve_local_model_snapshot(MODEL_ID)
    if snapshot.revision != MODEL_REVISION:
        raise StudyError("local snapshot revision changed")
    identity = _snapshot_identity(snapshot)
    if identity["snapshot_sha256"] != EXPECTED_SNAPSHOT_SHA256:
        raise StudyError("local snapshot hash changed")
    weight_hashes = identity.get("weight_sha256", {})
    if not weight_hashes or any(value != EXPECTED_WEIGHT_SHA256 for value in weight_hashes.values()):
        raise StudyError("local weight hash changed")
    swap = _swap_used_bytes()
    if swap is None:
        raise ResourceError("swap usage unavailable")
    return revision, dirty, power, identity, swap, bindings


def _environment(
    identity: dict[str, Any],
    block: int,
    order: tuple[str, ...],
    prereg_hash: str,
    bindings: dict[str, Any] | None = None,
) -> dict[str, str]:
    if type(block) is not int or not 1 <= block <= PAIR_COUNT or order != PAIR_SCHEDULE[block - 1]:
        raise StudyError("worker block/order binding is invalid")
    bindings = _validate_protocol_contract() if bindings is None else bindings
    if _current_code_fingerprints() != bindings["code_fingerprints"]:
        raise StudyError("study code changed after preflight")
    if _environment_fingerprint() != bindings["environment_sha256"]:
        raise StudyError("study environment changed after preflight")
    environment = os.environ.copy()
    for key in UNSAFE_ENVIRONMENT:
        environment.pop(key, None)
    environment.update(WORKER_ENVIRONMENT)
    values = {
        "PARENT_PID": str(os.getpid()), "RUN_ID": RUN_ID, "MODEL_KEY": MODEL_KEY,
        "NONCE": "cycle17-fixed-compiled-batched-readback-v1", "BLOCK": str(block),
        "ARM_ORDER": json.dumps(list(order), separators=(",", ":")),
        "SNAPSHOT_PATH": str(identity["snapshot_path"]), "SNAPSHOT_REVISION": MODEL_REVISION,
        "SNAPSHOT_SHA256": identity["snapshot_sha256"],
        "WEIGHT_SHA256": _canonical(identity["weight_sha256"]).decode("ascii"),
        "SNAPSHOT_STAT_MANIFEST": _canonical(identity["execution_stat_manifest"]).decode("ascii"),
        "PREREG_SHA256": prereg_hash, "PROMPT_SHA256": EXPECTED_PROMPT_SHA256,
        "PROTOCOL_VERSION": str(bindings["protocol_version"]),
        "PROTOCOL_SHA256": bindings["protocol_sha256"],
        "CODE_FINGERPRINTS": _canonical(bindings["code_fingerprints"]).decode("ascii"),
        "CODE_SHA256": bindings["code_sha256"],
        "ENVIRONMENT_SHA256": bindings["environment_sha256"],
    }
    for key, value in values.items():
        environment[f"FRIDAY_BRB_{key}"] = value
    if not bindings["auth_required_env_names"].issubset(environment):
        raise StudyError("worker authorisation environment is incomplete")
    return environment


def _environment_fingerprint() -> str:
    return _sha256_bytes(_canonical({"fixed": WORKER_ENVIRONMENT, "removed": UNSAFE_ENVIRONMENT,
                                     "python": str(Path(sys.executable).resolve()), "machine": platform.machine()}))


def _remaining(deadline: float | None) -> float | None:
    return None if deadline is None else max(0.0, deadline - time.monotonic())


def _kill(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        try: process.terminate()
        except OSError: pass


def _terminate(process: subprocess.Popen[bytes], deadline: float | None) -> None:
    _kill(process)
    remaining = _remaining(deadline)
    if remaining is not None and remaining <= 0:
        try: os.killpg(process.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            try: process.kill()
            except OSError: pass
        return
    try:
        process.wait(timeout=min(1.0, remaining) if remaining is not None else 1.0)
    except subprocess.TimeoutExpired:
        try: os.killpg(process.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            try: process.kill()
            except OSError: pass
        remaining = _remaining(deadline)
        if remaining is not None and remaining <= 0:
            return
        try: process.wait(timeout=min(1.0, remaining) if remaining is not None else 1.0)
        except subprocess.TimeoutExpired: pass


def _read_capped(stream: Any, process: subprocess.Popen[bytes], limit: int,
                 result: dict[str, Any], key: str, deadline: float | None) -> None:
    data = bytearray()
    try:
        while True:
            chunk = stream.read(min(64 * 1024, limit + 1 - len(data)))
            if not chunk: break
            data.extend(chunk)
            if len(data) > limit:
                result["overflow"] = True
                _terminate(process, deadline)
                break
    except Exception as exc:
        result["read_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        _terminate(process, deadline)
    result[key] = bytes(data)


def _decode_event(payload: bytes) -> dict[str, Any]:
    if not payload or len(payload) > MAX_EVENT_BYTES:
        raise WorkerError("worker output is empty or oversized")
    if b"\r" in payload:
        raise WorkerError("worker output contains a carriage return")
    line = payload[:-1] if payload.endswith(b"\n") else payload
    if not line or b"\n" in line:
        raise WorkerError("worker must emit one JSON line")
    def unique(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result: raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    try:
        value = json.loads(line.decode("utf-8"), object_pairs_hook=unique,
                           parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise WorkerError("worker emitted invalid JSON") from exc
    if not isinstance(value, dict):
        raise WorkerError("worker event is not an object")
    return value


def _budget_schema(value: Any, label: str, *, allow_overrun: bool = False) -> None:
    fields = {"gpu_work_seconds", "max_continuous_gpu_seconds", "cooldown_seconds",
              "required_break_seconds", "wall_seconds", "gpu_work_limit_seconds",
              "continuous_gpu_limit_seconds", "duty_cycle_limit", "wall_limit_seconds",
              "candidate_cooldown_seconds", "required_break_limit_seconds"}
    if not isinstance(value, dict) or set(value) != fields:
        raise WorkerError(f"{label} budget schema invalid")
    if any(not _finite(value[field]) or float(value[field]) < 0 for field in fields):
        raise WorkerError(f"{label} budget contains invalid numbers")
    if (value["gpu_work_limit_seconds"] != 120.0 or value["continuous_gpu_limit_seconds"] != 6.0
            or value["duty_cycle_limit"] != 0.15 or value["wall_limit_seconds"] != 1200.0
            or value["required_break_limit_seconds"] != 4.0):
        raise WorkerError(f"{label} budget limits changed")
    if not allow_overrun and (value["gpu_work_seconds"] > 120.0
            or value["max_continuous_gpu_seconds"] > 6.0 or value["wall_seconds"] > 1200.0):
        raise WorkerError(f"{label} budget limit exceeded")


def _arm_budget_schema(value: Any, label: str, status: str) -> None:
    fields = {"observed_model_work_ns", "charged_model_work_ns", "charge_accepted",
              "guard_gpu_work_before_seconds", "guard_gpu_work_after_seconds",
              "guard_recorded_model_work_ns", "duty_formula_break_seconds",
              "required_break_blocks"}
    if not isinstance(value, dict) or set(value) != fields:
        raise WorkerError(f"{label} arm budget schema invalid")
    observed, charged, guard_recorded = (value[key] for key in
                                          ("observed_model_work_ns", "charged_model_work_ns", "guard_recorded_model_work_ns"))
    if (type(observed) is not int or observed <= 0 or type(charged) is not int or charged < 0
            or type(guard_recorded) is not int or guard_recorded < 0
            or charged > observed or guard_recorded > observed or type(value["charge_accepted"]) is not bool):
        raise WorkerError(f"{label} arm duration invalid")
    if status != "resource_or_budget_failed" and observed > int(6e9):
        raise WorkerError(f"{label} continuous duration exceeded")
    if value["charge_accepted"] and (charged != observed or guard_recorded != observed):
        raise WorkerError(f"{label} accepted charge mismatch")
    if not value["charge_accepted"] and (
        charged != guard_recorded or status != "resource_or_budget_failed"
    ):
        raise WorkerError(f"{label} rejected charge has invalid status")
    before, after = value["guard_gpu_work_before_seconds"], value["guard_gpu_work_after_seconds"]
    if (not _finite(before) or not _finite(after) or before < 0 or after < before
            or abs(after - before - guard_recorded / 1e9) > 1e-6):
        raise WorkerError(f"{label} guard accounting invalid")
    theoretical = observed / 1e9 * (1.0 - 0.15) / 0.15
    if not _finite(value["duty_formula_break_seconds"]) or abs(value["duty_formula_break_seconds"] - theoretical) > 1e-6:
        raise WorkerError(f"{label} duty formula invalid")
    expected_blocks = max(13, math.ceil(theoretical / 4.0))
    if type(value["required_break_blocks"]) is not int or value["required_break_blocks"] != expected_blocks:
        raise WorkerError(f"{label} break projection invalid")


def _hash_ints(values: Any) -> str:
    return _sha256_bytes(_canonical(values))


def _event_percentile(values: list[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] * (upper - position) + ordered[upper] * (position - lower))


def _validate_resource(value: Any, label: str, *, allow_limit_failure: bool) -> None:
    worker = _worker_module()
    if not isinstance(value, dict) or set(value) != set(worker.RESOURCE_REQUIRED_FIELDS):
        raise WorkerError(f"{label} resource schema invalid")
    for field in ("rss_peak_bytes", "mlx_peak_bytes", "swap_before_bytes"):
        if type(value[field]) is not int or value[field] < 0:
            raise WorkerError(f"{label} resource value invalid: {field}")
    after, delta = value["swap_after_bytes"], value["swap_delta_bytes"]
    if (after is None) != (delta is None):
        raise WorkerError(f"{label} swap nullability invalid")
    if after is not None and (
        type(after) is not int
        or type(delta) is not int
        or after < 0
        or delta != after - value["swap_before_bytes"]
    ):
        raise WorkerError(f"{label} swap evidence inconsistent")
    if not allow_limit_failure and (
        value["rss_peak_bytes"] > MAX_RSS_BYTES
        or value["mlx_peak_bytes"] > MAX_MLX_BYTES
        or delta != 0
    ):
        raise WorkerError(f"{label} resource limit exceeded")


def _validate_arm(
    arm: str,
    value: Any,
    budget: dict[str, Any],
    prompt_hashes: dict[str, str] | None = None,
    *,
    resources: dict[str, Any] | None = None,
    bindings: dict[str, Any] | None = None,
) -> None:
    worker = _worker_module()
    bindings = _validate_protocol_contract() if bindings is None else bindings
    if arm not in ARM_NAMES or not isinstance(value, dict):
        raise WorkerError(f"{arm} arm is invalid")
    if set(value) != set(bindings["arm_required_fields"]):
        raise WorkerError(f"{arm} arm field set invalid")
    interval = ARM_INTERVALS[arm]
    if (
        value["arm"] != arm
        or value["readback_interval"] != interval
        or value["max_physical_tokens"] != OUTPUT_TOKEN_LIMIT
        or value["eos_token_ids"] != [1, 106]
        or value["cache_capacity"] != CAPACITY
        or value["fixed_cache"] is not True
        or value["fixed_compile"] is not True
        or value["compile_config"] != dict(worker.COMPILE_CONFIG)
        or value["compile_callable_shared"] is not True
        or value["greedy"] is not True
        or value["sampler_temperature"] != 0.0
    ):
        raise WorkerError(f"{arm} fixed-path identity mismatch")

    physical = value["physical_tokens"]
    if (
        not isinstance(physical, list)
        or not 1 <= len(physical) <= OUTPUT_TOKEN_LIMIT
        or any(type(token) is not int or token < 0 for token in physical)
    ):
        raise WorkerError(f"{arm} physical tokens invalid")
    eos_position = next((i for i, token in enumerate(physical) if token in {1, 106}), None)
    logical = physical if eos_position is None else physical[: eos_position + 1]
    visible = physical if eos_position is None else physical[:eos_position]
    expected = {
        "logical_tokens": logical,
        "visible_tokens": visible,
        "physical_token_count": len(physical),
        "logical_token_count": len(logical),
        "visible_token_count": len(visible),
        "overproduced_tokens": len(physical) - len(logical),
        "eos_found": eos_position is not None,
        "eos_position": eos_position,
        "eos_token_id": None if eos_position is None else physical[eos_position],
        "finish_reason": "length" if eos_position is None else "stop",
    }
    if any(value[field] != expected_value for field, expected_value in expected.items()):
        raise WorkerError(f"{arm} EOS/logical token contract invalid")
    for field, tokens in (
        ("physical_token_sha256", physical),
        ("logical_token_sha256", logical),
        ("visible_token_sha256", visible),
    ):
        if value[field] != _hash_ints(tokens):
            raise WorkerError(f"{arm} token hash invalid: {field}")
    text = value["visible_text"]
    text_hash = _sha256_bytes(text.encode("utf-8")) if isinstance(text, str) else None
    if (
        text_hash is None
        or value["text_sha256"] != text_hash
        or value["text_utf8_sha256"] != text_hash
        or value["token_sha256"] != value["logical_token_sha256"]
    ):
        raise WorkerError(f"{arm} visible text/hash invalid")
    expected_prompt = {
        "prompt_sha256": EXPECTED_PROMPT_SHA256,
        "prompt_token_sha256": EXPECTED_PROMPT_TOKEN_SHA256,
        "rendered_prompt_sha256": EXPECTED_RENDERED_PROMPT_SHA256,
    }
    if prompt_hashes:
        expected_prompt.update(prompt_hashes)
    if any(value[field] != expected_prompt[field] for field in expected_prompt):
        raise WorkerError(f"{arm} prompt identity mismatch")

    sizes = value["readback_block_sizes"]
    records = value["readback_records"]
    count = len(sizes) if isinstance(sizes, list) else -1
    if (
        not isinstance(sizes, list)
        or not isinstance(records, list)
        or count < 1
        or value["readback_count"] != count
        or count != math.ceil(len(physical) / interval)
        or any(type(size) is not int or not 1 <= size <= interval for size in sizes)
        or sum(sizes) != len(physical)
        or value["readback_boundaries"] != records
        or len(records) != count
    ):
        raise WorkerError(f"{arm} readback blocks invalid")
    cursor = 0
    boundary_times: list[int] = []
    readback_ns: list[int] = []
    block_latency_ns: list[int] = []
    expected_eos_block: int | None = None
    for index, (size, record) in enumerate(zip(sizes, records)):
        if not isinstance(record, dict) or set(record) != set(bindings["boundary_required_fields"]):
            raise WorkerError(f"{arm} boundary schema invalid")
        block_tokens = physical[cursor : cursor + size]
        eos_offset = next((i for i, token in enumerate(block_tokens) if token in {1, 106}), None)
        if eos_offset is not None and expected_eos_block is None:
            expected_eos_block = index
        if (
            record["boundary_index"] != index
            or record["physical_start_index"] != cursor
            or record["physical_end_index"] != cursor + size - 1
            or record["block_size"] != size
            or record["eos_offset_in_block"] != eos_offset
            or record["readback_measurement_scope"]
            != "pending_device_eval_plus_sync_plus_single_vector_host_conversion"
            or record["host_transfer_api_calls"] != 1
            or record["host_transfer_method"] != "single_vector_tolist"
            or record["host_transfer_physical_dma_count"] is not None
            or record["vector_block_readback_supported"] is not True
        ):
            raise WorkerError(f"{arm} boundary identity invalid")
        for field in ("block_latency_ns", "host_available_ns", "readback_ns"):
            if type(record[field]) is not int or record[field] <= 0:
                raise WorkerError(f"{arm} boundary timing invalid: {field}")
        boundary_times.append(record["host_available_ns"])
        readback_ns.append(record["readback_ns"])
        block_latency_ns.append(record["block_latency_ns"])
        cursor += size
    if any(later <= earlier for earlier, later in zip(boundary_times, boundary_times[1:])):
        raise WorkerError(f"{arm} boundaries are not strictly monotone")
    host_by_token = value["host_available_ns_by_physical_token"]
    expanded = [time_ns for time_ns, size in zip(boundary_times, sizes) for _ in range(size)]
    gaps = [later - earlier for earlier, later in zip(boundary_times, boundary_times[1:])]
    if (
        host_by_token != expanded
        or value["host_boundary_available_ns"] != boundary_times
        or value["host_available_total_ns"] != boundary_times[-1]
        or value["first_host_token_ns"] != boundary_times[0]
        or value["readback_ns"] != readback_ns
        or value["readback_total_ns"] != sum(readback_ns)
        or value["block_latency_ns"] != block_latency_ns
        or value["boundary_interarrival_ns"] != gaps
        or value["boundary_interarrival_p50_ns"] != _event_percentile(gaps, 0.50)
        or value["boundary_interarrival_p95_ns"] != _event_percentile(gaps, 0.95)
        or value["boundary_interarrival_p99_ns"] != _event_percentile(gaps, 0.99)
        or value["readback_measurement_scope"]
        != "pending_device_eval_plus_sync_plus_single_vector_host_conversion"
        or value["host_transfer_api_call_count"] != count
        or value["host_transfer_physical_dma_count"] is not None
        or value["host_transfer_method"] != "single_vector_tolist"
        or value["host_boundary_available"] is not True
        or value["vector_block_readback_supported"] is not True
    ):
        raise WorkerError(f"{arm} host/readback evidence invalid")
    if (
        value["eos_block"] != expected_eos_block
        or value["eos_readback_block"] != expected_eos_block
    ):
        raise WorkerError(f"{arm} EOS block invalid")

    forwards = len(physical) - 1
    if (
        value["physical_forwards"] != forwards
        or value["decode_forwards"] != forwards
        or not isinstance(value["forward_submit_ns"], list)
        or len(value["forward_submit_ns"]) != forwards
        or any(type(number) is not int or number <= 0 for number in value["forward_submit_ns"])
        or value["cache_discarded"] is not True
    ):
        raise WorkerError(f"{arm} forward/cache evidence invalid")
    positive_ns = (
        "decode_critical_path_ns", "prefill_ns", "ttft_ns", "arm_wall_ns",
        "observed_model_work_ns", "charged_model_work_ns", "warmup_total_ns",
        "first_warmup_materialization_ns", "warmup_prefill_ns",
        "warmup_cache_conversion_ns", "compile_wrapper_ns",
    )
    if any(type(value[field]) is not int or value[field] <= 0 for field in positive_ns):
        raise WorkerError(f"{arm} positive timing evidence invalid")
    if type(value["cache_conversion_ns"]) is not int or value["cache_conversion_ns"] <= 0:
        raise WorkerError(f"{arm} cache conversion timing invalid")
    if value["compile_cold_ns"] is not None and (
        type(value["compile_cold_ns"]) is not int or value["compile_cold_ns"] <= 0
    ):
        raise WorkerError(f"{arm} compile-cold timing invalid")
    if (
        type(value["stop_decision_ns"]) is not int
        or value["stop_decision_ns"] < 0
        or value["stop_decision_ns"]
        != value["decode_critical_path_ns"] - boundary_times[-1]
        or value["ttft_ns"]
        < value["prefill_ns"] + value["cache_conversion_ns"] + boundary_times[0]
        or value["arm_wall_ns"] < value["decode_critical_path_ns"]
        or not _finite(value["token_rate"])
        or value["token_rate"] <= 0
        or abs(value["token_rate"] - len(physical) / (value["decode_critical_path_ns"] / 1e9)) > 1e-9
    ):
        raise WorkerError(f"{arm} primary timing/rate invalid")
    if (
        value["warmup_forwards"] != WARMUP_FORWARDS
        or not isinstance(value["warmup_forward_submit_ns"], list)
        or len(value["warmup_forward_submit_ns"]) != WARMUP_FORWARDS
        or any(type(number) is not int or number <= 0 for number in value["warmup_forward_submit_ns"])
        or not isinstance(value["warmup_readback_ns"], list)
        or len(value["warmup_readback_ns"]) != 2
        or any(type(number) is not int or number <= 0 for number in value["warmup_readback_ns"])
        or value["warmup_cache_discarded"] is not True
    ):
        raise WorkerError(f"{arm} warmup evidence invalid")
    if (
        value["observed_model_work_ns"] != budget["observed_model_work_ns"]
        or value["charged_model_work_ns"] != budget["charged_model_work_ns"]
        or value["charge_accepted"] != budget["charge_accepted"]
        or value["arm_wall_ns"] != value["observed_model_work_ns"]
        or value["charge_accepted"] is not True
    ):
        raise WorkerError(f"{arm} budget/arm mismatch")
    _budget_schema(value["budget_summary"], f"{arm} embedded", allow_overrun=False)
    if value["budget_summary"]["required_break_seconds"] + 1e-6 < budget["required_break_blocks"] * 4.0:
        raise WorkerError(f"{arm} required break was not completed")
    if resources is not None:
        _validate_resource(resources, arm, allow_limit_failure=False)
        if value["resource_snapshot"] != resources:
            raise WorkerError(f"{arm} resource snapshot mismatch")


def _validate_event(
    event: dict[str, Any],
    pid: int,
    identity: dict[str, Any],
    order: tuple[str, ...],
    returncode: int,
    *,
    block: int | None = None,
    bindings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    worker = _worker_module()
    bindings = _validate_protocol_contract() if bindings is None else bindings
    if not isinstance(event, dict) or set(event) != set(bindings["event_required_fields"]):
        raise WorkerError("worker event schema has unexpected fields")
    status = event["status"]
    terminal_statuses = {
        "complete", "correctness_failed", "candidate_not_runnable",
        "resource_or_budget_failed", "error",
    }
    if status not in terminal_statuses or event["event"] != "complete":
        raise WorkerError("worker status invalid")
    expected_block = block if block is not None else event["process_index"]
    if (
        returncode not in (0, 1)
        or (returncode == 0) != (status == "complete")
        or type(expected_block) is not int
        or not 1 <= expected_block <= PAIR_COUNT
        or event["process_index"] != expected_block
        or event["study_id"] != STUDY_ID
        or event["run_id"] != RUN_ID
        or event["candidate_id"] != CANDIDATE_ID
        or event["formal_claim"] is not False
        or event["protocol_version"] != bindings["protocol_version"]
        or event["model_key"] != MODEL_KEY
        or event["model_id"] != MODEL_ID
        or event["snapshot_revision"] != MODEL_REVISION
        or event["snapshot_path"] != identity["snapshot_path"]
        or event["snapshot_sha256"] != identity["snapshot_sha256"]
        or event["weight_sha256"] != identity["weight_sha256"]
        or event["device"] != "Device(gpu, 0)"
        or event["power_source"] != "ac_power"
        or event["pid"] != pid
        or event["load_count"] != 1
        or event["cache_capacity"] != CAPACITY
        or event["max_physical_tokens"] != OUTPUT_TOKEN_LIMIT
        or event["warmup_forwards_per_arm"] != WARMUP_FORWARDS
        or event["sampler_temperature"] != 0.0
        or event["greedy"] is not True
        or event["worker_watchdog_seconds"] != CONTINUOUS_MODEL_LIMIT_SECONDS
        or tuple(event["arm_order"]) != order
        or order != PAIR_SCHEDULE[expected_block - 1]
    ):
        raise WorkerError("worker event identity mismatch")
    prompt_ids = event["prompt_token_ids"]
    try:
        rendered = base64.b64decode(event["rendered_prompt_b64"], validate=True)
    except (TypeError, ValueError) as exc:
        raise WorkerError("rendered prompt is not strict base64") from exc
    if (
        event["prompt_sha256"] != EXPECTED_PROMPT_SHA256
        or not isinstance(prompt_ids, list)
        or len(prompt_ids) != 322
        or any(type(token) is not int or token < 0 for token in prompt_ids)
        or event["prompt_tokens"] != len(prompt_ids)
        or event["prompt_token_sha256"] != EXPECTED_PROMPT_TOKEN_SHA256
        or _hash_ints(prompt_ids) != EXPECTED_PROMPT_TOKEN_SHA256
        or event["rendered_prompt_sha256"] != EXPECTED_RENDERED_PROMPT_SHA256
        or _sha256_bytes(rendered) != EXPECTED_RENDERED_PROMPT_SHA256
        or event["eos_token_ids"] != [1, 106]
    ):
        raise WorkerError("worker prompt/token identity mismatch")
    integrity = event["snapshot_integrity"]
    if not isinstance(integrity, dict) or set(integrity) != {
        "before_load_stat_manifest", "after_load_stat_manifest",
        "bound_snapshot_sha256", "bound_weight_sha256",
    } or (
        integrity["before_load_stat_manifest"] != identity["execution_stat_manifest"]
        or integrity["after_load_stat_manifest"] != identity["execution_stat_manifest"]
        or integrity["bound_snapshot_sha256"] != identity["snapshot_sha256"]
        or integrity["bound_weight_sha256"] != identity["weight_sha256"]
    ):
        raise WorkerError("worker snapshot integrity mismatch")
    if (
        event["preregistration_sha256"] != FROZEN_PREREGISTRATION_SHA256
        or event["code_fingerprints"] != bindings["code_fingerprints"]
        or event["code_sha256"] != bindings["code_sha256"]
        or event["environment_sha256"] != bindings["environment_sha256"]
        or _current_code_fingerprints() != bindings["code_fingerprints"]
    ):
        raise WorkerError("worker code/spec/environment fingerprint invalid")

    arms, arm_budget, arm_resources = event["arms"], event["arm_budget"], event["arm_resources"]
    if (
        not isinstance(arms, dict)
        or not isinstance(arm_budget, dict)
        or not isinstance(arm_resources, dict)
        or not set(arms).issubset(ARM_NAMES)
        or not set(arm_budget).issubset(ARM_NAMES)
        or set(arm_budget) != set(arm_resources)
        or not set(arm_budget).issuperset(arms)
        or (status in {"complete", "correctness_failed"} and set(arms) != set(ARM_NAMES))
    ):
        raise WorkerError("worker arm evidence sets invalid")
    _budget_schema(event["budget"], "worker", allow_overrun=status == "resource_or_budget_failed")
    for arm_name in arm_budget:
        _arm_budget_schema(arm_budget[arm_name], arm_name, status)
        _validate_resource(
            arm_resources[arm_name], arm_name,
            allow_limit_failure=status == "resource_or_budget_failed",
        )
    for arm_name, arm_value in arms.items():
        _validate_arm(
            arm_name, arm_value, arm_budget[arm_name],
            resources=arm_resources[arm_name], bindings=bindings,
        )
    observed = sum(value["observed_model_work_ns"] for value in arm_budget.values())
    charged = sum(value["charged_model_work_ns"] for value in arm_budget.values())
    recorded = sum(value["guard_recorded_model_work_ns"] for value in arm_budget.values())
    if (
        event["model_work_ns"] != observed
        or event["observed_model_work_ns"] != observed
        or event["charged_model_work_ns"] != charged
        or event["guard_recorded_model_work_ns"] != recorded
        or abs(event["budget"]["gpu_work_seconds"] - recorded / 1e9) > 1e-6
    ):
        raise WorkerError("worker model-work accounting invalid")
    if status == "complete" and any(not value["charge_accepted"] for value in arm_budget.values()):
        raise WorkerError("complete event contains a rejected charge")

    for field in ("model_load_ns", "compile_wrapper_ns", "process_wall_ns"):
        if type(event[field]) is not int or event[field] <= 0:
            raise WorkerError(f"worker timing invalid: {field}")
    if event["compile_cold_ns"] is not None and (
        type(event["compile_cold_ns"]) is not int or event["compile_cold_ns"] <= 0
    ):
        raise WorkerError("worker compile-cold timing invalid")
    if arms:
        first = arms.get(order[0])
        second = arms.get(order[1])
        if first is not None and (
            first["compile_cold_ns"] != first["first_warmup_materialization_ns"]
            or event["compile_cold_ns"] != first["compile_cold_ns"]
        ):
            raise WorkerError("first-arm compile-cold evidence invalid")
        if second is not None and second["compile_cold_ns"] is not None:
            raise WorkerError("second arm unexpectedly reports compile-cold time")
    _validate_resource(
        {field: event[field] for field in worker.RESOURCE_REQUIRED_FIELDS},
        "worker", allow_limit_failure=status == "resource_or_budget_failed",
    )
    if arm_resources and (
        event["rss_peak_bytes"] < max(value["rss_peak_bytes"] for value in arm_resources.values())
        or event["mlx_peak_bytes"] < max(value["mlx_peak_bytes"] for value in arm_resources.values())
    ):
        raise WorkerError("worker peak is below arm peak")
    if event["host_transfer_claim"] != {
        "api_call_unit": "one flattened vector tolist call per boundary",
        "physical_dma_count_observable": False,
        "physical_dma_count": None,
    } or event["determinism"] != {
        "seed": BOOTSTRAP_SEED,
        "greedy_no_sampling_randomness": True,
        "within_arm_across_processes_checked_by_parent": True,
    }:
        raise WorkerError("worker host-transfer/determinism claim changed")

    correctness = event["correctness"]
    if not isinstance(correctness, dict) or set(correctness) != set(bindings["correctness_required_fields"]):
        raise WorkerError("worker correctness schema invalid")
    complete_arms = set(arms) == set(ARM_NAMES)
    logical_equal = complete_arms and len({arms[name]["logical_token_sha256"] for name in ARM_NAMES}) == 1
    visible_equal = complete_arms and len({arms[name]["visible_token_sha256"] for name in ARM_NAMES}) == 1
    text_equal = complete_arms and len({arms[name]["text_sha256"] for name in ARM_NAMES}) == 1
    prompt_equal = complete_arms and len({arms[name]["prompt_sha256"] for name in ARM_NAMES}) == 1
    no_eos = complete_arms and all(not arms[name]["eos_found"] for name in ARM_NAMES)
    physical_equal_if_required = not no_eos or len(
        {arms[name]["physical_token_sha256"] for name in ARM_NAMES}
    ) == 1
    first_mismatch = None
    if complete_arms and not (logical_equal and visible_equal and text_equal):
        left, right = (arms[name]["logical_tokens"] for name in ARM_NAMES)
        mismatch_index = next(
            (index for index, pair in enumerate(zip(left, right)) if pair[0] != pair[1]),
            min(len(left), len(right)),
        )
        first_mismatch = {
            "left_arm": ARM_NAMES[0], "right_arm": ARM_NAMES[1],
            "token_index": mismatch_index,
        }
    expected_correctness = {
        "logical_tokens_equal": logical_equal,
        "visible_tokens_equal": visible_equal,
        "visible_text_equal": text_equal,
        "prompt_identity_equal": prompt_equal,
        "physical_tokens_equal_when_no_eos": physical_equal_if_required,
        "first_mismatch": first_mismatch,
        "pass": bool(
            complete_arms and logical_equal and visible_equal and text_equal
            and prompt_equal and physical_equal_if_required
        ),
    }
    if correctness != expected_correctness:
        raise WorkerError("worker correctness evidence is inconsistent")
    if status == "complete" and (correctness["pass"] is not True or event["error"] is not None):
        raise WorkerError("complete worker event failed functional gates")
    if status == "correctness_failed" and (correctness["pass"] is not False or not isinstance(event["error"], dict)):
        raise WorkerError("correctness terminal event is inconsistent")
    if status in {"candidate_not_runnable", "resource_or_budget_failed", "error"} and not isinstance(event["error"], dict):
        raise WorkerError("failed worker event lacks bounded error evidence")
    if isinstance(event["error"], dict) and (
        set(event["error"]) != {"type", "message"}
        or not isinstance(event["error"]["type"], str)
        or not isinstance(event["error"]["message"], str)
        or len(event["error"]["type"]) > 120
        or len(event["error"]["message"]) > 500
    ):
        raise WorkerError("worker error evidence is invalid")
    return event


def _run_block(
    block: int,
    order: tuple[str, ...],
    identity: dict[str, Any],
    prereg_hash: str,
    deadline: float,
    bindings: dict[str, Any],
) -> dict[str, Any]:
    if (_remaining(deadline) or 0) <= 0: raise ResourceError("worker deadline exhausted")
    power = require_ac_power()
    if power != "ac_power":
        raise ResourceError("AC power was lost before worker start")
    swap_before = _swap_used_bytes()
    if swap_before is None: raise ResourceError("swap unavailable before worker")
    started = time.perf_counter_ns()
    process = subprocess.Popen([sys.executable, str(WORKER), "--worker", "--model-key", MODEL_KEY], cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0, env=_environment(identity, block, order, prereg_hash, bindings), start_new_session=True)
    if process.stdout is None or process.stderr is None: _terminate(process, deadline); raise WorkerError("worker pipes unavailable")
    out: dict[str, Any] = {}; err: dict[str, Any] = {}
    out_thread = threading.Thread(target=_read_capped, args=(process.stdout, process, MAX_EVENT_BYTES, out, "payload", deadline), daemon=True)
    err_thread = threading.Thread(target=_read_capped, args=(process.stderr, process, MAX_STDERR_BYTES, err, "stderr", deadline), daemon=True)
    out_thread.start(); err_thread.start()
    try:
        timeout = min(WORKER_TIMEOUT_SECONDS, _remaining(deadline) or 0)
        if timeout <= 0: _terminate(process, deadline); raise WorkerError("worker deadline exhausted")
        try: process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc: _terminate(process, deadline); raise WorkerError("worker timed out") from exc
        remaining = _remaining(deadline)
        out_thread.join(timeout=remaining); err_thread.join(timeout=_remaining(deadline))
        if out_thread.is_alive() or err_thread.is_alive(): _terminate(process, deadline); raise WorkerError("worker reader exceeded deadline")
        if out.get("overflow") or err.get("overflow"): raise WorkerError("worker output exceeded limit")
        if out.get("read_error") or err.get("read_error"): raise WorkerError("worker output read failed")
        event = _decode_event(out.get("payload", b""))
        if event.get("event") == "error":
            raise WorkerEventError("worker error event", {"event": "error", "block": block, "error_type": str(event.get("error_type", "WorkerError"))[:120], "message": str(event.get("message", "worker error"))[:500]})
        event = _validate_event(
            event, process.pid, identity, order, process.returncode,
            block=block, bindings=bindings,
        )
        swap_after = _swap_used_bytes()
        if swap_after is None: raise ResourceError("swap unavailable after worker")
        parent_status = event["status"]
        parent_delta = swap_after - swap_before
        if parent_delta != 0 or event["rss_peak_bytes"] > MAX_RSS_BYTES or event["mlx_peak_bytes"] > MAX_MLX_BYTES:
            parent_status = "resource_or_budget_failed"
        worker_status = event["status"]
        event.update({
            "block": block,
            "worker_returncode": process.returncode,
            "worker_status": worker_status,
            "terminal_status": parent_status,
            "parent_process_wall_ns": time.perf_counter_ns() - started,
            "parent_swap_before_bytes": swap_before,
            "parent_swap_after_bytes": swap_after,
            "parent_swap_delta_bytes": parent_delta,
            "stderr_tail": err.get("stderr", b"").decode("utf-8", errors="replace")[-4000:],
        })
        event["status"] = parent_status
        return event
    finally:
        try: process.stdout.close()
        except OSError: pass
        try: process.stderr.close()
        except OSError: pass


def _percentile(values: list[float], fraction: float) -> float:
    if not values or any(not math.isfinite(value) for value in values): raise StudyError("invalid statistic")
    ordered = sorted(values); position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper: return float(ordered[lower])
    return float(ordered[lower] * (upper - position) + ordered[upper] * (position - lower))


def summarize(values: list[float]) -> dict[str, Any]:
    """Return the preregistered median/MAD and descriptive percentiles."""
    data = [float(value) for value in values]
    if not data or any(not math.isfinite(value) for value in data): raise ValueError("values must be finite and non-empty")
    median = float(statistics.median(data))
    mad = float(statistics.median([abs(value - median) for value in data]))
    return {"values": data, "median": median, "mad": mad, "p50": _percentile(data, .50), "p95": _percentile(data, .95), "p99": _percentile(data, .99)}


def paired_bootstrap(baseline: list[float], candidate: list[float], seed: int = BOOTSTRAP_SEED, iterations: int = BOOTSTRAP_RESAMPLES) -> dict[str, Any]:
    """Bootstrap paired candidate/baseline median ratios without outlier removal."""
    if len(baseline) != len(candidate) or not baseline or any(not _finite(value) or float(value) <= 0 for value in baseline + candidate): raise ValueError("paired positive finite series required")
    ratios = [float(c) / float(b) for b, c in zip(baseline, candidate)]
    rng = random.Random(seed); size = len(ratios)
    samples = [float(statistics.median([ratios[rng.randrange(size)] for _ in range(size)])) for _ in range(iterations)]
    lower, upper = _percentile(samples, .025), _percentile(samples, .975)
    return {"ratios": ratios, "median": float(statistics.median(ratios)), "lower": lower, "upper": upper, "bootstrap_95_ci": {"lower": lower, "upper": upper, "seed": seed, "resamples": iterations, "method": "paired median-ratio percentile bootstrap"}}


def _arm_stats(runs: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    values = [run["arms"][arm] for run in runs if arm in run.get("arms", {})]
    resources = [run.get("arm_resources", {}).get(arm, {}) for run in runs if arm in run.get("arms", {})]
    def series(getter: Any) -> dict[str, Any] | None:
        data = [float(getter(item)) for item in values]
        return summarize(data) if data else None
    def nested_percentile(field: str, fraction: float) -> dict[str, Any] | None:
        data = [_percentile([float(item) / 1e9 for item in value[field]], fraction) for value in values if value[field]]
        return summarize(data) if data else None
    def boundary_gap(item: dict[str, Any], fraction: float) -> float | None:
        gaps = item["boundary_interarrival_ns"]
        return (float(_percentile(gaps, fraction)) / 1e9) if gaps else None
    def boundary_summary(fraction: float) -> dict[str, Any] | None:
        data = [value for item in values if (value := boundary_gap(item, fraction)) is not None]
        return summarize(data) if data else None
    return {"complete": len(values) == PAIR_COUNT,
            "decode_critical_path": series(lambda item: item["decode_critical_path_ns"] / 1e9),
            "token_rate": series(lambda item: item["token_rate"]),
            "ttft": series(lambda item: item["ttft_ns"] / 1e9),
            "prefill": series(lambda item: item["prefill_ns"] / 1e9),
            "cache_conversion": series(lambda item: item["cache_conversion_ns"] / 1e9),
            "arm_wall": series(lambda item: item["arm_wall_ns"] / 1e9),
            "observed_model_work": series(lambda item: item["observed_model_work_ns"] / 1e9),
            "readback": series(lambda item: item["readback_total_ns"] / 1e9),
            "readback_block": {"p50": nested_percentile("readback_ns", .50), "p95": nested_percentile("readback_ns", .95), "p99": nested_percentile("readback_ns", .99)},
            "block_latency": {"p50": nested_percentile("block_latency_ns", .50), "p95": nested_percentile("block_latency_ns", .95), "p99": nested_percentile("block_latency_ns", .99)},
            "host_boundary": {"p50": nested_percentile("host_boundary_available_ns", .50), "p95": nested_percentile("host_boundary_available_ns", .95), "p99": nested_percentile("host_boundary_available_ns", .99)},
            "host_available_total": series(lambda item: item["host_available_total_ns"] / 1e9),
            "host_boundary_gap": {"p50": boundary_summary(.50), "p95": boundary_summary(.95), "p99": boundary_summary(.99)},
            "worker_process_wall": summarize([float(run["process_wall_ns"]) / 1e9 for run in runs if arm in run.get("arms", {})]),
            "parent_process_wall": summarize([float(run["parent_process_wall_ns"]) / 1e9 for run in runs if arm in run.get("arms", {})]),
            "rss_peak_bytes": summarize([float(item["rss_peak_bytes"]) for item in resources]) if resources and all("rss_peak_bytes" in item for item in resources) else None,
            "mlx_peak_bytes": summarize([float(item["mlx_peak_bytes"]) for item in resources]) if resources and all("mlx_peak_bytes" in item for item in resources) else None,
            "swap_delta_bytes": summarize([float(item["swap_delta_bytes"]) for item in resources]) if resources and all(item.get("swap_delta_bytes") is not None for item in resources) else None}


def _complete_blocks(runs: list[dict[str, Any]]) -> bool:
    return len(runs) == PAIR_COUNT and [run.get("block") for run in runs] == list(range(1, PAIR_COUNT + 1)) and all(run.get("terminal_status", run.get("status")) == "complete" for run in runs)


def _determinism(runs: list[dict[str, Any]], arm: str) -> bool:
    values = [run["arms"][arm] for run in runs if arm in run.get("arms", {})]
    if len(values) != PAIR_COUNT: return False
    return len({(tuple(value["physical_tokens"]), value["physical_token_sha256"]) for value in values}) == 1


def _correctness(runs: list[dict[str, Any]]) -> bool:
    if not _complete_blocks(runs): return False
    first = runs[0]["arms"]
    for run in runs:
        a, b = run["arms"][ARM_NAMES[0]], run["arms"][ARM_NAMES[1]]
        if (a["logical_tokens"], a["visible_tokens"], a["visible_text"]) != (b["logical_tokens"], b["visible_tokens"], b["visible_text"]): return False
        if not a["eos_found"] and a["physical_tokens"] != b["physical_tokens"]: return False
        for arm in ARM_NAMES:
            value = run["arms"][arm]
            if value["prompt_sha256"] != EXPECTED_PROMPT_SHA256: return False
            if value["prompt_token_sha256"] != EXPECTED_PROMPT_TOKEN_SHA256 or value["rendered_prompt_sha256"] != EXPECTED_RENDERED_PROMPT_SHA256: return False
            if value["logical_tokens"] != first[arm]["logical_tokens"] or value["visible_tokens"] != first[arm]["visible_tokens"] or value["visible_text"] != first[arm]["visible_text"]: return False
            if value["physical_tokens"] != first[arm]["physical_tokens"]: return False
    return all(_determinism(runs, arm) for arm in ARM_NAMES)


def _paired(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not _complete_blocks(runs): return {"complete": False, "ratios": {}}
    baseline = [run["arms"][ARM_NAMES[0]]["decode_critical_path_ns"] / 1e9 for run in runs]
    candidate = [run["arms"][ARM_NAMES[1]]["decode_critical_path_ns"] / 1e9 for run in runs]
    return {"complete": True, "primary": paired_bootstrap(baseline, candidate), "baseline_arm": ARM_NAMES[0], "candidate_arm": ARM_NAMES[1]}


def _decision(*, resource_pass: bool, budget_pass: bool, correctness_pass: bool, candidate_runnable: bool, paired: dict[str, Any]) -> str:
    if not resource_pass or not budget_pass: return "resource_or_budget_failed"
    if not candidate_runnable: return "candidate_not_runnable"
    if not correctness_pass: return "correctness_failed"
    primary = paired.get("primary", {})
    if primary.get("median", math.inf) <= .95 and primary.get("upper", math.inf) < 1.0: return "runtime_readback8_wins_exact_scope"
    if primary.get("lower", -math.inf) > 1.0: return "readback8_regression_baseline_retained"
    return "no_clear_speedup_baseline_retained"


def decision_for(*, resource_pass: bool, budget_pass: bool, correctness_pass: bool, candidate_runnable: bool, paired: dict[str, Any]) -> str:
    return _decision(resource_pass=resource_pass, budget_pass=budget_pass, correctness_pass=correctness_pass, candidate_runnable=candidate_runnable, paired=paired)


def _aggregate_budget(runs: list[dict[str, Any]], parent_wall: float) -> dict[str, Any]:
    budgets = [run.get("budget") for run in runs]
    if not budgets or any(not isinstance(value, dict) for value in budgets): return {"valid": False, "error": "missing_child_budget", "parent_wall_seconds": parent_wall}
    try:
        for run, value in zip(runs, budgets): _budget_schema(value, "child", allow_overrun=run.get("terminal_status", run.get("status")) == "resource_or_budget_failed")
        total = sum(float(value["gpu_work_seconds"]) for value in budgets)
        maximum = max((float(value["max_continuous_gpu_seconds"]) for value in budgets), default=0.0)
        required_break = sum(float(value["required_break_seconds"]) for value in budgets)
        accepted_arms = sum(
            1 for run in runs for value in run.get("arm_budget", {}).values()
            if value.get("charge_accepted") is True
        )
        minimum_break = accepted_arms * 13 * 4.0
        valid = (total <= 120.0 and maximum <= 6.0 and parent_wall <= 1200.0
                 and required_break + 1e-6 >= minimum_break
                 and all(float(value["duty_cycle_limit"]) == .15 for value in budgets))
        return {"valid": valid, "gpu_work_seconds": total, "max_continuous_gpu_seconds": maximum, "duty_cycle_limit": .15, "required_break_seconds": required_break, "minimum_preregistered_break_seconds": minimum_break, "accepted_arms": accepted_arms, "child_wall_seconds": sum(float(value["wall_seconds"]) for value in budgets), "parent_wall_seconds": parent_wall, "gpu_work_limit_seconds": 120.0, "continuous_gpu_limit_seconds": 6.0, "wall_limit_seconds": 1200.0, "child_budgets": budgets}
    except (KeyError, TypeError, ValueError, WorkerError) as exc:
        return {"valid": False, "error": f"invalid_child_budget: {exc}", "parent_wall_seconds": parent_wall}


def _budget_gate_pass(error: Any, terminal_status: str | None, budget: dict[str, Any]) -> bool:
    return error is None and terminal_status != "resource_or_budget_failed" and bool(budget.get("valid"))


def _provenance(revision: str, dirty: str, power: str, identity: dict[str, Any], prereg_hash: str, bindings: dict[str, Any]) -> dict[str, Any]:
    files = dict(bindings["code_fingerprints"])
    packages = {name: importlib.metadata.version(name) if importlib.metadata.packages_distributions() else None for name in ()}
    return {"git_revision": revision, "git_status": dirty, "git_dirty_state": bool(dirty), "code_files_sha256": files, "code_sha256": bindings["code_sha256"], "protocol_version": bindings["protocol_version"], "protocol_sha256": bindings["protocol_sha256"], "preregistration_sha256": prereg_hash, "prompt_sha256": EXPECTED_PROMPT_SHA256, "prompt_token_sha256": EXPECTED_PROMPT_TOKEN_SHA256, "rendered_prompt_sha256": EXPECTED_RENDERED_PROMPT_SHA256, "environment_sha256": bindings["environment_sha256"], "environment": {"python": platform.python_version(), "executable": str(Path(sys.executable).resolve()), "packages": {name: (importlib.metadata.version(name) if name in {dist.metadata["Name"] for dist in importlib.metadata.distributions()} else None) for name in ("mlx", "mlx-lm", "numpy", "psutil")}, "fixed_worker_environment": WORKER_ENVIRONMENT}, "hardware": {"machine": platform.machine(), "cpu_brand": _sysctl("machdep.cpu.brand_string"), "memory_bytes": _sysctl("hw.memsize"), "macos": platform.mac_ver()[0], "mlx_default_device": "Device(gpu, 0)"}, "power_source": power, "model_snapshot": identity, "schedule": [list(order) for order in PAIR_SCHEDULE]}


def _execute(run_id: str) -> dict[str, Any]:
    started = time.monotonic(); hard_deadline = started + MAX_WALL_SECONDS; worker_deadline = hard_deadline - FINALIZATION_RESERVE_SECONDS
    parent_guard = BudgetGuard(POLICY)
    revision, dirty, power, identity, swap_start, bindings = _preflight(run_id)
    prereg_hash = _sha256(PREREGISTRATION)
    ATTEMPT_DIR.mkdir(mode=0o700, parents=True, exist_ok=True); _private_dir(ATTEMPT_DIR)
    provenance = _provenance(revision, dirty, power, identity, prereg_hash, bindings)
    _exclusive_json(ATTEMPT_PATH, {"study_id": STUDY_ID, "run_id": run_id, "formal_claim": False, "started_at_unix_ns": time.time_ns(), "schedule": [list(order) for order in PAIR_SCHEDULE], "provenance": provenance}, 0o600)
    state: dict[str, Any] = {"schema_version": 1, "study_id": STUDY_ID, "run_id": RUN_ID, "candidate_id": CANDIDATE_ID, "formal_claim": False, "runs": [], "worker_events": [], "error": None, "partial_result": True, "provenance": provenance}
    _atomic_result(state)
    error = None; terminal = None
    try:
        for block, order in enumerate(PAIR_SCHEDULE, 1):
            parent_guard.check_wall()
            if time.monotonic() >= worker_deadline: raise ResourceError("finalization reserve reached")
            run = _run_block(block, order, identity, prereg_hash, worker_deadline, bindings); state["runs"].append(run)
            terminal_status = run.get("terminal_status", run.get("status"))
            state["partial_result"] = len(state["runs"]) != PAIR_COUNT
            _atomic_result(state, replace=True)
            if terminal_status != "complete": terminal = terminal_status; break
    except WorkerEventError as exc:
        state["worker_events"].append(exc.event); error = {"type": type(exc).__name__, "message": str(exc)[:500]}
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc)[:500]}
    try:
        swap_end = _swap_used_bytes()
    except BaseException as exc:
        swap_end = None
        if error is None:
            error = {"type": type(exc).__name__, "message": str(exc)[:500]}
    if swap_end is None and error is None:
        error = {"type": "ResourceError", "message": "swap unavailable after study"}
    swap_delta = swap_end - swap_start if swap_end is not None else None
    try:
        parent_guard.check_wall()
    except Exception as exc:
        if error is None: error = {"type": type(exc).__name__, "message": str(exc)[:500]}
    parent_wall = time.monotonic() - started
    budget = _aggregate_budget(state["runs"], parent_wall)
    resource_pass = (error is None and swap_delta == 0 and terminal != "resource_or_budget_failed"
                     and all(run.get("parent_swap_delta_bytes") == 0 for run in state["runs"])
                     and all(run.get("rss_peak_bytes", MAX_RSS_BYTES + 1) <= MAX_RSS_BYTES and run.get("mlx_peak_bytes", MAX_MLX_BYTES + 1) <= MAX_MLX_BYTES for run in state["runs"]))
    budget_pass = _budget_gate_pass(error, terminal, budget)
    candidate_runnable = error is None and terminal != "candidate_not_runnable" and all(run.get("terminal_status", run.get("status")) != "candidate_not_runnable" for run in state["runs"])
    correctness = _correctness(state["runs"])
    if error or terminal in {"resource_or_budget_failed", "error"}: decision = "resource_or_budget_failed"
    elif terminal == "correctness_failed": decision = "correctness_failed"
    elif terminal == "candidate_not_runnable": decision = "candidate_not_runnable"
    elif not _complete_blocks(state["runs"]): decision = "resource_or_budget_failed"
    else: decision = _decision(resource_pass=resource_pass, budget_pass=budget_pass, correctness_pass=correctness, candidate_runnable=candidate_runnable, paired=_paired(state["runs"]))
    post_identity: dict[str, Any]
    if time.monotonic() >= hard_deadline:
        post_identity = {"error": "hard wall deadline exhausted before postflight"}
        resource_pass = False
    else:
        try:
            post_snapshot = resolve_local_model_snapshot(MODEL_ID)
            post_identity = _snapshot_identity(post_snapshot)
            if (post_identity.get("snapshot_sha256") != identity.get("snapshot_sha256") or
                    post_identity.get("weight_sha256") != identity.get("weight_sha256") or
                    post_identity.get("execution_files_sha256") != identity.get("execution_files_sha256") or
                    post_identity.get("execution_stat_manifest") != identity.get("execution_stat_manifest") or
                    _sha256(PREREGISTRATION) != prereg_hash or
                    _current_code_fingerprints() != bindings["code_fingerprints"] or
                    _environment_fingerprint() != bindings["environment_sha256"] or
                    require_ac_power() != "ac_power"):
                resource_pass = False
        except BaseException as exc:
            post_identity = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
            resource_pass = False
    if not resource_pass:
        decision = "resource_or_budget_failed"
    state.update({"completed_at_unix_ns": time.time_ns(), "decision": decision, "error": error, "partial_result": len(state["runs"]) != PAIR_COUNT or error is not None, "budget": budget, "resources": {"swap_before_bytes": swap_start, "swap_after_bytes": swap_end, "swap_delta_bytes": swap_delta, "max_rss_peak_bytes": max((run.get("rss_peak_bytes", 0) for run in state["runs"]), default=None), "max_mlx_peak_bytes": max((run.get("mlx_peak_bytes", 0) for run in state["runs"]), default=None)}, "gates": {"resource_pass": resource_pass, "budget_pass": budget_pass, "correctness_pass": correctness, "candidate_runnable": candidate_runnable, "all_pairs_completed": len(state["runs"]) == PAIR_COUNT}, "metrics": {"arms": {arm: _arm_stats(state["runs"], arm) for arm in ARM_NAMES}, "paired": _paired(state["runs"]), "runs_completed": len(state["runs"])}, "provenance": provenance, "thresholds": {"median_ratio_max": .95, "bootstrap_upper_max": 1.0, "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_resamples": BOOTSTRAP_RESAMPLES, "no_outlier_removal": True}})
    state["snapshot_postflight"] = post_identity
    _atomic_result(state, replace=True)
    return state


def _self_check() -> int:
    before = _evidence_state(); _validate_evidence_state(before)
    bindings = _validate_protocol_contract()
    assert bindings["protocol_sha256"] == EXPECTED_PROTOCOL_SHA256
    assert _sha256(PREREGISTRATION) == FROZEN_PREREGISTRATION_SHA256
    assert _current_code_fingerprints() == bindings["code_fingerprints"]
    assert len(PAIR_SCHEDULE) == 6 and all(sorted(order) == sorted(ARM_NAMES) for order in PAIR_SCHEDULE)
    assert summarize([1.0, 2.0, 3.0])["median"] == 2.0
    assert paired_bootstrap([1.0] * 6, [1.0] * 6)["upper"] == 1.0
    assert _decision(resource_pass=False, budget_pass=True, correctness_pass=True, candidate_runnable=True, paired={}) == "resource_or_budget_failed"
    assert _decision(resource_pass=True, budget_pass=True, correctness_pass=True, candidate_runnable=True, paired={"primary": {"median": .9, "upper": .99}}) == "runtime_readback8_wins_exact_scope"
    assert _decision(resource_pass=True, budget_pass=True, correctness_pass=True, candidate_runnable=True, paired={"primary": {"median": 1.1, "lower": 1.01}}) == "readback8_regression_baseline_retained"
    after = _evidence_state(); _validate_evidence_state(after); assert before == after
    print(json.dumps({"checks": 9, "self_check": "pass"}, sort_keys=True)); return 0


def _show() -> int:
    before = _evidence_state(); _validate_evidence_state(before)
    if not RESULT_PATH.is_file() or RESULT_PATH.is_symlink():
        print(json.dumps({"status": "unavailable", "study_id": STUDY_ID, "formal_claim": False}, sort_keys=True)); return 78
    try:
        value = _decode_event(RESULT_PATH.read_bytes())
    except Exception as exc:
        print(json.dumps({"status": "invalid", "error": type(exc).__name__}, sort_keys=True)); return 2
    summary = {"study_id": value.get("study_id"), "run_id": value.get("run_id"), "candidate_id": value.get("candidate_id"), "decision": value.get("decision"), "formal_claim": False, "runs_completed": len(value.get("runs", [])), "partial_result": value.get("partial_result")}
    after = _evidence_state(); _validate_evidence_state(after)
    if before != after:
        print(json.dumps({"status": "invalid", "error": "evidence_changed"}, sort_keys=True)); return 2
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True)); return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="measure_batched_readback", allow_abbrev=False)
    modes = parser.add_mutually_exclusive_group(); modes.add_argument("--execute", action="store_true"); modes.add_argument("--self-check", action="store_true"); modes.add_argument("--show", action="store_true")
    parser.add_argument("--run-id", default=RUN_ID); args = parser.parse_args(argv)
    if args.self_check:
        try: return _self_check()
        except Exception as exc:
            print(json.dumps({"self_check": "failed", "error": str(exc)[:500]}, sort_keys=True)); return 2
    if args.show: return _show()
    if not args.execute:
        print(json.dumps({"state": "not_released", "required_flag": "--execute", "formal_claim": False}, sort_keys=True)); return 78
    try: report = _execute(args.run_id)
    except BaseException as exc:
        print(json.dumps({"state": "not_started_or_partial", "error_type": type(exc).__name__[:120], "error": str(exc)[:500], "formal_claim": False}, sort_keys=True)); return 2
    print(json.dumps({"decision": report["decision"], "formal_claim": False, "result": str(RESULT_PATH.relative_to(PROJECT_ROOT)), "run_id": RUN_ID, "pairs_completed": len(report["runs"])}, sort_keys=True))
    return 0 if report["decision"] == "runtime_readback8_wins_exact_scope" else 1


if __name__ == "__main__":
    raise SystemExit(main())
