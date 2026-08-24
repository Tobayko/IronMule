#!/usr/bin/env python3
"""Fail-closed parent harness for the cycle-16 MLX fixed-cache study.

The worker owns one model load and one complete three-arm block, including
GPU charging and duty-cycle pacing.  This parent owns all admission checks,
process isolation, wall-time checks, evidence and statistics.
Importing or invoking this file without ``--execute`` never imports MLX or
starts a child process.
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
import tempfile
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from _bench import require_ac_power, resolve_local_model_snapshot  # noqa: E402
from friday_evidence.budget import BudgetError, BudgetGuard  # noqa: E402
from friday_evidence.registry import BudgetPolicy  # noqa: E402

STUDY_ID = "matmul-compile-ab-20260824-01"
RUN_ID = "matmul-compile-validation-20260824-01"
MODEL_KEY = "4b"
MODEL_ID = "mlx-community/gemma-3-4b-it-4bit"
MODEL_REVISION = "93724907d4ed1745d2fe50baadf3b0b01a65abf2"
EXPECTED_MACHINE = "arm64"
EXPECTED_CPU_BRAND = "Apple M1 Max"
EXPECTED_MEMORY_BYTES = 32 * 1024**3
REQUIRED_PACKAGES = {"mlx": "0.32.0", "mlx-lm": "0.31.3"}
FROZEN_PREREGISTRATION_SHA256 = "dc84020e9bdf07043c5395d3d21d7941f466eae1007ab15cd031f78479696fcf"
PREREGISTRATION = Path(__file__).with_name("PREREGISTRATION.md")
WORKER = Path(__file__).with_name("worker.py")
RESULT_PATH = Path(__file__).with_name("results.json")
ATTEMPT_DIR = PROJECT_ROOT / ".friday-data" / "matmul-compile-ab"
ATTEMPT_PATH = ATTEMPT_DIR / "attempt.json"
MAX_EVENT_BYTES = 1_000_000
MAX_STDERR_BYTES = 64_000
WORKER_TIMEOUT_SECONDS = 300.0
FINALIZATION_RESERVE_SECONDS = 15.0
CONTINUOUS_MODEL_LIMIT_SECONDS = 6.0
MAX_MODEL_WORK_SECONDS = 120.0
MAX_WALL_SECONDS = 1_200.0
MAX_RSS_BYTES = 6 * 1024**3
MAX_MLX_BYTES = 5 * 1024**3
PAIR_COUNT = 6
RUN_COUNT = PAIR_COUNT
ARM_COUNT = 3
DECODE_FORWARDS = 31
OUTPUT_TOKENS = 32
EXPECTED_PROMPT_SHA256 = "c746eca8644a18fc75673acb9b3dbdf03825cbfba6c76faede5d909cf3d2ea0b"
BOOTSTRAP_SEED = 20260824
BOOTSTRAP_RESAMPLES = 10_000
ARM_NAMES = ("standard_eager", "fixed_eager", "fixed_compiled")
ARM_PERMUTATIONS = (
    ("standard_eager", "fixed_eager", "fixed_compiled"),
    ("standard_eager", "fixed_compiled", "fixed_eager"),
    ("fixed_eager", "standard_eager", "fixed_compiled"),
    ("fixed_eager", "fixed_compiled", "standard_eager"),
    ("fixed_compiled", "standard_eager", "fixed_eager"),
    ("fixed_compiled", "fixed_eager", "standard_eager"),
)
PAIR_SCHEDULE = ARM_PERMUTATIONS
MAX_MEMORY_BYTES = MAX_RSS_BYTES
WORKER_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "PYTHONNOUSERSITE": "1",
}
UNSAFE_ENVIRONMENT = ("PYTHONHOME", "PYTHONINSPECT", "PYTHONPATH", "PYTHONSTARTUP")
POLICY = BudgetPolicy(
    gpu_work_limit_s=120.0,
    continuous_gpu_limit_s=6.0,
    required_break_s=4.0,
    duty_window_s=60.0,
    duty_cycle_limit=0.15,
    wall_limit_s=1_200.0,
    candidate_cooldown_s=60.0,
)


class StudyError(RuntimeError):
    pass


class WorkerError(StudyError):
    pass


class WorkerEventError(WorkerError):
    def __init__(self, message: str, event: dict[str, Any]):
        super().__init__(message)
        self.event = event


class ResourceError(StudyError):
    pass


@lru_cache(maxsize=1)
def _worker_module() -> Any:
    spec = importlib.util.spec_from_file_location("matmul_compile_ab_worker", WORKER)
    if spec is None or spec.loader is None:
        raise StudyError("worker module is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canonical(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, ensure_ascii=True,
                      separators=(",", ":"), sort_keys=True).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _evidence_state() -> dict[str, dict[str, Any]]:
    """Capture immutable state for the self-check evidence lifecycle gate."""
    state: dict[str, dict[str, Any]] = {}
    for label, path in (("result", RESULT_PATH), ("marker", ATTEMPT_PATH)):
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            state[label] = {"exists": False, "regular": False, "symlink": False,
                            "sha256": None, "mode": None}
            continue
        is_symlink = stat.S_ISLNK(metadata.st_mode)
        is_regular = stat.S_ISREG(metadata.st_mode)
        if is_symlink or not is_regular:
            raise StudyError(f"{label} evidence path is not a regular file")
        state[label] = {"exists": True, "regular": True, "symlink": False,
                        "sha256": _sha256(path), "mode": stat.S_IMODE(metadata.st_mode)}
    return state


def _validate_evidence_state(state: dict[str, dict[str, Any]]) -> None:
    marker = state["marker"]
    for label, evidence in state.items():
        if evidence["exists"] != evidence["regular"]:
            raise StudyError(f"{label} evidence lifecycle state is invalid")
    if marker["exists"] and marker["mode"] != 0o600:
        raise StudyError("attempt marker is not private")


def _git(*args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=PROJECT_ROOT, check=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               timeout=10)
    return completed.stdout.decode("utf-8", errors="strict").strip()


def _sysctl(name: str) -> str | None:
    try:
        completed = subprocess.run(["/usr/sbin/sysctl", "-n", name], check=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                   timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.decode("utf-8", errors="replace").strip() or None


def _swap_used_bytes() -> int | None:
    try:
        import psutil
        value = psutil.swap_memory().used
    except Exception:
        return None
    return value if type(value) is int and value >= 0 else None


def _exclusive_json(path: Path, value: dict[str, Any], mode: int) -> None:
    payload = _canonical(value) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _atomic_result(value: dict[str, Any]) -> None:
    if RESULT_PATH.exists() or RESULT_PATH.is_symlink():
        raise StudyError("result path already exists")
    temporary = RESULT_PATH.with_name(f".{RESULT_PATH.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise StudyError("result temporary path already exists")
    _exclusive_json(temporary, value, 0o600)
    os.chmod(temporary, 0o644)
    os.replace(temporary, RESULT_PATH)


def _require_private_directory(path: Path) -> None:
    metadata = os.lstat(path)
    if (not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700):
        raise StudyError("attempt directory is not private")


def _snapshot_identity(snapshot: Any) -> dict[str, Any]:
    root = Path(snapshot.path).resolve(strict=True)
    try:
        repository = root.parent.parent.resolve(strict=True)
        root.relative_to(repository)
    except (OSError, ValueError) as exc:
        raise StudyError("snapshot is outside its local repository") from exc
    if root.parent.name != "snapshots":
        raise StudyError("unexpected local snapshot layout")
    required = ["config.json", "tokenizer_config.json"]
    for name in ("tokenizer.json", "tokenizer.model"):
        if (root / name).is_file() or (root / name).is_symlink():
            required.append(name)
            break
    required.extend(snapshot.weight_files)
    files: dict[str, str] = {}
    manifest: dict[str, dict[str, Any]] = {}
    for relative in dict.fromkeys(required):
        candidate = (root / relative).resolve(strict=True)
        try:
            candidate.relative_to(repository)
        except ValueError as exc:
            raise StudyError(f"snapshot file escaped repository: {relative}") from exc
        if not candidate.is_file():
            raise StudyError(f"snapshot file is not regular: {relative}")
        files[relative] = _sha256(candidate)
        metadata = candidate.stat()
        manifest[relative] = {"dev": int(metadata.st_dev), "inode": int(metadata.st_ino),
                              "mtime_ns": int(metadata.st_mtime_ns),
                              "path": str(candidate), "size": int(metadata.st_size)}
    weights = {name: files.get(name) for name in snapshot.weight_files}
    if any(value is None for value in weights.values()):
        raise StudyError("weight hash is missing")
    identity = dict(snapshot.report_identity())
    identity.update({"snapshot_path": str(root), "snapshot_files_sha256": files,
                     "snapshot_sha256": _sha256_bytes(_canonical(files)),
                     "execution_stat_manifest": manifest,
                     "weight_sha256": weights})
    return identity


def _clean_worktree() -> tuple[str, str]:
    revision = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=all", "--",
                  ".", ":(exclude)ProjectAtlas")
    if status:
        raise StudyError("project worktree is dirty")
    return revision, status


def _require_target() -> None:
    if platform.machine() != EXPECTED_MACHINE:
        raise StudyError("architecture is not arm64")
    if _sysctl("machdep.cpu.brand_string") != EXPECTED_CPU_BRAND:
        raise StudyError("CPU is not Apple M1 Max")
    if _sysctl("hw.memsize") != str(EXPECTED_MEMORY_BYTES):
        raise StudyError("memory is not 32 GiB")
    for package, expected in REQUIRED_PACKAGES.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise StudyError(f"package unavailable: {package}") from exc
        if actual != expected:
            raise StudyError(f"package version changed: {package}")
    import mlx.core as mx
    if str(mx.default_device()) != "Device(gpu, 0)":
        raise StudyError("MLX is not using the registered GPU")


def _preflight(run_id: str) -> tuple[str, str, str, dict[str, Any], int]:
    if run_id != RUN_ID:
        raise StudyError("run ID is not registered")
    expected_python = (PROJECT_ROOT / ".venv" / "bin" / "python").resolve(strict=True)
    if Path(sys.executable).resolve() != expected_python:
        raise StudyError("study must use the project virtual environment")
    prereg_hash = _sha256(PREREGISTRATION)
    if not FROZEN_PREREGISTRATION_SHA256 or prereg_hash != FROZEN_PREREGISTRATION_SHA256:
        raise StudyError("preregistration hash is not sealed")
    if _worker_module().PROMPT_SHA256 != EXPECTED_PROMPT_SHA256:
        raise StudyError("worker prompt hash is not the preregistered prompt")
    if RESULT_PATH.exists() or RESULT_PATH.is_symlink():
        raise StudyError("result already exists; no retry")
    if ATTEMPT_PATH.exists() or ATTEMPT_PATH.is_symlink():
        raise StudyError("attempt marker already exists; no retry")
    revision, dirty = _clean_worktree()
    _require_target()
    try:
        power = require_ac_power()
    except SystemExit as exc:
        raise ResourceError(str(exc)) from exc
    snapshot = resolve_local_model_snapshot(MODEL_ID)
    if snapshot.revision != MODEL_REVISION:
        raise StudyError("local snapshot revision changed")
    identity = _snapshot_identity(snapshot)
    swap = _swap_used_bytes()
    if swap is None:
        raise ResourceError("swap usage unavailable")
    return revision, dirty, power, identity, swap


def _environment(identity: dict[str, Any], block: int, order: tuple[str, ...]) -> dict[str, str]:
    environment = os.environ.copy()
    for name in UNSAFE_ENVIRONMENT:
        environment.pop(name, None)
    environment.update(WORKER_ENVIRONMENT)
    environment.update({
        "FRIDAY_MATMUL_PARENT_PID": str(os.getpid()),
        "FRIDAY_MATMUL_RUN_ID": RUN_ID,
        "FRIDAY_MATMUL_MODEL_KEY": MODEL_KEY,
        "FRIDAY_MATMUL_NONCE": "cycle16-fixed-cache-v1",
        "FRIDAY_MATMUL_BLOCK": str(block),
        "FRIDAY_MATMUL_ARM_ORDER": json.dumps(list(order), separators=(",", ":")),
        "FRIDAY_MATMUL_SNAPSHOT_PATH": str(identity["snapshot_path"]),
        "FRIDAY_MATMUL_SNAPSHOT_REVISION": MODEL_REVISION,
        "FRIDAY_MATMUL_SNAPSHOT_SHA256": identity["snapshot_sha256"],
        "FRIDAY_MATMUL_WEIGHT_SHA256": _canonical(identity["weight_sha256"]).decode("ascii"),
        "FRIDAY_MATMUL_SNAPSHOT_STAT_MANIFEST": _canonical(identity["execution_stat_manifest"]).decode("ascii"),
    })
    return environment


def _environment_fingerprint() -> str:
    value = {"fixed": WORKER_ENVIRONMENT, "removed": UNSAFE_ENVIRONMENT,
             "python": str(Path(sys.executable).resolve()), "machine": platform.machine()}
    return _sha256_bytes(_canonical(value))


def _read_capped(stream: Any, process: subprocess.Popen[bytes], limit: int,
                 result: dict[str, Any], key: str,
                 deadline_monotonic: float | None) -> None:
    data = bytearray()
    try:
        while True:
            chunk = stream.read(min(64 * 1024, limit + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > limit:
                result["overflow"] = True
                _terminate(process, deadline_monotonic)
                break
    except Exception as exc:
        result["read_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        _terminate(process, deadline_monotonic)
    result[key] = bytes(data)


def _remaining(deadline_monotonic: float | None) -> float | None:
    if deadline_monotonic is None:
        return None
    return max(0.0, deadline_monotonic - time.monotonic())


def _terminate(process: subprocess.Popen[bytes], deadline_monotonic: float | None = None) -> None:
    """Terminate without spending time beyond the shared hard deadline."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        try:
            process.terminate()
        except OSError:
            pass
    remaining = _remaining(deadline_monotonic)
    if remaining is not None and remaining <= 0:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            try:
                process.kill()
            except OSError:
                pass
        return
    try:
        process.wait(timeout=min(2.0, remaining) if remaining is not None else 2.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            try:
                process.kill()
            except OSError:
                pass
        remaining = _remaining(deadline_monotonic)
        if remaining is not None and remaining <= 0:
            return
        try:
            process.wait(timeout=min(2.0, remaining) if remaining is not None else 2.0)
        except subprocess.TimeoutExpired:
            return


def _decode_event(payload: bytes) -> dict[str, Any]:
    if not payload or len(payload) > MAX_EVENT_BYTES:
        raise WorkerError("worker output exceeds limit")
    lines = payload.splitlines()
    if len(lines) != 1:
        raise WorkerError("worker must emit exactly one JSON event")
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON field")
            value[key] = item
        return value
    try:
        event = json.loads(lines[0].decode("utf-8"), object_pairs_hook=unique,
                           parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise WorkerError("worker emitted invalid JSON") from exc
    if not isinstance(event, dict):
        raise WorkerError("worker event is not an object")
    return event


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _validate_budget_summary(value: Any, *, label: str, model_work_ns: int | None = None,
                             allow_terminal_overrun: bool = False) -> None:
    required = {
        "gpu_work_seconds", "max_continuous_gpu_seconds", "cooldown_seconds",
        "required_break_seconds", "wall_seconds", "gpu_work_limit_seconds",
        "continuous_gpu_limit_seconds", "duty_cycle_limit", "wall_limit_seconds",
        "candidate_cooldown_seconds", "required_break_limit_seconds",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise WorkerError(f"{label} budget schema is invalid")
    for field in required:
        if not _finite_number(value[field]) or float(value[field]) < 0:
            raise WorkerError(f"{label} budget field is invalid: {field}")
    if (value["gpu_work_limit_seconds"] != MAX_MODEL_WORK_SECONDS or
            value["continuous_gpu_limit_seconds"] != CONTINUOUS_MODEL_LIMIT_SECONDS or
            value["duty_cycle_limit"] != POLICY.duty_cycle_limit or
            value["wall_limit_seconds"] != MAX_WALL_SECONDS or
            value["required_break_limit_seconds"] != POLICY.required_break_s):
        raise WorkerError(f"{label} budget limits are not preregistered")
    if ((value["gpu_work_seconds"] > MAX_MODEL_WORK_SECONDS or
         value["max_continuous_gpu_seconds"] > CONTINUOUS_MODEL_LIMIT_SECONDS or
         value["wall_seconds"] > MAX_WALL_SECONDS) and not allow_terminal_overrun) or \
            value["duty_cycle_limit"] != 0.15:
        raise WorkerError(f"{label} budget limit failed")
    if model_work_ns is not None and abs(value["gpu_work_seconds"] - model_work_ns / 1e9) > 1e-6:
        raise WorkerError(f"{label} budget does not equal model work")


def _validate_event(event: dict[str, Any], pid: int, identity: dict[str, Any],
                    order: tuple[str, ...], returncode: int | None = None) -> dict[str, Any]:
    worker = _worker_module()
    required = {"arm_order", "arms", "cache_capacity", "correctness", "device", "error",
                "event", "fixed_steps", "load_count", "model_id", "model_key", "model_load_ns",
                "model_work_ns", "observed_model_work_ns", "charged_model_work_ns", "guard_recorded_model_work_ns", "mlx_peak_bytes", "pid", "prompt_sha256", "prompt_token_ids",
                "prompt_tokens", "rendered_prompt_b64", "rendered_prompt_sha256", "rss_peak_bytes",
                "sampler_temperature", "snapshot_integrity", "snapshot_path", "snapshot_revision",
                "snapshot_sha256", "status", "study_id", "text_sha256_by_arm", "token_sha256_by_arm",
                "worker_watchdog_seconds", "weight_sha256", "power_source", "arm_budget",
                "arm_resources", "budget", "swap_before_bytes", "swap_after_bytes",
                "swap_delta_bytes"}
    ints = ("cache_capacity", "fixed_steps", "load_count", "model_load_ns", "mlx_peak_bytes",
            "pid", "prompt_tokens", "rss_peak_bytes", "swap_before_bytes")
    status = event.get("status")
    allowed_statuses = {"complete", "correctness_failed", "candidate_not_runnable",
                        "resource_or_budget_failed", "error"}
    terminal = status != "complete"
    if (set(event) != required or event.get("event") != "complete" or
        event.get("device") != "Device(gpu, 0)" or event.get("model_key") != MODEL_KEY or
        event.get("model_id") != MODEL_ID or event.get("study_id") != STUDY_ID or
        event.get("snapshot_revision") != MODEL_REVISION or
        event.get("snapshot_path") != identity["snapshot_path"] or
        event.get("snapshot_sha256") != identity["snapshot_sha256"] or
        event.get("weight_sha256") != identity["weight_sha256"] or
        event.get("load_count") != 1 or event.get("pid") != pid or
        event.get("prompt_sha256") != EXPECTED_PROMPT_SHA256 or event.get("fixed_steps") != OUTPUT_TOKENS or
        event.get("prompt_tokens") != worker.EXPECTED_PROMPT_TOKENS or
        not isinstance(event.get("prompt_token_ids"), list) or
        len(event["prompt_token_ids"]) != worker.EXPECTED_PROMPT_TOKENS or
        any(type(token) is not int for token in event["prompt_token_ids"]) or
        event.get("cache_capacity") != worker.CAPACITY or event.get("sampler_temperature") != 0.0 or
        event.get("worker_watchdog_seconds") != CONTINUOUS_MODEL_LIMIT_SECONDS or
        status not in allowed_statuses or
        any(type(event.get(field)) is not int or event[field] <= 0 for field in ints) or
        type(event.get("model_work_ns")) is not int or event["model_work_ns"] < 0 or
        type(event.get("observed_model_work_ns")) is not int or event["observed_model_work_ns"] < 0 or
        type(event.get("charged_model_work_ns")) is not int or event["charged_model_work_ns"] < 0 or
        type(event.get("guard_recorded_model_work_ns")) is not int or event["guard_recorded_model_work_ns"] < 0 or
        event["model_work_ns"] != event["observed_model_work_ns"] or
        (status == "complete" and event["observed_model_work_ns"] > int(MAX_MODEL_WORK_SECONDS * 1e9)) or
        event["charged_model_work_ns"] > event["observed_model_work_ns"] or
        event["guard_recorded_model_work_ns"] > event["observed_model_work_ns"] or
        (returncode is not None and returncode not in (0, 1)) or
        (returncode == 0 and status != "complete") or
        (returncode == 1 and status == "complete") or
        tuple(event.get("arm_order", ())) != order):
        raise WorkerError("worker event identity/schema invalid")
    if event["power_source"] != "ac_power":
        raise WorkerError("worker power source is not AC")
    if event["swap_after_bytes"] is not None and (type(event["swap_after_bytes"]) is not int or event["swap_after_bytes"] < 0):
        raise WorkerError("worker swap-after field is invalid")
    if event["swap_delta_bytes"] is not None and (type(event["swap_delta_bytes"]) is not int):
        raise WorkerError("worker swap-delta field is invalid")
    if not terminal and (event["swap_after_bytes"] is None or event["swap_delta_bytes"] is None):
        raise WorkerError("complete worker event lacks swap evidence")
    if not isinstance(event["arms"], dict) or not set(event["arms"]).issubset(set(ARM_NAMES)):
        raise WorkerError("worker arm set is invalid")
    if status == "complete" and set(event["arms"]) != set(ARM_NAMES):
        raise WorkerError("worker did not return all three arms")
    _validate_budget_summary(
        event["budget"], label="worker",
        model_work_ns=event["guard_recorded_model_work_ns"],
        allow_terminal_overrun=status == "resource_or_budget_failed",
    )
    if not isinstance(event["arm_budget"], dict) or not set(event["arm_budget"]).issubset(set(ARM_NAMES)):
        raise WorkerError("worker arm budget set is invalid")
    if not isinstance(event["arm_resources"], dict) or not set(event["arm_resources"]).issubset(set(ARM_NAMES)):
        raise WorkerError("worker arm resource set is invalid")
    total_arm_work = 0
    for arm, budget in event["arm_budget"].items():
        if not isinstance(budget, dict) or set(budget) != {"observed_model_work_ns", "charged_model_work_ns", "charge_accepted", "guard_gpu_work_before_seconds", "guard_gpu_work_after_seconds", "guard_recorded_model_work_ns", "duty_formula_break_seconds", "required_break_blocks"}:
            raise WorkerError(f"arm budget schema is invalid: {arm}")
        observed = budget["observed_model_work_ns"]
        charged = budget["charged_model_work_ns"]
        accepted = budget["charge_accepted"]
        guard_recorded = budget["guard_recorded_model_work_ns"]
        if (type(observed) is not int or observed <= 0 or (status == "complete" and observed > int(CONTINUOUS_MODEL_LIMIT_SECONDS * 1e9)) or
                type(charged) is not int or charged < 0 or charged > observed or type(accepted) is not bool or
                (accepted and charged != observed) or (not accepted and charged != 0) or
                type(guard_recorded) is not int or guard_recorded < 0 or guard_recorded > observed or
                (accepted and guard_recorded != observed) or
                not _finite_number(budget["guard_gpu_work_before_seconds"]) or
                not _finite_number(budget["guard_gpu_work_after_seconds"]) or
                budget["guard_gpu_work_before_seconds"] < 0 or
                budget["guard_gpu_work_after_seconds"] < budget["guard_gpu_work_before_seconds"] or
                abs((budget["guard_gpu_work_after_seconds"] - budget["guard_gpu_work_before_seconds"]) - guard_recorded / 1e9) > 1e-6):
            raise WorkerError(f"arm budget duration is invalid: {arm}")
        if not _finite_number(budget["duty_formula_break_seconds"]) or budget["duty_formula_break_seconds"] <= 0:
            raise WorkerError(f"arm duty formula is invalid: {arm}")
        if type(budget["required_break_blocks"]) is not int or budget["required_break_blocks"] < 1:
            raise WorkerError(f"arm break count is invalid: {arm}")
        # This is the theoretical duty-cycle break required by the observed
        # arm duration.  It is not evidence that a pause happened: rejected
        # charges have charged_model_work_ns=0 and do not call required_break.
        expected_break = observed / 1e9 * (1.0 - 0.15) / 0.15
        if abs(float(budget["duty_formula_break_seconds"]) - expected_break) > 1e-6:
            raise WorkerError(f"arm duty formula does not use 0.15: {arm}")
        expected_blocks = max(13, math.ceil(expected_break / POLICY.required_break_s))
        if budget["required_break_blocks"] != expected_blocks:
            raise WorkerError(f"arm break count does not match registered duty formula: {arm}")
        total_arm_work += observed
    if total_arm_work != event["observed_model_work_ns"]:
        raise WorkerError("worker observed arm work does not sum to observed model work")
    if sum(value["charged_model_work_ns"] for value in event["arm_budget"].values()) != event["charged_model_work_ns"]:
        raise WorkerError("worker accepted arm work does not sum to charged model work")
    if sum(value["guard_recorded_model_work_ns"] for value in event["arm_budget"].values()) != event["guard_recorded_model_work_ns"]:
        raise WorkerError("worker guard-recorded work does not sum to top-level evidence")
    if set(event["arm_budget"]) != set(event["arm_resources"]):
        raise WorkerError("worker arm budget/resource records do not match")
    if not set(event["arms"]).issubset(set(event["arm_budget"])):
        raise WorkerError("completed arms are not a subset of terminal arm evidence")
    if status == "complete" and set(event["arm_budget"]) != set(ARM_NAMES):
        raise WorkerError("complete worker event lacks an arm evidence record")
    if any(not value["charge_accepted"] for value in event["arm_budget"].values()) and status != "resource_or_budget_failed":
        raise WorkerError("an unaccepted arm is only valid for a resource/budget terminal event")
    if status == "complete" and any(value["charge_accepted"] is not True for value in event["arm_budget"].values()):
        raise WorkerError("complete worker event contains an unaccepted arm")
    for arm, resources in event["arm_resources"].items():
        if (not isinstance(resources, dict) or
                set(resources) != {"rss_peak_bytes", "mlx_peak_bytes", "swap_after_bytes", "swap_delta_bytes"} or
                any(type(resources[field]) is not int or resources[field] < 0 for field in resources) or
                resources["rss_peak_bytes"] > MAX_RSS_BYTES or
                resources["mlx_peak_bytes"] > MAX_MLX_BYTES or
                resources["swap_delta_bytes"] != 0):
            raise WorkerError(f"arm resource schema invalid: {arm}")
    correctness = event["correctness"]
    if (not isinstance(correctness, dict) or set(correctness) != {"all_arms_text_equal", "all_arms_token_equal", "first_mismatch", "required_arm_count"}
            or correctness["required_arm_count"] != ARM_COUNT):
        raise WorkerError("worker correctness record is invalid")
    for arm in event["arms"]:
        value = event["arms"][arm]
        if not isinstance(value, dict):
            raise WorkerError("arm record is not an object")
        for field in ("decode_forward_ns", "intertoken_ns", "tokens", "text", "observed_model_work_ns", "charged_model_work_ns", "charge_accepted", "arm_wall_ns", "finish_reason", "decode_forwards", "warmup_forwards", "warmup_decode_forward_ns", "warmup_intertoken_ns", "prefill_ns", "ttft_ns", "model_work_ns", "token_rate", "compile_wrapper_ns", "compile_cold_ns", "cache_conversion_ns", "decode_forward_total_ns"):
            if field not in value:
                raise WorkerError(f"arm field is missing: {arm}.{field}")
        compile_shape_valid = (
            (arm == "fixed_compiled" and type(value["compile_wrapper_ns"]) is int and value["compile_wrapper_ns"] > 0 and
             type(value["compile_cold_ns"]) is int and value["compile_cold_ns"] >= 0) or
            (arm != "fixed_compiled" and value["compile_wrapper_ns"] == 0 and value["compile_cold_ns"] is None)
        )
        if arm == "standard_eager" and value["cache_conversion_ns"] != 0:
            compile_shape_valid = False
        if (value["finish_reason"] != "fixed_steps" or value["decode_forwards"] != DECODE_FORWARDS or
                value["warmup_forwards"] != 8 or len(value["warmup_decode_forward_ns"]) != 8 or
                len(value["warmup_intertoken_ns"]) != 8 or
                any(type(x) is not int or x <= 0 for x in value["warmup_decode_forward_ns"]) or
                any(type(x) is not int or x <= 0 for x in value["warmup_intertoken_ns"]) or
                type(value["prefill_ns"]) is not int or value["prefill_ns"] <= 0 or
                type(value["ttft_ns"]) is not int or value["ttft_ns"] <= 0 or
                type(value["model_work_ns"]) is not int or value["model_work_ns"] != value["prefill_ns"] + sum(value["decode_forward_ns"]) or
                not _finite_number(value["token_rate"]) or value["token_rate"] <= 0 or
                type(value["cache_conversion_ns"]) is not int or value["cache_conversion_ns"] < 0 or
                value["decode_forward_total_ns"] != sum(value["decode_forward_ns"]) or not compile_shape_valid or
                len(value["decode_forward_ns"]) != DECODE_FORWARDS or
                len(value["intertoken_ns"]) != DECODE_FORWARDS or
                len(value["tokens"]) != OUTPUT_TOKENS or not isinstance(value["text"], str) or
                not value["text"] or any(type(x) is not int or x <= 0 for x in value["decode_forward_ns"]) or
                any(type(x) is not int or x <= 0 for x in value["intertoken_ns"]) or
                any(type(x) is not int for x in value["tokens"]) or
                type(value["observed_model_work_ns"]) is not int or
                value["observed_model_work_ns"] != event["arm_budget"][arm]["observed_model_work_ns"] or
                type(value["charged_model_work_ns"]) is not int or
                value["charged_model_work_ns"] != event["arm_budget"][arm]["charged_model_work_ns"] or
                value["charge_accepted"] is not True or
                type(value["arm_wall_ns"]) is not int or value["arm_wall_ns"] != value["charged_model_work_ns"] or
                value.get("prompt_sha256") != EXPECTED_PROMPT_SHA256 or
                value.get("prompt_token_sha256") != _sha256_bytes(_canonical(event["prompt_token_ids"])) or
                value.get("rendered_prompt_sha256") != event["rendered_prompt_sha256"] or
                _sha256_bytes(_canonical(value["tokens"])) != value.get("token_sha256") or
                _sha256_bytes(value["text"].encode("utf-8")) != value.get("text_utf8_sha256")):
            raise WorkerError(f"arm timing/token schema invalid: {arm}")
        if event["token_sha256_by_arm"].get(arm) != value["token_sha256"] or event["text_sha256_by_arm"].get(arm) != value["text_utf8_sha256"]:
            raise WorkerError(f"arm hash map mismatch: {arm}")
    if set(event["token_sha256_by_arm"]) != set(event["arms"]) or set(event["text_sha256_by_arm"]) != set(event["arms"]):
        raise WorkerError("worker arm hash maps do not match arms")
    if event["swap_after_bytes"] is not None and event["swap_delta_bytes"] is not None and \
            event["swap_after_bytes"] - event["swap_before_bytes"] != event["swap_delta_bytes"]:
        raise WorkerError("worker swap fields are inconsistent")
    if not isinstance(event["error"], (dict, type(None))):
        raise WorkerError("worker error record is invalid")
    if isinstance(event["error"], dict) and (
            set(event["error"]) != {"type", "message"} or
            not isinstance(event["error"]["type"], str) or
            not isinstance(event["error"]["message"], str)):
        raise WorkerError("worker error record is invalid")
    integrity = event["snapshot_integrity"]
    if (not isinstance(integrity, dict) or
        integrity.get("bound_snapshot_sha256") != identity["snapshot_sha256"] or
        integrity.get("bound_weight_sha256") != identity["weight_sha256"] or
        integrity.get("before_load_stat_manifest") != identity["execution_stat_manifest"] or
        integrity.get("after_load_stat_manifest") != identity["execution_stat_manifest"]):
        raise WorkerError("snapshot integrity proof is invalid")
    try:
        rendered = base64.b64decode(event["rendered_prompt_b64"], validate=True)
    except (ValueError, TypeError) as exc:
        raise WorkerError("rendered prompt is invalid base64") from exc
    if not rendered or _sha256_bytes(rendered) != event["rendered_prompt_sha256"]:
        raise WorkerError("rendered prompt hash is invalid")
    return event


def _run_block(block: int, order: tuple[str, ...], identity: dict[str, Any],
               deadline_monotonic: float | None = None) -> dict[str, Any]:
    power = require_ac_power()
    swap_before = _swap_used_bytes()
    if swap_before is None:
        raise ResourceError("swap unavailable before block")
    started = time.perf_counter_ns()
    process = subprocess.Popen([sys.executable, str(WORKER), "--worker", "--model-key", MODEL_KEY],
                               cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
                               env=_environment(identity, block, order), start_new_session=True)
    if process.stdout is None or process.stderr is None:
        _terminate(process, deadline_monotonic)
        raise WorkerError("worker pipes unavailable")
    stdout_result: dict[str, Any] = {}
    stderr_result: dict[str, Any] = {}
    stdout_thread = threading.Thread(target=_read_capped, args=(process.stdout, process, MAX_EVENT_BYTES, stdout_result, "payload", deadline_monotonic), daemon=True)
    stderr_thread = threading.Thread(target=_read_capped, args=(process.stderr, process, MAX_STDERR_BYTES, stderr_result, "stderr", deadline_monotonic), daemon=True)
    stdout_thread.start(); stderr_thread.start()
    try:
        try:
            remaining = WORKER_TIMEOUT_SECONDS
            if deadline_monotonic is not None:
                remaining = min(remaining, deadline_monotonic - time.monotonic())
            if remaining <= 0:
                _terminate(process, deadline_monotonic)
                raise WorkerError("parent wall deadline exhausted before worker wait")
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as exc:
            _terminate(process, deadline_monotonic)
            raise WorkerError("worker timed out") from exc
        join_remaining = _remaining(deadline_monotonic)
        stdout_thread.join(timeout=join_remaining)
        join_remaining = _remaining(deadline_monotonic)
        stderr_thread.join(timeout=join_remaining)
        if stdout_thread.is_alive() or stderr_thread.is_alive():
            _terminate(process, deadline_monotonic)
            raise WorkerError("worker output reader did not stop")
        if stdout_result.get("overflow") or stderr_result.get("overflow"):
            raise WorkerError("worker output exceeded live limit")
        if stdout_result.get("read_error") or stderr_result.get("read_error"):
            raise WorkerError("worker pipe read failed")
        returncode = process.returncode
        event = _decode_event(stdout_result.get("payload", b""))
        if event.get("event") == "error":
            raise WorkerEventError(
                "worker emitted a terminal error event",
                {
                    "event": "error",
                    "error_type": str(event.get("error_type", "WorkerError"))[:120],
                    "message": str(event.get("message", "worker error"))[:500],
                    "model_key": str(event.get("model_key", MODEL_KEY))[:32],
                    "block": block,
                },
            )
        event = _validate_event(event, process.pid, identity, order, returncode)
        finished = time.perf_counter_ns()
        swap_after = _swap_used_bytes()
        if swap_after is None:
            raise ResourceError("swap unavailable after block")
        process_wall = finished - started
        abort: list[str] = []
        if event["observed_model_work_ns"] > process_wall:
            abort.append("model_work_exceeds_process_wall")
        if event["rss_peak_bytes"] > MAX_RSS_BYTES:
            abort.append("rss_limit_exceeded")
        if event["mlx_peak_bytes"] > MAX_MLX_BYTES:
            abort.append("mlx_limit_exceeded")
        if swap_after - swap_before > 0:
            abort.append("swap_growth")
        event.update({"block": block, "arm_order": list(order), "power_source": power,
                      "process_wall_ns": process_wall, "swap_before_bytes": swap_before,
                      "swap_after_bytes": swap_after,
                      "swap_delta_bytes": swap_after - swap_before,
                      "child_budget": event["budget"],
                      "abort_reason": ";".join(abort) if abort else None,
                      "stderr_tail": stderr_result.get("stderr", b"").decode("utf-8", errors="replace")[-4000:]})
        return event
    finally:
        try: process.stdout.close()
        except OSError: pass
        try: process.stderr.close()
        except OSError: pass


def _median(values: list[float]) -> float:
    if not values or any(not math.isfinite(v) for v in values):
        raise StudyError("invalid empty/non-finite statistic")
    return float(statistics.median(values))


def _mad(values: list[float]) -> float:
    center = _median(values)
    return _median([abs(v - center) for v in values])


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered: raise StudyError("empty percentile")
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position); upper = math.ceil(position)
    if lower == upper: return float(ordered[lower])
    return float(ordered[lower] * (upper - position) + ordered[upper] * (position - lower))


def _bootstrap(values: list[float]) -> dict[str, Any] | None:
    if len(values) != PAIR_COUNT or any(not math.isfinite(v) or v <= 0 for v in values):
        return None
    rng = random.Random(BOOTSTRAP_SEED)
    medians = [_median([values[rng.randrange(PAIR_COUNT)] for _ in range(PAIR_COUNT)])
               for _ in range(BOOTSTRAP_RESAMPLES)]
    return {"lower": _percentile(medians, .025), "upper": _percentile(medians, .975),
            "method": "paired six-block median-ratio bootstrap percentile",
            "resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED,
            "percentiles": {"lower": .025, "upper": .975, "interpolation": "linear"}}


_bootstrap_ci = _bootstrap


def _arm_stats(runs: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    values = [run["arms"][arm] for run in runs if isinstance(run.get("arms"), dict) and arm in run["arms"]]
    complete = len(values) == PAIR_COUNT
    metrics: dict[str, Any] = {}
    for name, getter in (("decode_total", lambda x: sum(x["decode_forward_ns"]) / 1e9),
                         ("intertoken_p50", lambda x: x["intertoken_p50_ns"] / 1e9),
                         ("intertoken_p95", lambda x: x["intertoken_p95_ns"] / 1e9),
                         ("intertoken_p99", lambda x: x["intertoken_p99_ns"] / 1e9)):
        series = [float(getter(x)) for x in values]
        metrics[name] = {"values": series, "median": _median(series) if complete else None,
                         "mad": _mad(series) if complete else None,
                         "p50": _percentile(series, .5) if complete else None,
                         "p95": _percentile(series, .95) if complete else None,
                         "p99": _percentile(series, .99) if complete else None}
    def auxiliary(name: str, getter: Any) -> None:
        series: list[float] = []
        for run in runs:
            arm_value = run.get("arms", {}).get(arm) if isinstance(run.get("arms"), dict) else None
            try:
                value = getter(run, arm_value)
            except (KeyError, TypeError, ValueError):
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                series.append(float(value))
        ready = len(series) == PAIR_COUNT
        metrics[name] = {"values": series, "median": _median(series) if ready else None,
                         "mad": _mad(series) if ready else None,
                         "p50": _percentile(series, .5) if ready else None,
                         "p95": _percentile(series, .95) if ready else None,
                         "p99": _percentile(series, .99) if ready else None}
    auxiliary("ttft_seconds", lambda _run, value: value["ttft_ns"] / 1e9)
    auxiliary("prefill_seconds", lambda _run, value: value["prefill_ns"] / 1e9)
    auxiliary("model_work_seconds", lambda _run, value: value["model_work_ns"] / 1e9)
    auxiliary("arm_wall_seconds", lambda _run, value: value["arm_wall_ns"] / 1e9)
    auxiliary("process_wall_seconds", lambda run, _value: run["process_wall_ns"] / 1e9)
    auxiliary("token_rate", lambda _run, value: value["token_rate"])
    auxiliary("rss_peak_bytes", lambda run, _value: run["rss_peak_bytes"])
    auxiliary("mlx_peak_bytes", lambda run, _value: run["mlx_peak_bytes"])
    auxiliary("swap_delta_bytes", lambda run, _value: run["swap_delta_bytes"])
    return {"arm": arm, "runs": len(values), "metrics": metrics,
            "token_sha256": [x.get("token_sha256") for x in values],
            "text_sha256": [x.get("text_utf8_sha256") for x in values],
            "prompt_sha256": [x.get("prompt_sha256") for x in values],
            "prompt_token_sha256": [x.get("prompt_token_sha256") for x in values],
            "rendered_prompt_sha256": [x.get("rendered_prompt_sha256") for x in values],
            "peak_rss_bytes": max((r.get("rss_peak_bytes", 0) for r in runs), default=None),
            "peak_mlx_bytes": max((r.get("mlx_peak_bytes", 0) for r in runs), default=None),
            "swap_deltas_bytes": [r.get("swap_delta_bytes") for r in runs]}


def _strict_blocks(runs: list[dict[str, Any]]) -> tuple[dict[int, dict[str, Any]] | None, str | None]:
    by_id: dict[int, dict[str, Any]] = {}
    errors: list[str] = []
    for run in runs:
        block = run.get("block") if isinstance(run, dict) else None
        if type(block) is not int or not 1 <= block <= PAIR_COUNT:
            errors.append(f"bad_block={block!r}"); continue
        if block in by_id:
            errors.append(f"duplicate_block={block}"); continue
        by_id[block] = run
    if len(runs) != PAIR_COUNT: errors.append(f"run_count={len(runs)}")
    if set(by_id) != set(range(1, PAIR_COUNT + 1)): errors.append("missing_block")
    for block, run in by_id.items():
        expected = ARM_PERMUTATIONS[block - 1]
        if tuple(run.get("arm_order", ())) != expected: errors.append(f"block_{block}:order")
    return (None, ";".join(errors)) if errors else (by_id, None)


def _determinism(runs: list[dict[str, Any]], arm: str) -> bool:
    values = [r["arms"][arm] for r in runs if isinstance(r.get("arms"), dict) and arm in r["arms"]]
    if len(values) != PAIR_COUNT: return False
    first = values[0]
    return all(v.get("tokens") == first.get("tokens") and v.get("text") == first.get("text") and
               v.get("prompt_sha256") == first.get("prompt_sha256") and
               v.get("prompt_token_sha256") == first.get("prompt_token_sha256") and
               v.get("rendered_prompt_sha256") == first.get("rendered_prompt_sha256")
               for v in values) and all(
                   run.get("prompt_token_ids") == runs[0].get("prompt_token_ids") and
                   run.get("rendered_prompt_b64") == runs[0].get("rendered_prompt_b64") and
                   run.get("rendered_prompt_sha256") == runs[0].get("rendered_prompt_sha256")
                   for run in runs
               )


def _block_correctness(runs: list[dict[str, Any]]) -> bool:
    if len(runs) != PAIR_COUNT:
        return False
    for run in runs:
        arms = run.get("arms", {})
        if (run.get("status") != "complete" or set(arms) != set(ARM_NAMES) or
                run.get("correctness", {}).get("all_arms_token_equal") is not True or
                run.get("correctness", {}).get("all_arms_text_equal") is not True):
            return False
        token_sets = {tuple(arms[name].get("tokens", ())) for name in ARM_NAMES}
        text_sets = {arms[name].get("text") for name in ARM_NAMES}
        if len(token_sets) != 1 or len(text_sets) != 1:
            return False
    return True


def _paired(runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_id, error = _strict_blocks(runs)
    if error or by_id is None:
        return {"complete": False, "error": error or "invalid_blocks", "ratios": {}}
    ratios: dict[str, dict[str, Any]] = {}
    for label, getter in (("decode_total", lambda x: sum(x["decode_forward_ns"])),
                          ("intertoken_p50", lambda x: x["intertoken_p50_ns"]),
                          ("intertoken_p95", lambda x: x["intertoken_p95_ns"]),
                          ("intertoken_p99", lambda x: x["intertoken_p99_ns"])):
        values = [float(getter(by_id[i]["arms"]["fixed_compiled"])) /
                  float(getter(by_id[i]["arms"]["standard_eager"])) for i in range(1, PAIR_COUNT + 1)]
        eager_values = [float(getter(by_id[i]["arms"]["fixed_compiled"])) /
                        float(getter(by_id[i]["arms"]["fixed_eager"])) for i in range(1, PAIR_COUNT + 1)]
        ratios[label] = {"fixed_compiled_div_standard_eager": {"values": values, "median": _median(values), "bootstrap_95_ci": _bootstrap(values)},
                         "fixed_compiled_div_fixed_eager": {"values": eager_values, "median": _median(eager_values), "bootstrap_95_ci": _bootstrap(eager_values)}}
    return {"complete": True, "ratios": ratios,
            "blocks": {str(i): {"standard_pid": by_id[i]["pid"], "arm_order": by_id[i]["arm_order"]} for i in range(1, PAIR_COUNT + 1)}}


def _derived_metrics(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Report arithmetic projections separately from measured timings.

    These are calculated upper bounds, not additional measurements.  The cold
    setup deliberately includes both the compile wrapper and the first cold
    compile call; using the wrapper alone would understate cold cost.
    """
    by_id, error = _strict_blocks(runs)
    if error or by_id is None:
        return {"complete": False, "error": error or "invalid_blocks", "calculated_only": True}
    warmed: list[float] = []
    cold: list[float] = []
    break_even: list[float | None] = []
    cold_setup_values: list[float] = []
    measured_forward_savings: list[float] = []
    for index in range(1, PAIR_COUNT + 1):
        standard_arm = by_id[index]["arms"]["standard_eager"]
        compiled_arm = by_id[index]["arms"]["fixed_compiled"]
        standard_decode = sum(standard_arm["decode_forward_ns"])
        compiled_decode = sum(compiled_arm["decode_forward_ns"])
        standard_prefill = standard_arm.get("prefill_ns")
        compiled_prefill = compiled_arm.get("prefill_ns")
        conversion = compiled_arm.get("cache_conversion_ns")
        compile_wrapper_ns = compiled_arm.get("compile_wrapper_ns")
        compile_cold_ns = compiled_arm.get("compile_cold_ns")
        if (type(standard_decode) is not int or standard_decode <= 0 or
                type(compiled_decode) is not int or compiled_decode <= 0 or
                type(standard_prefill) is not int or standard_prefill < 0 or
                type(compiled_prefill) is not int or compiled_prefill < 0 or
                type(conversion) is not int or conversion < 0 or
                type(compile_wrapper_ns) is not int or compile_wrapper_ns < 0 or
                type(compile_cold_ns) is not int or compile_cold_ns < 0):
            return {"complete": False, "error": f"invalid_block_{index}", "calculated_only": True}
        standard = standard_prefill + standard_decode
        compiled = compiled_prefill + conversion + compiled_decode
        if standard <= 0 or compiled <= 0:
            return {"complete": False, "error": f"invalid_block_{index}", "calculated_only": True}
        warmed.append(compiled / standard)
        cold_setup = compile_wrapper_ns + compile_cold_ns
        cold_setup_values.append(cold_setup / 1e9)
        cold.append((compiled + cold_setup) / standard)
        saving_per_forward = (standard_decode - compiled_decode) / DECODE_FORWARDS
        measured_forward_savings.append(saving_per_forward / 1e9)
        break_even.append(cold_setup / saving_per_forward if saving_per_forward > 0 else None)
    return {"complete": True, "calculated_only": True,
            "method": "calculated upper-bound; warmed=(fixed prefill+conversion+31 decode)/(standard prefill+31 decode); cold adds compile_wrapper+compile_cold; break-even uses measured per-forward decode saving",
            "warmed_decode_ratio_median": _median(warmed),
            "cold_decode_ratio_median": _median(cold),
            "break_even_decode_forwards": break_even,
            "warmed_decode_ratio_values": warmed,
            "cold_decode_ratio_values": cold,
            "cold_setup_seconds": cold_setup_values,
            "measured_decode_saving_per_forward_seconds": measured_forward_savings}


def _decision(*, resource_pass: bool, budget_pass: bool, correctness_pass: bool,
              candidate_runnable: bool, paired: dict[str, Any]) -> str:
    if not resource_pass or not budget_pass: return "resource_or_budget_failed"
    if not correctness_pass: return "correctness_failed"
    if not candidate_runnable: return "candidate_not_runnable"
    if paired.get("complete") is not True: return "no_clear_speedup_baseline_retained"
    # Only total measured decode time is confirmatory.  Inter-token p50/p95/p99
    # remain report-only and must not change the decision.
    fixed_wins = all(
        paired["ratios"]["decode_total"][name]["median"] <= .95 and
        paired["ratios"]["decode_total"][name]["bootstrap_95_ci"]["upper"] < 1.0
        for name in ("fixed_compiled_div_standard_eager", "fixed_compiled_div_fixed_eager")
    )
    standard = (
        paired["ratios"]["decode_total"]["fixed_compiled_div_standard_eager"]["median"] <= .95 and
        paired["ratios"]["decode_total"]["fixed_compiled_div_standard_eager"]["bootstrap_95_ci"]["upper"] < 1.0
    )
    eager = (
        paired["ratios"]["decode_total"]["fixed_compiled_div_fixed_eager"]["median"] <= .95 and
        paired["ratios"]["decode_total"]["fixed_compiled_div_fixed_eager"]["bootstrap_95_ci"]["upper"] < 1.0
    )
    if fixed_wins: return "runtime_compile_wins_exact_scope"
    if eager and not standard: return "compile_gain_no_system_gain"
    if standard and not eager: return "fixed_cache_gain_not_compile_gain"
    clear_regression = any(
        paired["ratios"]["decode_total"][name]["bootstrap_95_ci"]["lower"] > 1.0
        for name in ("fixed_compiled_div_standard_eager", "fixed_compiled_div_fixed_eager")
    )
    return "compile_regression_baseline_retained" if clear_regression else "no_clear_speedup_baseline_retained"


def decision_for(*, resource_pass: bool, budget_pass: bool, correctness_pass: bool,
                 candidate_runnable: bool, paired: dict[str, Any]) -> str:
    """Public testable wrapper for the preregistered decision table."""
    return _decision(resource_pass=resource_pass, budget_pass=budget_pass,
                     correctness_pass=correctness_pass,
                     candidate_runnable=candidate_runnable, paired=paired)


def _provenance(revision: str, dirty: str, power: str, identity: dict[str, Any]) -> dict[str, Any]:
    code_files = {p.relative_to(PROJECT_ROOT).as_posix(): _sha256(p) for p in (Path(__file__), WORKER, PREREGISTRATION)}
    packages: dict[str, str | None] = {}
    for name in ("mlx", "mlx-lm", "numpy", "psutil"):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = None
    return {"git_revision": revision, "git_status": dirty, "git_dirty_state": bool(dirty),
            "code_files_sha256": code_files, "code_sha256": _sha256_bytes(_canonical(code_files)),
            "preregistration_sha256": _sha256(PREREGISTRATION), "prompt_sha256": _worker_module().PROMPT_SHA256,
            "environment_sha256": _environment_fingerprint(),
            "environment": {"python": platform.python_version(), "executable": str(Path(sys.executable).resolve()), "packages": packages, "fixed_worker_environment": WORKER_ENVIRONMENT},
            "hardware": {"machine": platform.machine(), "cpu_brand": _sysctl("machdep.cpu.brand_string"), "memory_bytes": _sysctl("hw.memsize"), "macos": platform.mac_ver()[0], "mlx_default_device": "Device(gpu, 0)"},
            "power_source": power, "model_snapshot": identity, "schedule": [list(x) for x in ARM_PERMUTATIONS]}


def _aggregate_budget(runs: list[dict[str, Any]], parent_wall_seconds: float) -> dict[str, Any]:
    """Aggregate child BudgetGuard evidence without charging or pacing twice."""
    child_budgets = [run.get("budget") for run in runs]
    if any(not isinstance(value, dict) for value in child_budgets):
        return {"valid": False, "error": "missing_child_budget", "parent_wall_seconds": parent_wall_seconds}
    try:
        for run, value in zip(runs, child_budgets):
            _validate_budget_summary(
                value,
                label="child",
                allow_terminal_overrun=run.get("status") == "resource_or_budget_failed",
            )
        gpu_work = sum(float(value["gpu_work_seconds"]) for value in child_budgets)
        max_continuous = max((float(value["max_continuous_gpu_seconds"]) for value in child_budgets), default=0.0)
        duty_values = {float(value["duty_cycle_limit"]) for value in child_budgets}
        valid = (len(duty_values) == 1 and 0.15 in duty_values and
                 gpu_work <= MAX_MODEL_WORK_SECONDS and
                 max_continuous <= CONTINUOUS_MODEL_LIMIT_SECONDS and
                 parent_wall_seconds <= MAX_WALL_SECONDS)
        return {
            "valid": valid,
            "gpu_work_seconds": round(gpu_work, 6),
            "max_continuous_gpu_seconds": round(max_continuous, 6),
            "duty_cycle_limit": 0.15 if len(duty_values) == 1 else None,
            "required_break_seconds": round(sum(float(value["required_break_seconds"]) for value in child_budgets), 6),
            "child_wall_seconds": round(sum(float(value["wall_seconds"]) for value in child_budgets), 6),
            "parent_wall_seconds": round(parent_wall_seconds, 6),
            "gpu_work_limit_seconds": MAX_MODEL_WORK_SECONDS,
            "continuous_gpu_limit_seconds": CONTINUOUS_MODEL_LIMIT_SECONDS,
            "wall_limit_seconds": MAX_WALL_SECONDS,
            "child_budgets": child_budgets,
        }
    except (KeyError, TypeError, ValueError, WorkerError) as exc:
        return {"valid": False, "error": f"invalid_child_budget: {exc}", "parent_wall_seconds": parent_wall_seconds}


def _budget_gate_pass(error: dict[str, str] | None, terminal_status: str | None,
                      budget: dict[str, Any]) -> bool:
    """A terminal budget/resource failure cannot report a passing budget gate."""
    return (
        error is None
        and terminal_status != "resource_or_budget_failed"
        and bool(budget.get("valid"))
    )


def _write_fail_safe(state: dict[str, Any]) -> None:
    try:
        _atomic_result(state)
    except BaseException as exc:
        fallback = {"schema_version": 1, "study_id": STUDY_ID, "run_id": RUN_ID,
                    "formal_claim": False, "decision": "resource_or_budget_failed",
                    "partial_result": True, "error": {"type": type(exc).__name__, "message": str(exc)[:400]},
                    "runs": state.get("runs", []), "worker_events": state.get("worker_events", []), "provenance": state.get("provenance")}
        _atomic_result(fallback)


def execute(run_id: str) -> dict[str, Any]:
    parent_started = time.monotonic()
    hard_deadline = parent_started + MAX_WALL_SECONDS
    worker_deadline = hard_deadline - FINALIZATION_RESERVE_SECONDS
    guard: BudgetGuard = BudgetGuard(POLICY)
    revision, dirty, power, identity, swap_start = _preflight(run_id)
    ATTEMPT_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    _require_private_directory(ATTEMPT_DIR)
    provenance = _provenance(revision, dirty, power, identity)
    _exclusive_json(ATTEMPT_PATH, {"study_id": STUDY_ID, "run_id": run_id,
                                   "formal_claim": False, "started_at_unix_ns": time.time_ns(),
                                   "schedule": [list(x) for x in ARM_PERMUTATIONS], "provenance": provenance}, 0o600)
    state: dict[str, Any] = {"schema_version": 1, "study_id": STUDY_ID, "run_id": run_id,
                             "formal_claim": False, "runs": [], "worker_events": [], "decision": "resource_or_budget_failed",
                             "provenance": provenance, "partial_result": True, "error": None}
    error: dict[str, str] | None = None
    terminal_status: str | None = None
    try:
        for block, order in enumerate(ARM_PERMUTATIONS, start=1):
            guard.check_wall()
            if hard_deadline - time.monotonic() <= FINALIZATION_RESERVE_SECONDS:
                raise ResourceError("reserved finalization budget is exhausted")
            run = _run_block(block, order, identity, worker_deadline)
            state["runs"].append(run)
            if run.get("abort_reason"):
                raise ResourceError(run["abort_reason"])
            if run.get("status") != "complete":
                terminal_status = str(run.get("status"))
                break
            # The child already charged and paced each arm.  The parent only
            # checks aggregate wall time and must not charge or sleep again.
            if run.get("observed_model_work_ns", 0) < 0 or not math.isfinite(run.get("observed_model_work_ns", 0) / 1e9):
                raise BudgetError("invalid model-work duration")
    except WorkerEventError as exc:
        state["worker_events"].append(exc.event)
        error = {"type": type(exc).__name__, "message": str(exc)[:500]}
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc)[:500]}
    finally:
        if guard is not None:
            try: guard.check_wall()
            except BaseException as exc:
                if error is None: error = {"type": type(exc).__name__, "message": str(exc)[:500]}
    swap_end = _swap_used_bytes()
    swap_delta = swap_end - swap_start if swap_end is not None else None
    resource_pass = (len(state["runs"]) == PAIR_COUNT and swap_delta is not None and swap_delta <= 0 and
                     all(r.get("swap_delta_bytes") == 0 and r.get("rss_peak_bytes", MAX_RSS_BYTES + 1) <= MAX_RSS_BYTES and
                         r.get("mlx_peak_bytes", MAX_MLX_BYTES + 1) <= MAX_MLX_BYTES for r in state["runs"]))
    parent_wall_seconds = time.monotonic() - parent_started
    budget = _aggregate_budget(state["runs"], parent_wall_seconds)
    budget_pass = _budget_gate_pass(error, terminal_status, budget)
    paired = _paired(state["runs"])
    correctness = _block_correctness(state["runs"]) and all(_determinism(state["runs"], arm) for arm in ARM_NAMES)
    candidate_runnable = all(r.get("status") == "complete" for r in state["runs"]) and len(state["runs"]) == PAIR_COUNT
    if error is not None or terminal_status in {"error", "resource_or_budget_failed"}:
        decision = "resource_or_budget_failed"
    elif terminal_status == "correctness_failed":
        decision = "correctness_failed"
    elif terminal_status == "candidate_not_runnable":
        decision = "candidate_not_runnable"
    else:
        decision = _decision(resource_pass=resource_pass, budget_pass=budget_pass,
                             correctness_pass=correctness, candidate_runnable=candidate_runnable,
                             paired=paired)
    if error is not None and decision not in ("resource_or_budget_failed",):
        decision = "resource_or_budget_failed"
    summaries = {arm: _arm_stats(state["runs"], arm) for arm in ARM_NAMES} if state["runs"] else {}
    post_identity = None
    if time.monotonic() >= hard_deadline:
        post_identity = {"error": "hard wall deadline exhausted before postflight"}
        resource_pass = False; decision = "resource_or_budget_failed"
    else:
        try:
            snapshot = resolve_local_model_snapshot(MODEL_ID)
            post_identity = _snapshot_identity(snapshot)
            if (post_identity["snapshot_sha256"] != identity["snapshot_sha256"] or
                    post_identity["weight_sha256"] != identity["weight_sha256"]):
                resource_pass = False; decision = "resource_or_budget_failed"
            if time.monotonic() >= hard_deadline:
                post_identity = {"error": "hard wall deadline exhausted during postflight"}
                resource_pass = False; decision = "resource_or_budget_failed"
        except Exception as exc:
            post_identity = {"error": str(exc)[:400]}; resource_pass = False; decision = "resource_or_budget_failed"
    state.update({"completed_at_unix_ns": time.time_ns(), "decision": decision, "error": error,
                  "partial_result": len(state["runs"]) != PAIR_COUNT or error is not None,
                  "worker_events": state["worker_events"],
                  "budget": budget, "resources": {"swap_before_bytes": swap_start, "swap_after_bytes": swap_end, "swap_delta_bytes": swap_delta,
                                                     "max_rss_peak_bytes": max((r.get("rss_peak_bytes", 0) for r in state["runs"]), default=None),
                                                     "max_mlx_peak_bytes": max((r.get("mlx_peak_bytes", 0) for r in state["runs"]), default=None)},
                  "gates": {"all_blocks_completed": len(state["runs"]) == PAIR_COUNT, "resource_pass": resource_pass,
                            "budget_pass": budget_pass, "block_correctness_pass": _block_correctness(state["runs"]),
                            "determinism_pass": all(_determinism(state["runs"], arm) for arm in ARM_NAMES),
                            "candidate_runnable": candidate_runnable, "pairing_pass": paired.get("complete") is True,
                            "snapshot_content_pass": isinstance(post_identity, dict) and "error" not in post_identity},
                  "metrics": {"arms": summaries, "paired": paired,
                               "derived": _derived_metrics(state["runs"]),
                               "runs_completed": len(state["runs"])},
                  "snapshot_postflight": post_identity,
                  "thresholds": {"bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                                  "max_rss_bytes": MAX_RSS_BYTES, "max_mlx_bytes": MAX_MLX_BYTES,
                                  "continuous_model_limit_seconds": CONTINUOUS_MODEL_LIMIT_SECONDS,
                                  "max_model_work_seconds": MAX_MODEL_WORK_SECONDS, "max_wall_seconds": MAX_WALL_SECONDS}})
    _write_fail_safe(state)
    return state


def _self_check() -> int:
    before = _evidence_state()
    _validate_evidence_state(before)
    # The sealed hash is deliberately resolved from the exact preregistration,
    # so this check stays offline and cannot mutate a study.
    assert len(ARM_PERMUTATIONS) == 6 and len(set(ARM_PERMUTATIONS)) == 6
    assert all(sorted(order) == sorted(ARM_NAMES) for order in ARM_PERMUTATIONS)
    assert OUTPUT_TOKENS == 32 and DECODE_FORWARDS == 31
    assert FINALIZATION_RESERVE_SECONDS == 15.0
    assert EXPECTED_PROMPT_SHA256 == "c746eca8644a18fc75673acb9b3dbdf03825cbfba6c76faede5d909cf3d2ea0b"
    assert _bootstrap([1.0] * 6)["lower"] == 1.0
    assert _decision(resource_pass=False, budget_pass=True, correctness_pass=True,
                     candidate_runnable=True, paired={}) == "resource_or_budget_failed"
    assert _decision(resource_pass=True, budget_pass=True, correctness_pass=False,
                     candidate_runnable=True, paired={}) == "correctness_failed"
    assert _decision(resource_pass=True, budget_pass=True, correctness_pass=True,
                     candidate_runnable=False, paired={}) == "candidate_not_runnable"
    primary_only = {
        "complete": True,
        "ratios": {
            "decode_total": {
                "fixed_compiled_div_standard_eager": {"median": .90, "bootstrap_95_ci": {"lower": .80, "upper": .99}},
                "fixed_compiled_div_fixed_eager": {"median": .91, "bootstrap_95_ci": {"lower": .81, "upper": .995}},
            },
            "intertoken_p50": {
                "fixed_compiled_div_standard_eager": {"median": 2.0, "bootstrap_95_ci": {"lower": 1.5, "upper": 3.0}},
                "fixed_compiled_div_fixed_eager": {"median": 2.0, "bootstrap_95_ci": {"lower": 1.5, "upper": 3.0}},
            },
            "intertoken_p95": {}, "intertoken_p99": {},
        },
    }
    assert _decision(resource_pass=True, budget_pass=True, correctness_pass=True,
                     candidate_runnable=True, paired=primary_only) == "runtime_compile_wins_exact_scope"
    regression = {"complete": True, "ratios": {"decode_total": {
        "fixed_compiled_div_standard_eager": {"median": 1.1, "bootstrap_95_ci": {"lower": 1.01, "upper": 1.2}},
        "fixed_compiled_div_fixed_eager": {"median": 1.0, "bootstrap_95_ci": {"lower": .99, "upper": 1.1}},
    }}}
    assert _decision(resource_pass=True, budget_pass=True, correctness_pass=True,
                     candidate_runnable=True, paired=regression) == "compile_regression_baseline_retained"
    unclear = {"complete": True, "ratios": {"decode_total": {
        "fixed_compiled_div_standard_eager": {"median": 1.1, "bootstrap_95_ci": {"lower": .9, "upper": 1.2}},
        "fixed_compiled_div_fixed_eager": {"median": 1.0, "bootstrap_95_ci": {"lower": .8, "upper": 1.1}},
    }}}
    assert _decision(resource_pass=True, budget_pass=True, correctness_pass=True,
                     candidate_runnable=True, paired=unclear) == "no_clear_speedup_baseline_retained"
    budget_sample = {
        "gpu_work_seconds": 1.0, "max_continuous_gpu_seconds": 0.5,
        "cooldown_seconds": 0.0, "required_break_seconds": 4.0,
        "wall_seconds": 5.0, "gpu_work_limit_seconds": 120.0,
        "continuous_gpu_limit_seconds": 6.0, "duty_cycle_limit": 0.15,
        "wall_limit_seconds": 1200.0, "candidate_cooldown_seconds": 0.0,
        "required_break_limit_seconds": 4.0,
    }
    _validate_budget_summary(budget_sample, label="selfcheck", model_work_ns=1_000_000_000)
    assert _aggregate_budget([{"budget": budget_sample}], 5.0)["valid"] is True
    assert _budget_gate_pass(None, "resource_or_budget_failed", {"valid": True}) is False
    assert _budget_gate_pass(None, "candidate_not_runnable", {"valid": True}) is True
    overrun = dict(budget_sample, gpu_work_seconds=121.0)
    _validate_budget_summary(overrun, label="terminal", model_work_ns=121_000_000_000,
                             allow_terminal_overrun=True)
    derived_arms = {}
    for arm in ARM_NAMES:
        derived_arms[arm] = {
            "decode_forward_ns": ([20] if arm == "standard_eager" else [10]) * DECODE_FORWARDS,
            "prefill_ns": 100,
            "cache_conversion_ns": 5 if arm == "fixed_compiled" else 0,
            "compile_wrapper_ns": 20 if arm == "fixed_compiled" else 0,
            "compile_cold_ns": 30 if arm == "fixed_compiled" else 0,
        }
    derived_runs = [{"block": i, "arm_order": list(ARM_PERMUTATIONS[i - 1]), "pid": i,
                     "arms": derived_arms} for i in range(1, PAIR_COUNT + 1)]
    derived = _derived_metrics(derived_runs)
    assert derived["calculated_only"] is True and derived["complete"] is True
    assert derived["cold_setup_seconds"][0] == 50 / 1e9
    assert derived["break_even_decode_forwards"][0] == 5.0
    assert _strict_blocks([])[0] is None
    after = _evidence_state()
    _validate_evidence_state(after)
    assert after == before
    print(json.dumps({"checks": 18, "self_check": "pass"}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="measure_matmul_compile", allow_abbrev=False)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--self-check", action="store_true")
    modes.add_argument("--show", action="store_true")
    parser.add_argument("--run-id", default=RUN_ID)
    args = parser.parse_args(argv)
    if args.self_check:
        try: return _self_check()
        except Exception as exc:
            print(json.dumps({"error": str(exc), "self_check": "failed"}, sort_keys=True)); return 2
    if args.show:
        if not RESULT_PATH.is_file() or RESULT_PATH.is_symlink(): raise SystemExit("result unavailable")
        print(RESULT_PATH.read_text(encoding="utf-8"), end=""); return 0
    if not args.execute:
        print(json.dumps({"state": "not_released", "required_flag": "--execute"}, sort_keys=True)); return 78
    try: report = execute(args.run_id)
    except StudyError as exc:
        print(json.dumps({"error": str(exc), "state": "not_started"}, sort_keys=True)); return 2
    print(json.dumps({"decision": report.get("decision", "resource_or_budget_failed"), "formal_claim": False,
                      "result": RESULT_PATH.relative_to(PROJECT_ROOT).as_posix(), "run_id": RUN_ID,
                      "blocks_completed": len(report.get("runs", []))}, sort_keys=True))
    return 0 if report.get("decision") == "runtime_compile_wins_exact_scope" else 1


if __name__ == "__main__":
    raise SystemExit(main())
