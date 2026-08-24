#!/usr/bin/env python3
"""One-shot, offline-gated six-pair comparison of the local 1B and 4B planners.

The parent owns the schedule, process isolation, evidence, pacing, statistics,
and terminal decision.  A normal invocation never starts a worker; hardware is
reachable only with the fixed ``--execute`` flag and registered run ID.
"""

from __future__ import annotations

import argparse
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

STUDY_ID = "dual-model-evidence-planner-20260824-01"
RUN_ID = "dual-model-evidence-planner-validation-20260824-01"
MODEL_SPECS: dict[str, dict[str, str]] = {
    "1b": {
        "model_id": "mlx-community/gemma-3-1b-it-4bit",
        "revision": "2d44e83dc9e80843d22fb941d3d699a0b1351aa6",
    },
    "4b": {
        "model_id": "mlx-community/gemma-3-4b-it-4bit",
        "revision": "93724907d4ed1745d2fe50baadf3b0b01a65abf2",
    },
}
EXPECTED_CANDIDATE = "persistent_service_qualification"
EXPECTED_CPU_BRAND = "Apple M1 Max"
EXPECTED_MACHINE = "arm64"
EXPECTED_MEMORY_BYTES = 32 * 1024**3
REQUIRED_PACKAGES = {"mlx": "0.32.0", "mlx-lm": "0.31.3"}
FROZEN_PREREGISTRATION_SHA256 = (
    "246357735be8adaf2c275c36eb0d5bcd6fadef8dc267c3a5c612cbae15422cfe"
)
PREREGISTRATION = Path(__file__).with_name("PREREGISTRATION.md")
WORKER = Path(__file__).with_name("worker.py")
RESULT_PATH = Path(__file__).with_name("results.json")
ATTEMPT_DIR = PROJECT_ROOT / ".friday-data" / "dual-model-planner"
ATTEMPT_PATH = ATTEMPT_DIR / "attempt.json"
MAX_EVENT_BYTES = 1_000_000
WORKER_TIMEOUT_SECONDS = 90.0
RUN_COUNT = 12
PAIR_COUNT = 6
MAX_OUTPUT_TOKENS = 32
MAX_RESPONSE_BYTES = 8_192
MAX_MEMORY_BYTES = 5 * 1024**3
WORKER_WATCHDOG_SECONDS = 6.0
BOOTSTRAP_SEED = 20260824
BOOTSTRAP_RESAMPLES = 10_000
PACING_TARGET = 0.10
WORKER_ENVIRONMENT = {
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "PYTHONNOUSERSITE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
UNSAFE_PYTHON_ENVIRONMENT = (
    "PYTHONHOME",
    "PYTHONINSPECT",
    "PYTHONPATH",
    "PYTHONSTARTUP",
)

# This literal is the complete, frozen balanced schedule from the preregistration.
PAIR_SCHEDULE: tuple[tuple[str, str], ...] = (
    ("1b", "4b"),
    ("1b", "4b"),
    ("1b", "4b"),
    ("4b", "1b"),
    ("4b", "1b"),
    ("4b", "1b"),
)

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
    """The study cannot continue safely or reproducibly."""


class WorkerError(StudyError):
    """A child violated the fixed one-event process protocol."""


class ResourceError(StudyError):
    """A hard memory, swap, or target-resource boundary was crossed."""


@lru_cache(maxsize=1)
def _load_worker_module() -> Any:
    specification = importlib.util.spec_from_file_location(
        "dual_model_planner_worker", WORKER
    )
    if specification is None or specification.loader is None:
        raise StudyError("worker module is unavailable")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _token_sha256(tokens: list[int]) -> str:
    return _sha256_bytes(_canonical_json(tokens))


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
    )
    return completed.stdout.decode("utf-8", errors="strict").strip()


def _require_clean_worktree() -> tuple[str, str]:
    revision = _git("rev-parse", "HEAD")
    status = _git(
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        ".",
        ":(exclude)ProjectAtlas",
    )
    if status:
        raise StudyError("project worktree is dirty; commit the study before execution")
    return revision, status


def _swap_used_bytes() -> int | None:
    try:
        import psutil

        value = psutil.swap_memory().used
    except Exception:
        return None
    return value if type(value) is int and value >= 0 else None


def _sysctl(name: str) -> str | None:
    try:
        completed = subprocess.run(
            ["/usr/sbin/sysctl", "-n", name],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.decode("utf-8", errors="replace").strip()
    return value or None


def _exclusive_json(path: Path, value: dict[str, Any], mode: int) -> None:
    payload = _canonical_json(value) + b"\n"
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
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise StudyError("attempt directory cannot be inspected") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise StudyError("attempt directory is unsafe")


def _worker_environment(
    model_key: str, snapshot_identity: dict[str, Any]
) -> dict[str, str]:
    environment = os.environ.copy()
    for name in UNSAFE_PYTHON_ENVIRONMENT:
        environment.pop(name, None)
    environment.update(WORKER_ENVIRONMENT)
    environment["FRIDAY_DUAL_PARENT_PID"] = str(os.getpid())
    environment["FRIDAY_DUAL_RUN_ID"] = RUN_ID
    environment["FRIDAY_DUAL_MODEL_KEY"] = model_key
    snapshot_path = snapshot_identity.get("snapshot_path")
    snapshot_revision = snapshot_identity.get("model_revision")
    snapshot_sha256 = snapshot_identity.get("snapshot_sha256")
    weight_sha256 = snapshot_identity.get("weight_sha256")
    stat_manifest = snapshot_identity.get("execution_stat_manifest")
    if not all(
        isinstance(value, str) and value
        for value in (snapshot_path, snapshot_revision, snapshot_sha256)
    ) or not isinstance(weight_sha256, dict) or not isinstance(stat_manifest, dict):
        raise StudyError("parent snapshot binding is incomplete")
    environment["FRIDAY_DUAL_SNAPSHOT_PATH"] = snapshot_path
    environment["FRIDAY_DUAL_SNAPSHOT_REVISION"] = snapshot_revision
    environment["FRIDAY_DUAL_SNAPSHOT_SHA256"] = snapshot_sha256
    environment["FRIDAY_DUAL_WEIGHT_SHA256"] = _canonical_json(weight_sha256).decode(
        "ascii"
    )
    environment["FRIDAY_DUAL_SNAPSHOT_STAT_MANIFEST"] = _canonical_json(
        stat_manifest
    ).decode("ascii")
    return environment


def _environment_fingerprint() -> str:
    selected = {
        "fixed": WORKER_ENVIRONMENT,
        "removed_python_variables": UNSAFE_PYTHON_ENVIRONMENT,
        "python": str(Path(sys.executable).resolve()),
        "cwd": str(PROJECT_ROOT),
        "platform_machine": platform.machine(),
    }
    return _sha256_bytes(_canonical_json(selected))


def _require_target_environment() -> None:
    if platform.machine() != EXPECTED_MACHINE:
        raise StudyError("machine architecture is not the registered arm64 target")
    if _sysctl("machdep.cpu.brand_string") != EXPECTED_CPU_BRAND:
        raise StudyError("CPU is not the registered Apple M1 Max target")
    if _sysctl("hw.memsize") != str(EXPECTED_MEMORY_BYTES):
        raise StudyError("memory size is not the registered 32 GiB target")
    for package, expected in REQUIRED_PACKAGES.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise StudyError(f"required package is unavailable: {package}") from exc
        if actual != expected:
            raise StudyError(f"required package version changed: {package}")
    import mlx.core as mx

    if str(mx.default_device()) != "Device(gpu, 0)":
        raise StudyError("MLX default device is not the registered GPU")


def _snapshot_identity(snapshot: Any) -> dict[str, Any]:
    """Hash the exact local execution files, resolving safe HF-cache links."""

    root = snapshot.path.resolve(strict=True)
    try:
        repository = (root.parent.parent).resolve(strict=True)
        root.relative_to(repository)
    except (OSError, ValueError) as exc:
        raise StudyError("model snapshot root is outside its local repository") from exc
    if root.parent.name != "snapshots":
        raise StudyError("model snapshot root has an unexpected layout")

    def execution_path(relative: str) -> Path:
        candidate = root / relative
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(repository)
        except (OSError, ValueError) as exc:
            raise StudyError(f"model execution file is outside local repository: {relative}") from exc
        if not resolved.is_file():
            raise StudyError(f"model execution path is not a file: {relative}")
        return resolved

    required = ["config.json", "tokenizer_config.json"]
    for tokenizer_name in ("tokenizer.json", "tokenizer.model"):
        candidate = root / tokenizer_name
        if candidate.is_file() or candidate.is_symlink():
            required.append(tokenizer_name)
            break
    required.extend(snapshot.weight_files)
    files: dict[str, str] = {}
    resolved_files: dict[str, Path] = {}
    for relative in dict.fromkeys(required):
        resolved = execution_path(relative)
        resolved_files[relative] = resolved
        files[relative] = _sha256(resolved)
    if not files:
        raise StudyError("model snapshot has no files")
    weight_hashes = {
        name: files.get(name)
        for name in snapshot.weight_files
    }
    if any(value is None for value in weight_hashes.values()):
        raise StudyError("model weight hash is missing")
    identity = dict(snapshot.report_identity())
    identity.update(
        {
            "snapshot_path": str(root),
            "snapshot_files_sha256": files,
            "snapshot_sha256": _sha256_bytes(_canonical_json(files)),
            "execution_stat_manifest": {
                relative: {
                    "dev": int(metadata.st_dev),
                    "inode": int(metadata.st_ino),
                    "mtime_ns": int(metadata.st_mtime_ns),
                    "path": str(resolved),
                    "size": int(metadata.st_size),
                }
                for relative, resolved in resolved_files.items()
                for metadata in (resolved.stat(),)
            },
            "weight_sha256": weight_hashes,
        }
    )
    return identity


def _provenance(
    revision: str,
    dirty_status: str,
    power_source: str,
    snapshots: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    code_files = {
        path.relative_to(PROJECT_ROOT).as_posix(): _sha256(path)
        for path in (Path(__file__), WORKER, PREREGISTRATION)
    }
    packages: dict[str, str | None] = {}
    for name in ("mlx", "mlx-lm", "numpy", "psutil"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "code_files_sha256": code_files,
        "code_sha256": _sha256_bytes(_canonical_json(code_files)),
        "environment": {
            "executable": str(Path(sys.executable).resolve()),
            "packages": packages,
            "python": platform.python_version(),
            "worker_fixed_environment": dict(WORKER_ENVIRONMENT),
            "worker_removed_environment": list(UNSAFE_PYTHON_ENVIRONMENT),
        },
        "environment_sha256": _environment_fingerprint(),
        "git_dirty_state": bool(dirty_status),
        "git_status": dirty_status,
        "git_revision": revision,
        "hardware": {
            "cpu_brand": _sysctl("machdep.cpu.brand_string"),
            "machine": platform.machine(),
            "macos": platform.mac_ver()[0],
            "memory_bytes": _sysctl("hw.memsize"),
            "model": _sysctl("hw.model"),
            "mlx_default_device": "Device(gpu, 0)",
        },
        "model_snapshots": snapshots,
        "power_source": power_source,
        "preregistration_sha256": FROZEN_PREREGISTRATION_SHA256,
        "prompt_sha256": _load_worker_module().PROMPT_SHA256,
        "schedule": [list(order) for order in PAIR_SCHEDULE],
    }


def _preflight(
    run_id: str,
) -> tuple[str, str, str, dict[str, dict[str, Any]], int]:
    if run_id != RUN_ID:
        raise StudyError("run ID is not registered")
    expected_python = (PROJECT_ROOT / ".venv" / "bin" / "python").resolve(strict=True)
    if Path(sys.executable).resolve() != expected_python:
        raise StudyError("study must use the project virtual environment")
    if _sha256(PREREGISTRATION) != FROZEN_PREREGISTRATION_SHA256:
        raise StudyError("preregistration changed")
    if RESULT_PATH.exists() or RESULT_PATH.is_symlink():
        raise StudyError("result already exists")
    if ATTEMPT_PATH.exists() or ATTEMPT_PATH.is_symlink():
        raise StudyError("the one-shot hardware attempt was already started")
    revision, dirty_status = _require_clean_worktree()
    _require_target_environment()
    power_source = require_ac_power()
    snapshots: dict[str, dict[str, Any]] = {}
    for key, spec in MODEL_SPECS.items():
        snapshot = resolve_local_model_snapshot(spec["model_id"])
        if snapshot.revision != spec["revision"]:
            raise StudyError(f"registered model revision changed: {key}")
        snapshots[key] = _snapshot_identity(snapshot)
    swap_before = _swap_used_bytes()
    if swap_before is None:
        raise StudyError("swap usage is unavailable")
    return revision, dirty_status, power_source, snapshots, swap_before


def _decode_event(payload: bytes) -> dict[str, Any]:
    if not payload or len(payload) > MAX_EVENT_BYTES:
        raise WorkerError("worker output size is invalid")
    lines = payload.splitlines()
    if len(lines) != 1 or not lines[0]:
        raise WorkerError("worker must emit exactly one JSON event")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate worker event field")
            value[key] = item
        return value

    try:
        value = json.loads(
            lines[0].decode("utf-8", errors="strict"),
            object_pairs_hook=unique_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {item}")
            ),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise WorkerError("worker emitted invalid JSON") from exc
    if not isinstance(value, dict):
        raise WorkerError("worker event is not an object")
    if value.get("event") == "error":
        raise WorkerError(
            f"worker failed: {value.get('error_type', 'unknown')}: "
            f"{str(value.get('message', ''))[:300]}"
        )
    return value


def _stderr_tail(stream: Any) -> str:
    try:
        stream.flush()
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(max(0, size - 4_000))
        return stream.read().decode("utf-8", errors="replace")[-4_000:]
    except Exception:
        return ""


def _read_bounded_stdout(
    stream: Any, process: subprocess.Popen[bytes], result: dict[str, Any]
) -> None:
    """Read the event pipe with a live byte ceiling.

    ``communicate()`` only lets the parent notice an oversized event after the
    child has finished.  This reader stops at ``MAX_EVENT_BYTES + 1`` and kills
    the child process group while it is still writing.
    """

    payload = bytearray()
    overflow = False
    read_error: str | None = None
    try:
        while True:
            remaining = MAX_EVENT_BYTES + 1 - len(payload)
            if remaining <= 0:
                overflow = True
                _terminate_process(process)
                break
            chunk = stream.read(min(64 * 1024, remaining))
            if not chunk:
                break
            payload.extend(chunk)
            if len(payload) > MAX_EVENT_BYTES:
                overflow = True
                _terminate_process(process)
                break
    except Exception as exc:
        read_error = f"{type(exc).__name__}: {str(exc)[:300]}"
        _terminate_process(process)
    result.update(
        {
            "overflow": overflow,
            "payload": bytes(payload),
            "read_error": read_error,
        }
    )


def _validate_event(
    event: dict[str, Any],
    process_pid: int,
    model_key: str,
    snapshot_identity: dict[str, Any],
) -> dict[str, Any]:
    worker = _load_worker_module()
    spec = MODEL_SPECS[model_key]
    expected_fields = {
        "device",
        "event",
        "finish_reason",
        "load_count",
        "max_output_tokens",
        "model_key",
        "model_id",
        "model_load_ns",
        "model_work_ns",
        "mlx_peak_bytes",
        "output_tokens",
        "pid",
        "prefill_step_size",
        "prompt_sha256",
        "prompt_token_ids",
        "prompt_tokens",
        "rendered_prompt_b64",
        "rendered_prompt_sha256",
        "rss_peak_bytes",
        "sampler_temperature",
        "snapshot_integrity",
        "snapshot_path",
        "snapshot_sha256",
        "snapshot_revision",
        "text",
        "token_rate",
        "tokens",
        "ttft_ns",
        "weight_sha256",
        "worker_watchdog_seconds",
    }
    tokens = event.get("tokens")
    prompt_ids = event.get("prompt_token_ids")
    text = event.get("text")
    number_fields = (
        "model_load_ns",
        "model_work_ns",
        "ttft_ns",
        "rss_peak_bytes",
        "mlx_peak_bytes",
        "prompt_tokens",
        "output_tokens",
    )
    if (
        set(event) != expected_fields
        or event.get("event") != "complete"
        or event.get("device") != "Device(gpu, 0)"
        or event.get("model_key") != model_key
        or event.get("model_id") != spec["model_id"]
        or event.get("snapshot_revision") != spec["revision"]
        or event.get("snapshot_path") != snapshot_identity.get("snapshot_path")
        or event.get("snapshot_sha256") != snapshot_identity.get("snapshot_sha256")
        or event.get("weight_sha256") != snapshot_identity.get("weight_sha256")
        or event.get("prompt_sha256") != worker.PROMPT_SHA256
        or event.get("load_count") != 1
        or event.get("max_output_tokens") != MAX_OUTPUT_TOKENS
        or event.get("prefill_step_size") != worker.PREFILL_STEP_SIZE
        or event.get("sampler_temperature") != 0.0
        or event.get("pid") != process_pid
        or any(type(event.get(field)) is not int or event[field] <= 0 for field in number_fields)
        or event["ttft_ns"] > event["model_work_ns"]
        or not isinstance(prompt_ids, list)
        or not prompt_ids
        or any(type(token) is not int for token in prompt_ids)
        or event["prompt_tokens"] != len(prompt_ids)
        or not isinstance(event.get("rendered_prompt_b64"), str)
        or len(event["rendered_prompt_b64"]) > 256_000
        or not isinstance(event.get("rendered_prompt_sha256"), str)
        or len(event["rendered_prompt_sha256"]) != 64
        or not isinstance(tokens, list)
        or not 1 <= len(tokens) <= MAX_OUTPUT_TOKENS
        or event["output_tokens"] != len(tokens)
        or any(type(token) is not int for token in tokens)
        or not isinstance(text, str)
        or not text
        or len(text.encode("utf-8")) > MAX_RESPONSE_BYTES
        or not isinstance(event.get("finish_reason"), str)
        or event.get("finish_reason") not in {"stop", "length"}
        or type(event.get("token_rate")) is not float
        or not math.isfinite(event["token_rate"])
        or event["token_rate"] <= 0
        or event.get("worker_watchdog_seconds") != WORKER_WATCHDOG_SECONDS
        or not isinstance(event.get("snapshot_integrity"), dict)
    ):
        raise WorkerError("worker completion event is invalid")

    integrity = event["snapshot_integrity"]
    if set(integrity) != {
        "bound_snapshot_sha256",
        "bound_weight_sha256",
        "before_load_stat_manifest",
        "after_load_stat_manifest",
    } or any(
        integrity.get(field) != snapshot_identity.get("snapshot_sha256")
        for field in ("bound_snapshot_sha256",)
    ) or any(
        integrity.get(field) != snapshot_identity.get("weight_sha256")
        for field in ("bound_weight_sha256",)
    ) or any(
        integrity.get(field) != snapshot_identity.get("execution_stat_manifest")
        for field in ("before_load_stat_manifest", "after_load_stat_manifest")
    ):
        raise WorkerError("worker snapshot integrity proof is invalid")

    import base64

    try:
        rendered_prompt_bytes = base64.b64decode(
            event["rendered_prompt_b64"], validate=True
        )
    except (ValueError, TypeError) as exc:
        raise WorkerError("rendered chat-template prompt is not valid base64") from exc
    if (
        not rendered_prompt_bytes
        or hashlib.sha256(rendered_prompt_bytes).hexdigest()
        != event["rendered_prompt_sha256"]
    ):
        raise WorkerError("rendered chat-template prompt hash is invalid")

    try:
        structural_candidate = worker.parse_structure(text)
        structural = {
            "accepted": True,
            "candidate_id": structural_candidate,
            "error": None,
        }
    except worker.WorkerError as exc:
        structural_candidate = None
        structural = {
            "accepted": False,
            "candidate_id": None,
            "error": str(exc)[:300],
        }
    try:
        contract_candidate = worker.parse_choice(text)
        contract = {
            "accepted": True,
            "candidate_id": contract_candidate,
            "error": None,
        }
    except worker.WorkerError as exc:
        contract = {
            "accepted": False,
            "candidate_id": None,
            "error": str(exc)[:300],
        }
    return {
        **event,
        "candidate_id": structural_candidate,
        "model_snapshot": snapshot_identity,
        "parser": {
            "contract_ok": contract["accepted"],
            "contract_error": contract["error"],
            "contract_candidate_id": contract["candidate_id"],
            "structural_ok": structural["accepted"],
            "structural_error": structural["error"],
            "structural_candidate_id": structural["candidate_id"],
        },
        "text_utf8_sha256": _sha256_bytes(text.encode("utf-8")),
        "token_sha256": _token_sha256(tokens),
    }


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        try:
            process.terminate()
        except OSError:
            pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            try:
                process.kill()
            except OSError:
                pass
        process.wait(timeout=2)


def _run_worker(
    *, pair_id: int, schedule_position: int, model_key: str, snapshot_identity: dict[str, Any]
) -> dict[str, Any]:
    try:
        power_source = require_ac_power()
    except SystemExit as exc:
        raise ResourceError(str(exc)) from exc
    swap_before = _swap_used_bytes()
    if swap_before is None:
        raise ResourceError("swap usage unavailable before worker")
    started_ns = time.perf_counter_ns()
    stderr = tempfile.TemporaryFile(mode="w+b")
    process = subprocess.Popen(
        [sys.executable, str(WORKER), "--worker", "--model-key", model_key],
        cwd=PROJECT_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=stderr,
        bufsize=0,
        env=_worker_environment(model_key, snapshot_identity),
        start_new_session=True,
    )
    if process.stdout is None:
        stderr.close()
        raise WorkerError("worker stdout is unavailable")
    reader_result: dict[str, Any] = {}
    reader = threading.Thread(
        target=_read_bounded_stdout,
        args=(process.stdout, process, reader_result),
        name="dual-planner-stdout-reader",
        daemon=True,
    )
    reader.start()
    try:
        try:
            process.wait(timeout=WORKER_TIMEOUT_SECONDS)
            completed_ns = time.perf_counter_ns()
        except subprocess.TimeoutExpired as exc:
            _terminate_process(process)
            completed_ns = time.perf_counter_ns()
            raise WorkerError("worker timed out") from exc
        reader.join(timeout=3)
        if reader.is_alive():
            _terminate_process(process)
            try:
                process.stdout.close()
            except OSError:
                pass
            reader.join(timeout=1)
            raise WorkerError("worker stdout reader did not stop")
        payload = reader_result.get("payload", b"")
        if reader_result.get("overflow"):
            raise WorkerError("worker output exceeded the live size limit")
        if reader_result.get("read_error"):
            raise WorkerError(f"worker stdout read failed: {reader_result['read_error']}")
        if process.returncode != 0:
            detail = ""
            try:
                _decode_event(payload)
            except WorkerError as inner:
                detail = str(inner)
            tail = _stderr_tail(stderr)
            message = "; ".join(value for value in (detail, tail) if value)
            raise WorkerError(f"worker exited with {process.returncode}: {message[:500]}")
        value = _validate_event(
            _decode_event(payload), process.pid, model_key, snapshot_identity
        )
        process_wall_ns = completed_ns - started_ns
        abort_reasons: list[str] = []
        if value["model_work_ns"] > process_wall_ns:
            abort_reasons.append("model_duration_exceeds_process_wall")
        if value["model_work_ns"] > int(WORKER_WATCHDOG_SECONDS * 1_000_000_000):
            abort_reasons.append("continuous_model_work_budget_exceeded")
        swap_after = _swap_used_bytes()
        if swap_after is None:
            abort_reasons.append("swap_usage_unavailable_after_worker")
        value.update(
            {
                "abort_reason": ";".join(abort_reasons) if abort_reasons else None,
                "pair_id": pair_id,
                "power_source": power_source,
                "process_wall_ns": process_wall_ns,
                "schedule_position": schedule_position,
                "swap_after_bytes": swap_after,
                "swap_before_bytes": swap_before,
                "swap_delta_bytes": (
                    swap_after - swap_before if swap_after is not None else None
                ),
            }
        )
        return value
    finally:
        try:
            process.stdout.close()
        except OSError:
            pass
        stderr.close()


def _record_gpu(guard: BudgetGuard, model_work_ns: int) -> float:
    """Charge only the already-stopped worker duration; never pause here."""

    seconds = model_work_ns / 1_000_000_000
    if not math.isfinite(seconds) or seconds <= 0:
        raise BudgetError("worker model duration is invalid")
    guard.record_gpu(seconds)
    return seconds


def _required_breaks(guard: BudgetGuard, seconds: float) -> None:
    """Apply registered cooling only after immediate resource checks pass."""

    required = seconds * (1.0 - PACING_TARGET) / PACING_TARGET
    for _ in range(max(1, math.ceil(required / POLICY.required_break_s))):
        guard.required_break()


def _mark_abort(run: dict[str, Any], reason: str) -> None:
    existing = run.get("abort_reason")
    run["abort_reason"] = f"{existing};{reason}" if isinstance(existing, str) else reason


def _median(values: list[float]) -> float:
    if not values or any(not math.isfinite(value) for value in values):
        raise StudyError("cannot summarize empty or non-finite measurements")
    return float(statistics.median(values))


def _mad(values: list[float]) -> float:
    center = _median(values)
    return _median([abs(value - center) for value in values])


def _strict_pair_map(
    runs: list[dict[str, Any]],
) -> tuple[dict[int, dict[str, dict[str, Any]]] | None, dict[str, Any] | None]:
    """Build the six pairs without ever overwriting a duplicate record."""

    by_pair: dict[int, dict[str, dict[str, Any]]] = {}
    errors: list[str] = []
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            errors.append(f"run_{index}:not_object")
            continue
        pair_id = run.get("pair_id")
        model_key = run.get("model_key")
        if type(pair_id) is not int or not 1 <= pair_id <= PAIR_COUNT:
            errors.append(f"run_{index}:unexpected_pair_id={pair_id!r}")
            continue
        if model_key not in MODEL_SPECS:
            errors.append(f"run_{index}:unexpected_model={model_key!r}")
            continue
        pair = by_pair.setdefault(pair_id, {})
        if model_key in pair:
            errors.append(f"pair_{pair_id}:duplicate_model={model_key}")
            continue
        pair[model_key] = run

    expected_ids = set(range(1, PAIR_COUNT + 1))
    observed_ids = set(by_pair)
    missing = sorted(expected_ids - observed_ids)
    unexpected = sorted(observed_ids - expected_ids)
    if missing:
        errors.append(f"missing_pair_ids={missing}")
    if unexpected:
        errors.append(f"unexpected_pair_ids={unexpected}")
    for pair_id in sorted(observed_ids & expected_ids):
        models = set(by_pair[pair_id])
        if models != set(MODEL_SPECS):
            errors.append(f"pair_{pair_id}:models={sorted(models)!r}")
    if len(runs) != RUN_COUNT:
        errors.append(f"run_count={len(runs)}")
    if errors:
        return None, {
            "error": "incomplete_or_invalid_pairing: " + "; ".join(errors[:12]),
            "observed_pair_ids": sorted(observed_ids),
            "observed_run_count": len(runs),
        }
    return by_pair, None


def _determinism_gate(runs: list[dict[str, Any]]) -> bool:
    if len(runs) != 6:
        return False
    first = runs[0]
    return bool(
        all(run.get("tokens") == first.get("tokens") for run in runs)
        and all(run.get("text") == first.get("text") for run in runs)
        and all(run.get("finish_reason") == first.get("finish_reason") for run in runs)
        and all(run.get("prompt_token_ids") == first.get("prompt_token_ids") for run in runs)
        and all(run.get("rendered_prompt_b64") == first.get("rendered_prompt_b64") for run in runs)
        and all(run.get("rendered_prompt_sha256") == first.get("rendered_prompt_sha256") for run in runs)
    )


def _prompt_identity_gate(runs: list[dict[str, Any]]) -> bool:
    """Require one rendered chat-template byte sequence across both models."""

    if len(runs) != RUN_COUNT:
        return False
    first = runs[0]
    return bool(
        all(run.get("prompt_sha256") == first.get("prompt_sha256") for run in runs)
        and all(
            run.get("rendered_prompt_b64") == first.get("rendered_prompt_b64")
            for run in runs
        )
        and all(
            run.get("rendered_prompt_sha256") == first.get("rendered_prompt_sha256")
            for run in runs
        )
    )


def _fresh_process_gate(runs: list[dict[str, Any]]) -> bool:
    """Require twelve distinct child PIDs for the twelve fresh processes."""

    pids = [run.get("pid") for run in runs]
    return bool(
        len(runs) == RUN_COUNT
        and all(type(pid) is int and pid > 0 for pid in pids)
        and len(pids) == len(set(pids))
    )


def _identity_gate(model_key: str, runs: list[dict[str, Any]]) -> bool:
    worker = _load_worker_module()
    spec = MODEL_SPECS[model_key]
    pids = [run.get("pid") for run in runs]
    if len(runs) != 6 or not all(type(pid) is int and pid > 0 for pid in pids):
        return False
    return bool(
        len(pids) == len(set(pids))
        and all(run.get("model_key") == model_key for run in runs)
        and all(run.get("model_id") == spec["model_id"] for run in runs)
        and all(run.get("snapshot_revision") == spec["revision"] for run in runs)
        and all(run.get("load_count") == 1 for run in runs)
        and all(run.get("prompt_sha256") == worker.PROMPT_SHA256 for run in runs)
        and all(run.get("rendered_prompt_b64") == runs[0].get("rendered_prompt_b64") for run in runs)
        and all(run.get("rendered_prompt_sha256") == runs[0].get("rendered_prompt_sha256") for run in runs)
        and all(run.get("device") == "Device(gpu, 0)" for run in runs)
    )


def _contract_gate(runs: list[dict[str, Any]]) -> bool:
    allowed = _load_worker_module().ALLOWED_CANDIDATES
    return bool(
        len(runs) == 6
        and all(run.get("candidate_id") in allowed for run in runs)
        and all(run.get("parser", {}).get("contract_ok") is True for run in runs)
    )


def _priority_gate(runs: list[dict[str, Any]]) -> bool:
    return bool(
        len(runs) == 6
        and all(run.get("candidate_id") == EXPECTED_CANDIDATE for run in runs)
    )


def _resource_gate(runs: list[dict[str, Any]], study_swap_delta: int | None) -> bool:
    if len(runs) != RUN_COUNT or study_swap_delta is None or study_swap_delta > 0:
        return False
    for run in runs:
        if run.get("abort_reason") is not None:
            return False
        for field in ("rss_peak_bytes", "mlx_peak_bytes"):
            if type(run.get(field)) is not int or not 0 < run[field] <= MAX_MEMORY_BYTES:
                return False
        if type(run.get("swap_delta_bytes")) is not int or run["swap_delta_bytes"] > 0:
            return False
    return True


def _model_summary(model_key: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    def seconds(field: str) -> list[float]:
        return [run[field] / 1_000_000_000 for run in runs]

    contract_count = sum(
        1
        for run in runs
        if run.get("parser", {}).get("contract_ok") is True
        and run.get("candidate_id") in _load_worker_module().ALLOWED_CANDIDATES
    )
    priority_count = sum(
        1 for run in runs if run.get("candidate_id") == EXPECTED_CANDIDATE
    )
    complete = len(runs) == 6
    metrics: dict[str, Any] = {}
    for label, field in (
        ("ttft_seconds", "ttft_ns"),
        ("model_work_seconds", "model_work_ns"),
        ("process_wall_seconds", "process_wall_ns"),
    ):
        if complete:
            values = seconds(field)
            metrics[label] = {"mad": _mad(values), "median": _median(values), "values": values}
        else:
            metrics[label] = {"mad": None, "median": None, "values": []}
    rss = [run.get("rss_peak_bytes") for run in runs if type(run.get("rss_peak_bytes")) is int]
    mlx = [run.get("mlx_peak_bytes") for run in runs if type(run.get("mlx_peak_bytes")) is int]
    swap = [run.get("swap_delta_bytes") for run in runs if type(run.get("swap_delta_bytes")) is int]
    return {
        "candidate_ids": [run.get("candidate_id") for run in runs],
        "contract_pass": contract_count == 6,
        "contract_successes": contract_count,
        "correctness_pass": bool(
            complete and _identity_gate(model_key, runs) and _determinism_gate(runs)
        ),
        "deterministic": _determinism_gate(runs),
        "identity_pass": _identity_gate(model_key, runs),
        "metrics": metrics,
        "model_key": model_key,
        "peak_mlx_bytes": max(mlx) if mlx else None,
        "peak_rss_bytes": max(rss) if rss else None,
        "priority_pass": priority_count == 6,
        "priority_successes": priority_count,
        "runs_completed": len(runs),
        "swap_deltas_bytes": swap,
        "functional_pass": bool(
            complete
            and _identity_gate(model_key, runs)
            and _determinism_gate(runs)
            and contract_count == 6
            and priority_count == 6
        ),
    }


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise StudyError("cannot calculate percentile of empty data")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap_ci(ratios: list[float]) -> dict[str, Any] | None:
    if len(ratios) != PAIR_COUNT or any(not math.isfinite(value) for value in ratios):
        return None
    rng = random.Random(BOOTSTRAP_SEED)
    medians: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sample = [ratios[rng.randrange(PAIR_COUNT)] for _ in range(PAIR_COUNT)]
        medians.append(_median(sample))
    return {
        "percentiles": {"lower": 0.025, "upper": 0.975, "interpolation": "linear"},
        "lower": _percentile(medians, 0.025),
        "upper": _percentile(medians, 0.975),
        "method": "paired six-pair median-ratio bootstrap percentile",
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
    }


def _pairwise(runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_pair, pairing_error = _strict_pair_map(runs)
    if pairing_error is not None or by_pair is None:
        return {
            "complete": False,
            "pair_ids": [],
            "pairs": {},
            "ratios_1b_div_4b": {},
            **(pairing_error or {"error": "incomplete_or_invalid_pairing"}),
        }
    ratios: dict[str, list[float]] = {}
    for label, field in (
        ("ttft", "ttft_ns"),
        ("model_work", "model_work_ns"),
        ("process_wall", "process_wall_ns"),
        ("token_rate", "token_rate"),
    ):
        values: list[float] = []
        for pair_id in range(1, PAIR_COUNT + 1):
            numerator = by_pair[pair_id]["1b"].get(field)
            denominator = by_pair[pair_id]["4b"].get(field)
            if (
                type(numerator) not in (int, float)
                or type(denominator) not in (int, float)
                or not math.isfinite(float(numerator))
                or not math.isfinite(float(denominator))
                or numerator <= 0
                or denominator <= 0
            ):
                return {
                    "complete": False,
                    "error": f"pairwise metric is invalid: {label}",
                    "pair_ids": [],
                    "pairs": {},
                    "ratios_1b_div_4b": {},
                }
            values.append(float(numerator) / float(denominator))
        ratios[label] = values
        if any(
            not math.isfinite(value) or value <= 0 for value in ratios[label]
        ):
            return {
                "complete": False,
                "error": f"pairwise metric is invalid: {label}",
                "pair_ids": [],
                "pairs": {},
                "ratios_1b_div_4b": {},
            }
    return {
        "complete": True,
        "pair_ids": list(range(1, PAIR_COUNT + 1)),
        "pairs": {
            str(pair_id): {
                "1b_pid": by_pair[pair_id]["1b"]["pid"],
                "4b_pid": by_pair[pair_id]["4b"]["pid"],
            }
            for pair_id in range(1, PAIR_COUNT + 1)
        },
        "ratios_1b_div_4b": {
            label: {
                "bootstrap_95_ci": _bootstrap_ci(values),
                "median": _median(values),
                "statistic": "median of 1B/4B pair ratios",
                "values": values,
            }
            for label, values in ratios.items()
        },
    }


def _cross_model_text(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Report concrete cross-model byte equality without making it a gate."""

    by_pair, pairing_error = _strict_pair_map(runs)
    if pairing_error is not None or by_pair is None:
        return {
            "complete": False,
            "exact_text_equal_count": 0,
            "exact_text_equal_total": "0/6",
            "pairs": [],
            "informational_only": True,
            **(pairing_error or {"error": "incomplete_or_invalid_pairing"}),
        }
    rows: list[dict[str, Any]] = []
    for pair_id in range(1, PAIR_COUNT + 1):
        pair = by_pair.get(pair_id, {})
        text_1b = pair["1b"].get("text")
        text_4b = pair["4b"].get("text")
        if not isinstance(text_1b, str) or not isinstance(text_4b, str):
            return {
                "complete": False,
                "error": f"pair_{pair_id}:missing_decoded_text",
                "exact_text_equal_count": 0,
                "exact_text_equal_total": "0/6",
                "pairs": rows,
                "informational_only": True,
            }
        rows.append(
            {
                "4b_text_utf8_sha256": _sha256_bytes(text_4b.encode("utf-8")),
                "1b_text_utf8_sha256": _sha256_bytes(text_1b.encode("utf-8")),
                "exact_text_equal": text_1b.encode("utf-8") == text_4b.encode("utf-8"),
                "pair_id": pair_id,
                "4b_token_sha256": pair["4b"].get("token_sha256"),
                "1b_token_sha256": pair["1b"].get("token_sha256"),
            }
        )
    equal_count = sum(1 for row in rows if row["exact_text_equal"])
    return {
        "complete": len(rows) == PAIR_COUNT,
        "exact_text_equal_count": equal_count,
        "exact_text_equal_total": f"{equal_count}/{PAIR_COUNT}",
        "pairs": rows,
        "informational_only": True,
    }


def decision_for(
    *,
    one_b_pass: bool,
    four_b_pass: bool,
    pairwise: dict[str, Any] | None,
    one_b_peak_rss: int | None,
    four_b_peak_rss: int | None,
    correctness_failure: bool = False,
    terminal_failure: bool = False,
) -> str:
    """Apply the immutable cycle-15 decision table."""

    if terminal_failure:
        return "resource_or_budget_failed"
    if pairwise is not None and pairwise.get("complete") is not True:
        return "resource_or_budget_failed"
    if correctness_failure:
        return "correctness_failed"
    if one_b_pass and not four_b_pass:
        return "planner_1b_qualified_exact_case"
    if four_b_pass and not one_b_pass:
        return "planner_4b_qualified_exact_case"
    if not one_b_pass and not four_b_pass:
        return "no_planner_qualified"
    ratio = None
    if pairwise and pairwise.get("complete"):
        ratio = pairwise.get("ratios_1b_div_4b", {}).get("process_wall", {}).get("median")
    memory_reduction = None
    if one_b_peak_rss is not None and four_b_peak_rss is not None and four_b_peak_rss > 0:
        memory_reduction = 1.0 - one_b_peak_rss / four_b_peak_rss
    if ratio is not None and ratio <= 1.05 and memory_reduction is not None and memory_reduction >= 0.25:
        return "both_qualified_1b_preferred"
    return "both_qualified_no_automatic_preference"


def _exception_record(exc: BaseException) -> dict[str, str]:
    return {"message": str(exc)[:500], "type": type(exc).__name__}


def _empty_model_summary(model_key: str) -> dict[str, Any]:
    return {
        "candidate_ids": [],
        "contract_pass": False,
        "contract_successes": 0,
        "correctness_pass": False,
        "deterministic": False,
        "identity_pass": False,
        "metrics": {
            label: {"mad": None, "median": None, "values": []}
            for label in (
                "ttft_seconds",
                "model_work_seconds",
                "process_wall_seconds",
            )
        },
        "model_key": model_key,
        "peak_mlx_bytes": None,
        "peak_rss_bytes": None,
        "priority_pass": False,
        "priority_successes": 0,
        "runs_completed": 0,
        "swap_deltas_bytes": [],
        "functional_pass": False,
    }


def _guard_summary(guard: BudgetGuard | None) -> tuple[dict[str, Any], str | None]:
    if guard is None:
        return {}, "budget guard was not initialized"
    try:
        return guard.summary(), None
    except BaseException as exc:
        return {}, f"guard.summary failed: {type(exc).__name__}: {str(exc)[:300]}"


def _write_result_fail_safe(state: dict[str, Any]) -> None:
    """Write a complete or minimal atomic result after the start marker."""

    try:
        _atomic_result(state)
        return
    except BaseException as first:
        temporary = RESULT_PATH.with_name(f".{RESULT_PATH.name}.tmp-{os.getpid()}")
        try:
            if temporary.exists() and not temporary.is_symlink():
                temporary.unlink()
        except OSError:
            pass
        fallback = {
            "completed_at_unix_ns": time.time_ns(),
            "decision": "resource_or_budget_failed",
            "error": {
                "message": f"result finalization failed: {str(first)[:300]}",
                "type": "FinalizationError",
            },
            "formal_claim": False,
            "partial_result": True,
            "provenance": state.get("provenance"),
            "run_id": state.get("run_id"),
            "runs": state.get("runs", []),
            "schema_version": state.get("schema_version", 1),
            "study_id": state.get("study_id", STUDY_ID),
        }
        _atomic_result(fallback)


def execute(run_id: str) -> dict[str, Any]:
    revision, dirty_status, power_source, snapshots, swap_start = _preflight(run_id)
    ATTEMPT_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    _require_private_directory(ATTEMPT_DIR)
    provenance = _provenance(revision, dirty_status, power_source, snapshots)
    _exclusive_json(
        ATTEMPT_PATH,
        {
            "formal_claim": False,
            "provenance": provenance,
            "run_id": run_id,
            "schedule": [list(order) for order in PAIR_SCHEDULE],
            "started_at_unix_ns": time.time_ns(),
            "study_id": STUDY_ID,
        },
        0o600,
    )

    state: dict[str, Any] = {
        "decision": "hardware_run_failed",
        "formal_claim": False,
        "provenance": provenance,
        "run_id": run_id,
        "runs": [],
        "schema_version": 1,
        "study_id": STUDY_ID,
    }
    error: dict[str, str] | None = None
    guard: BudgetGuard | None = None
    try:
        guard = BudgetGuard(POLICY)
        guard.before_candidate()
        position = 0
        for pair_index, order in enumerate(PAIR_SCHEDULE, start=1):
            for model_key in order:
                position += 1
                value = _run_worker(
                    pair_id=pair_index,
                    schedule_position=position,
                    model_key=model_key,
                    snapshot_identity=snapshots[model_key],
                )
                value["study_fingerprints"] = {
                    "code_sha256": provenance["code_sha256"],
                    "environment_sha256": provenance["environment_sha256"],
                    "preregistration_sha256": FROZEN_PREREGISTRATION_SHA256,
                    "prompt_sha256": provenance["prompt_sha256"],
                }
                value["execution_context"] = {
                    "code_sha256": provenance["code_sha256"],
                    "environment_sha256": provenance["environment_sha256"],
                    "git_dirty_state": provenance["git_dirty_state"],
                    "git_revision": provenance["git_revision"],
                    "hardware": provenance["hardware"],
                    "model_snapshot_sha256": value["model_snapshot"].get(
                        "snapshot_sha256"
                    ),
                    "snapshot_path": value["model_snapshot"].get("snapshot_path"),
                    "snapshot_stat_manifest": value["model_snapshot"].get(
                        "execution_stat_manifest"
                    ),
                    "package_versions": provenance["environment"]["packages"],
                    "power_source": value["power_source"],
                    "preregistration_sha256": FROZEN_PREREGISTRATION_SHA256,
                    "prompt_sha256": provenance["prompt_sha256"],
                    "weight_sha256": value["model_snapshot"].get("weight_sha256"),
                }
                state["runs"].append(value)
                try:
                    charged_seconds = _record_gpu(guard, value["model_work_ns"])
                except BaseException as exc:
                    _mark_abort(value, f"budget_record_failed:{type(exc).__name__}")
                    raise
                if value.get("abort_reason") is not None:
                    raise ResourceError(
                        f"validated worker requested abort: {value['abort_reason']}"
                    )
                if type(value.get("swap_delta_bytes")) is not int:
                    _mark_abort(value, "swap_delta_unavailable")
                    raise ResourceError("swap delta unavailable after worker")
                if value["swap_delta_bytes"] > 0:
                    _mark_abort(value, "swap_growth_detected")
                    raise ResourceError("swap grew during worker")
                if (
                    value["rss_peak_bytes"] > MAX_MEMORY_BYTES
                    or value["mlx_peak_bytes"] > MAX_MEMORY_BYTES
                ):
                    _mark_abort(value, "worker_memory_limit_exceeded")
                    raise ResourceError("worker memory limit exceeded")
                try:
                    _required_breaks(guard, charged_seconds)
                except BaseException as exc:
                    _mark_abort(value, f"required_break_failed:{type(exc).__name__}")
                    raise
    except BaseException as exc:
        error = _exception_record(exc)
    finally:
        if guard is not None:
            try:
                guard.finish_candidate()
            except BaseException as exc:
                if error is None:
                    error = _exception_record(exc)

    # Every operation after the start marker is fail-safe.  A malformed raw
    # event, a guard implementation failure, or a statistics error must still
    # leave one atomic partial result instead of losing the evidence path.
    try:
        runs = state["runs"]
        finalization_errors: list[str] = []
        post_snapshots: dict[str, dict[str, Any]] = {}
        snapshot_content_pass = True
        try:
            for key, spec in MODEL_SPECS.items():
                post_snapshot = resolve_local_model_snapshot(spec["model_id"])
                if post_snapshot.revision != spec["revision"]:
                    raise StudyError(f"post-study model revision changed: {key}")
                post_identity = _snapshot_identity(post_snapshot)
                post_snapshots[key] = post_identity
                expected_identity = snapshots[key]
                if any(
                    post_identity.get(field) != expected_identity.get(field)
                    for field in (
                        "snapshot_path",
                        "snapshot_sha256",
                        "snapshot_files_sha256",
                        "weight_sha256",
                    )
                ):
                    raise ResourceError(f"post-study snapshot content changed: {key}")
        except BaseException as exc:
            snapshot_content_pass = False
            finalization_errors.append(f"snapshot postflight failed: {exc}")
        swap_end = _swap_used_bytes()
        study_swap_delta = (
            swap_end - swap_start if swap_end is not None else None
        )

        try:
            resources_pass = _resource_gate(runs, study_swap_delta)
        except BaseException as exc:
            resources_pass = False
            finalization_errors.append(f"resource gate failed: {exc}")

        budget, guard_error = _guard_summary(guard)
        if guard_error is not None:
            finalization_errors.append(guard_error)
        budget_pass = bool(
            error is None
            and not finalization_errors
            and budget.get("duty_cycle_limit") == 0.15
            and budget.get("gpu_work_seconds", math.inf) <= POLICY.gpu_work_limit_s
            and budget.get("max_continuous_gpu_seconds", math.inf)
            <= POLICY.continuous_gpu_limit_s
            and budget.get("wall_seconds", math.inf) <= POLICY.wall_limit_s
        )

        by_model: dict[str, dict[str, Any]] = {}
        for key in MODEL_SPECS:
            try:
                model_runs = [run for run in runs if run.get("model_key") == key]
                by_model[key] = _model_summary(key, model_runs)
            except BaseException as exc:
                by_model[key] = _empty_model_summary(key)
                finalization_errors.append(f"{key} summary failed: {exc}")

        try:
            paired = _pairwise(runs)
        except BaseException as exc:
            paired = {
                "complete": False,
                "error": f"pairwise aggregation failed: {exc}",
                "pair_ids": [],
                "pairs": {},
                "ratios_1b_div_4b": {},
            }
            finalization_errors.append(f"pairwise aggregation failed: {exc}")
        try:
            cross_model_text = _cross_model_text(runs)
        except BaseException as exc:
            cross_model_text = {
                "complete": False,
                "error": f"cross-model aggregation failed: {exc}",
                "exact_text_equal_count": 0,
                "exact_text_equal_total": "0/6",
                "pairs": [],
                "informational_only": True,
            }
            finalization_errors.append(f"cross-model aggregation failed: {exc}")

        try:
            fresh_process_pass = _fresh_process_gate(runs)
            prompt_identity_pass = _prompt_identity_gate(runs)
        except BaseException as exc:
            fresh_process_pass = False
            prompt_identity_pass = False
            finalization_errors.append(f"global correctness gates failed: {exc}")
        correctness_failure = bool(
            len(runs) == RUN_COUNT
            and (
                not prompt_identity_pass
                or not fresh_process_pass
                or any(
                    not by_model[key]["correctness_pass"] for key in MODEL_SPECS
                )
            )
        )
        if finalization_errors and error is None:
            error = {
                "message": "; ".join(finalization_errors)[:500],
                "type": "FinalizationError",
            }
        terminal_failure = bool(
            error is not None
            or finalization_errors
            or not resources_pass
            or not budget_pass
            or not snapshot_content_pass
            or not paired.get("complete")
            or not cross_model_text.get("complete")
        )
        decision = decision_for(
            one_b_pass=by_model["1b"]["functional_pass"],
            four_b_pass=by_model["4b"]["functional_pass"],
            pairwise=paired,
            one_b_peak_rss=by_model["1b"]["peak_rss_bytes"],
            four_b_peak_rss=by_model["4b"]["peak_rss_bytes"],
            correctness_failure=correctness_failure,
            terminal_failure=terminal_failure,
        )
        max_mlx = [
            run.get("mlx_peak_bytes")
            for run in runs
            if isinstance(run, dict) and type(run.get("mlx_peak_bytes")) is int
        ]
        max_rss = [
            run.get("rss_peak_bytes")
            for run in runs
            if isinstance(run, dict) and type(run.get("rss_peak_bytes")) is int
        ]
        state.update(
            {
                "budget": budget,
                "completed_at_unix_ns": time.time_ns(),
                "decision": decision,
                "error": error,
                "finalization_errors": finalization_errors,
                "gates": {
                    "all_runs_completed": len(runs) == RUN_COUNT,
                    "resource_pass": resources_pass,
                    "budget_pass": budget_pass,
                    "correctness_failure": correctness_failure,
                    "fresh_process_pass": fresh_process_pass,
                    "prompt_identity_pass": prompt_identity_pass,
                    "pairing_pass": paired.get("complete") is True,
                    "cross_model_text_complete": cross_model_text.get("complete") is True,
                    "snapshot_content_pass": snapshot_content_pass,
                    "model_1b": by_model["1b"]["functional_pass"],
                    "model_4b": by_model["4b"]["functional_pass"],
                },
                "metrics": {
                    "model_1b": by_model["1b"],
                    "model_4b": by_model["4b"],
                    "pairwise": paired,
                    "cross_model_text": cross_model_text,
                    "runs_completed": len(runs),
                    "study_swap_delta_bytes": study_swap_delta,
                },
                "partial_result": bool(
                    len(runs) != RUN_COUNT
                    or error is not None
                    or finalization_errors
                    or not paired.get("complete")
                    or not cross_model_text.get("complete")
                ),
                "snapshot_postflight": post_snapshots,
                "resources": {
                    "max_mlx_peak_bytes": max(max_mlx) if max_mlx else None,
                    "max_rss_peak_bytes": max(max_rss) if max_rss else None,
                    "swap_after_bytes": swap_end,
                    "swap_before_bytes": swap_start,
                    "swap_delta_bytes": study_swap_delta,
                },
                "thresholds": {
                    "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "max_event_bytes": MAX_EVENT_BYTES,
                    "max_memory_bytes": MAX_MEMORY_BYTES,
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                    "required_pairs": PAIR_COUNT,
                    "required_runs": RUN_COUNT,
                    "worker_watchdog_seconds": WORKER_WATCHDOG_SECONDS,
                },
            }
        )
    except BaseException as exc:
        state.update(
            {
                "completed_at_unix_ns": time.time_ns(),
                "decision": "resource_or_budget_failed",
                "error": _exception_record(exc),
                "finalization_errors": [
                    f"unhandled finalization failure: {type(exc).__name__}: {str(exc)[:400]}"
                ],
                "partial_result": True,
            }
        )
    _write_result_fail_safe(state)
    return state


def _self_check() -> int:
    if _sha256(PREREGISTRATION) != FROZEN_PREREGISTRATION_SHA256:
        raise StudyError("preregistration hash mismatch")
    worker = _load_worker_module()
    assert worker.PROMPT_SHA256 == (
        "c746eca8644a18fc75673acb9b3dbdf03825cbfba6c76faede5d909cf3d2ea0b"
    )
    assert PAIR_SCHEDULE == (
        ("1b", "4b"),
        ("1b", "4b"),
        ("1b", "4b"),
        ("4b", "1b"),
        ("4b", "1b"),
        ("4b", "1b"),
    )
    fake_runs: list[dict[str, Any]] = []
    for pair_id in range(1, PAIR_COUNT + 1):
        for model_key, pid, rate in (("1b", pair_id * 2, 20.0), ("4b", pair_id * 2 + 1, 10.0)):
            fake_runs.append(
                {
                    "pair_id": pair_id,
                    "model_key": model_key,
                    "pid": pid,
                    "ttft_ns": 2_000_000 if model_key == "1b" else 1_000_000,
                    "model_work_ns": 4_000_000 if model_key == "1b" else 2_000_000,
                    "process_wall_ns": 8_000_000 if model_key == "1b" else 4_000_000,
                    "token_rate": rate,
                    "text": "same",
                    "token_sha256": "a" * 64,
                }
            )
    paired = _pairwise(fake_runs)
    assert paired["complete"] is True
    assert paired["ratios_1b_div_4b"]["token_rate"]["median"] == 2.0
    assert paired["ratios_1b_div_4b"]["token_rate"]["bootstrap_95_ci"] is not None
    duplicate_pair = _pairwise(fake_runs + [fake_runs[0]])
    duplicate_text = _cross_model_text(fake_runs + [fake_runs[0]])
    assert duplicate_pair["complete"] is False and "duplicate" in duplicate_pair["error"]
    assert duplicate_text["complete"] is False and "duplicate" in duplicate_text["error"]
    abort_probe: dict[str, Any] = {"abort_reason": None}
    _mark_abort(abort_probe, "first")
    _mark_abort(abort_probe, "second")
    assert abort_probe["abort_reason"] == "first;second"
    assert decision_for(
        one_b_pass=True,
        four_b_pass=True,
        pairwise=paired,
        one_b_peak_rss=70,
        four_b_peak_rss=100,
        terminal_failure=True,
    ) == "resource_or_budget_failed"
    assert decision_for(
        one_b_pass=True,
        four_b_pass=False,
        pairwise=duplicate_pair,
        one_b_peak_rss=None,
        four_b_peak_rss=None,
    ) == "resource_or_budget_failed"
    assert decision_for(
        one_b_pass=True,
        four_b_pass=False,
        pairwise=None,
        one_b_peak_rss=None,
        four_b_peak_rss=None,
    ) == "planner_1b_qualified_exact_case"
    assert decision_for(
        one_b_pass=False,
        four_b_pass=True,
        pairwise=None,
        one_b_peak_rss=None,
        four_b_peak_rss=None,
    ) == "planner_4b_qualified_exact_case"
    assert decision_for(
        one_b_pass=False,
        four_b_pass=False,
        pairwise=None,
        one_b_peak_rss=None,
        four_b_peak_rss=None,
    ) == "no_planner_qualified"
    assert decision_for(
        one_b_pass=True,
        four_b_pass=True,
        pairwise={
            "complete": True,
            "ratios_1b_div_4b": {"process_wall": {"median": 1.04}},
        },
        one_b_peak_rss=70,
        four_b_peak_rss=100,
    ) == "both_qualified_1b_preferred"
    assert decision_for(
        one_b_pass=True,
        four_b_pass=True,
        pairwise={
            "complete": True,
            "ratios_1b_div_4b": {"process_wall": {"median": 1.06}},
        },
        one_b_peak_rss=70,
        four_b_peak_rss=100,
    ) == "both_qualified_no_automatic_preference"
    assert decision_for(
        one_b_pass=True,
        four_b_pass=True,
        pairwise=None,
        one_b_peak_rss=70,
        four_b_peak_rss=100,
        terminal_failure=True,
    ) == "resource_or_budget_failed"
    assert _bootstrap_ci([1.0] * 6) == {
        "percentiles": {"lower": 0.025, "upper": 0.975, "interpolation": "linear"},
        "lower": 1.0,
        "upper": 1.0,
        "method": "paired six-pair median-ratio bootstrap percentile",
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
    }
    print(json.dumps({"checks": 25, "self_check": "pass"}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="measure_dual_model_planner", allow_abbrev=False)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--self-check", action="store_true")
    modes.add_argument("--show", action="store_true")
    parser.add_argument("--run-id", default=RUN_ID)
    args = parser.parse_args(argv)
    if args.self_check:
        try:
            return _self_check()
        except StudyError as exc:
            print(json.dumps({"error": str(exc), "self_check": "failed"}, sort_keys=True))
            return 2
    if args.show:
        if not RESULT_PATH.is_file() or RESULT_PATH.is_symlink():
            raise SystemExit("result is unavailable")
        print(RESULT_PATH.read_text(encoding="utf-8"), end="")
        return 0
    if not args.execute:
        print(json.dumps({"state": "not_released", "required_flag": "--execute"}))
        return 78
    try:
        report = execute(args.run_id)
    except StudyError as exc:
        print(json.dumps({"error": str(exc), "state": "not_started"}, sort_keys=True))
        return 2
    decision = report.get("decision")
    if type(decision) is not str:
        decision = "resource_or_budget_failed"
    report_run_id = report.get("run_id")
    if type(report_run_id) is not str:
        report_run_id = RUN_ID
    report_runs = report.get("runs")
    fallback_runs_completed = len(report_runs) if isinstance(report_runs, list) else 0
    report_metrics = report.get("metrics")
    runs_completed = (
        report_metrics.get("runs_completed")
        if isinstance(report_metrics, dict)
        and type(report_metrics.get("runs_completed")) is int
        and 0 <= report_metrics["runs_completed"] <= RUN_COUNT
        else fallback_runs_completed
    )
    print(
        json.dumps(
            {
                "decision": decision,
                "formal_claim": False,
                "result": RESULT_PATH.relative_to(PROJECT_ROOT).as_posix(),
                "run_id": report_run_id,
                "runs_completed": runs_completed,
            },
            sort_keys=True,
        )
    )
    successful_decisions = {
        "planner_1b_qualified_exact_case",
        "planner_4b_qualified_exact_case",
        "both_qualified_1b_preferred",
        "both_qualified_no_automatic_preference",
    }
    return 0 if decision in successful_decisions else 1


if __name__ == "__main__":
    raise SystemExit(main())
