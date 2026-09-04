#!/usr/bin/env python3
"""Fail-closed parent harness for Cycle 18.

The parent never imports MLX and never accounts GPU work itself.  The child
owns model loading, BudgetGuard and pacing; this process validates the one-event
protocol, enforces the hard wall clock, and aggregates the six pairs.
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
import secrets
import stat
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKER = Path(__file__).with_name("worker.py")
PREREGISTRATION = Path(__file__).with_name("PREREGISTRATION.md")
RESULT_PATH = Path(__file__).with_name("results.json")
ATTEMPT_DIR = PROJECT_ROOT / ".friday-data" / "fused-greedy-compile"
ATTEMPT_PATH = ATTEMPT_DIR / "attempt.json"
STUDY_ID = "fused-greedy-compile-20260825-01"
RUN_ID = "fused-greedy-compile-validation-20260825-01"
CANDIDATE_ID = "fixed_compiled_fused_greedy"
MODEL_ID = "mlx-community/gemma-3-4b-it-4bit"
MODEL_REVISION = "93724907d4ed1745d2fe50baadf3b0b01a65abf2"
MODEL_KEY = "4b"
EXPECTED_SNAPSHOT_SHA256 = "e6edcd46c52b4cf5580f095185a94858565896df7f31c23522294e8f73b3edae"
EXPECTED_WEIGHT_SHA256 = "94d3d701367d78584a9334ca00672b1c86e4aefa6a94167556c0485381e74af3"
EXPECTED_PROMPT_SHA256 = "c746eca8644a18fc75673acb9b3dbdf03825cbfba6c76faede5d909cf3d2ea0b"
EXPECTED_PROMPT_TOKEN_SHA256 = "80ecf700cf0dfdc82616c73f1b6a5fccc137b68e9bb9586ca376c3f2adb260ad"
EXPECTED_RENDERED_PROMPT_SHA256 = "9e18d10b7b101bda3d28593190e622544d474655872aed826c9cbc44211a2cca"
FROZEN_PREREGISTRATION_SHA256 = "ac59015a2e5cd6635468bc85704cf2418b9eef6616043201e454c41c0eed1399"
AUTH_PREFIX = "FRIDAY_FGC_"
AUTH_NONCE = "cycle18-fused-greedy-compile-v1"
PROTOCOL_VERSION = 1
PAIR_COUNT = 6
ARM_NAMES = ("fixed_compiled_external_greedy", "fixed_compiled_fused_greedy")
PAIR_ORDERS = tuple((ARM_NAMES[0], ARM_NAMES[1]) if i % 2 == 0 else (ARM_NAMES[1], ARM_NAMES[0]) for i in range(PAIR_COUNT))
OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONNOUSERSITE": "1"}
UNSAFE_ENV = ("PYTHONHOME", "PYTHONPATH", "PYTHONINSPECT", "PYTHONSTARTUP")
MAX_EVENT_BYTES = 1_000_000
WORKER_TIMEOUT_SECONDS = 280.0
MAX_WALL_SECONDS = 1200.0
BOOTSTRAP_SEED = 20260825
BOOTSTRAP_RESAMPLES = 10_000
EXPECTED_MACHINE = "arm64"
EXPECTED_CPU_BRAND = "Apple M1 Max"
EXPECTED_MEMORY_BYTES = 32 * 1024**3
REQUIRED_PACKAGES = {"mlx": "0.32.0", "mlx-lm": "0.31.3"}
MAX_STDOUT_BYTES = 1_000_000
MAX_STDERR_BYTES = 64_000


class StudyError(RuntimeError):
    pass


class WorkerError(StudyError):
    pass


def _module():
    if not WORKER.is_file() or WORKER.is_symlink(): raise StudyError("worker unavailable")
    spec = importlib.util.spec_from_file_location("fgc_worker", WORKER)
    if spec is None or spec.loader is None: raise StudyError("worker import unavailable")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024): digest.update(chunk)
    return digest.hexdigest()


def code_fingerprints() -> dict[str, str]:
    return {path.relative_to(PROJECT_ROOT).as_posix(): _sha256_file(path) for path in (PREREGISTRATION, WORKER, Path(__file__))}


def environment_fingerprint() -> str:
    return _sha256_bytes(_canonical({"offline": OFFLINE_ENV, "removed": UNSAFE_ENV, "python": str(Path(sys.executable).resolve()), "machine": platform.machine()}))


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
        value = psutil.swap_memory().used
    except Exception:
        return None
    return int(value) if type(value) is int and value >= 0 else None


def _snapshot_identity(snapshot: Any) -> dict[str, Any]:
    root = Path(snapshot.path).resolve(strict=True)
    repository = root.parent.parent.resolve(strict=True)
    if root.parent.name != "snapshots": raise StudyError("unexpected snapshot layout")
    required = ["config.json", "tokenizer_config.json"]
    required += [name for name in ("tokenizer.json", "tokenizer.model") if (root / name).is_file()][:1]
    required.extend(snapshot.weight_files)
    files: dict[str, str] = {}; manifest: dict[str, Any] = {}
    for relative in dict.fromkeys(required):
        path = (root / relative).resolve(strict=True)
        path.relative_to(repository)
        if not path.is_file(): raise StudyError(f"snapshot file missing: {relative}")
        files[relative] = _sha256_file(path); metadata = path.stat()
        manifest[relative] = {"dev": int(metadata.st_dev), "inode": int(metadata.st_ino), "mtime_ns": int(metadata.st_mtime_ns), "path": str(path), "size": int(metadata.st_size)}
    generation = (root / "generation_config.json").resolve(strict=True)
    generation.relative_to(repository)
    generation_metadata = generation.stat()
    manifest["generation_config.json"] = {"dev": int(generation_metadata.st_dev), "inode": int(generation_metadata.st_ino), "mtime_ns": int(generation_metadata.st_mtime_ns), "path": str(generation), "size": int(generation_metadata.st_size)}
    execution_files = {name: _sha256_file(Path(item["path"])) for name, item in manifest.items()}
    return {"model_id": MODEL_ID, "model_revision": snapshot.revision, "snapshot_path": str(root), "snapshot_files_sha256": files, "execution_files_sha256": execution_files,
            "snapshot_sha256": _sha256_bytes(_canonical(files)), "weight_sha256": {name: files[name] for name in snapshot.weight_files}, "execution_stat_manifest": manifest,
            "model_snapshot_weight_files": list(snapshot.weight_files), "model_snapshot_weight_bytes": int(snapshot.weight_bytes), "model_source": "validated_project_local_snapshot"}


def _clean_worktree() -> tuple[str, str]:
    revision = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain", "--untracked-files=all", "--", ".", ":(exclude)ProjectAtlas")
    if dirty: raise StudyError("project worktree is dirty")
    return revision, dirty


def _require_target() -> None:
    if platform.machine() != EXPECTED_MACHINE or _sysctl("machdep.cpu.brand_string") != EXPECTED_CPU_BRAND or _sysctl("hw.memsize") != str(EXPECTED_MEMORY_BYTES):
        raise StudyError("registered Apple M1 Max/32 GB target not present")
    expected_python = (PROJECT_ROOT / ".venv" / "bin" / "python").resolve(strict=True)
    if Path(sys.executable).resolve() != expected_python: raise StudyError("wrong project Python")
    for package, expected in REQUIRED_PACKAGES.items():
        if importlib.metadata.version(package) != expected: raise StudyError(f"wrong package version: {package}")
    # This is the only MLX import in the parent and is reachable only from --execute.
    import mlx.core as mx
    if str(mx.default_device()) != "Device(gpu, 0)": raise StudyError("MLX GPU device gate failed")


def _target_info() -> dict[str, Any]:
    return {"machine": platform.machine(), "cpu_brand": _sysctl("machdep.cpu.brand_string"), "memory_bytes": _sysctl("hw.memsize"), "packages": {name: importlib.metadata.version(name) for name in REQUIRED_PACKAGES}, "python": str(Path(sys.executable).resolve()), "device": "Device(gpu, 0)"}


def _preflight() -> tuple[str, str, str, dict[str, Any], int]:
    if _sha256_file(PREREGISTRATION) != FROZEN_PREREGISTRATION_SHA256: raise StudyError("preregistration hash is not sealed")
    module = _module(); contract = module.protocol_contract()
    if (module.PROMPT_SHA256 != EXPECTED_PROMPT_SHA256 or contract.get("study_id") != STUDY_ID or contract.get("run_id") != RUN_ID
            or tuple(contract.get("arms", ())) != ARM_NAMES or contract.get("capacity") != 512 or contract.get("warmups") != 8): raise StudyError("worker protocol/prompt fingerprint failed")
    if RESULT_PATH.exists() or RESULT_PATH.is_symlink() or ATTEMPT_PATH.exists() or ATTEMPT_PATH.is_symlink(): raise StudyError("existing evidence blocks execution")
    revision, dirty = _clean_worktree(); _require_target()
    sys.path.insert(0, str(PROJECT_ROOT)); sys.path.insert(0, str(PROJECT_ROOT / "tools"))
    from _bench import require_ac_power, resolve_local_model_snapshot
    power = require_ac_power()
    snapshot = resolve_local_model_snapshot(MODEL_ID)
    if snapshot.revision != MODEL_REVISION: raise StudyError("local snapshot revision changed")
    identity = _snapshot_identity(snapshot); swap = _swap_used_bytes()
    if identity["snapshot_sha256"] != EXPECTED_SNAPSHOT_SHA256 or identity["weight_sha256"] != {"model.safetensors": EXPECTED_WEIGHT_SHA256}:
        raise StudyError("local snapshot or weight hash does not match the registered values")
    if swap is None: raise StudyError("swap usage unavailable")
    return revision, dirty, power, identity, swap


def _evidence_state() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, path in (("marker", ATTEMPT_PATH), ("result", RESULT_PATH)):
        try: metadata = path.lstat()
        except FileNotFoundError:
            result[name] = {"exists": False, "mode": None, "sha256": None, "regular": False, "symlink": False}; continue
        result[name] = {"exists": True, "mode": stat.S_IMODE(metadata.st_mode), "sha256": _sha256_file(path), "regular": stat.S_ISREG(metadata.st_mode), "symlink": stat.S_ISLNK(metadata.st_mode)}
    return result


def _validate_evidence_state(state: dict[str, Any]) -> None:
    for name, value in state.items():
        if value["exists"] and (not value["regular"] or value["symlink"]): raise StudyError(f"{name} is not a regular file")
    if state["marker"]["exists"] and state["marker"]["mode"] != 0o600: raise StudyError("marker must be 0600")
    if state["result"]["exists"] and state["result"]["mode"] != 0o644: raise StudyError("result must be 0644")


def _strict_json(payload: bytes) -> dict[str, Any]:
    raw = payload[:-1] if payload.endswith(b"\n") else payload
    if not raw or len(raw) > MAX_EVENT_BYTES or b"\n" in raw or b"\r" in raw: raise WorkerError("invalid worker output framing")
    def unique(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result: raise WorkerError("duplicate JSON key")
            result[key] = value
        return result
    value = json.loads(raw.decode("utf-8", errors="strict"), object_pairs_hook=unique, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    if not isinstance(value, dict): raise WorkerError("worker output is not one object")
    return value


def _hash_ints(values: Any) -> str:
    return _sha256_bytes(_canonical(values))


def _validate_arm(arm: dict[str, Any], name: str) -> None:
    required = ("arm", "fixed_cache", "fixed_compile", "fused_selection", "cache_capacity", "warmup_forwards", "decode_forwards", "physical_forwards", "finish_reason", "cache_discarded", "physical_tokens", "logical_tokens", "visible_tokens", "physical_token_count", "logical_token_count", "visible_token_count", "overproduced_tokens", "eos_found", "eos_position", "eos_token_id", "physical_token_sha256", "logical_token_sha256", "visible_token_sha256", "visible_text", "text_sha256", "prompt_sha256", "prompt_token_sha256", "rendered_prompt_sha256", "ttft_ns", "decode_critical_path_ns", "host_boundary_count", "host_transfer_api_call_count", "intertoken_ns", "token_rate", "timing_scopes")
    if set(required) - set(arm): raise WorkerError(f"arm {name} missing fields")
    if arm["arm"] != name or arm["fixed_cache"] is not True or arm["fixed_compile"] is not True or arm["cache_capacity"] != 512 or arm["warmup_forwards"] != 8 or arm["cache_discarded"] is not True: raise WorkerError("fixed arm invariant failed")
    if arm["fused_selection"] is not (name == ARM_NAMES[1]): raise WorkerError("fused arm invariant failed")
    physical = arm["physical_tokens"]
    if not isinstance(physical, list) or len(physical) > 32 or any(type(item) is not int or item < 0 for item in physical): raise WorkerError("physical tokens invalid")
    eos = next((i for i, token in enumerate(physical) if token in (1, 106)), None)
    logical = physical if eos is None else physical[:eos + 1]; visible = physical if eos is None else physical[:eos]
    for field, expected in (("logical_tokens", logical), ("visible_tokens", visible), ("physical_token_count", len(physical)), ("logical_token_count", len(logical)), ("visible_token_count", len(visible)), ("overproduced_tokens", len(physical) - len(logical)), ("eos_found", eos is not None), ("eos_position", eos), ("eos_token_id", None if eos is None else physical[eos])):
        if arm[field] != expected: raise WorkerError(f"token contract failed: {field}")
    for field, value in (("physical_token_sha256", physical), ("logical_token_sha256", logical), ("visible_token_sha256", visible)):
        if not isinstance(arm[field], str) or arm[field] != _hash_ints(value): raise WorkerError(f"token hash failed: {field}")
    if not isinstance(arm["visible_text"], str) or arm["text_sha256"] != _sha256_bytes(arm["visible_text"].encode()): raise WorkerError("text hash failed")
    if arm["decode_forwards"] != arm["physical_forwards"] or arm["physical_forwards"] != len(physical) - 1: raise WorkerError("forward count failed")
    if not isinstance(arm["timing_scopes"], dict) or not all(isinstance(value, str) and value for value in arm["timing_scopes"].values()): raise WorkerError("timing scope labels missing")
    if arm["host_boundary_count"] != len(physical) or arm["host_transfer_api_call_count"] != len(physical): raise WorkerError("boundary count failed")
    if not isinstance(arm["intertoken_ns"], list) or len(arm["intertoken_ns"]) != len(physical) - 1: raise WorkerError("timing count failed")
    for field in ("prompt_sha256", "prompt_token_sha256", "rendered_prompt_sha256", "physical_token_sha256", "logical_token_sha256", "visible_token_sha256", "text_sha256"):
        if len(arm[field]) != 64: raise WorkerError(f"invalid hash: {field}")
    if (arm["prompt_sha256"], arm["prompt_token_sha256"], arm["rendered_prompt_sha256"]) != (EXPECTED_PROMPT_SHA256, EXPECTED_PROMPT_TOKEN_SHA256, EXPECTED_RENDERED_PROMPT_SHA256): raise WorkerError("arm prompt identity failed")


def _validate_event(event: dict[str, Any], process_index: int, order: tuple[str, str], *, identity: dict[str, Any], git_revision: str, marker_token: str, expected_pid: int) -> dict[str, Any]:
    if event.get("study_id") != STUDY_ID or event.get("run_id") != RUN_ID or event.get("candidate_id") != CANDIDATE_ID or event.get("formal_claim") is not False: raise WorkerError("event identity failed")
    if event.get("process_index") != process_index or tuple(event.get("arm_order", ())) != order: raise WorkerError("process/order failed")
    status = event.get("status")
    if status not in {"complete", "candidate_not_runnable", "correctness_failed", "resource_or_budget_failed", "error"}: raise WorkerError("unknown terminal status")
    if event.get("protocol_version") != PROTOCOL_VERSION or not isinstance(event.get("error"), (dict, type(None))): raise WorkerError("terminal protocol/error schema failed")
    if event.get("error") is not None and (set(event["error"]) - {"type", "message"} or not isinstance(event["error"].get("message", ""), str) or len(event["error"].get("message", "")) > 500): raise WorkerError("bounded error schema failed")
    if event.get("load_count") not in (0, 1) or event.get("pid") != expected_pid: raise WorkerError("terminal process schema failed")
    if status != "complete":
        if event.get("event") not in {"terminal", "error"}: raise WorkerError("terminal event kind failed")
        if not isinstance(event.get("error"), dict): raise WorkerError("terminal error evidence missing")
        if (event.get("preregistration_sha256") != FROZEN_PREREGISTRATION_SHA256 or event.get("environment_sha256") != environment_fingerprint()
                or event.get("code_fingerprints") != code_fingerprints() or event.get("code_sha256") != _sha256_bytes(_canonical(code_fingerprints()))
                or not isinstance(event.get("git_revision"), str) or not isinstance(event.get("dirty_state"), str)):
            raise WorkerError("terminal provenance identity failed")
        if not event.get("partial_result", True): raise WorkerError("terminal event must preserve partial-result flag")
        event["partial_result"] = True
        return event
    if event.get("event") != "complete" or event.get("load_count") != 1 or set(event.get("arms", {})) != set(ARM_NAMES): raise WorkerError("complete arm set failed")
    if event.get("marker_token_sha256") != _sha256_bytes(marker_token.encode()): raise WorkerError("marker binding failed")
    if event.get("git_revision") != git_revision or event.get("dirty_state") != "clean": raise WorkerError("worker Git provenance failed")
    if event.get("snapshot_path") != identity["snapshot_path"] or event.get("weight_sha256") != identity["weight_sha256"] or event.get("execution_files_sha256") != identity["execution_files_sha256"]: raise WorkerError("snapshot binding path/weight failed")
    integrity = event.get("snapshot_integrity")
    if not isinstance(integrity, dict) or integrity.get("before_load_stat_manifest") != identity["execution_stat_manifest"] or integrity.get("after_load_stat_manifest") != identity["execution_stat_manifest"] or integrity.get("post_arm_stat_manifest") != identity["execution_stat_manifest"] or integrity.get("snapshot_files_sha256") != identity["snapshot_files_sha256"] or integrity.get("execution_files_sha256") != identity["execution_files_sha256"]: raise WorkerError("snapshot stat/hash provenance failed")
    if event.get("swap_available") is not True: raise WorkerError("swap availability evidence missing")
    for field in ("rss_peak_bytes", "mlx_peak_bytes", "swap_before_bytes", "swap_after_bytes", "swap_delta_bytes"):
        if type(event.get(field)) is not int: raise WorkerError("numeric resource evidence missing")
    if event.get("snapshot_revision") != MODEL_REVISION or event.get("snapshot_sha256") != EXPECTED_SNAPSHOT_SHA256: raise WorkerError("snapshot identity failed")
    if event.get("weight_sha256") != {"model.safetensors": EXPECTED_WEIGHT_SHA256}: raise WorkerError("weight identity failed")
    if event.get("device") != "Device(gpu, 0)" or event.get("greedy") is not True or event.get("sampler_temperature") != 0.0: raise WorkerError("device/sampler gate failed")
    if event.get("dirty_state") != "clean": raise WorkerError("worker dirty-state gate failed")
    if event.get("preregistration_sha256") != FROZEN_PREREGISTRATION_SHA256 or event.get("environment_sha256") != environment_fingerprint(): raise WorkerError("provenance fingerprint failed")
    if event.get("code_fingerprints") != code_fingerprints() or event.get("code_sha256") != _sha256_bytes(_canonical(code_fingerprints())): raise WorkerError("code fingerprint failed")
    budget = event.get("budget")
    if not isinstance(budget, dict) or budget.get("duty_cycle_limit") != .15 or budget.get("continuous_gpu_limit_seconds") != 6.0 or budget.get("gpu_work_limit_seconds") != 120.0 or budget.get("wall_limit_seconds") != 1200.0 or float(budget.get("gpu_work_seconds", 999.0)) > 120.0 or float(budget.get("max_continuous_gpu_seconds", 999.0)) > 6.0 or float(budget.get("wall_seconds", 9999.0)) > 1200.0: raise WorkerError("budget gate failed")
    if event.get("swap_delta_bytes") != 0 or int(event.get("rss_peak_bytes", 0)) > 5 * 1024**3 or int(event.get("mlx_peak_bytes", 0)) > 5 * 1024**3: raise WorkerError("resource gate failed")
    if event.get("prompt_sha256") != EXPECTED_PROMPT_SHA256 or event.get("prompt_token_sha256") != EXPECTED_PROMPT_TOKEN_SHA256 or event.get("rendered_prompt_sha256") != EXPECTED_RENDERED_PROMPT_SHA256: raise WorkerError("prompt identity failed")
    prompt_ids = event.get("prompt_token_ids")
    if not isinstance(prompt_ids, list) or len(prompt_ids) != 322 or any(type(item) is not int or item < 0 for item in prompt_ids) or _hash_ints(prompt_ids) != EXPECTED_PROMPT_TOKEN_SHA256: raise WorkerError("prompt token evidence failed")
    try: rendered_prompt = base64.b64decode(event.get("rendered_prompt_b64", ""), validate=True)
    except (ValueError, TypeError): raise WorkerError("rendered prompt encoding failed")
    if _sha256_bytes(rendered_prompt) != EXPECTED_RENDERED_PROMPT_SHA256: raise WorkerError("rendered prompt evidence failed")
    if event.get("prompt_tokens") != 322: raise WorkerError("prompt token count failed")
    for arm in ARM_NAMES:
        _validate_arm(event["arms"][arm], arm)
        resource = event.get("arm_resources", {}).get(arm)
        budget_arm = event.get("arm_budget", {}).get(arm)
        if not isinstance(resource, dict) or resource.get("swap_available") is not True or type(resource.get("swap_delta_bytes")) is not int or resource.get("swap_delta_bytes") != 0 or type(resource.get("rss_peak_bytes")) is not int or type(resource.get("mlx_peak_bytes")) is not int or resource.get("rss_peak_bytes") > 5 * 1024**3 or resource.get("mlx_peak_bytes") > 5 * 1024**3: raise WorkerError("arm resource gate failed")
        if not isinstance(budget_arm, dict) or budget_arm.get("charge_accepted") is not True or int(budget_arm.get("observed_model_work_ns", 0)) != int(budget_arm.get("charged_model_work_ns", -1)) or int(budget_arm.get("observed_model_work_ns", 0)) > 6_000_000_000 or budget_arm.get("required_break_blocks") != 13: raise WorkerError("arm budget gate failed")
    if not event.get("correctness", {}).get("pass") or any(event["arms"][ARM_NAMES[0]][field] != event["arms"][ARM_NAMES[1]][field] for field in ("physical_token_sha256", "logical_token_sha256", "visible_token_sha256", "text_sha256")): raise WorkerError("correctness gate failed")
    return event


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values: return {"n": 0, "median": None, "mad": None, "p50": None, "p95": None, "p99": None}
    ordered = sorted(float(value) for value in values)
    median = statistics.median(ordered)
    def percentile(q: float) -> float: return ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]
    return {"n": len(ordered), "median": median, "mad": statistics.median([abs(value - median) for value in ordered]), "p50": percentile(.50), "p95": percentile(.95), "p99": percentile(.99)}


def paired_bootstrap(baseline: list[float], candidate: list[float], seed: int = BOOTSTRAP_SEED, iterations: int = BOOTSTRAP_RESAMPLES) -> dict[str, Any]:
    if len(baseline) != len(candidate) or not baseline: raise ValueError("paired samples required")
    ratios = [candidate[i] / baseline[i] for i in range(len(baseline))]
    rng = random.Random(seed); medians = []
    for _ in range(iterations):
        sample = [ratios[rng.randrange(len(ratios))] for _ in ratios]; medians.append(statistics.median(sample))
    medians.sort(); lo = medians[max(0, int(.025 * len(medians)) - 1)]; hi = medians[min(len(medians) - 1, int(.975 * len(medians)))]
    return {"ratios": ratios, "median": statistics.median(ratios), "lower": lo, "upper": hi, "seed": seed, "iterations": iterations}


def decision_for(*, resource_pass: bool, budget_pass: bool, correctness_pass: bool, candidate_runnable: bool, paired: dict[str, Any], complete: bool = True) -> str:
    if not resource_pass or not budget_pass: return "resource_or_budget_failed"
    if not correctness_pass: return "correctness_failed"
    if not candidate_runnable: return "candidate_not_runnable"
    if not complete: return "incomplete_evidence"
    primary = paired.get("primary", paired)
    if primary.get("median", 2.0) <= .99 and primary.get("upper", 2.0) < 1.0: return "fused_greedy_compile_wins_exact_scope"
    if primary.get("median", 0.0) > 1.0 and primary.get("lower", 0.0) > 1.0: return "fused_greedy_compile_regression_baseline_retained"
    return "fused_greedy_compile_inconclusive"


def _write_exclusive(path: Path, value: dict[str, Any], mode: int) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(fd, "wb") as stream: fd = -1; stream.write(_canonical(value) + b"\n"); stream.flush(); os.fsync(stream.fileno())
    finally:
        if fd >= 0: os.close(fd)


def _atomic_result(value: dict[str, Any], *, replace: bool = False) -> None:
    if RESULT_PATH.exists() and not replace: raise StudyError("result already exists")
    if RESULT_PATH.is_symlink(): raise StudyError("result symlink is forbidden")
    ATTEMPT_DIR.mkdir(parents=True, exist_ok=True)
    temp = RESULT_PATH.with_name(f".{RESULT_PATH.name}.{os.getpid()}.tmp")
    if temp.exists() or temp.is_symlink(): raise StudyError("result temporary path already exists")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "wb") as stream:
            fd = -1; stream.write(_canonical(value) + b"\n"); stream.flush(); os.fsync(stream.fileno())
    finally:
        if fd >= 0: os.close(fd)
    os.replace(temp, RESULT_PATH)


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        try: process.kill()
        except OSError: pass


def _read_pipe(stream: Any, limit: int, result: dict[str, Any], key: str, process: subprocess.Popen[bytes]) -> None:
    data = bytearray()
    try:
        while True:
            chunk = stream.read(min(64 * 1024, limit + 1 - len(data)))
            if not chunk: break
            data.extend(chunk)
            if len(data) > limit:
                result[f"overflow_{key}"] = True
                _kill_process_group(process)
                break
    except Exception as exc:
        result["read_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    result[key] = bytes(data)


def _run_child(index: int, order: tuple[str, str], deadline: float, *, identity: dict[str, Any], git_revision: str, marker_token: str) -> dict[str, Any]:
    if deadline - time.monotonic() <= 0: raise WorkerError(f"worker {index} deadline exhausted before spawn")
    env = {key: value for key, value in os.environ.items() if key not in UNSAFE_ENV}; env.update(OFFLINE_ENV)
    env.update({AUTH_PREFIX + "PARENT_PID": str(os.getpid()), AUTH_PREFIX + "RUN_ID": RUN_ID, AUTH_PREFIX + "MODEL_KEY": MODEL_KEY, AUTH_PREFIX + "NONCE": AUTH_NONCE,
                AUTH_PREFIX + "BLOCK": str(index), AUTH_PREFIX + "ARM_ORDER": json.dumps(list(order)), AUTH_PREFIX + "PROTOCOL_VERSION": str(PROTOCOL_VERSION), AUTH_PREFIX + "PROTOCOL_SHA256": _module().PROTOCOL_SHA256,
                AUTH_PREFIX + "PREREG_SHA256": FROZEN_PREREGISTRATION_SHA256, AUTH_PREFIX + "PROMPT_SHA256": EXPECTED_PROMPT_SHA256, AUTH_PREFIX + "ENVIRONMENT_SHA256": environment_fingerprint(),
                AUTH_PREFIX + "SNAPSHOT_PATH": identity["snapshot_path"], AUTH_PREFIX + "SNAPSHOT_REVISION": MODEL_REVISION, AUTH_PREFIX + "SNAPSHOT_SHA256": identity["snapshot_sha256"],
                AUTH_PREFIX + "WEIGHT_SHA256": _canonical(identity["weight_sha256"]).decode(), AUTH_PREFIX + "SNAPSHOT_FILES_SHA256": _canonical(identity["snapshot_files_sha256"]).decode(),
                AUTH_PREFIX + "EXECUTION_FILES_SHA256": _canonical(identity["execution_files_sha256"]).decode(),
                AUTH_PREFIX + "SNAPSHOT_STAT_MANIFEST": _canonical(identity["execution_stat_manifest"]).decode(), AUTH_PREFIX + "GIT_REVISION": git_revision,
                AUTH_PREFIX + "MARKER_TOKEN": marker_token, AUTH_PREFIX + "CODE_FINGERPRINTS": _canonical(code_fingerprints()).decode(), AUTH_PREFIX + "CODE_SHA256": _sha256_bytes(_canonical(code_fingerprints()))})
    process = subprocess.Popen([sys.executable, str(WORKER), "--execute", "--model-key", MODEL_KEY], cwd=PROJECT_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    remaining = min(WORKER_TIMEOUT_SECONDS, deadline - time.monotonic())
    if remaining <= 0:
        _kill_process_group(process); raise WorkerError(f"worker {index} deadline exhausted")
    captured: dict[str, Any] = {}
    stdout_thread = threading.Thread(target=_read_pipe, args=(process.stdout, MAX_STDOUT_BYTES, captured, "stdout", process), daemon=True)
    stderr_thread = threading.Thread(target=_read_pipe, args=(process.stderr, MAX_STDERR_BYTES, captured, "stderr", process), daemon=True)
    stdout_thread.start(); stderr_thread.start()
    try: process.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        _kill_process_group(process); process.wait(timeout=max(0.01, deadline - time.monotonic())); raise WorkerError(f"worker {index} timeout")
    join_budget = deadline - time.monotonic()
    if join_budget <= 0: _kill_process_group(process); raise WorkerError(f"worker {index} reader deadline exhausted")
    stdout_thread.join(timeout=join_budget); stderr_thread.join(timeout=max(0.0, deadline - time.monotonic()))
    if stdout_thread.is_alive() or stderr_thread.is_alive(): _kill_process_group(process); raise WorkerError(f"worker {index} reader did not finish")
    stdout, stderr = captured.get("stdout", b""), captured.get("stderr", b"")
    if any(key in captured for key in ("overflow_stdout", "overflow_stderr", "read_error")) or len(stdout) > MAX_STDOUT_BYTES: raise WorkerError("worker output reader/cap failed")
    if process.returncode not in (0, 1): raise WorkerError(f"worker returncode {process.returncode}")
    event = _strict_json(stdout)
    event = _validate_event(event, index, order, identity=identity, git_revision=git_revision, marker_token=marker_token, expected_pid=process.pid)
    if process.returncode == 0 and event.get("status") != "complete": raise WorkerError("returncode 0 for terminal failure")
    if process.returncode == 1 and event.get("status") == "complete": raise WorkerError("returncode 1 for complete event")
    event["stderr_sha256"] = _sha256_bytes(stderr); event["process_returncode"] = process.returncode; return event


def _aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {"arms": {}}
    for arm in ARM_NAMES:
        values = [run["arms"][arm] for run in runs if run.get("status") == "complete"]
        metrics["arms"][arm] = {"decode_critical_path": summarize([item["decode_critical_path_ns"] / 1e9 for item in values]), "ttft": summarize([item["ttft_ns"] / 1e9 for item in values]), "token_rate": summarize([item["token_rate"] for item in values]), "rss_peak_bytes": max((item.get("rss_peak_bytes", 0) for run in runs for item in [run.get("arm_resources", {}).get(arm, {})]), default=0)}
    complete = [run for run in runs if run.get("status") == "complete" and set(run.get("arms", {})) == set(ARM_NAMES)]
    baseline = [run["arms"][ARM_NAMES[0]]["decode_critical_path_ns"] / 1e9 for run in complete]; candidate = [run["arms"][ARM_NAMES[1]]["decode_critical_path_ns"] / 1e9 for run in complete]
    metrics["paired"] = paired_bootstrap(baseline, candidate) if baseline else {}
    return metrics


def execute() -> dict[str, Any]:
    before = _evidence_state(); _validate_evidence_state(before)
    if before["marker"]["exists"] or before["result"]["exists"]: raise StudyError("existing marker/result blocks run")
    revision, dirty, power, snapshot_identity, swap_before = _preflight()
    if not WORKER.is_file() or WORKER.is_symlink(): raise StudyError("worker unavailable")
    if ATTEMPT_DIR.exists() or ATTEMPT_DIR.is_symlink():
        directory_metadata = ATTEMPT_DIR.lstat()
        if stat.S_ISLNK(directory_metadata.st_mode) or not stat.S_ISDIR(directory_metadata.st_mode): raise StudyError("marker directory path is unsafe")
    ATTEMPT_DIR.mkdir(parents=True, exist_ok=True); os.chmod(ATTEMPT_DIR, 0o700)
    metadata = ATTEMPT_DIR.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o700 or metadata.st_uid != os.geteuid(): raise StudyError("marker directory is not private")
    marker_token = secrets.token_urlsafe(32)
    _write_exclusive(ATTEMPT_PATH, {"study_id": STUDY_ID, "run_id": RUN_ID, "started_at_unix_ns": time.time_ns(), "formal_claim": False, "token_sha256": _sha256_bytes(marker_token.encode()), "code_fingerprints": code_fingerprints()}, 0o600)
    provenance = {"git_revision": revision, "dirty_state": dirty, "power_source": power, "snapshot": snapshot_identity, "swap_before_bytes": swap_before, "target": _target_info(), "code_fingerprints": code_fingerprints(), "environment_sha256": environment_fingerprint(), "preregistration_sha256": FROZEN_PREREGISTRATION_SHA256}
    _atomic_result({"schema_version": 1, "study_id": STUDY_ID, "run_id": RUN_ID, "candidate_id": CANDIDATE_ID, "formal_claim": False, "status": "running", "decision": "incomplete_evidence", "partial_result": True, "runs": [], "provenance": provenance}, replace=False)
    started = time.monotonic(); runs = []; error = None
    try:
        for index, order in enumerate(PAIR_ORDERS, 1):
            if time.monotonic() - started >= MAX_WALL_SECONDS: raise WorkerError("study wall deadline exceeded")
            try:
                runs.append(_run_child(index, order, started + MAX_WALL_SECONDS, identity=snapshot_identity, git_revision=revision, marker_token=marker_token))
                _atomic_result({"schema_version": 1, "study_id": STUDY_ID, "run_id": RUN_ID, "candidate_id": CANDIDATE_ID, "formal_claim": False, "status": "running", "decision": "incomplete_evidence", "partial_result": True, "runs": runs, "provenance": provenance, "checkpoint_pair": index}, replace=True)
                if runs[-1].get("status") != "complete": break
            except Exception as exc:
                error = {"type": type(exc).__name__, "message": str(exc)[:400]}
                _atomic_result({"schema_version": 1, "study_id": STUDY_ID, "run_id": RUN_ID, "candidate_id": CANDIDATE_ID, "formal_claim": False, "status": "resource_or_budget_failed", "decision": "resource_or_budget_failed", "partial_result": True, "runs": runs, "error": error, "provenance": provenance, "checkpoint_pair": index - 1}, replace=True)
                break
    finally:
        after = _evidence_state()
    complete = len(runs) == PAIR_COUNT and all(run.get("status") == "complete" for run in runs)
    if any(run.get("git_revision") not in (None, revision) for run in runs):
        error = error or {"type": "ProvenanceError", "message": "worker Git revision changed"}
    identity = complete and all(run["correctness"].get("pass") for run in runs)
    deterministic = complete and all(all(runs[0]["arms"][arm][field] == run["arms"][arm][field] for field in ("physical_token_sha256", "logical_token_sha256", "visible_token_sha256", "text_sha256")) for run in runs for arm in ARM_NAMES)
    metrics = _aggregate(runs) if runs else {"arms": {}, "paired": {}}
    paired = metrics.get("paired", {})
    decision = decision_for(resource_pass=error is None, budget_pass=error is None, correctness_pass=identity and deterministic, candidate_runnable=not any(run.get("status") == "candidate_not_runnable" for run in runs), paired=paired, complete=complete)
    statuses = {run.get("status") for run in runs}
    if "resource_or_budget_failed" in statuses: decision = "resource_or_budget_failed"
    elif "correctness_failed" in statuses: decision = "correctness_failed"
    elif "candidate_not_runnable" in statuses: decision = "candidate_not_runnable"
    elif "error" in statuses: decision = "incomplete_evidence"
    swap_after = _swap_used_bytes()
    if swap_after is None or swap_after != swap_before:
        decision = "resource_or_budget_failed"; error = error or {"type": "ResourceFailure", "message": "swap unavailable or changed"}
    try:
        if code_fingerprints() != provenance["code_fingerprints"] or _sha256_file(PREREGISTRATION) != FROZEN_PREREGISTRATION_SHA256:
            decision = "resource_or_budget_failed"; error = error or {"type": "ProvenanceError", "message": "code/specification changed postflight"}
        post_dirty = _git("status", "--porcelain", "--untracked-files=all", "--", ".", ":(exclude)ProjectAtlas", ":(exclude).friday-data")
        if post_dirty or _git("rev-parse", "HEAD") != revision or environment_fingerprint() != provenance["environment_sha256"] or _target_info() != provenance["target"]:
            decision = "resource_or_budget_failed"; error = error or {"type": "ProvenanceError", "message": "Git/environment/target changed postflight"}
        from _bench import require_ac_power
        if require_ac_power() != power:
            decision = "resource_or_budget_failed"; error = error or {"type": "ResourceFailure", "message": "power source changed postflight"}
    except Exception as exc:
        decision = "resource_or_budget_failed"; error = error or {"type": type(exc).__name__, "message": str(exc)[:300]}
    try:
        sys.path.insert(0, str(PROJECT_ROOT)); sys.path.insert(0, str(PROJECT_ROOT / "tools"))
        from _bench import resolve_local_model_snapshot
        if _snapshot_identity(resolve_local_model_snapshot(MODEL_ID)) != snapshot_identity:
            decision = "resource_or_budget_failed"; error = error or {"type": "ResourceFailure", "message": "snapshot changed postflight"}
    except Exception as exc:
        decision = "resource_or_budget_failed"; error = error or {"type": type(exc).__name__, "message": str(exc)[:300]}
    report = {"schema_version": 1, "study_id": STUDY_ID, "run_id": RUN_ID, "candidate_id": CANDIDATE_ID, "formal_claim": False, "status": decision, "decision": decision,
              "partial_result": not complete, "runs": runs, "metrics": metrics, "gates": {"resource_pass": error is None and decision != "resource_or_budget_failed", "budget_pass": error is None and decision != "resource_or_budget_failed", "correctness_pass": identity and deterministic, "candidate_runnable": not any(run.get("status") == "candidate_not_runnable" for run in runs), "all_pairs_completed": complete},
              "thresholds": {"median_ratio_max": .99, "bootstrap_upper_max": 1.0, "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_resamples": BOOTSTRAP_RESAMPLES, "no_outlier_removal": True}, "error": error,
              "provenance": {**provenance, "swap_after_bytes": swap_after, "swap_delta_bytes": None if swap_after is None else swap_after - swap_before, "model_id": MODEL_ID, "model_revision": MODEL_REVISION}, "completed_at_unix_ns": time.time_ns()}
    _atomic_result(report, replace=True)
    return report


def _self_check() -> int:
    before = _evidence_state(); _validate_evidence_state(before)
    assert summarize([1.0, 2.0, 3.0])["median"] == 2.0
    assert paired_bootstrap([1.0] * 6, [1.0] * 6)["upper"] == 1.0
    assert decision_for(resource_pass=False, budget_pass=True, correctness_pass=True, candidate_runnable=True, paired={}) == "resource_or_budget_failed"
    assert decision_for(resource_pass=True, budget_pass=True, correctness_pass=True, candidate_runnable=True, paired={"median": .9, "upper": .99}) == "fused_greedy_compile_wins_exact_scope"
    assert before == _evidence_state()
    print(json.dumps({"self_check": "pass", "checks": 5}, sort_keys=True)); return 0


def _show() -> int:
    before = _evidence_state(); _validate_evidence_state(before)
    if not RESULT_PATH.is_file(): print(json.dumps({"status": "unavailable", "study_id": STUDY_ID, "formal_claim": False}, sort_keys=True)); return 78
    value = _strict_json(RESULT_PATH.read_bytes()); after = _evidence_state()
    if before != after: raise StudyError("evidence changed")
    print(json.dumps({"study_id": value.get("study_id"), "run_id": value.get("run_id"), "decision": value.get("decision"), "formal_claim": False, "runs_completed": len(value.get("runs", []))}, sort_keys=True)); return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False); group = parser.add_mutually_exclusive_group(); group.add_argument("--execute", action="store_true"); group.add_argument("--self-check", action="store_true"); group.add_argument("--show", action="store_true"); args = parser.parse_args(argv)
    try:
        if args.self_check: return _self_check()
        if args.show: return _show()
        if not args.execute: print(json.dumps({"state": "not_released", "required_flag": "--execute", "formal_claim": False})); return 78
        report = execute(); print(json.dumps({"decision": report["decision"], "formal_claim": False, "pairs_completed": len(report["runs"])}, sort_keys=True)); return 0 if report["decision"] == "fused_greedy_compile_wins_exact_scope" else 1
    except Exception as exc:
        print(json.dumps({"state": "not_started_or_partial", "error_type": type(exc).__name__, "error": str(exc)[:500], "formal_claim": False}, sort_keys=True)); return 2


if __name__ == "__main__": raise SystemExit(main())
