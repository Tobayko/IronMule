#!/usr/bin/env python3
"""Q3d: one model-free, fail-closed stability gate before the Q3c run.

The parent of this module is deliberately standard-library-only.  Q3b/Q3c are
loaded as source modules only to reuse their existing preflight and OS probes;
neither module is allowed to load IronMule, MLX, a model, or an inference
process on this path.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping

EXPERIMENT_ID = "Q3d-model-free-stability-gate"
SCHEMA = "ironmule.q3d_stability_gate.v1"
MODEL_ID = "mlx-community/gemma-3-4b-it-4bit"
EXPECTED_REVISION = "93724907d4ed1745d2fe50baadf3b0b01a65abf2"
EXPECTED_MODEL_MANIFEST_SHA256 = "a405b1a73ee9fac816ed7cfeab45b70a26f031843467a4aa4030edc663e857ae"
SCHEDULED_SAMPLES = 60
SAMPLE_COUNT = SCHEDULED_SAMPLES + 1
SAMPLE_INTERVAL_SECONDS = 1.0
GATE_DEADLINE_SECONDS = 90.0
COMMAND_TIMEOUT_SECONDS = 1.0
OUTPUT_CAP_BYTES = 512 * 1024
Q3C_STUDY_MAX_SECONDS = 600.0
Q3C_WRAPPER_TIMEOUT_SECONDS = 630.0
CAPTURED_OUTPUT_BYTES = 8192
TERMINAL_RESERVE_SECONDS = 30.0
OUTER_MAX_SECONDS = GATE_DEADLINE_SECONDS + Q3C_STUDY_MAX_SECONDS + TERMINAL_RESERVE_SECONDS
MIN_FIRST_LAST_SECONDS = 60.0
MAX_FIRST_LAST_SECONDS = 62.5
MAX_ADJACENT_GAP_SECONDS = 2.5
START_SWAP_LIMIT_BYTES = 4 * 1024**3
START_FREE_PERCENT = 35
LOAD_MAX = 8.0
LOAD_SPREAD_MAX = 2.0
ALLOWED_UNTRACKED = frozenset({"research/data/squad-dev-v1.1.json"})
PREREGISTRATION = Path(__file__).resolve().parent / "raw" / "Q3d_preregistration.md"
PREREGISTRATION_SHA = Path(__file__).resolve().parent / "raw" / "Q3d_preregistration.sha256"


class Q3dRefused(RuntimeError):
    """Raised when a required stability or identity fact is unknown."""


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_code_sha256(root: Path | None = None) -> str:
    """Hash the complete runtime surface bound by the Q3d record."""
    root = Path(__file__).resolve().parents[1] if root is None else root.resolve()
    paths = list((root / "ironmule").rglob("*.py"))
    paths.extend(root / "research" / name for name in (
        "q3b_residual_swap_canary.py", "q3c_performance_replication.py",
        "q3d_stability_gate.py"))
    digest = hashlib.sha256()
    for path in sorted({path.resolve() for path in paths if path.is_file()},
                       key=lambda item: item.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _preregistration_matches() -> bool:
    try:
        fields = PREREGISTRATION_SHA.read_text().strip().split()
        return (PREREGISTRATION.exists() and len(fields) == 2
                and re.fullmatch(r"[0-9a-f]{64}", fields[0]) is not None
                and _sha256(PREREGISTRATION) == fields[0]
                and fields[1] == PREREGISTRATION.name)
    except (OSError, IndexError):
        return False


def _load_source_module(name: str, path: Path) -> Any:
    if not path.is_file():
        raise Q3dRefused(f"required policy module is unavailable: {name}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise Q3dRefused(f"required policy module cannot be loaded: {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_q3b(root: Path) -> Any:
    return _load_source_module("ironmule_q3d_q3b_policy", root / "research" / "q3b_residual_swap_canary.py")


def _load_q3c(root: Path) -> Any:
    return _load_source_module("ironmule_q3d_q3c_policy", root / "research" / "q3c_performance_replication.py")


def _untracked_inventory(root: Path, q3b: Any, bounded: Callable[[list[str]], str]) -> dict[str, Any]:
    try:
        output = bounded([q3b.COMMANDS["git"], "-C", str(root), "ls-files", "--others", "--exclude-standard"])
    except BaseException as exc:
        return {"passed": False, "paths": [], "unexpected": [],
                "reason": f"untracked inventory failed: {type(exc).__name__}"}
    if not isinstance(output, str) or not getattr(output, "ok", True):
        return {"passed": False, "paths": [], "unexpected": [], "reason": "untracked inventory unavailable"}
    paths = [line for line in output.splitlines() if line]
    unexpected = sorted(set(paths) - ALLOWED_UNTRACKED)
    return {"passed": not unexpected, "paths": paths, "unexpected": unexpected,
            "allowed": sorted(ALLOWED_UNTRACKED)}


def _valid_load(load: Any) -> bool:
    if not isinstance(load, Mapping) or load.get("passed") is not True:
        return False
    samples = load.get("samples")
    if not (isinstance(samples, list) and len(samples) == 3
            and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(float(value)) and value >= 0 for value in samples)
            and isinstance(load.get("max"), (int, float))
            and not isinstance(load.get("max"), bool)
            and math.isfinite(float(load.get("max")))
            and isinstance(load.get("spread"), (int, float))
            and not isinstance(load.get("spread"), bool)
            and math.isfinite(float(load.get("spread")))):
        return False
    expected_max, expected_spread = max(samples), max(samples) - min(samples)
    return (load["max"] == expected_max and load["spread"] == expected_spread
            and expected_max <= LOAD_MAX and expected_spread <= LOAD_SPREAD_MAX)


def preflight(*, root: Path | None = None, deadline: float | None = None,
              q3b_module: Any | None = None, q3c_module: Any | None = None) -> dict[str, Any]:
    """Reuse Q3c/Q3b preflight without importing MLX or starting a model."""
    root = Path(__file__).resolve().parents[1] if root is None else root.resolve()
    q3b = _load_q3b(root) if q3b_module is None else q3b_module
    q3c = _load_q3c(root) if q3c_module is None else q3c_module
    try:
        base = q3c.preflight(root=root, deadline=deadline)
    except BaseException as exc:
        return {"passed": False, "checks": {}, "error": f"preflight failed: {type(exc).__name__}"}
    environment = base.get("environment") if isinstance(base, Mapping) else None
    identity = base.get("identity") if isinstance(base, Mapping) else None
    git = base.get("git") if isinstance(base, Mapping) else None
    untracked = base.get("untracked") if isinstance(base, Mapping) else None
    load = base.get("loadavg") if isinstance(base, Mapping) else None
    memory = base.get("installed_memory_bytes") if isinstance(base, Mapping) else None
    commit = git.get("commit") if isinstance(git, Mapping) else None
    checks = {
        "ac_power": isinstance(environment, Mapping) and environment.get("power_source") == "AC",
        "low_power_off": isinstance(environment, Mapping) and environment.get("low_power_mode") is False,
        "thermal_nominal": isinstance(environment, Mapping) and environment.get("thermal_state") == "nominal",
        "start_memory_free_at_least_35_percent": isinstance(environment, Mapping)
            and type(environment.get("memory_free_percent")) is int
            and environment["memory_free_percent"] >= START_FREE_PERCENT,
        "start_swap_known": isinstance(environment, Mapping) and type(environment.get("swap_used_bytes")) is int
            and environment["swap_used_bytes"] >= 0,
        "start_swap_within_4gib": isinstance(environment, Mapping)
            and type(environment.get("swap_used_bytes")) is int
            and environment["swap_used_bytes"] <= START_SWAP_LIMIT_BYTES,
        "loadavg_gate": _valid_load(load),
        "no_competing_model_process": isinstance(base, Mapping)
            and isinstance(base.get("checks"), Mapping)
            and base["checks"].get("no_competing_model_process") is True,
        "model_cache_identity_exact": isinstance(identity, Mapping)
            and identity.get("model_id") == MODEL_ID
            and identity.get("model_revision") == EXPECTED_REVISION
            and identity.get("model_manifest_sha256") == EXPECTED_MODEL_MANIFEST_SHA256,
        "git_commit_known": isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit) is not None,
        "git_clean_and_bound": isinstance(git, Mapping) and git.get("clean") is True
            and isinstance(commit, str) and bool(commit),
        "untracked_runtime_clean": isinstance(untracked, Mapping) and untracked.get("passed") is True,
        "preregistration_matches": _preregistration_matches(),
        "runtime_code_hash_known": bool(re.fullmatch(r"[0-9a-f]{64}", runtime_code_sha256(root))),
        "installed_memory_known": type(memory) is int and memory > 0,
    }
    return {"environment": environment, "identity": identity,
            "model_cache_identity": identity, "git": git, "untracked": untracked,
            "loadavg": load, "installed_memory_bytes": memory,
            "git_commit": commit, "runtime_code_sha256": runtime_code_sha256(root),
            "checks": checks, "passed": all(checks.values())}


def _preflight_is_complete(value: Any) -> bool:
    """Do not let an injected or stale ``passed`` flag bypass Q3d gates."""
    if not isinstance(value, Mapping) or value.get("passed") is not True:
        return False
    checks = value.get("checks")
    required_checks = _preflight_check_names()
    if (not isinstance(checks, Mapping) or set(checks) != required_checks
            or not all(item is True for item in checks.values())):
        return False
    environment = value.get("environment")
    identity = value.get("model_cache_identity", value.get("identity"))
    git = value.get("git")
    commit = value.get("git_commit", git.get("commit") if isinstance(git, Mapping) else None)
    return bool(
        isinstance(environment, Mapping)
        and environment.get("power_source") == "AC"
        and environment.get("low_power_mode") is False
        and environment.get("thermal_state") == "nominal"
        and type(environment.get("memory_free_percent")) is int
        and environment["memory_free_percent"] >= START_FREE_PERCENT
        and type(environment.get("swap_used_bytes")) is int
        and 0 <= environment["swap_used_bytes"] <= START_SWAP_LIMIT_BYTES
        and isinstance(identity, Mapping)
        and identity.get("model_id") == MODEL_ID
        and identity.get("model_revision") == EXPECTED_REVISION
        and identity.get("model_manifest_sha256") == EXPECTED_MODEL_MANIFEST_SHA256
        and isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit) is not None
        and isinstance(git, Mapping) and git.get("clean") is True and git.get("commit") == commit
        and _valid_load(value.get("loadavg"))
        and isinstance(value.get("untracked"), Mapping) and value["untracked"].get("passed") is True
        and set(value["untracked"].get("unexpected", [])) == set()
        and set(value["untracked"].get("paths", [])) <= ALLOWED_UNTRACKED
        and re.fullmatch(r"[0-9a-f]{64}", str(value.get("runtime_code_sha256", ""))) is not None
        and type(value.get("installed_memory_bytes")) is int and value["installed_memory_bytes"] > 0
        and value.get("preregistration_matches", value.get("checks", {}).get("preregistration_matches")) is True
    )


def _preflight_check_names() -> frozenset[str]:
    return frozenset({
        "ac_power", "low_power_off", "thermal_nominal",
        "start_memory_free_at_least_35_percent", "start_swap_known", "start_swap_within_4gib",
        "loadavg_gate", "no_competing_model_process", "model_cache_identity_exact",
        "git_commit_known", "git_clean_and_bound", "untracked_runtime_clean",
        "preregistration_matches", "runtime_code_hash_known", "installed_memory_known",
    })


def _decode_swap_result(raw_value: Any) -> tuple[int, Mapping[str, Any]]:
    metadata = raw_value if isinstance(raw_value, Mapping) else {"value": raw_value}
    value = metadata.get("value")
    output_length = metadata.get("output_length", len(str(value)))
    if (type(value) is not int or value < 0 or metadata.get("result_ok", True) is not True
            or metadata.get("returncode", 0) != 0
            or type(output_length) is not int or output_length < 0 or output_length > OUTPUT_CAP_BYTES):
        raise Q3dRefused("swap command result is unknown or malformed")
    return value, metadata


def run_stability_gate(preflight_result: Mapping[str, Any], *, root: Path | None = None,
                       q3b_module: Any | None = None, clock: Callable[[], float] = time.monotonic,
                       sleeper: Callable[[float], None] = time.sleep,
                       swap_reader: Callable[[], Any] | None = None,
                       environment_reader: Callable[[], Mapping[str, Any]] | None = None,
                       process_checker: Callable[[], Any] | None = None,
                       load_reader: Callable[[], Mapping[str, Any]] | None = None,
                       deadline: float | None = None) -> dict[str, Any]:
    """Take exactly one t0 sample and 60 samples on the frozen 1-second grid."""
    root = Path(__file__).resolve().parents[1] if root is None else root.resolve()
    q3b = _load_q3b(root) if q3b_module is None else q3b_module
    started = clock()
    gate_deadline = started + GATE_DEADLINE_SECONDS if deadline is None else deadline
    if not _preflight_is_complete(preflight_result):
        return {"schema": "ironmule.q3d_stability_evidence.v1", "t0_monotonic": started,
                "samples": [], "sample_times": [], "sample_offsets": [], "commands": [],
                "elapsed_seconds": None, "adjacent_gaps_seconds": [], "start_swap_bytes": None,
                "max_swap_bytes": None, "highwater_delta_bytes": None, "post_environment": None,
                "post_loadavg": None, "post_competing_model_process": None,
                "errors": ["preflight evidence incomplete or failed"], "passed": False}
    run = getattr(q3b, "_run_text", None)
    if swap_reader is None:
        if not callable(run):
            raise Q3dRefused("swap command runner unavailable")
        bounded = q3b._deadline_runner(gate_deadline, run)
        def default_swap_reader():
            output = bounded([q3b.COMMANDS["sysctl"], "-n", "vm.swapusage"])
            if not isinstance(output, str) or not getattr(output, "ok", True):
                raise Q3dRefused("swap command failed")
            value = q3b._swap_bytes(output)
            if type(value) is not int or value < 0:
                raise Q3dRefused("swap command output malformed")
            return {"value": value, "result_ok": True, "output_length": len(output)}
        swap_reader = default_swap_reader
    if environment_reader is None:
        if not callable(run):
            raise Q3dRefused("environment command runner unavailable")
        environment_reader = lambda: q3b.system_environment(q3b._deadline_runner(gate_deadline, run))
    if process_checker is None:
        if not callable(run):
            raise Q3dRefused("process inventory runner unavailable")
        process_checker = lambda: q3b.competing_model_process(q3b._deadline_runner(gate_deadline, run))
    if load_reader is None:
        if not callable(run):
            raise Q3dRefused("load runner unavailable")
        load_reader = lambda: q3b.loadavg_gate(deadline=gate_deadline)

    samples: list[int] = []
    times: list[float] = []
    offsets: list[float] = []
    commands: list[dict[str, Any]] = []
    errors: list[str] = []
    t0: float | None = None
    try:
        command_start = clock()
        raw_value = swap_reader()
        command_end = clock()
        value, metadata = _decode_swap_result(raw_value)
        t0 = command_end
        samples.append(value); times.append(t0); offsets.append(0.0)
        commands.append({"index": 0, "target_offset_seconds": 0.0,
                         "command_start_monotonic": command_start,
                         "command_end_monotonic": command_end,
                         "command_duration_seconds": command_end - command_start,
                         "timeout_seconds": COMMAND_TIMEOUT_SECONDS,
                         "returncode": metadata.get("returncode", 0),
                         "result_ok": metadata.get("result_ok", True) is True,
                         "known": True, "output_length": metadata.get("output_length", len(str(value))),
                         "value": value})
    except BaseException as exc:
        errors.append(f"sample-0:{type(exc).__name__}")
        observed = clock()
        t0 = observed
        commands.append({"index": 0, "target_offset_seconds": 0.0,
                         "command_start_monotonic": observed, "command_end_monotonic": observed,
                         "command_duration_seconds": 0.0, "timeout_seconds": COMMAND_TIMEOUT_SECONDS,
                         "returncode": -1,
                         "result_ok": False, "known": False, "output_length": 0,
                         "value": None, "error": str(exc) or type(exc).__name__})
    for index in range(1, SCHEDULED_SAMPLES + 1):
        target = t0 + index * SAMPLE_INTERVAL_SECONDS
        if errors:
            break
        if clock() >= gate_deadline:
            errors.append("gate deadline exhausted before scheduled sample")
            break
        try:
            remaining = target - clock()
            if remaining > 0:
                sleeper(remaining)
            if clock() >= gate_deadline:
                raise Q3dRefused("gate deadline exhausted before scheduled sample")
            command_start = clock()
            raw_value = swap_reader()
            command_end = clock()
            value, metadata = _decode_swap_result(raw_value)
            observed = command_end
            samples.append(value); times.append(observed); offsets.append(observed - t0)
            commands.append({"index": index, "target_offset_seconds": float(index),
                             "command_start_monotonic": command_start,
                             "command_end_monotonic": command_end,
                             "command_duration_seconds": command_end - command_start,
                             "timeout_seconds": COMMAND_TIMEOUT_SECONDS,
                             "returncode": metadata.get("returncode", 0),
                             "result_ok": metadata.get("result_ok", True) is True,
                             "known": True, "output_length": metadata.get("output_length", len(str(value))),
                             "value": value})
        except BaseException as exc:
            errors.append(f"sample-{index}:{type(exc).__name__}")
            observed = clock()
            commands.append({"index": index, "target_offset_seconds": float(index),
                             "command_start_monotonic": observed, "command_end_monotonic": observed,
                             "command_duration_seconds": 0.0, "timeout_seconds": COMMAND_TIMEOUT_SECONDS,
                             "returncode": -1,
                             "result_ok": False, "known": False, "output_length": 0,
                             "value": None, "error": str(exc) or type(exc).__name__})
            break
    post: Mapping[str, Any] | None = None
    post_load: Mapping[str, Any] | None = None
    post_competing: Any = None
    if not errors and len(samples) == SAMPLE_COUNT:
        try:
            post = environment_reader()
            post_load = load_reader()
            post_competing = process_checker()
        except BaseException as exc:
            errors.append(f"post-environment:{type(exc).__name__}")
    elapsed = times[-1] - times[0] if len(times) >= 2 else None
    gaps = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    start_swap = samples[0] if samples else None
    highwater = max(samples) if samples else None
    highwater_delta = highwater - start_swap if isinstance(highwater, int) and isinstance(start_swap, int) else None
    temporal = (len(samples) == SAMPLE_COUNT and isinstance(elapsed, (int, float))
                and math.isfinite(float(elapsed)) and MIN_FIRST_LAST_SECONDS <= elapsed <= MAX_FIRST_LAST_SECONDS
                and all(math.isfinite(float(gap)) and 0 <= gap <= MAX_ADJACENT_GAP_SECONDS for gap in gaps))
    post_known = (isinstance(post, Mapping)
                  and all(post.get(key) is not None for key in ("power_source", "low_power_mode", "thermal_state", "swap_used_bytes", "memory_free_percent"))
                  and post.get("power_source") == "AC" and post.get("low_power_mode") is False
                  and post.get("thermal_state") == "nominal"
                  and type(post.get("memory_free_percent")) is int and post["memory_free_percent"] >= START_FREE_PERCENT
                  and type(post.get("swap_used_bytes")) is int and 0 <= post["swap_used_bytes"] <= START_SWAP_LIMIT_BYTES
                  and _valid_load(post_load)
                  and post_competing is None)
    passed = (not errors and len(samples) == SAMPLE_COUNT and temporal
              and highwater_delta == 0 and post_known)
    if not temporal and len(samples) == SAMPLE_COUNT:
        errors.append("sample timing or adjacent gap bound failed")
    if highwater_delta != 0:
        errors.append("swap high-water increase was not exactly zero")
    if not post_known:
        errors.append("post environment or process state is unknown")
    return {"schema": "ironmule.q3d_stability_evidence.v1", "t0_monotonic": t0,
            "samples": samples, "sample_times": times, "sample_offsets": offsets,
            "commands": commands, "elapsed_seconds": elapsed, "adjacent_gaps_seconds": gaps,
            "start_swap_bytes": start_swap, "max_swap_bytes": highwater,
            "highwater_delta_bytes": highwater_delta, "post_environment": post,
            "post_loadavg": post_load,
            "post_competing_model_process": post_competing, "errors": errors,
            "passed": passed}


def validate_stability_evidence(value: Any) -> tuple[bool, str]:
    required = {"schema", "t0_monotonic", "samples", "sample_times", "sample_offsets", "commands",
                "elapsed_seconds", "adjacent_gaps_seconds", "start_swap_bytes", "max_swap_bytes",
                "highwater_delta_bytes", "post_environment", "post_loadavg",
                "post_competing_model_process", "errors", "passed"}
    if not isinstance(value, dict) or set(value) != required or value["schema"] != "ironmule.q3d_stability_evidence.v1":
        return False, "evidence schema mismatch"
    samples, times, offsets, commands = value["samples"], value["sample_times"], value["sample_offsets"], value["commands"]
    if (not isinstance(samples, list) or len(samples) != SAMPLE_COUNT
            or not isinstance(times, list) or len(times) != SAMPLE_COUNT
            or not isinstance(offsets, list) or len(offsets) != SAMPLE_COUNT
            or not isinstance(commands, list) or len(commands) != SAMPLE_COUNT):
        return False, "sample count incomplete"
    if (any(type(item) is not int or item < 0 for item in samples)
            or any(not isinstance(item, (int, float)) or isinstance(item, bool) or not math.isfinite(float(item)) for item in times + offsets)
            or any(times[i + 1] < times[i] or offsets[i + 1] < offsets[i] for i in range(SAMPLE_COUNT - 1))
            or offsets[0] != 0.0
            or any(times[i + 1] - times[i] > MAX_ADJACENT_GAP_SECONDS for i in range(SAMPLE_COUNT - 1))
            or times[0] != value["t0_monotonic"]
            or offsets != [timestamp - value["t0_monotonic"] for timestamp in times]
            or value["elapsed_seconds"] != times[-1] - times[0]
            or value["adjacent_gaps_seconds"] != [times[i + 1] - times[i] for i in range(SAMPLE_COUNT - 1)]):
        return False, "sample values or gaps invalid"
    if (not isinstance(value["t0_monotonic"], (int, float)) or isinstance(value["t0_monotonic"], bool) or not math.isfinite(float(value["t0_monotonic"]))
            or not isinstance(value["elapsed_seconds"], (int, float)) or not math.isfinite(float(value["elapsed_seconds"]))
            or value["elapsed_seconds"] < MIN_FIRST_LAST_SECONDS or value["elapsed_seconds"] > MAX_FIRST_LAST_SECONDS
            or value["start_swap_bytes"] != samples[0] or value["max_swap_bytes"] != max(samples)
            or value["highwater_delta_bytes"] != 0 or not isinstance(value["post_environment"], Mapping)
            or value["post_environment"].get("power_source") != "AC"
            or value["post_environment"].get("low_power_mode") is not False
            or value["post_environment"].get("thermal_state") != "nominal"
            or type(value["post_environment"].get("memory_free_percent")) is not int
            or value["post_environment"]["memory_free_percent"] < START_FREE_PERCENT
            or type(value["post_environment"].get("swap_used_bytes")) is not int
            or not (0 <= value["post_environment"]["swap_used_bytes"] <= START_SWAP_LIMIT_BYTES)
            or not _valid_load(value["post_loadavg"])
            or value["post_competing_model_process"] is not None
            or value["errors"] != [] or value["passed"] is not True):
        return False, "stability decision evidence invalid"
    for index, command in enumerate(commands):
        if (not isinstance(command, dict) or set(command) != {"index", "target_offset_seconds", "command_start_monotonic", "command_end_monotonic", "command_duration_seconds", "timeout_seconds", "returncode", "result_ok", "known", "output_length", "value"}
                or command["index"] != index or command["target_offset_seconds"] != float(index)
                or command["known"] is not True or command["result_ok"] is not True or command["value"] != samples[index]
                or type(command["returncode"]) is not int or command["returncode"] != 0
                or type(command["output_length"]) is not int or command["output_length"] < 0
                or command["output_length"] > OUTPUT_CAP_BYTES
                or command["timeout_seconds"] != COMMAND_TIMEOUT_SECONDS
                or not all(isinstance(command[key], (int, float)) and not isinstance(command[key], bool)
                           and math.isfinite(float(command[key])) for key in ("command_start_monotonic", "command_end_monotonic", "command_duration_seconds"))
                or command["command_end_monotonic"] < command["command_start_monotonic"]
                or command["command_duration_seconds"] != command["command_end_monotonic"] - command["command_start_monotonic"]
                or command["command_duration_seconds"] > COMMAND_TIMEOUT_SECONDS):
            return False, "command evidence invalid"
    return True, "ok"


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
        os.chmod(path, 0o600)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def plan(root: Path | None = None) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1] if root is None else root.resolve()
    return {"schema": SCHEMA, "experiment": EXPERIMENT_ID, "model": MODEL_ID,
            "revision": EXPECTED_REVISION, "sample_count": SAMPLE_COUNT,
            "scheduled_samples": SCHEDULED_SAMPLES, "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
            "gate_deadline_seconds": GATE_DEADLINE_SECONDS, "command_timeout_seconds": COMMAND_TIMEOUT_SECONDS,
            "terminal_reserve_seconds": TERMINAL_RESERVE_SECONDS,
            "outer_max_seconds": OUTER_MAX_SECONDS,
            "first_last_elapsed_seconds": [MIN_FIRST_LAST_SECONDS, MAX_FIRST_LAST_SECONDS],
            "adjacent_gap_max_seconds": MAX_ADJACENT_GAP_SECONDS,
            "start_swap_limit_bytes": START_SWAP_LIMIT_BYTES, "start_free_percent": START_FREE_PERCENT,
            "load_max": LOAD_MAX, "load_spread_max": LOAD_SPREAD_MAX,
            "output_cap_bytes": OUTPUT_CAP_BYTES, "promotion_allowed": False,
            "q3c_study_max_seconds": Q3C_STUDY_MAX_SECONDS,
            "runtime_code_sha256": runtime_code_sha256(root)}


def _expected_q3c_plan(q3c: Any, root: Path) -> dict[str, Any]:
    return {"schema": "ironmule.q3c_plan.v1", "experiment": q3c.EXPERIMENT_ID,
            "model": q3c.MODEL_ID, "revision": q3c.EXPECTED_REVISION,
            "phases": [q3c.phase_plan(phase) for phase in q3c.PHASES],
            "study_max_seconds": q3c.STUDY_MAX_SECONDS,
            "phase_max_seconds": q3c.PHASE_MAX_SECONDS,
            "post_phase_seconds": q3c.POST_PHASE_SECONDS,
            "final_reserve_seconds": q3c.FINAL_RESERVE_SECONDS,
            "worker_max_seconds": q3c.WORKER_MAX_SECONDS,
            "child_timeout_seconds": q3c.CHILD_TIMEOUT_SECONDS,
            "bootstrap_resamples": q3c.BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": q3c.BOOTSTRAP_SEED,
            "sampler_interval_seconds": q3c.SAMPLE_INTERVAL_SECONDS,
            "max_sampler_gap_seconds": q3c.MAX_SWAP_SAMPLE_GAP_SECONDS,
            "max_sampler_samples": q3c.MAX_SWAP_SAMPLES,
            "start_swap_limit_bytes": q3c.START_SWAP_LIMIT_BYTES,
            "swap_delta_limit_bytes": q3c.SWAP_DELTA_LIMIT_BYTES,
            "start_free_percent": q3c.START_FREE_PERCENT,
            "after_free_percent": q3c.AFTER_FREE_PERCENT,
            "peak_ceiling_fraction": q3c.PEAK_CEILING_FRACTION,
            "prompt_tokens": q3c.PROMPT_TOKENS, "max_tokens": q3c.MAX_TOKENS,
            "command_timeout_seconds": q3c.COMMAND_TIMEOUT_SECONDS,
            "capability_max_bytes": q3c.CAPABILITY_MAX_BYTES,
            "worker_output_max_bytes": q3c.MAX_WORKER_OUTPUT,
            "promotion_allowed": False}


def _strict_q3c_result(raw: Any, *, root: Path, expected_commit: str,
                       expected_identity: Mapping[str, Any], q3c: Any) -> tuple[bool, str, dict[str, Any] | None]:
    """Accept Q3c only when its complete raw evidence passes every hard gate."""
    if not isinstance(raw, dict) or raw.get("schema") != "ironmule.q3c_result.v1":
        return False, "q3c result schema mismatch", None
    if (raw.get("experiment") != q3c.EXPERIMENT_ID or raw.get("fallback") != "BASE/current incumbent"
            or raw.get("promotion_allowed") is not False or raw.get("status") != "COMPLETE_PASS"):
        return False, "q3c status or safety contract mismatch", None
    try:
        if raw.get("preregistration_sha256") != q3c._sha256(q3c.PREREGISTRATION) or not q3c._preregistration_matches():
            return False, "q3c preregistration binding mismatch", None
        if not callable(getattr(q3c, "_strict_equal", None)) \
                or not q3c._strict_equal(raw.get("plan"), _expected_q3c_plan(q3c, root)):
            return False, "q3c frozen plan mismatch", None
    except (AttributeError, OSError, TypeError, ValueError):
        return False, "q3c plan or preregistration unavailable", None
    pre = raw.get("preflight")
    if not isinstance(pre, Mapping) or pre.get("passed") is not True:
        return False, "q3c preflight did not pass", None
    git = pre.get("git")
    if not isinstance(git, Mapping) or git.get("commit") != expected_commit or git.get("clean") is not True:
        return False, "q3c commit binding mismatch", None
    identity = pre.get("identity")
    if (not isinstance(expected_identity, Mapping)
            or expected_identity.get("model_id") != MODEL_ID
            or expected_identity.get("model_revision") != EXPECTED_REVISION
            or expected_identity.get("model_manifest_sha256") != EXPECTED_MODEL_MANIFEST_SHA256):
        return False, "expected model identity is not the pinned Gemma snapshot", None
    if not isinstance(identity, Mapping) or dict(identity) != dict(expected_identity):
        return False, "q3c model identity mismatch", None
    runtime_hash = q3c.runtime_code_sha256(root)
    if pre.get("runtime_code_sha256") != runtime_hash:
        return False, "q3c runtime hash mismatch", None
    phases = raw.get("phases")
    if not isinstance(phases, list) or len(phases) != 2 or [item.get("phase") for item in phases if isinstance(item, Mapping)] != list(q3c.PHASES):
        return False, "q3c phases incomplete", None
    binding = {**dict(expected_identity), "runtime_code_sha256": runtime_hash}
    for phase in phases:
        if not isinstance(phase, Mapping) or "failure" in phase:
            return False, "q3c phase failure present", None
        try:
            valid, reason = q3c.validate_phase_result(dict(phase), phase["phase"], binding)
        except BaseException as exc:
            return False, f"q3c phase validation failed: {type(exc).__name__}", None
        if not valid:
            return False, f"q3c phase invalid: {reason}", None
    resources = raw.get("resource_history")
    if (not isinstance(resources, list) or len(resources) != 3
            or [item.get("phase") for item in resources if isinstance(item, Mapping)] != ["R", "N", "FINAL"]
            or any(not isinstance(item, Mapping) or not isinstance(item.get("gate"), Mapping)
                   or item["gate"].get("passed") is not True for item in resources)):
        return False, "q3c resource history gate failed", None
    phase_gates = raw.get("phase_gates")
    if (not isinstance(phase_gates, list) or len(phase_gates) != 1
            or phase_gates[0].get("phase") != "R"
            or not isinstance(phase_gates[0].get("performance"), Mapping)
            or phase_gates[0]["performance"].get("passed") is not True):
        return False, "q3c performance gate failed", None
    decision = raw.get("decision")
    identities = raw.get("cross_phase_identity")
    if (not isinstance(decision, Mapping) or decision.get("status") != "COMPLETE_PASS"
            or decision.get("promotion_allowed") is not False
            or decision.get("phase_R_reproduced") is not True
            or decision.get("phase_N_preserved") is not True
            or not isinstance(identities, Mapping) or not identities or not all(item is True for item in identities.values())):
        return False, "q3c decision or cross-phase identity failed", None
    return True, "ok", {"status": raw["status"], "model_identity": dict(identity),
                         "runtime_code_sha256": runtime_hash, "decision_status": decision["status"]}


def _cleanup_nested_q3c_workers(q3b: Any, uid: int, run: Callable[[list[str]], str],
                                baseline: Mapping[str, Any] | None = None,
                                wrapper_pid: int | None = None) -> dict[str, Any]:
    """Best-effort cleanup for Q3c phase-worker sessions after wrapper timeout."""
    first = q3b._cleanup_ps_snapshot(run)
    if (first.get("command_ok") is not True or first.get("parse_ok") is not True
            or getattr(q3b, "_cleanup_comm_map", lambda _snapshot: None)(first) is None):
        return {"known": False, "groups": [], "errors": ["nested worker inventory unknown"]}
    required_fields = {"pid", "ppid", "pgid", "uid", "stat", "start", "args"}
    allowed_fields = (required_fields, required_fields | {"sid"})
    if any(not isinstance(row, Mapping) or set(row) not in allowed_fields
           or (row.get("uid") == uid and ("sid" not in row or type(row.get("sid")) is not int))
           for row in first.get("records", [])):
        return {"known": False, "groups": [], "errors": ["nested worker inventory schema unknown"]}
    if (not isinstance(baseline, Mapping) or baseline.get("valid") is not True
            or not isinstance(baseline.get("identities"), list)
            or any(not isinstance(item, Mapping) or set(item) != {"pid", "start", "uid", "sid", "pgid"}
                   or type(item.get("pid")) is not int or item["pid"] <= 0
                   or type(item.get("uid")) is not int or item["uid"] < 0
                   or type(item.get("sid")) is not int or item["sid"] < 0
                   or type(item.get("pgid")) is not int or item["pgid"] < 0
                   or not isinstance(item.get("start"), str) or not item["start"]
                   for item in baseline["identities"])):
        return {"known": False, "groups": [], "errors": ["nested worker baseline unknown"]}
    baseline_pairs = {(item.get("pid"), item.get("start"), item.get("uid"), item.get("sid"), item.get("pgid")) for item in baseline["identities"]
                      if isinstance(item, Mapping)}
    rows = [row for row in first.get("records", [])
            if row.get("uid") == uid and "q3c_performance_replication.py" in row.get("args", "")
            and "--phase-worker" in row.get("args", "")
            and (wrapper_pid is None or row.get("ppid") == wrapper_pid)
            and (row.get("pid"), row.get("start"), row.get("uid"), row.get("sid"), row.get("pgid")) not in baseline_pairs]
    comm_map = q3b._cleanup_comm_map(first)
    if comm_map is None or any("python" not in comm_map.get(row.get("pid"), {}).get("comm", "").casefold()
                               for row in rows):
        return {"known": False, "groups": [], "remaining": [], "errors": ["nested worker comm identity unknown"]}
    candidate_keys = {(row.get("pid"), row.get("start"), row.get("uid"), row.get("sid"), row.get("pgid")) for row in rows}
    groups: list[dict[str, Any]] = []
    errors: list[str] = []
    original_starts = {row.get("pid"): row.get("start") for row in rows}
    for row in rows:
        members = [item for item in first["records"] if item.get("pgid") == row.get("pgid")]
        if (row.get("pgid") != row.get("sid") or any(item.get("uid") != uid for item in members)
                or any(item.get("start") == "" for item in members)
                or any(item.get("pid") not in original_starts or item.get("start") != original_starts[item.get("pid")]
                       for item in members)):
            errors.append(f"nested group identity unsafe: {row.get('pid')}")
            continue
        attempt = {"pid": row["pid"], "pgid": row["pgid"], "start": row["start"], "status": "sent"}
        try:
            os.killpg(row["pgid"], signal.SIGTERM)
        except ProcessLookupError:
            attempt["status"] = "not_found"
        except OSError as exc:
            attempt["status"] = "error"; attempt["error"] = f"{type(exc).__name__}: {exc}"[:256]
            errors.append(f"nested group signal failed: {row['pid']}")
        groups.append(attempt)
    second = q3b._cleanup_ps_snapshot(run)
    if (second.get("command_ok") is not True or second.get("parse_ok") is not True
            or getattr(q3b, "_cleanup_comm_map", lambda _snapshot: None)(second) is None
            or any(not isinstance(row, Mapping) or set(row) not in allowed_fields
           or (row.get("uid") == uid and ("sid" not in row or type(row.get("sid")) is not int))
           for row in second.get("records", []))):
        return {"known": False, "groups": groups, "remaining": [], "errors": errors + ["nested worker inventory schema unknown"]}
    remaining = [row for row in second.get("records", [])
                 if row.get("uid") == uid and "q3c_performance_replication.py" in row.get("args", "")
                 and "--phase-worker" in row.get("args", "")
                 and (row.get("pid"), row.get("start"), row.get("uid"), row.get("sid"), row.get("pgid")) in candidate_keys]
    if remaining and second.get("command_ok") is True and second.get("parse_ok") is True:
        for row in remaining:
            current = [item for item in second["records"] if item.get("pgid") == row.get("pgid")]
            if (row.get("start") != original_starts.get(row.get("pid"))
                    or row.get("start") != next((item.get("start") for item in current if item.get("pid") == row.get("pid")), None)
                    or row.get("sid") != row.get("pgid")
                    or any(item.get("uid") != uid for item in current)):
                errors.append(f"nested group start identity changed: {row.get('pid')}")
                continue
            try:
                os.killpg(row["pgid"], signal.SIGKILL)
            except OSError as exc:
                errors.append(f"nested group kill failed: {type(exc).__name__}")
        final = q3b._cleanup_ps_snapshot(run)
        remaining = [row for row in final.get("records", [])
                     if row.get("uid") == uid and "q3c_performance_replication.py" in row.get("args", "")
                     and "--phase-worker" in row.get("args", "")
                     and (row.get("pid"), row.get("start"), row.get("uid"), row.get("sid"), row.get("pgid")) in candidate_keys]
        if remaining:
            errors.append("nested worker remains after kill")
    return {"known": second.get("command_ok") is True and second.get("parse_ok") is True,
            "groups": groups, "remaining": remaining, "errors": errors}


def _strict_q3c_cleanup(q3b: Any, process: Any, identity: Mapping[str, Any],
                        *, guard_proof: Any = None, child_ledger: Any = None) -> dict[str, Any]:
    """Call v2 cleanup only when its Q3f proof channel is available."""
    helper = getattr(q3b, "_cleanup_worker_evidence", None)
    if not callable(helper):
        return {"schema": "ironmule.cleanup.v2", "worker_reaped": False,
                "verification": {"group_gone": False},
                "unresolved_errors": ["strict Q3f cleanup helper unavailable"]}
    try:
        parameters = inspect.signature(helper).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "guard_proof" not in parameters or "child_ledger" not in parameters:
        return {"schema": "ironmule.cleanup.v2", "worker_reaped": False,
                "verification": {"group_gone": False},
                "unresolved_errors": ["strict Q3f cleanup proof channel unavailable"]}
    try:
        evidence = helper(process, identity, global_inventory=True,
                          guard_proof=guard_proof, child_ledger=child_ledger)
    except BaseException as exc:
        return {"schema": "ironmule.cleanup.v2", "worker_reaped": False,
                "verification": {"group_gone": False},
                "unresolved_errors": [f"strict cleanup failed: {type(exc).__name__}"]}
    if not isinstance(evidence, dict):
        return {"schema": "ironmule.cleanup.v2", "worker_reaped": False,
                "verification": {"group_gone": False},
                "unresolved_errors": ["strict cleanup returned malformed evidence"]}
    return evidence


def _invoke_q3c_once(root: Path, output: Path, *, expected_commit: str | None = None,
                     expected_identity: Mapping[str, Any] | None = None,
                     runner: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Invoke unchanged Q3c once and accept only a complete strict PASS."""
    execute = runner
    command = [sys.executable, str(root / "research" / "q3c_performance_replication.py"),
               "--execute", "--output", str(output)]
    try:
        q3c = _load_q3c(root)
    except BaseException as exc:
        return {"invoked": True, "command": command,
                "offline_env": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
                "inherited_environment_preserved": True, "returncode": None,
                "output_bounded": False, "output_path": str(output),
                "output_sha256": None, "output_size_bytes": None,
                "parsed_identity": None, "parsed_status": None, "status": "FAILED",
                "error": f"q3c policy load failed: {type(exc).__name__}"}
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONPATH": str(root)}
    cleanup = None
    nested_cleanup = None
    process = None
    worker_identity = None
    q3b = None
    try:
        if execute is not None:
            completed = execute(command, cwd=str(root), capture_output=True, text=True,
                                timeout=Q3C_WRAPPER_TIMEOUT_SECONDS, check=False, env=env)
        else:
            q3b = _load_q3b(root)
            baseline = q3b._capture_process_baseline()
            if not isinstance(baseline, Mapping) or baseline.get("valid") is not True:
                return {"invoked": False, "command": command,
                        "offline_env": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
                        "inherited_environment_preserved": True, "status": "FAILED",
                        "output_path": str(output), "output_bounded": False,
                        "error": "q3c pre-spawn baseline unavailable"}
            process = subprocess.Popen(command, cwd=str(root), stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True, env=env,
                                       start_new_session=True)
            try:
                worker_identity = q3b._capture_worker_identity(process, baseline=baseline)
            except BaseException as exc:
                # Identity capture is immediate and fail-closed.  Do not use
                # an unverified PGID for cleanup; preserve this as unknown.
                try:
                    process.terminate()
                    process.wait(timeout=2)
                except (OSError, subprocess.TimeoutExpired, ValueError):
                    try:
                        process.kill()
                        process.wait(timeout=2)
                    except (OSError, subprocess.TimeoutExpired, ValueError):
                        pass
                completed = type("Completed", (), {"returncode": process.poll(), "stdout": "", "stderr": ""})()
                cleanup = {"schema": "ironmule.cleanup.v2", "error": f"identity capture failed: {type(exc).__name__}"}
            else:
                try:
                    stdout_value, stderr_value = process.communicate(timeout=Q3C_WRAPPER_TIMEOUT_SECONDS)
                except subprocess.TimeoutExpired as exc:
                    if hasattr(q3b, "_cleanup_ps_snapshot"):
                        nested_cleanup = _cleanup_nested_q3c_workers(q3b, worker_identity["uid"], q3b._run_text, baseline, process.pid)
                    cleanup = _strict_q3c_cleanup(q3b, process, worker_identity,
                                                  guard_proof=[], child_ledger=[])
                    completed = type("Completed", (), {
                        "returncode": process.poll(), "stdout": getattr(exc, "output", "") or "",
                        "stderr": getattr(exc, "stderr", "") or "",
                    })()
                else:
                    cleanup = _strict_q3c_cleanup(q3b, process, worker_identity,
                                                  guard_proof=[], child_ledger=[])
                    completed = type("Completed", (), {"returncode": process.returncode,
                                                       "stdout": stdout_value, "stderr": stderr_value})()
    except BaseException as exc:
        # Every exception after Popen must pass through the same bounded
        # cleanup proof as timeout.  This includes OSError, interrupted
        # communicate(), malformed streams, and unexpected BaseExceptions.
        if process is not None and cleanup is None and q3b is not None and worker_identity is not None:
            try:
                if hasattr(q3b, "_cleanup_ps_snapshot"):
                    try:
                        nested_cleanup = _cleanup_nested_q3c_workers(
                            q3b, worker_identity["uid"], q3b._run_text, baseline, process.pid)
                    except BaseException as nested_exc:
                        nested_cleanup = {"known": False, "errors": [f"nested cleanup failed: {type(nested_exc).__name__}"]}
                cleanup = _strict_q3c_cleanup(q3b, process, worker_identity,
                                              guard_proof=[], child_ledger=[])
            except BaseException as cleanup_exc:
                cleanup = {"schema": "ironmule.cleanup.v2",
                           "error": f"cleanup failed: {type(cleanup_exc).__name__}"}
        return {"invoked": True, "command": command,
                "offline_env": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
                "inherited_environment_preserved": True, "returncode": None,
                "output_bounded": False, "output_path": str(output),
                "output_sha256": None, "output_size_bytes": None,
                "parsed_identity": None, "parsed_status": None, "cleanup": cleanup,
                "nested_cleanup": nested_cleanup,
                "status": "FAILED",
                "error": f"q3c invocation failed: {type(exc).__name__}"}
    stdout, stderr = getattr(completed, "stdout", ""), getattr(completed, "stderr", "")
    bounded = (isinstance(stdout, str) and isinstance(stderr, str)
               and len(stdout) <= OUTPUT_CAP_BYTES and len(stderr) <= OUTPUT_CAP_BYTES)
    try:
        raw_bounded = output.is_file() and output.stat().st_size <= OUTPUT_CAP_BYTES
        raw = json.loads(output.read_text(), parse_constant=_reject_json_constant) if raw_bounded else None
    except OSError:
        raw_bounded, raw = False, None
    except (UnicodeError, ValueError, TypeError, json.JSONDecodeError):
        raw_bounded, raw = raw_bounded, None
    strict_ok, strict_reason, identity = (False, "q3c binding unavailable", None)
    if raw_bounded and expected_commit is not None and expected_identity is not None:
        strict_ok, strict_reason, identity = _strict_q3c_result(
            raw, root=root, expected_commit=expected_commit,
            expected_identity=expected_identity, q3c=q3c)
    try:
        raw_sha = _sha256(output) if raw_bounded else None
        raw_size = output.stat().st_size if raw_bounded else None
    except OSError:
        raw_sha, raw_size = None, None
    return {"invoked": True, "command": command,
            "returncode": getattr(completed, "returncode", None),
            "stdout": stdout[:CAPTURED_OUTPUT_BYTES] if bounded else "<output exceeded cap>",
            "stderr": stderr[:CAPTURED_OUTPUT_BYTES] if bounded else "<output exceeded cap>",
            "stdout_size_bytes": len(stdout) if isinstance(stdout, str) else None,
            "stderr_size_bytes": len(stderr) if isinstance(stderr, str) else None,
            "offline_env": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            "inherited_environment_preserved": all(key in env for key in ("HOME", "PATH")),
            "output_bounded": bounded and raw_bounded,
            "cleanup": cleanup,
            "nested_cleanup": nested_cleanup,
            "output_path": str(output), "output_sha256": raw_sha, "output_size_bytes": raw_size,
            "parsed_identity": identity, "parsed_status": raw.get("status") if isinstance(raw, dict) else None,
            "error": None if strict_ok else strict_reason,
            "status": "PASS" if bounded and raw_bounded and getattr(completed, "returncode", None) == 0
            and strict_ok and (execute is not None or isinstance(cleanup, dict)
                               and cleanup.get("verification", {}).get("group_gone") is True
                               and isinstance(cleanup.get("guard_proof"), list)
                               and bool(cleanup.get("guard_proof"))
                               and isinstance(cleanup.get("child_ledger"), list)
                               and bool(cleanup.get("child_ledger"))
                               and (nested_cleanup is None or nested_cleanup.get("known") is True
                                    and not nested_cleanup.get("errors"))) else "FAILED"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--q3c-output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"schema": SCHEMA, "experiment": EXPERIMENT_ID,
                          "estimated_wall_seconds": OUTER_MAX_SECONDS, "plan": plan()}, indent=2, sort_keys=True))
        return 0
    if args.output is None:
        parser.error("--output is required with --execute")
    if not args.output.parent.is_dir() or os.path.lexists(args.output):
        print("q3d: output path must have an existing parent and must not already exist", file=sys.stderr)
        return 2
    q3c_output = args.q3c_output or args.output.with_name(args.output.stem + "_q3c.json")
    if not q3c_output.parent.is_dir() or os.path.lexists(q3c_output) or q3c_output == args.output:
        print("q3d: q3c output path must have an existing parent and must not already exist", file=sys.stderr)
        return 2
    summary_output = args.summary_output or args.output.with_name(args.output.stem + "_summary.json")
    if not summary_output.parent.is_dir() or os.path.lexists(summary_output) or summary_output in {args.output, q3c_output}:
        print("q3d: summary output path must have an existing parent and must not already exist", file=sys.stderr)
        return 2
    root = Path(__file__).resolve().parents[1]
    started = time.monotonic()
    try:
        pre = preflight(root=root, deadline=started + GATE_DEADLINE_SECONDS)
    except BaseException as exc:
        pre = {"passed": False, "checks": {}, "error": f"preflight failed: {type(exc).__name__}"}
    gate = None
    status = "FAILED"
    if pre.get("passed") is True:
        try:
            gate = run_stability_gate(pre, root=root, deadline=started + GATE_DEADLINE_SECONDS)
            if gate.get("passed") is True:
                valid_gate, reason = validate_stability_evidence(gate)
                if not valid_gate:
                    gate = dict(gate)
                    gate.setdefault("errors", []).append(f"gate evidence validation failed: {reason}")
                status = "PASS" if valid_gate else "FAILED"
            else:
                status = "FAILED"
        except BaseException as exc:
            gate = {"schema": "ironmule.q3d_stability_evidence.v1", "samples": [], "sample_times": [],
                    "sample_offsets": [], "commands": [], "errors": [f"gate failed: {type(exc).__name__}"],
                    "passed": False}
    record = {"schema": SCHEMA, "experiment": EXPERIMENT_ID, "status": status,
              "fallback": "BASE/current incumbent", "promotion_allowed": False,
              "plan": plan(root), "preregistration_sha256": _sha256(PREREGISTRATION) if PREREGISTRATION.exists() else None,
              "preflight": pre, "stability": gate}
    encoded = (json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if len(encoded) > OUTPUT_CAP_BYTES:
        raise Q3dRefused("stability record exceeds bounded output")
    _write_exclusive(args.output, encoded)
    gate_raw = {"path": str(args.output), "sha256": _sha256(args.output),
                "size_bytes": args.output.stat().st_size}
    if status != "PASS":
        summary = {"schema": "ironmule.q3d_summary.v1", "experiment": EXPERIMENT_ID,
                       "status": "FAILED", "promotion_allowed": False, "invoked": False,
                       "gate_raw": gate_raw, "q3c": {"invoked": False, "status": "NOT_INVOKED"}}
        summary_bytes = (json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        if len(summary_bytes) > OUTPUT_CAP_BYTES:
            raise Q3dRefused("final summary exceeds bounded output")
        _write_exclusive(summary_output, summary_bytes)
        print(json.dumps({"status": status, "output": str(args.output), "promotion_allowed": False,
                          "summary_output": str(summary_output), "q3c_invoked": False}, sort_keys=True))
        return 2
    # This is the sole permitted Q3c invocation.  A non-zero Q3c result is
    # retained in its own exclusive raw file and never retried or promoted.
    try:
        q3c = _invoke_q3c_once(root, q3c_output, expected_commit=pre["git_commit"],
                                expected_identity=pre["model_cache_identity"])
    except BaseException as exc:
        q3c = {"invoked": True, "status": "FAILED",
               "error": f"q3c invocation failed: {type(exc).__name__}"}
    summary = {"schema": "ironmule.q3d_summary.v1", "experiment": EXPERIMENT_ID,
               "status": "Q3C_" + q3c["status"], "promotion_allowed": False,
               "invoked": q3c.get("invoked") is True, "gate_raw": gate_raw, "q3c": q3c}
    summary_bytes = (json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if len(summary_bytes) > OUTPUT_CAP_BYTES:
        raise Q3dRefused("final summary exceeds bounded output")
    _write_exclusive(summary_output, summary_bytes)
    print(json.dumps({"status": "Q3C_" + q3c["status"], "output": str(args.output),
                      "q3c_output": str(q3c_output), "summary_output": str(summary_output), "promotion_allowed": False,
                      "q3c_invoked": q3c["invoked"]}, sort_keys=True))
    return 0 if q3c["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
