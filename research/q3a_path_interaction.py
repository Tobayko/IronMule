#!/usr/bin/env python3
"""Q3a: dry-run-first, capability-bound path-interaction pilot.

The parent is stdlib-only and owns the complete 300-second deadline. It resolves
the exact cached model snapshot, checks the machine, creates a one-shot pipe
capability, and only then starts the worker process group. The worker verifies that
capability before importing IronMule/MLX and calls the existing ``ab.run`` hooks;
its direct children inherit the worker process group.
No execution is possible through the worker entry point alone.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import resource
import secrets
import signal
import stat
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

EXPERIMENT_ID = "Q3a-path-interaction"
MODEL_ID = "mlx-community/gemma-3-4b-it-4bit"
EXPECTED_REVISION = "93724907d4ed1745d2fe50baadf3b0b01a65abf2"
PROCESSES = 6
REPEATS = 7
WARMUP = 2
MAX_TOKENS = 32
PILOT_DEADLINE_SECONDS = 300
WORKER_MAX_SECONDS = 240
CHILD_TIMEOUT_SECONDS = 35
POSTFLIGHT_RESERVE_SECONDS = 10
COMMAND_TIMEOUT_SECONDS = 1.0
SWAP_LIMIT_BYTES = 256 * 1024**2
LOAD_MAX = 4.0
LOAD_SPREAD_MAX = 1.0
PEAK_CEILING_FRACTION = 0.60
MAX_COMMAND_OUTPUT = 64 * 1024
MAX_WORKER_OUTPUT = 512 * 1024
CAPABILITY_MAX_BYTES = 16 * 1024
KNOWN_MODEL_ACTIVITY = ("mlx", "llama.cpp", "ollama", "lm studio", "vllm", "gemma", "qwen", "claude")
NATIVE_INFERENCE_ACTIVITY = ("llama-server", "llama_server", "llama-cli", "llama_cli", "llama.cpp", "mlx_lm", "mlx-lm", "ollama", "vllm", "lm studio")
COMMANDS = {"pmset": "/usr/bin/pmset", "sysctl": "/usr/sbin/sysctl", "ps": "/bin/ps", "git": "/usr/bin/git"}
PREREGISTRATION = Path(__file__).resolve().parent / "raw" / "Q3a_preregistration.md"
PREREGISTRATION_SHA = Path(__file__).resolve().parent / "raw" / "Q3a_preregistration.sha256"

Q2_INCUMBENT = {
    "fuse_projections": False, "compiled_fixed_cache": True, "fused_argmax": False,
    "head_skip_prefill": True, "prefill_into_fixed": False, "readback_every": 2,
    "speculate_k": 0, "speculate_ngram": 3, "capacity_slack": 0, "wired_fraction": 0.0,
}
Q3A_CANDIDATE = {**Q2_INCUMBENT, "fused_argmax": True}


class PilotRefused(RuntimeError):
    """Raised when a start or runtime safety gate cannot be proven."""


class _CommandText(str):
    """String output carrying command success without changing test injectors."""

    def __new__(cls, value: str, ok: bool):
        result = super().__new__(cls, value)
        result.ok = ok
        return result


def _absolute(name: str) -> str:
    return COMMANDS[name]


def _run_text(command: list[str], timeout: float = COMMAND_TIMEOUT_SECONDS) -> str:
    if not command or not Path(command[0]).is_absolute() or not Path(command[0]).is_file() or not os.access(command[0], os.X_OK):
        return _CommandText("", False)
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return _CommandText("", False)
    if completed.returncode != 0 or not isinstance(completed.stdout, str) or len(completed.stdout) > MAX_COMMAND_OUTPUT:
        return _CommandText("", False)
    return _CommandText(completed.stdout, True)


def _deadline_runner(deadline: float, run: Callable[[list[str]], str] = _run_text) -> Callable[[list[str]], str]:
    """Prevent OS commands from starting after the shared monotonic deadline."""
    def bounded(command: list[str]) -> str:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return _CommandText("", False)
        if run is _run_text:
            return run(command, timeout=min(COMMAND_TIMEOUT_SECONDS, remaining))
        return run(command)
    return bounded


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_code_sha256(root: Path | None = None) -> str:
    """Hash the complete Python execution surface deterministically."""
    root = Path(__file__).resolve().parents[1] if root is None else root.resolve()
    paths = sorted((root / "ironmule").rglob("*.py"))
    q3a_path = root / "research" / "q3a_path_interaction.py"
    if q3a_path.is_file():
        paths.append(q3a_path)
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _load_identity_module(root: Path) -> Any:
    path = root / "ironmule" / "model_identity.py"
    name = "ironmule_q3a_model_identity"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PilotRefused("cannot load stdlib model identity contract")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _cache_roots() -> tuple[Path, ...]:
    roots = []
    if os.environ.get("HF_HUB_CACHE"):
        roots.append(Path(os.environ["HF_HUB_CACHE"]).expanduser())
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]).expanduser() / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    return tuple(dict.fromkeys(roots))


def resolve_exact_local_identity(root: Path | None = None) -> dict[str, Any]:
    """Resolve the pinned 4B snapshot with no Hugging Face/network import."""
    root = Path(__file__).resolve().parents[1] if root is None else root.resolve()
    encoded = "models--" + MODEL_ID.replace("/", "--")
    snapshots = [cache / encoded / "snapshots" / EXPECTED_REVISION for cache in _cache_roots()]
    existing = [path for path in snapshots if path.is_dir()]
    if len(existing) != 1:
        raise PilotRefused(f"expected exactly one local 4B snapshot at pinned revision, found {len(existing)}")
    try:
        identity = _load_identity_module(root).build_model_identity(MODEL_ID, existing[0], EXPECTED_REVISION)
        binding = {"model_id": identity.model_id, "model_revision": identity.revision, "model_manifest_sha256": identity.model_manifest_sha256}
    except (AttributeError, ImportError, OSError, ValueError, TypeError, KeyError) as exc:
        raise PilotRefused(f"local model identity failed: {exc}") from exc
    if binding["model_id"] != MODEL_ID or binding["model_revision"] != EXPECTED_REVISION:
        raise PilotRefused("local model identity does not match the pinned 4B binding")
    if not re.fullmatch(r"[0-9a-f]{64}", binding["model_manifest_sha256"]):
        raise PilotRefused("local model manifest is not a lowercase SHA-256 digest")
    return binding


def _swap_bytes(text: str) -> int | None:
    if not isinstance(text, str):
        return None
    match = re.fullmatch(r"\s*.*?used\s*=\s*([0-9]+(?:[.,][0-9]+)?)\s*([KMG]?).*", text, re.IGNORECASE)
    if not match:
        return None
    scale = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3}[match.group(2).upper()]
    return int(float(match.group(1).replace(",", ".")) * scale)


def _thermal_nominal(text: str) -> bool:
    if not isinstance(text, str):
        return False
    raw_lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
    lines = set(raw_lines)
    if len(raw_lines) != len(lines):
        return False
    required = {"no thermal warning level has been recorded", "no performance warning level has been recorded"}
    if not required.issubset(lines):
        return False
    allowed_cpu_status = {
        "no cpu power status has been recorded",
        "no cpu power status is available",
        "no cpu power status available",
    }
    zero_value = re.compile(r"^[a-z0-9 _-]*(?:speed[_ ]limit|pressure)[a-z0-9 _-]*\s*[:=]\s*0(?:\.0+)?$")
    for line in lines - required:
        if line in allowed_cpu_status or zero_value.fullmatch(line):
            continue
        return False
    return True


def system_environment(run: Callable[[list[str]], str] = _run_text) -> dict[str, Any]:
    try:
        battery = run([_absolute("pmset"), "-g", "batt"])
        low_power = run([_absolute("pmset"), "-g", "lowpowermode"])
        thermal = run([_absolute("pmset"), "-g", "therm"])
        swap_text = run([_absolute("sysctl"), "-n", "vm.swapusage"])
    except Exception:
        battery = low_power = thermal = swap_text = ""
    battery = battery if isinstance(battery, str) else ""
    low_power = low_power if isinstance(low_power, str) else ""
    thermal = thermal if isinstance(thermal, str) else ""
    swap_text = swap_text if isinstance(swap_text, str) else ""
    low_match = re.search(r"(?:lowpowermode|low\s+power\s+mode).*?\b([01])\b", low_power.lower())
    return {"power_source": "AC" if "ac power" in battery.lower() else "battery" if battery else "unknown", "low_power_mode": False if low_match and low_match.group(1) == "0" else None, "thermal_state": "nominal" if _thermal_nominal(thermal) else "unknown", "swap_used_bytes": _swap_bytes(swap_text), "platform": platform.platform(), "python": platform.python_version()}


def _loadavg1() -> float:
    try:
        return float(os.getloadavg()[0])
    except (AttributeError, OSError, TypeError, ValueError):
        return math.nan


def loadavg_gate(sample: Callable[[], float] = _loadavg1, sleeper: Callable[[float], None] = time.sleep, *, deadline: float | None = None) -> dict[str, Any]:
    samples = []
    for index in range(3):
        if deadline is not None and time.monotonic() >= deadline:
            samples.extend([math.nan] * (3 - index))
            break
        if index:
            if deadline is not None and time.monotonic() + 1.0 >= deadline:
                samples.append(math.nan)
                continue
            try:
                sleeper(1.0)
            except (OSError, TypeError, ValueError):
                samples.append(math.nan)
                continue
        try:
            value = float(sample())
        except (TypeError, ValueError, OverflowError, OSError):
            value = math.nan
        samples.append(value)
    valid = all(math.isfinite(value) and value >= 0 for value in samples)
    spread = max(samples) - min(samples) if valid else None
    return {"samples": samples, "max": max(samples) if valid else None, "spread": spread, "passed": valid and max(samples) <= LOAD_MAX and spread <= LOAD_SPREAD_MAX}


def competing_model_process(run: Callable[[list[str]], str] = _run_text) -> str | None:
    """Parse ``ps`` strictly and return no argv, only a generic reason."""
    try:
        output = run([_absolute("ps"), "-Ao", "pid=,rss=,%cpu=,args="])
    except Exception:
        return "process inventory unavailable or command failed"
    if not isinstance(output, str) or not output:
        return "process inventory unavailable or command failed"
    for line in output.splitlines():
        parts = line.split(None, 3)
        if len(parts) != 4:
            return "process inventory malformed"
        try:
            pid, rss_kb, cpu = int(parts[0]), int(parts[1]), float(parts[2])
        except (TypeError, ValueError, OverflowError):
            return "process inventory malformed"
        if pid <= 0 or rss_kb < 0 or not math.isfinite(cpu) or cpu < 0:
            return "process inventory malformed"
        if pid == os.getpid():
            continue
        lowered = parts[3].lower()
        if any(token in lowered for token in NATIVE_INFERENCE_ACTIVITY):
            return "competing model activity detected"
        if any(token in lowered for token in KNOWN_MODEL_ACTIVITY) and (cpu > 1.0 or rss_kb >= 900_000):
            return "competing model activity detected"
    return None


def installed_memory_bytes(run: Callable[[list[str]], str] = _run_text) -> int | None:
    try:
        output = run([_absolute("sysctl"), "-n", "hw.memsize"])
        value = int(output.strip()) if isinstance(output, str) else 0
    except (TypeError, ValueError, OSError):
        return None
    return value if value > 0 else None


def _git_binding(root: Path, run: Callable[[list[str]], str]) -> dict[str, Any]:
    try:
        commit_output = run([_absolute("git"), "-C", str(root), "rev-parse", "HEAD"])
        status = run([_absolute("git"), "-C", str(root), "status", "--porcelain", "--untracked-files=all"])
        commit = commit_output.strip() if isinstance(commit_output, str) else ""
    except Exception:
        return {"clean": False, "commit": None, "reason": "git command failed"}
    if (not isinstance(status, str) or not getattr(commit, "ok", True)
            or not getattr(status, "ok", True)
            or not re.fullmatch(r"[0-9a-f]{40}", commit)):
        return {"clean": False, "commit": None, "reason": "git commit unavailable"}
    dirty = []
    for line in status.splitlines():
        if len(line) < 4 or not line[3:].strip():
            return {"clean": False, "commit": commit, "dirty_paths": [], "reason": "git status malformed"}
        path = line[3:]
        if path != "research/data/squad-dev-v1.1.json":
            dirty.append(path)
    return {"clean": not dirty, "commit": commit, "dirty_paths": dirty}


def preflight(*, root: Path | None = None, run: Callable[[list[str]], str] = _run_text, identity_resolver: Callable[[Path | None], dict[str, Any]] = resolve_exact_local_identity, load_sample: Callable[[], float] = _loadavg1, deadline: float | None = None, load_sleeper: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1] if root is None else root.resolve()
    estimate = PROCESSES * CHILD_TIMEOUT_SECONDS + 60
    if deadline is not None:
        run = _deadline_runner(deadline, run)
    try:
        env = system_environment(run)
        memory = installed_memory_bytes(run)
    except Exception:
        env = {"power_source": "unknown", "low_power_mode": None, "thermal_state": "unknown", "swap_used_bytes": None, "platform": "unknown", "python": "unknown"}
        memory = None
    if deadline is not None and time.monotonic() >= deadline:
        identity, identity_error = None, "preflight deadline exhausted"
    else:
        try:
            identity, identity_error = identity_resolver(root), None
        except Exception as exc:
            identity, identity_error = None, str(exc)
    try:
        load = loadavg_gate(load_sample, load_sleeper, deadline=deadline)
    except Exception:
        load = {"samples": [], "max": None, "spread": None, "passed": False}
    git = _git_binding(root, run)
    checks = {
        "model_identity_exact": identity is not None and identity.get("model_id") == MODEL_ID and identity.get("model_revision") == EXPECTED_REVISION and bool(re.fullmatch(r"[0-9a-f]{64}", str(identity.get("model_manifest_sha256", "")))),
        "runtime_bound_within_300s": estimate <= PILOT_DEADLINE_SECONDS,
        "ac_power": env["power_source"] == "AC",
        "low_power_off": env["low_power_mode"] is False,
        "thermal_nominal": env["thermal_state"] == "nominal",
        "no_competing_model_process": competing_model_process(run) is None,
        "loadavg_gate": load["passed"],
        "start_swap_within_256mib": isinstance(env["swap_used_bytes"], int) and env["swap_used_bytes"] <= SWAP_LIMIT_BYTES,
        "installed_memory_known": memory is not None,
        "git_clean_and_bound": git["clean"] and bool(git["commit"]),
    }
    try:
        declared = PREREGISTRATION_SHA.read_text().strip().split()
        checks["preregistration_matches"] = PREREGISTRATION.exists() and PREREGISTRATION_SHA.exists() and len(declared) == 2 and re.fullmatch(r"[0-9a-f]{64}", declared[0]) is not None and _sha256(PREREGISTRATION) == declared[0] and declared[1] == PREREGISTRATION.name
    except (OSError, IndexError):
        checks["preregistration_matches"] = False
    return {"environment": env, "identity": identity, "identity_error": identity_error, "git": git, "loadavg": load, "installed_memory_bytes": memory, "peak_ceiling_bytes": int(memory * PEAK_CEILING_FRACTION) if memory else None, "estimated_wall_seconds": estimate, "checks": checks, "passed": all(checks.values())}


def postflight(*, deadline: float, run: Callable[[list[str]], str] = _run_text, load_sample: Callable[[], float] = _loadavg1, load_sleeper: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    """Collect postflight gates only while time remains in the shared budget."""
    bounded = _deadline_runner(deadline, run)
    try:
        environment = system_environment(bounded)
        load = loadavg_gate(load_sample, load_sleeper, deadline=deadline)
        process_reason = competing_model_process(bounded)
    except Exception:
        environment = {"power_source": "unknown", "low_power_mode": None, "thermal_state": "unknown", "swap_used_bytes": None}
        load = {"samples": [], "max": None, "spread": None, "passed": False}
        process_reason = "postflight gate unavailable"
    environment["loadavg"] = load
    environment["competing_model_process"] = process_reason
    environment["deadline_remaining"] = max(0.0, deadline - time.monotonic())
    return environment


def _read_capability() -> dict[str, Any]:
    fd_text, nonce, expected_text = os.environ.get("IRONMULE_Q3A_CAP_FD"), os.environ.get("IRONMULE_Q3A_CAP_NONCE"), os.environ.get("IRONMULE_Q3A_EXPECTED_IDENTITY")
    if not fd_text or not nonce or not expected_text:
        raise PilotRefused("worker capability is absent")
    fd = -1
    try:
        fd = int(fd_text)
        if fd < 0:
            raise ValueError
        chunks, total = [], 0
        while True:
            chunk = os.read(fd, min(4096, CAPABILITY_MAX_BYTES - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total >= CAPABILITY_MAX_BYTES:
                raise ValueError
        payload, expected = json.loads(b"".join(chunks).decode("utf-8")), json.loads(expected_text)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise PilotRefused("worker capability is malformed") from exc
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
    if (not isinstance(payload, dict) or not isinstance(expected, dict)
            or set(expected) != {"identity", "runtime_code_sha256"}
            or payload.get("nonce") != nonce
            or payload.get("identity") != expected["identity"]
            or payload.get("runtime_code_sha256") != expected["runtime_code_sha256"]):
        raise PilotRefused("worker capability or expected identity mismatch")
    return expected


def _max_rss_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    return raw if sys.platform == "darwin" else raw * 1024


def _worker() -> int:
    """Verify parent capability before the first IronMule/MLX import."""
    progress: list[dict[str, Any]] = []
    try:
        capability = _read_capability()
        expected = capability["identity"]
        expected_code_hash = capability["runtime_code_sha256"]
        if (not re.fullmatch(r"[0-9a-f]{64}", str(expected_code_hash))
                or runtime_code_sha256() != expected_code_hash):
            raise PilotRefused("runtime code manifest changed after preflight")
        from ironmule import ab
        from ironmule.model_identity import ModelIdentityError
        from ironmule.runtime import Knobs
        from ironmule.tune import resolve_local_model
        try:
            resolved = resolve_local_model(MODEL_ID, revision=EXPECTED_REVISION)
        except (OSError, ValueError, ModelIdentityError) as exc:
            raise PilotRefused(f"worker model resolution failed: {exc}") from exc
        identity = {"model_id": resolved.identity.model_id, "model_revision": resolved.identity.revision, "model_manifest_sha256": resolved.identity.model_manifest_sha256}
        if identity != expected:
            raise PilotRefused("worker model identity differs from parent binding")
        worker_deadline = float(os.environ.get("IRONMULE_Q3A_WORKER_DEADLINE", "nan"))
        if not math.isfinite(worker_deadline):
            raise PilotRefused("worker deadline is absent")
        orders = {}
        worker_run = _deadline_runner(worker_deadline)

        def before_child(index: int, order: list[str]) -> None:
            orders[index] = list(order)
            if time.monotonic() >= worker_deadline:
                raise PilotRefused(f"worker deadline exhausted before child {index}")
            env, quick, reason = (system_environment(worker_run),
                                  loadavg_gate(deadline=worker_deadline),
                                  competing_model_process(worker_run))
            if env["power_source"] != "AC" or env["low_power_mode"] is not False or env["thermal_state"] != "nominal" or not isinstance(env["swap_used_bytes"], int) or env["swap_used_bytes"] > SWAP_LIMIT_BYTES or not quick["passed"] or reason is not None:
                raise PilotRefused(f"before_child gate failed at child {index}")

        def on_child(index: int, child: dict[str, Any]) -> None:
            marker = {"index": index, "pid": child.get("pid"), "order": orders.get(index, []), "arms": sorted(child.get("arms", {}))}
            progress.append(marker)
            print("@PROGRESS" + json.dumps(marker, separators=(",", ":")), flush=True)

        result = ab.run({"q2_incumbent": Knobs(**Q2_INCUMBENT), "fused_argmax_path": Knobs(**Q3A_CANDIDATE)}, processes=PROCESSES, repeats=REPEATS, warmup=WARMUP, max_tokens=MAX_TOKENS, model=MODEL_ID, child_timeout_seconds=CHILD_TIMEOUT_SECONDS, before_child=before_child, on_child=on_child)
        result["binding"] = {**identity, "runtime_code_sha256": expected_code_hash}
        result["rss_peak_bytes"], result["progress_markers"] = _max_rss_bytes(), progress
        print("@@" + json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
        return 0
    except BaseException as exc:
        print("@@" + json.dumps({"failure": f"{type(exc).__name__}: {exc}", "progress_markers": progress, "partial_children": getattr(exc, "partial_children", [])}, sort_keys=True), flush=True)
        return 2


def _finite(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _arm_valid(arm: Any) -> bool:
    expected = {"total_ns", "prefill_ns", "decode_ns", "logical_tokens", "logical_tokens_per_repeat", "physical_tokens_per_repeat", "token_counts", "stop_reasons", "capacities", "deterministic", "decode_steps", "prompt_tokens", "mlx_peak_bytes"}
    if not isinstance(arm, dict) or set(arm) != expected or any(not isinstance(arm[name], list) or len(arm[name]) != REPEATS for name in ("total_ns", "prefill_ns", "decode_ns", "logical_tokens_per_repeat", "physical_tokens_per_repeat", "token_counts", "stop_reasons", "capacities")):
        return False
    if any(not all(_finite(value) and value > 0 for value in arm[name]) for name in ("total_ns", "prefill_ns", "decode_ns")):
        return False
    if (not isinstance(arm["logical_tokens"], list)
            or not all(isinstance(token, int) and not isinstance(token, bool) and token >= 0
                       for token in arm["logical_tokens"])):
        return False
    for logical, physical in zip(arm["logical_tokens_per_repeat"], arm["physical_tokens_per_repeat"]):
        if (not isinstance(logical, list) or not isinstance(physical, list)
                or not physical
                or not all(isinstance(token, int) and not isinstance(token, bool) and token >= 0
                           for token in logical + physical)):
            return False
    if arm["logical_tokens_per_repeat"][0] != arm["logical_tokens"] or any(not isinstance(item, dict) or set(item) != {"logical", "physical"} or any(not isinstance(item[key], int) or isinstance(item[key], bool) or item[key] < 0 for key in ("logical", "physical")) or item["logical"] != len(logical) or item["physical"] != len(physical) for item, logical, physical in zip(arm["token_counts"], arm["logical_tokens_per_repeat"], arm["physical_tokens_per_repeat"])):
        return False
    if any(stop not in {"eos", "length"} for stop in arm["stop_reasons"]):
        return False
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in arm["capacities"] + [arm["decode_steps"], arm["prompt_tokens"], arm["mlx_peak_bytes"]]):
        return False
    return (isinstance(arm["deterministic"], bool)
            and arm["deterministic"] is True
            and arm["decode_steps"] == len(arm["physical_tokens_per_repeat"][0]) - 1
            and all(item == arm["logical_tokens_per_repeat"][0]
                    for item in arm["logical_tokens_per_repeat"])
            and all(item == arm["physical_tokens_per_repeat"][0]
                     for item in arm["physical_tokens_per_repeat"])
            and all(item == arm["token_counts"][0] for item in arm["token_counts"])
            and all(item == arm["stop_reasons"][0] for item in arm["stop_reasons"])
            and all(item == arm["capacities"][0] for item in arm["capacities"]))


def _raw_complete(result: dict[str, Any]) -> bool:
    raw = result.get("raw")
    if not isinstance(raw, list) or len(raw) != PROCESSES:
        return False
    pids = set()
    for index, child in enumerate(raw):
        if (not isinstance(child, dict)
                or set(child) != {"pid", "arms", "order", "mlx_peak_bytes"}
                or not isinstance(child["arms"], dict)
                or not isinstance(child["order"], list)):
            return False
        if not isinstance(child["pid"], int) or isinstance(child["pid"], bool) or child["pid"] <= 0 or child["pid"] in pids:
            return False
        pids.add(child["pid"])
        order = ["q2_incumbent", "fused_argmax_path"] if index % 2 == 0 else ["fused_argmax_path", "q2_incumbent"]
        if child["order"] != order or set(child["arms"]) != set(order) or not _arm_valid(child["arms"][order[0]]) or not _arm_valid(child["arms"][order[1]]) or not isinstance(child["mlx_peak_bytes"], int) or child["mlx_peak_bytes"] <= 0:
            return False
    return True


def _summary_valid(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {"n", "median", "min", "max", "p95", "stdev"} and value["n"] == PROCESSES and all(_finite(value[key]) and value[key] >= 0 for key in ("median", "min", "max", "p95", "stdev"))


def _ratio_valid(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {"median_ratio", "ci_low", "ci_high", "pairs"} and all(_finite(value[key]) for key in ("median_ratio", "ci_low", "ci_high")) and value["ci_low"] <= value["median_ratio"] <= value["ci_high"] and isinstance(value["pairs"], list) and len(value["pairs"]) == PROCESSES and all(_finite(item) and item > 0 for item in value["pairs"])


def _summary_from_raw(samples: list[float]) -> dict[str, Any]:
    ordered = sorted(samples)
    return {
        "n": len(ordered),
        "median": statistics.median(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "p95": ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))],
        "stdev": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
    }


def _ratio_from_raw(candidate: list[float], baseline: list[float]) -> dict[str, Any] | None:
    try:
        pairs = [c / b for c, b in zip(candidate, baseline)]
        if len(pairs) != PROCESSES or not all(_finite(value) and value > 0 for value in pairs):
            return None
        rng = __import__("random").Random(20260825)
        medians = []
        for _ in range(10000):
            draw = [pairs[rng.randrange(len(pairs))] for _ in pairs]
            medians.append(statistics.median(draw))
        medians.sort()
        return {"median_ratio": statistics.median(pairs), "ci_low": medians[250], "ci_high": medians[9750], "pairs": pairs}
    except (IndexError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None


def _derived_values_match(result: dict[str, Any]) -> bool:
    raw = result["raw"]
    for name in ("q2_incumbent", "fused_argmax_path"):
        for metric in ("total_ns", "prefill_ns", "decode_ns"):
            samples = [statistics.median(child["arms"][name][metric]) for child in raw]
            if result["per_arm"][name][metric] != _summary_from_raw(samples):
                return False
    for metric in ("total_ns", "prefill_ns", "decode_ns"):
        candidate = [statistics.median(child["arms"]["fused_argmax_path"][metric]) for child in raw]
        baseline = [statistics.median(child["arms"]["q2_incumbent"][metric]) for child in raw]
        expected = _ratio_from_raw(candidate, baseline)
        if expected is None or result["ratios"]["fused_argmax_path/q2_incumbent"][metric] != expected:
            return False
    return True


def _raw_identity_flags(result: dict[str, Any]) -> dict[str, bool]:
    """Recompute correctness flags from every raw arm, never from declarations."""
    arms = [child["arms"][name] for child in result["raw"] for name in ("q2_incumbent", "fused_argmax_path")]
    logical = [arm["logical_tokens"] for arm in arms]
    counts = [arm["token_counts"] for arm in arms]
    stops = [arm["stop_reasons"] for arm in arms]
    def consistent(arm: dict[str, Any]) -> bool:
        return (arm["logical_tokens_per_repeat"]
                and all(item == arm["logical_tokens_per_repeat"][0] for item in arm["logical_tokens_per_repeat"])
                and all(item == arm["physical_tokens_per_repeat"][0] for item in arm["physical_tokens_per_repeat"])
                and all(item == arm["token_counts"][0] for item in arm["token_counts"])
                and all(item == arm["stop_reasons"][0] for item in arm["stop_reasons"])
                and all(item == arm["capacities"][0] for item in arm["capacities"])
                and arm["decode_steps"] == len(arm["physical_tokens_per_repeat"][0]) - 1)
    return {
        "token_identity": all(item == logical[0] for item in logical),
        "token_count_identity": all(item == counts[0] for item in counts),
        "stop_reason_identity": all(item == stops[0] for item in stops),
        "deterministic": all(consistent(arm) for arm in arms),
    }


def _schema_valid(result: Any) -> tuple[bool, str]:
    if not isinstance(result, dict):
        return False, "worker result is not an object"
    expected = {"arms", "processes", "repeats", "warmup", "raw", "per_arm", "token_identity", "token_count_identity", "stop_reason_identity", "deterministic", "reference_tokens", "ratios", "binding", "rss_peak_bytes", "progress_markers"}
    if (set(result) != expected or result["processes"] != PROCESSES
            or result["repeats"] != REPEATS or result["warmup"] != WARMUP
            or result["arms"] != {"q2_incumbent": Q2_INCUMBENT, "fused_argmax_path": Q3A_CANDIDATE}):
        return False, "worker result fields, counts, or arms are not exact"
    if not _raw_complete(result):
        return False, "raw child/arm records are incomplete"
    flags = _raw_identity_flags(result)
    if any(result[key] is not flags[key] for key in flags) or not all(flags.values()):
        return False, "identity or determinism gate failed"
    if (not isinstance(result["reference_tokens"], list)
            or not all(isinstance(token, int) and not isinstance(token, bool)
                       for token in result["reference_tokens"])
            or result["reference_tokens"] != result["raw"][0]["arms"]["q2_incumbent"]["logical_tokens"]):
        return False, "reference tokens are malformed"
    if not isinstance(result["per_arm"], dict) or set(result["per_arm"]) != {"q2_incumbent", "fused_argmax_path"} or any(not isinstance(result["per_arm"][name], dict) or set(result["per_arm"][name]) != {"total_ns", "prefill_ns", "decode_ns"} or any(not _summary_valid(value) for value in result["per_arm"][name].values()) for name in result["per_arm"]):
        return False, "per-arm summaries are malformed"
    if not isinstance(result["ratios"], dict) or set(result["ratios"]) != {"fused_argmax_path/q2_incumbent"} or any(not _ratio_valid(value) for value in result["ratios"]["fused_argmax_path/q2_incumbent"].values()):
        return False, "paired ratios are malformed"
    try:
        if not _derived_values_match(result):
            return False, "summaries or paired ratios do not match raw samples"
    except (IndexError, KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return False, "summaries or paired ratios cannot be recomputed from raw samples"
    binding = result["binding"]
    if (not isinstance(binding, dict)
            or set(binding) != {"model_id", "model_revision", "model_manifest_sha256", "runtime_code_sha256"}
            or binding["model_id"] != MODEL_ID
            or binding["model_revision"] != EXPECTED_REVISION
            or any(not re.fullmatch(r"[0-9a-f]{64}", str(binding[key]))
                   for key in ("model_manifest_sha256", "runtime_code_sha256"))):
        return False, "model or runtime binding is malformed"
    markers = result["progress_markers"]
    if (not isinstance(result["rss_peak_bytes"], int)
            or isinstance(result["rss_peak_bytes"], bool)
            or result["rss_peak_bytes"] <= 0
            or not isinstance(markers, list)
            or len(markers) != PROCESSES
            or any(not isinstance(marker, dict)
                   or set(marker) != {"index", "pid", "order", "arms"}
                   or not isinstance(marker["index"], int)
                   or not 0 <= marker["index"] < PROCESSES
                   or not isinstance(marker["pid"], int)
                   or marker["pid"] <= 0
                   or marker["order"] not in (["q2_incumbent", "fused_argmax_path"], ["fused_argmax_path", "q2_incumbent"])
                   or marker["arms"] != ["fused_argmax_path", "q2_incumbent"]
                   for marker in markers)
            or len({marker["index"] for marker in markers}) != len(markers)
            or len({marker["pid"] for marker in markers}) != len(markers)):
        return False, "RSS or progress markers are malformed"
    for marker in markers:
        child = result["raw"][marker["index"]]
        if marker["pid"] != child["pid"] or marker["order"] != child["order"]:
            return False, "progress markers do not bind one-to-one to raw children"
    return True, "ok"


def analyze_ratio(result: dict[str, Any]) -> dict[str, Any]:
    ratio = result["ratios"]["fused_argmax_path/q2_incumbent"]["total_ns"]
    low, high = ratio["ci_low"], ratio["ci_high"]
    if high < 0.995:
        classification = "GAIN"
    elif low > 1.005:
        classification = "LOSS"
    elif low >= 0.995 and high <= 1.005:
        classification = "PRACTICALLY_NEUTRAL"
    else:
        classification = "INCONCLUSIVE"
    return {"metric": "total_ns", "candidate_over_incumbent": ratio["median_ratio"], "ci_low": low, "ci_high": high, "classification": classification, "thresholds_preregistered": {"gain_ci_high_lt": 0.995, "loss_ci_low_gt": 1.005, "neutral_interval": [0.995, 1.005]}}


def validate_result(result: Any, before: dict[str, Any], after: dict[str, Any], expected_runtime_code_sha256: str, *, deadline: float | None = None) -> dict[str, Any]:
    """Fail closed with structured errors; malformed input must never raise KeyError."""
    try:
        valid, reason = _schema_valid(result)
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return {"checks": {"schema": False}, "passed": False, "errors": ["worker result schema raised during validation"]}
    if not valid:
        return {"checks": {"schema": False}, "passed": False, "errors": [reason]}
    if not isinstance(before, dict) or not isinstance(after, dict):
        return {"checks": {"evidence_context": False}, "passed": False, "errors": ["pre/postflight evidence is malformed"]}
    ceiling = before.get("peak_ceiling_bytes")
    peaks = [arm["mlx_peak_bytes"] for child in result["raw"] for arm in child["arms"].values()]
    before_environment = before.get("environment")
    before_environment = before_environment if isinstance(before_environment, dict) else {}
    start_swap, end_swap = before_environment.get("swap_used_bytes"), after.get("swap_used_bytes")
    binding = result["binding"]
    before_identity = before.get("identity")
    if (not isinstance(before_identity, dict)
            or set(before_identity) != {"model_id", "model_revision", "model_manifest_sha256"}
            or before_identity.get("model_id") != MODEL_ID
            or before_identity.get("model_revision") != EXPECTED_REVISION
            or not re.fullmatch(r"[0-9a-f]{64}", str(before_identity.get("model_manifest_sha256")))):
        return {"checks": {"evidence_identity": False}, "passed": False, "errors": ["preflight identity binding is missing or not exact"]}
    expected_manifest = before_identity.get("model_manifest_sha256")
    manifest_bound = binding["model_manifest_sha256"] == expected_manifest
    after_load = after.get("loadavg")
    checks = {"schema": True, "runtime_code_hash_bound": binding["runtime_code_sha256"] == expected_runtime_code_sha256, "model_identity_bound": binding["model_id"] == MODEL_ID and binding["model_revision"] == EXPECTED_REVISION and manifest_bound, "mlx_peak_conservative": isinstance(ceiling, int) and max(peaks) <= ceiling, "rss_peak_conservative": isinstance(ceiling, int) and result["rss_peak_bytes"] <= ceiling, "swap_delta_within_256mib": isinstance(start_swap, int) and isinstance(end_swap, int) and end_swap - start_swap <= SWAP_LIMIT_BYTES, "environment_still_safe": after.get("power_source") == "AC" and after.get("low_power_mode") is False and after.get("thermal_state") == "nominal", "postflight_loadavg_safe": isinstance(after_load, dict) and after_load.get("passed") is True, "postflight_process_safe": after.get("competing_model_process") is None, "deadline_within_bound": deadline is None or time.monotonic() <= deadline, "no_timeout_or_crash": True}
    return {"checks": checks, "passed": all(checks.values()), "errors": [] if all(checks.values()) else [name for name, passed in checks.items() if not passed]}


def _markers(output: str | bytes | None) -> list[dict[str, Any]]:
    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")
    if not isinstance(output, str):
        return []
    found = []
    for line in output.splitlines():
        if not line.startswith("@PROGRESS"):
            continue
        try:
            value = json.loads(line[len("@PROGRESS"):])
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if (isinstance(value, dict) and set(value) == {"index", "pid", "order", "arms"}
                and isinstance(value["index"], int) and 0 <= value["index"] < PROCESSES
                and isinstance(value["pid"], int) and value["pid"] > 0
                and value["order"] in (["q2_incumbent", "fused_argmax_path"], ["fused_argmax_path", "q2_incumbent"])
                and value["arms"] == ["fused_argmax_path", "q2_incumbent"]
                and all(value["index"] != marker["index"] for marker in found)
                and len(found) < PROCESSES):
            found.append(value)
    return found


def _process_group_gone(pgid: int) -> bool:
    """Return true only when the entire worker process group is gone."""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    return False


def _wait_process_group_gone(pgid: int, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        if _process_group_gone(pgid):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.05, remaining))


def _cleanup_worker(process: subprocess.Popen[str]) -> tuple[list[str], str]:
    """Terminate the worker group, reap its leader, and drain bounded output."""
    cleanup_errors: list[str] = []
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except OSError as error:
        cleanup_errors.append(f"killpg(SIGTERM): {type(error).__name__}")
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    except OSError as error:
        cleanup_errors.append(f"worker wait: {type(error).__name__}")
    group_gone = _wait_process_group_gone(process.pid)
    if group_gone:
        try:
            process.wait(timeout=2)
        except (subprocess.TimeoutExpired, OSError) as error:
            cleanup_errors.append(f"worker reap: {type(error).__name__}")
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as error:
            cleanup_errors.append(f"killpg(SIGKILL): {type(error).__name__}")
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            cleanup_errors.append("worker did not exit after SIGKILL")
        except OSError as error:
            cleanup_errors.append(f"worker wait: {type(error).__name__}")
        if not _wait_process_group_gone(process.pid):
            cleanup_errors.append("worker process group did not disappear")
    drained = ""
    try:
        drained, _ = process.communicate(timeout=2)
    except (subprocess.TimeoutExpired, OSError) as error:
        cleanup_errors.append(f"communicate: {type(error).__name__}")
    if process.poll() is None:
        cleanup_errors.append("worker process-group cleanup failed")
    return cleanup_errors, drained


def _bounded_partial(value: Any) -> list[dict[str, Any]]:
    if (not isinstance(value, list) or len(value) > PROCESSES
            or any(not isinstance(item, dict)
                   or set(item) != {"pid", "arms", "order", "mlx_peak_bytes"}
                   or not isinstance(item["pid"], int)
                   or isinstance(item["pid"], bool)
                   or item["pid"] <= 0
                   or not isinstance(item["arms"], dict)
                   or not isinstance(item["order"], list)
                   or not isinstance(item["mlx_peak_bytes"], int)
                   or isinstance(item["mlx_peak_bytes"], bool)
                   or item["mlx_peak_bytes"] <= 0
                   for item in value)):
        return []
    try:
        if len(json.dumps(value, sort_keys=True, allow_nan=False)) > MAX_WORKER_OUTPUT:
            return []
    except (TypeError, ValueError):
        return []
    return value


def _bounded_markers(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return _markers("\n".join("@PROGRESS" + json.dumps(item, sort_keys=True) for item in value))


def _failure(plan: dict[str, Any], preflight_result: dict[str, Any], reason: str, partial: Any = None, markers: Any = None) -> dict[str, Any]:
    return {"schema": "ironmule.q3a_result.v2", "experiment": EXPERIMENT_ID, "status": "FAILED", "fallback": "BASE", "promotion_allowed": False, "result_type": "INFORMATION_GAIN", "interpretation": "PATH_INTERACTION_ONLY", "reason": reason, "partial_children": _bounded_partial(partial), "progress_markers": _bounded_markers(markers), "plan": plan, "preflight": preflight_result}


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _start_worker(identity: dict[str, Any], deadline: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    nonce = secrets.token_urlsafe(24)
    read_fd, write_fd = os.pipe()
    remaining = deadline - time.monotonic() - POSTFLIGHT_RESERVE_SECONDS
    if remaining <= 0:
        os.close(read_fd)
        os.close(write_fd)
        raise PilotRefused("global deadline exhausted before worker start")
    timeout = min(float(WORKER_MAX_SECONDS), remaining)
    expected = {"identity": identity, "runtime_code_sha256": runtime_code_sha256()}
    payload = json.dumps({"nonce": nonce, **expected}, sort_keys=True, separators=(",", ":")).encode()
    if len(payload) > CAPABILITY_MAX_BYTES:
        os.close(read_fd)
        os.close(write_fd)
        raise PilotRefused("worker capability payload exceeds bound")
    os.set_inheritable(read_fd, True)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(write_fd, payload[offset:])
        os.close(write_fd)
        write_fd = -1
        env = {**os.environ, "IRONMULE_Q3A_CAP_FD": str(read_fd), "IRONMULE_Q3A_CAP_NONCE": nonce, "IRONMULE_Q3A_EXPECTED_IDENTITY": json.dumps(expected, sort_keys=True, separators=(",", ":")), "IRONMULE_Q3A_WORKER_DEADLINE": str(time.monotonic() + timeout)}
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, pass_fds=(read_fd,), start_new_session=True, cwd=str(Path(__file__).resolve().parents[1]))
    except (OSError, ValueError) as exc:
        raise PilotRefused("worker could not be started") from exc
    finally:
        if write_fd >= 0:
            os.close(write_fd)
        os.close(read_fd)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except (subprocess.TimeoutExpired, OSError) as exc:
        cleanup_errors, drained = _cleanup_worker(process)
        markers = _markers(getattr(exc, "stdout", None)) + _markers(drained)
        if cleanup_errors:
            suffix = "; " + "; ".join(cleanup_errors)
        else:
            suffix = ""
        if isinstance(exc, subprocess.TimeoutExpired):
            reason = f"worker timeout after {timeout:.1f}s"
        else:
            reason = "worker communicate failed"
        return {"failure": reason + suffix}, markers
    if len(stdout) > MAX_WORKER_OUTPUT or len(stderr) > MAX_WORKER_OUTPUT:
        return {"failure": "worker output exceeded bounded limit"}, _markers(stdout)
    marker = next((line[len("@@"): ] for line in stdout.splitlines() if line.startswith("@@")), None)
    if marker is None:
        return {"failure": f"worker returned no result (status {process.returncode})"}, _markers(stdout)
    try:
        result = json.loads(marker)
    except (TypeError, ValueError, json.JSONDecodeError):
        result = {"failure": "worker result JSON is malformed"}
    if process.returncode != 0 and "failure" not in result:
        result = {"failure": f"worker exited with status {process.returncode}"}
    markers = _markers(stdout)
    if not markers and isinstance(result, dict):
        markers = _markers(json.dumps(result.get("progress_markers", [])))
    return result, markers


def main(argv: list[str] | None = None) -> int:
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="run the capability-authorized local-model pilot")
    parser.add_argument("--output", type=Path, help="exclusive 0600 result path; required with --execute")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return _worker()
    plan = {"model": MODEL_ID, "revision": EXPECTED_REVISION, "baseline": Q2_INCUMBENT, "candidate": Q3A_CANDIDATE, "processes": PROCESSES, "repeats": REPEATS, "warmup": WARMUP, "max_tokens": MAX_TOKENS, "child_timeout_seconds": CHILD_TIMEOUT_SECONDS, "worker_max_seconds": WORKER_MAX_SECONDS, "postflight_reserve_seconds": POSTFLIGHT_RESERVE_SECONDS, "deadline_seconds": PILOT_DEADLINE_SECONDS, "runtime_code_sha256": runtime_code_sha256(), "promotion_allowed": False}
    if not args.execute:
        print(json.dumps({"schema": "ironmule.q3a_plan.v2", "experiment": EXPERIMENT_ID, "estimated_wall_seconds": PROCESSES * CHILD_TIMEOUT_SECONDS + 60, "plan": plan}, indent=2, sort_keys=True))
        return 0
    if args.output is None:
        parser.error("--output is required with --execute")
    if not args.output.parent.is_dir() or os.path.lexists(args.output):
        print("q3a: output path must have an existing parent and must not already exist", file=sys.stderr)
        return 2
    deadline = started + PILOT_DEADLINE_SECONDS
    pre = preflight(deadline=deadline)
    if not pre["passed"]:
        result = _failure(plan, pre, "preflight gate failed; no model process started")
    elif time.monotonic() >= deadline:
        result = _failure(plan, pre, "global deadline exhausted before worker start")
    else:
        try:
            worker, partial = _start_worker(pre["identity"], deadline)
            after = postflight(deadline=deadline)
            if "failure" in worker:
                result = _failure(plan, pre, worker["failure"], worker.get("partial_children"), partial)
            else:
                gates = validate_result(worker, pre, after, plan["runtime_code_sha256"], deadline=deadline)
                if gates["passed"]:
                    result = {"schema": "ironmule.q3a_result.v2", "experiment": EXPERIMENT_ID, "status": "INFORMATION_GAIN", "fallback": None, "promotion_allowed": False, "result_type": "INFORMATION_GAIN", "interpretation": "PATH_INTERACTION_ONLY", "analysis": analyze_ratio(worker), "plan": plan, "preflight": pre, "post_environment": after, "gates": gates, "raw_result": worker}
                else:
                    result = _failure(plan, pre, "postflight gate failed", worker.get("partial_children"), worker.get("progress_markers", []))
        except BaseException as exc:
            result = _failure(plan, pre, f"execution failed: {type(exc).__name__}")
    _write_exclusive(args.output, (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    print(json.dumps({"status": result["status"], "output": str(args.output), "promotion_allowed": False}, sort_keys=True))
    return 0 if result["status"] == "INFORMATION_GAIN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
