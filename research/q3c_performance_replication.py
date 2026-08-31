#!/usr/bin/env python3
"""Q3c: exact, two-phase Q2 replication and fused-argmax preservation.

The parent is deliberately stdlib-only.  IronMule/MLX is loaded only inside the
capability-bound worker, after the inherited Q3b gates and exact repository
root check have passed.  Results are fail-closed and never promote a profile.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import re
import resource
import secrets
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping

EXPERIMENT_ID = "Q3c-performance-replication"
MODEL_ID = "mlx-community/gemma-3-4b-it-4bit"
EXPECTED_REVISION = "93724907d4ed1745d2fe50baadf3b0b01a65abf2"
PHASES = ("R", "N")
PROCESSES = 6
REPEATS = 7
WARMUP = 2
MAX_TOKENS = 32
PROMPT_TOKENS = 322
CHILD_TIMEOUT_SECONDS = 35
WORKER_MAX_SECONDS = 240
PHASE_MAX_SECONDS = 270
POST_PHASE_SECONDS = 30
FINAL_RESERVE_SECONDS = 60
STUDY_MAX_SECONDS = 600
COMMAND_TIMEOUT_SECONDS = 1.0
START_SWAP_LIMIT_BYTES = 4 * 1024**3
SWAP_DELTA_LIMIT_BYTES = 128 * 1024**2
START_FREE_PERCENT = 35
AFTER_FREE_PERCENT = 20
PEAK_CEILING_FRACTION = 0.60
SAMPLE_INTERVAL_SECONDS = 0.25
MAX_SWAP_SAMPLE_GAP_SECONDS = SAMPLE_INTERVAL_SECONDS + COMMAND_TIMEOUT_SECONDS + 0.5
# 240 s / 0.25 s plus start/final samples is ~962; retain bounded headroom.
MAX_SWAP_SAMPLES = 2048
MAX_WORKER_OUTPUT = 512 * 1024
CAPABILITY_MAX_BYTES = 16 * 1024
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260825
MAX_PID = 2**31 - 1
MAX_EVIDENCE_INTEGER = 2**63 - 1
MAX_TIMING_NS = CHILD_TIMEOUT_SECONDS * 1_000_000_000
HISTORICAL_RATIO = 0.8568
HISTORICAL_CI = (0.8549, 0.9402)

BASE = {
    "fuse_projections": False, "compiled_fixed_cache": False, "fused_argmax": False,
    "head_skip_prefill": False, "prefill_into_fixed": False, "readback_every": 1,
    "speculate_k": 0, "speculate_ngram": 3, "capacity_slack": 0, "wired_fraction": 0.0,
}
INCUMBENT = {
    **BASE, "compiled_fixed_cache": True, "head_skip_prefill": True, "readback_every": 2,
}
CANDIDATE = {**INCUMBENT, "fused_argmax": True}
PHASE_ARMS = {"R": {"baseline": BASE, "incumbent": INCUMBENT},
              "N": {"baseline": BASE, "candidate": CANDIDATE}}
PHASE_CANDIDATE = {"R": "incumbent", "N": "candidate"}
PREREGISTRATION = Path(__file__).resolve().parent / "raw" / "Q3c_preregistration.md"
PREREGISTRATION_SHA = Path(__file__).resolve().parent / "raw" / "Q3c_preregistration.sha256"
ALLOWED_UNTRACKED = frozenset({"research/data/squad-dev-v1.1.json"})


class Q3cRefused(RuntimeError):
    """A missing or untrusted safety/evidence fact."""


def _reject_json_constant(value: str) -> None:
    """Reject NaN/Infinity so all worker evidence remains strict JSON."""
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


_Q3B: Any = None


def _load_q3b_helpers() -> Any:
    """Load the committed Q3b policy module only on an execution path."""
    global _Q3B
    if _Q3B is not None:
        return _Q3B
    root = Path(__file__).resolve().parents[1]
    package = root / "ironmule" / "__init__.py"
    if not package.is_file():
        raise Q3cRefused("exact IronMule package root is unavailable")
    for name, module in tuple(sys.modules.items()):
        if module is None or not (name == "ironmule" or name.startswith("ironmule.")):
            continue
        origin = getattr(module, "__file__", None) or getattr(getattr(module, "__spec__", None), "origin", None)
        if not isinstance(origin, str) or Path(origin).resolve() != package.resolve() if name == "ironmule" else (not isinstance(origin, str) or root not in Path(origin).resolve().parents):
            raise Q3cRefused("preloaded foreign IronMule module")
    root_text = str(root)
    sys.path[:] = [entry for entry in sys.path if entry != root_text]
    sys.path.insert(0, root_text)
    try:
        package_spec = importlib.util.find_spec("ironmule")
    except (ImportError, AttributeError, ValueError) as exc:
        raise Q3cRefused("exact IronMule module spec is unavailable") from exc
    if package_spec is None or Path(str(getattr(package_spec, "origin", ""))).resolve() != package.resolve():
        raise Q3cRefused("IronMule module spec resolves outside exact repository root")
    path = Path(__file__).resolve().with_name("q3b_residual_swap_canary.py")
    spec = importlib.util.spec_from_file_location("ironmule_q3c_q3b_policy", path)
    if spec is None or spec.loader is None:
        raise Q3cRefused("committed Q3b policy is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _Q3B = module
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_code_sha256(root: Path | None = None) -> str:
    root = Path(__file__).resolve().parents[1] if root is None else root.resolve()
    paths = sorted((root / "ironmule").rglob("*.py"))
    for extra in (root / "research" / "q3a_path_interaction.py", root / "research" / "q3b_residual_swap_canary.py", Path(__file__).resolve()):
        if extra.is_file() and extra not in paths:
            paths.append(extra)
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.relative_to(root).as_posix()):
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


def phase_plan(phase: str) -> dict[str, Any]:
    if phase not in PHASES:
        raise ValueError("unknown Q3c phase")
    return {"phase": phase, "arms": PHASE_ARMS[phase], "candidate": PHASE_CANDIDATE[phase],
            "processes": PROCESSES, "repeats": REPEATS, "warmup": WARMUP,
            "prompt_tokens": PROMPT_TOKENS, "max_tokens": MAX_TOKENS,
            "order": [["baseline", PHASE_CANDIDATE[phase]] if i % 2 == 0
                       else [PHASE_CANDIDATE[phase], "baseline"] for i in range(PROCESSES)]}


def _read_capability() -> dict[str, Any]:
    fd_text = os.environ.get("IRONMULE_Q3C_CAP_FD")
    nonce = os.environ.get("IRONMULE_Q3C_CAP_NONCE")
    encoded = os.environ.get("IRONMULE_Q3C_EXPECTED")
    if not fd_text or not nonce or not encoded:
        raise Q3cRefused("worker capability is absent")
    fd = -1
    try:
        fd = int(fd_text)
        if fd < 0:
            raise ValueError
        data = os.read(fd, CAPABILITY_MAX_BYTES)
        if len(data) >= CAPABILITY_MAX_BYTES:
            raise ValueError
        payload = json.loads(data.decode(), parse_constant=_reject_json_constant)
        expected = json.loads(encoded, parse_constant=_reject_json_constant)
    except (OSError, ValueError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
        raise Q3cRefused("worker capability is malformed") from exc
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
    fields = {"identity", "runtime_code_sha256", "phase", "initial_swap", "installed_memory", "phase_plan"}
    if (not isinstance(payload, dict) or not isinstance(expected, dict)
            or payload.get("nonce") != nonce or not _strict_equal(payload.get("expected"), expected)
            or set(expected) != fields):
        raise Q3cRefused("worker capability or expected binding mismatch")
    return expected


def _activate_exact_repo_root() -> Path:
    return _load_q3b_helpers()._activate_exact_repo_root()


def _verify_exact_repo_root() -> Path:
    """Check the worker import boundary without importing IronMule."""
    root = Path(__file__).resolve().parents[1]
    package = root / "ironmule" / "__init__.py"
    if not package.is_file():
        raise Q3cRefused("exact IronMule package root is unavailable")
    for name, module in tuple(sys.modules.items()):
        if module is None or not (name == "ironmule" or name.startswith("ironmule.")):
            continue
        origin = getattr(module, "__file__", None) or getattr(getattr(module, "__spec__", None), "origin", None)
        if not isinstance(origin, str) or root not in Path(origin).resolve().parents:
            raise Q3cRefused("preloaded foreign IronMule module")
    root_text = str(root)
    sys.path[:] = [entry for entry in sys.path if entry != root_text]
    sys.path.insert(0, root_text)
    try:
        spec = importlib.util.find_spec("ironmule")
    except (ImportError, AttributeError, ValueError) as exc:
        raise Q3cRefused("exact IronMule module spec is unavailable") from exc
    if spec is None or Path(str(getattr(spec, "origin", ""))).resolve() != package.resolve():
        raise Q3cRefused("IronMule module spec resolves outside exact repository root")
    return root


def _q3b_runtime() -> Any:
    return _load_q3b_helpers()


def _paired_ratio(candidate: list[float], baseline: list[float]) -> dict[str, Any]:
    import random
    pairs = [c / b for c, b in zip(candidate, baseline)]
    rng = random.Random(BOOTSTRAP_SEED)
    medians = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        medians.append(statistics.median([pairs[rng.randrange(len(pairs))] for _ in pairs]))
    medians.sort()
    return {"median_ratio": statistics.median(pairs), "ci_low": medians[int(0.025 * BOOTSTRAP_RESAMPLES)],
            "ci_high": medians[int(0.975 * BOOTSTRAP_RESAMPLES)], "pairs": pairs}


def _summarise(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    return {"n": len(ordered), "median": statistics.median(ordered), "min": ordered[0], "max": ordered[-1],
            "p95": ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))],
            "stdev": statistics.stdev(ordered) if len(ordered) > 1 else 0.0}


def _untracked_runtime_inventory(root: Path, bounded: Callable[[list[str]], str],
                                 q3b: Any) -> dict[str, Any]:
    """Fail closed when any untracked file beyond the licensed fixture exists."""
    try:
        output = bounded([q3b.COMMANDS["git"], "-C", str(root),
                          "ls-files", "--others", "--exclude-standard"])
        if not isinstance(output, str) or not getattr(output, "ok", True):
            return {"passed": False, "paths": [], "unexpected": [],
                    "reason": "git untracked inventory unavailable"}
        paths = [line for line in output.splitlines() if line]
        unexpected = sorted(set(paths) - ALLOWED_UNTRACKED)
        return {"passed": not unexpected, "paths": paths, "unexpected": unexpected,
                "allowed": sorted(ALLOWED_UNTRACKED)}
    except BaseException as exc:
        return {"passed": False, "paths": [], "unexpected": [],
                "reason": f"git untracked inventory failed: {type(exc).__name__}"}


def preflight(*, root: Path | None = None, deadline: float | None = None) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1] if root is None else root.resolve()
    q3b = _q3b_runtime()
    bounded = q3b._deadline_runner(deadline, q3b._run_text) if deadline is not None else q3b._run_text
    env = q3b.system_environment(bounded)
    memory = q3b.installed_memory_bytes(bounded)
    try:
        identity = q3b.resolve_exact_local_identity(root)
        identity_error = None
    except Exception as exc:
        identity, identity_error = None, str(exc)
    load = q3b.loadavg_gate(deadline=deadline)
    git = q3b._git_binding(root, bounded)
    untracked = _untracked_runtime_inventory(root, bounded, q3b)
    checks = {
        "model_identity_exact": isinstance(identity, dict) and identity.get("model_id") == MODEL_ID and identity.get("model_revision") == EXPECTED_REVISION and bool(re.fullmatch(r"[0-9a-f]{64}", str(identity.get("model_manifest_sha256", "")))),
        "ac_power": env.get("power_source") == "AC", "low_power_off": env.get("low_power_mode") is False,
        "thermal_nominal": env.get("thermal_state") == "nominal",
        "no_competing_model_process": q3b.competing_model_process(bounded) is None,
        "untracked_runtime_clean": untracked["passed"],
        "loadavg_gate": load.get("passed") is True,
        "start_swap_known": isinstance(env.get("swap_used_bytes"), int),
        "start_swap_within_4gib": isinstance(env.get("swap_used_bytes"), int) and env["swap_used_bytes"] <= START_SWAP_LIMIT_BYTES,
        "installed_memory_known": isinstance(memory, int) and memory > 0,
        "start_memory_free_at_least_35_percent": isinstance(env.get("memory_free_percent"), int) and env["memory_free_percent"] >= START_FREE_PERCENT,
        "git_clean_and_bound": git.get("clean") is True and bool(git.get("commit")),
        "preregistration_matches": _preregistration_matches(),
        "runtime_code_hash_known": bool(re.fullmatch(r"[0-9a-f]{64}", runtime_code_sha256(root))),
    }
    return {"environment": env, "identity": identity, "identity_error": identity_error,
            "git": git, "untracked": untracked, "loadavg": load, "installed_memory_bytes": memory,
            "runtime_code_sha256": runtime_code_sha256(root),
            "peak_ceiling_bytes": int(memory * PEAK_CEILING_FRACTION) if memory else None,
            "checks": checks, "passed": all(checks.values())}


def _finite_positive(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value)) and float(value) > 0
    except (OverflowError, ValueError, TypeError):
        return False


def _bounded_nonnegative_integer(value: Any, *, upper: int = MAX_EVIDENCE_INTEGER) -> bool:
    return type(value) is int and 0 <= value <= upper


def _strict_equal(actual: Any, expected: Any) -> bool:
    """Compare decoded evidence without Python's bool/int or int/float coercion."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return (len(actual) == len(expected)
                and all(key in actual and _strict_equal(actual[key], value)
                        for key, value in expected.items()))
    if isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(
            _strict_equal(item, value) for item, value in zip(actual, expected)
        )
    return actual == expected


def _timing_value(value: Any) -> bool:
    return _finite_positive(value) and float(value) <= MAX_TIMING_NS


def _rate_arrays(child: Mapping[str, Any], arm_name: str) -> dict[str, list[float]]:
    arm = child["arms"][arm_name]
    physical = arm["physical_tokens_per_repeat"]
    total = arm["total_ns"]
    decode = arm["decode_ns"]
    steps = arm["decode_steps"]
    try:
        tok_rates = [len(ids) / (float(ns) / 1e9) for ids, ns in zip(physical, total)]
        step_rates = [steps / (float(ns) / 1e9) for ns in decode]
    except (TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
        raise ValueError("rate formula is not finite") from exc
    if any(not _finite_positive(v) for v in tok_rates + step_rates):
        raise ValueError("rate formula is not finite")
    return {"physical_output_tokens_per_s": tok_rates, "decode_steps_per_s": step_rates}


def derive_rates(result: Mapping[str, Any], candidate: str) -> dict[str, Any]:
    names = ["baseline", candidate]
    by_arm = {name: [] for name in names}
    process_medians = {name: {"physical_output_tokens_per_s": [], "decode_steps_per_s": []} for name in names}
    for child in result["raw"]:
        for name in names:
            rates = _rate_arrays(child, name)
            by_arm[name].append(rates)
            for metric, values in rates.items():
                process_medians[name][metric].append(float(statistics.median(values)))
    q3b = _q3b_runtime()
    ratios = {metric: _paired_ratio(process_medians[candidate][metric], process_medians["baseline"][metric])
              for metric in process_medians["baseline"]}
    return {"formula": {"physical_output_tokens_per_s": "len(physical_tokens_per_repeat) / (total_ns / 1e9)",
                         "decode_steps_per_s": "decode_steps / (decode_ns / 1e9)"},
            "per_repeat": by_arm, "process_medians": process_medians,
            "ratios": {f"{candidate}/baseline": ratios}}


def _ab_validate(result: Any, phase: str, candidate: str) -> tuple[bool, str]:
    """Pure evaluator-owned equivalent of ``ab.validate_result``."""
    if (not isinstance(result, dict) or phase not in PHASES
            or candidate != PHASE_CANDIDATE.get(phase)):
        return False, "top_fields"
    expected_top = {"arms", "processes", "repeats", "warmup", "raw", "per_arm", "token_identity",
                    "token_count_identity", "stop_reason_identity", "deterministic", "reference_tokens", "ratios"}
    if (set(result) != expected_top or not _strict_equal(result["arms"], PHASE_ARMS[phase])
            or type(result["processes"]) is not int or result["processes"] != PROCESSES
            or type(result["repeats"]) is not int or result["repeats"] != REPEATS
            or type(result["warmup"]) is not int or result["warmup"] != WARMUP
            or not isinstance(result["raw"], list)
            or len(result["raw"]) != PROCESSES):
        return False, "top_fields"
    pids: set[int] = set()
    names = ("baseline", candidate)
    if (not isinstance(result["per_arm"], dict) or set(result["per_arm"]) != set(names)
            or any(not isinstance(result["per_arm"].get(name), dict)
                   or set(result["per_arm"][name]) != {"total_ns", "prefill_ns", "decode_ns"}
                   for name in names)):
        return False, "summary_fields"
    for index, child in enumerate(result["raw"]):
        if (not isinstance(child, dict) or set(child) != {"pid", "arms", "order", "mlx_peak_bytes"}
                or child["order"] != phase_plan(phase)["order"][index]
                or not isinstance(child["arms"], dict)
                or set(child["arms"]) != set(names)
                or not _bounded_nonnegative_integer(child["pid"], upper=MAX_PID)
                or child["pid"] <= 0
                or child["pid"] in pids
                or not _bounded_nonnegative_integer(child["mlx_peak_bytes"])
                or child["mlx_peak_bytes"] <= 0):
            return False, "child_fields"
        pids.add(child["pid"])
        for name in names:
            arm = child["arms"][name]
            expected_arm = {"total_ns", "prefill_ns", "decode_ns", "logical_tokens", "logical_tokens_per_repeat",
                            "physical_tokens_per_repeat", "token_counts", "stop_reasons", "capacities", "deterministic",
                            "decode_steps", "prompt_tokens", "mlx_peak_bytes"}
            if not isinstance(arm, dict) or set(arm) != expected_arm:
                return False, "arm_fields"
            array_fields = ("total_ns", "prefill_ns", "decode_ns", "logical_tokens_per_repeat",
                            "physical_tokens_per_repeat", "token_counts", "stop_reasons", "capacities")
            if any(not isinstance(arm[field], list) or len(arm[field]) != REPEATS for field in array_fields):
                return False, "arm_arrays"
            if any(not _timing_value(value) for field in ("total_ns", "prefill_ns", "decode_ns") for value in arm[field]):
                return False, "timing"
            if (not isinstance(arm["logical_tokens"], list) or not isinstance(arm["deterministic"], bool)
                    or not _bounded_nonnegative_integer(arm["decode_steps"])
                    or not _bounded_nonnegative_integer(arm["prompt_tokens"])
                    or not _bounded_nonnegative_integer(arm["mlx_peak_bytes"])
                    or arm["mlx_peak_bytes"] <= 0):
                return False, "arm_scalars"
            if (any(not isinstance(rep, list) or any(not _bounded_nonnegative_integer(token) for token in rep)
                    for rep in arm["logical_tokens_per_repeat"] + arm["physical_tokens_per_repeat"])
                    or any(not rep for rep in arm["logical_tokens_per_repeat"] + arm["physical_tokens_per_repeat"])
                    or arm["logical_tokens"] != arm["logical_tokens_per_repeat"][0]
                    or arm["decode_steps"] != len(arm["physical_tokens_per_repeat"][0]) - 1):
                return False, "token_reference"
            for logical, physical, counts in zip(arm["logical_tokens_per_repeat"], arm["physical_tokens_per_repeat"], arm["token_counts"]):
                if (not isinstance(counts, dict) or set(counts) != {"logical", "physical"}
                        or not _bounded_nonnegative_integer(counts["logical"])
                        or not _bounded_nonnegative_integer(counts["physical"])
                        or counts != {"logical": len(logical), "physical": len(physical)}):
                    return False, "token_counts"
            if (any(stop not in {"eos", "length"} for stop in arm["stop_reasons"])
                    or any(not _bounded_nonnegative_integer(cap) or cap <= 0 for cap in arm["capacities"])):
                return False, "stop_or_capacity"
            deterministic = all(logical == arm["logical_tokens_per_repeat"][0]
                                and physical == arm["physical_tokens_per_repeat"][0]
                                and counts == arm["token_counts"][0]
                                and stop == arm["stop_reasons"][0]
                                and cap == arm["capacities"][0]
                                for logical, physical, counts, stop, cap in zip(arm["logical_tokens_per_repeat"], arm["physical_tokens_per_repeat"], arm["token_counts"], arm["stop_reasons"], arm["capacities"]))
            if arm["deterministic"] is not deterministic:
                return False, "determinism"
    for name in names:
        for metric in ("total_ns", "prefill_ns", "decode_ns"):
            values = [statistics.median(child["arms"][name][metric]) for child in result["raw"]]
            if not _strict_equal(result["per_arm"][name][metric], _summarise(values)):
                return False, "summary"
    reference = result["raw"][0]["arms"]["baseline"]
    for name in names:
        for child in result["raw"]:
            arm = child["arms"][name]
            if any(arm[field] != reference[field] for field in ("logical_tokens_per_repeat", "physical_tokens_per_repeat", "token_counts", "stop_reasons", "capacities", "decode_steps", "prompt_tokens")):
                return False, "identity"
    if (result["reference_tokens"] != reference["logical_tokens"]
            or result["token_identity"] is not True or result["token_count_identity"] is not True
            or result["stop_reason_identity"] is not True or result["deterministic"] is not True):
        return False, "identity_flags"
    pair_name = f"{candidate}/baseline"
    if set(result["ratios"]) != {pair_name}:
        return False, "ratio_pairs"
    for metric in ("total_ns", "prefill_ns", "decode_ns"):
        expected = _paired_ratio([statistics.median(child["arms"][candidate][metric]) for child in result["raw"]], [statistics.median(child["arms"]["baseline"][metric]) for child in result["raw"]])
        if result["ratios"][pair_name].get(metric) != expected:
            return False, "ratio"
    return True, "ok"


def validate_phase_result(result: Any, phase: str, expected_binding: Mapping[str, Any] | None = None) -> tuple[bool, str]:
    if type(phase) is not str or phase not in PHASES or not isinstance(result, dict):
        return False, "phase or result malformed"
    candidate = PHASE_CANDIDATE[phase]
    required = {"phase", "phase_plan", "arms", "processes", "repeats", "warmup", "raw", "per_arm",
                "token_identity", "token_count_identity", "stop_reason_identity", "deterministic",
                "reference_tokens", "ratios", "binding", "child_rss_peak_bytes", "swap_samples",
                "swap_sample_times", "swap_sample_offsets", "sampler_errors", "max_swap_used_bytes",
                "derived_rates", "cleanup", "phase_initial_swap_bytes"}
    if (set(result) != required or not _strict_equal(result["phase"], phase)
            or not _strict_equal(result["phase_plan"], phase_plan(phase))):
        return False, "phase schema or plan mismatch"
    ok, reason = _ab_validate({k: result[k] for k in ("arms", "processes", "repeats", "warmup", "raw", "per_arm", "token_identity", "token_count_identity", "stop_reason_identity", "deterministic", "reference_tokens", "ratios")}, phase, candidate)
    if not ok:
        return False, f"ab.validate_result:{reason}"
    ratio_values = result["ratios"].get(f"{candidate}/baseline")
    if not isinstance(ratio_values, dict):
        return False, "ratio pair is missing"
    for metric in ("total_ns", "prefill_ns", "decode_ns"):
        ratio = ratio_values.get(metric)
        if (not isinstance(ratio, dict) or not _finite_positive(ratio.get("median_ratio"))
                or not _finite_positive(ratio.get("ci_low")) or not _finite_positive(ratio.get("ci_high"))
                or not isinstance(ratio.get("pairs"), list) or len(ratio["pairs"]) != PROCESSES
                or any(not _finite_positive(value) for value in ratio["pairs"])):
            return False, "ratio contains non-finite or invalid evidence"
    if not isinstance(result["raw"], list) or len(result["raw"]) != PROCESSES or len({c.get("pid") for c in result["raw"] if isinstance(c, dict)}) != PROCESSES:
        return False, "PIDs are incomplete or duplicated"
    if any(not isinstance(child, dict) or not _bounded_nonnegative_integer(child.get("pid"), upper=MAX_PID)
           or not _bounded_nonnegative_integer(child.get("mlx_peak_bytes"))
           for child in result["raw"]):
        return False, "child PID or peak has invalid type/bound"
    if (type(result["phase_initial_swap_bytes"]) is not int
            or result["phase_initial_swap_bytes"] < 0
            or result["phase_initial_swap_bytes"] > START_SWAP_LIMIT_BYTES):
        return False, "phase initial swap evidence is invalid"
    expected_orders = phase_plan(phase)["order"]
    if any(not isinstance(child, dict) or child.get("order") != expected_orders[i] for i, child in enumerate(result["raw"])):
        return False, "AB/BA process order mismatch"
    for child in result["raw"]:
        for name in ("baseline", candidate):
            arm = child["arms"][name]
            if (arm["prompt_tokens"] != PROMPT_TOKENS or arm["deterministic"] is not True
                    or not _bounded_nonnegative_integer(arm["decode_steps"])
                    or arm["decode_steps"] != len(arm["physical_tokens_per_repeat"][0]) - 1
                    or not _bounded_nonnegative_integer(arm["mlx_peak_bytes"])):
                return False, "exact prompt/token metadata gate failed"
            if any(any(not _bounded_nonnegative_integer(token) for token in repeat)
                   for repeat in arm["logical_tokens_per_repeat"] + arm["physical_tokens_per_repeat"]):
                return False, "token ID has invalid type/bound"
            if any(not _bounded_nonnegative_integer(item) for item in arm["capacities"]):
                return False, "capacity has invalid type/bound"
    reference = result["raw"][0]["arms"]["baseline"]
    for child in result["raw"]:
        for name in ("baseline", candidate):
            arm = child["arms"][name]
            for key in ("logical_tokens_per_repeat", "physical_tokens_per_repeat", "token_counts", "stop_reasons", "capacities", "prompt_tokens", "decode_steps"):
                if arm[key] != reference[key]:
                    return False, "cross-arm token metadata mismatch"
    binding = result["binding"]
    current_hash = runtime_code_sha256()
    if (not isinstance(binding, dict) or set(binding) != {"model_id", "model_revision", "model_manifest_sha256", "runtime_code_sha256"}
            or binding.get("model_id") != MODEL_ID or binding.get("model_revision") != EXPECTED_REVISION
            or not re.fullmatch(r"[0-9a-f]{64}", str(binding.get("model_manifest_sha256")))
            or binding.get("runtime_code_sha256") != current_hash
            or (expected_binding is not None and any(binding.get(key) != expected_binding.get(key)
                                                       for key in ("model_id", "model_revision", "model_manifest_sha256", "runtime_code_sha256")))):
        return False, "model/runtime binding mismatch"
    samples, times, offsets = result["swap_samples"], result["swap_sample_times"], result["swap_sample_offsets"]
    if (result["sampler_errors"] != [] or not isinstance(samples, list) or len(samples) < 2 or len(samples) > MAX_SWAP_SAMPLES
            or not isinstance(times, list) or not isinstance(offsets, list) or len(times) != len(samples) or len(offsets) != len(samples)
            or any(not _bounded_nonnegative_integer(v) for v in samples)
            or any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(float(v)) or v < 0 for v in times) or any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(float(v)) or v < 0 for v in offsets)
            or any(times[i] > times[i + 1] for i in range(len(times) - 1)) or any(offsets[i] > offsets[i + 1] for i in range(len(offsets) - 1))
            or any(times[i + 1] - times[i] > MAX_SWAP_SAMPLE_GAP_SECONDS for i in range(len(times) - 1))
            or any(offsets[i + 1] - offsets[i] > MAX_SWAP_SAMPLE_GAP_SECONDS for i in range(len(offsets) - 1))
            or result["phase_initial_swap_bytes"] != samples[0]
            or not _bounded_nonnegative_integer(result["max_swap_used_bytes"])
            or result["max_swap_used_bytes"] != max(samples)
            or result["max_swap_used_bytes"] - result["phase_initial_swap_bytes"] > SWAP_DELTA_LIMIT_BYTES):
        return False, "swap sampler evidence is invalid"
    if (not _bounded_nonnegative_integer(result["child_rss_peak_bytes"]) or result["child_rss_peak_bytes"] <= 0
            or not isinstance(result["cleanup"], dict) or set(result["cleanup"]) != {"worker_group_gone", "cleanup_errors", "child_cleanup_errors"}
            or result["cleanup"]["worker_group_gone"] is not True or result["cleanup"]["cleanup_errors"] != []
            or result["cleanup"]["child_cleanup_errors"] != []):
        return False, "cleanup evidence is invalid"
    try:
        derived = derive_rates(result, candidate)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        return False, "derived rate formula is invalid"
    if result["derived_rates"] != derived:
        return False, "derived rates do not match raw evidence"
    return True, "ok"


def cross_phase_identity(replication: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, bool]:
    rb, nc = replication["raw"], candidate["raw"]
    ra, ca = rb[0]["arms"]["baseline"], nc[0]["arms"]["baseline"]
    def field(name: str) -> bool:
        return all(r["arms"]["baseline"][name] == n["arms"]["baseline"][name] for r, n in zip(rb, nc))
    return {"model_binding": replication["binding"] == candidate["binding"],
            "pids_disjoint": not ({row["pid"] for row in rb} & {row["pid"] for row in nc}),
            "prompt_tokens": field("prompt_tokens"), "logical_tokens": field("logical_tokens_per_repeat"),
            "physical_tokens": field("physical_tokens_per_repeat"), "token_counts": field("token_counts"),
            "stop_reasons": field("stop_reasons"), "capacities": field("capacities"),
            "decode_steps": field("decode_steps"), "deterministic": field("deterministic")}


def decide(replication: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    rratio = replication["ratios"]["incumbent/baseline"]["total_ns"]
    nratio = candidate["ratios"]["candidate/baseline"]["total_ns"]
    rpass = abs(rratio["median_ratio"] - HISTORICAL_RATIO) <= 0.03 and rratio["ci_low"] <= HISTORICAL_RATIO <= rratio["ci_high"] and rratio["ci_high"] < 1.0
    n_checks = {
        "ci_high_below_one": nratio["ci_high"] < 1.0,
        "median_within_replicated_phase_R": nratio["median_ratio"] <= rratio["median_ratio"] + 0.005,
    }
    npass = all(n_checks.values())
    def directionally_labelled(ratios: Mapping[str, Any], *, higher_is_better: bool) -> dict[str, Any]:
        def label(value: Mapping[str, Any]) -> dict[str, Any]:
            if higher_is_better:
                low, high = 100.0 * (float(value["ci_low"]) - 1.0), 100.0 * (float(value["ci_high"]) - 1.0)
                median = 100.0 * (float(value["median_ratio"]) - 1.0)
            else:
                # Inverting a lower-is-better ratio reverses the CI endpoints.
                low, high = 100.0 * (1.0 - float(value["ci_high"])), 100.0 * (1.0 - float(value["ci_low"]))
                median = 100.0 * (1.0 - float(value["median_ratio"]))
            return {**value, "percent_faster": median, "percent_ci_low": low,
                    "percent_ci_high": high, "direction": "higher_is_better" if higher_is_better else "lower_is_better"}
        return {metric: label(value) for metric, value in ratios.items()}
    return {"phase_R_reproduced": rpass, "phase_N_preserved": npass,
            "phase_N_checks": n_checks,
            "status": "COMPLETE_PASS" if rpass and npass else "CRITERIA_MISS",
            "fallback": "BASE/current incumbent", "promotion_allowed": False,
            "metrics": {"phase_R": replication["ratios"], "phase_N": candidate["ratios"],
                        "phase_R_rates": replication["derived_rates"]["ratios"],
                        "phase_N_rates": candidate["derived_rates"]["ratios"],
                        "time_percent_faster": directionally_labelled(candidate["ratios"]["candidate/baseline"] if "candidate/baseline" in candidate["ratios"] else candidate["ratios"].get("incumbent/baseline", {}), higher_is_better=False),
            "rate_percent_faster": directionally_labelled(candidate["derived_rates"]["ratios"]["candidate/baseline"], higher_is_better=True)}}


def phase_r_performance_gate(result: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate and record all three preregistered Phase-R criteria."""
    ratio = result["ratios"]["incumbent/baseline"]["total_ns"]
    checks = {
        "historical_absolute_tolerance": abs(float(ratio["median_ratio"]) - HISTORICAL_RATIO) <= 0.03,
        "historical_ratio_in_ci": float(ratio["ci_low"]) <= HISTORICAL_RATIO <= float(ratio["ci_high"]),
        "ci_high_below_one": float(ratio["ci_high"]) < 1.0,
    }
    return {"checks": checks, "passed": all(checks.values()), "ratio": ratio}


def _group_gone(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    return False


def _record_completed_child(records: list[dict[str, Any]], lock: threading.Lock,
                            record: Mapping[str, Any]) -> None:
    """Append a defensive completed-child snapshot under its own lock."""
    if not isinstance(record, dict):
        raise Q3cRefused("completed child record is not an object")
    snapshot = copy.deepcopy(record)
    with lock:
        if len(records) >= PROCESSES:
            raise Q3cRefused("completed child record limit exceeded")
        records.append(snapshot)


def _write_bounded_safety_marker(event: Mapping[str, Any],
                                 records: list[dict[str, Any]], lock: threading.Lock,
                                 emit: Callable[[dict[str, Any]], None]) -> None:
    """Publish safety evidence with full completed children and no argv field."""
    with lock:
        completed = copy.deepcopy(records)
    bounded_event = {
        "reason": str(event.get("reason", "safety abort"))[:256],
        "samples": list(event.get("samples", []))[:MAX_SWAP_SAMPLES],
        "times": list(event.get("times", []))[:MAX_SWAP_SAMPLES],
        "offsets": list(event.get("offsets", []))[:MAX_SWAP_SAMPLES],
        "errors": [str(item)[:512] for item in list(event.get("errors", []))[:32]],
        "partial_children": completed,
    }
    encoded = json.dumps(bounded_event, sort_keys=True, allow_nan=False).encode()
    if len(encoded) > MAX_WORKER_OUTPUT or b"argv" in encoded:
        raise Q3cRefused("safety marker exceeds bound or contains argv")
    emit(bounded_event)


def _start_phase(phase: str, identity: dict[str, Any], initial_swap: int, installed: int, deadline: float) -> dict[str, Any]:
    nonce = secrets.token_urlsafe(24)
    read_fd, write_fd = os.pipe()
    expected = {"identity": identity, "runtime_code_sha256": runtime_code_sha256(), "phase": phase,
                "initial_swap": initial_swap, "installed_memory": installed, "phase_plan": phase_plan(phase)}
    payload = json.dumps({"nonce": nonce, "expected": expected}, sort_keys=True, separators=(",", ":")).encode()
    if len(payload) >= CAPABILITY_MAX_BYTES:
        os.close(read_fd); os.close(write_fd); raise Q3cRefused("capability payload exceeds bound")
    os.set_inheritable(read_fd, True)
    try:
        os.write(write_fd, payload); os.close(write_fd); write_fd = -1
        timeout = min(float(WORKER_MAX_SECONDS), deadline - time.monotonic() - POST_PHASE_SECONDS)
        if timeout <= 0:
            raise Q3cRefused("phase deadline exhausted")
        env = {**os.environ, "IRONMULE_Q3C_CAP_FD": str(read_fd), "IRONMULE_Q3C_CAP_NONCE": nonce,
               "IRONMULE_Q3C_EXPECTED": json.dumps(expected, sort_keys=True, separators=(",", ":")),
               "IRONMULE_Q3C_WORKER_DEADLINE": str(time.monotonic() + timeout)}
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--phase-worker"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, pass_fds=(read_fd,), start_new_session=True, cwd=str(Path(__file__).resolve().parents[1]))
    finally:
        if write_fd >= 0:
            os.close(write_fd)
        os.close(read_fd)

    def cleanup_errors() -> list[str]:
        try:
            errors = _q3b_runtime()._cleanup_worker(process)
            if not isinstance(errors, list):
                return ["cleanup returned malformed evidence"]
            return [str(error)[:512] for error in errors]
        except BaseException as exc:
            return [f"cleanup:{type(exc).__name__}: {exc}"[:512]]

    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except BaseException as exc:
        errors = cleanup_errors()
        return {"failure": "phase worker timeout" if isinstance(exc, subprocess.TimeoutExpired) else "phase communication failed", "partial_evidence": [], "cleanup": {"worker_group_gone": not errors, "cleanup_errors": errors, "child_cleanup_errors": []}}
    if not isinstance(stdout, str) or not isinstance(stderr, str):
        errors = cleanup_errors()
        return {"failure": "phase worker communication returned non-text output", "partial_evidence": [], "cleanup": {"worker_group_gone": not errors, "cleanup_errors": errors, "child_cleanup_errors": []}}
    if len(stdout) > MAX_WORKER_OUTPUT or len(stderr) > MAX_WORKER_OUTPUT:
        errors = cleanup_errors()
        return {"failure": "phase worker output exceeded bounded limit", "cleanup": {"worker_group_gone": not errors, "cleanup_errors": errors, "child_cleanup_errors": []}}
    marker = next((line[2:] for line in stdout.splitlines() if line.startswith("@@")), None)
    safety = None
    for line in stdout.splitlines():
        if line.startswith("@SAFETY"):
            try:
                safety = json.loads(line[len("@SAFETY"):].strip(), parse_constant=_reject_json_constant)
            except (TypeError, ValueError, json.JSONDecodeError):
                safety = {"reason": "malformed_safety_marker"}
            break
    if safety is not None:
        errors = cleanup_errors()
        return {"failure": "live safety abort", "safety_event": safety, "partial_evidence": safety, "cleanup": {"worker_group_gone": not errors, "cleanup_errors": errors, "child_cleanup_errors": []}}
    if marker is None:
        errors = cleanup_errors()
        return {"failure": "phase worker returned no result", "cleanup": {"worker_group_gone": not errors, "cleanup_errors": errors, "child_cleanup_errors": []}}
    try:
        result = json.loads(marker, parse_constant=_reject_json_constant)
    except (TypeError, ValueError, json.JSONDecodeError):
        result = {"failure": "phase worker result JSON malformed"}
    if not isinstance(result, dict):
        result = {"failure": "phase worker result JSON malformed"}
    if process.returncode != 0:
        partial = dict(result)
        errors = cleanup_errors()
        result = {"failure": f"phase worker exited with status {process.returncode}", "partial_result": partial,
                  "cleanup": {"worker_group_gone": not errors, "cleanup_errors": errors,
                               "child_cleanup_errors": partial.get("child_cleanup_errors", [])}}
    elif "failure" in result:
        errors = cleanup_errors()
        result["cleanup"] = {"worker_group_gone": not errors, "cleanup_errors": errors,
                              "child_cleanup_errors": result.get("child_cleanup_errors", [])}
    else:
        gone = _group_gone(process.pid)
        cleanup_failure: list[str] = []
        if not gone:
            cleanup_failure = cleanup_errors()
            gone = not cleanup_failure
        result["cleanup"] = {"worker_group_gone": gone,
                              "cleanup_errors": cleanup_failure if cleanup_failure else ([] if gone else ["worker process group still alive"]),
                              "child_cleanup_errors": result.get("child_cleanup_errors", [])}
    return result


def _phase_worker() -> int:
    q3b = None
    samples: list[int] = []
    times: list[float] = []
    offsets: list[float] = []
    errors: list[str] = []
    safety: dict[str, Any] = {}
    partial_children: list[dict[str, Any]] = []
    partial_children_lock = threading.Lock()

    def partial_children_snapshot() -> list[dict[str, Any]]:
        with partial_children_lock:
            return copy.deepcopy(partial_children)

    try:
        expected = _read_capability()
        if runtime_code_sha256() != expected["runtime_code_sha256"]:
            raise Q3cRefused("runtime code changed after preflight")
        phase = expected["phase"]
        if phase not in PHASES or not _strict_equal(expected["phase_plan"], phase_plan(phase)):
            raise Q3cRefused("phase plan mismatch")
        worker_deadline = float(os.environ.get("IRONMULE_Q3C_WORKER_DEADLINE", "nan"))
        if not math.isfinite(worker_deadline):
            raise Q3cRefused("worker deadline malformed")
        _verify_exact_repo_root()
        _activate_exact_repo_root()
        q3b = _q3b_runtime()
        if not _preregistration_matches():
            raise Q3cRefused("Q3c preregistration SHA mismatch in worker")
        started = time.monotonic()
        lock = threading.Lock()
        def sample(label: str) -> None:
            try:
                value = q3b._read_swap_sample(q3b._deadline_runner(worker_deadline))
                stamp = time.monotonic()
                with lock:
                    if len(samples) >= MAX_SWAP_SAMPLES:
                        raise Q3cRefused("swap sampler sample limit exceeded")
                    samples.append(value); times.append(stamp); offsets.append(stamp - started)
            except BaseException as exc:
                with lock:
                    errors.append(f"{label}: {type(exc).__name__}")
                raise
        sample("worker-start")
        phase_initial_swap = samples[0]
        if phase_initial_swap > START_SWAP_LIMIT_BYTES:
            raise Q3cRefused("phase initial swap exceeds 4 GiB")
        if samples[-1] - phase_initial_swap > SWAP_DELTA_LIMIT_BYTES:
            raise Q3cRefused("worker-start swap delta exceeded 128 MiB")
        stop = threading.Event()

        def on_child(_index: int, record: Mapping[str, Any]) -> None:
            """Retain completed child raw evidence independently of sampler state."""
            _record_completed_child(partial_children, partial_children_lock, record)

        def marker_writer(event: Mapping[str, Any]) -> None:
            """Emit bounded safety evidence, including completed children but no argv."""
            _write_bounded_safety_marker(event, partial_children, partial_children_lock,
                                         q3b._emit_safety_marker)

        def monitor() -> None:
            while not stop.wait(SAMPLE_INTERVAL_SECONDS):
                try:
                    sample("periodic")
                    if max(samples) - phase_initial_swap > SWAP_DELTA_LIMIT_BYTES:
                        q3b._capture_live_safety(reason="swap_delta_exceeded", samples=samples, sample_times=times, sample_offsets=offsets, sampler_errors=errors, lock=lock, state=safety, marker_writer=marker_writer)
                        stop.set(); return
                except BaseException as exc:
                    if "event" not in safety:
                        q3b._capture_live_safety(reason="swap_sampler_error", samples=samples, sample_times=times, sample_offsets=offsets, sampler_errors=errors or [type(exc).__name__], lock=lock, state=safety, marker_writer=marker_writer)
                    stop.set(); return
        thread = threading.Thread(target=monitor, daemon=True); thread.start()
        from ironmule import ab
        from ironmule.runtime import Knobs
        from ironmule.tune import resolve_local_model
        resolved = resolve_local_model(MODEL_ID, revision=EXPECTED_REVISION)
        actual = {"model_id": resolved.identity.model_id, "model_revision": resolved.identity.revision, "model_manifest_sha256": resolved.identity.model_manifest_sha256}
        if actual != expected["identity"]:
            raise Q3cRefused("worker model identity differs from parent")
        def before_child(_index: int, _order: list[str]) -> None:
            if errors or "event" in safety:
                raise Q3cRefused("sampler failed before child")
            env = q3b.system_environment(q3b._deadline_runner(worker_deadline)); env["loadavg"] = q3b.loadavg_gate(deadline=worker_deadline); env["competing_model_process"] = q3b.competing_model_process(q3b._deadline_runner(worker_deadline))
            gate = q3b._stage_gate(env, phase_initial_swap, installed=expected["installed_memory"], peak=1, rss=1, max_swap_used_bytes=max(samples))
            if not all(gate["checks"].get(k) for k in ("swap_endpoint_known", "swap_delta_within_128mib", "memory_free_at_least_20_percent", "ac_power", "low_power_off", "thermal_nominal", "loadavg_gate", "no_competing_model_process")):
                raise Q3cRefused("pre-child resource gate failed")
        # ``ab._child`` passes this value to ``load_engine``; using the resolved
        # snapshot path binds every fresh child to the already-verified revision
        # without a download or a second cache selection.
        result = ab.run({name: Knobs(**knobs) for name, knobs in PHASE_ARMS[phase].items()}, processes=PROCESSES, repeats=REPEATS, warmup=WARMUP, max_tokens=MAX_TOKENS, model=str(resolved.path), child_timeout_seconds=CHILD_TIMEOUT_SECONDS, before_child=before_child, on_child=on_child)
        stop.set(); thread.join(timeout=2)
        if thread.is_alive():
            with lock:
                errors.append("sampler-join: thread did not stop")
        final_error = None
        try:
            sample("worker-final")
        except BaseException as exc:
            final_error = exc
        final_event = q3b._finalize_stage_safety(initial_swap=phase_initial_swap, samples=samples,
                                                   sample_times=times, sample_offsets=offsets,
                                                   sampler_errors=errors, lock=lock, state=safety,
                                                   final_error=final_error, marker_writer=marker_writer)
        if final_error is not None or errors or final_event is not None or "event" in safety or len(samples) < 2:
            raise Q3cRefused("final sampler gate failed")
        result.update({"phase": phase, "phase_plan": phase_plan(phase), "binding": {**actual, "runtime_code_sha256": expected["runtime_code_sha256"]}, "child_rss_peak_bytes": q3b._load_q3a_helpers()._max_rss_bytes(), "swap_samples": samples, "swap_sample_times": times, "swap_sample_offsets": offsets, "sampler_errors": [], "max_swap_used_bytes": max(samples), "phase_initial_swap_bytes": samples[0], "derived_rates": derive_rates(result, PHASE_CANDIDATE[phase]), "cleanup": {"worker_group_gone": True, "cleanup_errors": [], "child_cleanup_errors": []}})
        print("@@" + json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
        return 0
    except BaseException as exc:
        completed = partial_children_snapshot()
        inherited = getattr(exc, "partial_children", [])
        if isinstance(inherited, list) and inherited:
            completed.extend(copy.deepcopy(inherited[len(completed):]))
        print("@@" + json.dumps({"failure": f"{type(exc).__name__}: {exc}", "partial_children": completed[:PROCESSES], "child_cleanup_errors": [], "partial_evidence": {"swap_samples": samples, "swap_sample_times": times, "swap_sample_offsets": offsets, "sampler_errors": errors}, "safety_event": safety.get("event")}, sort_keys=True), flush=True)
        return 2


def _write_exclusive(path: Path, payload: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
        os.chmod(path, 0o600)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        raise


def _finalize_two_phases(phases: list[Any], resources: list[Any], deadline: float, installed: int) -> tuple[str, dict[str, bool]]:
    """Run the bounded terminal snapshot and cross-phase decision fail-closed."""
    final_after = _q3b_runtime()._post_environment(deadline, phases[-1]["raw"][-1]["pid"])
    final_gate = _q3b_runtime()._stage_gate(final_after, phases[-1]["phase_initial_swap_bytes"], peak=max(a["arms"][name]["mlx_peak_bytes"] for a in phases[-1]["raw"] for name in phases[-1]["arms"]), rss=phases[-1]["child_rss_peak_bytes"], installed=installed, max_swap_used_bytes=phases[-1]["max_swap_used_bytes"])
    resources.append({"phase": "FINAL", "environment": final_after, "gate": final_gate})
    if not final_gate["passed"]:
        return "FAILED", {}
    identities = cross_phase_identity(phases[0], phases[1])
    return (decide(phases[0], phases[1])["status"] if all(identities.values()) else "INCONCLUSIVE"), identities


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--phase-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.phase_worker:
        return _phase_worker()
    plan = {"schema": "ironmule.q3c_plan.v1", "experiment": EXPERIMENT_ID, "model": MODEL_ID, "revision": EXPECTED_REVISION,
            "phases": [phase_plan(p) for p in PHASES], "study_max_seconds": STUDY_MAX_SECONDS, "phase_max_seconds": PHASE_MAX_SECONDS,
            "post_phase_seconds": POST_PHASE_SECONDS, "final_reserve_seconds": FINAL_RESERVE_SECONDS,
            "worker_max_seconds": WORKER_MAX_SECONDS, "child_timeout_seconds": CHILD_TIMEOUT_SECONDS,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES, "bootstrap_seed": BOOTSTRAP_SEED,
            "sampler_interval_seconds": SAMPLE_INTERVAL_SECONDS,
            "max_sampler_gap_seconds": MAX_SWAP_SAMPLE_GAP_SECONDS,
            "max_sampler_samples": MAX_SWAP_SAMPLES,
            "start_swap_limit_bytes": START_SWAP_LIMIT_BYTES,
            "swap_delta_limit_bytes": SWAP_DELTA_LIMIT_BYTES,
            "start_free_percent": START_FREE_PERCENT, "after_free_percent": AFTER_FREE_PERCENT,
            "peak_ceiling_fraction": PEAK_CEILING_FRACTION,
            "prompt_tokens": PROMPT_TOKENS, "max_tokens": MAX_TOKENS,
            "command_timeout_seconds": COMMAND_TIMEOUT_SECONDS,
            "capability_max_bytes": CAPABILITY_MAX_BYTES,
            "worker_output_max_bytes": MAX_WORKER_OUTPUT,
            "promotion_allowed": False}
    if not args.execute:
        print(json.dumps({"schema": "ironmule.q3c_plan.v1", "experiment": EXPERIMENT_ID, "estimated_wall_seconds": STUDY_MAX_SECONDS, "plan": plan}, indent=2, sort_keys=True))
        return 0
    if args.output is None:
        parser.error("--output is required with --execute")
    if not args.output.parent.is_dir() or os.path.lexists(args.output):
        print("q3c: output path must have an existing parent and must not already exist", file=sys.stderr)
        return 2
    started = time.monotonic(); global_deadline = started + STUDY_MAX_SECONDS
    try:
        pre = preflight(deadline=started + PHASE_MAX_SECONDS)
    except BaseException as exc:
        pre = {"passed": False, "checks": {}, "error": f"preflight failed: {type(exc).__name__}"}
    phases: list[Any] = []; resources: list[Any] = []; phase_gates: list[Any] = []; status = "FAILED"
    if pre.get("passed"):
        initial = pre["environment"]["swap_used_bytes"]; installed = pre["installed_memory_bytes"]
        for index, phase in enumerate(PHASES):
            phase_now = time.monotonic()
            if index and phase_now >= started + PHASE_MAX_SECONDS:
                phases.append({"phase": phase,
                               "failure": "phase not started: phase time boundary exhausted",
                               "failure_evidence": {
                                   "reason": "phase deadline exhausted before worker start",
                                   "phase_deadline_seconds": PHASE_MAX_SECONDS,
                                   "elapsed_seconds": phase_now - started,
                               }})
                break
            phase_deadline = started + (index + 1) * PHASE_MAX_SECONDS
            try:
                result = _start_phase(phase, pre["identity"], initial, installed, phase_deadline)
            except BaseException as exc:
                phases.append({"phase": phase, "failure": f"phase execution failed: {type(exc).__name__}", "error": str(exc)[:256]})
                break
            if "failure" in result:
                phases.append(result); break
            try:
                expected_binding = {**pre["identity"], "runtime_code_sha256": pre["runtime_code_sha256"]}
                valid, reason = validate_phase_result(result, phase, expected_binding)
            except BaseException as exc:
                phases.append({"phase": phase, "failure": f"validation failed: {type(exc).__name__}", "raw_result": result})
                break
            if not valid:
                phases.append({"phase": phase, "failure": reason, "raw_result": result}); break
            try:
                after = _q3b_runtime()._post_environment(min(phase_deadline, global_deadline), result["raw"][-1]["pid"])
                gate = _q3b_runtime()._stage_gate(after, result["phase_initial_swap_bytes"], peak=max(a["arms"][name]["mlx_peak_bytes"] for a in result["raw"] for name in result["arms"]), rss=result["child_rss_peak_bytes"], installed=installed, max_swap_used_bytes=result["max_swap_used_bytes"])
            except BaseException as exc:
                phases.append({"phase": phase, "failure": f"post-phase gate failed: {type(exc).__name__}", "raw_result": result})
                break
            resources.append({"phase": phase, "environment": after, "gate": gate})
            phases.append(result)
            if not gate["passed"]:
                phases.append({"phase": phase, "failure": "post-phase safety/resource gate failed", "resource_gate": gate})
                status = "FAILED"
                break
            if phase == "R":
                try:
                    r_gate = phase_r_performance_gate(result)
                except BaseException as exc:
                    phases.append({"phase": "R", "failure": f"R performance gate failed: {type(exc).__name__}", "raw_result": result})
                    break
                phase_gates.append({"phase": "R", "performance": r_gate})
        if len(phases) == 2 and all(isinstance(item, dict) and "failure" not in item for item in phases):
            try:
                status, identities = _finalize_two_phases(phases, resources, global_deadline, installed)
                if status == "FAILED":
                    phases.append({"failure": "final resource/cleanup gate failed"})
                elif status == "INCONCLUSIVE":
                    phases.append({"failure": "cross-phase identity mismatch", "identity": identities})
            except BaseException as exc:
                status = "FAILED"
                phases.append({"failure": f"finalization failed: {type(exc).__name__}", "error": str(exc)[:256]})
    result = {"schema": "ironmule.q3c_result.v1", "experiment": EXPERIMENT_ID, "status": status, "fallback": "BASE/current incumbent", "promotion_allowed": False, "plan": plan, "preregistration_sha256": _sha256(PREREGISTRATION) if PREREGISTRATION.exists() else None, "preflight": pre, "phases": phases, "phase_gates": phase_gates, "resource_history": resources}
    if len(phases) == 2 and all(isinstance(item, dict) and "failure" not in item for item in phases):
        result["decision"] = decide(phases[0], phases[1]); result["cross_phase_identity"] = cross_phase_identity(phases[0], phases[1])
    _write_exclusive(args.output, (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    print(json.dumps({"status": status, "output": str(args.output), "promotion_allowed": False}, sort_keys=True))
    return 0 if status == "COMPLETE_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
