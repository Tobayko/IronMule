#!/usr/bin/env python3
"""Sealed prospective Cycle-12 study for skipping unused prefill LM-head rows."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import platform
import resource
import sqlite3
import stat
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, quote, urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from friday_evidence.budget import BudgetGuard  # noqa: E402
from friday_evidence.registry import BudgetPolicy  # noqa: E402
from friday_evidence.canonical import canonical_json, canonical_sha256  # noqa: E402
from friday_h1.statistics import (  # noqa: E402
    balanced_orders,
    hierarchical_bootstrap,
    session_metrics,
)
from _bench import require_ac_power, resolve_local_model_snapshot  # noqa: E402


STUDY_ID = "head-skip-prefill-v1-20260824"
CANDIDATE_ID = "prefill-head-skip-20260824-02"
MODEL_ID = "mlx-community/gemma-3-4b-it-4bit"
MODEL_REVISION = "93724907d4ed1745d2fe50baadf3b0b01a65abf2"
DEFAULT_DATABASE = PROJECT_ROOT / ".friday-data" / "head-skip-v1.sqlite3"
PREREGISTRATION_PATH = "experiments/head_skip_formal/PREREGISTRATION.md"
SCRIPT_PATH = "experiments/head_skip_formal/study.py"

SCHEMA_VERSION = 1
SQLITE_APPLICATION_ID = 0x48535631  # HSV1
CALIBRATION = "calibration"
CONFIRMATION = "confirmation"
STAGES = frozenset({CALIBRATION, CONFIRMATION})
SESSION_ORDER = ("C0", "V0", "C1", "V1", "C2", "V2")
SESSION_COHORT = {
    "C0": ("characterization", 0),
    "V0": ("validation", 0),
    "C1": ("characterization", 1),
    "V1": ("validation", 1),
    "C2": ("characterization", 2),
    "V2": ("validation", 2),
}
SESSION_SEEDS = {
    "C0": 15510830734782369641,
    "V0": 13859906320662629798,
    "C1": 3290811032693642639,
    "V1": 14366515575250128902,
    "C2": 13587802099656419680,
    "V2": 12362147029480673024,
}
BOOTSTRAP_SEEDS = {
    "calibration": 3434287716142173047,
    "characterization": 16895945304681056598,
    "validation": 16493265756820087568,
    "all": 7407874620929745004,
}

PROMPT_UNIT = (
    "You are a careful engineering assistant working in a Python repository. "
    "Follow the existing style and explain your reasoning briefly. "
)
PROMPT_CONTENT = PROMPT_UNIT * 40 + "\n\nWhy is false sharing slow?"
PROMPT_CONTENT_SHA256 = "73675a7043bd40e61586757d8252cf1fb69bfb53b8747ff47f1c08d5fb8f69e5"
PROMPT_TOKENS = 897
CHUNK = 256
CORRECTNESS_TOKENS = 32
WARMUP_PAIRS = 2
MEASUREMENT_BLOCKS = 4
INTER_SESSION_COOLDOWN_SECONDS = 20

BOOTSTRAP_DRAWS = 10_000
CONFIDENCE = 0.95
MDE_FLOOR = 0.05
MDE_CAP = 0.15

GPU_WORK_LIMIT_SECONDS = 120.0
CONTINUOUS_GPU_LIMIT_SECONDS = 6.0
REQUIRED_BREAK_SECONDS = 4.0
DUTY_WINDOW_SECONDS = 60.0
DUTY_CYCLE_LIMIT = 0.15
PACING_TARGET = 0.14
WALL_LIMIT_SECONDS = 1_200.0
CANDIDATE_COOLDOWN_SECONDS = 60.0
POLICY = BudgetPolicy(
    gpu_work_limit_s=GPU_WORK_LIMIT_SECONDS,
    continuous_gpu_limit_s=CONTINUOUS_GPU_LIMIT_SECONDS,
    required_break_s=REQUIRED_BREAK_SECONDS,
    duty_window_s=DUTY_WINDOW_SECONDS,
    duty_cycle_limit=DUTY_CYCLE_LIMIT,
    wall_limit_s=WALL_LIMIT_SECONDS,
    candidate_cooldown_s=CANDIDATE_COOLDOWN_SECONDS,
)

MAX_FILE_BYTES = 4 * 1024 * 1024
MAX_PAYLOAD_BYTES = 8 * 1024 * 1024
MAX_UI_ROWS = 64
ZERO_SHA256 = "0" * 64
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
SELF_HASH_FIELDS = {
    "preregistration": "preregistration_sha256",
    "calibration_session": "session_sha256",
    "confirmation_session": "session_sha256",
    "calibration_summary": "calibration_summary_sha256",
    "confirmation_seal": "confirmation_seal_sha256",
    "study_decision": "decision_sha256",
    "session_failure": "failure_sha256",
}


class StudyError(RuntimeError):
    """The prospective study cannot continue without weakening its contract."""


class ProtocolError(ValueError):
    """A record does not replay against the preregistered protocol."""


class StorageError(RuntimeError):
    """The append-only evidence store failed closed."""


class CorrectnessError(StudyError):
    """Greedy token identity failed before timing."""


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False), flush=True)


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    body = dict(value)
    digest = body.pop(field, None)
    if not isinstance(digest, str) or len(digest) != 64:
        raise ProtocolError(f"invalid {field}")
    if digest != canonical_sha256(body):
        raise ProtocolError(f"{field} does not replay")
    return digest


def _with_hash(value: dict[str, Any], field: str) -> dict[str, Any]:
    value[field] = canonical_sha256(value)
    return value


def study_specification() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "candidate_id": CANDIDATE_ID,
        "scope": {
            "device_count": 1,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "sampling": "greedy_fixed_horizon",
            "prompt_logprobs": False,
            "claim_scope": "one-device-one-model-one-prompt-one-prefill-plan",
        },
        "workload": {
            "prompt_content_sha256": PROMPT_CONTENT_SHA256,
            "prompt_tokens": PROMPT_TOKENS,
            "prefill_chunk": CHUNK,
            "batch": 1,
            "correctness_tokens": CORRECTNESS_TOKENS,
            "arm_a": "full_lm_head_all_prefill_positions",
            "arm_b": "lm_head_last_position_of_final_prefill_block_only",
            "primary_endpoint": "paired_prefill_duration_ratio_B_over_A",
        },
        "schedule": {
            "session_order": list(SESSION_ORDER),
            "session_seeds": dict(SESSION_SEEDS),
            "warmup_pairs": WARMUP_PAIRS,
            "measurement_blocks": MEASUREMENT_BLOCKS,
            "calibration_design": "A/A_distinct_full_head_callables",
            "confirmation_design": "A/B_full_head_vs_last_only",
            "separate_process_per_session": True,
            "inter_session_cooldown_seconds": INTER_SESSION_COOLDOWN_SECONDS,
            "retry_failed_session": False,
            "discard_outliers": False,
        },
        "statistics": {
            "session_estimator": "exp(median(log(B_ns/A_ns)))",
            "study_estimator": "median(session_log_medians)",
            "interval": "deterministic_hierarchical_percentile_bootstrap",
            "draws": BOOTSTRAP_DRAWS,
            "confidence": CONFIDENCE,
            "bootstrap_seeds": dict(BOOTSTRAP_SEEDS),
            "mde_formula": "max(0.05,2*sd(session_ratio)*sqrt(2/3))",
            "mde_floor": MDE_FLOOR,
            "mde_cap": MDE_CAP,
        },
        "correctness": {
            "identical_token_ids": True,
            "identical_finish_reason": True,
            "prompt_truncated": False,
            "silent_fallback": False,
            "failure": "correctness_failed_terminal",
            "timing_only_after_gate": True,
        },
        "budgets": {
            "gpu_work_limit_seconds": GPU_WORK_LIMIT_SECONDS,
            "continuous_gpu_limit_seconds": CONTINUOUS_GPU_LIMIT_SECONDS,
            "required_break_seconds": REQUIRED_BREAK_SECONDS,
            "duty_window_seconds": DUTY_WINDOW_SECONDS,
            "duty_cycle_limit": DUTY_CYCLE_LIMIT,
            "pacing_target": PACING_TARGET,
            "minimum_break_seconds_per_recorded_operation": 16.0,
            "wall_limit_seconds": WALL_LIMIT_SECONDS,
            "candidate_cooldown_seconds": CANDIDATE_COOLDOWN_SECONDS,
            "power_source": "ac_power",
        },
        "decision": {
            "gain": "all C,V,all upper bounds strictly below 1-MDE",
            "regression": "all C,V,all lower bounds strictly above 1+MDE",
            "equivalence": "all C,V,all intervals inside [1-MDE,1+MDE]",
            "otherwise": "inconclusive",
            "formal_claim_records": 1,
            "automatic_product_activation": False,
        },
    }


def orders_for(stage: str, session_id: str, *, warmup: bool) -> list[str]:
    if stage not in STAGES or session_id not in SESSION_SEEDS:
        raise ProtocolError("unknown stage or session")
    count = WARMUP_PAIRS if warmup else MEASUREMENT_BLOCKS
    return balanced_orders(
        count,
        seed=SESSION_SEEDS[session_id],
        domain=f"head-skip-v1:{stage}:{session_id}:{'warmup' if warmup else 'measurement'}",
    )


def _git(*args: str) -> bytes:
    try:
        result = subprocess.run(
            ["/usr/bin/git", "-C", str(PROJECT_ROOT), *args],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise StudyError("Git provenance is unavailable") from exc
    if result.returncode != 0:
        raise StudyError("Git provenance failed")
    return result.stdout


def _regular_hash(relative: str) -> str:
    path = PROJECT_ROOT / relative
    try:
        info = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_size > MAX_FILE_BYTES:
            raise StudyError(f"unsafe provenance input: {relative}")
        data = path.read_bytes()
    except OSError as exc:
        raise StudyError(f"unavailable provenance input: {relative}") from exc
    if len(data) != info.st_size:
        raise StudyError(f"provenance input changed while reading: {relative}")
    return hashlib.sha256(data).hexdigest()


def _sysctl(name: str) -> str | None:
    try:
        result = subprocess.run(
            ["/usr/sbin/sysctl", "-n", name],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=3.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.decode("utf-8", errors="strict").strip() or None


def collect_provenance(*, require_clean: bool = True) -> dict[str, Any]:
    revision = _git("rev-parse", "HEAD").decode("ascii", errors="strict").strip()
    status = _git(
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        ".",
        ":(exclude)ProjectAtlas",
    ).decode("utf-8", errors="strict")
    dirty = bool(status.strip())
    if require_clean and dirty:
        raise StudyError("project worktree must be clean before sealing or measuring")
    diff = _git("diff", "--binary", "HEAD") + _git("diff", "--cached", "--binary", "HEAD")
    code_paths = (
        SCRIPT_PATH,
        "friday_evidence/budget.py",
        "friday_evidence/registry.py",
        "friday_evidence/canonical.py",
        "friday_h1/statistics.py",
        "tools/_bench.py",
    )
    spec_paths = (PREREGISTRATION_PATH, "requirements-apple-silicon.txt")
    code_files = {path: _regular_hash(path) for path in sorted(code_paths)}
    spec_files = {path: _regular_hash(path) for path in sorted(spec_paths)}
    packages: dict[str, str | None] = {}
    for name in ("mlx", "mlx-metal", "mlx-lm", "numpy", "psutil"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    snapshot = resolve_local_model_snapshot(MODEL_ID)
    if snapshot.revision != MODEL_REVISION:
        raise StudyError("local model revision differs from the preregistration")
    environment = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": str(Path(sys.executable).resolve()),
        "packages": packages,
    }
    hardware = {
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0] or None,
        "model": _sysctl("hw.model"),
        "memory_bytes": _sysctl("hw.memsize"),
        "cpu_brand": _sysctl("machdep.cpu.brand_string"),
    }
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "git_revision": revision,
        "git_dirty": dirty,
        "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "code_files": code_files,
        "code_sha256": canonical_sha256(code_files),
        "spec_files": spec_files,
        "spec_sha256": canonical_sha256(spec_files),
        "environment": environment,
        "environment_sha256": canonical_sha256(environment),
        "hardware": hardware,
        "hardware_sha256": canonical_sha256(hardware),
        "model": snapshot.report_identity(),
    }
    value["model_sha256"] = canonical_sha256(value["model"])
    value["provenance_sha256"] = canonical_sha256(value)
    return value


def _validated_provenance(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ProtocolError("provenance must be an object")
    value = dict(raw)
    digest = value.pop("provenance_sha256", None)
    if digest != canonical_sha256(value):
        raise ProtocolError("provenance digest does not replay")
    required = {
        "schema_version",
        "study_id",
        "git_revision",
        "git_dirty",
        "git_diff_sha256",
        "code_files",
        "code_sha256",
        "spec_files",
        "spec_sha256",
        "environment",
        "environment_sha256",
        "hardware",
        "hardware_sha256",
        "model",
        "model_sha256",
    }
    if set(value) != required:
        raise ProtocolError("provenance fields differ from the sealed contract")
    if (
        value["schema_version"] != SCHEMA_VERSION
        or value["study_id"] != STUDY_ID
        or value["git_dirty"] is not False
        or value["git_diff_sha256"] != EMPTY_SHA256
    ):
        raise ProtocolError("provenance does not bind a clean study checkout")
    projections = {
        "code_sha256": "code_files",
        "spec_sha256": "spec_files",
        "environment_sha256": "environment",
        "hardware_sha256": "hardware",
        "model_sha256": "model",
    }
    if any(value[digest_key] != canonical_sha256(value[source_key]) for digest_key, source_key in projections.items()):
        raise ProtocolError("provenance projection digest differs")
    model = value["model"]
    if not isinstance(model, Mapping) or model.get("model_id") != MODEL_ID or model.get("model_revision") != MODEL_REVISION:
        raise ProtocolError("provenance model binding differs")
    value["provenance_sha256"] = digest
    return value


def build_preregistration(provenance: Mapping[str, Any]) -> dict[str, Any]:
    checked_provenance = _validated_provenance(provenance)
    spec = study_specification()
    value = {
        "kind": "preregistration",
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "candidate_id": CANDIDATE_ID,
        "study_specification": spec,
        "study_spec_sha256": canonical_sha256(spec),
        "provenance": checked_provenance,
        "provenance_sha256": checked_provenance["provenance_sha256"],
        "status": "sealed_before_measurement",
        "formal_claim": False,
    }
    return _with_hash(value, "preregistration_sha256")


def _validated_blocks(
    raw: Sequence[Mapping[str, Any]], stage: str, session_id: str, *, warmup: bool
) -> list[dict[str, Any]]:
    expected = orders_for(stage, session_id, warmup=warmup)
    if len(raw) != len(expected):
        raise ProtocolError("timing block count differs from the preregistration")
    result: list[dict[str, Any]] = []
    for index, (block, order) in enumerate(zip(raw, expected, strict=True)):
        if set(block) != {"block_index", "order", "a_ns", "b_ns"}:
            raise ProtocolError("timing block keys are not exact")
        if block["block_index"] != index or block["order"] != order:
            raise ProtocolError("timing blocks are reordered")
        if any(type(block[key]) is not int or block[key] <= 0 for key in ("a_ns", "b_ns")):
            raise ProtocolError("timing duration is invalid")
        result.append(dict(block))
    return result


def _validated_correctness(raw: Mapping[str, Any]) -> dict[str, Any]:
    keys = {
        "status",
        "token_ids",
        "token_sha256",
        "token_count",
        "finish_reason",
        "prompt_truncated",
        "silent_fallback",
        "candidate_path_exercised",
    }
    if set(raw) != keys or raw.get("status") != "token_identical":
        raise ProtocolError("correctness payload is invalid")
    tokens = raw.get("token_ids")
    if (
        not isinstance(tokens, list)
        or len(tokens) != CORRECTNESS_TOKENS
        or any(type(token) is not int or token < 0 for token in tokens)
    ):
        raise ProtocolError("correctness token trace is invalid")
    if raw.get("token_count") != len(tokens):
        raise ProtocolError("correctness token count differs")
    digest = hashlib.sha256(canonical_json(tokens).encode("utf-8")).hexdigest()
    if raw.get("token_sha256") != digest:
        raise ProtocolError("correctness token digest differs")
    if (
        raw.get("finish_reason") != "fixed_horizon"
        or raw.get("prompt_truncated") is not False
        or raw.get("silent_fallback") is not False
        or raw.get("candidate_path_exercised") is not True
    ):
        raise ProtocolError("correctness gates did not all pass")
    return dict(raw)


def _validated_budget(raw: Mapping[str, Any]) -> dict[str, Any]:
    expected_limits = {
        "gpu_work_limit_seconds": GPU_WORK_LIMIT_SECONDS,
        "continuous_gpu_limit_seconds": CONTINUOUS_GPU_LIMIT_SECONDS,
        "duty_cycle_limit": DUTY_CYCLE_LIMIT,
        "wall_limit_seconds": WALL_LIMIT_SECONDS,
        "candidate_cooldown_seconds": CANDIDATE_COOLDOWN_SECONDS,
        "required_break_limit_seconds": REQUIRED_BREAK_SECONDS,
    }
    if raw.get("passed") is not True or any(raw.get(k) != v for k, v in expected_limits.items()):
        raise ProtocolError("budget limits differ from the preregistration")
    for key in (
        "gpu_work_seconds",
        "max_continuous_gpu_seconds",
        "cooldown_seconds",
        "required_break_seconds",
        "wall_seconds",
    ):
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ProtocolError("budget measurement is invalid")
    if (
        float(raw["gpu_work_seconds"]) > GPU_WORK_LIMIT_SECONDS
        or float(raw["max_continuous_gpu_seconds"]) > CONTINUOUS_GPU_LIMIT_SECONDS
        or float(raw["wall_seconds"]) > WALL_LIMIT_SECONDS
    ):
        raise ProtocolError("hardware budget was exceeded")
    return dict(raw)


def build_session(
    *,
    stage: str,
    session_id: str,
    preregistration_sha256: str,
    confirmation_seal_sha256: str | None,
    provenance_sha256: str,
    warmups: Sequence[Mapping[str, Any]],
    measurements: Sequence[Mapping[str, Any]],
    correctness: Mapping[str, Any],
    budget: Mapping[str, Any],
    resources: Mapping[str, Any],
) -> dict[str, Any]:
    if stage not in STAGES or session_id not in SESSION_COHORT:
        raise ProtocolError("unknown session")
    checked_warmups = _validated_blocks(warmups, stage, session_id, warmup=True)
    checked_measurements = _validated_blocks(measurements, stage, session_id, warmup=False)
    checked_correctness = _validated_correctness(correctness)
    checked_budget = _validated_budget(budget)
    if stage == CALIBRATION and confirmation_seal_sha256 is not None:
        raise ProtocolError("calibration session cannot bind a confirmation seal")
    if stage == CONFIRMATION and not isinstance(confirmation_seal_sha256, str):
        raise ProtocolError("confirmation session lacks its seal")
    resource_keys = {
        "cpu_process_ns",
        "rss_peak_bytes",
        "mlx_active_memory_bytes",
        "mlx_peak_memory_bytes",
        "mlx_cache_memory_bytes",
    }
    if set(resources) != resource_keys:
        raise ProtocolError("resource payload keys differ")
    if any(type(resources[key]) is not int or resources[key] < 0 for key in ("cpu_process_ns", "rss_peak_bytes")):
        raise ProtocolError("process resource measurement is invalid")
    for key in resource_keys - {"cpu_process_ns", "rss_peak_bytes"}:
        if resources[key] is not None and (type(resources[key]) is not int or resources[key] < 0):
            raise ProtocolError("MLX resource measurement is invalid")
    value = {
        "kind": f"{stage}_session",
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "candidate_id": CANDIDATE_ID,
        "stage": stage,
        "session_id": session_id,
        "cohort": SESSION_COHORT[session_id][0],
        "cohort_index": SESSION_COHORT[session_id][1],
        "preregistration_sha256": preregistration_sha256,
        "confirmation_seal_sha256": confirmation_seal_sha256,
        "provenance_sha256": provenance_sha256,
        "power_source": "ac_power",
        "warmups": checked_warmups,
        "measurements": checked_measurements,
        "metrics": session_metrics(checked_measurements),
        "correctness": checked_correctness,
        "budget": checked_budget,
        "resources": dict(resources),
        "status": "session_complete",
        "formal_claim": False,
    }
    return _with_hash(value, "session_sha256")


def _ordered_sessions(values: Sequence[Mapping[str, Any]], stage: str) -> list[dict[str, Any]]:
    if len(values) != len(SESSION_ORDER):
        raise ProtocolError("study stage requires exactly six sessions")
    result = [dict(value) for value in values]
    if [value.get("session_id") for value in result] != list(SESSION_ORDER):
        raise ProtocolError("sessions are missing, duplicated, or reordered")
    if any(value.get("stage") != stage for value in result):
        raise ProtocolError("study stage is mixed")
    return result


def build_calibration_summary(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    sessions = _ordered_sessions(values, CALIBRATION)
    ratios = [float(value["metrics"]["ratio"]) for value in sessions]
    ratio_sd = statistics.stdev(ratios)
    raw_mde = 2.0 * ratio_sd * math.sqrt(2.0 / 3.0)
    mde = max(MDE_FLOOR, raw_mde)
    aggregate = hierarchical_bootstrap(
        [value["measurements"] for value in sessions],
        seed=BOOTSTRAP_SEEDS["calibration"],
        draws=BOOTSTRAP_DRAWS,
        confidence=CONFIDENCE,
    )
    gates = {
        "aggregate_interval_contains_one": bool(aggregate["ci_low"] <= 1.0 <= aggregate["ci_high"]),
        "aggregate_bias_within_floor": abs(float(aggregate["ratio"]) - 1.0) <= MDE_FLOOR,
        "mde_within_cap": mde <= MDE_CAP,
        "all_sessions_token_identical": all(
            value["correctness"]["status"] == "token_identical" for value in sessions
        ),
    }
    value = {
        "kind": "calibration_summary",
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "candidate_id": CANDIDATE_ID,
        "stage": CALIBRATION,
        "preregistration_sha256": sessions[0]["preregistration_sha256"],
        "provenance_sha256": sessions[0]["provenance_sha256"],
        "session_sha256": [session["session_sha256"] for session in sessions],
        "session_ratios": ratios,
        "session_ratio_sd": ratio_sd,
        "raw_mde": raw_mde,
        "mde": mde,
        "mde_floor": MDE_FLOOR,
        "mde_cap": MDE_CAP,
        "aggregate": aggregate,
        "gates": gates,
        "status": "calibration_passed" if all(gates.values()) else "calibration_failed",
        "formal_claim": False,
    }
    return _with_hash(value, "calibration_summary_sha256")


def build_confirmation_seal(
    summary: Mapping[str, Any], sessions: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected = build_calibration_summary(sessions)
    if canonical_json(summary) != canonical_json(expected) or expected["status"] != "calibration_passed":
        raise ProtocolError("calibration cannot open confirmation")
    value = {
        "kind": "confirmation_seal",
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "candidate_id": CANDIDATE_ID,
        "stage": CONFIRMATION,
        "preregistration_sha256": expected["preregistration_sha256"],
        "calibration_summary_sha256": expected["calibration_summary_sha256"],
        "provenance_sha256": expected["provenance_sha256"],
        "mde": expected["mde"],
        "session_order": list(SESSION_ORDER),
        "status": "confirmation_sealed",
        "formal_claim": False,
    }
    return _with_hash(value, "confirmation_seal_sha256")


def _inside(interval: Mapping[str, Any], low: float, high: float) -> bool:
    return float(interval["ci_low"]) >= low and float(interval["ci_high"]) <= high


def build_study_decision(
    values: Sequence[Mapping[str, Any]], seal: Mapping[str, Any]
) -> dict[str, Any]:
    sessions = _ordered_sessions(values, CONFIRMATION)
    seal_hash = seal.get("confirmation_seal_sha256")
    if any(session.get("confirmation_seal_sha256") != seal_hash for session in sessions):
        raise ProtocolError("confirmation sessions do not bind one seal")
    mde = float(seal["mde"])
    characterization = [value for value in sessions if value["cohort"] == "characterization"]
    validation = [value for value in sessions if value["cohort"] == "validation"]
    intervals = {
        "characterization": hierarchical_bootstrap(
            [value["measurements"] for value in characterization],
            seed=BOOTSTRAP_SEEDS["characterization"],
            draws=BOOTSTRAP_DRAWS,
            confidence=CONFIDENCE,
        ),
        "validation": hierarchical_bootstrap(
            [value["measurements"] for value in validation],
            seed=BOOTSTRAP_SEEDS["validation"],
            draws=BOOTSTRAP_DRAWS,
            confidence=CONFIDENCE,
        ),
        "all": hierarchical_bootstrap(
            [value["measurements"] for value in sessions],
            seed=BOOTSTRAP_SEEDS["all"],
            draws=BOOTSTRAP_DRAWS,
            confidence=CONFIDENCE,
        ),
    }
    low, high = 1.0 - mde, 1.0 + mde
    gain = all(float(interval["ci_high"]) < low for interval in intervals.values())
    regression = all(float(interval["ci_low"]) > high for interval in intervals.values())
    equivalent = all(_inside(interval, low, high) for interval in intervals.values())
    if gain:
        status = "head_skip_gain_confirmed"
        claim = "prefill_head_skip_is_faster_beyond_mde"
        action = "permit_bounded_architecture_review"
    elif regression:
        status = "head_skip_regression_confirmed"
        claim = "prefill_head_skip_is_slower_beyond_mde"
        action = "reject_candidate"
    elif equivalent:
        status = "head_skip_equivalent_within_mde"
        claim = "prefill_paths_are_equivalent_within_mde"
        action = "reject_candidate"
    else:
        status = "head_skip_inconclusive"
        claim = "no_confirmatory_direction_cleared_all_split_gates"
        action = "stop_without_promotion"
    value = {
        "kind": "study_decision",
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "candidate_id": CANDIDATE_ID,
        "stage": CONFIRMATION,
        "preregistration_sha256": sessions[0]["preregistration_sha256"],
        "confirmation_seal_sha256": seal_hash,
        "provenance_sha256": sessions[0]["provenance_sha256"],
        "session_sha256": [session["session_sha256"] for session in sessions],
        "mde": mde,
        "equivalence_bounds": {"low": low, "high": high},
        "intervals": intervals,
        "effect_percent": 100.0 * (float(intervals["all"]["ratio"]) - 1.0),
        "gates": {
            "gain_all_splits": gain,
            "regression_all_splits": regression,
            "equivalence_all_splits": equivalent,
            "all_sessions_token_identical": all(
                session["correctness"]["status"] == "token_identical" for session in sessions
            ),
        },
        "status": status,
        "claim": claim,
        "action": action,
        "claim_scope": "one-device-one-model-one-prompt-one-prefill-plan",
        "limitations": [
            "single_device",
            "single_model_snapshot",
            "single_prompt_length",
            "greedy_without_prompt_logprobs_only",
            "no_automatic_product_activation",
        ],
        "formal_claim": True,
    }
    return _with_hash(value, "decision_sha256")


def build_failure(
    *,
    stage: str,
    session_id: str,
    preregistration_sha256: str,
    confirmation_seal_sha256: str | None,
    provenance_sha256: str,
    failure_type: str,
) -> dict[str, Any]:
    if stage not in STAGES or session_id not in SESSION_ORDER:
        raise ProtocolError("failure identifies an unknown session")
    if stage == CALIBRATION and confirmation_seal_sha256 is not None:
        raise ProtocolError("calibration failure cannot bind a confirmation seal")
    if stage == CONFIRMATION and not isinstance(confirmation_seal_sha256, str):
        raise ProtocolError("confirmation failure lacks its seal")
    if not isinstance(failure_type, str) or not failure_type:
        raise ProtocolError("failure type is invalid")
    value = {
        "kind": "session_failure",
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "candidate_id": CANDIDATE_ID,
        "stage": stage,
        "session_id": session_id,
        "preregistration_sha256": preregistration_sha256,
        "confirmation_seal_sha256": confirmation_seal_sha256,
        "provenance_sha256": provenance_sha256,
        "failure_type": failure_type[:96],
        "status": "correctness_failed_terminal" if failure_type == "CorrectnessError" else "measurement_failed_terminal",
        "formal_claim": False,
    }
    return _with_hash(value, "failure_sha256")


def _rebuild_session(raw: Mapping[str, Any]) -> dict[str, Any]:
    return build_session(
        stage=raw.get("stage"),
        session_id=raw.get("session_id"),
        preregistration_sha256=raw.get("preregistration_sha256"),
        confirmation_seal_sha256=raw.get("confirmation_seal_sha256"),
        provenance_sha256=raw.get("provenance_sha256"),
        warmups=raw.get("warmups"),
        measurements=raw.get("measurements"),
        correctness=raw.get("correctness"),
        budget=raw.get("budget"),
        resources=raw.get("resources"),
    )


def validate_history(values: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not values:
        return []
    prereg = dict(values[0])
    if prereg.get("kind") != "preregistration":
        raise ProtocolError("history does not begin with preregistration")
    _self_hash(prereg, "preregistration_sha256")
    rebuilt_prereg = build_preregistration(_validated_provenance(prereg.get("provenance")))
    if canonical_json(prereg) != canonical_json(rebuilt_prereg):
        raise ProtocolError("preregistration does not replay")
    checked = [prereg]
    calibration_sessions: list[dict[str, Any]] = []
    confirmation_sessions: list[dict[str, Any]] = []
    summary: dict[str, Any] | None = None
    seal: dict[str, Any] | None = None
    terminal = False
    formal_count = 0
    for raw_value in values[1:]:
        if terminal:
            raise ProtocolError("terminal history has later records")
        raw = dict(raw_value)
        kind = raw.get("kind")
        field = SELF_HASH_FIELDS.get(kind)
        if field is None:
            raise ProtocolError("unknown record kind")
        _self_hash(raw, field)
        formal_count += raw.get("formal_claim") is True
        if kind in {"calibration_session", "confirmation_session"}:
            rebuilt = _rebuild_session(raw)
            if canonical_json(raw) != canonical_json(rebuilt):
                raise ProtocolError("session does not replay")
            target = calibration_sessions if raw["stage"] == CALIBRATION else confirmation_sessions
            if len(target) >= len(SESSION_ORDER) or raw["session_id"] != SESSION_ORDER[len(target)]:
                raise ProtocolError("session order differs")
            if raw["preregistration_sha256"] != prereg["preregistration_sha256"]:
                raise ProtocolError("session binds another preregistration")
            if raw["provenance_sha256"] != prereg["provenance_sha256"]:
                raise ProtocolError("session binds another provenance")
            if raw["stage"] == CALIBRATION and (summary is not None or seal is not None):
                raise ProtocolError("calibration is already closed")
            if raw["stage"] == CONFIRMATION and seal is None:
                raise ProtocolError("confirmation started before its seal")
            target.append(raw)
        elif kind == "calibration_summary":
            expected = build_calibration_summary(calibration_sessions)
            if summary is not None or canonical_json(raw) != canonical_json(expected):
                raise ProtocolError("calibration summary does not replay")
            summary = raw
        elif kind == "confirmation_seal":
            if summary is None or seal is not None:
                raise ProtocolError("confirmation seal is misplaced")
            expected = build_confirmation_seal(summary, calibration_sessions)
            if canonical_json(raw) != canonical_json(expected):
                raise ProtocolError("confirmation seal does not replay")
            seal = raw
        elif kind == "study_decision":
            if seal is None:
                raise ProtocolError("decision lacks confirmation seal")
            expected = build_study_decision(confirmation_sessions, seal)
            if canonical_json(raw) != canonical_json(expected):
                raise ProtocolError("study decision does not replay")
            terminal = True
        elif kind == "session_failure":
            expected_stage = raw.get("stage")
            target = calibration_sessions if expected_stage == CALIBRATION else confirmation_sessions
            if raw.get("session_id") != SESSION_ORDER[len(target)]:
                raise ProtocolError("failure is not the next session")
            expected_failure = build_failure(
                stage=raw.get("stage"),
                session_id=raw.get("session_id"),
                preregistration_sha256=raw.get("preregistration_sha256"),
                confirmation_seal_sha256=raw.get("confirmation_seal_sha256"),
                provenance_sha256=raw.get("provenance_sha256"),
                failure_type=raw.get("failure_type"),
            )
            if canonical_json(raw) != canonical_json(expected_failure):
                raise ProtocolError("session failure does not replay")
            if (
                raw["preregistration_sha256"] != prereg["preregistration_sha256"]
                or raw["provenance_sha256"] != prereg["provenance_sha256"]
                or (expected_stage == CONFIRMATION and (seal is None or raw["confirmation_seal_sha256"] != seal["confirmation_seal_sha256"]))
            ):
                raise ProtocolError("session failure binds another study state")
            terminal = True
        checked.append(raw)
    if formal_count > 1:
        raise ProtocolError("history contains multiple formal claim records")
    if terminal and checked[-1]["kind"] == "study_decision" and formal_count != 1:
        raise ProtocolError("terminal decision must be the sole formal claim")
    return checked


DDL = """
CREATE TABLE metadata(
  singleton INTEGER PRIMARY KEY CHECK(singleton=1),
  schema_version INTEGER NOT NULL,
  created_at_unix_ns INTEGER NOT NULL
);
CREATE TABLE records(
  seq INTEGER PRIMARY KEY,
  kind TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  payload_sha256 TEXT NOT NULL,
  previous_record_sha256 TEXT,
  recorded_at_unix_ns INTEGER NOT NULL,
  record_sha256 TEXT NOT NULL UNIQUE
);
CREATE TRIGGER records_no_update BEFORE UPDATE ON records BEGIN
  SELECT RAISE(ABORT, 'append-only');
END;
CREATE TRIGGER records_no_delete BEFORE DELETE ON records BEGIN
  SELECT RAISE(ABORT, 'append-only');
END;
CREATE TRIGGER metadata_no_update BEFORE UPDATE ON metadata BEGIN
  SELECT RAISE(ABORT, 'append-only');
END;
CREATE TRIGGER metadata_no_delete BEFORE DELETE ON metadata BEGIN
  SELECT RAISE(ABORT, 'append-only');
END;
"""


def _database_path(path: str | Path, *, create_parent: bool) -> Path:
    candidate = Path(os.path.abspath(Path(path).expanduser()))
    if create_parent:
        candidate.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        parent = candidate.parent.lstat()
    except OSError as exc:
        raise StorageError("database parent is unavailable") from exc
    if parent.st_uid != os.geteuid() or stat.S_IMODE(parent.st_mode) & 0o077 or not stat.S_ISDIR(parent.st_mode):
        raise StorageError("database parent permissions are unsafe")
    resolved = candidate.parent.resolve(strict=True) / candidate.name
    if resolved.exists():
        info = resolved.lstat()
        if resolved.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o600:
            raise StorageError("database file permissions are unsafe")
    return resolved


def _configure(connection: sqlite3.Connection, *, read_only: bool) -> None:
    connection.row_factory = sqlite3.Row
    required = (
        "SQLITE_DBCONFIG_DEFENSIVE",
        "SQLITE_DBCONFIG_TRUSTED_SCHEMA",
        "SQLITE_DBCONFIG_DQS_DDL",
        "SQLITE_DBCONFIG_DQS_DML",
        "SQLITE_DBCONFIG_ENABLE_LOAD_EXTENSION",
    )
    if not hasattr(connection, "setconfig") or any(not hasattr(sqlite3, name) for name in required):
        raise StorageError("required SQLite defensive controls are unavailable")
    connection.setconfig(sqlite3.SQLITE_DBCONFIG_DEFENSIVE, True)
    connection.setconfig(sqlite3.SQLITE_DBCONFIG_TRUSTED_SCHEMA, False)
    connection.setconfig(sqlite3.SQLITE_DBCONFIG_DQS_DDL, False)
    connection.setconfig(sqlite3.SQLITE_DBCONFIG_DQS_DML, False)
    connection.setconfig(sqlite3.SQLITE_DBCONFIG_ENABLE_LOAD_EXTENSION, False)
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA trusted_schema=OFF")
    connection.execute("PRAGMA busy_timeout=5000")
    if read_only:
        connection.execute("PRAGMA query_only=ON")
        if connection.execute("PRAGMA query_only").fetchone()[0] != 1:
            raise StorageError("SQLite query-only mode did not take effect")


class Storage:
    def __init__(self, path: Path, connection: sqlite3.Connection, *, read_only: bool) -> None:
        self.path = path
        self.connection = connection
        self.read_only = read_only

    @classmethod
    def open(cls, path: str | Path, *, initialize: bool = False, read_only: bool = False) -> "Storage":
        if initialize and read_only:
            raise StorageError("read-only initialization is forbidden")
        checked = _database_path(path, create_parent=initialize)
        existed = checked.exists()
        if initialize and not existed:
            flags = os.O_CREAT | os.O_EXCL | os.O_RDWR
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(checked, flags, 0o600)
            os.close(descriptor)
        if not checked.exists():
            raise StorageError("evidence database does not exist")
        try:
            if read_only:
                uri = f"file:{quote(str(checked), safe='/')}?mode=ro&immutable=0"
                connection = sqlite3.connect(uri, uri=True, timeout=5.0)
            else:
                connection = sqlite3.connect(str(checked), timeout=5.0)
            _configure(connection, read_only=read_only)
            storage = cls(checked, connection, read_only=read_only)
            if initialize and not existed:
                storage._initialize()
            storage.verify_schema()
            return storage
        except Exception:
            if "connection" in locals():
                connection.close()
            if initialize and not existed:
                checked.unlink(missing_ok=True)
            raise

    def __enter__(self) -> "Storage":
        return self

    def __exit__(self, *_args: object) -> None:
        self.connection.close()

    def _initialize(self) -> None:
        try:
            self.connection.executescript("BEGIN EXCLUSIVE;\n" + DDL)
            self.connection.execute(f"PRAGMA application_id={SQLITE_APPLICATION_ID}")
            self.connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            self.connection.execute(
                "INSERT INTO metadata(singleton,schema_version,created_at_unix_ns) VALUES(1,?,?)",
                (SCHEMA_VERSION, time.time_ns()),
            )
            self.connection.commit()
        except sqlite3.Error as exc:
            self.connection.rollback()
            raise StorageError("cannot initialize evidence schema") from exc

    def verify_schema(self) -> None:
        try:
            app_id = self.connection.execute("PRAGMA application_id").fetchone()[0]
            version_value = self.connection.execute("PRAGMA user_version").fetchone()[0]
            integrity = self.connection.execute("PRAGMA integrity_check(1)").fetchone()[0]
            tables = {
                row[0]
                for row in self.connection.execute(
                    "SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
            }
            triggers = {
                row[0] for row in self.connection.execute("SELECT name FROM sqlite_schema WHERE type='trigger'")
            }
            metadata = self.connection.execute("SELECT schema_version FROM metadata WHERE singleton=1").fetchall()
            metadata_columns = tuple(
                row[1] for row in self.connection.execute("PRAGMA table_info(metadata)")
            )
            record_columns = tuple(
                row[1] for row in self.connection.execute("PRAGMA table_info(records)")
            )
        except sqlite3.Error as exc:
            raise StorageError("cannot verify evidence schema") from exc
        if (
            app_id != SQLITE_APPLICATION_ID
            or version_value != SCHEMA_VERSION
            or integrity != "ok"
            or tables != {"metadata", "records"}
            or triggers != {"records_no_update", "records_no_delete", "metadata_no_update", "metadata_no_delete"}
            or len(metadata) != 1
            or metadata[0][0] != SCHEMA_VERSION
            or metadata_columns != ("singleton", "schema_version", "created_at_unix_ns")
            or record_columns
            != (
                "seq",
                "kind",
                "payload_json",
                "payload_sha256",
                "previous_record_sha256",
                "recorded_at_unix_ns",
                "record_sha256",
            )
        ):
            raise StorageError("evidence schema differs from the sealed schema")

    def verified_records(self) -> list[dict[str, Any]]:
        try:
            rows = self.connection.execute(
                "SELECT seq,kind,payload_json,payload_sha256,previous_record_sha256,recorded_at_unix_ns,record_sha256 FROM records ORDER BY seq"
            ).fetchall()
        except sqlite3.Error as exc:
            raise StorageError("cannot read evidence history") from exc
        payloads: list[dict[str, Any]] = []
        previous: str | None = None
        for expected_seq, row in enumerate(rows, start=1):
            if row["seq"] != expected_seq or row["previous_record_sha256"] != previous:
                raise StorageError("record sequence or predecessor differs")
            try:
                payload = json.loads(row["payload_json"])
            except (TypeError, json.JSONDecodeError) as exc:
                raise StorageError("record payload JSON is invalid") from exc
            if not isinstance(payload, dict) or canonical_json(payload) != row["payload_json"]:
                raise StorageError("record payload is not canonical")
            payload_hash = canonical_sha256(payload)
            if payload_hash != row["payload_sha256"] or payload.get("kind") != row["kind"]:
                raise StorageError("record payload hash differs")
            body = {
                "seq": expected_seq,
                "kind": row["kind"],
                "payload_sha256": payload_hash,
                "previous_record_sha256": previous,
                "recorded_at_unix_ns": row["recorded_at_unix_ns"],
            }
            if canonical_sha256(body) != row["record_sha256"]:
                raise StorageError("record hash chain differs")
            previous = row["record_sha256"]
            payloads.append(payload)
        validate_history(payloads)
        return payloads

    def append(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self.read_only:
            raise StorageError("read-only storage cannot append")
        payload_dict = dict(payload)
        encoded = canonical_json(payload_dict)
        if len(encoded.encode("utf-8")) > MAX_PAYLOAD_BYTES:
            raise StorageError("record payload is too large")
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            existing = self.verified_records()
            validate_history([*existing, payload_dict])
            previous_row = self.connection.execute(
                "SELECT record_sha256 FROM records ORDER BY seq DESC LIMIT 1"
            ).fetchone()
            previous = None if previous_row is None else previous_row[0]
            seq = len(existing) + 1
            observed = time.time_ns()
            payload_hash = canonical_sha256(payload_dict)
            body = {
                "seq": seq,
                "kind": payload_dict["kind"],
                "payload_sha256": payload_hash,
                "previous_record_sha256": previous,
                "recorded_at_unix_ns": observed,
            }
            record_hash = canonical_sha256(body)
            self.connection.execute(
                "INSERT INTO records(seq,kind,payload_json,payload_sha256,previous_record_sha256,recorded_at_unix_ns,record_sha256) VALUES(?,?,?,?,?,?,?)",
                (seq, payload_dict["kind"], encoded, payload_hash, previous, observed, record_hash),
            )
            self.connection.commit()
            self.verified_records()
            return {"state": "inserted", "seq": seq, "record_sha256": record_hash, "kind": payload_dict["kind"]}
        except Exception:
            self.connection.rollback()
            raise


def _single(records: Sequence[Mapping[str, Any]], kind: str) -> dict[str, Any] | None:
    matches = [dict(value) for value in records if value.get("kind") == kind]
    if len(matches) > 1:
        raise StudyError(f"duplicate {kind} record")
    return matches[0] if matches else None


def seal_preregistration(database: str | Path = DEFAULT_DATABASE) -> dict[str, Any]:
    provenance = collect_provenance(require_clean=True)
    with Storage.open(database, initialize=True) as storage:
        records = storage.verified_records()
        if records:
            prereg = _single(records, "preregistration")
            if prereg and prereg["provenance_sha256"] == provenance["provenance_sha256"]:
                return {"state": "already_sealed", "preregistration_sha256": prereg["preregistration_sha256"]}
            raise StudyError("existing study database binds another provenance")
        payload = build_preregistration(provenance)
        return storage.append(payload)


def _stage_sessions(records: Sequence[Mapping[str, Any]], stage: str) -> list[dict[str, Any]]:
    return [dict(value) for value in records if value.get("kind") == f"{stage}_session"]


def preflight_session(
    stage: str,
    session_id: str,
    *,
    database: str | Path = DEFAULT_DATABASE,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if stage not in STAGES or session_id not in SESSION_ORDER:
        raise StudyError("unknown stage or session")
    current = dict(provenance or collect_provenance(require_clean=True))
    with Storage.open(database, read_only=True) as storage:
        records = storage.verified_records()
    prereg = _single(records, "preregistration")
    if prereg is None or prereg["provenance_sha256"] != current["provenance_sha256"]:
        raise StudyError("current provenance differs from the sealed preregistration")
    if records[-1]["kind"] in {"session_failure", "study_decision"}:
        raise StudyError("study history is terminal")
    sessions = _stage_sessions(records, stage)
    if len(sessions) >= len(SESSION_ORDER) or SESSION_ORDER[len(sessions)] != session_id:
        raise StudyError("session is not the next preregistered session")
    summary = _single(records, "calibration_summary")
    seal = _single(records, "confirmation_seal")
    if stage == CALIBRATION and (summary is not None or seal is not None):
        raise StudyError("calibration is closed")
    if stage == CONFIRMATION and seal is None:
        raise StudyError("confirmation is locked")
    return {
        "preregistration_sha256": prereg["preregistration_sha256"],
        "confirmation_seal_sha256": None if seal is None else seal["confirmation_seal_sha256"],
        "provenance_sha256": current["provenance_sha256"],
        "stage": stage,
        "session_id": session_id,
    }


def _rss_peak_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _memory_snapshot(mx: Any) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for key, name in (
        ("mlx_active_memory_bytes", "get_active_memory"),
        ("mlx_peak_memory_bytes", "get_peak_memory"),
        ("mlx_cache_memory_bytes", "get_cache_memory"),
    ):
        function = getattr(mx, name, None)
        try:
            value = function() if callable(function) else None
        except Exception:
            value = None
        result[key] = value if type(value) is int and value >= 0 else None
    return result


def _pace(guard: BudgetGuard, seconds: float) -> None:
    guard.record_gpu(seconds)
    required = seconds * (1.0 - PACING_TARGET) / PACING_TARGET
    # Four full breaks keep even five ~2 s prefills below the absolute 9 s
    # allowance of a trailing 60 s window; the ratio-only calculation can be
    # insufficient at that finite-window boundary.
    for _ in range(max(4, math.ceil(required / REQUIRED_BREAK_SECONDS))):
        guard.required_break()


def measure_live_session(stage: str, session_id: str, binding: Mapping[str, Any]) -> dict[str, Any]:
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx

    process_started = time.process_time_ns()
    guard = BudgetGuard(POLICY)
    guard.before_candidate()
    snapshot = resolve_local_model_snapshot(MODEL_ID)
    model, tokenizer = load(str(snapshot.path))
    inner = model.language_model if hasattr(model, "language_model") else model
    body, head = inner.model, inner.lm_head
    sampler = make_sampler(temp=0.0)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT_CONTENT}], add_generation_prompt=True
    )
    token_ids = list(rendered if isinstance(rendered, list) else tokenizer.encode(rendered))
    if hashlib.sha256(PROMPT_CONTENT.encode("utf-8")).hexdigest() != PROMPT_CONTENT_SHA256:
        raise StudyError("prompt content digest differs")
    if len(token_ids) != PROMPT_TOKENS:
        raise StudyError(f"prompt token count differs: {len(token_ids)}")
    candidate_blocks = 0
    candidate_last_head_calls = 0

    def prefill(skip_head: bool) -> tuple[Any, Any, int]:
        nonlocal candidate_blocks, candidate_last_head_calls
        cache = make_prompt_cache(model)
        logits = None
        started = time.perf_counter_ns()
        for offset in range(0, len(token_ids), CHUNK):
            piece = mx.array([token_ids[offset : offset + CHUNK]])
            if skip_head:
                candidate_blocks += 1
                hidden = body(piece, cache=cache)
                is_last = offset + CHUNK >= len(token_ids)
                candidate_last_head_calls += int(is_last)
                logits = head(hidden[:, -1:, :]) if is_last else None
                mx.eval(logits if is_last else hidden)
            else:
                logits = model(piece, cache=cache)
                mx.eval(logits)
            mx.synchronize()
        ended = time.perf_counter_ns()
        duration_ns = ended - started  # stop before _pace(): guard sleeps are excluded
        if duration_ns <= 0 or logits is None:
            raise StudyError("prefill duration or logits are invalid")
        _pace(guard, duration_ns / 1e9)
        return cache, logits, duration_ns

    def generate(skip_head: bool) -> list[int]:
        cache, logits, _ = prefill(skip_head)
        started = time.perf_counter_ns()
        y = sampler(logits[:, -1, :].astype(mx.float32))[:, None]
        mx.eval(y)
        output = [int(y[0, 0])]
        for _ in range(CORRECTNESS_TOKENS - 1):
            next_logits = model(y, cache=cache)
            y = sampler(next_logits[:, -1, :].astype(mx.float32))[:, None]
            mx.eval(y)
            output.append(int(y[0, 0]))
        mx.synchronize()
        ended = time.perf_counter_ns()
        decode_ns = ended - started  # stop before _pace()
        _pace(guard, decode_ns / 1e9)
        return output

    reference_tokens = generate(False)
    before_candidate_blocks = candidate_blocks
    before_candidate_heads = candidate_last_head_calls
    candidate_tokens = generate(True)
    expected_blocks = math.ceil(len(token_ids) / CHUNK)
    candidate_path_exercised = (
        candidate_blocks - before_candidate_blocks == expected_blocks
        and candidate_last_head_calls - before_candidate_heads == 1
    )
    if reference_tokens != candidate_tokens:
        raise CorrectnessError("greedy token identity mismatch")
    if not candidate_path_exercised:
        raise CorrectnessError("registered candidate path was not exercised exactly")
    correctness = {
        "status": "token_identical",
        "token_ids": reference_tokens,
        "token_sha256": hashlib.sha256(canonical_json(reference_tokens).encode("utf-8")).hexdigest(),
        "token_count": len(reference_tokens),
        "finish_reason": "fixed_horizon",
        "prompt_truncated": False,
        "silent_fallback": False,
        "candidate_path_exercised": candidate_path_exercised,
    }

    def baseline_a() -> int:
        return prefill(False)[2]

    def baseline_b() -> int:
        return prefill(False)[2]

    def candidate_b() -> int:
        return prefill(True)[2]

    arm_a: Callable[[], int] = baseline_a
    arm_b: Callable[[], int] = baseline_b if stage == CALIBRATION else candidate_b

    def pair(order: str) -> tuple[int, int]:
        if order == "ab":
            return arm_a(), arm_b()
        if order == "ba":
            b_ns = arm_b()
            a_ns = arm_a()
            return a_ns, b_ns
        raise StudyError("unknown arm order")

    warmups = []
    for index, order in enumerate(orders_for(stage, session_id, warmup=True)):
        a_ns, b_ns = pair(order)
        warmups.append({"block_index": index, "order": order, "a_ns": a_ns, "b_ns": b_ns})
    measurements = []
    for index, order in enumerate(orders_for(stage, session_id, warmup=False)):
        a_ns, b_ns = pair(order)
        measurements.append({"block_index": index, "order": order, "a_ns": a_ns, "b_ns": b_ns})
    guard.finish_candidate()
    budget = {"passed": True, **guard.summary()}
    resources = {
        "cpu_process_ns": time.process_time_ns() - process_started,
        "rss_peak_bytes": _rss_peak_bytes(),
        **_memory_snapshot(mx),
    }
    return build_session(
        stage=stage,
        session_id=session_id,
        preregistration_sha256=binding["preregistration_sha256"],
        confirmation_seal_sha256=binding["confirmation_seal_sha256"],
        provenance_sha256=binding["provenance_sha256"],
        warmups=warmups,
        measurements=measurements,
        correctness=correctness,
        budget=budget,
        resources=resources,
    )


def execute_session(stage: str, session_id: str, database: str | Path = DEFAULT_DATABASE) -> dict[str, Any]:
    provenance = collect_provenance(require_clean=True)
    binding = preflight_session(stage, session_id, database=database, provenance=provenance)
    require_ac_power()
    try:
        payload = measure_live_session(stage, session_id, binding)
    except (Exception, SystemExit) as exc:
        current = collect_provenance(require_clean=True)
        if current["provenance_sha256"] == provenance["provenance_sha256"]:
            failure = build_failure(
                stage=stage,
                session_id=session_id,
                preregistration_sha256=binding["preregistration_sha256"],
                confirmation_seal_sha256=binding["confirmation_seal_sha256"],
                provenance_sha256=binding["provenance_sha256"],
                failure_type=type(exc).__name__,
            )
            with Storage.open(database) as storage:
                storage.append(failure)
        raise
    current = collect_provenance(require_clean=True)
    if current["provenance_sha256"] != provenance["provenance_sha256"]:
        raise StudyError("provenance changed during the hardware session")
    with Storage.open(database) as storage:
        return storage.append(payload)


def summarize_calibration(database: str | Path = DEFAULT_DATABASE) -> dict[str, Any]:
    provenance = collect_provenance(require_clean=True)
    with Storage.open(database) as storage:
        records = storage.verified_records()
        prereg = _single(records, "preregistration")
        if prereg is None or prereg["provenance_sha256"] != provenance["provenance_sha256"]:
            raise StudyError("calibration provenance differs")
        if _single(records, "calibration_summary") is not None:
            raise StudyError("calibration is already summarized")
        return storage.append(build_calibration_summary(_stage_sessions(records, CALIBRATION)))


def seal_confirmation(database: str | Path = DEFAULT_DATABASE) -> dict[str, Any]:
    provenance = collect_provenance(require_clean=True)
    with Storage.open(database) as storage:
        records = storage.verified_records()
        prereg = _single(records, "preregistration")
        summary = _single(records, "calibration_summary")
        if prereg is None or summary is None or prereg["provenance_sha256"] != provenance["provenance_sha256"]:
            raise StudyError("confirmation seal lacks matching calibration")
        if _single(records, "confirmation_seal") is not None:
            raise StudyError("confirmation is already sealed")
        return storage.append(
            build_confirmation_seal(summary, _stage_sessions(records, CALIBRATION))
        )


def decide_study(database: str | Path = DEFAULT_DATABASE) -> dict[str, Any]:
    provenance = collect_provenance(require_clean=True)
    with Storage.open(database) as storage:
        records = storage.verified_records()
        prereg = _single(records, "preregistration")
        seal = _single(records, "confirmation_seal")
        if prereg is None or seal is None or prereg["provenance_sha256"] != provenance["provenance_sha256"]:
            raise StudyError("decision lacks matching sealed provenance")
        if _single(records, "study_decision") is not None:
            raise StudyError("study already has a decision")
        return storage.append(build_study_decision(_stage_sessions(records, CONFIRMATION), seal))


def snapshot(database: str | Path = DEFAULT_DATABASE, limit: int = 32) -> dict[str, Any]:
    if type(limit) is not int or not 1 <= limit <= MAX_UI_ROWS:
        raise StudyError("snapshot limit is invalid")
    with Storage.open(database, read_only=True) as storage:
        records = storage.verified_records()
        rows = storage.connection.execute(
            "SELECT seq,kind,recorded_at_unix_ns,record_sha256 FROM records ORDER BY seq DESC LIMIT ?",
            (limit,),
        ).fetchall()
    payload_by_kind = {value["kind"]: value for value in records}
    recent = []
    for row in rows:
        payload = records[row["seq"] - 1]
        metrics: dict[str, Any] = {}
        if payload["kind"].endswith("_session"):
            metrics["ratio"] = payload["metrics"]["ratio"]
        if payload["kind"] == "calibration_summary":
            metrics.update({"mde": payload["mde"], "ratio": payload["aggregate"]["ratio"]})
        if payload["kind"] == "study_decision":
            metrics.update({"mde": payload["mde"], "effect_percent": payload["effect_percent"]})
        recent.append(
            {
                "seq": row["seq"],
                "kind": row["kind"],
                "status": payload["status"],
                "recorded_at_unix_ns": row["recorded_at_unix_ns"],
                "record_sha256": row["record_sha256"],
                "formal_claim": payload["formal_claim"],
                "metrics": metrics,
            }
        )
    return {
        "study_id": STUDY_ID,
        "read_only": True,
        "records": len(records),
        "formal_claim_records": sum(value.get("formal_claim") is True for value in records),
        "current_status": records[-1]["status"] if records else "unsealed",
        "chain_head": None if not recent else recent[0]["record_sha256"],
        "calibration": payload_by_kind.get("calibration_summary"),
        "decision": payload_by_kind.get("study_decision"),
        "recent": recent,
    }


def _html_page(value: Mapping[str, Any]) -> bytes:
    rows = "".join(
        "<tr>"
        f"<td>{item['seq']}</td><td>{html.escape(item['kind'])}</td>"
        f"<td>{html.escape(item['status'])}</td><td>{html.escape(json.dumps(item['metrics'], sort_keys=True))}</td>"
        f"<td><code>{item['record_sha256'][:12]}</code></td></tr>"
        for item in value["recent"]
    )
    document = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width'><title>Friday head-skip study</title>
<style>body{{font:15px system-ui;margin:2rem;background:#f6f8fa;color:#17212b}}.card{{background:white;border:1px solid #d8dee4;border-radius:10px;padding:1rem;margin:1rem 0}}table{{width:100%;border-collapse:collapse}}th,td{{padding:.5rem;border-bottom:1px solid #ddd;text-align:left}}code{{font-size:.85em}}</style></head><body>
<h1>LM-head prefill study</h1><div class='card'><strong>{html.escape(value['current_status'])}</strong> — {value['records']} records, {value['formal_claim_records']} formal claim record.</div>
<div class='card'><table><thead><tr><th>#</th><th>Kind</th><th>Status</th><th>Metrics</th><th>Hash</th></tr></thead><tbody>{rows}</tbody></table></div></body></html>"""
    return document.encode("utf-8")


def serve(database: str | Path, port: int) -> None:
    if not 1024 <= port <= 65535:
        raise StudyError("dashboard port is invalid")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            try:
                parsed = urlsplit(self.path)
                query = parse_qs(parsed.query, strict_parsing=True, max_num_fields=1)
                if parsed.path == "/" and not query:
                    payload = _html_page(snapshot(database))
                    kind = "text/html; charset=utf-8"
                elif parsed.path == "/api/snapshot" and set(query) <= {"limit"}:
                    raw = query.get("limit", ["32"])[0]
                    payload = json.dumps(snapshot(database, int(raw)), sort_keys=True).encode()
                    kind = "application/json; charset=utf-8"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(payload)
            except Exception:
                self.send_error(400)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def _verified_sleep(seconds: float) -> None:
    started = time.monotonic()
    time.sleep(seconds)
    if time.monotonic() - started + 1e-9 < seconds:
        raise StudyError("inter-session cooldown did not elapse")


def run_stage(stage: str, database: Path) -> int:
    with Storage.open(database, read_only=True) as storage:
        completed = len(_stage_sessions(storage.verified_records(), stage))
    for index in range(completed, len(SESSION_ORDER)):
        if index > completed or completed > 0:
            _verified_sleep(INTER_SESSION_COOLDOWN_SECONDS)
        session_id = SESSION_ORDER[index]
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--database",
                str(database),
                "session",
                "--stage",
                stage,
                "--id",
                session_id,
                "--execute",
            ],
            cwd=PROJECT_ROOT,
            check=False,
        )
        if result.returncode != 0:
            _print({"state": "session_failed", "stage": stage, "session_id": session_id})
            return result.returncode
    outcome = summarize_calibration(database) if stage == CALIBRATION else decide_study(database)
    _print(outcome)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="head_skip_formal", allow_abbrev=False)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("spec", allow_abbrev=False)
    commands.add_parser("self-check", allow_abbrev=False)
    seal = commands.add_parser("seal", allow_abbrev=False)
    seal.add_argument("--execute", action="store_true")
    preflight = commands.add_parser("preflight", allow_abbrev=False)
    preflight.add_argument("--stage", choices=tuple(sorted(STAGES)), required=True)
    preflight.add_argument("--id", choices=SESSION_ORDER, required=True)
    session = commands.add_parser("session", allow_abbrev=False)
    session.add_argument("--stage", choices=tuple(sorted(STAGES)), required=True)
    session.add_argument("--id", choices=SESSION_ORDER, required=True)
    session.add_argument("--execute", action="store_true")
    summary = commands.add_parser("summarize-calibration", allow_abbrev=False)
    summary.add_argument("--execute", action="store_true")
    confirmation = commands.add_parser("seal-confirmation", allow_abbrev=False)
    confirmation.add_argument("--execute", action="store_true")
    decision = commands.add_parser("decide", allow_abbrev=False)
    decision.add_argument("--execute", action="store_true")
    stage = commands.add_parser("run-stage", allow_abbrev=False)
    stage.add_argument("--stage", choices=tuple(sorted(STAGES)), required=True)
    stage.add_argument("--execute", action="store_true")
    commands.add_parser("verify", allow_abbrev=False)
    snap = commands.add_parser("snapshot", allow_abbrev=False)
    snap.add_argument("--limit", type=int, default=32)
    dashboard = commands.add_parser("serve", allow_abbrev=False)
    dashboard.add_argument("--port", type=int, default=8772)
    return parser


def _released(args: argparse.Namespace) -> bool:
    if getattr(args, "execute", False):
        return True
    _print({"state": "not_released", "hint": "pass --execute"})
    return False


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "spec":
            _print(study_specification())
            return 0
        if args.command == "self-check":
            spec = study_specification()
            if hashlib.sha256(PROMPT_CONTENT.encode()).hexdigest() != PROMPT_CONTENT_SHA256:
                raise StudyError("prompt hash self-check failed")
            for stage in STAGES:
                for session_id in SESSION_ORDER:
                    if sorted(orders_for(stage, session_id, warmup=False)) != ["ab", "ab", "ba", "ba"]:
                        raise StudyError("measurement order is not balanced")
                    if sorted(orders_for(stage, session_id, warmup=True)) != ["ab", "ba"]:
                        raise StudyError("warmup order is not balanced")
            _print({"state": "self_check_passed", "study_id": STUDY_ID, "study_spec_sha256": canonical_sha256(spec)})
            return 0
        if args.command == "seal":
            if not _released(args):
                return 78
            _print(seal_preregistration(args.database))
            return 0
        if args.command == "preflight":
            _print({"state": "preflight_ok", **preflight_session(args.stage, args.id, database=args.database)})
            return 0
        if args.command == "session":
            if not _released(args):
                return 78
            _print(execute_session(args.stage, args.id, args.database))
            return 0
        if args.command == "summarize-calibration":
            if not _released(args):
                return 78
            _print(summarize_calibration(args.database))
            return 0
        if args.command == "seal-confirmation":
            if not _released(args):
                return 78
            _print(seal_confirmation(args.database))
            return 0
        if args.command == "decide":
            if not _released(args):
                return 78
            _print(decide_study(args.database))
            return 0
        if args.command == "run-stage":
            if not _released(args):
                return 78
            return run_stage(args.stage, args.database)
        if args.command == "verify":
            with Storage.open(args.database, read_only=True) as storage:
                records = storage.verified_records()
                head = storage.connection.execute("SELECT record_sha256 FROM records ORDER BY seq DESC LIMIT 1").fetchone()
            _print({"state": "verified", "records": len(records), "formal_claim_records": sum(value.get("formal_claim") is True for value in records), "chain_head": None if head is None else head[0], "read_only": True})
            return 0
        if args.command == "snapshot":
            _print(snapshot(args.database, args.limit))
            return 0
        if args.command == "serve":
            serve(args.database, args.port)
            return 0
    except Exception as exc:
        _print({"state": "failed", "failure_type": type(exc).__name__})
        return 1
    return 64


if __name__ == "__main__":
    raise SystemExit(main())
