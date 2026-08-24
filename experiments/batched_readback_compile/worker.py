#!/usr/bin/env python3
"""One fresh, authorised MLX process for the Cycle-17 readback study.

The two arms use the same Gemma 4B snapshot, greedy sampler, fixed 512-token
cache and one shared explicit-state ``mx.compile`` callable.  Their sole
experimental difference is the number of device tokens retained before the
boundary helper materialises one vector on the host: one versus at most eight.

There is deliberately no MLX or mlx-lm import at module import time.  Direct
unauthorised execution and ``--self-check`` are offline-only and cannot load a
model.  Every real arm gets fresh prefill/cache state, performs exactly eight
warmup forwards, and discards its measured cache after the stop decision.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREREGISTRATION_PATH = Path(__file__).with_name("PREREGISTRATION.md")
HARNESS_PATH = Path(__file__).with_name("measure_batched_readback.py")
STUDY_ID = "fixed-compiled-batched-readback-20260824-01"
RUN_ID = "fixed-compiled-batched-readback-validation-20260824-01"
CANDIDATE_ID = "fixed_compiled_batched_readback_n8_v1"
MODEL_KEY = "4b"
MODEL_ID = "mlx-community/gemma-3-4b-it-4bit"
MODEL_REVISION = "93724907d4ed1745d2fe50baadf3b0b01a65abf2"
EXPECTED_SNAPSHOT_SHA256 = (
    "e6edcd46c52b4cf5580f095185a94858565896df7f31c23522294e8f73b3edae"
)
EXPECTED_WEIGHT_SHA256 = (
    "94d3d701367d78584a9334ca00672b1c86e4aefa6a94167556c0485381e74af3"
)

CACHE_CAPACITY = 512
CAPACITY = CACHE_CAPACITY
FIXED_CACHE = True
FIXED_COMPILE = True
COMPILE_CONFIG = {"shapeless": False, "explicit_state": True}
EXPECTED_PROMPT_TOKENS = 322
MAX_PHYSICAL_TOKENS = 32
OUTPUT_TOKENS = MAX_PHYSICAL_TOKENS
MAX_DECODE_FORWARDS = MAX_PHYSICAL_TOKENS - 1
WARMUP_FORWARDS = 8
MAX_RESPONSE_BYTES = 16_384
MAX_EVENT_BYTES = 1_000_000
CONTINUOUS_GPU_LIMIT_SECONDS = 6.0
TOTAL_GPU_LIMIT_SECONDS = 120.0
MAX_WALL_SECONDS = 1_200.0
MIN_REQUIRED_BREAK_BLOCKS = 13
REQUIRED_BREAK_SECONDS = 4.0
MAX_RSS_BYTES = 5 * 1024**3
MAX_MLX_BYTES = 5 * 1024**3
BOOTSTRAP_SEED = 20260824
BOOTSTRAP_RESAMPLES = 10_000
FROZEN_PREREGISTRATION_SHA256 = (
    "74f63c36ddd141c4b4666d9f15d7b17d3ac9294e2d63cb29f6d9e35a80db21b1"
)

ARM_NAMES = ("fixed_compiled_readback_1", "fixed_compiled_readback_8")
ARMS = ARM_NAMES
INTERVALS = (1, 8)
ARM_INTERVALS = {
    "fixed_compiled_readback_1": 1,
    "fixed_compiled_readback_8": 8,
}
ARM_ORDERS = (
    ("fixed_compiled_readback_1", "fixed_compiled_readback_8"),
    ("fixed_compiled_readback_8", "fixed_compiled_readback_1"),
)
EXPECTED_EOS_TOKEN_IDS = (1, 106)
PROTOCOL_VERSION = 1
AUTH_ENV_PREFIX = "FRIDAY_BRB_"
AUTH_NONCE = "cycle17-fixed-compiled-batched-readback-v1"
AUTH_REQUIRED_ENV_NAMES = frozenset(
    {
        "FRIDAY_BRB_PARENT_PID",
        "FRIDAY_BRB_RUN_ID",
        "FRIDAY_BRB_MODEL_KEY",
        "FRIDAY_BRB_NONCE",
        "FRIDAY_BRB_BLOCK",
        "FRIDAY_BRB_ARM_ORDER",
        "FRIDAY_BRB_SNAPSHOT_PATH",
        "FRIDAY_BRB_SNAPSHOT_REVISION",
        "FRIDAY_BRB_SNAPSHOT_SHA256",
        "FRIDAY_BRB_WEIGHT_SHA256",
        "FRIDAY_BRB_SNAPSHOT_STAT_MANIFEST",
        "FRIDAY_BRB_PREREG_SHA256",
        "FRIDAY_BRB_PROMPT_SHA256",
        "FRIDAY_BRB_PROTOCOL_VERSION",
        "FRIDAY_BRB_PROTOCOL_SHA256",
        "FRIDAY_BRB_CODE_FINGERPRINTS",
        "FRIDAY_BRB_CODE_SHA256",
        "FRIDAY_BRB_ENVIRONMENT_SHA256",
    }
)
OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "PYTHONNOUSERSITE": "1",
}
UNSAFE_ENVIRONMENT = ("PYTHONHOME", "PYTHONINSPECT", "PYTHONPATH", "PYTHONSTARTUP")

# Byte-identical to the Cycle-14/15/16 planner prompt.  Never add a newline.
PLANNER_PROMPT = """You choose exactly one next Project Friday experiment.

Hardware: Apple M1 Max, 32 GB unified memory. Use only the evidence below.

Measured evidence:
- persistent_service_qualification: keeping Gemma 4B loaded reduced paired time to first output by 65.3032%; all greedy outputs matched exactly. Multi-turn and parallel-request qualification are still missing.
- batched_readback: isolated decode readback accounts for 12.98% per output token, but batching the checks can emit extra tokens and therefore needs a later correctness study.
- host_readback_upper_bound: 15.3% is only an upper bound, not a directly usable implementation.
- kv_cache_preallocation_ab: 4.4263% of decode time is correlated with reallocations, but the first step is confounded and the cache change still requires separate architecture permission.

Fixed selection policy:
1. Prefer the largest already confirmed end-to-end lever that also closes a required missing workload.
2. Do not choose a diagnostic upper bound.
3. Do not choose a permission-blocked cache change.
4. Choose exactly one ID from this list: persistent_service_qualification, batched_readback, host_readback_upper_bound, kv_cache_preallocation_ab.

Return only a JSON object with exactly one key named candidate_id and no prose, markdown, or explanation."""
PROMPT_SHA256 = hashlib.sha256(PLANNER_PROMPT.encode("utf-8")).hexdigest()
EXPECTED_PROMPT_SHA256 = (
    "c746eca8644a18fc75673acb9b3dbdf03825cbfba6c76faede5d909cf3d2ea0b"
)
EXPECTED_PROMPT_TOKEN_SHA256 = (
    "80ecf700cf0dfdc82616c73f1b6a5fccc137b68e9bb9586ca376c3f2adb260ad"
)
EXPECTED_RENDERED_PROMPT_SHA256 = (
    "9e18d10b7b101bda3d28593190e622544d474655872aed826c9cbc44211a2cca"
)

BOUNDARY_REQUIRED_FIELDS = frozenset(
    {
        "boundary_index",
        "physical_start_index",
        "physical_end_index",
        "block_size",
        "block_latency_ns",
        "host_available_ns",
        "eos_offset_in_block",
        "readback_ns",
        "readback_measurement_scope",
        "host_transfer_api_calls",
        "host_transfer_method",
        "host_transfer_physical_dma_count",
        "vector_block_readback_supported",
    }
)
RESOURCE_REQUIRED_FIELDS = frozenset(
    {
        "rss_peak_bytes",
        "mlx_peak_bytes",
        "swap_before_bytes",
        "swap_after_bytes",
        "swap_delta_bytes",
    }
)
ARM_BUDGET_REQUIRED_FIELDS = frozenset(
    {
        "observed_model_work_ns",
        "charged_model_work_ns",
        "charge_accepted",
        "guard_gpu_work_before_seconds",
        "guard_gpu_work_after_seconds",
        "guard_recorded_model_work_ns",
        "duty_formula_break_seconds",
        "required_break_blocks",
    }
)
BUDGET_REQUIRED_FIELDS = frozenset(
    {
        "gpu_work_seconds",
        "max_continuous_gpu_seconds",
        "cooldown_seconds",
        "required_break_seconds",
        "wall_seconds",
        "gpu_work_limit_seconds",
        "continuous_gpu_limit_seconds",
        "duty_cycle_limit",
        "wall_limit_seconds",
        "candidate_cooldown_seconds",
        "required_break_limit_seconds",
    }
)
CORRECTNESS_REQUIRED_FIELDS = frozenset(
    {
        "logical_tokens_equal",
        "visible_tokens_equal",
        "visible_text_equal",
        "prompt_identity_equal",
        "physical_tokens_equal_when_no_eos",
        "first_mismatch",
        "pass",
    }
)
ARM_REQUIRED_FIELDS = frozenset(
    {
        "arm",
        "readback_interval",
        "max_physical_tokens",
        "eos_token_ids",
        "physical_tokens",
        "logical_tokens",
        "visible_tokens",
        "physical_token_count",
        "logical_token_count",
        "visible_token_count",
        "overproduced_tokens",
        "eos_found",
        "eos_position",
        "eos_token_id",
        "finish_reason",
        "physical_token_sha256",
        "logical_token_sha256",
        "visible_token_sha256",
        "eos_block",
        "eos_readback_block",
        "cache_discarded",
        "visible_text",
        "text_sha256",
        "text_utf8_sha256",
        "token_sha256",
        "decode_critical_path_ns",
        "physical_forwards",
        "decode_forwards",
        "forward_submit_ns",
        "readback_count",
        "readback_block_sizes",
        "readback_records",
        "readback_boundaries",
        "readback_ns",
        "readback_total_ns",
        "readback_measurement_scope",
        "host_transfer_api_call_count",
        "host_transfer_physical_dma_count",
        "host_transfer_method",
        "host_boundary_available",
        "vector_block_readback_supported",
        "host_available_ns_by_physical_token",
        "host_boundary_available_ns",
        "host_available_total_ns",
        "first_host_token_ns",
        "block_latency_ns",
        "boundary_interarrival_ns",
        "boundary_interarrival_p50_ns",
        "boundary_interarrival_p95_ns",
        "boundary_interarrival_p99_ns",
        "stop_decision_ns",
        "token_rate",
        "ttft_ns",
        "prefill_ns",
        "cache_conversion_ns",
        "warmup_forwards",
        "warmup_forward_submit_ns",
        "warmup_total_ns",
        "first_warmup_materialization_ns",
        "warmup_readback_ns",
        "warmup_cache_discarded",
        "warmup_prefill_ns",
        "warmup_cache_conversion_ns",
        "cache_capacity",
        "fixed_cache",
        "fixed_compile",
        "compile_config",
        "compile_callable_shared",
        "greedy",
        "sampler_temperature",
        "arm_wall_ns",
        "observed_model_work_ns",
        "charged_model_work_ns",
        "charge_accepted",
        "prompt_sha256",
        "prompt_token_sha256",
        "rendered_prompt_sha256",
        "compile_wrapper_ns",
        "compile_cold_ns",
        "budget_summary",
        "resource_snapshot",
    }
)
EVENT_REQUIRED_FIELDS = frozenset(
    {
        "event",
        "status",
        "study_id",
        "run_id",
        "candidate_id",
        "formal_claim",
        "protocol_version",
        "process_index",
        "arm_order",
        "arms",
        "arm_budget",
        "arm_resources",
        "correctness",
        "error",
        "pid",
        "load_count",
        "model_key",
        "model_id",
        "snapshot_revision",
        "snapshot_path",
        "snapshot_sha256",
        "weight_sha256",
        "snapshot_integrity",
        "model_load_ns",
        "compile_wrapper_ns",
        "compile_cold_ns",
        "cache_capacity",
        "max_physical_tokens",
        "warmup_forwards_per_arm",
        "prompt_sha256",
        "prompt_token_ids",
        "prompt_token_sha256",
        "rendered_prompt_b64",
        "rendered_prompt_sha256",
        "prompt_tokens",
        "eos_token_ids",
        "sampler_temperature",
        "greedy",
        "device",
        "power_source",
        "model_work_ns",
        "observed_model_work_ns",
        "charged_model_work_ns",
        "guard_recorded_model_work_ns",
        "budget",
        "rss_peak_bytes",
        "mlx_peak_bytes",
        "swap_before_bytes",
        "swap_after_bytes",
        "swap_delta_bytes",
        "worker_watchdog_seconds",
        "host_transfer_claim",
        "determinism",
        "preregistration_sha256",
        "code_fingerprints",
        "code_sha256",
        "environment_sha256",
        "process_wall_ns",
    }
)


class WorkerError(RuntimeError):
    """The closed worker protocol cannot continue safely."""


class CandidateNotRunnable(WorkerError):
    """The fixed-compiled vector-readback API is unavailable or invalid."""


class ResourceFailure(WorkerError):
    """A resource, budget, swap, or memory failure occurred."""


class BudgetGuard:
    """Offline-only public description of the registered worker limits.

    The authorised path imports and uses ``friday_evidence.budget.BudgetGuard``.
    This tiny value object lets contract tests inspect the limits without
    importing MLX, touching a device, or sleeping.
    """

    def __init__(
        self,
        *,
        duty_cycle: float = 0.15,
        continuous_limit_seconds: float = 6.0,
    ) -> None:
        if duty_cycle != 0.15 or continuous_limit_seconds != 6.0:
            raise ValueError("only the preregistered BudgetGuard policy is allowed")
        self.duty_cycle = duty_cycle
        self.duty_cycle_limit = duty_cycle
        self.continuous_limit_seconds = continuous_limit_seconds
        self.continuous_gpu_limit_seconds = continuous_limit_seconds


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def code_fingerprints() -> dict[str, str]:
    """Hash every executable/spec byte shared by parent and worker."""

    paths = (Path(__file__), HARNESS_PATH, PREREGISTRATION_PATH)
    return {
        path.relative_to(PROJECT_ROOT).as_posix(): _sha256_file(path)
        for path in paths
    }


def environment_fingerprint() -> str:
    return _sha256_bytes(
        _canonical_json(
            {
                "fixed": OFFLINE_ENVIRONMENT,
                "removed": UNSAFE_ENVIRONMENT,
                "python": str(Path(sys.executable).resolve()),
                "machine": platform.machine(),
            }
        )
    )


def protocol_contract() -> dict[str, Any]:
    """Return the complete offline parent/worker protocol contract."""

    return {
        "protocol_version": PROTOCOL_VERSION,
        "auth_env_prefix": AUTH_ENV_PREFIX,
        "auth_nonce": AUTH_NONCE,
        "auth_required_env_names": sorted(AUTH_REQUIRED_ENV_NAMES),
        "study_id": STUDY_ID,
        "run_id": RUN_ID,
        "candidate_id": CANDIDATE_ID,
        "model_key": MODEL_KEY,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "arm_names": list(ARM_NAMES),
        "arm_intervals": dict(ARM_INTERVALS),
        "arm_orders": [list(order) for order in ARM_ORDERS],
        "cache_capacity": CACHE_CAPACITY,
        "max_physical_tokens": MAX_PHYSICAL_TOKENS,
        "warmup_forwards": WARMUP_FORWARDS,
        "prompt_sha256": EXPECTED_PROMPT_SHA256,
        "prompt_token_sha256": EXPECTED_PROMPT_TOKEN_SHA256,
        "rendered_prompt_sha256": EXPECTED_RENDERED_PROMPT_SHA256,
        "preregistration_sha256": FROZEN_PREREGISTRATION_SHA256,
        "event_required_fields": sorted(EVENT_REQUIRED_FIELDS),
        "arm_required_fields": sorted(ARM_REQUIRED_FIELDS),
        "boundary_required_fields": sorted(BOUNDARY_REQUIRED_FIELDS),
        "resource_required_fields": sorted(RESOURCE_REQUIRED_FIELDS),
        "arm_budget_required_fields": sorted(ARM_BUDGET_REQUIRED_FIELDS),
        "budget_required_fields": sorted(BUDGET_REQUIRED_FIELDS),
        "correctness_required_fields": sorted(CORRECTNESS_REQUIRED_FIELDS),
    }


PROTOCOL_SHA256 = _sha256_bytes(_canonical_json(protocol_contract()))


def parse_one_json(payload: str | bytes) -> dict[str, Any]:
    """Parse one capped JSON object, rejecting duplicates, NaN and multiline data."""

    if isinstance(payload, str):
        encoded = payload.encode("utf-8")
    elif isinstance(payload, bytes):
        encoded = payload
    else:
        raise TypeError("JSON event must be text or bytes")
    if not encoded or len(encoded) > MAX_EVENT_BYTES:
        raise ValueError("JSON event is empty or oversize")
    if encoded.endswith(b"\n"):
        encoded = encoded[:-1]
    if not encoded or b"\n" in encoded or b"\r" in encoded:
        raise ValueError("multiline JSON is forbidden")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=unique,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {item}")
            ),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("invalid strict JSON event") from exc
    if not isinstance(value, dict):
        raise ValueError("JSON event must be one object")
    return value


parse_event = parse_one_json
decode_event = parse_one_json


def _emit(value: dict[str, Any]) -> None:
    payload = _canonical_json(value)
    if len(payload) > MAX_EVENT_BYTES:
        raise WorkerError("worker JSON event is oversize")
    sys.stdout.buffer.write(payload + b"\n")
    sys.stdout.buffer.flush()


def _valid_token_list(value: Any, *, label: str) -> list[int]:
    if not isinstance(value, list) or any(type(item) is not int or item < 0 for item in value):
        raise ValueError(f"{label} must contain nonnegative integer token IDs")
    return list(value)


def normalize_tokens(event: dict[str, Any]) -> dict[str, Any]:
    """Apply the sealed physical/logical/visible EOS contract as a pure helper."""

    if not isinstance(event, dict):
        raise TypeError("token event must be an object")
    raw = _valid_token_list(event.get("tokens"), label="tokens")
    eos_ids = _valid_token_list(event.get("eos_ids"), label="eos_ids")
    maximum = event.get("max_tokens", MAX_PHYSICAL_TOKENS)
    if type(maximum) is not int or not 1 <= maximum <= MAX_PHYSICAL_TOKENS:
        raise ValueError("max_tokens is outside the sealed physical cap")
    if not eos_ids or len(set(eos_ids)) != len(eos_ids):
        raise ValueError("eos_ids must be a nonempty unique list")

    # This pure helper deliberately clips supplied test vectors at the sealed
    # physical cap; the real loop itself can never schedule beyond that cap.
    physical = raw[:maximum]
    eos_set = set(eos_ids)
    eos_position = next(
        (index for index, token in enumerate(physical) if token in eos_set),
        None,
    )
    if eos_position is None:
        logical = list(physical)
        visible = list(physical)
        overproduced = 0
        eos_token_id = None
        finish_reason = "length"
    else:
        logical = physical[: eos_position + 1]
        visible = physical[:eos_position]
        overproduced = len(physical) - len(logical)
        eos_token_id = physical[eos_position]
        finish_reason = "stop"
    return {
        "physical_tokens": physical,
        "logical_tokens": logical,
        "visible_tokens": visible,
        "physical_token_count": len(physical),
        "logical_token_count": len(logical),
        "visible_token_count": len(visible),
        "overproduced_tokens": overproduced,
        "eos_found": eos_position is not None,
        "eos_position": eos_position,
        "eos_token_id": eos_token_id,
        "finish_reason": finish_reason,
        "physical_token_sha256": _sha256_bytes(_canonical_json(physical)),
        "logical_token_sha256": _sha256_bytes(_canonical_json(logical)),
        "visible_token_sha256": _sha256_bytes(_canonical_json(visible)),
    }


finalize_tokens = normalize_tokens


def validate_terminal_event(event: dict[str, Any]) -> dict[str, Any]:
    """Reject retry-like pseudo states; terminal worker states are closed."""

    if not isinstance(event, dict):
        raise TypeError("terminal event must be an object")
    status = event.get("status")
    if status in {"timeout", "partial", "retry"}:
        raise ValueError("timeout/partial/retry is not a valid worker status")
    if status not in {
        "complete",
        "candidate_not_runnable",
        "correctness_failed",
        "resource_or_budget_failed",
        "error",
    }:
        raise ValueError("unknown worker status")
    return event


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_event(event: dict[str, Any]) -> dict[str, Any]:
    """Offline structural validation used by tests and the parent harness."""

    if not isinstance(event, dict) or set(event) != EVENT_REQUIRED_FIELDS:
        raise ValueError("worker event field set changed")
    validate_terminal_event(event)
    if event.get("status") != "complete":
        return event
    if event.get("event") != "complete" or event.get("load_count") != 1:
        raise ValueError("complete worker event has invalid identity")
    arms = event.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(ARM_NAMES):
        raise ValueError("complete worker event has invalid arms")
    for arm_name in ARM_NAMES:
        arm = arms[arm_name]
        if (
            not isinstance(arm, dict)
            or set(arm) != ARM_REQUIRED_FIELDS
            or arm.get("arm") != arm_name
        ):
            raise ValueError("arm identity mismatch")
        if (
            arm.get("readback_interval") != ARM_INTERVALS[arm_name]
            or arm.get("cache_capacity") != CACHE_CAPACITY
            or arm.get("fixed_cache") is not True
            or arm.get("fixed_compile") is not True
            or arm.get("compile_callable_shared") is not True
            or arm.get("warmup_forwards") != WARMUP_FORWARDS
            or arm.get("cache_discarded") is not True
            or arm.get("host_boundary_available") is not True
            or arm.get("vector_block_readback_supported") is not True
        ):
            raise ValueError("arm fixed/readback invariants changed")
        for field in (
            "physical_token_sha256",
            "logical_token_sha256",
            "visible_token_sha256",
            "text_sha256",
        ):
            if not _is_sha256(arm.get(field)):
                raise ValueError(f"arm hash mismatch: {field}")
        normalized = normalize_tokens(
            {
                "tokens": arm.get("physical_tokens"),
                "eos_ids": arm.get("eos_token_ids"),
                "max_tokens": MAX_PHYSICAL_TOKENS,
            }
        )
        for field in (
            "physical_tokens",
            "logical_tokens",
            "visible_tokens",
            "physical_token_sha256",
            "logical_token_sha256",
            "visible_token_sha256",
        ):
            if arm.get(field) != normalized[field]:
                raise ValueError(f"arm token contract mismatch: {field}")
        if any(
            arm.get(field) != normalized[field]
            for field in (
                "physical_token_count",
                "logical_token_count",
                "visible_token_count",
                "overproduced_tokens",
                "eos_found",
                "eos_position",
                "eos_token_id",
                "finish_reason",
            )
        ):
            raise ValueError("arm EOS/count contract mismatch")
        blocks = arm.get("readback_block_sizes")
        if (
            not isinstance(blocks, list)
            or any(type(size) is not int or not 1 <= size <= ARM_INTERVALS[arm_name] for size in blocks)
            or sum(blocks) != normalized["physical_token_count"]
            or arm.get("readback_count") != len(blocks)
            or arm.get("host_transfer_api_call_count") != len(blocks)
            or arm.get("physical_forwards") != normalized["physical_token_count"] - 1
            or arm.get("decode_forwards") != arm.get("physical_forwards")
        ):
            raise ValueError("arm readback/forward counts mismatch")
        records = arm.get("readback_records")
        if (
            not isinstance(records, list)
            or any(
                not isinstance(record, dict)
                or set(record) != BOUNDARY_REQUIRED_FIELDS
                for record in records
            )
            or arm.get("readback_boundaries") != records
        ):
            raise ValueError("arm readback boundary schema changed")
        text = arm.get("visible_text")
        if (
            not isinstance(text, str)
            or _sha256_bytes(text.encode("utf-8")) != arm.get("text_sha256")
            or arm.get("text_utf8_sha256") != arm.get("text_sha256")
            or arm.get("token_sha256") != arm.get("logical_token_sha256")
        ):
            raise ValueError("arm visible text/hash mismatch")
    correctness = event.get("correctness")
    if not isinstance(correctness, dict) or correctness.get("pass") is not True:
        raise ValueError("complete event did not pass correctness")
    return event


validate_result = validate_event


def decode_loop(
    model: Any,
    initial_tokens: Iterable[Any] | None = None,
    *,
    interval: int,
    max_tokens: int = MAX_PHYSICAL_TOKENS,
    eos_ids: set[int] | tuple[int, ...] = EXPECTED_EOS_TOKEN_IDS,
    boundary_helper: Callable[[list[Any]], list[int]] | None = None,
) -> dict[str, Any]:
    """Small offline fake loop proving that host scalar reads occur at boundaries.

    The authorised MLX path uses ``_run_device_decode`` below.  This helper is
    intentionally pure enough for unit fakes and never imports or allocates MLX.
    """

    if interval not in INTERVALS or not 1 <= max_tokens <= MAX_PHYSICAL_TOKENS:
        raise ValueError("invalid readback interval or cap")
    pending = list(initial_tokens or [])[:max_tokens]
    if not pending:
        # Exercise a callable fake once without manufacturing a device result.
        try:
            response = model(0)
            logits = getattr(response, "logits", [])
            pending = list(logits[:1]) if isinstance(logits, list) else []
        except Exception:
            pending = []

    def default_boundary(block: list[Any]) -> list[int]:
        host: list[int] = []
        for token in block:
            value = token.item() if hasattr(token, "item") else token
            while hasattr(value, "value"):
                value = value.value
            host.append(int(value))
        return host

    materialize = boundary_helper or default_boundary
    physical: list[int] = []
    readback_blocks: list[int] = []
    for start in range(0, len(pending), interval):
        host = materialize(pending[start : start + interval])
        readback_blocks.append(len(host))
        physical.extend(host)
        if any(token in eos_ids for token in host):
            break
    result = normalize_tokens(
        {"tokens": physical, "eos_ids": sorted(eos_ids), "max_tokens": max_tokens}
    )
    result.update(
        {
            "readback_interval": interval,
            "readback_count": len(readback_blocks),
            "readback_block_sizes": readback_blocks,
            "cache_discarded": True,
        }
    )
    return result


run_decode = decode_loop
batched_decode = decode_loop


def _resource_failure(exc: BaseException) -> bool:
    if isinstance(exc, (MemoryError, ResourceFailure)):
        return True
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "out of memory",
            "oom",
            "killed",
            "abort",
            "resource exhausted",
            "swap",
            "budget",
        )
    )


def _classify_candidate_failure(context: str, exc: BaseException) -> None:
    if _resource_failure(exc):
        raise ResourceFailure(
            f"{context} resource failure: {type(exc).__name__}: {exc}"
        ) from exc
    raise CandidateNotRunnable(
        f"{context} is not runnable: {type(exc).__name__}: {exc}"
    ) from exc


class FixedKVCache:
    """Fixed-cache layer with shared outer-only offset, identical to Cycle 16."""

    def __init__(self, state: dict[str, Any], position: dict[str, Any], mx: Any):
        if set(state) != {"keys", "values"} or set(position) != {"offset"}:
            raise CandidateNotRunnable("fixed-cache state tree is not closed")
        self._state = state
        self._position = position
        self._mx = mx

    @property
    def offset(self) -> Any:
        return self._position["offset"]

    @property
    def keys(self) -> Any:
        return self._state["keys"]

    @property
    def values(self) -> Any:
        return self._state["values"]

    def update_and_fetch(self, keys: Any, values: Any) -> tuple[Any, Any]:
        if keys.shape[2] < 1 or keys.shape[2] > CACHE_CAPACITY:
            raise CandidateNotRunnable("fixed cache received an invalid step shape")
        zero = self._mx.array(0, dtype=self.offset.dtype)
        starts = self._mx.stack((zero, zero, self.offset, zero))
        self._state["keys"] = self._mx.slice_update(
            self._state["keys"], keys, start_indices=starts, axes=(0, 1, 2, 3)
        )
        self._state["values"] = self._mx.slice_update(
            self._state["values"], values, start_indices=starts, axes=(0, 1, 2, 3)
        )
        # Cache layers never mutate the shared offset.  The outer model call
        # advances it once, preserving global and sliding-layer semantics.
        return self._state["keys"], self._state["values"]

    def make_mask(
        self,
        n_tokens: int,
        *,
        window_size: int | None = None,
        return_array: bool = False,
    ) -> Any:
        del return_array
        if type(n_tokens) is not int or n_tokens < 1:
            raise CandidateNotRunnable("fixed mask received invalid token count")
        positions = self._mx.arange(CACHE_CAPACITY, dtype=self.offset.dtype)
        query_positions = self.offset + self._mx.arange(
            n_tokens, dtype=self.offset.dtype
        )
        mask = positions[None, :] <= query_positions[:, None]
        mask = mask & (positions[None, :] < self.offset + n_tokens)
        if window_size is not None:
            mask = mask & (
                positions[None, :] >= query_positions[:, None] - window_size + 1
            )
        return mask[None, None, :, :]


def _fixed_state_from_standard_cache(
    cache: list[Any], prompt_tokens: int, mx: Any
) -> dict[str, Any]:
    """Copy a fresh standard prefill cache into fixed 512-position storage."""

    if not cache or type(prompt_tokens) is not int or prompt_tokens < 1:
        raise CandidateNotRunnable("standard prefill cache is empty")
    layers: list[dict[str, Any]] = []
    for layer in cache:
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if keys is None or values is None:
            raise CandidateNotRunnable("standard cache lacks keys or values")
        if len(keys.shape) != 4 or len(values.shape) != 4:
            raise CandidateNotRunnable("standard cache tensors have invalid rank")
        if keys.shape[2] < prompt_tokens or values.shape[2] < prompt_tokens:
            raise CandidateNotRunnable("standard cache is shorter than the prompt")
        if keys.shape[2] > CACHE_CAPACITY or values.shape[2] > CACHE_CAPACITY:
            raise CandidateNotRunnable("standard cache exceeds fixed capacity")
        key_shape = (keys.shape[0], keys.shape[1], CACHE_CAPACITY, keys.shape[3])
        value_shape = (
            values.shape[0],
            values.shape[1],
            CACHE_CAPACITY,
            values.shape[3],
        )
        padded_keys = mx.zeros(key_shape, dtype=keys.dtype)
        padded_values = mx.zeros(value_shape, dtype=values.dtype)
        start = mx.array((0, 0, 0, 0), dtype=mx.int32)
        padded_keys = mx.slice_update(
            padded_keys,
            keys[..., :prompt_tokens, :],
            start_indices=start,
            axes=(0, 1, 2, 3),
        )
        padded_values = mx.slice_update(
            padded_values,
            values[..., :prompt_tokens, :],
            start_indices=start,
            axes=(0, 1, 2, 3),
        )
        layers.append({"keys": padded_keys, "values": padded_values})
    return {
        "position": {"offset": mx.array(prompt_tokens, dtype=mx.int32)},
        "layers": layers,
    }


def _fixed_caches(state_tree: dict[str, Any], mx: Any) -> list[FixedKVCache]:
    position = state_tree.get("position")
    layers = state_tree.get("layers")
    if not isinstance(position, dict) or not isinstance(layers, list):
        raise CandidateNotRunnable("fixed state is not a position/layers tree")
    return [FixedKVCache(layer, position, mx) for layer in layers]


def _fixed_forward(
    model: Any, input_ids: Any, state_tree: dict[str, Any], mx: Any
) -> tuple[Any, dict[str, Any]]:
    """Run identical fixed-cache mathematics and advance the offset once."""

    caches = _fixed_caches(state_tree, mx)
    logits = model(input_ids, cache=caches)
    old_offset = state_tree["position"]["offset"]
    new_state = {
        "position": {"offset": old_offset + input_ids.shape[1]},
        "layers": [
            {"keys": cache.keys, "values": cache.values} for cache in caches
        ],
    }
    return logits, new_state


def _make_compiled_forward(model: Any, mx: Any) -> Callable[..., Any]:
    """Create one fixed-compile callable with complete explicit state."""

    def body(input_ids: Any, state: dict[str, Any]):
        return _fixed_forward(model, input_ids, state, mx)

    try:
        return mx.compile(body, shapeless=False)
    except Exception as exc:
        _classify_candidate_failure("mx.compile explicit-state wrapper", exc)
    raise AssertionError("unreachable")


def _tree_leaves(value: Any) -> list[Any]:
    leaves: list[Any] = []
    if isinstance(value, dict):
        for key in sorted(value):
            leaves.extend(_tree_leaves(value[key]))
    elif isinstance(value, (list, tuple)):
        for child in value:
            leaves.extend(_tree_leaves(child))
    else:
        leaves.append(value)
    return leaves


def _selected_device_token(logits: Any, sampler: Callable[[Any], Any]) -> Any:
    selected = sampler(logits[:, -1, :])
    try:
        if int(selected.size) != 1:
            raise CandidateNotRunnable("greedy sampler did not return one device token")
        return selected.reshape((1,))
    except CandidateNotRunnable:
        raise
    except Exception as exc:
        _classify_candidate_failure("device greedy-token shape", exc)
    raise AssertionError("unreachable")


def _schedule_device_dependencies(token: Any, state: dict[str, Any], mx: Any) -> None:
    """Schedule lazy device dependencies without synchronising or reading host data."""

    try:
        mx.async_eval(token, *_tree_leaves(state))
    except Exception as exc:
        _classify_candidate_failure("asynchronous fixed-state evaluation", exc)


def _host_readback_boundary(
    device_tokens: list[Any],
    state: dict[str, Any],
    mx: Any,
) -> tuple[list[int], dict[str, Any]]:
    """The only decode helper allowed to synchronise and read tokens on host.

    One flattened vector is passed to one ``tolist`` host-conversion call.  The
    public MLX API does not expose a physical-DMA counter, so evidence records
    one host API call and leaves the physical transfer count as ``None``.
    """

    if not device_tokens:
        raise CandidateNotRunnable("readback boundary received an empty block")
    started_ns = time.perf_counter_ns()
    try:
        block = mx.stack(tuple(device_tokens), axis=0).reshape((-1,))
        leaves = _tree_leaves(state)
        mx.async_eval(block, *leaves)
        mx.eval(block, *leaves)
        mx.synchronize()
        # Exactly one vector host conversion.  Never replace this with a loop
        # of scalar .item() calls: that would fake the N=8 treatment.
        raw = block.tolist()
    except Exception as exc:
        _classify_candidate_failure("single-vector readback boundary", exc)
    finished_ns = time.perf_counter_ns()
    if not isinstance(raw, list) or len(raw) != len(device_tokens):
        raise CandidateNotRunnable("vector readback returned the wrong block shape")
    if any(type(token) is not int or token < 0 for token in raw):
        raise CandidateNotRunnable("vector readback returned invalid token IDs")
    return list(raw), {
        "readback_ns": finished_ns - started_ns,
        "readback_measurement_scope": (
            "pending_device_eval_plus_sync_plus_single_vector_host_conversion"
        ),
        "host_transfer_api_calls": 1,
        "host_transfer_method": "single_vector_tolist",
        "host_transfer_physical_dma_count": None,
        "vector_block_readback_supported": True,
    }


def _prompt_ids(tokenizer: Any) -> tuple[list[int], bytes]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": PLANNER_PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str):
        raise WorkerError("chat template did not return rendered text")
    rendered_bytes = rendered.encode("utf-8")
    try:
        values = tokenizer.encode(rendered, add_special_tokens=False)
    except TypeError as exc:
        raise WorkerError("tokenizer encode lacks fixed special-token option") from exc
    if not isinstance(values, list):
        try:
            values = values.tolist()
        except AttributeError as exc:
            raise WorkerError("tokenizer returned invalid prompt tokens") from exc
    if not values or any(type(value) is not int or value < 0 for value in values):
        raise WorkerError("tokenizer returned invalid prompt token IDs")
    return list(values), rendered_bytes


def _materialize_fixed_state(state: dict[str, Any], mx: Any) -> None:
    try:
        leaves = _tree_leaves(state)
        mx.eval(*leaves)
        mx.synchronize()
    except Exception as exc:
        _classify_candidate_failure("fixed-cache conversion materialisation", exc)


def _prepare_prefill(
    model: Any,
    prompt_ids: list[int],
    mx: Any,
) -> dict[str, Any]:
    """Create a fresh standard prefill and independently converted fixed state."""

    total_started_ns = time.perf_counter_ns()
    prompt_array = mx.array([prompt_ids])
    cache = model.make_cache() if hasattr(model, "make_cache") else None
    if cache is None:
        raise CandidateNotRunnable("model did not expose a fresh standard cache")
    prefill_started_ns = time.perf_counter_ns()
    try:
        logits = model(prompt_array, cache=cache)
        mx.eval(logits)
        mx.synchronize()
    except Exception as exc:
        _classify_candidate_failure("fresh fixed-path prefill", exc)
    prefill_finished_ns = time.perf_counter_ns()
    conversion_started_ns = time.perf_counter_ns()
    try:
        state = _fixed_state_from_standard_cache(cache, len(prompt_ids), mx)
        _materialize_fixed_state(state, mx)
    except (CandidateNotRunnable, ResourceFailure):
        raise
    except Exception as exc:
        _classify_candidate_failure("standard-to-fixed cache conversion", exc)
    conversion_finished_ns = time.perf_counter_ns()
    return {
        "fixed_state": state,
        "prefill_logits": logits,
        "prefill_ns": prefill_finished_ns - prefill_started_ns,
        "cache_conversion_ns": conversion_finished_ns - conversion_started_ns,
        "preparation_total_ns": conversion_finished_ns - total_started_ns,
    }


def _compiled_step(
    callable_forward: Callable[..., Any],
    token: Any,
    state: dict[str, Any],
    sampler: Callable[[Any], Any],
    mx: Any,
) -> tuple[Any, dict[str, Any], int]:
    started_ns = time.perf_counter_ns()
    try:
        result = callable_forward(token.reshape((1, 1)), state)
        if not isinstance(result, tuple) or len(result) != 2:
            raise CandidateNotRunnable("compiled forward returned wrong structure")
        logits, next_state = result
        if not isinstance(next_state, dict):
            raise CandidateNotRunnable("compiled forward returned invalid state")
        selected = _selected_device_token(logits, sampler)
        _schedule_device_dependencies(selected, next_state, mx)
    except (CandidateNotRunnable, ResourceFailure):
        raise
    except Exception as exc:
        _classify_candidate_failure("compiled fixed forward", exc)
    return selected, next_state, time.perf_counter_ns() - started_ns


def _run_warmup(
    model: Any,
    prompt_ids: list[int],
    callable_forward: Callable[..., Any],
    sampler: Callable[[Any], Any],
    mx: Any,
) -> dict[str, Any]:
    """Run exactly eight fixed-compiled forwards on fresh disposable state."""

    prepared = _prepare_prefill(model, prompt_ids, mx)
    state = prepared["fixed_state"]
    token = _selected_device_token(prepared["prefill_logits"], sampler)
    submit_ns: list[int] = []
    first_materialized_ns: int | None = None
    readback_ns: list[int] = []
    warmup_started_ns = time.perf_counter_ns()
    for index in range(WARMUP_FORWARDS):
        first_forward_started_ns = time.perf_counter_ns() if index == 0 else None
        token, state, elapsed_ns = _compiled_step(
            callable_forward, token, state, sampler, mx
        )
        submit_ns.append(elapsed_ns)
        if index == 0:
            _, evidence = _host_readback_boundary([token], state, mx)
            if first_forward_started_ns is None:
                raise AssertionError("first warmup timer was not started")
            first_materialized_ns = time.perf_counter_ns() - first_forward_started_ns
            readback_ns.append(evidence["readback_ns"])
    _, evidence = _host_readback_boundary([token], state, mx)
    readback_ns.append(evidence["readback_ns"])
    warmup_finished_ns = time.perf_counter_ns()
    state = None
    return {
        "warmup_forwards": WARMUP_FORWARDS,
        "warmup_forward_submit_ns": submit_ns,
        "warmup_total_ns": warmup_finished_ns - warmup_started_ns,
        "first_warmup_materialization_ns": first_materialized_ns,
        "warmup_readback_ns": readback_ns,
        "warmup_cache_discarded": state is None,
        "warmup_prefill_ns": prepared["prefill_ns"],
        "warmup_cache_conversion_ns": prepared["cache_conversion_ns"],
    }


def _percentile(values: list[int | float], fraction: float) -> float | None:
    if not values:
        return None
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("percentile fraction is invalid")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _run_device_decode(
    prepared: dict[str, Any],
    callable_forward: Callable[..., Any],
    sampler: Callable[[Any], Any],
    tokenizer: Any,
    mx: Any,
    *,
    readback_interval: int,
) -> dict[str, Any]:
    """Run the common measured decode, varying only the host boundary size."""

    if readback_interval not in INTERVALS:
        raise WorkerError("readback interval is not preregistered")
    state = prepared["fixed_state"]
    physical_tokens: list[int] = []
    device_pending: list[Any] = []
    readback_records: list[dict[str, Any]] = []
    host_available_ns_by_physical_token: list[int] = []
    forward_submit_ns: list[int] = []
    physical_forwards = 0
    current_device: Any | None = None
    eos_set = set(EXPECTED_EOS_TOKEN_IDS)

    # Primary timing begins before the first post-prefill sampling action and
    # stops before any BudgetGuard charge or required break.
    decode_started_ns = time.perf_counter_ns()
    current_device = _selected_device_token(prepared["prefill_logits"], sampler)
    _schedule_device_dependencies(current_device, state, mx)
    device_pending.append(current_device)
    stop = False
    while not stop and len(physical_tokens) < MAX_PHYSICAL_TOKENS:
        block_started_ns = time.perf_counter_ns()
        block_capacity = min(
            readback_interval,
            MAX_PHYSICAL_TOKENS - len(physical_tokens),
        )
        while len(device_pending) < block_capacity:
            if current_device is None:
                raise WorkerError("device token state disappeared")
            current_device, state, submit_ns = _compiled_step(
                callable_forward,
                current_device,
                state,
                sampler,
                mx,
            )
            forward_submit_ns.append(submit_ns)
            physical_forwards += 1
            device_pending.append(current_device)

        host_tokens, transfer = _host_readback_boundary(device_pending, state, mx)
        host_available_ns = time.perf_counter_ns() - decode_started_ns
        physical_start = len(physical_tokens)
        physical_tokens.extend(host_tokens)
        host_available_ns_by_physical_token.extend(
            [host_available_ns] * len(host_tokens)
        )
        eos_in_block = next(
            (
                offset
                for offset, token_id in enumerate(host_tokens)
                if token_id in eos_set
            ),
            None,
        )
        record = {
            "boundary_index": len(readback_records),
            "physical_start_index": physical_start,
            "physical_end_index": len(physical_tokens) - 1,
            "block_size": len(host_tokens),
            "block_latency_ns": time.perf_counter_ns() - block_started_ns,
            "host_available_ns": host_available_ns,
            "eos_offset_in_block": eos_in_block,
            **transfer,
        }
        readback_records.append(record)
        stop = eos_in_block is not None or len(physical_tokens) >= MAX_PHYSICAL_TOKENS
        device_pending = []

    normalized = normalize_tokens(
        {
            "tokens": physical_tokens,
            "eos_ids": list(EXPECTED_EOS_TOKEN_IDS),
            "max_tokens": MAX_PHYSICAL_TOKENS,
        }
    )
    try:
        visible_text = tokenizer.decode(normalized["visible_tokens"])
    except Exception as exc:
        raise WorkerError("tokenizer could not decode visible tokens") from exc
    if not isinstance(visible_text, str):
        raise WorkerError("tokenizer returned non-text visible output")
    text_bytes = visible_text.encode("utf-8")
    if len(text_bytes) > MAX_RESPONSE_BYTES:
        raise ResourceFailure("visible model text exceeds evidence cap")

    eos_position = normalized["eos_position"]
    eos_block = None
    if eos_position is not None:
        eos_block = next(
            record["boundary_index"]
            for record in readback_records
            if record["physical_start_index"]
            <= eos_position
            <= record["physical_end_index"]
        )
    # Explicitly discard the speculative tail state.  This study makes no
    # cache-continuation or multi-turn claim.
    state = None
    cache_discarded = state is None
    decode_finished_ns = time.perf_counter_ns()
    decode_critical_path_ns = decode_finished_ns - decode_started_ns
    if decode_critical_path_ns <= 0:
        raise WorkerError("decode critical-path duration is not positive")
    if physical_forwards != len(physical_tokens) - 1:
        raise WorkerError("physical forward count does not match sampled tokens")
    if len(host_available_ns_by_physical_token) != len(physical_tokens):
        raise WorkerError("host availability count does not match physical tokens")

    boundary_times = [record["host_available_ns"] for record in readback_records]
    boundary_interarrival_ns = [
        later - earlier for earlier, later in zip(boundary_times, boundary_times[1:])
    ]
    readback_ns = [record["readback_ns"] for record in readback_records]
    block_latency_ns = [record["block_latency_ns"] for record in readback_records]
    first_host_ns = boundary_times[0] if boundary_times else None
    result = {
        "readback_interval": readback_interval,
        "max_physical_tokens": MAX_PHYSICAL_TOKENS,
        "eos_token_ids": list(EXPECTED_EOS_TOKEN_IDS),
        **normalized,
        "eos_block": eos_block,
        "eos_readback_block": eos_block,
        "cache_discarded": cache_discarded,
        "visible_text": visible_text,
        "text_sha256": _sha256_bytes(text_bytes),
        "text_utf8_sha256": _sha256_bytes(text_bytes),
        "token_sha256": normalized["logical_token_sha256"],
        "decode_critical_path_ns": decode_critical_path_ns,
        "physical_forwards": physical_forwards,
        "decode_forwards": physical_forwards,
        "forward_submit_ns": forward_submit_ns,
        "readback_count": len(readback_records),
        "readback_block_sizes": [record["block_size"] for record in readback_records],
        "readback_records": readback_records,
        "readback_boundaries": readback_records,
        "readback_ns": readback_ns,
        "readback_total_ns": sum(readback_ns),
        "readback_measurement_scope": (
            "pending_device_eval_plus_sync_plus_single_vector_host_conversion"
        ),
        "host_transfer_api_call_count": sum(
            record["host_transfer_api_calls"] for record in readback_records
        ),
        "host_transfer_physical_dma_count": None,
        "host_transfer_method": "single_vector_tolist",
        "host_boundary_available": True,
        "vector_block_readback_supported": all(
            record["vector_block_readback_supported"]
            for record in readback_records
        ),
        "host_available_ns_by_physical_token": host_available_ns_by_physical_token,
        "host_boundary_available_ns": boundary_times,
        "host_available_total_ns": boundary_times[-1] if boundary_times else None,
        "first_host_token_ns": first_host_ns,
        "block_latency_ns": block_latency_ns,
        "boundary_interarrival_ns": boundary_interarrival_ns,
        "boundary_interarrival_p50_ns": _percentile(boundary_interarrival_ns, 0.50),
        "boundary_interarrival_p95_ns": _percentile(boundary_interarrival_ns, 0.95),
        "boundary_interarrival_p99_ns": _percentile(boundary_interarrival_ns, 0.99),
        "stop_decision_ns": (
            decode_critical_path_ns - boundary_times[-1] if boundary_times else None
        ),
        "token_rate": len(physical_tokens) / (decode_critical_path_ns / 1e9),
        "ttft_ns": (
            prepared["preparation_total_ns"] + first_host_ns
            if first_host_ns is not None
            else None
        ),
        "prefill_ns": prepared["prefill_ns"],
        "cache_conversion_ns": prepared["cache_conversion_ns"],
    }
    return result


def _run_arm(
    model: Any,
    tokenizer: Any,
    prompt_ids: list[int],
    callable_forward: Callable[..., Any],
    sampler: Callable[[Any], Any],
    mx: Any,
    arm_name: str,
    readback_interval: int,
) -> dict[str, Any]:
    if arm_name not in ARM_NAMES or ARM_INTERVALS[arm_name] != readback_interval:
        raise WorkerError("arm/readback binding is invalid")
    warmup = _run_warmup(
        model,
        prompt_ids,
        callable_forward,
        sampler,
        mx,
    )
    prepared = _prepare_prefill(model, prompt_ids, mx)
    measured = _run_device_decode(
        prepared,
        callable_forward,
        sampler,
        tokenizer,
        mx,
        readback_interval=readback_interval,
    )
    measured.update(
        {
            "arm": arm_name,
            "cache_capacity": CACHE_CAPACITY,
            "fixed_cache": FIXED_CACHE,
            "fixed_compile": FIXED_COMPILE,
            "compile_config": dict(COMPILE_CONFIG),
            "compile_callable_shared": True,
            "greedy": True,
            "sampler_temperature": 0.0,
            **warmup,
        }
    )
    return measured


def _parse_arm_order() -> tuple[str, str]:
    raw = os.environ.get(f"{AUTH_ENV_PREFIX}ARM_ORDER")
    if not isinstance(raw, str) or not raw:
        raise WorkerError("parent did not bind an arm order")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise WorkerError("parent arm order is not JSON") from exc
    order = tuple(value) if isinstance(value, list) else ()
    if order not in ARM_ORDERS:
        raise WorkerError("arm order is not one of the registered pair orders")
    return order  # type: ignore[return-value]


def _snapshot_stat_manifest(snapshot: Any) -> tuple[str, dict[str, dict[str, Any]]]:
    root = Path(snapshot.path).resolve(strict=True)
    try:
        repository = root.parent.parent.resolve(strict=True)
        root.relative_to(repository)
    except (OSError, ValueError) as exc:
        raise WorkerError("snapshot is outside its local repository") from exc
    if root.parent.name != "snapshots":
        raise WorkerError("snapshot directory layout is unexpected")

    required = ["config.json", "tokenizer_config.json", "generation_config.json"]
    for name in ("tokenizer.json", "tokenizer.model"):
        if (root / name).is_file() or (root / name).is_symlink():
            required.append(name)
            break
    required.extend(snapshot.weight_files)
    manifest: dict[str, dict[str, Any]] = {}
    for relative in dict.fromkeys(required):
        resolved = (root / relative).resolve(strict=True)
        try:
            resolved.relative_to(repository)
        except ValueError as exc:
            raise WorkerError("snapshot execution file escaped repository") from exc
        if not resolved.is_file():
            raise WorkerError(f"snapshot execution file is missing: {relative}")
        metadata = resolved.stat()
        manifest[relative] = {
            "dev": int(metadata.st_dev),
            "inode": int(metadata.st_ino),
            "mtime_ns": int(metadata.st_mtime_ns),
            "path": str(resolved),
            "size": int(metadata.st_size),
        }
    return str(root), manifest


def _verify_snapshot_binding(
    snapshot: Any, spec: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    values = {
        "path": os.environ.get(f"{AUTH_ENV_PREFIX}SNAPSHOT_PATH"),
        "revision": os.environ.get(f"{AUTH_ENV_PREFIX}SNAPSHOT_REVISION"),
        "snapshot_sha256": os.environ.get(f"{AUTH_ENV_PREFIX}SNAPSHOT_SHA256"),
        "weight_sha256": os.environ.get(f"{AUTH_ENV_PREFIX}WEIGHT_SHA256"),
        "stat_manifest": os.environ.get(f"{AUTH_ENV_PREFIX}SNAPSHOT_STAT_MANIFEST"),
    }
    if not all(isinstance(value, str) and value for value in values.values()):
        raise WorkerError("parent snapshot binding is incomplete")
    try:
        weight_hashes = json.loads(str(values["weight_sha256"]))
        expected_stats = json.loads(str(values["stat_manifest"]))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise WorkerError("parent snapshot binding is invalid JSON") from exc
    resolved_path, actual_stats = _snapshot_stat_manifest(snapshot)
    if (
        snapshot.revision != spec["revision"]
        or values["revision"] != spec["revision"]
        or values["path"] != resolved_path
        or values["snapshot_sha256"] != EXPECTED_SNAPSHOT_SHA256
        or actual_stats != expected_stats
        or not isinstance(weight_hashes, dict)
        or set(weight_hashes.values()) != {EXPECTED_WEIGHT_SHA256}
    ):
        raise WorkerError("snapshot revision/content/stat binding changed")
    return resolved_path, {
        "snapshot_sha256": values["snapshot_sha256"],
        "weight_sha256": weight_hashes,
        "stat_manifest": expected_stats,
    }


def _authorise(model_key: str) -> None:
    if os.environ.get(f"{AUTH_ENV_PREFIX}PARENT_PID") != str(os.getppid()):
        raise WorkerError("unauthorised worker parent PID")
    if os.environ.get(f"{AUTH_ENV_PREFIX}RUN_ID") != RUN_ID:
        raise WorkerError("unauthorised worker run ID")
    if os.environ.get(f"{AUTH_ENV_PREFIX}MODEL_KEY") != model_key or model_key != MODEL_KEY:
        raise WorkerError("unauthorised worker model key")
    if os.environ.get(f"{AUTH_ENV_PREFIX}NONCE") != AUTH_NONCE:
        raise WorkerError("unauthorised worker nonce")
    try:
        process_index = int(os.environ.get(f"{AUTH_ENV_PREFIX}BLOCK", ""))
    except ValueError as exc:
        raise WorkerError("unauthorised worker process index") from exc
    if not 1 <= process_index <= 6:
        raise WorkerError("unauthorised worker process index")
    for name, expected in OFFLINE_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise WorkerError(f"offline environment gate missing: {name}")
    if os.environ.get(f"{AUTH_ENV_PREFIX}PROTOCOL_VERSION") != str(PROTOCOL_VERSION):
        raise WorkerError("worker protocol version is not authorised")
    if os.environ.get(f"{AUTH_ENV_PREFIX}PROTOCOL_SHA256") != PROTOCOL_SHA256:
        raise WorkerError("worker protocol hash is not authorised")
    if (
        os.environ.get(f"{AUTH_ENV_PREFIX}PREREG_SHA256")
        != FROZEN_PREREGISTRATION_SHA256
        or _sha256_file(PREREGISTRATION_PATH) != FROZEN_PREREGISTRATION_SHA256
    ):
        raise WorkerError("preregistration binding changed")
    if os.environ.get(f"{AUTH_ENV_PREFIX}PROMPT_SHA256") != EXPECTED_PROMPT_SHA256:
        raise WorkerError("raw prompt binding changed")
    actual_environment = environment_fingerprint()
    if os.environ.get(f"{AUTH_ENV_PREFIX}ENVIRONMENT_SHA256") != actual_environment:
        raise WorkerError("worker environment fingerprint changed")
    raw_fingerprints = os.environ.get(f"{AUTH_ENV_PREFIX}CODE_FINGERPRINTS")
    try:
        expected_fingerprints = parse_one_json(raw_fingerprints or "")
    except ValueError as exc:
        raise WorkerError("worker code fingerprint binding is invalid") from exc
    actual_fingerprints = code_fingerprints()
    if expected_fingerprints != actual_fingerprints:
        raise WorkerError("worker code fingerprint binding changed")
    if (
        os.environ.get(f"{AUTH_ENV_PREFIX}CODE_SHA256")
        != _sha256_bytes(_canonical_json(actual_fingerprints))
    ):
        raise WorkerError("worker code aggregate hash changed")


def _read_eos_ids(snapshot_path: str) -> tuple[int, ...]:
    path = Path(snapshot_path) / "generation_config.json"
    try:
        payload = path.read_bytes()

        def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            document: dict[str, Any] = {}
            for key, item in pairs:
                if key in document:
                    raise ValueError("duplicate generation configuration key")
                document[key] = item
            return document

        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=unique,
            parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
        )
        if not isinstance(value, dict):
            raise ValueError("generation configuration is not an object")
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise WorkerError("generation EOS configuration is unavailable") from exc
    raw = value.get("eos_token_id")
    if type(raw) is int:
        ids = (raw,)
    elif isinstance(raw, list) and all(type(item) is int for item in raw):
        ids = tuple(raw)
    else:
        raise WorkerError("generation EOS configuration is invalid")
    if ids != EXPECTED_EOS_TOKEN_IDS:
        raise WorkerError("generation EOS IDs changed")
    return ids


def _swap_used_bytes() -> int:
    try:
        import psutil

        value = psutil.swap_memory().used
    except Exception as exc:
        raise ResourceFailure("swap usage is unavailable") from exc
    if type(value) is not int or value < 0:
        raise ResourceFailure("swap usage is invalid")
    return value


def _resource_snapshot(mx: Any, swap_before: int) -> dict[str, int]:
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    mlx_peak = int(mx.get_peak_memory())
    swap_after = _swap_used_bytes()
    if rss > MAX_RSS_BYTES:
        raise ResourceFailure("RSS resource limit exceeded")
    if mlx_peak > MAX_MLX_BYTES:
        raise ResourceFailure("MLX resource limit exceeded")
    if swap_after != swap_before:
        raise ResourceFailure("swap delta is nonzero")
    return {
        "rss_peak_bytes": rss,
        "mlx_peak_bytes": mlx_peak,
        "swap_before_bytes": swap_before,
        "swap_after_bytes": swap_after,
        "swap_delta_bytes": swap_after - swap_before,
    }


def _resource_evidence(mx: Any, swap_before: int) -> dict[str, int]:
    try:
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        rss = 0
    try:
        mlx_peak = int(mx.get_peak_memory())
    except Exception:
        mlx_peak = 0
    try:
        swap_after = _swap_used_bytes()
    except Exception:
        swap_after = swap_before
    return {
        "rss_peak_bytes": max(0, rss),
        "mlx_peak_bytes": max(0, mlx_peak),
        "swap_before_bytes": swap_before,
        "swap_after_bytes": max(0, int(swap_after)),
        "swap_delta_bytes": int(swap_after) - swap_before,
    }


class _ChargeRejected(ResourceFailure):
    def __init__(self, evidence: dict[str, Any], cause: BaseException):
        super().__init__(str(cause))
        self.evidence = evidence
        self.cause = cause


def _arm_budget_evidence(
    observed_ns: int,
    *,
    guard_before_seconds: float = 0.0,
    guard_after_seconds: float = 0.0,
    accepted: bool = False,
) -> dict[str, Any]:
    seconds = observed_ns / 1e9
    recorded_ns = max(
        0,
        round((guard_after_seconds - guard_before_seconds) * 1e9),
    )
    duty_break = seconds * (1.0 - 0.15) / 0.15
    return {
        "observed_model_work_ns": observed_ns,
        # A guard can reject after its internal counter has advanced.  Preserve
        # that observed charge instead of incorrectly rewriting it to zero.
        "charged_model_work_ns": observed_ns if accepted else recorded_ns,
        "charge_accepted": accepted,
        "guard_gpu_work_before_seconds": guard_before_seconds,
        "guard_gpu_work_after_seconds": guard_after_seconds,
        "guard_recorded_model_work_ns": recorded_ns,
        "duty_formula_break_seconds": duty_break,
        "required_break_blocks": max(
            MIN_REQUIRED_BREAK_BLOCKS,
            math.ceil(duty_break / REQUIRED_BREAK_SECONDS),
        ),
    }


def _charge_arm(guard: Any, stopped_arm_ns: int) -> dict[str, Any]:
    """Record already-stopped work; never include Guard sleep in arm timing."""

    seconds = stopped_arm_ns / 1e9
    if not math.isfinite(seconds) or seconds <= 0:
        raise ResourceFailure("stopped arm duration is invalid")
    before = float(guard.gpu_work_seconds)
    try:
        guard.record_gpu(seconds)
    except Exception as exc:
        after = float(guard.gpu_work_seconds)
        raise _ChargeRejected(
            _arm_budget_evidence(
                stopped_arm_ns,
                guard_before_seconds=before,
                guard_after_seconds=after,
                accepted=False,
            ),
            exc,
        ) from exc
    after = float(guard.gpu_work_seconds)
    return _arm_budget_evidence(
        stopped_arm_ns,
        guard_before_seconds=before,
        guard_after_seconds=after,
        accepted=True,
    )


def _pause_arm(guard: Any, evidence: dict[str, Any]) -> None:
    for _ in range(int(evidence["required_break_blocks"])):
        guard.required_break()


def _correctness(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    complete = set(arms) == set(ARM_NAMES)
    logical_equal = complete and len(
        {arms[name]["logical_token_sha256"] for name in ARM_NAMES}
    ) == 1
    visible_equal = complete and len(
        {arms[name]["visible_token_sha256"] for name in ARM_NAMES}
    ) == 1
    text_equal = complete and len({arms[name]["text_sha256"] for name in ARM_NAMES}) == 1
    prompt_equal = complete and len(
        {arms[name]["prompt_sha256"] for name in ARM_NAMES}
    ) == 1
    no_eos = complete and all(not arms[name]["eos_found"] for name in ARM_NAMES)
    physical_equal_if_required = not no_eos or len(
        {arms[name]["physical_token_sha256"] for name in ARM_NAMES}
    ) == 1
    first_mismatch = None
    if complete and not (logical_equal and visible_equal and text_equal):
        left = arms[ARM_NAMES[0]]["logical_tokens"]
        right = arms[ARM_NAMES[1]]["logical_tokens"]
        index = next(
            (i for i, pair in enumerate(zip(left, right)) if pair[0] != pair[1]),
            min(len(left), len(right)),
        )
        first_mismatch = {"left_arm": ARM_NAMES[0], "right_arm": ARM_NAMES[1], "token_index": index}
    return {
        "logical_tokens_equal": logical_equal,
        "visible_tokens_equal": visible_equal,
        "visible_text_equal": text_equal,
        "prompt_identity_equal": prompt_equal,
        "physical_tokens_equal_when_no_eos": physical_equal_if_required,
        "first_mismatch": first_mismatch,
        "pass": bool(
            complete
            and logical_equal
            and visible_equal
            and text_equal
            and prompt_equal
            and physical_equal_if_required
        ),
    }


def _run_worker(model_key: str) -> int:
    process_started_ns = time.perf_counter_ns()
    _authorise(model_key)
    arm_order = _parse_arm_order()
    bound_code_fingerprints = code_fingerprints()
    bound_code_sha256 = _sha256_bytes(_canonical_json(bound_code_fingerprints))
    bound_environment_sha256 = environment_fingerprint()

    # Hardware libraries are reachable only after every authorisation and
    # offline-environment gate above has passed.
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))
    from _bench import require_ac_power, resolve_local_model_snapshot
    from friday_evidence.budget import BudgetError, BudgetGuard as EvidenceBudgetGuard
    from friday_evidence.registry import BudgetPolicy
    from mlx_lm import load
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx

    power_source = require_ac_power()
    policy = BudgetPolicy(
        gpu_work_limit_s=TOTAL_GPU_LIMIT_SECONDS,
        continuous_gpu_limit_s=CONTINUOUS_GPU_LIMIT_SECONDS,
        required_break_s=REQUIRED_BREAK_SECONDS,
        duty_window_s=60.0,
        duty_cycle_limit=0.15,
        wall_limit_s=MAX_WALL_SECONDS,
        candidate_cooldown_s=0.0,
    )
    guard = EvidenceBudgetGuard(policy)
    swap_before = _swap_used_bytes()
    spec = {"model_id": MODEL_ID, "revision": MODEL_REVISION}
    snapshot = resolve_local_model_snapshot(MODEL_ID)
    snapshot_path, binding = _verify_snapshot_binding(snapshot, spec)
    eos_ids = _read_eos_ids(snapshot_path)
    mx.reset_peak_memory()
    load_count = 0
    load_started_ns = time.perf_counter_ns()
    model, tokenizer = load(str(snapshot.path))
    load_count += 1
    load_finished_ns = time.perf_counter_ns()
    if load_count != 1:
        raise WorkerError("model load_count is not exactly one")
    after_path, after_stats = _snapshot_stat_manifest(snapshot)
    if after_path != snapshot_path or after_stats != binding["stat_manifest"]:
        raise WorkerError("snapshot changed during the single model load")
    if str(mx.default_device()) != "Device(gpu, 0)":
        raise ResourceFailure("MLX default device is not the registered GPU")

    prompt_ids, rendered_prompt_bytes = _prompt_ids(tokenizer)
    prompt_token_sha256 = _sha256_bytes(_canonical_json(prompt_ids))
    rendered_prompt_sha256 = _sha256_bytes(rendered_prompt_bytes)
    if (
        PROMPT_SHA256 != EXPECTED_PROMPT_SHA256
        or len(prompt_ids) != EXPECTED_PROMPT_TOKENS
        or prompt_token_sha256 != EXPECTED_PROMPT_TOKEN_SHA256
        or rendered_prompt_sha256 != EXPECTED_RENDERED_PROMPT_SHA256
        or eos_ids != EXPECTED_EOS_TOKEN_IDS
    ):
        raise WorkerError("prompt/tokenizer/EOS identity gate failed")
    if len(prompt_ids) + MAX_DECODE_FORWARDS > CACHE_CAPACITY:
        raise WorkerError("prompt plus physical decode exceeds fixed cache")
    sampler = make_sampler(temp=0.0)
    compile_started_ns = time.perf_counter_ns()
    callable_forward = _make_compiled_forward(model, mx)
    compile_wrapper_ns = time.perf_counter_ns() - compile_started_ns

    arms: dict[str, dict[str, Any]] = {}
    arm_budget: dict[str, dict[str, Any]] = {}
    arm_resources: dict[str, dict[str, int]] = {}
    status = "complete"
    error: dict[str, str] | None = None
    observed_model_work_ns = 0
    charged_model_work_ns = 0
    guard_recorded_model_work_ns = 0

    for arm_name in arm_order:
        arm_started_ns = time.perf_counter_ns()
        arm_value: dict[str, Any] | None = None
        arm_exception: BaseException | None = None
        try:
            arm_value = _run_arm(
                model,
                tokenizer,
                prompt_ids,
                callable_forward,
                sampler,
                mx,
                arm_name,
                ARM_INTERVALS[arm_name],
            )
        except BaseException as exc:
            arm_exception = exc

        # Timer stop precedes charge -> resource snapshot -> required break.
        arm_finished_ns = time.perf_counter_ns()
        arm_ns = arm_finished_ns - arm_started_ns
        if arm_ns <= 0:
            arm_exception = ResourceFailure("arm timer did not advance")
            arm_ns = max(1, arm_ns)
        observed_model_work_ns += arm_ns
        arm_budget[arm_name] = _arm_budget_evidence(arm_ns)
        try:
            arm_budget[arm_name] = _charge_arm(guard, arm_ns)
            charged_model_work_ns += arm_ns
            guard_recorded_model_work_ns += arm_budget[arm_name][
                "guard_recorded_model_work_ns"
            ]
            arm_resources[arm_name] = _resource_snapshot(mx, swap_before)
            _pause_arm(guard, arm_budget[arm_name])
        except _ChargeRejected as exc:
            arm_budget[arm_name] = exc.evidence
            charged_model_work_ns += exc.evidence["charged_model_work_ns"]
            guard_recorded_model_work_ns += exc.evidence[
                "guard_recorded_model_work_ns"
            ]
            arm_resources[arm_name] = _resource_evidence(mx, swap_before)
            status = "resource_or_budget_failed"
            error = {"type": type(exc.cause).__name__, "message": str(exc.cause)[:500]}
            break
        except (BudgetError, ResourceFailure, WorkerError) as exc:
            arm_resources[arm_name] = _resource_evidence(mx, swap_before)
            status = "resource_or_budget_failed"
            error = {"type": type(exc).__name__, "message": str(exc)[:500]}
            break

        if arm_exception is not None:
            if isinstance(arm_exception, CandidateNotRunnable):
                status = "candidate_not_runnable"
            elif _resource_failure(arm_exception):
                status = "resource_or_budget_failed"
            else:
                # A non-resource failure while constructing, compiling or
                # synchronising this exact candidate means that the candidate
                # API is not runnable in the sealed scope.  It is not retried
                # or reinterpreted as a quality result.
                status = "candidate_not_runnable"
            error = {
                "type": type(arm_exception).__name__,
                "message": str(arm_exception)[:500],
            }
            break
        if arm_value is None:
            status = "candidate_not_runnable"
            error = {"type": "WorkerError", "message": "arm returned no evidence"}
            break

        arm_value.update(
            {
                "arm_wall_ns": arm_ns,
                "observed_model_work_ns": arm_ns,
                "charged_model_work_ns": arm_ns,
                "charge_accepted": True,
                "prompt_sha256": PROMPT_SHA256,
                "prompt_token_sha256": prompt_token_sha256,
                "rendered_prompt_sha256": rendered_prompt_sha256,
                "compile_wrapper_ns": compile_wrapper_ns,
                "compile_cold_ns": (
                    arm_value["first_warmup_materialization_ns"]
                    if not arms
                    else None
                ),
                "budget_summary": guard.summary(),
                "resource_snapshot": arm_resources[arm_name],
            }
        )
        arms[arm_name] = arm_value

    correctness = _correctness(arms)
    if status == "complete" and not correctness["pass"]:
        status = "correctness_failed"
        error = {
            "type": "CorrectnessError",
            "message": "logical tokens, visible tokens/text, or prompt identity differ",
        }

    # Every successful warmup, prefill and decode boundary has already been
    # synchronised inside its charged arm interval.  Do not add an uncharged
    # post-arm synchronisation here.
    try:
        swap_after: int | None = _swap_used_bytes()
    except ResourceFailure as exc:
        swap_after = None
        status = "resource_or_budget_failed"
        error = {"type": type(exc).__name__, "message": str(exc)[:500]}
    if swap_after is not None and swap_after != swap_before:
        status = "resource_or_budget_failed"
        error = {"type": "ResourceFailure", "message": "swap delta is nonzero"}

    try:
        process_index = int(os.environ.get(f"{AUTH_ENV_PREFIX}BLOCK", ""))
    except ValueError as exc:
        raise WorkerError("parent process index is invalid") from exc
    if not 1 <= process_index <= 6:
        raise WorkerError("parent process index is outside the registered schedule")

    event = {
        "event": "complete",
        "status": status,
        "study_id": STUDY_ID,
        "run_id": RUN_ID,
        "candidate_id": CANDIDATE_ID,
        "formal_claim": False,
        "protocol_version": PROTOCOL_VERSION,
        "process_index": process_index,
        "arm_order": list(arm_order),
        "arms": arms,
        "arm_budget": arm_budget,
        "arm_resources": arm_resources,
        "correctness": correctness,
        "error": error,
        "pid": os.getpid(),
        "load_count": load_count,
        "model_key": model_key,
        "model_id": MODEL_ID,
        "snapshot_revision": MODEL_REVISION,
        "snapshot_path": snapshot_path,
        "snapshot_sha256": binding["snapshot_sha256"],
        "weight_sha256": binding["weight_sha256"],
        "snapshot_integrity": {
            "before_load_stat_manifest": binding["stat_manifest"],
            "after_load_stat_manifest": after_stats,
            "bound_snapshot_sha256": binding["snapshot_sha256"],
            "bound_weight_sha256": binding["weight_sha256"],
        },
        "model_load_ns": load_finished_ns - load_started_ns,
        "compile_wrapper_ns": compile_wrapper_ns,
        "compile_cold_ns": (
            arms.get(arm_order[0], {}).get("first_warmup_materialization_ns")
        ),
        "cache_capacity": CACHE_CAPACITY,
        "max_physical_tokens": MAX_PHYSICAL_TOKENS,
        "warmup_forwards_per_arm": WARMUP_FORWARDS,
        "prompt_sha256": PROMPT_SHA256,
        "prompt_token_ids": prompt_ids,
        "prompt_token_sha256": prompt_token_sha256,
        "rendered_prompt_b64": base64.b64encode(rendered_prompt_bytes).decode("ascii"),
        "rendered_prompt_sha256": rendered_prompt_sha256,
        "prompt_tokens": len(prompt_ids),
        "eos_token_ids": list(eos_ids),
        "sampler_temperature": 0.0,
        "greedy": True,
        "device": str(mx.default_device()),
        "power_source": power_source,
        "model_work_ns": observed_model_work_ns,
        "observed_model_work_ns": observed_model_work_ns,
        "charged_model_work_ns": charged_model_work_ns,
        "guard_recorded_model_work_ns": guard_recorded_model_work_ns,
        "budget": guard.summary(),
        "rss_peak_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "mlx_peak_bytes": int(mx.get_peak_memory()),
        "swap_before_bytes": swap_before,
        "swap_after_bytes": swap_after,
        "swap_delta_bytes": (
            swap_after - swap_before if swap_after is not None else None
        ),
        "worker_watchdog_seconds": CONTINUOUS_GPU_LIMIT_SECONDS,
        "host_transfer_claim": {
            "api_call_unit": "one flattened vector tolist call per boundary",
            "physical_dma_count_observable": False,
            "physical_dma_count": None,
        },
        "determinism": {
            "seed": BOOTSTRAP_SEED,
            "greedy_no_sampling_randomness": True,
            "within_arm_across_processes_checked_by_parent": True,
        },
        "preregistration_sha256": FROZEN_PREREGISTRATION_SHA256,
        "code_fingerprints": bound_code_fingerprints,
        "code_sha256": bound_code_sha256,
        "environment_sha256": bound_environment_sha256,
        "process_wall_ns": time.perf_counter_ns() - process_started_ns,
    }
    validate_event(event)
    _emit(event)
    return 0 if status == "complete" and correctness["pass"] else 1


def _self_check() -> int:
    source = Path(__file__).read_text(encoding="utf-8")
    assert PROMPT_SHA256 == EXPECTED_PROMPT_SHA256
    assert CACHE_CAPACITY == 512
    assert MAX_PHYSICAL_TOKENS == 32 and MAX_DECODE_FORWARDS == 31
    assert WARMUP_FORWARDS == 8
    assert ARM_NAMES == (
        "fixed_compiled_readback_1",
        "fixed_compiled_readback_8",
    )
    assert ARM_INTERVALS == {
        "fixed_compiled_readback_1": 1,
        "fixed_compiled_readback_8": 8,
    }
    assert AUTH_ENV_PREFIX == "FRIDAY_BRB_"
    assert AUTH_NONCE == "cycle17-fixed-compiled-batched-readback-v1"
    assert AUTH_REQUIRED_ENV_NAMES >= {
        "FRIDAY_BRB_PARENT_PID",
        "FRIDAY_BRB_RUN_ID",
        "FRIDAY_BRB_MODEL_KEY",
        "FRIDAY_BRB_NONCE",
        "FRIDAY_BRB_BLOCK",
        "FRIDAY_BRB_ARM_ORDER",
    }
    assert PROTOCOL_SHA256 == _sha256_bytes(_canonical_json(protocol_contract()))
    assert _sha256_file(PREREGISTRATION_PATH) == FROZEN_PREREGISTRATION_SHA256
    assert EXPECTED_EOS_TOKEN_IDS == (1, 106)
    assert MIN_REQUIRED_BREAK_BLOCKS == 13
    assert CONTINUOUS_GPU_LIMIT_SECONDS == 6.0
    assert TOTAL_GPU_LIMIT_SECONDS == 120.0
    assert MAX_WALL_SECONDS == 1200.0
    assert BOOTSTRAP_SEED == 20260824 and BOOTSTRAP_RESAMPLES == 10000
    assert "mx.compile(body, shapeless=False)" in source
    assert "mx.async_eval" in source
    assert "block.tolist()" in source
    scalar_host_call = "." + "item()"
    assert scalar_host_call not in source[
        source.index("def _run_device_decode") : source.index("def _self_check")
    ]
    assert "slice_update" in FixedKVCache.update_and_fetch.__code__.co_names
    assert "concatenate" not in FixedKVCache.update_and_fetch.__code__.co_names
    sample = normalize_tokens(
        {"tokens": [7, 106, 8, 9], "eos_ids": [1, 106], "max_tokens": 32}
    )
    assert sample["logical_tokens"] == [7, 106]
    assert sample["visible_tokens"] == [7]
    assert sample["overproduced_tokens"] == 2
    parse_one_json('{"event":"ok"}')
    for bad in ('{"a":1,"a":2}', '{"a":NaN}', '{}\n{}'):
        try:
            parse_one_json(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("strict JSON self-check accepted invalid input")
    print(json.dumps({"checks": 30, "self_check": "pass"}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="batched_readback_compile_worker",
        allow_abbrev=False,
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--worker", action="store_true")
    modes.add_argument("--self-check", action="store_true")
    parser.add_argument("--model-key", default=None)
    args = parser.parse_args(argv)
    if args.self_check:
        return _self_check()
    if not args.worker or args.model_key != MODEL_KEY:
        # No model import or load can occur on an unauthorised direct call.
        print(json.dumps({"error": "worker authorization failed", "event": "error"}))
        return 2
    try:
        return _run_worker(args.model_key)
    except BaseException as exc:
        _emit(
            {
                "error_type": type(exc).__name__,
                "event": "error",
                "message": str(exc)[:500],
                "model_key": args.model_key,
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
