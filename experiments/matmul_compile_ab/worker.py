#!/usr/bin/env python3
"""One fresh MLX process for the cycle-16 fixed-cache compile study.

The parent harness owns the schedule, snapshot binding, resource accounting and
the result file.  This module deliberately has no MLX import at module import
time: an unauthorised direct invocation and ``--self-check`` are offline-only.
The fixed cache is a functional, fixed-shape state tree.  All layers share one
tensor offset; a complete model forward advances that offset exactly once.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "matmul-compile-ab-20260824-01"
RUN_ID = "matmul-compile-validation-20260824-01"
MODEL_KEY = "4b"
MODEL_ID = "mlx-community/gemma-3-4b-it-4bit"
MODEL_REVISION = "93724907d4ed1745d2fe50baadf3b0b01a65abf2"

CAPACITY = 512
EXPECTED_PROMPT_TOKENS = 322
OUTPUT_TOKENS = 32
DECODE_FORWARDS = OUTPUT_TOKENS - 1
WARMUP_FORWARDS = 8
MAX_RESPONSE_BYTES = 16_384
MAX_EVENT_BYTES = 1_000_000
CONTINUOUS_GPU_LIMIT_SECONDS = 6.0
MIN_REQUIRED_BREAK_BLOCKS = 13
PREFILL_STEP_SIZE = 256
MAX_RSS_BYTES = 6 * 1024**3
MAX_MLX_BYTES = 5 * 1024**3

ARM_NAMES = ("standard_eager", "fixed_eager", "fixed_compiled")

# This is byte-for-byte the cycle-14/15 prompt.  Do not add a newline or
# interpolate values into it.  The raw prompt hash is a preregistered gate.
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


class WorkerError(RuntimeError):
    """The closed worker protocol cannot continue safely."""


class CandidateNotRunnable(WorkerError):
    """The local fixed-state/compile API could not run correctly."""


def _resource_failure(exc: BaseException) -> bool:
    if isinstance(exc, (MemoryError, OSError)):
        return True
    text = str(exc).lower()
    return any(marker in text for marker in ("out of memory", "oom", "killed", "abort", "resource exhausted"))


def _candidate_conversion_failure(exc: BaseException) -> None:
    """Classify the complete lazy standard-cache -> fixed-cache conversion."""
    if _resource_failure(exc):
        raise WorkerError(
            f"fixed-cache conversion resource failure: {type(exc).__name__}: {exc}"
        ) from exc
    raise CandidateNotRunnable(
        f"fixed-cache conversion failed: {type(exc).__name__}: {exc}"
    ) from exc


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


def _emit(value: dict[str, Any]) -> None:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    if len(payload.encode("utf-8")) > MAX_EVENT_BYTES:
        raise WorkerError("worker event exceeds the registered output limit")
    sys.stdout.write(payload + "\n")
    sys.stdout.flush()


class FixedKVCache:
    """A fixed-shape layer cache backed by one shared tensor position.

    ``state`` is a dict containing only the fixed ``keys`` and ``values``
    arrays for one layer.  ``position`` is shared by every layer and contains a
    scalar MLX array at ``position["offset"]``.  A layer writes at that offset,
    but does not increment it; the outer fixed forward increments it once
    after every complete model call.
    """

    def __init__(self, state: dict[str, Any], position: dict[str, Any], mx: Any):
        if set(state) != {"keys", "values"}:
            raise CandidateNotRunnable("fixed layer state keys are not closed")
        if set(position) != {"offset"}:
            raise CandidateNotRunnable("fixed position state is not closed")
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
        if keys.shape[2] < 1 or keys.shape[2] > CAPACITY:
            raise CandidateNotRunnable("fixed cache received an invalid step shape")
        # Dynamic start indices are tensor values.  No Python slice or growing
        # allocation occurs on the decode path.
        zero = self._mx.array(0, dtype=self.offset.dtype)
        starts = self._mx.stack((zero, zero, self.offset, zero))
        self._state["keys"] = self._mx.slice_update(
            self._state["keys"], keys, start_indices=starts, axes=(0, 1, 2, 3)
        )
        self._state["values"] = self._mx.slice_update(
            self._state["values"], values, start_indices=starts, axes=(0, 1, 2, 3)
        )
        # The shared offset is intentionally not changed here.  The outer
        # fixed forward changes it once after all layers have run.
        return self._state["keys"], self._state["values"]

    def make_mask(
        self,
        n_tokens: int,
        *,
        window_size: int | None = None,
        return_array: bool = False,
    ) -> Any:
        del return_array
        if not isinstance(n_tokens, int) or n_tokens < 1:
            raise CandidateNotRunnable("fixed mask received invalid token count")
        positions = self._mx.arange(CAPACITY, dtype=self.offset.dtype)
        query_positions = self.offset + self._mx.arange(
            n_tokens, dtype=self.offset.dtype
        )
        mask = positions[None, :] <= query_positions[:, None]
        mask = mask & (positions[None, :] < (self.offset + n_tokens))
        if window_size is not None:
            mask = mask & (
                positions[None, :] >= query_positions[:, None] - window_size + 1
            )
        # MLX attention broadcasts this [1, 1, query, fixed-key] tensor over
        # batch and heads.  Padded positions are therefore never attended.
        return mask[None, None, :, :]


def _fixed_state_from_standard_cache(
    cache: list[Any], prompt_tokens: int, mx: Any
) -> dict[str, Any]:
    """Pad a standard prefill cache to 512 without changing existing values."""

    if not cache or not isinstance(prompt_tokens, int) or prompt_tokens < 1:
        raise CandidateNotRunnable("standard prefill cache is empty")
    layers: list[dict[str, Any]] = []
    for layer in cache:
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if keys is None or values is None:
            raise CandidateNotRunnable("standard cache did not expose keys/values")
        if len(keys.shape) != 4 or len(values.shape) != 4:
            raise CandidateNotRunnable("standard cache tensors have invalid rank")
        if keys.shape[2] < prompt_tokens or values.shape[2] < prompt_tokens:
            raise CandidateNotRunnable("standard cache is shorter than the prompt")
        if keys.shape[2] > CAPACITY or values.shape[2] > CAPACITY:
            raise CandidateNotRunnable("standard cache exceeds fixed capacity")
        key_shape = (keys.shape[0], keys.shape[1], CAPACITY, keys.shape[3])
        value_shape = (values.shape[0], values.shape[1], CAPACITY, values.shape[3])
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
    position = {"offset": mx.array(prompt_tokens, dtype=mx.int32)}
    return {"position": position, "layers": layers}


def _fixed_caches(state_tree: dict[str, Any], mx: Any) -> list[FixedKVCache]:
    position = state_tree.get("position")
    layers = state_tree.get("layers")
    if not isinstance(position, dict) or not isinstance(layers, list):
        raise CandidateNotRunnable("fixed state tree is not a position/layers tree")
    return [FixedKVCache(layer, position, mx) for layer in layers]


def _fixed_forward(model: Any, input_ids: Any, state_tree: dict[str, Any], mx: Any):
    """Run one fixed-cache model forward and advance position exactly once."""

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


def _make_compiled_forward(model: Any, mx: Any):
    """Compile a function whose complete cache state is explicit and functional."""

    def body(input_ids: Any, state: dict[str, Any]):
        return _fixed_forward(model, input_ids, state, mx)

    try:
        return mx.compile(body, shapeless=False)
    except Exception as exc:
        if _resource_failure(exc):
            raise WorkerError(f"mx.compile resource failure: {type(exc).__name__}: {exc}") from exc
        raise CandidateNotRunnable(
            f"mx.compile could not bind fixed state tree: {type(exc).__name__}: {exc}"
        ) from exc


def _prompt_ids(tokenizer: Any) -> tuple[list[int], bytes]:
    messages = [{"role": "user", "content": PLANNER_PROMPT}]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str):
        raise WorkerError("chat template did not return rendered text")
    rendered_bytes = rendered.encode("utf-8")
    try:
        values = tokenizer.encode(rendered, add_special_tokens=False)
    except TypeError as exc:
        raise WorkerError("tokenizer.encode lacks fixed special-token option") from exc
    if not isinstance(values, list):
        try:
            values = values.tolist()
        except AttributeError as exc:
            raise WorkerError("tokenizer returned non-list prompt tokens") from exc
    if not values or any(type(value) is not int for value in values):
        raise WorkerError("tokenizer returned invalid prompt token IDs")
    return list(values), rendered_bytes


def _select_token(logits: Any, sampler: Callable[[Any], Any], mx: Any) -> tuple[int, Any]:
    selected = sampler(logits[:, -1, :])
    mx.eval(selected)
    mx.synchronize()
    try:
        token = int(selected.item())
    except (AttributeError, TypeError, ValueError) as exc:
        raise WorkerError("greedy sampler did not return one scalar token") from exc
    return token, selected


def _prepare_prefill(
    model: Any,
    prompt_ids: list[int],
    arm: str,
    tokenizer: Any,
    sampler: Callable[[Any], Any],
    mx: Any,
) -> dict[str, Any]:
    del tokenizer
    prompt_array = mx.array([prompt_ids])
    cache = model.make_cache() if hasattr(model, "make_cache") else None
    if cache is None:
        raise CandidateNotRunnable("model did not provide a standard cache")
    started_ns = time.perf_counter_ns()
    logits = model(prompt_array, cache=cache)
    mx.eval(logits)
    mx.synchronize()
    first_token, _ = _select_token(logits, sampler, mx)
    finished_ns = time.perf_counter_ns()
    fixed_state: dict[str, Any] | None = None
    conversion_ns = 0
    if arm != "standard_eager":
        conversion_started_ns = time.perf_counter_ns()
        try:
            fixed_state = _fixed_state_from_standard_cache(cache, len(prompt_ids), mx)
            # Conversion is outside the decode primary metric but must be fully
            # materialised before its duration is recorded.  MLX may defer both
            # slice_update shape errors and allocation failures until eval/sync.
            mx.eval(fixed_state["position"]["offset"])
            for layer in fixed_state["layers"]:
                mx.eval(layer["keys"], layer["values"])
            mx.synchronize()
        except Exception as exc:
            _candidate_conversion_failure(exc)
        conversion_ns = time.perf_counter_ns() - conversion_started_ns
    return {
        "cache": cache,
        "conversion_ns": conversion_ns,
        "fixed_state": fixed_state,
        "first_token": first_token,
        "prefill_ns": finished_ns - started_ns,
        "ttft_ns": finished_ns - started_ns,
    }


def _run_arm(
    model: Any,
    tokenizer: Any,
    prompt_ids: list[int],
    arm: str,
    sampler: Callable[[Any], Any],
    mx: Any,
) -> dict[str, Any]:
    if arm not in ARM_NAMES:
        raise WorkerError(f"unknown arm: {arm}")

    first_prepare = _prepare_prefill(model, prompt_ids, arm, tokenizer, sampler, mx)
    compiled = None
    compile_wrapper_ns = 0
    compile_cold_ns: int | None = None
    if arm == "fixed_compiled":
        state_tree = first_prepare["fixed_state"]
        if not isinstance(state_tree, dict):
            raise CandidateNotRunnable("compiled arm has no fixed state tree")
        compile_started_ns = time.perf_counter_ns()
        compiled = _make_compiled_forward(model, mx)
        compile_wrapper_ns = time.perf_counter_ns() - compile_started_ns

    def step(
        token: int, state: dict[str, Any] | None, cache: Any
    ) -> tuple[int, dict[str, Any] | None, int, int]:
        input_ids = mx.array([[token]])
        started_ns = time.perf_counter_ns()
        if arm == "standard_eager":
            logits = model(input_ids, cache=cache)
            next_state = None
        elif arm == "fixed_eager":
            if state is None:
                raise CandidateNotRunnable("fixed eager state disappeared")
            try:
                logits, next_state = _fixed_forward(model, input_ids, state, mx)
            except Exception as exc:
                if _resource_failure(exc):
                    raise WorkerError(
                        f"fixed eager forward resource failure: {type(exc).__name__}: {exc}"
                    ) from exc
                raise CandidateNotRunnable(
                    f"fixed eager forward failed: {type(exc).__name__}: {exc}"
                ) from exc
        else:
            if state is None or compiled is None:
                raise CandidateNotRunnable("compiled fixed state disappeared")
            try:
                result = compiled(input_ids, state)
            except Exception as exc:
                # A compile API/shape failure is a candidate failure, but OOM,
                # an explicit abort, or a killed/resource failure remains
                # fail-closed as a resource error.
                if _resource_failure(exc):
                    raise WorkerError(
                        f"compiled cold forward resource failure: {type(exc).__name__}: {exc}"
                    ) from exc
                raise CandidateNotRunnable(
                    f"compiled cold forward failed: {type(exc).__name__}: {exc}"
                ) from exc
            if not isinstance(result, tuple) or len(result) != 2:
                raise CandidateNotRunnable("compiled forward returned wrong structure")
            logits, next_state = result
        if arm in {"fixed_eager", "fixed_compiled"}:
            try:
                # MLX is lazy: the actual fixed-cache compile/shape failure
                # can surface only while materialising and synchronizing logits.
                mx.eval(logits)
                mx.synchronize()
            except Exception as exc:
                if _resource_failure(exc):
                    raise WorkerError(
                        f"fixed forward materialization resource failure: {type(exc).__name__}: {exc}"
                    ) from exc
                raise CandidateNotRunnable(
                    f"fixed forward materialization failed: {type(exc).__name__}: {exc}"
                ) from exc
        else:
            mx.eval(logits)
            mx.synchronize()
        forward_finished_ns = time.perf_counter_ns()
        next_token, _ = _select_token(logits, sampler, mx)
        intertoken_finished_ns = time.perf_counter_ns()
        return (
            next_token,
            next_state,
            forward_finished_ns - started_ns,
            intertoken_finished_ns - started_ns,
        )

    # Eight warmup forwards are charged and recorded by the parent but never
    # enter the primary metric.  The second prefill below resets every arm.
    warmup_forward_ns: list[int] = []
    warmup_intertoken_ns: list[int] = []
    warmup_token = int(first_prepare["first_token"])
    warmup_state = first_prepare["fixed_state"]
    warmup_cache = first_prepare["cache"]
    for warmup_index in range(WARMUP_FORWARDS):
        warmup_token, warmup_state, forward_ns, intertoken_ns = step(
            warmup_token, warmup_state, warmup_cache
        )
        warmup_forward_ns.append(forward_ns)
        warmup_intertoken_ns.append(intertoken_ns)
        if arm == "fixed_compiled" and compile_cold_ns is None:
            compile_cold_ns = intertoken_ns

    measurement_prepare = _prepare_prefill(
        model, prompt_ids, arm, tokenizer, sampler, mx
    )
    measured_state = measurement_prepare["fixed_state"]
    measured_cache = measurement_prepare["cache"]
    tokens = [int(measurement_prepare["first_token"])]
    decode_forward_ns: list[int] = []
    intertoken_ns: list[int] = []
    token = tokens[0]
    for _ in range(DECODE_FORWARDS):
        token, measured_state, forward_ns, intertoken_duration_ns = step(
            token, measured_state, measured_cache
        )
        decode_forward_ns.append(forward_ns)
        intertoken_ns.append(intertoken_duration_ns)
        tokens.append(token)
    if len(tokens) != OUTPUT_TOKENS:
        raise WorkerError("fixed-step generation did not produce 32 tokens")
    try:
        text = tokenizer.decode(tokens)
    except Exception as exc:
        raise WorkerError("tokenizer could not decode fixed-step tokens") from exc
    if not isinstance(text, str) or not text:
        raise WorkerError("fixed-step decoded text is empty")
    text_bytes = text.encode("utf-8")
    if len(text_bytes) > MAX_RESPONSE_BYTES:
        raise WorkerError("fixed-step decoded text is too large")
    return {
        "arm": arm,
        "cache_capacity": CAPACITY,
        "cache_conversion_ns": int(measurement_prepare["conversion_ns"]),
        "compile_cold_ns": compile_cold_ns,
        "compile_wrapper_ns": compile_wrapper_ns,
        "decode_forward_ns": decode_forward_ns,
        "decode_forward_total_ns": sum(decode_forward_ns),
        "decode_forwards": DECODE_FORWARDS,
        "finish_reason": "fixed_steps",
        "intertoken_ns": intertoken_ns,
        "intertoken_p50_ns": _percentile(intertoken_ns, 0.50),
        "intertoken_p95_ns": _percentile(intertoken_ns, 0.95),
        "intertoken_p99_ns": _percentile(intertoken_ns, 0.99),
        "model_work_ns": int(
            measurement_prepare["prefill_ns"] + sum(decode_forward_ns)
        ),
        "prefill_ns": int(measurement_prepare["prefill_ns"]),
        "text": text,
        "text_utf8_sha256": _sha256_bytes(text_bytes),
        "token_rate": DECODE_FORWARDS / (sum(decode_forward_ns) / 1_000_000_000),
        "token_sha256": _sha256_bytes(_canonical_json(tokens)),
        "tokens": tokens,
        "ttft_ns": int(measurement_prepare["ttft_ns"]),
        "warmup_decode_forward_ns": warmup_forward_ns,
        "warmup_intertoken_ns": warmup_intertoken_ns,
        "warmup_forwards": WARMUP_FORWARDS,
    }


def _percentile(values: list[int | float], fraction: float) -> float:
    if not values:
        raise WorkerError("cannot calculate percentile of an empty list")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _parse_arm_order() -> tuple[str, ...]:
    raw = os.environ.get("FRIDAY_MATMUL_ARM_ORDER")
    if not isinstance(raw, str) or not raw:
        raise WorkerError("parent did not bind an arm order")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise WorkerError("parent arm order is not JSON") from exc
    if not isinstance(value, list) or tuple(value) not in _all_permutations():
        raise WorkerError("parent arm order is not one of the six registered permutations")
    return tuple(value)


def _all_permutations() -> tuple[tuple[str, ...], ...]:
    return (
        ("standard_eager", "fixed_eager", "fixed_compiled"),
        ("standard_eager", "fixed_compiled", "fixed_eager"),
        ("fixed_eager", "standard_eager", "fixed_compiled"),
        ("fixed_eager", "fixed_compiled", "standard_eager"),
        ("fixed_compiled", "standard_eager", "fixed_eager"),
        ("fixed_compiled", "fixed_eager", "standard_eager"),
    )


def _snapshot_stat_manifest(snapshot: Any) -> tuple[str, dict[str, dict[str, Any]]]:
    root = Path(snapshot.path).resolve(strict=True)
    try:
        repository = root.parent.parent.resolve(strict=True)
        root.relative_to(repository)
    except (OSError, ValueError) as exc:
        raise WorkerError("snapshot is outside its local repository") from exc
    if root.parent.name != "snapshots":
        raise WorkerError("snapshot directory layout is unexpected")

    def execution_path(relative: str) -> Path:
        candidate = root / relative
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(repository)
        except ValueError as exc:
            raise WorkerError("snapshot execution file escaped local repository") from exc
        if not resolved.is_file():
            raise WorkerError(f"snapshot execution path is not a file: {relative}")
        return resolved

    required = ["config.json", "tokenizer_config.json"]
    for name in ("tokenizer.json", "tokenizer.model"):
        if (root / name).is_file() or (root / name).is_symlink():
            required.append(name)
            break
    required.extend(snapshot.weight_files)
    manifest: dict[str, dict[str, Any]] = {}
    for relative in dict.fromkeys(required):
        resolved = execution_path(relative)
        metadata = resolved.stat()
        manifest[relative] = {
            "dev": int(metadata.st_dev),
            "inode": int(metadata.st_ino),
            "mtime_ns": int(metadata.st_mtime_ns),
            "path": str(resolved),
            "size": int(metadata.st_size),
        }
    return str(root), manifest


def _verify_snapshot_binding(snapshot: Any, spec: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    expected_path = os.environ.get("FRIDAY_MATMUL_SNAPSHOT_PATH")
    expected_revision = os.environ.get("FRIDAY_MATMUL_SNAPSHOT_REVISION")
    expected_snapshot_sha = os.environ.get("FRIDAY_MATMUL_SNAPSHOT_SHA256")
    expected_weight_sha = os.environ.get("FRIDAY_MATMUL_WEIGHT_SHA256")
    expected_stats = os.environ.get("FRIDAY_MATMUL_SNAPSHOT_STAT_MANIFEST")
    if not all(
        isinstance(value, str) and value
        for value in (
            expected_path,
            expected_revision,
            expected_snapshot_sha,
            expected_weight_sha,
            expected_stats,
        )
    ):
        raise WorkerError("parent snapshot binding is incomplete")
    try:
        weight_hashes = json.loads(expected_weight_sha)
        stat_manifest = json.loads(expected_stats)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise WorkerError("parent snapshot binding is invalid JSON") from exc
    resolved_path, actual_stats = _snapshot_stat_manifest(snapshot)
    if (
        snapshot.revision != spec["revision"]
        or expected_revision != spec["revision"]
        or resolved_path != expected_path
        or actual_stats != stat_manifest
        or not isinstance(weight_hashes, dict)
    ):
        raise WorkerError("snapshot revision/path/stat binding changed")
    return resolved_path, {
        "snapshot_sha256": expected_snapshot_sha,
        "weight_sha256": weight_hashes,
        "stat_manifest": stat_manifest,
    }


def _authorise(model_key: str) -> None:
    expected_parent = os.environ.get("FRIDAY_MATMUL_PARENT_PID")
    if expected_parent != str(os.getppid()):
        raise WorkerError("worker parent PID is not registered")
    if os.environ.get("FRIDAY_MATMUL_RUN_ID") != RUN_ID:
        raise WorkerError("worker run ID is not registered")
    if os.environ.get("FRIDAY_MATMUL_MODEL_KEY") != model_key or model_key != MODEL_KEY:
        raise WorkerError("worker model key is not registered")
    if os.environ.get("FRIDAY_MATMUL_NONCE") != "cycle16-fixed-cache-v1":
        raise WorkerError("worker nonce is not registered")
    for name in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "PYTHONNOUSERSITE",
    ):
        if os.environ.get(name) != "1":
            raise WorkerError(f"offline environment gate missing: {name}")


def _swap_used_bytes() -> int:
    """Read swap usage through the installed local, read-only psutil API."""

    try:
        import psutil

        value = psutil.swap_memory().used
    except Exception as exc:
        raise WorkerError("swap usage is unavailable") from exc
    if type(value) is not int or value < 0:
        raise WorkerError("swap usage is invalid")
    return value


def _resource_snapshot(mx: Any, swap_before: int) -> dict[str, int]:
    """Return post-arm resource counters and fail closed at registered limits."""

    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    mlx = int(mx.get_peak_memory())
    swap_after = _swap_used_bytes()
    if rss > MAX_RSS_BYTES:
        raise WorkerError("RSS resource limit exceeded")
    if mlx > MAX_MLX_BYTES:
        raise WorkerError("MLX resource limit exceeded")
    if swap_after != swap_before:
        raise WorkerError("swap delta is nonzero")
    return {
        "rss_peak_bytes": rss,
        "mlx_peak_bytes": mlx,
        "swap_after_bytes": swap_after,
        "swap_delta_bytes": swap_after - swap_before,
    }


def _resource_evidence(mx: Any, swap_before: int) -> dict[str, int]:
    """Capture a terminal arm's resource fields without raising again."""
    try:
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        rss = 0
    try:
        mlx = int(mx.get_peak_memory())
    except Exception:
        mlx = 0
    try:
        swap_after = _swap_used_bytes()
    except Exception:
        swap_after = swap_before
    return {
        "rss_peak_bytes": max(0, rss),
        "mlx_peak_bytes": max(0, mlx),
        "swap_after_bytes": max(0, int(swap_after)),
        "swap_delta_bytes": int(swap_after) - swap_before,
    }


class _ChargeRejected(WorkerError):
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
    seconds = observed_ns / 1_000_000_000
    recorded_ns = max(0, round((guard_after_seconds - guard_before_seconds) * 1_000_000_000))
    return {
        "observed_model_work_ns": observed_ns,
        "charged_model_work_ns": observed_ns if accepted else 0,
        "charge_accepted": accepted,
        "guard_gpu_work_before_seconds": guard_before_seconds,
        "guard_gpu_work_after_seconds": guard_after_seconds,
        "guard_recorded_model_work_ns": recorded_ns,
        "duty_formula_break_seconds": seconds * (1.0 - 0.15) / 0.15,
        "required_break_blocks": max(
            MIN_REQUIRED_BREAK_BLOCKS,
            math.ceil((seconds * (1.0 - 0.15) / 0.15) / 4.0),
        ),
    }


def _charge_arm(guard: Any, charged_ns: int) -> dict[str, Any]:
    """Book stopped arm work and calculate, but do not perform, the pause."""

    seconds = charged_ns / 1_000_000_000
    if not math.isfinite(seconds) or seconds <= 0:
        raise WorkerError("arm duration is invalid")
    before = float(guard.gpu_work_seconds)
    try:
        guard.record_gpu(seconds)
    except Exception as exc:
        after = float(guard.gpu_work_seconds)
        evidence = _arm_budget_evidence(
            charged_ns,
            guard_before_seconds=before,
            guard_after_seconds=after,
            accepted=False,
        )
        if isinstance(exc, Exception):
            raise _ChargeRejected(evidence, exc) from exc
        raise
    after = float(guard.gpu_work_seconds)
    required_break_seconds = seconds * (1.0 - guard.policy.duty_cycle_limit) / guard.policy.duty_cycle_limit
    evidence = _arm_budget_evidence(
        charged_ns,
        guard_before_seconds=before,
        guard_after_seconds=after,
        accepted=True,
    )
    if evidence["required_break_blocks"] < max(
        MIN_REQUIRED_BREAK_BLOCKS,
        math.ceil(required_break_seconds / guard.policy.required_break_s),
    ):
        raise WorkerError("registered rolling-duty break projection is invalid")
    return evidence


def _pause_arm(guard: Any, budget: dict[str, Any]) -> None:
    """Sleep only after the post-charge resource checks have passed."""

    for _ in range(int(budget["required_break_blocks"])):
        guard.required_break()


def _run_worker(model_key: str) -> int:
    _authorise(model_key)
    arm_order = _parse_arm_order()

    # All imports below this line are after the direct-invocation gates.
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))
    from _bench import require_ac_power
    from _bench import resolve_local_model_snapshot
    from friday_evidence.budget import BudgetError, BudgetGuard
    from friday_evidence.registry import BudgetPolicy
    from mlx_lm import load
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx

    power_source = require_ac_power()
    policy = BudgetPolicy(
        gpu_work_limit_s=120.0,
        continuous_gpu_limit_s=6.0,
        required_break_s=4.0,
        duty_window_s=60.0,
        duty_cycle_limit=0.15,
        wall_limit_s=1200.0,
        candidate_cooldown_s=0.0,
    )
    guard = BudgetGuard(policy)
    swap_before = _swap_used_bytes()
    spec = {"model_id": MODEL_ID, "revision": MODEL_REVISION}
    snapshot = resolve_local_model_snapshot(MODEL_ID)
    snapshot_path, binding = _verify_snapshot_binding(snapshot, spec)
    mx.reset_peak_memory()
    load_started_ns = time.perf_counter_ns()
    model, tokenizer = load(str(snapshot.path))
    load_finished_ns = time.perf_counter_ns()
    after_path, after_stats = _snapshot_stat_manifest(snapshot)
    if after_path != snapshot_path or after_stats != binding["stat_manifest"]:
        raise WorkerError("snapshot changed during model load")
    if str(mx.default_device()) != "Device(gpu, 0)":
        raise WorkerError("MLX default device is not the registered GPU")

    prompt_ids, rendered_prompt_bytes = _prompt_ids(tokenizer)
    if len(prompt_ids) != EXPECTED_PROMPT_TOKENS:
        raise WorkerError(
            f"prompt token count changed: expected {EXPECTED_PROMPT_TOKENS}, got {len(prompt_ids)}"
        )
    if len(prompt_ids) + DECODE_FORWARDS > CAPACITY:
        raise WorkerError("fixed cache capacity gate failed")
    sampler = make_sampler(temp=0.0)

    arms: dict[str, dict[str, Any]] = {}
    arm_budget: dict[str, dict[str, Any]] = {}
    arm_resources: dict[str, dict[str, int]] = {}
    first_mismatch: dict[str, Any] | None = None
    status = "complete"
    error: dict[str, str] | None = None
    observed_model_work_ns = 0
    charged_model_work_ns = 0
    guard_recorded_model_work_ns = 0
    try:
        for arm in arm_order:
            arm_started_ns = time.perf_counter_ns()
            arm_value: dict[str, Any] | None = None
            arm_exception: Exception | None = None
            try:
                arm_value = _run_arm(model, tokenizer, prompt_ids, arm, sampler, mx)
            except Exception as exc:
                arm_exception = exc

            # Stop the arm clock before charging or sleeping.  Pauses are never
            # included in model-work or arm timing.
            arm_finished_ns = time.perf_counter_ns()
            arm_ns = arm_finished_ns - arm_started_ns
            if arm_ns <= 0:
                raise WorkerError("arm duration is not positive")
            # Count this stopped arm immediately.  This remains in the
            # partial event even if the subsequent budget/resource gate fails.
            observed_model_work_ns += arm_ns
            # Bind both terminal evidence maps before charging.  If charging
            # or the resource gate fails, the parent still receives a
            # structurally complete record for this attempted arm.
            arm_budget[arm] = _arm_budget_evidence(arm_ns)
            try:
                arm_budget[arm] = _charge_arm(guard, arm_ns)
                charged_model_work_ns += arm_ns
                guard_recorded_model_work_ns += arm_budget[arm]["guard_recorded_model_work_ns"]
                # Resource checks must happen before any duty-cycle sleep.
                arm_resources[arm] = _resource_snapshot(mx, swap_before)
                _pause_arm(guard, arm_budget[arm])
            except _ChargeRejected as exc:
                arm_budget[arm] = exc.evidence
                guard_recorded_model_work_ns += exc.evidence["guard_recorded_model_work_ns"]
                arm_resources[arm] = _resource_evidence(mx, swap_before)
                status = "resource_or_budget_failed"
                error = {"type": type(exc.cause).__name__, "message": str(exc.cause)[:500]}
                break
            except BudgetError as exc:
                arm_resources[arm] = _resource_evidence(mx, swap_before)
                status = "resource_or_budget_failed"
                error = {"type": type(exc).__name__, "message": str(exc)[:500]}
                break
            except WorkerError as exc:
                arm_resources[arm] = _resource_evidence(mx, swap_before)
                status = "resource_or_budget_failed"
                error = {"type": type(exc).__name__, "message": str(exc)[:500]}
                break
            if arm_value is None:
                if isinstance(arm_exception, CandidateNotRunnable):
                    raise arm_exception
                if arm_exception is not None:
                    raise arm_exception
                raise WorkerError("arm returned no result")
            arm_value["observed_model_work_ns"] = arm_ns
            arm_value["charged_model_work_ns"] = arm_ns
            arm_value["charge_accepted"] = True
            arm_value["arm_wall_ns"] = arm_ns
            arm_value["prompt_sha256"] = PROMPT_SHA256
            arm_value["prompt_token_sha256"] = _sha256_bytes(_canonical_json(prompt_ids))
            arm_value["rendered_prompt_sha256"] = _sha256_bytes(rendered_prompt_bytes)
            arm_value["budget_summary"] = guard.summary()
            arms[arm] = arm_value
            if len(arms) > 1:
                reference_arm = next(iter(arms))
                reference_tokens = arms[reference_arm]["tokens"]
                if arm_value["tokens"] != reference_tokens:
                    first_difference = next(
                        (
                            index
                            for index, pair in enumerate(
                                zip(reference_tokens, arm_value["tokens"])
                            )
                            if pair[0] != pair[1]
                        ),
                        min(len(reference_tokens), len(arm_value["tokens"])),
                    )
                    first_mismatch = {
                        "arm": arm,
                        "reference_arm": reference_arm,
                        "token_index": first_difference,
                    }
                    status = "correctness_failed"
                    break
    except CandidateNotRunnable as exc:
        status = "candidate_not_runnable"
        error = {"type": type(exc).__name__, "message": str(exc)[:500]}
    except BudgetError as exc:
        status = "resource_or_budget_failed"
        error = {"type": type(exc).__name__, "message": str(exc)[:500]}
    except Exception as exc:
        status = "error"
        error = {"type": type(exc).__name__, "message": str(exc)[:500]}
    mx.synchronize()

    token_hashes = {arm: value.get("token_sha256") for arm, value in arms.items()}
    text_hashes = {arm: value.get("text_utf8_sha256") for arm, value in arms.items()}
    all_token_equal = bool(
        len(arms) == len(ARM_NAMES)
        and len(set(token_hashes.values())) == 1
        and all(len(value.get("tokens", [])) == OUTPUT_TOKENS for value in arms.values())
    )
    all_text_equal = bool(
        len(arms) == len(ARM_NAMES)
        and len(set(text_hashes.values())) == 1
    )
    if first_mismatch is not None:
        all_token_equal = False
        all_text_equal = False
    if status == "complete" and len(arms) == len(ARM_NAMES) and not (
        all_token_equal and all_text_equal
    ):
        status = "correctness_failed"
        error = {"type": "CorrectnessError", "message": "arm tokens or text differ"}
    swap_after: int | None
    try:
        swap_after = _swap_used_bytes()
    except WorkerError as exc:
        swap_after = None
        status = "resource_or_budget_failed"
        error = {"type": type(exc).__name__, "message": str(exc)[:500]}
    if swap_after is not None and swap_after != swap_before:
        status = "resource_or_budget_failed"
        error = {"type": "WorkerError", "message": "swap delta is nonzero"}
    event = {
        "arm_order": list(arm_order),
        "arms": arms,
        "cache_capacity": CAPACITY,
        "correctness": {
            "all_arms_text_equal": all_text_equal,
            "all_arms_token_equal": all_token_equal,
            "first_mismatch": first_mismatch,
            "required_arm_count": len(ARM_NAMES),
        },
        "device": str(mx.default_device()),
        "error": error,
        "event": "complete",
        "fixed_steps": OUTPUT_TOKENS,
        "load_count": 1,
        "model_id": MODEL_ID,
        "model_key": model_key,
        "model_load_ns": load_finished_ns - load_started_ns,
        "model_work_ns": observed_model_work_ns,
        "observed_model_work_ns": observed_model_work_ns,
        "charged_model_work_ns": charged_model_work_ns,
        "guard_recorded_model_work_ns": guard_recorded_model_work_ns,
        "mlx_peak_bytes": int(mx.get_peak_memory()),
        "pid": os.getpid(),
        "prompt_sha256": PROMPT_SHA256,
        "prompt_token_ids": prompt_ids,
        "power_source": power_source,
        "prompt_tokens": len(prompt_ids),
        "rendered_prompt_b64": base64.b64encode(rendered_prompt_bytes).decode("ascii"),
        "rendered_prompt_sha256": _sha256_bytes(rendered_prompt_bytes),
        "rss_peak_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "sampler_temperature": 0.0,
        "snapshot_integrity": {
            "after_load_stat_manifest": after_stats,
            "before_load_stat_manifest": binding["stat_manifest"],
            "bound_snapshot_sha256": binding["snapshot_sha256"],
            "bound_weight_sha256": binding["weight_sha256"],
        },
        "snapshot_path": snapshot_path,
        "snapshot_revision": MODEL_REVISION,
        "snapshot_sha256": binding["snapshot_sha256"],
        "status": status,
        "study_id": STUDY_ID,
        "text_sha256_by_arm": text_hashes,
        "token_sha256_by_arm": token_hashes,
        "arm_budget": arm_budget,
        "arm_resources": arm_resources,
        "budget": guard.summary(),
        "swap_before_bytes": swap_before,
        "swap_after_bytes": swap_after,
        "swap_delta_bytes": (
            swap_after - swap_before if swap_after is not None else None
        ),
        "worker_watchdog_seconds": CONTINUOUS_GPU_LIMIT_SECONDS,
        "weight_sha256": binding["weight_sha256"],
    }
    _emit(event)
    # A nonzero worker status makes the parent preserve this event as a
    # terminal partial result rather than charging a false successful block.
    return 0 if status == "complete" and all_token_equal and all_text_equal else 1


def _self_check() -> int:
    source = Path(__file__).read_text(encoding="utf-8")
    assert PROMPT_SHA256 == (
        "c746eca8644a18fc75673acb9b3dbdf03825cbfba6c76faede5d909cf3d2ea0b"
    )
    assert OUTPUT_TOKENS == 32
    assert DECODE_FORWARDS == 31
    assert CAPACITY == 512
    assert MIN_REQUIRED_BREAK_BLOCKS == 13
    assert ARM_NAMES == ("standard_eager", "fixed_eager", "fixed_compiled")
    assert len(_all_permutations()) == 6
    assert len(set(_all_permutations())) == 6
    assert all(sorted(order) == sorted(ARM_NAMES) for order in _all_permutations())
    assert "slice_update" in FixedKVCache.update_and_fetch.__code__.co_names
    assert "concatenate" not in FixedKVCache.update_and_fetch.__code__.co_names
    assert "mx.compile(body, shapeless=False)" in source
    capture_inputs = "inputs" + "="
    capture_outputs = "outputs" + "="
    assert capture_inputs not in source and capture_outputs not in source
    assert "def body(input_ids: Any, state: dict[str, Any])" in source
    assert "return _fixed_forward(model, input_ids, state, mx)" in source
    assert "duty_cycle_limit=0.15" in source
    assert "required_break_s" in source and "required_break_blocks" in source
    assert "guard_recorded_model_work_ns" in source
    assert "status = \"resource_or_budget_failed\"" in source
    assert "status == \"complete\" and all_token_equal and all_text_equal" in source
    assert "mx." + "concatenate" not in source
    charge_start = source.index("def _charge_arm")
    pause_start = source.index("def _pause_arm")
    run_start = source.index("def _run_worker")
    charge_source = source[charge_start:pause_start]
    run_source = source[run_start:]
    assert "guard.record_gpu(seconds)" in charge_source
    assert "guard.required_break()" not in charge_source
    assert run_source.index("charged_model_work_ns += arm_ns") < run_source.index(
        "arm_resources[arm] = _resource_snapshot"
    ) < run_source.index("_pause_arm(guard, arm_budget[arm])")
    class FakeTokenizer:
        def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
            assert messages == [{"role": "user", "content": PLANNER_PROMPT}]
            assert kwargs == {"tokenize": False, "add_generation_prompt": True}
            return "rendered"

        def encode(self, value: str, *, add_special_tokens: bool) -> list[int]:
            assert value == "rendered" and add_special_tokens is False
            return [1, 2, 3]

    values, rendered = _prompt_ids(FakeTokenizer())
    assert values == [1, 2, 3] and rendered == b"rendered"
    print(json.dumps({"checks": 21, "self_check": "pass"}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="matmul_compile_ab_worker", allow_abbrev=False)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--worker", action="store_true")
    modes.add_argument("--self-check", action="store_true")
    parser.add_argument("--model-key", default=None)
    args = parser.parse_args(argv)
    if args.self_check:
        return _self_check()
    if not args.worker or args.model_key != MODEL_KEY:
        # Do not import MLX on malformed direct calls.
        print(json.dumps({"error": "worker authorization failed", "event": "error"}))
        return 2
    try:
        return _run_worker(args.model_key)
    except Exception as exc:
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
