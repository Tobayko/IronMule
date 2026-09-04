#!/usr/bin/env python3
"""Authorised child for Cycle 21 fused-greedy fixed-cache measurement.

The module is intentionally importable without MLX.  MLX, mlx-lm and the model
are imported only after the parent nonce, offline environment and snapshot
binding have been checked.  The pure helpers are used by the offline tests.
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
import stat
import sys
import time
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREREGISTRATION = Path(__file__).with_name("PREREGISTRATION.md")
HARNESS = Path(__file__).with_name("measure_fused_greedy_compile_v4.py")
STUDY_ID = "fused-greedy-compile-20260825-04"
RUN_ID = "fused-greedy-compile-validation-20260825-04"
CANDIDATE_ID = "fixed_compiled_fused_greedy"
MODEL_KEY = "4b"
MODEL_ID = "mlx-community/gemma-3-4b-it-4bit"
MODEL_REVISION = "93724907d4ed1745d2fe50baadf3b0b01a65abf2"
EXPECTED_SNAPSHOT_SHA256 = "e6edcd46c52b4cf5580f095185a94858565896df7f31c23522294e8f73b3edae"
EXPECTED_WEIGHT_SHA256 = "94d3d701367d78584a9334ca00672b1c86e4aefa6a94167556c0485381e74af3"
CAPACITY = 512
MAX_PHYSICAL_TOKENS = 32
MAX_DECODE_FORWARDS = 31
WARMUP_FORWARDS = 8
EXPECTED_EOS_TOKEN_IDS = (1, 106)
CONTINUOUS_GPU_LIMIT_SECONDS = 6.0
TOTAL_GPU_LIMIT_SECONDS = 120.0
WALL_LIMIT_SECONDS = 1200.0
MAX_RSS_BYTES = 5 * 1024**3
MAX_MLX_BYTES = 5 * 1024**3
BOOTSTRAP_SEED = 20260825
PROTOCOL_VERSION = 1
AUTH_PREFIX = "FRIDAY_FGC_"
AUTH_NONCE = "cycle21-fused-greedy-compile-v4"
OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONNOUSERSITE": "1"}
UNSAFE_ENV = ("PYTHONHOME", "PYTHONPATH", "PYTHONINSPECT", "PYTHONSTARTUP")
RESULT_RELATIVE = "experiments/fused_greedy_compile_v4/results.json"
RESULT_PATH = PROJECT_ROOT / RESULT_RELATIVE
ARM_NAMES = ("fixed_compiled_external_greedy", "fixed_compiled_fused_greedy")

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
PROMPT_SHA256 = hashlib.sha256(PLANNER_PROMPT.encode()).hexdigest()
EXPECTED_PROMPT_SHA256 = "c746eca8644a18fc75673acb9b3dbdf03825cbfba6c76faede5d909cf3d2ea0b"
EXPECTED_PROMPT_TOKEN_SHA256 = "80ecf700cf0dfdc82616c73f1b6a5fccc137b68e9bb9586ca376c3f2adb260ad"
EXPECTED_RENDERED_PROMPT_SHA256 = "9e18d10b7b101bda3d28593190e622544d474655872aed826c9cbc44211a2cca"
FROZEN_PREREGISTRATION_SHA256 = "a734975191de7c77a4966c42c0225d8bdbe89d215e24ff63600affef0599dadf"


class WorkerError(RuntimeError):
    pass


class CandidateNotRunnable(WorkerError):
    pass


class ResourceFailure(WorkerError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_one_json(payload: str | bytes) -> dict[str, Any]:
    raw = payload.encode() if isinstance(payload, str) else payload
    if not isinstance(raw, bytes) or not raw or len(raw) > 1_000_000:
        raise ValueError("empty or oversize JSON")
    raw = raw[:-1] if raw.endswith(b"\n") else raw
    if b"\n" in raw or b"\r" in raw:
        raise ValueError("multiline JSON is forbidden")

    def unique(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    value = json.loads(raw.decode("utf-8", errors="strict"), object_pairs_hook=unique,
                       parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    if not isinstance(value, dict):
        raise ValueError("event is not an object")
    return value


def normalize_tokens(tokens: Any, eos_ids: Any = EXPECTED_EOS_TOKEN_IDS, maximum: int = MAX_PHYSICAL_TOKENS) -> dict[str, Any]:
    if not isinstance(tokens, list) or any(type(item) is not int or item < 0 for item in tokens):
        raise ValueError("tokens must be nonnegative integers")
    if not isinstance(eos_ids, (list, tuple)) or not eos_ids or any(type(item) is not int for item in eos_ids):
        raise ValueError("EOS IDs are invalid")
    physical = list(tokens[:maximum])
    eos_position = next((i for i, token in enumerate(physical) if token in set(eos_ids)), None)
    logical = physical if eos_position is None else physical[: eos_position + 1]
    visible = physical if eos_position is None else physical[:eos_position]
    return {
        "physical_tokens": physical, "logical_tokens": logical, "visible_tokens": visible,
        "physical_token_count": len(physical), "logical_token_count": len(logical),
        "visible_token_count": len(visible), "overproduced_tokens": len(physical) - len(logical),
        "eos_found": eos_position is not None, "eos_position": eos_position,
        "eos_token_id": None if eos_position is None else physical[eos_position],
        "finish_reason": "stop" if eos_position is not None else "length",
        "physical_token_sha256": _sha256_bytes(_canonical(physical)),
        "logical_token_sha256": _sha256_bytes(_canonical(logical)),
        "visible_token_sha256": _sha256_bytes(_canonical(visible)),
    }


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def protocol_contract() -> dict[str, Any]:
    return {"version": PROTOCOL_VERSION, "study_id": STUDY_ID, "run_id": RUN_ID,
            "candidate_id": CANDIDATE_ID, "arms": list(ARM_NAMES), "capacity": CAPACITY,
            "max_physical_tokens": MAX_PHYSICAL_TOKENS, "warmups": WARMUP_FORWARDS,
            "nonce": AUTH_NONCE, "prompt_sha256": EXPECTED_PROMPT_SHA256}


PROTOCOL_SHA256 = _sha256_bytes(_canonical(protocol_contract()))


def code_fingerprints() -> dict[str, str]:
    return {path.relative_to(PROJECT_ROOT).as_posix(): _sha256_file(path)
            for path in (PREREGISTRATION, HARNESS, Path(__file__))}


def environment_fingerprint() -> str:
    return _sha256_bytes(_canonical({"offline": OFFLINE_ENV, "removed": UNSAFE_ENV, "python": str(Path(sys.executable).resolve()), "machine": platform.machine()}))


def allowed_post_marker_status(lines: list[str]) -> bool:
    return lines == [f"?? {RESULT_RELATIVE}"]


def _validate_post_marker_git_state() -> None:
    import subprocess
    marker_path = PROJECT_ROOT / ".friday-data" / "fused-greedy-compile-v4" / "attempt.json"
    marker_directory = marker_path.parent.lstat()
    marker = marker_path.lstat()
    if (not stat.S_ISDIR(marker_directory.st_mode) or stat.S_ISLNK(marker_directory.st_mode)
            or stat.S_IMODE(marker_directory.st_mode) != 0o700 or marker_directory.st_uid != os.geteuid()):
        raise WorkerError("marker directory is not private")
    if (not stat.S_ISREG(marker.st_mode) or stat.S_ISLNK(marker.st_mode)
            or stat.S_IMODE(marker.st_mode) != 0o600 or marker.st_uid != os.geteuid()):
        raise WorkerError("marker file is not private")
    result = RESULT_PATH.lstat()
    if (not stat.S_ISREG(result.st_mode) or stat.S_ISLNK(result.st_mode)
            or stat.S_IMODE(result.st_mode) != 0o644 or result.st_uid != os.geteuid()):
        raise WorkerError("result file is not a regular 0644 runner-owned file")
    if any(path.name.startswith(f".{RESULT_PATH.name}.") and path.name.endswith(".tmp") for path in RESULT_PATH.parent.iterdir()):
        raise WorkerError("result temporary file remains")
    status = subprocess.run(["git", "status", "--porcelain=v1", "--untracked-files=all", "--", ".", ":(exclude)ProjectAtlas"], cwd=PROJECT_ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=5).stdout.decode()
    if not allowed_post_marker_status(status.splitlines() if status else []):
        raise WorkerError("unexpected Git change outside authorized evidence")


def _tree_leaves(value: Any) -> list[Any]:
    if isinstance(value, dict):
        return [leaf for key in sorted(value) for leaf in _tree_leaves(value[key])]
    if isinstance(value, (list, tuple)):
        return [leaf for child in value for leaf in _tree_leaves(child)]
    return [value]


class FixedKVCache:
    def __init__(self, state: dict[str, Any], position: dict[str, Any], mx: Any):
        self._state, self._position, self._mx = state, position, mx

    @property
    def keys(self): return self._state["keys"]

    @property
    def values(self): return self._state["values"]

    @property
    def offset(self): return self._position["offset"]

    def update_and_fetch(self, keys: Any, values: Any):
        if keys.shape[2] < 1 or keys.shape[2] > CAPACITY:
            raise CandidateNotRunnable("invalid fixed-cache step shape")
        z = self._mx.array(0, dtype=self.offset.dtype)
        starts = self._mx.stack((z, z, self.offset, z))
        self._state["keys"] = self._mx.slice_update(self._state["keys"], keys, start_indices=starts, axes=(0, 1, 2, 3))
        self._state["values"] = self._mx.slice_update(self._state["values"], values, start_indices=starts, axes=(0, 1, 2, 3))
        return self._state["keys"], self._state["values"]

    def make_mask(self, n_tokens: int, *, window_size: int | None = None, return_array: bool = False):
        del return_array
        positions = self._mx.arange(CAPACITY, dtype=self.offset.dtype)
        queries = self.offset + self._mx.arange(n_tokens, dtype=self.offset.dtype)
        mask = (positions[None, :] <= queries[:, None]) & (positions[None, :] < self.offset + n_tokens)
        if window_size is not None:
            mask = mask & (positions[None, :] >= queries[:, None] - window_size + 1)
        return mask[None, None, :, :]


def _fixed_state_from_standard_cache(cache: list[Any], prompt_tokens: int, mx: Any) -> dict[str, Any]:
    if not cache or prompt_tokens < 1:
        raise CandidateNotRunnable("empty standard cache")
    layers = []
    for layer in cache:
        keys, values = getattr(layer, "keys", None), getattr(layer, "values", None)
        if keys is None or values is None or len(keys.shape) != 4 or len(values.shape) != 4:
            raise CandidateNotRunnable("invalid standard cache tensors")
        if keys.shape[2] < prompt_tokens or keys.shape[2] > CAPACITY:
            raise CandidateNotRunnable("cache length outside fixed capacity")
        padded_keys = mx.zeros((keys.shape[0], keys.shape[1], CAPACITY, keys.shape[3]), dtype=keys.dtype)
        padded_values = mx.zeros((values.shape[0], values.shape[1], CAPACITY, values.shape[3]), dtype=values.dtype)
        start = mx.array((0, 0, 0, 0), dtype=mx.int32)
        layers.append({"keys": mx.slice_update(padded_keys, keys[..., :prompt_tokens, :], start_indices=start, axes=(0, 1, 2, 3)),
                       "values": mx.slice_update(padded_values, values[..., :prompt_tokens, :], start_indices=start, axes=(0, 1, 2, 3))})
    return {"position": {"offset": mx.array(prompt_tokens, dtype=mx.int32)}, "layers": layers}


def _fixed_caches(state: dict[str, Any], mx: Any) -> list[FixedKVCache]:
    if not isinstance(state.get("position"), dict) or not isinstance(state.get("layers"), list):
        raise CandidateNotRunnable("invalid fixed state")
    return [FixedKVCache(layer, state["position"], mx) for layer in state["layers"]]


def _fixed_forward(model: Any, input_ids: Any, state: dict[str, Any], mx: Any):
    caches = _fixed_caches(state, mx)
    logits = model(input_ids, cache=caches)
    old_offset = state["position"]["offset"]
    new_state = {"position": {"offset": old_offset + input_ids.shape[1]},
                 "layers": [{"keys": cache.keys, "values": cache.values} for cache in caches]}
    return logits, new_state


def _make_compiled(model: Any, mx: Any, fused: bool):
    def body(input_ids: Any, state: dict[str, Any]):
        logits, new_state = _fixed_forward(model, input_ids, state, mx)
        if fused:
            return mx.argmax(logits[:, -1, :], axis=-1).reshape((1,)), new_state
        return logits, new_state
    try:
        return mx.compile(body, shapeless=False)
    except Exception as exc:
        if _resource_failure(exc):
            raise ResourceFailure(str(exc)) from exc
        raise CandidateNotRunnable(str(exc)) from exc


def _resource_failure(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return isinstance(exc, (MemoryError, OSError)) or any(word in text for word in ("out of memory", "oom", "killed", "budget", "swap"))


def _host_boundary(token: Any, state: dict[str, Any], mx: Any) -> tuple[int, int]:
    started = time.perf_counter_ns()
    leaves = _tree_leaves(state)
    mx.async_eval(token, *leaves)
    mx.eval(token, *leaves)
    mx.synchronize()
    raw = token.reshape((-1,)).tolist()
    if not isinstance(raw, list) or len(raw) != 1 or type(raw[0]) is not int or raw[0] < 0:
        raise CandidateNotRunnable("host token boundary returned invalid token")
    return raw[0], time.perf_counter_ns() - started


def _prompt_ids(tokenizer: Any) -> tuple[list[int], bytes]:
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": PLANNER_PROMPT}], tokenize=False, add_generation_prompt=True)
    values = tokenizer.encode(rendered, add_special_tokens=False)
    values = values if isinstance(values, list) else values.tolist()
    if len(values) != 322 or any(type(item) is not int for item in values):
        raise WorkerError("prompt token gate failed")
    return values, rendered.encode()


def _materialize(state: dict[str, Any], mx: Any) -> None:
    values = [state["position"]["offset"]]
    values.extend(leaf for layer in state["layers"] for leaf in (layer["keys"], layer["values"]))
    mx.eval(*values); mx.synchronize()


def _prepare(model: Any, prompt_ids: list[int], mx: Any) -> tuple[dict[str, Any], Any, int, int, Any, int, int]:
    ids = mx.array(prompt_ids)[None, :]
    started = time.perf_counter_ns()
    cache = model.make_cache() if hasattr(model, "make_cache") else None
    if cache is None:
        raise CandidateNotRunnable("model did not expose a fresh standard cache")
    logits = model(ids, cache=cache)
    mx.eval(logits); mx.synchronize()
    prefill_ns = time.perf_counter_ns() - started
    conversion_started = time.perf_counter_ns()
    state = _fixed_state_from_standard_cache(cache, len(prompt_ids), mx)
    _materialize(state, mx)
    conversion_ns = time.perf_counter_ns() - conversion_started
    return state, None, time.perf_counter_ns() - started, 0, logits, prefill_ns, conversion_ns


def _run_arm(model: Any, tokenizer: Any, prompt_ids: list[int], arm: str, mx: Any, sampler: Any) -> dict[str, Any]:
    del sampler
    fused = arm == "fixed_compiled_fused_greedy"
    compile_started = time.perf_counter_ns()
    compiled = _make_compiled(model, mx, fused)
    compile_wrapper_ns = time.perf_counter_ns() - compile_started
    warmup_state, _, warmup_total_ns, _, warmup_logits, warmup_prefill_ns, warmup_conversion_ns = _prepare(model, prompt_ids, mx)
    warmup_first, _ = _host_boundary(mx.argmax(warmup_logits[:, -1, :], axis=-1).reshape((1,)), warmup_state, mx)
    state, _, prep_total_ns, _, measurement_logits, prefill_ns, conversion_ns = _prepare(model, prompt_ids, mx)
    current = warmup_first
    warmup_ns = []
    compile_cold_ns = None
    for warmup_index in range(WARMUP_FORWARDS):
        warmup_started = time.perf_counter_ns()
        output = compiled(mx.array([[current]]), warmup_state)
        token = output[0] if fused else mx.argmax(output[0][:, -1, :], axis=-1).reshape((1,))
        current, elapsed = _host_boundary(token, output[1], mx)
        warmup_state = output[1]
        elapsed = time.perf_counter_ns() - warmup_started
        if warmup_index == 0: compile_cold_ns = elapsed
        warmup_ns.append(elapsed)
    primary_started = time.perf_counter_ns()
    current, first_boundary_ns = _host_boundary(mx.argmax(measurement_logits[:, -1, :], axis=-1).reshape((1,)), state, mx)
    physical = [current]
    decode_ns: list[int] = []
    host_ns: list[int] = []
    for _ in range(MAX_DECODE_FORWARDS):
        started = time.perf_counter_ns()
        output = compiled(mx.array([[current]]), state)
        token = output[0] if fused else mx.argmax(output[0][:, -1, :], axis=-1).reshape((1,))
        current, boundary_ns = _host_boundary(token, output[1], mx)
        state = output[1]
        elapsed = time.perf_counter_ns() - started
        physical.append(current); decode_ns.append(elapsed); host_ns.append(boundary_ns)
        if current in EXPECTED_EOS_TOKEN_IDS:
            break
    state = None
    primary_ns = time.perf_counter_ns() - primary_started
    normalized = normalize_tokens(physical)
    visible_text = tokenizer.decode(normalized["visible_tokens"], skip_special_tokens=False)
    total_ns = sum(decode_ns)
    return {"arm": arm, "fixed_cache": True, "fixed_compile": True, "fused_selection": fused,
            "cache_capacity": CAPACITY, "warmup_forwards": WARMUP_FORWARDS,
            "decode_forwards": len(physical) - 1, "physical_forwards": len(physical) - 1,
            "finish_reason": normalized["finish_reason"], "cache_discarded": True,
            "physical_tokens": physical, "logical_tokens": normalized["logical_tokens"], "visible_tokens": normalized["visible_tokens"],
            "physical_token_count": normalized["physical_token_count"], "logical_token_count": normalized["logical_token_count"], "visible_token_count": normalized["visible_token_count"],
            "overproduced_tokens": normalized["overproduced_tokens"], "eos_found": normalized["eos_found"], "eos_position": normalized["eos_position"], "eos_token_id": normalized["eos_token_id"],
            "physical_token_sha256": normalized["physical_token_sha256"], "logical_token_sha256": normalized["logical_token_sha256"], "visible_token_sha256": normalized["visible_token_sha256"],
            "visible_text": visible_text, "text_sha256": _sha256_bytes(visible_text.encode()),
            "prompt_sha256": EXPECTED_PROMPT_SHA256, "prompt_token_sha256": EXPECTED_PROMPT_TOKEN_SHA256, "rendered_prompt_sha256": EXPECTED_RENDERED_PROMPT_SHA256,
            "ttft_ns": prep_total_ns + first_boundary_ns, "prefill_ns": prefill_ns, "model_work_ns": total_ns,
            "decode_critical_path_ns": primary_ns, "host_readback_ns": sum(host_ns) + first_boundary_ns, "host_boundary_count": len(host_ns) + 1,
            "host_transfer_api_call_count": len(host_ns) + 1, "intertoken_ns": decode_ns,
            "token_rate": (len(normalized["logical_tokens"]) / (primary_ns / 1e9)) if primary_ns else 0.0,
            "compile_wrapper_ns": compile_wrapper_ns, "compile_cold_ns": compile_cold_ns,
            "first_token_boundary_ns": first_boundary_ns, "warmup_boundary_ns": warmup_ns,
            "warmup_prefill_ns": warmup_prefill_ns, "warmup_conversion_ns": warmup_conversion_ns,
            "warmup_preparation_total_ns": warmup_total_ns, "measurement_preparation_total_ns": prep_total_ns,
            "measurement_prefill_ns": prefill_ns, "cache_conversion_ns": conversion_ns,
            "timing_scopes": {"ttft": "measurement_prefill_plus_conversion_plus_first_host_boundary", "primary": "measurement_first_host_boundary_through_decode_eos_and_state_discard", "model_work": "whole_arm_until_before_budget_charge"}}


def _snapshot_manifest(snapshot: Any) -> tuple[str, dict[str, Any]]:
    root = Path(snapshot.path).resolve(strict=True)
    repository = root.parent.parent.resolve(strict=True)
    if root.parent.name != "snapshots": raise WorkerError("unexpected snapshot layout")
    required = ["config.json", "tokenizer_config.json"]
    required += [name for name in ("tokenizer.json", "tokenizer.model") if (root / name).is_file()][:1]
    required.extend(snapshot.weight_files)
    manifest = {}
    for name in dict.fromkeys(required):
        path = (root / name).resolve(strict=True)
        path.relative_to(repository)
        if not path.is_file(): raise WorkerError(f"missing snapshot file {name}")
        item = path.stat(); manifest[name] = {"dev": int(item.st_dev), "inode": int(item.st_ino), "mtime_ns": int(item.st_mtime_ns), "path": str(path), "size": int(item.st_size)}
    generation = (root / "generation_config.json").resolve(strict=True)
    generation.relative_to(repository)
    if not generation.is_file(): raise WorkerError("missing snapshot file generation_config.json")
    generation_stat = generation.stat()
    manifest["generation_config.json"] = {"dev": int(generation_stat.st_dev), "inode": int(generation_stat.st_ino), "mtime_ns": int(generation_stat.st_mtime_ns), "path": str(generation), "size": int(generation_stat.st_size)}
    return str(root), manifest


def _snapshot_identity(snapshot: Any) -> tuple[str, dict[str, Any], dict[str, str], dict[str, str]]:
    root, manifest = _snapshot_manifest(snapshot)
    files = {name: _sha256_file(Path(item["path"])) for name, item in manifest.items() if name != "generation_config.json"}
    execution_files = {name: _sha256_file(Path(item["path"])) for name, item in manifest.items()}
    return root, manifest, files, execution_files


def _authorise(model_key: str) -> int:
    expected = {"PARENT_PID": str(os.getppid()), "RUN_ID": RUN_ID, "MODEL_KEY": model_key,
                "NONCE": AUTH_NONCE, "PROTOCOL_VERSION": str(PROTOCOL_VERSION), "PROTOCOL_SHA256": PROTOCOL_SHA256,
                "PREREG_SHA256": FROZEN_PREREGISTRATION_SHA256, "PROMPT_SHA256": EXPECTED_PROMPT_SHA256,
                "ENVIRONMENT_SHA256": environment_fingerprint()}
    for key, value in expected.items():
        if os.environ.get(AUTH_PREFIX + key) != value:
            raise WorkerError(f"authorisation failed: {key}")
    for key, value in OFFLINE_ENV.items():
        if os.environ.get(key) != value: raise WorkerError(f"offline gate failed: {key}")
    try: block = int(os.environ[AUTH_PREFIX + "BLOCK"])
    except (KeyError, ValueError) as exc: raise WorkerError("invalid block") from exc
    if not 1 <= block <= 6: raise WorkerError("invalid block")
    order = json.loads(os.environ.get(AUTH_PREFIX + "ARM_ORDER", "[]"))
    if tuple(order) not in ((ARM_NAMES[0], ARM_NAMES[1]), (ARM_NAMES[1], ARM_NAMES[0])): raise WorkerError("invalid arm order")
    marker_token = os.environ.get(AUTH_PREFIX + "MARKER_TOKEN", "")
    marker_path = PROJECT_ROOT / ".friday-data" / "fused-greedy-compile-v4" / "attempt.json"
    try:
        marker_metadata = marker_path.lstat()
        if (not stat.S_ISREG(marker_metadata.st_mode) or stat.S_ISLNK(marker_metadata.st_mode)
                or stat.S_IMODE(marker_metadata.st_mode) != 0o600 or marker_metadata.st_uid != os.geteuid()):
            raise WorkerError("marker file mode/type is unsafe")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise WorkerError("marker binding is unavailable") from exc
    if marker.get("token_sha256") != _sha256_bytes(marker_token.encode()) or not marker_token:
        raise WorkerError("marker token binding failed")
    expected_git = os.environ.get(AUTH_PREFIX + "GIT_REVISION")
    if expected_git:
        import subprocess
        actual_git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=5).stdout.decode().strip()
        if actual_git != expected_git: raise WorkerError("git revision binding failed")
        _validate_post_marker_git_state()
    bound_fingerprints = os.environ.get(AUTH_PREFIX + "CODE_FINGERPRINTS")
    if bound_fingerprints:
        try: bound_fingerprints_value = json.loads(bound_fingerprints)
        except json.JSONDecodeError as exc: raise WorkerError("code fingerprint binding is invalid") from exc
        actual_fingerprints = code_fingerprints()
        if bound_fingerprints_value != actual_fingerprints or os.environ.get(AUTH_PREFIX + "CODE_SHA256") != _sha256_bytes(_canonical(actual_fingerprints)):
            raise WorkerError("code fingerprint binding failed")
    return block


def _read_eos(snapshot_path: str) -> tuple[int, ...]:
    value = json.loads((Path(snapshot_path) / "generation_config.json").read_text())
    raw = value.get("eos_token_id")
    ids = (raw,) if type(raw) is int else tuple(raw) if isinstance(raw, list) else ()
    if ids != EXPECTED_EOS_TOKEN_IDS: raise WorkerError("EOS configuration changed")
    return ids


def _resource_evidence(mx: Any, swap_before: int) -> dict[str, Any]:
    try:
        import psutil
        swap_after = int(psutil.swap_memory().used)
    except Exception:
        swap_after = None
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    mlx = int(mx.get_peak_memory())
    return {"rss_peak_bytes": rss, "mlx_peak_bytes": mlx, "swap_before_bytes": swap_before,
            "swap_after_bytes": swap_after, "swap_delta_bytes": None if swap_after is None else swap_after - swap_before,
            "swap_available": swap_after is not None}


def _emit(event: dict[str, Any]) -> None:
    payload = json.dumps(event, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    if len(payload) > 1_000_000 or b"\n" in payload or b"\r" in payload: raise WorkerError("worker event exceeds strict output cap")
    sys.stdout.buffer.write(payload + b"\n")
    sys.stdout.flush()


def _run_worker(model_key: str) -> int:
    block = _authorise(model_key)
    sys.path.insert(0, str(PROJECT_ROOT)); sys.path.insert(0, str(PROJECT_ROOT / "tools"))
    from _bench import require_ac_power, resolve_local_model_snapshot
    from friday_evidence.budget import BudgetGuard, BudgetError
    from friday_evidence.registry import BudgetPolicy
    from mlx_lm import load
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx
    process_started = time.perf_counter_ns()
    power = require_ac_power()
    policy = BudgetPolicy(gpu_work_limit_s=TOTAL_GPU_LIMIT_SECONDS, continuous_gpu_limit_s=CONTINUOUS_GPU_LIMIT_SECONDS,
                          required_break_s=4.0, duty_window_s=60.0, duty_cycle_limit=0.15, wall_limit_s=WALL_LIMIT_SECONDS, candidate_cooldown_s=0.0)
    guard = BudgetGuard(policy)
    swap_before = int(__import__("psutil").swap_memory().used)
    snapshot = resolve_local_model_snapshot(MODEL_ID)
    if snapshot.revision != MODEL_REVISION: raise ResourceFailure("snapshot revision mismatch")
    snapshot_path, stat_manifest, snapshot_files, execution_files = _snapshot_identity(snapshot)
    bound_path = os.environ.get(AUTH_PREFIX + "SNAPSHOT_PATH")
    bound_stats = os.environ.get(AUTH_PREFIX + "SNAPSHOT_STAT_MANIFEST")
    bound_files = os.environ.get(AUTH_PREFIX + "SNAPSHOT_FILES_SHA256")
    if (bound_path != snapshot_path or os.environ.get(AUTH_PREFIX + "SNAPSHOT_REVISION") != MODEL_REVISION
            or os.environ.get(AUTH_PREFIX + "SNAPSHOT_SHA256") != EXPECTED_SNAPSHOT_SHA256
            or os.environ.get(AUTH_PREFIX + "WEIGHT_SHA256") != _canonical({name: snapshot_files[name] for name in snapshot.weight_files}).decode()
            or os.environ.get(AUTH_PREFIX + "EXECUTION_FILES_SHA256") != _canonical(execution_files).decode()
            or (bound_stats and json.loads(bound_stats) != stat_manifest)
            or (bound_files and json.loads(bound_files) != snapshot_files)):
        raise ResourceFailure("snapshot binding changed")
    load_started = time.perf_counter_ns(); model, tokenizer = load(snapshot_path); load_ns = time.perf_counter_ns() - load_started
    if str(mx.default_device()) != "Device(gpu, 0)": raise ResourceFailure("wrong MLX device")
    after_load_path, after_load_stats, after_load_files, after_load_execution_files = _snapshot_identity(snapshot)
    if (after_load_path, after_load_stats, after_load_files, after_load_execution_files) != (snapshot_path, stat_manifest, snapshot_files, execution_files):
        raise ResourceFailure("snapshot changed during load")
    prompt_ids, rendered = _prompt_ids(tokenizer)
    if _sha256_bytes(rendered) != EXPECTED_RENDERED_PROMPT_SHA256: raise WorkerError("rendered prompt hash mismatch")
    eos_ids = _read_eos(snapshot_path)
    order = tuple(json.loads(os.environ[AUTH_PREFIX + "ARM_ORDER"]))
    arms: dict[str, Any] = {}; arm_budget: dict[str, Any] = {}; arm_resources: dict[str, Any] = {}; status = "complete"; error = None
    observed = charged = recorded = 0
    try:
        for arm in order:
            started = time.perf_counter_ns()
            try:
                value = _run_arm(model, tokenizer, prompt_ids, arm, mx, make_sampler(temp=0.0))
                stopped = time.perf_counter_ns(); arm_ns = stopped - started
                observed += arm_ns
                before_guard = guard.gpu_work_seconds
                try:
                    guard.record_gpu(arm_ns / 1e9); accepted = True
                except BudgetError as exc:
                    accepted = False; status = "resource_or_budget_failed"; error = {"type": type(exc).__name__, "message": str(exc)[:300]}
                after_guard = guard.gpu_work_seconds; guard_ns = max(0, int(round((after_guard - before_guard) * 1e9)))
                recorded += guard_ns
                charged += arm_ns if accepted else 0
                arm_budget[arm] = {"observed_model_work_ns": arm_ns, "charged_model_work_ns": arm_ns if accepted else 0,
                                   "guard_recorded_model_work_ns": guard_ns, "charge_accepted": accepted,
                                   "required_break_blocks": 13, "required_break_seconds": 52.0}
                arm_resources[arm] = _resource_evidence(mx, swap_before); arms[arm] = value
                arm_resource = arm_resources[arm]
                if (not arm_resource.get("swap_available") or arm_resource.get("swap_delta_bytes") != 0
                        or arm_resource.get("rss_peak_bytes", MAX_RSS_BYTES + 1) > MAX_RSS_BYTES
                        or arm_resource.get("mlx_peak_bytes", MAX_MLX_BYTES + 1) > MAX_MLX_BYTES):
                    status = "resource_or_budget_failed"
                    error = {"type": "ResourceFailure", "message": "per-arm RSS/MLX/swap gate failed"}
                    break
                for _ in range(13): guard.required_break()
                if not accepted: break
            except Exception as exc:
                status = "resource_or_budget_failed" if _resource_failure(exc) else "candidate_not_runnable" if arm == CANDIDATE_ID else "error"
                error = {"type": type(exc).__name__, "message": str(exc)[:300]}; arm_budget.setdefault(arm, {"observed_model_work_ns": 0, "charged_model_work_ns": 0, "guard_recorded_model_work_ns": 0, "charge_accepted": False, "required_break_blocks": 0, "required_break_seconds": 0.0}); break
    finally:
        try: mx.synchronize()
        except Exception: pass
    post_arm_path, post_arm_stats, post_arm_files, post_arm_execution_files = _snapshot_identity(snapshot)
    if (post_arm_path, post_arm_stats, post_arm_files, post_arm_execution_files) != (snapshot_path, stat_manifest, snapshot_files, execution_files):
        status = "resource_or_budget_failed"; error = {"type": "ResourceFailure", "message": "snapshot changed after arm"}
    if len(arms) == 2 and status == "complete":
        a, b = arms[ARM_NAMES[0]], arms[ARM_NAMES[1]]
        if any(a[field] != b[field] for field in ("physical_token_sha256", "logical_token_sha256", "visible_token_sha256", "text_sha256")):
            status = "correctness_failed"; error = {"type": "CorrectnessError", "message": "baseline and fused tokens/text differ"}
    resources = _resource_evidence(mx, swap_before)
    if (not resources.get("swap_available") or resources["rss_peak_bytes"] > MAX_RSS_BYTES
            or resources["mlx_peak_bytes"] > MAX_MLX_BYTES or resources["swap_delta_bytes"] != 0):
        status = "resource_or_budget_failed"
        error = {"type": "ResourceFailure", "message": "RSS/MLX/swap resource gate failed"}
    import subprocess
    git_revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=5).stdout.decode().strip()
    event = {"event": "complete" if status == "complete" else "terminal", "status": status, "study_id": STUDY_ID, "run_id": RUN_ID, "candidate_id": CANDIDATE_ID, "formal_claim": False,
             "protocol_version": PROTOCOL_VERSION, "process_index": block, "arm_order": list(order), "arms": arms, "arm_budget": arm_budget, "arm_resources": arm_resources,
             "correctness": {"pass": status != "correctness_failed" and len(arms) == 2, "physical_identity": len(arms) == 2 and arms[ARM_NAMES[0]].get("physical_token_sha256") == arms[ARM_NAMES[1]].get("physical_token_sha256"), "logical_identity": len(arms) == 2 and arms[ARM_NAMES[0]].get("logical_token_sha256") == arms[ARM_NAMES[1]].get("logical_token_sha256"), "visible_identity": len(arms) == 2 and arms[ARM_NAMES[0]].get("visible_token_sha256") == arms[ARM_NAMES[1]].get("visible_token_sha256"), "text_identity": len(arms) == 2 and arms[ARM_NAMES[0]].get("text_sha256") == arms[ARM_NAMES[1]].get("text_sha256")},
             "error": error, "pid": os.getpid(), "load_count": 1, "model_key": MODEL_KEY, "model_id": MODEL_ID, "snapshot_revision": MODEL_REVISION,
             "snapshot_path": snapshot_path, "snapshot_sha256": _sha256_bytes(_canonical(snapshot_files)), "weight_sha256": {name: snapshot_files[name] for name in snapshot.weight_files},
             "snapshot_integrity": {"before_load_stat_manifest": stat_manifest, "after_load_stat_manifest": after_load_stats, "post_arm_stat_manifest": post_arm_stats, "snapshot_files_sha256": snapshot_files, "execution_files_sha256": execution_files}, "execution_files_sha256": execution_files, "model_load_ns": load_ns,
             "cache_capacity": CAPACITY, "max_physical_tokens": MAX_PHYSICAL_TOKENS, "warmup_forwards_per_arm": WARMUP_FORWARDS,
             "prompt_sha256": EXPECTED_PROMPT_SHA256, "prompt_token_ids": prompt_ids, "prompt_token_sha256": EXPECTED_PROMPT_TOKEN_SHA256,
             "rendered_prompt_b64": base64.b64encode(rendered).decode(), "rendered_prompt_sha256": EXPECTED_RENDERED_PROMPT_SHA256, "prompt_tokens": len(prompt_ids), "eos_token_ids": list(eos_ids),
             "sampler_temperature": 0.0, "greedy": True, "device": str(mx.default_device()), "power_source": power,
             "model_work_ns": observed, "observed_model_work_ns": observed, "charged_model_work_ns": charged, "guard_recorded_model_work_ns": recorded,
             "budget": guard.summary(), **resources, "worker_watchdog_seconds": CONTINUOUS_GPU_LIMIT_SECONDS,
             "determinism": {"greedy": True, "within_arm_checked_by_parent": True}, "git_revision": git_revision, "dirty_state": "clean",
             "marker_token_sha256": _sha256_bytes(os.environ.get(AUTH_PREFIX + "MARKER_TOKEN", "").encode()),
             "preregistration_sha256": FROZEN_PREREGISTRATION_SHA256,
             "code_fingerprints": code_fingerprints(), "code_sha256": _sha256_bytes(_canonical(code_fingerprints())), "environment_sha256": environment_fingerprint(),
             "process_wall_ns": time.perf_counter_ns() - process_started}
    _emit(event)
    return 0 if status == "complete" else 1


def _self_check() -> int:
    assert PROMPT_SHA256 == EXPECTED_PROMPT_SHA256
    assert normalize_tokens([3, 1, 9])["visible_tokens"] == [3]
    assert normalize_tokens([3, 4], [1])["finish_reason"] == "length"
    assert parse_one_json(b'{"ok":true}') == {"ok": True}
    try: parse_one_json(b'{"x":1,"x":2}')
    except ValueError: pass
    else: raise AssertionError("duplicate key accepted")
    assert allowed_post_marker_status([f"?? {RESULT_RELATIVE}"])
    assert not allowed_post_marker_status([" M experiments/fused_greedy_compile_v4/worker.py"])
    print(json.dumps({"self_check": "pass", "checks": 7}, sort_keys=True)); return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False); parser.add_argument("--execute", action="store_true"); parser.add_argument("--self-check", action="store_true"); parser.add_argument("--model-key", default=MODEL_KEY)
    args = parser.parse_args(argv)
    if args.self_check: return _self_check()
    if not args.execute: print(json.dumps({"state": "not_released", "formal_claim": False})); return 78
    try: return _run_worker(args.model_key)
    except Exception as exc:
        try:
            block = int(os.environ.get(AUTH_PREFIX + "BLOCK", "0"))
        except ValueError:
            block = 0
        try:
            order = json.loads(os.environ.get(AUTH_PREFIX + "ARM_ORDER", "[]"))
        except json.JSONDecodeError:
            order = []
        try:
            import subprocess
            git_revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=5).stdout.decode().strip()
        except Exception:
            git_revision = os.environ.get(AUTH_PREFIX + "GIT_REVISION", "unknown")
        status = "resource_or_budget_failed" if isinstance(exc, ResourceFailure) or _resource_failure(exc) else "candidate_not_runnable" if isinstance(exc, CandidateNotRunnable) else "error"
        _emit({"event": "terminal" if status != "error" else "error", "status": status, "protocol_version": PROTOCOL_VERSION, "pid": os.getpid(), "load_count": 0, "study_id": STUDY_ID,
               "run_id": RUN_ID, "candidate_id": CANDIDATE_ID, "formal_claim": False,
               "process_index": block, "arm_order": order, "arms": {}, "arm_budget": {},
               "arm_resources": {}, "correctness": {"pass": False}, "load_count": 0,
               "error": {"type": type(exc).__name__, "message": str(exc)[:300]}, "partial_result": True,
               "preregistration_sha256": FROZEN_PREREGISTRATION_SHA256, "environment_sha256": environment_fingerprint(),
               "code_fingerprints": code_fingerprints(), "code_sha256": _sha256_bytes(_canonical(code_fingerprints())),
               "git_revision": git_revision, "dirty_state": "unknown"})
        return 1


if __name__ == "__main__": raise SystemExit(main())
