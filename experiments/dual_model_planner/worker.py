#!/usr/bin/env python3
"""One fresh, fixed MLX process for the dual-model planner study.

The parent harness supplies only the immutable model key.  This worker does not
accept a prompt, code, path, or candidate from the outside and never executes
model output.  Its normal entry point is deliberately gated by a registered
parent PID and study run ID; ``--self-check`` is the only offline mode.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import resource
import signal
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "dual-model-evidence-planner-validation-20260824-01"
MAX_OUTPUT_TOKENS = 32
PREFILL_STEP_SIZE = 256
MAX_RESPONSE_BYTES = 8_192
CONTINUOUS_GPU_LIMIT_SECONDS = 6.0

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
EXACT_RESPONSE = '{"candidate_id":"persistent_service_qualification"}'
ALLOWED_CANDIDATES = (
    EXPECTED_CANDIDATE,
    "batched_readback",
    "host_readback_upper_bound",
    "kv_cache_preallocation_ab",
)

# This is byte-for-byte the prompt registered for cycle 14.  Do not format,
# interpolate, or append a newline: both models must see identical input bytes.
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
    """The fixed child cannot complete its closed protocol."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _emit(value: dict[str, Any]) -> None:
    sys.stdout.write(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )
    sys.stdout.flush()


def _snapshot_stat_manifest(snapshot: Any) -> tuple[str, dict[str, dict[str, Any]]]:
    """Return resolved execution paths and cheap TOCTOU metadata.

    Content hashes are deliberately computed once by the parent before and
    once after the study.  The child binds each measured load to the parent's
    content hashes using this inexpensive resolved-path/stat manifest so those
    provenance checks do not inflate ``process_wall_ns``.
    """

    root = Path(snapshot.path).resolve(strict=True)
    if not root.is_dir():
        raise WorkerError("resolved model snapshot is not a directory")
    try:
        repository = (root.parent.parent).resolve(strict=True)
        root.relative_to(repository)
    except (OSError, ValueError) as exc:
        raise WorkerError("model snapshot root is outside its local repository") from exc
    if root.parent.name != "snapshots":
        raise WorkerError("model snapshot root has an unexpected layout")

    def execution_path(relative: str) -> Path:
        candidate = root / relative
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(repository)
        except (OSError, ValueError) as exc:
            raise WorkerError(
                f"model execution file is outside local repository: {relative}"
            ) from exc
        if not resolved.is_file():
            raise WorkerError(f"model execution path is not a file: {relative}")
        return resolved

    required = ["config.json", "tokenizer_config.json"]
    for tokenizer_name in ("tokenizer.json", "tokenizer.model"):
        candidate = root / tokenizer_name
        if candidate.is_file() or candidate.is_symlink():
            required.append(tokenizer_name)
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
    if not manifest:
        raise WorkerError("model snapshot has no execution files")
    return str(root), manifest


def parse_structure(text: str) -> str:
    """Parse the structural JSON shape without repairing model text.

    This result is recorded separately from ``parse_choice``.  Insignificant
    JSON whitespace may be structurally valid, but it is never contract-valid.
    """

    if not isinstance(text, str) or not text:
        raise WorkerError("planner answer is empty or not text")
    if len(text.encode("utf-8")) > MAX_RESPONSE_BYTES:
        raise WorkerError("planner answer is too large")
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = item
        return result

    try:
        value = json.loads(
            text,
            object_pairs_hook=unique_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {item}")
            ),
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise WorkerError("planner answer is not strict JSON") from exc
    if not isinstance(value, dict) or set(value) != {"candidate_id"}:
        raise WorkerError("planner answer fields are invalid")
    candidate = value.get("candidate_id")
    if not isinstance(candidate, str) or candidate not in ALLOWED_CANDIDATES:
        raise WorkerError("planner candidate is not in the fixed list")
    return candidate


def parse_choice(text: str) -> str:
    """Accept only the pre-registered UTF-8 answer bytes."""

    candidate = parse_structure(text)
    if text != EXACT_RESPONSE:
        raise WorkerError("planner answer is structurally valid but not byte-exact")
    return candidate


def _prompt_ids(tokenizer: Any) -> tuple[list[int], bytes]:
    messages = [{"role": "user", "content": PLANNER_PROMPT}]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str):
        raise WorkerError("chat template did not return rendered text")
    rendered_prompt_bytes = rendered.encode("utf-8")
    try:
        value = tokenizer.encode(rendered, add_special_tokens=False)
    except TypeError as exc:
        raise WorkerError(
            "tokenizer.encode does not support add_special_tokens=False"
        ) from exc
    if not isinstance(value, list):
        try:
            value = value.tolist()
        except AttributeError as exc:
            raise WorkerError("tokenizer returned a non-list prompt") from exc
    if not value or any(type(item) is not int for item in value):
        raise WorkerError("tokenizer returned invalid prompt token IDs")
    return list(value), rendered_prompt_bytes


def _run_worker(model_key: str) -> int:
    expected_parent = os.environ.get("FRIDAY_DUAL_PARENT_PID")
    if expected_parent != str(os.getppid()):
        raise WorkerError("worker is not owned by the registered parent")
    if os.environ.get("FRIDAY_DUAL_RUN_ID") != RUN_ID:
        raise WorkerError("worker run ID is invalid")
    if os.environ.get("FRIDAY_DUAL_MODEL_KEY") != model_key:
        raise WorkerError("worker model key is not bound by the parent")
    if model_key not in MODEL_SPECS:
        raise WorkerError("worker model key is not registered")

    # These imports are intentionally below all protocol gates: self-checks and
    # malformed direct invocations never load MLX or a model.
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))
    from _bench import resolve_local_model_snapshot
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx

    spec = MODEL_SPECS[model_key]
    snapshot = resolve_local_model_snapshot(spec["model_id"])
    expected_path = os.environ.get("FRIDAY_DUAL_SNAPSHOT_PATH")
    expected_revision = os.environ.get("FRIDAY_DUAL_SNAPSHOT_REVISION")
    expected_snapshot_sha = os.environ.get("FRIDAY_DUAL_SNAPSHOT_SHA256")
    expected_weight_sha = os.environ.get("FRIDAY_DUAL_WEIGHT_SHA256")
    expected_stat_manifest = os.environ.get("FRIDAY_DUAL_SNAPSHOT_STAT_MANIFEST")
    if not all(
        isinstance(value, str) and value
        for value in (
            expected_path,
            expected_revision,
            expected_snapshot_sha,
            expected_weight_sha,
            expected_stat_manifest,
        )
    ):
        raise WorkerError("parent snapshot binding is incomplete")
    try:
        resolved_path = str(Path(snapshot.path).resolve(strict=True))
        expected_weight_hashes = json.loads(expected_weight_sha)
        expected_stats = json.loads(expected_stat_manifest)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise WorkerError("parent snapshot binding is invalid") from exc
    if (
        snapshot.revision != spec["revision"]
        or expected_revision != spec["revision"]
        or resolved_path != expected_path
        or not isinstance(expected_weight_hashes, dict)
        or not isinstance(expected_stats, dict)
    ):
        raise WorkerError("registered model revision changed")

    resolved_path, stats_before_load = _snapshot_stat_manifest(snapshot)
    if resolved_path != expected_path or stats_before_load != expected_stats:
        raise WorkerError("child snapshot path/stat binding does not match parent")

    mx.reset_peak_memory()
    load_started_ns = time.perf_counter_ns()
    model, tokenizer = load(str(snapshot.path))
    load_finished_ns = time.perf_counter_ns()
    resolved_after_load, stats_after_load = _snapshot_stat_manifest(snapshot)
    if (
        resolved_after_load != resolved_path
        or stats_after_load != stats_before_load
        or stats_after_load != expected_stats
    ):
        raise WorkerError("model snapshot changed during load")
    prompt, rendered_prompt_bytes = _prompt_ids(tokenizer)
    sampler = make_sampler(temp=0.0)

    generation_started_ns = time.perf_counter_ns()
    first_token_ns: int | None = None
    tokens: list[int] = []
    text_parts: list[str] = []
    finish_reason: str | None = None
    previous_alarm = signal.getsignal(signal.SIGALRM)

    def generation_alarm(_signum: int, _frame: Any) -> None:
        raise WorkerError(
            "continuous generation watchdog exceeded 6 seconds before final synchronization"
        )

    try:
        signal.signal(signal.SIGALRM, generation_alarm)
        signal.setitimer(signal.ITIMER_REAL, CONTINUOUS_GPU_LIMIT_SECONDS)
        for response in stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=MAX_OUTPUT_TOKENS,
            sampler=sampler,
            prefill_step_size=PREFILL_STEP_SIZE,
        ):
            if first_token_ns is None:
                first_token_ns = time.perf_counter_ns()
            tokens.append(int(response.token))
            text_parts.append(response.text)
            finish_reason = response.finish_reason
        mx.synchronize()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_alarm)
    generation_finished_ns = time.perf_counter_ns()

    if not tokens or first_token_ns is None:
        raise WorkerError("planner generated no response")
    model_work_ns = generation_finished_ns - generation_started_ns
    ttft_ns = first_token_ns - generation_started_ns
    if model_work_ns <= 0 or ttft_ns <= 0 or ttft_ns > model_work_ns:
        raise WorkerError("worker timing is invalid")
    text = "".join(text_parts)
    token_rate = len(tokens) / (model_work_ns / 1_000_000_000)
    if not text or len(text.encode("utf-8")) > MAX_RESPONSE_BYTES:
        raise WorkerError("planner answer is empty or too large")

    _emit(
        {
            "device": str(mx.default_device()),
            "event": "complete",
            "finish_reason": finish_reason,
            "load_count": 1,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "model_key": model_key,
            "model_id": spec["model_id"],
            "model_load_ns": load_finished_ns - load_started_ns,
            "model_work_ns": model_work_ns,
            "mlx_peak_bytes": int(mx.get_peak_memory()),
            "output_tokens": len(tokens),
            "pid": os.getpid(),
            "prefill_step_size": PREFILL_STEP_SIZE,
            "prompt_sha256": PROMPT_SHA256,
            "rendered_prompt_b64": base64.b64encode(rendered_prompt_bytes).decode("ascii"),
            "rendered_prompt_sha256": hashlib.sha256(rendered_prompt_bytes).hexdigest(),
            "prompt_token_ids": prompt,
            "prompt_tokens": len(prompt),
            "rss_peak_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "sampler_temperature": 0.0,
            "snapshot_revision": snapshot.revision,
            "snapshot_path": resolved_after_load,
            "snapshot_sha256": expected_snapshot_sha,
            "snapshot_integrity": {
                "bound_snapshot_sha256": expected_snapshot_sha,
                "bound_weight_sha256": expected_weight_hashes,
                "before_load_stat_manifest": stats_before_load,
                "after_load_stat_manifest": stats_after_load,
            },
            "text": text,
            "token_rate": token_rate,
            "tokens": tokens,
            "ttft_ns": ttft_ns,
            "weight_sha256": expected_weight_hashes,
            "worker_watchdog_seconds": CONTINUOUS_GPU_LIMIT_SECONDS,
        }
    )
    return 0


def _self_check() -> int:
    assert PROMPT_SHA256 == (
        "c746eca8644a18fc75673acb9b3dbdf03825cbfba6c76faede5d909cf3d2ea0b"
    )
    assert parse_choice(EXACT_RESPONSE) == (
        EXPECTED_CANDIDATE
    )
    assert parse_structure('{"candidate_id": "batched_readback"}') == "batched_readback"
    class FakeTokenizer:
        def __init__(self) -> None:
            self.render_calls = 0
            self.encode_calls: list[tuple[str, bool]] = []

        def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
            self.render_calls += 1
            assert kwargs == {"tokenize": False, "add_generation_prompt": True}
            assert messages == [{"role": "user", "content": PLANNER_PROMPT}]
            return "rendered-prompt"

        def encode(self, value: str, *, add_special_tokens: bool) -> list[int]:
            self.encode_calls.append((value, add_special_tokens))
            return [11, 12]

    fake_tokenizer = FakeTokenizer()
    assert _prompt_ids(fake_tokenizer) == ([11, 12], b"rendered-prompt")
    assert fake_tokenizer.render_calls == 1
    assert fake_tokenizer.encode_calls == [("rendered-prompt", False)]
    try:
        parse_choice('{"candidate_id": "batched_readback"}')
    except WorkerError:
        pass
    else:
        raise AssertionError("non-exact candidate answer was accepted")
    rejected = 0
    for value in (
        "persistent_service_qualification",
        '```json\n{"candidate_id":"persistent_service_qualification"}\n```',
        '{"candidate_id":"unknown"}',
        '{"candidate_id":"persistent_service_qualification","command":"run"}',
        '{"candidate_id":"persistent_service_qualification","candidate_id":"batched_readback"}',
        "NaN",
        "",
        ' {"candidate_id":"persistent_service_qualification"}',
    ):
        try:
            parse_choice(value)
        except WorkerError:
            rejected += 1
    assert rejected == 8
    assert tuple(MODEL_SPECS) == ("1b", "4b")
    assert len(ALLOWED_CANDIDATES) == 4
    print(json.dumps({"checks": 17, "self_check": "pass"}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dual_model_planner_worker", allow_abbrev=False)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--worker", action="store_true")
    modes.add_argument("--self-check", action="store_true")
    parser.add_argument("--model-key", choices=tuple(MODEL_SPECS), default=None)
    args = parser.parse_args(argv)
    if args.self_check:
        return _self_check()
    if args.model_key is None:
        parser.error("--model-key is required with --worker")
    try:
        return _run_worker(args.model_key)
    except Exception as exc:
        _emit(
            {
                "error_type": type(exc).__name__,
                "event": "error",
                "message": str(exc)[:300],
                "model_key": args.model_key,
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
