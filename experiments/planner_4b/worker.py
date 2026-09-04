#!/usr/bin/env python3
"""Fixed one-shot Gemma-4B planner worker for the prospective study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "mlx-community/gemma-3-4b-it-4bit"
MODEL_REVISION = "93724907d4ed1745d2fe50baadf3b0b01a65abf2"
RUN_ID = "planner-4b-validation-20260824-01"
MAX_OUTPUT_TOKENS = 32
PREFILL_STEP_SIZE = 256
EXPECTED_CANDIDATE = "persistent_service_qualification"
ALLOWED_CANDIDATES = (
    EXPECTED_CANDIDATE,
    "batched_readback",
    "host_readback_upper_bound",
    "kv_cache_preallocation_ab",
)

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
    """The fixed worker cannot safely complete its closed protocol."""


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


def parse_choice(text: str) -> str:
    """Parse the full model answer; surrounding prose and extra fields fail."""

    if not isinstance(text, str) or not text or len(text.encode("utf-8")) > 512:
        raise WorkerError("planner answer size is invalid")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON field")
            value[key] = item
        return value

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


def _prompt_ids(tokenizer: Any) -> list[int]:
    templated = tokenizer.apply_chat_template(
        [{"role": "user", "content": PLANNER_PROMPT}],
        add_generation_prompt=True,
    )
    value = templated if isinstance(templated, list) else tokenizer.encode(templated)
    if not isinstance(value, list) or not value or any(type(item) is not int for item in value):
        raise WorkerError("tokenizer returned an invalid prompt")
    return list(value)


def run_worker() -> int:
    expected_parent = os.environ.get("FRIDAY_PLANNER_4B_PARENT_PID")
    if expected_parent != str(os.getppid()):
        raise WorkerError("worker is not owned by the registered parent")
    if os.environ.get("FRIDAY_PLANNER_4B_RUN_ID") != RUN_ID:
        raise WorkerError("worker run id is invalid")

    sys.path.insert(0, str(PROJECT_ROOT / "tools"))
    from _bench import resolve_local_model_snapshot
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    snapshot = resolve_local_model_snapshot(MODEL_ID)
    if snapshot.revision != MODEL_REVISION:
        raise WorkerError("registered model revision changed")
    model, tokenizer = load(str(snapshot.path))
    prompt = _prompt_ids(tokenizer)
    sampler = make_sampler(temp=0.0)

    import mlx.core as mx

    started_ns = time.perf_counter_ns()
    tokens: list[int] = []
    text_parts: list[str] = []
    finish_reason: str | None = None
    for response in stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=MAX_OUTPUT_TOKENS,
        sampler=sampler,
        prefill_step_size=PREFILL_STEP_SIZE,
    ):
        tokens.append(int(response.token))
        text_parts.append(response.text)
        finish_reason = response.finish_reason
    mx.synchronize()
    compute_ns = time.perf_counter_ns() - started_ns
    if not tokens:
        raise WorkerError("planner generated no response")
    text = "".join(text_parts)
    _emit(
        {
            "compute_ns": compute_ns,
            "event": "complete",
            "finish_reason": finish_reason,
            "load_count": 1,
            "mlx_peak_bytes": int(mx.get_peak_memory()),
            "model_id": MODEL_ID,
            "output_tokens": len(tokens),
            "pid": os.getpid(),
            "prompt_sha256": PROMPT_SHA256,
            "prompt_tokens": len(prompt),
            "rss_peak_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "snapshot_revision": snapshot.revision,
            "text": text,
            "tokens": tokens,
        }
    )
    return 0


def _self_check() -> int:
    assert parse_choice('{"candidate_id":"persistent_service_qualification"}') == (
        EXPECTED_CANDIDATE
    )
    assert parse_choice('{"candidate_id": "batched_readback"}') == "batched_readback"
    rejected = 0
    for value in (
        "persistent_service_qualification",
        '```json\n{"candidate_id":"persistent_service_qualification"}\n```',
        '{"candidate_id":"unknown"}',
        '{"candidate_id":"persistent_service_qualification","command":"run"}',
        '{"candidate_id":"persistent_service_qualification","candidate_id":"batched_readback"}',
        "NaN",
        "",
    ):
        try:
            parse_choice(value)
        except WorkerError:
            rejected += 1
    assert rejected == 7
    assert len(ALLOWED_CANDIDATES) == 4
    assert len(PROMPT_SHA256) == 64
    print(json.dumps({"checks": 11, "self_check": "pass"}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="planner_4b_worker", allow_abbrev=False)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--worker", action="store_true")
    modes.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if args.self_check:
        return _self_check()
    try:
        return run_worker()
    except Exception as exc:
        _emit(
            {
                "error_type": type(exc).__name__,
                "event": "error",
                "message": str(exc)[:300],
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
