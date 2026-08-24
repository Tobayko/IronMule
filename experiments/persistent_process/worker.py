#!/usr/bin/env python3
"""Fixed-protocol worker for the persistent-process study.

The module imports only the standard library until a hardware mode is explicitly
selected.  Prompts and generation settings are closed; callers cannot inject text,
code, paths, or model identifiers through the line protocol.
"""

from __future__ import annotations

import argparse
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
OUTPUT_TOKENS = 32
PREFILL_CHUNK = 256
MAX_REQUEST_LINE_BYTES = 512

FILLER = (
    "You are a careful engineering assistant working in a Python repository. "
    "Follow the existing style and explain your reasoning briefly. "
) * 40
QUESTIONS = {
    "P": "Why is false sharing slow?",
    "Q": "What are TLB misses?",
    "R": "When does store forwarding fail?",
    "S": "Why can branch prediction fail?",
}
EXPECTED_PROMPT_TOKENS = {key: 897 for key in QUESTIONS}


class WorkerProtocolError(RuntimeError):
    """A caller or worker message does not match the frozen protocol."""


def _strict_json(line: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    return json.loads(line, parse_constant=reject_constant)


def parse_request(line: str) -> dict[str, str]:
    """Validate one closed request without importing MLX."""

    if not isinstance(line, str) or not line or len(line.encode("utf-8")) > MAX_REQUEST_LINE_BYTES:
        raise WorkerProtocolError("request line is empty or too large")
    try:
        value = _strict_json(line)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise WorkerProtocolError("request is not strict JSON") from exc
    if not isinstance(value, dict):
        raise WorkerProtocolError("request must be an object")
    command = value.get("command")
    if command == "shutdown":
        if set(value) != {"command"}:
            raise WorkerProtocolError("shutdown request has extra fields")
        return {"command": "shutdown"}
    if command != "request" or set(value) != {"command", "request_id", "prompt_key"}:
        raise WorkerProtocolError("request fields do not match the protocol")
    request_id = value.get("request_id")
    prompt_key = value.get("prompt_key")
    if (
        not isinstance(request_id, str)
        or not 1 <= len(request_id) <= 80
        or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in request_id)
    ):
        raise WorkerProtocolError("request id is invalid")
    if not isinstance(prompt_key, str) or prompt_key not in QUESTIONS:
        raise WorkerProtocolError("prompt key is invalid")
    return {"command": "request", "request_id": request_id, "prompt_key": prompt_key}


def _emit(value: dict[str, Any]) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    sys.stdout.write(payload + "\n")
    sys.stdout.flush()


class ModelRunner:
    """Load the registered model once and create a fresh KV-cache per request."""

    def __init__(self) -> None:
        sys.path.insert(0, str(PROJECT_ROOT / "tools"))
        from _bench import resolve_local_model_snapshot
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler

        snapshot = resolve_local_model_snapshot(MODEL_ID)
        if snapshot.revision != MODEL_REVISION:
            raise WorkerProtocolError("registered model revision changed")
        self.model, self.tokenizer = load(str(snapshot.path))
        self.sampler = make_sampler(temp=0.0)
        self.snapshot = snapshot
        self.load_count = 1
        self.request_count = 0
        lengths = {key: len(self._ids_for(key)) for key in QUESTIONS}
        if lengths != EXPECTED_PROMPT_TOKENS:
            raise WorkerProtocolError(f"prompt token counts changed: {lengths}")

    def _ids_for(self, key: str) -> list[int]:
        templated = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": FILLER + "\n\n" + QUESTIONS[key]}],
            add_generation_prompt=True,
        )
        value = templated if isinstance(templated, list) else self.tokenizer.encode(templated)
        if not isinstance(value, list) or not value or any(type(item) is not int for item in value):
            raise WorkerProtocolError("tokenizer returned an invalid prompt")
        return list(value)

    def ready_event(self) -> dict[str, Any]:
        return {
            "event": "ready",
            "load_count": self.load_count,
            "model_id": MODEL_ID,
            "pid": os.getpid(),
            "prompt_tokens": dict(EXPECTED_PROMPT_TOKENS),
            "snapshot_revision": self.snapshot.revision,
        }

    def answer(self, *, request_id: str, prompt_key: str) -> None:
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        ids = self._ids_for(prompt_key)
        cache = make_prompt_cache(self.model)
        self.request_count += 1
        compute_started_ns = time.perf_counter_ns()
        logits = None
        for start in range(0, len(ids), PREFILL_CHUNK):
            logits = self.model(mx.array([ids[start : start + PREFILL_CHUNK]]), cache=cache)
            mx.eval(logits)
            mx.synchronize()
        if logits is None:
            raise WorkerProtocolError("prompt unexpectedly produced no logits")
        token = self.sampler(logits[:, -1, :].astype(mx.float32))[:, None]
        mx.eval(token)
        mx.synchronize()
        first_id = int(token[0, 0])
        first_compute_ns = time.perf_counter_ns() - compute_started_ns
        _emit(
            {
                "event": "first_token",
                "first_compute_ns": first_compute_ns,
                "request_id": request_id,
                "token_id": first_id,
            }
        )

        tokens = [first_id]
        for _ in range(OUTPUT_TOKENS - 1):
            logits = self.model(token, cache=cache)
            token = self.sampler(logits[:, -1, :].astype(mx.float32))[:, None]
            mx.eval(token)
            tokens.append(int(token[0, 0]))
        mx.synchronize()
        compute_ns = time.perf_counter_ns() - compute_started_ns
        _emit(
            {
                "cache_instances": 1,
                "compute_ns": compute_ns,
                "event": "complete",
                "load_count": self.load_count,
                "mlx_peak_bytes": int(mx.get_peak_memory()),
                "pid": os.getpid(),
                "prompt_key": prompt_key,
                "prompt_tokens": len(ids),
                "request_count": self.request_count,
                "request_id": request_id,
                "rss_peak_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                "tokens": tokens,
            }
        )


def _run_once(prompt_key: str, request_id: str) -> int:
    runner = ModelRunner()
    _emit(runner.ready_event())
    runner.answer(request_id=request_id, prompt_key=prompt_key)
    return 0


def _run_server() -> int:
    runner = ModelRunner()
    _emit(runner.ready_event())
    for raw in sys.stdin:
        request = parse_request(raw.rstrip("\n"))
        if request["command"] == "shutdown":
            _emit({"event": "stopped", "pid": os.getpid(), "requests": runner.request_count})
            return 0
        runner.answer(request_id=request["request_id"], prompt_key=request["prompt_key"])
    raise WorkerProtocolError("server input closed without shutdown")


def _self_check() -> int:
    assert parse_request('{"command":"shutdown"}') == {"command": "shutdown"}
    request = parse_request('{"command":"request","prompt_key":"P","request_id":"p-1"}')
    assert request == {"command": "request", "request_id": "p-1", "prompt_key": "P"}
    rejected = 0
    for value in (
        "[]",
        '{"command":"request","prompt_key":"X","request_id":"p-1"}',
        '{"command":"request","prompt_key":"P","request_id":"P"}',
        '{"command":"shutdown","extra":1}',
        "NaN",
    ):
        try:
            parse_request(value)
        except WorkerProtocolError:
            rejected += 1
    assert rejected == 5
    print(json.dumps({"self_check": "pass", "checks": 7}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="persistent_process_worker", allow_abbrev=False)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--server", action="store_true")
    modes.add_argument("--once", choices=tuple(QUESTIONS))
    parser.add_argument("--request-id", default="once-1")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if args.self_check:
        if args.server or args.once:
            parser.error("--self-check cannot be combined with a hardware mode")
        return _self_check()
    if not args.server and args.once is None:
        parser.error("one hardware mode is required")
    if args.once is not None:
        parse_request(
            json.dumps(
                {"command": "request", "request_id": args.request_id, "prompt_key": args.once}
            )
        )
    try:
        return _run_server() if args.server else _run_once(args.once, args.request_id)
    except Exception as exc:
        _emit(
            {
                "error_type": type(exc).__name__,
                "event": "error",
                "message": str(exc)[:300],
                "pid": os.getpid(),
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
