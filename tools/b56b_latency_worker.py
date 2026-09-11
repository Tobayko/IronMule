#!/usr/bin/env python3
"""One loaded model that serves a single latency request when told to, and not before.

A separate OS process, because that is the only way to submit work to the GPU that is
independent of another process's submission without sharing any Python state. It loads
once, reports ready, then waits on stdin for an absolute `perf_counter_ns` deadline.

`time.perf_counter_ns` is system-wide on macOS, so the timestamps this process reports
are directly comparable with the parent's. Nothing here derives a latency: every number
is a stamp taken at the moment it names.
"""

from __future__ import annotations

import json
import os
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ironmule.plans import StrictOneShotPlan  # noqa: E402
from ironmule.service import InteractiveMode, Request, Runtime  # noqa: E402


def _resources() -> dict[str, int]:
    """This process's own cost. `ru_maxrss` is bytes on macOS and needs no subprocess,
    so it can be read next to a measurement without being part of one."""
    import mlx.core as mx

    return {"peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "mlx_peak_memory_bytes": int(mx.get_peak_memory()),
            "mlx_active_memory_bytes": int(mx.get_active_memory())}


def _wait_until(deadline_ns: int) -> None:
    """Sleep to within two milliseconds, then spin. The deadline is the arrival."""
    while True:
        remaining = deadline_ns - time.perf_counter_ns()
        if remaining <= 0:
            return
        if remaining > 2_000_000:
            time.sleep((remaining - 2_000_000) / 1e9)


def main() -> int:
    model_id = sys.argv[1]
    load_started = time.perf_counter_ns()
    runtime = Runtime.load(model_id, mode=InteractiveMode())
    prompts = json.loads(sys.argv[2])
    prompt_ids = [runtime.encode(prompt) for prompt in prompts]
    print(json.dumps({"ready": True, "load_ns": time.perf_counter_ns() - load_started,
                      "pid": os.getpid(), "resources": _resources()}), flush=True)

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            command = json.loads(line)
            if command.get("cmd") == "quit":
                break
            if command.get("cmd") == "stats":
                print(json.dumps({"resources": _resources()}), flush=True)
                continue
            if command.get("cmd") != "serve":
                print(json.dumps({"error": "unknown command"}), flush=True)
                continue

            request = Request(prompt_ids=list(prompt_ids[command["prompt_index"]]),
                              max_tokens=command["max_tokens"],
                              plan=StrictOneShotPlan())
            deadline = command.get("at_ns")
            if deadline is not None:
                _wait_until(deadline)
            dispatch_ns = time.perf_counter_ns()
            result = runtime.serve([request])[0]
            returned_ns = time.perf_counter_ns()
            metrics = runtime.telemetry.requests[0]
            print(json.dumps({
                "rid": result.rid,
                "tokens": list(result.tokens),
                "stop_reason": result.stop_reason,
                "requested_at_ns": deadline,
                "dispatch_ns": dispatch_ns,
                "arrival_ns": metrics.arrival_ns,
                "engine_start_ns": metrics.engine_start_ns,
                "first_token_ns": metrics.first_token_ns,
                "finished_ns": metrics.finished_ns,
                "returned_ns": returned_ns,
                "generated_tokens": metrics.generated_tokens,
                "fell_back": metrics.fell_back,
                "fallbacks": runtime.telemetry.fallbacks,
                "resources": _resources(),
            }), flush=True)
    finally:
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
