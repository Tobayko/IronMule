#!/usr/bin/env python3
"""B24, kernel level: count the dispatches a decode step actually issues.

`Metal System Trace` alone records encoders, and one compute encoder can hold many
kernels. Adding the `Metal GPU Counters` instrument turns on the shader timeline, which
records every dispatch with its shader name and duration. That is the honest kernel
count `B10` asks for, and it also shows how much of a step is copy kernels.

Counter recording slows execution down, so this reports structure and shares, never
latency. Timings come from the counter-free runs.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b24_trace_report import (  # noqa: E402
    _export,
    _launched_pid,
    _measured_phase,
    _number,
    _rows,
)

# MLX names its copy kernels after the layout pair they move between.
COPY = re.compile(r"_copy|^copy")


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--wall", type=Path, required=True)
    parser.add_argument("--gap-ms", type=float, default=200.0)
    parser.add_argument("--matmul-calls", type=int, default=None,
                        help="quantized matmul module calls per step, counted separately")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    wall = json.loads(args.wall.read_text())
    steps = int(wall["steps"])

    pid = _launched_pid(args.trace)
    shaders = [
        row
        for row in _rows(_export(args.trace, "metal-shader-profiler-intervals"))
        if row.get("shader-type") == "Compute" and row.get("process") == pid
    ]
    measured = _measured_phase(shaders, args.gap_ms * 1e6)

    counts: Counter[str] = Counter()
    durations: defaultdict[str, float] = defaultdict(float)
    for row in measured:
        name = re.sub(r"\s*\(\d+\)$", "", row.get("name", ""))
        counts[name] += 1
        durations[name] += _number(row.get("duration", "0"))

    total_ns = sum(durations.values())
    copies = [name for name in counts if COPY.search(name)]
    copy_count = sum(counts[name] for name in copies)
    copy_ns = sum(durations[name] for name in copies)

    kernels = [
        {
            "kernel": name,
            "intervals": counts[name],
            "intervals_per_step": counts[name] / steps,
            "gpu_ns_total": durations[name],
            "share_of_gpu_time": durations[name] / total_ns if total_ns else 0.0,
            "is_copy": bool(COPY.search(name)),
        }
        for name in sorted(counts, key=lambda key: -durations[key])
    ]

    report = {
        "schema": "ironmule.b24_shader_timeline.v1",
        "experiment_id": args.out.stem,
        "agent": "claude",
        "kind": "gpu_dispatch_instrumentation",
        "status": "measured",
        "instrument": "Instruments 16.0, Metal System Trace plus Metal GPU Counters",
        "caveat": (
            "counter recording perturbs execution, so time shares are the result and absolute "
            "latency is not; an interval aggregates dispatches of one shader, so the interval "
            "count is a floor on kernels, not the dispatch count"
        ),
        "traced_pid": pid,
        "model": wall["model"],
        "decode_steps": steps,
        "shader_intervals_total": sum(counts.values()),
        "shader_intervals_per_step": sum(counts.values()) / steps,
        "quantized_matmul_calls_per_step": args.matmul_calls,
        "distinct_kernels": len(counts),
        "gpu_ns_total": total_ns,
        "copy_intervals_per_step": copy_count / steps,
        "copy_share_of_intervals": copy_count / sum(counts.values()) if counts else 0.0,
        "copy_share_of_gpu_time": copy_ns / total_ns if total_ns else 0.0,
        "kernels": kernels,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    brief = {key: value for key, value in report.items() if key != "kernels"}
    print(json.dumps(brief, indent=2, sort_keys=True))
    for kernel in kernels[:12]:
        print(f"  {kernel['share_of_gpu_time']*100:5.1f}%  {kernel['intervals_per_step']:6.1f}/step"
              f"  {'copy ' if kernel['is_copy'] else '     '}{kernel['kernel'][:64]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
