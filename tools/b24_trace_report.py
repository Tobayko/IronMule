#!/usr/bin/env python3
"""B24: read real GPU time out of a Metal System Trace instead of inferring it.

`LIMITS.md` retired kernel counts because MLX exposes no dispatch counter, and
`completion_wait` is a wait, not GPU time. Instruments records both: every command
buffer submission carries its encoder count, and the GPU timeline carries the interval
the device actually ran. This reads those tables and reports the decode phase.

The measured phase is everything after the workload's quiet gap, so the model load and
the warmup steps stay out of the numbers.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

TABLES = (
    "metal-application-command-buffer-submissions",
    "metal-gpu-intervals",
)


def _export(trace: Path, schema: str) -> str:
    xpath = f'/trace-toc/run[@number="1"]/data/table[@schema="{schema}"]'
    result = subprocess.run(
        ["xcrun", "xctrace", "export", "--input", str(trace), "--xpath", xpath],
        capture_output=True, text=True, check=True,
    )
    return result.stdout


def _rows(xml: str) -> list[dict[str, str]]:
    """xctrace emits each value once and references it by id afterwards."""

    root = ET.fromstring(xml)
    node = root.find("node")
    schema = node.find("schema")
    columns = [col.find("mnemonic").text for col in schema.findall("col")]
    seen: dict[str, str] = {}
    rows = []
    for row in node.findall("row"):
        values = {}
        for column, element in zip(columns, list(row)):
            reference = element.get("ref")
            if reference is not None:
                values[column] = seen.get(reference, "")
                continue
            identifier = element.get("id")
            text = element.text if element.text and element.text.strip() else ""
            if not text and element.tag == "process":
                pid = element.find("pid")
                text = pid.text if pid is not None else ""
            if identifier is not None:
                seen[identifier] = text
            values[column] = text
        rows.append(values)
    return rows


def _launched_pid(trace: Path) -> str:
    """The workload's own pid; the Metal tables carry every process on the machine."""

    result = subprocess.run(
        ["xcrun", "xctrace", "export", "--input", str(trace), "--toc"],
        capture_output=True, text=True, check=True,
    )
    root = ET.fromstring(result.stdout)
    for process in root.iter("process"):
        if process.get("type") == "launched":
            return process.get("pid", "")
    raise SystemExit("no launched process in trace")


def _number(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _measured_phase(rows: list[dict], gap_ns: float) -> list[dict]:
    """Split on the workload's quiet gap and keep what follows it."""

    ordered = sorted(rows, key=lambda row: _number(row.get("start", "0")))
    split = 0
    previous = None
    for index, row in enumerate(ordered):
        start = _number(row.get("start", "0"))
        if previous is not None and start - previous > gap_ns:
            split = index
        previous = start
    return ordered[split:]


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--wall", type=Path, required=True, help="workload JSON")
    parser.add_argument("--gap-ms", type=float, default=200.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    wall = json.loads(args.wall.read_text())
    gap_ns = args.gap_ms * 1e6

    pid = _launched_pid(args.trace)
    submitted = [
        row
        for row in _rows(_export(args.trace, TABLES[0]))
        if row.get("process") == pid and row.get("event-type") == "CommandBufferSubmission"
    ]
    submissions = _measured_phase(submitted, gap_ns)
    # The GPU timeline carries no pid; command buffer ids tie its rows back to ours, and
    # MLX runs on the compute channel while the window server owns vertex and fragment.
    identifiers = {row.get("cmdbuffer-id") for row in submissions if row.get("cmdbuffer-id")}
    intervals = [
        row
        for row in _rows(_export(args.trace, TABLES[1]))
        if row.get("channel-name") == "Compute" and row.get("cmdbuffer-id") in identifiers
    ]
    intervals.sort(key=lambda row: _number(row.get("start", "0")))

    encoders = [int(_number(row.get("num-encoders", "0"))) for row in submissions]
    encoder_time = [_number(row.get("encoder-time", "0")) for row in submissions]
    gpu_duration = [_number(row.get("duration", "0")) for row in intervals]
    latency = [_number(row.get("start-latency", "0")) for row in intervals]

    steps = int(wall["steps"])
    span_ns = 0.0
    if intervals:
        first = _number(intervals[0].get("start", "0"))
        last = _number(intervals[-1].get("start", "0")) + gpu_duration[-1]
        span_ns = last - first

    report = {
        "schema": "ironmule.b24_metal_system_trace.v1",
        "experiment_id": args.out.stem,
        "agent": "claude",
        "kind": "gpu_counter_instrumentation",
        "status": "measured",
        "instrument": "Instruments 16.0, Metal System Trace",
        "traced_pid": pid,
        "gpu_channel": "Compute",
        "model": wall["model"],
        "decode_steps": steps,
        "command_buffers": len(submissions),
        "command_buffers_per_step": len(submissions) / steps if steps else 0.0,
        "encoders_total": sum(encoders),
        "encoders_per_step": sum(encoders) / steps if steps else 0.0,
        "encoder_time_ns_total": sum(encoder_time),
        "encoder_time_ns_per_step": sum(encoder_time) / steps if steps else 0.0,
        "gpu_intervals": len(intervals),
        "gpu_time_ns_total": sum(gpu_duration),
        "gpu_time_ns_per_step": sum(gpu_duration) / steps if steps else 0.0,
        "cpu_to_gpu_latency_ns_mean": sum(latency) / len(latency) if latency else 0.0,
        "gpu_span_ns": span_ns,
        "gpu_busy_fraction_of_span": sum(gpu_duration) / span_ns if span_ns else 0.0,
        "wall_total_ns_per_step": wall["total_ns_mean"],
        "wall_submit_ns_per_step": wall["submit_ns_mean"],
        "wall_wait_ns_per_step": wall["wait_ns_mean"],
    }
    per_step = report["gpu_time_ns_per_step"]
    report["gpu_share_of_step"] = per_step / wall["total_ns_mean"] if wall["total_ns_mean"] else 0.0
    report["host_only_ns_per_step"] = wall["total_ns_mean"] - per_step
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
