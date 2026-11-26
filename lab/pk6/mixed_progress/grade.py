"""Audit fixed-work HTTP results by width and seed. Missing observations remain unverified."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import median

from lab.pk6.mixed_progress.trial import ARMS, SEEDS
from lab.pk6.replay_http import checksum
from lab.pk6.warm_repeat import GGUF
from lab.portable_gpu.fetch_awq import REVISION, _manifest_digest

MEASURES = ("tok_s", "first", "visible", "first_p95", "eighth", "gap_p95", "end_p95")


def lines(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def percentile95(values):
    return sorted(values)[math.ceil(0.95 * len(values)) - 1]


def measure(wave):
    rows = wave["requests"]
    if (
        len(rows) != wave["width"]
        or len({r["server_id"] for r in rows}) != len(rows)
        or not all(r["server_id"] for r in rows)
        or not 0 < wave["elapsed_s"] < math.inf
    ):
        raise ValueError("lost requests or invalid elapsed time")
    for r in rows:
        ts = r["chunk_times"]
        if (
            not ts
            or not all(map(math.isfinite, ts))
            or sorted(ts) != ts
            or not 0 < r["t_send"] <= ts[0] <= ts[-1] <= r["t_done"]
            or not 0 < r["ttft_ms"] < math.inf
            or abs((ts[0] - r["t_send"]) * 1000 - r["ttft_ms"]) > 5
            or not 0 < r["tokens"] <= 32
        ):
            raise ValueError("inconsistent stream timestamps or work")
    if (
        max(r["t_done"] for r in rows) - min(r["t_send"] for r in rows)
        > wave["elapsed_s"] + 0.001
    ):
        raise ValueError("HTTP elapsed time excludes requests")
    return {
        "tok_s": sum(r["tokens"] for r in rows) / wave["elapsed_s"],
        "first": median(r["ttft_ms"] for r in rows) / 1000,
        "visible": median(r["visible_ttft_ms"] for r in rows) / 1000
        if all(r["visible_ttft_ms"] for r in rows)
        else None,
        "first_p95": percentile95([r["ttft_ms"] / 1000 for r in rows]),
        "eighth": median(r["chunk_times"][7] - r["t_send"] for r in rows)
        if all(len(r["chunk_times"]) >= 8 for r in rows)
        else None,
        "gap_p95": percentile95(
            [
                max(b - a for a, b in zip(r["chunk_times"], r["chunk_times"][1:]))
                for r in rows
            ]
        )
        if all(len(r["chunk_times"]) >= 2 for r in rows)
        else None,
        "end_p95": percentile95([r["t_done"] - r["t_send"] for r in rows]),
    }


def bindings(folder, arm, waves):
    """Project native prompt/decode calls only from the counted, ID-matched HTTP windows."""
    path = folder / f"trace-{arm}.txt"
    ids = {r["server_id"] for w in waves for r in w["requests"]}
    jobs, mixed = {}, []
    for line in path.read_text().splitlines():
        if line.startswith("# job "):
            fields = dict(x.split("=", 1) for x in line.split()[2:])
            tag = fields["tag"]
            if tag in ids:
                if tag in jobs:
                    raise ValueError("duplicate native request")
                jobs[tag] = {
                    "prompt": int(fields["prompt"]),
                    "seq": int(fields["seq"]),
                    "start": float(fields["seen"]),
                    "end": float(fields["end"]),
                }
        elif line and line[0].isdigit():
            t, seconds, _, _, _, prompt, decode, _, seqs, *_ = line.split()
            if int(prompt) and int(decode):
                mixed.append(
                    (
                        float(t),
                        float(seconds),
                        int(prompt),
                        int(decode),
                        {int(n) for n in seqs.removeprefix("seqs=").split(",")},
                    )
                )
    if set(jobs) != ids:
        raise ValueError("unmatched HTTP/native ID")
    cells = []
    for w in waves:
        rows = w["requests"]
        spans = [jobs[r["server_id"]] for r in rows]
        if any(
            s["prompt"] != r["prompt_tokens"]
            or not r["t_send"] <= s["start"] <= s["end"] <= r["t_done"] + 0.000001
            for r, s in zip(rows, spans)
        ):
            raise ValueError("native identity, work or request clock differs")
        lo, hi = min(r["t_send"] for r in rows), max(r["t_done"] for r in rows)
        calls = []
        for t, seconds, p, d, seqs in mixed:
            if lo <= t <= hi:
                joined = {
                    s["seq"]
                    for s in spans
                    if s["start"] <= t and t + seconds <= s["end"] + 0.000002
                }
                if not seqs <= joined:
                    raise ValueError(
                        "mixed call includes a sequence outside the HTTP wave"
                    )
                calls.append({"prompt_rows": p, "decode_rows": d, "seconds": seconds})
        cells.append(
            {
                "width": w["width"],
                "seed": w["seed"],
                "repeat": w["repeat"],
                "matched": len(rows),
                "mixed": calls,
            }
        )
    return {
        "trace_sha256": checksum(path),
        "waves_sha256": checksum(folder / f"waves-{arm}.jsonl"),
        "cells": cells,
    }


def paired(a, b):
    if len(a) != len(b) or len(a) != 3:
        raise ValueError("incomplete paired waves")
    ratios = {k: [] for k in MEASURES}
    exact = []
    valid = True
    for x, y in zip(a, b, strict=True):
        if (x["seed"], x["width"], x["repeat"], x["input_digest"]) != (
            y["seed"],
            y["width"],
            y["repeat"],
            y["input_digest"],
        ):
            raise ValueError("paired inputs or wave order differ")
        valid &= all(
            r["tokens"] == 32 and r["finish_reason"] == "length"
            for wave in (x, y)
            for r in wave["requests"]
        )
        valid &= [r["prompt_tokens"] for r in x["requests"]] == [
            r["prompt_tokens"] for r in y["requests"]
        ]
        exact.append(
            all(
                p["output_sha256"] == q["output_sha256"]
                for p, q in zip(x["requests"], y["requests"], strict=True)
            )
        )
        m, n = measure(x), measure(y)
        for k in MEASURES:
            ratios[k].append(n[k] / m[k] if m[k] and n[k] else None)
    return {
        "same_fixed_work": bool(valid),
        "text_digests_match": exact,
        "candidate_over_reference": {
            k: median(v) if None not in v else None for k, v in ratios.items()
        },
    }


def summarize(cells):
    return {
        k: median([m[k] for m in cells])
        if all(m[k] is not None for m in cells)
        else None
        for k in MEASURES
    }


def telemetry_samples(path):
    telemetry = []
    for dt, util, mem, power in csv.reader(path.read_text().splitlines()):
        t = (
            datetime.strptime(dt.strip(), "%Y/%m/%d %H:%M:%S.%f")
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )
        telemetry.append((t, float(mem), float(power)))
    return telemetry


def sampled_resources(telemetry, rows):
    samples = [
        (m, p)
        for t, m, p in telemetry
        if rows[0]["unix_start"] <= t <= rows[-1]["unix_end"]
    ]
    return {
        "samples": len(samples),
        "peak_gpu_mib": max(m for m, _ in samples) if samples else None,
        "mean_sampled_watts": sum(p for _, p in samples) / len(samples)
        if samples
        else None,
    }


def grade(folder):
    read = lambda name: json.loads((folder / name).read_bytes())
    target = read("target.json")
    if (
        target["hardware"]["name"] != "NVIDIA L40S"
        or target["source"] != "7fa0e4d8974c190192f056f6bc5b30dc72b82571"
        or target["engine_sha256"]
        != "a1985d5af074571d1cafd59e918f95f22f1bd16754d2229488da49d2d69288fd"
        or target["rows"] != 2048
        or target["gguf_sha256"] != GGUF
        or target["awq_revision"] != REVISION
        or target["awq_manifest_sha256"] != _manifest_digest()
        or target["vllm_version"] != "0.11.0"
    ):
        raise ValueError("foreign hardware, model or backend")
    complete = read("completed.json")
    if complete["order"] != list(ARMS) or complete["hardware"] != target["hardware"]:
        raise ValueError("incomplete run or changed device")
    data, stats, resources, quality, starts, proof = {}, {}, {}, {}, {}, {}
    prior = 0.0
    telemetry = telemetry_samples(folder / "device-load.csv")
    for arm in ARMS:
        start, ready, end = (read(f"{s}-{arm}.json") for s in ("start", "ready", "end"))
        if (
            start["hardware"] != target["hardware"]
            or end["hardware"] != target["hardware"]
            or not prior <= start["unix"] < end["unix"]
        ):
            raise ValueError("overlapping processes or hardware change")
        starts[arm] = ready["startup_s"]
        prior = end["unix"]
        rows = lines(folder / f"waves-{arm}.jsonl")
        seeds = SEEDS if arm == "vllm" else (SEEDS[int(arm[1]) - 1],)
        expected = [
            (s, i, w)
            for s in seeds
            for i in range(4)
            for w in ((1, 8, 32) if i < 3 else (32,))
        ]
        if [(r["seed"], r["repeat"], r["width"]) for r in rows] != expected or any(
            r["arm"] != arm for r in rows
        ):
            raise ValueError("missing or unexpected measured wave")
        if any(
            not start["unix"] <= r["unix_start"] < r["unix_end"] <= end["unix"]
            for r in rows
        ):
            raise ValueError("wave outside measured start")
        data[arm] = rows
        quality[arm] = {}
        for r in rows:
            measure(r)
            if r["repeat"] == 3:
                quality[arm][str(r["seed"])] = [
                    sum(x["correct"] is True for x in r["requests"]),
                    len(r["requests"]),
                ]
        stats[arm] = {}
        for width in (1, 8, 32):
            ms = [measure(w) for w in rows if w["repeat"] < 3 and w["width"] == width]
            stats[arm][str(width)] = summarize(ms)
        resources[arm] = sampled_resources(telemetry, rows)
        if arm != "vllm":
            if ready["health"]["gate"]["passes"] is not True:
                raise ValueError("native product gate failed")
            native = read(f"native-{arm}.json")
            if native["waves_sha256"] != checksum(folder / f"waves-{arm}.jsonl") or len(
                native["cells"]
            ) != len(rows):
                raise ValueError("wrong native timing projection")
            for c, w in zip(native["cells"], rows, strict=True):
                if (c["seed"], c["repeat"], c["width"], c["matched"]) != (
                    w["seed"],
                    w["repeat"],
                    w["width"],
                    len(w["requests"]),
                ):
                    raise ValueError("native wave disagrees with client")
            proof[arm] = {
                "performance_waves_with_mixed_calls": dict(
                    Counter(
                        str(c["width"])
                        for c in native["cells"]
                        if c["mixed"] and c["repeat"] < 3
                    )
                ),
                "quality_waves_with_mixed_calls": sum(
                    bool(c["mixed"]) for c in native["cells"] if c["repeat"] == 3
                ),
            }
    comparisons = {}
    for i, seed in enumerate(SEEDS, 1):
        for ref in ("vllm", f"R{i}"):
            cand = f"C{i}"
            q1, q2 = (
                next(w for w in data[arm] if w["seed"] == seed and w["repeat"] == 3)
                for arm in (ref, cand)
            )
            if q1["input_digest"] != q2["input_digest"] or [
                x["case"] for x in q1["requests"]
            ] != [x["case"] for x in q2["requests"]]:
                raise ValueError("different quality tasks")
            for width in (1, 8, 32):

                def block(arm):
                    return [
                        w
                        for w in data[arm]
                        if w["seed"] == seed and w["width"] == width and w["repeat"] < 3
                    ]

                comparisons[f"{cand}/{ref}/{width}"] = paired(block(ref), block(cand))

    def clear(row):
        return row["same_fixed_work"] and all(
            v is not None and (v >= 1.1 if k == "tok_s" else v <= 0.9)
            for k, v in row["candidate_over_reference"].items()
        )

    candidate_quality = all(
        q == [32, 32] for a in ("C1", "C2") for q in quality[a].values()
    )
    all_perf = all(clear(row) for key, row in comparisons.items() if "/vllm/" in key)
    native_gain = all(
        row["same_fixed_work"]
        and row["candidate_over_reference"]["tok_s"] >= 1.0
        and row["candidate_over_reference"]["first_p95"] <= 1.0
        for key, row in comparisons.items()
        if "/R" in key
    )
    exercised = all(
        proof[arm]["performance_waves_with_mixed_calls"].get("32", 0) > 0
        and proof[arm]["quality_waves_with_mixed_calls"] > 0
        for arm in ("C1", "C2")
    )
    return {
        "target": target,
        "startup_s": starts,
        "median_by_arm_width": stats,
        "quality": quality,
        "resources": resources,
        "native_mechanism": proof,
        "comparisons": comparisons,
        "full_candidate_quality": candidate_quality,
        "clear_performance_win_at_all_widths": all_perf,
        "measured_mixed_performance_and_quality": exercised,
        "preserves_incumbent_in_all_blocks": native_gain,
        "decision": "holdout_required"
        if candidate_quality and all_perf and native_gain and exercised
        else "retain_reference",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument(
        "--project",
        action="store_true",
        help="create native projections from original traces",
    )
    args = p.parse_args()
    if args.project:
        for arm in ARMS[1:]:
            wave = lines(args.folder / f"waves-{arm}.jsonl")
            (args.folder / f"native-{arm}.json").write_text(
                json.dumps(bindings(args.folder, arm, wave)) + "\n"
            )
    print(json.dumps(grade(args.folder), indent=2, allow_nan=False))
