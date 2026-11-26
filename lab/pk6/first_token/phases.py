"""Join PR 57 streaming requests to native prefill calls using request and KV sequence IDs."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import median

from lab.pk6.replay_http import checksum, waves
from lab.pk6.warm_result import check_timeline

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "lab/autonomous/results"
HISTORICAL = RESULTS / "a100-pr57-warm-repeat"
JOB_TIMES = ("seen", "prefill_start", "prefill_end", "first", "end")
PHASES = ("send_to_seen", "seen_to_prefill", "prefill", "prefill_to_first", "first_to_client_content")


def clients(directory, label):
    path = directory / f"timeline-{label}.jsonl"
    engine = "vllm" if label == "vllm" else "ironmule"
    check_timeline(path, label, engine, waves(directory, label, engine, 32))
    records = [json.loads(line) for line in path.read_bytes().splitlines()]
    if (len({r["server_id"] for r in records}) != len(records)
            or any(not isinstance(r["server_id"], str) or not r["server_id"] for r in records)):
        raise ValueError("duplicate or missing HTTP request ID")
    return records


def project(trace, records):
    """Keep only matched jobs and prompt calls within counted waves, without text or host paths."""
    ids = {r["server_id"] for r in records}
    windows = [(r[0]["wave_start"], max(v["t_done"] for v in r))
               for r in (records[i:i + 32] for i in range(0, len(records), 32))]
    jobs, calls = {}, []
    startup = None
    for line in trace.read_text().splitlines():
        if line.startswith("# init result "):
            if calls:
                raise ValueError("engine reinitialised after measured service began")
            info = json.loads(line.removeprefix("# init result "))
            startup = {"pilot": info.get("pilot"), "argmax_only": info.get("argmax_only"),
                       "kg_tokens": info.get("kg_tokens"), "pad_to": info.get("pad_to"),
                       "pk2_min_rows": info.get("pk2", {}).get("min_rows")}
        elif line.startswith("# job "):
            fields = dict(token.split("=", 1) for token in line.split()[2:])
            tag = fields["tag"]
            if tag in ids:
                if tag in jobs:
                    raise ValueError("duplicate native request ID")
                jobs[tag] = {k: float(fields[k]) for k in JOB_TIMES}
                jobs[tag].update({k: int(fields[k]) for k in ("seq", "prompt", "tokens", "reused")})
                jobs[tag]["reason"] = fields["reason"]
        elif line and line[0].isdigit():
            t, seconds, rows, width, logits, prompt, decode, filler, seqs, *rest = line.split()
            start = float(t)
            if int(prompt) and any(lo <= start <= hi for lo, hi in windows):
                calls.append({"start": start, "seconds": float(seconds), "rows": int(rows), "width": int(width),
                              "logits": int(logits), "prompt": int(prompt), "decode": int(decode), "filler": int(filler),
                              "seqs": [int(n) for n in seqs.removeprefix("seqs=").split(",")]})
    if jobs.keys() != ids:
        raise ValueError("native trace does not cover the HTTP request IDs")
    if not startup or not startup["pilot"] or not startup["pk2_min_rows"]:
        raise ValueError("missing active Pilot or PK2 selection")
    return {"trace_sha256": checksum(trace), "serving_profile": startup, "jobs": jobs, "calls": calls}


def summarize(records, native=None):
    if native is not None and set(native["jobs"]) != {r["server_id"] for r in records}:
        raise ValueError("native projection does not match the client request IDs")
    by_wave, used = [], set()
    for i in range(0, len(records), 32):
        group = records[i:i + 32]
        lengths = Counter(r["prompt_tokens"] for r in group)
        if (lengths != {33: 16, 444: 16} or any(r["tokens"] != 32
                or r["prompt_tokens"] != (444 if r["index"] % 2 else 33) for r in group)):
            raise ValueError("different input or output work")
        ttfts = [r["ttft_ms"] for r in group]
        entry = {"repeat": group[0]["repeat"], "requests": len(group), "p50_ms": round(median(ttfts), 1),
                 "p95_ms": round(sorted(ttfts)[30], 1), "max_ms": round(max(ttfts), 1),
                 "p50_by_prompt_ms": {str(n): round(median(r["ttft_ms"] for r in group
                                                           if r["prompt_tokens"] == n), 1) for n in sorted(lengths)}}
        if all(len(r["chunk_times"]) >= 2 for r in group):
            second = [1000 * (r["chunk_times"][1] - r["t_send"]) for r in group]
            gaps = [1000 * max(b - a for a, b in zip(r["chunk_times"], r["chunk_times"][1:])) for r in group]
            entry["second_content_p50_ms"] = round(median(second), 1)
            entry["longest_content_gap_p95_ms"] = round(sorted(gaps)[30], 1)
        if native is not None:
            phases, calls = [], {}
            for row in group:
                job = native["jobs"][row["server_id"]]
                stamps = [row["t_send"], job["seen"], job["prefill_start"], job["prefill_end"],
                          job["first"], row["chunk_times"][0]]
                if (not all(map(math.isfinite, stamps)) or stamps != sorted(stamps)
                        or not job["first"] <= job["end"] <= row["t_done"] + .000001
                        or (job["prompt"], job["tokens"], job["reason"]) != (row["prompt_tokens"], 32, "length")
                        or not 0 <= job["reused"] < job["prompt"]):
                    raise ValueError("inconsistent native and HTTP clocks or work")
                phases.append([1000 * (b - a) for a, b in zip(stamps, stamps[1:])])
                calls.setdefault(job["prefill_start"], []).append(job)
            shapes = []
            for start, members in sorted(calls.items()):
                matches = [c for c in native["calls"] if c["start"] == start]
                if len(matches) != 1 or start in used:
                    raise ValueError("missing, duplicate or reused native call")
                call = matches[0]
                if (not 0 < call["seconds"] < math.inf
                        or any(abs(j["prefill_end"] - (start + call["seconds"])) > .0000011 for j in members)
                        or sorted(j["seq"] for j in members) != call["seqs"]
                        or len(set(call["seqs"])) != len(members)
                        or call["width"] != len(members) or call["logits"] != len(members)
                        or call["decode"] or call["filler"] or call["rows"] != call["prompt"]
                        or call["prompt"] != sum(j["prompt"] - j["reused"] for j in members)):
                    raise ValueError("native call disagrees with its completed jobs")
                used.add(start)
                shapes.append({"width": len(members), "prompt_rows": call["prompt"],
                               "short": sum(j["prompt"] == 33 for j in members),
                               "elapsed_ms": round(1000 * call["seconds"], 1)})
            entry["prefill_calls"] = shapes
            entry["median_phase_ms"] = {k: round(median(col), 1) for k, col in zip(PHASES, zip(*phases), strict=True)}
        by_wave.append(entry)
    if native is not None and len(used) != len(native["calls"]):
        raise ValueError("unaccounted native prompt call")
    return by_wave


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--series", type=Path, default=HISTORICAL)
    p.add_argument("--projections", required=True, type=Path)
    p.add_argument("--traces", type=Path, help="extract projections from original private traces")
    p.add_argument("--labels", nargs="+", default=["K1", "vllm", "K2"])
    args = p.parse_args()
    result = {}
    for label in args.labels:
        records = clients(args.series, label)
        native = None
        if label != "vllm":
            path = args.projections / f"native-{label}.json"
            if args.traces:
                native = project(args.traces / f"trace-{label}.txt", records)
            else:
                native = json.loads(path.read_bytes())
            if label in ("K1", "K2"):
                receipt = json.loads((RESULTS / f"a100-dynamic-calibration/init-{label}.json").read_bytes())
                if native["trace_sha256"] != receipt["trace_sha256"]:
                    raise ValueError("trace differs from the earlier startup receipt")
        result[label] = summarize(records, native)
        if args.traces and native:
            args.projections.mkdir(exist_ok=True, parents=True)
            path.write_text(json.dumps(native, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
