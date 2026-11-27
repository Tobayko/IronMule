"""Locate the native row counts where real HTTP requests spent GPU time."""

import argparse
from collections import Counter
import gzip
from pathlib import Path

from lab.pk6.mixed_progress import grade
from lab.pk6.native_dispatch.score import waves, read
from lab.pk6.warm_repeat import save


def profile(primary, arm):
    rows = waves(primary, arm)
    joined = grade.bindings(primary, arm, rows)
    raw = primary / f"trace-{arm}.txt"
    contents = (
        raw.read_bytes()
        if raw.exists()
        else gzip.decompress(raw.with_suffix(".txt.gz").read_bytes())
    )
    spans = [
        (
            min(q["t_send"] for q in w["requests"]),
            max(q["t_done"] for q in w["requests"]),
            w["repeat"] == 3,
        )
        for w in rows
    ]
    seconds, count = [Counter(), Counter()], [Counter(), Counter()]
    for l in contents.decode().splitlines():
        if not l or not l[0].isdigit():
            continue
        time, elapsed, size = l.split()[:3]
        for lo, hi, quality in spans:
            if lo <= float(time) <= hi:
                seconds[quality][int(size)] += float(elapsed)
                count[quality][int(size)] += 1
                break
    return seconds, count, joined["trace_sha256"]


def order(primary, prior):
    meta, grid = read(primary / "target.json"), read(prior)
    if meta["hardware"] != grid["hardware"] or meta["gguf_sha256"] != grid["gguf"]:
        raise ValueError("unmatched grid and HTTP target")
    seconds, count, trace = profile(primary, "R1")

    # Spend at most 22 new row probes per geometry. Retain a quality-workload slice too.
    def largest(group, n):
        return [w for w, _ in group.most_common() if 8 <= w <= 512][:n]

    widths = sorted(set(largest(seconds[0], 18) + largest(seconds[1], 4)))
    old = {k for _, _, lo, hi, _ in grid["ranges"] for k in range(lo, hi + 1)}
    return {
        "hardware": meta["hardware"],
        "gguf": grid["gguf"],
        "reference_cuda": grid["reference_cuda"],
        "widths": widths,
        "source_trace": trace,
        "source_arm": "R1",
        "prior_ranges": grid["ranges"],
        "prior_http_calls": [sum(c[w] for w in old) for c in count],
        "total_http_calls": [sum(c.values()) for c in count],
        "proposed_fraction_of_observed_call_time": [
            sum(c[w] for w in widths) / sum(c.values()) for c in seconds
        ],
        "meaning": "Outer native shapes seen during fixed performance and quality HTTP windows. Internal graph splits, actual kernel dispatch and whole-model benefit require measurement.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for f in ("primary", "previous", "out"):
        p.add_argument(f, type=Path)
    a = p.parse_args()
    save(a.out, order(a.primary, a.previous))
