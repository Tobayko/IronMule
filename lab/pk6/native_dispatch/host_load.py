"""Join passive host CPU samples to study process windows without imputing missing samples."""

import argparse
import gzip
import json
from pathlib import Path
from statistics import median

from lab.pk6.native_dispatch.score import read, waves
from lab.pk6.warm_repeat import save


def summarize(root, samples):
    contents = samples.read_bytes()
    if samples.suffix == ".gz":
        contents = gzip.decompress(contents)
    rows = [json.loads(line) for line in contents.splitlines()]
    if not rows or any(a["unix"] >= b["unix"] for a, b in zip(rows, rows[1:])):
        raise ValueError("missing or unordered host observations")
    out = {}
    for end in sorted(root.rglob("end-*.json")):
        arm = end.stem[4:]
        folder = end.parent
        start = read(folder / f"start-{arm}.json")["unix"]
        stop = read(end)["unix"]
        spans = [(w["unix_start"], w["unix_end"]) for w in waves(folder, arm)]
        during = [r for r in rows if start <= r["unix"] <= stop]
        served = [r for r in during if any(lo <= r["unix"] <= hi for lo, hi in spans)]
        out[str(end.relative_to(root))] = {
            "process_samples": len(during),
            "request_samples": len(served),
            "median_load1_in_requests": median(r["load1"] for r in served)
            if served
            else None,
            "request_samples_with_native_compiler": sum(
                r["native_compilers"] > 0 for r in served
            )
            if served
            else None,
        }
    return {
        "sampling_interval_s": 5,
        "first_unix": rows[0]["unix"],
        "last_unix": rows[-1]["unix"],
        "scope": "Host load average and compiler processes visible inside the pod. Five-second samples can miss short CPU bursts. Earlier server runs have no host observations.",
        "processes": out,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("root", "samples", "out"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    save(args.out, summarize(args.root, args.samples))
