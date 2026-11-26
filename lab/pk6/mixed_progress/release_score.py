"""Compare a subsequent vLLM start with the same-board, independently graded primary arms."""

import argparse
import json
from pathlib import Path

from lab.pk6.mixed_progress import grade
from lab.pk6.mixed_progress.current import VERSION
from lab.pk6.mixed_progress.trial import ARMS, SEEDS
from lab.pk6.replay_http import checksum


def score(primary, current):
    def read(folder, name):
        return json.loads((folder / name).read_bytes())

    original = read(primary, "target.json")
    result = grade.grade(primary)
    target = read(current, "target.json")
    if (
        target["vllm_version"] != target["versions"]["vllm"]
        or target["versions"]["vllm"] != VERSION
        or any(target[k] != v for k, v in original.items() if k != "vllm_version")
        or read(current, "completed.json")
        != {"hardware": original["hardware"], "version": VERSION}
    ):
        raise ValueError("supplement target differs from the primary experiment")
    start, ready, end = (
        read(current, f"{k}-vllm.json") for k in ("start", "ready", "end")
    )
    if (
        any(r["hardware"] != target["hardware"] for r in (start, end))
        or start["unix"] < read(primary, "end-R2.json")["unix"]
        or not 0 < ready["startup_s"] <= end["unix"] - start["unix"]
    ):
        raise ValueError("supplement timing or hardware differs")
    new = grade.lines(current / "waves-vllm.jsonl")
    original_rows = grade.lines(primary / "waves-vllm.jsonl")
    if len(new) != len(original_rows):
        raise ValueError("incomplete supplement")
    for a, b in zip(original_rows, new, strict=True):
        if (
            any(
                a[k] != b[k] for k in ("seed", "repeat", "width", "input_digest", "arm")
            )
            or not start["unix"] <= b["unix_start"] < b["unix_end"] <= end["unix"]
            or [(r["prompt_tokens"], r.get("case")) for r in a["requests"]]
            != [(r["prompt_tokens"], r.get("case")) for r in b["requests"]]
        ):
            raise ValueError("supplement inputs, ordering or time differs")
        grade.measure(b)
    by_width = {}
    for width in (1, 8, 32):
        cells = [
            grade.measure(w) for w in new if w["repeat"] < 3 and w["width"] == width
        ]
        by_width[str(width)] = grade.summarize(cells)
    pairs = {}
    for arm in ARMS[1:]:
        seed = SEEDS[int(arm[1]) - 1]
        native = grade.lines(primary / f"waves-{arm}.jsonl")
        for width in (1, 8, 32):

            def subset(waves):
                return [
                    w
                    for w in waves
                    if w["seed"] == seed and w["width"] == width and w["repeat"] < 3
                ]

            pairs[f"{arm}/vllm/{width}"] = grade.paired(subset(new), subset(native))
    telemetry = grade.telemetry_samples(current / "current-device-load.csv")
    return {
        "target": target,
        "primary_decision": result["decision"],
        "startup_s": ready["startup_s"],
        "median_by_width": by_width,
        "resources": grade.sampled_resources(telemetry, new),
        "quality": {
            str(w["seed"]): [
                sum(r["correct"] is True for r in w["requests"]),
                len(w["requests"]),
            ]
            for w in new
            if w["repeat"] == 3
        },
        "native_over_current_vllm": pairs,
        "waves_sha256": checksum(current / "waves-vllm.jsonl"),
        "scope": "One subsequent current-version start. Different quantizations. The primary decision is frozen.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("primary", type=Path)
    p.add_argument("current", type=Path)
    args = p.parse_args()
    print(json.dumps(score(args.primary, args.current), indent=2, allow_nan=False))
