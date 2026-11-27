"""Replay the shared-pool admission follow-up and distinguish actual decisions from ordinary mixed calls."""

import argparse
import gzip
import json
from pathlib import Path

from lab.pk6.mixed_progress import grade
from lab.pk6.mixed_progress.pool_trial import ENGINE


def markers(raw, waves):
    changes = []
    for line in raw.splitlines():
        if line.startswith("# shared_budget "):
            _, _, old, new, queued, t = line.split()
            low, high, depth = (int(v.split("=")[1]) for v in (old, new, queued))
            if not 0 < low < high <= 2048 or depth < 4:
                raise ValueError("invalid shared-pool admission marker")
            changes.append(float(t))
    return [
        {
            "width": w["width"],
            "seed": w["seed"],
            "repeat": w["repeat"],
            "changes": sum(
                min(r["t_send"] for r in w["requests"])
                <= t
                <= max(r["t_done"] for r in w["requests"])
                for t in changes
            ),
        }
        for w in waves
    ]


def score(primary, extra):
    def read(folder, name):
        return json.loads((folder / name).read_bytes())

    base = read(primary, "target.json")
    target = read(extra, "target.json")
    if target != {**base, "pool_engine_sha256": ENGINE} or read(
        extra, "completed.json"
    ) != {"hardware": base["hardware"], "order": ["S1", "S2"]}:
        raise ValueError("foreign follow-up target")
    out = {}
    last_end = read(primary, "end-R2.json")["unix"]
    for idx, seed in enumerate(grade.SEEDS, 1):
        tag = f"S{idx}"
        native = grade.lines(extra / f"waves-{tag}.jsonl")
        reference = grade.lines(primary / f"waves-R{idx}.jsonl")
        start = read(extra, f"start-{tag}.json")
        end = read(extra, f"end-{tag}.json")
        health = read(extra, f"ready-{tag}.json")["health"]
        old = read(primary, f"ready-R{idx}.json")["health"]
        if (
            start["hardware"] != base["hardware"]
            or end["hardware"] != base["hardware"]
            or not last_end <= start["unix"] < end["unix"]
            or len(native) != len(reference)
            or not health.get("shared_pool")
            or health["kernel_profile"] != old["kernel_profile"]
            or health["sequences"]["sequences"] != old["sequences"]["sequences"]
            or health["gate"]["passes"] is not True
            or health.get("kernel_generation", {}).get("status") != "disabled"
        ):
            raise ValueError("incompatible product configuration")
        last_end = end["unix"]
        for w, r in zip(native, reference, strict=True):
            if (
                w["seed"] != seed
                or w["arm"] != tag
                or any(w[k] != r[k] for k in ("repeat", "width", "input_digest"))
                or not start["unix"] <= w["unix_start"] < w["unix_end"] <= end["unix"]
            ):
                raise ValueError("altered or incomplete follow-up workload")
        f = extra / f"trace-{tag}.txt"
        raw = (
            f.read_text()
            if f.exists()
            else gzip.decompress(f.with_suffix(".txt.gz").read_bytes()).decode()
        )
        evidence = markers(raw, native)
        original = grade.bindings(extra, tag, native)
        q, qr = native[-1], reference[-1]
        if [(r["case"], r["prompt_tokens"]) for r in q["requests"]] != [
            (r["case"], r["prompt_tokens"]) for r in qr["requests"]
        ]:
            raise ValueError("different quality cases")
        out[tag] = {
            "trace_sha256": original["trace_sha256"],
            "waves_sha256": original["waves_sha256"],
            "changes": evidence,
            "quality": [
                sum(r["correct"] is True for r in q["requests"]),
                len(q["requests"]),
            ],
            "by_width": {},
            "vs_primary_reference": {},
        }
        for width in (1, 8, 32):

            def block(rows):
                return [w for w in rows if w["width"] == width and w["repeat"] < 3]

            out[tag]["vs_primary_reference"][str(width)] = grade.paired(
                block(reference), block(native)
            )
            out[tag]["by_width"][str(width)] = grade.summarize(
                [grade.measure(w) for w in block(native)]
            )
    return {
        "target": target,
        "starts": out,
        "promotion": False,
        "scope": "Discovery seeds followed the incumbent. "
        "A separate paired holdout, net-benefit gate and rollback are required for promotion.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("primary", type=Path)
    p.add_argument("extra", type=Path)
    a = p.parse_args()
    print(json.dumps(score(a.primary, a.extra), indent=2, allow_nan=False))
