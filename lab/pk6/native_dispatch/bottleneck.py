"""Replay mixed-step attribution, KV-span controls and the held-out HTTP retention trial."""

import argparse
import csv
import gzip
from pathlib import Path
from statistics import median

from lab.pk6.mixed_progress import grade
from lab.pk6.native_dispatch import reference, score
from lab.pk6.warm_repeat import save


SEEDS = (180341, 943819)
FLOORS = (0, 8192, 16384, 32768)


def long_intervals(folder, arm):
    spans = []
    with gzip.open(folder / f"trace-{arm}.txt.gz", "rt") as trace:
        for line in trace:
            fields = line.split()
            if (
                fields
                and line[0].isdigit()
                and int(fields[5]) > 300
                and int(fields[6]) > 0
            ):
                start, seconds = map(float, fields[:2])
                spans.append((start, start + seconds))
    return spans


def short_stream_pauses(requests, spans, count):
    lo = min(r["t_send"] for r in requests)
    hi = max(r["t_done"] for r in requests)
    selected = [(a, b) for a, b in spans if lo <= a <= b <= hi]
    if len(selected) != count or any(
        x[1] > y[0] for x, y in zip(selected, selected[1:])
    ):
        raise ValueError("overlapping or missing large native intervals")
    covered = []
    for r in requests:
        if r["prompt_tokens"] > 64:
            continue
        times = r["chunk_times"]
        a, b = max(zip(times, times[1:]), key=lambda pair: pair[1] - pair[0])
        overlap = sum(max(0.0, min(b, end) - max(a, start)) for start, end in selected)
        covered.append(overlap / (b - a))
    return {
        "requests": len(covered),
        "median_overlap_fraction": median(covered),
        "at_least_80_percent_covered": sum(x >= 0.8 for x in covered),
    }


def step_cost(archive, probe):
    primary = archive / "http/primary"
    result = []
    for arm, seed in zip(("R1", "R2"), SEEDS, strict=True):
        waves = score.waves(primary, arm)
        joined = grade.bindings(primary, arm, waves)
        spans = long_intervals(primary, arm)
        for w, c in zip(waves, joined["cells"], strict=True):
            rep, width = w["repeat"], w["width"]
            if rep == 3:
                continue
            fresh = score.read(probe / f"wave-{seed}-{rep}-{width}.json")
            if (
                (seed, rep, width, w["input_digest"])
                != (
                    fresh["seed"],
                    fresh["repeat"],
                    fresh["width"],
                    fresh["input_digest"],
                )
                or fresh["output_counts"] != [32] * width
                or [r["tokens"] for r in w["requests"]] != fresh["output_counts"]
            ):
                raise ValueError("changed native workload")
            if width != 32:
                continue
            big = [r for r in c["mixed"] if r["prompt_rows"] > 300]
            natural = [
                r for r in fresh["calls"] if r["prompt_rows"] > 300 and r["decode_rows"]
            ]
            if len(big) != 16 or len(natural) != 16:
                raise ValueError("missing a document step")
            sec = sum(r["seconds"] for r in big)
            native = {k: v["llama_decode_s"] for k, v in c["phases"].items()}
            result.append(
                dict(
                    seed=seed,
                    repeat=rep,
                    requests=width,
                    trace=joined["trace_sha256"],
                    http_s=w["elapsed_s"],
                    archived_phases_s=native,
                    large_calls=len(big),
                    longest_short_stream_pauses=short_stream_pauses(
                        w["requests"], spans, len(big)
                    ),
                    all_native_fraction=sum(native.values()) / w["elapsed_s"],
                    large_call_s=sec,
                    large_call_wall_fraction=sec / w["elapsed_s"],
                    large_prompt_range=[
                        min(r["prompt_rows"] for r in big),
                        max(r["prompt_rows"] for r in big),
                    ],
                    median_physical_span=median(r["kv"] for r in natural),
                    median_natural_step_s=median(r["seconds"] for r in natural),
                )
            )
    return result


def intervention(probe):
    groups = {}
    for seed in SEEDS:
        rows = score.read(probe / f"intervention-{seed}.json")
        if len(rows) != 20 or len({r["rows_sha256"] for r in rows}) != 1:
            raise ValueError("changed fixed input or missing trials")
        groups[seed] = {}
        for floor in FLOORS:
            group = [r for r in rows if r["floor"] == floor]
            if sorted(r["repeat"] for r in group) != list(range(5)) or any(
                r["natural"] != 1024
                or r["effective"] != max(1024, floor)
                or ((r["seconds"] is None) != (r["repeat"] == 4))
                for r in group
            ):
                raise ValueError("span or order mismatch")
            events = score.read(probe / f"cuda-{seed}-{floor}.json")
            group_times = [r["seconds"] for r in group if r["seconds"] is not None]
            groups[seed][floor] = dict(
                actual_span=group[0]["effective"],
                median_s=median(group_times),
                min_s=min(group_times),
                max_s=max(group_times),
                all_17_picks_match_control=all(
                    r["result_sha256"] == rows[0]["result_sha256"] for r in group
                ),
                attention_device_s=sum(
                    r["device_s"] for r in events if "flash_attn" in r["symbol"]
                ),
                q4_main_device_s=sum(
                    r["device_s"]
                    for r in events
                    if "void mul_mat_q<(ggml_type)12," in r["symbol"]
                ),
                q6_main_device_s=sum(
                    r["device_s"]
                    for r in events
                    if "void mul_mat_q<(ggml_type)14," in r["symbol"]
                ),
            )
    return groups


def projection_estimates(archive):
    operators = archive / "operator-calibration"
    target = score.read(operators / "completed.json")
    primary = score.read(archive / "http/primary/target.json")
    if (primary["hardware"], primary["gguf_sha256"]) != (
        target["hardware"],
        target["gguf"],
    ):
        raise ValueError("operator calibration targets a different device or GGUF")
    estimates = {}
    for seed in target["seeds"]:
        groups = {}
        for file in operators.glob(f"shape-{seed}-*.json"):
            m = score.read(file)
            name = (
                "ffn_gate_up"
                if (m["k"], m["n"]) == (5120, 25600)
                else (
                    "ffn_down"
                    if (m["k"], m["n"]) == (25600, 5120)
                    else "attention_projections"
                )
            )
            groups[name] = (
                groups.get(name, 0)
                + m["matrix_count"] * m["cells"]["448"]["stock"]["us"] / 1e6
            )
        estimates[seed] = groups
    return estimates


def mask_coverage(folder):
    results = []
    done = score.read(folder / "complete.json")
    if (done["query_tile"], done["key_tile"]) != (8, 256):
        raise ValueError("different mask tiling")
    for seed in SEEDS:
        for repeat in range(3):
            rows = list(
                csv.reader(
                    (folder / f"{seed}-{repeat}-32.csv").read_text().splitlines()
                )
            )
            counts = [0] * 4
            for row in rows:
                queries, keys, total, visible, trailing, below = map(int, row)
                if (
                    not 440 <= queries <= 485
                    or total != ((queries + 7) // 8) * (keys // 256)
                    or min(visible, trailing, below) < 0
                    or visible + trailing + below != total
                ):
                    raise ValueError("invalid mask-cell counts")
                counts = [
                    a + b
                    for a, b in zip(
                        counts, (total, visible, trailing, below), strict=True
                    )
                ]
            if len(rows) != 16:
                raise ValueError(f"expected 16 document masks, got {len(rows)}")
            results.append(
                dict(
                    seed=seed,
                    repeat=repeat,
                    calls=len(rows),
                    tiles=counts[0],
                    visible=counts[1],
                    trailing_masked=counts[2],
                    other_masked=counts[3],
                )
            )
    return results


def served(folder):
    cells = []
    board = None
    for i, seed in enumerate(SEEDS, start=1):
        a, b = f"R{i}", f"H{i}"
        pairs = []
        factual = {}
        for arm, keep in ((a, 1), (b, 0.03125)):
            root = folder / arm
            plan = score.read(root / "plan.json")
            hardware = plan["hardware"]
            if (
                plan["seed"] != seed
                or plan["finished_fraction"] != keep
                or (board is not None and board != hardware)
                or any(
                    score.read(root / f"{stage}-{arm}.json")["hardware"] != hardware
                    for stage in ("start", "end")
                )
            ):
                raise ValueError("unexpected retention trial device or setting")
            board = hardware
            health = score.read(root / f"ready-{arm}.json")["health"]
            if not health["gate"]["passes"]:
                raise ValueError("failed native gate")
            waves = score.waves(root, arm)
            grade.bindings(root, arm, waves)
            facts = [r for w in waves if w["repeat"] == 3 for r in w["requests"]]
            if len(facts) != 32 or any(w["seed"] != seed for w in waves):
                raise ValueError("wrong facts or seed")
            factual[arm] = sum(r["correct"] for r in facts)
            pairs.append(waves)
        for width in (1, 8, 32):
            r, h = [
                [w for w in group if w["width"] == width and w["repeat"] < 3]
                for group in pairs
            ]
            paired = grade.paired(r, h)
            if not paired["same_fixed_work"]:
                raise ValueError("different HTTP work")
            cells.append(
                dict(
                    seed=seed,
                    width=width,
                    reference=grade.summarize([grade.measure(w) for w in r]),
                    limited=grade.summarize([grade.measure(w) for w in h]),
                    paired=paired,
                    facts=factual,
                )
            )
    return cells


def replay(archive, root):
    old = reference.lookup(archive)
    if old is None:
        raise ValueError("missing reference")
    return dict(
        archived_vllm_32=old["metrics"]["32"]["tok_s"],
        archive_and_new_native=step_cost(archive, root / "span"),
        fixed_rows=intervention(root / "span"),
        tile_counts=mask_coverage(root / "masks"),
        estimated_projection_seconds_per_448_row_graph=projection_estimates(archive),
        paired_http=served(root / "http"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("archive", "results", "out"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    save(args.out, replay(args.archive, args.results))
