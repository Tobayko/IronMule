"""Learn an Ada prompt dispatch policy from native operator experiments, then freeze it for a new product gate."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import median
import time

import numpy as np

from ironmule.pilot import Pilot, predict
from lab.pk6.native_dispatch.operators import WIDTHS
from lab.pk6.replay_http import checksum
from lab.pk6.warm_repeat import GGUF, save
from training.pilot_w4_profiles import fit_head

NAMES = ("stock", "fp8")
# Hold each projection geometry together, including different GGUF types at the same dimensions.
# Train on attention Q and FFN down, calibrate confidence using attention output, test K/V and FFN up.
FAMILIES = {
    "train": {(8192, 5120), (5120, 25600)},
    "calibrate": {(5120, 8192)},
    "test": {(1024, 5120), (25600, 5120)},
}


def features(key):
    qt, n, k, b = key
    return np.asarray(
        [
            math.log2(n / 4096),
            math.log2(k / 4096),
            math.log2(b / 256),
            float(qt == 14),
            math.log2(b / 256) ** 2 / 4,
        ],
        dtype=np.float32,
    )


def learn(directory, out):
    started = time.perf_counter()
    protocol = json.loads((directory / "protocol.json").read_bytes())
    if (
        protocol != json.loads((directory / "completed.json").read_bytes())
        or protocol["gguf"] != GGUF
    ):
        raise ValueError("incomplete target calibration")
    observed = defaultdict(dict)
    sources = {}
    for path in sorted(directory.glob("shape-*.json")):
        row = json.loads(path.read_bytes())
        sources[path.name] = checksum(path)
        qt, n, k = (row[v] for v in ("type", "n", "k"))
        if set(map(int, row["cells"])) != set(WIDTHS):
            raise ValueError("missing widths")
        for b, cell in row["cells"].items():
            key = qt, n, k, int(b)
            if (
                row["seed"] in observed[key]
                or cell["fp8"]["dispatches"] != cell["fp8"]["graphs"]
                or not 0 < cell["fp8"]["graphs"]
                or any(
                    not math.isfinite(v["us"])
                    or v["us"] <= 0
                    or not math.isfinite(v["relative_rms_to_stock"])
                    for v in cell.values()
                )
            ):
                raise ValueError("unmatched operator")
            observed[key][row["seed"]] = cell
    if len(observed) != 7 * len(WIDTHS) or any(
        set(rows) != set(protocol["seeds"]) for rows in observed.values()
    ):
        raise ValueError("incomplete paired experiment")
    durations = {
        key: [median(row[name]["us"] for row in replicates.values()) for name in NAMES]
        for key, replicates in observed.items()
    }
    inputs = {key: features(key) for key in observed}
    groups = {
        s: sorted(key for key in observed if key[1:3] in shapes)
        for s, shapes in FAMILIES.items()
    }
    if not all(groups.values()) or sum(map(len, groups.values())) != len(observed):
        raise ValueError("unknown projection family")
    seeds = (631, 632, 633)
    heads = [fit_head(durations, groups["train"], seed, inputs) for seed in seeds]
    before = [
        fit_head(durations, groups["train"], seed, inputs, epochs=0) for seed in seeds
    ]
    error = max(
        abs(
            float(np.diff(predict(h, inputs[key]))[0])
            - math.log(durations[key][0] / durations[key][1])
        )
        for key in groups["calibrate"]
        for h in heads
    )
    runs, good = [], set()
    for group, keys in groups.items():
        for key in keys:
            decisions = [
                Pilot.startup_profile(
                    ensemble,
                    inputs[key],
                    NAMES,
                    NAMES,
                    "stock",
                    in_support=True,
                    margin_error=error,
                )
                for ensemble in (before, heads)
            ]
            d = decisions[1]
            fp8 = d["profile"] == "fp8"
            # Compiling a new mixed plan requires target evidence on both observed seeds. The final
            # HTTP gate will independently check model-level precision and service behavior.
            profitable = all(
                row["fp8"]["us"] * 1.1 < min(row["stock"]["us"], row["cublas"]["us"])
                and row["fp8"]["relative_rms_to_stock"] < 0.06
                for row in observed[key].values()
            )
            if fp8 and profitable and key[-1] >= 128:
                good.add(key)
            runs.append(
                {
                    "shape": key,
                    "set": group,
                    "median_us": dict(zip(NAMES, durations[key], strict=True)),
                    "cuBLAS_us": median(
                        r["cublas"]["us"] for r in observed[key].values()
                    ),
                    "random_profile": decisions[0]["profile"],
                    "trained_profile": d["profile"],
                    "trained_scores": d["scores"],
                    "decision_us": d["cpu_us"],
                    "eligible": fp8 and profitable and key[-1] >= 128,
                }
            )
    intervals = []
    for qt, n, k in sorted({key[:3] for key in observed}):
        for b in sorted(WIDTHS):
            if (qt, n, k, b) not in good:
                continue
            previous = next((v for v in reversed(sorted(WIDTHS)) if v < b), None)
            if (
                intervals
                and intervals[-1][:3] == [qt, n, k]
                and intervals[-1][-1] == previous
            ):
                intervals[-1][-1] = b
            else:
                intervals.append([qt, n, k, b, b])
    out.mkdir(parents=True, exist_ok=False)
    arrays = {
        f"head_{i}_{j}": v for i, head in enumerate(heads) for j, v in enumerate(head)
    }
    np.savez(out / "pilot.npz", **arrays)
    header = "struct PromptRange { int type, n, k, low, high; };\nconstexpr PromptRange qualified_prompt_rows[] = {\n"
    # An empty plan can still be built and always falls back.
    for values in intervals or [[0, 0, 0, 0, 0]]:
        header += "    {" + ", ".join(map(str, values)) + "},\n"
    (out / "target_rows.h").write_text(header + "};\n")
    cpu_s = time.perf_counter() - started
    save(
        out / "fit.json",
        {
            "sources": sources,
            "target": protocol,
            "training_seeds": seeds,
            "cpu_s": cpu_s,
            "calibration_max_log_error": error,
            "rows": runs,
            "candidate_intervals": intervals,
            "weights_sha256": checksum(out / "pilot.npz"),
            "header_sha256": checksum(out / "target_rows.h"),
            "interpretation": "Interpolated interior widths are hypotheses pending the system gate. The test groups only score generalization before the target confirmation filter. No product qualification is implied.",
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("measurements", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    learn(a.measurements, a.output)
