"""Update the CPU step-cost model on one L40S start and predict a different seed and restart."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median

from ironmule.ggml_agent import StepCost
from lab.pk6.mixed_progress.grade import grade
from lab.pk6.mixed_progress.plan import plan
from lab.pk6.replay_http import checksum


def fit(train, test, prior):
    def valid(c):
        return (
            c["prompt_rows"] > 0
            and c["decode_rows"] > 0
            and math.isfinite(c["seconds"])
            and c["seconds"] > 0
        )

    if not train or not test:
        return {
            "status": "insufficient_observed_mixed_calls",
            "train_calls": len(train),
            "holdout_calls": len(test),
        }
    if not all(valid(c) for c in [*train, *test]):
        raise ValueError("invalid mixed call")
    # The original forecast priced every physical row at the same marginal cost.
    original = StepCost([prior[0], prior[2] / 1000, prior[2], 0.0])
    model = StepCost.from_json(original.to_json())
    for c in train:
        model.update(c["decode_rows"], c["prompt_rows"], 0, c["seconds"])
    fixed = median(c["seconds"] for c in train)
    predictions = {
        "unadapted_a100": [
            original.predict(c["decode_rows"], c["prompt_rows"], 0) for c in test
        ],
        "l40s_constant": [fixed] * len(test),
        "l40s_updated": [
            model.predict(c["decode_rows"], c["prompt_rows"], 0) for c in test
        ],
    }
    return {
        "status": "measured_held_out_restart",
        "train_calls": model.n,
        "holdout_calls": len(test),
        "prior": original.to_json(),
        "updated": model.to_json(),
        "train_median_s": fixed,
        "holdout_mean_absolute_error_s": {
            k: mean(abs(v - c["seconds"]) for v, c in zip(vals, test, strict=True))
            for k, vals in predictions.items()
        },
        "uncertainty": "One board and two starts. Only executed mixed-call durations have labels. "
        "Attended KV was unobserved. No serving-speed or policy-promotion inference.",
    }


def replay(folder):
    result = grade(folder)
    data, sources = [], {}
    for arm in ("C1", "C2"):
        file = folder / f"native-{arm}.json"
        native = json.loads(file.read_bytes())
        sources[arm] = {
            "native_sha256": checksum(file),
            "trace_sha256": native["trace_sha256"],
            "waves_sha256": native["waves_sha256"],
        }
        data.append(native["cells"])
    if {c["seed"] for c in data[0]} & {c["seed"] for c in data[1]}:
        raise ValueError("holdout seed leaked into the supervised update")
    trained = fit(
        *([s for c in cells for s in c["mixed"]] for cells in data),
        plan()["step_cost_theta"],
    )
    return {
        **trained,
        "hardware": result["target"]["hardware"],
        "train": "C1",
        "holdout": "C2",
        "sources": sources,
        "production_decision": result["decision"],
        "source_path": "ordinary shared KV"
        if result["native_mechanism"]["C1"]["shared_pool"]
        else "qualified split streams",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    print(json.dumps(replay(parser.parse_args().folder), indent=2, allow_nan=False))
