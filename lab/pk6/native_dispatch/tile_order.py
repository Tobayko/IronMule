"""Issue a single code experiment from target-attributed GPU activity and the existing CPU world-model trace."""

import argparse
import json
from pathlib import Path
import re
from statistics import median

from lab.pk6.mixed_progress.grade import bindings, lines
from lab.pk6.replay_http import checksum
from lab.pk6.warm_repeat import save
from training.code_author import PendingAuthor, hosted_proposal, digest

REFERENCE = """static int prompt_tile_limit(int type, int cc, int matrices, int rows, int stock_limit) {
    return stock_limit;
}"""


def order(primary, state):
    state.mkdir(parents=True, exist_ok=True)
    target = json.loads((primary / "target.json").read_bytes())
    waves = lines(primary / "waves-D1.jsonl")
    measured = bindings(primary, "D1", waves)
    sample = json.loads((primary / "activity-mixed.json").read_bytes())
    matches = [
        w
        for w in waves
        if w["repeat"] < 3
        and set(sample["request_ids"]).issubset({r["server_id"] for r in w["requests"]})
        and min(r["t_send"] for r in w["requests"])
        <= sample["start_monotonic"]
        <= max(r["t_done"] for r in w["requests"])
    ]
    if len(matches) != 1 or len(sample["request_ids"]) < 12 or sample["prompt"] < 300:
        raise ValueError("sample is outside the busy HTTP window")
    # Use unprofiled stock wave times to price a hypothetical 10% saving in each source family.
    # One CUPTI call estimates shares, so the forecast only ranks the next experiment.
    stock = lines(primary / "waves-R1.jsonl")
    base = bindings(primary, "R1", stock)
    mixed = median(
        c["phases"]["mixed"]["llama_decode_s"]
        for c in base["cells"]
        if c["width"] == 32 and c["repeat"] < 3
    )
    options = []
    for e in sample["kernels"]:
        m = re.match(r"void mul_mat_q<\(ggml_type\)(\d+), (\d+), false>", e["symbol"])
        if m:
            options.append(
                {
                    "ggml_type": int(m[1]),
                    "executed_tile": int(m[2]),
                    "observed_s": e["device_s"],
                    "simulated_wave_saving_s": 0.1
                    * mixed
                    * e["device_s"]
                    / sample["elapsed_s"],
                }
            )
    if not options:
        raise ValueError("no measured quantized native kernel")
    options.sort(key=lambda o: o["simulated_wave_saving_s"], reverse=True)
    selected = options[0]
    work = {
        "schema": "ironmule.native-prompt-order.v1",
        "base_commit": target["source"],
        "hardware": target["hardware"],
        "gguf": target["gguf_sha256"],
        "kind": "simulated experiment ranking",
        "matched_request_ids": len(sample["request_ids"]),
        "trace_sha256": measured["trace_sha256"],
        "sample_sha256": checksum(primary / "activity-mixed.json"),
        "fixed_work_stock_mixed_s": mixed,
        "options": options,
        "target": selected,
        "editable_symbol": "prompt_tile_limit",
        "evaluator": "Same native MMQ operation, device, tensors and completed output. Confirm in an independent product gate and paired HTTP run.",
        "task": "Return one bounded tile-limit experiment for the selected quantization on Ada. Keep native launch validation. Empty/default configuration must preserve stock. Do not modify quality or budget.",
    }
    work["order_id"] = digest(work)
    save(state / "order.json", work)
    try:
        return hosted_proposal(
            state,
            work,
            REFERENCE,
            knowledge=[
                "native MMQ minimizes the number of column tiles under its shared-memory limit",
                "small and pure decode phases have different measured tile sizes from the mixed phase",
            ],
        )
    except PendingAuthor:
        return {"status": "awaiting_codex", "order": work["order_id"]}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("primary", type=Path)
    p.add_argument("state", type=Path)
    args = p.parse_args()
    print(json.dumps(order(args.primary, args.state)))
