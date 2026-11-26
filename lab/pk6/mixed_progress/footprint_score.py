"""Check real work, quality and native dispatch for the memory-bounded W4 screen."""

import argparse
import json
from pathlib import Path
from statistics import median

from lab.pk6.mixed_progress import footprint
from lab.pk6.warm_repeat import GGUF


def score(primary, folder):
    def read(path):
        return json.loads(path.read_bytes())

    target = read(primary / "target.json")
    inventory = read(primary / "w4-inventory.json")
    selection = read(folder / "selection.json")
    base = read(primary / "result.json")
    peak = max(base["resources"][a]["peak_gpu_mib"] for a in ("R1", "R2"))
    expected = footprint.choose(inventory, target["hardware"]["memory_mib"], peak)
    if (
        any(selection[k] != v for k, v in expected.items())
        or selection["hardware"] != target["hardware"]
        or selection["gguf_sha256"] != GGUF
        or selection["table_sha256"] != inventory["table_sha256"]
        or selection["stock_peak_mib"] != peak
    ):
        raise ValueError("memory selection disagrees with original device receipts")
    if (folder / "outcome.json").exists():
        outcome = read(folder / "outcome.json")
    else:
        interrupted = read(folder / "collector-stop.json")
        if (
            interrupted["reason"] != "mutable_attempt_map_rejected"
            or interrupted["candidate_worker_receipt"] is not None
        ):
            raise ValueError("unrecognized collector stop")
        outcome = {
            "status": "collection_interrupted",
            "attempts": read(folder / "attempts.json"),
            "production": False,
            "collector": interrupted,
        }
    attempts = outcome.get("attempts", {})
    if attempts and read(folder / "attempts.json") != attempts:
        raise ValueError("inconsistent native worker outcomes")
    if selection["table_rows"]:
        table = (folder / "selected.table").read_text().splitlines()
        if len(table) != selection["table_rows"] or {
            tuple(map(int, l.split()[:3])) for l in table
        } != {tuple(s) for s in expected["allowed_shapes"]}:
            raise ValueError("different shape population")
    pairs = {}
    for i, seed in enumerate((825323, 629317), 1):
        ref, cand = (f"{tag}{i}" for tag in ("R", "M"))
        if attempts.get(ref) != 0 or attempts.get(cand) != 0:
            continue
        r, c = [read(folder / f"worker-{a}.json") for a in (ref, cand)]
        for v in (r, c):
            if (
                v["seed"] != seed
                or len(v["quality"]) != 18
                or [(w["width"], w["repeat"]) for w in v["waves"]]
                != [(w, j) for w in (1, 8) for j in range(3)]
            ):
                raise ValueError("invalid scored worker")
        if r["native_repacked_bytes"] != 0 or set(r["quality"]) != set(c["quality"]):
            raise ValueError("invalid native stock comparison")
        fixed = all(
            w["tokens"] == [32] * w["width"]
            and w["reasons"] == ["length"] * w["width"]
            and w["seconds"] > 0
            for v in (r, c)
            for w in v["waves"]
        )
        pairs[cand] = {
            "seed": seed,
            "fixed_work": fixed,
            "same_output_ids": all(
                a["output_ids_sha256"] == b["output_ids_sha256"]
                for a, b in zip(r["waves"], c["waves"], strict=True)
            ),
            "correct": {
                a: sum(v["quality"].values()) for a, v in ((ref, r), (cand, c))
            },
            "native_repacked_bytes": c["native_repacked_bytes"],
            "tokens_served": c["tokens_served"],
            "q6k_tokens_served": c["q6k_tokens_served"],
            "paired_output_rate_ratio": {
                str(n): median(
                    a["seconds"] / b["seconds"]
                    for a, b in zip(r["waves"], c["waves"], strict=True)
                    if a["width"] == n
                )
                for n in (1, 8)
            },
        }
    return {
        "hardware": target["hardware"],
        "selection": selection,
        "outcome": outcome,
        "pairs": pairs,
        "promotion": False,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("primary", type=Path)
    p.add_argument("folder", type=Path)
    a = p.parse_args()
    print(json.dumps(score(a.primary, a.folder), indent=2, allow_nan=False))
