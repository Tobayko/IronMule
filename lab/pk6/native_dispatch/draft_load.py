"""Rank a bounded drafting experiment from real target-verified proposals and mixed HTTP waves."""

import argparse
from pathlib import Path


def prioritize(wants, slots, held):
    """Keep incumbent slots and offer free ones to the cheapest prompt catch-ups first."""
    from ironmule.draft_model import _common

    cold = [key for key in wants if key not in slots]
    free = [history for slot, history in held.items() if slot not in slots.values()]
    if not free or len(cold) <= len(free):
        return wants

    def missing(key):
        context = wants[key][0]
        reused = max(_common(history, context) for history in free)
        return max(1, len(context) - reused)

    order = [key for key in wants if key in slots] + sorted(cold, key=missing)
    return {key: wants[key] for key in order}


def propose(http, metadata):
    from lab.pk6.native_dispatch import score
    from lab.pk6.native_dispatch.run import DRAFT_SHA256
    from lab.pk6.replay_http import checksum

    m = score.read(metadata)
    if m["gguf"] != DRAFT_SHA256:
        raise ValueError("foreign draft model")
    results = score.run(http)
    k, stock, external = (results["arms"][a] for a in ("K1", "R1", "vllm"))
    waves = next(http.rglob("waves-K1.jsonl"))
    target = score.read(waves.parent / "target.json")
    if (
        k["excluded_from_promotion"]
        or not all(q == 32 for q in k["quality"].values())
        or target["draft_gguf_sha256"] != DRAFT_SHA256
        or not results["pairs"]["K1/R1/8"]["same_fixed_work"]
    ):
        raise ValueError("missing a qualified fixed-work drafting observation")
    checked = k["proof"]["draft_verified_by_width"]
    difference = k["metrics"]["8"]["tok_s"] / external["metrics"]["8"]["tok_s"]
    # Double the two incumbent slots for the first under-served measured load.
    # FP16 K and V, using the draft GGUF's own metadata and the measured context.
    slots = 4
    extra_bytes = (
        (slots - 2)
        * 2048
        * 2
        * m["qwen3.block_count"]
        * m["qwen3.attention.head_count_kv"]
        * (m["qwen3.attention.key_length"] + m["qwen3.attention.value_length"])
    )
    headroom = (
        target["hardware"]["memory_mib"] - k["resources"]["peak_gpu_mib"]
    ) * 2**20
    selected = (
        k["metrics"]["1"]["tok_s"] > stock["metrics"]["1"]["tok_s"]
        and 0 < checked["8"]["model"]["rows"] < checked["1"]["model"]["rows"]
        and difference < 1
        and headroom > 2 * extra_bytes
    )
    return {
        "hardware": target["hardware"],
        "gguf": target["gguf_sha256"],
        "draft": m["gguf"],
        "metadata": m,
        "source": checksum(waves),
        "selected": selected,
        "verified_by_width": checked,
        "eight_client_fraction_of_vllm": difference,
        "additional_kv_bytes_estimate": extra_bytes,
        "measured_headroom_bytes": int(headroom),
        "slots": slots,
        "max_running": 8,
        "scope": "One discovery seed. The Ada target accepted drafts mostly at low load. Four slots and catch-up ordering are a new experiment. Only a full product measurement can establish benefit.",
    }


if __name__ == "__main__":
    from lab.pk6.warm_repeat import save

    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("http", "metadata", "out"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    save(args.out, propose(args.http, args.metadata))
