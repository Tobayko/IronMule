"""Report deployment-level comparisons, keeping cold-start and resource evidence explicit."""

import argparse
from pathlib import Path

from lab.pk6.mixed_progress import grade
from lab.pk6.native_dispatch import score
from lab.pk6.warm_repeat import save


def matching(rows, seed, width):
    return [
        r for r in rows if r["seed"] == seed and r["repeat"] < 3 and r["width"] == width
    ]


def verdict(root):
    comparison = score.run(root / "http")
    arms = comparison["arms"]
    initial = arms["vllm"]
    later = root / "controls/late"
    repeat = score.run(later)["arms"].get("vllm") if later.exists() else None
    groups = {}
    for name, measured in comparison["two_seed_evaluation"].items():
        ids = [name + str(i) for i in (1, 2)]
        pairs = [comparison["pairs"][f"{a}/vllm/{w}"] for a in ids for w in (1, 8, 32)]
        native = [arms[a] for a in ids]
        energy = []
        comparisons = list(pairs)
        for arm, a in zip(ids, native, strict=True):
            seed = next(iter(a["quality"]))
            for w in ("1", "8", "32"):
                ours = a["resources"][
                    "estimated_board_joules_per_output_token_by_seed"
                ][seed][w]
                for ext in [initial, repeat] if repeat else [initial]:
                    theirs = ext["resources"][
                        "estimated_board_joules_per_output_token_by_seed"
                    ][seed][w]
                    energy.append(
                        ours / theirs if ours is not None and theirs else None
                    )
            if repeat:
                side = next((root / "http").rglob(f"end-{arm}.json")).parent
                extdir = next(later.rglob("end-vllm.json")).parent
                observed = score.waves(side, arm)
                external = score.waves(extdir, "vllm")
                for width in (1, 8, 32):
                    comparisons.append(
                        grade.paired(
                            matching(external, int(seed), width),
                            matching(observed, int(seed), width),
                        )
                    )
        fixed = all(p["same_fixed_work"] for p in comparisons)
        peaks = [a["resources"]["peak_gpu_mib"] for a in native]
        # Only the first stock and first vLLM processes followed first-time GPU calibration in this series.
        # A cache hit on one side cannot establish cold-start superiority.
        timing = {
            key: all(
                p["candidate_over_reference"][key] is not None
                and (
                    p["candidate_over_reference"][key] >= 1.1
                    if key == "tok_s"
                    else p["candidate_over_reference"][key] <= 0.9
                )
                for p in comparisons
            )
            for key in initial["metrics"]["32"]
        }
        cold = native[0]["startup_s"] / initial["startup_s"] if name == "R" else None
        warm = native[1]["startup_s"] / repeat["startup_s"] if repeat else None
        groups[name] = {
            "same_fixed_answers": fixed,
            "same_factual_accuracy": measured["both_seeds_32_facts"]
            and set(initial["quality"].values()) == {32},
            "profile_eligible": not measured["excluded"],
            "ten_percent_on_every_width_and_seed": timing,
            "highest_serving_vram_ratio": max(peaks)
            / initial["resources"]["peak_gpu_mib"]
            if all(peaks)
            else None,
            "worst_sampled_energy_ratio": max(energy)
            if all(e is not None for e in energy)
            else None,
            "first_stock_start_ratio": cold,
            "second_start_over_late_comparator": warm,
        }
    return {
        "target": comparison["binding"][:5],
        "families": groups,
        "deployment": "GGUF Q4_K_M in IronMule and official AWQ-Marlin in vLLM. Timings cover three waves at each width on two seeds. Energy uses one-second board samples.",
        "startup": "First and later process starts are reported separately. Only R1 is the initial IronMule qualification. Matched cold repeats of other stacks are absent.",
        "scope": "Two input seeds and one GPU cannot establish a win over unmeasured hardware or workloads.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    save(a.out, verdict(a.root))
