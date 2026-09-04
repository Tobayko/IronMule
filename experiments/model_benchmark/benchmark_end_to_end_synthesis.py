#!/usr/bin/env python3
"""Roll up the model-benchmark result files into one synthesis document.

This script computes nothing on the GPU. It reads the JSON artifacts the other
harnesses wrote and derives a compact summary. A source file that is absent
produces ``"not measured"`` for its section — never a placeholder number.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = PROJECT_ROOT / "experiments" / "model_benchmark"

SOURCES = {
    "family_benchmark": "gemma_family_benchmark.json",
    "long_tasks": "long_tasks_benchmark_results.json",
    "prompt_lookup": "prompt_lookup_results.json",
}


def _family_section(data: dict) -> dict:
    out = {}
    for tag, prompts in data.items():
        rows = []
        for prompt_name, cfgs in prompts.items():
            base = cfgs.get("baseline", {})
            for cand_name in ("dispatched", "combined_r8"):
                cand = cfgs.get(cand_name)
                if not cand:
                    continue
                rows.append({
                    "prompt": prompt_name,
                    "config": cand_name,
                    "end_to_end_speedup_pct": cand.get("end_to_end_speedup_percent"),
                    "tps_gain_pct": cand.get("tps_gain_percent"),
                    "token_identical": cand.get("token_identical_to_baseline"),
                })
        out[tag] = rows
    return out


def _long_tasks_section(data: dict) -> dict:
    out = {}
    for key, model in data.items():
        lengths = {}
        for length, cfgs in model.get("lengths", {}).items():
            base = next(iter(cfgs.values()))
            lengths[length] = {
                name: {
                    "wall_s": r.get("wall_s"),
                    "decode_tps": r.get("decode_tps"),
                    "token_identical": r.get("tokens_identical"),
                }
                for name, r in cfgs.items()
            }
        out[key] = lengths
    return out


def main():
    print("=== ROLLING UP MODEL-BENCHMARK RESULTS ===")
    loaded = {}
    for key, name in SOURCES.items():
        path = BENCH_DIR / name
        if path.exists():
            loaded[key] = json.loads(path.read_text())
            print(f"[loaded]  {key}: {name}")
        else:
            print(f"[missing] {key}: {name}")

    synthesis = {
        "note": "derived from the result files listed under 'sources'; no GPU work here",
        "sources": {k: (SOURCES[k] if k in loaded else None) for k in SOURCES},
        "family_benchmark": _family_section(loaded["family_benchmark"]) if "family_benchmark" in loaded else "not measured",
        "long_tasks": _long_tasks_section(loaded["long_tasks"]) if "long_tasks" in loaded else "not measured",
        "prompt_lookup": loaded.get("prompt_lookup", "not measured"),
    }

    out_file = BENCH_DIR / "end_to_end_synthesis_results.json"
    out_file.write_text(json.dumps(synthesis, indent=2))
    print(f"\n[ok] synthesis written to: {out_file}")


if __name__ == "__main__":
    main()
