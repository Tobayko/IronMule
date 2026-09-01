"""Aggregate every recorded token-identity divergence and locate it.

Offline forensics over sealed experiment output already on disk.  No model,
no hardware, no writes outside stdout.  It answers one question: do the
identity failures of the prefill candidates look structural, or do they look
like a single degenerate position in the greedy path?
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SOURCES = (
    ("chunk_identity", "experiments/chunk_identity/results.json", ("rows",)),
    ("chunk_confirmation", "experiments/chunk_identity/confirmation.json", ("rows", "combined")),
    ("prefix_reuse", "experiments/ttft/prefix_correctness.json", ("rows",)),
    ("prefill_chunking", "experiments/ttft/prefill_chunking.json", ("rows",)),
)


def position(row: dict) -> int | None:
    # The two studies spell the field differently; neither file is edited.
    return row.get("first_diff") if "first_diff" in row else row.get("erste_abweichung")


def main() -> int:
    observations: list[tuple[str, dict]] = []
    for label, relative, sections in SOURCES:
        data = json.loads((ROOT / relative).read_text())
        for section in sections:
            for row in data.get(section, []):
                if "identical" in row or "identisch_zu_ein_block" in row:
                    normalised = dict(row)
                    if "identisch_zu_ein_block" in normalised:
                        normalised["identical"] = normalised["identisch_zu_ein_block"]
                    observations.append((label, normalised))

    diverged = [(label, row) for label, row in observations if not row["identical"]]
    print(f"observations {len(observations)}   identical {len(observations) - len(diverged)}"
          f"   diverged {len(diverged)}")
    print()
    for label, row in diverged:
        detail = {k: v for k, v in row.items()
                  if k not in ("identical", "identisch_zu_ein_block", "first_diff",
                               "erste_abweichung", "note")}
        print(f"  {label:20s} first_diff={position(row)!s:>5s}  {detail}")
    counts = Counter(position(row) for _, row in diverged)
    print()
    print("divergence positions:", dict(sorted(counts.items(), key=lambda item: (item[0] is None, item[0]))))
    print()
    print("A structural prefill or KV error diverges at generated position 0 or 1 and")
    print("scatters. A single repeated late position across different prompt lengths,")
    print("chunk sizes and two independent mechanisms is the signature of a near-tie")
    print("in the greedy argmax that any change of accumulation order flips.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
