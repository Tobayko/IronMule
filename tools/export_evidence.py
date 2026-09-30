"""Export redacted, deterministic evidence summaries from the private raw measurements.

The runs under `experiments/kaggle_compat/results/`, `research/raw/` and
`experiments/head_skip_formal/` are gitignored: they stay on the machine that measured
them. This script reads exactly the files `tools/make_figures.py`, `ironmule/numeric_plans.py`
and the two claim tests open, keeps only the numbers and identity fields those readers
use, and writes the result under the tracked `evidence/` directory so a fresh clone can
run the tests and render the figures without the raw data.

Dropped on purpose: prompt and generated text, token-id lists, local/container paths
(`model_path` becomes `model_id`/`revision`), hostnames, logs and unused timestamps.

    python tools/export_evidence.py               # read from the repo root, write evidence/
    python tools/export_evidence.py /path/to/raw   # read raw files from another root
    python tools/export_evidence.py --check        # re-export into a temp dir, diff, fail on drift

`--check` needs the private raw data to be present under the raw root; it is meant for
the machine that measured the data, not for CI or a fresh clone.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Every raw file a tracked reader opens (`tools/make_figures.py`, `ironmule/numeric_plans.py`,
#: `tests/test_numeric_plans.py`, `tests/test_documented_claims.py`), relative to the raw root.
#: Derived by reading those four files; not the same as "every file a run produced".
KAGGLE_SOURCES = (
    "apple-abcd/abcd-1b.json", "apple-abcd/abcd-4b.json", "apple-abcd/abcd-12b.json",
    "port1-run6-c3af42bd/cross-1b.json", "port1-run6-c3af42bd/cross-4b.json",
    "port1-run7-1f40ad2b/cross-12b.json",
    "perf1-run2-baddcb2b/e2e-qwen3-8b-stock.json", "perf1-run2-baddcb2b/e2e-qwen3-8b-kernel+p16.json",
    "perf1-run2-baddcb2b/e2e-qwen3-14b-stock.json", "perf1-run2-baddcb2b/e2e-qwen3-14b-kernel+p16.json",
    "perf1-run8-aa90d4d2/server-qwen3-8b-kernel+p16-free.json",
    "perf1-run8-aa90d4d2/server-qwen3-8b-kernel+mma+p16-free.json",
    "perf1-run5-a9559a15/gate-qwen3-8b-kernel-decode.json",
    "perf1-run5-a9559a15/gate-qwen3-8b-stock-decode.json",
    "perf1-run5-a9559a15/gate-qwen3-8b-p16-prefill.json",
    "perf1-run5-a9559a15/gate-qwen3-8b-stock-prefill.json",
    "perf1-run5-a9559a15/gate-qwen3-14b-p16-prefill.json",
    "perf1-run5-a9559a15/gate-qwen3-14b-stock-prefill.json",
    "perf1-run18-863237d6/cross-gemma3-1b.json", "perf1-run18-863237d6/cross-gemma3-4b.json",
    "perf1-run18-863237d6/cross-gemma3-12b.json",
    "port2k-run1-fe76f8df/quality-gemma4-e2b-float16.json",
    "port2k-run1-fe76f8df/quality-gemma4-e2b-bf16.json",
    "next1c-run1-1497adb8/gate-gemma4-e2b-fp16-decode.json",
    "next1c-run1-1497adb8/gate-gemma4-e2b-stock-decode.json",
    "port2k-run1-fe76f8df/quality-gemma4-e2b-float32.json",
    "next1c-run1-1497adb8/gate-gemma4-e2b-fp32-decode.json",
    "port2k-run1-fe76f8df/quality-gemma3-4b-float32.json",
    "port2k-run1-fe76f8df/quality-gemma3-4b-bf16.json",
    "next1c-run1-1497adb8/gate-gemma3-4b-fp32-decode.json",
    "next1c-run1-1497adb8/gate-gemma3-4b-stock-decode.json",
    "port2-run8-da1a6469/quality-mistral-24b-float32-8.json",
    "port2-run8-da1a6469/quality-mistral-24b-bf16-8.json",
    "next1c-run1-1497adb8/gate-mistral-24b-fp32-decode.json",
    "next1c-run1-1497adb8/gate-mistral-24b-stock-decode.json",
    "gate-q38-run1-a8f88fc2/gate-qwen38-27b-p16-prefill.json",
    "gate-q38-run1-a8f88fc2/gate-qwen38-27b-stock-prefill.json",
    "gate-q38-run1-a8f88fc2/gate-qwen38-27b-kernel-decode.json",
    "gate-q38-run1-a8f88fc2/gate-qwen38-27b-stock-decode.json",
    "perf1u-run1-b0cce87b/perf1u-summary.json",
    "port2-run6-59ce8efc/cross-fp16-gemma3-4b.json",
    "port2k-run2-39af179b/quality-gemma3-4b-float16.json",
    "port2k-run2-39af179b/quality-gemma3-4b-bf16.json",
    "port2-run4-c86664a3/cross-fused-llama31-8b.json",
    "port2-run2-74fe1a6d/quality-llama31-8b.json",
    "port2-run6-59ce8efc/cross-fp16-qwen3-8b.json",
    "port2-run2-74fe1a6d/quality-qwen3-8b.json",
    "port2-run6-59ce8efc/quality16-qwen3-8b-float16.json",
    "port2-run6-59ce8efc/quality16-qwen3-8b-bf16.json",
    "port2-run6-59ce8efc/cross-fp16-gptoss-20b.json",
    "port2-run7-cddae1f9/quality-gptoss-20b-float32-24.json",
    "port2-run7-cddae1f9/quality-gptoss-20b-float16-16.json",
    "port2-run7-cddae1f9/quality-gptoss-20b-bf16-16.json",
    "port2-run3-281b971a/cross-mistral-lean.json",
    "perf1-run7-080bfab7/cross-native-qwen3-8b.json",
    "perf1-run7-080bfab7/cross-native-qwen3-14b.json",
    "backlog2-run1-17b2ca39/gate-qwen3-14b-kernel-decode.json",
    "backlog2-run1-17b2ca39/gate-qwen3-14b-stock-decode.json",
    "backlog8-run1-1665f2ae/gate-gemma3-12b-kernel-decode.json",
    "backlog8-run1-1665f2ae/gate-gemma3-12b-stock-decode.json",
    "backlog8-run1-1665f2ae/gate-gemma3-12b-p16-prefill.json",
    "backlog8-run1-1665f2ae/gate-gemma3-12b-stock-prefill.json",
    "port2-run9b-e8751c84/cross-gemma4-e2b.json",
    "port2-run9b-e8751c84/cross-gemma4-e4b.json",
    "port2-run9b-e8751c84/cross-gemma4-e4b-qat.json",
    "port2-run4-c86664a3/cross-fused-gemma3-4b.json",
    "port2-run4-c86664a3/cross-fused-qwen3-8b.json",
    "port2-run4-c86664a3/cross-fused-qwen3-14b.json",
    "port2-run2-74fe1a6d/cross-gptoss-20b.json",
    "perf1-run11-7b29bb97/e2e-mistral-24b-stock.json",
    "perf1-run11-7b29bb97/e2e-mistral-24b-kernel.json",
    "perf1-run11-7b29bb97/server-mistral-24b-kernel+mma.json",
    "perf1-run11-7b29bb97/server-mistral-24b-kernel.json",
    "perf1-run12-c4c35978/e2e-qwen3-32b-stock.json",
    "perf1-run12-c4c35978/e2e-qwen3-32b-kernel+p16.json",
    "perf1-run12-c4c35978/server-qwen3-32b-kernel+mma+p16.json",
    "perf1-run13-93ae1f80/e2e-mistral-24b-kernel+p16.json",
    "perf1-run13-93ae1f80/e2e-mistral-24b-kernel.json",
    "perf1-run13-93ae1f80/server-mistral-24b-kernel+mma+p16.json",
)

#: Run directories `tests/test_numeric_plans.py` checks the `pip freeze` of (mlx, mlx-lm
#: versions), read from `<run>/logs/freeze.log`. Exported as a tiny parsed sidecar
#: (`freeze.json`) instead of the raw log, which also carries every other package pinned
#: that run.
FREEZE_RUN_DIRS = (
    "port2-run6-59ce8efc", "port2k-run1-fe76f8df", "port2k-run2-39af179b",
    "port2-run4-c86664a3", "port2-run2-74fe1a6d", "port2-run7-cddae1f9",
    "port2-run3-281b971a", "port2-run8-da1a6469", "perf1-run7-080bfab7",
    "perf1-run5-a9559a15", "backlog2-run1-17b2ca39", "perf1-run18-863237d6",
    "backlog8-run1-1665f2ae", "port2-run9b-e8751c84",
)
#: `ironmule.numeric_plans.MEASURED_WITH` names these; that is what the freeze sidecar keeps.
FRAMEWORK_PACKAGES = ("mlx", "mlx-lm")

APPLE_SOURCES = ("E10-prefix-cache-session-ab.json", "E1-prefill-breakdown.json")

KAGGLE_PREFIX = "experiments/kaggle_compat/results"
APPLE_PREFIX = "research/raw"
HEAD_SKIP_SOURCE = "experiments/head_skip_formal/results.json"

#: Per-row fields kept from a quality file's `rows` list: every NLL a gate or a paired
#: bootstrap reads, never the `kl`/`top1` columns nobody in the tracked code touches.
_ROW_PREFIX = "nll_"
_QUALITY_SCALARS = ("bos", "ppl_bf16", "ppl_fp32", "ppl_float32", "ppl_float16",
                    "ppl_ratio_fp32_over_bf16", "ppl_ratio_fp16_over_bf16", "ppl_ratio_ci",
                    "quality_gate_upper_below_1_005")


def _identity(data: dict) -> tuple[str | None, str | None]:
    """(model_id, revision) from whichever the raw file recorded.

    Some runs name both directly; others only ever wrote the local HuggingFace cache
    path (`.../models--ORG--NAME/snapshots/REVISION`), which is exactly the kind of
    local path this export must not publish. Either way the caller gets the two public
    facts and nothing that names a machine.
    """
    if data.get("model_id") and data.get("revision"):
        return data["model_id"], data["revision"]
    model_path = data.get("model_path")
    if not model_path or "/models--" not in model_path:
        return None, None
    repo, _, revision = model_path.split("/models--", 1)[1].partition("/snapshots/")
    return repo.replace("--", "/", 1), revision.strip("/")


def _redact_head_skip(data: dict) -> dict:
    workload = data["workload"]
    calibration = data["calibration_from_measured_blocks"]
    confirmation = data["confirmation_from_measured_blocks"]
    decision = data["calculated_decision"]
    return {
        "sealed_identity": {"model_id": data["sealed_identity"]["model_id"]},
        "workload": {key: workload[key] for key in
                    ("calibration_sessions", "confirmation_sessions",
                     "measurement_pairs_per_session")},
        "calibration_from_measured_blocks": {
            "aggregate_ratio": calibration["aggregate_ratio"],
            "ci95": calibration["ci95"],
            "session_ratios": calibration["session_ratios"],
        },
        "confirmation_from_measured_blocks": {
            "session_ratios": confirmation["session_ratios"],
            "token_identity": confirmation["token_identity"],
        },
        "calculated_decision": {
            "intervals": {"all": {"ratio": decision["intervals"]["all"]["ratio"],
                                  "ci95": decision["intervals"]["all"]["ci95"]}},
        },
    }


def redact(data: dict) -> dict:
    """Keep only the fields a tracked reader opens; drop everything else.

    Dispatches on which shape the file has rather than its name, because the same
    handful of harness shapes (a wall-ratio square, a stock/candidate summary, a
    per-chunk NLL gate, a quality run with rows, a server sweep, an end-to-end run)
    repeats across every run directory.
    """
    if "sealed_identity" in data:
        return _redact_head_skip(data)

    out: dict = {}
    if "wall_ratios" in data:
        out["wall_ratios"] = data["wall_ratios"]
    if "summary" in data:
        out["summary"] = data["summary"]
    if "chunk_nll" in data:
        out["chunk_nll"] = data["chunk_nll"]
    if "rows" in data and "chunks" in data:
        out["rows"] = [{k: v for k, v in row.items() if k.startswith(_ROW_PREFIX)}
                       for row in data["rows"]]
        for key in _QUALITY_SCALARS:
            if key in data:
                out[key] = data[key]
    if "phases_ms" in data:
        out["phases_ms"] = data["phases_ms"]
        out["repeats"] = data["repeats"]
        out["prompt_tokens"] = data["prompt_tokens"]
    if "ratio_warm" in data:
        out["ratio_warm"] = data["ratio_warm"]
        out["ratio_cold"] = data["ratio_cold"]
        out["processes"] = data["processes"]
    if "pairs" in data and "runs" in data:
        out["pairs"] = {key: [{"flip_rate": chunk["flip_rate"], "cells": chunk["cells"],
                               "flips": chunk["flips"]} for chunk in chunks]
                        for key, chunks in data["pairs"].items()}
    if "decode_tps_median" in data:
        out["decode_tps_median"] = data["decode_tps_median"]
    if "ttft_ms_median" in data:
        out["ttft_ms_median"] = data["ttft_ms_median"]
    if "widths" in data:
        out["widths"] = {width: {"aggregate_tps": row["aggregate_tps"]}
                         for width, row in data["widths"].items()}
    pipeline = data.get("pipeline")
    if isinstance(pipeline, dict) and "gpus_after_load" in pipeline:
        out["pipeline"] = {"gpus_after_load": pipeline["gpus_after_load"]}

    model_id, revision = _identity(data)
    if model_id:
        out["model_id"] = model_id
    if revision:
        out["revision"] = revision

    if not out:
        raise ValueError("unrecognised evidence schema; nothing to keep")
    return out


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Sorted keys and json's own float repr (exact round-trip) make a re-export
    # byte-identical; a trailing newline matches the repository's other tracked files.
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _export_freeze(raw_root: Path, out_root: Path, run_dir: str) -> None:
    freeze_log = raw_root / KAGGLE_PREFIX / run_dir / "logs" / "freeze.log"
    pinned = dict(line.split("==", 1) for line in freeze_log.read_text().split() if "==" in line)
    versions = {name: pinned[name] for name in FRAMEWORK_PACKAGES}
    _write_json(out_root / "evidence" / "kaggle" / run_dir / "freeze.json", versions)


def export_all(raw_root: Path, out_root: Path) -> list[Path]:
    written = []
    for relative in KAGGLE_SOURCES:
        data = json.loads((raw_root / KAGGLE_PREFIX / relative).read_text())
        dest = out_root / "evidence" / "kaggle" / relative
        _write_json(dest, redact(data))
        written.append(dest)
    for relative in APPLE_SOURCES:
        data = json.loads((raw_root / APPLE_PREFIX / relative).read_text())
        dest = out_root / "evidence" / "apple" / relative
        _write_json(dest, redact(data))
        written.append(dest)
    head_skip = json.loads((raw_root / HEAD_SKIP_SOURCE).read_text())
    dest = out_root / "evidence" / "apple" / "head_skip_formal.json"
    _write_json(dest, redact(head_skip))
    written.append(dest)
    for run_dir in FREEZE_RUN_DIRS:
        dest = out_root / "evidence" / "kaggle" / run_dir / "freeze.json"
        _export_freeze(raw_root, out_root, run_dir)
        written.append(dest)
    return written


def _check(raw_root: Path) -> int:
    with tempfile.TemporaryDirectory() as directory:
        fresh_root = Path(directory)
        export_all(raw_root, fresh_root)
        drifted = []
        for fresh in sorted((fresh_root / "evidence").rglob("*.json")):
            committed = ROOT / fresh.relative_to(fresh_root)
            if not committed.is_file():
                drifted.append(f"{committed.relative_to(ROOT)}: not committed")
            elif not filecmp.cmp(fresh, committed, shallow=False):
                drifted.append(f"{committed.relative_to(ROOT)}: differs from a fresh export")
        committed_count = len(list((ROOT / "evidence").rglob("*.json")))
        exported_count = len(list((fresh_root / "evidence").rglob("*.json")))
        if committed_count != exported_count:
            drifted.append(f"{committed_count} files committed but this export writes "
                           f"{exported_count}")
        if drifted:
            print("evidence has drifted from the raw measurements:", file=sys.stderr)
            for line in drifted:
                print(f"  {line}", file=sys.stderr)
            return 1
        print(f"{exported_count} evidence files match a fresh export")
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("raw_root", nargs="?", default=str(ROOT),
                        help="root that holds the private experiments/ and research/raw/ "
                             "trees (default: this repository's root)")
    parser.add_argument("--check", action="store_true",
                        help="re-export into a temp dir and fail on any difference; "
                             "needs the private raw data, so it only runs on the "
                             "machine that measured it")
    args = parser.parse_args(argv)
    raw_root = Path(args.raw_root).resolve()

    if args.check:
        return _check(raw_root)

    written = export_all(raw_root, ROOT)
    print(f"wrote {len(written)} evidence files under {(ROOT / 'evidence').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
