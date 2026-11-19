"""Evaluate the width model on entire held-out GPU types. Reads local imported reports only."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ironmule.gpu_experience import HINT_SCHEMA, _design, _fit, _targets


def load_corpus(root):
    groups = defaultdict(dict)
    for path in sorted((root / "gpu-width-experience").glob("*/*.json")):
        report = json.loads(path.read_text())
        if (report["schema"] != HINT_SCHEMA or report["source_sha256"] != path.stem
                or report["model"] != path.parent.name or len(report["devices"]) != 1):
            raise ValueError(f"Invalid single-GPU import {path}")
        group = groups[(report["model"], report["provider"])]
        device = report["devices"][0]
        for pair in report["pairs"]:
            trial = pair["trial"]
            if trial not in group:
                group[trial] = (device, pair.copy())
                continue
            previous_device, previous = group[trial]
            if previous_device != device:
                raise ValueError("The same physical trial was attributed to two GPU types")
            # Duplicate exports cannot cross splits or erase a quality failure.
            if pair["role"] == "holdout":
                previous["role"] = "holdout"
            previous["matched"] = previous["matched"] and pair["matched"]
    return groups


def metrics(cases):
    eligible = [c for c in cases if c["matched"]]
    result = {"scored_pairs": len(eligible), "holdout_pairs": len(cases),
              "quality_rejections": len(cases) - len(eligible),
              "single_slot": sum(c["selected_width"] == 1 for c in cases),
              "wider": sum(c["selected_width"] > 1 for c in cases)}
    result["unsafe_wider_quality"] = sum(c["selected_width"] > 1 and not c["matched"] for c in cases)
    wider = [c for c in eligible if c["selected_width"] > 1]
    result["matched_wider_lost_throughput"] = sum(c["actual"][0] < 0 for c in wider)
    result["matched_wider_later_ttft"] = sum(c["actual"][1] > 0 for c in wider)
    result["matched_wider_met_both_targets"] = sum(c["actual"][0] >= math.log(1.02) and c["actual"][1] <= 0
                                                   for c in wider)
    if eligible:
        for label, field in (("throughput", 0), ("ttft", 1)):
            result[label] = {
                "model_mae": statistics.fmean(abs(c["actual"][field] - c["predicted"][field]) for c in eligible),
                "equal_cost_mae": statistics.fmean(abs(c["actual"][field]) for c in eligible)}
    return result


def evaluate(root):
    folds, skipped, corpus = [], [], []
    for (gguf, backend), trials in sorted(load_corpus(root).items()):
        devices = sorted({device for device, _ in trials.values()})
        for target in devices:
            train = [p for device, p in trials.values() if device != target and p["role"] == "calibration"]
            test = [p for device, p in trials.values() if device == target and p["role"] == "holdout"]
            fit = _fit(train)
            base = {"gguf_sha256": gguf, "backend": backend, "target_gpu": target,
                    "other_gpus": [d for d in devices if d != target], "training_pairs": fit["training_pairs"],
                    "target_holdout_pairs": len(test), "target_quality_rejections": sum(not p["matched"] for p in test)}
            if not fit["weights"] or not any(p["repeats"] >= 2 for p in test):
                skipped.append({**base, "reason": "no eligible other-GPU calibration or no repeated target holdout"})
                continue
            cases = []
            for p in test:
                if p["repeats"] < 2:
                    continue
                narrow, wide = sorted(a["seqs"] for a in p["arms"])
                if narrow != 1:
                    raise ValueError("Transfer evaluation requires a single-slot reference arm")
                x, w = _design(p), fit["weights"]
                predicted = [sum(v * weights[i] for v, weights in zip(x, w)) for i in range(2)]
                # This hypothetical choice sees only shape and source weights. The target's quality
                # flag and measured outcomes enter evaluation strictly after selection.
                action = wide if predicted[0] >= math.log(1.02) and predicted[1] <= 0 else narrow
                cases.append({"trial": p["trial"], "prompt_tokens": p["prompt_tokens"], "concurrency": p["concurrency"],
                              "selected_width": action, "matched": p["matched"], "predicted": predicted,
                              "actual": _targets(p) if p["matched"] else None})
            corpus.extend(cases)
            folds.append({**base, "metrics": metrics(cases), "cases": cases})
    return {"schema": "ironmule.gpu-width-transfer.v1", "folds": folds, "skipped": skipped, "corpus": metrics(corpus)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True, help="IronMule state containing imported GPU reports")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.state_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, delete=False) as tmp:
        try:
            json.dump(report, tmp, indent=2, allow_nan=False)
            tmp.write("\n")
            tmp.close()
            Path(tmp.name).replace(args.output)
        finally:
            Path(tmp.name).unlink(missing_ok=True)
    print(json.dumps({"folds": len(report["folds"]), "skipped": len(report["skipped"]), **report["corpus"]}))


if __name__ == "__main__":
    main()
