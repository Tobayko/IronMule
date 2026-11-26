"""Verify the recorded five EPYC serving stages against the original copy answers and checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from lab.rp.cpu_pilot_withdrawal import COMMIT, HEAD, SMOL

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("live-steady", "live-under_load", "live-withdrawn", "restart-prequalification", "restart-steady")


def checksum(data):
    return hashlib.sha256(data).hexdigest()


def answer_hash(value):
    return checksum(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def replay(study: Path, prior: Path):
    raw = study / "raw"
    original = json.loads(prior.read_bytes())
    manifest = json.loads((raw / "manifest.json").read_bytes())
    if (manifest["outcome"] != "completed" or manifest["commit"] != COMMIT or manifest["head_sha256"] != HEAD
            or manifest["gguf_sha256"] != SMOL or manifest["prior_sha256"] != checksum(prior.read_bytes())):
        raise ValueError("foreign study or incomplete execution")
    rows = {}
    for path in raw.glob("*.json"):
        if path.name == "manifest.json":
            continue
        data = path.read_bytes()
        if manifest["files"][path.name] != {"bytes": len(data), "sha256": checksum(data)}:
            raise ValueError(f"altered artifact: {path.name}")
        rows[path.stem] = json.loads(data)
    learner = original["binding"]["scratch_thread_head"]["learner_binding"]
    if (rows["live-boot"]["pid"] == rows["restart-boot"]["pid"]
            or any(rows[phase + "-boot"]["status"]["binding"] != learner for phase in ("live", "restart"))):
        raise ValueError("process restart or common checkpoint binding is missing")
    summary = {}
    for stage in STAGES:
        episode = rows[stage]
        state = episode["state"]
        hashes, times, ttfts = [], [], []
        for wave, reference in zip(episode["waves"], original["waves"], strict=True):
            answers = wave["answers"]
            if (answers != reference["answers"] or wave["requests"] != reference["requests"]
                    or len(answers) != 4 or any(not a["correct"] for a in answers)
                    or wave["output_sha256"] != answer_hash(answers)
                    or wave["correct"] != 4 or wave["tokens"] != 12 or wave["completed"] != 4):
                raise ValueError(f"different delivered work at {stage}")
            hashes.append(answer_hash(answers))
            times.append(wave["wall_s"])
            ttfts.extend(wave["ttfts_s"])
        if (len(hashes) != 4 or answer_hash(hashes) != original["output_sha256"]
                or episode["digest"] != answer_hash(hashes) or state["status"]["binding"] != learner
                or any(not math.isfinite(t) or t <= 0 for t in [*times, *ttfts])):
            raise ValueError(f"incomplete or foreign serving work at {stage}")
        summary[stage] = {"correct": 16, "tokens": 48, "output_sha256": answer_hash(hashes),
                          "qualified_decisions": episode["qualified_decisions"], "head_available": state["head_available"],
                          "active_offsets": state["status"]["weights"], "serving_wall_s": sum(times),
                          "ttft_p95_s": sorted(ttfts)[math.ceil(len(ttfts) * .95) - 1]}
    for phase in ("live", "restart"):
        steady = summary[phase + "-steady"]
        if (not steady["head_available"] or not steady["qualified_decisions"] or steady["active_offsets"]
                != rows[phase + "-qualification"]["binding"]["qualified_offsets"]):
            raise ValueError("steady traffic did not use the newly qualified head")
    withdrawn, cold = summary["live-withdrawn"], summary["restart-prequalification"]
    if (withdrawn["head_available"] or not cold["head_available"] or any(s["qualified_decisions"] or
            s["active_offsets"] for s in (withdrawn, cold))):
        raise ValueError("ineligible scratch profile was applied")
    return {"elapsed_s": manifest["elapsed_s"], "stages": summary, "verified_correct": 80,
            "load_recheck": rows["live-load_recheck"]["after"]["status"]["last"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=Path, default=ROOT / "lab/autonomous/results/runpod-epyc-pilot/withdrawal")
    p.add_argument("--prior", type=Path, default=ROOT / "evidence/runtime/cpu-pilot-epyc-discovery.json")
    args = p.parse_args()
    print(json.dumps(replay(args.study, args.prior), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
