"""Replay the registered A100 test from public request and native timing projections."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

from lab.pk6.first_token import phases, trial
from lab.pk6.replay_http import checksum, waves
from lab.pk6.warm_repeat import GGUF
from lab.pk6.warm_result import read


def paired(ref, cand):
    quality_ok = all(r["quality"] == "18/18" for r in ref["quality"] + cand["quality"])
    throughput, p50, p95 = [], [], []
    checks = {"quality": quality_ok}
    records = []
    for a, b, ar, br in zip(ref["waves"], cand["waves"], ref["phases"], cand["phases"], strict=True):
        throughput.append(b["tok_s"] / a["tok_s"])
        p50.append(b["ttft_p50_ms"] / a["ttft_p50_ms"])
        p95.append(b["ttft_p95_ms"] / a["ttft_p95_ms"])
        records.append({"repeat": a["repeat"], "tok_s": [a["tok_s"], b["tok_s"]],
                        "p50_ms": [ar["p50_ms"], br["p50_ms"]], "p95_ms": [ar["p95_ms"], br["p95_ms"]],
                        "max_ms": [ar["max_ms"], br["max_ms"]], "text_digest_match": a["texts"] == b["texts"]})
    forms = [[(c["width"], c["short"]) for c in wave["prefill_calls"]] for wave in ref["phases"] + cand["phases"]]
    checks["prefill_groups"] = forms == [[(32, 16)]] * 3 + [[(17, 16), (15, 0)]] * 3
    checks["paired_work"] = all((a["tokens"], a["prompt_tokens"]) == (b["tokens"], b["prompt_tokens"])
                                and a["tokens"] == width * 32
                                for width, field in ((32, "waves"), (8, "quality"))
                                for a, b in zip(ref[field], cand[field], strict=True))
    checks.update(throughput=median(throughput) >= 1, p50=median(p50) <= .90, p95=median(p95) <= 1)
    return {"waves": records, "checks": checks, "median_ratios": {"throughput": median(throughput),
            "p50": median(p50), "p95": median(p95)}, "reference_candidate_groups": forms,
            "factual": [[w["quality"] for w in arm["quality"]] for arm in (ref, cand)]}


def grade(folder):
    target = read(folder / "target.json")
    hw = target["hardware"]
    complete = read(folder / "completed.json")
    if (target["base"] != trial.BASE or target["model_sha256"] != GGUF
            or target["candidate_patch_sha256"] != checksum(trial.HERE / "lead.patch")
            or target["protocol_sha256"] != checksum(trial.HERE / "protocol.md")
            or hw["name"] != "NVIDIA A100-SXM4-80GB" or complete["hardware"] != hw
            or target["arguments"] != list(trial.ARMS) or complete["order"] != list(trial.ARMS)):
        raise ValueError("unbound target, code or protocol")
    measured, startup = {}, {}
    previous = 0
    for arm in trial.ARMS:
        start, ready, end = (read(folder / f"{p}-{arm}.json") for p in ("start", "ready", "end"))
        cfg = ready["ggml"]
        if (start["hardware"] != hw or end["hardware"] != hw or start["source_commit"] != trial.BASE
                or not previous <= start["time"] < ready["time"] < end["time"]
                or ready["model"] != "gguf/Qwen3-32B-Q4_K_M" or ready["device"] != "gpu"
                or not cfg["gate"]["passes"] or not cfg["stream_qualification"]["passes"]
                or cfg["sequences"]["sequences"] != 32):
            raise ValueError("invalid arm order, identity or server qualification")
        previous = end["time"]
        records = phases.clients(folder, arm)
        native = read(folder / f"native-{arm}.json")
        stats = phases.summarize(records, native)
        measured[arm] = {"waves": waves(folder, arm, "ironmule", 32),
                         "quality": waves(folder, arm + "q", "ironmule", 8), "phases": stats}
        startup[arm] = ready["startup_s"]
    comparisons = {f"R{i}/C{i}": paired(measured[f"R{i}"], measured[f"C{i}"]) for i in (1, 2)}
    passed = all(all(p["checks"].values()) for p in comparisons.values())
    return {"decision": "nominate_holdout" if passed else "retain_reference", "hardware": hw,
            "startup_s": startup, "pairs": comparisons, "protocol_sha256": target["protocol_sha256"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    print(json.dumps(grade(parser.parse_args().folder), indent=2, allow_nan=False))
