"""Summarize observed GPU controller replays independently of serving qualifications."""
from __future__ import annotations

import math
import re
import statistics

PROBE_SCHEMA = "ironmule.ggml-switch-probe.v1"
EVIDENCE_SCHEMA = "ironmule.gpu-switch-experience.v1"
POLICIES = ("unpriced", "cost_aware")


def _time(value):
    return type(value) in (int, float) and 0 < value and math.isfinite(value)


def _arm(run):
    waves = run.get("waves")
    if not isinstance(waves, list) or not waves:
        raise ValueError("missing switch replay waves")
    workloads, outputs, ttfts, durations = [], [], [], []
    correct = tokens = switches = 0
    for wave in waves:
        if not isinstance(wave, dict) or not _time(wave.get("elapsed_s")) or not isinstance(wave.get("wave"), str):
            raise ValueError("invalid switch replay wave")
        requests = wave.get("requests")
        if not isinstance(requests, list) or not requests:
            raise ValueError("missing switch replay requests")
        durations.append(wave["elapsed_s"])
        switched = wave.get("physical_switch_s")
        if switched is not None and not _time(switched):
            raise ValueError("invalid measured switch time")
        switches += int(switched is not None)
        targets = []
        for request in requests:
            if (not isinstance(request, dict) or not isinstance(request.get("expected"), str)
                    or not _time(request.get("ttft_ms")) or type(request.get("correct")) is not bool
                    or request.get("stop") not in ("length", "eos") or not isinstance(request.get("tokens"), list)
                    or any(type(token) is not int for token in request["tokens"])):
                raise ValueError("invalid switch replay request")
            targets.append(request["expected"])
            outputs.append((request["tokens"], request["stop"]))
            correct += int(request["correct"])
            tokens += len(request["tokens"])
            ttfts.append(request["ttft_ms"])
        workloads.append((wave["wave"], targets))
    elapsed = sum(durations)
    metrics = {"requests": len(outputs), "correct": correct, "output_tokens": tokens, "switches": switches,
               "elapsed_s": elapsed, "tokens_per_second": tokens / elapsed, "median_ttft_ms": statistics.median(ttfts)}
    if not _time(metrics["tokens_per_second"]):
        raise ValueError("invalid measured switch replay throughput")
    return metrics, workloads, outputs


def summarize(report):
    repeats, runs = report.get("repeats"), report.get("replays")
    if (report.get("cost_aware_gpu_branch_active") is not True or type(repeats) is not int
            or repeats < 1 or not isinstance(runs, list) or len(runs) != 2 * repeats):
        raise ValueError("need a completed pair of GPU policy replays for each repeat")
    sources = report.get("source_sha256")
    if (not isinstance(sources, dict) or "lab/portable_gpu/switch_probe.py" not in sources
            or not all(isinstance(v, str) and re.fullmatch("[0-9a-f]{64}", v) for v in sources.values())):
        raise ValueError("missing switch probe source identities")
    settings = {key: report.get(key) for key in ("long_prompt", "max_output_tokens", "threads", "seq_ctx")}
    if not all(type(v) is int and v > 0 for v in settings.values()):
        raise ValueError("missing switch replay workload settings")
    pairs, reference_tasks, reference_outputs = [], None, None
    exact = correct = True
    for seed in range(repeats):
        arms, order = {}, []
        for run in runs[2 * seed:2 * seed + 2]:
            if (not isinstance(run, dict) or type(run.get("seed")) is not int or run["seed"] != seed
                    or run.get("policy") not in POLICIES or run["policy"] in arms):
                raise ValueError("switch replay needs one arm per policy with the same seed")
            label = run["policy"]
            metrics, tasks, outputs = _arm(run)
            if reference_tasks is None:
                reference_tasks, reference_outputs = tasks, outputs
            if tasks != reference_tasks:
                raise ValueError("switch replay workloads differ")
            exact = exact and outputs == reference_outputs
            correct = correct and metrics["correct"] == metrics["requests"]
            arms[label] = metrics
            order.append(label)
        ratios = {key: arms["cost_aware"][key] / arms["unpriced"][key] for key in ("tokens_per_second", "median_ttft_ms")}
        if not all(_time(ratio) for ratio in ratios.values()):
            raise ValueError("invalid switch replay ratio")
        pairs.append({"seed": seed, "order": order, "arms": arms, "cost_aware_over_unpriced": ratios})
    if report.get("all_output_ids_and_stops_match") is not exact or report.get("all_copy_correct") is not correct:
        raise ValueError("switch quality summary disagrees with measured requests")
    rate, ttft = (statistics.median(p["cost_aware_over_unpriced"][key] for p in pairs)
                  for key in ("tokens_per_second", "median_ttft_ms"))
    if not (exact and correct):
        outcome = "quality_rejected"
    elif rate == ttft == 1:
        outcome = "unchanged"
    elif rate >= 1 and ttft <= 1:
        outcome = "improvement"
    elif rate <= 1 and ttft >= 1:
        outcome = "regression"
    else:
        outcome = "tradeoff"
    return {"source_code_sha256": sources, "workload": settings, "repeats": repeats,
            "both_orders_measured": len({tuple(p["order"]) for p in pairs}) == 2,
            "all_output_ids_and_stops_match": exact, "all_copy_correct": correct, "outcome": outcome,
            "qualification": "observation_only", "cost_aware_throughput_change_pct": 100 * (rate - 1),
            "cost_aware_median_ttft_change_pct": 100 * (ttft - 1), "paired_replays": pairs}
