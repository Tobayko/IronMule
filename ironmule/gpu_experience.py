"""Convert paired GPU serving measurements into workload hints for a fresh local width search.

Only the calibration split contributes. Imported rates are never used as local step times
or qualifications; the local backend measures every shape that enters its cost model.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import sys
import tempfile
from pathlib import Path

SCHEMA = "ironmule.gpu-width-bench.v1"
HINT_SCHEMA = "ironmule.gpu-width-hint.v1"


def _positive_series(values):
    return (isinstance(values, list) and bool(values)
            and all(type(v) in (int, float) and 0 < v <= sys.float_info.max for v in values))


def _device_models(devices):
    if not isinstance(devices, (tuple, list)):
        return []
    return sorted({str(row.get("description") or row.get("name")) for row in devices
                   if isinstance(row, dict) and (row.get("description") or row.get("name"))})


def _design(pair):
    narrow, wide = sorted(pair["arms"], key=lambda arm: arm["seqs"])
    n = pair["concurrency"]
    spread = math.log2(wide["seqs"] / narrow["seqs"])
    occupancy = math.log2(min(n, wide["seqs"]) / min(n, narrow["seqs"]))
    context = math.log2(1 + pair["prompt_tokens"]) / 10
    return [spread, occupancy, spread * context, occupancy * context, spread * math.log2(n) / 4]


def _targets(pair):
    narrow, wide = sorted(pair["arms"], key=lambda arm: arm["seqs"])
    return [math.log(wide[k]) - math.log(narrow[k]) for k in ("rate", "ttft_ms")]


def _fit(rows):
    """Fit paired log throughput and latency ratios from zero with a ridge prior towards equal cost."""
    import numpy as np

    train = [r for r in rows if r["role"] == "calibration" and r["matched"] and r["repeats"] >= 2]
    holdout = [r for r in rows if r["role"] == "holdout" and r["matched"] and r["repeats"] >= 2]
    if not train:
        return {"training_pairs": 0, "holdout_pairs": 0, "weights": []}
    x, y = np.asarray([_design(r) for r in train]), np.asarray([_targets(r) for r in train])
    w = np.linalg.solve(x.T @ x + 0.5 * np.eye(x.shape[1]), x.T @ y)
    result = {"training_pairs": len(train), "holdout_pairs": len(holdout), "weights": w.tolist()}
    if holdout:
        actual = np.asarray([_targets(r) for r in holdout])
        predicted = np.asarray([_design(r) for r in holdout]) @ w
        result.update(holdout_mae=np.abs(predicted - actual).mean(axis=0).tolist(),
                      equal_cost_mae=np.abs(actual).mean(axis=0).tolist())
    return result


def _transition_evidence(replay):
    """Keep paired, switch-inclusive replay outcomes apart from the width training data."""
    if not isinstance(replay, dict) or type(replay.get("quality_matched")) is not bool:
        raise ValueError("invalid GPU transition replay")
    arms = replay.get("arms")
    if not isinstance(arms, dict) or not all(isinstance(arms.get(k), list) for k in ("fixed", "selected")):
        raise ValueError("GPU transition needs fixed and selected arms")
    fixed, selected = arms["fixed"], arms["selected"]
    if not fixed or len(fixed) != len(selected):
        raise ValueError("GPU transition needs paired repeats")
    repeats, elapsed_ratios, switches, quality = [], [], 0, True
    for control, choice in zip(fixed, selected):
        if not isinstance(control, list) or not isinstance(choice, list) or not control or len(control) != len(choice):
            raise ValueError("GPU transition needs paired waves")
        waves = []
        elapsed = [0.0, 0.0]
        for baseline, candidate in zip(control, choice):
            wave = {}
            for i, (key, row) in enumerate((("fixed", baseline), ("selected", candidate))):
                if not isinstance(row, dict):
                    raise ValueError("invalid GPU transition wave")
                ctx, n, width = (row.get(k) for k in ("prompt_tokens", "concurrency", "seqs"))
                correct, tokens = row.get("correct"), row.get("output_tokens")
                if (not all(type(v) is int and v > 0 for v in (ctx, n, width))
                        or not _positive_series([row.get("elapsed_s"), row.get("ttft_ms")])
                        or type(row.get("reconfigured")) is not bool
                        or not isinstance(correct, list) or len(correct) != n or any(type(v) is not bool for v in correct)
                        or not isinstance(tokens, list) or len(tokens) != n or any(not isinstance(v, list) for v in tokens)):
                    raise ValueError("invalid GPU transition measurements")
                if i and (ctx, n) != (wave["prompt_tokens"], wave["concurrency"]):
                    raise ValueError("GPU transition arms must replay the same workload")
                wave.update(prompt_tokens=ctx, concurrency=n)
                wave[key] = {k: row[k] for k in ("seqs", "reconfigured", "elapsed_s", "ttft_ms")}
                elapsed[i] += row["elapsed_s"]
            wave["matched"] = (baseline["output_tokens"] == candidate["output_tokens"]
                               and all(baseline["correct"]) and all(candidate["correct"]))
            quality = quality and wave["matched"]
            switches += int(candidate["reconfigured"])
            waves.append(wave)
        repeats.append(waves)
        ratio = elapsed[1] / elapsed[0]
        if not math.isfinite(ratio):
            raise ValueError("non-finite GPU transition elapsed ratio")
        elapsed_ratios.append(ratio)
    if (replay["quality_matched"] and not quality
            or type(replay.get("switches")) is not int or switches != replay["switches"]):
        raise ValueError("GPU transition summary disagrees with its measured waves")
    return {"quality_matched": replay["quality_matched"], "switches": switches,
            "selected_over_fixed_elapsed": elapsed_ratios, "waves": repeats}


def import_report(path: Path, state: Path) -> dict:
    """Persist compact calibration outcomes, including unsuccessful comparisons. Import is idempotent."""
    raw = path.read_bytes()
    report = json.loads(raw)
    if not isinstance(report, dict) or report.get("schema") != SCHEMA:
        raise ValueError(f"expected {SCHEMA}")
    if report.get("status") != "complete" or report.get("stage") != "compare":
        raise ValueError("only completed GPU width comparisons can be imported")
    offload = report.get("offloaded_layers")
    if not isinstance(offload, list) or not any(
        isinstance(a, list) and len(a) == 2 and all(type(k) is int for k in a) and a[0] > 0 and a[0] == a[1]
        for a in offload
    ):
        raise ValueError("GPU measurement needs confirmed layer offload")
    model = report.get("model_sha256")
    provider, machine = report.get("provider"), report.get("machine")
    devices = report.get("devices")
    if not isinstance(model, str) or not re.fullmatch("[0-9a-f]{64}", model):
        raise ValueError("GPU measurements need the GGUF SHA-256")
    if (provider not in ("cuda", "rocm", "vulkan", "metal", "sycl") or not _device_models(devices)
            or not isinstance(machine, str) or not machine
            or any(not isinstance(d, dict) or d.get("provider") != provider for d in devices)):
        raise ValueError("GPU measurements need a matching backend, device inventory and machine identity")
    summary = []
    runs = report.get("runs")
    if not isinstance(runs, list):
        raise ValueError("missing paired GPU runs")
    for row in runs:
        if not isinstance(row, dict) or row.get("role") not in ("calibration", "holdout"):
            continue
        ctx, concurrency = row.get("prompt_tokens"), row.get("concurrency")
        if (type(ctx) is not int or ctx <= 0 or type(concurrency) is not int or concurrency <= 0
                or type(row.get("quality_matched")) is not bool or not isinstance(row.get("arms"), list)):
            raise ValueError("invalid calibration workload or quality result")
        arms, times, repeats = [], [], None
        for arm in row["arms"]:
            if not isinstance(arm, dict):
                raise ValueError("invalid calibration arm")
            width, rates, ttft = arm.get("seqs"), arm.get("tokens_per_second"), arm.get("ttft_ms")
            if (type(width) is not int or not 1 <= width <= 64 or not _positive_series(rates)
                    or not _positive_series(ttft) or len(rates) != len(ttft)
                    or repeats is not None and repeats != len(rates)):
                raise ValueError("each width needs paired, positive throughput and first-token times")
            median_rate, median_ttft = statistics.median(rates), statistics.median(ttft)
            if not math.isfinite(median_rate) or not math.isfinite(median_ttft):
                raise ValueError("non-finite median in GPU measurement")
            repeats = len(rates)
            times.append([width, rates, ttft])
            arms.append({"seqs": width, "rate": median_rate, "ttft_ms": median_ttft})
        if len(arms) != 2 or arms[0]["seqs"] == arms[1]["seqs"]:
            raise ValueError("a calibration pair must compare two distinct sequence widths")
        matched = row["quality_matched"]
        if matched and not (row.get("exact_output_match") is True and row.get("all_copy_tasks_correct") is True):
            raise ValueError("quality_matched disagrees with the reported token and correctness checks")
        trial = json.dumps([machine, model, ctx, concurrency, sorted(times)], separators=(",", ":"))
        summary.append({"trial": hashlib.sha256(trial.encode()).hexdigest(), "role": row["role"],
                        "prompt_tokens": ctx, "concurrency": concurrency, "matched": matched,
                        "repeats": repeats, "arms": arms})
    if not any(r["role"] == "calibration" for r in summary):
        raise ValueError("no calibration measurements")
    digest = hashlib.sha256(raw).hexdigest()
    record = {"schema": HINT_SCHEMA, "model": model, "provider": provider, "machine": machine,
              "devices": _device_models(devices), "source_sha256": digest, "pairs": summary, "fit": _fit(summary)}
    replay = {}
    if "transition" in report:
        record["transition"] = _transition_evidence(report["transition"])
        replay["transition"] = {k: v for k, v in record["transition"].items() if k != "waves"}
    from ironmule_product.state import _ensure_directory

    directory = state / "gpu-width-experience" / model
    for private in (state, directory.parent, directory):
        _ensure_directory(private)
    with tempfile.NamedTemporaryFile(mode="w", dir=directory, delete=False) as handle:
        try:
            json.dump(record, handle, allow_nan=False)
            handle.close()
            Path(handle.name).replace(directory / f"{digest}.json")
        finally:
            Path(handle.name).unlink(missing_ok=True)
    return {"source_sha256": digest, "model_sha256": model, "fit": record["fit"], **replay,
            "pooled_fit": _fit(_pooled_pairs(state, model, devices, machine)),
            "quality_mismatches": sum(not row["matched"] for row in summary),
            "coverage": {role: {"pairs": sum(r["role"] == role for r in summary),
                                "quality_rejections": sum(r["role"] == role and not r["matched"] for r in summary)}
                         for role in ("calibration", "holdout")}}


def _pooled_pairs(state: Path, model: str, devices, machine: str):
    """Use the closest scope available and count each measured trial once."""
    models = _device_models(devices)
    if not models or len({device["provider"] for device in devices}) != 1:
        return []
    provider = devices[0]["provider"]
    records = []
    for path in sorted((state / "gpu-width-experience" / model).glob("*.json")):
        try:
            record = json.loads(path.read_text())
            if (record.get("schema") == HINT_SCHEMA and record["model"] == model and record["provider"] == provider
                    and record["source_sha256"] == path.stem and isinstance(record["pairs"], list)):
                scope = (record["machine"] == machine, record["devices"] == models)
                records.append((scope, record))
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            continue
    if not records:
        return []
    # Pool all observations in the closest hardware scope. Source reimports or overlapping reports
    # must not amplify a result. A holdout stays held out even if another report relabels it.
    scope = max(rank for rank, _ in records)
    unique = {}
    for rank, record in records:
        if rank == scope:
            for p in record["pairs"]:
                prev = unique.get(p["trial"])
                if prev is None or p["role"] == "holdout" or not p["matched"] and prev["role"] == "calibration":
                    unique[p["trial"]] = p
    return [unique[k] for k in sorted(unique)]


def suggest(state: Path, model: str, devices, machine: str, seq_ctx: int):
    """Pick one transferable workload from a joint fit in the closest available hardware scope."""
    try:
        pairs = _pooled_pairs(state, model, devices, machine)
        fit = _fit(pairs)
        weights = fit["weights"]
        if not weights:
            return None
        def shape(pair):
            return (pair["prompt_tokens"], pair["concurrency"], *sorted(a["seqs"] for a in pair["arms"]))
        bad = {shape(p) for p in pairs if p["role"] == "calibration" and not p["matched"]}
        best, largest = None, 0.0
        for pair in pairs:
            ctx, n, narrow, wide = shape(pair)
            if (pair["role"] != "calibration" or pair["repeats"] < 2 or shape(pair) in bad
                    or not (0 < ctx <= min(seq_ctx - 16, 1024) and 0 < n <= 16 and 1 <= narrow < wide <= 64)):
                continue
            features = _design(pair)
            gain, latency = [sum(x * w[i] for x, w in zip(features, weights)) for i in range(2)]
            if not (math.isfinite(gain) and math.isfinite(latency) and abs(gain) > largest and gain * latency <= 0):
                continue
            largest = abs(gain)
            best = {"prompt_tokens": ctx, "concurrency": n, "widths": [wide, narrow] if gain > 0 else [narrow, wide],
                    "training_pairs": fit["training_pairs"]}
        if best:
            # Bind the search to the pooled evidence, not to a single historical best run.
            best["experience_sha256"] = hashlib.sha256(json.dumps(pairs, sort_keys=True).encode()).hexdigest()
        return best
    except (TypeError, ValueError, KeyError, AttributeError, IndexError, OverflowError, ZeroDivisionError):
        return None
