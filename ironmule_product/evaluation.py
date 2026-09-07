"""Pure fail-closed evaluator for the frozen calibration report."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from friday_evidence.statistics import paired_ratio

from .calibration_plan import POLICY, plan_id, schedule
from friday_evidence.registry import DEFAULT_BUDGET_POLICY


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HASH_FIELDS = ("model_sha256", "environment_sha256", "code_sha256", "hardware_sha256")
_MIB = 1024 * 1024


def _invalid(reasons: list[str]) -> dict[str, Any]:
    return {
        "verdict": "invalid",
        "reasons": reasons,
        "per_limit": {},
        "activation_allowed": False,
        "performance_claim_allowed": False,
    }


def _finite_positive(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return False
    return math.isfinite(number) and number > 0


def _finite_nonnegative(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return False
    return math.isfinite(number) and number >= 0


def _sha(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _sample_field(sample: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in sample:
            return sample[name]
    return None


def _identity_valid(identity: Any) -> bool:
    if not isinstance(identity, Mapping):
        return False
    if not _sha(identity.get("identity_sha256")):
        return False
    return all(_sha(identity.get(field)) for field in _HASH_FIELDS)


def _identity_equal(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    fields = ("identity_sha256", "model_sha256", "environment_sha256", "code_sha256", "hardware_sha256")
    return all(before.get(field) == after.get(field) for field in fields)


def _output_key(sample: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        sample.get("identity_sha256"),
        sample.get("token_sha256"),
        sample.get("output_sha256"),
        sample.get("finish_reason"),
        sample.get("prompt_tokens"),
        sample.get("completion_tokens"),
    )


def _actual_comparison_valid(sample: Mapping[str, Any]) -> bool:
    token_values = [sample[name] for name in ("tokens_match", "actual_tokens_match", "tokens_equal") if name in sample]
    text_values = [sample[name] for name in ("text_match", "actual_text_match", "text_equal") if name in sample]
    if (len(token_values) == 0 or len(text_values) == 0
            or any(type(value) is not bool for value in token_values + text_values)
            or len(set(token_values)) != 1 or len(set(text_values)) != 1):
        return False
    tokens_match, text_match = token_values[0], text_values[0]
    if not tokens_match or not text_match:
        return False
    # If drivers retain actual values, verify the comparison rather than trusting
    # a claimed boolean.  Their absence is allowed because the frozen report's
    # required contract stores the already-computed comparison booleans.
    for left, right in (("reference_tokens", "candidate_tokens"), ("reference_text", "candidate_text")):
        if left in sample or right in sample:
            if left not in sample or right not in sample or sample[left] != sample[right]:
                return False
    return True


def _sample_valid(sample: Any, descriptor: Mapping[str, Any], report: Mapping[str, Any], workers: Mapping[int, Mapping[str, Any]]) -> str | None:
    if not isinstance(sample, Mapping):
        return "sample is not an object"
    if (type(sample.get("sample_index")) is not int
            or type(sample.get("worker_index")) is not int
            or type(sample.get("limit")) is not int
            or type(sample.get("trace_forwards")) is not bool
            or not isinstance(sample.get("phase"), str)
            or not isinstance(sample.get("variant"), str)):
        return f"sample descriptor types are invalid at {descriptor.get('sample_index')}"
    for key in ("sample_index", "worker_index", "limit", "phase", "variant", "trace_forwards"):
        if sample.get(key) != descriptor.get(key):
            return f"sample descriptor mismatch at {descriptor.get('sample_index')} ({key})"
    if sample.get("status") != "passed" or type(sample.get("correctness")) is not bool or sample.get("correctness") is not True:
        return f"sample {descriptor.get('sample_index')} failed correctness/status"
    if not _actual_comparison_valid(sample):
        return f"sample {descriptor.get('sample_index')} failed actual token/text comparison"
    if not _finite_positive(sample.get("http_wall_seconds")):
        return f"sample {descriptor.get('sample_index')} has invalid HTTP wall time"
    memory_total = report.get("memory_total_bytes")
    if type(memory_total) is not int or memory_total <= 0:
        return "report memory_total_bytes is invalid"
    peak = sample.get("peak_memory_bytes")
    if type(peak) is not int or peak < 0 or peak > POLICY.peak_memory_fraction * memory_total:
        return f"sample {descriptor.get('sample_index')} exceeds peak-memory envelope"
    baseline_swap = report.get("swap_baseline_bytes")
    swap = sample.get("swap_used_bytes")
    if type(baseline_swap) is not int or baseline_swap < 0 or type(swap) is not int or swap < 0 or swap > baseline_swap + POLICY.swap_delta_limit_bytes:
        return f"sample {descriptor.get('sample_index')} exceeds swap envelope"
    prompt_tokens = sample.get("prompt_tokens")
    completion_tokens = sample.get("completion_tokens")
    if type(prompt_tokens) is not int or prompt_tokens <= 0 or type(completion_tokens) is not int or not 0 < completion_tokens <= descriptor["limit"]:
        return f"sample {descriptor.get('sample_index')} has invalid token counts"
    finish_reason = sample.get("finish_reason")
    if finish_reason not in ("stop", "length") or (finish_reason == "length" and completion_tokens != descriptor["limit"]):
        return f"sample {descriptor.get('sample_index')} has invalid finish reason"
    if not _sha(sample.get("token_sha256")) or not _sha(sample.get("output_sha256")):
        return f"sample {descriptor.get('sample_index')} has invalid output hashes"
    identity = sample.get("identity_sha256")
    if not _sha(identity):
        return f"sample {descriptor.get('sample_index')} has invalid identity hash"
    if identity != report["identity_before"]["identity_sha256"]:
        return f"sample {descriptor.get('sample_index')} identity mismatch"
    worker_index = descriptor["worker_index"]
    worker = workers.get(worker_index)
    if worker is None or worker.get("started") is not True or worker.get("closed") is not True:
        return f"sample {descriptor.get('sample_index')} lacks a closed worker proof"
    count = _sample_field(sample, "forward_count", "model_forward_invocations")
    if descriptor["trace_forwards"]:
        if type(count) is not int or count < 0:
            return f"traced sample {descriptor.get('sample_index')} has invalid forward count"
    elif count is not None:
        return f"untraced sample {descriptor.get('sample_index')} exposes a forward count"
    return None


def _stats(samples: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_phase: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for sample in samples:
        by_phase[str(sample["phase"])][str(sample["variant"])] = sample
    pairs: list[tuple[float, float]] = []
    for phase in ("ab", "ba"):
        group = by_phase.get(phase, {})
        if "reference" not in group or "bounded_prefetch" not in group:
            raise ValueError("missing AB/BA variant pair")
        pairs.append((float(group["bounded_prefetch"]["http_wall_seconds"]), float(group["reference"]["http_wall_seconds"])))
    within = paired_ratio([item[0] for item in pairs], [item[1] for item in pairs], resamples=POLICY.bootstrap_resamples, seed=POLICY.bootstrap_seed)
    aa = by_phase.get("aa", {})
    if len(aa) != 1:
        raise ValueError("invalid A/A cell")
    # The two same-arm A/A requests are represented by separate rows; retain
    # their order rather than overwriting by variant.
    aa_rows = [sample for sample in samples if sample["phase"] == "aa"]
    if len(aa_rows) != 2:
        raise ValueError("invalid A/A sample count")
    aa_ratio = paired_ratio([float(aa_rows[1]["http_wall_seconds"])], [float(aa_rows[0]["http_wall_seconds"])], resamples=POLICY.bootstrap_resamples, seed=POLICY.bootstrap_seed)
    return {"within_worker": within, "aa": aa_ratio}


def _budget_valid(report: Mapping[str, Any], samples: list[Any]) -> list[str]:
    reasons: list[str] = []
    events = report.get("resource_events")
    if not isinstance(events, list) or len(events) != len(schedule()):
        return [f"resource_events must contain exactly {len(schedule())} rows"]
    previous_end: float | None = None
    parsed: list[tuple[float, float, float]] = []
    tolerance = 1e-3
    for index, event in enumerate(events):
        if not isinstance(event, Mapping) or set(event) != {"sample_index", "start_monotonic", "end_monotonic", "upper_bound_seconds"}:
            reasons.append(f"resource event {index} shape is invalid")
            continue
        if event.get("sample_index") != index or type(event.get("sample_index")) is not int:
            reasons.append(f"resource event {index} index is invalid")
        values = [event.get("start_monotonic"), event.get("end_monotonic"), event.get("upper_bound_seconds")]
        if (not _finite_nonnegative(values[0]) or not _finite_positive(values[1])
                or not _finite_positive(values[2])):
            reasons.append(f"resource event {index} timing is invalid")
            continue
        start, end, upper = (float(value) for value in values)
        if abs((end - start) - upper) > tolerance:
            reasons.append(f"resource event {index} duration disagrees with upper bound")
        if upper > DEFAULT_BUDGET_POLICY.continuous_gpu_limit_s:
            reasons.append(f"resource event {index} exceeds continuous bound")
        sample = samples[index] if index < len(samples) else None
        wall = sample.get("http_wall_seconds") if isinstance(sample, Mapping) else None
        if not _finite_positive(wall) or upper + tolerance < float(wall):
            reasons.append(f"resource event {index} does not cover HTTP wall time")
        if previous_end is not None and start + tolerance < previous_end + DEFAULT_BUDGET_POLICY.required_break_s:
            reasons.append(f"resource event {index} violates required break")
        previous_end = end
        parsed.append((start, end, upper))
    if reasons:
        return reasons
    timings = report.get("worker_timings")
    timing_by_worker: dict[int, tuple[float, float, float]] = {}
    if not isinstance(timings, list) or len(timings) != 3:
        reasons.append("worker_timings must contain exactly three rows")
    else:
        for timing in timings:
            if not isinstance(timing, Mapping) or set(timing) != {"worker_index", "load_started_monotonic", "ready_monotonic", "closed_monotonic"}:
                reasons.append("worker timing shape is invalid")
                continue
            index = timing.get("worker_index")
            values = (timing.get("load_started_monotonic"), timing.get("ready_monotonic"), timing.get("closed_monotonic"))
            if type(index) is not int or index in timing_by_worker or index not in (0, 1, 2) or any(not _finite_nonnegative(value) for value in values):
                reasons.append("worker timing values are invalid")
                continue
            load_started, ready, closed = (float(value) for value in values)
            if not load_started <= ready <= closed:
                reasons.append(f"worker {index} lifecycle timestamps are unordered")
            timing_by_worker[index] = (load_started, ready, closed)
        if len(timing_by_worker) == 3:
            for index in (1, 2):
                if timing_by_worker[index][0] + tolerance < timing_by_worker[index - 1][2] + 60.0:
                    reasons.append(f"worker {index} started before the registered cooldown")
    if reasons:
        return reasons
    for index, (start, end, _) in enumerate(parsed):
        worker_index = samples[index].get("worker_index") if isinstance(samples[index], Mapping) else None
        timing = timing_by_worker.get(worker_index) if isinstance(worker_index, int) else None
        if timing is None or start + tolerance < timing[1] or end - tolerance > timing[2]:
            reasons.append(f"resource event {index} lies outside its worker lifetime")
    if reasons:
        return reasons
    total = sum(item[2] for item in parsed)
    if total > DEFAULT_BUDGET_POLICY.gpu_work_limit_s + tolerance:
        reasons.append("resource total exceeds GPU-work limit")
    for _, endpoint, _ in parsed:
        rolling = sum(max(0.0, min(end, endpoint) - max(start, endpoint - DEFAULT_BUDGET_POLICY.duty_window_s)) for start, end, _ in parsed if start < endpoint)
        if rolling > DEFAULT_BUDGET_POLICY.duty_window_s * DEFAULT_BUDGET_POLICY.duty_cycle_limit + tolerance:
            reasons.append("resource rolling duty window exceeds limit")
            break
    budget = report.get("budget")
    if not isinstance(budget, Mapping):
        return reasons + ["budget summary is missing"]
    exact_limits = {
        "gpu_work_limit_seconds": DEFAULT_BUDGET_POLICY.gpu_work_limit_s,
        "continuous_gpu_limit_seconds": DEFAULT_BUDGET_POLICY.continuous_gpu_limit_s,
        "duty_cycle_limit": DEFAULT_BUDGET_POLICY.duty_cycle_limit,
        "wall_limit_seconds": DEFAULT_BUDGET_POLICY.wall_limit_s,
        "candidate_cooldown_seconds": DEFAULT_BUDGET_POLICY.candidate_cooldown_s,
        "required_break_limit_seconds": DEFAULT_BUDGET_POLICY.required_break_s,
    }
    for key, expected in exact_limits.items():
        if budget.get(key) != expected:
            reasons.append(f"budget policy field {key} changed")
    for key in ("gpu_work_seconds", "max_continuous_gpu_seconds", "required_break_seconds", "cooldown_seconds", "wall_seconds"):
        valid_number = _finite_nonnegative(budget.get(key)) if key == "cooldown_seconds" else _finite_positive(budget.get(key))
        if not valid_number and not (key == "max_continuous_gpu_seconds" and budget.get(key) == 0):
            reasons.append(f"budget summary field {key} is invalid")
    if _finite_positive(budget.get("gpu_work_seconds")) and abs(float(budget["gpu_work_seconds"]) - total) > tolerance:
        reasons.append("budget GPU work does not match resource-event accounting")
    if _finite_positive(budget.get("max_continuous_gpu_seconds")) and float(budget["max_continuous_gpu_seconds"]) > DEFAULT_BUDGET_POLICY.continuous_gpu_limit_s + tolerance:
        reasons.append("budget continuous work exceeds limit")
    if _finite_positive(budget.get("required_break_seconds")) and float(budget["required_break_seconds"]) + tolerance < len(events) * DEFAULT_BUDGET_POLICY.required_break_s:
        reasons.append("budget required breaks are below the mandatory minimum")
    if not _finite_nonnegative(budget.get("cooldown_seconds")):
        reasons.append("budget cooldown is invalid")
    if _finite_positive(budget.get("wall_seconds")) and float(budget["wall_seconds"]) > DEFAULT_BUDGET_POLICY.wall_limit_s + tolerance:
        reasons.append("budget wall time exceeds limit")
    return reasons


def evaluate_report(report: dict) -> dict[str, Any]:
    """Evaluate a measured calibration report without raising on bad input."""
    reasons: list[str] = []
    try:
        if not isinstance(report, Mapping):
            return _invalid(["report is not an object"])
        if report.get("schema") != "ironmule.calibration.v1":
            reasons.append("schema mismatch")
        if report.get("plan_id") != plan_id():
            reasons.append("calibration plan mismatch")
        if report.get("status") != "measured":
            reasons.append("report status is not measured")
        before, after = report.get("identity_before"), report.get("identity_after")
        if not _identity_valid(before) or not _identity_valid(after) or not _identity_equal(before, after):
            reasons.append("identity before/after mismatch or incomplete hashes")
        memory_total = report.get("memory_total_bytes")
        baseline_swap = report.get("swap_baseline_bytes")
        if type(memory_total) is not int or memory_total <= 0:
            reasons.append("invalid memory_total_bytes")
        if type(baseline_swap) is not int or baseline_swap < 0:
            reasons.append("invalid swap_baseline_bytes")
        if report.get("resource_valid") is not True:
            reasons.append("resource_valid is not true")
        workers_value = report.get("workers")
        workers: dict[int, Mapping[str, Any]] = {}
        if not isinstance(workers_value, list) or len(workers_value) != 3:
            reasons.append("worker proof must contain exactly three workers")
        else:
            for worker in workers_value:
                if not isinstance(worker, Mapping) or set(worker) != {"worker_index", "started", "closed", "pid"}:
                    reasons.append("worker proof shape is invalid")
                    continue
                index, pid = worker.get("worker_index"), worker.get("pid")
                if type(index) is not int or index in workers or index not in (0, 1, 2):
                    reasons.append("worker indexes are not exactly 0,1,2")
                if worker.get("started") is not True or worker.get("closed") is not True or type(pid) is not int or pid <= 0:
                    reasons.append("worker lifecycle proof is invalid")
                if type(index) is int:
                    workers[index] = worker
            if len({worker.get("pid") for worker in workers.values()}) != 3:
                reasons.append("worker pids are not distinct")
        expected = schedule()
        samples = report.get("samples")
        if not isinstance(samples, list) or len(samples) != len(expected):
            reasons.append(f"samples must contain exactly {len(expected)} schedule rows")
            samples = samples if isinstance(samples, list) else []
        valid_samples: list[Mapping[str, Any]] = []
        for descriptor, sample in zip(expected, samples):
            error = _sample_valid(sample, descriptor, report, workers)
            if error:
                reasons.append(error)
            elif isinstance(sample, Mapping):
                valid_samples.append(sample)
        # Require exact schedule length/order, including duplicate worker indexes
        # that might otherwise be hidden by a dictionary projection.
        if len(samples) == len(expected):
            for index, (descriptor, sample) in enumerate(zip(expected, samples)):
                if not isinstance(sample, Mapping) or sample.get("sample_index") != index:
                    reasons.append(f"sample order/index mismatch at {index}")

        reasons.extend(_budget_valid(report, samples))

        # Every variant's identity/output/count tuple must be stable across the
        # three fresh workers for each limit.
        for limit in (1, 8, 32):
            for variant in ("reference", "bounded_prefetch"):
                rows = [sample for sample in valid_samples if sample.get("limit") == limit and sample.get("variant") == variant]
                keys = {_output_key(sample) for sample in rows}
                if len(keys) > 1:
                    reasons.append(f"identity/output/counts differ across workers at limit {limit} {variant}")
            for worker_index in (0, 1, 2):
                traced = [sample for sample in valid_samples if sample.get("limit") == limit
                          and sample.get("worker_index") == worker_index
                          and sample.get("phase") == "warmup" and sample.get("trace_forwards") is True]
                by_variant = {sample.get("variant"): sample for sample in traced}
                if len(by_variant) == 2:
                    ref_count = _sample_field(by_variant["reference"], "forward_count", "model_forward_invocations")
                    cand_count = _sample_field(by_variant["bounded_prefetch"], "forward_count", "model_forward_invocations")
                    completion = by_variant["reference"].get("completion_tokens")
                    expected_delta = 1 if completion == limit else 0
                    if ref_count - cand_count != expected_delta:
                        reasons.append(f"forward delta mismatch at limit {limit} worker {worker_index}")

        if reasons:
            return _invalid(reasons)

        per_limit: dict[str, Any] = {}
        signals = []
        for limit in (1, 8, 32):
            workers_for_limit = []
            aa_deviations = []
            for worker_index in (0, 1, 2):
                rows = [sample for sample in valid_samples if sample.get("limit") == limit and sample.get("worker_index") == worker_index]
                stats = _stats(rows)
                workers_for_limit.append(stats["within_worker"]["median_ratio"])
                aa_deviations.append(abs(stats["aa"]["median_ratio"] - 1.0))
            cluster = paired_ratio(workers_for_limit, [1.0, 1.0, 1.0], resamples=POLICY.bootstrap_resamples, seed=POLICY.bootstrap_seed)
            noise = max(POLICY.min_effect_fraction, POLICY.aa_noise_multiplier * max(aa_deviations))
            mechanism = any(sample.get("completion_tokens") == limit and sample.get("variant") == "reference" and sample.get("trace_forwards") is True for sample in valid_samples if sample.get("limit") == limit)
            signal = cluster["ci_high"] < 1.0 - POLICY.min_effect_fraction and (1.0 - cluster["median_ratio"]) > noise and mechanism
            signals.append(signal)
            per_limit[str(limit)] = {
                "cluster": cluster,
                "aa_max_abs_deviation": max(aa_deviations),
                "noise_threshold": noise,
                "mechanism_forward_delta_observed": mechanism,
                "signal": signal,
                "classification": "calibration_signal" if signal else "inconclusive",
            }
        if all(signals):
            verdict = "calibration_signal"
        elif all(per_limit[str(limit)]["cluster"]["median_ratio"] >= 1.0 for limit in (1, 8, 32)):
            verdict = "no_gain"
        else:
            verdict = "inconclusive"
        return {
            "verdict": verdict,
            "reasons": [],
            "per_limit": per_limit,
            "activation_allowed": False,
            "performance_claim_allowed": False,
        }
    except Exception as exc:
        return _invalid(reasons + [f"evaluator rejected malformed report: {type(exc).__name__}"])


__all__ = ["evaluate_report"]
