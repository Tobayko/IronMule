"""Reproducible, end-to-end benchmark of IronMule's service modes.

The public number is measured around ``Runtime.serve`` (``outer_wall_ms``). The
executor's own wall clock and prefill/decode timings remain diagnostics. Both arms
use independent plans, the same warmup/repeat protocol, and alternating order. A
stock ``mlx_lm`` comparison arm is deliberately not part of this benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
import time
from pathlib import Path
from typing import Any, Iterable

DOCUMENT = (
    "Apple silicon uses a unified memory architecture in which the CPU and the GPU "
    "address the same physical memory. MLX evaluates arrays lazily and executes them "
    "on the GPU, so a timer must synchronise before it can be trusted. Quantised "
    "weights are stored in four bits with a scale and a bias for every group of 64 "
    "values. Decoding one token reads the whole weight set, which is why decode is "
    "bound by memory bandwidth rather than by arithmetic. "
) * 5

QUESTIONS = [
    "What does unified memory avoid?",
    "How are quantised weights stored?",
    "Why must a timer synchronise?",
    "What bounds decoding?",
    "What does MLX do lazily?",
    "Which two units share the memory?",
    "How large is a quantisation group?",
    "What is read when decoding one token?",
]
ARM_NAMES = ("interactive", "throughput")


def _value(result: Any, name: str, default: Any = None) -> Any:
    if isinstance(result, dict):
        return result.get(name, default)
    return getattr(result, name, default)


def _json_value(value: Any) -> Any:
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _output_record(result: Any, eos_ids: Iterable[int] = ()) -> dict[str, Any]:
    """Keep raw token IDs, stop reason, and physical/visible counts in evidence."""
    tokens = _json_value(_value(result, "tokens", []))
    metrics = _value(result, "metrics", {}) or {}
    physical = len(tokens) if isinstance(tokens, list) else None
    visible = metrics.get("visible_generated_tokens") if isinstance(metrics, dict) else None
    if visible is None and isinstance(tokens, list) and eos_ids:
        visible = sum(token not in set(eos_ids) for token in tokens)
    return {
        "rid": str(_value(result, "rid", "")), "tokens": tokens,
        "stop_reason": _json_value(_value(result, "stop_reason")),
        "text": _json_value(_value(result, "text")),
        "physical_token_count": physical, "visible_token_count": visible,
    }


def output_diff(expected: Iterable[Any], actual: Iterable[Any]) -> list[dict[str, Any]]:
    """Return a structured, JSON-safe diff for generated answers."""
    expected_by_rid = {
        str(_value(result, "rid", index)): result
        for index, result in enumerate(expected)
    }
    actual_by_rid = {
        str(_value(result, "rid", index)): result
        for index, result in enumerate(actual)
    }
    differences: list[dict[str, Any]] = []
    for rid in sorted(set(expected_by_rid) | set(actual_by_rid)):
        if rid not in expected_by_rid:
            differences.append({"rid": rid, "kind": "unexpected_result"})
            continue
        if rid not in actual_by_rid:
            differences.append({"rid": rid, "kind": "missing_result"})
            continue
        want = expected_by_rid[rid]
        got = actual_by_rid[rid]
        fields: dict[str, dict[str, Any]] = {}
        expected_tokens = _json_value(_value(want, "tokens", []))
        actual_tokens = _json_value(_value(got, "tokens", []))
        if isinstance(expected_tokens, list) and isinstance(actual_tokens, list):
            first_difference = next(
                (index for index, (want_token, got_token)
                 in enumerate(zip(expected_tokens, actual_tokens))
                 if want_token != got_token),
                min(len(expected_tokens), len(actual_tokens)),
            )
            if (expected_tokens != actual_tokens):
                fields["first_token_difference"] = {
                    "position": first_difference,
                    "expected": expected_tokens[first_difference]
                    if first_difference < len(expected_tokens) else None,
                    "actual": actual_tokens[first_difference]
                    if first_difference < len(actual_tokens) else None,
                }
        for field in ("tokens", "stop_reason", "text"):
            expected_value = _json_value(_value(want, field))
            actual_value = _json_value(_value(got, field))
            if expected_value != actual_value:
                fields[field] = {"expected": expected_value, "actual": actual_value}
        want_metrics = _value(want, "metrics", {}) or {}
        got_metrics = _value(got, "metrics", {}) or {}
        if isinstance(want_metrics, dict) and isinstance(got_metrics, dict):
            count_diff = {
                "physical": {
                    "expected": len(expected_tokens) if isinstance(expected_tokens, list) else None,
                    "actual": len(actual_tokens) if isinstance(actual_tokens, list) else None,
                },
                "visible": {
                    "expected": want_metrics.get("visible_generated_tokens"),
                    "actual": got_metrics.get("visible_generated_tokens"),
                },
            }
            for field in ("generated_tokens", "physical_tokens"):
                expected_value = _json_value(want_metrics.get(field))
                actual_value = _json_value(got_metrics.get(field))
                if expected_value != actual_value and (expected_value is not None or actual_value is not None):
                    fields[f"metrics.{field}"] = {
                        "expected": expected_value, "actual": actual_value,
                    }
            if fields:
                fields["token_counts"] = count_diff
        if fields:
            differences.append({"rid": rid, "kind": "field_mismatch", "fields": fields})
    return differences


def _run(rt: Any, ironmule: Any, mode: Any, requests: list[Any]) -> tuple[list[Any], dict[str, Any]]:
    """Run a complete service request set and preserve executor diagnostics."""
    rt.mode = mode
    started = time.perf_counter_ns()
    results = rt.serve(requests)
    outer_wall_ms = (time.perf_counter_ns() - started) / 1e6
    snapshot = dict(rt.telemetry.snapshot())
    # The executor wall clock starts inside the service and is diagnostic only.
    snapshot["outer_wall_ms"] = outer_wall_ms
    snapshot["primary_wall_ms"] = outer_wall_ms
    physical_count = _metric_number(
        snapshot, "physical_token_count", "physical_tokens", "generated_tokens"
    )
    if physical_count is None:
        physical_count = sum(len(_value(result, "tokens", ())) for result in results)
    visible_count = _metric_number(
        snapshot, "visible_token_count", "visible_tokens", "visible_generated_tokens"
    )
    if visible_count is None:
        eos_ids = tuple(getattr(getattr(rt, "backend", None), "eos_ids", ()))
        visible_count = sum(
            _output_record(result, eos_ids)["visible_token_count"] or 0 for result in results
        )
    wall_s = outer_wall_ms / 1000.0
    snapshot["physical_token_count"] = physical_count
    snapshot["visible_token_count"] = visible_count
    snapshot["physical_tokens_per_second"] = physical_count / wall_s if wall_s > 0 else 0.0
    snapshot["visible_tokens_per_second"] = visible_count / wall_s if wall_s > 0 else 0.0
    return results, snapshot


def _sample_summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("benchmark requires at least one measured sample")
    ordered = sorted(float(value) for value in values)
    return {
        "n": len(ordered), "median": st.median(ordered), "min": ordered[0],
        "max": ordered[-1], "stdev": st.stdev(ordered) if len(ordered) > 1 else 0.0,
    }


def _paired_ratio(candidate: list[float], baseline: list[float]) -> dict[str, Any]:
    """Bootstrap a paired median ratio while retaining raw paired ratios."""
    if len(candidate) != len(baseline) or not candidate:
        raise ValueError("paired benchmark samples must be non-empty and equal in length")
    import random

    pairs = [float(c) / float(b) for c, b in zip(candidate, baseline)]
    if any(not pair or pair != pair for pair in pairs):
        raise ValueError("benchmark sample contains an invalid ratio")
    rng = random.Random(20260827)
    resamples = 10_000
    medians = sorted(
        st.median(rng.choice(pairs) for _ in pairs) for _ in range(resamples)
    )
    return {
        "median_ratio": st.median(pairs),
        "ci_low": medians[int(0.025 * resamples)],
        "ci_high": medians[int(0.975 * resamples)],
        "ratios": pairs,
    }


def _new_plan(rt: Any, ironmule: Any, plan_name: str, arm: str) -> Any:
    """Create one plan per arm; no plan or prefix cache crosses the arm boundary."""
    if plan_name == "strict":
        return ironmule.StrictOneShotPlan()
    return rt.session_plan(DOCUMENT, name=f"benchmark-{arm}")


def _build_requests(rt: Any, ironmule: Any, plan: Any, plan_name: str, count: int,
                    max_tokens: int) -> list[Any]:
    questions = (QUESTIONS * ((count + len(QUESTIONS) - 1) // len(QUESTIONS)))[:count]
    prompts = [
        (DOCUMENT + "\n\nQuestion: " + question) if plan_name == "reusable" else question
        for question in questions
    ]
    return [
        ironmule.Request(
            prompt_ids=rt.encode(prompt), max_tokens=max_tokens, plan=plan,
            rid=f"q{index}",
        )
        for index, prompt in enumerate(prompts)
    ]


def _validate_protocol(warmup: int, repeats: int) -> None:
    if warmup < 2:
        raise ValueError("benchmark requires at least two warmup runs per arm")
    if repeats < 2:
        raise ValueError("benchmark requires at least two measured repeats per arm")
    if repeats % 2:
        raise ValueError("benchmark repeats must be even for balanced AB/BA positions")


def _workload(rt: Any, plan_name: str, requests: int, max_tokens: int,
              warmup: int, repeats: int) -> dict[str, Any]:
    questions = (QUESTIONS * ((requests + len(QUESTIONS) - 1) // len(QUESTIONS)))[:requests]
    prompts = [
        (DOCUMENT + "\n\nQuestion: " + question) if plan_name == "reusable" else question
        for question in questions
    ]
    return {
        "requests": requests, "max_tokens": max_tokens, "plan": plan_name,
        "prompt_token_lengths": [len(rt.encode(prompt)) for prompt in prompts],
        "warmup": warmup, "repeats": repeats,
    }


def _runtime_fingerprint(rt: Any, plan: Any, workload: dict[str, Any], mode: Any) -> Any:
    previous_mode = getattr(rt, "mode", None)
    rt.mode = mode
    try:
        return rt.fingerprint(plan, workload)
    except (AttributeError, TypeError):
        return {"status": "unavailable_in_test_runtime"}
    finally:
        rt.mode = previous_mode


def _metric_number(snapshot: dict[str, Any], *names: str) -> int | float | None:
    for name in names:
        value = snapshot.get(name)
        if isinstance(value, (int, float)):
            return value
    return None


def phase_roofline_diagnostic(
    *,
    prefill_ns: int | float | None,
    decode_ns: int | float | None,
    decode_steps: int | None,
    effective_bandwidth_gbps: int | float | None,
    active_weight_bytes_per_token: int | float | None,
    kv_read_bytes_per_token: int | float | None,
    kv_write_bytes_per_token: int | float | None,
    extra_bytes_per_token: int | float | None,
    bandwidth_source: str | None,
    bandwidth_source_kind: str | None,
) -> dict[str, Any]:
    """Return a fail-closed, per-run prefill/decode roofline diagnostic.

    This is deliberately a calculation over supplied measurements, not a probe or
    a tuning decision. ``decode_steps`` excludes the first token produced by
    prefill, matching ``Engine.generate``'s phase clock. Missing traffic components
    remain inconclusive instead of being silently treated as zero. Only a
    measured-effective bandwidth may produce a roofline efficiency; nominal peak
    bandwidth is retained as provenance but remains inconclusive.
    """
    missing: list[str] = []
    invalid: list[str] = []

    def numeric(name: str, value: Any) -> float | None:
        if value is None:
            missing.append(name)
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            invalid.append(f"{name}: expected a finite number")
            return None
        try:
            converted = float(value)
        except (OverflowError, ValueError):
            invalid.append(f"{name}: expected a finite number")
            return None
        if not math.isfinite(converted):
            invalid.append(f"{name}: expected a finite number")
            return None
        if converted < 0:
            invalid.append(f"{name}: must be non-negative")
            return None
        return converted

    def positive_duration(name: str, value: Any) -> tuple[float | None, str]:
        before = len(invalid)
        converted = numeric(name, value)
        if converted is None:
            return None, "invalid" if len(invalid) > before else "unavailable"
        if converted <= 0:
            invalid.append(f"{name}: must be positive when measured")
            return converted, "invalid"
        return converted, "measured"

    prefill_value, prefill_status = positive_duration("prefill_ns", prefill_ns)
    steps: int | None
    if decode_steps is None:
        missing.append("decode_steps")
        steps = None
    elif isinstance(decode_steps, bool) or not isinstance(decode_steps, int):
        invalid.append("decode_steps: expected an integer")
        steps = None
    elif decode_steps < 0:
        invalid.append("decode_steps: must be non-negative")
        steps = None
    elif decode_steps > 2**63 - 1:
        invalid.append("decode_steps: exceeds 2**63-1")
        steps = None
    else:
        steps = decode_steps

    if steps == 0:
        if isinstance(decode_ns, bool):
            invalid.append("decode_ns: expected zero, not a boolean")
            decode_value, decode_status = None, "invalid"
        elif decode_ns is None:
            decode_value, decode_status = None, "not_applicable"
        elif isinstance(decode_ns, (int, float)) and decode_ns == 0:
            decode_value, decode_status = 0.0, "not_applicable"
        else:
            decode_value, decode_status = positive_duration("decode_ns", decode_ns)
            invalid.append("decode_ns: must be zero for a zero-step decode")
            decode_status = "invalid"
    elif steps is None:
        before = len(invalid)
        decode_value = numeric("decode_ns", decode_ns)
        decode_status = "invalid" if len(invalid) > before else "unavailable"
    else:
        decode_value, decode_status = positive_duration("decode_ns", decode_ns)

    def checked_divide(name: str, numerator: float, denominator: float) -> float | None:
        if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator == 0:
            invalid.append(f"{name}: non-finite or zero denominator")
            return None
        try:
            result = numerator / denominator
        except (OverflowError, ZeroDivisionError):
            invalid.append(f"{name}: non-finite or zero denominator")
            return None
        if not math.isfinite(result):
            invalid.append(f"{name}: result is not finite")
            return None
        if result <= 0:
            invalid.append(f"{name}: result is not positive")
            return None
        return result

    prefill_ms = (
        checked_divide("prefill_ms", prefill_value, 1e6)
        if prefill_value is not None else None
    )
    decode_tokens_per_second = (
        checked_divide("decode_tokens_per_second", float(steps), decode_value / 1e9)
        if steps and decode_value is not None else None
    )
    phases: dict[str, Any] = {
        "prefill": {
            "duration_ns": prefill_value,
            "duration_ms": prefill_ms,
            "status": prefill_status,
        },
        "decode": {
            "duration_ns": decode_value,
            "decode_steps": steps,
            "tokens_per_second": decode_tokens_per_second,
            "status": decode_status,
        },
    }

    if decode_status == "not_applicable":
        roofline: dict[str, Any] = {"status": "not_applicable"}
    else:
        bandwidth = numeric("effective_bandwidth_gbps", effective_bandwidth_gbps)
        weights = numeric("active_weight_bytes_per_token", active_weight_bytes_per_token)
        kv_read = numeric("kv_read_bytes_per_token", kv_read_bytes_per_token)
        kv_write = numeric("kv_write_bytes_per_token", kv_write_bytes_per_token)
        extra = numeric("extra_bytes_per_token", extra_bytes_per_token)
        if bandwidth_source is None:
            missing.append("bandwidth_source")
        elif not isinstance(bandwidth_source, str) or not bandwidth_source.strip():
            invalid.append("bandwidth_source: expected a non-empty string")
        if bandwidth_source_kind is None:
            missing.append("bandwidth_source_kind")
        elif not isinstance(bandwidth_source_kind, str):
            invalid.append("bandwidth_source_kind: expected a string")
        elif bandwidth_source_kind not in {"measured_effective", "nominal_peak"}:
            invalid.append("bandwidth_source_kind: unknown value")
        traffic_values = (bandwidth, weights, kv_read, kv_write, extra)
        if all(value is not None for value in traffic_values):
            traffic = {
                "weights": weights,
                "kv_read": kv_read,
                "kv_write": kv_write,
                "extra": extra,
            }
            try:
                total_bytes = math.fsum(traffic.values())
            except (OverflowError, ValueError):
                total_bytes = None
                invalid.append("bytes_per_token: sum is not finite")
            if total_bytes is not None and not math.isfinite(total_bytes):
                invalid.append("bytes_per_token: sum is not finite")
            if total_bytes is not None and total_bytes <= 0:
                invalid.append("bytes_per_token: denominator must be positive")
            if bandwidth is not None and bandwidth <= 0:
                invalid.append("effective_bandwidth_gbps: denominator must be positive")
        if invalid:
            roofline = {"status": "invalid", "invalid": list(invalid)}
        elif decode_status != "measured" or missing:
            roofline = {"status": "inconclusive", "missing": list(dict.fromkeys(missing))}
        elif bandwidth_source_kind == "nominal_peak":
            roofline = {
                "status": "inconclusive",
                "reason": "nominal_peak_not_valid_for_efficiency",
                "bandwidth_gbps": bandwidth,
                "bandwidth_source": {
                    "kind": bandwidth_source_kind,
                    "label": bandwidth_source.strip(),
                },
                "bytes_per_token": dict(traffic, total=total_bytes),
            }
        else:
            try:
                ideal_numerator = bandwidth * 1e9
            except OverflowError:
                ideal_numerator = float("inf")
            if not math.isfinite(ideal_numerator):
                invalid.append("ideal_tokens_per_second: numerator is not finite")
                ideal = None
            else:
                ideal = checked_divide("ideal_tokens_per_second", ideal_numerator, total_bytes)
            measured = phases["decode"]["tokens_per_second"]
            efficiency = (
                checked_divide("efficiency", measured, ideal)
                if measured is not None and ideal is not None else None
            )
            if invalid or ideal is None or efficiency is None:
                roofline = {"status": "invalid", "invalid": list(invalid)}
            else:
                roofline = {
                    "status": "ok",
                    "bandwidth_gbps": bandwidth,
                    "bandwidth_source": {
                        "kind": bandwidth_source_kind,
                        "label": bandwidth_source.strip(),
                    },
                    "bytes_per_token": dict(traffic, total=total_bytes),
                    "ideal_tokens_per_second": ideal,
                    "efficiency": efficiency,
                }
                if efficiency > 1:
                    roofline["warnings"] = ["efficiency_above_one_input_consistency"]

    status = "invalid" if invalid else (
        "ok" if prefill_status == "measured"
        and decode_status == "measured"
        and roofline["status"] == "ok"
        else "inconclusive"
    )
    return {
        "schema": "ironmule.phase_roofline.v1",
        "diagnostic_only": True,
        "status": status,
        "missing": list(dict.fromkeys(missing)),
        "phases": phases,
        "roofline": roofline,
    }


def run_protocol(rt: Any, ironmule: Any, *, requests: int, max_tokens: int,
                 plan_name: str = "strict", warmup: int = 2, repeats: int = 6
                 ) -> dict[str, Any]:
    """Run balanced warmups/repeats for both arms on one loaded model.

    Process/model isolation is intentionally not claimed: one loaded runtime is shared
    to avoid doubling peak memory. R3 therefore remains open for a future fresh-process
    protocol; plan/cache state is isolated per arm within this process.
    """
    _validate_protocol(warmup, repeats)
    plans = {arm: _new_plan(rt, ironmule, plan_name, arm) for arm in ARM_NAMES}
    modes = {
        "interactive": ironmule.InteractiveMode(),
        "throughput": ironmule.ThroughputMode(),
    }
    workload = _workload(rt, plan_name, requests, max_tokens, warmup, repeats)
    # The same two orders are used repeatedly so each position receives both arms.
    orders = [list(ARM_NAMES), list(reversed(ARM_NAMES))]
    raw: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARM_NAMES}
    measured_results: dict[int, dict[str, list[Any]]] = {}

    def execute(arm: str, phase: str, index: int, order: list[str]) -> list[Any]:
        result, snapshot = _run(
            rt, ironmule, modes[arm],
            _build_requests(rt, ironmule, plans[arm], plan_name, requests, max_tokens),
        )
        raw[arm].append({
            "phase": phase, "repeat": index, "order": list(order), "snapshot": snapshot,
            "outputs": [_output_record(
                item, tuple(getattr(getattr(rt, "backend", None), "eos_ids", ()))
            ) for item in result],
        })
        return result

    for index in range(warmup):
        order = orders[index % 2]
        for arm in order:
            execute(arm, "warmup", index, order)
    for index in range(repeats):
        order = orders[(warmup + index) % 2]
        measured_results[index] = {}
        for arm in order:
            measured_results[index][arm] = execute(arm, "measure", index, order)

    summaries: dict[str, dict[str, Any]] = {}
    for arm in ARM_NAMES:
        samples = [entry["snapshot"] for entry in raw[arm] if entry["phase"] == "measure"]
        summaries[arm] = {
            "outer_wall_ms": _sample_summary([s["outer_wall_ms"] for s in samples]),
            "physical_tokens_per_second": _sample_summary(
                [s["physical_tokens_per_second"] for s in samples]
            ),
            "visible_tokens_per_second": _sample_summary(
                [s["visible_tokens_per_second"] for s in samples]
            ),
            "executor_wall_ms": _sample_summary([s.get("wall_ms", 0.0) for s in samples]),
        }

    differences: list[dict[str, Any]] = []
    for index, pair in measured_results.items():
        for difference in output_diff(pair["interactive"], pair["throughput"]):
            differences.append(dict(difference, repeat=index))
    baseline_wall = [
        entry["snapshot"]["outer_wall_ms"]
        for entry in raw["interactive"] if entry["phase"] == "measure"
    ]
    candidate_wall = [
        entry["snapshot"]["outer_wall_ms"]
        for entry in raw["throughput"] if entry["phase"] == "measure"
    ]
    baseline_rate = [
        entry["snapshot"]["physical_tokens_per_second"]
        for entry in raw["interactive"] if entry["phase"] == "measure"
    ]
    candidate_rate = [
        entry["snapshot"]["physical_tokens_per_second"]
        for entry in raw["throughput"] if entry["phase"] == "measure"
    ]
    wall_ratio = _paired_ratio(candidate_wall, baseline_wall)
    rate_ratio = _paired_ratio(candidate_rate, baseline_rate)
    return {
        "protocol": {
            "warmup": warmup, "repeats": repeats, "orders": orders,
            "primary_metric": "outer_wall_ms",
            "primary_rate_metric": "physical_tokens_per_second",
            "token_metric_definition": {
                "physical": "Result.tokens, including the prefill choice and EOS when emitted",
                "visible": "physical tokens excluding EOS, as reported by telemetry",
            },
            "available_diagnostic_metrics": [
                "wall_ms", "outer_wall_ms", "service_ttft_p50_ms", "service_ttft_p95_ms",
                "engine_ttft_p50_ms", "engine_ttft_p95_ms", "latency_p50_ms",
                "latency_p95_ms", "queue_wait_p95_ms", "inter_token_p50_ms",
                "inter_token_p95_ms", "mean_realised_width", "max_realised_width",
                "fallbacks", "correctness_errors", "peak_memory_bytes",
            ],
            "planned_diagnostic_metrics": [
                "prefill_ns", "decode_ns", "per_request_raw_phase_timings",
            ],
        },
        "arms": {
            arm: {
                "summary": summaries[arm], "raw": raw[arm],
                "runtime_fingerprint": _runtime_fingerprint(rt, plans[arm], workload, modes[arm]),
                "workload": workload,
            }
            for arm in ARM_NAMES
        },
        "comparison": {
            "primary_wall_ratio_throughput_over_interactive": wall_ratio,
            "primary_rate_ratio_throughput_over_interactive": rate_ratio,
            "throughput_gain": 1.0 - wall_ratio["median_ratio"],
            "token_identity": not differences,
            "token_differences": differences,
        },
        "limitations": {
            "shared_loaded_runtime": True, "shared_loaded_model": True,
            "fresh_plan_per_arm": True, "fresh_process_per_arm": False,
            "r3_status": "open_shared_process_protocol",
        },
        "plan": plan_name, "requests": requests, "max_tokens": max_tokens,
    }


def _print_report(result: dict[str, Any], model_id: str) -> None:
    print(f"model {model_id}")
    protocol = result["protocol"]
    print(
        f"plan {result['plan']}   requests {result['requests']}   "
        f"warmup {protocol['warmup']}   repeats {protocol['repeats']}\n"
    )
    row = "{:<14}{:>14}{:>15}{:>15}{:>15}{:>15}"
    print(row.format("mode", "outer ms p50", "physical tok/s", "visible tok/s", "exec ms p50", "outer spread"))
    for arm in ARM_NAMES:
        summary = result["arms"][arm]["summary"]
        wall = summary["outer_wall_ms"]
        print(row.format(
            arm, f"{wall['median']:.1f}",
            f"{summary['physical_tokens_per_second']['median']:.1f}",
            f"{summary['visible_tokens_per_second']['median']:.1f}",
            f"{summary['executor_wall_ms']['median']:.1f}",
            f"[{wall['min']:.1f}; {wall['max']:.1f}]",
        ))
    comparison = result["comparison"]
    wall_ratio = comparison["primary_wall_ratio_throughput_over_interactive"]
    print(
        f"\nthroughput gain {comparison['throughput_gain'] * 100:+.2f}%   "
        f"outer-wall ratio {wall_ratio['median_ratio']:.4f} "
        f"95% CI [{wall_ratio['ci_low']:.4f}; {wall_ratio['ci_high']:.4f}]"
    )
    print(f"identical answers in both modes: {comparison['token_identity']}")
    if comparison["token_differences"]:
        print(
            "structured token differences: "
            + json.dumps(comparison["token_differences"], sort_keys=True),
            file=sys.stderr,
        )


def main(argv: list[str] | None = None) -> int:
    import ironmule

    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--plan", choices=["strict", "reusable"], default="strict")
    parser.add_argument("--model", default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.requests < 1 or args.max_tokens < 1:
        parser.error("--requests and --max-tokens must be positive")

    rt = ironmule.Runtime.load(model_id=args.model)
    result = run_protocol(
        rt, ironmule, requests=args.requests, max_tokens=args.max_tokens,
        plan_name=args.plan, warmup=args.warmup, repeats=args.repeats,
    )
    result["model"] = getattr(rt, "model_id", args.model)
    _print_report(result, result["model"])
    if args.json:
        args.json.write_text(json.dumps(result, indent=1, sort_keys=True, default=str))
        print(f"wrote {args.json}")
    return 0 if result["comparison"]["token_identity"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
