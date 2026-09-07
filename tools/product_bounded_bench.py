"""Registered PROD2 bounded-prefetch pilot harness.

The harness is intentionally inert unless invoked with ``--execute``.  It is
an opt-in real-hardware study: no CPU substitute, fake model, or fabricated
completion is used.  See ``docs/PROD2_BOUNDED_PREFETCH_2026-09-07.md`` for the
registered design and interpretation limits.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import http.client
import importlib.metadata
import json
import os
import math
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _bench import harness_preconditions, resolve_local_model_snapshot  # noqa: E402


LIMITS = (1, 8, 32)
WORKER_ORDERS = ((1, 8, 32), (8, 32, 1), (32, 1, 8))
FULL_WORKERS = 3
CONTEXT_LIMIT = 8192
PROMPT = "Write a short sentence about apples."
SCHEMA = "ironmule.product_bounded_prefetch_pilot.v1"
PREREGISTRATION = ROOT / "docs" / "PROD2_BOUNDED_PREFETCH_2026-09-07.md"
MAX_CPU_LOAD_RATIO = 0.8
ALLOWED_THERMAL_STATES = (0, 1)


def _manifest() -> dict[str, str]:
    paths = [
        *sorted((ROOT / "ironmule_product").glob("*.py")),
        *sorted((ROOT / "friday_evidence").glob("*.py")),
        *sorted((ROOT / "ironmule").glob("*.py")),
        ROOT / "tools" / "_bench.py",
        Path(__file__).resolve(),
        PREREGISTRATION,
        ROOT / "pyproject.toml",
        ROOT / "BACKLOG.md",
    ]
    result: dict[str, str] = {}
    for path in paths:
        result[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in ("mlx", "mlx-lm", "numpy")}


def _process_info_public_api() -> dict[str, int | bool]:
    """Read public Low Power and thermal state flags from Foundation."""
    probe = subprocess.run(
        ["/usr/bin/swift", "-e", "import Foundation; let p=ProcessInfo.processInfo; print(p.isLowPowerModeEnabled ? \"1\" : \"0\", p.thermalState.rawValue)"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    fields = probe.stdout.split()
    if probe.returncode != 0 or len(fields) != 2 or fields[0] not in {"0", "1"} or fields[1] not in {"0", "1", "2", "3"}:
        raise RuntimeError("could not read ProcessInfo power/thermal state")
    return {"low_power_mode": fields[0] == "1", "thermal_state": int(fields[1])}


def _cell_checkpoint(environment: Any) -> dict[str, Any]:
    state = _process_info_public_api()
    snapshot = dict(environment())
    logical = os.cpu_count()
    load = snapshot.get("loadavg", [None])[0]
    load_ratio = float(load) / logical if isinstance(load, (int, float)) and logical else None
    snapshot["low_power_mode_public_api"] = state["low_power_mode"]
    snapshot["thermal_state_public_api"] = state["thermal_state"]
    snapshot["load_ratio"] = load_ratio
    snapshot["observed_unix_ns"] = time.time_ns()
    reasons = []
    if snapshot.get("power_source") != "AC":
        reasons.append("AC power is required")
    if state["low_power_mode"] or state["thermal_state"] not in ALLOWED_THERMAL_STATES:
        reasons.append("power/thermal state failed the registered cell gate")
    if load_ratio is None or not math.isfinite(load_ratio) or load_ratio > MAX_CPU_LOAD_RATIO:
        reasons.append("CPU load exceeded the registered cell gate")
    snapshot["gate_reasons"] = reasons
    snapshot["eligible"] = not reasons
    return snapshot


def _request_json(port: int, payload: dict[str, Any]) -> dict[str, Any]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=125)
    try:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        connection.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
        response = connection.getresponse()
        raw = response.read()
        if response.status != 200:
            raise RuntimeError(f"HTTP generation returned {response.status}: {raw[:256]!r}")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise RuntimeError("HTTP completion was not an object")
        return value
    finally:
        connection.close()


class _RecordingLease:
    """One fixed-variant forwarding lease over one shared worker."""

    def __init__(self, client: Any, variant: str):
        self.client = client
        self.variant = variant
        self.trace_forwards = False
        self.records: list[dict[str, Any]] = []

    @property
    def ready(self) -> bool:
        return bool(self.client.ready)

    def stream(self, request: Any, cancel: Any = None, timeout: float = 120.0) -> Iterator[dict[str, Any]]:
        record: dict[str, Any] = {
            "request_id": request.request_id,
            "variant": self.variant,
            "trace_forwards": self.trace_forwards,
            "events": [],
        }
        self.records.append(record)
        stream = self.client.stream(
            request,
            cancel=cancel,
            timeout=timeout,
            variant=self.variant,
            trace_forwards=self.trace_forwards,
        )
        try:
            for event in stream:
                record["events"].append(dict(event))
                yield event
        finally:
            stream.close()

    def close(self) -> None:
        # The shared MLXWorkerClient owns the process and is closed once after
        # both ProductService instances have been stopped.
        return None


@contextmanager
def _servers(spec: Any, client: Any, state_root: Path):
    from ironmule_product.http_server import create_server
    from ironmule_product.service import ProductService
    from ironmule_product.state import ProductStore

    leases = {variant: _RecordingLease(client, variant) for variant in ("reference", "bounded_prefetch")}
    services = []
    servers = []
    threads = []
    try:
        for variant in ("reference", "bounded_prefetch"):
            store = ProductStore(state_root / variant)
            store.setup("server")
            store.register_model(spec)
            service = ProductService(store, backend=leases[variant], spec=spec)
            server = create_server(service, host="127.0.0.1", port=0)
            thread = threading.Thread(target=server.serve_forever, name=f"prod2-{variant}", daemon=True)
            thread.start()
            services.append(service)
            servers.append(server)
            threads.append(thread)
        yield leases, {variant: server.server_address[1] for variant, server in zip(("reference", "bounded_prefetch"), servers)}
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=5)
        for service in services:
            service.close()


def _underlying_output(record: dict[str, Any]) -> dict[str, Any]:
    events = record.get("events")
    if not isinstance(events, list) or not events or not isinstance(events[-1], dict):
        raise RuntimeError("forwarding lease captured no completion")
    tokens = [event for event in events if event.get("type") == "token"]
    done = events[-1]
    if done.get("type") != "done":
        raise RuntimeError("forwarding lease did not capture a terminal done frame")
    metrics = done.get("metrics", {})
    peak = metrics.get("peak_memory") if isinstance(metrics, dict) else None
    peak = float(peak) if isinstance(peak, (int, float)) and not isinstance(peak, bool) and math.isfinite(float(peak)) else None
    return {
        "tokens": [event.get("token_id") for event in tokens],
        "text": "".join(event.get("text", "") for event in tokens),
        "finish_reason": done.get("finish_reason"),
        "prompt_tokens": done.get("prompt_tokens"),
        "completion_tokens": done.get("completion_tokens"),
        "peak_memory_gb": peak,
        "model_forward_invocations": done.get("model_forward_invocations"),
    }


def _compare_output(actual: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    keys = ("tokens", "text", "finish_reason", "prompt_tokens", "completion_tokens")
    for key in keys:
        if actual[key] != expected[key]:
            raise RuntimeError(f"{label}: {key} differs from reference")


def _pace_for_duty_reserve(guard: Any) -> None:
    """Reserve one full 6-second block before starting another HTTP call."""
    reserve = 6.0
    limit = guard.policy.duty_window_s * guard.policy.duty_cycle_limit
    while True:
        now = guard._clock()  # shared BudgetGuard clock; no policy copy
        cutoff = now - guard.policy.duty_window_s
        current = sum(max(0.0, end - max(start, cutoff)) for start, end in guard._gpu_events)
        if current + reserve <= limit:
            return
        guard.required_break()


def _call(
    *,
    leases: dict[str, _RecordingLease],
    ports: dict[str, int],
    variant: str,
    limit: int,
    phase: str,
    trace: bool,
    guard: Any,
    memory_gate: Any,
    block_index: int,
    persist_sample: Any,
) -> dict[str, Any]:
    lease = leases[variant]
    before = len(lease.records)
    lease.trace_forwards = trace
    payload = {"model": "", "messages": [{"role": "user", "content": PROMPT}], "max_tokens": limit}
    payload["model"] = lease.client.spec.model_id
    _pace_for_duty_reserve(guard)
    started = time.monotonic()
    completion = _request_json(ports[variant], payload)
    wall = time.monotonic() - started
    if len(lease.records) != before + 1:
        raise RuntimeError("forwarding lease record count changed unexpectedly")
    lease_record = lease.records[-1]
    actual = _underlying_output(lease_record)
    choice = completion.get("choices", [{}])[0] if isinstance(completion.get("choices"), list) and completion.get("choices") else {}
    message = choice.get("message", {}) if isinstance(choice, dict) else {}
    usage = completion.get("usage", {})
    prompt_count = actual["prompt_tokens"]
    completion_count = actual["completion_tokens"]
    http_total = prompt_count + completion_count if isinstance(prompt_count, int) and isinstance(completion_count, int) else None
    http_matches_underlying = (isinstance(message, dict) and message.get("content") == actual["text"]
                               and choice.get("finish_reason") == actual["finish_reason"]
                               and isinstance(usage, dict)
                               and usage.get("prompt_tokens") == actual["prompt_tokens"]
                               and usage.get("completion_tokens") == actual["completion_tokens"]
                               and usage.get("total_tokens") == http_total)
    peak_bytes = int(actual["peak_memory_gb"] * 1_000_000_000) if actual["peak_memory_gb"] is not None else None
    result = {
        "status": "measured",
        "variant": variant,
        "limit": limit,
        "phase": phase,
        "trace_forwards": trace,
        "http_wall_seconds": wall,
        "http_completion": completion,
        "underlying": actual,
        "request_id": lease_record["request_id"],
    }
    persist_sample(result)
    try:
        if not http_matches_underlying:
            raise RuntimeError(f"{variant} limit {limit}: HTTP completion differs from underlying done")
        if peak_bytes is None:
            raise RuntimeError("done frame lacks actual peak_memory metric")
        guard.record_gpu(wall)
        reason = memory_gate.check(block_index, peak_bytes)
        if memory_gate.record["blocks"][-1]["swap_bytes"] is None:
            raise RuntimeError("swap telemetry disappeared during pilot")
        if reason:
            raise RuntimeError(f"memory gate rejected block {block_index}: {reason}")
        guard.required_break()
        result["status"] = "passed"
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        persist_sample(result, update=True)
        raise
    persist_sample(result, update=True)
    return result


def _run_worker(
    spec: Any,
    worker_index: int,
    order: tuple[int, ...],
    guard: Any,
    memory_gate: Any,
    *,
    audit_only: bool,
    worker_result: dict[str, Any] | None = None,
    progress: Any = None,
    checkpoint: Any = None,
) -> dict[str, Any]:
    from ironmule_product.backend import MLXWorkerClient
    from ironmule.bench import paired_ratio, swap_used_bytes

    if worker_result is None:
        worker_result = {"worker_index": worker_index, "limit_order": list(order), "status": "started", "limits": {}}
    client = MLXWorkerClient(spec)
    try:
        started = time.monotonic()
        client.start()
        worker_result["load_wall_seconds"] = time.monotonic() - started
        reason = memory_gate.check(sum(len(item.get("calls", [])) for item in worker_result["limits"].values()), None)
        if reason or memory_gate.record["blocks"][-1]["swap_bytes"] is None:
            raise RuntimeError(f"memory gate rejected after worker load: {reason}")
        guard.required_break()
        with tempfile.TemporaryDirectory(prefix=f"ironmule-prod2-{worker_index}-") as state_dir:
            with _servers(spec, client, Path(state_dir)) as (leases, ports):
                for limit in order:
                    cell: dict[str, Any] = {"calls": []}
                    worker_result["limits"][str(limit)] = cell
                    cell["environment_before"] = checkpoint() if checkpoint is not None else None
                    if cell["environment_before"] is not None and cell["environment_before"].get("power_source") != "AC":
                        raise RuntimeError("AC power was lost at cell start")

                    def persist_sample(sample: dict[str, Any], update: bool = False) -> None:
                        if not update:
                            cell["calls"].append(sample)
                        if progress is not None:
                            progress(worker_result)

                    def run_call(
                        variant: str,
                        phase: str,
                        trace: bool,
                        expected: dict[str, Any] | None = None,
                        label: str = "",
                    ) -> dict[str, Any]:
                        before_calls = len(cell["calls"])
                        try:
                            result = _call(leases=leases, ports=ports, variant=variant, limit=limit,
                                           phase=phase, trace=trace, guard=guard, memory_gate=memory_gate,
                                           block_index=sum(len(item["calls"]) for item in worker_result["limits"].values()),
                                           persist_sample=persist_sample)
                        except Exception as exc:
                            if len(cell["calls"]) == before_calls:
                                cell["calls"].append({"status": "failed", "variant": variant, "limit": limit,
                                                      "phase": phase, "trace_forwards": trace,
                                                      "error_type": type(exc).__name__, "error": str(exc)})
                                if progress is not None:
                                    progress(worker_result)
                            raise
                        try:
                            if expected is not None:
                                _compare_output(result["underlying"], expected, label)
                            return result
                        except Exception as exc:
                            result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
                            if progress is not None:
                                progress(worker_result)
                            raise

                    reference = run_call("reference", "warmup", True)
                    expected = reference["underlying"]
                    candidate = run_call("bounded_prefetch", "warmup", True, expected,
                                         f"worker {worker_index} limit {limit} candidate warmup")
                    if candidate["underlying"]["model_forward_invocations"] is None:
                        raise RuntimeError("traced candidate omitted model_forward_invocations")
                    reference_forwards = expected["model_forward_invocations"]
                    candidate_forwards = candidate["underlying"]["model_forward_invocations"]
                    expected_delta = 1 if expected["completion_tokens"] == limit else 0
                    if not isinstance(reference_forwards, int) or not isinstance(candidate_forwards, int):
                        raise RuntimeError("traced reference/candidate forward count is not an integer")
                    if reference_forwards - candidate_forwards != expected_delta:
                        raise RuntimeError(f"worker {worker_index} limit {limit}: forward delta mismatch")
                    if audit_only:
                        cell["environment_after"] = checkpoint() if checkpoint is not None else None
                        if cell["environment_after"] is not None and cell["environment_after"].get("power_source") != "AC":
                            raise RuntimeError("AC power was lost at cell end")
                        continue

                    aa_first = run_call("reference", "aa", False, expected,
                                        f"worker {worker_index} limit {limit} AA-1")
                    aa_second = run_call("reference", "aa", False, expected,
                                         f"worker {worker_index} limit {limit} AA-2")
                    cell["aa_wall_seconds"] = [aa_first["http_wall_seconds"], aa_second["http_wall_seconds"]]
                    cell["aa_summary"] = summarise(cell["aa_wall_seconds"])
                    cell["aa_pair_ratio"] = paired_ratio(
                        [aa_second["http_wall_seconds"]],
                        [aa_first["http_wall_seconds"]],
                        resamples=2000,
                    )
                    pairs: list[dict[str, Any]] = []
                    for pair_name, pair_order in (("ab", ("reference", "bounded_prefetch")), ("ba", ("bounded_prefetch", "reference"))):
                        pair: dict[str, Any] = {"name": pair_name, "calls": []}
                        for variant in pair_order:
                            call = run_call(variant, pair_name, False, expected,
                                            f"worker {worker_index} limit {limit} {pair_name}")
                            pair["calls"].append(call)
                        pairs.append(pair)
                    cell["pairs"] = pairs
                    if len(cell["calls"]) != 8:
                        raise RuntimeError("pilot cell did not execute exactly eight calls")
                    cell["environment_after"] = checkpoint() if checkpoint is not None else None
                    if cell["environment_after"] is not None and cell["environment_after"].get("power_source") != "AC":
                        raise RuntimeError("AC power was lost at cell end")
                expected_calls = 6 if audit_only else 24
                worker_result["real_call_count"] = sum(len(cell["calls"]) for cell in worker_result["limits"].values())
                if worker_result["real_call_count"] != expected_calls:
                    raise RuntimeError("worker call count does not match registered schedule")
                worker_result["memory_gate"] = memory_gate.record
            worker_result["memory_after_swap_bytes"] = swap_used_bytes()
            client.close()
            reason = memory_gate.check(sum(len(item.get("calls", [])) for item in worker_result["limits"].values()) + 1, None)
            if reason or memory_gate.record["blocks"][-1]["swap_bytes"] is None:
                raise RuntimeError(f"memory gate rejected after worker cleanup: {reason}")
        worker_result["status"] = "passed"
        if progress is not None:
            progress(worker_result)
        return worker_result
    finally:
        client.close()


def run(model_id: str, output: Path, *, audit_only: bool) -> int:
    if output.exists():
        raise ValueError("report exists; choose a new output path")
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "running",
        "model_id": model_id,
        "audit_only": audit_only,
        "performance_claim": False,
        "activation_allowed": False,
        "limits": list(LIMITS),
        "worker_orders": [list(order) for order in WORKER_ORDERS],
        "expected_real_calls": 6 if audit_only else FULL_WORKERS * len(LIMITS) * 8,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "source_manifest_before": _manifest(),
        "started_unix_ns": time.time_ns(),
        "workers": [],
    }

    def persist() -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    # Persist before model resolution, power checks, or any hardware probe so
    # failed attempts remain evidence rather than disappearing at preflight.
    persist()
    guard = None
    environment = None
    memory_gate = None
    try:
        if "gemma" not in model_id.lower():
            raise ValueError("PROD2 pilot accepts an exact cached Gemma model ID only")
        snapshot = resolve_local_model_snapshot(model_id)
        from ironmule_product.types import ModelSpec

        spec = ModelSpec(model_id, snapshot.revision, str(snapshot.path), snapshot.weight_bytes)
        report["model"] = snapshot.report_identity()
        # The shared resolver also supports the user's ordinary HF cache.
        # Its historical label must not imply that every snapshot is project-local.
        report["model"]["model_source"] = "validated_local_snapshot"
        persist()
        guard = harness_preconditions()
        from ironmule.bench import MemoryGate, environment as bench_environment, paired_ratio, summarise

        environment = bench_environment
        import mlx.core as mx
        report["hardware"] = dict(mx.device_info())
        stock_file = importlib.metadata.distribution("mlx-lm").locate_file("mlx_lm/generate.py")
        report["stock_generate_sha256"] = hashlib.sha256(stock_file.read_bytes()).hexdigest()
        report["budget"] = guard.summary()
        report["environment_before"] = environment()
        persist()
        process_state = _process_info_public_api()
        report["process_info_before"] = process_state
        if report["environment_before"].get("power_source") != "AC" or process_state["low_power_mode"] or process_state["thermal_state"] not in ALLOWED_THERMAL_STATES:
            raise RuntimeError("pilot requires reported AC power and ProcessInfo low power mode 0")
        report["versions"] = _versions()
        report["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        memory_gate = MemoryGate()
        if memory_gate.inert or memory_gate.baseline_swap is None:
            raise RuntimeError("pilot requires an active swap/memory gate with a readable swap baseline")
        report["memory_gate"] = memory_gate.record
        persist()

        def checkpoint() -> dict[str, Any]:
            # Keep the actual refused observation as well as its reason. A
            # failed readiness sample is evidence, not a missing context.
            sample = _cell_checkpoint(environment)
            report.setdefault("readiness_samples", []).append(sample)
            persist()
            if not sample["eligible"]:
                raise RuntimeError("; ".join(sample["gate_reasons"]))
            return sample

        orders = WORKER_ORDERS[:1] if audit_only else WORKER_ORDERS
        for index, order in enumerate(orders):
            guard.before_candidate()
            checkpoint()  # Refuse before loading weights, not after load.
            worker_result = {"worker_index": index, "limit_order": list(order), "status": "started", "limits": {}}
            report["workers"].append(worker_result)
            persist()
            try:
                _run_worker(spec, index, order, guard, memory_gate, audit_only=audit_only,
                            worker_result=worker_result, progress=lambda _: persist(),
                            checkpoint=checkpoint)
            except Exception as exc:
                worker_result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
                persist()
                raise
            finally:
                guard.finish_candidate()
                report["budget"] = guard.summary()
                persist()
        report["environment_after"] = environment()
        report["source_manifest_after"] = _manifest()
        if report["source_manifest_after"] != report["source_manifest_before"]:
            raise RuntimeError("source manifest changed during pilot")
        if not audit_only:
            cluster_ratios: dict[str, Any] = {}
            for limit in LIMITS:
                worker_medians: list[float] = []
                aa_relative: list[float] = []
                for worker in report["workers"]:
                    cell = worker["limits"][str(limit)]
                    pairs = cell["pairs"]
                    baseline = []
                    candidate = []
                    for pair in pairs:
                        by_variant = {call["variant"]: call for call in pair["calls"]}
                        baseline.append(by_variant["reference"]["http_wall_seconds"])
                        candidate.append(by_variant["bounded_prefetch"]["http_wall_seconds"])
                    worker_ratio = paired_ratio(candidate, baseline, resamples=2000)
                    cell["paired_ratio_within_worker"] = worker_ratio
                    worker_medians.append(worker_ratio["median_ratio"])
                    aa_ratio = cell["aa_pair_ratio"]
                    aa_relative.append(abs(aa_ratio["median_ratio"] - 1.0))
                cluster = paired_ratio(worker_medians, [1.0] * len(worker_medians), resamples=4000)
                noise_threshold = max(0.02, 3.0 * max(aa_relative))
                cluster_ratios[str(limit)] = cluster
                cluster_ratios[str(limit)]["aa_max_abs_relative"] = max(aa_relative)
                cluster_ratios[str(limit)]["noise_threshold"] = noise_threshold
                cluster_ratios[str(limit)]["classification"] = (
                    "inconclusive_noise_overlap"
                    if cluster["ci_low"] <= 1.0 <= cluster["ci_high"]
                    or abs(cluster["median_ratio"] - 1.0) <= noise_threshold
                    else "pilot_signal_only"
                )
            report["cluster_ratio"] = cluster_ratios
            report["cluster_ratio_method"] = "three-worker cluster medians per limit; no six-independent-worker claim"
        report["budget"] = guard.summary()
        report["status"] = "passed"
        return 0
    except BaseException as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        if guard is not None:
            report["budget"] = guard.summary()
        raise
    finally:
        report.setdefault("environment_after", None)
        report.setdefault("source_manifest_after", _manifest())
        persist()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--model", required=True, help="exact cached Gemma model ID")
    parser.add_argument("--output", type=Path, required=True, help="new incremental JSON report path")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"state": "not_released", "hint": "pass --execute"}))
        return 78
    return run(args.model, args.output, audit_only=args.audit_only)


if __name__ == "__main__":
    raise SystemExit(main())
