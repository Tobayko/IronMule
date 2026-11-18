"""Read-only monitor for local inference and benchmark progress."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

STATUSES = {"preparing", "running", "complete", "error", "timeout"}
STAGES = {"initializing", "preflight", "starting", "warmup", "benchmark", "quality", "learning", "evaluating", "finished"}
ERRORS = {"launch_failed", "identity_unverified", "measurement_failed", "learning_failed", "budget_exhausted",
          "invalid_input"}
DEPLOYMENT_SCHEMA = "ironmule.deployment-comparison.v1"
DEPLOYMENTS = {"ironmule_gguf", "vllm_awq"}
DEPLOYMENT_STAGES = STAGES | {"hashing", "verifying_prompts", "verified"}
VERDICTS = {"pending", "incomplete", "does_not_qualify", "qualifies"}
ENGINES = {"ironmule", "vllm", "llama_cpp"}
REASONS = {"insufficient_repeats_or_loads", "missing_arm", "throughput_margin", "first_token_regressed",
           "request_rate_regressed", "quality_regressed", "uninformative_quality_reference",
           "served_prompt_lengths_differ", "failed_arm", "prompt_ids_unmatched", "unverified_device_placement",
           "stock_provenance_unverified", "generated_lengths_or_stops_differ", *ERRORS}
LEARNING = {"deferred", "qualified", "retained_incumbent", "error"}
PROVIDER_EVENTS = {
    "fetch_awq": "Downloading Qwen3-32B AWQ", "fetch_gguf": "Downloading Qwen3-32B GGUF",
    "tokenizer": "Preparing the shared tokenizer", "production_compare": "Comparing GGUF and AWQ deployments",
    "error": "Remote workload failed", "cleanup_verified": "Confirmed instance shutdown",
    "local_preflight": "Checking local prerequisites", "preflight": "Checking runtime",
    "create_requested": "Requested a GPU", "pod_booting": "Starting GPU instance",
    "container_boot": "Starting container", "awaiting_bundle": "Waiting for benchmark files",
    "verify_bundle": "Verifying benchmark files", "bootstrap_runtime": "Preparing runtime",
    "fetch_qwen3_06b": "Downloading Qwen3-0.6B", "tokenizer_qwen3_06b": "Preparing Qwen3-0.6B tokenizer",
    "compare_qwen3_06b": "Comparing Qwen3-0.6B", "fetch_qwen3_32b": "Downloading Qwen3-32B",
    "tokenizer_qwen3_32b": "Preparing Qwen3-32B tokenizer", "compare_qwen3_32b": "Comparing Qwen3-32B",
    "readiness_qwen3_06b": "Checking Qwen3-0.6B server readiness",
    "readiness_qwen3_32b": "Checking Qwen3-32B server readiness",
    "readiness_saved_qwen3_06b": "Saved Qwen3-0.6B readiness results",
    "readiness_saved_qwen3_32b": "Saved Qwen3-32B readiness results",
    "bundle_timeout": "Benchmark file transfer timed out", "finished": "Finished remote workload",
    "cleanup_requested": "Requested instance shutdown", "operator_deadline": "Reached the time limit",
    "operator_interrupted": "Remote run interrupted", "operator_error": "Remote setup failed",
}
LABEL = re.compile(r"[\w ./+():@-]{1,180}\Z")
CONFIG = re.compile(r"w\d+(?:-(?:fa|uni|f32|kv8))*\Z")


def provider_event(value):
    if isinstance(value, str):
        return PROVIDER_EVENTS.get(value.replace("-", "_"))
    return None


def number(value):
    return value if type(value) in (float, int) and math.isfinite(value) and value >= 0 else None


def label(value):
    if not isinstance(value, str):
        return None
    if value.startswith(("/", "~", "\\")) or "\\" in value:
        value = value.replace("\\", "/").rsplit("/", 1)[-1]
    return value if LABEL.fullmatch(value) else None


def timestamp(value):
    if not isinstance(value, str) or len(value) > 45:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.isoformat() if parsed.tzinfo is not None else None


def progress_path(service=None, explicit=None):
    given = explicit or os.environ.get("IRONMULE_BENCH_PROGRESS")
    if given:
        return Path(given).expanduser()
    root = getattr(getattr(service, "store", None), "root", None)
    if root is None:
        from .state import ProductStore

        root = ProductStore().root
    return Path(root) / "benchmark-progress.json"


def _deployment_pairs(pairs):
    """Descriptive ratios are emitted only for pairs verified by the deployment runner."""
    if not isinstance(pairs, list):
        raise TypeError("invalid paired workloads")
    result = []
    for item in pairs[-80:]:
        if not isinstance(item, dict):
            raise TypeError("invalid pair")
        row = {k: number(item.get(k)) for k in ("repeat", "concurrency", "tokens_per_second_ratio",
                                                "requests_per_second_ratio", "ttft_p95_ratio")}
        quality = item.get("quality_correct") or {}
        total = item.get("quality_total")
        counts = [quality.get(k) for k in ("ironmule_gguf", "vllm_awq")] if isinstance(quality, dict) else []
        if (len(counts) != 2 or type(total) is not int or total <= 0 or
                any(type(x) is not int or not 0 <= x <= total for x in counts)):
            continue
        match = item.get("output_tokens_and_stops_match")
        row.update(gguf_correct=counts[0], awq_correct=counts[1], quality_total=total,
                   equal_work=match if type(match) is bool else None)
        result.append(row)
    return result


def benchmark(path, expected_schema=None):
    try:
        with path.open("rb") as f:
            raw = f.read((4 << 20) + 1)
        if len(raw) > 4 << 20:
            raise ValueError("oversized")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise TypeError("invalid report")
        schema = data.get("schema")
        if schema not in ("ironmule.compare.v1", DEPLOYMENT_SCHEMA) or (expected_schema and expected_schema != schema):
            raise ValueError("invalid schema")
        deployment = schema == DEPLOYMENT_SCHEMA
        engine_key, engines = ("deployment", DEPLOYMENTS) if deployment else ("engine", ENGINES)
        stages = DEPLOYMENT_STAGES if deployment else STAGES
        status = data.get("status")
        if status not in ({"preparing", "running", "measured", "incomplete"} if deployment else STATUSES):
            raise ValueError("invalid status")
        current = data.get("current") or {}
        runs = data.get("measurements" if deployment else "runs", [])
        if not isinstance(current, dict) or not isinstance(runs, list):
            raise TypeError("invalid rows")
        rows = []
        for item in runs[-80:]:
            if not isinstance(item, dict):
                raise TypeError("invalid row")
            quality = item.get("quality")
            row = {k: number(item.get(k)) for k in ("repeat", "concurrency", "requests", "completion_tokens",
                                                     "tokens_per_second", "requests_per_second", "ttft_median_ms", "ttft_p95_ms")}
            row.update(engine=item.get(engine_key) if item.get(engine_key) in engines else None, quality=None)
            if isinstance(quality, dict):
                passed, total = (quality.get(k) for k in ("passed", "total"))
                if type(passed) is int and type(total) is int and 0 <= passed <= total and total > 0:
                    row["quality"] = {"passed": passed, "total": total}
            rows.append(row)
        operator = data.get("operator") or {}
        provider = {}
        if isinstance(operator, dict) and operator.get("provider") == "Runpod":
            provider = {"name": "Runpod", "phase": provider_event(current.get("phase")),
                        "event": provider_event(operator.get("last_remote_event")),
                        "updated_at": timestamp(operator.get("last_remote_event_at"))}
        verdict = data.get("verdict") or {}
        reasons = verdict.get("reasons") if isinstance(verdict, dict) else None
        failures = data.get("failures") or []
        failure = failures[-1] if isinstance(failures, list) and failures and isinstance(failures[-1], dict) else {}
        learning = data.get("learning") or {}
        device = label(data.get("device"))
        for arm in reversed(data.get("deployments" if deployment else "arms") or []):
            identity = arm.get("identity") if isinstance(arm, dict) else None
            if isinstance(identity, dict) and identity.get("device") == device and label(identity.get("device_name")):
                device = label(identity["device_name"])
                break
        common = {"status": status, "stage": data.get("stage") if data.get("stage") in stages else None,
                "model": "Qwen3-32B" if deployment else label(data.get("model")), "device": device,
                "updated_at": timestamp(data.get("updated_at")),
                "engine": current.get(engine_key) if current.get(engine_key) in engines else None,
                "repeat": number(current.get("repeat")), "concurrency": number(current.get("concurrency")),
                "runs": rows, "run_count": len(runs), "provider": provider}
        if deployment:
            failure = data.get("failure") or {}
            failed_stage = failure.get("stage") if isinstance(failure, dict) else None
            return {**common, "kind": "deployment", "pairs": _deployment_pairs(data.get("pairs", [])),
                    "failed_stage": failed_stage if failed_stage in DEPLOYMENT_STAGES else None}
        return {**common, "verdict": verdict.get("status") if isinstance(verdict, dict) and verdict.get("status") in VERDICTS else None,
                "reasons": [r for r in reasons if isinstance(r, str) and r in REASONS] if isinstance(reasons, list) else [],
                "failed_engine": failure.get("engine") if failure.get("engine") in ENGINES else None,
                "failed_stage": failure.get("stage") if failure.get("stage") in STAGES else None,
                "learning": learning.get("status") if isinstance(learning, dict) and learning.get("status") in LEARNING else None,
                "error_code": data.get("error_code") if data.get("error_code") in ERRORS else None}
    except FileNotFoundError:
        return {"status": "idle", "runs": [], "run_count": 0}
    except (OSError, TypeError, OverflowError, UnicodeError, ValueError, RecursionError):
        return {"status": "unavailable", "runs": [], "run_count": 0}


def runtime(service):
    if service is None:
        return {"status": "detached", "samples": []}
    health = service.health()
    active, queued = [number(health.get(k)) for k in ("active_requests", "queued_requests")]
    ready = health.get("ready") is True
    state = ("running" if active or queued else "idle") if ready else "not_ready"
    agent = (health.get("ggml") or {}).get("agent") or {}
    decisions = []
    for entry in (agent.get("last_decisions") or [])[-5:]:
        if entry.get("action") in ("switch", "stay", "revert", "watch", "keep"):
            dest = str(entry.get("to", ""))
            decisions.append({"at": number(entry.get("at")), "action": entry["action"],
                              "to": dest if CONFIG.fullmatch(dest) else None})
    shards = getattr(service, "shards", [service])
    history = sorted((s for shard in shards if hasattr(shard, "live_samples") for s in shard.live_samples()),
                     key=lambda sample: sample["at"])[-32:]
    samples = [{k: number(s.get(k)) for k in ("at", "completion_tokens", "tokens_per_second", "ttft_ms")}
               for s in history]
    group_agent = health.get("agent") or {}
    schedule = health.get("schedule") or {}
    ggml = health.get("ggml") or {}
    gate = (ggml.get("gate") or {}).get("passes")
    return {"status": state, "model": label(health.get("loaded_model")), "engine": label(health.get("execution")),
            "backend": label(health.get("backend")), "device": label(health.get("device")),
            "active": active, "queued": queued, "completed": number(health.get("completed_requests")),
            "failed": number(health.get("failed_requests")), "samples": samples, "decisions": decisions,
            "config": label(agent.get("serving")) or label(group_agent.get("serving")),
            "planner": group_agent.get("planner") if group_agent.get("planner") in ("idle", "running") else None,
            "schedule_width": number(schedule.get("max_width")),
            "schedule_order": schedule.get("order") if schedule.get("order") in ("fifo", "sjf") else None,
            "workers": len(shards), "live_windows": number(agent.get("live_windows")),
            "decision_us": number(group_agent.get("decision_us")), "gate": gate if type(gate) is bool else None}


def snapshot(service: Any, path: Path):
    return {"schema": "ironmule.live.v1", "observed_at": time.time(), "runtime": runtime(service),
            "benchmark": benchmark(path)}


def main():
    from .http_server import create_server
    from .state import ProductStore

    parser = argparse.ArgumentParser(description="Watch benchmark progress on a local read-only dashboard.")
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--progress", type=Path, help="path to a same-GGUF or deployment-comparison progress file")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
    path = (args.progress or os.environ.get("IRONMULE_BENCH_PROGRESS")
            or ProductStore(args.state_dir).root / "benchmark-progress.json")
    with create_server(None, port=args.port, progress=path) as server:
        print(f"http://127.0.0.1:{server.server_port}/live", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
