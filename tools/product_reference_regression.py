"""Run one guarded installed-product regression against a frozen Gemma reference.

This is a correctness regression, not a benchmark.  It executes exactly one
warmup and three greedy eight-token requests through the installed reference
worker, compares each complete result to the historical real reference, and
stores only digests of prompt and output material.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
import math
from pathlib import Path
import threading
import time
from typing import Any
import uuid

# This module deliberately does not add the repository root to sys.path.  When
# invoked from tools/, product_load_screen is a sibling helper; its installed
# package proof rejects source-tree product/evidence imports before model work.
from product_load_screen import (  # type: ignore[import-not-found]
    DEFAULT_WAIT_READY,
    READINESS_POLICY,
    _exclusive_write,
    _installed_package_proof,
    _sleep_cancel,
    _wait_ready,
)

from friday_evidence.budget import BudgetError, BudgetGuard
from friday_evidence.canonical import canonical_sha256
from friday_evidence.events import EventJournal
from friday_evidence.identity import runtime_identity
from ironmule_product.backend import MLXWorkerClient
from ironmule_product.calibration import model_lease
from ironmule_product.calibration_plan import PROMPT
from ironmule_product.memory import LoadMemoryGuard, MemoryGuardError, SWAP_DELTA_LIMIT_BYTES
from ironmule_product.readiness import hardware_identity
from ironmule_product.state import ProductStore
from ironmule_product.types import GenerationRequest


SCHEMA = "ironmule.product_reference_regression.v1"
JOURNAL_NAME = "reference-regression.sqlite3"
REFERENCE = Path(__file__).resolve().parents[1] / "research" / "raw" / "PROD1_GEMMA_SMOKE_20260907_attempt3.json"
EXPECTED_MODEL = "mlx-community/gemma-3-4b-it-4bit"
EXPECTED_REVISION = "93724907d4ed1745d2fe50baadf3b0b01a65abf2"
FROZEN_MODELS = {
    EXPECTED_MODEL: EXPECTED_REVISION,
    "mlx-community/gemma-3-12b-it-4bit": "86cc6a8dedbc456dd0e4af01a9d09f396f77e558",
}
MAX_TOKENS = 8
REPEATS = 3
REQUEST_TIMEOUT_SECONDS = 6.0
MAX_RUN_SECONDS = 120.0


class RegressionFailure(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise RegressionFailure("source_unavailable") from exc


def _source_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = (
        Path(__file__).resolve(), root / "tools" / "product_load_screen.py",
        root / "ironmule_product" / "backend.py", root / "ironmule_product" / "memory.py",
        root / "ironmule_product" / "readiness.py", root / "ironmule_product" / "types.py",
        root / "ironmule_product" / "calibration.py", root / "ironmule_product" / "calibration_plan.py",
        root / "friday_evidence" / "budget.py", root / "friday_evidence" / "events.py",
        root / "friday_evidence" / "identity.py", REFERENCE,
        root / "docs" / "PROD4_PROCESS_MEMORY_SPEC.md",
        root / "docs" / "PROD6_MODEL_POLICY_SPEC.md",
    )
    result: dict[str, str] = {}
    for path in paths:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
            raise RegressionFailure("source_manifest_invalid")
        result[path.relative_to(root).as_posix()] = _sha256_file(path)
    return result


def _output_digest(tokens: list[int], text: str, done: dict[str, Any]) -> str:
    return canonical_sha256({
        "tokens": tokens, "text": text,
        "finish_reason": done.get("finish_reason"),
        "prompt_tokens": done.get("prompt_tokens"),
        "completion_tokens": done.get("completion_tokens"),
    })


def _reference_samples(model_id: str = EXPECTED_MODEL, revision: str = EXPECTED_REVISION) -> tuple[list[str], str]:
    """Return full-result hashes for one exact historical real Gemma reference."""
    if PROMPT != "Write a short sentence about apples.":
        raise RegressionFailure("fixed_prompt_changed")
    try:
        value = json.loads(REFERENCE.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RegressionFailure("reference_invalid") from exc
    models = value.get("models") if isinstance(value, dict) else None
    if (not isinstance(models, list) or value.get("schema") != "ironmule.product_gemma_smoke.v1"
            or value.get("status") != "passed" or value.get("performance_claim") is not False
            or value.get("source_sha256_before") != value.get("source_sha256_after")):
        raise RegressionFailure("reference_invalid")
    if FROZEN_MODELS.get(model_id) != revision:
        raise RegressionFailure("reference_model_not_exact")
    matches = [row for row in models if isinstance(row, dict) and row.get("model_id") == model_id
               and row.get("revision") == revision]
    if len(matches) != 1:
        raise RegressionFailure("reference_model_not_exact")
    row = matches[0]
    samples = row.get("reference", {}).get("samples")
    if (row.get("status") != "passed" or not isinstance(samples, list)
            or len(samples) != REPEATS + 1):
        raise RegressionFailure("reference_samples_invalid")
    result: list[str] = []
    for sample in samples:
        if not isinstance(sample, dict) or not isinstance(sample.get("tokens"), list) or not all(
            type(token) is int and token >= 0 for token in sample["tokens"]
        ) or not isinstance(sample.get("text"), str):
            raise RegressionFailure("reference_samples_invalid")
        if (sample.get("finish_reason") not in ("stop", "length")
                or type(sample.get("prompt_tokens")) is not int or sample["prompt_tokens"] <= 0
                or type(sample.get("completion_tokens")) is not int
                or sample["completion_tokens"] != len(sample["tokens"])
                or not 1 <= sample["completion_tokens"] <= MAX_TOKENS):
            raise RegressionFailure("reference_samples_invalid")
        result.append(canonical_sha256({
            "tokens": sample["tokens"], "text": sample["text"],
            "finish_reason": sample["finish_reason"], "prompt_tokens": sample["prompt_tokens"],
            "completion_tokens": sample["completion_tokens"],
        }))
    return result, _sha256_file(REFERENCE)


def _installed_import_proof() -> dict[str, Any]:
    """Prove every product/evidence module this harness uses is installed."""
    proof = _installed_package_proof()
    root = Path(__file__).resolve().parents[1]
    names = (
        "ironmule_product.backend", "ironmule_product.calibration",
        "ironmule_product.calibration_plan", "ironmule_product.memory",
        "ironmule_product.readiness", "ironmule_product.state", "ironmule_product.types",
        "friday_evidence.budget", "friday_evidence.canonical", "friday_evidence.events",
        "friday_evidence.identity",
    )
    hashes: dict[str, str] = {}
    for name in names:
        module = importlib.import_module(name)
        raw_origin = getattr(module, "__file__", None)
        if not isinstance(raw_origin, str):
            raise RegressionFailure("installed_package_origin_missing")
        origin = Path(raw_origin).resolve()
        try:
            origin.relative_to(root)
        except ValueError:
            pass
        else:
            raise RegressionFailure("source_tree_package_in_use")
        if not origin.is_file() or origin.is_symlink() or origin.stat().st_size > 16 * 1024 * 1024:
            raise RegressionFailure("installed_package_origin_invalid")
        hashes[name] = _sha256_file(origin)
    return {**proof, "all_product_evidence_module_sha256": canonical_sha256(hashes)}


def _collect(client: MLXWorkerClient, request: GenerationRequest, check) -> tuple[str, float, int]:
    """Consume one complete worker response; callers retain no raw output."""
    tokens: list[int] = []
    text: list[str] = []
    done: dict[str, Any] | None = None
    started = time.monotonic()
    stream = client.stream(request, timeout=REQUEST_TIMEOUT_SECONDS, variant="reference")
    try:
        for event in stream:
            # This runs on the EventJournal owner's thread. LoadMemoryGuard
            # internally throttles the actual host poll to its fixed cadence.
            check()
            if event["type"] == "token":
                tokens.append(event["token_id"])
                text.append(event["text"])
            elif event["type"] == "done":
                done = event
    finally:
        stream.close()
    elapsed = time.monotonic() - started
    if done is None or done.get("finish_reason") not in ("stop", "length"):
        raise RegressionFailure("generation_not_terminal")
    peak = done.get("metrics", {}).get("peak_memory")
    if type(peak) not in (int, float) or not math.isfinite(peak) or peak < 0:
        raise RegressionFailure("peak_telemetry_missing")
    return _output_digest(tokens, "".join(text), done), elapsed, int(peak * 1_000_000_000)


def run(state_dir: Path, model_id: str, output: Path) -> dict[str, Any]:
    if model_id not in FROZEN_MODELS:
        raise RegressionFailure("model_is_not_frozen_reference")
    output = output.expanduser()
    if not output.is_absolute():
        output = Path.cwd() / output
    if output.exists() or output.is_symlink():
        raise RegressionFailure("output_exists")
    state_dir = state_dir.expanduser().absolute()
    store = ProductStore(state_dir)
    store.settings()
    spec = store.model(model_id)
    if spec.revision != FROZEN_MODELS[model_id]:
        raise RegressionFailure("registered_revision_is_not_frozen_reference")
    expected_hashes, reference_sha256 = _reference_samples(model_id, spec.revision)
    manifest_before = _source_manifest()
    source_harness_sha256 = _sha256_file(Path(__file__).resolve())
    specification_sha256 = canonical_sha256({
        "schema": SCHEMA, "prompt_sha256": canonical_sha256(PROMPT),
        "max_tokens": MAX_TOKENS, "warmups": 1, "repeats": REPEATS,
        "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        "readiness_policy": asdict(READINESS_POLICY),
        "swap_delta_limit_bytes": SWAP_DELTA_LIMIT_BYTES,
    })
    run_id = uuid.uuid4().hex
    report: dict[str, Any] = {
        "schema": SCHEMA, "run_id": run_id, "status": "started",
        "model_id": spec.model_id, "revision": spec.revision,
        "reference_sha256": reference_sha256,
        "source_harness_sha256": source_harness_sha256,
        "specification_sha256": specification_sha256,
        "source_manifest_before": manifest_before,
        "source_manifest_sha256": canonical_sha256(manifest_before),
        "prompt_sha256": canonical_sha256(PROMPT),
        "expected_output_sha256": expected_hashes,
        "observed_output_sha256": [], "matches_reference": [], "peak_memory_bytes": [],
        "performance_claim": False, "activation_allowed": False,
        "readiness_samples": [], "load_memory_samples": [], "worker": None,
    }
    client: MLXWorkerClient | None = None
    process = None
    guard: BudgetGuard | None = None
    load_guard: LoadMemoryGuard | None = None
    deadline = time.monotonic() + MAX_RUN_SECONDS

    def check() -> None:
        if time.monotonic() >= deadline:
            raise RegressionFailure("run_deadline")
        if store.settings()["optimization_paused"]:
            raise RegressionFailure("paused")
        if guard is not None:
            try:
                guard.check_wall()
            except BudgetError as exc:
                report["budget_error_code"] = "budget_error"
                raise RegressionFailure("budget_error") from exc
        if load_guard is not None and process is not None:
            try:
                load_guard(process.pid)
            except MemoryGuardError as exc:
                raise RegressionFailure(exc.code) from exc

    def sleep(seconds: float) -> None:
        _sleep_cancel(seconds, check, threading.Event())

    try:
        with EventJournal(state_dir / JOURNAL_NAME) as journal:
            journal.append(run_id, "run_started", {
                "model_id": spec.model_id, "revision": spec.revision,
                "source_harness_sha256": source_harness_sha256,
                "source_manifest_sha256": report["source_manifest_sha256"],
                "reference_sha256": reference_sha256,
                "spec_sha256": specification_sha256,
            })
            # This verifies imports are installed package files before a worker
            # can be created; it deliberately fails when run from this checkout.
            report["installed_package"] = _installed_import_proof()
            with model_lease(store):
                initial, _ = _wait_ready(
                    deadline=min(deadline, time.monotonic() + DEFAULT_WAIT_READY), global_deadline=deadline,
                    check=check, sleep=sleep, journal=journal, run_id=run_id,
                    phase="before_load", sample_sink=report["readiness_samples"],
                )
                hardware = hardware_identity()
                memory_total = initial.get("memory_total_bytes")
                swap_baseline = initial.get("swap_used_bytes")
                if type(memory_total) is not int or memory_total <= 0 or type(swap_baseline) is not int or swap_baseline < 0:
                    raise RegressionFailure("memory_telemetry_invalid")
                host = {**initial, **hardware, "chip_name": hardware.get("chip_name"), "memory_total_bytes": memory_total}
                report["identity_before"] = runtime_identity(spec.as_dict(), host)
                journal.append(run_id, "validation", {"state": "identity_before", "identity": report["identity_before"]})
                after_identity, _ = _wait_ready(
                    deadline=min(deadline, time.monotonic() + DEFAULT_WAIT_READY), global_deadline=deadline,
                    check=check, sleep=sleep, journal=journal, run_id=run_id,
                    phase="after_identity", sample_sink=report["readiness_samples"],
                )
                if (type(after_identity.get("swap_used_bytes")) is not int
                        or after_identity["swap_used_bytes"] - swap_baseline > SWAP_DELTA_LIMIT_BYTES):
                    raise RegressionFailure("swap_growth")
                guard = BudgetGuard(sleeper=sleep)

                def on_memory(sample: dict[str, Any]) -> None:
                    report["load_memory_samples"].append(dict(sample))
                    journal.append(run_id, "validation", {"state": "load_memory_sample", "observation": dict(sample)})

                load_guard = LoadMemoryGuard(swap_baseline_bytes=swap_baseline, memory_total_bytes=memory_total, on_sample=on_memory)
                client = MLXWorkerClient(spec, startup_timeout=DEFAULT_WAIT_READY)
                worker_record = {"pid": None, "returncode": None, "ready": False, "closed": False}
                report["worker"] = worker_record

                def startup_guard(pid: int) -> None:
                    nonlocal process
                    process = client._process
                    if process is None or process.pid != pid:
                        raise RegressionFailure("worker_start_handle_missing")
                    worker_record["pid"] = pid
                    check()

                active_error: BaseException | None = None
                try:
                    ready = client.start(startup_guard=startup_guard)
                    if process is None:
                        raise RegressionFailure("worker_start_handle_missing")
                    raw_ready = {key: ready.get(key) for key in (
                        "startup_wall_seconds", "process_peak_rss_bytes", "mlx_active_bytes",
                        "mlx_peak_bytes", "mlx_cache_bytes", "recommended_working_set_bytes",
                    )}
                    report["ready_telemetry_raw"] = raw_ready
                    journal.append(run_id, "validation", {"state": "load_memory_ready_raw", "ready": raw_ready})
                    report["ready_telemetry"] = load_guard.validate_ready(ready)
                    report["load_memory_ready_observation"] = load_guard(process.pid, force=True)
                    worker_record["ready"] = True
                    journal.append(run_id, "worker_started", {"pid": process.pid, "ready": report["ready_telemetry"]})
                    guard.required_break()
                    after_load, _ = _wait_ready(
                        deadline=min(deadline, time.monotonic() + DEFAULT_WAIT_READY), global_deadline=deadline,
                        check=check, sleep=sleep, journal=journal, run_id=run_id,
                        phase="after_load", memory_guard=load_guard, sample_sink=report["readiness_samples"],
                    )
                    for index in range(REPEATS + 1):
                        check()
                        request = GenerationRequest(spec.model_id, (("user", PROMPT),), max_tokens=MAX_TOKENS,
                                                    temperature=0.0, top_p=1.0)
                        started_request = time.monotonic()
                        try:
                            actual, _elapsed, peak_bytes = _collect(client, request, check)
                        except BaseException as exc:
                            active_error = exc
                            raise
                        finally:
                            elapsed = time.monotonic() - started_request
                            try:
                                guard.record_gpu(elapsed)  # conservative full request wall time, including polls
                            except BudgetError as exc:
                                report["budget_error_code"] = "budget_error"
                                if active_error is None:
                                    raise RegressionFailure("budget_error") from exc
                        report["peak_memory_bytes"].append(peak_bytes)
                        report["observed_output_sha256"].append(actual)
                        matched = actual == expected_hashes[index]
                        report["matches_reference"].append(matched)
                        journal.append(run_id, "validation", {
                            "state": "output", "sample_index": index, "warmup": index == 0,
                            "expected_sha256": expected_hashes[index], "observed_sha256": actual,
                            "peak_memory_bytes": peak_bytes,
                            "status": "passed" if matched and peak_bytes <= 0.60 * memory_total else "failed",
                        })
                        if peak_bytes > 0.60 * memory_total:
                            raise RegressionFailure("peak_memory_limit")
                        if not matched:
                            raise RegressionFailure("reference_output_mismatch")
                        guard.required_break()
                except BaseException as exc:
                    active_error = exc
                    raise
                finally:
                    active_process = process
                    process = None
                    load_guard = None
                    cleanup_error: BaseException | None = None
                    if client is not None:
                        try:
                            client.close()
                        except BaseException as exc:
                            cleanup_error = exc
                    worker_record["returncode"] = active_process.poll() if active_process is not None else None
                    worker_record["closed"] = active_process is not None and active_process.poll() is not None
                    try:
                        journal.append(run_id, "validation", {"state": "worker_cleanup", "worker": dict(worker_record)})
                    except BaseException as exc:
                        if cleanup_error is None:
                            cleanup_error = exc
                    if active_error is None and cleanup_error is not None:
                        raise cleanup_error
                    if active_error is None and worker_record["returncode"] != 0:
                        raise RegressionFailure("worker_exit_nonzero")

                report["identity_after"] = runtime_identity(
                    spec.as_dict(), {**after_load, **hardware, "chip_name": hardware.get("chip_name"), "memory_total_bytes": memory_total},
                )
                if report["identity_after"]["identity_sha256"] != report["identity_before"]["identity_sha256"]:
                    raise RegressionFailure("identity_changed")
                journal.append(run_id, "validation", {"state": "identity_after", "identity": report["identity_after"]})
                report["budget"] = guard.summary()
            report["source_manifest_after"] = _source_manifest()
            if report["source_manifest_after"] != manifest_before:
                raise RegressionFailure("source_changed")
            report["status"] = "passed"
            journal.append(run_id, "run_finished", {"status": "passed", "worker": report["worker"],
                                                       "identity_sha256": report["identity_after"]["identity_sha256"]})
    except BaseException as exc:
        report["status"] = "failed"
        report["error_code"] = getattr(exc, "code", type(exc).__name__.lower())
        report["error_type"] = type(exc).__name__
        if guard is not None:
            report["budget"] = guard.summary()
        report.setdefault("source_manifest_after", _source_manifest())
        try:
            with EventJournal(state_dir / JOURNAL_NAME) as journal:
                journal.append(run_id, "run_finished", {"status": "failed", "error_code": report["error_code"],
                                                           "error_type": report["error_type"], "worker": report["worker"]})
        except Exception:
            report["journal_terminal_error"] = True
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"state": "not_released", "hint": "pass --execute"}))
        return 78
    try:
        report = run(args.state_dir, args.model, args.output)
        _exclusive_write(args.output, report)
        print(json.dumps({"status": report["status"], "run_id": report["run_id"], "output": str(args.output)}))
        return 0 if report["status"] == "passed" else 1
    except BaseException as exc:
        print(json.dumps({"status": "failed", "error_code": getattr(exc, "code", type(exc).__name__.lower())}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
