"""Run the preregistered real 12B long-context exact-reference screen.

This harness is intentionally inert until ``--execute`` is supplied.  Its
stock-reference child speaks a bounded JSON pipe and emits hashes/counts only.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import selectors
from typing import Any
import uuid


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "ironmule.prod8_long_context_reference.v1"
MODEL_ID = "mlx-community/gemma-3-12b-it-4bit"
REVISION = "86cc6a8dedbc456dd0e4af01a9d09f396f77e558"
MAX_TOKENS = 8
REPEATS = 3
REQUEST_TIMEOUT_SECONDS = 6.0
STOCK_STARTUP_TIMEOUT_SECONDS = 120.0
MAX_RUN_SECONDS = 20.0 * 60.0
MIN_PROMPT_TOKENS = 1000
MAX_PROMPT_TOKENS = 1100
EXPECTED_PROMPT_TOKENS = 1077
MAX_CHILD_OUTPUT_BYTES = 1024 * 1024
MAX_HTTP_OUTPUT_BYTES = 1024 * 1024
CALIBRATION = ROOT / "research" / "raw" / "PROD6_12B_installed_20260907_attempt1.json"
CALIBRATION_RUN_ID = "ef5b8f0273404145be3f03a4a793f499"
CALIBRATION_REPORT_SHA256 = "e3febb770142615d01dac19bbf32b19d8be4b137806ed8af606044f89334cab9"
CALIBRATION_EVENTS_SHA256 = "183760f276bfe01aaacd283bed21db37bc4ec30fd2276c81f17ddd235a2dbda0"


class LongContextFailure(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _message() -> tuple[tuple[str, str], ...]:
    """Return a benign deterministic message; never write its content to disk."""
    # This is deliberately fixed before execution. The token-count range lives
    # in PROD8, not in a value adjusted from a native observation.
    unit = "Public orchard note: apples grow on trees and need sunlight. "
    return (("user", unit * 88 + "END-OF-PUBLIC-ORCHARD-NOTE."),)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _stock_child() -> int:
    """Load stock MLX-LM in a private process and return bounded metadata only."""
    try:
        request = json.loads(sys.stdin.buffer.readline(MAX_CHILD_OUTPUT_BYTES + 1))
        if not isinstance(request, dict) or request.get("revision") != REVISION:
            raise LongContextFailure("stock_request_invalid")
        snapshot = request.get("snapshot_path")
        messages = request.get("messages")
        if not isinstance(snapshot, str) or not isinstance(messages, list):
            raise LongContextFailure("stock_request_invalid")
        from ironmule_product.model_policy import validate_model_config
        model_config = validate_model_config(snapshot, REVISION)
        with redirect_stdout(sys.stderr):
            import mlx.core as mx
            from mlx_lm import load, stream_generate
            mx.set_default_device(mx.gpu)
            model, tokenizer = load(snapshot, tokenizer_config={"trust_remote_code": False},
                                    model_config={**model_config, "model_file": None}, revision=REVISION)
            prompt_ids = list(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
        prompt_count = len(prompt_ids)
        if not MIN_PROMPT_TOKENS <= prompt_count <= MAX_PROMPT_TOKENS or prompt_count != EXPECTED_PROMPT_TOKENS:
            raise LongContextFailure("context_count_out_of_range")
        ready = {
            "schema": "ironmule.prod8_stock_reference.v1", "prompt_tokens": prompt_count,
            "prompt_ids_sha256": _sha256(json.dumps(prompt_ids, separators=(",", ":")).encode("ascii")),
            "state": "ready", "device": str(mx.default_device()),
            "mlx_active_bytes": int(mx.get_active_memory()), "mlx_peak_bytes": int(mx.get_peak_memory()),
        }
        encoded = json.dumps(ready, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if len(encoded) > MAX_CHILD_OUTPUT_BYTES:
            raise LongContextFailure("stock_metadata_oversize")
        sys.stdout.buffer.write(encoded + b"\n")
        sys.stdout.buffer.flush()
        for line in sys.stdin.buffer:
            command = json.loads(line)
            if command == {"type": "shutdown"}:
                return 0
            if command != {"type": "generate"}:
                raise LongContextFailure("stock_command_invalid")
            started = time.monotonic()
            with redirect_stdout(sys.stderr):
                responses = list(stream_generate(model, tokenizer, prompt_ids, max_tokens=MAX_TOKENS))
            if not responses:
                raise LongContextFailure("stock_generation_empty")
            tokens, text, final = [r.token for r in responses], "".join(r.text for r in responses), responses[-1]
            result = {"state": "sample", "elapsed_seconds": time.monotonic() - started,
                "peak_memory_bytes": int(mx.get_peak_memory()), "output_sha256": _sha256(json.dumps({"tokens": tokens, "text": text, "finish_reason": final.finish_reason, "prompt_tokens": final.prompt_tokens, "completion_tokens": final.generation_tokens}, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")),
                "text_sha256": _sha256(text.encode("utf-8")), "finish_reason": final.finish_reason,
                "prompt_tokens": final.prompt_tokens, "completion_tokens": final.generation_tokens}
            sys.stdout.buffer.write(json.dumps(result, separators=(",", ":")).encode("utf-8") + b"\n")
            sys.stdout.buffer.flush()
    except BaseException as exc:
        code = getattr(exc, "code", type(exc).__name__.lower())
        sys.stdout.write(json.dumps({"status": "failed", "error_code": code}, separators=(",", ":")) + "\n")
        return 1


def _read_stock_event(child, *, check, load_guard, timeout: float) -> dict[str, Any]:
    """Read one bounded child frame while the parent enforces readiness/memory."""
    assert child.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(child.stdout, selectors.EVENT_READ)
    deadline = time.monotonic() + timeout
    buffered = getattr(child, "_prod8_buffer", b"")
    try:
        while True:
            check()
            if time.monotonic() >= deadline:
                raise LongContextFailure("stock_request_timeout")
            if child.poll() is None:
                load_guard(child.pid)
            if b"\n" in buffered:
                line, buffered = buffered.split(b"\n", 1)
                setattr(child, "_prod8_buffer", buffered)
                try:
                    value = json.loads(line)
                except (TypeError, json.JSONDecodeError) as exc:
                    raise LongContextFailure("stock_reference_invalid") from exc
                if isinstance(value, dict) and value.get("status") == "failed":
                    raise LongContextFailure(str(value.get("error_code", "stock_reference_failed")))
                return value
            if selector.select(min(0.25, max(0.0, deadline - time.monotonic()))):
                chunk = os.read(child.stdout.fileno(), 65536)
                if not chunk:
                    raise LongContextFailure("stock_reference_failed")
                buffered += chunk
                if len(buffered) > MAX_CHILD_OUTPUT_BYTES:
                    raise LongContextFailure("stock_metadata_oversize")
            if child.poll() is not None and b"\n" not in buffered:
                raise LongContextFailure("stock_reference_failed")
    finally:
        selector.close()


def _stock_reference(snapshot_path: str, messages: tuple[tuple[str, str], ...], *, check, load_guard,
                     worker: dict[str, Any], guard, pause, after_ready, bind_process, result,
                     record_event) -> dict[str, Any]:
    """Run the independent stock child; prompt bytes travel only over its pipe."""
    child = subprocess.Popen(
        [sys.executable, "-I", "-u", str(Path(__file__).resolve()), "--stock-child"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        cwd=Path(__file__).resolve().parent,
    )
    worker["pid"] = child.pid
    bind_process(child, load_guard)
    assert child.stdin is not None
    try:
        child.stdin.write((json.dumps({"revision": REVISION, "snapshot_path": snapshot_path,
            "messages": [{"role": role, "content": content} for role, content in messages]}, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")); child.stdin.flush()
        raw_ready = _read_stock_event(child, check=check, load_guard=load_guard,
                                      timeout=STOCK_STARTUP_TIMEOUT_SECONDS)
        if isinstance(raw_ready, dict):
            result.update(raw_ready)
        ready = _validate_stock_metadata(raw_ready)
        if ready["mlx_peak_bytes"] > load_guard.memory_total_bytes * .60:
            raise LongContextFailure("stock_peak_memory_limit")
        after_ready(child.pid)
        samples = result.setdefault("samples", [])
        for _ in range(REPEATS + 1):
            started = time.monotonic()
            primary: BaseException | None = None
            try:
                child.stdin.write(b'{"type":"generate"}\n'); child.stdin.flush()
                sample = _read_stock_event(child, check=check, load_guard=load_guard,
                                           timeout=REQUEST_TIMEOUT_SECONDS)
            except BaseException as exc:
                primary = exc
            parent_elapsed = time.monotonic() - started
            event = {"label": f"stock_{len(samples)}", "elapsed_seconds": parent_elapsed,
                     "start_monotonic": started, "end_monotonic": started + parent_elapsed,
                     "status": "failed" if primary is not None else "completed"}
            record_event(event)
            try:
                guard.record_gpu(parent_elapsed)
            except BaseException:
                if primary is None:
                    raise
            if primary is not None:
                raise primary
            if sample.get("state") != "sample" or not isinstance(sample.get("elapsed_seconds"), (int, float)):
                raise LongContextFailure("stock_reference_invalid")
            sample["parent_elapsed_seconds"] = parent_elapsed
            if (type(sample.get("peak_memory_bytes")) is not int or sample["peak_memory_bytes"] < 0
                    or sample["peak_memory_bytes"] > load_guard.memory_total_bytes * .60):
                raise LongContextFailure("stock_peak_memory_limit")
            sample["phase"] = "warmup" if not samples else "recorded"
            sample["sample_index"] = len(samples)
            samples.append(sample)
            record_event({"label": f"stock_{len(samples) - 1}", "sample": dict(sample)})
            pause()
        child.stdin.write(b'{"type":"shutdown"}\n'); child.stdin.close(); child.wait(timeout=5)
        result = _validate_stock_samples({**ready, "samples": samples})
    finally:
        primary_error = sys.exc_info()[1]
        cleanup_failed = False
        if child.poll() is None:
            try:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait(timeout=5)
            except BaseException:
                cleanup_failed = True
        worker["returncode"] = child.returncode
        worker["closed"] = child.returncode is not None
        bind_process(None, None)
        if cleanup_failed and primary_error is None:
            raise LongContextFailure("stock_cleanup_failed")
    if not load_guard.samples:
        raise LongContextFailure("stock_memory_sample_missing")
    if child.returncode != 0:
        raise LongContextFailure("stock_reference_failed")
    return result


def _validate_stock_metadata(result: Any) -> dict[str, Any]:
    """Pure schema gate for bounded stock-child metadata, with no MLX import."""
    if (not isinstance(result, dict) or result.get("schema") != "ironmule.prod8_stock_reference.v1"
            or type(result.get("prompt_tokens")) is not int
            or not MIN_PROMPT_TOKENS <= result["prompt_tokens"] <= MAX_PROMPT_TOKENS
            or not _is_sha256(result.get("prompt_ids_sha256"))
            or "gpu" not in str(result.get("device", "")).lower()
            or type(result.get("mlx_active_bytes")) is not int or result["mlx_active_bytes"] <= 0
            or type(result.get("mlx_peak_bytes")) is not int or result["mlx_peak_bytes"] <= 0
            or result.get("state") != "ready"):
        raise LongContextFailure("stock_reference_invalid")
    return result


def _validate_stock_samples(result: dict[str, Any]) -> dict[str, Any]:
    if (not isinstance(result.get("samples"), list) or len(result["samples"]) != REPEATS + 1
            or not all(isinstance(value, dict) and _is_sha256(value.get("output_sha256"))
                          and _is_sha256(value.get("text_sha256"))
                          and value.get("finish_reason") in ("stop", "length")
                          and value.get("prompt_tokens") == result["prompt_tokens"]
                          and type(value.get("completion_tokens")) is int and 1 <= value["completion_tokens"] <= MAX_TOKENS
                          and value.get("phase") == ("warmup" if value.get("sample_index") == 0 else "recorded")
                          and type(value.get("sample_index")) is int and 0 <= value["sample_index"] <= REPEATS
                          and isinstance(value.get("parent_elapsed_seconds"), (int, float))
                          and math.isfinite(value["parent_elapsed_seconds"])
                          and 0 <= value["parent_elapsed_seconds"] <= REQUEST_TIMEOUT_SECONDS + 0.25
                          for value in result["samples"])):
        raise LongContextFailure("stock_reference_invalid")
    return result


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _canonical_sha256(value: Any) -> str:
    return _sha256(json.dumps(value, ensure_ascii=False, allow_nan=False,
                              sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _provider_distribution_proof() -> dict[str, str]:
    """Hash MLX/MLX-LM provider files without importing native controller code."""
    from importlib import metadata
    proof: dict[str, str] = {}
    for distribution_name in ("mlx", "mlx-lm"):
        distribution = metadata.distribution(distribution_name)
        files = distribution.files or ()
        selected = [entry for entry in files if str(entry).endswith(".so")
                    or (str(entry).startswith("mlx_lm/") and str(entry).endswith(".py"))]
        if not selected:
            raise LongContextFailure("provider_distribution_invalid")
        hashes = {}
        for entry in selected:
            path = Path(distribution.locate_file(entry)).resolve()
            if not path.is_file() or path.is_symlink() or path.stat().st_size > 64 * 1024 * 1024:
                raise LongContextFailure("provider_distribution_invalid")
            hashes[str(entry)] = _sha256(path.read_bytes())
        proof[distribution_name] = _canonical_sha256(hashes)
    return proof


def _snapshot_metadata_proof(snapshot_path: str) -> dict[str, str]:
    """Bind tokenizer/template/config bytes in addition to the weight identity."""
    root = Path(snapshot_path)
    names = ("config.json", "tokenizer.json", "tokenizer_config.json",
             "special_tokens_map.json", "chat_template.jinja", "added_tokens.json")
    hashes = {}
    for name in names:
        path = root / name
        if not path.exists():
            if name in names[:3]:
                raise LongContextFailure("model_metadata_missing")
            continue
        if not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
            raise LongContextFailure("model_metadata_invalid")
        hashes[name] = _sha256(path.read_bytes())
    return hashes


def _calibration_gate(path: Path = CALIBRATION, *, evaluator=None) -> dict[str, str]:
    """Bind PROD8 to the one completed, resource-valid 12B calibration export."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LongContextFailure("calibration_prerequisite_invalid") from exc
    if not isinstance(value, dict):
        raise LongContextFailure("calibration_prerequisite_invalid")
    report = value.get("report")
    if not isinstance(report, dict):
        raise LongContextFailure("calibration_prerequisite_invalid")
    exits = report.get("worker_exit_codes") if isinstance(report, dict) else None
    if (value.get("schema") != "ironmule.calibration_export.v1"
            or value.get("report_sha256") != CALIBRATION_REPORT_SHA256
            or value.get("events_sha256") != CALIBRATION_EVENTS_SHA256
            or _canonical_sha256(report) != value.get("report_sha256")
            or _canonical_sha256(value.get("events")) != value.get("events_sha256")
            or report.get("schema") != "ironmule.calibration.v1"
            or report.get("run_id") != CALIBRATION_RUN_ID
            or report.get("model_id") != MODEL_ID or report.get("revision") != REVISION
            or report.get("status") != "measured" or report.get("resource_valid") is not True
            or report.get("error_code") is not None
            or report.get("evaluation", {}).get("verdict") != "inconclusive"
            or not isinstance(report.get("samples"), list) or len(report["samples"]) != 90
            or not all(row.get("status") == "passed" and row.get("correctness") is True
                       for row in report["samples"])
            or report.get("identity_before") != report.get("identity_after")
            or not isinstance(exits, list) or len(exits) != 3
            or not all(row.get("normal_shutdown") is True and row.get("returncode") == 0 for row in exits)):
        raise LongContextFailure("calibration_prerequisite_invalid")
    if evaluator is not None and evaluator(report) != report.get("evaluation"):
        raise LongContextFailure("calibration_prerequisite_invalid")
    return {"run_id": CALIBRATION_RUN_ID, "report_sha256": CALIBRATION_REPORT_SHA256,
            "events_sha256": CALIBRATION_EVENTS_SHA256, "artifact_sha256": _sha256(path.read_bytes())}


def _off_main_thread(call, *, check, load_guard, pid: int, on_abort):
    """Run blocking loopback I/O away from the journal/controller owner thread."""
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prod8-http")
    future = pool.submit(call)
    deadline = time.monotonic() + REQUEST_TIMEOUT_SECONDS
    try:
        while not future.done():
            check(); load_guard(pid)
            if time.monotonic() >= deadline:
                raise LongContextFailure("http_request_timeout")
            time.sleep(0.05)
        return future.result()
    except BaseException:
        try:
            on_abort()
        except BaseException:
            pass
        try:
            future.result(timeout=0.5)
        except BaseException:
            pass
        raise
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _request_json(port: int, payload: dict[str, Any]) -> dict[str, Any]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=REQUEST_TIMEOUT_SECONDS)
    try:
        connection.request("POST", "/v1/chat/completions", json.dumps(payload), {"Content-Type": "application/json"})
        response = connection.getresponse()
        raw = response.read(MAX_HTTP_OUTPUT_BYTES + 1)
        if len(raw) > MAX_HTTP_OUTPUT_BYTES:
            raise LongContextFailure("http_response_oversize")
        if response.status != 200:
            raise LongContextFailure("http_status")
        return json.loads(raw)
    finally:
        connection.close()


def _assert_http(value: dict[str, Any], expected: dict[str, Any]) -> None:
    try:
        choice = value["choices"][0]
        usage = value["usage"]
        if (_sha256(choice["message"]["content"].encode("utf-8")) != expected["text_sha256"]
                or choice["finish_reason"] != expected["finish_reason"]
                or usage["prompt_tokens"] != expected["prompt_tokens"]
                or usage["completion_tokens"] != expected["completion_tokens"]):
            raise LongContextFailure("http_output_mismatch")
    except (KeyError, IndexError, TypeError) as exc:
        raise LongContextFailure("http_response_invalid") from exc


def _sse(port: int, payload: dict[str, Any]) -> tuple[str, str, int, int]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=REQUEST_TIMEOUT_SECONDS)
    try:
        connection.request("POST", "/v1/chat/completions", json.dumps({**payload, "stream": True}), {"Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status != 200 or response.getheader("Connection", "").lower() != "close":
            raise LongContextFailure("sse_framing")
        raw = response.read(MAX_HTTP_OUTPUT_BYTES + 1)
        if len(raw) > MAX_HTTP_OUTPUT_BYTES:
            raise LongContextFailure("http_response_oversize")
        wire = raw.decode("utf-8")
    finally:
        connection.close()
    if wire.count("data: [DONE]") != 1:
        raise LongContextFailure("sse_done_count")
    chunks = [json.loads(line[6:]) for line in wire.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]
    text = "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks)
    final = [chunk for chunk in chunks if chunk["choices"][0].get("finish_reason") is not None]
    if len(final) != 1:
        raise LongContextFailure("sse_terminal_invalid")
    usage = final[0].get("usage", {})
    return text, final[0]["choices"][0]["finish_reason"], usage.get("prompt_tokens"), usage.get("completion_tokens")


def run(state_dir: Path, output: Path) -> dict[str, Any]:
    # Deliberately import installed package modules only after the caller has
    # opted in. This reuses PROD4/PROD6 guards rather than copying them.
    from product_load_screen import DEFAULT_WAIT_READY, _installed_package_proof, _sleep_cancel, _wait_ready
    from product_reference_regression import _collect
    from friday_evidence.budget import BudgetError, BudgetGuard
    from friday_evidence.canonical import canonical_sha256
    from friday_evidence.events import EventJournal
    from friday_evidence.identity import runtime_identity
    from ironmule_product.backend import MLXWorkerClient
    from ironmule_product.calibration import model_lease
    from ironmule_product.memory import LoadMemoryGuard, MemoryGuardError, SWAP_DELTA_LIMIT_BYTES
    from ironmule_product.readiness import hardware_identity
    from ironmule_product.state import ProductStore
    from ironmule_product.types import GenerationRequest
    from ironmule_product.service import ProductService
    from ironmule_product.http_server import create_server
    from ironmule_product.evaluation import evaluate_report

    if output.exists() or output.is_symlink():
        raise LongContextFailure("output_exists")
    store = ProductStore(state_dir.expanduser().absolute())
    spec = store.model(MODEL_ID)
    if spec.revision != REVISION:
        raise LongContextFailure("registered_revision_not_frozen")
    messages = _message()
    source_paths = (Path(__file__).resolve(), ROOT / "docs" / "PROD8_LONG_CONTEXT_SPEC.md",
                    ROOT / "tools" / "product_load_screen.py", ROOT / "tools" / "product_reference_regression.py",
                    CALIBRATION)
    manifest_before = {path.relative_to(ROOT).as_posix(): _sha256(path.read_bytes()) for path in source_paths}
    run_id = uuid.uuid4().hex
    deadline = time.monotonic() + MAX_RUN_SECONDS
    report: dict[str, Any] = {"schema": SCHEMA, "run_id": run_id, "status": "started", "model_id": MODEL_ID,
        "revision": REVISION, "prompt_sha256": canonical_sha256(messages), "performance_claim": False,
        "activation_allowed": False, "readiness_samples": [], "load_memory_samples": [], "worker": None,
        "request_resource_events": [],
        "source_manifest_before": manifest_before, "source_manifest_sha256": _sha256(json.dumps(manifest_before, sort_keys=True).encode())}
    guard: BudgetGuard | None = None
    load_guard: Any = None
    process = None

    def check() -> None:
        if time.monotonic() >= deadline:
            raise LongContextFailure("run_deadline")
        if store.settings()["optimization_paused"]:
            raise LongContextFailure("paused")
        if guard is not None:
            try: guard.check_wall()
            except BudgetError as exc: raise LongContextFailure("budget_error") from exc
        if load_guard is not None and process is not None:
            try: load_guard(process.pid)
            except MemoryGuardError as exc: raise LongContextFailure(exc.code) from exc

    def sleep(seconds: float) -> None:
        _sleep_cancel(seconds, check, threading.Event())

    def budget_pause() -> None:
        # A fixed 24 s post-call pause keeps even a valid 6 s call below 25%
        # steady-state duty; required_break accounts for its first four seconds.
        assert guard is not None
        guard.required_break()
        sleep(20.0)

    def measured_call(label: str, call, journal):
        assert guard is not None
        started = time.monotonic()
        primary: BaseException | None = None
        try:
            return call()
        except BaseException as exc:
            primary = exc
            raise
        finally:
            elapsed = time.monotonic() - started
            row = {"label": label, "elapsed_seconds": elapsed,
                   "start_monotonic": started, "end_monotonic": started + elapsed,
                   "status": "failed" if primary is not None else "completed"}
            report["request_resource_events"].append(row)
            journal.append(run_id, "validation", {"state": "request_resource", **row})
            try:
                guard.record_gpu(elapsed)
            except BaseException:
                if primary is None:
                    raise

    try:
        with EventJournal(store.root / "long-context-reference.sqlite3") as journal, model_lease(store):
            journal.append(run_id, "run_started", {"model_id": MODEL_ID, "revision": REVISION,
                "source_manifest_sha256": report["source_manifest_sha256"], "prompt_sha256": report["prompt_sha256"]})
            report["calibration_prerequisite"] = _calibration_gate(evaluator=evaluate_report)
            report["installed_package"] = _installed_package_proof((
                "ironmule_product.backend", "ironmule_product.calibration", "ironmule_product.http_server",
                "ironmule_product.model_policy", "ironmule_product.readiness", "ironmule_product.service",
                "ironmule_product.state", "ironmule_product.types", "friday_evidence.budget",
                "friday_evidence.canonical", "friday_evidence.events", "friday_evidence.identity",
            ))
            installed_before = report["installed_package"]
            provider_before = _provider_distribution_proof()
            report["provider_distribution"] = provider_before
            metadata_before = _snapshot_metadata_proof(spec.snapshot_path)
            report["model_metadata_before"] = metadata_before
            initial, _ = _wait_ready(deadline=min(deadline, time.monotonic() + DEFAULT_WAIT_READY), global_deadline=deadline,
                check=check, sleep=sleep, journal=journal, run_id=run_id, phase="before_load", sample_sink=report["readiness_samples"])
            memory_total, swap_baseline = initial.get("memory_total_bytes"), initial.get("swap_used_bytes")
            if type(memory_total) is not int or memory_total <= 0 or type(swap_baseline) is not int or swap_baseline < 0:
                raise LongContextFailure("memory_telemetry_invalid")
            hardware = hardware_identity()
            report["identity_before"] = runtime_identity(spec.as_dict(), {**initial, **hardware, "memory_total_bytes": memory_total})
            calibration_identity = json.loads(CALIBRATION.read_text(encoding="utf-8"))["report"]["identity_before"]
            for field in ("model_sha256", "environment_sha256", "hardware_sha256", "code_sha256"):
                if report["identity_before"].get(field) != calibration_identity.get(field):
                    raise LongContextFailure("calibration_identity_mismatch")
            initial, _ = _wait_ready(deadline=min(deadline, time.monotonic() + DEFAULT_WAIT_READY),
                global_deadline=deadline, check=check, sleep=sleep, journal=journal, run_id=run_id,
                phase="after_identity_before_load", sample_sink=report["readiness_samples"])
            # Stock is complete before product startup, so weights/workers never overlap.
            guard = BudgetGuard(sleeper=sleep)
            stock_worker = {"pid": None, "returncode": None, "closed": False}
            report["stock_worker"] = stock_worker
            def record_stock(sample: dict[str, Any]) -> None:
                report["load_memory_samples"].append({"phase": "stock", **dict(sample)})
                journal.append(run_id, "validation", {"state": "stock_memory", "sample": dict(sample)})
            stock_guard = LoadMemoryGuard(swap_baseline_bytes=swap_baseline, memory_total_bytes=memory_total, on_sample=record_stock)
            def bind_stock(active_process, active_guard) -> None:
                nonlocal process, load_guard
                process, load_guard = active_process, active_guard
            def after_stock_ready(pid: int) -> None:
                stock_guard(pid, force=True)
                _wait_ready(deadline=min(deadline, time.monotonic() + DEFAULT_WAIT_READY),
                    global_deadline=deadline, check=check, sleep=sleep, journal=journal, run_id=run_id,
                    phase="after_stock_load", memory_guard=stock_guard, sample_sink=report["readiness_samples"])
            stock: dict[str, Any] = {"samples": []}
            report["stock"] = stock
            def record_request_event(row: dict[str, Any]) -> None:
                state = "stock_sample" if "sample" in row else "request_resource"
                if state == "request_resource":
                    report["request_resource_events"].append(row)
                journal.append(run_id, "validation", {"state": state, **row})
            stock = _stock_reference(spec.snapshot_path, messages, check=check, load_guard=stock_guard,
                                     worker=stock_worker, guard=guard, pause=budget_pause,
                                     after_ready=after_stock_ready, bind_process=bind_stock, result=stock,
                                     record_event=record_request_event)
            stock = _validate_stock_samples(stock)
            report["stock"] = stock
            guard.finish_candidate()
            guard.before_candidate()
            def record(sample: dict[str, Any]) -> None:
                report["load_memory_samples"].append(dict(sample)); journal.append(run_id, "validation", {"state": "memory", "sample": dict(sample)})
            load_guard = LoadMemoryGuard(swap_baseline_bytes=swap_baseline, memory_total_bytes=memory_total, on_sample=record)
            client = MLXWorkerClient(spec, startup_timeout=DEFAULT_WAIT_READY)
            worker = {"pid": None, "returncode": None, "closed": False}; report["worker"] = worker
            try:
                def startup(pid: int) -> None:
                    nonlocal process
                    process = client._process
                    if process is None or process.pid != pid: raise LongContextFailure("worker_start_handle_missing")
                    worker["pid"] = pid; check()
                ready = client.start(startup_guard=startup)
                worker["ready_observation"] = dict(ready)
                if process is None: raise LongContextFailure("worker_start_handle_missing")
                load_guard.validate_ready(ready); load_guard(process.pid, force=True)
                after, _ = _wait_ready(deadline=min(deadline, time.monotonic() + DEFAULT_WAIT_READY), global_deadline=deadline,
                    check=check, sleep=sleep, journal=journal, run_id=run_id, phase="after_load", memory_guard=load_guard, sample_sink=report["readiness_samples"])
                if after.get("swap_used_bytes", -1) - swap_baseline > SWAP_DELTA_LIMIT_BYTES: raise LongContextFailure("swap_growth")
                request = GenerationRequest(MODEL_ID, messages, max_tokens=MAX_TOKENS, temperature=0.0, top_p=1.0)
                observed: list[dict[str, Any]] = []
                report["product_samples"] = observed
                for index in range(REPEATS + 1):
                    actual, _elapsed, peak = measured_call(
                        f"product_{index}", lambda: _collect(client, request, check), journal)
                    sample = {"phase": "warmup" if index == 0 else "recorded",
                              "sample_index": index, "output_sha256": actual,
                              "peak_memory_bytes": peak,
                              "matches_reference": actual == stock["samples"][index]["output_sha256"]}
                    observed.append(sample)
                    journal.append(run_id, "validation", {"state": "product_sample", "sample": sample})
                    if peak > .60 * memory_total: raise LongContextFailure("peak_memory_limit")
                    if actual != stock["samples"][index]["output_sha256"]: raise LongContextFailure("reference_output_mismatch")
                    budget_pause()
                report["product_output_sha256"] = observed
                # The HTTP service uses the same already-loaded worker; transport calls are separately guarded.
                with tempfile.TemporaryDirectory(prefix="ironmule-prod8-") as temp:
                    transport_store = ProductStore(Path(temp)); transport_store.setup("server"); transport_store.register_model(spec)
                    service = ProductService(transport_store, backend=client, spec=spec)
                    server = create_server(service, host="127.0.0.1", port=0)
                    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
                    try:
                        payload = {"model": MODEL_ID, "messages": [{"role": r, "content": c} for r, c in messages], "max_tokens": MAX_TOKENS}
                        response = measured_call("http_json", lambda: _off_main_thread(
                            lambda: _request_json(server.server_address[1], payload), check=check,
                            load_guard=load_guard, pid=process.pid, on_abort=client.close), journal)
                        _assert_http(response, stock["samples"][-1])
                        report["transport"] = {"json_usage": response["usage"], "json_matches_reference": True}
                        journal.append(run_id, "validation", {"state": "http_json", **report["transport"]})
                        budget_pause()
                        text, finish, prompt_count, completion = measured_call("http_sse", lambda: _off_main_thread(
                            lambda: _sse(server.server_address[1], payload), check=check,
                            load_guard=load_guard, pid=process.pid, on_abort=client.close), journal)
                        expected = stock["samples"][-1]
                        if (_sha256(text.encode("utf-8")) != expected["text_sha256"] or finish != expected["finish_reason"]
                                or prompt_count != expected["prompt_tokens"] or completion != expected["completion_tokens"]):
                            raise LongContextFailure("sse_output_mismatch")
                        report["transport"].update(sse_text_sha256=_sha256(text.encode()), sse_finish_reason=finish,
                            sse_prompt_tokens=prompt_count, sse_completion_tokens=completion,
                            sse_done_count=1, sse_eof=True, sse_matches_reference=True)
                        journal.append(run_id, "validation", {"state": "http_sse", **report["transport"]})
                        budget_pause()
                    finally:
                        transport_primary = sys.exc_info()[1]
                        transport_cleanup_error: BaseException | None = None
                        try:
                            server.shutdown()
                            server.server_close()
                            thread.join(timeout=5)
                            if thread.is_alive():
                                raise LongContextFailure("http_server_cleanup_failed")
                        except BaseException as exc:
                            transport_cleanup_error = exc
                        if transport_cleanup_error is not None and transport_primary is None:
                            raise transport_cleanup_error
            finally:
                primary_error = sys.exc_info()[1]
                active = process; process = None; load_guard = None
                try:
                    client.close()
                except BaseException:
                    if primary_error is None:
                        raise
                worker["returncode"] = active.poll() if active is not None else None; worker["closed"] = active is not None and active.poll() is not None
                if (not worker["closed"] or worker["returncode"] != 0) and primary_error is None:
                    raise LongContextFailure("worker_cleanup_failed")
            report["budget"] = guard.summary()
            installed_after = _installed_package_proof((
                "ironmule_product.backend", "ironmule_product.calibration", "ironmule_product.http_server",
                "ironmule_product.model_policy", "ironmule_product.readiness", "ironmule_product.service",
                "ironmule_product.state", "ironmule_product.types", "friday_evidence.budget",
                "friday_evidence.canonical", "friday_evidence.events", "friday_evidence.identity",
            ))
            if installed_after != installed_before:
                raise LongContextFailure("installed_package_changed")
            if _provider_distribution_proof() != provider_before:
                raise LongContextFailure("provider_distribution_changed")
            report["model_metadata_after"] = _snapshot_metadata_proof(spec.snapshot_path)
            if report["model_metadata_after"] != metadata_before:
                raise LongContextFailure("model_metadata_changed")
            report["installed_package_after"] = installed_after
            report["identity_after"] = runtime_identity(spec.as_dict(), {**after, **hardware, "memory_total_bytes": memory_total})
            if report["identity_after"]["identity_sha256"] != report["identity_before"]["identity_sha256"]: raise LongContextFailure("identity_changed")
            report["source_manifest_after"] = {path.relative_to(ROOT).as_posix(): _sha256(path.read_bytes()) for path in source_paths}
            if report["source_manifest_after"] != manifest_before: raise LongContextFailure("source_changed")
            report["status"] = "passed"
            journal.append(run_id, "run_finished", {"status": "passed", "worker": worker, "stock_worker": stock_worker,
                "budget": report["budget"], "identity_sha256": report["identity_after"]["identity_sha256"],
                "source_manifest_sha256": report["source_manifest_sha256"]})
    except BaseException as exc:
        report.update(status="failed", error_code=getattr(exc, "code", type(exc).__name__.lower()), error_type=type(exc).__name__)
        if str(report["error_code"]).endswith("readiness_timeout"):
            report["status"] = "deferred"
        if guard is not None:
            report["budget"] = guard.summary()
        workers_closed = all(not isinstance(item, dict) or item.get("pid") is None or item.get("closed") is True
                             for item in (report.get("stock_worker"), report.get("worker")))
        if workers_closed:
            try:
                report["source_manifest_after"] = {
                    path.relative_to(ROOT).as_posix(): _sha256(path.read_bytes()) for path in source_paths
                }
            except OSError:
                report["source_manifest_after_error"] = "source_unavailable"
        try:
            with EventJournal(store.root / "long-context-reference.sqlite3") as journal:
                journal.append(run_id, "run_finished", {"status": report["status"], "error_code": report["error_code"],
                    "worker": report.get("worker"), "stock_worker": report.get("stock_worker"),
                    "budget": report.get("budget"), "request_resource_events": report["request_resource_events"]})
        except BaseException as journal_exc:
            report["terminal_journal_error"] = type(journal_exc).__name__
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--stock-child", action="store_true")
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.stock_child: return _stock_child()
    if not args.execute or args.state_dir is None or args.output is None: parser.error("native screen requires --execute, --state-dir, and --output")
    report = run(args.state_dir, args.output)
    from product_load_screen import _exclusive_write
    _exclusive_write(args.output, report)
    print(json.dumps({"status": report["status"], "run_id": report["run_id"]}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
