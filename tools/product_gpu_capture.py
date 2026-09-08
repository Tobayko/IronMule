"""PROD10 diagnostic-only MLX Metal capture for the registered stock 12B case.

The controller deliberately never imports MLX.  A private child makes exactly
four finite stock requests: three capture-enabled warmups and one captured
``long_8`` request.  Its pipe contains only bounded hashes and counters.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "ironmule.prod10_gpu_capture.v1"
MODEL_ID = "mlx-community/gemma-3-12b-it-4bit"
REVISION = "86cc6a8dedbc456dd0e4af01a9d09f396f77e558"
RAW_REFERENCE = ROOT / "research/raw/PROD10_12B_open_20260907_attempt2.json"
FRAME_LIMIT = 1024 * 1024


class CaptureFailure(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


# Reuse the exact manifest-bound sibling even under the worker's ``-I``.  This
# loads one named source file; it does not add the repository to ``sys.path`` or
# make source packages eligible to shadow the installed runtime.
_REFERENCE_PATH = ROOT / "tools/product_open_validation.py"
_REFERENCE_SPEC = importlib.util.spec_from_file_location(
    "_ironmule_prod10_open_validation_reference", _REFERENCE_PATH,
)
if _REFERENCE_SPEC is None or _REFERENCE_SPEC.loader is None:
    raise ImportError("product_open_validation_spec_unavailable")
_reference = importlib.util.module_from_spec(_REFERENCE_SPEC)
_REFERENCE_SPEC.loader.exec_module(_reference)
ReferenceProcess = _reference.ReferenceProcess
_emit = _reference._emit
cases = _reference.cases
digest = _reference.digest
output_metadata = _reference.output_metadata


def _manifest() -> dict[str, str]:
    paths = (Path(__file__), ROOT / "docs/PROD10_GPU_CAPTURE_SPEC.md",
             ROOT / "tools/product_open_validation.py", ROOT / "tools/product_load_screen.py",
             ROOT / "tools/product_long_context_reference.py", RAW_REFERENCE)
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def _private_trace(path: Path) -> Path:
    """Create one fresh private run directory below the sole trace root."""
    private_root = ROOT / ".friday-data"
    profiles = private_root / "profiles"
    raw = path.expanduser()
    if not raw.is_absolute():
        raw = Path.cwd() / raw
    if raw.name != "capture.gputrace" or raw.is_symlink() or raw.parent.is_symlink():
        raise CaptureFailure("trace_path_not_private")
    if private_root.is_symlink() or profiles.is_symlink():
        raise CaptureFailure("trace_path_not_private")
    profiles.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        profiles = profiles.resolve(strict=True)
        if not profiles.is_dir() or profiles.is_symlink() or raw.parent.parent.resolve(strict=True) != profiles:
            raise CaptureFailure("trace_path_not_private")
    except (OSError, RuntimeError) as exc:
        raise CaptureFailure("trace_path_not_private") from exc
    if raw.parent.exists() or raw.exists():
        raise CaptureFailure("trace_exists")
    raw.parent.mkdir(mode=0o700)
    os.chmod(raw.parent, 0o700)
    return raw.parent.resolve(strict=True) / raw.name


def _expected_stock() -> tuple[dict[str, object], dict[str, object]]:
    try:
        raw = json.loads(RAW_REFERENCE.read_text(encoding="utf-8"))
        if (not isinstance(raw, dict) or raw.get("status") != "passed"
                or raw.get("model_id") != MODEL_ID or raw.get("revision") != REVISION):
            raise CaptureFailure("stock_reference_scope_invalid")
        rows = [value for value in raw["samples"]
                if value.get("backend") == "stock" and value.get("case") == "long_8"]
        if ([value.get("repeat") for value in rows] != [0, 1, 2, 3]
                or [value.get("warmup") for value in rows] != [True, False, False, False]):
            raise CaptureFailure("stock_reference_schedule_invalid")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CaptureFailure("stock_reference_row_invalid") from exc
    keys = ("output_sha256", "token_sha256", "text_sha256", "finish_reason",
            "prompt_tokens", "completion_tokens")
    expected_rows = [{key: row.get(key) for key in keys} for row in rows]
    if any(value is None for row in expected_rows for value in row.values()):
        raise CaptureFailure("stock_reference_row_invalid")
    if any(row != expected_rows[0] for row in expected_rows[1:]):
        raise CaptureFailure("stock_reference_not_stable")
    binding_keys = ("model_sha256", "environment_sha256", "hardware_sha256", "code_sha256")
    before, after = raw.get("identity_before"), raw.get("identity_after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise CaptureFailure("stock_reference_binding_invalid")
    bindings = {key: before.get(key) for key in binding_keys}
    if any(not isinstance(value, str) or len(value) != 64 for value in bindings.values()):
        raise CaptureFailure("stock_reference_binding_invalid")
    if any(after.get(key) != value for key, value in bindings.items()):
        raise CaptureFailure("stock_reference_binding_changed")
    return expected_rows[0], bindings


def _sample(model, tokenizer, mx, stream_generate, expected: dict[str, object], *, capture: Path | None):
    case = cases()[0]
    prompt = tokenizer.apply_chat_template(case["messages"], tokenize=True, add_generation_prompt=True)
    if not 1000 <= len(prompt) <= 1100:
        raise CaptureFailure("long_context_out_of_range")
    tokens, text, final = [], [], None
    started = time.monotonic()
    started_capture = False
    stream = None
    primary = close_error = stop_error = None
    try:
        if capture is not None:
            mx.metal.start_capture(str(capture))
            started_capture = True
        stream = stream_generate(model, tokenizer, prompt, max_tokens=case["max_tokens"])
        for response in stream:
            tokens.append(response.token)
            text.append(response.text)
            final = response
    except BaseException as exc:
        primary = exc
    try:
        if stream is not None:
            stream.close()
    except BaseException as exc:
        close_error = exc
    finally:
        try:
            if started_capture:
                mx.metal.stop_capture()
        except BaseException as exc:
            stop_error = exc
    failure = primary or close_error or stop_error
    if failure is not None:
        cleanup = []
        if close_error is not None and close_error is not failure:
            cleanup.append("stream_close:" + getattr(close_error, "code", type(close_error).__name__))
        if stop_error is not None and stop_error is not failure:
            cleanup.append("capture_stop:" + getattr(stop_error, "code", type(stop_error).__name__))
        if cleanup:
            setattr(failure, "cleanup_errors", cleanup)
        raise failure
    if final is None:
        raise CaptureFailure("stock_no_output")
    sample = output_metadata(tokens, "".join(text), {"finish_reason": final.finish_reason,
                             "prompt_tokens": final.prompt_tokens,
                             "completion_tokens": final.generation_tokens})
    if any(sample[key] != expected[key] for key in expected):
        raise CaptureFailure("stock_reference_mismatch")
    return {**sample, "wall_seconds": time.monotonic() - started,
            "mlx_peak_bytes": int(mx.get_peak_memory())}


def worker(state_dir: Path, trace: Path) -> int:
    """Child entry point.  Validation happens before the sole MLX import."""
    os.umask(0o077)
    try:
        from ironmule_product.model_policy import validate_model_config
        from ironmule_product.state import ProductStore
        spec = ProductStore(state_dir).model(MODEL_ID)
        if spec.revision != REVISION:
            raise CaptureFailure("snapshot_not_frozen")
        config = validate_model_config(spec.snapshot_path, spec.revision)
        expected, _bindings = _expected_stock()
        with redirect_stdout(sys.stderr):
            import mlx.core as mx
            from mlx_lm import load, stream_generate
            if not mx.metal.is_available():
                raise CaptureFailure("metal_unavailable")
            mx.set_default_device(mx.gpu)
            model, tokenizer = load(spec.snapshot_path, model_config=config,
                                    tokenizer_config={"trust_remote_code": False}, revision=REVISION)
        _emit({"type": "ready", "pid": os.getpid(), "device": str(mx.default_device()),
               "mlx_active_bytes": int(mx.get_active_memory()), "mlx_peak_bytes": int(mx.get_peak_memory())})
        for index in range(3):
            sample = _sample(model, tokenizer, mx, stream_generate, expected, capture=None)
            _emit({"type": "sample", "index": index, "captured": False, "sample": sample})
        sample = _sample(model, tokenizer, mx, stream_generate, expected, capture=trace)
        _emit({"type": "sample", "index": 3, "captured": True, "sample": sample})
        _emit({"type": "finished", "pid": os.getpid(), "mlx_peak_bytes": int(mx.get_peak_memory())})
        # Parent owns the terminal close and only inspects the artifact afterward.
        command = json.loads(sys.stdin.buffer.readline(FRAME_LIMIT))
        if command != {"type": "shutdown"}:
            raise CaptureFailure("shutdown_command_invalid")
        return 0
    except BaseException as exc:
        cleanup_errors = getattr(exc, "cleanup_errors", None)
        if cleanup_errors:
            _emit({"type": "cleanup", "errors": cleanup_errors})
        _emit({"type": "error", "code": getattr(exc, "code", type(exc).__name__)})
        return 1


class CaptureProcess(ReferenceProcess):
    """ReferenceProcess framing/robust close with an argument-driven child."""
    def __init__(self, state_dir: Path, trace: Path, observer):
        import subprocess
        self.observer = observer
        self.buffer = bytearray()
        env = {**os.environ, "MTL_CAPTURE_ENABLED": "1"}
        self.process = subprocess.Popen([sys.executable, "-I", "-u", str(Path(__file__).resolve()),
                                         "--worker", "--state-dir", str(state_dir), "--trace", str(trace)],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env)
        self.observer.bind(self.process, "stock_capture")

    # The proven bounded framing and terminate/wait/kill cleanup are inherited
    # unchanged from ReferenceProcess.


def _trace_metadata(trace: Path) -> dict[str, object]:
    result: dict[str, object] = {"basename": trace.name, "exists": trace.exists(), "size_bytes": 0}
    try:
        root_stat = os.lstat(trace)
        regular_inodes: set[tuple[int, int]] = set()
        allocated_inodes: set[tuple[int, int]] = set()
        regular_file_count = 0
        symlink_count = 0

        def visit(path: Path, entry_stat: os.stat_result) -> None:
            nonlocal regular_file_count, symlink_count
            inode = (entry_stat.st_dev, entry_stat.st_ino)
            if stat.S_ISLNK(entry_stat.st_mode):
                symlink_count += 1
            if inode not in allocated_inodes:
                allocated_inodes.add(inode)
                result["allocated_bytes"] = result.get("allocated_bytes", 0) + entry_stat.st_blocks * 512
            if stat.S_ISREG(entry_stat.st_mode):
                if inode not in regular_inodes:
                    regular_inodes.add(inode)
                    regular_file_count += 1
                    result["size_bytes"] += entry_stat.st_size
            elif stat.S_ISDIR(entry_stat.st_mode):
                with os.scandir(path) as entries:
                    for entry in entries:
                        visit(Path(entry.path), entry.stat(follow_symlinks=False))

        visit(trace, root_stat)
        result["regular_file_count"] = regular_file_count
        result["symlink_count"] = symlink_count
        result.setdefault("allocated_bytes", 0)
    except OSError as exc:
        result["inspection_error"] = type(exc).__name__
    return result


def run(state_dir: Path, output: Path, trace: Path) -> dict[str, object]:
    from friday_evidence.canonical import canonical_sha256
    from friday_evidence.events import EventJournal
    from friday_evidence.identity import runtime_identity
    from friday_evidence.open_observation import OpenObservation
    from ironmule_product.calibration import model_lease
    from ironmule_product.readiness import hardware_identity, probe
    from product_load_screen import _exclusive_write, _installed_package_proof
    from product_long_context_reference import _provider_distribution_proof, _snapshot_metadata_proof
    if output.exists() or output.is_symlink(): raise CaptureFailure("output_exists")
    trace = _private_trace(trace); run_id = uuid.uuid4().hex
    store = __import__("ironmule_product.state", fromlist=["ProductStore"]).ProductStore(state_dir)
    spec = store.model(MODEL_ID)
    if spec.revision != REVISION: raise CaptureFailure("snapshot_not_frozen")
    # Installed origins prove the runtime path; this source file is bound by
    # the separate source manifest and must not be mistaken for an installed
    # package module.
    used = ("friday_evidence.canonical", "friday_evidence.events", "friday_evidence.identity",
            "friday_evidence.open_observation", "friday_evidence.process_memory",
            "ironmule_product.calibration", "ironmule_product.model_policy",
            "ironmule_product.readiness", "ironmule_product.state")
    expected, reference_bindings = _expected_stock()
    report: dict[str, object] = {"schema": SCHEMA, "run_id": run_id, "status": "started", "model_id": MODEL_ID,
        "revision": REVISION, "diagnostic_only": True, "performance_claim": False, "activation_allowed": False,
        "capture": {"basename": trace.name, "captured_request_index": 3}, "samples": [],
        "source_before": _manifest(), "reference_bindings": reference_bindings,
        "reference_sha256": hashlib.sha256(RAW_REFERENCE.read_bytes()).hexdigest(), "host_observations": []}
    child = None
    observer = None
    terminal_recorded = False
    try:
        with EventJournal(state_dir / "gpu-capture.sqlite3") as journal, model_lease(store):
            try:
                journal.append(run_id, "run_started", {"model_id": MODEL_ID, "revision": REVISION,
                    "source_sha256": canonical_sha256(report["source_before"]), "reference_sha256": report["reference_sha256"]})
                host = {**probe(), **hardware_identity()}; report["host_observations"].append(host)
                journal.append(run_id, "validation", {"state": "host_observation", "observation": host})
                report["identity_before"] = runtime_identity(spec.as_dict(), host)
                if any(report["identity_before"].get(key) != value for key, value in reference_bindings.items()):
                    raise CaptureFailure("stock_reference_binding_mismatch")
                report["provider_before"] = _provider_distribution_proof()
                report["metadata_before"] = _snapshot_metadata_proof(spec.snapshot_path)
                report["installed_before"] = _installed_package_proof(used)
                observer = OpenObservation(on_sample=lambda row: journal.append(run_id, "validation",
                    {"state": "resource_observation", "observation": row}), interval_seconds=1.0)
                child = CaptureProcess(state_dir, trace, observer)
                ready_seen = False
                while True:
                    event = child.read(); kind = event.get("type")
                    if kind == "ready":
                        if (ready_seen or report["samples"] or event.get("pid") != child.process.pid
                                or "gpu" not in str(event.get("device", "")).lower()
                                or type(event.get("mlx_active_bytes")) is not int or event["mlx_active_bytes"] <= 0
                                or type(event.get("mlx_peak_bytes")) is not int or event["mlx_peak_bytes"] <= 0):
                            raise CaptureFailure("worker_not_gpu_ready")
                        ready_seen = True; report["worker_ready"] = event
                    elif kind == "sample":
                        index = len(report["samples"]); sample = event.get("sample")
                        if (not ready_seen or index >= 4 or event.get("index") != index
                                or event.get("captured") is not (index == 3) or not isinstance(sample, dict)
                                or any(sample.get(key) != value for key, value in expected.items())
                                or not isinstance(sample.get("wall_seconds"), (int, float))
                                or isinstance(sample.get("wall_seconds"), bool)
                                or not math.isfinite(sample["wall_seconds"]) or sample["wall_seconds"] <= 0
                                or type(sample.get("mlx_peak_bytes")) is not int or sample["mlx_peak_bytes"] <= 0):
                            raise CaptureFailure("sample_schedule_or_reference_invalid")
                        report["samples"].append(event)
                        journal.append(run_id, "sample", {"index": index, "captured": event["captured"], "sample": sample})
                    elif kind == "finished":
                        if (not ready_seen or len(report["samples"]) != 4 or event.get("pid") != child.process.pid
                                or type(event.get("mlx_peak_bytes")) is not int or event["mlx_peak_bytes"] <= 0):
                            raise CaptureFailure("worker_finished_invalid")
                        report["worker_finished"] = event; break
                    elif kind == "cleanup" and isinstance(event.get("errors"), list):
                        report.setdefault("worker_cleanup_errors", []).extend(event["errors"])
                    else: raise CaptureFailure("worker_protocol_invalid")
            except BaseException as exc:
                report.update(status="failed", error_code=getattr(exc, "code", type(exc).__name__))
            finally:
                if child is not None:
                    buffer = bytes(child.buffer); process = child.process
                    report["worker_partial_frame"] = {"bytes": len(buffer), "sha256": hashlib.sha256(buffer).hexdigest()}
                    try: child.close()
                    except BaseException as exc:
                        report.setdefault("cleanup_errors", []).append(getattr(exc, "code", type(exc).__name__))
                        if report["status"] == "started": report.update(status="failed", error_code=getattr(exc, "code", type(exc).__name__))
                    finally:
                        report["worker"] = {"pid": process.pid, "returncode": process.poll(), "closed": process.poll() is not None}
                        child = None
                if observer is not None: report["observation_summary"] = observer.summary()
                if isinstance(report.get("worker"), dict) and not report["worker"].get("closed"):
                    report["capture"] = {**report["capture"], "inspection_error": "worker_not_reaped"}
                else: report["capture"] = {**report["capture"], **_trace_metadata(trace)}
                for name, call in (("source_after", _manifest), ("provider_after", _provider_distribution_proof),
                                   ("metadata_after", lambda: _snapshot_metadata_proof(spec.snapshot_path)),
                                   ("installed_after", lambda: _installed_package_proof(used))):
                    try: report[name] = call()
                    except BaseException as exc: report.setdefault("after_errors", {})[name] = type(exc).__name__
                try:
                    host = {**probe(), **hardware_identity()}; report["host_observations"].append(host)
                    journal.append(run_id, "validation", {"state": "host_observation", "observation": host})
                    report["identity_after"] = runtime_identity(spec.as_dict(), host)
                except BaseException as exc: report.setdefault("after_errors", {})["identity_after"] = type(exc).__name__
                if report["status"] == "started":
                    if not report["capture"].get("exists") or report["capture"].get("size_bytes", 0) <= 0:
                        report.update(status="failed", error_code="capture_missing_or_empty")
                    elif any(report.get(key + "_before") != report.get(key + "_after")
                             for key in ("identity", "provider", "metadata", "installed", "source")):
                        report.update(status="failed", error_code="evidence_changed")
                    else: report["status"] = "passed"
                journal.append(run_id, "run_finished", {"status": report["status"],
                    "error_code": report.get("error_code"), "samples": len(report["samples"]),
                    "report_sha256": digest(report)})
                terminal_recorded = True
    except BaseException as exc:
        if not terminal_recorded:
            report.update(status="failed", error_code=getattr(exc, "code", type(exc).__name__))
    finally:
        if child is not None and not terminal_recorded:
            process = child.process
            try: child.close()
            except BaseException as exc: report.setdefault("cleanup_errors", []).append(getattr(exc, "code", type(exc).__name__))
            finally: report["worker"] = {"pid": process.pid, "returncode": process.poll(), "closed": process.poll() is not None}
        if observer is not None and not terminal_recorded: report.setdefault("observation_summary", observer.summary())
        if not terminal_recorded:
            if isinstance(report.get("worker"), dict) and not report["worker"].get("closed"):
                report["capture"] = {**report["capture"], "inspection_error": "worker_not_reaped"}
            else:
                report["capture"] = {**report["capture"], **_trace_metadata(trace)}
        if report["status"] != "passed" and not terminal_recorded:
            for name, call in (("source_after", _manifest), ("provider_after", _provider_distribution_proof),
                               ("metadata_after", lambda: _snapshot_metadata_proof(spec.snapshot_path)),
                               ("installed_after", lambda: _installed_package_proof(used))):
                try: report[name] = call()
                except BaseException as exc: report.setdefault("after_errors", {})[name] = type(exc).__name__
            try:
                host = {**probe(), **hardware_identity()}; report["host_observations"].append(host)
                report["identity_after"] = runtime_identity(spec.as_dict(), host)
            except BaseException as exc: report.setdefault("after_errors", {})["identity_after"] = type(exc).__name__
        _exclusive_write(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--worker", action="store_true"); parser.add_argument("--execute", action="store_true")
    parser.add_argument("--state-dir", type=Path); parser.add_argument("--output", type=Path); parser.add_argument("--trace", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.state_dir is None or args.trace is None: parser.error("--worker requires --state-dir and --trace")
        return worker(args.state_dir, args.trace)
    if not args.execute or None in (args.state_dir, args.output, args.trace): parser.error("--execute, --state-dir, --output and --trace are required")
    result = run(args.state_dir, args.output, args.trace)
    print(json.dumps({"status": result["status"], "run_id": result["run_id"], "error_code": result.get("error_code")}))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__": raise SystemExit(main())
