"""PROD12: observe the real HTTP server and its model worker separately.

Native execution is opt-in.  This controller never imports MLX and never hosts
the product service.  Its isolated server child owns ProductStore,
ProductService, the HTTP server, and the model-worker lifecycle.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "tools/product_open_validation.py"
LOAD_HELPER = ROOT / "tools/product_load_screen.py"
PROOF_HELPER = ROOT / "tools/product_long_context_reference.py"
SPEC_PATH = ROOT / "docs/PROD12_SERVER_MEMORY_SPEC.md"
REFERENCE_PATH = ROOT / "research/raw/PROD10_12B_open_20260907_attempt2.json"
SCHEMA = "ironmule.prod12.server-memory.v1"
MODEL_ID = "mlx-community/gemma-3-12b-it-4bit"
REVISION = "86cc6a8dedbc456dd0e4af01a9d09f396f77e558"
FRAME_LIMIT = 1024 * 1024
_EMIT_LOCK = threading.Lock()
POLICY = {
    "request_timeout_s": None, "startup_timeout_s": None,
    "generation_limit": None, "work_budget_s": None,
    "rss_stop_bytes": None, "swap_stop_bytes": None,
    "duty_cycle_limit": None, "required_pause_s": 0,
    "automatic_retry": False,
}
REFERENCE = {
    "long_8": {"prompt_tokens": 1077, "completion_tokens": 8, "finish_reason": "length",
               "output_sha256": "923ba7f75ef72975774057063f36d4917c0fce06e8a8e36ce7bb5e8ccac60922",
               "token_sha256": "0a9d453c03cb2da6b9546a710d889c87d40a3b4094c7607d3ff5299fbf37f49f",
               "text_sha256": "4967edda1afffdeede4b0cd4991a82d91a8e1ee9ab1c5d457df317866a8ca1c2"},
    "short_32": {"prompt_tokens": 16, "completion_tokens": 13, "finish_reason": "stop",
                 "output_sha256": "147e10117bf25ba41b2b2af9825bbff1f1fe40c68af0d5b90f699c98fe17aa1c",
                 "token_sha256": "a378c42360f6070c3350d4ebb4580482318f9fe61cdadef7856a75f3d9c1131d",
                 "text_sha256": "6e3740d57c904144108c70cb3731dde63e9a5fea27e168be8ace06928feb804c"},
    "long_32": {"prompt_tokens": 1077, "completion_tokens": 30, "finish_reason": "stop",
                "output_sha256": "fdb6fdccaf14cff45f70125865c9b82989d763dbe40a9e0c080997cb9d9b9462",
                "token_sha256": "f27a69fc8a6b92a8a2eb9ba5ff5dcd18872b4ea792d7b55c94bdf812c36ee467",
                "text_sha256": "b5f984a7822a4929a7a8431fd094edc7eae78f5ba723566f1ec072db547ce8c7"},
}


class ValidationFailure(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_module(path: Path, expected_sha256: str, name: str):
    """Load a manifest-bound helper by absolute path (safe under ``-I``)."""
    if sha256_file(path) != expected_sha256:
        raise ValidationFailure("helper_changed")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValidationFailure("helper_import_failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_helper(expected_sha256: str):
    return _load_module(HELPER, expected_sha256, "prod12_open_helper")


def request_plan() -> list[dict]:
    """Finite plan: three cases x four (first warmup), then one 4-client burst."""
    rows = [
        {"case_index": case_index, "case": name, "repeat": repeat,
         "warmup": repeat == 0, "batch": f"single-{case_index}-{repeat}"}
        for case_index, name in enumerate(("long_8", "short_32", "long_32"))
        for repeat in range(4)
    ]
    rows.extend({"case_index": index % 3, "case": ("long_8", "short_32", "long_32")[index % 3],
                 "repeat": None, "warmup": False, "batch": "burst-4", "slot": index}
                for index in range(4))
    return rows


def _emit(value: dict) -> None:
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    if len(raw) >= FRAME_LIMIT:
        raise ValidationFailure("frame_too_large")
    with _EMIT_LOCK:
        sys.stdout.buffer.write(raw + b"\n")
        sys.stdout.buffer.flush()


def server_child() -> int:
    """Own the complete product server stack and exactly one model worker."""
    server = client = thread = None
    process = observer = sample_stop = sample_thread = None
    observation_summary = {"sample_count": 0, "memory_sample_count": 0,
        "max_memory_bytes": {key: None for key in ("rss_bytes", "physical_footprint_bytes",
                                                     "peak_footprint_bytes", "wired_bytes")},
        "errors": {}}
    try:
        initial = json.loads(sys.stdin.buffer.readline(FRAME_LIMIT))
        _load_helper(initial["helper_sha256"])
        load_helper = _load_module(LOAD_HELPER, initial["load_helper_sha256"], "prod12_child_load")
        proof_helper = _load_module(PROOF_HELPER, initial["proof_helper_sha256"], "prod12_child_proof")
        from friday_evidence.identity import runtime_identity
        from friday_evidence.open_observation import OpenObservation
        from ironmule_product.backend import MLXWorkerClient
        from ironmule_product.calibration import model_lease
        from ironmule_product.http_server import create_server
        from ironmule_product.service import ProductService
        from ironmule_product.state import ProductStore
        from ironmule_product.readiness import hardware_identity, probe

        persistent = ProductStore(Path(initial["state_dir"]))
        spec = persistent.model(MODEL_ID)
        if spec.revision != REVISION:
            raise ValidationFailure("snapshot_not_frozen")
        lease = model_lease(persistent)
        lease.__enter__()
        try:
            with tempfile.TemporaryDirectory(prefix="ironmule-prod12-server-") as tmp:
                store = ProductStore(Path(tmp))
                store.setup("server")
                store.set_request_timeout(None)
                store.register_model(spec)
                client = MLXWorkerClient(spec, startup_timeout=None)
                ready = client.start()
                process = client._process
                if process is None or process.poll() is not None:
                    raise ValidationFailure("model_worker_not_running")
                def emit_observation(row):
                    observation_summary["sample_count"] += 1
                    if type(row["rss_bytes"]) is int: observation_summary["memory_sample_count"] += 1
                    for key in observation_summary["max_memory_bytes"]:
                        value = row[key]
                        if type(value) is int:
                            previous = observation_summary["max_memory_bytes"][key]
                            observation_summary["max_memory_bytes"][key] = value if previous is None else max(previous, value)
                    for code in row["errors"]:
                        observation_summary["errors"][code] = observation_summary["errors"].get(code, 0) + 1
                    _emit({"type": "model_observation", "observation": row})
                observer = OpenObservation(emit_observation, interval_seconds=1.0)
                observer.bind(process, "model_worker")
                sample_stop = threading.Event()
                def sampling():
                    while not sample_stop.wait(0.1):
                        observer.sample()
                        observer.rows.clear()
                sample_thread = threading.Thread(target=sampling, daemon=True)
                observer.sample(force=True); sample_thread.start()
                service = ProductService(store, backend=client, spec=spec)
                server = create_server(service, host="127.0.0.1", port=0)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                used = ("friday_evidence.identity", "friday_evidence.open_observation",
                    "friday_evidence.process_memory", "ironmule_product.backend",
                    "ironmule_product.calibration", "ironmule_product.http_server",
                    "ironmule_product.model_policy", "ironmule_product.service",
                    "ironmule_product.state", "ironmule_product.types")
                host = {**probe(), **hardware_identity()}
                bindings_before = {"identity": runtime_identity(spec.as_dict(), host),
                    "installed": load_helper._installed_package_proof(used),
                    "provider": proof_helper._provider_distribution_proof(),
                    "metadata": proof_helper._snapshot_metadata_proof(spec.snapshot_path)}
                _emit({"type": "ready", "port": server.server_address[1],
                       "server_pid": os.getpid(), "model_pid": process.pid,
                       "model_ready": ready, "model_id": spec.model_id,
                       "revision": spec.revision, "weight_bytes": spec.weight_bytes,
                       "bindings_before": bindings_before,
                       "helper_sha256": sha256_file(HELPER)})
                command = json.loads(sys.stdin.buffer.readline(FRAME_LIMIT))
                if command != {"type": "shutdown"}:
                    raise ValidationFailure("command_invalid")
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
                if thread.is_alive():
                    raise ValidationFailure("server_thread_not_closed")
                server = None
                sample_stop.set(); sample_thread.join(timeout=5)
                client.close()
                client = None
                final_host = {**probe(), **hardware_identity()}
                bindings_after = {"identity": runtime_identity(spec.as_dict(), final_host),
                    "installed": load_helper._installed_package_proof(used),
                    "provider": proof_helper._provider_distribution_proof(),
                    "metadata": proof_helper._snapshot_metadata_proof(spec.snapshot_path)}
                _emit({"type": "closed", "server_pid": os.getpid(),
                       "model_pid": process.pid, "model_returncode": process.poll(),
                       "model_observation_summary": observation_summary,
                       "bindings_after": bindings_after})
        finally:
            if sample_stop is not None:
                sample_stop.set()
            if sample_thread is not None:
                sample_thread.join(timeout=5)
            if server is not None:
                try: server.shutdown(); server.server_close()
                except BaseException: pass
                server = None
            if client is not None:
                try: client.close()
                except BaseException: pass
                client = None
            lease.__exit__(*sys.exc_info())
        return 0
    except BaseException as exc:
        try:
            _emit({"type": "error", "code": getattr(exc, "code", type(exc).__name__),
                   "model_pid": process.pid if process is not None else None,
                   "model_returncode": process.poll() if process is not None else None,
                   "model_observation_summary": observation_summary if observer is not None else None})
        except BaseException:
            pass
        return 1
    finally:
        if server is not None:
            try: server.shutdown(); server.server_close()
            except BaseException: pass
        if client is not None:
            try: client.close()
            except BaseException: pass


class ServerProcess:
    def __init__(self, state_dir: Path, helper_sha256: str):
        self.process = subprocess.Popen(
            [sys.executable, "-I", "-u", str(Path(__file__).resolve()), "--server-child"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self.buffer = bytearray()
        self.last_event = None
        self.pending = []
        self.on_event = None
        self.send({"state_dir": str(state_dir), "helper_sha256": helper_sha256,
                   "load_helper_sha256": sha256_file(LOAD_HELPER),
                   "proof_helper_sha256": sha256_file(PROOF_HELPER)})

    def send(self, value: dict) -> None:
        raw = json.dumps(value, separators=(",", ":")).encode() + b"\n"
        if len(raw) >= FRAME_LIMIT or self.process.stdin is None:
            raise ValidationFailure("frame_too_large")
        self.process.stdin.write(raw); self.process.stdin.flush()

    def read(self, observers=(), deadline: float | None = None, on_event=None) -> dict:
        if self.process.stdout is None:
            raise ValidationFailure("server_protocol_closed")
        fd = self.process.stdout.fileno(); os.set_blocking(fd, False)
        on_event = on_event or self.on_event
        with selectors.DefaultSelector() as selector:
            selector.register(fd, selectors.EVENT_READ)
            while True:
                if self.pending:
                    return self.pending.pop(0)
                if deadline is not None and time.monotonic() >= deadline:
                    raise ValidationFailure("server_protocol_timeout")
                for observer in observers: observer.sample()
                if b"\n" in self.buffer:
                    line, rest = self.buffer.split(b"\n", 1); self.buffer = bytearray(rest)
                    value = json.loads(line)
                    self.last_event = value
                    if value.get("type") == "model_observation" and on_event is not None:
                        on_event(value["observation"])
                        continue
                    if value.get("type") == "error": raise ValidationFailure(value["code"])
                    return value
                wait = 0.05 if deadline is None else min(0.05, max(0.0, deadline - time.monotonic()))
                if selector.select(wait):
                    chunk = os.read(fd, 65536)
                    if not chunk: raise ValidationFailure("server_exited_before_protocol")
                    self.buffer.extend(chunk)
                    if len(self.buffer) > FRAME_LIMIT: raise ValidationFailure("server_frame_too_large")

    def drain(self, on_event) -> None:
        """Drain only already-available frames; never wait opportunistically."""
        if self.process.stdout is None: return
        fd = self.process.stdout.fileno(); os.set_blocking(fd, False)
        while True:
            try: chunk = os.read(fd, 65536)
            except BlockingIOError: break
            if not chunk: break
            self.buffer.extend(chunk)
            if len(self.buffer) > FRAME_LIMIT: raise ValidationFailure("server_frame_too_large")
        while b"\n" in self.buffer:
            line, rest = self.buffer.split(b"\n", 1); self.buffer = bytearray(rest)
            value = json.loads(line); self.last_event = value
            if value.get("type") == "model_observation": on_event(value["observation"])
            else: self.pending.append(value)

    def close(self) -> dict:
        if self.process.poll() is None:
            try: self.send({"type": "shutdown"})
            except (BrokenPipeError, OSError, ValueError): pass
            try: closed = self.read(deadline=time.monotonic() + 10, on_event=self.on_event)
            except BaseException: closed = None
            try: self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if self.process.stdin is not None:
                    try: self.process.stdin.close()
                    except OSError: pass
                try: self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGTERM)
                    try: self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(self.process.pid, signal.SIGKILL)
                        self.process.wait(timeout=5)
        else: closed = None
        return {"pid": self.process.pid, "returncode": self.process.poll(),
                "closed": self.process.poll() is not None, "protocol": closed or self.last_event}


def _assert_http(actual: dict, expected: dict) -> None:
    keys = ("text_sha256", "prompt_tokens", "completion_tokens", "finish_reason")
    if any(actual.get(key) != expected[key] for key in keys):
        raise ValidationFailure("http_reference_mismatch")


def _observe_call(call, observers, drain=None):
    done = threading.Event()
    result, failure = [], []
    def target():
        try: result.append(call())
        except BaseException as exc: failure.append(exc)
        finally: done.set()
    threading.Thread(target=target, daemon=True).start()
    while not done.wait(0.05):
        for observer in observers: observer.sample()
        if drain is not None: drain()
    if failure: raise failure[0]
    return result[0]


def _reference_manifest(helper) -> dict:
    raw = json.loads(REFERENCE_PATH.read_text())
    if raw.get("status") != "passed" or raw.get("model_id") != MODEL_ID or raw.get("revision") != REVISION:
        raise ValidationFailure("reference_artifact_invalid")
    for name, expected in REFERENCE.items():
        rows = [row for row in raw["samples"] if row.get("backend") == "stock" and row.get("case") == name]
        if len(rows) != 4 or any(any(row[key] != expected[key] for key in expected) for row in rows):
            raise ValidationFailure("reference_artifact_changed")
    return {"path": REFERENCE_PATH.relative_to(ROOT).as_posix(), "sha256": sha256_file(REFERENCE_PATH),
            "cases": REFERENCE, "case_definition_sha256": helper.digest(helper.cases())}


def run(state_dir: Path, output: Path) -> dict:
    from friday_evidence.canonical import canonical_sha256
    from friday_evidence.events import EventJournal
    from friday_evidence.open_observation import OpenObservation
    from ironmule_product.state import ProductStore
    helper_sha = sha256_file(HELPER)
    helper = _load_helper(helper_sha)
    load_helper = _load_module(LOAD_HELPER, sha256_file(LOAD_HELPER), "prod12_load_helper")
    if output.exists() or output.is_symlink(): raise ValidationFailure("output_exists")
    run_id = uuid.uuid4().hex
    sources = (Path(__file__), SPEC_PATH, HELPER, LOAD_HELPER, PROOF_HELPER, REFERENCE_PATH)
    manifest = lambda: {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in sources}
    report = {"schema": SCHEMA, "run_id": run_id, "status": "started", "model_id": MODEL_ID,
              "revision": REVISION, "policy": POLICY, "performance_claim": False,
              "activation_allowed": False, "source_before": manifest(), "samples": [],
              "processes": [], "observations": {"server": [], "model_worker": []}}
    server = None
    ready = None
    journal = None
    try:
        journal = EventJournal(state_dir / "server-memory.sqlite3")
        # Keep the owner journal open through failure cleanup: closing the
        # server may still deliver real model-resource frames to its callback.
        with nullcontext(journal):
            report["reference"] = _reference_manifest(helper)
            parent_spec = ProductStore(state_dir).model(MODEL_ID)
            if parent_spec.revision != REVISION: raise ValidationFailure("snapshot_not_frozen")
            def record_model(row):
                report["observations"]["model_worker"].append(row)
                journal.append(run_id, "validation", {"state": "resource_observation", "observation": row})
            server = ServerProcess(state_dir, helper_sha)
            server.on_event = record_model
            def record_server(row):
                report["observations"]["server"].append(row)
                journal.append(run_id, "validation", {"state": "resource_observation", "observation": row})
            server_observer = OpenObservation(record_server, interval_seconds=1.0)
            server_observer.bind(server.process, "http_server")
            ready = server.read((server_observer,), on_event=record_model)
            if ready.get("type") != "ready" or ready["server_pid"] != server.process.pid:
                raise ValidationFailure("server_ready_invalid")
            if (ready["model_id"], ready["revision"], ready["weight_bytes"]) != (parent_spec.model_id, parent_spec.revision, parent_spec.weight_bytes):
                raise ValidationFailure("server_model_binding_mismatch")
            model_ready = ready.get("model_ready", {})
            if (model_ready.get("device") != "gpu"
                    or type(model_ready.get("mlx_active_bytes")) is not int
                    or model_ready["mlx_active_bytes"] <= 0
                    or type(model_ready.get("mlx_peak_bytes")) is not int
                    or model_ready["mlx_peak_bytes"] <= 0):
                raise ValidationFailure("model_ready_invalid")
            observers = (server_observer,)
            for key, value in ready["bindings_before"].items(): report[f"{key}_before"] = value
            journal.append(run_id, "run_started", {"model_id": MODEL_ID, "plan_sha256": canonical_sha256(request_plan()),
                           "server_pid": ready["server_pid"], "model_pid": ready["model_pid"]})
            cases = helper.cases()
            singles = request_plan()[:12]
            for row in singles:
                sample = _observe_call(lambda r=row: helper.http_sample(ready["port"], MODEL_ID, cases[r["case_index"]]),
                                       observers, lambda: server.drain(record_model))
                _assert_http(sample, REFERENCE[row["case"]]); sample.update(row)
                report["samples"].append(sample)
                journal.append(run_id, "validation", {"state": "request_sample", "sample": sample})
            burst = request_plan()[12:]
            pool = ThreadPoolExecutor(max_workers=4)
            try:
                futures = [pool.submit(helper.http_sample, ready["port"], MODEL_ID, cases[row["case_index"]]) for row in burst]
                while not all(f.done() for f in futures):
                    for observer in observers: observer.sample()
                    server.drain(record_model)
                    time.sleep(0.05)
                first_error = None
                for row, future in zip(burst, futures):
                    try:
                        sample = future.result(); _assert_http(sample, REFERENCE[row["case"]]); sample.update(row)
                    except BaseException as exc:
                        sample = {**row, "status": "failed", "error_code": getattr(exc, "code", type(exc).__name__)}
                        first_error = first_error or exc
                    report["samples"].append(sample); journal.append(run_id, "validation", {"state": "request_sample", "sample": sample})
                if first_error is not None: raise first_error
            finally:
                pool.shutdown(wait=False, cancel_futures=True)
            report["health_final"] = helper.idle_health(ready["port"], server_observer)
            health = report["health_final"]
            if (not health["ready"] or health["completed_requests"] != 16 or health["failed_requests"] != 0
                    or health["active_requests"] != 0 or health["queued_requests"] != 0):
                raise ValidationFailure("server_final_counters_invalid")
            closed = server.close(); server = None
            protocol = closed.get("protocol") or {}
            report["processes"] = [{"role": "http_server", **closed},
                {"role": "model_worker", "pid": ready["model_pid"],
                 "returncode": protocol.get("model_returncode"),
                 "closed": protocol.get("model_returncode") is not None}]
            report["source_after"] = manifest()
            for key, value in protocol.get("bindings_after", {}).items(): report[f"{key}_after"] = value
            for key in ("identity", "installed", "provider", "metadata", "source"):
                if report.get(f"{key}_before") != report.get(f"{key}_after"): raise ValidationFailure(f"{key}_changed")
            if closed["returncode"] != 0 or not all(row["closed"] and row["returncode"] == 0 for row in report["processes"]):
                raise ValidationFailure("process_cleanup_failed")
            report["observation_summary"] = {"server": server_observer.summary(),
                                             "model_worker": protocol.get("model_observation_summary", {})}
            if (ready["server_pid"] == ready["model_pid"]
                    or any(row.get("pid") != ready["model_pid"]
                           for row in report["observations"]["model_worker"])
                    or any(report["observation_summary"][role]["memory_sample_count"] < 1
                           for role in ("server", "model_worker"))
                    or any(report["observation_summary"][role].get("errors")
                           for role in ("server", "model_worker"))):
                raise ValidationFailure("resource_observation_incomplete")
            report["status"] = "passed"
            journal.append(run_id, "run_finished", {"status": "passed", "samples": 16,
                "processes": report["processes"], "report_sha256": helper.digest(report)})
    except BaseException as exc:
        report.update(status="failed", error_code=getattr(exc, "code", type(exc).__name__), error_type=type(exc).__name__)
    finally:
        if server is not None:
            closed = server.close()
            report["processes"].append({"role": "http_server", **closed})
            if ready is not None:
                protocol = closed.get("protocol") or {}
                if protocol.get("model_observation_summary") is not None:
                    report.setdefault("observation_summary", {})["model_worker"] = protocol["model_observation_summary"]
                for key, value in protocol.get("bindings_after", {}).items():
                    report[f"{key}_after"] = value
                report["processes"].append({"role": "model_worker", "pid": ready["model_pid"],
                    "returncode": protocol.get("model_returncode"),
                    "closed": protocol.get("model_returncode") is not None})
        report.setdefault("source_after", manifest())
        if report["status"] != "passed":
            try:
                if journal is not None:
                    journal.append(run_id, "run_finished", {"status": "failed",
                        "error_code": report.get("error_code"), "samples": len(report["samples"]),
                        "processes": report["processes"], "report_sha256": helper.digest(report)})
            except BaseException as exc:
                report.setdefault("cleanup_errors", []).append(type(exc).__name__)
        if journal is not None:
            journal.close()
        load_helper._exclusive_write(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--server-child", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.server_child: return server_child()
    if not args.execute or args.state_dir is None or args.output is None:
        parser.error("--execute, --state-dir and --output are required")
    result = run(args.state_dir, args.output)
    print(json.dumps({"status": result["status"], "run_id": result["run_id"],
                      "error_code": result.get("error_code"), "samples": len(result["samples"])}))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
