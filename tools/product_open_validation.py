"""PROD10: real installed-model validation without artificial hardware gates.

No model is imported in the controller. The reference child uses unmodified
MLX-LM and a finite, explicit request list. Native execution is opt-in. Timing,
memory and load are observations, not reasons to throttle or reject a request.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import hashlib
import http.client
import json
import os
from pathlib import Path
import selectors
import socket
import struct
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "ironmule.prod10_open_validation.v1"
FRAME_LIMIT = 1024 * 1024
MODELS = {
    "mlx-community/gemma-3-1b-it-4bit": "2d44e83dc9e80843d22fb941d3d699a0b1351aa6",
    "mlx-community/gemma-3-4b-it-4bit": "93724907d4ed1745d2fe50baadf3b0b01a65abf2",
    "mlx-community/gemma-3-12b-it-4bit": "86cc6a8dedbc456dd0e4af01a9d09f396f77e558",
}
POLICY = {"request_timeout_s": None, "startup_timeout_s": None,
          "work_budget_s": None, "continuous_budget_s": None,
          "duty_cycle_limit": None, "required_pause_s": 0,
          "rss_stop_bytes": None, "swap_stop_bytes": None,
          "host_readiness_gate": False, "automatic_retry": False}


class ValidationFailure(RuntimeError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def cases():
    long = "Public orchard note: apples grow on trees and need sunlight. " * 88
    long += "END-OF-PUBLIC-ORCHARD-NOTE."
    return [
        {"name": "long_8", "messages": [{"role": "user", "content": long}], "max_tokens": 8},
        {"name": "short_32", "messages": [{"role": "user", "content": "Write a short sentence about apples."}], "max_tokens": 32},
        {"name": "long_32", "messages": [{"role": "user", "content": long}], "max_tokens": 32},
    ]


def output_metadata(tokens, text, done):
    count = done["completion_tokens"]
    if type(count) is not int or count != len(tokens) or not tokens:
        raise ValidationFailure("output_count_invalid")
    if done["finish_reason"] not in ("stop", "length"):
        raise ValidationFailure("output_not_complete")
    content = {"tokens": tokens, "text": text, "finish_reason": done["finish_reason"],
               "prompt_tokens": done["prompt_tokens"], "completion_tokens": count}
    return {"output_sha256": digest(content), "token_sha256": digest(tokens),
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "prompt_tokens": content["prompt_tokens"], "completion_tokens": count,
            "finish_reason": content["finish_reason"]}


def _emit(value):
    encoded = json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    if len(encoded) >= FRAME_LIMIT:
        raise ValidationFailure("frame_too_large")
    sys.stdout.buffer.write(encoded + b"\n")
    sys.stdout.buffer.flush()


def reference_child():
    """Independent stock worker; all prompt/output material stays in memory."""
    try:
        initial = json.loads(sys.stdin.buffer.readline(FRAME_LIMIT))
        spec = initial["spec"]
        if MODELS.get(spec["model_id"]) != spec["revision"]:
            raise ValidationFailure("snapshot_not_frozen")
        from ironmule_product.model_policy import validate_model_config
        config = validate_model_config(spec["snapshot_path"], spec["revision"])
        started = time.monotonic()
        with redirect_stdout(sys.stderr):
            import mlx.core as mx
            from mlx_lm import load, stream_generate
            if not mx.metal.is_available():
                raise ValidationFailure("metal_unavailable")
            mx.set_default_device(mx.gpu)
            model, tokenizer = load(spec["snapshot_path"], model_config=config,
                                    tokenizer_config={"trust_remote_code": False})
        _emit({"type": "ready", "load_wall_seconds": time.monotonic() - started,
               "device": str(mx.default_device()), "mlx_active_bytes": mx.get_active_memory(),
               "mlx_peak_bytes": mx.get_peak_memory()})
        while True:
            line = sys.stdin.buffer.readline(FRAME_LIMIT)
            if not line:
                return 0
            command = json.loads(line)
            if command == {"type": "shutdown"}:
                return 0
            if command.get("type") != "generate" or command.get("case_index") not in range(len(cases())):
                raise ValidationFailure("command_invalid")
            case = cases()[command["case_index"]]
            tokenizing = time.monotonic()
            prompt = tokenizer.apply_chat_template(case["messages"], tokenize=True, add_generation_prompt=True)
            tokenize_seconds = time.monotonic() - tokenizing
            if case["name"].startswith("long") and not 1000 <= len(prompt) <= 1100:
                raise ValidationFailure("long_context_out_of_range")
            phases, tokens, pieces = [], [], []
            started = time.monotonic()
            first = None
            def progress(processed, total):
                # Installed stock generate_step invokes this after mx.eval(cache)
                # and after the first-token eval. These are wall boundaries, not
                # hardware timestamp counters or newly inserted GPU barriers.
                phases.append({"processed": int(processed), "total": int(total),
                               "elapsed_seconds": time.monotonic() - started})
            stream = stream_generate(model, tokenizer, prompt, max_tokens=case["max_tokens"],
                                     prompt_progress_callback=progress)
            final = None
            try:
                with redirect_stdout(sys.stderr):
                    for response in stream:
                        if first is None:
                            first = time.monotonic() - started
                        tokens.append(response.token)
                        pieces.append(response.text)
                        final = response
            finally:
                stream.close()
            if final is None:
                raise ValidationFailure("reference_no_output")
            done = {"finish_reason": final.finish_reason, "prompt_tokens": final.prompt_tokens,
                    "completion_tokens": final.generation_tokens}
            sample = output_metadata(tokens, "".join(pieces), done)
            sample.update(wall_seconds=time.monotonic() - started, first_token_seconds=first,
                          tokenize_seconds=tokenize_seconds, prompt_progress=phases,
                          prompt_ids_sha256=digest(prompt), peak_memory_bytes=mx.get_peak_memory(),
                          prompt_tps=final.prompt_tps, generation_tps=final.generation_tps)
            _emit({"type": "result", "sample": sample})
    except BaseException as exc:
        _emit({"type": "error", "code": getattr(exc, "code", type(exc).__name__)})
        return 1


class ReferenceProcess:
    def __init__(self, spec, observer):
        self.observer = observer
        self.buffer = bytearray()
        self.process = subprocess.Popen([sys.executable, "-I", "-u", str(Path(__file__).resolve()), "--reference-child"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        observer.bind(self.process, "stock")
        try:
            self.send({"spec": spec.as_dict()})
        except BaseException:
            self.process.terminate()
            self.process.wait(timeout=5)
            observer.bind(None, "stock_failed_to_start")
            raise

    def send(self, value):
        frame = json.dumps(value, separators=(",", ":")).encode() + b"\n"
        if len(frame) >= FRAME_LIMIT:
            raise ValidationFailure("frame_too_large")
        self.process.stdin.write(frame)
        self.process.stdin.flush()

    def read(self):
        fd = self.process.stdout.fileno()
        os.set_blocking(fd, False)
        with selectors.DefaultSelector() as selector:
            selector.register(fd, selectors.EVENT_READ)
            while True:
                self.observer.sample()
                if b"\n" in self.buffer:
                    line, rest = self.buffer.split(b"\n", 1)
                    self.buffer = bytearray(rest)
                    event = json.loads(line)
                    if event.get("type") == "error":
                        raise ValidationFailure(event["code"])
                    return event
                if not selector.select(0.1):
                    continue
                chunk = os.read(fd, 65536)
                if not chunk:
                    raise ValidationFailure("reference_exited_before_result")
                self.buffer.extend(chunk)
                if len(self.buffer) > FRAME_LIMIT:
                    raise ValidationFailure("reference_frame_too_large")

    def close(self):
        active_error = sys.exc_info()[1]
        try:
            if self.process.poll() is None:
                try:
                    self.send({"type": "shutdown"})
                    self.process.stdin.close()
                    self.process.wait(timeout=5)
                except (BrokenPipeError, OSError, ValueError, subprocess.TimeoutExpired):
                    pass
                finally:
                    if self.process.poll() is None:
                        self.process.terminate()
                        try:
                            self.process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            self.process.kill()
                            self.process.wait(timeout=5)
        finally:
            self.observer.bind(None, "stock_closed")
        row = {"pid": self.process.pid, "returncode": self.process.poll(),
               "closed": self.process.poll() is not None}
        if active_error is None and row["returncode"] != 0:
            raise ValidationFailure("reference_shutdown_failed")
        return row


def observed_call(call, observer):
    # Only this owner thread observes resources and appends to SQLite.
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(call)
    try:
        while not future.done():
            observer.sample()
            time.sleep(0.05)
        return future.result()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def product_sample(client, spec, case, partial=None):
    from ironmule_product.types import GenerationRequest
    request = GenerationRequest(spec.model_id, tuple((m["role"], m["content"]) for m in case["messages"]),
                                max_tokens=case["max_tokens"], temperature=0.0, top_p=1.0)
    tokens, pieces, done, first = [], [], None, None
    started = time.monotonic()
    stream = client.stream(request, timeout=None)
    try:
        for event in stream:
            if event["type"] == "token":
                if first is None:
                    first = time.monotonic() - started
                tokens.append(event["token_id"])
                pieces.append(event["text"])
                if partial is not None:
                    partial.update(token_count=len(tokens), token_sha256=digest(tokens),
                        text_sha256=hashlib.sha256("".join(pieces).encode()).hexdigest(),
                        elapsed_seconds=time.monotonic() - started,
                        prompt_tokens=event.get("prompt_tokens"), completed=False)
            elif event["type"] == "done":
                done = event
    finally:
        stream.close()
    if done is None:
        raise ValidationFailure("product_not_complete")
    result = output_metadata(tokens, "".join(pieces), done)
    result.update(wall_seconds=time.monotonic() - started, first_token_seconds=first,
                  metrics=done.get("metrics", {}))
    return result


def http_sample(port, model, case, *, streaming=False):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=None)
    payload = {"model": model, "messages": case["messages"], "max_tokens": case["max_tokens"],
               "temperature": 0.0, "top_p": 1.0, "stream": streaming}
    started = time.monotonic()
    try:
        connection.request("POST", "/v1/chat/completions", json.dumps(payload), {"Content-Type": "application/json"})
        response = connection.getresponse()
        raw = response.read(FRAME_LIMIT + 1)
        if len(raw) > FRAME_LIMIT or response.status != 200:
            raise ValidationFailure(f"http_status_{response.status}" if response.status != 200 else "http_oversize")
        if not streaming:
            value = json.loads(raw)
            choice, usage = value["choices"][0], value["usage"]
            text, finish = choice["message"]["content"], choice["finish_reason"]
        else:
            if response.getheader("Connection", "").lower() != "close":
                raise ValidationFailure("sse_not_finite")
            lines = [line for line in raw.decode().splitlines() if line]
            if not lines or lines[-1] != "data: [DONE]" or lines.count("data: [DONE]") != 1:
                raise ValidationFailure("sse_done_invalid")
            values = [json.loads(line[6:]) for line in lines[:-1] if line.startswith("data: ")]
            if any("error" in value for value in values):
                raise ValidationFailure("sse_error")
            finished = [value for value in values if value["choices"][0]["finish_reason"] is not None]
            if len(finished) != 1:
                raise ValidationFailure("sse_finish_invalid")
            text = "".join(value["choices"][0]["delta"].get("content", "") for value in values)
            usage, finish = finished[0]["usage"], finished[0]["choices"][0]["finish_reason"]
        if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
            raise ValidationFailure("http_usage_invalid")
        return {"text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "prompt_tokens": usage["prompt_tokens"], "completion_tokens": usage["completion_tokens"],
                "finish_reason": finish, "wall_seconds": time.monotonic() - started,
                "streaming": streaming, "sse_done_count": 1 if streaming else None}
    finally:
        connection.close()


def assert_http(actual, expected):
    if any(actual[key] != expected[key] for key in ("text_sha256", "prompt_tokens", "completion_tokens", "finish_reason")):
        raise ValidationFailure("http_reference_mismatch")


def health(port):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=None)
    try:
        connection.request("GET", "/health")
        response = connection.getresponse()
        raw = response.read(FRAME_LIMIT)
        if response.status != 200:
            raise ValidationFailure("health_failed")
        return json.loads(raw)
    finally:
        connection.close()


def disconnect_request(port, model):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=None)
    payload = {"model": model, "messages": [{"role": "user", "content": "Count from 1 to 500, writing every integer in order, separated by commas. Do not skip any number."}],
               "max_tokens": 512, "temperature": 0, "stream": True}
    response = None
    try:
        connection.request("POST", "/v1/chat/completions", json.dumps(payload), {"Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            raise ValidationFailure("disconnect_request_rejected")
        while True:
            line = response.readline(FRAME_LIMIT)
            if not line or line.strip() == b"data: [DONE]":
                raise ValidationFailure("disconnect_no_content")
            if line.startswith(b"data: "):
                value = json.loads(line[6:])
                if value.get("choices", [{}])[0].get("delta", {}).get("content"):
                    # The finite SSE response may transfer ownership of the socket
                    # from HTTPConnection to HTTPResponse; close both references.
                    sock = connection.sock or response.fp.raw._sock
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                    sock.shutdown(socket.SHUT_RDWR)
                    return {"disconnected_after_content": True}
    finally:
        if response is not None:
            response.close()
        connection.close()


def run(state_dir, model_id, output, *, soak_seconds=0):
    from friday_evidence.canonical import canonical_sha256
    from friday_evidence.events import EventJournal
    from friday_evidence.identity import runtime_identity
    from friday_evidence.open_observation import OpenObservation
    from ironmule_product.backend import MLXWorkerClient
    from ironmule_product.calibration import model_lease
    from ironmule_product.http_server import create_server
    from ironmule_product.readiness import hardware_identity, probe
    from ironmule_product.service import ProductService
    from ironmule_product.state import ProductStore
    from product_load_screen import _exclusive_write, _installed_package_proof
    from product_long_context_reference import _provider_distribution_proof, _snapshot_metadata_proof

    if output.exists() or output.is_symlink():
        raise ValidationFailure("output_exists")
    store = ProductStore(state_dir)
    spec = store.model(model_id)
    if MODELS.get(model_id) != spec.revision:
        raise ValidationFailure("snapshot_not_frozen")
    run_id = uuid.uuid4().hex
    source_paths = [Path(__file__), ROOT / "docs/PROD10_OPEN_VALIDATION_SPEC.md",
                    ROOT / "tools/product_long_context_reference.py", ROOT / "tools/product_load_screen.py"]
    manifest = lambda: {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    report = {"schema": SCHEMA, "run_id": run_id, "model_id": model_id, "revision": spec.revision,
              "status": "started", "policy": POLICY, "source_before": manifest(),
              "soak_seconds_requested": soak_seconds, "performance_claim": False, "activation_allowed": False,
              "samples": [], "workers": [], "host_observations": []}
    client = reference = server = server_thread = process = None
    observer = None
    try:
        with EventJournal(state_dir / "open-validation.sqlite3") as journal, model_lease(store):
            journal.append(run_id, "run_started", {"model_id": model_id, "revision": spec.revision,
                "policy": POLICY, "source_sha256": canonical_sha256(report["source_before"]),
                "soak_seconds": soak_seconds})
            used = ("friday_evidence.open_observation", "friday_evidence.process_memory", "ironmule_product.service",
                    "ironmule_product.state", "ironmule_product.http_server", "friday_evidence.canonical")
            report["installed_before"] = _installed_package_proof(used)
            report["provider_before"] = _provider_distribution_proof()
            report["metadata_before"] = _snapshot_metadata_proof(spec.snapshot_path)
            host = {**probe(), **hardware_identity()}
            report["host_observations"].append(host)
            report["identity_before"] = runtime_identity(spec.as_dict(), host)
            host = {**probe(), **hardware_identity()}
            report["host_observations"].append(host)
            journal.append(run_id, "validation", {"state": "host_observation", "observation": host})
            observer = OpenObservation(on_sample=lambda row: journal.append(run_id, "validation", {"state": "resource_observation", "observation": row}))
            def save(row):
                report["samples"].append(row)
                journal.append(run_id, "validation", {"state": "request_sample", "sample": row})
            baseline = {}
            reference = ReferenceProcess(spec, observer)
            try:
                ready = reference.read()
                report["reference_ready"] = ready
                if ready.get("type") != "ready" or "gpu" not in ready.get("device", ""):
                    raise ValidationFailure("reference_not_gpu_ready")
                for case_index, case in enumerate(cases()):
                    for repeat in range(4):
                        started = time.monotonic()
                        journal.append(run_id, "validation", {"state": "request_started", "backend": "stock",
                            "case": case["name"], "repeat": repeat})
                        reference.send({"type": "generate", "case_index": case_index})
                        try:
                            event = reference.read()
                        except BaseException as exc:
                            save({"backend": "stock", "case": case["name"], "repeat": repeat,
                                "status": "failed", "partial_output": "not_returned_by_reference",
                                "controller_wall_seconds": time.monotonic() - started,
                                "error_code": getattr(exc, "code", type(exc).__name__)})
                            raise
                        if event.get("type") != "result":
                            raise ValidationFailure("reference_protocol_invalid")
                        row = {"backend": "stock", "case": case["name"], "repeat": repeat,
                               "warmup": repeat == 0, **event["sample"], "controller_wall_seconds": time.monotonic() - started}
                        save(row)
                        if case["name"] in baseline and baseline[case["name"]]["output_sha256"] != row["output_sha256"]:
                            raise ValidationFailure("reference_not_repeatable")
                        baseline[case["name"]] = row
            finally:
                try:
                    reference.close()
                finally:
                    report["workers"].append({"backend": "stock", "pid": reference.process.pid,
                                              "returncode": reference.process.poll(), "closed": reference.process.poll() is not None})
                reference = None
            client = MLXWorkerClient(spec, startup_timeout=None)
            def startup(pid):
                nonlocal process
                process = client._process
                observer.bind(process, "product")
                observer.sample()
            report["product_ready"] = client.start(startup_guard=startup)
            for case in cases():
                for repeat in range(4):
                    partial = {"token_count": 0, "completed": False}
                    journal.append(run_id, "validation", {"state": "request_started", "backend": "product",
                        "case": case["name"], "repeat": repeat})
                    try:
                        sample = observed_call(lambda: product_sample(client, spec, case, partial), observer)
                    except BaseException as exc:
                        save({"backend": "product", "case": case["name"], "repeat": repeat,
                            "status": "failed", "partial": dict(partial),
                            "error_code": getattr(exc, "code", type(exc).__name__)})
                        raise
                    sample.update(backend="product", case=case["name"], repeat=repeat, warmup=repeat == 0)
                    sample["matches_reference"] = sample["output_sha256"] == baseline[case["name"]]["output_sha256"]
                    save(sample)
                    if not sample["matches_reference"]:
                        raise ValidationFailure("product_reference_mismatch")
            # Isolated mutable service settings; the user's persistent configuration
            # and the frozen state of prior studies are not changed by a test.
            import tempfile
            with tempfile.TemporaryDirectory(prefix="ironmule-prod10-service-") as tmp:
                transport = ProductStore(Path(tmp))
                transport.setup("server")
                transport.set_request_timeout(None)
                transport.register_model(spec)
                service = ProductService(transport, backend=client, spec=spec)
                server = create_server(service, port=0)
                server_thread = threading.Thread(target=server.serve_forever, daemon=True)
                server_thread.start()
                port = server.server_address[1]
                for case in cases():
                    for streaming in (False, True):
                        sample = observed_call(lambda: http_sample(port, model_id, case, streaming=streaming), observer)
                        sample.update(backend="http", case=case["name"])
                        save(sample)
                        assert_http(sample, baseline[case["name"]])
                with ThreadPoolExecutor(max_workers=4) as pool:
                    futures = [pool.submit(http_sample, port, model_id, cases()[index % 3]) for index in range(4)]
                    while not all(f.done() for f in futures):
                        observer.sample()
                        time.sleep(0.05)
                    for index, future in enumerate(futures):
                        sample = future.result()
                        sample.update(backend="http_concurrent", case=cases()[index % 3]["name"], slot=index)
                        save(sample)
                        assert_http(sample, baseline[sample["case"]])
                before_cancel = health(port)
                report["disconnect"] = observed_call(lambda: disconnect_request(port, model_id), observer)
                while True:
                    observer.sample()
                    current = health(port)
                    if current["active_requests"] == 0 and current["queued_requests"] == 0:
                        break
                    time.sleep(0.05)
                report["health_after_disconnect"] = current
                if current["cancelled_requests"] <= before_cancel["cancelled_requests"]:
                    raise ValidationFailure("disconnect_cancellation_not_observed")
                sample = observed_call(lambda: http_sample(port, model_id, cases()[1]), observer)
                save({"backend": "recovery", "case": "short_32", **sample})
                assert_http(sample, baseline["short_32"])
                soak_started = time.monotonic()
                index = 0
                while time.monotonic() - soak_started < soak_seconds:
                    case = cases()[index % 3]
                    sample = observed_call(lambda: http_sample(port, model_id, case, streaming=index % 2 == 1), observer)
                    save({"backend": "soak", "case": case["name"], "index": index, **sample})
                    assert_http(sample, baseline[case["name"]])
                    index += 1
                report["soak"] = {"wall_seconds": time.monotonic() - soak_started, "requests": index}
                report["health_final"] = health(port)
                server.shutdown()
                server.server_close()
                server_thread.join(timeout=5)
                server = None
            client.close()
            report["workers"].append({"backend": "product", "pid": process.pid, "returncode": process.poll(), "closed": process.poll() is not None})
            if process.poll() != 0:
                raise ValidationFailure("product_shutdown_failed")
            observer.bind(None, "complete")
            report["observation_summary"] = observer.summary()
            if report["observation_summary"]["errors"]:
                raise ValidationFailure("resource_observation_incomplete")
            report["identity_after"] = runtime_identity(spec.as_dict(), host)
            report["provider_after"] = _provider_distribution_proof()
            report["metadata_after"] = _snapshot_metadata_proof(spec.snapshot_path)
            report["installed_after"] = _installed_package_proof(used)
            report["source_after"] = manifest()
            for key in ("identity", "provider", "metadata", "installed", "source"):
                if report[f"{key}_before"] != report[f"{key}_after"]:
                    raise ValidationFailure(f"{key}_changed")
            report["status"] = "passed"
            journal.append(run_id, "run_finished", {"status": "passed", "samples": len(report["samples"]),
                "report_sha256": digest(report), "workers": report["workers"], "soak": report["soak"]})
    except BaseException as exc:
        report.update(status="failed", error_code=getattr(exc, "code", type(exc).__name__), error_type=type(exc).__name__)
    finally:
        primary = report.get("error_code")
        for action in ((lambda: server.shutdown()) if server else None,
                       (lambda: server.server_close()) if server else None,
                       (lambda: client.close()) if client else None):
            if action is not None:
                try:
                    action()
                except BaseException as exc:
                    report.setdefault("cleanup_errors", []).append(type(exc).__name__)
        if reference is not None:
            try:
                reference.close()
            except BaseException as exc:
                report.setdefault("cleanup_errors", []).append(type(exc).__name__)
        if process is not None and not any(row["pid"] == process.pid for row in report["workers"]):
            report["workers"].append({"backend": "product", "pid": process.pid, "returncode": process.poll(), "closed": process.poll() is not None})
        if observer is not None:
            report["observation_summary"] = observer.summary()
        if report.get("cleanup_errors"):
            report.update(status="failed", error_code=primary or "cleanup_failed")
        if report["status"] != "passed":
            report["source_after"] = manifest()
            with EventJournal(state_dir / "open-validation.sqlite3") as journal:
                journal.append(run_id, "run_finished", {"status": report["status"], "error_code": report.get("error_code"),
                    "samples": len(report["samples"]), "workers": report["workers"]})
        _exclusive_write(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--reference-child", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--soak-seconds", type=int, choices=(0, 3600), default=0)
    args = parser.parse_args()
    if args.reference_child:
        return reference_child()
    if not args.execute or None in (args.state_dir, args.model, args.output):
        parser.error("--execute, --state-dir, --model and --output are required")
    result = run(args.state_dir, args.model, args.output, soak_seconds=args.soak_seconds)
    print(json.dumps({"status": result["status"], "run_id": result["run_id"], "error_code": result.get("error_code"), "samples": len(result["samples"])}))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
