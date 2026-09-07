"""Run the registered real-Gemma product correctness screen (explicit --execute)."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import http.client
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from _bench import harness_preconditions

PROMPT = "Write a short sentence about apples."
REPEATS = 3
MAX_TOKENS = 8


def digest() -> str:
    result = hashlib.sha256()
    paths = [*sorted((ROOT / "ironmule_product").glob("*.py")),
             *sorted((ROOT / "friday_evidence").glob("*.py")), Path(__file__),
             ROOT / "tools/_bench.py", ROOT / "ironmule_inventory.py",
             ROOT / "docs/PROD1_GEMMA_SMOKE_2026-09-07.md"]
    for path in paths:
        result.update(str(path.relative_to(ROOT)).encode())
        result.update(path.read_bytes())
    return result.hexdigest()


def reference(snapshot: str) -> int:
    guard = harness_preconditions()
    started = time.monotonic()
    with redirect_stdout(sys.stderr):
        import mlx.core as mx
        hardware = dict(mx.device_info())
        mx.set_default_device(mx.gpu)
        from mlx_lm import load, stream_generate
        model, tokenizer = load(snapshot, tokenizer_config={"trust_remote_code": False})
        ids = tokenizer.apply_chat_template([{"role": "user", "content": PROMPT}],
                                             tokenize=True, add_generation_prompt=True)
    report = {"load_wall_seconds": time.monotonic() - started, "samples": [], "hardware": hardware}
    guard.required_break()
    for index in range(REPEATS + 1):
        started = time.monotonic()
        with redirect_stdout(sys.stderr):
            responses = list(stream_generate(model, tokenizer, list(ids), max_tokens=MAX_TOKENS))
        elapsed = time.monotonic() - started
        guard.record_gpu(elapsed)
        report["samples"].append({"warmup": index == 0, "tokens": [r.token for r in responses],
                                  "text": "".join(r.text for r in responses),
                                  "finish_reason": responses[-1].finish_reason,
                                  "prompt_tokens": responses[-1].prompt_tokens,
                                  "completion_tokens": responses[-1].generation_tokens,
                                  "wall_seconds": elapsed})
        guard.required_break()
    report["guard"] = guard.summary()
    print(json.dumps(report, allow_nan=False))
    return 0


def request_json(port: int, payload: dict) -> dict:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    try:
        connection.request("POST", "/v1/chat/completions", json.dumps(payload), {"Content-Type": "application/json"})
        response = connection.getresponse()
        raw = response.read()
        if response.status != 200:
            raise RuntimeError(f"HTTP generation returned {response.status}: {raw[:256]!r}")
        return json.loads(raw)
    finally:
        connection.close()


def run(output: Path) -> int:
    from ironmule_product.backend import MLXWorkerClient
    from ironmule_product.errors import InvalidRequest
    from ironmule_product.cli import inventory_rows
    from ironmule_product.http_server import create_server
    from ironmule_product.service import ProductService
    from ironmule_product.state import ProductStore
    from ironmule_product.types import GenerationRequest, ModelSpec

    if output.exists():
        raise ValueError("report exists; use a new attempt filename")
    guard = harness_preconditions()
    rows = [row for row in inventory_rows() if row["family"].lower().startswith("gemma") and row["status"] == "available"]
    models = {(row["model_id"], row["revision"]): ModelSpec.from_dict(row) for row in rows}
    if not models:
        raise RuntimeError("no local Gemma snapshot; screen cannot pass")
    report = {"schema": "ironmule.product_gemma_smoke.v1", "models": [],
              "source_sha256_before": digest(), "performance_claim": False,
              "activation_allowed": False, "status": "running",
              "versions": {name: importlib.metadata.version(name) for name in ("mlx", "mlx-lm", "numpy")},
              "os": platform.platform(), "python": platform.python_version(),
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())}

    def persist():
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")

    persist()
    try:
        for spec in sorted(models.values(), key=lambda spec: spec.weight_bytes):
            guard.before_candidate()
            result = {"model_id": spec.model_id, "revision": spec.revision, "status": "started"}
            report["models"].append(result)
            persist()
            print(f"testing {spec.model_id}", flush=True)
            process = subprocess.run([sys.executable, str(Path(__file__)), "--reference", spec.snapshot_path],
                                     cwd=ROOT, text=True, capture_output=True, timeout=240)
            if process.returncode != 0:
                result.update(status="failed", stage="reference", returncode=process.returncode,
                              diagnostic=process.stderr[-1200:])
                persist()
                raise RuntimeError("stock reference failed")
            baseline = json.loads(process.stdout)
            result["reference"] = baseline
            backend = MLXWorkerClient(spec)
            started = time.monotonic()
            try:
                backend.start()
                result["product_load_wall_seconds"] = time.monotonic() - started
                guard.required_break()
                product = []
                for index in range(REPEATS + 1):
                    request = GenerationRequest(spec.model_id, (("user", PROMPT),), max_tokens=MAX_TOKENS)
                    started = time.monotonic()
                    events = list(backend.stream(request))
                    elapsed = time.monotonic() - started
                    guard.record_gpu(elapsed)
                    tokens = [event for event in events if event["type"] == "token"]
                    done = events[-1]
                    sample = {"warmup": index == 0, "tokens": [t["token_id"] for t in tokens],
                              "text": "".join(t["text"] for t in tokens),
                              "finish_reason": done["finish_reason"], "prompt_tokens": done["prompt_tokens"],
                              "completion_tokens": done["completion_tokens"], "wall_seconds": elapsed}
                    for key in ("tokens", "text", "finish_reason", "prompt_tokens", "completion_tokens"):
                        if sample[key] != baseline["samples"][index][key]:
                            raise RuntimeError(f"reference mismatch in {key}")
                    product.append(sample)
                    guard.required_break()
                result["product"] = product
                # Test an actual mid-generation cancel, then reuse the same
                # loaded worker. No synthesized model responses are involved.
                cancel = threading.Event()
                request = GenerationRequest(spec.model_id, (("user", PROMPT),), max_tokens=128)
                started = time.monotonic()
                stream = backend.stream(request, cancel=cancel)
                first = next(stream)
                if first["type"] != "token":
                    raise RuntimeError("cancellation screen received no first token")
                cancel.set()
                remaining = list(stream)
                elapsed = time.monotonic() - started
                guard.record_gpu(elapsed)
                if not remaining or remaining[-1].get("finish_reason") != "cancelled" or not backend.ready:
                    raise RuntimeError("mid-generation cancellation did not preserve the worker")
                result["cancellation"] = {"wall_seconds": elapsed,
                                          "completion_tokens": remaining[-1]["completion_tokens"],
                                          "warm_worker_preserved": True}
                guard.required_break()
                # A real tokenized context overflow must reject before model
                # forward and must not poison the next valid request.
                try:
                    list(backend.stream(GenerationRequest(spec.model_id, (("user", PROMPT),), max_tokens=8192)))
                except InvalidRequest:
                    result["context_overflow"] = {"rejected": True, "warm_worker_preserved": backend.ready}
                else:
                    raise RuntimeError("context overflow was not rejected")
                if not backend.ready:
                    raise RuntimeError("context rejection retired the warm worker")
                started = time.monotonic()
                recovery = list(backend.stream(GenerationRequest(spec.model_id, (("user", PROMPT),), max_tokens=MAX_TOKENS)))
                guard.record_gpu(time.monotonic() - started)
                if ([e["token_id"] for e in recovery if e["type"] == "token"] != baseline["samples"][-1]["tokens"]
                        or "".join(e["text"] for e in recovery if e["type"] == "token") != baseline["samples"][-1]["text"]):
                    raise RuntimeError("recovery request differs from reference")
                result["recovery"] = {"exact": True}
                guard.required_break()
                with tempfile.TemporaryDirectory(prefix="ironmule-live-") as state_dir:
                    store = ProductStore(Path(state_dir))
                    store.setup("server")
                    store.register_model(spec)
                    service = ProductService(store, backend=backend, spec=spec)
                    server = create_server(service, host="127.0.0.1", port=0)
                    thread = threading.Thread(target=server.serve_forever, daemon=True)
                    thread.start()
                    port = server.server_address[1]
                    payload = {"model": spec.model_id, "messages": [{"role": "user", "content": PROMPT}], "max_tokens": MAX_TOKENS}
                    try:
                        started = time.monotonic()
                        completion = request_json(port, payload)
                        guard.record_gpu(time.monotonic() - started)
                        if completion["choices"][0]["message"]["content"] != baseline["samples"][-1]["text"]:
                            raise RuntimeError("HTTP text differs from reference")
                        guard.required_break()
                        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
                        started = time.monotonic()
                        try:
                            connection.request("POST", "/v1/chat/completions", json.dumps({**payload, "stream": True}), {"Content-Type": "application/json"})
                            response = connection.getresponse()
                            if response.getheader("Connection", "").lower() != "close":
                                raise RuntimeError("finite SSE response must close its HTTP body")
                            wire = response.read().decode("utf-8")
                            if response.status != 200 or "data: [DONE]" not in wire:
                                raise RuntimeError("SSE failed or did not terminate")
                            chunks = [json.loads(line[6:]) for line in wire.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]
                            if any("error" in chunk for chunk in chunks):
                                raise RuntimeError("SSE emitted an error")
                            text = "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks)
                            if text != baseline["samples"][-1]["text"]:
                                raise RuntimeError("SSE text differs from reference")
                            result["sse"] = {"text": text, "done_markers": wire.count("data: [DONE]"),
                                             "wall_seconds": time.monotonic() - started}
                        finally:
                            connection.close()
                        guard.record_gpu(time.monotonic() - started)
                        guard.required_break()
                        stop = baseline["samples"][-1]["text"][:3]
                        if stop:
                            started = time.monotonic()
                            stopped = request_json(port, {**payload, "stop": [stop]})
                            guard.record_gpu(time.monotonic() - started)
                            if stopped["choices"][0]["message"]["content"] != "" or stopped["choices"][0]["finish_reason"] != "stop":
                                raise RuntimeError("stop filtering leaked output or finish reason")
                            result["stop"] = {"matched": True}
                            guard.required_break()
                        result["http"] = completion
                        result["service_health"] = service.health()
                    finally:
                        server.shutdown()
                        server.server_close()
                        thread.join(timeout=5)
                result["status"] = "passed"
            except Exception as exc:
                result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
                raise
            finally:
                backend.close()
                persist()
            guard.finish_candidate()
            print(f"passed {spec.model_id}", flush=True)
        report["source_sha256_after"] = digest()
        if report["source_sha256_after"] != report["source_sha256_before"]:
            raise RuntimeError("source changed during hardware screen")
        report["guard"] = guard.summary()
        report["status"] = "passed"
        return 0
    except BaseException as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        if report["models"] and report["models"][-1]["status"] == "started":
            report["models"][-1].update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        persist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--reference", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.reference:
        raise SystemExit(reference(args.reference))
    if not args.execute or args.output is None:
        parser.error("real hardware screen requires --execute and a new --output path")
    raise SystemExit(run(args.output))
