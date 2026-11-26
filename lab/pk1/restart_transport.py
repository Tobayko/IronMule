"""Capture one immutable v3 incumbent HTTP arm for diagnosis without making release or timing decisions."""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import threading
import time
import traceback
from contextlib import contextmanager
from unittest.mock import patch

from lab.pk1 import restart_gate_broker, restart_http as startup
from training import scoped_release as release
from training.code_author import write_once
from training.code_trial import read, save
from training.trial_release import require


class Trace:
    """Observe the exact comparison client and ordinary service with bounded payloads."""

    def __init__(self, folder):
        self.folder = folder
        self.requests = []
        self.phases = []
        self.server_errors = []
        self.closed = False
        self.lock = threading.RLock()
        self.thread = threading.local()

    def flush(self):
        with self.lock:
            if not self.closed:
                save(self.folder / "requests.json", {"phases": self.phases, "requests": self.requests, "server_errors":self.server_errors})

    def observe(self):
        compare = startup.compare
        batch, stream, readline = compare.batch, compare.stream, http.client.HTTPResponse.readline

        def group(engine, server, prompts, width, limit, deadline, *args, **kw):
            with self.lock:
                number = len(self.phases)
                name, expected = (("cold", 18), ("warm", 18), ("performance", 32))[number]
                require(width == expected and len(prompts) == expected and limit == 32, "foreign HTTP workload")
                self.phases.append({"name": name, "requested": width, "returned": False})
                self.flush()
            result = batch(engine, server, prompts, width, limit, deadline, *args, **kw)
            with self.lock:
                self.phases[number]["returned"] = True
                self.flush()
            return result

        def request(engine, server, prompt, limit, deadline, **kw):
            row = {"phase": self.phases[-1]["name"], "prompt_sha256": hashlib.sha256(
                json.dumps(prompt, ensure_ascii=False).encode()).hexdigest(), "chunks": 0, "text_prefix": ""}
            with self.lock:
                self.requests.append(row)
            self.thread.row = row
            try:
                result = stream(engine, server, prompt, limit, deadline, **kw)
                with self.lock:
                    row["completion"] = {k:result[k] for k in ("text", "tokens", "prompt_tokens", "finish_reason")}
                return result
            except Exception as exc:
                with self.lock:
                    row["client_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
                raise
            finally:
                self.thread.row = None
                self.flush()

        def line(response, *args, **kw):
            raw = readline(response, *args, **kw)
            row = getattr(self.thread, "row", None)
            if row is not None and raw.startswith(b"data:"):
                text = raw[5:].strip()
                with self.lock:
                    row["status"] = response.status
                    if text == b"[DONE]":
                        row["done"] = True
                    else:
                        try:
                            msg = json.loads(text)
                        except (TypeError, ValueError):
                            return raw
                        if msg.get("id"):
                            row["server_id"] = msg["id"]
                        if msg.get("error"):
                            row["sse_error"] = msg["error"]
                            self.flush()
                        for choice in msg.get("choices") or []:
                            piece = (choice.get("delta") or {}).get("content") or ""
                            if piece:
                                row["chunks"] += 1
                                row["text_prefix"] = (row["text_prefix"] + piece)[:1024]
            return raw

        from contextlib import ExitStack
        from ironmule_product.http_server import _ProductRequestHandler

        finish_error = _ProductRequestHandler._finish_sse_error
        def server_error(handler, error, request):
            record = {"id":request.request_id, "chain":[]}
            exc = error
            for _ in range(4):
                if exc is None:
                    break
                frames = traceback.extract_tb(exc.__traceback__)[-8:]
                record["chain"].append({"type":type(exc).__name__, "message":str(exc)[:500],
                    "frames":[[Path(f.filename).name,f.lineno,f.name] for f in frames]})
                exc = exc.__cause__ or exc.__context__
            with self.lock:
                self.server_errors.append(record)
                self.flush()
            return finish_error(handler, error, request)

        stack = ExitStack()
        stack.enter_context(patch.object(compare, "batch", group))
        stack.enter_context(patch.object(compare, "stream", request))
        stack.enter_context(patch.object(http.client.HTTPResponse, "readline", line))
        stack.enter_context(patch.object(_ProductRequestHandler, "_finish_sse_error", server_error))
        return stack


def run(a):
    a.deadline = time.monotonic() + 114
    receipt = release.pinned(a.state, a.receipt_sha256)
    require(release.checksum(a.geometry) == a.geometry_sha256 and release.checksum(a.head) == a.head_sha256,
            "changed HTTP policy or geometry")
    a.geometry_record = read(a.geometry)
    binding = startup.key_for(a, receipt)
    require(binding == a.binding, "different HTTP evaluator")
    evidence = a.state / "deployment" / ("restart-http-"+binding)
    gate = evidence / "gate-reference.json"
    journal = read(evidence / "gate-devices-reference.json")
    prior = read(evidence / "decision.json")
    require(prior == {"passed": False, "reason": "ironmule returned a stream error"}
            and release.checksum(gate) == a.gate_sha256 and read(gate)["source_sha256"] == receipt["qualification"]["reference_sha256"]
            and read(gate)["qualified_width"]["sequences"] == 32 and read(gate)["passes"] is True
            and len(journal) == 7 and {x["stage"] for x in journal} == set((*restart_gate_broker.STAGES,*restart_gate_broker.WIDTH_STAGES))
            and all(x["child_status"] == 0 and x["provider"] == "cuda" and x["gpu_layers"] == 999 and x["default_sequences"] == 8 for x in journal),
            "missing original stock quality and width qualification")
    a.out.mkdir(mode=0o700, parents=True, exist_ok=True)
    opened = a.out/"opened.json"
    with opened.open("x") as f:
        json.dump({"schema":1, "binding":binding, "source":read(gate)["source_sha256"], "gate":a.gate_sha256,
                   "diagnostic":release.checksum(__file__),"seed":startup.SEEDS[0],"price_label":False,"started_s":time.time()},f)
    trace = Trace(a.out)
    endpoint = startup.cold_endpoint

    @contextmanager
    def retaining(*args, **kw):
        with endpoint(*args, **kw) as result:
            _, outer, selection = result
            try:
                yield result
            finally:
                client = outer.client
                service = outer.backend
                if service:
                    logs = service.store.root / "stability-log.jsonl"
                    raw = logs.read_bytes()[-32768:] if logs.is_file() else b""
                    (a.out/"stability-tail.jsonl").write_bytes(raw)
                if client:
                    with client._stderr_lock:
                        (a.out/"worker-stderr.log").write_bytes(bytes(client._stderr_buffer))
                save(a.out/"worker.json",{"cold_failure":outer.failure, "selection":selection,
                    "worker_stderr":client.stderr_tail() if client else None,
                    "last_exit":getattr(client,"last_exit",None), "health":service.health() if service else None})

    error = None
    try:
        with trace.observe(), patch.object(startup, "cold_endpoint", retaining):
            startup.measure(a, receipt, False, read(gate), startup.SEEDS[0])
    except Exception as exc:
        error = {"type":type(exc).__name__, "message":str(exc)[:400]}
    finally:
        with trace.lock:
            trace.flush()
            trace.closed = True
    note = {"schema":1, "diagnostic_only":True, "completed_arm":error is None, "error":error,
            "request_report_sha256":release.checksum(a.out/"requests.json"),
            "worker_report_sha256":release.checksum(a.out/"worker.json") if (a.out/"worker.json").exists() else None}
    write_once(a.out/"result.json",note)
    return note


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("state","trial","checkout","candidate-source","gguf","worker-root","reference-lib","table","geometry","head","out"):
        p.add_argument("--"+key,type=Path,required=True)
    for key in ("receipt-sha256","geometry-sha256","head-sha256","gate-sha256","binding"):
        p.add_argument("--"+key,required=True)
    p.set_defaults(port=0, reference_url=None, dedicated=True, worker_uid=60000, timeout=114)
    os.umask(0o077)
    print(json.dumps(run(p.parse_args())),flush=True)
