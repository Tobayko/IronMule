"""ADM1 (2026-11-25): the product HTTP server and ProductService with a stub worker, to time a burst's admission.

The stub stands where WorkerBackend stands (streams_batches, joins, slot reuse, ggml configuration): it records when
the first group and every join reached it, polls joins every millisecond for `--gather-ms`, then streams `--tokens`
tokens per request `--step-ms` apart. No model; only the service's own path is real (HTTP threads, health, the
stability log, the scheduler, the join path).

  python lab/adm1/stub_server.py --port 8010 --state /tmp/st --log /tmp/arrivals.jsonl
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

from ironmule_product.http_server import create_server
from ironmule_product.service import ProductService
from ironmule_product.state import ProductStore
from ironmule_product.types import ModelSpec


class StubBackend:
    ready = True
    streams_batches = True
    reuses_slots = True
    engine_configuration = "ggml"
    execution_variant = "reference"
    batch_capacity = 32

    def __init__(self, log: str, gather_ms: float, tokens: int, step_ms: float):
        self.log, self.gather, self.tokens, self.step = log, gather_ms / 1000, tokens, step_ms / 1000
        self.ready_payload = {"model_id": "stub/model", "revision": "r", "device": "cpu",
                              "execution_variant": "reference", "sources_sha256": None, "engine": "ggml"}
        self._lock = threading.Lock()
        # a done frame's engine section about as large as a GPU worker's (about 8 KB on a CPU build)
        self.engine = {"configuration": "ggml", "plan": "strict_one_shot",
                       "tables": {str(i): {"step_ms": [round(0.1 * j, 2) for j in range(40)]} for i in range(24)}}

    def _wire(self, frame):
        """the protocol's cost: the worker encodes the frame and the client decodes it"""
        return json.loads(json.dumps(frame, separators=(",", ":")))

    def can_batch_requests(self, requests) -> bool:
        return True

    def complete_batch(self, requests, cancels=None, timeout=None, on_event=None, joins=None):
        requests = list(requests)
        arrivals = [(r.request_id, time.monotonic(), "group") for r in requests]
        started = last = time.monotonic()
        while time.monotonic() - started < self.gather and len(requests) < 32:
            fresh = list(joins()) if joins is not None else []
            now = time.monotonic()
            for request, _ in fresh:
                requests.append(request)
                arrivals.append((request.request_id, now, "join"))
                last = now
            if not fresh and len(requests) > 1 and now - last > 0.025:
                break
            time.sleep(0.001)
        with open(self.log, "a") as fh:
            for rid, t, kind in arrivals:
                fh.write(json.dumps({"id": rid, "t": t, "kind": kind}) + "\n")
            fh.write(json.dumps({"prefill_start": time.monotonic(), "n": len(requests)}) + "\n")
        for k in range(self.tokens):
            time.sleep(self.step)
            for r in requests:
                if k >= r.max_tokens:
                    continue
                on_event(r.request_id, self._wire({"type": "token", "request_id": r.request_id, "text": f" t{k}",
                                                   "token_id": 7, "prompt_tokens": 5, "completion_tokens": k + 1}))
        rows = []
        ended = time.monotonic()
        for r in requests:
            done = self._wire({"type": "done", "request_id": r.request_id, "finish_reason": "length", "prompt_tokens": 5,
                               "completion_tokens": min(self.tokens, r.max_tokens),
                               "engine": {**self.engine, "computed_tokens": 37}})
            on_event(r.request_id, done)
            rows.append({"request_id": r.request_id, "events": [], "metadata": done})
        with open(self.log, "a") as fh:
            fh.write(json.dumps({"last_step_end": ended}) + "\n")
        return rows

    def close(self, *a):
        self.ready = False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8010)
    p.add_argument("--state", required=True)
    p.add_argument("--log", required=True)
    p.add_argument("--gather-ms", type=float, default=150)
    p.add_argument("--tokens", type=int, default=4)
    p.add_argument("--step-ms", type=float, default=30)
    a = p.parse_args()
    store = ProductStore(Path(a.state))
    if not (Path(a.state) / "settings.json").exists():
        store.setup("server")
    spec = ModelSpec("stub/model", "r", str(Path(a.state).resolve()), 1)
    backend = StubBackend(a.log, a.gather_ms, a.tokens, a.step_ms)
    service = ProductService(store, backend=backend, spec=spec)
    from ironmule_product.lab_switch import start as lab_switch

    lab_switch()
    server = create_server(service, port=a.port)
    print(json.dumps({"ready": True, "port": a.port}), flush=True)
    try:
        server.serve_forever()
    finally:
        service.close()


if __name__ == "__main__":
    main()
