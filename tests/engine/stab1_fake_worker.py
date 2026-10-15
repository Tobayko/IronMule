"""A protocol-faithful stand-in for the grouped engine worker (STAB1 fault scenarios; no model).

Speaks the real JSON-lines protocol of `ironmule_product.worker` for `current_engine`: the ready
handshake, `generate_batch` with tokens streamed round-robin across the group, `cancel` honoured at
the next token, and `batch_done`. Like the real worker (`_BatchStream`), each request's last token and
its `done` are held until the whole run has finished, so a short member waits for its run's end. A request's last user message selects a fault:

    crash      the worker writes one stderr line and exits (status 9) after this request's first token
    hang       the worker stops emitting and ignores cancellation (alive, no progress)
    error      the worker reports a generation error for the group
    slow:S     S seconds per token for the whole group

Anything else is served normally: max_tokens tokens whose text derives only from the message, so
outputs are comparable across workers. `FAKE_STAB1_CONTROL` names a JSON file read at start:
{"fail_start": true} exits before the handshake, {"digest": "<64 hex>"} changes the source identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import sys
import threading
import time

OUT = sys.stdout


def emit(value: dict) -> None:
    OUT.write(json.dumps(value, separators=(",", ":")) + "\n")
    OUT.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    args, _ = parser.parse_known_args()
    spec = json.loads(args.spec)
    control = {}
    if os.environ.get("FAKE_STAB1_CONTROL"):
        try:
            with open(os.environ["FAKE_STAB1_CONTROL"]) as stream:
                control = json.load(stream)
        except (OSError, ValueError):
            control = {}
    if control.get("fail_start"):
        sys.stderr.write("fake worker: start refused by control file\n")
        return 4
    emit({"type": "ready", "protocol_version": 1, "model_id": spec["model_id"], "revision": spec["revision"],
          "device": "gpu", "stop_handling": "parent", "context_limit": 8192, "execution_variant": "current_engine",
          "engine": control.get("engine", {"configuration": "current_profile"}),
          "sources_sha256": control.get("digest", "a" * 64)})
    commands: queue.Queue = queue.Queue()
    cancelled: set[str] = set()
    lock = threading.Lock()

    def read() -> None:
        for line in sys.stdin.buffer:
            command = json.loads(line)
            if command.get("type") == "cancel":
                with lock:
                    cancelled.add(command["request_id"])
            else:
                commands.put(command)
        commands.put(None)

    threading.Thread(target=read, daemon=True).start()
    while True:
        command = commands.get()
        if command is None or command.get("type") == "shutdown":
            # os._exit: a normal interpreter exit while the daemon reader holds stdin's buffer aborts
            # with "_enter_buffered_busy" (SIGABRT, a macOS crash report); the real worker avoids it too.
            OUT.flush()
            os._exit(0)
        if command.get("type") != "generate_batch":
            emit({"type": "error", "code": "protocol_error"})
            continue
        batch_id, requests = command["batch_id"], command["requests"]
        faults = [request["messages"][-1]["content"] for request in requests]
        if "error" in faults:
            emit({"type": "error", "code": "generation_error", "batch_id": batch_id})
            continue
        delay = max([float(f.split(":", 1)[1]) for f in faults if f.startswith("slow:")] or [0.01])
        sent = {request["request_id"]: 0 for request in requests}
        open_ids = [request["request_id"] for request in requests]
        by_id = {request["request_id"]: request for request in requests}
        held: dict[str, dict] = {}
        ends: dict[str, str] = {}
        while open_ids:
            for request_id in list(open_ids):
                request = by_id[request_id]
                with lock:
                    stop = request_id in cancelled
                if stop or sent[request_id] >= request["max_tokens"]:
                    ends[request_id] = "cancelled" if stop else "length"
                    open_ids.remove(request_id)
                    continue
                content = request["messages"][-1]["content"]
                if content == "hang":
                    while True:
                        time.sleep(3600)
                index = sent[request_id]
                if request_id in held:  # one token stays back, as in the real worker
                    emit(held.pop(request_id))
                text = hashlib.sha256(f"{content}/{index}".encode()).hexdigest()[:4] + " "
                held[request_id] = {"type": "token", "batch_id": batch_id, "request_id": request_id, "text": text,
                                    "token_id": index + 1, "prompt_tokens": 3, "completion_tokens": index + 1}
                sent[request_id] += 1
                if content == "crash":
                    emit(held.pop(request_id))
                    sys.stderr.write("fake worker: fatal out of memory during decode\n")
                    sys.stderr.flush()
                    os._exit(9)
            if open_ids:
                time.sleep(delay)
        for request in requests:  # the run is over: last tokens and completions
            request_id = request["request_id"]
            if request_id in held and ends[request_id] != "cancelled":
                emit(held.pop(request_id))
            delivered = sent[request_id] - (request_id in held)
            emit({"type": "done", "batch_id": batch_id, "request_id": request_id, "finish_reason": ends[request_id],
                  "prompt_tokens": 3, "completion_tokens": delivered if ends[request_id] == "cancelled" else sent[request_id],
                  "variant": "current_engine", "metrics": {}, "engine": {"configuration": "current_profile"}})
        emit({"type": "batch_done", "batch_id": batch_id, "request_ids": [r["request_id"] for r in requests],
              "memory": {"active": 1, "cache": 2, "peak": 3}})


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
