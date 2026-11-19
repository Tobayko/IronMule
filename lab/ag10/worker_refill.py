"""Exercise an oversized initial GGUF batch through the real worker, with its checked GGUF options."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import threading
import time

from ironmule.ggml_engine import gate_key
from ironmule.gguf_tokenizer import GgufTokenizer
from ironmule_product import worker
from ironmule_product.ggml_bridge import GgmlBridge


def measure(model, cache):
    key = gate_key(str(model), str(model))
    gate = json.loads((cache / (key + ".json")).read_text())
    if not gate["passes"]:
        raise ValueError("the GGUF must pass its local gate")
    tokenizer = GgufTokenizer(str(model))
    bridge = GgmlBridge(str(model), tokenizer, speculate=False, gate=gate)
    try:
        request_rows = []
        for i in range(16):
            prompt = ("List every integer from 1 to 1000 in ascending order, separated by commas. Start at 1."
                      if i == 0 else f"Begin with the word Paris. {'Read this sample. ' * (i % 5)}Sample {i}.")
            request_rows.append({"request_id": str(i), "model": "local/refill", "max_tokens": 384 if i == 0 else 3,
                                 "stream": True, "messages": [{"role": "user", "content": prompt}]})
        command = {"variant": "current_engine", "batch_id": "refill", "requests": request_rows}
        bridge.generate([1, 2, 3], 2)
        rows = {r["request_id"]: {"token_ids": []} for r in request_rows}
        order = []
        groups = []
        def collect(frame):
            now = time.monotonic() - start
            kind = frame["type"]
            if kind == "batch_done":
                groups.append(frame["request_ids"])
                return
            if kind not in ("done", "token"):
                raise RuntimeError(f"unexpected worker frame {kind}")
            row = rows[frame["request_id"]]
            if kind == "token":
                row.setdefault("ttft_s", now)
                row["token_ids"].append(frame["token_id"])
            else:
                order.append(frame["request_id"])
                row.update(end_s=now, prompt_tokens=frame["prompt_tokens"], completion_tokens=frame["completion_tokens"],
                           steps=frame["engine"]["ggml_steps"], finish=frame["finish_reason"])

        worker._emit = collect
        start = time.monotonic()
        worker._run_engine_batch(command, "local/refill", tokenizer, bridge, {}, set(), threading.Lock())
        elapsed = time.monotonic() - start
        if groups != [list(rows)] or sorted(order) != sorted(rows):
            raise RuntimeError("incomplete worker run")
        for row in rows.values():
            ids = row.pop("token_ids")
            if len(ids) != row["completion_tokens"]:
                raise RuntimeError("invalid token count")
            row["ids_sha256"] = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        return {"gate_key": key, "gguf_sha256": bridge.gguf_sha256, "free_sequences": bridge.free_slots(),
                "gate_settings": gate["settings"], "wall_s": elapsed, "order": order, "rows": rows}
    finally:
        bridge.close()
        tokenizer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("gate_cache", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    with args.out.open("x") as target:
        json.dump(measure(args.gguf.resolve(strict=True), args.gate_cache), target, indent=2)
        target.write("\n")
