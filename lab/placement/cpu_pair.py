"""Reproduce the placement comparison on one CPU with separately qualified one- and three-thread workers."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from ironmule.gguf_tokenizer import gguf_model_id, gguf_revision
from ironmule_product.backend import MLXWorkerClient
from ironmule_product.calibration import model_lease
from ironmule_product.cli import _pinned_ggml_options
from ironmule_product.device_pool import start_pool
from ironmule_product.placement import Placement
from ironmule_product.placement_training import wait_for_calibration
from ironmule_product.service import ProductService
from ironmule_product.state import ProductStore
from ironmule_product.types import GenerationRequest, ModelSpec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--short-traffic", action="store_true",
                        help="send 32 short public requests after the initial comparison and observe requalification")
    args = parser.parse_args()
    path = args.gguf.resolve()
    spec = ModelSpec(gguf_model_id(str(path)), gguf_revision(str(path)), str(path), path.stat().st_size)
    store = ProductStore(args.state_dir)
    store.setup("server")
    args.gguf, args.draft_gguf, args.speculate, args.agent, args.model = str(path), None, "off", "off", None
    os.environ.update(IRONMULE_PILOT="0", IRONMULE_GGML_GPU_LAYERS="0", CUDA_VISIBLE_DEVICES="")
    os.environ.setdefault("IRONMULE_GGML_SEQS", "4")
    options = {}
    with model_lease(store):
        for label, threads in (("local", 1), ("cpu", 3)):
            os.environ["IRONMULE_GGML_THREADS"] = str(threads)
            arm = ProductStore(store.root / f"{threads}-thread")
            arm.setup("server")
            opt = _pinned_ggml_options("cpu", args, spec, arm)
            opt["ggml_worker_env"].update(IRONMULE_GGML_THREADS=str(threads), IRONMULE_PILOT="0")
            options[label] = opt
        def worker(label):
            return MLXWorkerClient(spec, execution_variant="current_engine", **options[label])
        pool = start_pool(store, spec, list(options), worker, ProductService)
        try:
            pool.hardware_placement = Placement(pool, options, store.root)
            result = wait_for_calibration(pool, 20)
            if result == 0 and args.short_traffic:
                learner = pool.hardware_placement
                with learner.lock:
                    previous = learner.trace["started"]
                    already_short = max(case[1] for case in learner.trace["plan"]) == 8
                if not already_short:
                    for i in range(32):
                        size, limit = (96, 1024)[i % 2], (4, 8)[(i // 2) % 2]
                        text = ("List as many world capitals as possible, in alphabetical order. " * 20)[:size]
                        list(pool.stream(GenerationRequest(spec.model_id, (("user", text),), limit)))
                    deadline = time.monotonic() + 60
                    while time.monotonic() < deadline:
                        with learner.lock:
                            if learner.trace["started"] != previous:
                                break
                        time.sleep(1)
                    else:
                        raise RuntimeError("the workload shift did not start a fresh campaign")
                    result = wait_for_calibration(pool, 20)
            if result == 0:
                prompt = ("Write a short list of five countries in Europe and their capitals. " * 2)[:96]
                request = GenerationRequest(spec.model_id, (("user", prompt),), 8)
                done = next(event for event in pool.stream(request) if event.get("type") == "done")
                print(json.dumps({"foreground": pool.hardware_placement.status()["last_decision"],
                                  "completed_tokens": done["completion_tokens"]}))
            return result
        finally:
            pool.close()


if __name__ == "__main__":
    raise SystemExit(main())
