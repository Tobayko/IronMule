"""Sample real product calls in a dedicated discovery server using CUDA activity tracing."""

from pathlib import Path
import os
import time

from lab.pk1.native_order import activity
from lab.pk6.warm_repeat import save


def install(engine_class):
    directory = Path(os.environ["IRONMULE_LAB_ACTIVITY"])
    original = engine_class._decode
    seen = set()

    def captured(engine, rows, **kwargs):
        jobs = getattr(engine, "_tl_jobs", {}) or {}
        prompt = sum(
            seq in jobs and pos < len(jobs[seq].ids) for _, pos, seq, _ in rows
        )
        decode = sum(
            seq in jobs and pos >= len(jobs[seq].ids) for _, pos, seq, _ in rows
        )
        # Capture an actual short first prefill, substantial mixed calls and a busy pure decode.
        kind = (
            "mixed"
            if prompt >= 300 and decode >= 12
            else "decode"
            if prompt == 0 and decode >= 8
            else "short"
            if 15 <= prompt <= 64 and decode == 0
            else None
        )
        if (
            kind is None
            or kind in seen
            or any(j.hardware_probe or j.limit != 32 for j in jobs.values())
        ):
            return original(engine, rows, **kwargs)
        seen.add(kind)
        import torch
        from torch.profiler import ProfilerActivity, profile

        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            begin = time.monotonic()
            result = original(engine, rows, **kwargs)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - begin
        events = activity(prof)
        filename = directory / f"activity-{kind}.json"
        save(
            filename,
            {
                "prompt": prompt,
                "decode": decode,
                "elapsed_s": elapsed,
                "sequences": sorted({r[2] for r in rows}),
                "kernels": events,
                "pid": os.getpid(),
                "start_monotonic": begin,
                "shared": bool(engine.pool),
                "logical_batch_capacity": engine.batch_cap,
                "request_ids": sorted(
                    {jobs[r[2]].tag for r in rows if r[2] in jobs and jobs[r[2]].tag}
                ),
                "occupied_kv_cells": engine.pool.used if engine.pool else None,
            },
        )
        return result

    engine_class._decode = captured


def count_dispatches(engine_class, trace):
    """Bind native plugin launches to each measured HTTP call in this worker process."""
    import ctypes

    library = os.environ.get("IRONMULE_KG_LIB")
    tile = os.getenv("IRONMULE_LAB_MMQ_TILE")
    if tile:
        library = str(Path(os.environ["LLAMA_CPP_LIB_PATH"]) / "libggml-cuda.so")
    if not library:
        return
    lib = ctypes.CDLL(library)
    if hasattr(lib, "im_prompt_begin"):
        begin, end = lib.im_prompt_begin, lib.im_prompt_end
        begin.restype = end.restype = None
        original_init, original_close = engine_class.__init__, engine_class.close

        def open_model(engine, *args, **kwargs):
            begin()
            engine._lab_prompt_session = True
            try:
                original_init(engine, *args, **kwargs)
            except BaseException:
                if engine._lab_prompt_session:
                    engine._lab_prompt_session = False
                    end()
                raise

        def close_model(engine):
            if getattr(engine, "_lab_prompt_session", False):
                engine._lab_prompt_session = False
                end()
            return original_close(engine)

        engine_class.__init__ = open_model
        engine_class.close = close_model

    counter = lib.im_mmq_tile_calls if tile else lib.im_prompt_dispatches
    counter.restype = ctypes.c_uint64
    original = engine_class._decode

    def counted(engine, rows, **kwargs):
        start = time.monotonic()
        before = counter()
        result = original(engine, rows, **kwargs)
        launches = counter() - before
        if launches:
            with open(trace, "a") as handle:
                handle.write(
                    f"# prompt_kernel t={start:.9f} rows={len(rows)} launches={launches}\n"
                )
        return result

    engine_class._decode = counted
