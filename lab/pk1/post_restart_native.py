"""Trace actual GGML ABI calls below PR60's measured decode wrapper without replacing native work."""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path


def digest(val):
    return hashlib.sha256(json.dumps(val, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def cell(record):
    return [record[k] for k in ("phase", "owner", "row_bucket", "sequences", "logits")]


def select_cells(records):
    costs = Counter()
    for r in records:
        key = tuple(cell(r))
        costs[key] += r["decode_s"] + r["readback_s"]
    result = []
    for startup in (True, False):
        options = {k: v for k, v in costs.items() if (k[0] == "startup") == startup}
        if options:
            key = max(sorted(options), key=options.get)
            result.append({"cell": list(key), "aggregate_abi_s": options[key]})
    return result


def abi_images(library):
    from ironmule import ggml_binary as binary
    from ironmule_devices import _function

    dladdr = _function(ctypes.CDLL(None), "dladdr", ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(binary._Object))
    mappings = binary._mappings()
    result = {}
    for name in ("llama_decode", "llama_get_logits_ith"):
        location = binary._Object()
        pointer = ctypes.cast(getattr(library, name), ctypes.c_void_p)
        if not dladdr(pointer, ctypes.byref(location)) or not location.path:
            raise ValueError("unresolved native symbol")
        path = Path(os.fsdecode(location.path)).resolve(strict=True)
        stat = path.stat()
        if mappings.get(location.base) != (stat.st_dev, stat.st_ino):
            raise ValueError("mapped native symbol no longer matches the on-disk image")
        result[name] = binary._digest(str(path), binary._stat_id(stat))
    return result


def main():
    import restart_arm
    import serving_trial
    from post_restart_activity import Activity

    from ironmule import ggml_engine as ge

    final = Path(sys.argv[sys.argv.index("--output") + 1])
    raw = final.with_name("original-arm.json")
    sys.argv[sys.argv.index("--output") + 1] = str(raw)
    source = Path(ge.__file__).resolve()
    L = ge.bindings()
    program = ctypes.CDLL(os.environ["IRONMULE_KG_LIB"])
    mask_getters = [program.ironmule_kg_get_mask, program.ironmule_kg_get_mask6]
    for getter in mask_getters:
        getter.argtypes, getter.restype = [], ctypes.c_uint32
    profile_path = os.environ.get("IRONMULE_NATIVE_ACTIVITY_CELLS")
    selected = json.loads(Path(profile_path).read_bytes()) if profile_path else []
    pending = [r["cell"] for r in selected]
    activity = Activity() if pending else None
    phase = "startup"
    waves = iter(("cold", "warm", "factual"))
    records, events = [], []
    kv = {}
    known = True
    current = None
    saved = {}
    orig_wave = serving_trial.run_wave
    orig_once = ge.GgmlEngine._decode_once

    def wave(*args, **kw):
        nonlocal phase
        phase = next(waves)
        return orig_wave(*args, **kw)

    def instrument(name, fn):
        def observe(*args, **kw):
            nonlocal known
            if name in ("llama_decode", "llama_get_logits_ith"):
                before = time.monotonic_ns()
                result = fn(*args, **kw)
                if current is not None:
                    counter = "decode" if name == "llama_decode" else "readback"
                    current[counter + "_s"] += (time.monotonic_ns() - before) / 1e9
                    current[counter + "_calls"] += 1
                    if counter == "decode":
                        current["native_statuses"].append(int(result))
                return result
            result = fn(*args, **kw)
            if name == "llama_memory_clear":
                kv.clear()
                known = True
            elif name == "llama_memory_seq_rm":
                _, seq, low, high = args
                if result is None:
                    known = False
                elif result:
                    for key in list(kv):
                        if (seq < 0 or key[0] == seq) and (low < 0 or key[1] >= low) and (high < 0 or key[1] < high):
                            del kv[key]
            else:
                # Copy, change, divide or compact sequence positions. Do not infer unknown token histories.
                known = False
            return result
        return observe

    def once(engine, rows, *a, **kw):
        nonlocal current
        frame = sys._getframe(1)
        chain = []
        locations = []
        while frame:
            co = frame.f_code
            if (co.co_filename == str(source) and co.co_qualname.startswith("GgmlEngine.")
                    and co.co_qualname not in ("GgmlEngine._decode", "GgmlEngine._decode_once")):
                chain.append(co.co_qualname)
                locations.append([co.co_qualname, frame.f_lineno])
            frame = frame.f_back
        owner = chain[0] if chain else "outside-engine"
        abi = [f() for f in mask_getters]
        item = {"phase": phase, "owner": owner, "source_callers": chain[:12], "source_lines": locations[:12],
                "native_statuses": [], "rows": len(rows),
                "row_bucket": len(rows).bit_length(), "sequences": len({r[2] for r in rows}),
                "logits": sum(bool(r[3]) for r in rows), "decode_s": 0., "readback_s": 0.,
                "decode_calls": 0, "readback_calls": 0, "mask_bits": abi, "threads": engine.threads,
                "argmax_on_device": bool(getattr(engine, "_argmax_only", None)), "known_logical_kv": known,
                "row_sha256": digest(rows), "kv_sha256": digest(sorted(kv.items())) if known else None}
        item["input_sha256"] = digest([item[k] for k in ("row_sha256", "kv_sha256", "mask_bits", "threads", "argmax_on_device")]) if known else None
        current = item
        tick = time.monotonic_ns()
        try:
            key = cell(item)
            if key in pending:
                pending.remove(key)
                picks, evidence = activity.measure(orig_once, engine, rows, *a, **kw)
                events.append({"cell": key, "record_index": len(records), "activity": evidence})
                item["activity_sampled"] = True
            else:
                picks = orig_once(engine, rows, *a, **kw)
                item["activity_sampled"] = False
        finally:
            current = None
        item["wrapper_s"] = (time.monotonic_ns() - tick) / 1e9
        item["output_sha256"] = digest(picks)
        for token, pos, seq, _ in rows:
            kv[seq, pos] = token
        records.append(item)
        return picks

    hooks = ["llama_decode", "llama_get_logits_ith", "llama_memory_clear", "llama_memory_seq_rm", "llama_memory_seq_cp",
             "llama_memory_seq_keep", "llama_memory_seq_add", "llama_memory_seq_div"]
    for name in hooks:
        if hasattr(L, name):
            saved[name] = getattr(L, name)
            setattr(L, name, instrument(name, saved[name]))
    ge.GgmlEngine._decode_once, serving_trial.run_wave = once, wave
    try:
        restart_arm.main()
    finally:
        ge.GgmlEngine._decode_once, serving_trial.run_wave = orig_once, orig_wave
        for name, fn in saved.items():
            setattr(L, name, fn)
    arm = json.loads(raw.read_bytes())
    result = {"arm": arm, "abi_images": abi_images(L.llama_cpp._lib), "records": records, "events": events, "selected_cells": selected,
              "not_observed": pending, "abi_symbols": sorted(saved), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "profiler_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    final.write_text(json.dumps(result, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
