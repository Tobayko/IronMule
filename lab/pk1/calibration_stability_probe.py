"""Observe complete return values and explicit Python field writes of real startup calibrations."""
from __future__ import annotations

import ast
import hashlib
import json
import sys
import time
from pathlib import Path


def calibration_methods(source):
    tree = ast.parse(source.read_bytes())
    engine = next(c for c in tree.body if isinstance(c, ast.ClassDef) and c.name == "GgmlEngine")
    init = next(f for f in engine.body if getattr(f, "name", None) == "__init__")
    calls = {n.func.attr for n in ast.walk(init) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and isinstance(n.func.value, ast.Name) and n.func.value.id == "self"}
    registered = {}
    text = source.read_text()
    for fn in engine.body:
        if not isinstance(fn, ast.FunctionDef) or not fn.name.startswith("measure_") or fn.name not in calls:
            continue
        writes = sorted({a.attr for a in ast.walk(fn) if isinstance(a, ast.Attribute) and isinstance(a.ctx, ast.Store)
                         and isinstance(a.value, ast.Name) and a.value.id == "self"})
        body = ast.get_source_segment(text, fn)
        registered[fn.name] = {"line": fn.lineno, "direct_python_writes": writes,
                               "method_sha256": hashlib.sha256(body.encode()).hexdigest()}
    return registered


def value(x):
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, (tuple, list)):
        return [value(v) for v in x]
    if isinstance(x, dict):
        result = {str(k): value(v) for k, v in x.items()}
        if len(result) != len(x):
            raise ValueError("ambiguous serialized calibration keys")
        return result
    if type(x).__module__ == "ironmule.ggml_engine" and hasattr(x, "__dict__"):
        return {"object_type": type(x).__qualname__, "fields": value(vars(x))}
    raise ValueError("unrecordable calibration return value")


def main():
    import restart_arm

    from ironmule import ggml_engine as ge

    dest = Path(sys.argv[sys.argv.index("--output") + 1])
    raw = dest.with_name("original-arm.json")
    sys.argv[sys.argv.index("--output") + 1] = str(raw)
    source = Path(ge.__file__).resolve()
    inventory = calibration_methods(source)
    defined = {getattr(ge.GgmlEngine, name).__code__: name for name in inventory}
    current, captured = {}, []
    timings = {}
    original = ge.GgmlEngine.__init__

    def observe(frame, event, result):
        name = defined.get(frame.f_code)
        if name is None:
            return
        clock = time.monotonic_ns()
        if event == "call":
            current[frame] = clock
            timings[frame] = {}
        if event == "c_return" and getattr(result, "__name__", "") == "append":
            local = frame.f_locals
            samples = local.get("times")
            width = local.get("t")
            if isinstance(samples, list) and type(width) is int and getattr(result, "__self__", None) is samples:
                timings[frame][str(width)] = list(samples)
        if event != "return":
            return
        elapsed = (clock - current.pop(frame)) / 1e9
        engine = frame.f_locals["self"]
        side = {key: value(getattr(engine, key)) for key in inventory[name]["direct_python_writes"] if hasattr(engine, key)}
        captured.append({"method": name, "method_sha256": inventory[name]["method_sha256"], "elapsed_s": elapsed,
                         "return_value": value(result), "direct_field_values": side, "timing_samples_by_width": timings.pop(frame)})

    def init(self, *args, **kwargs):
        previous = sys.getprofile()
        if previous is not None:
            raise ValueError("an independent outer profiler is already active")
        sys.setprofile(observe)
        try:
            return original(self, *args, **kwargs)
        finally:
            sys.setprofile(previous)

    ge.GgmlEngine.__init__ = init
    try:
        restart_arm.main()
    finally:
        ge.GgmlEngine.__init__ = original
    report = {"arm": json.loads(raw.read_bytes()), "inventory": inventory, "calibrations": captured,
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "profiler_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "unobserved": "Native KV, graph/kernel caches, memory pools and indirect mutable side effects are not captured by this recorder."}
    dest.write_text(json.dumps(report, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
