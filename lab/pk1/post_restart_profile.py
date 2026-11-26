"""Measure exclusive owned-source time inside a complete, independently graded PR60 startup."""
from __future__ import annotations

import ast
import cProfile
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path


def eligibility(node):
    name = node.name
    calls = {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
    names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
    if name.startswith("__") or any(word in name.lower() for word in
                                   ("qualif", "verify", "check", "measure", "identit", "receipt", "digest", "hash", "probe")):
        return False, "construction, qualification or binding"
    if names.intersection({"hashlib", "ctypes", "subprocess"}) or any(
            f.startswith(("llama_", "ggml_", "cuda", "cuBLAS")) or f in {"CDLL", "synchronize", "join", "wait", "sleep"}
            for f in calls):
        return False, "native, wait or protected verification time cannot be priced as Python work"
    return True, "exclusive owned implementation time"


def sites(source):
    cls = next(c for c in ast.parse(source.read_bytes()).body if isinstance(c, ast.ClassDef) and c.name == "GgmlEngine")
    return {f"GgmlEngine.{n.name}": (n.lineno, *eligibility(n)) for n in cls.body if isinstance(n, ast.FunctionDef)}


def main():
    import restart_arm
    import serving_trial

    from ironmule import ggml_engine as ge

    dest = Path(sys.argv[sys.argv.index("--output") + 1])
    raw = dest.with_name("original-restart-arm.json")
    sys.argv[sys.argv.index("--output") + 1] = str(raw)
    source = Path(ge.__file__).resolve()
    known = sites(source)
    records = []

    def capture(phase, fn, *a, **kw):
        prof = cProfile.Profile()
        start = time.monotonic()
        result = prof.runcall(fn, *a, **kw)
        wall = time.monotonic() - start
        times = defaultdict(float)
        counts = defaultdict(int)
        for entry in prof.getstats():
            co = entry.code
            if isinstance(co, str) or co.co_filename != str(source):
                continue
            owner = ".".join(co.co_qualname.split(".")[:2])
            if owner in known:
                times[owner] += entry.inlinetime
                if co.co_qualname == owner:
                    counts[owner] += entry.callcount
        captured = [{"path": "ironmule/ggml_engine.py", "function": name, "line": known[name][0],
                     "self_s": seconds, "calls": counts[name], "eligible": known[name][1], "reason": known[name][2]}
                    for name, seconds in sorted(times.items())]
        unused = wall - sum(times.values())
        if unused < -1e-6:
            raise ValueError("overlapping exclusive times")
        records.append({"phase": phase, "wall_s": wall, "sites": captured, "unassigned_s": max(0., unused)})
        return result

    init = ge.GgmlEngine.__init__
    wave = serving_trial.run_wave
    waves = iter(("cold", "warm", "factual"))
    ge.GgmlEngine.__init__ = lambda self, *a, **kw: capture("startup", init, self, *a, **kw)
    serving_trial.run_wave = lambda *a, **kw: capture(next(waves), wave, *a, **kw)
    try:
        restart_arm.main()
    finally:
        ge.GgmlEngine.__init__ = init
        serving_trial.run_wave = wave
    if [r["phase"] for r in records] != ["startup", "cold", "warm", "factual"]:
        raise ValueError("incomplete measured work")
    result = {"arm": json.loads(raw.read_bytes()), "phases": records,
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "profiler_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    dest.write_text(json.dumps(result, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
