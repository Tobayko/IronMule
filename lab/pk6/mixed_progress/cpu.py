"""Exercise the isolated candidate with real GGUF inference and identical seeded copy tasks."""
import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import platform

from ironmule import ggml_engine as ge
from ironmule.gguf_tokenizer import GgufTokenizer
from training.serving_trial import fixture, run_wave, file_digest


def screen(gguf, *, seed0=1729, pairs=2):
    if (ge.GPU_LAYERS != "0" or not ge.PACK or os.environ.get("IRONMULE_LAB_CPU_PACK") != "1"):
        raise ValueError("use the patched lab CPU path with GGML_PACK=1 and GPU_LAYERS=0")
    rows = []
    with closing(ge.GgmlEngine(str(gguf), threads=2, seqs=8, seq_ctx=1024, speculate=False)) as engine, \
            closing(GgufTokenizer(str(gguf), engine.L)) as tokenizer:
        if not engine.pack or engine.gpu_layers:
            raise ValueError("experiment checkout or CPU placement differs")
        for i in range(pairs):
            seed = seed0 + i
            order = (192, 0) if i % 2 else (0, 192)
            short, long = (fixture(seed, 8, key) for key in ("short", "document"))
            fixtures = [pool[i] for i, pool in enumerate((short, long) * 4)]
            for budget in order:
                engine._clear()
                engine._free = list(range(engine.seqs))
                os.environ["IRONMULE_LAB_MIXED_ROWS"] = str(budget)
                calls = []
                engine.observer = lambda d, p, kv, sec: calls.append((d, p))
                result = run_wave(engine, tokenizer, fixtures, 12, {})
                rows.append({"seed": seed, "row_budget": budget, "mixed_calls": sum(bool(d and p) for d,p in calls),
                             "native_calls": calls, **{k: result[k] for k in
                                ("completed", "correct", "tokens", "output_sha256", "wall_s", "requests")}})
    valid = all(a["output_sha256"] == b["output_sha256"] and a["correct"] == b["correct"] == 8
                for a, b in zip(rows[::2], rows[1::2], strict=True))
    mixed = all(bool(r["mixed_calls"]) == bool(r["row_budget"]) for r in rows)
    return {"passed": valid and mixed, "all_graded_correct": all(w["correct"] == w["completed"] == 8 for w in rows),
            "same_work": all(a["requests"] == b["requests"] for a,b in zip(rows[::2],rows[1::2],strict=True)),
            "mixed_steps_verified": mixed, "cpu": platform.machine(), "affinity": sorted(os.sched_getaffinity(0)),
            "prefer_whole_prompts": os.environ.get("IRONMULE_LAB_WHOLE") == "1", "threads": 2, "gguf_sha256": file_digest(gguf), "waves": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--pairs", type=int, default=2)
    args = parser.parse_args()
    result = screen(args.gguf, seed0=args.seed, pairs=args.pairs)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
