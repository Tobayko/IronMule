"""Exercise the shared-pool admission candidate with actual cached KV and fresh CPU tasks."""

import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import tempfile

from ironmule import ggml_engine as ge
from ironmule.gguf_tokenizer import GgufTokenizer
from training.serving_trial import file_digest, fixture, run_wave


def screen(path, chunk_rows=0):
    def run(seed, candidate, trace):
        if chunk_rows:
            os.environ["IRONMULE_LAB_SHARED_CHUNK"] = str(
                chunk_rows if candidate else 0
            )
        else:
            os.environ["IRONMULE_LAB_POOL_BURST"] = str(int(candidate))
        ge.TRACE = str(trace)
        with (
            closing(
                ge.GgmlEngine(
                    str(path),
                    threads=2,
                    seqs=8,
                    seq_ctx=1024,
                    shared=True,
                    cap=8,
                    flash_attn=True,
                    speculate=False,
                )
            ) as e,
            closing(GgufTokenizer(str(path), e.L)) as tok,
        ):
            if e.gpu_layers or not e.pool:
                raise ValueError("CPU shared pool required")
            # Force the low-budget case on this CPU. The L40S experiment uses the trained Pilot intact.
            normal = e.pilot.decide
            e.pilot.decide = lambda *args: {**normal(*args), "budget": 128, "cap": 8}
            short, long = (fixture(seed, 8, k) for k in ("short", "document"))
            r = run_wave(
                e, tok, [a[i] for i, a in enumerate((short, long) * 4)], 12, {}
            )
            if e.pool.active or e.pool.reserve:
                raise ValueError("completed requests retained live KV reservations")
            marker = "# shared_chunks " if chunk_rows else "# shared_budget "
            return {
                "seed": seed,
                "candidate": candidate,
                "marked": marker in trace.read_text(),
                **{
                    k: r[k]
                    for k in (
                        "correct",
                        "completed",
                        "requests",
                        "output_sha256",
                        "tokens",
                        "wall_s",
                    )
                },
            }

    rows = []
    with tempfile.TemporaryDirectory() as d:
        for i, seed in enumerate((274041, 898013) if chunk_rows else (970113, 970117)):
            for candidate in (False, True) if not i else (True, False):
                rows.append(run(seed, candidate, Path(d) / f"{seed}-{candidate}.txt"))
    return {
        "gguf_sha256": file_digest(path),
        "waves": rows,
        "all_correct": all(r["correct"] == r["completed"] == 8 for r in rows),
        "matched_work": all(
            a["requests"] == b["requests"] for a, b in zip(rows[::2], rows[1::2])
        ),
        "witnessed": all(r["marked"] == r["candidate"] for r in rows),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("gguf", type=Path)
    p.add_argument("--chunk-rows", type=int, default=0)
    args = p.parse_args()
    result = screen(args.gguf, args.chunk_rows)
    print(json.dumps(result, indent=2))
    raise SystemExit(
        0 if all(result[k] for k in ("all_correct", "matched_work", "witnessed")) else 1
    )
