"""Screen a device-measured W4 dispatch table inside the L40S stock path's observed memory margin."""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from lab.pk6.mixed_progress.trial import GGUF_NAME, board
from lab.pk6.replay_http import checksum
from lab.pk6.warm_repeat import GGUF, save


def choose(inventory, total_mib, stock_peak_mib):
    if not 0 < stock_peak_mib < total_mib:
        raise ValueError("missing or inconsistent stock memory sample")
    # Reserve another 5% of the board and price each native tile copy at twice its original bytes.
    budget = max(0, int((total_mib * 0.95 - stock_peak_mib) * 2**20))
    selected, estimate = [], 0
    for shape in sorted(
        inventory["measured_shapes"], key=lambda s: s["total_weight_bytes"]
    ):
        cost = 2 * shape["total_weight_bytes"]
        if cost <= 0:
            raise ValueError("unknown shape footprint")
        if estimate + cost <= budget:
            selected.append([shape["code"], shape["N"], shape["K"]])
            estimate += cost
    return {
        "allowed_shapes": selected,
        "budget_bytes": budget,
        "estimated_bytes": estimate,
    }


def worker(args):
    from ironmule.factual_gate import ggml_scores
    from ironmule.ggml_engine import GgmlEngine, Job
    from ironmule.gguf_tokenizer import GgufTokenizer
    from contextlib import closing

    t = time.monotonic()
    with (
        closing(GgmlEngine(str(args.gguf), seqs=8, seq_ctx=2048, speculate=False)) as e,
        closing(GgufTokenizer(str(args.gguf), e.L)) as tok,
    ):
        if not e.gpu_layers:
            raise ValueError("missing GPU offload")
        startup = time.monotonic() - t
        quality = ggml_scores(e, tok, seed=args.seed)
        waves = []
        for n in (1, 8):
            for repeat in range(3):
                # This probe prices a native decode workload; HTTP measurements live in the primary series.
                text = (
                    "Continue with single number words in sequence. "
                    f"Trial {args.seed + repeat} starts with one two three four five six"
                )
                ids = tok.apply_chat_template(
                    [{"role": "user", "content": text}],
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                jobs = [Job(list(ids), 32) for _ in range(n)]
                e._clear()
                start = time.monotonic()
                e.run(jobs)
                waves.append(
                    {
                        "width": n,
                        "repeat": repeat,
                        "seconds": time.monotonic() - start,
                        "tokens": [len(j.tokens) for j in jobs],
                        "reasons": [j.reason for j in jobs],
                        "output_ids_sha256": hashlib.sha256(
                            json.dumps([j.tokens for j in jobs]).encode()
                        ).hexdigest(),
                    }
                )
        packed = 0
        if os.environ["IRONMULE_KG_LIB"]:
            fn = ctypes.CDLL(os.environ["IRONMULE_KG_LIB"]).im_w4_repacked_bytes
            fn.restype = ctypes.c_size_t
            packed = fn()
        save(
            args.out / f"worker-{args.arm}.json",
            {
                "startup_s": startup,
                "seed": args.seed,
                "quality": quality,
                "waves": waves,
                "native_repacked_bytes": packed,
                "tokens_served": e.kg.get("tokens_served", []),
                "q6k_tokens_served": e.kg.get("q6k_tokens_served", []),
            },
        )


def run(args):
    args.out.mkdir(exist_ok=False)
    target = json.loads((args.primary / "target.json").read_bytes())
    if (
        board() != target["hardware"]
        or checksum(args.gguf) != GGUF
        or args.gguf.name != GGUF_NAME
    ):
        raise ValueError("different board or model")
    inventory = json.loads(args.inventory.read_bytes())
    if checksum(args.table) != inventory["table_sha256"]:
        raise ValueError("different target calibration")
    selected = choose(inventory, target["hardware"]["memory_mib"], args.stock_peak_mib)
    included = {tuple(v) for v in selected["allowed_shapes"]}
    lines = [
        line
        for line in args.table.read_text().splitlines()
        if tuple(map(int, line.split()[:3])) in included
    ]
    save(
        args.out / "selection.json",
        {
            **selected,
            "hardware": board(),
            "gguf_sha256": GGUF,
            "table_sha256": checksum(args.table),
            "table_rows": len(lines),
            "stock_peak_mib": args.stock_peak_mib,
            "library_sha256": checksum(args.table.with_name("kernel.so")),
        },
    )
    if not lines:
        save(
            args.out / "outcome.json",
            {"status": "no_shape_fits_screen", "production": False},
        )
        return
    table = args.out / "selected.table"
    table.write_text("\n".join(lines) + "\n")
    out = {}
    for label, seed in (("R1", 825323), ("M1", 825323), ("M2", 629317), ("R2", 629317)):
        limit = min(150, int(float(os.environ["RP_DEADLINE"]) - time.time() - 60))
        if limit < 60:
            break
        env = {
            **os.environ,
            "IRONMULE_KG_LIB": str(args.table.with_name("kernel.so"))
            if label.startswith("M")
            else "",
            "IRONMULE_W4_TABLE": str(table),
            "IRONMULE_GGML_PACK": "1",
            "IRONMULE_GGML_UBATCH": "8192",
            "IRONMULE_CUDA_GRAPHS_ROWS": "1",
            "IRONMULE_PS1": "0",
            "IRONMULE_PK6": "0",
            "IRONMULE_GGML_DH2": "0",
            "IRONMULE_PS1_CONCAT": "0",
            "IRONMULE_PS1_WARM": "0",
            "IRONMULE_SV5": "0",
            "IRONMULE_GGML_SEQ_CTX": "2048",
        }
        with (args.out / f"log-{label}.txt").open("xb") as f:
            try:
                p = subprocess.run(
                    [
                        sys.executable,
                        __file__,
                        "--out",
                        str(args.out),
                        "--gguf",
                        str(args.gguf),
                        "--arm",
                        label,
                        "--seed",
                        str(seed),
                    ],
                    env=env,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    timeout=limit,
                )
                out[label] = p.returncode
            except subprocess.TimeoutExpired:
                out[label] = "timeout"
        save(args.out / "attempts.json", out)
        if out[label] != 0:
            break
    save(
        args.out / "outcome.json",
        {
            "status": "complete_native_screen"
            if len(out) == 4 and all(v == 0 for v in out.values())
            else "incomplete_native_screen",
            "production": False,
            "attempts": out,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("primary", "inventory", "table", "out", "gguf"):
        p.add_argument("--" + name, type=Path, required=name in ("out", "gguf"))
    p.add_argument("--stock-peak-mib", type=float)
    p.add_argument("--arm")
    p.add_argument("--seed", type=int)
    a = p.parse_args()
    worker(a) if a.arm else run(a)
