"""ADM2 (2026-11-25): paired HTTP waves that alternate arms inside one server start (IRONMULE_LAB_SWITCH).

Every round runs one wave per arm, in an order rotated from round to round; every wave has prompts of its own (a
seed per round and position: the benchmark's mix of 16 short and 16 long prompts, so every wave carries the same
shape of work and no wave finds its prompts kept from an earlier one). Before a wave the switch file names the
arm's switches; the server and its worker read it within 20 ms. One JSON line per wave (arm, round, tok/s, wave
seconds, first-token p50 / max, the client timeline is kept apart).

  python lab/adm2/paired.py --url http://127.0.0.1:8002 --model gguf/Qwen3-32B-Q4_K_M --switch /work/arms.json \\
      --arms 'R={"ADM1":0,"ADM2":0,"PX2":0,"BF1":0}' 'A={"ADM1":1,"ADM2":0,"PX2":0,"BF1":0}' ... \\
      --rounds 6 --width 32 --out waves.jsonl --timeline tl.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import time
from pathlib import Path

from ironmule_product import compare


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--engine", default="ironmule")
    p.add_argument("--switch", required=True)
    p.add_argument("--arms", nargs="+", required=True, help="NAME=JSON")
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--seed", type=int, default=20261125)
    p.add_argument("--out", required=True)
    p.add_argument("--timeline", default="")
    p.add_argument("--quality", action="store_true", help="also grade the factual cases once per arm (last round)")
    p.add_argument("--settle-s", type=float, default=0.3)
    p.add_argument("--discard", type=int, default=1, help="waves at --width before the rounds, not recorded")
    a = p.parse_args()
    arms = []
    for spec in a.arms:
        name, _, values = spec.partition("=")
        arms.append((name, json.loads(values)))
    server = {"base_url": a.url, "model_id": a.model}
    deadline = time.monotonic() + 7200
    compare.stream(a.engine, server, compare.WARMUP, 8, deadline)
    Path(a.switch).write_text(json.dumps(arms[0][1]))
    time.sleep(a.settle_s)
    for k in range(a.discard):  # the first wave of this width after a start (or a width change) is not an arm's
        prompts = compare.performance_prompts(a.seed + 777 + k, a.width, max(a.width, 8))
        compare.batch(a.engine, server, prompts, a.width, a.max_tokens, deadline)
    rows_out = []
    for rnd in range(a.rounds):
        order = arms[rnd % len(arms):] + arms[:rnd % len(arms)]
        for pos, (name, values) in enumerate(order):
            Path(a.switch).write_text(json.dumps(values))
            time.sleep(a.settle_s)
            seed = a.seed + 1009 * rnd + 101 * pos + a.width
            prompts = compare.performance_prompts(seed, a.width, max(a.width, 8))
            w0 = time.monotonic()
            rows, elapsed = compare.batch(a.engine, server, prompts, a.width, a.max_tokens, deadline,
                                          timeline=bool(a.timeline))
            m = compare.performance_metrics(rows, elapsed)
            rec = {"arm": name, "round": rnd, "pos": pos, "seed": seed, "width": a.width, "wave_start": w0,
                   "tok_s": round(m["tokens_per_second"], 2), "wall_s": round(m["wall_s"], 4),
                   "ttft_p50_ms": round(m["ttft_median_ms"], 1), "ttft_max_ms": round(max(r["ttft_ms"] for r in rows), 1),
                   "tokens": m["completion_tokens"]}
            if a.quality and rnd == a.rounds - 1:
                from ironmule.factual_gate import MAX_TOKENS, cases, correct

                qs = cases(seed)
                qrows, _ = compare.batch(a.engine, server, [c.prompt for c in qs], a.width, MAX_TOKENS, deadline)
                rec["quality"] = f"{sum(correct(c, r['text']) for c, r in zip(qs, qrows))}/{len(qs)}"
            rows_out.append(rec)
            with open(a.out, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            if a.timeline:
                with open(a.timeline, "a") as fh:
                    for index, r in enumerate(rows):
                        fh.write(json.dumps({"label": name, "engine": a.engine, "repeat": rnd, "width": a.width,
                                             "index": index, "wave_start": w0, "server_id": r["server_id"],
                                             "prompt_tokens": r["prompt_tokens"], "tokens": r["tokens"],
                                             "ttft_ms": r["ttft_ms"], "t_send": r["t_send"], "t_done": r["t_done"],
                                             "chunk_times": r["chunk_times"]}) + "\n")
            print(json.dumps(rec), flush=True)
    summary = {}
    for name, _ in arms:
        mine = [r for r in rows_out if r["arm"] == name]
        summary[name] = {"waves": len(mine), "tok_s_median": st.median(r["tok_s"] for r in mine),
                         "wall_s_median": st.median(r["wall_s"] for r in mine),
                         "ttft_p50_median_ms": st.median(r["ttft_p50_ms"] for r in mine),
                         **({"quality": [r["quality"] for r in mine if "quality" in r]} if a.quality else {})}
    ref = arms[0][0]
    for name, _ in arms[1:]:  # paired by round: the same round's reference wave
        d = [next(r["wall_s"] for r in rows_out if r["arm"] == name and r["round"] == k)
             - next(r["wall_s"] for r in rows_out if r["arm"] == ref and r["round"] == k) for k in range(a.rounds)]
        summary[name]["paired_wall_ms_vs_" + ref] = {"median": round(1e3 * st.median(d), 1),
                                                     "faster_rounds": sum(1 for x in d if x < 0), "rounds": len(d)}
    print("SUMMARY " + json.dumps(summary), flush=True)
    with open(a.out, "a") as fh:
        fh.write(json.dumps({"summary": summary}) + "\n")


if __name__ == "__main__":
    main()
