"""PORT1: stock against IronMule in separate processes, the like-for-like CUDA comparison.

Usage: python cross.py MODEL_ID REVISION CONFIGS_JSON OUT.json [REPS] [MEASURE]

`abcd.py` runs every arm in one process, so an environment setting such as IronMule's CUDA
graph default reaches its baseline arm too. Here each configuration runs in fresh
processes interleaved by repetition: "stock" pins MLX's own graph limits, baseline knobs
and interactive mode; IronMule configurations use whatever IronMule applies by itself.
The first configuration is the reference. A configuration with `"dtype": "float32"` changes
numerics, so it reports token agreement instead of claiming identity.
CONFIGS_JSON: [{"name", "env": {...}, "knobs": {...}, "mode": "interactive"|"throughput",
"dtype": null|"float32"}, ...]
"""
import hashlib
import json
import os
import statistics as st
import subprocess
import sys
import time


def child(model_id, revision, config_json, measure):
    import mlx.core as mx

    import ironmule
    from ironmule import benchmark as bench
    from ironmule.tune import load_engine

    config = json.loads(config_json)
    # The product path: IronMule's own opt-in numeric plan, not a post-load cast.
    engine, tokenizer = load_engine(model_id, ironmule.Knobs(**config["knobs"]), revision=revision,
                                    compute_dtype=config.get("dtype"))
    mode = ironmule.ThroughputMode if config["mode"] == "throughput" else ironmule.InteractiveMode
    with ironmule.Runtime(engine, tokenizer, model_id=model_id) as rt:
        walls, outputs = [], None
        for index in range(1 + measure):
            requests = bench._build_requests(rt, ironmule, ironmule.StrictOneShotPlan(), "strict", 6, 48)
            results, snapshot = bench._run(rt, ironmule, mode(), requests)
            outputs = [[int(t) for t in r.tokens] for r in results]
            if index:
                walls.append(snapshot["outer_wall_ms"])
    print(json.dumps({"walls_ms": walls, "median_ms": st.median(walls), "tokens": outputs,
                      "outputs_sha256": hashlib.sha256(json.dumps(outputs).encode()).hexdigest(),
                      "peak_memory_bytes": int(mx.get_peak_memory()),
                      "graph_env": {k: os.environ.get(k) for k in ("MLX_MAX_OPS_PER_BUFFER", "MLX_MAX_MB_PER_BUFFER")}}))


def main(model_id, revision, configs_json, out, reps, measure):
    configs = json.loads(configs_json)
    runs = {c["name"]: [] for c in configs}
    started = time.time()
    for rep in range(reps):
        order = configs[rep % len(configs):] + configs[:rep % len(configs)]
        for config in order:
            env = {k: v for k, v in os.environ.items() if not k.startswith("MLX_MAX_")}
            env.update(config.get("env", {}))
            proc = subprocess.run([sys.executable, __file__, "child", model_id, revision, json.dumps(config),
                                   str(measure)], env=env, capture_output=True, text=True, timeout=1800)
            row = {"rep": rep, "exit": proc.returncode, "stderr_tail": proc.stderr[-1500:]}
            if proc.returncode == 0:
                row.update(json.loads(proc.stdout.strip().splitlines()[-1]))
            runs[config["name"]].append(row)
            print(config["name"], rep, row.get("median_ms"), row["exit"], flush=True)
    base = configs[0]["name"]
    reference = next((r["tokens"] for r in runs[base] if r.get("tokens")), None)
    summary = {}
    for config in configs:
        name = config["name"]
        pairs = [r["median_ms"] / b["median_ms"] for r, b in zip(runs[name], runs[base])
                 if r.get("median_ms") and b.get("median_ms")]
        tokens = next((r["tokens"] for r in runs[name] if r.get("tokens")), None)
        summary[name] = {
            "ratio_vs_reference_per_rep": pairs,
            "median_ratio": st.median(pairs) if pairs else None,
            "max_ratio": max(pairs) if pairs else None,
            "identical_requests": (sum(a == b for a, b in zip(tokens, reference))
                                   if tokens and reference else None),
            "deterministic_across_processes": len({r.get("outputs_sha256") for r in runs[name]}) == 1,
            "failed_processes": sum(r["exit"] != 0 for r in runs[name]),
        }
    report = {"schema": "ironmule.port1-cross.v1", "model_id": model_id, "revision": revision,
              "configs": configs, "reps": reps, "measure": measure, "runs": runs, "summary": summary,
              "seconds": time.time() - started, "performance_claim": False}
    with open(out, "w") as stream:
        json.dump(report, stream, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "child":
        child(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
    else:
        main(*sys.argv[1:5], int(sys.argv[5]) if len(sys.argv) > 5 else 3,
             int(sys.argv[6]) if len(sys.argv) > 6 else 3)
