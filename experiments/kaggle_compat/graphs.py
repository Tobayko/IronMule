"""PORT1: does a larger CUDA graph per commit make IronMule's best arm faster on a T4?

Usage: python graphs.py MODEL_ID REVISION KNOBS_JSON OUT.json [REPS]

MLX's CUDA backend commits a CUDA graph every MLX_MAX_OPS_PER_BUFFER ops or
MLX_MAX_MB_PER_BUFFER MB. The defaults are 20/100 for compute capability 7.5 and
100/1000 for H100-class devices. Both are read once per process, so each setting runs in
fresh child processes, interleaved by repetition so drift hits every setting alike.
The measured arm is D (tuned knobs + throughput mode, `ironmule benchmark`'s workload);
every setting must reproduce the default setting's tokens.
"""
import hashlib
import json
import os
import statistics as st
import subprocess
import sys
import time

ALL_SETTINGS = {
    # Before IronMule set CUDA defaults, "default" meant MLX's own 20/100 on a T4.
    "default": {}, "graphs_off": {"MLX_USE_CUDA_GRAPHS": "0"},
    "ops100_mb1000": {"MLX_MAX_OPS_PER_BUFFER": "100", "MLX_MAX_MB_PER_BUFFER": "1000"},
    "ops400_mb4000": {"MLX_MAX_OPS_PER_BUFFER": "400", "MLX_MAX_MB_PER_BUFFER": "4000"},
    "mlx_default": {"MLX_MAX_OPS_PER_BUFFER": "20", "MLX_MAX_MB_PER_BUFFER": "100"},
    "ops400": {"MLX_MAX_OPS_PER_BUFFER": "400", "MLX_MAX_MB_PER_BUFFER": "100"},
}
SELECTED = os.environ.get("GRAPH_SETTINGS", "default,graphs_off,ops100_mb1000,ops400_mb4000").split(",")
SETTINGS = [(name, ALL_SETTINGS[name]) for name in SELECTED]
MEASURE = int(os.environ.get("GRAPH_MEASURE", "3"))
REQUESTS = int(os.environ.get("GRAPH_REQUESTS", "6"))
TOKENS = int(os.environ.get("GRAPH_TOKENS", "48"))


def child(model_id, revision, knobs):
    import mlx.core as mx

    import ironmule
    from ironmule import benchmark as bench
    from ironmule.tune import load_engine

    engine, tokenizer = load_engine(model_id, ironmule.Knobs(**json.loads(knobs)), revision=revision)
    with ironmule.Runtime(engine, tokenizer, model_id=model_id) as rt:
        walls, digest = [], None
        for index in range(1 + MEASURE):  # one warmup, then the measured runs
            requests = bench._build_requests(rt, ironmule, ironmule.StrictOneShotPlan(), "strict", REQUESTS, TOKENS)
            results, snapshot = bench._run(rt, ironmule, ironmule.ThroughputMode(), requests)
            outputs = [bench._output_record(r, tuple(rt.backend.eos_ids)) for r in results]
            digest = hashlib.sha256(json.dumps(outputs, sort_keys=True, default=str).encode()).hexdigest()
            if index:
                walls.append(snapshot["outer_wall_ms"])
    print(json.dumps({"walls_ms": walls, "median_ms": st.median(walls), "outputs_sha256": digest,
                      "peak_memory_bytes": int(mx.get_peak_memory()),
                      "graph_env": {k: os.environ.get(k) for k in ALL_SETTINGS["ops400_mb4000"]}}))


def main(model_id, revision, knobs, out, reps):
    runs = {name: [] for name, _ in SETTINGS}
    started = time.time()
    for rep in range(reps):
        order = SETTINGS[rep % len(SETTINGS):] + SETTINGS[:rep % len(SETTINGS)]
        for name, extra in order:
            proc = subprocess.run([sys.executable, __file__, "child", model_id, revision, knobs],
                                  env={**os.environ, **extra}, capture_output=True, text=True, timeout=600)
            row = {"rep": rep, "exit": proc.returncode, "stderr_tail": proc.stderr[-800:]}
            if proc.returncode == 0:
                row.update(json.loads(proc.stdout.strip().splitlines()[-1]))
            runs[name].append(row)
            print(name, rep, row.get("median_ms"), row["exit"], flush=True)
    base = SETTINGS[0][0]
    reference = [r.get("outputs_sha256") for r in runs[base]]
    summary = {}
    for name, _ in SETTINGS:
        pairs = [r["median_ms"] / d["median_ms"] for r, d in zip(runs[name], runs[base])
                 if r.get("median_ms") and d.get("median_ms")]
        summary[name] = {"ratio_vs_first_setting_per_rep": pairs,
                         "peak_memory_bytes": [r.get("peak_memory_bytes") for r in runs[name]],
                         "median_ratio": st.median(pairs) if pairs else None,
                         "token_identity": all(r.get("outputs_sha256") in reference[:1] for r in runs[name])}
    report = {"schema": "ironmule.port1-cuda-graphs.v1", "model_id": model_id, "revision": revision,
              "knobs": json.loads(knobs), "settings": dict(SETTINGS), "reps": reps, "runs": runs,
              "measure": MEASURE, "requests": REQUESTS, "max_tokens": TOKENS,
              "summary": summary, "seconds": time.time() - started, "performance_claim": False}
    with open(out, "w") as stream:
        json.dump(report, stream, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "child":
        child(*sys.argv[2:5])
    else:
        main(*sys.argv[1:5], int(sys.argv[5]) if len(sys.argv) > 5 else 3)
