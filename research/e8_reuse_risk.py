"""E8: calibrate the real flip risk of prefix reuse on a realistic prefix length."""
import os, statistics, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import mlx.core as mx
from ironmule import bench
from ironmule.runtime import FixedKVCache, Knobs, _fixed_state_from_standard, _leaves, _project, _trunk
from ironmule.tune import DEFAULT_MODEL, _eos_ids, gpu_busy, load_engine
assert gpu_busy() is None, gpu_busy()

PREFIX = """You are the experiment selector for an inference-performance research project.

Hardware context: Apple M1 Max, 32 GB unified memory, 32 GPU cores, macOS, MLX backend.
The model, its weights and its quantisation are fixed and may never be changed to
produce a performance gain. Only the execution layer below the model may change.

Fixed selection policy, applied in this order:
1. Prefer the candidate with the largest confirmed end-to-end effect that also closes
   a workload gap the project has not yet covered.
2. Never choose a candidate whose only evidence is a diagnostic upper bound, because an
   upper bound is not an implementation.
3. Never choose a candidate that is blocked on a permission you do not have.
4. Never choose a candidate whose correctness gate has already failed, unless the
   evidence explicitly states that the failure was resolved.
5. If two candidates are otherwise equal, prefer the one with the lower implementation
   risk and the shorter measurement cost.

Output contract: return exactly one JSON object with exactly one key named candidate_id,
whose value is one of the identifiers listed in the evidence. Emit no prose, no markdown,
no code fence, and no explanation of any kind.

Measured evidence for this request:
"""

VARIANTS = [
    ("persistent_service", "keeping the model loaded removed 65.3032% of paired time to first output, all greedy outputs matched"),
    ("batched_readback", "reading the stop token every eight steps was 4.1893% faster but can emit extra tokens"),
    ("upper_bound_probe", "15.3% is a diagnostic ceiling only, with no implementation behind it"),
    ("kv_realloc", "4.4263% of decode correlates with cache reallocation, blocked on an architecture permission"),
    ("prefix_reuse", "reusing a shared prompt prefix removed 47% of end-to-end time, correctness gate passed 4 of 4"),
    ("projection_fusion", "1.10% faster prefill, decode unchanged, output bit identical by construction"),
    ("speculative_ngram", "acceptance 0.17 per drafted token made decode 2.9 times slower, tokens identical"),
    ("width_batching", "7.577 ms per token against 11.909 at width one, requires concurrent requests"),
    ("compiled_fixed_cache", "7.04% faster decode with identical tokens, already qualified end to end"),
    ("capacity_sizing", "cache sized to the workload instead of a fixed 512, never measured in isolation"),
    ("wired_residency", "untested, expected to matter only under memory pressure"),
    ("head_skip_prefill", "projecting only the read prompt position removed 15.3615% of prefill, gate passed"),
]

def suffix(rot):
    picks = [VARIANTS[(rot + k) % len(VARIANTS)] for k in (0, 3, 6, 9)]
    body = "\n".join(f"- {name}: {text}" for name, text in picks)
    return f"{body}\n\nChoose exactly one of: {', '.join(n for n, _ in picks)}."

SUFFIXES = [suffix(r) for r in range(12)]
MAX_TOKENS = 32

BEST = Knobs(compiled_fixed_cache=True, fused_argmax=False, head_skip_prefill=True,
             fuse_projections=True)
engine, tok = load_engine(DEFAULT_MODEL, BEST)
eos = _eos_ids(tok)
model, trunk = engine.model, _trunk(engine.model)
render = lambda t: tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True)
encode = lambda t: list(tok.encode(t, add_special_tokens=False))

full_ids = [encode(render(PREFIX + s)) for s in SUFFIXES]
prefix_ids = encode(render(PREFIX + "@@CUT@@").split("@@CUT@@")[0])
for i, ids in enumerate(full_ids):
    assert ids[:len(prefix_ids)] == prefix_ids, f"request {i}: tokenisation gate failed"
CAP = ((max(len(i) for i in full_ids) + MAX_TOKENS + 63)//64)*64
print(f"prefix {len(prefix_ids)} tokens, full {min(map(len, full_ids))}-{max(map(len, full_ids))}, "
      f"prefix share {len(prefix_ids)/statistics.mean(map(len, full_ids)):.1%}, capacity {CAP}")

snap_state, _ = engine._prefill(prefix_ids, CAP)
SNAP = [{"keys": l["keys"], "values": l["values"]} for l in snap_state["layers"]]

def prefill_full(i):
    t0 = time.perf_counter_ns()
    cache = engine.model.make_cache()
    hidden = trunk(mx.array(full_ids[i])[None, :], cache=cache)
    logits = _project(model, hidden[:, -1:, :])
    mx.eval(logits)
    state = _fixed_state_from_standard(cache, len(full_ids[i]), CAP)
    mx.eval(*_leaves(state)); mx.synchronize()
    return state, logits[:, -1, :], time.perf_counter_ns() - t0

def prefill_reuse(i):
    t0 = time.perf_counter_ns()
    state = {"position": {"offset": mx.array(len(prefix_ids), dtype=mx.int32)},
             "layers": [{"keys": l["keys"], "values": l["values"]} for l in SNAP]}
    caches = [FixedKVCache(l, state["position"], CAP) for l in state["layers"]]
    hidden = trunk(mx.array(full_ids[i][len(prefix_ids):])[None, :], cache=caches)
    logits = _project(model, hidden[:, -1:, :])
    state = {"position": {"offset": mx.array(len(full_ids[i]), dtype=mx.int32)},
             "layers": [{"keys": c.keys, "values": c.values} for c in caches]}
    mx.eval(logits, *_leaves(state)); mx.synchronize()
    return state, logits[:, -1, :], time.perf_counter_ns() - t0

def risk(lf, lr):
    """How far the observed perturbation is from flipping this token."""
    f, r = lf.astype(mx.float32), lr.astype(mx.float32)
    order = mx.argsort(-f, axis=-1)
    i1, i2 = int(order[0, 0].item()), int(order[0, 1].item())
    gap_full = (f[0, i1] - f[0, i2]).item()
    gap_reuse = (r[0, i1] - r[0, i2]).item()
    delta = abs(gap_full - gap_reuse)
    return {"top1": i1, "top2": i2, "gap_full": gap_full, "gap_reuse": gap_reuse,
            "delta_gap": delta, "headroom": (gap_reuse / delta) if delta > 0 else float("inf"),
            "argmax_reuse": int(mx.argmax(r, -1).item()), "flip": gap_reuse <= 0}

for i in range(3):
    prefill_full(i); prefill_reuse(i)

body = engine._body(CAP, 1)
rows, all_head = [], []
for i in range(len(SUFFIXES)):
    sf, lf, tf = prefill_full(i)
    sr, lr, tr = prefill_reuse(i)
    steps, flips = [], 0
    while True:
        m = risk(lf, lr)
        steps.append(m); flips += m["flip"]
        if m["headroom"] != float("inf"):
            all_head.append(m["headroom"])
        tokv = int(mx.argmax(lf.astype(mx.float32), -1).item())
        if m["argmax_reuse"] != tokv or tokv in eos or len(steps) >= MAX_TOKENS:
            break
        inp = mx.array([[tokv]])
        of = body(inp, sf); sf, lf = of[1], of[0][:, -1, :]
        orr = body(inp, sr); sr, lr = orr[1], orr[0][:, -1, :]
        mx.eval(lf, lr, *_leaves(sf), *_leaves(sr)); mx.synchronize()
    rows.append({"request": i, "steps": len(steps), "flips": flips,
                 "full_ttft_ms": tf/1e6, "reuse_ttft_ms": tr/1e6,
                 "min_headroom": min(s["headroom"] for s in steps),
                 "min_gap_reuse": min(s["gap_reuse"] for s in steps),
                 "max_delta_gap": max(s["delta_gap"] for s in steps),
                 "identical": flips == 0, "detail": steps})

print(f"\n{'req':>3} {'steps':>6} {'flips':>6} {'full ms':>9} {'reuse ms':>9} {'ratio':>7} {'min gap':>9} {'max dgap':>9} {'headroom':>9}")
for r in rows:
    h = r["min_headroom"]
    print(f"{r['request']:3d} {r['steps']:6d} {r['flips']:6d} {r['full_ttft_ms']:8.1f} {r['reuse_ttft_ms']:8.1f} "
          f"{r['reuse_ttft_ms']/r['full_ttft_ms']:7.4f} {r['min_gap_reuse']:9.3f} {r['max_delta_gap']:9.3f} "
          f"{('inf' if h==float('inf') else f'{h:9.1f}'):>9}")
tot = sum(r["steps"] for r in rows)
print(f"\n{sum(r['identical'] for r in rows)}/{len(rows)} requests token identical, "
      f"{sum(r['flips'] for r in rows)} flips over {tot} steps")
print(f"decisive headroom over all steps: min {min(all_head):.1f}  median {statistics.median(all_head):.1f}  "
      f"p5 {sorted(all_head)[max(0,int(0.05*len(all_head))-1)]:.1f}")
print(f"TTFT ratio median {statistics.median(r['reuse_ttft_ms']/r['full_ttft_ms'] for r in rows):.4f}")
bench.record("E8-prefix-reuse-risk", {"prefix_tokens": len(prefix_ids), "capacity": CAP,
             "requests": len(SUFFIXES), "total_steps": tot, "rows": rows,
             "headroom_min": min(all_head), "headroom_median": statistics.median(all_head)})
