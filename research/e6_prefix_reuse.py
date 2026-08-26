"""E6: can a prefix KV snapshot be reused without changing a single token?

Feasibility and correctness only. Performance is claimed after a paired A/B.
"""
import os, statistics, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import mlx.core as mx
from ironmule import bench
from ironmule.runtime import FixedKVCache, Knobs, _leaves, _project, _trunk
from ironmule.tune import DEFAULT_MODEL, _eos_ids, gpu_busy, load_engine

assert gpu_busy() is None, gpu_busy()

# Shared instruction block. Fixed for every request of this service.
PREFIX = """You choose exactly one next Project Friday experiment.

Hardware: Apple M1 Max, 32 GB unified memory. Use only the evidence below.

Fixed selection policy:
1. Prefer the largest already confirmed end-to-end lever that also closes a required missing workload.
2. Do not choose a diagnostic upper bound.
3. Do not choose a permission-blocked cache change.
4. Choose exactly one ID from the list given in the evidence.

Return only a JSON object with exactly one key named candidate_id and no prose, markdown, or explanation.

Measured evidence:
"""

SUFFIXES = [
"""- persistent_service_qualification: keeping the model loaded reduced paired time to first output by 65.3032%; all greedy outputs matched exactly.
- batched_readback: isolated decode readback accounts for 12.98% per output token, but batching can emit extra tokens.
- host_readback_upper_bound: 15.3% is only an upper bound, not a usable implementation.
Choose from: persistent_service_qualification, batched_readback, host_readback_upper_bound.""",

"""- prefix_kv_reuse: reusing a shared prompt prefix removes 63% of end-to-end time, correctness not yet established.
- projection_fusion: measured 1.10% faster prefill, decode neutral, bit identical.
- kernel_tiling_sweep: no candidate beat the vendor kernel on any tested shape.
Choose from: prefix_kv_reuse, projection_fusion, kernel_tiling_sweep.""",

"""- speculative_ngram: acceptance 0.17 per drafted token, decode 2.9x slower, tokens identical.
- width_four_batching: 7.577 ms per token against 11.909 at width one, needs concurrent requests.
- lm_head_streaming: already at 288 GB/s, the measured ceiling is 324 GB/s.
Choose from: speculative_ngram, width_four_batching, lm_head_streaming.""",

"""- wired_memory_residency: untested, expected to matter only under memory pressure.
- capacity_right_sizing: cache sized to the workload instead of a fixed 512, untested in isolation.
- compiled_fixed_cache: measured 7.04% faster decode with identical tokens, already qualified.
Choose from: wired_memory_residency, capacity_right_sizing, compiled_fixed_cache.""",
]

MAX_TOKENS = 32
BEST = Knobs(compiled_fixed_cache=True, fused_argmax=True, head_skip_prefill=True,
             fuse_projections=True)
engine, tok = load_engine(DEFAULT_MODEL, BEST)
eos = _eos_ids(tok)
model, trunk = engine.model, _trunk(engine.model)


def render(text):
    return tok.apply_chat_template([{"role": "user", "content": text}],
                                   tokenize=False, add_generation_prompt=True)


def encode(text):
    return list(tok.encode(text, add_special_tokens=False))


# --- tokenisation gate -------------------------------------------------------
full_ids = [encode(render(PREFIX + s)) for s in SUFFIXES]
rendered_prefix = render(PREFIX + "@@CUT@@").split("@@CUT@@")[0]
prefix_ids = encode(rendered_prefix)
for i, ids in enumerate(full_ids):
    assert ids[:len(prefix_ids)] == prefix_ids, \
        f"request {i}: prefix is not a token prefix of the full prompt"
print(f"tokenisation gate ok: prefix {len(prefix_ids)} tokens, "
      f"full {[len(i) for i in full_ids]} tokens")

CAP = ((max(len(i) for i in full_ids) + MAX_TOKENS + 63) // 64) * 64


def snapshot_prefix():
    """Prefill the shared prefix once into a fixed-shape cache."""
    state, _ = engine._prefill(prefix_ids, CAP)
    # slice_update is functional, so these arrays stay valid however often the
    # decode loop advances a *copy* of this structure.
    return {"position": {"offset": mx.array(len(prefix_ids), dtype=mx.int32)},
            "layers": [{"keys": l["keys"], "values": l["values"]} for l in state["layers"]]}


def restore(snap):
    return {"position": {"offset": mx.array(int(snap["position"]["offset"].item()), dtype=mx.int32)},
            "layers": [{"keys": l["keys"], "values": l["values"]} for l in snap["layers"]]}


def prefill_suffix(snap, suffix_ids):
    """Continue an existing fixed-cache state with the request-specific tail."""
    state = restore(snap)
    caches = [FixedKVCache(l, state["position"], CAP) for l in state["layers"]]
    hidden = trunk(mx.array(suffix_ids)[None, :], cache=caches)
    logits = _project(model, hidden[:, -1:, :])
    token = mx.argmax(logits[:, -1, :], axis=-1).reshape((1, 1))
    state = {"position": {"offset": mx.array(len(prefix_ids) + len(suffix_ids), dtype=mx.int32)},
             "layers": [{"keys": c.keys, "values": c.values} for c in caches]}
    mx.eval(token, *_leaves(state)); mx.synchronize()
    return state, token


def generate_from(state, token):
    physical = [int(token.reshape((-1,)).item())]
    body = engine._body(CAP, 1)
    tokenv = token
    for _ in range(MAX_TOKENS - 1):
        out = body(tokenv, state)
        tokenv, state = out[0][:, -1:], out[1]
        mx.eval(tokenv, *_leaves(state)); mx.synchronize()
        physical.append(int(tokenv.reshape((-1,)).item()))
        if physical[-1] in eos:
            break
    logical = []
    for v in physical:
        logical.append(v)
        if v in eos:
            break
    return logical


def arm_full(i):
    t0 = time.perf_counter_ns()
    state, token = engine._prefill(full_ids[i], CAP)
    ttft = time.perf_counter_ns() - t0
    return generate_from(state, token), ttft


def arm_reuse(i, snap):
    t0 = time.perf_counter_ns()
    state, token = prefill_suffix(snap, full_ids[i][len(prefix_ids):])
    ttft = time.perf_counter_ns() - t0
    return generate_from(state, token), ttft


snap = snapshot_prefix()
for i in range(len(SUFFIXES)):          # warmup both paths
    arm_full(i); arm_reuse(i, snap)

REPEATS = 3
rows = []
for i in range(len(SUFFIXES)):
    fa = [arm_full(i) for _ in range(REPEATS)]
    fb = [arm_reuse(i, snap) for _ in range(REPEATS)]
    same = fa[0][0] == fb[0][0]
    det = (all(r[0] == fa[0][0] for r in fa) and all(r[0] == fb[0][0] for r in fb))
    rows.append({
        "request": i, "suffix_tokens": len(full_ids[i]) - len(prefix_ids),
        "full_ttft_ms": statistics.median(r[1] for r in fa)/1e6,
        "reuse_ttft_ms": statistics.median(r[1] for r in fb)/1e6,
        "identical": same, "deterministic": det,
        "full_tokens": fa[0][0], "reuse_tokens": fb[0][0],
        "text_full": tok.decode([t for t in fa[0][0] if t not in eos]),
        "text_reuse": tok.decode([t for t in fb[0][0] if t not in eos]),
    })

print(f"\n{'req':>3} {'suffix':>7} {'full TTFT':>10} {'reuse TTFT':>11} {'ratio':>7} {'identical':>10} {'determ':>7}")
for r in rows:
    print(f"{r['request']:3d} {r['suffix_tokens']:7d} {r['full_ttft_ms']:9.2f}ms {r['reuse_ttft_ms']:10.2f}ms "
          f"{r['reuse_ttft_ms']/r['full_ttft_ms']:7.4f} {str(r['identical']):>10} {str(r['deterministic']):>7}")
ok = sum(r["identical"] for r in rows)
print(f"\ncorrectness: {ok}/{len(rows)} requests token identical")
for r in rows:
    if not r["identical"]:
        print(f"  req {r['request']} full : {r['text_full'][:90]!r}")
        print(f"  req {r['request']} reuse: {r['text_reuse'][:90]!r}")
print(f"prefix {len(prefix_ids)} tokens, capacity {CAP}, mlx peak {mx.get_peak_memory()/1e9:.2f} GB")
bench.record("E6-prefix-kv-reuse-feasibility", {
    "prefix_tokens": len(prefix_ids), "capacity": CAP, "repeats": REPEATS,
    "max_tokens": MAX_TOKENS, "rows": rows, "identical_count": ok,
    "mlx_peak_bytes": mx.get_peak_memory()})
