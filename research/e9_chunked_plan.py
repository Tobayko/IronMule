"""E9: does defining the plan as chunked prefill make prefix reuse exact?"""
import os, statistics, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import mlx.core as mx
from ironmule import bench
from ironmule.runtime import FixedKVCache, Knobs, _fixed_state_from_standard, _leaves, _project, _trunk
from ironmule.tune import DEFAULT_MODEL, _eos_ids, gpu_busy, load_engine
assert gpu_busy() is None, gpu_busy()

src = open(os.path.join(os.path.dirname(__file__), "e8_reuse_risk.py")).read()
ns = {"statistics": statistics}
exec(compile(src[src.index('PREFIX = """'):src.index('MAX_TOKENS = 32')], "e8_workload", "exec"), ns)
PREFIX, SUFFIXES = ns["PREFIX"], ns["SUFFIXES"]
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
P = len(prefix_ids)
for ids in full_ids:
    assert ids[:P] == prefix_ids
CAP = ((max(len(i) for i in full_ids) + MAX_TOKENS + 63)//64)*64


def _fresh_fixed(template):
    return {"position": {"offset": mx.array(0, dtype=mx.int32)},
            "layers": [{"keys": mx.zeros((k.shape[0], k.shape[1], CAP, k.shape[3]), dtype=k.dtype),
                        "values": mx.zeros((k.shape[0], k.shape[1], CAP, k.shape[3]), dtype=k.dtype)}
                       for k in (l.keys for l in template)]}


def _feed(state, ids):
    caches = [FixedKVCache(l, state["position"], CAP) for l in state["layers"]]
    hidden = trunk(mx.array(ids)[None, :], cache=caches)
    used = int(state["position"]["offset"].item()) + len(ids)
    return ({"position": {"offset": mx.array(used, dtype=mx.int32)},
             "layers": [{"keys": c.keys, "values": c.values} for c in caches]}, hidden)


def arm_single(i):
    t0 = time.perf_counter_ns()
    cache = engine.model.make_cache()
    hidden = trunk(mx.array(full_ids[i])[None, :], cache=cache)
    logits = _project(model, hidden[:, -1:, :]); mx.eval(logits)
    state = _fixed_state_from_standard(cache, len(full_ids[i]), CAP)
    mx.eval(*_leaves(state)); mx.synchronize()
    return state, logits[:, -1, :], time.perf_counter_ns() - t0


def arm_chunked(i):
    """Two chunks, split exactly at the prefix boundary."""
    t0 = time.perf_counter_ns()
    probe = engine.model.make_cache(); trunk(mx.array([[full_ids[i][0]]]), cache=probe)
    mx.eval([c.keys for c in probe])
    state = _fresh_fixed(probe)
    state, _ = _feed(state, full_ids[i][:P])
    state, hidden = _feed(state, full_ids[i][P:])
    logits = _project(model, hidden[:, -1:, :])
    mx.eval(logits, *_leaves(state)); mx.synchronize()
    return state, logits[:, -1, :], time.perf_counter_ns() - t0


PROBE = engine.model.make_cache(); trunk(mx.array([[full_ids[0][0]]]), cache=PROBE)
mx.eval([c.keys for c in PROBE])
_snap_state, _ = _feed(_fresh_fixed(PROBE), prefix_ids)
mx.eval(*_leaves(_snap_state)); mx.synchronize()
SNAP = [{"keys": l["keys"], "values": l["values"]} for l in _snap_state["layers"]]


def arm_reuse(i):
    t0 = time.perf_counter_ns()
    state = {"position": {"offset": mx.array(P, dtype=mx.int32)},
             "layers": [{"keys": l["keys"], "values": l["values"]} for l in SNAP]}
    state, hidden = _feed(state, full_ids[i][P:])
    logits = _project(model, hidden[:, -1:, :])
    mx.eval(logits, *_leaves(state)); mx.synchronize()
    return state, logits[:, -1, :], time.perf_counter_ns() - t0


def maxdiff(a, b):
    return mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()


for i in range(2):
    arm_single(i); arm_chunked(i); arm_reuse(i)

body = engine._body(CAP, 1)
rows = []
for i in range(len(SUFFIXES)):
    (ss, ls, ts), (sc, lc, tc), (sr, lr, tr) = arm_single(i), arm_chunked(i), arm_reuse(i)
    d_sc, d_cr, steps = [], [], 0
    while True:
        d_sc.append(maxdiff(ls, lc)); d_cr.append(maxdiff(lc, lr)); steps += 1
        a_s = int(mx.argmax(ls.astype(mx.float32), -1).item())
        a_c = int(mx.argmax(lc.astype(mx.float32), -1).item())
        a_r = int(mx.argmax(lr.astype(mx.float32), -1).item())
        if a_c != a_r or a_s != a_c or a_c in eos or steps >= MAX_TOKENS:
            break
        inp = mx.array([[a_c]])
        o = body(inp, ss); ss, ls = o[1], o[0][:, -1, :]
        o = body(inp, sc); sc, lc = o[1], o[0][:, -1, :]
        o = body(inp, sr); sr, lr = o[1], o[0][:, -1, :]
        mx.eval(ls, lc, lr, *_leaves(ss), *_leaves(sc), *_leaves(sr)); mx.synchronize()
    rows.append({"request": i, "steps": steps,
                 "max_diff_single_vs_chunked": max(d_sc), "max_diff_chunked_vs_reuse": max(d_cr),
                 "single_ms": ts/1e6, "chunked_ms": tc/1e6, "reuse_ms": tr/1e6,
                 "argmax_single": a_s, "argmax_chunked": a_c, "argmax_reuse": a_r})

print(f"{'req':>3} {'steps':>6} {'single|chunk':>13} {'chunk|reuse':>12} {'single ms':>10} {'chunk ms':>9} {'reuse ms':>9}")
for r in rows:
    print(f"{r['request']:3d} {r['steps']:6d} {r['max_diff_single_vs_chunked']:13.4e} "
          f"{r['max_diff_chunked_vs_reuse']:12.4e} {r['single_ms']:9.1f} {r['chunked_ms']:8.1f} {r['reuse_ms']:8.1f}")
exact = sum(r["max_diff_chunked_vs_reuse"] == 0.0 for r in rows)
print(f"\nchunked vs reuse bit identical: {exact}/{len(rows)} requests")
print(f"single vs chunked max diff overall: {max(r['max_diff_single_vs_chunked'] for r in rows):.4f}")
print(f"chunking cost: single {statistics.median(r['single_ms'] for r in rows):.1f} ms -> "
      f"chunked {statistics.median(r['chunked_ms'] for r in rows):.1f} ms "
      f"({statistics.median(r['chunked_ms'] for r in rows)/statistics.median(r['single_ms'] for r in rows):.4f}x); "
      f"reuse {statistics.median(r['reuse_ms'] for r in rows):.1f} ms "
      f"({statistics.median(r['reuse_ms'] for r in rows)/statistics.median(r['chunked_ms'] for r in rows):.4f}x of chunked)")
bench.record("E9-chunked-plan-exactness", {"prefix_tokens": P, "capacity": CAP, "rows": rows,
             "bit_identical_requests": exact})
