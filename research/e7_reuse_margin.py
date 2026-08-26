"""E7: is prefix reuse bit-identical, or only argmax-identical with a margin?"""
import os, json
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import mlx.core as mx
from ironmule import bench
from ironmule.runtime import FixedKVCache, Knobs, _leaves, _project, _trunk
from ironmule.tune import DEFAULT_MODEL, _eos_ids, gpu_busy, load_engine
import importlib.util
spec = importlib.util.spec_from_file_location("e6", os.path.join(os.path.dirname(__file__), "e6_prefix_reuse.py"))

assert gpu_busy() is None, gpu_busy()
# reuse E6's exact workload text without re-running it
src = open(os.path.join(os.path.dirname(__file__), "e6_prefix_reuse.py")).read()
ns = {}
exec(compile(src[src.index('PREFIX = """'):src.index('MAX_TOKENS = 32')], "e6_workload", "exec"), ns)
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
for ids in full_ids:
    assert ids[:len(prefix_ids)] == prefix_ids
CAP = ((max(len(i) for i in full_ids) + MAX_TOKENS + 63)//64)*64

snap_state, _ = engine._prefill(prefix_ids, CAP)
SNAP = {"position": {"offset": mx.array(len(prefix_ids), dtype=mx.int32)},
        "layers": [{"keys": l["keys"], "values": l["values"]} for l in snap_state["layers"]]}

def prefill_full(i):
    ids = mx.array(full_ids[i])[None, :]
    cache = engine.model.make_cache()
    hidden = trunk(ids, cache=cache)
    logits = _project(model, hidden[:, -1:, :])
    from ironmule.runtime import _fixed_state_from_standard
    mx.eval(logits)
    state = _fixed_state_from_standard(cache, len(full_ids[i]), CAP)
    mx.eval(*_leaves(state)); mx.synchronize()
    return state, logits[:, -1, :]

def prefill_reuse(i):
    suffix = full_ids[i][len(prefix_ids):]
    state = {"position": {"offset": mx.array(len(prefix_ids), dtype=mx.int32)},
             "layers": [{"keys": l["keys"], "values": l["values"]} for l in SNAP["layers"]]}
    caches = [FixedKVCache(l, state["position"], CAP) for l in state["layers"]]
    hidden = trunk(mx.array(suffix)[None, :], cache=caches)
    logits = _project(model, hidden[:, -1:, :])
    state = {"position": {"offset": mx.array(len(full_ids[i]), dtype=mx.int32)},
             "layers": [{"keys": c.keys, "values": c.values} for c in caches]}
    mx.eval(logits, *_leaves(state)); mx.synchronize()
    return state, logits[:, -1, :]

def compare(a, b):
    a32, b32 = a.astype(mx.float32), b.astype(mx.float32)
    diff = mx.max(mx.abs(a32 - b32)).item()
    order = mx.argsort(-a32, axis=-1)
    top1 = mx.take_along_axis(a32, order[:, :1], axis=-1).item()
    top2 = mx.take_along_axis(a32, order[:, 1:2], axis=-1).item()
    return diff, top1 - top2, int(mx.argmax(a32, -1).item()), int(mx.argmax(b32, -1).item())

body = engine._body(CAP, 1)
rows = []
for i in range(len(SUFFIXES)):
    sf, lf = prefill_full(i)
    sr, lr = prefill_reuse(i)
    steps, bitwise_equal = [], True
    while True:
        diff, margin, af, ar = compare(lf, lr)
        bitwise_equal &= (diff == 0.0)
        steps.append({"step": len(steps), "max_abs_diff": diff, "top1_top2_margin": margin,
                      "argmax_full": af, "argmax_reuse": ar,
                      "margin_over_diff": (margin/diff if diff > 0 else float("inf"))})
        if af != ar or af in eos or len(steps) >= MAX_TOKENS:
            break
        tokv = mx.array([[af]])
        of = body(tokv, sf); sf, lf = of[1], of[0][:, -1, :]
        orr = body(tokv, sr); sr, lr = orr[1], orr[0][:, -1, :]
        mx.eval(lf, lr, *_leaves(sf), *_leaves(sr)); mx.synchronize()
    finite = [s["margin_over_diff"] for s in steps if s["margin_over_diff"] != float("inf")]
    rows.append({"request": i, "steps": len(steps), "bitwise_equal": bitwise_equal,
                 "argmax_all_equal": all(s["argmax_full"] == s["argmax_reuse"] for s in steps),
                 "max_diff": max(s["max_abs_diff"] for s in steps),
                 "min_margin": min(s["top1_top2_margin"] for s in steps),
                 "min_margin_over_diff": min(finite) if finite else float("inf"),
                 "detail": steps})

print(f"{'req':>3} {'steps':>6} {'bit-equal':>10} {'argmax eq':>10} {'max|d|':>10} {'min margin':>11} {'margin/|d|':>11}")
for r in rows:
    mo = r["min_margin_over_diff"]
    print(f"{r['request']:3d} {r['steps']:6d} {str(r['bitwise_equal']):>10} {str(r['argmax_all_equal']):>10} "
          f"{r['max_diff']:10.3e} {r['min_margin']:11.4f} {('inf' if mo==float('inf') else f'{mo:11.1f}'):>11}")
print(f"\nlogits bit identical in {sum(r['bitwise_equal'] for r in rows)}/{len(rows)} requests")
print(f"argmax identical in {sum(r['argmax_all_equal'] for r in rows)}/{len(rows)} requests")
bench.record("E7-prefix-reuse-numeric-margin", {"prefix_tokens": len(prefix_ids),
             "capacity": CAP, "max_tokens": MAX_TOKENS, "rows": rows})
