"""E12 positive control 2: prove the *logit* comparison also fires.

Control 1 flipped a bit at prefix position 3 of a sliding layer at L=1024. The
logits did not move at all, because that position lies outside the 1024 window and
is never read. That validated the KV check but left the logit check unexercised.
Two further poisonings close that: one inside the window on a sliding layer, one on
a global layer which attends to everything.
"""
import json, os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import mlx.core as mx
from pathlib import Path
from ironmule import bench
from ironmule.tune import gpu_busy
import e12_window_falsification as E

assert gpu_busy() is None, gpu_busy()
h = E.Harness(E.DEFAULT_MODEL)
_, corpus = E.load_corpus()
L = 1024
preamble = E.build_preamble("natural", L, h.tok)
ids = h.render(preamble + "\n\nMeasured evidence for this request:\n" + corpus[0])
capacity = ((len(ids) + E.MAX_NEW_TOKENS + 63)//64)*64
snapshot, _ = h.snapshot(ids, L, capacity)
cb_state, cb_logits, _ = h.prefill_chunked(ids, L, capacity)
SLIDING_LAYER, GLOBAL_LAYER = 7, 5      # global layers are {5, 11, 17, 23, 29}

def poison(layer_index, position, channel=11):
    keys = snapshot[layer_index]["keys"]
    bits = E._bits(keys)
    flat = int(((0 * bits.shape[1] + 0) * bits.shape[2] + position) * bits.shape[3] + channel)
    orig = int(bits.reshape(-1)[flat].item())
    buf = mx.array(bits.reshape(-1)); buf[flat] = mx.array(orig ^ 1, dtype=bits.dtype)
    bad = [dict(l) for l in snapshot]
    bad[layer_index] = {"keys": buf.reshape(bits.shape).view(keys.dtype),
                        "values": snapshot[layer_index]["values"]}
    mx.eval(bad[layer_index]["keys"])
    _, logits, _ = h.prefill_chunked(ids, L, capacity, snapshot=bad)
    equal = E.bits_equal(cb_logits, logits)
    delta = float(mx.max(mx.abs(cb_logits.astype(mx.float32) - logits.astype(mx.float32))).item())
    return {"layer": layer_index, "position": position,
            "in_window_for_last_query": position > len(ids) - E.SLIDING_WINDOW,
            "layer_kind": "global" if layer_index in (5, 11, 17, 23, 29) else "sliding",
            "logits_bit_equal": equal, "logits_max_abs_diff": delta,
            "detected": not equal}

cases = {
    "sliding_layer_outside_window_pos3": poison(SLIDING_LAYER, 3),
    "sliding_layer_inside_window_pos1022": poison(SLIDING_LAYER, L - 2),
    "global_layer_outside_sliding_window_pos3": poison(GLOBAL_LAYER, 3),
}
result = {"experiment": "E12", "control": "positive_control_2", "prefix_length": L,
          "prompt_tokens": len(ids), "sliding_window": E.SLIDING_WINDOW,
          "window_starts_at_position": len(ids) - E.SLIDING_WINDOW, "cases": cases}
result["logit_detector_validated"] = bool(
    cases["sliding_layer_inside_window_pos1022"]["detected"]
    and cases["global_layer_outside_sliding_window_pos3"]["detected"])
result["dead_prefix_region_confirmed"] = not cases["sliding_layer_outside_window_pos3"]["detected"]

print(f"prompt {len(ids)} tokens, sliding window admits positions "
      f">= {len(ids)-E.SLIDING_WINDOW}\n")
for name, c in cases.items():
    print(f"{name:44s} layer={c['layer']:2d} {c['layer_kind']:8s} pos={c['position']:5d} "
          f"in_window={str(c['in_window_for_last_query']):5s} "
          f"detected={str(c['detected']):5s} max|d|={c['logits_max_abs_diff']:.5f}")
print(f"\nlogit detector validated : {result['logit_detector_validated']}")
print(f"dead prefix region       : {result['dead_prefix_region_confirmed']}")
Path(E.RAW / "E12_positive_control2.json").write_text(
    json.dumps({**result, "environment": bench.environment()}, indent=1, sort_keys=True, default=str))
