"""E12 positive control 3: do global layers actually read early prefix positions?

Control 2 flipped one mantissa bit at position 3 of global layer 5 and the logits
did not move at all. Two explanations: the perturbation was too small to survive
bf16 rounding through a 1194-way softmax, or the fixed cache is not letting global
layers see early positions at all. The second would be a real defect in the plan,
so it is settled with a perturbation far too large to be rounded away.
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
ids = h.render(E.build_preamble("natural", L, h.tok)
               + "\n\nMeasured evidence for this request:\n" + corpus[0])
capacity = ((len(ids) + E.MAX_NEW_TOKENS + 63)//64)*64
snapshot, _ = h.snapshot(ids, L, capacity)
_, ref_logits, _ = h.prefill_chunked(ids, L, capacity)
GLOBAL, SLIDING = 5, 7
window_start = len(ids) - E.SLIDING_WINDOW

def clobber(layer_index, position, magnitude):
    """Overwrite a whole key vector, far beyond anything bf16 could round away."""
    keys = snapshot[layer_index]["keys"]
    buf = mx.array(keys)
    buf[0, 0, position, :] = mx.full((keys.shape[3],), magnitude, dtype=keys.dtype)
    mx.eval(buf)
    bad = [dict(l) for l in snapshot]
    bad[layer_index] = {"keys": buf, "values": snapshot[layer_index]["values"]}
    _, logits, _ = h.prefill_chunked(ids, L, capacity, snapshot=bad)
    delta = float(mx.max(mx.abs(ref_logits.astype(mx.float32) - logits.astype(mx.float32))).item())
    return {"layer": layer_index,
            "layer_kind": "global" if layer_index in (5, 11, 17, 23, 29) else "sliding",
            "position": position, "magnitude": magnitude,
            "inside_sliding_window": position >= window_start,
            "detected": not E.bits_equal(ref_logits, logits), "logits_max_abs_diff": delta}

cases = {
    "global_pos3_clobbered":        clobber(GLOBAL, 3, 40.0),
    "global_pos1022_clobbered":     clobber(GLOBAL, L - 2, 40.0),
    "sliding_pos3_clobbered":       clobber(SLIDING, 3, 40.0),
    "sliding_pos1022_clobbered":    clobber(SLIDING, L - 2, 40.0),
}
result = {"experiment": "E12", "control": "positive_control_3", "prefix_length": L,
          "prompt_tokens": len(ids), "sliding_window": E.SLIDING_WINDOW,
          "window_starts_at_position": window_start, "cases": cases}
result["global_layers_read_early_positions"] = cases["global_pos3_clobbered"]["detected"]
result["sliding_layers_ignore_out_of_window"] = not cases["sliding_pos3_clobbered"]["detected"]
result["logit_detector_fires"] = any(c["detected"] for c in cases.values())

print(f"prompt {len(ids)} tokens; sliding window admits positions >= {window_start}\n")
for name, c in cases.items():
    print(f"{name:30s} {c['layer_kind']:8s} pos={c['position']:5d} "
          f"in_window={str(c['inside_sliding_window']):5s} "
          f"detected={str(c['detected']):5s} max|d|={c['logits_max_abs_diff']:9.4f}")
print(f"\nglobal layers read early positions   : {result['global_layers_read_early_positions']}")
print(f"sliding layers ignore out-of-window   : {result['sliding_layers_ignore_out_of_window']}")
print(f"logit detector fires at all           : {result['logit_detector_fires']}")
Path(E.RAW / "E12_positive_control3.json").write_text(
    json.dumps({**result, "environment": bench.environment()}, indent=1, sort_keys=True, default=str))
