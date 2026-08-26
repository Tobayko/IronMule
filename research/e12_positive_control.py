"""E12 positive control: does the comparison actually detect a difference?

26/26 PASS is consistent with reuse being exact AND with the detector being broken.
This flips exactly one bit in one element of one layer of the snapshot and requires
the harness to fail, and to localise the failure correctly. Not part of the matrix;
it validates the instrument, per preregistration class HARNESS_OR_STATE_FAILURE.
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
prompts = [h.render(preamble + "\n\nMeasured evidence for this request:\n" + s) for s in corpus[:1]]
ids = prompts[0]
capacity = ((len(ids) + E.MAX_NEW_TOKENS + 63)//64)*64
snapshot, _ = h.snapshot(ids, L, capacity)

# --- control 1: the detector must accept an untouched snapshot ---------------
cb_state, cb_logits, _ = h.prefill_chunked(ids, L, capacity)
cr_state, cr_logits, _ = h.prefill_chunked(ids, L, capacity, snapshot=snapshot)
clean_logits = E.bits_equal(cb_logits, cr_logits)
clean_kv = all(E.bits_equal(x, y) for x, y in
               zip(E.valid_kv(cb_state, len(ids)), E.valid_kv(cr_state, len(ids))))

# --- control 2: flip one bit in layer 7 keys, position 3, channel 11 ---------
LAYER, POS, CH = 7, 3, 11
keys = snapshot[LAYER]["keys"]
bits = E._bits(keys)
flat = int(((0 * bits.shape[1] + 0) * bits.shape[2] + POS) * bits.shape[3] + CH)
orig = int(bits.reshape(-1)[flat].item())
poisoned_flat = mx.array(bits.reshape(-1))
poisoned_flat[flat] = mx.array(orig ^ 1, dtype=bits.dtype)   # single-bit flip, lowest mantissa bit
poisoned = poisoned_flat.reshape(bits.shape).view(keys.dtype)
mx.eval(poisoned)
bad_snapshot = [dict(layer) for layer in snapshot]
bad_snapshot[LAYER] = {"keys": poisoned, "values": snapshot[LAYER]["values"]}

pb_state, pb_logits, _ = h.prefill_chunked(ids, L, capacity, snapshot=bad_snapshot)
poisoned_logits_equal = E.bits_equal(cb_logits, pb_logits)
pb_kv = E.valid_kv(pb_state, len(ids))
cb_kv = E.valid_kv(cb_state, len(ids))
kv_flags = [E.bits_equal(x, y) for x, y in zip(cb_kv, pb_kv)]
poisoned_kv_equal = all(kv_flags)
first_bad = next((i for i, ok in enumerate(kv_flags) if not ok), None)
detail = E.first_difference(cb_kv[first_bad], pb_kv[first_bad]) if first_bad is not None else None
hash_clean = E.bit_hash(cb_kv)
hash_poisoned = E.bit_hash(pb_kv)

result = {
    "experiment": "E12", "control": "positive_control", "prefix_length": L, "capacity": capacity,
    "clean_logits_bit_equal": clean_logits, "clean_kv_bit_equal": clean_kv,
    "poisoned_bit": {"layer": LAYER, "tensor": "keys", "token_position": POS, "channel": CH,
                     "bits_before": orig, "bits_after": orig ^ 1},
    "poisoned_logits_bit_equal": poisoned_logits_equal,
    "poisoned_kv_bit_equal": poisoned_kv_equal,
    "detected_first_bad_tensor_index": first_bad,
    "detected_layer": (first_bad // 2) if first_bad is not None else None,
    "detected_tensor": ("keys" if first_bad is not None and first_bad % 2 == 0 else "values"),
    "detection_detail": detail,
    "hash_clean": hash_clean, "hash_poisoned": hash_poisoned,
    "hashes_differ": hash_clean != hash_poisoned,
}
result["control_passed"] = bool(
    clean_logits and clean_kv                      # detector accepts identical input
    and not poisoned_kv_equal                      # and rejects a single flipped bit
    and result["detected_layer"] == LAYER
    and result["detected_tensor"] == "keys"
    and result["hashes_differ"])

print(json.dumps({k: v for k, v in result.items() if k != "detection_detail"}, indent=1, default=str))
print("detection_detail:", json.dumps(detail, default=str))
print("\nCONTROL", "PASSED" if result["control_passed"] else "FAILED")
Path(E.RAW / "E12_positive_control.json").write_text(
    json.dumps({**result, "environment": bench.environment()}, indent=1, sort_keys=True, default=str))
