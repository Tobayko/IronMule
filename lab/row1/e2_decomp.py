"""Where variant B's error comes from (numpy only): per-row int8 weights with float activations, and Q8_0 weights with
per-row int8 activations, and per-row int8 weights with 16-bit (hi/lo int8) per-row activations."""
import sys, numpy as np
sys.argv += []
exec(open(__file__.replace("e2_decomp.py", "e2_acc.py")).read().split("ids = [")[0].split("lib = ctypes")[0])
r = gguf.GGUFReader(sys.argv[1]); by = {t.name: t for t in r.tensors}
def deq(t):
    raw = np.ascontiguousarray(t.data); N = raw.shape[0]
    b = raw.reshape(N, -1, 34); d = b[:, :, :2].copy().view(np.float16).astype(np.float32)
    return (d * b[:, :, 2:].view(np.int8).astype(np.float32)).reshape(N, -1)
def rowq(a, bits=8):
    lim = 2 ** (bits - 1) - 1; s = np.abs(a).max(-1, keepdims=True) / lim; s[s == 0] = 1
    return np.round(a / s) * s
ids = [785, 6722, 315, 9625, 374, 12095, 13, 576, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562]
emb = deq(by["token_embd.weight"])[ids].astype(np.float64)
def rms(x, w): return x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6) * w
def rel(a, ref): return float(np.linalg.norm(a - ref) / np.linalg.norm(ref))
print("tensor             Wrow8+Xf   Wq8_0+Xrow8  Wrow8+Xrow16  Wrow8+Xq8_0(blocks)")
for L in (0, 14, 27):
    xa = rms(emb, by[f"blk.{L}.attn_norm.weight"].data); xf = rms(emb, by[f"blk.{L}.ffn_norm.weight"].data)
    g = xf @ deq(by[f"blk.{L}.ffn_gate.weight"]).T; u = xf @ deq(by[f"blk.{L}.ffn_up.weight"]).T
    xd = g / (1 + np.exp(-np.clip(g, -60, 60))) * u
    for name, x in (("attn_q", xa), ("ffn_up", xf), ("ffn_down", xd)):
        W = deq(by[f"blk.{L}.{name}.weight"]).astype(np.float64); ref = x @ W.T
        Wr = rowq(W); xb = x.reshape(16, -1, 32)
        xq8 = (np.round(xb / (np.abs(xb).max(-1, keepdims=True) / 127)) * (np.abs(xb).max(-1, keepdims=True) / 127)).reshape(x.shape)
        print(f"blk.{L}.{name:10s} {rel(x @ Wr.T, ref):9.2e} {rel(rowq(x) @ W.T, ref):11.2e} {rel(rowq(x, 16) @ Wr.T, ref):12.2e} {rel(xq8 @ Wr.T, ref):12.2e}", flush=True)
