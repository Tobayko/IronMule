"""E2 accuracy: ROW1 kernels (A: exact Q8_0 semantics, B: per-row int8) against float math on the dequantized weights,
with llama.cpp's own Q8_0 arithmetic (emulated in numpy) as the reference error. Activations: real embedding rows of
16 real tokens -> RMSNorm x the layer's norm weight (q/k/v/gate/up input); ffn_down input = SiLU(gate) * up computed in
float from that. Not hidden states of a full forward (deeper layers have larger outliers)."""
import ctypes, os, sys
import numpy as np
SP = "/tmp/claude-0/-home-user-IronMule/6e4a0a70-e763-5189-b9a9-9d8ce1e4fff1/scratchpad"
sys.path.insert(0, f"{SP}/row1/llama.cpp/gguf-py")
import gguf
lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libgemm.so"))
vp, ci = ctypes.c_void_p, ctypes.c_int
lib.row1_alloc.restype = vp; lib.row1_alloc.argtypes = [ctypes.c_size_t, ci]
lib.row1_size_a.restype = lib.row1_size_b.restype = ctypes.c_size_t
lib.row1_size_a.argtypes = lib.row1_size_b.argtypes = [ci, ci]
lib.row1_pack_a.argtypes = [vp, ci, ci, vp]; lib.row1_pack_b.argtypes = [vp, ci, ci, vp, vp]
lib.row1_quant_a.argtypes = lib.row1_quant_b.argtypes = [vp, ci, ci, vp, vp]; lib.row1_packx_b.argtypes = lib.row1_packx_a.argtypes = [vp, ci, ci, vp]
lib.row1_run.restype = ctypes.c_double; lib.row1_run.argtypes = [vp, ci, ci, ci, vp, vp, vp, ci]
class Mat(ctypes.Structure):
    _fields_ = [("W", vp), ("ws", vp), ("N", ci), ("K", ci)]
lib.row1_init(4)
r = gguf.GGUFReader(sys.argv[1]); by = {t.name: t for t in r.tensors}

def deq(t):
    raw = np.ascontiguousarray(t.data); N = raw.shape[0]
    b = raw.reshape(N, -1, 34); d = b[:, :, :2].copy().view(np.float16).astype(np.float32)
    return (d * b[:, :, 2:].view(np.int8).astype(np.float32)).reshape(N, -1)

def llama_q8(x):  # ggml quantize_row_q8_0 (ref): d = amax/127 stored as f16, q = round(x / d)
    R, K = x.shape; b = x.reshape(R, -1, 32); amax = np.abs(b).max(-1, keepdims=True)
    d = amax / 127; idv = np.where(d > 0, 1 / np.where(d > 0, d, 1), 0)
    q = np.round(b * idv).astype(np.int8); return q, d.astype(np.float16).astype(np.float32)

def llama_ref(x, t):
    raw = np.ascontiguousarray(t.data); N = raw.shape[0]; b = raw.reshape(N, -1, 34)
    dw = b[:, :, :2].copy().view(np.float16).astype(np.float64)[..., 0]; qw = b[:, :, 2:].view(np.int8).astype(np.int64)
    qx, dx = llama_q8(x); qx = qx.astype(np.int64); dx = dx[..., 0].astype(np.float64)
    dots = np.einsum("mbk,nbk->mnb", qx, qw)
    return (dots * dx[:, None, :] * dw[None, :, :]).sum(-1)

def ours(x, t, v):
    raw = np.ascontiguousarray(t.data); K, N = int(t.shape[0]), int(t.shape[1]); R = x.shape[0]
    if v == 0:
        W = lib.row1_alloc(lib.row1_size_a(N, K), 0); lib.row1_pack_a(raw.ctypes.data, N, K, W); ws = None
        q = np.zeros((R, K), np.int8); s = np.zeros(R * K // 32, np.float32)
        lib.row1_quant_a(np.ascontiguousarray(x, np.float32).ctypes.data, R, K, q.ctypes.data, s.ctypes.data)
        X = np.zeros(48 * K, np.int8); lib.row1_packx_a(q.ctypes.data, R, K, X.ctypes.data)
    else:
        W = lib.row1_alloc(lib.row1_size_b(N, K), 0); wsa = np.zeros(N, np.float32); ws = wsa.ctypes.data
        lib.row1_pack_b(raw.ctypes.data, N, K, W, ws)
        q = np.zeros((R, K), np.int8); s = np.zeros(R, np.float32)
        lib.row1_quant_b(np.ascontiguousarray(x, np.float32).ctypes.data, R, K, q.ctypes.data, s.ctypes.data)
        X = np.zeros(48 * K, np.int8); lib.row1_packx_b(q.ctypes.data, R, K, X.ctypes.data)
    m = (Mat * 1)(Mat(W, ws, N, K)); Y = np.zeros((48, N), np.float32)
    lib.row1_run(ctypes.cast(m, vp), 1, v, R, X.ctypes.data, s.ctypes.data, Y.ctypes.data, 1)
    return Y[:R].astype(np.float64)

ids = [785, 6722, 315, 9625, 374, 12095, 13, 576, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562]
emb = deq(by["token_embd.weight"])[ids]
def rms(x, w): return x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6) * w
def rel(a, ref): return float(np.linalg.norm(a - ref) / np.linalg.norm(ref))
print(f"{'tensor':24s} {'llama Q8_0':>11s} {'ours A':>9s} {'ours B':>9s} {'A vs llama':>11s}  (relative Frobenius error vs float64 on dequantized W)")
for L in (0, 14, 27):
    xa = rms(emb, by[f"blk.{L}.attn_norm.weight"].data); xf = rms(emb, by[f"blk.{L}.ffn_norm.weight"].data)
    g = xf @ deq(by[f"blk.{L}.ffn_gate.weight"]).T; u = xf @ deq(by[f"blk.{L}.ffn_up.weight"]).T
    xd = g / (1 + np.exp(-g)) * u
    for name, x in (("attn_q", xa), ("attn_k", xa), ("ffn_up", xf), ("ffn_down", xd)):
        t = by[f"blk.{L}.{name}.weight"]; x = x.astype(np.float32)
        ref = x.astype(np.float64) @ deq(t).astype(np.float64).T
        ll, a, b = llama_ref(x, t), ours(x, t, 0), ours(x, t, 1)
        print(f"blk.{L}.{name:16s} {rel(ll, ref):11.2e} {rel(a, ref):9.2e} {rel(b, ref):9.2e} {rel(a, ll):11.2e}", flush=True)
x = rms(emb, by["output_norm.weight"].data).astype(np.float32); t = by.get("output.weight", by["token_embd.weight"])
ref = x.astype(np.float64) @ deq(t).astype(np.float64).T
ll, a, b = llama_ref(x, t), ours(x, t, 0), ours(x, t, 1)
print(f"{'output':24s} {rel(ll, ref):11.2e} {rel(a, ref):9.2e} {rel(b, ref):9.2e} {rel(a, ll):11.2e}")
print("argmax agreement with float over 16 rows: llama", int((ll.argmax(1) == ref.argmax(1)).sum()), "A", int((a.argmax(1) == ref.argmax(1)).sum()), "B", int((b.argmax(1) == ref.argmax(1)).sum()))
