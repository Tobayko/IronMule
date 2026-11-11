"""Round 3, part 2 (offline): vocabulary shortlist for the output matmul. Hidden states = the output matmul's input,
captured by the hook (ROW1_DUMP) while scoring the fixed text (16 x 512 positions). Exact = float32 on the
dequantized Q8_0 output matrix. For each sketch: recall of the exact argmax in the sketch's top k, sketch size, and
for the rotation sketches the share of positions where a per-row error bound proves the top-k shortlist holds it."""
import sys, json, time
import numpy as np
SP = "/tmp/claude-0/-home-user-IronMule/6e4a0a70-e763-5189-b9a9-9d8ce1e4fff1/scratchpad"
sys.path.insert(0, f"{SP}/row1/llama.cpp/gguf-py")
import gguf
gg, dump = sys.argv[1], sys.argv[2]
r = gguf.GGUFReader(gg); by = {t.name: t for t in r.tensors}
t = by.get("output.weight", by["token_embd.weight"])
raw = np.ascontiguousarray(t.data); N = raw.shape[0]; b = raw.reshape(N, -1, 34)
W = (b[:, :, :2].copy().view(np.float16).astype(np.float32) * b[:, :, 2:].view(np.int8).astype(np.float32)).reshape(N, -1)
K = W.shape[1]
H = np.fromfile(dump, dtype=np.float32).reshape(-1, K)
print(f"{H.shape[0]} positions, W {N}x{K}", flush=True)
KS = (16, 64, 256, 1024)
def rowq(A, bits):
    lim = 2 ** (bits - 1) - 1; s = np.abs(A).max(1, keepdims=True) / lim; s[s == 0] = 1
    return np.clip(np.round(A / s), -lim, lim) * s, s[:, 0]
sk = {}
q4, _ = rowq(W, 4); sk["int4 per row"] = (q4, N * K / 2 + N * 2)
sg = np.sign(W) * np.abs(W).mean(1, keepdims=True); sk["sign bits x row mean"] = (sg, N * K / 8 + N * 2)
q2, _ = rowq(W, 2); sk["int2 per row"] = (q2, N * K / 4 + N * 2)
del sg, q2
t0 = time.time(); G = W.T.astype(np.float64) @ W.astype(np.float64); ev, V = np.linalg.eigh(G); V = V[:, ::-1].astype(np.float32)
print(f"eigh {time.time() - t0:.0f}s; energy in top 128/256/512 dims: " +
      " ".join(f"{ev[::-1][:d].sum() / ev.sum():.3f}" for d in (128, 256, 512)), flush=True)
WR = W @ V
rot = {}
for d in (128, 256, 512):
    q, s = rowq(WR[:, :d], 8)
    rest = np.linalg.norm(WR[:, d:], axis=1)
    rot[d] = (q, s, rest); sk[f"rotation d={d} int8"] = (None, N * d + N * 4)
res = {name: {k: 0 for k in KS} for name in sk}; cert = {d: {k: 0 for k in KS} for d in rot}
P = H.shape[0]
for p0 in range(0, P, 256):
    h = H[p0:p0 + 256]; ex = h @ W.T; am = ex.argmax(1); n = len(h)
    hr = h @ V
    for name, (A, _) in sk.items():
        if A is None:
            d = int(name.split("=")[1].split()[0]); q, s, rest = rot[d]; sc = hr[:, :d] @ q.T
            bound = np.linalg.norm(hr[:, d:], axis=1)[:, None] * rest[None, :] + np.abs(hr[:, :d]).sum(1)[:, None] * (s[None, :] / 2)
        else:
            sc = h @ A.T
        order = np.argpartition(-sc, max(KS), axis=1)[:, :max(KS)]
        osc = np.take_along_axis(sc, order, 1); order = np.take_along_axis(order, np.argsort(-osc, 1), 1)
        for k in KS:
            top = order[:, :k]; res[name][k] += int((top == am[:, None]).any(1).sum())
            if A is None:
                lstar = np.take_along_axis(ex, top, 1).max(1)
                ub = sc + bound; np.put_along_axis(ub, top, -np.inf, 1)
                cert[d][k] += int((ub.max(1) < lstar).sum())
    print(f"{p0 + n} positions done", flush=True)
out = {"positions": P, "recall": {n: {k: v / P for k, v in r_.items()} for n, r_ in res.items()},
       "size_MB": {n: v[1] / 1e6 for n, v in sk.items()}, "certified": {d: {k: v / P for k, v in c.items()} for d, c in cert.items()},
       "full_matrix_MB_q8_0": N * K / 32 * 34 / 1e6}
print(json.dumps(out, indent=1))
