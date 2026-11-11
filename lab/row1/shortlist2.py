"""Round 3, part 2 (offline), second try: rotate by the hidden states' own principal directions (PCA of the captured
output-matmul inputs, fitted on the first half of the positions, tested on the second half), keep d rotated
dimensions of the output matrix as int8 (or int4) per row. Recall of the exact argmax in the top k and a certified
rate (Cauchy-Schwarz bound on the dropped dimensions plus rounding)."""
import sys, json
import numpy as np
SP = "/tmp/claude-0/-home-user-IronMule/6e4a0a70-e763-5189-b9a9-9d8ce1e4fff1/scratchpad"
sys.path.insert(0, f"{SP}/row1/llama.cpp/gguf-py")
import gguf
r = gguf.GGUFReader(sys.argv[1]); by = {t.name: t for t in r.tensors}
t = by.get("output.weight", by["token_embd.weight"])
raw = np.ascontiguousarray(t.data); N = raw.shape[0]; b = raw.reshape(N, -1, 34)
W = (b[:, :, :2].copy().view(np.float16).astype(np.float32) * b[:, :, 2:].view(np.int8).astype(np.float32)).reshape(N, -1)
K = W.shape[1]
H = np.fromfile(sys.argv[2], dtype=np.float32).reshape(-1, K); P = H.shape[0]
fit, test = H[:P // 2], H[P // 2:]
C = fit.T.astype(np.float64) @ fit.astype(np.float64); ev, V = np.linalg.eigh(C); V = V[:, ::-1].astype(np.float32); ev = ev[::-1]
energy = {d: float(ev[:d].sum() / ev.sum()) for d in (64, 128, 256, 512)}
WR = W @ V
KS = (16, 64, 256, 1024)
def rowq(A, bits):
    lim = 2 ** (bits - 1) - 1; s = np.abs(A).max(1, keepdims=True) / lim; s[s == 0] = 1
    return np.clip(np.round(A / s), -lim, lim), s[:, 0]
configs = [(d, bits) for d in (128, 256, 512) for bits in (8, 4)]
rec = {c: {k: 0 for k in KS} for c in configs}; cert = {c: {k: 0 for k in KS} for c in configs}
prep = {}
for d, bits in configs:
    q, s = rowq(WR[:, :d], bits); prep[(d, bits)] = (q * s[:, None], s, np.linalg.norm(WR[:, d:], axis=1))
for p0 in range(0, len(test), 256):
    h = test[p0:p0 + 256]; ex = h @ W.T; am = ex.argmax(1); hr = h @ V
    for (d, bits), (A, s, rest) in prep.items():
        sc = hr[:, :d] @ A.T
        bound = np.linalg.norm(hr[:, d:], axis=1)[:, None] * rest[None, :] + np.abs(hr[:, :d]).sum(1)[:, None] * (s[None, :] / 2)
        order = np.argpartition(-sc, 1024, axis=1)[:, :1024]
        osc = np.take_along_axis(sc, order, 1); order = np.take_along_axis(order, np.argsort(-osc, 1), 1)
        for k in KS:
            top = order[:, :k]; rec[(d, bits)][k] += int((top == am[:, None]).any(1).sum())
            lstar = np.take_along_axis(ex, top, 1).max(1); ub = sc + bound; np.put_along_axis(ub, top, -np.inf, 1)
            cert[(d, bits)][k] += int((ub.max(1) < lstar).sum())
T = len(test)
print(json.dumps({"test_positions": T, "pca_energy": energy,
                  "rows": [{"d": d, "bits": bits, "size_MB": (N * d * bits / 8 + N * 4) / 1e6,
                            "recall": {k: rec[(d, bits)][k] / T for k in KS}, "certified": {k: cert[(d, bits)][k] / T for k in KS}}
                           for d, bits in configs]}, indent=1))
