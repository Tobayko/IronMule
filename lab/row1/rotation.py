"""Round 4 part 3 (offline): does a fixed Hadamard rotation make per-row int8 activations (variant B) accurate?
Real inputs of 9 matmuls per model (captured with the hook, one 512-token chunk, last 256 rows used). Relative
Frobenius error against float64 on the dequantized Q8_0 weights for: llama's Q8_0 arithmetic (= per-32 quantized
activations times the Q8_0 weights), B (per-row int8 W and x), B16 (per-row int8 W, 16-bit x), rotated B
(W R and R^T x, both per-row int8; R = random signs times block Hadamard, block = largest power of two dividing K)."""
import glob, os, sys
import numpy as np
SP = "/tmp/claude-0/-home-user-IronMule/6e4a0a70-e763-5189-b9a9-9d8ce1e4fff1/scratchpad"
sys.path.insert(0, f"{SP}/row1/llama.cpp/gguf-py")
import gguf
r = gguf.GGUFReader(sys.argv[1]); by = {t.name: t for t in r.tensors}; cap = sys.argv[2]
def deq(t):
    raw = np.ascontiguousarray(t.data); N = raw.shape[0]; b = raw.reshape(N, -1, 34)
    return (b[:, :, :2].copy().view(np.float16).astype(np.float64) * b[:, :, 2:].view(np.int8)).reshape(N, -1)
def rowq(a, bits):
    lim = 2 ** (bits - 1) - 1 if bits == 8 else 32639; s = np.abs(a).max(-1, keepdims=True) / lim; s[s == 0] = 1
    return np.round(a / s) * s
def q8_0(x):
    b = x.reshape(x.shape[0], -1, 32); d = (np.abs(b).max(-1, keepdims=True) / 127).astype(np.float16).astype(np.float64)
    return (np.round(b / np.where(d == 0, 1, d)) * d).reshape(x.shape)
def hadamard(n):
    H = np.ones((1, 1))
    while H.shape[0] < n: H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)
def rot(K):
    b = K & -K
    while b > 2048: b //= 2
    H = hadamard(b); R = np.kron(np.eye(K // b), H)
    s = np.random.default_rng(1).choice([-1.0, 1.0], K)
    return s[:, None] * R  # R = diag(s) . blockH ; orthogonal
def rel(a, ref): return np.linalg.norm(a - ref) / np.linalg.norm(ref)
print(f"{'tensor':22s} {'llama Q8_0':>10s} {'B':>9s} {'B16':>9s} {'rot B':>9s} {'rot B16':>9s}  kurtosis x / x rotated")
for f in sorted(glob.glob(f"{cap}/*.f32")):
    name = os.path.basename(f)[:-4]; W = deq(by[name]); K = W.shape[1]
    x = np.fromfile(f, dtype=np.float32).reshape(-1, K)[-256:].astype(np.float64)
    ref = x @ W.T
    Wr = rowq(W, 8); R = rot(K); xr = x @ R; WR = rowq(W @ R, 8)
    errs = [rel(q8_0(x) @ W.T, ref), rel(rowq(x, 8) @ Wr.T, ref), rel(rowq(x, 16) @ Wr.T, ref),
            rel(rowq(xr, 8) @ WR.T, ref), rel(rowq(xr, 16) @ WR.T, ref)]
    kur = lambda a: float((np.abs(a).max(1) / a.std(1)).mean())
    print(f"{name:22s} " + " ".join(f"{e:9.2e}" for e in errs) + f"   max/std {kur(x):5.1f} / {kur(xr):4.1f}", flush=True)
