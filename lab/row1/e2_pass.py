"""E2: full-pass timing of the ROW1 AMX kernel on Qwen3's real Q8_0 weights, variants A and B, R rows."""
import ctypes, os, sys, time, json
import numpy as np
SP = "/tmp/claude-0/-home-user-IronMule/6e4a0a70-e763-5189-b9a9-9d8ce1e4fff1/scratchpad"
sys.path.insert(0, f"{SP}/row1/llama.cpp/gguf-py")
import gguf

gguf_path = sys.argv[1]; huge = int(os.environ.get("ROW1_HUGE", "1"))
variants = [int(v) for v in os.environ.get("ROW1_VARIANTS", "0,1").split(",")]
RS = [int(x) for x in os.environ.get("ROW1_RS", "1,2,4,8,16,32,48").split(",")]
T = int(os.environ.get("ROW1_T", "4")); REPS = int(os.environ.get("ROW1_REPS", "7"))
lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libgemm.so"))
lib.row1_alloc.restype = ctypes.c_void_p; lib.row1_alloc.argtypes = [ctypes.c_size_t, ctypes.c_int]
lib.row1_size_a.restype = lib.row1_size_b.restype = ctypes.c_size_t
lib.row1_run.restype = ctypes.c_double
vp, ci = ctypes.c_void_p, ctypes.c_int
lib.row1_pack_a.argtypes = [vp, ci, ci, vp]; lib.row1_pack_b.argtypes = [vp, ci, ci, vp, vp]
lib.row1_size_a.argtypes = lib.row1_size_b.argtypes = [ci, ci]
lib.row1_packx_b.argtypes = lib.row1_packx_a.argtypes = [vp, ci, ci, vp]
lib.row1_quant_a.argtypes = lib.row1_quant_b.argtypes = [vp, ci, ci, vp, vp]
lib.row1_run.argtypes = [vp, ci, ci, ci, vp, vp, vp, ci]
class Mat(ctypes.Structure):
    _fields_ = [("W", ctypes.c_void_p), ("ws", ctypes.c_void_p), ("N", ctypes.c_int), ("K", ctypes.c_int)]
assert lib.row1_init(T) == 0
r = gguf.GGUFReader(gguf_path)
by = {t.name: t for t in r.tensors}
nl = max(int(n.split(".")[1]) for n in by if n.startswith("blk.")) + 1
names = [f"blk.{i}.{s}.weight" for i in range(nl) for s in
         ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")]
names.append("output.weight" if "output.weight" in by else "token_embd.weight")
if os.environ.get("ROW1_CACHED"):  # in-cache control: one small matrix (blk.0.attn_k) repeated, so no DRAM streaming
    names = ["blk.0.attn_k.weight"]
packed = {0: [], 1: []}; keep = []
t0 = time.time(); total_w = 0
for n in names:
    t = by[n]; K, N = int(t.shape[0]), int(t.shape[1]); src = np.ascontiguousarray(t.data)
    assert t.tensor_type.name == "Q8_0" and N % 16 == 0 and K % 64 == 0
    total_w += N * K
    if 0 in variants:
        a = lib.row1_alloc(lib.row1_size_a(N, K), huge); lib.row1_pack_a(src.ctypes.data, N, K, ctypes.c_void_p(a))
        packed[0].append(Mat(a, None, N, K))
    if 1 in variants:
        b = lib.row1_alloc(lib.row1_size_b(N, K), huge); s = np.zeros(N, np.float32); keep.append(s)
        lib.row1_pack_b(src.ctypes.data, N, K, ctypes.c_void_p(b), s.ctypes.data); packed[1].append(Mat(b, s.ctypes.data, N, K))
if os.environ.get("ROW1_CACHED"):
    packed = {v: packed[v] * 197 for v in packed}
print(f"packed {len(names)} matrices, {total_w / 1e9:.3f}e9 weights in {time.time() - t0:.0f}s", flush=True)
Kmax = max(m.K for m in (packed[0] or packed[1])); Nmax = max(m.N for m in (packed[0] or packed[1]))
rng = np.random.default_rng(0)
X = rng.integers(-127, 128, size=(48, Kmax), dtype=np.int8)
xs = rng.random(48 * Kmax // 32, dtype=np.float32) * 0.01 + 0.001
Y = np.zeros((48, Nmax), np.float32)
Xp = np.zeros(48 * Kmax, np.int8)  # packed activations for B; one Kmax layout serves every K up to Kmax only
# for timing, B reads its activations as [mt][K/64][16][64] with the matrix's own K; values are irrelevant
res = {"gguf": os.path.basename(gguf_path), "huge": huge, "threads": T, "weights": total_w}
for v in variants:
    arr = (Mat * len(packed[v]))(*packed[v]); nbytes = sum(lib.row1_size_a(m.N, m.K) if v == 0 else m.N * m.K + 4 * m.N for m in packed[v])
    res[f"bytes_{'AB'[v]}"] = nbytes
    for R in RS:
        XX = Xp
        lib.row1_run(ctypes.cast(arr, vp), len(packed[v]), v, R, XX.ctypes.data, xs.ctypes.data, Y.ctypes.data, 1)
        ts = sorted(lib.row1_run(ctypes.cast(arr, vp), len(packed[v]), v, R, XX.ctypes.data, xs.ctypes.data, Y.ctypes.data, 1) for _ in range(REPS))
        med = ts[len(ts) // 2]
        res[f"{'AB'[v]}_{R}"] = med * 1e3
        print(f"variant {'AB'[v]} R={R:2d}: {med * 1e3:6.1f} ms/pass (min {ts[0]*1e3:5.1f} max {ts[-1]*1e3:5.1f})  {nbytes / med / 1e9:5.1f} GB/s  {R * total_w / med:.2e} MAC/s", flush=True)
with open(f"/proc/{os.getpid()}/smaps_rollup") as f:
    res["AnonHugePages_kB"] = [int(l.split()[1]) for l in f if l.startswith("AnonHugePages")][0]
print(json.dumps(res))
