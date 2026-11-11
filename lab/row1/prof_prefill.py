"""Per-op profile of one 512-token prefill decode (instrumented build, ROW1_PROF=1)."""
import ctypes, os, sys, time
sys.path.insert(0, "/home/user/IronMule")
from ironmule.ggml_engine import GgmlEngine
from ironmule.numeric_choice import TEXT
eng = GgmlEngine(sys.argv[1], speculate=False, seqs=1, seq_ctx=1024)
cpu = ctypes.CDLL(os.path.join(os.environ["LLAMA_CPP_LIB_PATH"], "libggml-cpu.so"))
ids = eng.tokenize(TEXT.read_text())[:513]
eng.chunk_nll(ids, []); cpu.row1_reset(); t0 = time.perf_counter(); n = 3
for _ in range(n): eng.chunk_nll(ids, [])
print(f"mean chunk {((time.perf_counter() - t0) / n) * 1e3:.0f} ms (incl. logits readback)")
cpu.row1_dump(sys.argv[2].encode())
rows = sorted(([l.split()[0], float(l.split()[2]) / n / 4 * 1e3, float(l.split()[3]) / n / 4 * 1e3] for l in open(sys.argv[2])), key=lambda r: -r[1] - r[2])
for r in rows[:8]: print(f"  {r[0]:12s} compute {r[1]:7.1f} ms wait {r[2]:6.1f} ms")
