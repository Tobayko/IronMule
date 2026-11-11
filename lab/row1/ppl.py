"""Part 2: perplexity on the project's fixed text (numeric_choice: 16 chunks of 512 tokens, scored through one
multi-position decode each, GgmlEngine.chunk_nll), through the library at LLAMA_CPP_LIB_PATH."""
import json, math, sys, time
sys.path.insert(0, "/home/user/IronMule")
from ironmule.ggml_engine import GgmlEngine
from ironmule.numeric_choice import CHUNKS, CHUNK_TOKENS, TEXT
eng = GgmlEngine(sys.argv[1], speculate=False, seqs=1, seq_ctx=1024)
text = "".join(l for l in TEXT.read_text(encoding="utf-8").splitlines(True) if not l.startswith("#"))
ids = eng.tokenize(text)
t0 = time.perf_counter()
nll = [eng.chunk_nll(ids[i * CHUNK_TOKENS:(i + 1) * CHUNK_TOKENS + 1], []) for i in range(CHUNKS)]
dt = time.perf_counter() - t0
print(json.dumps({"nll": nll, "ppl": math.exp(sum(nll) / len(nll)), "seconds": dt, "tokens": len(ids)}))
