"""Round 4: per-chunk NLL of the first 4 chunks of the fixed text scored two ways: one 512-row decode (M=512 path) and
one token per call (M=1 path), to tell which matmul path changes a result."""
import json, math, sys
import numpy as np
sys.path.insert(0, "/home/user/IronMule")
from ironmule.ggml_engine import GgmlEngine
from ironmule.numeric_choice import CHUNK_TOKENS, TEXT
eng = GgmlEngine(sys.argv[1], speculate=False, seqs=1, seq_ctx=1024)
text = "".join(l for l in TEXT.read_text(encoding="utf-8").splitlines(True) if not l.startswith("#"))
ids = eng.tokenize(text); L = eng.L; n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
big, one = [], []
for c in range(n):
    ch = ids[c * CHUNK_TOKENS:(c + 1) * CHUNK_TOKENS + 1]
    big.append(eng.chunk_nll(ch, []))
    L.llama_memory_clear(eng.memory, True); tot = 0.0
    for i in range(CHUNK_TOKENS):
        b = eng.batch; b.n_tokens = 1; b.token[0], b.pos[0], b.n_seq_id[0], b.logits[0] = ch[i], i, 1, 1; b.seq_id[0][0] = 0
        assert L.llama_decode(eng.ctx, b) == 0
        row = np.ctypeslib.as_array(L.llama_get_logits_ith(eng.ctx, 0), shape=(eng.n_vocab,)).astype(np.float64)
        top = row.max(); tot += float(top + math.log(np.exp(row - top).sum()) - row[ch[i + 1]])
    one.append(tot / CHUNK_TOKENS)
print(json.dumps({"M512": big, "M1": one, "ppl_M512": math.exp(sum(big) / n), "ppl_M1": math.exp(sum(one) / n)}))
