"""Round 3, part 3: how often a draft copied from the context (ironmule.speculation.draft) would be right 1, 2, 3
tokens ahead. Fixed text: truth = the text's own next tokens (teacher forcing), and for 1 ahead also the model's
greedy argmax (from the captured output-matmul inputs, `argmax.npy` if given). Facts: truth = the greedy answers
(stock answers of facts.py), context = prompt + answer so far, which is exactly what a speculation step would see."""
import json, sys
import numpy as np
sys.path.insert(0, "/home/user/IronMule")
import ironmule.ggml_engine as ge
from ironmule.speculation import draft
from ironmule.numeric_choice import CHUNKS, CHUNK_TOKENS, TEXT
ge.GgmlEngine._warm_up = lambda self: None
eng = ge.GgmlEngine(sys.argv[1], speculate=False, seqs=1, seq_ctx=256)
def rates(seqs, start_of):  # seqs: list of token lists; positions from start_of(seq) on
    hit = np.zeros(3); avail = 0; n = 0
    for ids, s0 in seqs:
        for i in range(max(s0, 1), len(ids)):
            d, _ = draft(ids[:i], 3); n += 1
            if not d: continue
            avail += 1
            for j in range(3):
                if j < len(d) and i + j < len(ids) and all(d[x] == ids[i + x] for x in range(j + 1)): hit[j] += 1
    return {"positions": n, "draft_available": avail / n, "hit_1": hit[0] / n, "hit_2": hit[1] / n, "hit_3": hit[2] / n}
text = "".join(l for l in TEXT.read_text(encoding="utf-8").splitlines(True) if not l.startswith("#"))
ids = eng.tokenize(text)[:CHUNKS * CHUNK_TOKENS + 1]
chunks = [(ids[c * CHUNK_TOKENS:(c + 1) * CHUNK_TOKENS + 1], 1) for c in range(CHUNKS)]
out = {"fixed_text": rates(chunks, None)}
if len(sys.argv) > 3:
    am = np.load(sys.argv[3]); hits = n = 0
    for c in range(CHUNKS):
        ch = ids[c * CHUNK_TOKENS:(c + 1) * CHUNK_TOKENS + 1]
        for i in range(1, CHUNK_TOKENS):  # model argmax after ch[:i+1] is am[c*512 + i]
            d, _ = draft(ch[:i + 1], 1); n += 1
            hits += bool(d) and d[0] == int(am[c * CHUNK_TOKENS + i])
    out["fixed_text_vs_model_argmax_hit_1"] = hits / n
facts = json.load(open(sys.argv[2]))["answers"]
seqs = []
for a in facts:
    p = eng.tokenize(f"<|im_start|>user\n{a['q']} /no_think<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    seqs.append((p + eng.tokenize(a["a"]), len(p)))
out["fact_answers"] = rates(seqs, None)
print(json.dumps(out, indent=1))
