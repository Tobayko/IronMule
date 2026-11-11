"""Step 4 check: greedy answers of 4 prompts decoded together (M=4 rows per call, prefill M=prompt length) through the
library at LLAMA_CPP_LIB_PATH; prints tokens and text so two settings (ROW1_GGML=0/1) can be compared."""
import json, os, sys
sys.path.insert(0, "/home/user/IronMule")
import ironmule.ggml_engine as ge
ge.GgmlEngine._warm_up = lambda self: None
eng = ge.GgmlEngine(sys.argv[1], speculate=False, seqs=4, seq_ctx=256, max_rows=64)
qs = ["What is the capital of France? Answer in one sentence.", "How many legs does a spider have?",
      "Who wrote the play Romeo and Juliet?", "What is 17 times 3?"]
prompts = [eng.tokenize(f"<|im_start|>user\n{q} /no_think<|im_end|>\n<|im_start|>assistant\n") for q in qs]
nxt = [eng._prefill(p, s) for s, p in enumerate(prompts)]
pos = [len(p) for p in prompts]; out = [[t] for t in nxt]
for _ in range(40):
    nxt = eng._decode([(nxt[s], pos[s], s, True) for s in range(4)])
    for s in range(4): out[s].append(nxt[s]); pos[s] += 1
L = eng.L
def text(ids):
    buf = bytearray()
    for t in ids:
        b = (ctypes_buf := (__import__("ctypes").c_char * 64)()); n = L.llama_token_to_piece(eng.vocab, t, b, 64, 0, False)
        buf += bytes(b[:max(n, 0)])
    return buf.decode(errors="replace")
print(json.dumps({"tokens": out, "text": [text(o) for o in out]}, ensure_ascii=False))
