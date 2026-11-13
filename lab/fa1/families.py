"""FA1 on other model families (Gemma 3: head size 256 and sliding-window layers; Llama 3.2: head size 64; Mistral 7B):
(1) logits of four sequences and a decode step under llama.cpp without flash attention (the reference), llama.cpp's
flash attention (FA1=0) and FA1, each in its own process; (2) perplexity on the numeric gate's text (tokenized by the
GGUF itself, 16 chunks) for the reference and FA1, with the gate's ratio interval; (3) full-load decode steps at 8
sequences x 64 and x 1,024 tokens for the three. Usage: families.py GGUF OUT.json"""
import json, os, statistics, subprocess, sys, time
sys.path.insert(0, "/home/user/IronMule")
gguf, out = sys.argv[1], sys.argv[2]
CHILD = r'''
import json, os, statistics, sys, time
sys.path.insert(0, "/home/user/IronMule")
import numpy as np
from ironmule.ggml_engine import GgmlEngine, CHECK_TEXT
from ironmule.numeric_choice import CHUNKS, CHUNK_TOKENS, TEXT
cfg = json.loads(sys.argv[2])
e = GgmlEngine(sys.argv[1], seqs=8, seq_ctx=2048, speculate=False, flash_attn=cfg["fa"])
res = {}
base = e.tokenize(CHECK_TEXT * 60)
def logits(rows):
    b = e.batch; b.n_tokens = len(rows)
    for i, (t, p, s, l) in enumerate(rows):
        b.token[i], b.pos[i], b.n_seq_id[i], b.logits[i] = t, p, 1, int(l); b.seq_id[i][0] = s
    assert e.L.llama_decode(e.ctx, b) == 0
    return [np.ctypeslib.as_array(e.L.llama_get_logits_ith(e.ctx, i), shape=(e.n_vocab,)).copy() for i, r in enumerate(rows) if r[3]]
e._clear(); rows = []
for s, n in enumerate((700, 300, 50, 5)):
    rows += logits([(t, j, s, j >= n - 3) for j, t in enumerate(base[:n])])
rows += logits([(base[n], n, s, True) for s, n in enumerate((700, 300, 50, 5))])
np.save(cfg["npy"], np.stack(rows))
if cfg.get("nll"):
    text = "".join(l for l in TEXT.read_text(encoding="utf-8").splitlines(True) if not l.startswith("#"))
    ids = e.tokenize(text)
    bos = max(0, int(e.L.llama_vocab_bos(e.vocab)))
    res["nll"] = [e.chunk_nll(ids[i * CHUNK_TOKENS:(i + 1) * CHUNK_TOKENS + 1], [bos]) for i in range(CHUNKS)
                  if len(ids) >= (i + 1) * CHUNK_TOKENS + 1]
for ctx in (64, 1024):
    prompt = (base * 20)[:ctx]
    e._clear()
    for s in range(8):
        e._decode([(t, j, s, j == ctx - 1) for j, t in enumerate(prompt)])
    times = []
    for r in range(3):
        t0 = time.perf_counter()
        for k in range(8):
            e._decode([(prompt[k % ctx], ctx + r * 8 + k, s, True) for s in range(8)])
        times.append(time.perf_counter() - t0)
    res[f"step_8x{ctx}"] = statistics.median(times) / 8
e.close()
print(json.dumps(res))
'''
import numpy as np
from ironmule.numeric_choice import ratio_interval
work = os.path.dirname(out)
res = {"gguf": os.path.basename(gguf)}
arms = {"nofa": ({"fa": False, "nll": True}, "1"), "ggml_fa": ({"fa": True}, "0"), "fa1": ({"fa": True, "nll": True}, "1")}
for name, (cfg, fa1) in arms.items():
    cfg = {**cfg, "npy": os.path.join(work, f"{res['gguf']}.{name}.npy")}
    done = subprocess.run([sys.executable, "-c", CHILD, gguf, json.dumps(cfg)], env={**os.environ, "FA1": fa1},
                          capture_output=True, text=True)
    if done.returncode:
        res[name] = {"error": done.stderr[-400:]}
        continue
    res[name] = json.loads(done.stdout.strip().splitlines()[-1])
    print(name, {k: v for k, v in res[name].items() if k != "nll"}, flush=True)
ref = np.load(os.path.join(work, f"{res['gguf']}.nofa.npy"))
for name in ("ggml_fa", "fa1"):
    x = np.load(os.path.join(work, f"{res['gguf']}.{name}.npy"))
    res[name]["logits_vs_nofa"] = {"max": float(np.abs(x - ref).max()), "mean": float(np.abs(x - ref).mean()),
                                   "argmax_same": float((x.argmax(1) == ref.argmax(1)).mean())}
res["fa1_gate"] = ratio_interval(res["nofa"]["nll"], res["fa1"]["nll"])
for name in arms:
    res[name].pop("nll", None)
print(json.dumps({k: res[k] for k in ("fa1_gate",)}), json.dumps(res["fa1"]["logits_vs_nofa"]),
      json.dumps(res["ggml_fa"]["logits_vs_nofa"]), flush=True)
open(out, "w").write(json.dumps(res, indent=1))
