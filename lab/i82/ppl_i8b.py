# One process: calibrate (I8=5) on a text that is not the gate's, then perplexity on the gate's chunks and the benchmark
# prefill for several IRONMULE_I8 modes, switched in place (ironmule_i8_set_mode)
import ctypes, glob, json, os, sys, time
import torch
from ironmule.ggml_engine import GgmlEngine
from ironmule.gguf_tokenizer import GgufTokenizer
from ironmule.numeric_choice import CHUNKS, CHUNK_TOKENS, _text_ids, gate_bos, gate_lead
from ironmule_product import compare
G = '/models/Qwen3-32B-Q4_K_M.gguf'
tok = GgufTokenizer(G)
ids, bos = _text_ids(tok), gate_bos(tok)
e = GgmlEngine(G, seqs=32, seq_ctx=2048, flash_attn=True, speculate=False)
lib = ctypes.CDLL(glob.glob(os.environ['LLAMA_CPP_LIB_PATH'] + '/libggml-cuda.so*')[0])
setm = lib.ironmule_i8_set_mode; setm.argtypes, setm.restype = [ctypes.c_int], ctypes.c_int
def nll():
    out = []
    for i in range(CHUNKS):
        chunk = ids[i * CHUNK_TOKENS:(i + 1) * CHUNK_TOKENS + 1]
        out.append(e.chunk_nll(chunk, gate_lead(chunk, bos)))
    return out
toks = sorted((e.tokenize(p) for p in compare.performance_prompts(0, 32, 32)), key=len)
def prefill():
    best = 9e9
    for rep in range(3):
        e._clear(); torch.cuda.synchronize(); t = time.perf_counter()
        e._decode([(toks[s][j], j, s, j == len(toks[s]) - 1) for s in range(len(toks)) for j in range(len(toks[s]))])
        torch.cuda.synchronize(); best = min(best, time.perf_counter() - t)
    e._clear()
    return round(best, 3)
modes = [int(m) for m in sys.argv[1].split(',')]
for m in modes:
    if m == 5:  # calibration on other text: this repository's README and RUNTIME documents
        setm(5)
        cal = tok.encode(open('README.md').read() + open('docs/RUNTIME.md').read())
        for i in range(8):
            c = cal[i * 512:(i + 1) * 512 + 1]
            if len(c) == 513:
                e.chunk_nll(c, [])
        print("PPL " + json.dumps({"i8": "calibrated", "tokens": 8 * 512}), flush=True)
        continue
    setm(m)
    print("PPL " + json.dumps({"i8": str(m), "max_k": os.environ.get("IRONMULE_I8_MAX_K", "0"), "pk5": os.environ.get("IRONMULE_PK5", "1"),
                               "nll": nll(), "prefill_s": {"32": prefill()}}), flush=True)
e.close()
