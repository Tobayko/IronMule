"""KV1 probe: llama.cpp's 8-bit KV cache (q8_0) against f16 under flash attention, Qwen3-4B Q8_0 on this CPU.
(1) full-load decode rate per width at the default request length 2,048 (`measure_widths`), (2) perplexity ratio on the
numeric gate's chunks through the multi-position decode (as `gate` scores a llama.cpp option).
Usage: probe.py GGUF SNAPSHOT OUT.json"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, "/home/user/IronMule")
from ironmule.ggml_engine import GgmlEngine, measure_widths, memory_bytes
from ironmule.numeric_choice import CHUNKS, CHUNK_TOKENS, gate_lead, ratio_interval, _text_ids, gate_bos
from mlx_lm.tokenizer_utils import load
gguf, snap, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
res = {"memory": memory_bytes()}
tok = load(Path(snap)); ids = _text_ids(tok)[:CHUNKS * CHUNK_TOKENS + 1]; bos = gate_bos(tok)
def nll(settings):
    e = GgmlEngine(gguf, speculate=False, seqs=1, **settings); rows = []
    for i in range(CHUNKS):
        c = ids[i * CHUNK_TOKENS:(i + 1) * CHUNK_TOKENS + 1]
        if len(c) < CHUNK_TOKENS + 1: break
        rows.append(e.chunk_nll(c, gate_lead(c, bos)))
    e.close(); return rows
t = time.perf_counter()
base = nll({"flash_attn": True}); q8 = nll({"flash_attn": True, "kv_q8": True}); default = nll({})
res["ppl"] = {"q8_vs_fa": ratio_interval(base, q8), "q8_vs_default": ratio_interval(default, q8),
              "fa_vs_default": ratio_interval(default, base), "s": time.perf_counter() - t}
print(json.dumps(res["ppl"]), flush=True)
for name, st, widths in (("fa_f16", {"flash_attn": True}, (8, 16)), ("fa_q8", {"flash_attn": True, "kv_q8": True}, (8, 16, 32))):
    t = time.perf_counter(); res[name] = measure_widths(gguf, st, 2048, widths); res[name]["s"] = time.perf_counter() - t
    print(name, json.dumps(res[name]), flush=True)
out.write_text(json.dumps(res, indent=1))
