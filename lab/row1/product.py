"""Round 5 part 1: IronMule's product (ProductService + worker + ggml engine) end to end on a GGUF, with the library
at LLAMA_CPP_LIB_PATH and the ROW1_* variables passed through to the worker. Usage: product.py GGUF MODEL_ID OUT
[N,...]. Runs the engine's quality gate, then one worker for all blocks: per N one warm-up block and 3 blocks of N
streamed requests at once (64 tokens, serve.py's prompts), then the 50 facts (48 tokens, /no_think)."""
import json, os, sys, tempfile, threading, time
from pathlib import Path
ROOT = Path("/home/user/IronMule"); sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(ROOT / "training"), str(ROOT / "lab/row1")]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import gg1_measure
from ironmule.ggml_engine import gate
from ironmule_product.backend import MLXWorkerClient
from ironmule_product.service import ProductService
from ironmule_product.state import ProductStore
from ironmule_product.types import GenerationRequest
gguf, model, out_path = sys.argv[1], sys.argv[2], Path(sys.argv[3])
NS = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "1,8,16,32").split(",")]
src = open(ROOT / "lab/row1/serve.py").read()
TOPICS = eval(src[src.index("TOPICS = [") + 9:src.index("]", src.index("TOPICS = [")) + 1])
facts_src = open(ROOT / "lab/row1/facts.py").read()
Q = eval(facts_src[facts_src.index("Q = [") + 4:facts_src.index("]\nge.") + 1])
spec = gg1_measure.model_spec(model)
t0 = time.perf_counter()
saved = {k: os.environ.pop(k) for k in ("IRONMULE_GGML_SEQS", "IRONMULE_GGML_SEQ_CTX") if k in os.environ}
checked = gate(spec.snapshot_path, gguf, cache=out_path.parent / f"gate_{out_path.stem}")  # gate at the default size
os.environ.update(saved)
res = {"gguf": os.path.basename(gguf), "env": {k: v for k, v in os.environ.items() if k.startswith(("ROW1_", "IRONMULE_GGML"))},
       "gate": {k: checked.get(k) for k in ("ran", "passes", "ratio", "interval", "settings", "reason", "seconds")},
       "gate_tried": checked.get("tried"), "gate_s": time.perf_counter() - t0}
print(json.dumps(res["gate"]), flush=True)
store = ProductStore(Path(tempfile.mkdtemp(prefix="row1p-")) / "state"); store.setup("server"); store.set_request_timeout(3600)
store.register_model(spec)
backend = MLXWorkerClient(spec, execution_variant="current_engine", engine_configuration="ggml", ggml_gguf=gguf,
                          ggml_gate=checked, speculate=True, startup_timeout=1200)
backend.start(); service = ProductService(store, backend=backend, spec=spec)
def ask(text, limit, t0, out, i):
    req = GenerationRequest(spec.model_id, (("user", text),), max_tokens=limit, request_id=f"r{time.monotonic_ns()}-{i}", stream=True)
    first, pieces, tokens = None, [], None
    for ev in service.stream(req):
        if not isinstance(ev, dict): continue
        if ev.get("type") == "token":
            first = first if first is not None else time.perf_counter() - t0; pieces.append(ev.get("text") or "")
        if ev.get("type") == "done": tokens = ev.get("completion_tokens")
    out[i] = {"first_s": first, "tokens": tokens or len(pieces), "text": "".join(pieces)}
def block(texts, limit):
    out = [None] * len(texts); t0 = time.perf_counter()
    th = [threading.Thread(target=ask, args=(texts[i], limit, t0, out, i)) for i in range(len(texts))]
    for t in th: t.start()
    for t in th: t.join()
    wall = time.perf_counter() - t0
    return {"tok_s": sum(o["tokens"] for o in out) / wall, "ttft_mean_s": sum(o["first_s"] or wall for o in out) / len(out),
            "wall_s": wall, "sample": out[0]["text"][:80], "answers": [o["text"] for o in out]}
try:
    for n in NS:
        texts = [f"Write a short story about {TOPICS[i % len(TOPICS)]}. /no_think" for i in range(n)]
        block(texts, 64); res[f"N{n}"] = [block(texts, 64) for _ in range(3)]
        print(n, [round(b["tok_s"], 1) for b in res[f"N{n}"]], [round(b["ttft_mean_s"], 2) for b in res[f"N{n}"]], flush=True)
        for b in res[f"N{n}"]: b.pop("answers")
    answers = []
    for s in range(0, len(Q), 8):
        grp = Q[s:s + 8]; b = block([f"{q} /no_think" for q, _ in grp], 48)
        for (q, exp), a in zip(grp, b["answers"]):
            a = a.replace("<think>", "").replace("</think>", "").strip(); answers.append({"q": q, "a": a, "ok": any(e in a.lower() for e in exp)})
    res["facts"] = {"correct": sum(a["ok"] for a in answers), "answers": answers}
    print("facts", res["facts"]["correct"], flush=True)
    res["health_ggml"] = service.health().get("ggml")
finally:
    service.close()
out_path.write_text(json.dumps(res))
