"""Round 5 part 7: prefix reuse in the ggml engine. Usage: prefix_check.py GGUF MODEL_ID GATE_JSON.
Greedy answers of 8 chat prompts (shared template prefix, different topics), 48 tokens, each twice, with reuse on and
off; first-token times of single requests; answers compared."""
import json, os, sys, time
from pathlib import Path
ROOT = Path("/home/user/IronMule"); sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import gg1_measure
from mlx_lm.tokenizer_utils import load
from ironmule.ggml_engine import GgmlEngine, Job
gguf, model, gate = sys.argv[1], sys.argv[2], json.loads(open(sys.argv[3]).read())
tok = load(Path(gg1_measure.model_spec(model).snapshot_path))
sys_prompt = "You are a careful assistant. Answer briefly and correctly. " * 6
topics = ["a lighthouse keeper", "a lost robot", "a mountain village", "a clockmaker", "a beekeeper", "a moon base",
          "a violin maker", "a postman"]
prompts = [tok._tokenizer.apply_chat_template([{"role": "system", "content": sys_prompt},
           {"role": "user", "content": f"Write a short story about {t}. /no_think"}], add_generation_prompt=True,
           tokenize=True, return_dict=False) for t in topics]
print("prompt tokens", [len(p) for p in prompts], flush=True)
eng = GgmlEngine(gguf, speculate=True, **gate.get("settings", {}))
out = {}
for reuse in (False, True):
    eng.prefix_reuse = reuse; eng._clear(); eng.prompt_tokens = eng.reused_tokens = 0
    firsts, answers = [], []
    for rnd in range(2):
        for p in prompts:  # one at a time: first-token time of each
            t0 = time.perf_counter(); f = []
            j = Job(list(p), 48, emit=lambda t: f.append(time.perf_counter() - t0) if not f else None)
            eng.run([j]); firsts.append(f[0]); answers.append(list(j.tokens))
    t0 = time.perf_counter(); jobs = [Job(list(p), 48) for p in prompts]; eng.run(jobs); burst = time.perf_counter() - t0
    out[reuse] = answers + [list(j.tokens) for j in jobs]
    print(f"reuse={reuse}: first token mean {sum(firsts)/len(firsts):.3f} s (round 1 {sum(firsts[:8])/8:.3f}, round 2 "
          f"{sum(firsts[8:])/8:.3f}); burst of 8: {sum(len(j.tokens) for j in jobs)/burst:.1f} tok/s; "
          f"reused {eng.reused_tokens} of {eng.prompt_tokens} prompt tokens", flush=True)
same = sum(a == b for a, b in zip(out[False], out[True]))
print(f"answers equal with and without reuse: {same} of {len(out[True])}")
for a, b in zip(out[False], out[True]):
    if a != b:
        n = next(i for i in range(min(len(a), len(b))) if a[i] != b[i]) if any(x != y for x, y in zip(a, b)) else min(len(a), len(b))
        print("  first difference at token", n, repr(tok.decode(a)[:60]), "|", repr(tok.decode(b)[:60]))
eng.close()
