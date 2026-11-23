"""HC1 probe: does the CPU threadpool's busy-wait slow the server's Python work while the GPU decodes?

One process: the engine decodes 32 sequences in a thread while the main thread renders the benchmark's 32 prompts
with the worker's tokenizer over and over (the work the worker does before each group's prefill). Prints the render
time idle and during decoding, and the decode step time, for the threadpool setting in the environment
(IRONMULE_GGML_THREADS, IRONMULE_GGML_POLL)."""
import json, os, statistics as st, sys, threading, time

from ironmule.ggml_engine import GgmlEngine, THREADS, POLL
from ironmule.gguf_tokenizer import GgufTokenizer
from ironmule_product import compare

gguf = sys.argv[1]
tok = GgufTokenizer(gguf)
convs = [[{"role": "user", "content": p}] for p in compare.performance_prompts(0, 32, 32)]


def render_all():
    return [tok.apply_chat_template(c, add_generation_prompt=True, tokenize=True, enable_thinking=False) for c in convs]


idle = []
for _ in range(5):
    t = time.perf_counter(); render_all(); idle.append(time.perf_counter() - t)
ids = sorted(render_all(), key=len)
e = GgmlEngine(gguf, seqs=32, seq_ctx=2048, flash_attn=True, speculate=False)
e._clear()
e._decode([(ids[s][j], j, s, j == len(ids[s]) - 1) for s in range(32) for j in range(len(ids[s]))])
steps, busy, stop = [], [], threading.Event()


def decode():
    pos = {s: len(ids[s]) for s in range(32)}
    for k in range(200):
        t = time.perf_counter()
        e._decode([(8, pos[s], s, True) for s in range(32)])
        steps.append(time.perf_counter() - t)
        for s in pos:
            pos[s] += 1
        if stop.is_set():
            break


for _ in range(5):  # decode alone
    t = time.perf_counter(); e._decode([(8, len(ids[s]) + 300 + _, s, True) for s in range(32)]); steps.append(time.perf_counter() - t)
alone = st.median(steps[1:]); steps.clear()
e._clear()
e._decode([(ids[s][j], j, s, j == len(ids[s]) - 1) for s in range(32) for j in range(len(ids[s]))])
th = threading.Thread(target=decode); th.start()
time.sleep(0.5)
while th.is_alive() and len(busy) < 20:
    t = time.perf_counter(); render_all(); busy.append(time.perf_counter() - t)
stop.set(); th.join()
print("HC1 " + json.dumps({"threads": THREADS, "poll": POLL, "cpus": os.cpu_count(),
                           "render_idle_ms": round(1e3 * min(idle), 2), "render_while_decoding_ms": round(1e3 * st.median(busy), 2),
                           "step_alone_ms": round(1e3 * alone, 2), "step_while_rendering_ms": round(1e3 * st.median(steps), 2)}), flush=True)
e.close()
