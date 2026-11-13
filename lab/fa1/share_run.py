"""SP1 through the engine's run loop: N chat-like jobs (a shared system prompt + own question), greedy, all at once,
then a second wave that finds the first wave's cells kept; split sequences (prefix reuse per sequence) against the
shared pool. Prints tokens/s, mean first token, the pool summary and how many answers are token-identical.
Usage: share_run.py GGUF N TOKENS"""
import json, sys, time
sys.path.insert(0, "/home/user/IronMule")
from ironmule.ggml_engine import GgmlEngine, Job
gguf, n, limit = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
system = open("/home/user/IronMule/lab/fa1/system.txt").read()
out = {}
for mode in ("split", "shared"):
    e = GgmlEngine(gguf, seqs=16, seq_ctx=2048, speculate=True, flash_attn=True, shared=(mode == "shared"), cap=n)
    def prompt(i):
        return e.tokenize(f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\nName one service idea number {i} "
                          f"for the library. /no_think<|im_end|>\n<|im_start|>assistant\n")
    waves = []
    for w in range(2):
        jobs = [Job(prompt(w * n + i), limit) for i in range(n)]
        t0 = time.monotonic_ns()
        e.run(jobs)
        wall = (time.monotonic_ns() - t0) / 1e9
        waves.append({"tok_s": round(sum(len(j.tokens) for j in jobs) / wall, 1),
                      "ttft": round(sum(j.first_ns - t0 for j in jobs) / len(jobs) / 1e9, 3), "wall": round(wall, 1)})
        out.setdefault(mode, []).append([j.tokens for j in jobs])
    print(mode, json.dumps(waves), json.dumps(e.summary().get("shared_pool") or e.summary()["prefix_reuse"]), flush=True)
    e.close()
same = sum(a == b for wa, wb in zip(out["split"], out["shared"]) for a, b in zip(wa, wb))
print("identical answers", same, "of", 2 * n)
