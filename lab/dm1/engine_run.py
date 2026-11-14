"""DM1 through the engine's run loop: N chat requests at once (no system prompt, 128 tokens, greedy), with and without
the draft model; tokens/s, mean first token, the draft model's summary, and how many answers are token-identical.
Usage: engine_run.py TARGET.gguf DRAFT.gguf N[,N...]"""
import json, sys, time
sys.path.insert(0, "/home/user/IronMule")
from ironmule.ggml_engine import GgmlEngine, Job
target, draft, NS = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(",")]
ASK = ["Write a short story about a lighthouse keeper.", "Explain how a refrigerator keeps food cold.",
       "Write a Python function that checks whether a string is a palindrome, with a docstring.",
       "List five facts about the planet Mars.", "What causes the seasons on Earth? Answer in a paragraph.",
       "Write a JSON object describing a book with title, author, year and three tags.",
       "Summarize the plot of Romeo and Juliet in a few sentences.", "Give step by step instructions to make tea."]
answers = {}
for mode in (("plain", "draft") if len(sys.argv) < 5 else ("draft",)):
    e = GgmlEngine(target, seqs=8, seq_ctx=2048, speculate=True, flash_attn=True, shared=True, cap=16,
                   draft_gguf=draft if mode == "draft" else None)
    e.collective = None  # the comparison is the draft model's alone
    ids = [e.tokenize(f"<|im_start|>user\n{q} /no_think<|im_end|>\n<|im_start|>assistant\n") for q in ASK]
    e.run([Job(ids[0], 16)])
    for n in NS:
        rows = []
        for rep in range(8 // n if n < 8 else 1):
            jobs = [Job(ids[(rep * n + i) % 8], 128) for i in range(n)]
            t0 = time.monotonic_ns(); e.run(jobs); wall = (time.monotonic_ns() - t0) / 1e9
            rows.append((sum(len(j.tokens) for j in jobs), wall, sum(j.first_ns - t0 for j in jobs) / n / 1e9))
            answers.setdefault((n, rep), {})[mode] = [j.tokens for j in jobs]
        print(mode, "N", n, "tok/s", round(sum(r[0] for r in rows) / sum(r[1] for r in rows), 2), "first",
              round(sum(r[2] for r in rows) / len(rows), 3), flush=True)
    print(mode, json.dumps(e.summary().get("draft_model") or e.summary()["speculation"])[:700], flush=True)
    e.close()
if "plain" not in next(iter(answers.values())): sys.exit(0)
same = sum(a == b for v in answers.values() for a, b in zip(v["plain"], v["draft"]))
print("identical answers", same, "of", sum(len(v["plain"]) for v in answers.values()))
