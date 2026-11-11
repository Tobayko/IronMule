"""Part 2: fact check. ~50 questions with short checkable answers, greedy, chat template with /no_think, at most
48 tokens, decoded 8 at a time; correct when an expected string is in the answer (case-insensitive)."""
import json, sys
sys.path.insert(0, "/home/user/IronMule")
import ironmule.ggml_engine as ge
Q = [("What is the capital of France?", ["paris"]), ("What is the capital of Japan?", ["tokyo"]),
     ("What is the capital of Australia?", ["canberra"]), ("What is the capital of Canada?", ["ottawa"]),
     ("What is the capital of Italy?", ["rome"]), ("What is the capital of Egypt?", ["cairo"]),
     ("What is the capital of Spain?", ["madrid"]), ("What is the capital of Germany?", ["berlin"]),
     ("What is the capital of Kenya?", ["nairobi"]), ("What is the capital of Brazil?", ["brasília", "brasilia"]),
     ("What is 17 times 3?", ["51"]), ("What is 12 times 12?", ["144"]), ("What is 250 plus 375?", ["625"]),
     ("What is 1000 minus 357?", ["643"]), ("What is 81 divided by 9?", ["9"]), ("What is 15 squared?", ["225"]),
     ("What is the square root of 196?", ["14"]), ("What is 7 times 8?", ["56"]), ("What is 2 to the power of 10?", ["1024", "1,024"]),
     ("What is 45 plus 38?", ["83"]),
     ("In what year did World War II end?", ["1945"]), ("In what year did the Berlin Wall fall?", ["1989"]),
     ("In what year did humans first land on the Moon?", ["1969"]), ("In what year did Columbus first reach the Americas?", ["1492"]),
     ("In what year did the French Revolution begin?", ["1789"]), ("In what year was the US Declaration of Independence signed?", ["1776"]),
     ("What is the chemical symbol for gold?", ["au"]), ("What is the chemical symbol for sodium?", ["na"]),
     ("How many planets are in the Solar System?", ["8", "eight"]), ("What planet is known as the Red Planet?", ["mars"]),
     ("What gas do plants absorb from the air for photosynthesis?", ["carbon dioxide", "co2", "co₂"]),
     ("At what temperature in Celsius does water boil at sea level?", ["100"]), ("What is the largest planet in the Solar System?", ["jupiter"]),
     ("How many bones are in the adult human body?", ["206"]), ("What is the chemical formula of water?", ["h2o", "h₂o"]),
     ("What is the hardest natural substance?", ["diamond"]), ("How many legs does an insect have?", ["6", "six"]),
     ("What is the atomic number of carbon?", ["6", "six"]),
     ("How many centimeters are in a meter?", ["100"]), ("How many meters are in a kilometer?", ["1000", "1,000"]),
     ("How many minutes are in an hour?", ["60"]), ("How many seconds are in a minute?", ["60"]),
     ("How many grams are in a kilogram?", ["1000", "1,000"]), ("How many days are in a leap year?", ["366"]),
     ("How many hours are in a day?", ["24"]), ("How many millimeters are in a centimeter?", ["10"]),
     ("Who wrote Romeo and Juliet?", ["shakespeare"]), ("Who painted the Mona Lisa?", ["leonardo", "da vinci"]),
     ("What is the longest river in South America?", ["amazon"]), ("What ocean lies between Europe and North America?", ["atlantic"])]
ge.GgmlEngine._warm_up = lambda self: None
eng = ge.GgmlEngine(sys.argv[1], speculate=False, seqs=8, seq_ctx=256, max_rows=64)
L = eng.L
import ctypes
def text(ids):
    out = bytearray()
    for t in ids:
        b = (ctypes.c_char * 128)(); n = L.llama_token_to_piece(eng.vocab, t, b, 128, 0, False); out += bytes(b[:max(n, 0)])
    return out.decode(errors="replace")
answers = []
for start in range(0, len(Q), 8):
    group = Q[start:start + 8]
    L.llama_memory_clear(eng.memory, True)
    prompts = [eng.tokenize(f"<|im_start|>user\n{q} /no_think<|im_end|>\n<|im_start|>assistant\n") for q, _ in group]
    nxt = [eng._prefill(p, s) for s, p in enumerate(prompts)]
    pos = [len(p) for p in prompts]; out = [[t] for t in nxt]; live = [not eng.eog(t) for t in nxt]
    for _ in range(47):
        act = [s for s in range(len(group)) if live[s]]
        if not act: break
        picks = eng._decode([(out[s][-1], pos[s], s, True) for s in act])
        for s, t in zip(act, picks):
            pos[s] += 1
            if eng.eog(t): live[s] = False
            else: out[s].append(t)
    for s, (q, exp) in enumerate(group):
        a = text([t for t in out[s] if not eng.eog(t)]).replace("<think>", "").replace("</think>", "").strip()
        answers.append({"q": q, "a": a, "ok": any(e in a.lower() for e in exp)})
print(json.dumps({"correct": sum(x["ok"] for x in answers), "n": len(answers), "answers": answers}, ensure_ascii=False))
