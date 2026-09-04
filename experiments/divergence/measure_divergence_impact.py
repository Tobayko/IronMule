"""Aendert die Prefill-Zerteilung die Antwort oder nur die Formulierung?

Zyklus 2 zeigte, dass sie in rund 26 % der Faelle andere Token erzeugt. Ob das eine
andere Antwort bedeutet, war offen und wurde dem Nutzer ohne Daten vorgelegt. Dieser
Lauf beschafft die fehlende Zahl.

Er schlaegt KEINE Lockerung des Korrektheitsvertrags vor. Antwortgleichheit ist kein
Ersatz fuer Tokenidentitaet, sondern eine zusaetzliche Beobachtung.

Vorregistrierung: experiments/divergence/PREREGISTRATION.md
"""
import sys, json, re, time
sys.path.insert(0, "tools")
from pathlib import Path
from _bench import BudgetGuard, resolve_local_model_snapshot, require_ac_power
from measure_self_consistency import hard_problems, INSTRUCTION, extract
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler
import mlx.core as mx

PROBLEMS, GEN, CHUNK = 10, 160, 512
PREAMBLE = ("You are a careful engineering assistant. Read the problem, work through "
            "it step by step, and state the final answer exactly as instructed. "
            "Do not skip arithmetic steps. Do not restate the problem. ") * 28

require_ac_power()
guard = BudgetGuard()
_debt = 0.0
def charge(sec):
    global _debt
    guard.record_gpu(sec); _debt += sec * (1 - 0.15) / 0.15
    while _debt >= 4.0:
        guard.required_break(); _debt -= 4.0

snap = resolve_local_model_snapshot("mlx-community/gemma-3-4b-it-4bit")
model, tok = load(str(snap.path))
sampler = make_sampler(temp=0.0)

def run(ids, chunk):
    cache = make_prompt_cache(model)
    size = len(ids) if chunk is None else chunk
    logits = None
    for start in range(0, len(ids), size):
        at = time.perf_counter()
        logits = model(mx.array([ids[start:start + size]]), cache=cache)
        mx.eval(logits); mx.synchronize(); charge(time.perf_counter() - at)
    at = time.perf_counter()
    y = sampler(logits[:, -1, :].astype(mx.float32))[:, None]; mx.eval(y)
    out = [int(y[0, 0])]
    for _ in range(GEN - 1):
        logits = model(y, cache=cache)
        y = sampler(logits[:, -1, :].astype(mx.float32))[:, None]; mx.eval(y)
        out.append(int(y[0, 0]))
    mx.synchronize(); charge(time.perf_counter() - at)
    return out

rows = []
for question, truth in hard_problems(PROBLEMS):
    txt = tok.apply_chat_template(
        [{"role": "user", "content": PREAMBLE + "\n\n" + question + INSTRUCTION}],
        add_generation_prompt=True)
    ids = list(txt if isinstance(txt, list) else tok.encode(txt))
    ref = run(ids, None)
    alt = run(ids, CHUNK)
    same_tokens = ref == alt
    first = next((i for i, (a, b) in enumerate(zip(ref, alt)) if a != b), None)
    differing = sum(1 for a, b in zip(ref, alt) if a != b)
    ans_ref, ans_alt = extract(tok.decode(ref)), extract(tok.decode(alt))
    rows.append({"prompt_tokens": len(ids), "truth": truth,
                 "tokens_identical": same_tokens, "first_diff": first,
                 "differing_tokens": differing,
                 "answer_reference": ans_ref, "answer_chunked": ans_alt,
                 "answer_same": ans_ref == ans_alt,
                 "reference_correct": ans_ref == truth,
                 "chunked_correct": ans_alt == truth})
    print(rows[-1], flush=True)

diverged = [r for r in rows if not r["tokens_identical"]]
res = {"candidate_id": "divergence-impact-20260824-01", "formal_claim": False,
       "chunk": CHUNK, "generate_tokens": GEN, "problems": PROBLEMS, "rows": rows,
       "token_divergence_rate": round(len(diverged) / len(rows), 3),
       "answer_same_among_diverged": (
           None if not diverged else
           round(sum(r["answer_same"] for r in diverged) / len(diverged), 3)),
       "answer_extractable_both": sum(
           r["answer_reference"] is not None and r["answer_chunked"] is not None
           for r in rows),
       "accuracy_reference": round(sum(r["reference_correct"] for r in rows) / len(rows), 3),
       "accuracy_chunked": round(sum(r["chunked_correct"] for r in rows) / len(rows), 3),
       "budget": {k: v for k, v in guard.summary().items() if "limit" not in k}}
Path("experiments/divergence/results.json").write_text(json.dumps(res, indent=2))
print("\n", {k: v for k, v in res.items() if k not in ("rows", "budget")})
