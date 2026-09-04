"""Akzeptanz und Speedup auf echtem Projektinhalt.

Mit Aufwaermlauf und Wiederholungen: ein erster Lauf nach dem Modellladen zahlt
Allokation und Kernelaufbau und kam in einer frueheren Fassung dieser Messung als
54 % Speedup heraus, obwohl in dem Lauf gar nicht spekuliert wurde. Median aus
mehreren Durchlaeufen, Arme abwechselnd, damit thermische Drift beide gleich trifft.
"""
import sys, json, statistics
sys.path.insert(0, "tools")
from pathlib import Path
from _bench import BudgetGuard, resolve_local_model_snapshot, require_ac_power
from friday_hardware import HardwareProfile, speculative_generate
from mlx_lm import load
from mlx_lm.sample_utils import make_sampler

REPS = 2
TOKENS = 64

require_ac_power(); g = BudgetGuard()
def charge(w):
    g.record_gpu(w)
    for _ in range(int(-(-(w*(1-0.15)/0.15)//4))): g.required_break()

rows = []
WANT = sys.argv[1] if len(sys.argv) > 1 else "4b"
for pf, tag in (("profiles/m1max_gemma-3-4b-4bit-g64.json", "4B"),
                ("profiles/m1max_gemma-3-1b-4bit-g64.json", "1B")):
    if WANT.lower() not in tag.lower():
        continue
    prof = HardwareProfile.load(Path(pf))
    snap = resolve_local_model_snapshot(prof.model_id)
    m, tok = load(str(snap.path)); sampler = make_sampler(temp=0.0)
    for f in sorted(Path("experiments/prompt_lookup/real").glob("*.txt")):
        txt = tok.apply_chat_template([{"role": "user", "content": f.read_text()}],
                                      add_generation_prompt=True)
        ids = list(txt if isinstance(txt, list) else tok.encode(txt))
        # Aufwaermen, Ergebnis verworfen.
        charge(speculative_generate(m, sampler, ids, max_tokens=16, draft_length=0,
                                    ngram=prof.lookup_ngram).seconds)
        base, fix, ada = [], [], []
        last = None
        for _ in range(REPS):
            for bucket, kw in ((base, dict(ngram=prof.lookup_ngram, draft_length=0)),
                               (fix,  dict(profile=prof, adapt=False)),
                               (ada,  dict(profile=prof, adapt=True))):
                r = speculative_generate(m, sampler, ids, max_tokens=TOKENS, **kw)
                charge(r.seconds); bucket.append(r.seconds)
                if bucket is ada: last = r
                if bucket is base and last is None: last_base = r.tokens
        b, fx, ad = (statistics.median(x) for x in (base, fix, ada))
        rows.append({
            "model": tag, "prompt": f.stem, "prompt_tokens": len(ids),
            "greedy_ms": round(b * 1000, 1),
            "fixed_speedup": round(b / fx, 3),
            "adaptive_speedup": round(b / ad, 3),
            "acceptance": None if last.acceptance is None else round(last.acceptance, 3),
            "declined_steps": last.declined_steps,
            "unrewindable_steps": last.unrewindable_steps,
            "greedy_spread_pct": round((max(base) / min(base) - 1) * 100, 2),
        })
        print(rows[-1], flush=True)
    del m, tok
Path(f"experiments/prompt_lookup/real/results_{WANT.lower()}.json").write_text(
    json.dumps({"reps": REPS, "tokens": TOKENS, "rows": rows,
                "budget": {k: v for k, v in g.summary().items() if "limit" not in k}},
               indent=2))
