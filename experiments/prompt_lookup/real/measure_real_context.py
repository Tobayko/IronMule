"""Akzeptanz auf echtem Projektinhalt statt auf nachgebautem."""
import sys, json, glob
sys.path.insert(0, "tools")
from pathlib import Path
from _bench import BudgetGuard, resolve_local_model_snapshot, require_ac_power
from friday_hardware import HardwareProfile, speculative_generate
from mlx_lm import load
from mlx_lm.sample_utils import make_sampler

require_ac_power(); g = BudgetGuard()
def charge(w):
    g.record_gpu(w)
    for _ in range(int(-(-(w*(1-0.15)/0.15)//4))): g.required_break()

files = sorted(glob.glob("experiments/prompt_lookup/real/*.txt"))
out = []
for pf in ("profiles/m1max_gemma-3-4b-4bit-g64.json",
           "profiles/m1max_gemma-3-1b-4bit-g64.json"):
    profile = HardwareProfile.load(Path(pf))
    snap = resolve_local_model_snapshot(profile.model_id)
    m, tok = load(str(snap.path))
    sampler = make_sampler(temp=0.0)
    tag = "1B" if "3-1b" in pf else "4B"
    for f in files:
        prompt = Path(f).read_text()
        txt = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      add_generation_prompt=True)
        ids = list(txt if isinstance(txt, list) else tok.encode(txt))
        plain = speculative_generate(m, sampler, ids, max_tokens=96,
                                     ngram=profile.lookup_ngram, draft_length=0)
        charge(plain.seconds)
        fast = speculative_generate(m, sampler, ids, max_tokens=96, profile=profile)
        charge(fast.seconds)
        row = {"model": tag, "prompt": Path(f).stem, "prompt_tokens": len(ids),
               "greedy_ms": round(plain.seconds*1000, 1),
               "fast_ms": round(fast.seconds*1000, 1),
               "speedup": round(plain.seconds/fast.seconds, 3),
               "acceptance": None if fast.acceptance is None else round(fast.acceptance, 3),
               "tokens_per_step": round(fast.tokens_per_step, 3),
               "identical": plain.tokens == fast.tokens,
               "unrewindable_steps": fast.unrewindable_steps}
        out.append(row); print(row, flush=True)
    del m, tok
Path("experiments/prompt_lookup/real/results.json").write_text(
    json.dumps({"rows": out, "budget": {k: v for k, v in g.summary().items()
                                        if "limit" not in k}}, indent=2))
