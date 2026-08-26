"""E10: the prefix cache as a runtime feature, over a 12-request session."""
import json, os, statistics, subprocess, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def workload():
    src = open(os.path.join(HERE, "e8_reuse_risk.py")).read()
    ns = {"statistics": statistics}
    exec(compile(src[src.index('PREFIX = """'):src.index('MAX_TOKENS = 32')], "w", "exec"), ns)
    return ns["PREFIX"], ns["SUFFIXES"]


def child(order):
    import mlx.core as mx
    from ironmule.runtime import Knobs, PrefixCache
    from ironmule.tune import DEFAULT_MODEL, _eos_ids, load_engine

    PREFIX, SUFFIXES = workload()
    KNOBS = Knobs(compiled_fixed_cache=True, fused_argmax=True, head_skip_prefill=True,
                  fuse_projections=True)
    out = {"pid": os.getpid(), "order": order, "arms": {}}
    for arm in order:
        engine, tok = load_engine(DEFAULT_MODEL, KNOBS)
        render = lambda t: tok.apply_chat_template([{"role": "user", "content": t}],
                                                   tokenize=False, add_generation_prompt=True)
        enc = lambda t: list(tok.encode(t, add_special_tokens=False))
        prompts = [enc(render(PREFIX + s)) for s in SUFFIXES]
        prefix_ids = enc(render(PREFIX + "@@CUT@@").split("@@CUT@@")[0])
        eos = _eos_ids(tok)

        def session():
            if arm == "prefix_cache":
                engine.prefix_cache = PrefixCache(prefix_ids)
            started = time.perf_counter_ns()
            results = [engine.generate(p, 32, eos) for p in prompts]
            return time.perf_counter_ns() - started, results

        for _ in range(2):
            session()
        cold_ns, cold = session()          # arm B: cache starts empty here
        warm_ns, warm = session()          # arm B: every request is a hit
        out["arms"][arm] = {
            "cold_session_ns": cold_ns, "warm_session_ns": warm_ns,
            "cold_ttft_ns": [r["prefill_ns"] for r in cold],
            "warm_ttft_ns": [r["prefill_ns"] for r in warm],
            "cold_tokens": [r["logical_tokens"] for r in cold],
            "warm_tokens": [r["logical_tokens"] for r in warm],
            "cache_hits": warm[-1].get("prefix_cache_hits", 0),
            "prefix_tokens": len(prefix_ids),
            "prompt_tokens": [len(p) for p in prompts],
        }
        engine.prefix_cache = None
        del engine
    out["mlx_peak_bytes"] = mx.get_peak_memory()
    return out


if __name__ == "__main__" and len(sys.argv) > 1:
    print("@@" + json.dumps(child(json.loads(sys.argv[1]))))
    raise SystemExit(0)

from ironmule import bench
from ironmule.tune import gpu_busy
assert gpu_busy() is None, gpu_busy()

ARMS = ["single_shot", "prefix_cache"]
PROCESSES = 6
children = []
for order in bench.interleave(ARMS, PROCESSES):
    proc = subprocess.run([sys.executable, os.path.abspath(__file__), json.dumps(order)],
                          capture_output=True, text=True, cwd=ROOT,
                          env={**os.environ, "PYTHONPATH": ROOT, "HF_HUB_OFFLINE": "1"})
    line = next((l for l in proc.stdout.splitlines() if l.startswith("@@")), None)
    if line is None:
        raise RuntimeError(proc.stderr[-3000:])
    children.append(json.loads(line[2:]))
    print(f"process {len(children)}/{PROCESSES} done", flush=True)

warm = {a: [c["arms"][a]["warm_session_ns"] for c in children] for a in ARMS}
cold = {a: [c["arms"][a]["cold_session_ns"] for c in children] for a in ARMS}
r_warm = bench.paired_ratio(warm["prefix_cache"], warm["single_shot"])
r_cold = bench.paired_ratio(cold["prefix_cache"], cold["single_shot"])

pc = [c["arms"]["prefix_cache"] for c in children]
plan_stable = all(a["cold_tokens"] == a["warm_tokens"] for a in pc)
across = all(a["warm_tokens"] == pc[0]["warm_tokens"] for a in pc)
ss = [c["arms"]["single_shot"] for c in children]
ss_stable = all(a["warm_tokens"] == ss[0]["warm_tokens"] for a in ss)
differing = sum(1 for x, y in zip(pc[0]["warm_tokens"], ss[0]["warm_tokens"]) if x != y)

print(f"\ncorrectness gate (within plan): cold session == warm session: {plan_stable}")
print(f"prefix_cache deterministic across 6 processes: {across}")
print(f"single_shot deterministic across 6 processes: {ss_stable}")
print(f"cache hits in the warm session: {pc[0]['cache_hits']} (prefix {pc[0]['prefix_tokens']} tokens)")
print(f"plan difference (not a gate): {differing}/12 requests differ between plans\n")
for a in ARMS:
    print(f"  {a:14s} cold session {statistics.median(cold[a])/1e6:8.1f} ms   "
          f"warm session {statistics.median(warm[a])/1e6:8.1f} ms   "
          f"warm TTFT median {statistics.median(t for c in children for t in c['arms'][a]['warm_ttft_ns'])/1e6:7.1f} ms")
for label, r in (("warm session", r_warm), ("cold session", r_cold)):
    print(f"  {label}: ratio {r['median_ratio']:.4f}  95% CI [{r['ci_low']:.4f}; {r['ci_high']:.4f}]  "
          f"({(1-r['median_ratio'])*100:+.2f}%)")
print("raw:", bench.record("E10-prefix-cache-session-ab", {
    "arms": ARMS, "processes": PROCESSES, "raw": children,
    "ratio_warm": r_warm, "ratio_cold": r_cold,
    "plan_stable": plan_stable, "deterministic_across_processes": across,
    "single_shot_deterministic": ss_stable, "requests_differing_between_plans": differing}))
