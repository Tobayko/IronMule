"""PI3 probe on a Kaggle T4: where the Pilot's decisions can make an LLM on a GPU faster, before any GPU training.

One engine (Qwen3-4B Q4_K_M, every layer on GPU 0, shared pool, 16 x 2,048 cells, cap 32, PAD1, flash attention),
MG2's CUDA build (the dataset `ironmule-mg2-cuda-build`, no kg1 hook: the generated kernels stay off in every arm).
Each arm is a decision source swapped into the same engine between runs:

  defaults   the engine without the Pilot (all host cores, every request it can run, every waiting prompt)
  v6         the shipped Pilot (version 2, world v6), as the engine opens it on a GPU
  fixed:*    one decision held for every step (host threads, running cap, prompt budget, draft)

Parts (each cell twice, in alternating arm order):

  A  host CPU load on a GPU server: 0-3 busy processes or one busy half of every 20 ms on the 4 vCPUs, threads 1 / 2 /
     4 and v6, 1 / 8 / 32 requests at once, 64 tokens
  B  arriving requests: "chat" (48 short prompts, Poisson 4/s, answers 64-192 tokens) and "mixed" (48 prompts in
     clumps of 4 at 3/s, 30 % with a ~1,200-token document, answers 32-160); first token counted from arrival
  C  a draft model (Qwen3-0.6B Q8_0, DM1): defaults (drafts on), drafts off, v6; 1 and 8 requests and "chat"
  D  answers: the 50 facts of lab/row1/facts.py at 8 at a time, defaults against v6, checked as MG2 checks them

Every engine step's (decode rows, prompt tokens, attended tokens, seconds) is kept for the GPU dream world.
Writes /kaggle/working/pi3_probe.json. MG2_COMMIT names the commit to fetch.
"""
import json, math, multiprocessing as mp, os, random, sys, time
from pathlib import Path

COMMIT = os.environ["MG2_COMMIT"]
LOCAL = os.environ.get("PI3_LOCAL")  # a CPU smoke of this file: the repository, a local build and models in PI3_LOCAL
W = Path(LOCAL or "/kaggle/working")
SMOKE = bool(LOCAL)
OUT = {"commit": COMMIT, "started": time.time(), "A": [], "B": [], "C": [], "steps": {}}
CELL_LIMIT_S = 90.0


def save():
    (W / "pi3_probe.json").write_text(json.dumps(OUT))


def hog(stop, on, duty=1.0):
    x = 0
    while not stop.is_set():
        if not on.is_set():
            time.sleep(0.01)
        elif duty >= 1.0:
            for _ in range(100000):
                x += 1
        else:
            t = time.perf_counter()
            while time.perf_counter() - t < 0.02 * duty:
                pass
            time.sleep(0.02 * (1 - duty))


class Fixed:
    """One decision for every step, in the Pilot's interface."""

    def __init__(self, threads, cap, budget, draft=True):
        self.d = {"threads": threads, "cap": cap, "budget": budget, "draft": draft}

    def decide(self, *a):
        return dict(self.d)

    def observe(self, *a):
        pass

    def status(self):
        return {"fixed": self.d}

    def close(self):
        pass


def main():
    import subprocess

    OUT["gpu"] = subprocess.run("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader", shell=True,
                                capture_output=True, text=True).stdout
    OUT["cpu"] = subprocess.run("lscpu | head -20; nproc", shell=True, capture_output=True, text=True).stdout
    repo = Path(__file__).resolve().parents[2]
    if LOCAL:
        src = (repo / "lab/mg2/kaggle_gpu.py").read_bytes()
    else:
        src = __import__("urllib.request").request.urlopen(
            f"https://raw.githubusercontent.com/Tobayko/IronMule/{COMMIT}/lab/mg2/kaggle_gpu.py", timeout=60).read()
    mg2 = type(sys)("mg2")
    exec(compile(src, "mg2", "exec"), mg2.__dict__)
    t0 = time.time()
    if LOCAL:
        mg2.SRC = Path(os.environ["IRONMULE_GGML_BINDINGS"]).parent
        mg2.LIB = Path(os.environ["LLAMA_CPP_LIB_PATH"])
        (W / "pkg").mkdir(exist_ok=True)
        if not (W / "pkg/ironmule").exists():
            (W / "pkg/ironmule").symlink_to(repo / "ironmule")
        text = (repo / "lab/row1/facts.py").read_text()
        facts = eval(text[text.index("Q = [") + 4:text.index("]\nge.") + 1])
    else:
        facts = mg2.build()
    OUT["build_s"] = round(time.time() - t0, 1)
    OUT["build_cached"] = mg2.REPORT.get("build_cached")
    for k, v in mg2.env("0").items():
        os.environ[k] = v
    os.environ["IRONMULE_PILOT"] = "1"
    sys.path.insert(0, str(W / "pkg"))
    models = {}
    for hf, name in (("Qwen/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf"), ("Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf")):
        models[name] = str(W / name)
        if not Path(models[name]).exists():
            mg2.sh(f"cd {W} && curl -sSL -o {name}.part https://huggingface.co/{hf}/resolve/main/{name} && mv {name}.part {name}")
    gguf, draft = models["Qwen3-4B-Q4_K_M.gguf"], models["Qwen3-0.6B-Q8_0.gguf"]
    proc, port = mg2.server(gguf, 1, "0", "layer" if LOCAL else "none")
    try:
        fact_prompts = mg2.template(port, [q + " Answer in one short sentence." for q, _ in facts], "/no_think")
    finally:
        mg2.stop(proc)
    from ironmule.ggml_engine import GgmlEngine, Job

    tag = [0]  # every prompt of a run is new to the engine: no arm finds another arm's prompts in the pool
    topics = [t.split("about ", 1)[1].rstrip(".") for t in mg2.TEXTS]
    asks = ["Write a short story about {}.", "Explain in simple words how {} works.", "List five facts about {}.",
            "Write a short poem about {}."]

    def chat_text(i):
        tag[0] += 1
        return f"Ticket {tag[0]}. " + asks[i % len(asks)].format(topics[i % len(topics)])

    def doc_text(i):
        tag[0] += 1
        r = random.Random(tag[0])
        body = " ".join(f"Paragraph {k}: the harbour office of {('Arlen', 'Bree', 'Cosk', 'Dunmore')[r.randrange(4)]} "
                        f"logged {r.randrange(40)} ships, {r.randrange(12)} of them carrying grain, and the keeper noted "
                        f"the weather as {('calm', 'windy', 'rainy', 'foggy')[r.randrange(4)]}." for k in range(40))
        return f"Report {tag[0]}.\n{body}\n\nQuestion: how many ships were logged in paragraph {3 + i % 30}, and what was the weather?"

    def ids_of(text):
        return tok(f"<|im_start|>system\n/no_think<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n")

    def open_engine(draft_gguf=None):
        e = GgmlEngine(gguf, seqs=16, seq_ctx=2048, speculate=True, flash_attn=True, shared=True, cap=32,
                       draft_gguf=draft_gguf)
        e.collective = None
        ids = lambda t: e.tokenize(t)  # noqa: E731
        return e, ids

    def collect(e, tag):
        log = OUT["steps"].setdefault(tag, [])

        def obs(rows, prompt, kv, seconds):
            if len(log) < 60000:
                log.append([rows, prompt, kv, round(seconds * 1e4) / 1e4])
        e.observer = obs

    def block(e, n, tokens, blocks=2):
        tok = wall = 0.0
        firsts = []
        for b in range(blocks):
            js = [Job(ids_of(chat_text(b * n + i)), tokens) for i in range(n)]
            t = time.monotonic_ns()
            e.run(js)
            dt = (time.monotonic_ns() - t) / 1e9
            tok += sum(len(j.tokens) for j in js)
            wall += dt
            firsts += [(j.first_ns - t) / 1e9 for j in js]
            if wall > CELL_LIMIT_S:
                break
        firsts.sort()
        return {"tok_s": round(tok / wall, 2), "ttft": round(sum(firsts) / len(firsts), 3),
                "ttft_p90": round(firsts[math.ceil(0.9 * len(firsts)) - 1], 3), "wall": round(wall, 2)}

    def schedule(kind, seed):
        r = random.Random(seed)
        out, t = [], 0.0
        if kind == "chat":
            for i in range(6 if SMOKE else 48):
                t += r.expovariate(4.0)
                out.append((t, ids_of(chat_text(i)), 8 if SMOKE else r.randint(64, 192)))
        else:
            for c in range(2 if SMOKE else 12):
                t += r.expovariate(3.0 / 4)
                for i in range(4):
                    long = r.random() < 0.3
                    p = ids_of(doc_text(c * 4 + i) if long else chat_text(c * 4 + i))
                    out.append((t, p, 8 if SMOKE else r.randint(32, 160)))
        return out

    def stream(e, kind, seed):
        plan = schedule(kind, seed)
        jobs = [Job(p, lim) for _, p, lim in plan]
        born = {}
        nxt = [0]
        t0 = time.monotonic()

        def admit(room):
            now = time.monotonic() - t0
            if nxt[0] < len(plan) and all(j.finished_ns or j.reason for j in jobs[:nxt[0]]) and plan[nxt[0]][0] > now:
                time.sleep(plan[nxt[0]][0] - now)  # nothing running: wait for the next arrival
                now = time.monotonic() - t0
            got = []
            while nxt[0] < len(plan) and plan[nxt[0]][0] <= now and len(got) < max(room, 0):
                k = nxt[0]
                born[id(jobs[k])] = t0 + plan[k][0]
                got.append(jobs[k])
                nxt[0] += 1
            return got

        e.run([], admit=admit)
        wall = time.monotonic() - t0
        ttft = sorted((j.first_ns / 1e9 - born[id(j)]) for j in jobs if j.first_ns)
        e2e = sorted((j.finished_ns / 1e9 - born[id(j)]) for j in jobs if j.finished_ns)
        tok = sum(len(j.tokens) for j in jobs)
        q = lambda xs, p: round(xs[max(0, math.ceil(p * len(xs)) - 1)], 3) if xs else None  # noqa: E731
        return {"tok_s": round(tok / wall, 2), "tokens": tok, "wall": round(wall, 2), "done": len(e2e),
                "ttft_mean": round(sum(ttft) / max(1, len(ttft)), 3), "ttft_p50": q(ttft, 0.5), "ttft_p90": q(ttft, 0.9),
                "e2e_p50": q(e2e, 0.5), "e2e_p90": q(e2e, 0.9)}

    # ---- engine 1: no draft model ------------------------------------------------------------------------------------
    e, tok_ = open_engine()
    tok = tok_
    fact_ids = [tok(p) for p in fact_prompts]
    OUT["doc_tokens"] = len(ids_of(doc_text(0)))
    OUT["summary_open"] = e.summary()
    v6 = e.pilot
    OUT["v6_status"] = v6.status() if v6 is not None else e.summary().get("pilot")
    cores = e.threads
    OUT["costs"] = v6.costs if v6 is not None else e.measure_costs()
    big = 1 << 30
    e.pilot = None
    e.run([Job(ids_of(chat_text(0)), 16)])
    collect(e, "plain")
    save()

    # A: host CPU load
    arms_a = [("t1", Fixed(1, 32, big)), ("t2", Fixed(2, 32, big)), ("t4", Fixed(cores, 32, big)), ("v6", v6)]
    for load in (("0", "1") if SMOKE else ("0", "1", "2", "3", "d0.5")):
        stop, on = mp.Event(), mp.Event()
        part = load.startswith("d")
        ps = [mp.Process(target=hog, args=(stop, on, float(load[1:]) if part else 1.0), daemon=True)
              for _ in range(1 if part else int(load))]
        [p.start() for p in ps]
        on.set()
        time.sleep(1.0)
        for rep in range(2):
            for arm, src_ in (arms_a if rep == 0 else arms_a[::-1]):
                for n in ((1, 8) if SMOKE else (1, 8, 32)):
                    e.pilot = src_
                    row = {"load": load, "rep": rep, "arm": arm, "n": n, **block(e, n, 8 if SMOKE else 64)}
                    row["threads_now"] = e.threads
                    if arm == "v6" and v6 is not None:
                        st = v6.status()
                        row["decision"], row["decide_us"] = st.get("decision"), st.get("decide_us_mean")
                    OUT["A"].append(row)
                    print(json.dumps(row), flush=True)
                    save()
        stop.set()
        on.set()
        [p.join(timeout=5) for p in ps]
    e.pilot = None
    e.set_threads(cores)

    # B: arriving requests
    arms_b = [("defaults", None), ("v6", v6), ("cap4", Fixed(cores, 4, big)), ("cap8", Fixed(cores, 8, big)),
              ("cap16", Fixed(cores, 16, big)), ("budget128", Fixed(cores, 32, 128)),
              ("budget512", Fixed(cores, 32, 512)), ("t1", Fixed(1, 32, big))]
    for rep in range(1 if SMOKE else 3):
        for arm, src_ in (arms_b[:3] if SMOKE else arms_b if rep % 2 == 0 else arms_b[::-1]):
            for kind in ("chat", "mixed"):
                e.pilot = src_
                if src_ is None:
                    e.set_threads(cores)
                row = {"arm": arm, "rep": rep, "kind": kind, **stream(e, kind, 100 + rep)}
                if arm == "v6" and v6 is not None:
                    row["decision"] = v6.status().get("decision")
                OUT["B"].append(row)
                print(json.dumps(row), flush=True)
                save()

    # D: answers
    answers = {}
    for arm, src_ in (("defaults", None), ("v6", v6)):
        e.pilot = src_
        e.set_threads(cores)
        got = []
        for k in range(0, 8 if SMOKE else len(fact_ids), 8):
            js = [Job(i, 48) for i in fact_ids[k:k + 8]]
            e.run(js)
            got += [j.tokens for j in js]
        answers[arm] = got
    e.pilot = None
    OUT["summary_end"] = e.summary()
    e.observer = None
    e.close()
    save()

    # C: a draft model
    os.environ["IRONMULE_PILOT"] = "1"
    e, tok_ = open_engine(draft)
    tok = tok_
    collect(e, "draft")
    v6d = e.pilot
    OUT["draft_open"] = e.summary().get("draft_model")
    arms_c = [("defaults", None), ("nodraft", Fixed(cores, 32, big, False)), ("v6", v6d)]
    e.pilot = None
    e.run([Job(ids_of(chat_text(0)), 16)])
    for rep in range(1 if SMOKE else 2):
        for arm, src_ in (arms_c if rep == 0 else arms_c[::-1]):
            e.pilot = src_
            e.set_threads(cores)
            for n in (1, 8):
                row = {"arm": arm, "rep": rep, "n": n, **block(e, n, 8 if SMOKE else 128)}
                OUT["C"].append(row)
                print(json.dumps(row), flush=True)
            row = {"arm": arm, "rep": rep, "n": "chat", **stream(e, "chat", 200 + rep)}
            if arm == "v6" and v6d is not None:
                row["decision"] = v6d.status().get("decision")
            OUT["C"].append(row)
            print(json.dumps(row), flush=True)
            save()
    OUT["draft_end"] = e.summary().get("draft_model")
    e.observer = None
    e.close()
    proc, port = mg2.server(gguf, 1, "0", "layer" if LOCAL else "none")
    try:
        OUT["facts"] = {arm: mg2.correct([mg2.detok_server(port, i) for i in v], facts) for arm, v in answers.items()}
    finally:
        mg2.stop(proc)
    OUT["same_tokens"] = sum(a == b for a, b in zip(answers["defaults"], answers["v6"]))
    OUT["finished"] = time.time()
    save()


if __name__ == "__main__":
    main()
