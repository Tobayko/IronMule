"""Part 3/4: llama-server end to end. Usage: serve.py SERVER GGUF LABEL MODE [extra server flags...]
MODE "load:N1,N2,..": for each N a fresh server with -np N -c N*2048, one warm-up block then 3 blocks of N streaming
chat requests at once (64 tokens each, different short prompts); total tokens/s over each block's wall time, mean TTFT.
MODE "single": one request alone, 128 tokens, 1 warm-up + 3 runs. MODE "prefill": prompts of 512 and 2000 tokens of
the fixed text, n_predict 1, cache off; server-reported prompt tokens/s and wall time. Environment passes through
(ROW1_GGML, GLIBC_TUNABLES)."""
import http.client, json, os, socket, subprocess, sys, threading, time
sys.path.insert(0, "/home/user/IronMule")
server, gguf, label, mode, extra = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:]
TOPICS = ["a lighthouse keeper", "a lost robot", "a mountain village", "a river in winter", "a clockmaker", "a desert caravan",
          "a library at night", "a sailing ship", "a city garden", "an old bakery", "a train journey", "a forest fire watch",
          "a chess player", "a beekeeper", "a snowstorm", "a market day", "a violin maker", "a moon base", "a fisherman",
          "a school teacher", "a radio operator", "a glass blower", "a postman", "a tea plantation", "a castle ruin",
          "an island ferry", "a night nurse", "a bridge builder", "a museum guard", "a street musician", "a coal miner",
          "a weather station", "a kite festival", "a bicycle courier", "a toy shop", "a canal boat", "a stone mason", "a zoo keeper",
          "a printing press", "a hot air balloon", "a mapmaker", "a fire station", "a vineyard", "a cave explorer", "a ballet class",
          "a pottery studio", "a ski lodge", "a orchard"]
def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p
def start(np_, ctx):
    port = free_port()
    proc = subprocess.Popen([server, "-m", gguf, "--host", "127.0.0.1", "--port", str(port), "-t", "4", "-np", str(np_),
                             "-c", str(ctx), "--temp", "0", "--no-webui", *extra], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    while True:
        try:
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=5); c.request("GET", "/health")
            if c.getresponse().status == 200: return proc, port
        except Exception: pass
        if proc.poll() is not None: raise RuntimeError("server died")
        time.sleep(0.5)
def stop(proc):
    proc.terminate()
    try: proc.wait(timeout=30)
    except subprocess.TimeoutExpired: proc.kill()
def ask(port, text, limit, t0):
    sysm = [{"role": "system", "content": os.environ["ROW1_SYSTEM"]}] if os.environ.get("ROW1_SYSTEM") else []
    body = json.dumps({"messages": sysm + [{"role": "user", "content": text}], "max_tokens": limit, "temperature": 0,
                       "stream": True, "stream_options": {"include_usage": True}})
    first, pieces, usage = None, [], None
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=3600)
    c.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"}); r = c.getresponse()
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:") or line == "data: [DONE]": continue
        ch = json.loads(line[5:])
        for choice in ch.get("choices") or []:
            if (choice.get("delta") or {}).get("content"):
                first = first if first is not None else time.perf_counter() - t0
                pieces.append(choice["delta"]["content"])
        if ch.get("usage"): usage = ch["usage"]
    return {"first_s": first, "tokens": (usage or {}).get("completion_tokens") or len(pieces), "text": "".join(pieces)}
def block(port, texts, limit):
    out = [None] * len(texts); t0 = time.perf_counter()
    th = [threading.Thread(target=lambda i=i: out.__setitem__(i, ask(port, texts[i], limit, t0))) for i in range(len(texts))]
    for t in th: t.start()
    for t in th: t.join()
    wall = time.perf_counter() - t0; tok = sum(o["tokens"] for o in out)
    return {"tok_s": tok / wall, "tokens": tok, "wall_s": wall, "ttft_mean_s": sum(o["first_s"] for o in out) / len(out),
            "sample": out[0]["text"][:80]}
res = {"label": label, "mode": mode, "flags": extra, "env": {k: os.environ.get(k) for k in ("ROW1_GGML", "GLIBC_TUNABLES")}}
if mode.startswith("load:"):
    for n in [int(x) for x in mode[5:].split(",")]:
        proc, port = start(n, min(n * 2048, int(os.environ.get("ROW1_CTXCAP", "1000000000"))))
        try:
            texts = [f"Write a short story about {TOPICS[i % len(TOPICS)]}. /no_think" for i in range(n)]
            block(port, texts, 64)
            res[f"N{n}"] = [block(port, texts, 64) for _ in range(3)]
        finally: stop(proc)
        print(label, n, [round(b["tok_s"], 1) for b in res[f"N{n}"]], [round(b["ttft_mean_s"], 3) for b in res[f"N{n}"]], flush=True, file=sys.stderr)
elif mode == "single":
    proc, port = start(1, 4096)
    try:
        t = "Write a short story about a lighthouse keeper. /no_think"
        block(port, [t], 128); res["single"] = [block(port, [t], 128) for _ in range(3)]
    finally: stop(proc)
elif mode == "prefill":
    from ironmule.numeric_choice import TEXT
    proc, port = start(1, 4096)
    try:
        text = "".join(l for l in TEXT.read_text(encoding="utf-8").splitlines(True) if not l.startswith("#"))
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
        c.request("POST", "/tokenize", json.dumps({"content": text}), {"Content-Type": "application/json"})
        ids = json.loads(c.getresponse().read())["tokens"]
        for n in (512, 2000):
            runs = []
            for k in range(3):
                prompt = ids[k * 7:k * 7 + n]  # shifted so no prompt cache can help
                c = http.client.HTTPConnection("127.0.0.1", port, timeout=3600); t0 = time.perf_counter()
                c.request("POST", "/completion", json.dumps({"prompt": prompt, "n_predict": 1, "cache_prompt": False, "temperature": 0}),
                          {"Content-Type": "application/json"})
                d = json.loads(c.getresponse().read()); wall = time.perf_counter() - t0
                runs.append({"wall_s": wall, "prompt_n": d["timings"]["prompt_n"], "prompt_ms": d["timings"]["prompt_ms"],
                             "prompt_tok_s": d["timings"]["prompt_per_second"]})
            res[f"P{n}"] = runs
    finally: stop(proc)
print(json.dumps(res))
