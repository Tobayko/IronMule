"""SG1 v3 on a Kaggle T4: one request on one GPU with a draft model (DM1), against llama-server with its own draft model
(`-md`, the same small GGUF), on the GR1 build of SG1 v2 (the kernel output `ironmule-sg1-single`, attached as input).

A T4 computes a step of 9-16 tokens in 27-28 ms against 20.5 ms for one (MG2 v5's padding table on Qwen3-4B), so a step
that checks a draft model's guesses costs little more than a plain one. Pairs (same tokenizer): Qwen3-4B with
Qwen3-0.6B, Gemma 3 4B with Gemma 3 1B, Llama 3.1 8B with Llama 3.2 1B, all Q4_K_M with Q8_0 drafts. Per pair, in this
order twice (_a, _b): llama-server, llama-server -md, the engine, the engine with the draft model, with CUDA graphs on
or off as SG1_GRAPHS says (GR1). One warm-up block and 3 blocks of 128 tokens on 4 different prompts, greedy, SP4 off.

Answers: each arm's tokens against the plain engine's, and the 50 facts one request at a time through the engine with
the draft model and through llama-server -md. Writes /kaggle/working/sg1_draft.json.
"""
import glob, json, os, sys, time, urllib.request
from pathlib import Path

COMMIT = os.environ["MG2_COMMIT"]
W = Path("/kaggle/working")
src = urllib.request.urlopen(f"https://raw.githubusercontent.com/Tobayko/IronMule/{COMMIT}/lab/mg2/kaggle_gpu.py",
                             timeout=60).read()
(W / "kaggle_gpu.py").write_bytes(src)
mg2 = type(sys)("mg2")
exec(compile(src, str(W / "kaggle_gpu.py"), "exec"), mg2.__dict__)
OUT = {"commit": COMMIT, "started": time.time(), "models": {}}
GRAPHS = os.environ.get("SG1_GRAPHS", "1")
PAIRS = [("Qwen/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf", "Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf"),
         ("unsloth/gemma-3-4b-it-GGUF", "gemma-3-4b-it-Q4_K_M.gguf", "unsloth/gemma-3-1b-it-GGUF", "gemma-3-1b-it-Q8_0.gguf"),
         ("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
          "bartowski/Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q8_0.gguf")]


def save():
    (W / "sg1_draft.json").write_text(json.dumps(OUT, indent=1))


def fetch(repo, name):
    path = W / name
    if not path.exists():
        mg2.sh(f"cd {W} && curl -sSL -o {name}.part https://huggingface.co/{repo}/resolve/main/{name} && mv {name}.part {name}")
    return str(path)


def unpack_build():
    tar = next(iter(glob.glob("/kaggle/input/**/sg1_build.tar.gz", recursive=True)), None)
    if tar:
        mg2.SRC.mkdir(parents=True, exist_ok=True)
        mg2.sh(f"cd {mg2.SRC} && tar xzf {tar} && chmod +x {mg2.LIB}/llama-server")
        OUT["build_from"] = tar
    OUT["inputs"] = glob.glob("/kaggle/input/*/*")[:20]


def blocks(run, prompts):
    run(prompts[:1])
    out, texts = [], []
    for _ in range(3):
        rows, t = [], []
        for p in prompts:  # one request at a time: 4 prompts, one after the other
            b, tx = run([p])
            rows.append(b)
            t += tx
        wall = sum(b["wall_s"] for b in rows)
        out.append({"tok_s": sum(b["tokens"] for b in rows) / wall, "ttft_s": sum(b["ttft_s"] for b in rows) / len(rows),
                    "wall_s": wall, "tokens": sum(b["tokens"] for b in rows)})
        texts = t
    return out, texts


def run_pair(repo, name, drepo, dname, facts):
    gguf, draft = fetch(repo, name), fetch(drepo, dname)
    m = OUT["models"].setdefault(name, {"draft": dname, "arms": {}})
    proc, port = mg2.server(gguf, 1, "0", "none")
    try:
        system = "/no_think" if "Qwen3" in name else None
        prompts = mg2.template(port, mg2.TEXTS[:4], None)
        fact_prompts = mg2.template(port, [q + " Answer in one short sentence." for q, _ in facts], system)
    finally:
        mg2.stop(proc)
    md = ["-md", draft, "-ngld", "99"]
    texts = {}
    for rep in ("a", "b"):
        for arm in ("server", "server_md", "engine", "engine_dm1"):
            key = f"{arm}_{rep}"
            try:
                if arm.startswith("server"):
                    proc, port = mg2.server(gguf, 1, "0", "none", md if arm == "server_md" else ())
                    try:
                        r, tx = blocks(lambda pr: mg2.block_servers([port], pr, mg2.TOKENS), prompts)
                    finally:
                        mg2.stop(proc)
                    m["arms"][key] = {"blocks": r}
                else:
                    eng = mg2.Engines(gguf, 1, ["0"], draft=draft if arm == "engine_dm1" else None)
                    try:
                        r, tx = blocks(lambda pr: eng.block(pr, mg2.TOKENS), prompts)
                        m["arms"][key] = {"blocks": r, "diag": eng.last, "summary": eng.summary}
                    finally:
                        eng.close()
                texts[key] = tx
                mg2.log(name, key, json.dumps([{k: round(v, 3) for k, v in b.items()} for b in r]))
            except Exception as exc:  # noqa: BLE001 - recorded, the run goes on
                m["arms"][key] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
                mg2.log(name, key, "ERROR", exc)
            save()
    m["engine_same_tokens"] = {k: texts[k] == texts.get("engine_a") for k in texts if k.startswith("engine")}
    m["server_same_text"] = {k: texts[k] == texts.get("server_a") for k in texts if k.startswith("server")}
    answers = {}
    try:
        for arm, d in (("engine", None), ("engine_dm1", draft)):
            eng = mg2.Engines(gguf, 1, ["0"], draft=d)
            answers[arm] = [eng.block([p], 48)[1][0] for p in fact_prompts]
            eng.close()
        for arm, extra in (("server", ()), ("server_md", md)):
            proc, port = mg2.server(gguf, 1, "0", "none", extra)
            answers[arm] = [mg2.block_servers([port], [p], 48)[1][0] for p in fact_prompts]
            mg2.stop(proc)
        proc, port = mg2.server(gguf, 1, "0", "none")
        try:
            for arm, ans in answers.items():
                words = [mg2.detok_server(port, a) for a in ans] if arm.startswith("engine") else ans
                m[f"facts_{arm}"] = mg2.correct(words, facts)
        finally:
            mg2.stop(proc)
        m["facts_engine_dm1_same_tokens"] = sum(a == b for a, b in zip(answers["engine"], answers["engine_dm1"]))
        m["facts_server_md_same_text"] = sum(a == b for a, b in zip(answers["server"], answers["server_md"]))
    except Exception as exc:  # noqa: BLE001
        m["facts_error"] = f"{type(exc).__name__}: {exc}"[:300]
    mg2.log(name, "facts", {k: v for k, v in m.items() if k.startswith("facts")})
    save()


def main():
    os.environ["IRONMULE_CUDA_GRAPHS"] = GRAPHS
    OUT["graphs"] = GRAPHS
    unpack_build()
    facts = mg2.build()
    OUT["build_s"] = round(mg2.REPORT.get("build_s", 0), 1)
    save()
    for pair in PAIRS:
        try:
            run_pair(*pair, facts)
        except Exception as exc:  # noqa: BLE001
            OUT["models"].setdefault(pair[1], {})["error"] = f"{type(exc).__name__}: {exc}"[:300]
            mg2.log(pair[1], "ERROR", exc)
        save()
        for f in W.glob("*.gguf"):
            f.unlink()
    OUT["finished"] = time.time()
    save()


main()
