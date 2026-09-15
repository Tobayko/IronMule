"""DATA3: stock mlx_lm versus IronMule Runtime on one model, same process.

Usage: python infer.py MODEL_ID REVISION OUT.json
Runs identically on the Mac (Metal) and on Kaggle (MLX CUDA). Token identity is
judged only within one host; cross-host token differences are diagnostics.
"""
import gc
import json
import sys
import time
import traceback

MODEL_ID, REVISION, OUT = sys.argv[1:4]
PROMPTS = [
    "Explain in two sentences why the sky is blue.",
    "List three prime numbers and explain what makes them prime.",
]
MAX_TOKENS = 32
report = {"schema": "ironmule.data3-infer.v1", "model_id": MODEL_ID, "revision": REVISION,
          "max_tokens": MAX_TOKENS, "prompts": PROMPTS, "stages": {}}


def stage(name, fn):
    started = time.time()
    try:
        entry = {"ok": True, "value": fn()}
    except BaseException as exc:  # noqa: BLE001 - every stage failure is evidence
        entry = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:2000],
                 "traceback": traceback.format_exc()[-4000:]}
    entry["seconds"] = round(time.time() - started, 2)
    report["stages"][name] = entry
    with open(OUT, "w") as stream:
        json.dump(report, stream, indent=1, default=str)
    print(f"[{name}] ok={entry['ok']} {entry.get('error', '')[:400]}", flush=True)
    return entry.get("value")


import mlx.core as mx  # noqa: E402
import mlx_lm  # noqa: E402


def device():
    info = {"mlx": mx.__version__, "mlx_lm": mlx_lm.__version__,
            "default_device": str(mx.default_device()),
            "metal": bool(mx.metal.is_available())}
    cuda = getattr(mx, "cuda", None)
    info["cuda"] = bool(cuda.is_available()) if cuda is not None else None
    return info


stage("device", device)
state = {}


def prompt_ids():
    import os
    from huggingface_hub import try_to_load_from_cache
    from mlx_lm import load
    # snapshot_download(local_files_only) rejects snapshots lacking README/.gitattributes.
    path = os.path.dirname(try_to_load_from_cache(MODEL_ID, "config.json", revision=REVISION))
    model, tokenizer = load(path)
    state.update(model=model, tokenizer=tokenizer)
    ids = []
    for prompt in PROMPTS:
        rendered = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                 tokenize=False, add_generation_prompt=True)
        ids.append(list(tokenizer.encode(rendered, add_special_tokens=False)))
    state["ids"] = ids
    state["eos"] = sorted(int(t) for t in tokenizer.eos_token_ids)
    return {"path": path, "prompt_tokens": [len(i) for i in ids], "eos": state["eos"]}


def visible(tokens):
    return [int(t) for t in tokens if int(t) not in state["eos"]]


def stock():
    from mlx_lm import stream_generate
    rows = []
    for ids in state["ids"]:
        start = time.perf_counter()
        first = None
        tokens = []
        for response in stream_generate(state["model"], state["tokenizer"], ids, max_tokens=MAX_TOKENS):
            if first is None:
                first = time.perf_counter() - start
            tokens.append(int(response.token))
        total = time.perf_counter() - start
        rows.append({"tokens": visible(tokens), "ttft_s": first, "total_s": total,
                     "decode_tok_s": (len(tokens) - 1) / (total - first) if len(tokens) > 1 else None,
                     "text": state["tokenizer"].decode(visible(tokens))})
    state["stock"] = [r["tokens"] for r in rows]
    return {"rows": rows, "peak_memory_bytes": int(mx.get_peak_memory())}


def release():
    state.pop("model", None)
    gc.collect()
    mx.clear_cache()


def compare(results):
    got = [visible(r.tokens) for r in results]
    return {"identical_to_stock": got == state["stock"], "tokens": got,
            "stop_reasons": [r.stop_reason for r in results],
            "metrics": [r.metrics for r in results]}


def runtime_interactive():
    import ironmule
    with ironmule.Runtime.load(MODEL_ID, revision=REVISION, use_tuned_profile=False) as rt:
        results = [rt.generate(prompt_ids=ids, max_tokens=MAX_TOKENS) for ids in state["ids"]]
        out = compare(results)
        out["fallbacks"] = rt.telemetry.fallbacks
        return out


def runtime_throughput():
    import ironmule
    with ironmule.Runtime.load(MODEL_ID, mode=ironmule.ThroughputMode(), revision=REVISION,
                               use_tuned_profile=False) as rt:
        results = rt.serve([ironmule.Request(prompt_ids=ids, max_tokens=MAX_TOKENS)
                            for ids in state["ids"]])
        out = compare(results)
        out["fallbacks"] = rt.telemetry.fallbacks
        out["fallback_reasons"] = rt.telemetry.fallback_reasons
        return out


def runtime_shipped_knobs():
    # The delivery path (docs/FABLE_ERFOLGSPFAD.md, D4): head-skip + fixed compiled cache + readback 8.
    import ironmule
    from ironmule.tune import load_engine
    knobs = ironmule.Knobs(head_skip_prefill=True, compiled_fixed_cache=True, readback_every=8)
    engine, tokenizer = load_engine(MODEL_ID, knobs, revision=REVISION)
    with ironmule.Runtime(engine, tokenizer, model_id=MODEL_ID) as rt:
        results = [rt.generate(prompt_ids=ids, max_tokens=MAX_TOKENS) for ids in state["ids"]]
        out = compare(results)
        out["knobs"] = knobs.as_dict()
        out["fallbacks"] = rt.telemetry.fallbacks
        return out


def apple_runtime():
    import ironmule
    rt = ironmule.AppleRuntime.load(MODEL_ID, revision=REVISION)
    try:
        results = rt.serve([ironmule.Request(prompt_ids=ids, max_tokens=MAX_TOKENS)
                            for ids in state["ids"]])
        out = compare(results)
        decision = rt.last_decision or {}
        out["route"], out["reason"] = decision.get("route"), decision.get("reason")
        return out
    finally:
        close = getattr(rt, "close", None)
        if close:
            close()


if stage("load_stock", prompt_ids) is not None:
    stage("stock_mlx_lm", stock)
    release()
    stage("runtime_interactive", runtime_interactive)
    gc.collect(); mx.clear_cache()
    stage("runtime_throughput", runtime_throughput)
    gc.collect(); mx.clear_cache()
    stage("runtime_shipped_knobs", runtime_shipped_knobs)
    gc.collect(); mx.clear_cache()
    stage("apple_runtime", apple_runtime)
