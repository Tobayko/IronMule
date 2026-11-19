"""Qualify a scoped, graph-off Q4_K CUDA dispatch against bracketing stock native runs."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import tempfile
import time

SCHEMA = "ironmule.scoped-kg4-native.v1"
MODEL = "efd971561896866f0e910cce52761ca77b1b138090c7f15fe284676d57d1f689"
BACKEND = "bbc59e06670907330a9e39b111451560f7318153fdd0ce45d3d7d7afc9ffc816"
PLUGIN = "da0ff490ce4336e9b684166cfbe6486f7a9058ee68832114a9543d0b75c219cf"
REPEATS = 2
WIDTHS = (8, 32)
P95_LIMIT = 1.03


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def finite(value):
    return type(value) in (float, int) and 0 < value < math.inf


def typed_hits(hits, low):
    counts = hits["kg4"]
    return (hits["cublas"] == [0] and len(counts) == 64 and all(type(v) is int and v >= 0 for v in counts)
            and all(i in (5, 6, 7) or not value for i, value in enumerate(counts))
            and (sum(counts) > 0 if low else sum(counts) == 0))


def p95(values):
    return sorted(values)[math.ceil(0.95 * len(values)) - 1]


def qualify(stock, candidate, closing):
    """Read only three complete records from the same native runtime and held-out workload."""
    from ironmule import factual_gate as F, numeric_choice as NC

    reports = (stock, candidate, closing)
    result = {"eligible": False, "metrics": {}, "rows": [], "reasons": [], "quality": {}, "nll": {}}
    reasons = set()
    try:
        for arm, name in zip(reports, ("stock", "scoped", "stock-closing")):
            require(arm["schema"] == SCHEMA and arm["arm"] == name and arm["status"] == "measured", "incomplete_arm")
            require(arm["seed"] == stock["seed"] and type(arm["seed"]) is int and 0 <= arm["seed"] < 2**32,
                    "different_seed")
            require(arm["facts_seed"] == arm["seed"] + 65537, "different_factual_seed")
            require(arm["gguf_sha256"] == MODEL and arm["native"] == stock["native"] and
                    arm["native"]["backend/CUDA"] == BACKEND and all(map(sha256, arm["native"].values())),
                    "different_native_or_model")
            require({"llama_backend_init", "ggml_init", "ggml_backend_reg_count", "backend/CPU"} <= arm["native"].keys(),
                    "incomplete_native_binding")
            require(arm["gpu"] == stock["gpu"] and arm["gpu"]["cc"] == "89" and bool(arm["gpu"]["driver"])
                    and sha256(arm["gpu"]["uuid_sha256"]) and arm["offloaded_layers"] == 65
                    and finite(arm["cuda_model_mib"]) and arm["cuda_model_mib"] >= 15000,
                    "different_device_or_offload")
            require(arm["graph_capture"] is False and arm["context"] == 768 and arm["sequences"] == 32
                    and all(sha256(arm[x]) and arm[x] == stock[x] for x in ("engine_sha256", "scope_source_sha256", "harness_sha256")),
                    "different_runtime")
            require(arm["plugin_sha256"] == (PLUGIN if name == "scoped" else None), "wrong_plugin")
            for nll in ("nll", "nll_b32"):
                require(len(arm[nll]) == 16 and all(finite(v) for v in arm[nll]), "invalid_nll")
            require(typed_hits(arm["facts_low_hits"], name == "scoped"), "factual_dispatch")
        scope = candidate["scope"]
        require(scope["mode"] == "low-width8" and scope["parked"] is True and scope["low_runs"] >= 2
                and scope["wide_runs"] >= 2 and bool(scope["q4_buckets"])
                and set(scope["q4_buckets"]) <= {6, 7, 8} and candidate["masks_after"] == [0, 0]
                and typed_hits(candidate["disable_hits"], False), "scope_not_parked")
        for width in (6, 7, 8):
            key = str(width)
            for arm in reports:
                values = arm["nll_low"][key]
                hits = arm["nll_low_hits"][key]
                active = arm["arm"] == "scoped" and width in scope["q4_buckets"]
                require(len(values) == width and all(finite(v) for v in values), "invalid_low_nll")
                require(typed_hits(hits, active) and (not active or sum(hits["kg4"]) == hits["kg4"][width - 1]),
                        "low_nll_dispatch")
            result["nll"]["active_b" + key] = {}
            for name, control in (("stock", stock), ("closing", closing)):
                score = NC.ratio_interval(control["nll_low"][key], candidate["nll_low"][key])
                result["nll"]["active_b" + key][name] = score
                if not score["passes"]:
                    reasons.add("nll_low_b" + key)
        seed = stock["facts_seed"]
        for facts in ("facts", "facts_low"):
            reference, repeat = stock[facts], stock[facts + "_repeat"]
            scored = F.compare(reference, repeat, candidate[facts], seed=seed)
            final = F.compare(reference, repeat, closing[facts], seed=seed)
            lost = [key for key in reference if (reference[key] or repeat[key] or closing[facts][key])
                    and not candidate[facts][key]]
            if not scored["passes"] or not final["passes"] or lost:
                reasons.add(facts)
            result["quality"][facts] = {"candidate": scored, "closing": final, "lost": lost}
        for column in ("nll", "nll_b32"):
            result["nll"][column] = {}
            for name, control in (("stock", stock), ("closing", closing)):
                score = NC.ratio_interval(control[column], candidate[column])
                result["nll"][column][name] = score
                if not score["passes"]:
                    reasons.add(column)
        cells = {(r, w) for r in range(REPEATS) for w in WIDTHS}
        tables = []
        for arm in reports:
            rows = {(r["repeat"], r["width"]): r for r in arm["waves"]}
            require(rows.keys() == cells and len(arm["waves"]) == len(cells), "incomplete_native_waves")
            require(len({r["prompt_ids_sha256"] for r in rows.values()}) == len(cells), "repeated_timed_inputs")
            tables.append(rows)
        for cell in sorted(cells):
            a, b, c = (t[cell] for t in tables)
            rep, width = cell
            for name, row in zip(("stock", "scoped", "stock-closing"), (a, b, c)):
                times = row["ttft_ms"]
                require(len(times) == width and all(finite(x) for x in times) and
                        math.isclose(p95(times), row["ttft_p95_ms"], rel_tol=1e-6), "invalid_ttft")
                work = row["work"]
                require(len(work) == width and all(type(n) is int and 1 <= n <= 32
                        and stop in ("eos", "length") for n, stop in work) and finite(row["elapsed_s"])
                        and row["output_tokens"] == sum(n for n, _ in work)
                        and math.isclose(row["output_tok_s"], row["output_tokens"] / row["elapsed_s"], rel_tol=1e-6),
                        "invalid_output_work")
                require(sha256(row["prompt_ids_sha256"]) and sha256(row["output_ids_sha256"]), "invalid_token_proof")
                if not typed_hits(row["hits"], name == "scoped" and width == 8):
                    reasons.add(f"dispatch_w{width}")
            require(a["prompt_ids_sha256"] == b["prompt_ids_sha256"] == c["prompt_ids_sha256"], "different_inputs")
            if a["work"] != b["work"] or a["work"] != c["work"]:
                reasons.add(f"generation_w{width}")
            if a["output_ids_sha256"] != c["output_ids_sha256"]:
                reasons.add(f"unstable_reference_w{width}")
            exact = a["output_ids_sha256"] == b["output_ids_sha256"]
            if width == 32 and not exact:
                reasons.add("wide_output_changed")
            # Both controls individually bound the decision, including separate startup drift.
            rate = b["output_tok_s"] / max(a["output_tok_s"], c["output_tok_s"])
            latency = b["ttft_p95_ms"] / min(a["ttft_p95_ms"], c["ttft_p95_ms"])
            if rate < (1.03 if width == 8 else .97):
                reasons.add(f"throughput_w{width}")
            if latency > P95_LIMIT:
                reasons.add(f"p95_w{width}")
            result["rows"].append({"repeat": rep, "width": width, "throughput_ratio": rate, "p95_ratio": latency,
                                   "output_exact": exact, "accepted": sum(b["hits"]["kg4"])})
        for width in WIDTHS:
            group = [r for r in result["rows"] if r["width"] == width]
            result["metrics"][str(width)] = {k: statistics.median(row[k] for row in group)
                                              for k in ("throughput_ratio", "p95_ratio")}
        result["binding"] = {k: candidate[k] for k in ("seed", "gpu", "gguf_sha256", "native", "engine_sha256",
                                                         "plugin_sha256", "scope_source_sha256", "harness_sha256")}
    except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
        reasons.add(str(exc) if isinstance(exc, ValueError) else "invalid_record")
    result["reasons"] = sorted(reasons)
    result["eligible"] = not reasons
    return result


def low_facts(engine, tok, seed, gate, job_class):
    scores = {}
    cases = gate.cases(seed)
    for start in range(0, len(cases), 8):
        group = cases[start:start + 8]
        jobs = [job_class(gate._chat_ids(tok, case, False), gate.MAX_TOKENS) for case in group]
        engine._clear()
        engine.run(jobs)
        for case, job in zip(group, jobs):
            require(job.reason in ("eos", "length") and job.tokens, "incomplete_low_factual_answer")
            answer = tok.decode([t for t in job.tokens if not engine.eog(t)])
            scores[case.key] = gate.correct(case, answer)
    return scores


def low_nll(engine, group):
    """Teacher-force eight or fewer independent streams through each shape's actual native logits."""
    width = len(group)
    engine._clear()
    for seq, (lead, _) in enumerate(group):
        if lead:
            engine._decode([(token, i, seq, False) for i, token in enumerate(lead)])
    total = [0.0] * width
    steps = len(group[0][1]) - 1
    require(steps == 16 and all(len(ids) == steps + 1 for _, ids in group), "invalid_low_nll_fixture")
    scope = engine.kg4_scope
    if scope:
        scope.begin()
    try:
        if scope:
            scope.update(width)
        for i in range(steps):
            engine._decode([(ids[i], len(lead) + i, seq, True) for seq, (lead, ids) in enumerate(group)])
            for seq, (_, ids) in enumerate(group):
                ptr = engine.L.llama_get_logits_ith(engine.ctx, seq)
                logits = engine.np.ctypeslib.as_array(ptr, shape=(engine.n_vocab,)).astype(engine.np.float64)
                maximum = logits.max()
                total[seq] += float(maximum + math.log(engine.np.exp(logits - maximum).sum()) - logits[ids[i + 1]])
    finally:
        if scope:
            scope.end()
        engine._clear()
    return [v / steps for v in total]


def measure(args):
    # The prepared capsule verifies its immutable helpers, product tree, GGUF and native DSOs before launch.
    sys.path.insert(0, str(args.capsule_dir.resolve()))
    from native import delta, native_hits
    from select_layout import dump, gpu, identity, inputs, sha
    from ironmule import ggml_engine as G, factual_gate as F, numeric_choice as NC
    from ironmule.ggml_binary import loaded_objects
    from ironmule.gguf_tokenizer import GgufTokenizer
    from ironmule_product import compare

    scoped = args.arm == "scoped"
    require(0 <= args.seed < 2**32 and os.environ.get("GGML_CUDA_DISABLE_GRAPHS") == "1"
            and os.environ.get("IRONMULE_GGML_PREFILL_BATCH", "0") == "0"
            and os.environ.get("IRONMULE_PILOT") == "0" and G.THREADS == 2
            and os.environ.get("IRONMULE_GGML_STREAMS") == "1"
            and (os.environ.get("IRONMULE_KG4_SCOPE") == "low-width8") == scoped
            and bool(G.KG_LIB) == scoped, "incorrect_native_environment")
    os.close(os.open(args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
    report = {"schema": SCHEMA, "arm": args.arm, "seed": args.seed, "facts_seed": args.seed + 65537,
              "context": 768, "sequences": 32, "graph_capture": False, "status": "starting", "waves": [],
              "harness_sha256": sha(Path(__file__)), "engine_sha256": sha(Path(G.__file__)),
              "scope_source_sha256": sha(Path(G.__file__).with_name("kg4_scope.py"))}
    dump(args.output, report)
    try:
        stamp = identity(args.gguf)
        manifest = inputs(args.gguf)
        plugin = sha(G.KG_LIB) if scoped else None
        require(manifest["model_sha256"] == MODEL and (not scoped or plugin == PLUGIN), "unverified_gguf_or_plugin")
        report.update(gpu=gpu(), gguf_sha256=MODEL, plugin_sha256=plugin)
        with tempfile.TemporaryFile() as log:
            saved = os.dup(2)
            try:
                os.dup2(log.fileno(), 2)
                engine = G.GgmlEngine(str(args.gguf), seqs=32, seq_ctx=768, threads=2, shared=False,
                                      flash_attn=True, kv_unified=False, speculate=False)
            finally:
                os.dup2(saved, 2)
                os.close(saved)
            log.seek(0)
            text = log.read().decode(errors="replace")
        try:
            offload = re.findall(r"offloaded (\d+)/(\d+) layers to GPU", text)
            memory = re.findall(r"CUDA0\s+model buffer size\s*=\s*([\d.]+) MiB", text)
            devices = G.ggml_devices(engine.L)
            require(offload == [("65", "65")] and memory and float(memory[-1]) >= 15000 and engine.gpu_layers == 999
                    and len(devices) == 1 and devices[0]["provider"] == "cuda", "unverified_device_or_layers")
            native = loaded_objects(engine.L.llama_cpp._lib)
            require(native["backend/CUDA"] == BACKEND, "unverified_cuda_backend")
            report.update(status="running", native=native, offloaded_layers=65, cuda_model_mib=float(memory[-1]),
                          kg=engine.kg)
            dump(args.output, report)
            tok = GgufTokenizer(str(args.gguf))
            try:
                for rep in range(REPEATS):
                    for width in (WIDTHS if rep % 2 == 0 else reversed(WIDTHS)):
                        prompts = compare.performance_prompts(args.seed + rep * 1009 + width, width, width)
                        ids = [tok.apply_chat_template(compare.conversation(p), enable_thinking=False) for p in prompts]
                        jobs = [G.Job(tokens, 32) for tokens in ids]
                        engine._clear()
                        before = native_hits()
                        start = time.monotonic_ns()
                        engine.run(jobs)
                        elapsed = (time.monotonic_ns() - start) / 1e9
                        hits = delta(before, native_hits())
                        require(all(j.reason in ("eos", "length") and j.tokens
                                    and start <= j.first_ns <= j.finished_ns for j in jobs), "incomplete_wave")
                        times = [(j.first_ns - start) / 1e6 for j in jobs]
                        work = [[len(j.tokens), j.reason] for j in jobs]
                        count = sum(n for n, _ in work)
                        report["waves"].append({"repeat": rep, "width": width, "elapsed_s": elapsed,
                            "output_tokens": count, "output_tok_s": count / elapsed, "ttft_ms": times,
                            "ttft_p95_ms": p95(times), "work": work, "prompt_ids_sha256": digest(ids),
                            "output_ids_sha256": digest([j.tokens for j in jobs]), "hits": hits})
                        dump(args.output, report)
                for key, chunks in (("nll", NC.gate_chunks(tok)), ("nll_b32", NC.gate_chunks(tok, chunks=16, size=32))):
                    require(len(chunks) == 16, "incomplete_nll_fixture")
                    report[key] = []
                    for lead, ids in chunks:
                        report[key].append(engine.chunk_nll(ids, lead))
                        dump(args.output, report)
                chunks = NC.gate_chunks(tok, chunks=21, size=16)
                require(len(chunks) == 21, "incomplete_low_nll_fixture")
                report["nll_low"], report["nll_low_hits"] = {}, {}
                at = 0
                for width in (6, 7, 8):
                    before = native_hits()
                    report["nll_low"][str(width)] = low_nll(engine, chunks[at:at + width])
                    report["nll_low_hits"][str(width)] = delta(before, native_hits())
                    at += width
                    dump(args.output, report)
                for suffix in (("", "_repeat") if args.arm == "stock" else ("",)):
                    report["facts" + suffix] = F.ggml_scores(engine, tok, seed=report["facts_seed"])
                    dump(args.output, report)
                    before = native_hits()
                    report["facts_low" + suffix] = low_facts(engine, tok, report["facts_seed"], F, G.Job)
                    report["facts_low" + suffix + "_hits"] = delta(before, native_hits())
                    dump(args.output, report)
                if scoped:
                    report["scope"] = engine.summary()["kg4_scope"]
                    lib = ctypes.CDLL(G.KG_LIB)
                    masks = []
                    for getter in (lib.ironmule_kg_get_mask, lib.ironmule_kg_get_mask6):
                        getter.restype = ctypes.c_uint32
                        masks.append(getter())
                    report["masks_after"] = masks
                    # A real B8 decode outside a request cohort must now use only the stock backend.
                    engine._clear()
                    token = engine.tokenize("A")[0]
                    before = native_hits()
                    engine._decode([(token, 0, seq, True) for seq in range(8)])
                    report["disable_hits"] = delta(before, native_hits())
            finally:
                tok.close()
        finally:
            engine.close()
        require(identity(args.gguf) == stamp, "gguf_changed")
        report["status"] = "measured"
        dump(args.output, report)
    except Exception as exc:
        report.update(status="incomplete", error=type(exc).__name__)
        dump(args.output, report)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("stock", "scoped", "stock-closing"), required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--capsule-dir", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    measure(parser.parse_args())
