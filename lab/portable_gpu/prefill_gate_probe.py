"""Compare grouped and individual native prompts at qualification and production geometry without activating either."""

import argparse
import contextlib
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


@contextlib.contextmanager
def captured_stderr():
    with tempfile.TemporaryFile() as f:
        previous = os.dup(2)
        os.dup2(f.fileno(), 2)
        try:
            yield f
        finally:
            os.dup2(previous, 2)
            os.close(previous)


def first_difference(left, right):
    for pos, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return pos
    return min(len(left), len(right)) if len(left) != len(right) else None


def measure_jobs(engine, jobs, *, logits):
    """Identify actual native choices per sequence and prior token position, including their top-two margin."""
    import numpy as np

    shapes, top, slot = [], {}, {}
    sample = np.linspace(0, engine.n_vocab - 1, min(512, engine.n_vocab), dtype=np.intp)
    decode, group, claim = engine._decode, engine._group_prefill, engine._take_sequence

    def traced_decode(rows, **kwargs):
        picked = decode(rows, **kwargs)
        if logits:
            for index, (token, pos, seq, output) in enumerate(rows):
                if output:
                    view = np.ctypeslib.as_array(
                        engine.L.llama_get_logits_ith(engine.ctx, index),
                        shape=(engine.n_vocab,),
                    )
                    idx = np.argpartition(view, -2)[-2:]
                    hi, lo = sorted(idx, key=lambda i: float(view[i]), reverse=True)
                    top[(seq, pos)] = {
                        "top": int(hi),
                        "second": int(lo),
                        "margin": float(view[hi]) - float(view[lo]),
                        "sample": view[sample].copy(),
                    }
        return picked

    def grouped(rows):
        shapes.append(
            {
                "sequences": [seq for seq, _, _ in rows],
                "suffix_rows": [len(j.ids) - k for _, j, k in rows],
            }
        )
        return group(rows)

    def take(ids):
        seq, keep = claim(ids)
        slot[id(ids)] = seq
        return seq, keep

    engine._clear()
    engine._group_prefill, engine._decode, engine._take_sequence = (
        grouped,
        traced_decode,
        take,
    )
    try:
        engine.run(jobs)
    finally:
        engine._group_prefill, engine._decode, engine._take_sequence = (
            group,
            decode,
            claim,
        )
    if any(not j.tokens or j.reason not in ("eos", "length") for j in jobs):
        raise ValueError("an isolated native job did not finish")
    return [(j.tokens, j.reason, slot[id(j.ids)]) for j in jobs], top, shapes


def exact(engine, tokens, limit):
    import numpy as np
    from ironmule.ggml_engine import Job

    records = []
    for enabled in (False, True, False):
        engine.prefill_batch = enabled
        jobs = [Job(ids, limit) for ids in tokens]
        outputs, scores, groups = measure_jobs(engine, jobs, logits=True)
        records.append((outputs, scores, groups))
    base, trial, closing = records
    details = []
    for idx, ((a, stop, seq), (b, bst, bseq)) in enumerate(
        zip(base[0], trial[0], strict=True)
    ):
        pos = first_difference(a, b)
        if pos is not None or stop != bst:
            p = len(tokens[idx]) - 1 + pos if pos is not None else None
            ref_scores, b_scores = base[1].get((seq, p)), trial[1].get((bseq, p))
            evidence = {}
            if pos is not None and b_scores is not None and ref_scores is not None:
                # All requests have real copied logits at this prefix up to their respective stop.
                # A projected distance is a mapping diagnostic, not a proof of numerical equivalence.
                distances = {}
                candidate_sample = b_scores["sample"] - b_scores["sample"].mean()
                for j, (ref_tokens, _, sid) in enumerate(base[0]):
                    other = base[1].get((sid, len(tokens[j]) - 1 + pos))
                    if pos < len(ref_tokens) and other is not None:
                        sample = other["sample"]
                        distances[j] = float(
                            np.sqrt(
                                np.mean(
                                    (candidate_sample - (sample - sample.mean())) ** 2
                                )
                            )
                        )
                evidence = {
                    "nearest_reference_request": min(distances, key=distances.get),
                    "own_centered_sample_rmse": distances[idx],
                    "min_centered_sample_rmse": min(distances.values()),
                    "sample_size": len(candidate_sample),
                }
            summary = lambda score: (
                None
                if score is None
                else {k: v for k, v in score.items() if k != "sample"}
            )
            details.append(
                {
                    "request": idx,
                    "first_changed_generated_position": pos,
                    "reference_stop": stop,
                    "candidate_stop": bst,
                    "reference_length": len(a),
                    "candidate_length": len(b),
                    "reference": summary(ref_scores),
                    "candidate": summary(b_scores),
                    **evidence,
                }
            )
    result = lambda arm: [(t, s) for t, s, _ in arm[0]]
    return {
        "prompt_sha256": digest(tokens),
        "prompt_lengths": [len(t) for t in tokens],
        "reference_sha256": digest(result(base)),
        "candidate_sha256": digest(result(trial)),
        "stock_repeat_matches": result(base) == result(closing),
        "candidate_matches": result(base) == result(trial),
        "native_groups": trial[2],
        "differences": details,
    }


def cached(folder):
    """Extract allowlisted fields only. Cached artifacts describe the original run when available."""
    if folder is None or not folder.is_dir():
        return []
    rows = []
    for file in sorted(folder.glob("*.json")):
        if file.stat().st_size > 2**20:
            continue
        obj = json.loads(file.read_text())
        if not isinstance(obj, dict):
            continue
        if isinstance(obj.get("streams"), dict):

            def gate(val):
                return {
                    k: val[k]
                    for k in (
                        "settings",
                        "passes",
                        "error",
                        "ratio",
                        "interval",
                        "prefill_batch",
                        "factual",
                    )
                    if k in val
                }

            rows.append(
                {
                    "role": "gate",
                    "passes": obj.get("passes"),
                    "streams": gate(obj["streams"]),
                    "shared": gate(obj.get("shared") or {}),
                    "settings": obj.get("settings"),
                }
            )
        elif type(obj.get("sequences")) is int:
            rows.append(
                {
                    "role": "width",
                    "sequences": obj["sequences"],
                    "fitting": obj.get("fitting"),
                    "rates": obj.get("rates"),
                }
            )
    return rows


def run(a):
    from ironmule import factual_gate as F, ggml_engine as G, numeric_choice as N
    from ironmule.ggml_binary import loaded_objects
    from ironmule.gguf_tokenizer import GgufTokenizer
    from ironmule_product.compare import conversation, performance_prompts
    from ironmule_product.state import _atomic_write

    gpu = a.provider == "cuda"
    sha = G.file_sha256(str(a.gguf))
    if (
        gpu
        and sha != "efd971561896866f0e910cce52761ca77b1b138090c7f15fe284676d57d1f689"
    ):
        raise ValueError("expected the official Qwen3-32B Q4_K_M checkpoint")
    state = cached(a.cache)
    settings = next(
        (
            r["streams"]["settings"]
            for r in state
            if r["role"] == "gate" and r["streams"].get("settings")
        ),
        None,
    )
    settings = settings or {"flash_attn": True, "kv_unified": False, "kv_f32": False}
    if not settings.get("flash_attn") or settings.get("kv_unified") is not False:
        raise ValueError(
            "candidate settings must be flash attention with separate KV streams"
        )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    os.close(os.open(a.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
    report = {
        "schema": "ironmule.prefill-gate-diagnostic.v1",
        "status": "starting",
        "gguf_sha256": sha,
        "engine_sha256": G.file_sha256(G.__file__),
        "provider": a.provider,
        "source_cache": state,
        "contexts": [],
    }
    save = lambda: _atomic_write(a.output, report)
    save()
    engine = None
    try:
        with captured_stderr() as log:
            engine = G.GgmlEngine(
                str(a.gguf),
                seqs=8,
                seq_ctx=2048,
                speculate=False,
                shared=False,
                threads=a.threads,
            )
            log.seek(0)
            text = log.read().decode(errors="replace")
        off = [
            (int(x), int(y))
            for x, y in re.findall(r"offloaded (\d+)/(\d+) layers to GPU", text)
        ]
        reg = G.ggml_devices(engine.L)
        if gpu and (
            off != [(65, 65)]
            or len(reg) != 1
            or reg[0]["provider"] != "cuda"
            or not engine.gpu_layers
        ):
            raise ValueError("65/65 actual CUDA offload was not verified")
        if not gpu and engine.gpu_layers:
            raise ValueError("CPU smoke unexpectedly offloaded")
        report.update(
            offloaded_layers=off, native=loaded_objects(engine.L.llama_cpp._lib)
        )
        report["status"] = "running"
        save()
        if gpu:
            from ironmule.kernel_gen import _visible_cuda_devices

            (dev,) = _visible_cuda_devices()
            report.update(device=dev["name"], device_key=digest(dev["uuid"]))
        tok = GgufTokenizer(str(a.gguf))
        try:
            ids = N._text_ids(tok)
            chunks = N.gate_chunks(tok)
            seeds = (0, 764519)
            if len(chunks) < 4:
                raise ValueError("incomplete NLL fixture")
            baseline = {
                str(seed): [F.ggml_scores(engine, tok, seed=seed) for _ in range(2)]
                for seed in seeds
            }
            nll = [engine.chunk_nll(tokens, lead) for lead, tokens in chunks]
            report["reference"] = {
                "seqs": 8,
                "context": 2048,
                "settings": dict(engine.settings),
                "factual": baseline,
                "nll": nll,
            }
            save()
            for width in (8, 32):
                engine.close_context()
                engine._open(width, 2048, **settings)
                engine._warm_up()
                row = {
                    "seqs": engine.seqs,
                    "context": engine.seq_ctx,
                    "settings": dict(engine.settings),
                    "requested_ubatch": 512,
                    "effective_ubatch": int(engine.L.llama_n_ubatch(engine.ctx)),
                    "factual": {},
                    "exact": {},
                }
                report["contexts"].append(row)
                save()
                # Stock canary at the shipped eight streams x 33 tokens, then targeted per-request margin diagnostics.
                engine.prefill_batch = True
                row["product_canary"] = engine.check_prefill_batch(ids)
                row["exact"]["shipping_8x33"] = exact(
                    engine, [ids[i * 33 : (i + 1) * 33] for i in range(8)], 8
                )
                if width == 32:
                    prompts = performance_prompts(3591431646 + 32, 32, 32)
                    mixed = [
                        tok.apply_chat_template(conversation(p), enable_thinking=False)
                        for p in prompts
                    ]
                    row["exact"]["actual_mixed_32"] = exact(engine, mixed, 8)
                    long = [p for p in mixed if len(p) == 444][:4]
                    if len(long) == 4:
                        row["exact"]["long_4x444"] = exact(engine, long, 8)
                save()
                engine.prefill_batch = True
                measured = [engine.chunk_nll(tokens, lead) for lead, tokens in chunks]
                row["nll"] = N.ratio_interval(nll, measured)
                for seed in seeds:
                    results = []
                    for enabled in (False, True):
                        engine.prefill_batch = enabled
                        results.append(F.ggml_scores(engine, tok, seed=seed))
                    ref, repeat = baseline[str(seed)]
                    same = F.compare(results[0], results[0], results[1], seed=seed)
                    row["factual"][str(seed)] = {
                        "against_default": F.compare(
                            ref, repeat, results[1], seed=seed
                        ),
                        "against_ungrouped_same_context": same,
                        "lost_vs_same_context": [
                            k for k in results[0] if results[0][k] and not results[1][k]
                        ],
                        "gained_vs_same_context": [
                            k for k in results[0] if results[1][k] and not results[0][k]
                        ],
                    }
                    save()
            report["status"] = "measured"
        finally:
            tok.close()
    except BaseException as exc:
        report.update(status="incomplete", error=type(exc).__name__)
        raise
    finally:
        if engine is not None:
            engine.close()
        save()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gguf", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument(
        "--cache", type=Path, help="existing batch arm's ggml-gate directory"
    )
    p.add_argument("--provider", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    for key in ("IRONMULE_KG_LIB", "IRONMULE_KG5"):
        os.environ.pop(key, None)
    os.environ.update(
        IRONMULE_PILOT="0",
        IRONMULE_GGML_PREFILL_BATCH="0",
        IRONMULE_GGML_GPU_LAYERS="999" if args.provider == "cuda" else "0",
    )
    run(args)
