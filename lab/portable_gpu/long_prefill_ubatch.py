"""Bounded native-only physical ubatch study following a qualified CUDA long-444 measurement."""

from __future__ import annotations

import argparse
import ctypes as C
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from ironmule.long_prefill_study import (
    WIDTHS,
    local_identity,
    measured_shapes,
    qualified_receipt,
    uuid_hash,
)
from ironmule_product.state import _atomic_write

GIB = 1 << 30
MODEL = "efd971561896866f0e910cce52761ca77b1b138090c7f15fe284676d57d1f689"


def same_holdout(prior, current):
    if not current.get("matches") or set(current["shapes"]) != set(prior["shapes"]):
        return False
    for name, shape in current["shapes"].items():
        old = prior["shapes"][name]
        if not shape["matches"] or old["witnesses"] != shape["witnesses"]:
            return False
        # Identical held-out rendered prompts, IDs, stop, correctness in all three cold passes.
        if shape["cases"] != old["cases"]:
            return False
    return True


def memory(g, minimum):
    measured = g.gpu_memory_bytes(os.environ["CUDA_VISIBLE_DEVICES"])
    if measured is None or measured[0] < minimum:
        raise MemoryError("insufficient measured free CUDA memory")
    return int(measured[0])


def open_context(engine, physical, callback=None):
    engine.close_context()
    engine._study_ubatch = physical
    make = engine.L.llama_context_default_params

    def params():
        p = make()
        p.cb_eval = callback
        return p

    with patch.object(
        engine.L, "llama_context_default_params", params if callback else make
    ):
        engine._open(32, 2048, flash_attn=True, kv_unified=False)
        if int(engine.L.llama_n_ubatch(engine.ctx)) != physical:
            raise ValueError(
                "native context did not accept requested physical microbatch"
            )
        engine._warm_up()


def physical_witness(engine, tok, sizes, width, g):
    """Observe first-layer activation rows in the native scheduled graph, only in an untimed context."""
    from ironmule.long_prefill_gate import fixtures

    lib = (
        engine.L.llama_cpp._lib
    )  # resolve through the actual loaded libllama dependencies
    name, nrows = lib.ggml_get_name, lib.ggml_nrows
    name.argtypes, name.restype = [C.c_void_p], C.c_char_p
    nrows.argtypes, nrows.restype = [C.c_void_p], C.c_int64
    counts = Counter()
    callback_type = engine.L.llama_cpp.ggml_backend_sched_eval_callback

    @callback_type
    def observe(tensor, ask, _):
        if ask and name(tensor) == b"attn_norm-0":
            counts[int(nrows(tensor))] += 1
        return False  # host does not request tensor values or synchronization

    try:
        open_context(engine, sizes, observe)
        memory(g, GIB)
        counts.clear()
        engine._clear()
        rows = fixtures(tok, width)[:width]
        before = (engine.prefill_long_pairs, engine.prefill_long_quads)
        engine.run([g.Job(ids, 1) for _, ids in rows])
        after = (engine.prefill_long_pairs, engine.prefill_long_quads)
        hits = [b - a for a, b in zip(before, after, strict=True)]
        expected = [int(width == 2), int(width == 4)]
        threshold = 512 if sizes == 1024 else 1024
        return {
            "shape": [width, 444],
            "physical_cap": sizes,
            "group_hits": hits,
            "layer0_attention_norm_rows": dict(counts),
            "exercised": hits == expected
            and any(
                threshold < n <= sizes and count > 0 for n, count in counts.items()
            ),
        }
    finally:
        # Native retains callback pointer until the context is released.
        engine.close_context()


def speed_row(engine, ids, prior, g, digest):
    before = (engine.prefill_long_pairs, engine.prefill_long_quads)
    engine._clear()
    jobs = [g.Job(tokens, 32) for tokens in ids]
    t = time.perf_counter()
    engine.run(jobs)
    elapsed = time.perf_counter() - t
    after = (engine.prefill_long_pairs, engine.prefill_long_quads)
    outputs = [(j.tokens, j.reason) for j in jobs]
    hit = {str(w): n - b for w, n, b in zip((2, 4), after, before, strict=True)}
    row = {
        "width": prior["width"],
        "repeat": prior["repeat"],
        "seconds": elapsed,
        "prior_512_seconds": prior["seconds"],
        "prompts_sha256": digest(ids),
        "outputs_sha256": digest(outputs),
        "output_tokens": sum(len(j.tokens) for j in jobs),
        "hits": hit,
    }
    keys = ("prompts_sha256", "outputs_sha256", "output_tokens", "hits")
    row["matches"] = (
        all(row[k] == prior[k] for k in keys)
        and any(hit.values())
        and all(j.tokens and j.reason in ("length", "eos") for j in jobs)
        and math.isfinite(elapsed)
        and elapsed > 0
    )
    return row


def run(args, receipt, summary):
    from ironmule import ggml_engine as g
    from ironmule import numeric_choice as numeric
    from ironmule.compare_learning import digest
    from ironmule.gguf_tokenizer import GgufTokenizer
    from ironmule.long_prefill_gate import check
    from ironmule_product import compare, compare_servers
    from lab.portable_gpu.prefill_gate_probe import captured_stderr

    def save():
        _atomic_write(args.out_dir / "ubatch-study.json", summary)

    selected, _ = compare_servers._device(os.environ, "ironmule")
    if not selected.startswith("GPU-"):
        raise ValueError("physical CUDA placement required")
    sha = g.file_sha256(str(args.gguf))
    if sha != MODEL or not qualified_receipt(
        receipt, gguf_sha256=sha, **local_identity()
    ):
        summary.update(status="skipped", reason="local_identity")
        save()
        return
    # Exact pinned 32B model and default fp16 K/V, 64 layers × 8 KV heads × 128 dimensions.
    kv_bytes = 32 * 2048 * 64 * 8 * 128 * 4
    summary["free_before_model_bytes"] = memory(
        g, args.gguf.stat().st_size + kv_bytes + 2 * GIB
    )
    e = tok = None
    try:
        with captured_stderr() as log:
            e = g.GgmlEngine(
                str(args.gguf),
                seqs=32,
                seq_ctx=2048,
                threads=receipt["threads"],
                shared=False,
                flash_attn=True,
                kv_unified=False,
                speculate=False,
            )
            log.seek(0)
            placement = re.findall(rb"offloaded (\d+)/(\d+) layers to GPU", log.read())
        binding = local_identity(e)
        if (
            placement != [(b"65", b"65")]
            or not e.prefill_batch_long
            or e.shared
            or int(e.L.llama_n_ubatch(e.ctx)) != 512
            or e.settings != receipt["settings"]
            or not qualified_receipt(receipt, gguf_sha256=sha, **binding)
        ):
            raise ValueError("baseline context or device identity unverified")
        summary.update(
            offloaded_layers=[65, 65],
            gguf_sha256=sha,
            native_runtime_sha256=binding["native_runtime_sha256"],
            implementation_sha256=binding["implementation_sha256"],
            gpu_uuid_sha256=uuid_hash(binding["cuda_uuid"]),
        )
        tok = GgufTokenizer(str(args.gguf))
        e.prefill_batch = True
        e.prefill_long_shapes = measured_shapes(receipt["long_gate"], receipt["waves"])
        widths = e.prefill_long_shapes[444]
        chunks = numeric.gate_chunks(tok)
        ref = [e.chunk_nll(ids, lead) for lead, ids in chunks]
        free_512 = memory(g, GIB)
        e.close_context()
        context_bytes = max(kv_bytes, memory(g, GIB) - free_512)
        base = {
            (r["repeat"], r["width"]): r
            for r in receipt["waves"]
            if r["mode"] == "long-444"
        }
        bench = argparse.Namespace(
            concurrency=list(WIDTHS), requests=8, seed=3591431646
        )
        for physical in (1024, 2048):
            result = {"physical_ubatch": physical, "status": "running", "waves": []}
            path = args.out_dir / f"ubatch-{physical}.json"
            summary["candidates"][str(physical)] = "running"
            save()
            _atomic_write(path, result)
            try:
                result["free_before_context_bytes"] = memory(
                    g, context_bytes + (physical // 512) * GIB
                )
                width = max(widths)
                if physical == 2048 and width != 4:
                    result.update(status="skipped", reason="no_prior_quad_traffic")
                    continue
                actual = result["native_graph"] = physical_witness(
                    e, tok, physical, width, g
                )
                if not actual["exercised"]:
                    result.update(
                        status="rejected", reason="physical_rows_not_observed"
                    )
                    continue
                memory(g, context_bytes + (physical // 512) * GIB)
                open_context(
                    e, physical
                )  # callback-free context for cold comparison and timings
                result["free_after_context_bytes"] = memory(g, GIB)
                gate = check(e, tok)
                result["holdout"] = {
                    k: {
                        "matches": v["matches"],
                        "witnesses": v["witnesses"],
                        "cases": [
                            {
                                key: case[key]
                                for key in (
                                    "prompt_sha256",
                                    "reference",
                                    "repeat",
                                    "candidate",
                                )
                            }
                            for case in v["cases"]
                        ],
                    }
                    for k, v in gate["shapes"].items()
                }
                if not same_holdout(receipt["long_gate"], gate):
                    result.update(status="rejected", reason="cold_output_or_dispatch")
                    continue
                cand = [e.chunk_nll(ids, lead) for lead, ids in chunks]
                result["nll"] = {
                    "reference_512": ref,
                    "candidate": cand,
                    **numeric.ratio_interval(ref, cand),
                }
                if not result["nll"]["passes"]:
                    result.update(status="rejected", reason="paired_nll")
                    continue
                _atomic_write(path, result)
                for rep in range(3):
                    groups = compare.workloads(bench, rep)
                    for w in WIDTHS:
                        ids = [
                            tok.apply_chat_template(
                                compare.conversation(p), enable_thinking=False
                            )
                            for p in groups[w][0]
                        ]
                        row = speed_row(e, ids, base[rep, w], g, digest)
                        result["waves"].append(row)
                        _atomic_write(path, result)
                        if not row["matches"]:
                            break
                    if not result["waves"][-1]["matches"]:
                        break
                waves = result["waves"]
                if len(waves) == 6 and all(r["matches"] for r in waves):
                    ratios = {
                        str(w): statistics.median(
                            r["prior_512_seconds"] / r["seconds"]
                            for r in waves
                            if r["width"] == w
                        )
                        for w in WIDTHS
                    }
                    result.update(
                        status="measured",
                        median_ratio_vs_512=ratios,
                        candidate_gain=min(ratios.values()) >= 1
                        and max(ratios.values()) >= 1.03,
                    )
                else:
                    result.update(status="rejected", reason="workload_outputs_or_hits")
            except MemoryError:
                result.update(status="skipped", reason="memory_guard")
                break
            finally:
                e.close_context()
                summary["candidates"][str(physical)] = result["status"]
                _atomic_write(path, result)
                save()
        summary["status"] = "finished"
        save()
    finally:
        if tok:
            tok.close()
        if e:
            e.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("gguf", "receipt", "out-dir"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--budget-seconds", type=int, required=True)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if not 1 <= args.budget_seconds <= 3600:
        p.error("invalid time budget")
    args.out_dir.mkdir(exist_ok=True, parents=True)
    out = args.out_dir / "ubatch-study.json"
    if args.worker:
        run(args, json.loads(args.receipt.read_bytes()), json.loads(out.read_bytes()))
        return 0
    os.close(os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
    summary = {
        "schema": "ironmule.long-prefill-ubatch.v1",
        "status": "starting",
        "candidates": {},
    }
    _atomic_write(out, summary)
    try:
        if args.receipt.stat().st_size > 1_000_000 or not qualified_receipt(
            json.loads(args.receipt.read_bytes()), gguf_sha256=MODEL
        ):
            summary.update(status="skipped", reason="unqualified_512")
            _atomic_write(out, summary)
            return 0
        env = {
            k: v
            for k, v in os.environ.items()
            if k
            not in {
                "IRONMULE_KG_LIB",
                "IRONMULE_KG4_SCOPE",
                "IRONMULE_KG5",
                "IRONMULE_GGML_PREFILL_RECEIPT",
            }
        }
        env.update(
            IRONMULE_PILOT="0",
            IRONMULE_GGML_PREFILL_BATCH="long-444",
            IRONMULE_GGML_GPU_LAYERS="999",
        )
        cmd = [
            sys.executable,
            "-m",
            "lab.portable_gpu.long_prefill_ubatch",
            *(sys.argv[1:] if argv is None else argv),
            "--worker",
        ]
        child = subprocess.run(cmd, env=env, timeout=args.budget_seconds, check=False)
        if child.returncode:
            summary = json.loads(out.read_bytes())
            summary.update(status="incomplete", worker_exit=child.returncode)
            _atomic_write(out, summary)
            return 1
    except (TimeoutError, subprocess.TimeoutExpired, OSError, ValueError) as exc:
        summary = json.loads(out.read_bytes())
        summary.update(status="incomplete", reason=type(exc).__name__)
        _atomic_write(out, summary)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
