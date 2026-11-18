"""Time only an independently qualified cold long-prompt shape. No policy learning or deployment."""

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from ironmule.long_prefill_study import (
    SCHEMA,
    WIDTHS,
    local_identity,
    measured_shapes,
    protocol_hash,
    qualified_receipt,
    uuid_hash,
)


def run(args):
    from ironmule import factual_gate as F, ggml_engine as G, numeric_choice as N
    from ironmule.compare_learning import digest
    from ironmule.gguf_tokenizer import GgufTokenizer
    from ironmule.long_prefill_gate import check
    from ironmule_product import compare
    from ironmule_product.state import _atomic_write
    from lab.portable_gpu.prefill_gate_probe import captured_stderr

    gpu = args.provider == "cuda"
    report = {
        "schema": SCHEMA,
        "status": "running",
        "provider": args.provider,
        "waves": [],
        "seqs": 32,
        "seq_ctx": 2048,
        "threads": args.threads,
        "protocol_sha256": protocol_hash(),
        "stage": "binding",
    }

    def save(label):
        report["stage"] = label
        _atomic_write(args.output, report)

    save("binding")
    e = tok = None
    try:
        # Use the same single CUDA UUID mapping as the HTTP supervisor, before GGML opens a context.
        if gpu:
            from ironmule_product.compare_servers import _device

            env = os.environ.copy()
            selected, _ = _device(env, "ironmule")
            if not selected.startswith("GPU-"):
                raise ValueError("one NVIDIA CUDA GPU is required")
            os.environ["CUDA_VISIBLE_DEVICES"] = env["CUDA_VISIBLE_DEVICES"]
        sha = G.file_sha256(str(args.gguf))
        if (
            gpu
            and sha
            != "efd971561896866f0e910cce52761ca77b1b138090c7f15fe284676d57d1f689"
        ):
            raise ValueError("wrong Qwen3-32B GGUF")
        report["gguf_sha256"] = sha
        # Keep one loaded model; rebuild only its KV context between split and independent default.
        with captured_stderr() as log:
            e = G.GgmlEngine(
                str(args.gguf),
                seqs=32,
                seq_ctx=2048,
                speculate=False,
                threads=args.threads,
                flash_attn=True,
                kv_unified=False,
                shared=False,
            )
            log.seek(0)
            loads = re.findall(
                r"offloaded (\d+)/(\d+) layers to GPU",
                log.read().decode(errors="replace"),
            )
        if gpu:
            if loads != [("65", "65")] or G.ggml_devices(e.L)[0]["provider"] != "cuda":
                raise ValueError("65/65 CUDA placement unverified")
            binding = local_identity(e)
            uid = binding.pop("cuda_uuid")
            report.update(
                **binding, gpu_uuid_sha256=uuid_hash(uid), offloaded_layers=[65, 65]
            )
        elif e.gpu_layers:
            raise ValueError("CPU smoke unexpectedly offloaded")
        report.update(settings=dict(e.settings), ubatch=int(e.L.llama_n_ubatch(e.ctx)))
        if report["ubatch"] != 512:
            raise ValueError("unmatched native microbatch")
        tok = GgufTokenizer(str(args.gguf))
        e.prefill_batch_long = e.prefill_batch = True
        save("cold_holdout")
        gate = report["long_gate"] = check(e, tok)
        if not gate["matches"]:
            report.update(status="rejected", reason="cold_holdout")
            return
        e.prefill_long_shapes = measured_shapes(gate)
        e.prefill_batch = False
        save("paired_performance")
        bench = argparse.Namespace(
            concurrency=list(WIDTHS), requests=8, seed=3591431646
        )
        for rep in range(3):
            groups = compare.workloads(bench, rep)
            for width in WIDTHS:
                ids = [
                    tok.apply_chat_template(
                        compare.conversation(p), enable_thinking=False
                    )
                    for p in groups[width][0]
                ]
                for active in (
                    (False, True)
                    if (rep + WIDTHS.index(width)) % 2 == 0
                    else (True, False)
                ):
                    e.prefill_batch = active
                    e._clear()
                    jobs = [G.Job(tokens, 32) for tokens in ids]
                    before = (e.prefill_long_pairs, e.prefill_long_quads)
                    start = time.perf_counter()
                    e.run(jobs)
                    elapsed = time.perf_counter() - start
                    if not all(
                        j.tokens and j.reason in ("eos", "length") for j in jobs
                    ):
                        raise ValueError("incomplete native generation")
                    report["waves"].append(
                        {
                            "mode": "long-444" if active else "off",
                            "repeat": rep,
                            "width": width,
                            "prompts_sha256": digest(ids),
                            "outputs_sha256": digest(
                                [(j.tokens, j.reason) for j in jobs]
                            ),
                            "output_tokens": sum(len(j.tokens) for j in jobs),
                            "seconds": elapsed,
                            "hits": {
                                "2": e.prefill_long_pairs - before[0],
                                "4": e.prefill_long_quads - before[1],
                            },
                        }
                    )
                    save("paired_performance")
        # Ordinary independent default-reference comparison required as well as a same-settings exact shape holdout.
        chunks = N.gate_chunks(tok)
        seed = 190117
        e.prefill_batch = True
        candidate = [e.chunk_nll(tokens, lead) for lead, tokens in chunks]
        facts = F.ggml_scores(e, tok, seed=seed)
        e.prefill_batch = False
        e.close_context()
        e._open(32, 2048, flash_attn=False, kv_unified=False)
        e._warm_up()
        stock = [e.chunk_nll(tokens, lead) for lead, tokens in chunks]
        ref = F.ggml_scores(e, tok, seed=seed)
        repeat = F.ggml_scores(e, tok, seed=seed)
        report.update(
            nll=N.ratio_interval(stock, candidate),
            factual=F.compare(ref, repeat, facts, seed=seed),
            status="measured",
        )
        report["qualifies_for_http"] = qualified_receipt(report)
    except BaseException as exc:
        report.update(status="incomplete", error=type(exc).__name__)
        raise
    finally:
        if tok:
            tok.close()
        if e:
            e.close()
        save("finished")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("gguf", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--budget-seconds", type=int, default=360)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if not 1 <= args.threads <= 256 or not 1 <= args.budget_seconds <= 3600:
        p.error("invalid thread or time limit")
    if args.worker:
        run(args)
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.close(os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"IRONMULE_KG_LIB", "IRONMULE_KG5", "IRONMULE_GGML_PREFILL_RECEIPT"}
    }
    env.update(
        IRONMULE_PILOT="0",
        IRONMULE_GGML_PREFILL_BATCH="long-444",
        IRONMULE_GGML_GPU_LAYERS="999" if args.provider == "cuda" else "0",
    )
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "lab.portable_gpu.long_prefill_native",
            *sys.argv[1:],
            "--worker",
        ],
        env=env,
    )
    try:
        return proc.wait(timeout=args.budget_seconds)
    except (KeyboardInterrupt, subprocess.TimeoutExpired):
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
