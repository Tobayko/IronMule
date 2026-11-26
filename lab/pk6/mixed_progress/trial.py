"""Paired Qwen3-32B HTTP serving on a fresh Linux GPU. Run only in a disposable patched checkout."""

from __future__ import annotations

import argparse
from contextlib import closing, nullcontext
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ironmule.factual_gate import cases, correct
from ironmule.ggml_profiles import FLAGS
from ironmule.gguf_tokenizer import GgufTokenizer
from ironmule_product import compare
from ironmule_product.compare_tokenizer import rendered_prompt_ids
from lab.pk6.first_token import phases
from lab.pk6.mixed_progress import plan
from lab.pk6.warm_repeat import GGUF, fetch, output, save, stop
from lab.portable_gpu import fetch_awq

ARMS = ("vllm", "R1", "C1", "C2", "R2")
SEEDS = (761401, 961507)
GGUF_NAME = "Qwen3-32B-Q4_K_M.gguf"


def board():
    s = output(
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    )
    name, uuid, driver, mem = s.split(", ")
    if "\n" in s or name != "NVIDIA L40S":
        raise ValueError("expected one physical L40S")
    return {
        "name": name,
        "device_sha256": hashlib.sha256(uuid.encode()).hexdigest(),
        "driver": driver,
        "memory_mib": int(mem),
    }


def wait(proc, server, deadline):
    while proc.poll() is None and time.monotonic() < deadline:
        try:
            if server["model_id"] in {
                r["id"] for r in fetch(server["base_url"] + "/v1/models")["data"]
            }:
                if server["model_id"] == "compare":
                    return {}
                h = fetch(server["base_url"] + "/health")
                if h["ready"] and h["ggml"]["gate"]["passes"] and h["device"] == "gpu":
                    return h["ggml"]
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(2)
    raise RuntimeError(f"server failed readiness, exit={proc.poll()}")


def quality_cases(seed):
    facts = list(cases(seed))
    facts += [c for c in cases(seed + 1) if c.category != "facts"]
    facts += [c for c in cases(seed + 2) if c.category == "arithmetic"][:2]
    return [
        (
            f,
            (
                "Background notes." + " Recorded the site temperature." * 64 + "\n"
                if i % 2
                else ""
            )
            + f.prompt,
        )
        for i, f in enumerate(facts)
    ]


def measure(args, arm, board_id, row_budget, deadline):
    competitor = arm == "vllm"
    eng = "vllm" if competitor else "ironmule"
    model = "compare" if competitor else "gguf/Qwen3-32B-Q4_K_M"
    server = {"base_url": "http://127.0.0.1:8002", "model_id": model}
    env = dict(os.environ)
    if competitor:
        cmd = [
            str(args.vllm / "bin/vllm"),
            "serve",
            str(args.models / "qwen3-32b-awq"),
            "--served-model-name",
            model,
            "--tokenizer",
            str(args.models / "tokenizers/qwen3-32b"),
            "--dtype",
            "half",
            "--quantization",
            "awq_marlin",
            "--generation-config",
            "vllm",
            "--max-model-len",
            "2048",
            "--max-num-seqs",
            "32",
            "--gpu-memory-utilization",
            "0.88",
            "--tensor-parallel-size",
            "1",
            "--host",
            "127.0.0.1",
            "--port",
            "8002",
        ]
    else:
        env.pop("IRONMULE_COMPARE_LOCAL_KEY", None)
        env.update(
            IRONMULE_GGML_SEQ_CTX="2048",
            IRONMULE_ADMISSION_LEARN="0",
            IRONMULE_LAB_MIXED_ROWS=str(row_budget if arm.startswith("C") else 0),
            IRONMULE_GGML_TRACE=str(args.out / f"trace-{arm}.txt"),
        )
        cmd = [
            sys.executable,
            "-m",
            "ironmule_cli",
            "serve",
            "--gguf",
            str(args.models / GGUF_NAME),
            "--state-dir",
            str(args.state),
            "--agent",
            "on",
            "--no-thinking",
            "--host",
            "127.0.0.1",
            "--port",
            "8002",
            "--api-key-env",
            "IRONMULE_COMPARE_LOCAL_KEY",
        ]
    if board() != board_id:
        raise ValueError("physical device changed")
    started = time.monotonic()
    save(
        args.out / f"start-{arm}.json",
        {"arm": arm, "unix": time.time(), "hardware": board_id},
    )
    with (args.out / f"serve-{arm}.log").open("xb") as log:
        proc = subprocess.Popen(
            cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            health = wait(proc, server, min(deadline, time.monotonic() + 3600))
            save(
                args.out / f"ready-{arm}.json",
                {"startup_s": time.monotonic() - started, "health": health},
            )
            compare.stream(eng, server, compare.WARMUP, 8, deadline)
            seeds = SEEDS if competitor else [SEEDS[int(arm[1:]) - 1]]
            with (
                nullcontext(None)
                if competitor
                else closing(GgufTokenizer(str(args.models / GGUF_NAME))) as tok
            ):
                for seed in seeds:
                    for repeat in range(3):
                        for width in (1, 8, 32):
                            # At width one alternate short and document requests across the three blocks.
                            group = compare.performance_prompts(
                                seed + repeat * 1009 + width, width, max(width, 8)
                            )
                            prompts = [group[repeat % 2]] if width == 1 else group
                            wave(
                                args.out,
                                eng,
                                server,
                                arm,
                                seed,
                                repeat,
                                width,
                                prompts,
                                None,
                                tok,
                                deadline,
                            )
                    facts = quality_cases(seed)
                    wave(
                        args.out,
                        eng,
                        server,
                        arm,
                        seed,
                        3,
                        32,
                        [p for _, p in facts],
                        [c for c, _ in facts],
                        tok,
                        deadline,
                    )
        finally:
            stop(proc)
    save(
        args.out / f"end-{arm}.json",
        {"arm": arm, "unix": time.time(), "hardware": board()},
    )
    time.sleep(2)


def wave(
    folder, engine, server, arm, seed, repeat, width, prompts, facts, tok, deadline
):
    ids = rendered_prompt_ids(
        engine, server, [compare.conversation(p) for p in prompts], deadline, tok
    )
    start = time.time()
    rows, elapsed = compare.batch(
        engine, server, prompts, width, 32, deadline, timeline=True
    )
    if [r["prompt_tokens"] for r in rows] != ids["prompt_tokens"]:
        raise ValueError("usage differs from preflight tokenization")
    record = {
        "seed": seed,
        "arm": arm,
        "repeat": repeat,
        "width": width,
        "unix_start": start,
        "unix_end": time.time(),
        "input_digest": ids["digest"],
        "elapsed_s": elapsed,
        "requests": [],
    }
    for i, r in enumerate(rows):
        item = {
            k: r[k]
            for k in (
                "tokens",
                "prompt_tokens",
                "ttft_ms",
                "visible_ttft_ms",
                "finish_reason",
                "chunk_times",
                "server_id",
                "t_send",
                "t_done",
            )
        }
        item["output_sha256"] = hashlib.sha256(r["text"].encode()).hexdigest()
        if facts is not None:
            item.update(case=facts[i].key, correct=correct(facts[i], r["text"]))
        record["requests"].append(item)
    # Append only fully received waves. The off-pod collector reads this file throughout the lease.
    with (folder / f"waves-{arm}.jsonl").open("a") as stream:
        stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("models", "out", "vllm", "state"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(mode=0o700, parents=True, exist_ok=False)
    hw = board()
    if output("sha256sum", str(args.models / GGUF_NAME)).split()[0] != GGUF:
        raise ValueError("GGUF digest differs")
    awq = fetch_awq.verify(args.models / "qwen3-32b-awq")
    version = output(
        str(args.vllm / "bin/python"),
        "-c",
        "import importlib.metadata as m;print(m.version('vllm'))",
    )
    if version != "0.11.0":
        raise ValueError("vLLM version differs")
    selected = plan.plan()["selected_rows"]
    original_flags = {key: os.environ.get(key) for key in FLAGS}
    save(
        args.out / "target.json",
        {
            "hardware": hw,
            "gguf_sha256": GGUF,
            "awq_revision": awq["revision"],
            "awq_manifest_sha256": awq["manifest_sha256"],
            "vllm_version": version,
            "rows": selected,
            "launcher_overrides": original_flags,
            "source": output("git", "rev-parse", "HEAD"),
            "engine_sha256": phases.checksum(Path("ironmule/ggml_engine.py")),
        },
    )
    deadline = time.monotonic() + float(os.environ["RP_DEADLINE"]) - time.time() - 35
    with (args.out / "device-load.csv").open("xb") as telemetry:
        sampler = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,utilization.gpu,memory.used,power.draw",
                "--format=csv,noheader,nounits",
                "--loop=2",
            ],
            stdout=telemetry,
            env={**os.environ, "TZ": "UTC"},
        )
        try:
            for arm in ARMS:
                # Native tokenization can initialize raw kernel defaults in this client process.
                for key, value in original_flags.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
                measure(args, arm, hw, selected, deadline)
        finally:
            sampler.terminate()
            sampler.wait()
    save(args.out / "completed.json", {"order": ARMS, "hardware": board()})


if __name__ == "__main__":
    main()
