"""Native CUDA counterpart to the standalone-prefill CPU probe, using the same six bursts."""
import argparse
import contextlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

from prefill_observer_probe import SEED, WAVES, WIDTHS, digest, file_sha, run_wave


def layer_placement(log):
    layers = re.findall(r"offloaded (\d+)/(\d+) layers to GPU", log)
    buffers = re.findall(r"CUDA0\s+model buffer size\s*=\s*([\d.]+) MiB", log)
    if len(layers) != 1 or layers[0][0] != layers[0][1] or int(layers[0][0]) < 1 or not buffers:
        raise RuntimeError("full CUDA layer offload and model buffer did not appear in native logs")
    return {"offloaded_layers": int(layers[0][0]), "total_layers": int(layers[0][1]),
            "cuda_model_buffer_mib": float(buffers[-1])}


@contextlib.contextmanager
def capture_native_stderr():
    with tempfile.TemporaryFile() as stream:
        saved = os.dup(2)
        try:
            os.dup2(stream.fileno(), 2)
            yield stream
        finally:
            os.dup2(saved, 2)
            os.close(saved)


def physical(G, engine):
    from ironmule.kernel_gen import _visible_cuda_devices

    registry = G.ggml_devices(engine.L)
    devices = _visible_cuda_devices()
    if len(registry) != 1 or registry[0]["provider"] != "cuda" or len(devices) != 1:
        raise RuntimeError("the live GGML registry and CUDA driver must each identify one GPU")
    listing = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader"],
                                      text=True, timeout=5).splitlines()
    matched = [line.split(",")[1].strip() for line in listing
               if line.split(",")[0].strip().removeprefix("GPU-").replace("-", "").lower() == devices[0]["uuid"]]
    if len(matched) != 1:
        raise RuntimeError("CUDA driver GPU UUID differs from nvidia-smi inventory")
    return {"provider": "cuda", "name": devices[0]["name"], "compute_capability": devices[0]["cc"],
            "driver": matched[0], "cuda_uuid_matches_nvidia_smi": True,
            "device_key": digest(devices[0]["uuid"]), "backend": registry[0]["name"]}


def split_wave(engine, G):
    jobs = [G.Job(engine.tokenize(f"{i}. Next word:"), 3 if i % 2 == 0 else 1) for i in range(8)]
    native = engine._decode
    calls = []

    def decode(rows, **kwargs):
        # Every real prompt and every active decode sequence requests >=1 logit. A sequence with
        # zero logits in the entire native call can only be an injected inactive stream.
        has_output = {sid for _, _, sid, logits in rows if logits}
        gaps = sum(sid not in has_output for _, _, sid, _ in rows)
        calls.append({"rows": len(rows), "inactive_rows": gaps, "observer": []})
        return native(rows, **kwargs)

    def observe(decode_rows, prompt_rows, kv, seconds):
        calls[-1]["observer"].append([decode_rows, prompt_rows])

    engine._decode, engine.observer = decode, observe
    try:
        engine.run(jobs)
    finally:
        engine._decode, engine.observer = native, None
    gap_calls = [call for call in calls if call["inactive_rows"]]
    if not gap_calls or any(j.reason not in {"length", "eos"} or not j.tokens for j in jobs):
        raise ValueError("split-KV wave did not exercise interior inactive streams")
    return {"token_ids_and_stops_sha256": digest([(j.tokens, j.reason) for j in jobs]),
            "native_call_shapes_sha256": digest([(c["rows"], c["inactive_rows"]) for c in calls]),
            "requests": len(jobs), "output_tokens": sum(len(j.tokens) for j in jobs),
            "stop_reasons": dict(Counter(j.reason for j in jobs)), "gap_calls": gap_calls}


def stage(args):
    sys.path.insert(0, str(args.source.resolve()))
    from ironmule import ggml_engine as G
    from ironmule.ggml_binary import loaded_objects

    if Path(G.__file__).resolve() != args.source.resolve() / "ironmule/ggml_engine.py":
        raise ValueError("wrong engine source")
    waves, placements = [], []
    for width, shared in ((8, True), (32, True), (8, False)):
        with capture_native_stderr() as log:
            try:
                engine = G.GgmlEngine(str(args.gguf), threads=2, seqs=width, seq_ctx=128, shared=shared,
                                      cap=width, speculate=False, flash_attn=not shared)
            except Exception:
                log.seek(0)
                print(log.read()[-2500:].decode(errors="replace"), file=sys.stdout, flush=True)
                raise
            log.seek(0)
            native_placement = layer_placement(log.read().decode(errors="replace"))
        try:
            if engine.gpu_layers != 999 or engine.settings["kv_unified"] != shared:
                raise RuntimeError("GPU placement or KV layout differs")
            placements.append({"width": width, "shared_kv": shared, **native_placement, **physical(G, engine)})
            if shared:
                for wave in range(len(WAVES)):
                    started = time.perf_counter()
                    row = run_wave(engine, G, width, wave)
                    row["observations"]["wall_ms_with_tokenization"] = round((time.perf_counter() - started) * 1e3, 3)
                    waves.append(row)
            else:
                split = split_wave(engine, G)
            binary = loaded_objects(engine.L.llama_cpp._lib)
            if "backend/CUDA" not in binary:
                raise RuntimeError("registered backend did not resolve to a loaded CUDA image")
        finally:
            engine.close()
    return {"source_sha256": file_sha(G.__file__), "native": binary, "placements": placements,
            "waves": waves, "split": split}


def compare(a, b):
    if a["placements"] != b["placements"] or a["native"] != b["native"]:
        raise ValueError("actual devices, full layer offload or native binaries differ")
    waves = []
    for left, right in zip(a["waves"], b["waves"], strict=True):
        before, after = left.pop("observations"), right.pop("observations")
        if left != right:
            raise ValueError("model outputs, stops, prompt IDs or native call shapes changed")
        if not (before["calls"] == left["native_calls"] - left["standalone_calls"] and before["standalone_calls"] == 0
                and after["calls"] == left["native_calls"] and after["rows"] == left["native_rows"]
                and after["standalone_prompt_rows"] == left["standalone_rows"]):
            raise ValueError("before/after native coverage does not match expected observations")
        waves.append({**left, "before": before, "after": after})
    if any(not any(r["standalone_calls"] for r in waves if r["width"] == w) for w in WIDTHS):
        raise ValueError("standalone prefill did not occur at each measured width")
    old, new = a["split"].pop("gap_calls"), b["split"].pop("gap_calls")
    if a["split"] != b["split"] or len(old) != len(new):
        raise ValueError("uneven-length split-KV wave output or native call shapes changed")
    for l, r in zip(old, new, strict=True):
        n, gap = l["rows"], l["inactive_rows"]
        if (r["rows"], r["inactive_rows"]) != (n, gap) or l["observer"] != [[n-gap, gap]] or r["observer"] != [[n, 0]]:
            raise ValueError("inactive gap rows were not reclassified from prompt to decode")
    return {"placements": [{k: v for k, v in p.items() if k != "device_key"} for p in a["placements"]],
            "same_physical_gpu": True, "native_sha256": a["native"], "waves": waves,
            "split_kv": {**a["split"], "before": old, "after": new}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gguf", type=Path, required=True)
    p.add_argument("--before", type=Path)
    p.add_argument("--after", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--source", type=Path, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.source:
        print(json.dumps(stage(args)))
        return
    if not all([args.before, args.after, args.output, args.manifest]):
        p.error("--before, --after, --output, --manifest required")
    manifest = json.loads(args.manifest.read_text())
    if args.gguf.stat().st_size != manifest["model_bytes"] or file_sha(args.gguf) != manifest["model_sha256"]:
        raise ValueError("incorrect Qwen GGUF size or digest")
    stat = args.gguf.stat()
    env = {k: v for k, v in os.environ.items() if not k.startswith("IRONMULE_") or k == "IRONMULE_GGML_BINDINGS"}
    env.update(IRONMULE_GGML_GPU_LAYERS="999", IRONMULE_GGML_THREADS="2", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
    phases = []
    for side, path in (("before", args.before), ("after", args.after)):
        print(f"Starting {side} real CUDA workloads", file=sys.stderr, flush=True)
        child = subprocess.run([sys.executable, __file__, "--source", path, "--gguf", args.gguf],
                               env=env, stdout=subprocess.PIPE, text=True, check=False)
        if child.returncode:
            print(child.stdout[-2500:], file=sys.stderr)
            raise SystemExit(child.returncode)
        data = json.loads(child.stdout)
        if data["source_sha256"] != manifest[side]["engine_sha256"]:
            raise ValueError("measured engine differs from manifest")
        phases.append(data)
    now = args.gguf.stat()
    if (stat.st_dev, stat.st_ino, stat.st_size, stat.st_ctime_ns) != (now.st_dev, now.st_ino, now.st_size, now.st_ctime_ns):
        raise ValueError("GGUF changed during the paired run")
    result = {"schema": "ironmule.prefill-observer-cuda.v1", "model": args.gguf.name, "provenance": manifest,
              "protocol": {"scope": "real native inference and observation coverage", "seed": SEED, "seq_ctx": 128,
                           "shared_widths": list(WIDTHS), "split_width": 8, "max_output_tokens": 3,
                           "host_threads": 2, "speculation": False, "greedy": True}, **compare(*phases)}
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
