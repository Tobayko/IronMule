"""Immutable worker harness for a fresh GGUF engine start, first admitted wave and subsequent work."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import os
import resource
import time
from contextlib import closing
from pathlib import Path


def process_age():
    # /proc/self/stat starttime includes interpreter and import time, in ticks since boot.
    fields = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
    return time.clock_gettime(time.CLOCK_BOOTTIME) - int(fields[19]) / os.sysconf("SC_CLK_TCK")


def mapped_vendor():
    """Hash the actual CUDA mappings after serving, outside the timed windows."""
    import re

    objects = {}
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(None, 5)
        if len(fields) != 6 or int(fields[2], 16) != 0:
            continue
        path = Path(fields[5])
        matched = re.fullmatch(r"(libcublas(?:Lt)?|libcudart)\.so(?:\.\d+)+", path.name)
        if not matched:
            continue
        tag = matched.group(1)
        if tag in objects:
            raise ValueError(f"multiple mapped vendor images for {tag}")
        with path.open("rb") as stream:
            info = os.fstat(stream.fileno())
            major, minor = fields[3].split(":")
            if (info.st_dev, info.st_ino) != (os.makedev(int(major, 16), int(minor, 16)), int(fields[4])):
                raise ValueError(f"mapped {tag} differs from file")
            objects[tag] = {"path": str(path), "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}
            after = os.fstat(stream.fileno())
            if (after.st_size, after.st_ctime_ns) != (info.st_size, info.st_ctime_ns):
                raise ValueError(f"vendor image changed while hashing {tag}")
    return objects


def main():
    import ctypes

    from native_trial import FLAGS, TEMPLATE, checksum
    from serving_trial import digest, run_wave

    from ironmule import ggml_engine as ge
    from ironmule.factual_gate import MAX_TOKENS, cases, correct
    from ironmule.ggml_binary import identity
    from ironmule.gguf_tokenizer import GgufTokenizer
    from ironmule.kernel_gen import fingerprint
    from ironmule_devices import ggml_devices
    from ironmule_product.compare import performance_prompts
    from lab.portable_gpu.bench import save

    p = argparse.ArgumentParser()
    p.add_argument("--gguf", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    args = p.parse_args()
    lib, table = (Path(os.environ[k]) for k in ("IRONMULE_KG_LIB", "IRONMULE_W4_TABLE"))
    flags = {"IRONMULE_" + k: os.environ.get("IRONMULE_" + k) for k in FLAGS}
    if flags != {"IRONMULE_" + k: v for k, v in FLAGS.items()}:
        raise ValueError("changed fixed flags")
    with closing(ge.GgmlEngine(str(args.gguf), seqs=32, seq_ctx=2048, speculate=True, flash_attn=True)) as eng:
        if eng.pilot is None or not eng.gpu_layers == 999:
            raise ValueError("inactive Pilot or incomplete GPU configuration")
        initialized_s = process_age()
        initial_threads = eng.threads
        plugin = ctypes.CDLL(str(lib))
        def masks():
            result = []
            for method in (plugin.ironmule_kg_get_mask, plugin.ironmule_kg_get_mask6):
                method.restype = ctypes.c_uint32
                result.append(method())
            return result

        def memory():
            d, = ggml_devices(eng.L)
            return d["total_memory"] - d["free_memory"]

        def wave(seed):
            fixture = [(TEMPLATE.format(x), "") for x in performance_prompts(seed, 32, 32)]
            counters = ("steps", "padded", "pack_calls", "pack_prompts", "pack_rows", "reused_tokens")
            before = {k: getattr(eng, k) for k in counters}
            result = run_wave(eng, tok, fixture, 32, {})
            result["geometry"] = {k: getattr(eng, k) - before[k] for k in counters}
            return result

        initial_bits, memory_log = masks(), [memory()]
        with closing(GgufTokenizer(str(args.gguf), eng.L)) as tok:
            cold = wave(args.seed)
            first_s = process_age()
            memory_log.append(memory())
            warm = wave(args.seed ^ 317)
            memory_log.append(memory())
            questions = cases(args.seed)
            factual = run_wave(eng, tok, [(TEMPLATE.format(q.prompt), "") for q in questions], MAX_TOKENS, {})
            memory_log.append(memory())
        facts = [correct(q, r["answer"]) for q, r in zip(questions, factual["answers"], strict=True)]
        mask = dict(eng.kg)
        used = mask.pop("restart_profile_reused", False)
        finish_bits = masks()
        if initial_bits != finish_bits or not eng.pilot.decisions:
            raise ValueError("active mask changed or no CPU decisions")
        stat = args.gguf.stat()
        binding = {"model_sha256": checksum(args.gguf), "model_stat": [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns],
                   "native": identity(), "fp": fingerprint(), "w4_sha256": checksum(lib), "table_sha256": checksum(table),
                   "settings": eng.settings, "seqs": eng.seqs, "seq_ctx": eng.seq_ctx, "threads": initial_threads, "max_rows": eng.max_rows,
                   "speculate": eng.speculate, "gpu_layers": eng.gpu_layers, "affinity": sorted(os.sched_getaffinity(0)),
                   "flags": flags, "measure_kg_sha256": hashlib.sha256(inspect.getsource(ge.GgmlEngine.measure_kg).encode()).hexdigest()}
        def deliveries(raw):
            return [{k: row[k] for k in ("output_ids", "prompt_digest", "answer", "stop")} for row in raw["answers"]]

        def summarize(raw):
            answers = deliveries(raw)
            return {"wall_s": raw["wall_s"], "ttfts_s": raw["ttfts_s"], "answers": answers, "geometry": raw["geometry"],
                    "completed": raw["completed"], "tokens": raw["tokens"], "output_sha256": digest(answers)}

        save(args.output, {"schema": "ironmule.qwen-restart.arm.v1", "seed": args.seed, "cold_s": first_s,
                           "initialized_s": initialized_s, "mask": mask, "mask_bits": initial_bits, "reused": used,
                           "cold": summarize(cold), "warm": summarize(warm), "facts": facts, "cuda_vendor": mapped_vendor(),
                           "factual": deliveries(factual), "binding": binding, "pilot_decisions": eng.pilot.decisions,
                           "source_sha256": checksum(ge.__file__), "pilot_sha256": checksum(eng.pilot.weights_path),
                           "startup_prices": {"pilot": eng.pilot.costs, "padding": eng.pad_to,
                                              "passes": eng.pass_costs.points if eng.pass_costs else []},
                           "harness_sha256": checksum(__file__), "device_used_bytes": memory_log,
                           "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})


if __name__ == "__main__":
    main()
