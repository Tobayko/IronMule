"""Verify the independent same-device PR57 warm comparison from its persisted files."""
import argparse
import json
import math
from pathlib import Path
from statistics import median

from lab.pk6.replay_http import checksum, filenames, score, waves
from lab.pk6.warm_repeat import GGUF, SOURCE
from lab.portable_gpu.fetch_awq import REVISION, TOKENIZER_COMMIT, _manifest_digest

PROTOCOL = "f81300f9332e9a33afae7e2571ef6d367a8e95e9708937520f78063d1f0d9f95"
LABELS = ("K1", "vllm", "K2")


def read(path):
    return json.loads(path.read_bytes())


def check_timeline(path, label, engine, perf):
    """Reconcile each counted 32-client wave with the original streaming client trace."""
    records = [json.loads(line) for line in path.read_bytes().splitlines()]
    if len(records) != 32 * len(perf):
        raise ValueError("incomplete request timeline")
    previous_end = 0
    for i, wave in enumerate(perf):
        rows = records[32*i:32*(i+1)]
        starts = {r["wave_start"] for r in rows}
        if (len(starts) != 1 or min(starts) < previous_end
                or [r["index"] for r in rows] != list(range(32))):
            raise ValueError("unbound or overlapping timeline wave")
        for r in rows:
            chunks = r["chunk_times"]
            if (r["label"] != label or r["engine"] != engine or r["width"] != 32 or r["repeat"] != i
                    or not 0 < r["tokens"] <= 32 or not r["prompt_tokens"] > 0
                    or not chunks or any(not math.isfinite(t) for t in chunks)
                    or not r["wave_start"] <= r["t_send"] <= chunks[0] <= chunks[-1] <= r["t_done"]
                    or chunks != sorted(chunks) or not 0 < r["ttft_ms"] < math.inf):
                raise ValueError("invalid streamed request in timeline")
        firsts = sorted(r["ttft_ms"] for r in rows)
        if (sum(r["tokens"] for r in rows) != wave["tokens"]
                or sum(r["prompt_tokens"] for r in rows) != wave["prompt_tokens"]
                or [round(v, 1) for v in (median(firsts), firsts[30], firsts[-1])]
                != [wave[k] for k in ("ttft_p50_ms", "ttft_p95_ms", "ttft_max_ms")]):
            raise ValueError("HTTP aggregate differs from client timeline")
        previous_end = max(r["t_done"] for r in rows)


def result(directory):
    protocol = directory / "protocol.json"
    if checksum(protocol) != PROTOCOL:
        raise ValueError("different prospective protocol")
    bound = read(protocol)
    names = ["target.json", "completed.json", *filenames(LABELS)]
    for label in LABELS:
        names.extend(f"{phase}-{label}.json" for phase in ("start", "ready", "end"))
        names.append(f"timeline-{label}.jsonl")
    missing = [name for name in names if not (directory / name).is_file()]
    if missing:
        return {"status": "incomplete", "missing": missing, "protocol_sha256": PROTOCOL}
    target, final = (read(directory / name) for name in ("target.json", "completed.json"))
    hw = target["hardware"]
    if (target["source_commit"] != SOURCE or bound["source_commit"] != SOURCE or target["gguf_sha256"] != GGUF
            or hw["name"] != "NVIDIA A100-SXM4-80GB" or final["hardware"] != hw
            or final["process_order"] != list(LABELS) or target["awq"]["revision"] != REVISION
            or target["awq"]["verified"] is not True or target["awq"]["manifest_sha256"] != _manifest_digest()
            or target["awq"]["tokenizer_commit"] != TOKENIZER_COMMIT or target["awq"]["tokenizer_files_match"] is not True
            or target["versions"]["vllm"] != "0.11.0" or target["versions"]["transformers"] != "4.57.1"):
        raise ValueError("unbound hardware, model, source or software")
    previous = 0
    by_start = {}
    for label in LABELS:
        start, ready, end = (read(directory / f"{phase}-{label}.json") for phase in ("start", "ready", "end"))
        if (start["hardware"] != hw or end["hardware"] != hw or start["source_commit"] != SOURCE
                or not previous <= start["time"] < ready["time"] < end["time"] <= final["time"]):
            raise ValueError("starts overlap or differ from the measured device")
        expected_model = "compare" if label == "vllm" else "gguf/Qwen3-32B-Q4_K_M"
        if ready["model"] != expected_model or label != "vllm" and ready["ggml"]["gate"]["passes"] is not True:
            raise ValueError("server or quality gate differs")
        previous = end["time"]
        engine = "vllm" if label == "vllm" else "ironmule"
        perf = waves(directory, label, engine, 32)
        check_timeline(directory / f"timeline-{label}.jsonl", label, engine, perf)
        by_start[label] = {"median_tokens_s": median(row["tok_s"] for row in perf),
                           "rates": [row["tok_s"] for row in perf], "startup_s": ready["startup_s"]}
    scored = score(directory, LABELS)
    expected_work = all(set(row["completion_tokens"]) == {32 * int(key.rsplit("-", 1)[1])}
                        for key, row in scored["waves"].items())
    comparable = scored["facts_all_18"] and scored["fixed_token_work"] and expected_work
    by_start_ratios = {arm: by_start[arm]["median_tokens_s"] / by_start["vllm"]["median_tokens_s"] for arm in ("K1", "K2")}
    return {"status": "complete" if comparable else "work or factual grade differs", "comparable": comparable,
            "protocol_sha256": PROTOCOL, "hardware": hw, "versions": target["versions"],
            "gguf_sha256": GGUF, "awq_revision": REVISION, "quantization_match": False,
            "source_commit": SOURCE, "by_start": by_start,
            "ratio32": scored["original_scoring"]["K"]["tok_s_32_median"] / by_start["vllm"]["median_tokens_s"] if comparable else None,
            "ratios_by_start": by_start_ratios if comparable else None, **scored,
            "source_files": {name: checksum(directory / name) for name in names}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(result(parser.parse_args().directory), indent=2, allow_nan=False))
