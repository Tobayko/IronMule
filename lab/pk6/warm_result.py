"""Verify the independent same-device PR57 warm comparison from its persisted files."""
import argparse
import json
from pathlib import Path
from statistics import median

from lab.pk6.replay_http import checksum, filenames, score, waves
from lab.pk6.warm_repeat import GGUF, SOURCE
from lab.portable_gpu.fetch_awq import REVISION, _manifest_digest

PROTOCOL = "f81300f9332e9a33afae7e2571ef6d367a8e95e9708937520f78063d1f0d9f95"
LABELS = ("K1", "vllm", "K2")


def read(path):
    return json.loads(path.read_bytes())


def result(directory):
    protocol = directory / "protocol.json"
    if checksum(protocol) != PROTOCOL:
        raise ValueError("different prospective protocol")
    bound = read(protocol)
    names = ["target.json", "completed.json", *filenames(LABELS)]
    for label in LABELS:
        names.extend(f"{phase}-{label}.json" for phase in ("start", "ready", "end"))
    missing = [name for name in names if not (directory / name).is_file()]
    if missing:
        return {"status": "incomplete", "missing": missing, "protocol_sha256": PROTOCOL}
    target, final = (read(directory / name) for name in ("target.json", "completed.json"))
    hw = target["hardware"]
    if (target["source_commit"] != SOURCE or bound["source_commit"] != SOURCE or target["gguf_sha256"] != GGUF
            or hw["name"] != "NVIDIA A100-SXM4-80GB" or final["hardware"] != hw
            or final["process_order"] != list(LABELS) or target["awq"]["revision"] != REVISION
            or target["awq"]["verified"] is not True or target["awq"]["manifest_sha256"] != _manifest_digest()
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
