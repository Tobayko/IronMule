"""Replay the PR57 warmed HTTP comparison; independently check any complete fresh registered series."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import tarfile
from contextlib import redirect_stdout
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from lab.adm2.verdict import final

ROOT = Path(__file__).resolve().parents[2]
POD37 = ROOT / "lab/pk1/results/pod37/pk6_all.tgz"
POD37_SHA = "8f9611b216daa8545be194df9fe98d67528c2d9ceb77415cef59eba3a605fe6c"


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def filenames(labels):
    return [f"http_{label}{suffix}.jsonl" for label in labels for suffix in ("_32", "q_8")]


def waves(directory, label, engine, width):
    path = directory / f"http_{label}_{width}.jsonl"
    rows = [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]
    count = 3 if width == 32 else 2
    if (len(rows) != count + 1 or "tok_s" in rows[0] or not 0 < rows[0]["warmup_s"] < math.inf
            or any(r["label"] != label or r["engine"] != engine for r in rows)
            or [r["repeat"] for r in rows[1:]] != list(range(count))):
        raise ValueError(f"missing warm-up, label or measured repeats in {path.name}")
    for row in rows[1:]:
        tokens, wall, rate = row["tokens"], row["wall_s"], row["tok_s"]
        if (row["width"] != width or type(tokens) is not int or not 1 <= tokens <= 32 * width
                or type(row["prompt_tokens"]) is not int or row["prompt_tokens"] <= 0
                or not re.fullmatch("[0-9a-f]{16}", row["texts"]) or not 0 < wall < math.inf
                or not 0 < row["ttft_p50_ms"] <= row["ttft_p95_ms"] <= row["ttft_max_ms"] <= wall * 1000 + .6):
            raise ValueError(f"unverified work or latency in {path.name}")
        # http_bench.py writes rates to .01 token/s and wall to .001 s.
        if not (tokens / (wall + .0005) - .00501 <= rate <= tokens / max(1e-9, wall - .0005) + .00501):
            raise ValueError(f"token rate inconsistent with rounded wall and delivered tokens in {path.name}")
        if width == 8 and (not re.fullmatch(r"(?:[0-9]|1[0-8])/18", row["quality"])
                           or len(row["quality_misses"]) != 18 - int(row["quality"].split("/")[0])):
            raise ValueError(f"invalid factual grading in {path.name}")
    return rows[1:]


def score(directory, *, historical):
    labels = ("K1", "K2", "R1", "R2", "vllm") if historical else ("K1", "K2", "vllm")
    rows, summaries = {}, {}
    for label in labels:
        engine = "vllm" if label == "vllm" else "ironmule"
        for width in (32, 8):
            group = waves(directory, label + ("q" if width == 8 else ""), engine, width)
            rows[label, width] = group
            summaries[f"{label}-{width}"] = {
                "completion_tokens": [r["tokens"] for r in group], "prompt_tokens": [r["prompt_tokens"] for r in group],
                "text_hashes": [r["texts"] for r in group], "ttft_p95_ms_median": median(r["ttft_p95_ms"] for r in group),
                "ttft_p50_ms_median": median(r["ttft_p50_ms"] for r in group)}
    fixed = all(len({(rows[label, w][i]["prompt_tokens"], rows[label, w][i]["tokens"]) for label in labels}) == 1
                for w, count in ((32, 3), (8, 2)) for i in range(count))
    facts = all(r["quality"] == "18/18" for label in labels for r in rows[label, 8])
    # Use the original comparator on exactly the verified files. It includes first measured waves.
    with TemporaryDirectory() as tmp:
        scoring = Path(tmp)
        for name in filenames(labels):
            (scoring / name).write_bytes((directory / name).read_bytes())
        out = io.StringIO()
        with redirect_stdout(out):
            final(SimpleNamespace(dir=str(scoring), final="K", ref="R"))
    result = json.loads(out.getvalue())
    return {"original_scoring": result, "fixed_token_work": fixed, "facts_all_18": facts, "waves": summaries,
            "source_files": {name: checksum(directory / name) for name in filenames(labels)}}


def historical():
    if checksum(POD37) != POD37_SHA:
        raise ValueError("different historical source archive")
    labels = ("K1", "K2", "R1", "R2", "vllm")
    with TemporaryDirectory() as tmp, tarfile.open(POD37, "r:gz") as tar:
        root = Path(tmp)
        for name in filenames(labels):
            (root / name).write_bytes(tar.extractfile("./" + name).read())
        result = score(root, historical=True)
        def archived(name):
            return tar.extractfile("./" + name).read().decode()
        if (result["original_scoring"] != json.loads(archived("final_K.txt")) or not result["fixed_token_work"]
                or not result["facts_all_18"] or archived("K.txt").strip() != "P"
                or "A100-SXM4-80GB" not in archived("serve_K1.log") or "Qwen3-32B-Q4_K_M" not in archived("serve_K1.log")
                or "vLLM API server version 0.11.0" not in archived("serve_vllm.log")
                or "awq_marlin" not in archived("serve_vllm.log")):
            raise ValueError("historical recomputation, software, target or protocol differs")
        return {**result, "experiment": "PR57 pod 37 H", "archive_sha256": POD37_SHA,
                "source_revision": archived("work/IronMule/COMMIT").strip(), "hardware": "NVIDIA A100-SXM4-80GB",
                "ironmule_model": "Qwen3-32B GGUF Q4_K_M", "reference": "vLLM 0.11.0 Qwen3-32B AWQ-Marlin"}


def fresh(directory):
    registration = directory / "preregistration.json"
    doc = json.loads(registration.read_bytes())
    protocol = doc["http_protocol"]
    if (doc["historical_H_artifacts_reused"] or not doc["distinct_from_prior_H"] or doc["models"]["quantization_match"]
            or protocol["fixture"] != {"client_widths": [32, 8], "max_tokens": 32, "seed": 20261119}
            or protocol["order"] != ["product-K1", "vllm-AWQ", "product-K2"]
            or any(checksum(ROOT / path) != pin for path, pin in doc["immutable_source_manifest"]["files"].items())):
        raise ValueError("fresh protocol or frozen harness differs")
    names = filenames(("K1", "K2", "vllm"))
    missing = [name for name in names if not (directory / name).is_file()]
    base = {"experiment": doc["experiment_id"], "registration_sha256": checksum(registration),
            "declared_hardware": doc["target"]["gpu"], "reference_repo": doc["models"]["vllm_reference"]["repo"],
            "ironmule_model_sha256": doc["models"]["ironmule_product"]["sha256"], "matched_quantization": False}
    if missing:
        return {**base, "status": "incomplete", "missing": missing}
    result = score(directory, historical=False)
    if any(result["source_files"][name] == value for name, value in historical()["source_files"].items() if name in names):
        raise ValueError("historical raw file supplied as a fresh arm")
    # The new registration is descriptive. Only pod 37 used a predeclared 1.10 threshold.
    rates = result.pop("original_scoring")
    ratios = rates.pop("verdict")
    comparable = result["fixed_token_work"] and result["facts_all_18"]
    return {**base, "status": "completed HTTP statistics" if comparable else "quality or fixed work differs",
            "per_engine": rates, "throughput_ratio": ratios["ratio_tok_s"] if comparable else None, **result}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fresh", type=Path, help="directory with registered fresh measurements; defaults to historical replay")
    args = p.parse_args()
    result = fresh(args.fresh) if args.fresh else historical()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
