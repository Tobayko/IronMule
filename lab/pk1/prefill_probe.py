"""Select and run one bounded, synthetic-input cuBLASLt information experiment for a measured Qwen3 prefill."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import re
import subprocess
import tempfile
import time
from pathlib import Path
from statistics import median

from training.code_author import digest, write_once

SOURCE = Path(__file__).with_suffix(".cu")
SCHEMA = "ironmule.prefill-gemm-probe.v1"


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def plan(trace, order_path):
    raw = trace.read_text()
    order = json.loads(order_path.read_text())
    identity = order.pop("order_id")
    profile_path = order_path.with_suffix(".profile.json")
    if (identity != hashlib.sha256(json.dumps(order, sort_keys=True).encode()).hexdigest()
            or checksum(profile_path) != order["profile_sha256"]
            or f"Native order {identity} for " not in raw):
        raise ValueError("unmatched raw trace, measured profile or development order")
    profile = json.loads(profile_path.read_text())
    if (profile["kind"] != "measured_native_diagnostic" or profile["binding"] != order["binding"]
            or profile["base_commit"] != order["base_commit"] or not re.search(r"general\.architecture\s+str\s+= qwen3\b", raw)):
        raise ValueError("unsupported prefill graph or unbound profile")

    def match(pattern):
        found = re.findall(pattern, raw, re.MULTILINE)
        if len(found) != 1:
            raise ValueError(f"ambiguous or missing trace item: {pattern}")
        return found[0]

    def dimension(name):
        return int(match(r"^print_info:\s+" + name + r"\s+= (\d+)\s*$"))

    width, prompt = profile["width"], profile["prompt_tokens_per_client"]
    n, wall = match(r"^prefill (\d+) rows in ([\d.]+) s")
    layers, hidden, expansion = (dimension(key) for key in ("n_layer", "n_embd", "n_ff"))
    n = int(n)
    if not 1 < layers <= 256 or not 128 <= n <= 8192 or n != width * prompt or not 0 < hidden < expansion <= 32768:
        raise ValueError("unsupported prefill shape or row count")
    area = raw.split("== prefill:")[-1].split("decode ", 1)[0]
    events = [(float(ms), int(calls), kernel.strip()) for ms, calls, kernel in
              re.findall(r"^\s*([\d.]+) ms\s+[\d.]+% x(\d+)\s+(.+)$", area, re.MULTILINE)]
    reference_shapes = {(row[1], row[2]) for row in profile["dispatch_rows_type_n_k_batch_variant_ksplit"]}
    targets = []
    # llm_build_qwen3 gathers output rows before the last layer's FFN. The other FFNs use every prompt row.
    for family, m, k, multiplicity in (("gate_up", expansion, hidden, 2), ("down", hidden, expansion, 1)):
        calls = (layers - 1) * multiplicity
        candidates = [(ms, name) for ms, count, name in events
                      if count == calls and name.startswith("ampere_s16816gemm_fp16_")]
        if len(candidates) != 1 or (m, k) not in reference_shapes:
            raise ValueError(f"ambiguous dense GEMM attribution for {family}")
        ms, symbol = candidates[0]
        if not math.isfinite(ms) or ms <= 0:
            raise ValueError("invalid device duration")
        targets.append({"family": family, "m": m, "n": n, "k": k, "calls": calls, "trace_ms": ms, "symbol": symbol,
                        "simulated_wave_saving_ms_at_10pct": ms / 10})
    targets.sort(key=lambda row: -row["trace_ms"])
    return {"schema": SCHEMA, "kind": "information_experiment", "parent_order": identity, "base_commit": order["base_commit"],
            "trace_sha256": checksum(trace), "profile_sha256": order["profile_sha256"], "binding": profile["binding"],
            "source_sha256": checksum(SOURCE), "planner_sha256": checksum(Path(__file__)), "profiled_prefill_s": float(wall),
            "targets": targets, "contract": "column-major op(A)=T, op(B)=N, lda=ldb=K, ldc=M, fp16 inputs, "
            "fp32 accumulation and output, alpha=1, beta=0; synthetic deterministic inputs",
            "uncertainty": "Matching call counts, graph dimensions and dispatch shapes identify likely groups. "
            "Per-symbol CUPTI totals do not contain actual cuBLAS API arguments. An isolated speed difference "
            "does not establish a serving benefit or authorize a variant."}


def run(order, output):
    from ironmule.kernel_gen import cuda_compiler_for, fingerprint

    if order["schema"] != SCHEMA or order["source_sha256"] != checksum(SOURCE) or order["planner_sha256"] != checksum(Path(__file__)):
        raise ValueError("probe source or planner identity changed")
    request = digest(order)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(output) + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if output.exists():
            cached = json.loads(output.read_text())
            if cached["request_id"] != request:
                raise ValueError("existing outcome belongs to another plan")
            if cached["status"] != "completed":
                raise RuntimeError(f"cached failed probe: {cached.get('reason')}")
            return cached
        target = order["binding"]["fp"]
        hardware = fingerprint()
        if any(hardware[key] != target[key] for key in ("devices", "cc", "driver", "nvcc")) or len(hardware["devices"]) != 1:
            raise ValueError("GPU, driver or CUDA toolkit differs from observed prefill")
        attempt = Path(str(output) + ".attempt.json")
        if attempt.exists():
            raise RuntimeError("a prior attempt exists without its final receipt; inspect before starting a new run")
        write_once(attempt, {"request_id": request, "targets": order["targets"]})
        result = {"schema": SCHEMA, "request_id": request, "kind": "isolated_measurement", "targets": []}
        start = time.monotonic()
        try:
            compiler = cuda_compiler_for([hardware["cc"]])
            if not compiler:
                raise ValueError("no matching CUDA toolkit")
            with tempfile.TemporaryDirectory() as work:
                binary = Path(work) / "prefill-probe"
                subprocess.run([compiler, "-O3", "-std=c++17", f"-arch=sm_{hardware['cc']}", str(SOURCE),
                                "-lcublas", "-lcublasLt", "-o", str(binary)], check=True, timeout=120)
                result["compile_s"] = time.monotonic() - start
                result["binary_sha256"] = checksum(binary)
                for shape in order["targets"]:
                    command = [str(binary), *[str(shape[k]) for k in ("m", "n", "k")], "572"]
                    trial_start = time.monotonic()
                    done = subprocess.run(command, text=True, capture_output=True, timeout=90, check=False)
                    rows = [json.loads(line) for line in done.stdout.splitlines()]
                    measured = {"shape": shape, "return_code": done.returncode, "stderr": done.stderr[-2000:], "events": rows}
                    measured["elapsed_s"] = time.monotonic() - trial_start
                    result["targets"].append(measured)
                    if done.returncode or not rows or rows[0].get("kind") != "info":
                        raise RuntimeError("probe failed on real hardware")
                    savings = []
                    for index in range(rows[0]["heuristics"]):
                        pairs = [r for r in rows if r.get("index") == index and r.get("status") == "paired"]
                        if pairs:
                            if [r["pair"] for r in pairs] != list(range(4)):
                                raise ValueError("incomplete measured comparison")
                            savings.append({"index": index, "algo_id": pairs[0]["algo_id"],
                                            "projected_wave_ms": shape["calls"] * median(
                                                r["reference_ms"] - r["candidate_ms"] for r in pairs)})
                    measured["extrapolations"] = savings
            result["status"] = "completed"
        except Exception as error:
            result["status"] = "failed"
            result["reason"] = f"{type(error).__name__}: {error}"
            result["elapsed_s"] = time.monotonic() - start
            write_once(output, result)
            raise
        result["elapsed_s"] = time.monotonic() - start
        write_once(output, result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--order", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--measure", type=Path, help="execute the bounded GPU micro-screen and write this receipt")
    args = parser.parse_args()
    order = plan(args.trace, args.order)
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    write_once(args.plan, order)
    if args.measure:
        run(order, args.measure)


if __name__ == "__main__":
    main()
