#!/usr/bin/env python3
"""Put the bit-identical `K=3840` kernel into a real Gemma 12B decode and price it.

Only the projections whose reduction dimension is 3840 are swapped, and only their
matmul call: the module keeps its own weights, scales and biases, untouched and
unrequantised. Everything else in the step is the reference path.

Both arms decode the same prompt from a fresh cache, in balanced AB/BA blocks. Logits,
tokens and the KV cache are compared as bytes, because tokens agreeing is a weaker
statement than the state agreeing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b42_qmv_kernel import COMPILE_OPTIONS, K3840, run  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_K = 3840
PROMPT = "Explain in three sentences why unified memory changes how a laptop runs a language model."


class K3840QuantizedLinear(nn.Module):
    """The reference module with its matmul replaced. Same buffers, same bytes."""

    def __init__(self, source: nn.QuantizedLinear) -> None:
        super().__init__()
        self.weight = source.weight
        self.scales = source.scales
        self.biases = source.biases
        self.group_size = source.group_size
        self.bits = source.bits
        self.out_features = int(source.weight.shape[0])

    def __call__(self, x: mx.array) -> mx.array:
        shape = x.shape
        # Decode is (1, 1, K); anything wider is prefill and keeps the library path.
        if len(shape) != 3 or shape[0] != 1 or shape[1] != 1:
            return mx.quantized_matmul(
                x, self.weight, self.scales, self.biases,
                transpose=True, group_size=self.group_size, bits=self.bits,
            )
        out = run(K3840, self.weight, self.scales, self.biases, x[0],
                  self.out_features, TARGET_K)
        return out[None]


def _collect(module: nn.Module, found: list, prefix: str = "") -> None:
    """Every K=3840 projection, as (parent, attribute, original module, path)."""

    for name, child in list(module.children().items()):
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(child, list):
            for index, item in enumerate(child):
                if isinstance(item, nn.Module):
                    _collect(item, found, f"{path}.{index}")
            continue
        if isinstance(child, nn.QuantizedLinear):
            if child.weight.shape[1] * 32 // child.bits == TARGET_K:
                found.append((module, name, child, path))
        elif isinstance(child, nn.Module):
            _collect(child, found, path)


class Swapper:
    """One resident model, two states. The candidate shares the reference's buffers.

    Loading two models would put 14.5 GB in a 32 GB machine and push it into swap, which
    the ledger records as a cliff rather than a slope. The replacement module holds
    references to the same weight, scale and bias arrays, so switching costs nothing.
    """

    def __init__(self, language: nn.Module) -> None:
        self.sites: list = []
        _collect(language, self.sites)
        self.candidates = {
            id(original): K3840QuantizedLinear(original)
            for _, _, original, _ in self.sites
        }
        self.state = "reference"

    @property
    def paths(self) -> list[str]:
        return [path for _, _, _, path in self.sites]

    def use(self, state: str) -> None:
        if state == self.state:
            return
        for parent, name, original, _ in self.sites:
            setattr(
                parent, name,
                original if state == "reference" else self.candidates[id(original)],
            )
        self.state = state


def _vm_counters() -> dict:
    """Real paging, not occupancy: occupied swap can be old and never touched again."""

    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    values = {}
    for line in out.splitlines():
        match = re.match(r'"?([A-Za-z ,\-]+)"?:\s+(\d+)', line.strip())
        if match:
            values[match.group(1).strip()] = int(match.group(2))
    keep = ("Pageins", "Pageouts", "Swapins", "Swapouts", "Compressions", "Decompressions")
    return {key: values[key] for key in keep if key in values}


def _swap_used_bytes() -> int:
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    match = re.search(r"used\s*=\s*([\d.]+)M", out)
    return int(float(match.group(1)) * 1024 * 1024) if match else -1


def _decode(language, tokenizer, steps: int) -> dict:
    cache = language.make_cache()
    ids = mx.array(tokenizer.encode(PROMPT))[None]
    tokens, times, first = [], [], None
    for index in range(steps):
        began = time.perf_counter_ns()
        out = language.model(ids, cache=cache)
        logits = language.lm_head(out[:, -1, :])
        token = int(mx.argmax(logits, axis=-1).item())
        times.append(time.perf_counter_ns() - began)
        tokens.append(token)
        if index == 0:
            first = logits
        ids = mx.array([[token]])
    state = hashlib.sha256()
    for layer in cache:
        for part in (layer.keys, layer.values):
            if part is not None:
                mx.eval(part)
                state.update(bytes(memoryview(part[..., : layer.offset, :])))
    return {
        "tokens": tokens,
        "step_ns": times,
        "logits_sha256": hashlib.sha256(bytes(memoryview(first))).hexdigest(),
        "cache_sha256": state.hexdigest(),
        "hit_eos": any(token == tokenizer.eos_token_id for token in tokens),
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", default="mlx-community/gemma-3-12b-it-4bit")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--trace-arm", choices=["reference", "candidate"],
                        help="run a single arm for external tracing")
    parser.add_argument("--gap-s", type=float, default=0.5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.trace_arm:
        # One arm, one model resident, shaped for an external Metal System Trace.
        model, tokenizer = load(args.model)
        language = getattr(model, "language_model", model)
        swapper = Swapper(language)
        swapper.use(args.trace_arm)
        for _ in range(args.warmup):
            _decode(language, tokenizer, 4)
        time.sleep(args.gap_s)
        outcome = _decode(language, tokenizer, args.steps)
        totals = {
            "model": f"{args.model}:{args.trace_arm}",
            "steps": args.steps,
            "warmup": args.warmup,
            "gap_s": args.gap_s,
            "submit_ns_mean": 0.0,
            "wait_ns_mean": 0.0,
            "total_ns_mean": sum(outcome["step_ns"]) / len(outcome["step_ns"]),
            "total_ns_min": min(outcome["step_ns"]),
            "per_step": outcome["step_ns"],
            "tokens": outcome["tokens"],
            "logits_sha256": outcome["logits_sha256"],
            "cache_sha256": outcome["cache_sha256"],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(totals, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({key: value for key, value in totals.items()
                          if key not in ("per_step", "tokens")}, sort_keys=True))
        return 0

    model, tokenizer = load(args.model)
    language = getattr(model, "language_model", model)
    swapper = Swapper(language)
    changed = swapper.paths
    swapped = len(changed)
    print(f"swapped {swapped} projections, one resident model")

    for _ in range(args.warmup):
        for arm in ("reference", "candidate"):
            swapper.use(arm)
            _decode(language, tokenizer, 2)

    before_counters, before_swap = _vm_counters(), _swap_used_bytes()
    arms = {"reference": [], "reference_aa": [], "candidate": []}
    results = {}
    for block in range(args.blocks):
        forward = ("reference", "reference_aa", "candidate")
        order = forward if block % 2 == 0 else tuple(reversed(forward))
        for arm in order:
            swapper.use("candidate" if arm == "candidate" else "reference")
            outcome = _decode(language, tokenizer, args.steps)
            arms[arm].append(median(outcome["step_ns"]))
            results[arm] = outcome
    after_counters, after_swap = _vm_counters(), _swap_used_bytes()
    paging = {key: after_counters.get(key, 0) - value for key, value in before_counters.items()}

    def interval(numerator: str, denominator: str) -> dict:
        ratios = [n / d for n, d in zip(arms[numerator], arms[denominator])]
        rng = random.Random(20260909)
        draws = sorted(
            median([ratios[rng.randrange(len(ratios))] for _ in ratios]) for _ in range(10000)
        )
        return {
            "median": median(ratios),
            "min": min(ratios),
            "max": max(ratios),
            "ci95_low": draws[int(0.025 * len(draws))],
            "ci95_high": draws[int(0.975 * len(draws)) - 1],
            "blocks_below_one": sum(1 for value in ratios if value < 1.0),
            "ratios": ratios,
        }

    candidate_stats = interval("candidate", "reference")
    aa_stats = interval("reference_aa", "reference")
    reference_drift = max(arms["reference"]) / min(arms["reference"])
    aa_passes = aa_stats["ci95_low"] <= 1.0 <= aa_stats["ci95_high"]
    ratios = candidate_stats["ratios"]
    draws = None
    record = {
        "schema": "ironmule.qmv_k3840_model_ab.v1",
        "experiment_id": args.out.stem,
        "agent": "claude",
        "kind": "paired_ab",
        "status": "measured",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "question": "does the bit-identical K=3840 kernel make a real 12B decode step faster",
        "preregistered": {
            "sessions": 2,
            "blocks": args.blocks,
            "steps_per_arm": args.steps,
            "warmup_decodes_per_arm": args.warmup,
            "arms": ["reference", "reference_aa", "candidate"],
            "order": "balanced, forward on even blocks and reversed on odd",
            "statistical_unit": "the paired block, not the token",
            "confidence": "95% percentile bootstrap, 10000 resamples, seed 20260909",
            "success_rule": (
                "both sessions: the candidate's paired 95% interval entirely below 1.0, "
                "the A/A null control's interval containing 1.0, and tokens, logits, "
                "KV state and stop behaviour identical"
            ),
            "drift_rule": (
                "a session whose reference block medians span more than 1.5x is reported "
                "as drift-affected and cannot carry a GO"
            ),
            "abort_rule": "swapouts during a session mark it not evaluable",
            "budget": "two sessions, no extension, no block selection, no extra attempts",
        },
        "model": args.model,
        "swapped_projections": swapped,
        "swapped_paths": changed,
        "steps_per_arm": args.steps,
        "blocks": args.blocks,
        "compile_options": COMPILE_OPTIONS,
        "reference_step_ns_median": median(arms["reference"]),
        "candidate_step_ns_median": median(arms["candidate"]),
        "paired_ratio_median": candidate_stats["median"],
        "paired_ratio_min": candidate_stats["min"],
        "paired_ratio_max": candidate_stats["max"],
        "paired_ratio_ci95_low": candidate_stats["ci95_low"],
        "paired_ratio_ci95_high": candidate_stats["ci95_high"],
        "blocks_below_one": candidate_stats["blocks_below_one"],
        "aa_null_control": {key: value for key, value in aa_stats.items() if key != "ratios"},
        "aa_null_control_passes": aa_passes,
        "reference_drift_max_over_min": reference_drift,
        "reference_step_ns_first_block": arms["reference"][0],
        "reference_step_ns_last_block": arms["reference"][-1],
        "swap_used_before_bytes": before_swap,
        "swap_used_after_bytes": after_swap,
        "paging_during_run": paging,
        "eos_reference": results["reference"]["hit_eos"],
        "eos_candidate": results["candidate"]["hit_eos"],
        "reference_block_medians_ns": arms["reference"],
        "reference_aa_block_medians_ns": arms["reference_aa"],
        "candidate_block_medians_ns": arms["candidate"],
        "tokens_identical": results["reference"]["tokens"] == results["candidate"]["tokens"],
        "logits_identical": results["reference"]["logits_sha256"]
        == results["candidate"]["logits_sha256"],
        "kv_cache_identical": results["reference"]["cache_sha256"]
        == results["candidate"]["cache_sha256"],
        "reference_logits_sha256": results["reference"]["logits_sha256"],
        "candidate_logits_sha256": results["candidate"]["logits_sha256"],
        "reference_cache_sha256": results["reference"]["cache_sha256"],
        "candidate_cache_sha256": results["candidate"]["cache_sha256"],
        "peak_memory_bytes": int(mx.get_peak_memory()),
        "limits": [
            "one resident model; the candidate shares its buffers and only swaps the call",
            "one prompt, one machine, one MLX build",
            "prefill still uses the library path; only single-token decode is swapped",
        ],
        "mlx_version": mx.__version__,
        "device": {"platform": platform.platform(), "machine": platform.machine()},
        "git_revision": subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        ).stdout.strip(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: value for key, value in record.items()
                      if key in ("swapped_projections", "reference_step_ns_median",
                                 "candidate_step_ns_median", "paired_ratio_median",
                                 "paired_ratio_ci95_low", "paired_ratio_ci95_high",
                                 "blocks_below_one", "tokens_identical",
                                 "logits_identical", "kv_cache_identical",
                                 "peak_memory_bytes", "aa_null_control",
                                 "aa_null_control_passes", "reference_drift_max_over_min",
                                 "paging_during_run", "eos_reference", "eos_candidate")},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
