#!/usr/bin/env python3
"""Does the integration keep the small win the kernel study measured?

This times the shipped path, not the study prototype: the model is loaded through
`load_engine` with the knob on, so admission runs exactly as it would in service, and
the arms are switched with the module's own `enable` and `disable`. One resident model,
because the candidate shares the reference's buffers.

Preregistered before the run: 20 paired blocks of 32 greedy steps, three arms including
an A/A null control, balanced order, 95% percentile bootstrap over the paired blocks.
Success needs the candidate's interval entirely below 1.0 and the null control's
interval containing 1.0. Swapouts during a run make it not evaluable. No extension.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ironmule import qmv_k3840  # noqa: E402
from ironmule.runtime import Knobs  # noqa: E402
from ironmule.tune import load_engine  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT = "Explain in three sentences why unified memory changes how a laptop runs a language model."


def _vm_counters() -> dict:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    values = {}
    for line in out.splitlines():
        match = re.match(r'"?([A-Za-z ,\-]+)"?:\s+(\d+)', line.strip())
        if match:
            values[match.group(1).strip()] = int(match.group(2))
    keep = ("Pageins", "Pageouts", "Swapins", "Swapouts")
    return {key: values[key] for key in keep if key in values}


def _decode(language, tokenizer, steps: int, *, digests: bool = False) -> dict:
    """Timing and equality never share a loop: hashing 524 KB of logits per step would
    sit between the measured steps and change what the next one meets."""

    cache = language.make_cache()
    ids = mx.array(tokenizer.encode(PROMPT))[None]
    tokens, times, digest = [], [], hashlib.sha256()
    for _ in range(steps):
        began = time.perf_counter_ns()
        out = language.model(ids, cache=cache)
        logits = language.lm_head(out[:, -1, :])
        token = int(mx.argmax(logits, axis=-1).item())
        times.append(time.perf_counter_ns() - began)
        if digests:
            mx.eval(logits)
            digest.update(bytes(memoryview(logits)))
        tokens.append(token)
        ids = mx.array([[token]])
    state = hashlib.sha256()
    if digests:
        for layer in cache:
            for part in (layer.keys, layer.values):
                if part is not None:
                    mx.eval(part)
                    state.update(bytes(memoryview(part[..., : layer.offset, :])))
    return {
        "tokens": tokens,
        "step_ns": times,
        "logits_sha256": digest.hexdigest(),
        "cache_sha256": state.hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", default="mlx-community/gemma-3-12b-it-4bit")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    engine, tokenizer = load_engine(args.model, Knobs(k3840_matvec=True))
    admission = engine.k3840_admission
    if not admission:
        print(json.dumps({"state": "NOT STARTED", "reason": "admission refused"}))
        return 1
    language = getattr(engine.model, "language_model", engine.model)
    identity = engine.model_identity

    def use(arm: str) -> None:
        if arm == "candidate":
            if not any(isinstance(m, qmv_k3840.K3840QuantizedLinear)
                       for m in language.model.layers[0].self_attn.children().values()):
                qmv_k3840.enable(engine.model, identity)
        else:
            qmv_k3840.disable(engine.model)

    # Equality outside the timed region, so a claim rests on both arms being the same run.
    use("reference")
    reference_state = _decode(language, tokenizer, args.steps, digests=True)
    use("candidate")
    candidate_state = _decode(language, tokenizer, args.steps, digests=True)
    identical = {
        "tokens": reference_state["tokens"] == candidate_state["tokens"],
        "logits": reference_state["logits_sha256"] == candidate_state["logits_sha256"],
        "kv_cache": reference_state["cache_sha256"] == candidate_state["cache_sha256"],
    }

    for _ in range(args.warmup):
        for arm in ("reference", "candidate"):
            use(arm)
            _decode(language, tokenizer, 2)

    before = _vm_counters()
    arms: dict[str, list[float]] = {"reference": [], "reference_aa": [], "candidate": []}
    for block in range(args.blocks):
        forward = ("reference", "reference_aa", "candidate")
        order = forward if block % 2 == 0 else tuple(reversed(forward))
        for arm in order:
            use("candidate" if arm == "candidate" else "reference")
            arms[arm].append(median(_decode(language, tokenizer, args.steps)["step_ns"]))
    after = _vm_counters()
    paging = {key: after.get(key, 0) - value for key, value in before.items()}

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
        }

    candidate = interval("candidate", "reference")
    aa = interval("reference_aa", "reference")
    aa_passes = aa["ci95_low"] <= 1.0 <= aa["ci95_high"]
    drift = max(arms["reference"]) / min(arms["reference"])
    decision = (
        "INTEGRATION GO"
        if candidate["ci95_high"] < 1.0 and aa_passes and all(identical.values())
        and paging.get("Swapouts", 0) == 0 and drift <= 1.5
        else ("NO-GO" if not all(identical.values()) else "UNCLEAR")
    )

    record = {
        "schema": "ironmule.k3840_integration_bench.v1",
        "experiment_id": args.out.stem,
        "agent": "claude",
        "kind": "integration_ab",
        "status": "measured",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "question": "does the shipped opt-in path keep the confirmed 1.0 to 1.3 per cent",
        "decision": decision,
        "model": args.model,
        "knob": "k3840_matvec",
        "preregistered": {
            "blocks": args.blocks,
            "steps_per_arm": args.steps,
            "arms": ["reference", "reference_aa", "candidate"],
            "order": "balanced, forward on even blocks and reversed on odd",
            "statistical_unit": "the paired block",
            "confidence": "95% percentile bootstrap, 10000 resamples, seed 20260909",
            "success_rule": (
                "candidate interval entirely below 1.0, null control containing 1.0, "
                "identical tokens, logits and KV state, no swapouts, drift at most 1.5"
            ),
            "budget": "one run, no extension",
        },
        "admission": {key: value for key, value in admission.items()
                      if key not in ("admitted_paths", "declined_paths")},
        "identical": identical,
        "candidate_vs_reference": candidate,
        "aa_null_control": aa,
        "aa_null_control_passes": aa_passes,
        "reference_drift_max_over_min": drift,
        "reference_step_ns_median": median(arms["reference"]),
        "candidate_step_ns_median": median(arms["candidate"]),
        "reference_block_medians_ns": arms["reference"],
        "reference_aa_block_medians_ns": arms["reference_aa"],
        "candidate_block_medians_ns": arms["candidate"],
        "paging_during_run": paging,
        "peak_memory_bytes": int(mx.get_peak_memory()),
        "limits": [
            "one prompt, one machine, one MLX build, greedy, batch 1, ungrouped",
            "admission cost is paid at load and is not in these step times",
            "foreign load was measured, not removed",
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
                      if key in ("decision", "identical", "candidate_vs_reference",
                                 "aa_null_control", "aa_null_control_passes",
                                 "reference_drift_max_over_min", "paging_during_run",
                                 "reference_step_ns_median", "candidate_step_ns_median",
                                 "peak_memory_bytes", "admission")},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
