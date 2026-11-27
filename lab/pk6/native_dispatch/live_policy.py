"""Price the chosen prompt profile on CPU before a qualified engine opens its CUDA graphs."""

import json
import time

import numpy as np

from ironmule.pilot import Pilot
from lab.pk6.native_dispatch.learn import NAMES, features
from lab.pk6.replay_http import checksum
from lab.pk6.warm_repeat import GGUF


def choose(policy, hardware, native, receipt=None):
    started = time.perf_counter_ns()
    fit = json.loads((policy / "fit.json").read_bytes())
    if (
        fit["target"]["hardware"] != hardware
        or fit["target"]["gguf"] != GGUF
        or fit["target"]["ggml_cuda"] != native
        or checksum(policy / "pilot.npz") != fit["weights_sha256"]
    ):
        return {
            "profile": "stock",
            "reason": "unknown hardware or weights",
            "cpu_us": (time.perf_counter_ns() - started) / 1000,
        }
    with np.load(policy / "pilot.npz", allow_pickle=False) as f:
        heads = [[f[f"head_{i}_{j}"] for j in range(6)] for i in range(3)]
    decisions = []
    for r in fit["rows"]:
        decision = Pilot.startup_profile(
            heads,
            features(r["shape"]),
            NAMES,
            NAMES,
            "stock",
            in_support=True,
            margin_error=fit["calibration_max_log_error"],
        )
        decisions.append(decision["profile"])
        if decision["profile"] != r["trained_profile"]:
            raise ValueError("frozen hardware policy did not replay")
    possible = any(
        n == "fp8" and r["eligible"]
        for n, r in zip(decisions, fit["rows"], strict=True)
    )
    if possible:
        if (
            receipt is None
            or receipt["sources"]["fit.json"] != checksum(policy / "fit.json")
            or receipt["sources"]["target_rows.h"] != checksum(policy / "target_rows.h")
        ):
            raise ValueError("routed shared object was not built from this policy")
    return {
        "profile": "routed" if possible else "stock",
        "reason": "eligible intervals require product qualification"
        if possible
        else "no qualifying operator interval",
        "evaluated_shapes": len(decisions),
        "fp8_proposals": decisions.count("fp8"),
        "weights_sha256": fit["weights_sha256"],
        "cpu_us": (time.perf_counter_ns() - started) / 1000,
    }
