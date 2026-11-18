"""Opt-in Qwen3-32B/L40S workload guard for the measured KG4 Q4_K layout."""

from __future__ import annotations

import ctypes
import math
import os
from pathlib import Path
import statistics
import threading


MODE = "low-width8"
MODEL_SHA256 = "efd971561896866f0e910cce52761ca77b1b138090c7f15fe284676d57d1f689"
PLUGIN_SHA256 = "da0ff490ce4336e9b684166cfbe6486f7a9058ee68832114a9543d0b75c219cf"
BUCKETS = (6, 7, 8)
# The two masks belong to one loaded DSO and therefore to the process, even with multiple contexts.
_mask_lock = threading.Lock()


def _mask(buckets):
    return sum(1 << (b - 1) for b in buckets)


class Kg4Scope:
    def __init__(self, lib, buckets):
        self.lib = lib
        self.buckets = tuple(buckets)
        self.mask = _mask(self.buckets)
        self.low_runs = self.wide_runs = self.switches = 0
        self.enabled = self.low = self.wide = False

    @classmethod
    def calibrate(cls, engine, model, path):
        from .ggml_engine import file_sha256, ggml_devices

        devices = ggml_devices(engine.L)
        if (os.environ.get("GGML_CUDA_DISABLE_GRAPHS") != "1" or engine.gpu_layers <= 0
                or len(devices) != 1 or devices[0]["provider"] != "cuda" or devices[0]["description"] != "NVIDIA L40S"
                or file_sha256(model) != MODEL_SHA256 or file_sha256(path) != PLUGIN_SHA256):
            raise ValueError("scoped KG4 needs the measured model, L40S, plugin and disabled CUDA graphs")
        lib = ctypes.CDLL(path)
        for setter in (lib.ironmule_kg_set_mask, lib.ironmule_kg_set_mask6):
            setter.argtypes, setter.restype = [ctypes.c_uint32], None
        for getter in (lib.ironmule_kg_get_mask, lib.ironmule_kg_get_mask6):
            getter.argtypes, getter.restype = [], ctypes.c_uint32
        prices = {mode: {b: [] for b in BUCKETS} for mode in ("stock", "kg4")}
        with _mask_lock:
            lib.ironmule_kg_set_mask6(0)
            try:
                for rep in range(4):
                    for mode in (("stock", "kg4") if rep % 2 == 0 else ("kg4", "stock")):
                        lib.ironmule_kg_set_mask(_mask(BUCKETS) if mode == "kg4" else 0)
                        measured = engine._step_prices(BUCKETS, repeats=1)
                        for b in BUCKETS:
                            price = measured[b]
                            if not math.isfinite(price) or price <= 0:
                                raise ValueError("invalid local KG4 step measurement")
                            prices[mode][b].append(price)
            finally:
                lib.ironmule_kg_set_mask(0)
        medians = {mode: {b: statistics.median(v) for b, v in groups.items()} for mode, groups in prices.items()}
        use = [b for b in BUCKETS if medians["kg4"][b] <= .97 * medians["stock"][b]]
        scope = cls(lib, use)
        info = {"lib": Path(path).name, "tokens_served": use, "q6k_tokens_served": [],
                "step_ms_off": {str(b): 1000 * v for b, v in medians["stock"].items()},
                "step_ms_on": {str(b): 1000 * v for b, v in medians["kg4"].items()}}
        return scope, info

    def begin(self):
        _mask_lock.acquire()
        self.low = self.wide = False

    def _set(self, enabled):
        if self.enabled != enabled:
            self.lib.ironmule_kg_set_mask(self.mask if enabled else 0)
            self.enabled = enabled
            self.switches += 1

    def update(self, demand):
        if demand > 8 and not self.wide:
            self.wide = True
            self.wide_runs += 1
        eligible = not self.wide and 6 <= demand <= 8
        if eligible and not self.low:
            self.low = True
            self.low_runs += 1
        self._set(eligible and bool(self.mask))

    def end(self):
        try:
            self._set(False)
        finally:
            _mask_lock.release()

    def status(self):
        return {"mode": MODE, "q4_buckets": list(self.buckets), "low_runs": self.low_runs,
                "wide_runs": self.wide_runs, "switches": self.switches, "parked": not self.enabled}
