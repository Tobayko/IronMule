"""Observe the real per-width measurements of an opted-in performance-calibration experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path


def main():
    import followup_arm
    from ironmule import ggml_engine as ge
    from lab.portable_gpu.bench import save

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--sampling-screen", choices=("reference", "candidate"))
    parser.add_argument("--output", type=Path, required=True)
    a, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], "--output", str(a.output), *remaining]
    result = {}
    close = ge.GgmlEngine.close
    original_once = ge.GgmlEngine._decode_once
    jitter = False
    calls = Counter()
    spans = defaultdict(list)
    prefix = "IRONMULE_CALIBRATION_SAMPLING_"
    profile_env, pin_env = prefix + "PROFILE", prefix + "SHA256"
    ordinary = (os.environ[profile_env], os.environ[pin_env])
    cfg = json.loads(Path(ordinary[0]).read_bytes())
    method = cfg["method"]
    target = getattr(ge.GgmlEngine, method)

    def once(self, rows, *args, **kwargs):
        # Count actual calibration calls only in the separate diagnostic worker.
        frame = sys._getframe(1)
        while frame is not None and frame.f_code is not target.__code__:
            frame = frame.f_back
        if frame is not None and all(row[2] == 0 and row[3] for row in rows):
            width = len(rows)
            calls[width] += 1
            started = time.perf_counter()
            if jitter and width == 8 and calls[width] % 2 == 0:
                time.sleep(0.075)
            answer = original_once(self, rows, *args, **kwargs)
            spans[width].append(time.perf_counter() - started)
            return answer
        return original_once(self, rows, *args, **kwargs)

    def state(self):
        return {
            "prices": {str(k): v for k, v in self.pad_prices.items()},
            "dispatch": self.pad_to,
            "method": method,
            "telemetry": getattr(self, "calibration_sampling", None),
            "calls": {str(k): v for k, v in calls.items()},
            "native_seconds": {str(k): list(v) for k, v in spans.items()},
        }

    def at_start(frame, event, answer):
        nonlocal jitter
        if event != "return" or frame.f_code is not target.__code__:
            return
        # A serving Pilot may subsequently change the active CPU threads. Exercise all
        # controls here on the real startup context, before the first admitted request.
        sys.setprofile(None)
        engine = frame.f_locals["self"]
        result["startup"] = {**state(engine), "dispatch": answer}
        if a.sampling_screen != "candidate":
            return
        original_prices = engine.pad_prices
        original_telemetry = getattr(engine, "calibration_sampling", None)
        checks = {}
        try:
            for kind in (
                "valid_control",
                "missing",
                "wrong_pin",
                "expired",
                "foreign",
                "jitter",
            ):
                calls.clear()
                spans.clear()
                os.environ[profile_env], os.environ[pin_env] = ordinary
                if kind == "missing":
                    del os.environ[profile_env]
                    del os.environ[pin_env]
                elif kind == "wrong_pin":
                    os.environ[pin_env] = "0" * 64
                elif kind in ("expired", "foreign"):
                    logical = os.environ[prefix + kind.upper()]
                    os.environ[profile_env] = logical
                    os.environ[pin_env] = hashlib.sha256(
                        Path(logical).read_bytes()
                    ).hexdigest()
                jitter = kind == "jitter"
                engine.calibration_sampling = None
                dispatch = target(engine)
                checks[kind] = {**state(engine), "dispatch": dispatch}
        finally:
            jitter = False
            os.environ[profile_env], os.environ[pin_env] = ordinary
            engine.pad_prices, engine.calibration_sampling = (
                original_prices,
                original_telemetry,
            )
        result["negative"] = checks

    def finish(self):
        try:
            if not a.sampling_screen:
                result["startup"] = state(self)
        finally:
            close(self)

    # Timed workers read the effective profile at close; they install no per-call hook.
    ge.GgmlEngine.close = finish
    if a.sampling_screen:
        ge.GgmlEngine._decode_once = once
        sys.setprofile(at_start)
    try:
        followup_arm.main()
    finally:
        if a.sampling_screen:
            sys.setprofile(None)
        ge.GgmlEngine._decode_once, ge.GgmlEngine.close = original_once, close
        os.environ[profile_env], os.environ[pin_env] = ordinary
    row = json.loads(a.output.read_bytes())
    row["sampling"] = {
        "observed": result,
        "screen": a.sampling_screen,
        "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    save(a.output, row)


if __name__ == "__main__":
    main()
