"""Collect bounded CUDA device activity around at most two real calls in a dedicated diagnostic worker."""
from __future__ import annotations

import hashlib
import math
import re
import sys
import time
from pathlib import Path


def mapped_vendor():
    """Identify distinct mapped images cheaply. The surrounding serving harness hashes them after work."""
    objects = {}
    for line in Path("/proc/self/maps").read_text().splitlines():
        cols = line.split(None, 5)
        if len(cols) != 6:
            continue
        name = Path(cols[5]).name
        match = re.fullmatch(r"(libcudart|libcublasLt|libcublas)\.so(?:\.\d+)*", name)
        if match:
            objects.setdefault(match[1], set()).add((name, cols[3], cols[4]))
    return {tag: [list(image) for image in sorted(images)] for tag, images in sorted(objects.items())}


def setup(summary):
    from ironmule.kernel_gen import _visible_cuda_devices

    summary["visible_devices"] = _visible_cuda_devices()
    if len(summary["visible_devices"]) != 1 or not summary["visible_devices"][0].get("uuid"):
        raise ValueError("unsupported device visibility")
    import torch
    from torch.profiler import ProfilerActivity, profile

    summary["torch_version"] = str(torch.__version__)
    summary["mapped_after_import"] = mapped_vendor()
    if any(len(images) != 1 for images in summary["mapped_after_import"].values()):
        raise ValueError("ambiguous mapped CUDA vendor images")
    if ProfilerActivity.CUDA not in torch.profiler.supported_activities():
        raise ValueError("CUDA activity unavailable")
    # Drain prior work before the measured diagnostic, and fence only inside its device sample.
    torch.cuda.synchronize(0)
    return profile(activities=[ProfilerActivity.CUDA]), lambda: torch.cuda.synchronize(0)


def aggregate(prof):
    groups = {}
    count = 0
    total = 0.
    for event in prof.events():
        if getattr(event.device_type, "name", None) != "CUDA":
            continue
        duration = float(event.device_time_total) / 1e6
        if not math.isfinite(duration) or duration <= 0:
            continue
        name = event.name
        if not isinstance(name, str) or not name:
            continue
        total += duration
        count += 1
        if name in groups or len(groups) < 2048:
            row = groups.setdefault(name, [0, 0.])
            row[0] += 1
            row[1] += duration
    events = [{"symbol": name[:512], "symbol_sha256": hashlib.sha256(name.encode()).hexdigest(),
               "calls": calls, "device_s": seconds}
              for name, (calls, seconds) in sorted(groups.items(), key=lambda item: -item[1][1])[:48]]
    return {"events": events, "device_events": count, "summed_device_s": total,
            "omitted_calls": count - sum(r["calls"] for r in events),
            "omitted_device_s": max(0., total - sum(r["device_s"] for r in events))}


class Activity:
    def __init__(self):
        self.attempts = 0

    def measure(self, fn, *args, **kw):
        start = time.perf_counter()
        summary = {"schema": "ironmule.native-activity.v1", "sample": None, "status": "sample_limit",
                   "events": [], "device_events": 0, "summed_device_s": None, "detail": None,
                   "uncertainty": "CUPTI diagnostic with fencing. Device durations can overlap and do not equal wall savings."}
        prof = None
        if self.attempts < 2:
            self.attempts += 1
            summary["sample"] = self.attempts
            try:
                summary["mapped_before"] = mapped_vendor()
                ctx, sync = setup(summary)
                prof = ctx.__enter__()
            except Exception as exc:
                summary.update(status="unsupported", detail=type(exc).__name__)
        t0 = time.perf_counter()
        try:
            result = fn(*args, **kw)
        except BaseException:
            if prof is not None:
                try:
                    ctx.__exit__(*sys.exc_info())
                except Exception:
                    pass
            raise
        summary["native_call_wall_s"] = time.perf_counter() - t0
        if prof is not None:
            try:
                try:
                    sync()
                finally:
                    ctx.__exit__(*sys.exc_info())
                summary.update(aggregate(prof))
                summary["mapped_after"] = mapped_vendor()
                if any(len(images) != 1 for images in summary["mapped_after"].values()):
                    raise ValueError("ambiguous mapped CUDA vendor images")
                summary["status"] = "measured" if summary["device_events"] else "no_cuda_events"
            except Exception as exc:
                summary.update(status="unsupported", detail=type(exc).__name__, events=[], summed_device_s=None, device_events=0)
        summary["capture_wall_s"] = time.perf_counter() - start
        return result, summary
