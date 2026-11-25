"""Lab A/B switches inside one server start (diagnostics, off by default).

With IRONMULE_LAB_SWITCH naming a JSON file, a daemon thread in the server and one in its ggml worker read the file
whenever it changes (polled every 20 ms) and set the named switches of this process: "ADM1", "ADM2" (the service's
admission path and its logs), "PX2", "BF1" (the ggml engine's packed-prefill decision and batch fill). Paired waves
can then alternate the arms with the same server, worker, KV state and clocks; separate server starts had spread the
32-client prefill by about 0.25 s (A100 pod 32, PF2), more than the arms' effects. Every switch keeps its own default
until the file names it. Nothing reads the file without the variable.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

_started = False
_stop = threading.Event()
applied: dict[str, Any] = {}


def _apply(values: dict[str, Any]) -> None:
    import sys

    flags = {"ADM1": [("ironmule_product.http_server", "_ADM1"), ("ironmule_product.serving_policy", "_ADM1")],
             "ADM2": [("ironmule_product.serving_policy", "_ADM2")],
             "PX2": [("ironmule.ggml_engine", "PX2")],
             "BF1": [("ironmule.ggml_engine", "BF1")]}
    for key, targets in flags.items():
        if key not in values:
            continue
        for module, name in targets:
            mod = sys.modules.get(module)
            if mod is not None:
                setattr(mod, name, bool(values[key]))
        applied[key] = bool(values[key])


def _watch(path: str) -> None:
    seen, values = None, {}
    while not _stop.is_set():
        try:
            stamp = os.stat(path).st_mtime_ns
            if stamp != seen:
                with open(path) as fh:
                    read = json.load(fh)
                values = read if isinstance(read, dict) else values
                seen = stamp
        except (OSError, ValueError):
            pass
        _apply(values)  # again every poll: a module imported after the file changed gets it too
        _stop.wait(0.02)


def start() -> None:
    """Start the watcher once per process when IRONMULE_LAB_SWITCH is set; otherwise nothing."""
    global _started
    path = os.environ.get("IRONMULE_LAB_SWITCH")
    if not path or _started:
        return
    _started = True
    _stop.clear()
    threading.Thread(target=_watch, args=(path,), name="ironmule-lab-switch", daemon=True).start()


def stop() -> None:
    """End the watcher (tests); the switches keep their last values."""
    global _started
    _stop.set()
    _started = False
