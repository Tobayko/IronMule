"""Observation-only telemetry for an owned local worker.

This module deliberately records resource state without making an admission,
throttling, pause, or termination decision.  The caller owns worker lifecycle
and persistence; this helper owns only in-memory samples and aggregate facts.
"""

from __future__ import annotations

import math
import subprocess
import time
from collections import Counter
from collections.abc import Callable
from typing import Any

from friday_evidence.process_memory import ProcessMemoryError, sample_process_memory
from ironmule_product.readiness import _swap_used

SampleCallback = Callable[[dict[str, Any]], None]


class Observer:
    """Collect neutral, best-effort telemetry for one owned ``Popen`` worker.

    Sampling never controls the process.  Missing measurements are represented
    by ``None`` plus an error code rather than a synthetic numeric value.
    """

    def __init__(
        self,
        on_sample: SampleCallback | None = None,
        interval_seconds: float = 1.0,
    ) -> None:
        if on_sample is not None and not callable(on_sample):
            raise TypeError("on_sample must be callable or None")
        if (
            isinstance(interval_seconds, bool)
            or not isinstance(interval_seconds, (int, float))
            or not math.isfinite(float(interval_seconds))
            or interval_seconds < 0
        ):
            raise ValueError("interval_seconds must be a finite non-negative number")
        self.on_sample = on_sample
        self.interval_seconds = float(interval_seconds)
        self.process: subprocess.Popen[Any] | None = None
        self.label = ""
        self.rows: list[dict[str, Any]] = []
        self._last_sample_monotonic_ns: int | None = None
        self._swap_baseline_bytes: int | None = None
        self._errors: Counter[str] = Counter()
        self._skipped_interval = 0

    def bind(self, process: subprocess.Popen[Any] | None, label: str) -> None:
        """Bind an owned worker, or ``None`` when no worker is currently bound."""
        if process is not None and not isinstance(process, subprocess.Popen):
            raise TypeError("process must be subprocess.Popen or None")
        if not isinstance(label, str):
            raise TypeError("label must be str")
        self.process = process
        self.label = label

    def sample(self, force: bool = False) -> dict[str, Any] | None:
        """Record one due sample, returning ``None`` when interval-throttled.

        ``force`` bypasses only this helper's observation cadence.  It does not
        alter the owned process or imply that a measurement was available.
        """
        if not isinstance(force, bool):
            raise TypeError("force must be bool")
        observed_ns = time.monotonic_ns()
        if (
            not force
            and self._last_sample_monotonic_ns is not None
            and observed_ns - self._last_sample_monotonic_ns
            < int(self.interval_seconds * 1_000_000_000)
        ):
            self._skipped_interval += 1
            return None
        self._last_sample_monotonic_ns = observed_ns

        process = self.process
        pid: int | None = None
        errors: list[str] = []
        row: dict[str, Any] = {
            "observed_monotonic_ns": observed_ns,
            "label": self.label,
            "pid": None,
            "rss_bytes": None,
            "rss_source": None,
            "physical_footprint_bytes": None,
            "peak_footprint_bytes": None,
            "wired_bytes": None,
            "system_swap_used_bytes": None,
            "system_swap_delta_bytes": None,
            "errors": errors,
        }

        if process is None:
            errors.append("worker_unbound")
        else:
            pid = process.pid
            row["pid"] = pid
            # Poll before rusage: never request RSS for a reaped owned PID.
            returncode = process.poll()
            if returncode is not None:
                errors.append("worker_exited")
            else:
                try:
                    memory = sample_process_memory(pid)
                except ProcessMemoryError as exc:
                    errors.append(exc.code)
                except Exception:
                    errors.append("process_memory_failed")
                else:
                    row["rss_bytes"] = memory["resident_bytes"]
                    row["rss_source"] = "libproc"
                    row["physical_footprint_bytes"] = memory["physical_footprint_bytes"]
                    row["peak_footprint_bytes"] = memory["lifetime_peak_footprint_bytes"]
                    row["wired_bytes"] = memory["wired_bytes"]

        try:
            swap_used, swap_error = _swap_used()
        except Exception:
            swap_used, swap_error = None, "swap_probe_failed"
        if swap_error is not None:
            errors.append(swap_error)
        elif swap_used is not None:
            row["system_swap_used_bytes"] = swap_used
            if self._swap_baseline_bytes is None:
                self._swap_baseline_bytes = swap_used
            row["system_swap_delta_bytes"] = swap_used - self._swap_baseline_bytes

        self.rows.append(row)
        self._errors.update(errors)
        if self.on_sample is not None:
            try:
                self.on_sample(row)
            except Exception:
                errors.append("on_sample_failed")
                self._errors.update(("on_sample_failed",))
        return row

    def summary(self) -> dict[str, Any]:
        """Return in-memory counts and maxima without judging resource use."""
        def maximum(field: str) -> int | None:
            values = [row[field] for row in self.rows if type(row[field]) is int]
            return max(values) if values else None

        latest_swap_delta = next(
            (
                row["system_swap_delta_bytes"]
                for row in reversed(self.rows)
                if type(row["system_swap_delta_bytes"]) is int
            ),
            None,
        )
        return {
            "sample_count": len(self.rows),
            "skipped_interval_count": self._skipped_interval,
            "memory_sample_count": sum(row["rss_source"] == "libproc" for row in self.rows),
            "swap_sample_count": sum(
                type(row["system_swap_used_bytes"]) is int for row in self.rows
            ),
            "max_memory_bytes": {
                "rss_bytes": maximum("rss_bytes"),
                "physical_footprint_bytes": maximum("physical_footprint_bytes"),
                "peak_footprint_bytes": maximum("peak_footprint_bytes"),
                "wired_bytes": maximum("wired_bytes"),
            },
            "latest_system_swap_delta_bytes": latest_swap_delta,
            "errors": dict(sorted(self._errors.items())),
        }


# A neutral descriptive spelling for integrations that prefer it.
OpenObservation = Observer

__all__ = ["Observer", "OpenObservation", "SampleCallback"]
