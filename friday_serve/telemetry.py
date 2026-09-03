"""Telemetry tracker and hardware metric derivation for Friday LLM Runtime.

Captures real-time metrics per request and maintains a rolling history (50 requests):
- TTFT (ms) and prefix cache hit detection
- Generation count and decode rate (TPS)
- Memory bandwidth derivation based on model parameter bytes
- Peak VRAM allocation and macOS swap usage
- RL-selected strategy, speculative acceptance rate, and circuit breaker status
"""

from __future__ import annotations

import collections
import os
import platform
import re
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


def _read_swap_mb() -> float:
    """Read active macOS swap usage in megabytes via sysctl."""
    if platform.system() != "Darwin":
        return 0.0
    try:
        result = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "vm.swapusage"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"},
            timeout=1,
            check=False,
        )
        if result.returncode != 0:
            return 0.0
        text = (result.stdout or b"").decode("ascii", "ignore")
        match = re.search(r"used\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGTP])", text, re.I)
        if not match:
            return 0.0
        units = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4, "P": 1024**5}
        bytes_val = float(match.group(1)) * units[match.group(2).upper()]
        return round(bytes_val / (1024.0 * 1024.0), 2)
    except Exception:
        return 0.0


def _read_vram_mb() -> float:
    """Read peak Metal GPU allocated memory via MLX if available."""
    try:
        import mlx.core as mx

        metal = getattr(mx, "metal", None)
        if metal is not None and hasattr(metal, "get_peak_memory"):
            return round(metal.get_peak_memory() / (1024.0 * 1024.0), 2)
    except Exception:
        pass
    return 0.0


@dataclass(frozen=True)
class RequestMetrics:
    """Immutable snapshot of one request's execution telemetry."""

    model_id: str
    ttft_ms: float
    is_prefix_hit: bool
    tokens_generated: int
    decode_tps: float
    wall_s: float
    effective_bw_gbs: float
    bw_utilization_pct: float
    peak_vram_mb: float
    swap_mb: float
    action: str
    acceptance_rate: float
    breaker_status: str
    timestamp: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LiveState:
    """Dynamic real-time inference state for live dashboard animation."""

    status: str = "STANDBY"  # "STANDBY", "PREFILLING", "DECODING", "COMPLETED"
    model_id: str = "mlx-community/gemma-3-4b-it-4bit"
    tokens_generated: int = 0
    max_tokens: int = 0
    prompt_tokens: int = 0
    ttft_ms: float = 0.0
    is_prefix_hit: bool = False
    current_tps: float = 0.0
    current_bw_gbs: float = 0.0
    action: str = "device_profile_dispatch"
    vram_mb: float = 0.0
    swap_mb: float = 0.0
    decode_start_ns: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class TelemetryTracker:
    """Thread-safe telemetry aggregator and rolling history tracker."""

    def __init__(
        self,
        max_history: int = 50,
        peak_bandwidth_gbs: float = 400.0,
        host: str = "127.0.0.1",
        port: int = 8080,
    ) -> None:
        self.max_history = max_history
        self.peak_bandwidth_gbs = peak_bandwidth_gbs
        self.host = host
        self.port = port
        self._lock = threading.Lock()
        self.current: RequestMetrics | None = None
        self.live: LiveState = LiveState()
        self.history: collections.deque[RequestMetrics] = collections.deque(maxlen=max_history)

    def set_server_info(self, host: str, port: int) -> None:
        with self._lock:
            self.host = host
            self.port = port

    @staticmethod
    def model_size_gb(model_id: str) -> float:
        """Derive model memory footprint in GB for bandwidth calculation:

        1B = 0.77 GB, 4B = 2.56 GB, 12B = 7.19 GB.
        """
        low = model_id.lower()
        if "1b" in low:
            return 0.77
        if "12b" in low:
            return 7.19
        # Default for 4B and common IT/chat weights
        return 2.56

    def start_request(self, model_id: str, prompt_tokens: int, max_tokens: int = 128, action: str = "device_profile_dispatch") -> None:
        """Called when prefill begins."""
        with self._lock:
            self.live.status = "PREFILLING"
            self.live.model_id = model_id
            self.live.prompt_tokens = prompt_tokens
            self.live.max_tokens = max_tokens
            self.live.tokens_generated = 0
            self.live.action = action
            self.live.vram_mb = _read_vram_mb()
            self.live.swap_mb = _read_swap_mb()

    def update_first_token(self, ttft_ns: int, is_prefix_hit: bool) -> None:
        """Called immediately after first token is yielded."""
        with self._lock:
            self.live.status = "DECODING"
            self.live.tokens_generated = 1
            self.live.ttft_ms = round(max(0, ttft_ns) / 1_000_000.0, 2)
            self.live.is_prefix_hit = is_prefix_hit
            self.live.decode_start_ns = time.perf_counter_ns()

    def update_tokens(self, tokens_generated: int) -> None:
        """Called on every yielded decode chunk."""
        with self._lock:
            self.live.status = "DECODING"
            self.live.tokens_generated = tokens_generated
            if self.live.decode_start_ns > 0 and tokens_generated > 1:
                elapsed_s = (time.perf_counter_ns() - self.live.decode_start_ns) / 1e9
                if elapsed_s > 0:
                    self.live.current_tps = round((tokens_generated - 1) / elapsed_s, 1)
                    model_gb = self.model_size_gb(self.live.model_id)
                    self.live.current_bw_gbs = round(model_gb * self.live.current_tps, 1)

    def get_live(self) -> LiveState:
        with self._lock:
            return LiveState(**self.live.as_dict())

    def record_request(
        self,
        *,
        model_id: str,
        prefill_ns: int,
        decode_ns: int,
        tokens_generated: int,
        prefix_cache_hits: int = 0,
        action: str = "baseline",
        acceptance_rate: float = 0.0,
        breaker_status: str = "nominal",
        peak_vram_mb: float | None = None,
        swap_mb: float | None = None,
    ) -> RequestMetrics:
        """Record one request and calculate derived bandwidth and latency metrics."""
        ttft_ms = round(max(0, prefill_ns) / 1_000_000.0, 2)
        is_prefix_hit = prefix_cache_hits > 0

        total_ns = prefill_ns + decode_ns
        wall_s = round(max(0, total_ns) / 1_000_000_000.0, 4)

        decode_s = max(0, decode_ns) / 1_000_000_000.0
        if decode_s > 0 and tokens_generated > 1:
            decode_tokens = tokens_generated - 1
            decode_tps = round(decode_tokens / decode_s, 2)
        elif decode_s > 0 and tokens_generated == 1:
            decode_tps = round(1.0 / decode_s, 2)
        else:
            decode_tps = 0.0

        model_gb = self.model_size_gb(model_id)
        effective_bw_gbs = round(model_gb * decode_tps, 2)
        bw_utilization_pct = round(
            min(100.0, (effective_bw_gbs / self.peak_bandwidth_gbs) * 100.0), 2
        )

        vram_mb = peak_vram_mb if peak_vram_mb is not None else _read_vram_mb()
        swap = swap_mb if swap_mb is not None else _read_swap_mb()

        metrics = RequestMetrics(
            model_id=model_id,
            ttft_ms=ttft_ms,
            is_prefix_hit=is_prefix_hit,
            tokens_generated=tokens_generated,
            decode_tps=decode_tps,
            wall_s=wall_s,
            effective_bw_gbs=effective_bw_gbs,
            bw_utilization_pct=bw_utilization_pct,
            peak_vram_mb=vram_mb,
            swap_mb=swap,
            action=action,
            acceptance_rate=round(acceptance_rate, 4),
            breaker_status=breaker_status,
        )

        with self._lock:
            self.current = metrics
            self.history.append(metrics)
            self.live.status = "COMPLETED"
            self.live.tokens_generated = tokens_generated

        return metrics

    def get_current(self) -> RequestMetrics | None:
        with self._lock:
            return self.current

    def get_history(self) -> list[RequestMetrics]:
        with self._lock:
            return list(self.history)

    def clear(self) -> None:
        with self._lock:
            self.current = None
            self.live = LiveState()
            self.history.clear()


GLOBAL_TRACKER = TelemetryTracker()


def get_global_tracker() -> TelemetryTracker:
    return GLOBAL_TRACKER


__all__ = [
    "GLOBAL_TRACKER",
    "LiveState",
    "RequestMetrics",
    "TelemetryTracker",
    "get_global_tracker",
]
