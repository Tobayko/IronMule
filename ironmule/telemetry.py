"""Telemetry with the two time-to-first-token definitions kept apart.

Service TTFT is measured from when a request arrived. Engine TTFT is measured from
when the model actually started working on it. Under concurrency they differ by the
queue wait, and reporting only one of them hides exactly the effect a service cares
about: E16 measured service TTFT falling from ~800 ms to ~87 ms under grouping while
engine TTFT barely moved.

One rule is enforced by omission: there is no field here that divides a group's
wall time by its width. That number is not a caller latency and this module does
not compute it.
"""

from __future__ import annotations

import statistics as st
from dataclasses import dataclass, field


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


@dataclass
class RequestMetrics:
    """Timestamps in nanoseconds; every derived value in milliseconds."""

    rid: str
    arrival_ns: int
    engine_start_ns: int = 0
    first_token_ns: int = 0
    finished_ns: int = 0
    token_times_ns: list[int] = field(default_factory=list)
    prompt_tokens: int = 0
    generated_tokens: int = 0
    visible_generated_tokens: int = 0
    stop_reason: str = ""
    fell_back: bool = False

    @property
    def service_ttft_ms(self) -> float | None:
        """From arrival. What the caller experiences."""
        return (self.first_token_ns - self.arrival_ns) / 1e6 if self.first_token_ns else None

    @property
    def engine_ttft_ms(self) -> float | None:
        """From the model actually starting. What the engine is responsible for."""
        if not (self.first_token_ns and self.engine_start_ns):
            return None
        return (self.first_token_ns - self.engine_start_ns) / 1e6

    @property
    def queue_wait_ms(self) -> float | None:
        return (self.engine_start_ns - self.arrival_ns) / 1e6 if self.engine_start_ns else None

    @property
    def latency_ms(self) -> float | None:
        """Full request latency, from arrival to the last token."""
        return (self.finished_ns - self.arrival_ns) / 1e6 if self.finished_ns else None

    @property
    def inter_token_ms(self) -> list[float]:
        return [(b - a) / 1e6 for a, b in zip(self.token_times_ns, self.token_times_ns[1:])]

    def as_dict(self) -> dict:
        return {"rid": self.rid, "prompt_tokens": self.prompt_tokens,
                # `generated_tokens` remains the compatibility key; spell out the
                # physical count alongside it so EOS filtering is never implicit.
                "generated_tokens": self.generated_tokens,
                "physical_generated_tokens": self.generated_tokens,
                "stop_reason": self.stop_reason,
                "visible_generated_tokens": self.visible_generated_tokens,
                "service_ttft_ms": self.service_ttft_ms, "engine_ttft_ms": self.engine_ttft_ms,
                "queue_wait_ms": self.queue_wait_ms, "latency_ms": self.latency_ms,
                "inter_token_ms": self.inter_token_ms, "fell_back": self.fell_back}


@dataclass
class Telemetry:
    mode: str = ""
    plan_kinds: list[str] = field(default_factory=list)
    requests: list[RequestMetrics] = field(default_factory=list)
    realised_widths: list[int] = field(default_factory=list)
    wall_ns: int = 0
    fallbacks: int = 0
    fallback_reasons: list[str] = field(default_factory=list)
    correctness_errors: int = 0
    correctness_check_performed: bool = False
    correctness_checked_requests: int = 0
    plan_switch_attempts: int = 0
    peak_memory_bytes: int = 0

    def snapshot(self) -> dict:
        latencies = [m.latency_ms for m in self.requests if m.latency_ms is not None]
        service = [m.service_ttft_ms for m in self.requests if m.service_ttft_ms is not None]
        engine = [m.engine_ttft_ms for m in self.requests if m.engine_ttft_ms is not None]
        inter = [g for m in self.requests for g in m.inter_token_ms]
        generated = sum(m.generated_tokens for m in self.requests)
        visible_generated = sum(m.visible_generated_tokens for m in self.requests)
        wall_s = self.wall_ns / 1e9 if self.wall_ns else None
        return {
            "mode": self.mode, "plan_kinds": sorted(set(self.plan_kinds)),
            "requests": len(self.requests), "generated_tokens": generated,
            "physical_generated_tokens": generated,
            "visible_generated_tokens": visible_generated,
            "wall_ms": self.wall_ns / 1e6 if self.wall_ns else None,
            "aggregate_tokens_per_second": (generated / wall_s) if wall_s else None,
            "service_ttft_p50_ms": _pct(service, 0.50), "service_ttft_p95_ms": _pct(service, 0.95),
            "engine_ttft_p50_ms": _pct(engine, 0.50), "engine_ttft_p95_ms": _pct(engine, 0.95),
            "latency_p50_ms": _pct(latencies, 0.50), "latency_p95_ms": _pct(latencies, 0.95),
            "queue_wait_p95_ms": _pct([m.queue_wait_ms for m in self.requests
                                       if m.queue_wait_ms is not None], 0.95),
            "inter_token_p50_ms": _pct(inter, 0.50), "inter_token_p95_ms": _pct(inter, 0.95),
            "mean_realised_width": st.mean(self.realised_widths) if self.realised_widths else None,
            "max_realised_width": max(self.realised_widths) if self.realised_widths else None,
            "rounds": len(self.realised_widths),
            "fallbacks": self.fallbacks, "fallback_reasons": self.fallback_reasons[:10],
            "correctness_errors": self.correctness_errors,
            "correctness_check_performed": self.correctness_check_performed,
            "correctness_checked_requests": self.correctness_checked_requests,
            "plan_switch_attempts": self.plan_switch_attempts,
            "peak_memory_bytes": self.peak_memory_bytes,
            "per_request": [m.as_dict() for m in self.requests],
        }


def _self_check() -> None:
    m = RequestMetrics(rid="r0", arrival_ns=0, engine_start_ns=1_000_000,
                       first_token_ns=3_000_000, finished_ns=9_000_000,
                       token_times_ns=[3_000_000, 5_000_000, 9_000_000],
                       prompt_tokens=10, generated_tokens=3, stop_reason="eos")
    assert m.service_ttft_ms == 3.0, "service TTFT counts from arrival"
    assert m.engine_ttft_ms == 2.0, "engine TTFT counts from model start"
    assert m.queue_wait_ms == 1.0
    assert m.latency_ms == 9.0
    assert m.inter_token_ms == [2.0, 4.0]

    t = Telemetry(mode="throughput", plan_kinds=["strict_one_shot"], requests=[m],
                  realised_widths=[4, 4, 2], wall_ns=10_000_000)
    snap = t.snapshot()
    assert snap["generated_tokens"] == 3
    assert abs(snap["aggregate_tokens_per_second"] - 300.0) < 1e-9
    assert snap["mean_realised_width"] == 10 / 3
    assert snap["rounds"] == 3
    assert "per_request" in snap and snap["per_request"][0]["rid"] == "r0"
    assert not any("per_width" in k or "per_slot" in k for k in snap), \
        "no field may divide group time by width"
    print("telemetry self-check ok")


if __name__ == "__main__":
    _self_check()
