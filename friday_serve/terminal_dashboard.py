"""Terminal Live-Cockpit (ASCII/ANSI Telemetry Dashboard) for Friday LLM Runtime.

Zero external dependencies: pure Python standard library string and ANSI formatting.
Renders real-time hardware gauges in under 1 ms:
- Memory Bandwidth utilization bar (GB/s vs device peak, e.g. 400 GB/s on M1 Max)
- TTFT gauge with visual [⚡ PREFIX-CACHE HIT] / [COLD PREFILL] indicator
- Decode Rate (TPS) tachometer
- Hardware memory safety (VRAM peak, Swap nominal verification)
- Active RL strategy, speculative acceptance rate, and circuit breaker status
"""

from __future__ import annotations

import sys
from typing import Any

from .telemetry import RequestMetrics, TelemetryTracker


def make_bar(value: float, max_value: float, width: int = 20) -> str:
    """Generate a visual gauge bar with filled (█) and empty (░) blocks."""
    if max_value <= 0.0:
        pct = 0.0
    else:
        pct = max(0.0, min(1.0, value / max_value))
    filled = int(round(pct * width))
    empty = max(0, width - filled)
    return "█" * filled + "░" * empty


class Colors:
    """Standard ANSI color escapes."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"


def render_cockpit(
    tracker: TelemetryTracker,
    colored: bool = True,
    width: int = 76,
) -> str:
    """Render the full ASCII/ANSI cockpit dashboard string."""
    m: RequestMetrics | None = tracker.get_current()

    c_reset = Colors.RESET if colored else ""
    c_bold = Colors.BOLD if colored else ""
    c_cyan = Colors.CYAN if colored else ""
    c_green = Colors.GREEN if colored else ""
    c_yellow = Colors.YELLOW if colored else ""
    c_red = Colors.RED if colored else ""
    c_dim = Colors.DIM if colored else ""

    sep = "═" * (width - 2)
    thin_sep = "─" * (width - 2)

    lines: list[str] = [
        f"{c_cyan}╔{sep}╗{c_reset}",
        f"{c_cyan}║{c_bold}{'FRIDAY ULTIMATE INFERENCE COCKPIT':^{width - 2}}{c_reset}{c_cyan}║{c_reset}",
        f"{c_cyan}╠{sep}╣{c_reset}",
    ]

    if m is None:
        empty_bar = "░" * 20
        lines.extend([
            f"{c_cyan}║{c_reset} {c_dim}STATUS: STANDBY — WAITING FOR FIRST INFERENCE REQUEST...{' ' * (width - 60)}{c_cyan}║{c_reset}",
            f"{c_cyan}║{c_reset} Bandwidth Gauge: [{empty_bar}] 0.0 GB/s / {tracker.peak_bandwidth_gbs:.1f} GB/s{' ' * (width - 55)}{c_cyan}║{c_reset}",
            f"{c_cyan}╚{sep}╝{c_reset}",
        ])
        return "\n".join(lines)

    # 1. Header Line: Model & Breaker
    model_size = tracker.model_size_gb(m.model_id)
    breaker_color = c_green if m.breaker_status.lower() == "nominal" else c_red
    breaker_str = f"{breaker_color}{m.breaker_status.upper()}{c_reset}"
    header_content = f"Model: {c_bold}{m.model_id}{c_reset} ({model_size} GB) | Breaker: {breaker_str}"
    lines.append(f"{c_cyan}║{c_reset} {header_content:<{width + (len(c_bold) + len(c_reset) + len(breaker_color) + len(c_reset)) - 4}} {c_cyan}║{c_reset}")
    lines.append(f"{c_cyan}╠{thin_sep}╣{c_reset}")

    # 2. Memory Bandwidth Gauge
    bw_bar = make_bar(m.effective_bw_gbs, tracker.peak_bandwidth_gbs, width=20)
    bw_color = c_green if m.bw_utilization_pct >= 50.0 else (c_yellow if m.bw_utilization_pct >= 25.0 else c_dim)
    bw_text = f"[{bw_color}{bw_bar}{c_reset}] {m.effective_bw_gbs:.1f} GB/s / {tracker.peak_bandwidth_gbs:.1f} GB/s ({m.bw_utilization_pct:.1f} %)"
    lines.append(f"{c_cyan}║{c_reset} {c_bold}MEMORY BANDWIDTH UTILIZATION:{c_reset}{' ' * (width - 32)}{c_cyan}║{c_reset}")
    lines.append(f"{c_cyan}║{c_reset}   {bw_text:<{width + (len(bw_color) + len(c_reset)) - 6}} {c_cyan}║{c_reset}")
    lines.append(f"{c_cyan}║{c_reset}{' ' * (width - 2)}{c_cyan}║{c_reset}")

    # 3. TTFT Gauge
    ttft_bar = make_bar(m.ttft_ms, 250.0, width=20)
    if m.is_prefix_hit:
        hit_tag = f"{c_green}[⚡ PREFIX-CACHE HIT]{c_reset}"
    else:
        hit_tag = f"{c_yellow}[COLD PREFILL]{c_reset}"
    ttft_text = f"[{ttft_bar}] {m.ttft_ms:.1f} ms  {hit_tag}"
    lines.append(f"{c_cyan}║{c_reset} {c_bold}TIME TO FIRST TOKEN (TTFT):{c_reset}{' ' * (width - 30)}{c_cyan}║{c_reset}")
    lines.append(f"{c_cyan}║{c_reset}   {ttft_text:<{width + (len(hit_tag) - (len('[⚡ PREFIX-CACHE HIT]') if m.is_prefix_hit else len('[COLD PREFILL]'))) - 6}} {c_cyan}║{c_reset}")
    lines.append(f"{c_cyan}║{c_reset}{' ' * (width - 2)}{c_cyan}║{c_reset}")

    # 4. Decode TPS Gauge
    tps_bar = make_bar(m.decode_tps, 120.0, width=20)
    tps_text = f"[{c_cyan}{tps_bar}{c_reset}] {c_bold}{m.decode_tps:.1f} tok/s{c_reset} ({m.tokens_generated} tokens in {m.wall_s:.2f}s)"
    lines.append(f"{c_cyan}║{c_reset} {c_bold}DECODE RATE (TPS):{c_reset}{' ' * (width - 21)}{c_cyan}║{c_reset}")
    lines.append(f"{c_cyan}║{c_reset}   {tps_text:<{width + (len(c_cyan) + len(c_reset) + len(c_bold) + len(c_reset)) - 6}} {c_cyan}║{c_reset}")
    lines.append(f"{c_cyan}╠{thin_sep}╣{c_reset}")

    # 5. Hardware Safety (Critic Gates)
    if m.swap_mb == 0.0:
        swap_tag = f"{c_green}0 MB [NOMINAL]{c_reset}"
    else:
        swap_tag = f"{c_red}{m.swap_mb:.1f} MB [⚠️ SWAP WARNING]{c_reset}"
    safety_text = f"VRAM Peak: {m.peak_vram_mb:.0f} MB | SWAP: {swap_tag} | Concurrency Gate: 1/1"
    lines.append(f"{c_cyan}║{c_reset} {c_bold}HARDWARE & MEMORY SAFETY:{c_reset}{' ' * (width - 28)}{c_cyan}║{c_reset}")
    lines.append(f"{c_cyan}║{c_reset}   {safety_text:<{width + (len(c_green if m.swap_mb == 0.0 else c_red) + len(c_reset)) - 6}} {c_cyan}║{c_reset}")
    lines.append(f"{c_cyan}║{c_reset}{' ' * (width - 2)}{c_cyan}║{c_reset}")

    # 6. Dispatch & Optimization
    acc_pct = m.acceptance_rate * 100.0
    dispatch_text = f"RL Strategy: {c_bold}{m.action}{c_reset} | Speculation Acceptance: {acc_pct:.1f} %"
    lines.append(f"{c_cyan}║{c_reset} {c_bold}DISPATCH & CONTROLLER:{c_reset}{' ' * (width - 25)}{c_cyan}║{c_reset}")
    lines.append(f"{c_cyan}║{c_reset}   {dispatch_text:<{width + (len(c_bold) + len(c_reset)) - 6}} {c_cyan}║{c_reset}")

    lines.append(f"{c_cyan}╚{sep}╝{c_reset}")
    return "\n".join(lines)


def print_live_cockpit(
    tracker: TelemetryTracker,
    colored: bool = True,
    clear_screen: bool = False,
) -> None:
    """Print the cockpit to stdout with optional in-place screen clearing."""
    rendered = render_cockpit(tracker, colored=colored)
    if clear_screen and sys.stdout.isatty():
        sys.stdout.write("\033[H")
    sys.stdout.write(rendered + "\n")
    sys.stdout.flush()


__all__ = [
    "Colors",
    "make_bar",
    "print_live_cockpit",
    "render_cockpit",
]
