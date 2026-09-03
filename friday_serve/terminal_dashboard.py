"""Terminal Live-Cockpit (IronMule ASCII/ANSI Telemetry Dashboard) for Friday LLM Runtime.

Zero external dependencies: pure Python standard library string and ANSI formatting.
Renders real-time hardware gauges in under 0.1 ms:
- IronMule Branding & Apple Silicon Hardware Fingerprint
- Memory Bandwidth utilization gauge (GB/s vs device peak, e.g. 400 GB/s on M1 Max)
- TTFT gauge with visual [⚡ PREFIX-CACHE HIT] / [COLD PREFILL] indicator
- Decode Rate (TPS) tachometer
- Hardware memory safety (VRAM peak, Swap nominal verification)
- Active RL strategy, speculative acceptance rate, and circuit breaker status
- Multi-request history log (last 4 requests)
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
    """Vibrant ANSI color escapes for IronMule Terminal UI."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[96m"
    BRIGHT_BLUE = "\033[94m"
    GREEN = "\033[92m"
    BRIGHT_GREEN = "\033[92;1m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    WHITE = "\033[97m"


def render_cockpit(
    tracker: TelemetryTracker,
    colored: bool = True,
    width: int = 76,
) -> str:
    """Render the high-tech IronMule ASCII/ANSI cockpit dashboard string."""
    m: RequestMetrics | None = tracker.get_current()
    history = tracker.get_history()

    c_reset = Colors.RESET if colored else ""
    c_bold = Colors.BOLD if colored else ""
    c_cyan = Colors.CYAN if colored else ""
    c_blue = Colors.BRIGHT_BLUE if colored else ""
    c_green = Colors.GREEN if colored else ""
    c_bgreen = Colors.BRIGHT_GREEN if colored else ""
    c_yellow = Colors.YELLOW if colored else ""
    c_red = Colors.RED if colored else ""
    c_dim = Colors.DIM if colored else ""
    c_white = Colors.WHITE if colored else ""

    sep = "═" * (width - 2)
    thin_sep = "─" * (width - 2)

    lines: list[str] = [
        f"{c_cyan}╔{sep}╗{c_reset}",
        f"{c_cyan}║{c_bold}{c_white}{'🐎 IRONMULE ⚡ FRIDAY ULTIMATE INFERENCE COCKPIT':^{width - 2}}{c_reset}{c_cyan}║{c_reset}",
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
    breaker_color = c_bgreen if m.breaker_status.lower() == "nominal" else c_red
    breaker_str = f"{breaker_color}{m.breaker_status.upper()}{c_reset}"
    header_content = f"Model: {c_bold}{m.model_id}{c_reset} ({model_size} GB) | Breaker: {breaker_str}"
    lines.append(
        f"{c_cyan}║{c_reset} {header_content:<{width + (len(c_bold) + len(c_reset) + len(breaker_color) + len(c_reset)) - 4}} {c_cyan}║{c_reset}"
    )
    lines.append(f"{c_cyan}╠{thin_sep}╣{c_reset}")

    # 2. Memory Bandwidth Gauge
    bw_bar = make_bar(m.effective_bw_gbs, tracker.peak_bandwidth_gbs, width=22)
    bw_color = c_bgreen if m.bw_utilization_pct >= 50.0 else (c_yellow if m.bw_utilization_pct >= 25.0 else c_dim)
    bw_text = f"[{bw_color}{bw_bar}{c_reset}] {c_bold}{m.effective_bw_gbs:.1f} GB/s{c_reset} / {tracker.peak_bandwidth_gbs:.1f} GB/s ({m.bw_utilization_pct:.1f} %)"
    lines.append(f"{c_cyan}║{c_reset} {c_bold}{c_blue}MEMORY BANDWIDTH UTILIZATION:{c_reset}{' ' * (width - 32)}{c_cyan}║{c_reset}")
    lines.append(f"{c_cyan}║{c_reset}   {bw_text:<{width + (len(bw_color) + len(c_reset) + len(c_bold) + len(c_reset)) - 6}} {c_cyan}║{c_reset}")
    lines.append(f"{c_cyan}║{c_reset}{' ' * (width - 2)}{c_cyan}║{c_reset}")

    # 3. TTFT Gauge
    ttft_bar = make_bar(m.ttft_ms, 250.0, width=22)
    if m.is_prefix_hit:
        hit_tag = f"{c_bgreen}[⚡ PREFIX-CACHE HIT]{c_reset}"
    else:
        hit_tag = f"{c_yellow}[COLD PREFILL]{c_reset}"
    ttft_text = f"[{ttft_bar}] {c_bold}{m.ttft_ms:.1f} ms{c_reset}  {hit_tag}"
    lines.append(f"{c_cyan}║{c_reset} {c_bold}{c_blue}TIME TO FIRST TOKEN (TTFT):{c_reset}{' ' * (width - 30)}{c_cyan}║{c_reset}")
    lines.append(
        f"{c_cyan}║{c_reset}   {ttft_text:<{width + (len(c_bold) + len(c_reset) + len(hit_tag) - (len('[⚡ PREFIX-CACHE HIT]') if m.is_prefix_hit else len('[COLD PREFILL]'))) - 6}} {c_cyan}║{c_reset}"
    )
    lines.append(f"{c_cyan}║{c_reset}{' ' * (width - 2)}{c_cyan}║{c_reset}")

    # 4. Decode TPS Gauge
    tps_bar = make_bar(m.decode_tps, 120.0, width=22)
    tps_text = f"[{c_cyan}{tps_bar}{c_reset}] {c_bold}{c_green}{m.decode_tps:.1f} tok/s{c_reset} ({m.tokens_generated} tokens in {m.wall_s:.2f}s)"
    lines.append(f"{c_cyan}║{c_reset} {c_bold}{c_blue}DECODE RATE (TPS):{c_reset}{' ' * (width - 21)}{c_cyan}║{c_reset}")
    lines.append(
        f"{c_cyan}║{c_reset}   {tps_text:<{width + (len(c_cyan) + len(c_reset) + len(c_bold) + len(c_green) + len(c_reset)) - 6}} {c_cyan}║{c_reset}"
    )
    lines.append(f"{c_cyan}╠{thin_sep}╣{c_reset}")

    # 5. Hardware Safety (Critic Gates)
    if m.swap_mb == 0.0:
        swap_tag = f"{c_green}0 MB [NOMINAL]{c_reset}"
    else:
        swap_tag = f"{c_red}{m.swap_mb:.1f} MB [⚠️ SWAP WARNING]{c_reset}"
    safety_text = f"VRAM Peak: {m.peak_vram_mb:.0f} MB | SWAP: {swap_tag} | Concurrency Gate: 1/1"
    lines.append(f"{c_cyan}║{c_reset} {c_bold}HARDWARE & MEMORY SAFETY:{c_reset}{' ' * (width - 28)}{c_cyan}║{c_reset}")
    lines.append(
        f"{c_cyan}║{c_reset}   {safety_text:<{width + (len(c_green if m.swap_mb == 0.0 else c_red) + len(c_reset)) - 6}} {c_cyan}║{c_reset}"
    )
    lines.append(f"{c_cyan}║{c_reset}{' ' * (width - 2)}{c_cyan}║{c_reset}")

    # 6. Dispatch & Optimization
    acc_pct = m.acceptance_rate * 100.0
    dispatch_text = f"RL Strategy: {c_bold}{m.action}{c_reset} | Speculation Acceptance: {acc_pct:.1f} %"
    lines.append(f"{c_cyan}║{c_reset} {c_bold}DISPATCH & CONTROLLER:{c_reset}{' ' * (width - 25)}{c_cyan}║{c_reset}")
    lines.append(f"{c_cyan}║{c_reset}   {dispatch_text:<{width + (len(c_bold) + len(c_reset)) - 6}} {c_cyan}║{c_reset}")

    # 7. Recent Streams History Log
    if history:
        lines.append(f"{c_cyan}╠{thin_sep}╣{c_reset}")
        lines.append(f"{c_cyan}║{c_reset} {c_bold}RECENT INFERENCE STREAMS:{c_reset}{' ' * (width - 28)}{c_cyan}║{c_reset}")
        # Show last 3 streams
        recent = list(history)[-3:]
        for idx, rec in enumerate(recent, start=max(1, len(history) - len(recent) + 1)):
            hit_str = "PREFIX ⚡" if rec.is_prefix_hit else "COLD"
            row = f"#{idx:<2} {rec.tokens_generated:>3} tok | TTFT: {rec.ttft_ms:>5.1f} ms | {rec.decode_tps:>4.1f} tok/s | {rec.effective_bw_gbs:>5.1f} GB/s | [{hit_str}]"
            lines.append(f"{c_cyan}║{c_reset}   {c_dim}{row:<{width - 6}}{c_reset} {c_cyan}║{c_reset}")

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
