#!/usr/bin/env python3
"""Award-Winning Interactive IronMule Live Terminal Monitor for Friday LLM Runtime.

Connects to the Friday OpenAI runtime and renders a flicker-free,
in-place interactive terminal cockpit at 10-20 FPS.

Usage:
    python tools/friday.py monitor [--port 8080] [--hz 10]
    python tools/monitor.py [--port 8080] [--hz 10]
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from friday_serve.telemetry import LiveState, RequestMetrics, TelemetryTracker
from friday_serve.terminal_dashboard import Theme, make_bar, render_cockpit


def fetch_telemetry(url: str, timeout_s: float = 0.5) -> dict | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "IronMuleMonitor/1.0"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode("utf-8"))
    except Exception:
        pass
    return None


def run_monitor(host: str = "127.0.0.1", port: int = 8080, refresh_hz: float = 10.0) -> int:
    telemetry_url = f"http://{host}:{port}/telemetry"
    interval = 1.0 / max(1.0, min(30.0, refresh_hz))

    is_tty = sys.stdout.isatty()
    if is_tty:
        # Hide cursor and clear screen
        sys.stdout.write("\033[?25l\033[2J\033[H")
        sys.stdout.flush()

    tracker = TelemetryTracker(peak_bandwidth_gbs=400.0)
    stop = False

    def handle_sigint(sig, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    tick = 0
    try:
        while not stop:
            data = fetch_telemetry(telemetry_url)

            if is_tty:
                sys.stdout.write("\033[H")

            if data is None:
                # Waiting state
                pulse = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"][tick % 10]
                msg = (
                    f"\n  {Theme.CYAN_BOLD}🐎 IRONMULE LIVE MONITOR{Theme.RESET}\n"
                    f"  {Theme.MUTED}──────────────────────────────────────────────────────────{Theme.RESET}\n"
                    f"  {Theme.AMBER_BOLD}{pulse} Connecting to Friday runtime at {host}:{port}...{Theme.RESET}\n"
                    f"  {Theme.DIM}Start the server in another window: python tools/friday.py serve{Theme.RESET}\n"
                    f"  {Theme.DIM}Press Ctrl+C to exit.{Theme.RESET}\n\033[J"
                )
                sys.stdout.write(msg)
                sys.stdout.flush()
            else:
                # Update tracker with remote telemetry
                peak_bw = data.get("peak_bandwidth_gbs", 400.0)
                tracker.peak_bandwidth_gbs = peak_bw

                # Parse current
                curr_data = data.get("current")
                if curr_data:
                    tracker.current = RequestMetrics(
                        model_id=curr_data.get("model_id", "gemma-3-4b-it-4bit"),
                        ttft_ms=curr_data.get("ttft_ms", 0.0),
                        is_prefix_hit=curr_data.get("is_prefix_hit", False),
                        tokens_generated=curr_data.get("tokens_generated", 0),
                        decode_tps=curr_data.get("decode_tps", 0.0),
                        wall_s=curr_data.get("wall_s", 0.0),
                        effective_bw_gbs=curr_data.get("effective_bw_gbs", 0.0),
                        bw_utilization_pct=curr_data.get("bw_utilization_pct", 0.0),
                        peak_vram_mb=curr_data.get("peak_vram_mb", 0.0),
                        swap_mb=curr_data.get("swap_mb", 0.0),
                        action=curr_data.get("action", "baseline"),
                        acceptance_rate=curr_data.get("acceptance_rate", 0.0),
                        breaker_status=curr_data.get("breaker_status", "nominal"),
                    )

                # Parse live state
                live_data = data.get("live")
                if live_data:
                    tracker.live = LiveState(
                        status=live_data.get("status", "STANDBY"),
                        model_id=live_data.get("model_id", "gemma-3-4b-it-4bit"),
                        tokens_generated=live_data.get("tokens_generated", 0),
                        max_tokens=live_data.get("max_tokens", 0),
                        prompt_tokens=live_data.get("prompt_tokens", 0),
                        ttft_ms=live_data.get("ttft_ms", 0.0),
                        is_prefix_hit=live_data.get("is_prefix_hit", False),
                        current_tps=live_data.get("current_tps", 0.0),
                        current_bw_gbs=live_data.get("current_bw_gbs", 0.0),
                        action=live_data.get("action", "device_profile_dispatch"),
                        vram_mb=live_data.get("vram_mb", 0.0),
                        swap_mb=live_data.get("swap_mb", 0.0),
                        decode_start_ns=live_data.get("decode_start_ns", 0),
                    )

                # Parse history
                hist_data = data.get("history", [])
                tracker.history.clear()
                for h in hist_data:
                    tracker.history.append(
                        RequestMetrics(
                            model_id=h.get("model_id", "gemma-3-4b-it-4bit"),
                            ttft_ms=h.get("ttft_ms", 0.0),
                            is_prefix_hit=h.get("is_prefix_hit", False),
                            tokens_generated=h.get("tokens_generated", 0),
                            decode_tps=h.get("decode_tps", 0.0),
                            wall_s=h.get("wall_s", 0.0),
                            effective_bw_gbs=h.get("effective_bw_gbs", 0.0),
                            bw_utilization_pct=h.get("bw_utilization_pct", 0.0),
                            peak_vram_mb=h.get("peak_vram_mb", 0.0),
                            swap_mb=h.get("swap_mb", 0.0),
                            action=h.get("action", "baseline"),
                            acceptance_rate=h.get("acceptance_rate", 0.0),
                            breaker_status=h.get("breaker_status", "nominal"),
                        )
                    )

                cockpit = render_cockpit(tracker, colored=True, live_tick=tick)
                # Footer help line
                footer = (
                    f"  {Theme.MUTED}[Ctrl+C] Exit Monitor │ Refresh: {refresh_hz:.0f} Hz │ Host: {host}:{port}{Theme.RESET}\033[J\n"
                )
                sys.stdout.write(cockpit + "\n" + footer)
                sys.stdout.flush()

            tick += 1
            time.sleep(interval)
    finally:
        if is_tty:
            # Cleanly restore cursor and terminal state
            sys.stdout.write("\033[?25h\033[0m\n")
            sys.stdout.flush()

    return 0


def main():
    parser = argparse.ArgumentParser(description="IronMule Live Terminal Monitor")
    parser.add_argument("--host", default="127.0.0.1", help="Friday server host")
    parser.add_argument("--port", type=int, default=8080, help="Friday server port")
    parser.add_argument("--hz", type=float, default=10.0, help="Screen refresh frequency in Hz (default: 10)")
    args = parser.parse_args()

    sys.exit(run_monitor(host=args.host, port=args.port, refresh_hz=args.hz))


if __name__ == "__main__":
    main()
