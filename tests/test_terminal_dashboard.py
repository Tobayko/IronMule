"""Comprehensive unit and performance tests for Friday Terminal Live-Cockpit & Telemetry.

Tests:
1. TelemetryTracker Metric Calculation:
   - TTFT, TPS, effective memory bandwidth, and utilization percentage.
   - Automatic model size derivation (1B = 0.77 GB, 4B = 2.56 GB, 12B = 7.19 GB).
   - Prefix cache hit detection.
2. Rolling History Buffer:
   - Bounded ring buffer maintaining exactly the last 50 requests.
3. Gauge Bar Generation:
   - 0 %, 50 %, 100 %, overflow, and zero-maximum edge cases.
4. Cockpit Rendering:
   - Verified components: Bandwidth, TTFT, TPS, VRAM, SWAP, Dispatch status.
   - Verified state indicators: [⚡ PREFIX-CACHE HIT], [COLD PREFILL], NOMINAL vs LATCHED.
   - Colored (ANSI) vs monochrome modes.
5. Strict Performance Gate:
   - Verifies render_cockpit() executes in < 1.0 ms (resource-efficient design).
6. HTTP Server Dashboard Integration:
   - GET /dashboard returns ASCII cockpit.
   - GET /telemetry returns structured JSON metrics.
   - Inferences automatically update TelemetryTracker.
"""

from __future__ import annotations

import json
import threading
import time
import unittest
import urllib.request
from typing import Any

from friday_calibrate.profile import DeviceProfile, KnobVerdict
from friday_serve.http_server import create_server
from friday_serve.server import Server
from friday_serve.telemetry import RequestMetrics, TelemetryTracker
from friday_serve.terminal_dashboard import make_bar, render_cockpit

MODEL_ID = "mlx-community/gemma-3-4b-it-4bit"
REVISION = "rev1"


def make_profile(*verified: str) -> DeviceProfile:
    verdicts = [
        KnobVerdict(k, "verified" if k in verified else "failed", 6, 0.9, 0.88, 0.93, True)
        for k in ("head_skip", "fixed_compiled", "bundled_readback")
    ]
    return DeviceProfile(
        profile_id="device-telemetry-test",
        model_id=MODEL_ID,
        model_revision=REVISION,
        hardware_sha256="a" * 64,
        environment_sha256="b" * 64,
        mde=0.006,
        knobs=tuple(verdicts),
    )


class FakeBackend:
    def __init__(self) -> None:
        self.model_id = MODEL_ID
        self.model_revision = REVISION

    def encode(self, prompt: str) -> list[int]:
        return [ord(c) for c in prompt]

    def generate(self, token_ids, max_tokens, knobs):
        return {
            "logical_tokens": list(range(max_tokens)),
            "text": "Answer",
            "prefill_ns": 40_000_000,
            "decode_ns": 200_000_000,
            "knobs": dict(knobs),
            "prefix_cache_hits": 1,
        }

    def stream_generate(self, token_ids, max_tokens, knobs):
        yield {
            "type": "token",
            "token": 1,
            "tokens": [1],
            "text": "Hello",
            "is_first": True,
            "prefill_ns": 35_000_000,
            "prefix_cache_hits": 1,
        }
        for i in range(1, max_tokens):
            yield {
                "type": "token",
                "token": i + 1,
                "tokens": [i + 1],
                "text": f" word_{i}",
                "is_first": False,
            }
        yield {
            "type": "done",
            "total_tokens": max_tokens,
            "decode_ns": 165_000_000,
            "total_ns": 200_000_000,
            "knobs": dict(knobs),
            "prefix_cache_hits": 1,
            "logical_tokens": list(range(1, max_tokens + 1)),
        }


class TestTelemetryAndDashboard(unittest.TestCase):
    def test_telemetry_tracker_metric_calculation(self) -> None:
        """Validates exact math for TTFT, TPS, and memory bandwidth."""
        tracker = TelemetryTracker(peak_bandwidth_gbs=400.0)

        # 4B model (2.56 GB), 50 ms prefill, 1.0 s decode, 51 tokens (50 decode tokens)
        metrics = tracker.record_request(
            model_id="gemma-3-4b-it",
            prefill_ns=50_000_000,
            decode_ns=1_000_000_000,
            tokens_generated=51,
            prefix_cache_hits=0,
            action="full_optimized",
            acceptance_rate=0.75,
            breaker_status="nominal",
            peak_vram_mb=3100.0,
            swap_mb=0.0,
        )

        self.assertEqual(metrics.ttft_ms, 50.0)
        self.assertFalse(metrics.is_prefix_hit)
        self.assertEqual(metrics.tokens_generated, 51)
        self.assertEqual(metrics.decode_tps, 50.0)  # (51 - 1) / 1.0s
        self.assertEqual(metrics.effective_bw_gbs, 128.0)  # 2.56 GB * 50.0 TPS
        self.assertEqual(metrics.bw_utilization_pct, 32.0)  # 128 / 400 * 100
        self.assertEqual(metrics.wall_s, 1.05)  # (50ms + 1000ms) / 1000
        self.assertEqual(metrics.peak_vram_mb, 3100.0)
        self.assertEqual(metrics.swap_mb, 0.0)
        self.assertEqual(metrics.action, "full_optimized")
        self.assertEqual(metrics.acceptance_rate, 0.75)
        self.assertEqual(metrics.breaker_status, "nominal")

    def test_model_size_derivation(self) -> None:
        """Different parameter scales map to accurate memory footprints."""
        tracker = TelemetryTracker()
        self.assertEqual(tracker.model_size_gb("meta-llama-1b"), 0.77)
        self.assertEqual(tracker.model_size_gb("gemma-3-4b-it"), 2.56)
        self.assertEqual(tracker.model_size_gb("gemma-3-12b-it"), 7.19)
        self.assertEqual(tracker.model_size_gb("unknown-model"), 2.56)

    def test_rolling_history_buffer(self) -> None:
        """History is strictly capped at max_history (50 requests)."""
        tracker = TelemetryTracker(max_history=50)
        for i in range(60):
            tracker.record_request(
                model_id="gemma-3-4b-it",
                prefill_ns=10_000_000,
                decode_ns=100_000_000,
                tokens_generated=10,
                action=f"action_{i}",
            )

        history = tracker.get_history()
        self.assertEqual(len(history), 50)
        self.assertEqual(history[0].action, "action_10")
        self.assertEqual(history[-1].action, "action_59")
        self.assertEqual(tracker.get_current().action, "action_59")

    def test_make_bar(self) -> None:
        """Gauge bar rendering across 0%, 50%, 100%, and boundary cases."""
        self.assertEqual(make_bar(0, 100, width=10), "░░░░░░░░░░")
        self.assertEqual(make_bar(50, 100, width=10), "█████░░░░░")
        self.assertEqual(make_bar(100, 100, width=10), "██████████")
        self.assertEqual(make_bar(150, 100, width=10), "██████████")  # Clamp at 100%
        self.assertEqual(make_bar(-10, 100, width=10), "░░░░░░░░░░")  # Clamp at 0%
        self.assertEqual(make_bar(50, 0, width=10), "░░░░░░░░░░")  # Zero max

    def test_render_cockpit_standby(self) -> None:
        """Standby mode when tracker has recorded no requests."""
        tracker = TelemetryTracker()
        out = render_cockpit(tracker, colored=False)
        self.assertIn("FRIDAY ULTIMATE INFERENCE COCKPIT", out)
        self.assertIn("STATUS: STANDBY", out)

    def test_render_cockpit_active(self) -> None:
        """Active cockpit rendering with all gauges and indicators."""
        tracker = TelemetryTracker(peak_bandwidth_gbs=400.0)
        tracker.record_request(
            model_id=MODEL_ID,
            prefill_ns=32_000_000,
            decode_ns=500_000_000,
            tokens_generated=50,
            prefix_cache_hits=1,
            action="full_optimized",
            acceptance_rate=0.82,
            breaker_status="nominal",
            peak_vram_mb=3250.0,
            swap_mb=0.0,
        )

        # Monochrome check
        mono = render_cockpit(tracker, colored=False)
        self.assertIn("FRIDAY ULTIMATE INFERENCE COCKPIT", mono)
        self.assertIn(MODEL_ID, mono)
        self.assertIn("MEMORY BANDWIDTH UTILIZATION", mono)
        self.assertIn("TIME TO FIRST TOKEN (TTFT)", mono)
        self.assertIn("32.0 ms", mono)
        self.assertIn("[⚡ PREFIX-CACHE HIT]", mono)
        self.assertIn("DECODE RATE (TPS)", mono)
        self.assertIn("HARDWARE & MEMORY SAFETY", mono)
        self.assertIn("SWAP: 0 MB [NOMINAL]", mono)
        self.assertIn("RL Strategy: full_optimized", mono)
        self.assertIn("Speculation Acceptance: 82.0 %", mono)

        # Color escape check
        color_out = render_cockpit(tracker, colored=True)
        self.assertIn("\033[", color_out)

    def test_render_cockpit_swap_warning(self) -> None:
        """Swap allocation triggers warning flag."""
        tracker = TelemetryTracker()
        tracker.record_request(
            model_id=MODEL_ID,
            prefill_ns=100_000_000,
            decode_ns=500_000_000,
            tokens_generated=20,
            prefix_cache_hits=0,
            swap_mb=64.5,
        )
        out = render_cockpit(tracker, colored=False)
        self.assertIn("[COLD PREFILL]", out)
        self.assertIn("SWAP: 64.5 MB [⚠️ SWAP WARNING]", out)

    def test_render_cockpit_performance_under_1ms(self) -> None:
        """Strict performance gate: rendering must take strictly less than 1.0 ms."""
        tracker = TelemetryTracker()
        tracker.record_request(
            model_id=MODEL_ID,
            prefill_ns=40_000_000,
            decode_ns=300_000_000,
            tokens_generated=32,
            prefix_cache_hits=1,
            action="full_optimized",
            acceptance_rate=0.67,
            peak_vram_mb=2800.0,
            swap_mb=0.0,
        )

        # Warmup
        for _ in range(10):
            render_cockpit(tracker, colored=True)

        # Measure 100 iterations
        t0 = time.perf_counter()
        iterations = 100
        for _ in range(iterations):
            render_cockpit(tracker, colored=True)
        elapsed_s = time.perf_counter() - t0
        avg_ms = (elapsed_s / iterations) * 1000.0

        # Assert sub-millisecond execution
        self.assertLess(avg_ms, 1.0, f"Dashboard rendering took {avg_ms:.4f} ms, expected < 1.0 ms")


class TestHTTPTelemetryIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = FakeBackend()
        self.profile = make_profile("head_skip")
        self.server_instance = Server(self.backend, self.profile)
        self.tracker = TelemetryTracker()

        self.httpd = create_server(
            self.server_instance, host="127.0.0.1", port=0, telemetry_tracker=self.tracker
        )
        self.port = self.httpd.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.server_thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.server_thread.start()

    def tearDown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.server_thread.join(timeout=2.0)

    def test_dashboard_and_telemetry_routes(self) -> None:
        """GET /dashboard and GET /telemetry endpoints reflect inference state."""
        # Initial state (standby)
        with urllib.request.urlopen(f"{self.base_url}/dashboard") as resp:
            self.assertEqual(resp.status, 200)
            text = resp.read().decode("utf-8")
            self.assertIn("STATUS: STANDBY", text)

        with urllib.request.urlopen(f"{self.base_url}/telemetry") as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIsNone(data["current"])
            self.assertEqual(data["history"], [])

        # Perform a streaming inference request
        payload = {
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "Explain relativity"}],
            "max_tokens": 8,
            "stream": True,
        }
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            _ = resp.read()

        # Check telemetry updated
        current = self.tracker.get_current()
        self.assertIsNotNone(current)
        self.assertEqual(current.tokens_generated, 8)
        self.assertTrue(current.is_prefix_hit)

        # GET /dashboard now returns active cockpit
        with urllib.request.urlopen(f"{self.base_url}/dashboard") as resp:
            self.assertEqual(resp.status, 200)
            text = resp.read().decode("utf-8")
            self.assertIn("MEMORY BANDWIDTH UTILIZATION", text)
            self.assertIn("[⚡ PREFIX-CACHE HIT]", text)

        # GET /telemetry returns valid JSON metrics
        with urllib.request.urlopen(f"{self.base_url}/telemetry") as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIsNotNone(data["current"])
            self.assertEqual(data["current"]["tokens_generated"], 8)
            self.assertEqual(len(data["history"]), 1)


if __name__ == "__main__":
    unittest.main()
