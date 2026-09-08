"""Pure protocol tests for the opt-in current-engine bridge.

Native Engine construction and serving intentionally have no test double here:
they are qualified only by the dedicated native worker path.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
from types import SimpleNamespace
import unittest

from ironmule_product.engine_bridge import (
    _historical_configuration_metadata,
    _result_from_raw,
    _validate_configuration,
    _validate_request,
)


_METADATA = {
    "selected_knobs": {"width": 1},
    "profile_available": False,
    "profile_source": "baseline_no_compatible_profile",
    "current_identity": {"identity_sha256": "a" * 64},
    "plan": "strict_one_shot",
}


class EngineBridgeProtocolTests(unittest.TestCase):
    def test_eos_is_normalized_to_stop_and_eos_token_is_preserved(self):
        raw = SimpleNamespace(tokens=[19, 2], text="protocol example", stop_reason="eos",
                              metrics={"generated_tokens": 2})

        result = _result_from_raw(raw, 2, _METADATA)

        self.assertEqual(result.tokens, [19, 2])
        self.assertEqual(result.text, "protocol example")
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.token_count, 2)
        self.assertEqual(result.metrics, {"generated_tokens": 2})

    def test_length_is_normalized_to_length(self):
        raw = SimpleNamespace(tokens=[19], text="protocol example", stop_reason="length", metrics={})
        self.assertEqual(_result_from_raw(raw, 1, _METADATA).finish_reason, "length")

    def test_invalid_result_protocol_is_rejected(self):
        cases = (
            SimpleNamespace(tokens=[], text="x", stop_reason="length", metrics={}),
            SimpleNamespace(tokens=[True], text="x", stop_reason="length", metrics={}),
            SimpleNamespace(tokens=[-1], text="x", stop_reason="length", metrics={}),
            SimpleNamespace(tokens=[1, 2], text="x", stop_reason="length", metrics={}),
            SimpleNamespace(tokens=[1], text=None, stop_reason="length", metrics={}),
            SimpleNamespace(tokens=[1], text="x", stop_reason="error", metrics={}),
            SimpleNamespace(tokens=[1], text="x", stop_reason="length", metrics=[]),
        )
        for raw in cases:
            with self.subTest(raw=raw):
                with self.assertRaises(RuntimeError):
                    _result_from_raw(raw, 1, _METADATA)

    def test_request_ids_and_limit_reject_bool_and_invalid_values(self):
        self.assertEqual(_validate_request([0, 7], 2), [0, 7])
        for prompt_ids, max_tokens in (([True], 1), ([-1], 1), ([1], True), ([1], 0)):
            with self.subTest(prompt_ids=prompt_ids, max_tokens=max_tokens):
                with self.assertRaises(ValueError):
                    _validate_request(prompt_ids, max_tokens)

    def test_b39d_candidates_are_exact_historical_configuration_records(self):
        baseline = _historical_configuration_metadata("baseline_interactive")
        throughput = _historical_configuration_metadata("core_throughput")
        self.assertEqual(baseline["knobs"], {
            "capacity_slack": 0, "compiled_fixed_cache": False,
            "fuse_projections": False, "fused_argmax": False,
            "head_skip_prefill": False, "prefill_into_fixed": False,
            "readback_every": 1, "speculate_k": 0, "speculate_ngram": 3,
            "wired_fraction": 0.0,
        })
        self.assertEqual(baseline["mode"], "interactive")
        self.assertIsNone(baseline["max_width"])
        self.assertTrue(throughput["knobs"]["compiled_fixed_cache"])
        self.assertTrue(throughput["knobs"]["head_skip_prefill"])
        self.assertEqual(throughput["mode"], "throughput")
        self.assertEqual(throughput["max_width"], 4)
        self.assertFalse(throughput["activation_allowed"])
        self.assertTrue(throughput["requires_fresh_native_qualification"])

    def test_current_profile_is_not_a_historical_candidate_and_unknown_configuration_rejects(self):
        self.assertIsNone(_historical_configuration_metadata("current_profile"))
        self.assertEqual(_validate_configuration("current_profile"), "current_profile")
        with self.assertRaises(ValueError):
            _validate_configuration("automatic_best_b39d")

    def test_isolated_import_does_not_load_runtime_or_mlx(self):
        root = pathlib.Path(__file__).resolve().parents[2]
        command = (
            "import pathlib, sys; "
            f"sys.path.insert(0, {str(root)!r}); "
            "import ironmule_product.engine_bridge; "
            "assert 'mlx' not in sys.modules; "
            "assert 'ironmule.runtime' not in sys.modules; "
            "assert 'ironmule.service' not in sys.modules"
        )
        completed = subprocess.run(
            [sys.executable, "-I", "-c", command], check=False, capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
