from __future__ import annotations

import unittest

from friday_phase1b.constants import (
    CHARACTERIZE_BLOCKS,
    CHARACTERIZE_FIXTURE_SEEDS,
    CHARACTERIZE_ORDER_SEEDS,
    CONTRACT_ID,
    EXPECTED_DEVICE_NAME,
    EXPECTED_MLX_VERSION,
    SCHEMA_VERSION,
)
from friday_phase1b.kernel_source import KERNEL_NAME, KERNEL_SOURCE_SHA256
from friday_phase1b.supervisor import SupervisorError, _validate_result, run_worker


def result() -> dict[str, object]:
    arms = {
        "eager_transparent": [100.0] * CHARACTERIZE_BLOCKS,
        "compiled_transparent": [99.0] * CHARACTERIZE_BLOCKS,
        "fast_rms_norm": [98.0] * CHARACTERIZE_BLOCKS,
        "compiled_fast_rms_norm": [97.0] * CHARACTERIZE_BLOCKS,
    }
    names = list(arms)
    orders = [
        names[offset:] + names[:offset]
        for offset in (block % len(names) for block in range(CHARACTERIZE_BLOCKS))
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "mode": "characterize",
        "session_index": 0,
        "baseline": None,
        "status": "passed",
        "source_sha256": KERNEL_SOURCE_SHA256,
        "kernel_name": KERNEL_NAME,
        "limits": {
            "resource": {"cpu": [90, 90]},
            "mlx_memory_bytes": 1,
            "mlx_cache_bytes": 1,
            "previous_mlx_memory_bytes": 1,
            "previous_mlx_cache_bytes": 1,
        },
        "device": {
            "mlx_version": EXPECTED_MLX_VERSION,
            "metal_available": True,
            "device_info": {"device_name": EXPECTED_DEVICE_NAME},
            "python": "3.12",
            "macos": "26",
        },
        "evidence": {
            "fixture_sha256": "a" * 64,
            "fixture_seed": CHARACTERIZE_FIXTURE_SEEDS[0],
            "order_seed": CHARACTERIZE_ORDER_SEEDS[0],
            "correctness": {"passed": True},
            "timing": {"samples_ns": arms, "orders": orders},
            "passed": True,
        },
        "error": None,
        "process": {
            "wall_ns": 1,
            "cpu_ns": 1,
            "rss_peak_bytes": 1,
            "pid": 1,
            "power_source": "ac",
        },
    }


class Phase1BSupervisorTests(unittest.TestCase):
    def test_result_contract_accepts_only_frozen_identity(self) -> None:
        value = result()
        self.assertIs(
            _validate_result(value, mode="characterize", session_index=0, baseline=None),
            value,
        )
        value["source_sha256"] = "b" * 64
        with self.assertRaises(SupervisorError):
            _validate_result(value, mode="characterize", session_index=0, baseline=None)

    def test_unknown_nested_arm_and_bad_seed_are_rejected(self) -> None:
        value = result()
        value["evidence"]["timing"]["samples_ns"]["foreign"] = [1.0] * CHARACTERIZE_BLOCKS
        with self.assertRaises(SupervisorError):
            _validate_result(value, mode="characterize", session_index=0, baseline=None)
        value = result()
        value["evidence"]["fixture_seed"] = 1
        with self.assertRaises(SupervisorError):
            _validate_result(value, mode="characterize", session_index=0, baseline=None)

    def test_unregistered_worker_requests_fail_before_spawn(self) -> None:
        with self.assertRaises(SupervisorError):
            run_worker("foreign", 0)
        with self.assertRaises(SupervisorError):
            run_worker("qualification", 1)
        with self.assertRaises(SupervisorError):
            run_worker("aa", 0, None)

    def test_registered_worker_requires_frozen_provenance_before_spawn(self) -> None:
        with self.assertRaises(SupervisorError):
            run_worker("qualification", 0)


if __name__ == "__main__":
    unittest.main()
