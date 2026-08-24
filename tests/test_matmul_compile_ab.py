"""Offline contract tests for the cycle-16 matmul/compile study.

These tests deliberately do not import MLX, allocate device arrays, start a
model, or execute the hardware harness.  The worker event used below is a
complete synthetic event and is only used to exercise the closed validator.
"""

from __future__ import annotations

import ast
import base64
import copy
import hashlib
import http.client
import importlib.util
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = ROOT / "experiments" / "matmul_compile_ab"
PREREG_PATH = STUDY_DIR / "PREREGISTRATION.md"
WORKER_PATH = STUDY_DIR / "worker.py"
HARNESS_PATH = STUDY_DIR / "measure_matmul_compile.py"
DASHBOARD_PATH = STUDY_DIR / "dashboard.py"
PYTHON = ROOT / ".venv" / "bin" / "python"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


worker = _load_module(WORKER_PATH, "cycle16_test_worker")
harness = _load_module(HARNESS_PATH, "cycle16_test_harness")
dashboard = _load_module(DASHBOARD_PATH, "cycle16_test_dashboard")


ARM_NAMES = ("standard_eager", "fixed_eager", "fixed_compiled")
EXPECTED_EVENT_KEYS = {
    "arm_order",
    "arms",
    "cache_capacity",
    "correctness",
    "device",
    "error",
    "event",
    "fixed_steps",
    "load_count",
    "model_id",
    "model_key",
    "model_load_ns",
    "model_work_ns",
    "observed_model_work_ns",
    "charged_model_work_ns",
    "guard_recorded_model_work_ns",
    "mlx_peak_bytes",
    "pid",
    "prompt_sha256",
    "prompt_token_ids",
    "prompt_tokens",
    "rendered_prompt_b64",
    "rendered_prompt_sha256",
    "rss_peak_bytes",
    "sampler_temperature",
    "snapshot_integrity",
    "snapshot_path",
    "snapshot_revision",
    "snapshot_sha256",
    "status",
    "study_id",
    "text_sha256_by_arm",
    "token_sha256_by_arm",
    "worker_watchdog_seconds",
    "weight_sha256",
    "power_source",
    "arm_budget",
    "arm_resources",
    "budget",
    "swap_before_bytes",
    "swap_after_bytes",
    "swap_delta_bytes",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity() -> dict[str, object]:
    manifest = {
        "config.json": {
            "dev": 1,
            "inode": 2,
            "mtime_ns": 3,
            "path": "/local/models/snapshots/fixed/config.json",
            "size": 4,
        },
        "model.safetensors": {
            "dev": 1,
            "inode": 5,
            "mtime_ns": 6,
            "path": "/local/models/snapshots/fixed/model.safetensors",
            "size": 7,
        },
    }
    return {
        "snapshot_path": "/local/models/snapshots/fixed",
        "snapshot_sha256": "a" * 64,
        "weight_sha256": {"model.safetensors": "b" * 64},
        "execution_stat_manifest": manifest,
    }


def _budget(
    model_work_ns: int,
    *,
    charged_model_work_ns: int | None = None,
    guard_recorded_model_work_ns: int | None = None,
) -> dict[str, object]:
    charged = model_work_ns if charged_model_work_ns is None else charged_model_work_ns
    recorded = charged if guard_recorded_model_work_ns is None else guard_recorded_model_work_ns
    seconds = recorded / 1_000_000_000
    return {
        "gpu_work_seconds": seconds,
        "max_continuous_gpu_seconds": min(seconds, 5.0),
        "cooldown_seconds": 0.0,
        "required_break_seconds": 0.1,
        "wall_seconds": 1.0,
        "gpu_work_limit_seconds": 120.0,
        "continuous_gpu_limit_seconds": 6.0,
        "duty_cycle_limit": 0.15,
        "wall_limit_seconds": 1200.0,
        "candidate_cooldown_seconds": 0.0,
        "required_break_limit_seconds": 4.0,
    }


def _arm_value(arm: str, charged_ns: int = 1_000_000) -> dict[str, object]:
    tokens = list(range(1, 33))
    text = "synthetic fixed-step output"
    decode = [100] * 31
    intertoken = [110] * 31
    return {
        "arm": arm,
        "cache_capacity": 512,
        "cache_conversion_ns": 5 if arm != "standard_eager" else 0,
        "compile_cold_ns": 30 if arm == "fixed_compiled" else None,
        "compile_wrapper_ns": 20 if arm == "fixed_compiled" else 0,
        "decode_forward_ns": decode,
        "decode_forward_total_ns": sum(decode),
        "decode_forwards": 31,
        "finish_reason": "fixed_steps",
        "intertoken_ns": intertoken,
        "intertoken_p50_ns": 110.0,
        "intertoken_p95_ns": 110.0,
        "intertoken_p99_ns": 110.0,
        "model_work_ns": 3200,
        "prefill_ns": 100,
        "prompt_sha256": worker.PROMPT_SHA256,
        "prompt_token_sha256": _sha256_bytes(worker._canonical_json(list(range(322)))),
        "rendered_prompt_sha256": _sha256_bytes(b"synthetic rendered prompt"),
        "text": text,
        "text_utf8_sha256": _sha256_bytes(text.encode("utf-8")),
        "token_rate": 31 / (sum(decode) / 1_000_000_000),
        "token_sha256": _sha256_bytes(worker._canonical_json(tokens)),
        "tokens": tokens,
        "ttft_ns": 100,
        "warmup_decode_forward_ns": [100] * 8,
        "warmup_intertoken_ns": [110] * 8,
        "warmup_forwards": 8,
        "observed_model_work_ns": charged_ns,
        "charged_model_work_ns": charged_ns,
        "charge_accepted": True,
        "arm_wall_ns": charged_ns,
        "budget_summary": _budget(charged_ns),
    }


def _synthetic_event(
    *,
    status: str = "complete",
    arms_count: int = 3,
    order: tuple[str, ...] | None = None,
) -> tuple[dict[str, object], dict[str, object], int]:
    identity = _identity()
    selected_order = order or harness.ARM_PERMUTATIONS[0]
    selected = list(selected_order[:arms_count])
    arms = {arm: _arm_value(arm) for arm in selected}
    arm_budget = {
        arm: {
            "observed_model_work_ns": 1_000_000,
            "charged_model_work_ns": 1_000_000,
            "charge_accepted": True,
            "guard_gpu_work_before_seconds": 0.0,
            "guard_gpu_work_after_seconds": 0.001,
            "guard_recorded_model_work_ns": 1_000_000,
            "duty_formula_break_seconds": (1_000_000 / 1e9) * 0.85 / 0.15,
            "required_break_blocks": 13,
        }
        for arm in selected
    }
    arm_resources = {
        arm: {
            "rss_peak_bytes": 1_000_000,
            "mlx_peak_bytes": 2_000_000,
            "swap_after_bytes": 10,
            "swap_delta_bytes": 0,
        }
        for arm in selected
    }
    token_hashes = {arm: arms[arm]["token_sha256"] for arm in selected}
    text_hashes = {arm: arms[arm]["text_utf8_sha256"] for arm in selected}
    rendered = b"synthetic rendered prompt"
    prompt_token_ids = list(range(322))
    model_work_ns = len(selected) * 1_000_000
    event: dict[str, object] = {
        "arm_order": list(selected_order),
        "arms": arms,
        "cache_capacity": 512,
        "correctness": {
            "all_arms_text_equal": status == "complete",
            "all_arms_token_equal": status == "complete",
            "first_mismatch": None if status != "correctness_failed" else {"token_index": 1},
            "required_arm_count": 3,
        },
        "device": "Device(gpu, 0)",
        "error": None if status == "complete" else {"type": status, "message": "synthetic terminal event"},
        "event": "complete",
        "fixed_steps": 32,
        "load_count": 1,
        "model_id": harness.MODEL_ID,
        "model_key": "4b",
        "model_load_ns": 20,
        "model_work_ns": model_work_ns,
        "observed_model_work_ns": model_work_ns,
        "charged_model_work_ns": model_work_ns,
        "guard_recorded_model_work_ns": model_work_ns,
        "mlx_peak_bytes": 2_000_000,
        "pid": 4242,
        "prompt_sha256": worker.PROMPT_SHA256,
        "prompt_token_ids": prompt_token_ids,
        "prompt_tokens": 322,
        "rendered_prompt_b64": base64.b64encode(rendered).decode("ascii"),
        "rendered_prompt_sha256": _sha256_bytes(rendered),
        "rss_peak_bytes": 1_000_000,
        "sampler_temperature": 0.0,
        "snapshot_integrity": {
            "after_load_stat_manifest": identity["execution_stat_manifest"],
            "before_load_stat_manifest": identity["execution_stat_manifest"],
            "bound_snapshot_sha256": identity["snapshot_sha256"],
            "bound_weight_sha256": identity["weight_sha256"],
        },
        "snapshot_path": identity["snapshot_path"],
        "snapshot_revision": harness.MODEL_REVISION,
        "snapshot_sha256": identity["snapshot_sha256"],
        "status": status,
        "study_id": harness.STUDY_ID,
        "text_sha256_by_arm": text_hashes,
        "token_sha256_by_arm": token_hashes,
        "worker_watchdog_seconds": 6.0,
        "weight_sha256": identity["weight_sha256"],
        "power_source": "ac_power",
        "arm_budget": arm_budget,
        "arm_resources": arm_resources,
        "budget": _budget(model_work_ns),
        "swap_before_bytes": 10,
        "swap_after_bytes": 10 if status == "complete" else None,
        "swap_delta_bytes": 0 if status == "complete" else None,
    }
    return event, identity, 0 if status == "complete" else 1


def _dashboard_result() -> dict[str, object]:
    metric = {"median": 1.0, "mad": 0.1, "p50": 1.0, "p95": 1.1, "p99": 1.2}
    arms = {
        arm: {
            "arm": arm,
            "runs": 6,
            "metrics": {name: dict(metric) for name in dashboard.PAIR_METRICS},
            "token_sha256": ["a" * 64] * 6,
            "text_sha256": ["b" * 64] * 6,
            "peak_rss_bytes": 100,
            "peak_mlx_bytes": 200,
            "swap_deltas_bytes": [0] * 6,
        }
        for arm in ARM_NAMES
    }
    comparison = {
        "median": 0.9,
        "bootstrap_95_ci": {
            "lower": 0.8,
            "upper": 0.99,
            "method": "paired six-block median-ratio bootstrap percentile",
            "resamples": 10_000,
            "seed": 20260824,
        },
    }
    paired = {
        "complete": True,
        "ratios": {
            metric_name: {
                "fixed_compiled_div_standard_eager": dict(comparison),
                "fixed_compiled_div_fixed_eager": dict(comparison),
            }
            for metric_name in dashboard.PAIR_METRICS
        },
    }
    runs = [
        {
            "block": index,
            "arms": {arm: {} for arm in ARM_NAMES},
            "correctness": {"all_arms_token_equal": True, "all_arms_text_equal": True},
            "status": "complete",
        }
        for index in range(1, 7)
    ]
    return {
        "schema_version": 1,
        "study_id": dashboard.EXPECTED_STUDY_ID,
        "run_id": dashboard.EXPECTED_RUN_ID,
        "formal_claim": False,
        "decision": "runtime_compile_wins_exact_scope",
        "runs": runs,
        "provenance": {},
        "partial_result": False,
        "error": None,
        "budget": {},
        "resources": {},
        "gates": {
            "all_blocks_completed": True,
            "resource_pass": True,
            "budget_pass": True,
            "block_correctness_pass": True,
            "determinism_pass": True,
            "candidate_runnable": True,
            "pairing_pass": True,
            "snapshot_content_pass": True,
        },
        "metrics": {"arms": arms, "paired": paired, "derived": {
            "complete": True,
            "calculated_only": True,
            "warmed_decode_ratio_median": 0.9,
            "cold_decode_ratio_median": 1.1,
            "break_even_decode_forwards": [5.0] * 6,
        }, "runs_completed": 6},
        "snapshot_postflight": {},
        "thresholds": {},
        "completed_at_unix_ns": 1,
    }


def _paired_runs(
    *, standard: int = 100,
    fixed_eager: int = 100,
    compiled: int = 90,
) -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    for index, order in enumerate(harness.ARM_PERMUTATIONS, start=1):
        arms = {}
        for arm, value in (
            ("standard_eager", standard),
            ("fixed_eager", fixed_eager),
            ("fixed_compiled", compiled),
        ):
            arms[arm] = {
                "decode_forward_ns": [value] * 31,
                "intertoken_p50_ns": value,
                "intertoken_p95_ns": value,
                "intertoken_p99_ns": value,
                "prefill_ns": 100,
                "ttft_ns": 120,
                "model_work_ns": 100 + 31 * value,
                "arm_wall_ns": 2_000,
                "token_rate": 1_000_000_000 / value,
                "cache_conversion_ns": 5 if arm != "standard_eager" else 0,
                "compile_wrapper_ns": 20 if arm == "fixed_compiled" else 0,
                "compile_cold_ns": 30 if arm == "fixed_compiled" else 0,
            }
        runs.append({
            "block": index,
            "arm_order": list(order),
            "pid": index,
            "arms": arms,
            "process_wall_ns": 3_000,
            "rss_peak_bytes": 4_000,
            "mlx_peak_bytes": 5_000,
            "swap_delta_bytes": 0,
        })
    return runs


class MatmulCompileABTests(unittest.TestCase):
    def test_preregistration_and_workload_contract(self):
        prereg = PREREG_PATH.read_text(encoding="utf-8")
        self.assertEqual(_sha256_path(PREREG_PATH), harness.FROZEN_PREREGISTRATION_SHA256)
        self.assertEqual(worker.PROMPT_SHA256, "c746eca8644a18fc75673acb9b3dbdf03825cbfba6c76faede5d909cf3d2ea0b")
        self.assertEqual(_sha256_bytes(worker.PLANNER_PROMPT.encode("utf-8")), worker.PROMPT_SHA256)
        self.assertEqual(worker.EXPECTED_PROMPT_TOKENS, 322)
        self.assertEqual(worker.CAPACITY, 512)
        self.assertEqual(worker.OUTPUT_TOKENS, 32)
        self.assertEqual(worker.DECODE_FORWARDS, 31)
        self.assertEqual(worker.WARMUP_FORWARDS, 8)
        self.assertEqual(harness.OUTPUT_TOKENS, 32)
        self.assertEqual(harness.DECODE_FORWARDS, 31)
        self.assertEqual(harness.PAIR_COUNT, 6)
        self.assertEqual(harness.ARM_PERMUTATIONS, tuple(harness.PAIR_SCHEDULE))
        self.assertEqual(len(harness.PAIR_SCHEDULE), 6)
        self.assertIn("Genau 32 Ausgabetoken", prereg)
        self.assertIn("31 Decode-Forwards", prereg)
        self.assertIn("Warmup-Decode-Schritte", prereg)
        self.assertIn("decode_total", prereg)

    def test_worker_has_no_top_level_mlx_import_and_has_explicit_compile_state(self):
        tree = ast.parse(WORKER_PATH.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Import):
                self.assertTrue(all(not alias.name.startswith("mlx") for alias in node.names))
            elif isinstance(node, ast.ImportFrom):
                self.assertFalse((node.module or "").startswith("mlx"))
        compile_source = worker._make_compiled_forward.__code__
        self.assertIn("compile", compile_source.co_names)
        source = __import__("inspect").getsource(worker._make_compiled_forward)
        self.assertIn("mx.compile(body, shapeless=False)", source)
        self.assertNotIn("inputs=", source)
        self.assertNotIn("outputs=", source)
        self.assertIn("def body(input_ids: Any, state: dict[str, Any])", source)
        self.assertIn("return _fixed_forward(model, input_ids, state, mx)", source)

    def test_first_compiled_call_failure_is_candidate_not_runnable(self):
        """A compile/runtime-shape failure is not silently classified as a generic error."""

        class FakeMX:
            def array(self, value):
                return value

            def eval(self, *values):
                del values

            def synchronize(self):
                return None

        prepare = {
            "cache": object(),
            "conversion_ns": 0,
            "fixed_state": {"position": {"offset": 0}, "layers": []},
            "first_token": 1,
            "prefill_ns": 1,
            "ttft_ns": 1,
        }

        calls = {"compiled": 0}

        def broken_compiled(_input_ids, _state):
            calls["compiled"] += 1
            raise RuntimeError("compiled shape rejected")

        with mock.patch.object(worker, "_prepare_prefill", return_value=prepare), \
                mock.patch.object(worker, "_make_compiled_forward", return_value=broken_compiled):
            with self.assertRaises(worker.CandidateNotRunnable):
                worker._run_arm(
                    object(), object(), [1], "fixed_compiled", object(), FakeMX()
                )
        self.assertEqual(calls["compiled"], 1)

    def test_lazy_compiled_output_failure_during_eval_is_candidate_not_runnable(self):
        class LazyFailureMX:
            def array(self, value):
                return value

            def eval(self, *values):
                del values
                raise RuntimeError("lazy compiled evaluation failed")

            def synchronize(self):
                raise AssertionError("synchronize should not be reached after eval failure")

        prepare = {
            "cache": object(),
            "conversion_ns": 0,
            "fixed_state": {"position": {"offset": 0}, "layers": []},
            "first_token": 1,
            "prefill_ns": 1,
            "ttft_ns": 1,
        }

        def compiled(_input_ids, state):
            return object(), state

        with mock.patch.object(worker, "_prepare_prefill", return_value=prepare), \
                mock.patch.object(worker, "_make_compiled_forward", return_value=compiled):
            with self.assertRaises(worker.CandidateNotRunnable):
                worker._run_arm(
                    object(), object(), [1], "fixed_compiled", object(), LazyFailureMX()
                )

        class SyncFailureMX:
            def array(self, value):
                return value

            def eval(self, *values):
                del values

            def synchronize(self):
                raise RuntimeError("lazy compiled synchronization failed")

        with mock.patch.object(worker, "_prepare_prefill", return_value=prepare), \
                mock.patch.object(worker, "_make_compiled_forward", return_value=compiled):
            with self.assertRaises(worker.CandidateNotRunnable):
                worker._run_arm(
                    object(), object(), [1], "fixed_compiled", object(), SyncFailureMX()
                )

    def test_fixed_cache_conversion_slice_eval_and_sync_failures_are_classified(self):
        class Tensor:
            dtype = "float16"
            shape = (1, 1, 322, 4)

            def __getitem__(self, item):
                del item
                return self

        class Layer:
            def __init__(self):
                self.keys = Tensor()
                self.values = Tensor()

        class SliceMX:
            int32 = "int32"

            def zeros(self, shape, dtype=None):
                value = Tensor()
                value.shape = shape
                value.dtype = dtype
                return value

            def array(self, value, dtype=None):
                del dtype
                return value

            def eval(self, *values):
                del values

            def synchronize(self):
                return None

            def slice_update(self, *args, **kwargs):
                del args, kwargs
                raise RuntimeError("slice_update fixed-cache shape failure")

        class Model:
            def make_cache(self):
                return [Layer()]

            def __call__(self, prompt_array, *, cache):
                del prompt_array, cache
                return object()

        with mock.patch.object(worker, "_select_token", return_value=(1, None)):
            with self.assertRaises(worker.CandidateNotRunnable):
                worker._prepare_prefill(
                    Model(), [1] * 322, "fixed_eager", object(), object(), SliceMX()
                )

        class ResourceSliceMX(SliceMX):
            def slice_update(self, *args, **kwargs):
                del args, kwargs
                raise MemoryError("out of memory during fixed-cache conversion")

        with mock.patch.object(worker, "_select_token", return_value=(1, None)):
            with self.assertRaises(worker.WorkerError):
                worker._prepare_prefill(
                    Model(), [1] * 322, "fixed_eager", object(), object(), ResourceSliceMX()
                )

        fixed_state = {
            "position": {"offset": object()},
            "layers": [{"keys": object(), "values": object()}],
        }

        class ConversionMX:
            int32 = "int32"

            def __init__(self, phase, failure):
                self.phase = phase
                self.failure = failure
                self.eval_calls = 0
                self.sync_calls = 0

            def array(self, value, dtype=None):
                del dtype
                return value

            def eval(self, *values):
                del values
                self.eval_calls += 1
                if self.phase == "eval" and self.eval_calls == 2:
                    raise self.failure

            def synchronize(self):
                self.sync_calls += 1
                if self.phase == "sync" and self.sync_calls == 2:
                    raise self.failure

        for phase, exc_factory, expected in (
            ("eval", lambda: RuntimeError("lazy conversion eval failed"), worker.CandidateNotRunnable),
            ("sync", lambda: RuntimeError("lazy conversion sync failed"), worker.CandidateNotRunnable),
            ("eval", lambda: MemoryError("out of memory during conversion eval"), worker.WorkerError),
            ("sync", lambda: MemoryError("out of memory during conversion sync"), worker.WorkerError),
        ):
            with self.subTest(phase=phase, exception=type(exc_factory()).__name__):
                mx = ConversionMX(phase, exc_factory())
                with mock.patch.object(worker, "_fixed_state_from_standard_cache", return_value=fixed_state), \
                        mock.patch.object(worker, "_select_token", return_value=(1, None)):
                    with self.assertRaises(expected):
                        worker._prepare_prefill(
                            Model(), [1] * 322, "fixed_eager", object(), object(), mx
                        )

    def test_worker_fixed_cache_uses_slice_update_and_outer_only_offset(self):
        import inspect

        update_source = inspect.getsource(worker.FixedKVCache.update_and_fetch)
        forward_source = inspect.getsource(worker._fixed_forward)
        self.assertIn("slice_update", update_source)
        self.assertNotIn("concatenate", update_source)
        self.assertNotIn("self._position[\"offset\"] =", update_source)
        self.assertIn("old_offset = state_tree[\"position\"][\"offset\"]", forward_source)
        self.assertIn('"offset": old_offset + input_ids.shape[1]', forward_source)
        self.assertIn("The shared offset is intentionally not changed here", update_source)

    def test_worker_clock_order_and_return_contract(self):
        import inspect

        source = inspect.getsource(worker._run_arm)
        self.assertLess(source.index("if len(tokens) != OUTPUT_TOKENS"), source.index('"tokens": tokens'))
        self.assertLess(source.index("if not isinstance(text, str) or not text"), source.index('"text": text'))
        harness_source = inspect.getsource(harness._run_worker) if hasattr(harness, "_run_worker") else WORKER_PATH.read_text(encoding="utf-8")
        # The parent-side order is in the worker.  Keep the assertions bounded
        # to that function so helper definitions cannot satisfy them by chance.
        self.assertLess(harness_source.index("arm_finished_ns = time.perf_counter_ns()"), harness_source.index("charged_model_work_ns += arm_ns"))
        self.assertLess(harness_source.index("observed_model_work_ns += arm_ns"), harness_source.index("_charge_arm(guard, arm_ns)"))
        self.assertLess(harness_source.index("_charge_arm(guard, arm_ns)"), harness_source.index("charged_model_work_ns += arm_ns"))
        self.assertLess(harness_source.index("charged_model_work_ns += arm_ns"), harness_source.index("_resource_snapshot(mx, swap_before)"))
        self.assertLess(harness_source.index("_resource_snapshot(mx, swap_before)"), harness_source.index("_pause_arm(guard, arm_budget[arm])"))
        charge_source = inspect.getsource(worker._charge_arm)
        self.assertIn("guard.record_gpu(seconds)", charge_source)
        self.assertNotIn("guard.required_break()", charge_source)

    def test_budget_policy_is_preregistered(self):
        self.assertEqual(harness.POLICY.duty_cycle_limit, 0.15)
        self.assertEqual(harness.POLICY.continuous_gpu_limit_s, 6.0)
        self.assertEqual(harness.POLICY.gpu_work_limit_s, 120.0)
        self.assertEqual(harness.POLICY.wall_limit_s, 1200.0)
        source = WORKER_PATH.read_text(encoding="utf-8")
        for fragment in ("duty_cycle_limit=0.15", "continuous_gpu_limit_s=6.0", "gpu_work_limit_s=120.0", "wall_limit_s=1200.0"):
            self.assertIn(fragment, source)

    def test_budget_rejection_before_and_after_guard_booking_is_terminal_evidence(self):
        class FakeGuard:
            policy = types.SimpleNamespace(duty_cycle_limit=0.15)

            def __init__(self, books_before_failure):
                self.books_before_failure = books_before_failure
                self.gpu_work_seconds = 0.0

            def record_gpu(self, seconds):
                if self.books_before_failure:
                    self.gpu_work_seconds += seconds
                raise RuntimeError("synthetic BudgetGuard rejection")

        complete, identity, _ = _synthetic_event()
        for books_before_failure in (False, True):
            with self.subTest(books_before_failure=books_before_failure):
                guard = FakeGuard(books_before_failure)
                with self.assertRaises(worker._ChargeRejected) as raised:
                    worker._charge_arm(guard, 2_000_000)
                evidence = raised.exception.evidence
                self.assertEqual(evidence["observed_model_work_ns"], 2_000_000)
                self.assertEqual(evidence["charged_model_work_ns"], 0)
                self.assertFalse(evidence["charge_accepted"])
                self.assertEqual(evidence["guard_gpu_work_before_seconds"], 0.0)
                self.assertEqual(
                    evidence["guard_recorded_model_work_ns"],
                    2_000_000 if books_before_failure else 0,
                )

                event, _, returncode = _synthetic_event(
                    status="resource_or_budget_failed", arms_count=0
                )
                arm = "standard_eager"
                event["arm_budget"][arm] = copy.deepcopy(
                    complete["arm_budget"][arm]
                )
                event["arm_resources"][arm] = copy.deepcopy(
                    complete["arm_resources"][arm]
                )
                event["arm_budget"][arm].update(evidence)
                event["model_work_ns"] = 2_000_000
                event["observed_model_work_ns"] = 2_000_000
                event["charged_model_work_ns"] = 0
                recorded = evidence["guard_recorded_model_work_ns"]
                event["guard_recorded_model_work_ns"] = recorded
                event["budget"] = _budget(
                    2_000_000,
                    charged_model_work_ns=0,
                    guard_recorded_model_work_ns=recorded,
                )
                event["error"] = {
                    "type": "BudgetError",
                    "message": "synthetic BudgetGuard rejection",
                }
                accepted = harness._validate_event(
                    event, 4242, identity, harness.ARM_PERMUTATIONS[0], returncode
                )
                self.assertEqual(accepted["status"], "resource_or_budget_failed")
                self.assertEqual(
                    accepted["arm_budget"][arm]["charged_model_work_ns"], 0
                )
                self.assertEqual(
                    accepted["arm_budget"][arm]["observed_model_work_ns"], 2_000_000
                )

    def test_registered_pacing_uses_13_four_second_blocks_and_stays_within_limits(self):
        from friday_evidence.budget import BudgetGuard

        class Clock:
            def __init__(self):
                self.now = 0.0

            def __call__(self):
                return self.now

            def sleep(self, seconds):
                self.now += seconds

        clock = Clock()
        guard = BudgetGuard(harness.POLICY, clock=clock, sleeper=clock.sleep)
        evidence = worker._arm_budget_evidence(6_000_000_000)
        self.assertEqual(worker.MIN_REQUIRED_BREAK_BLOCKS, 13)
        self.assertEqual(evidence["required_break_blocks"], 13)
        self.assertAlmostEqual(evidence["duty_formula_break_seconds"], 34.0)

        for _ in range(3):
            clock.now += 6.0
            guard.record_gpu(6.0)
            for _ in range(evidence["required_break_blocks"]):
                guard.required_break()

        summary = guard.summary()
        self.assertEqual(summary["gpu_work_seconds"], 18.0)
        self.assertEqual(summary["max_continuous_gpu_seconds"], 6.0)
        self.assertLessEqual(summary["gpu_work_seconds"], 120.0)
        self.assertLessEqual(summary["max_continuous_gpu_seconds"], 6.0)
        self.assertLess(summary["wall_seconds"], 300.0)
        self.assertLess(summary["wall_seconds"], 1200.0)

    def test_complete_synthetic_worker_event_is_accepted(self):
        event, identity, returncode = _synthetic_event()
        self.assertEqual(set(event), EXPECTED_EVENT_KEYS)
        accepted = harness._validate_event(event, 4242, identity, harness.ARM_PERMUTATIONS[0], returncode)
        self.assertIs(accepted, event)
        self.assertEqual(len(event["arms"]), 3)
        self.assertTrue(all(len(value["tokens"]) == 32 for value in event["arms"].values()))
        self.assertTrue(all(len(value["intertoken_ns"]) == 31 for value in event["arms"].values()))

    def test_candidate_not_runnable_failed_arm_is_recorded_only_in_budget_and_resources(self):
        complete, identity, _ = _synthetic_event()
        event, _, returncode = _synthetic_event(status="candidate_not_runnable", arms_count=0)
        failed_arm = "fixed_compiled"
        event["arm_budget"][failed_arm] = copy.deepcopy(complete["arm_budget"][failed_arm])
        event["arm_resources"][failed_arm] = copy.deepcopy(complete["arm_resources"][failed_arm])
        event["model_work_ns"] = 1_000_000
        event["observed_model_work_ns"] = 1_000_000
        event["charged_model_work_ns"] = 1_000_000
        event["guard_recorded_model_work_ns"] = 1_000_000
        event["budget"] = _budget(1_000_000)
        event["error"] = {"type": "CandidateNotRunnable", "message": "compile API unavailable"}
        accepted = harness._validate_event(
            event, 4242, identity, harness.ARM_PERMUTATIONS[0], returncode
        )
        self.assertEqual(accepted["status"], "candidate_not_runnable")
        self.assertNotIn(failed_arm, accepted["arms"])
        self.assertIn(failed_arm, accepted["arm_budget"])
        self.assertIn(failed_arm, accepted["arm_resources"])

        missing_resources = copy.deepcopy(event)
        missing_resources["arm_resources"].pop(failed_arm)
        with self.assertRaises(harness.WorkerError):
            harness._validate_event(
                missing_resources, 4242, identity, harness.ARM_PERMUTATIONS[0], returncode
            )

    def test_budget_rejection_preserves_observed_vs_charged_time(self):
        complete, identity, _ = _synthetic_event()
        event, _, returncode = _synthetic_event(
            status="resource_or_budget_failed", arms_count=0
        )
        failed_arm = "standard_eager"
        event["arm_budget"][failed_arm] = copy.deepcopy(
            complete["arm_budget"][failed_arm]
        )
        event["arm_resources"][failed_arm] = copy.deepcopy(
            complete["arm_resources"][failed_arm]
        )
        observed = 2_000_000
        event["arm_budget"][failed_arm].update(
            observed_model_work_ns=observed,
            charged_model_work_ns=0,
            charge_accepted=False,
            guard_gpu_work_before_seconds=0.0,
            guard_gpu_work_after_seconds=0.0,
            guard_recorded_model_work_ns=0,
            duty_formula_break_seconds=(observed / 1e9) * 0.85 / 0.15,
        )
        event["model_work_ns"] = observed
        event["observed_model_work_ns"] = observed
        event["charged_model_work_ns"] = 0
        event["guard_recorded_model_work_ns"] = 0
        event["budget"] = _budget(
            observed, charged_model_work_ns=0, guard_recorded_model_work_ns=0
        )
        event["error"] = {
            "type": "BudgetError",
            "message": "continuous GPU work budget exceeded",
        }
        accepted = harness._validate_event(
            event, 4242, identity, harness.ARM_PERMUTATIONS[0], returncode
        )
        record = accepted["arm_budget"][failed_arm]
        self.assertEqual(record["observed_model_work_ns"], observed)
        self.assertEqual(record["charged_model_work_ns"], 0)
        self.assertFalse(record["charge_accepted"])
        self.assertEqual(accepted["charged_model_work_ns"], 0)

        not_resource_terminal = copy.deepcopy(event)
        not_resource_terminal["status"] = "candidate_not_runnable"
        with self.assertRaises(harness.WorkerError):
            harness._validate_event(
                not_resource_terminal,
                4242,
                identity,
                harness.ARM_PERMUTATIONS[0],
                returncode,
            )

    def test_prompt_sha_is_sealed_before_marker_and_mutation_fails(self):
        import inspect

        execute_source = inspect.getsource(harness.execute)
        self.assertLess(execute_source.index("_preflight(run_id)"), execute_source.index("_exclusive_json(ATTEMPT_PATH"))
        provenance = harness._provenance("git", "", "ac_power", _identity())
        self.assertEqual(provenance["prompt_sha256"], worker.PROMPT_SHA256)

        event, identity, returncode = _synthetic_event()
        for field in ("prompt_sha256", "rendered_prompt_sha256"):
            with self.subTest(field=field):
                mutated = copy.deepcopy(event)
                mutated[field] = "f" * 64
                with self.assertRaises(harness.WorkerError):
                    harness._validate_event(
                        mutated, 4242, identity, harness.ARM_PERMUTATIONS[0], returncode
                    )

    def test_per_arm_prompt_prompt_token_and_rendered_hashes_are_complete_and_strict(self):
        event, identity, returncode = _synthetic_event()
        for arm in ARM_NAMES:
            self.assertEqual(event["arms"][arm]["prompt_sha256"], worker.PROMPT_SHA256)
            self.assertEqual(
                event["arms"][arm]["prompt_token_sha256"],
                _sha256_bytes(worker._canonical_json(event["prompt_token_ids"])),
            )
            self.assertEqual(
                event["arms"][arm]["rendered_prompt_sha256"],
                event["rendered_prompt_sha256"],
            )
        accepted = harness._validate_event(
            event, 4242, identity, harness.ARM_PERMUTATIONS[0], returncode
        )

        for field in (
            "prompt_sha256",
            "prompt_token_sha256",
            "rendered_prompt_sha256",
        ):
            with self.subTest(field=field):
                mutated = copy.deepcopy(event)
                mutated["arms"]["fixed_eager"][field] = "0" * 64
                with self.assertRaises(harness.WorkerError):
                    harness._validate_event(
                        mutated, 4242, identity, harness.ARM_PERMUTATIONS[0], returncode
                    )
                missing = copy.deepcopy(event)
                missing["arms"]["fixed_eager"].pop(field)
                with self.assertRaises(harness.WorkerError):
                    harness._validate_event(
                        missing, 4242, identity, harness.ARM_PERMUTATIONS[0], returncode
                    )

    def test_event_schema_rejects_identity_and_shape_variations(self):
        base, identity, returncode = _synthetic_event()
        mutations = {
            "extra": lambda item: item.update(extra=True),
            "missing": lambda item: item.pop("arms"),
            "pid": lambda item: item.update(pid=4243),
            "order": lambda item: item.update(arm_order=list(harness.ARM_PERMUTATIONS[1])),
            "prompt_hash": lambda item: item.update(prompt_sha256="c" * 64),
            "load_count": lambda item: item.update(load_count=2),
            "timings": lambda item: item["arms"]["standard_eager"]["intertoken_ns"].pop(),
            "sum": lambda item: item.update(model_work_ns=3_000_001),
            "duty": lambda item: item["budget"].update(duty_cycle_limit=0.2),
            "continuous": lambda item: item["budget"].update(max_continuous_gpu_seconds=6.1),
            "arm_continuous": lambda item: item["arm_budget"]["standard_eager"].update(charged_model_work_ns=6_000_000_001),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                event = copy.deepcopy(base)
                mutate(event)
                with self.assertRaises(harness.WorkerError):
                    harness._validate_event(event, 4242, identity, harness.ARM_PERMUTATIONS[0], returncode)

    def test_arm_completion_and_compile_fields_are_strict(self):
        base, identity, returncode = _synthetic_event()
        accepted = harness._validate_event(
            base, 4242, identity, harness.ARM_PERMUTATIONS[0], returncode
        )
        self.assertEqual(accepted["arms"]["fixed_compiled"]["finish_reason"], "fixed_steps")
        self.assertEqual(accepted["arms"]["fixed_compiled"]["decode_forwards"], 31)
        self.assertEqual(accepted["arms"]["fixed_compiled"]["warmup_forwards"], 8)
        self.assertIsNone(accepted["arms"]["standard_eager"]["compile_cold_ns"])
        self.assertIsInstance(accepted["arms"]["fixed_compiled"]["compile_cold_ns"], int)

        mutations = {
            "finish_reason": lambda item: item["arms"]["fixed_compiled"].update(finish_reason="stop"),
            "decode_forwards": lambda item: item["arms"]["fixed_compiled"].update(decode_forwards=30),
            "warmup_forwards": lambda item: item["arms"]["fixed_compiled"].update(warmup_forwards=7),
            "warmup_decode_length": lambda item: item["arms"]["fixed_compiled"]["warmup_decode_forward_ns"].pop(),
            "compiled_cold_null": lambda item: item["arms"]["fixed_compiled"].update(compile_cold_ns=None),
            "eager_cold_not_null": lambda item: item["arms"]["standard_eager"].update(compile_cold_ns=0),
            "compiled_wrapper_zero": lambda item: item["arms"]["fixed_compiled"].update(compile_wrapper_ns=0),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                event = copy.deepcopy(base)
                mutate(event)
                with self.assertRaises(harness.WorkerError):
                    harness._validate_event(
                        event, 4242, identity, harness.ARM_PERMUTATIONS[0], returncode
                    )

        rejected_charge = copy.deepcopy(base)
        rejected_charge["arm_budget"]["standard_eager"]["charged_model_work_ns"] = 0
        rejected_charge["arm_budget"]["standard_eager"]["charge_accepted"] = False
        rejected_charge["charged_model_work_ns"] = 2_000_000
        with self.assertRaises(harness.WorkerError):
            harness._validate_event(
                rejected_charge, 4242, identity, harness.ARM_PERMUTATIONS[0], returncode
            )

    def test_terminal_partial_events_are_validated_and_classified(self):
        for status, arms_count in (("candidate_not_runnable", 0), ("correctness_failed", 1), ("resource_or_budget_failed", 1), ("error", 0)):
            with self.subTest(status=status):
                event, identity, returncode = _synthetic_event(status=status, arms_count=arms_count)
                accepted = harness._validate_event(event, 4242, identity, tuple(event["arm_order"]), returncode)
                self.assertEqual(accepted["status"], status)
        event, identity, returncode = _synthetic_event(status="candidate_not_runnable", arms_count=0)
        with self.assertRaises(harness.WorkerError):
            harness._validate_event(event, 4242, identity, tuple(event["arm_order"]), 0)
        with self.assertRaises(harness.WorkerError):
            harness._validate_event(_synthetic_event()[0], 4242, _identity(), harness.ARM_PERMUTATIONS[0], 1)

    def test_strict_worker_parser_rejects_duplicate_nan_multiline_and_oversize(self):
        cases = [
            b'{"a":1,"a":2}\n',
            b'{"a":NaN}\n',
            b'{}\n{}\n',
            b"x" * (harness.MAX_EVENT_BYTES + 1),
        ]
        for payload in cases:
            with self.subTest(payload=payload[:20]):
                with self.assertRaises(harness.WorkerError):
                    harness._decode_event(payload)
        self.assertEqual(harness._decode_event(b'{"a":1}\n'), {"a": 1})

    def test_default_harness_is_exit_78_and_does_not_create_or_change_evidence(self):
        paths = (harness.RESULT_PATH, harness.ATTEMPT_PATH)
        before = {(path, _sha256_path(path) if path.is_file() and not path.is_symlink() else None, path.is_symlink(), path.exists()) for path in paths}
        completed = subprocess.run([str(PYTHON), str(HARNESS_PATH)], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        self.assertEqual(completed.returncode, 78)
        self.assertIn(b"not_released", completed.stdout)
        after = {(path, _sha256_path(path) if path.is_file() and not path.is_symlink() else None, path.is_symlink(), path.exists()) for path in paths}
        self.assertEqual(before, after)

    def test_direct_unauthorised_worker_does_not_load_mlx_or_create_evidence(self):
        paths = (harness.RESULT_PATH, harness.ATTEMPT_PATH)
        before = {(path, _sha256_path(path) if path.is_file() and not path.is_symlink() else None, path.is_symlink(), path.exists()) for path in paths}
        completed = subprocess.run([str(PYTHON), str(WORKER_PATH), "--worker", "--model-key", "bad"], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        self.assertEqual(completed.returncode, 2)
        self.assertIn(b"authorization failed", completed.stdout)
        after = {(path, _sha256_path(path) if path.is_file() and not path.is_symlink() else None, path.is_symlink(), path.exists()) for path in paths}
        self.assertEqual(before, after)
        top_level = ast.parse(WORKER_PATH.read_text(encoding="utf-8")).body
        self.assertFalse(any(isinstance(node, (ast.Import, ast.ImportFrom)) and "mlx" in ast.unparse(node) for node in top_level))

    def test_all_offline_self_checks_pass(self):
        commands = (
            (WORKER_PATH, "--self-check"),
            (HARNESS_PATH, "--self-check"),
            (DASHBOARD_PATH, "--self-check"),
        )
        for path, flag in commands:
            with self.subTest(path=path.name):
                completed = subprocess.run([str(PYTHON), str(path), flag], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                self.assertEqual(completed.returncode, 0, completed.stderr.decode())
                if path != DASHBOARD_PATH:
                    self.assertIn(b"self_check", completed.stdout)

    def test_parser_and_worker_gates_can_be_checked_without_hardware(self):
        identity = _identity()
        order = harness.ARM_PERMUTATIONS[0]
        with mock.patch.dict(os.environ, {"PYTHONPATH": "unsafe", "PYTHONHOME": "unsafe", "HF_HUB_OFFLINE": "0"}, clear=True):
            environment = harness._environment(identity, 1, order)
        self.assertNotIn("PYTHONPATH", environment)
        self.assertNotIn("PYTHONHOME", environment)
        for name, value in harness.WORKER_ENVIRONMENT.items():
            self.assertEqual(environment[name], value)
        self.assertEqual(environment["FRIDAY_MATMUL_ARM_ORDER"], json.dumps(list(order), separators=(",", ":")))
        self.assertEqual(environment["FRIDAY_MATMUL_SNAPSHOT_SHA256"], identity["snapshot_sha256"])

        authorise_env = {
            "FRIDAY_MATMUL_PARENT_PID": str(os.getpid()),
            "FRIDAY_MATMUL_RUN_ID": worker.RUN_ID,
            "FRIDAY_MATMUL_MODEL_KEY": "4b",
            "FRIDAY_MATMUL_NONCE": "cycle16-fixed-cache-v1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "PYTHONNOUSERSITE": "1",
        }
        with mock.patch.dict(os.environ, authorise_env, clear=True), mock.patch.object(os, "getppid", return_value=os.getpid()):
            worker._authorise("4b")
        for missing in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PYTHONNOUSERSITE"):
            reduced = dict(authorise_env)
            reduced.pop(missing)
            with self.subTest(missing=missing), mock.patch.dict(os.environ, reduced, clear=True), mock.patch.object(os, "getppid", return_value=os.getpid()):
                with self.assertRaises(worker.WorkerError):
                    worker._authorise("4b")

    def test_hardware_target_gates_are_closed_with_fake_runtime(self):
        fake_core = types.ModuleType("mlx.core")
        fake_core.default_device = lambda: "Device(gpu, 0)"
        fake_mlx = types.ModuleType("mlx")
        fake_mlx.core = fake_core
        package_versions = {"mlx": "0.32.0", "mlx-lm": "0.31.3"}

        def fake_version(name: str) -> str:
            if name in package_versions:
                return package_versions[name]
            return "1.0.0"

        def fake_sysctl(name: str) -> str | None:
            return {"machdep.cpu.brand_string": "Apple M1 Max", "hw.memsize": str(32 * 1024**3)}.get(name)

        with mock.patch("platform.machine", return_value="arm64"), mock.patch.object(harness, "_sysctl", side_effect=fake_sysctl), mock.patch.object(harness.importlib.metadata, "version", side_effect=fake_version), mock.patch.dict(sys.modules, {"mlx": fake_mlx, "mlx.core": fake_core}):
            harness._require_target()
            fake_core.default_device = lambda: "Device(cpu, 0)"
            with self.assertRaises(harness.StudyError):
                harness._require_target()
        with mock.patch("platform.machine", return_value="x86_64"), mock.patch.object(harness, "_sysctl", side_effect=fake_sysctl):
            with self.assertRaises(harness.StudyError):
                harness._require_target()

    def test_preflight_marker_result_dirty_and_snapshot_gates(self):
        identity = _identity()
        snapshot = types.SimpleNamespace(revision=harness.MODEL_REVISION)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / "results.json"
            marker = root / "attempt.json"
            patches = mock.patch.multiple(
                harness,
                RESULT_PATH=result,
                ATTEMPT_PATH=marker,
                _clean_worktree=mock.DEFAULT,
                _require_target=mock.DEFAULT,
                _snapshot_identity=mock.DEFAULT,
                _swap_used_bytes=mock.DEFAULT,
            )
            with patches as values:
                values["_clean_worktree"].return_value = ("git", "")
                values["_require_target"].return_value = None
                values["_snapshot_identity"].return_value = identity
                values["_swap_used_bytes"].return_value = 0
                with mock.patch.object(harness, "require_ac_power", return_value="ac_power"), mock.patch.object(harness, "resolve_local_model_snapshot", return_value=snapshot), mock.patch.object(harness.sys, "executable", str(PYTHON)):
                    self.assertEqual(harness._preflight(harness.RUN_ID)[2], "ac_power")
                    result.write_text("x", encoding="utf-8")
                    with self.assertRaises(harness.StudyError):
                        harness._preflight(harness.RUN_ID)
                    result.unlink()
                    marker.symlink_to(root)
                    with self.assertRaises(harness.StudyError):
                        harness._preflight(harness.RUN_ID)
                    marker.unlink()
                    values["_clean_worktree"].side_effect = harness.StudyError("dirty")
                    with self.assertRaisesRegex(harness.StudyError, "dirty"):
                        harness._preflight(harness.RUN_ID)
                    values["_clean_worktree"].side_effect = None
                    snapshot.revision = "wrong"
                    with self.assertRaises(harness.StudyError):
                        harness._preflight(harness.RUN_ID)

    def test_atomic_result_and_fail_safe_preserve_partial_runs(self):
        worker_event = {
            "event": "error",
            "error_type": "RuntimeError",
            "message": "bounded worker failure",
            "model_key": "4b",
            "block": 1,
        }
        state = {
            "runs": [{"block": 1}],
            "worker_events": [worker_event],
            "provenance": {"x": 1},
        }
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "results.json"
            with mock.patch.object(harness, "RESULT_PATH", result):
                actual = harness._atomic_result
                calls = {"count": 0}

                def flaky(value):
                    calls["count"] += 1
                    if calls["count"] == 1:
                        raise OSError("synthetic first write failure")
                    return actual(value)

                with mock.patch.object(harness, "_atomic_result", side_effect=flaky):
                    harness._write_fail_safe(state)
                written = json.loads(result.read_text(encoding="utf-8"))
                self.assertTrue(written["partial_result"])
                self.assertEqual(written["decision"], "resource_or_budget_failed")
                self.assertEqual(written["runs"], state["runs"])
                self.assertEqual(written["worker_events"], state["worker_events"])
                self.assertEqual(stat.S_IMODE(result.stat().st_mode), 0o644)
                with self.assertRaises(harness.StudyError):
                    harness._atomic_result({})

    def test_stdout_cap_and_process_helpers_are_fail_closed(self):
        class Process:
            pid = 999_999

        result: dict[str, object] = {}
        with mock.patch.object(harness, "_terminate") as terminate:
            harness._read_capped(io.BytesIO(b"x" * 11), Process(), 10, result, "payload", None)
        self.assertTrue(result["overflow"])
        terminate.assert_called_once()

        class EmptyProcess:
            pid = 999_998

            def terminate(self):
                self.terminated = True

            def wait(self, timeout=None):
                del timeout
                return 0

        process = EmptyProcess()
        harness._terminate(process)
        self.assertTrue(getattr(process, "terminated", False))

    def test_block_timeout_is_capped_by_remaining_parent_wall_deadline(self):
        class DeadlineProcess:
            pid = 999_999_999

            def __init__(self):
                self.stdout = io.BytesIO()
                self.stderr = io.BytesIO()
                self.returncode = None
                self.timeouts: list[float | None] = []
                self.terminated = False

            def wait(self, timeout=None):
                self.timeouts.append(timeout)
                if len(self.timeouts) == 1:
                    raise subprocess.TimeoutExpired("synthetic-worker", timeout)
                self.returncode = -15
                return self.returncode

            def terminate(self):
                self.terminated = True

            def kill(self):
                self.terminated = True

        process = DeadlineProcess()
        with mock.patch.object(harness, "require_ac_power", return_value="ac_power"), \
                mock.patch.object(harness, "_swap_used_bytes", return_value=0), \
                mock.patch.object(harness.subprocess, "Popen", return_value=process), \
                mock.patch.object(harness.time, "monotonic", return_value=1000.0):
            with self.assertRaises(harness.WorkerError):
                harness._run_block(
                    1,
                    harness.ARM_PERMUTATIONS[0],
                    _identity(),
                    # The caller passes a deadline already reduced by the
                    # 15-second finalisation reserve.
                    deadline_monotonic=1000.25,
                )
        self.assertGreaterEqual(len(process.timeouts), 1)
        self.assertAlmostEqual(process.timeouts[0], 0.25, places=6)
        self.assertTrue(process.terminated)

        execute_source = __import__("inspect").getsource(harness.execute)
        self.assertIn(
            "worker_deadline = hard_deadline - FINALIZATION_RESERVE_SECONDS",
            execute_source,
        )
        self.assertIn("_run_block(block, order, identity, worker_deadline)", execute_source)

        class ExpiredProcess:
            pid = 999_999_998
            waited: list[float | None] = []

            def terminate(self):
                return None

            def kill(self):
                return None

            def wait(self, timeout=None):
                self.waited.append(timeout)

        expired = ExpiredProcess()
        with mock.patch.object(harness.time, "monotonic", return_value=2000.0):
            harness._terminate(expired, deadline_monotonic=2000.0)
        self.assertEqual(expired.waited, [])

    def test_minimal_worker_error_event_is_bounded_before_fail_safe_storage(self):
        raw = {
            "event": "error",
            "error_type": "E" * 1000,
            "message": "M" * 5000,
            "model_key": "K" * 1000,
            "ignored": "must not be copied",
        }

        class ErrorProcess:
            pid = 999_999_998

            def __init__(self):
                self.stdout = io.BytesIO(
                    json.dumps(raw, separators=(",", ":")).encode() + b"\n"
                )
                self.stderr = io.BytesIO()
                self.returncode = 0

            def wait(self, timeout=None):
                del timeout
                return self.returncode

            def terminate(self):
                return None

            def kill(self):
                return None

        process = ErrorProcess()
        with mock.patch.object(harness, "require_ac_power", return_value="ac_power"), \
                mock.patch.object(harness, "_swap_used_bytes", return_value=0), \
                mock.patch.object(harness.subprocess, "Popen", return_value=process):
            with self.assertRaises(harness.WorkerEventError) as raised:
                harness._run_block(1, harness.ARM_PERMUTATIONS[0], _identity())
        event = raised.exception.event
        self.assertEqual(
            set(event), {"event", "error_type", "message", "model_key", "block"}
        )
        self.assertLessEqual(len(event["error_type"]), 120)
        self.assertLessEqual(len(event["message"]), 500)
        self.assertLessEqual(len(event["model_key"]), 32)
        self.assertEqual(event["block"], 1)

    def test_reader_joins_are_bounded_by_the_same_hard_deadline(self):
        class Process:
            pid = 999_999_997
            returncode = 0

            def __init__(self):
                self.stdout = io.BytesIO()
                self.stderr = io.BytesIO()

            def wait(self, timeout=None):
                del timeout
                return 0

            def terminate(self):
                return None

            def kill(self):
                return None

        class Thread:
            joins: list[float | None] = []

            def __init__(self, *, target, args, daemon):
                del target, args, daemon

            def start(self):
                return None

            def join(self, timeout=None):
                self.joins.append(timeout)

            def is_alive(self):
                return False

        process = Process()
        with mock.patch.object(harness, "require_ac_power", return_value="ac_power"), \
                mock.patch.object(harness, "_swap_used_bytes", return_value=0), \
                mock.patch.object(harness.subprocess, "Popen", return_value=process), \
                mock.patch.object(harness.threading, "Thread", Thread), \
                mock.patch.object(harness.time, "monotonic", side_effect=[3000.0, 3010.0, 3030.0]):
            with self.assertRaises(harness.WorkerError):
                harness._run_block(
                    1,
                    harness.ARM_PERMUTATIONS[0],
                    _identity(),
                    deadline_monotonic=3030.0,
                )
        self.assertEqual(Thread.joins, [20.0, 0.0])
        self.assertTrue(all(timeout is not None and timeout >= 0 for timeout in Thread.joins))

    def test_paired_ratios_include_both_comparisons_all_ci_paths_and_no_outlier_filter(self):
        runs = _paired_runs()
        paired = harness._paired(runs)
        self.assertTrue(paired["complete"])
        for metric in ("decode_total", "intertoken_p50", "intertoken_p95", "intertoken_p99"):
            self.assertEqual(set(paired["ratios"][metric]), {"fixed_compiled_div_standard_eager", "fixed_compiled_div_fixed_eager"})
            for comparison in paired["ratios"][metric].values():
                self.assertEqual(comparison["bootstrap_95_ci"]["resamples"], 10_000)
                self.assertEqual(comparison["bootstrap_95_ci"]["seed"], 20260824)
        sample = [0.9, 0.9, 0.9, 0.9, 0.9, 1.2]
        first = harness._bootstrap(sample)
        second = harness._bootstrap(sample)
        self.assertEqual(first, second)
        self.assertEqual(first["resamples"], 10_000)
        self.assertEqual(len(sample), 6)

    def test_arm_stats_report_median_and_mad_for_timing_rate_wall_and_resources(self):
        stats = harness._arm_stats(_paired_runs(), "standard_eager")
        required_metrics = {
            "ttft_seconds",
            "prefill_seconds",
            "model_work_seconds",
            "arm_wall_seconds",
            "process_wall_seconds",
            "token_rate",
            "rss_peak_bytes",
            "mlx_peak_bytes",
            "swap_delta_bytes",
        }
        self.assertTrue(required_metrics.issubset(stats["metrics"]))
        for name in required_metrics:
            with self.subTest(metric=name):
                metric = stats["metrics"][name]
                self.assertEqual(metric["median"], metric["values"][0])
                self.assertEqual(metric["mad"], 0.0)
                self.assertEqual(len(metric["values"]), 6)
        self.assertEqual(stats["metrics"]["ttft_seconds"]["median"], 120 / 1e9)
        self.assertEqual(stats["metrics"]["process_wall_seconds"]["median"], 3000 / 1e9)
        self.assertEqual(stats["peak_rss_bytes"], 4000)
        self.assertEqual(stats["peak_mlx_bytes"], 5000)
        self.assertEqual(stats["swap_deltas_bytes"], [0] * 6)

    def test_decision_table_uses_only_decode_total(self):
        def paired(dec_std, upper_std, dec_eager, upper_eager, intertoken=2.0, lower_std=None, lower_eager=None):
            comparison = lambda median, upper, lower: {"median": median, "bootstrap_95_ci": {"lower": lower, "upper": upper}}
            if lower_std is None:
                lower_std = min(dec_std, 0.9)
            if lower_eager is None:
                lower_eager = min(dec_eager, 0.9)
            return {
                "complete": True,
                "ratios": {
                    "decode_total": {
                        "fixed_compiled_div_standard_eager": comparison(dec_std, upper_std, lower_std),
                        "fixed_compiled_div_fixed_eager": comparison(dec_eager, upper_eager, lower_eager),
                    },
                    "intertoken_p50": {
                        "fixed_compiled_div_standard_eager": comparison(intertoken, intertoken, 0.5),
                        "fixed_compiled_div_fixed_eager": comparison(intertoken, intertoken, 0.5),
                    },
                    "intertoken_p95": {},
                    "intertoken_p99": {},
                },
            }

        gates = dict(resource_pass=True, budget_pass=True, correctness_pass=True, candidate_runnable=True)
        self.assertEqual(harness.decision_for(**gates, paired=paired(.90, .99, .91, .995, 2.0)), "runtime_compile_wins_exact_scope")
        self.assertEqual(harness.decision_for(**gates, paired=paired(1.01, 1.1, .90, .99)), "compile_gain_no_system_gain")
        self.assertEqual(harness.decision_for(**gates, paired=paired(.90, .99, 1.01, 1.1)), "fixed_cache_gain_not_compile_gain")
        self.assertEqual(harness.decision_for(**gates, paired=paired(.94, 1.01, .94, 1.01)), "no_clear_speedup_baseline_retained")
        self.assertEqual(harness.decision_for(**gates, paired=paired(1.10, 1.2, 1.01, 1.1, lower_std=1.01, lower_eager=1.001)), "compile_regression_baseline_retained")
        self.assertEqual(harness.decision_for(**gates, paired=paired(1.10, 1.2, 1.01, 1.1, lower_std=0.9, lower_eager=0.9)), "no_clear_speedup_baseline_retained")
        self.assertEqual(harness.decision_for(**gates, paired={"complete": False}), "no_clear_speedup_baseline_retained")
        for key, expected in (("resource_pass", "resource_or_budget_failed"), ("budget_pass", "resource_or_budget_failed"), ("correctness_pass", "correctness_failed"), ("candidate_runnable", "candidate_not_runnable")):
            failed = dict(gates)
            failed[key] = False
            self.assertEqual(harness.decision_for(**failed, paired=paired(.9, .9, .9, .9)), expected)

    def test_terminal_resource_status_forces_budget_gate_without_reclassifying_other_terminals(self):
        identity = _identity()
        cases = (
            ("resource_or_budget_failed", 1, "resource_or_budget_failed", False),
            ("candidate_not_runnable", 0, "candidate_not_runnable", True),
            ("correctness_failed", 1, "correctness_failed", True),
        )
        for status, arms_count, expected_decision, expected_budget_pass in cases:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                captured: dict[str, object] = {}
                run_template, _, _ = _synthetic_event(
                    status=status, arms_count=arms_count
                )
                run_template.update(
                    block=1,
                    power_source="ac_power",
                    process_wall_ns=2_000_000,
                    swap_before_bytes=10,
                    swap_after_bytes=10,
                    swap_delta_bytes=0,
                    abort_reason=None,
                )

                def fake_run_block(block, order, bound_identity, deadline):
                    del deadline
                    run = copy.deepcopy(run_template)
                    run["block"] = block
                    run["arm_order"] = list(order)
                    run["snapshot_path"] = bound_identity["snapshot_path"]
                    return run

                with mock.patch.multiple(
                    harness,
                    ATTEMPT_DIR=root / "attempt-dir",
                    ATTEMPT_PATH=root / "attempt-dir" / "attempt.json",
                    _preflight=mock.DEFAULT,
                    _provenance=mock.DEFAULT,
                    _run_block=fake_run_block,
                    _write_fail_safe=lambda state: captured.update(state),
                    _snapshot_identity=mock.DEFAULT,
                    _swap_used_bytes=mock.DEFAULT,
                    resolve_local_model_snapshot=mock.DEFAULT,
                ) as values:
                    values["_preflight"].return_value = (
                        "git", "", "ac_power", identity, 0
                    )
                    values["_provenance"].return_value = {"synthetic": True}
                    values["_snapshot_identity"].return_value = identity
                    values["_swap_used_bytes"].return_value = 0
                    values["resolve_local_model_snapshot"].return_value = object()
                    result = harness.execute(harness.RUN_ID)

                self.assertEqual(result["decision"], expected_decision)
                self.assertEqual(
                    result["gates"]["budget_pass"], expected_budget_pass
                )
                self.assertEqual(captured["gates"]["budget_pass"], expected_budget_pass)

    def test_derived_metrics_use_prefill_conversion_decode_and_cold_compile(self):
        derived = harness._derived_metrics(_paired_runs(standard=20, fixed_eager=20, compiled=10))
        self.assertTrue(derived["complete"])
        self.assertTrue(derived["calculated_only"])
        self.assertAlmostEqual(derived["warmed_decode_ratio_values"][0], (100 + 5 + 31 * 10) / (100 + 31 * 20))
        self.assertAlmostEqual(derived["cold_decode_ratio_values"][0], (100 + 5 + 31 * 10 + 20 + 30) / (100 + 31 * 20))
        self.assertAlmostEqual(derived["cold_setup_seconds"][0], 50 / 1e9)
        self.assertAlmostEqual(derived["break_even_decode_forwards"][0], 5.0)
        self.assertIn("calculated upper-bound", derived["method"])

    def test_dashboard_projection_is_strict_read_only_and_redacts_raw_fields(self):
        value = _dashboard_result()
        projected = dashboard._project_result(value)
        forbidden = {"text", "tokens", "prompt", "rendered_prompt_b64", "snapshot_path", "error", "path", "base64"}

        def walk(node):
            if isinstance(node, dict):
                for key, item in node.items():
                    self.assertNotIn(key.lower(), forbidden)
                    yield from walk(item)
            elif isinstance(node, list):
                for item in node:
                    yield from walk(item)
            elif isinstance(node, str):
                self.assertNotIn("raw-model-sentinel", node)

        list(walk(projected))
        self.assertNotIn("innerHTML", dashboard.HTML.decode("utf-8"))
        self.assertIn("textContent", dashboard.HTML.decode("utf-8"))
        self.assertEqual(projected["formal_claim"], False)
        self.assertTrue(projected["read_only"])

    def test_dashboard_accepts_resource_or_budget_failed_as_safe_terminal_decision(self):
        value = _dashboard_result()
        value["decision"] = "resource_or_budget_failed"
        projected = dashboard._project_result(value)
        self.assertEqual(projected["decision"], "resource_or_budget_failed")
        self.assertFalse(projected["formal_claim"])
        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory) / "results.json"
            history_path = Path(directory) / "history.json"
            result_path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
            snapshot = dashboard.snapshot(result_path, history_path)
        self.assertEqual(snapshot["status"], "available")
        self.assertEqual(snapshot["decision"], "resource_or_budget_failed")

    def test_dashboard_http_methods_hosts_headers_and_hashes(self):
        value = _dashboard_result()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = root / "results.json"
            history_path = root / "history.json"
            result_path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
            history_path.write_text("{}", encoding="utf-8")
            before = (_sha256_path(result_path), _sha256_path(history_path))
            server = dashboard.ThreadingHTTPServer(("127.0.0.1", 0), dashboard._handler(result_path, history_path))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            port = server.server_address[1]
            host = f"127.0.0.1:{port}"

            def request(method, path, headers=None):
                connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                connection.request(method, path, headers={"Host": host, **(headers or {})})
                response = connection.getresponse()
                body = response.read()
                connection.close()
                return response, body

            try:
                for path in ("/", "/api/snapshot"):
                    response, body = request("GET", path)
                    self.assertEqual(response.status, 200)
                    self.assertTrue(body)
                    self.assertEqual(response.getheader("Cache-Control"), "no-store")
                    response, head_body = request("HEAD", path)
                    self.assertEqual(response.status, 200)
                    self.assertEqual(head_body, b"")
                    self.assertEqual(response.getheader("Content-Length"), str(len(body)))
                for method in ("POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE"):
                    response, body = request(method, "/api/snapshot")
                    self.assertEqual(response.status, 405)
                    self.assertEqual(response.getheader("Allow"), "GET, HEAD")
                    self.assertEqual(body, b"")
                response, _ = request("GET", "/api/snapshot", {"Host": "evil.example"})
                self.assertEqual(response.status, 421)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
            self.assertEqual(before, (_sha256_path(result_path), _sha256_path(history_path)))

    def test_dashboard_missing_result_is_safe_and_offline_helpers_do_not_touch_cycle15(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.json"
            no_history = Path(directory) / "no-history.json"
            snapshot = dashboard.snapshot(missing, no_history)
        self.assertEqual(snapshot["status"], "not_available")
        self.assertFalse(snapshot["formal_claim"])
        cycle15 = ROOT / "experiments" / "dual_model_planner" / "results.json"
        before = _sha256_path(cycle15) if cycle15.is_file() else None
        dashboard.snapshot(Path("/definitely/missing/cycle16.json"), Path("/definitely/missing/history.json"))
        after = _sha256_path(cycle15) if cycle15.is_file() else None
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
