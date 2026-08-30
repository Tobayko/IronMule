"""No-hardware protocol tests for the IronMule stage worker."""

from __future__ import annotations

import hashlib
import importlib
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

from friday_optimizer import ironmule_stage_worker as worker
from friday_optimizer.evaluator import MetricSample


def _source_digest(paths: tuple[str, ...], hashes: dict[str, str], registry: str) -> str:
    digest = hashlib.sha256(b"execution-registry\0" + registry.encode() + b"\0")
    for path in paths:
        digest.update(path.encode() + b"\0" + bytes.fromhex(hashes[path]))
    return digest.hexdigest()


def _profile(*, complete: bool = True) -> dict:
    confirmation = {
        "confirmed": True,
        "token_identity": True,
        "pair_count": 3,
        "pairs": [{"id": 1}, {"id": 2}, {"id": 3}],
        "ratio": {"median_ratio": 0.9, "ci_low": 0.85, "ci_high": 0.95},
        "resources": {
            "ttft_ms": 1.0, "decode_tokens_per_second": 10.0,
            "peak_memory_bytes": 100, "peak_rss_bytes": 100,
            "swap_delta_bytes": 0, "resource_gate_passed": True,
        },
    }
    result = {
        "tokens": [1, 2, 3], "token_count": 3, "tuned_decode_ns": 300_000_000,
        "ttft_ms": 1.0,
        "knobs": dict(worker.TUNE_SEARCH_CONTRACT["knobs_defaults"]),
        "confirmation": confirmation, "gain": 0.1,
    }
    if not complete:
        result.pop("tokens")
        result["confirmation"] = None
    return result


class StageWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "stage"
        (self.root / "ironmule").mkdir(parents=True)
        source = b"# fixed fake tune source\n"
        (self.root / "ironmule" / "tune.py").write_bytes(source)
        worker_source = Path(worker.__file__).read_bytes()
        (self.root / worker.WORKER_FILENAME).write_bytes(worker_source)
        relative = "ironmule/tune.py"
        source_hash = hashlib.sha256(source).hexdigest()
        registry = worker._registry_hash((relative,))
        source_digest = _source_digest((relative,), {relative: source_hash}, registry)
        self.spec = {
            "schema": worker.SCHEMA, "stage": "test", "candidate": "combined_core_profile",
            "model": {"model_id": "fake/gemma", "revision": "main", "manifest": "a" * 64,
                      "architecture": "gemma3", "quant_bits": 4, "quant_group_size": 64,
                      "tokenizer": "b" * 64},
            "workload": {"prompt_family": "default", "tokenizer": "b" * 64, "generator": "g",
                         "context_bucket": "short", "batch": 1, "concurrency": 1, "max_tokens": 3,
                         "greedy": True, "prompt_logprobs": False, "power_mode": "ac", "mode": "interactive"},
            "expected": {"commit": "c" * 40, "source_digest": source_digest,
                          "registry_hash": registry, "fingerprint": "d" * 64,
                          "worker_sha256": hashlib.sha256(worker_source).hexdigest(),
                          "tune_search_contract_sha256": worker.TUNE_SEARCH_CONTRACT_SHA256,
                          "pythonpath_sha256": hashlib.sha256(b"").hexdigest()},
            "limits": {"max_seconds": 30.0, "max_output_bytes": 64 * 1024,
                       "max_rss_bytes": 2**40, "max_peak_memory_bytes": 2**40,
                       "max_swap_delta_bytes": 0, "ac_connected": True, "low_power": False,
                       "processes": 6, "repeats": 7, "warmup": 2, "ttft_contract": worker.TTFT_CONTRACT},
            "session": {"session_id": "s1"},
            "source_manifest": [{"relative_path": relative, "sha256": source_hash, "size_bytes": len(source)}],
        }
        (self.root / worker.SPEC_FILENAME).write_bytes(worker._canonical(self.spec))
        self.old_cwd = Path.cwd()
        os.chdir(self.root)
        self.old_mlx = __import__("sys").modules.get("mlx.core")
        self.old_tune = __import__("sys").modules.get("ironmule.tune")
        self.old_ab = __import__("sys").modules.get("ironmule.ab")
        self.old_runtime = __import__("sys").modules.get("ironmule.runtime")
        self.power_gate = mock.patch.object(worker, "_read_power_state", return_value=(True, False))
        self.power_gate.start()

    def tearDown(self) -> None:
        import sys
        if self.old_mlx is None:
            sys.modules.pop("mlx.core", None)
        else:
            sys.modules["mlx.core"] = self.old_mlx
        if self.old_tune is None:
            sys.modules.pop("ironmule.tune", None)
        else:
            sys.modules["ironmule.tune"] = self.old_tune
        for name, old in (("ironmule.ab", self.old_ab), ("ironmule.runtime", self.old_runtime)):
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
        os.chdir(self.old_cwd)
        self.power_gate.stop()
        self.tmp.cleanup()

    def _install_fakes(self, profile: dict) -> None:
        import sys
        model = self.spec["model"]
        identity = {
            "schema": "ironmule.model_identity.v1", "model_id": model["model_id"],
            "revision": model["revision"], "model_manifest_sha256": model["manifest"],
            "architecture": model["architecture"],
            "quantisation": {"bits": model["quant_bits"], "group_size": model["quant_group_size"]},
            "quantisation_sha256": "q" * 64, "tokenizer_sha256": model["tokenizer"],
            "identity_sha256": "i" * 64,
        }
        profile.setdefault("model_identity", identity)
        profile.setdefault("conditions", {"model_id": identity["model_id"], "model_revision": identity["revision"],
                                           "model_manifest_sha256": identity["model_manifest_sha256"], "model_architecture": identity["architecture"],
                                           "quantisation": identity["quantisation"], "quantisation_sha256": identity["quantisation_sha256"],
                                           "tokenizer_sha256": identity["tokenizer_sha256"], "model_identity_sha256": identity["identity_sha256"]})
        mx = types.ModuleType("mlx.core")
        mx.reset_peak_memory = lambda: None
        mx.get_peak_memory = lambda: 123
        tune = types.ModuleType("ironmule.tune")
        tune.calls = []
        def fake_tune(*args, **kwargs):
            tune.calls.append((args, kwargs))
            if kwargs.get("confirm_winner"):
                tune.confirm(*args, **kwargs)
            return profile
        tune.tune = fake_tune
        runtime = types.ModuleType("ironmule.runtime")
        class FakeKnobs:
            def __init__(self, **kwargs):
                self.values = dict(worker.TUNE_SEARCH_CONTRACT["knobs_defaults"])
                self.values.update(kwargs)
            def as_dict(self):
                return dict(self.values)
        runtime.Knobs = FakeKnobs
        runtime.BASELINE = FakeKnobs()
        tune.SEARCH = [(item["name"], list(item["values"])) for item in worker.TUNE_SEARCH_CONTRACT["search"]]
        tune.KEEP_IF_RATIO_BELOW = worker.TUNE_SEARCH_CONTRACT["keep_if_ratio_below"]
        tune.CONFIRM_PROCESSES = worker.TUNE_SEARCH_CONTRACT["confirm_processes"]
        tune.CONFIRM_REPEATS = worker.TUNE_SEARCH_CONTRACT["confirm_repeats"]
        tune.CONFIRM_WARMUP = worker.TUNE_SEARCH_CONTRACT["confirm_warmup"]
        tune.resolve_local_model = lambda *args, **kwargs: types.SimpleNamespace(identity=identity)
        ab = types.ModuleType("ironmule.ab")
        def fake_run(arms, **kwargs):
            names = tuple(arms)
            children = []
            for index in range(6):
                order = list(names) if index % 2 == 0 else list(reversed(names))
                children.append({"pid": 1000 + index, "order": order, "arms": {
                    name: {"total_ns": [1000.0] * 7, "prefill_ns": [200.0] * 7,
                           "decode_ns": [800.0] * 7, "logical_tokens": [1, 2, 3],
                           "deterministic": True, "decode_steps": 3,
                           "mlx_peak_bytes": 123, "stop_reason": "eos"}
                    for name in names
                }})
            def ratio(value):
                return {"median_ratio": value, "ci_low": value - 0.01,
                        "ci_high": value + 0.01, "pairs": [value, value + 0.01, value - 0.01]}
            return {"raw": children, "token_identity": True, "deterministic": True,
                    "ratios": {"candidate/baseline": {"total_ns": ratio(0.9),
                                                         "prefill_ns": ratio(0.95),
                                                         "decode_ns": ratio(0.88)}}}
        ab.run = fake_run
        tune.confirm = lambda *args, **kwargs: fake_run({"baseline": object(), "candidate": object()})
        sys.modules["mlx.core"] = mx
        sys.modules["ironmule.tune"] = tune
        sys.modules["ironmule.ab"] = ab
        sys.modules["ironmule.runtime"] = runtime

    def _set_stage(self, stage: str) -> None:
        self.spec["stage"] = stage
        (self.root / worker.SPEC_FILENAME).write_bytes(worker._canonical(self.spec))

    def test_complete_profile_is_qualified_and_private_state_is_removed(self) -> None:
        self._install_fakes(_profile())
        with mock.patch.object(worker, "_read_swap_bytes", side_effect=[0, 0]):
            code, result = worker.run_argv(["--spec-file", worker.SPEC_FILENAME])
        self.assertEqual(code, 0)
        self.assertEqual(result["outcome"], "qualified")
        self.assertTrue(result["payload"]["confirmation"]["confirmed"])
        self.assertNotIn("gain", result["payload"])
        self.assertEqual(list(self.root.glob(".ironmule-session-*")), [])

    def test_missing_token_or_confirmation_is_inconclusive(self) -> None:
        self._install_fakes(_profile(complete=False))
        with mock.patch.object(worker, "_read_swap_bytes", side_effect=[0, 0]):
            code, result = worker.run_argv(["--spec-file", worker.SPEC_FILENAME])
        self.assertEqual(code, 0)
        self.assertEqual(result["outcome"], "inconclusive")
        self.assertFalse(result["payload"]["confirmed"])
        self.assertFalse(result["payload"]["resources"]["resource_gate_passed"])

    def test_calibration_screening_can_complete_but_never_qualifies(self) -> None:
        self._set_stage("calibrate")
        profile = _profile()
        profile.update({
            "token_identity": True,
            "trials": [{"knob": "fixed_compiled_cache", "ratio": 0.95,
                         "total_ns": 950, "prefill_ns": 190, "decode_ns": 760,
                         "disposition": "accepted", "verdict": "kept"}],
            "baseline_ns": 1000, "baseline_prefill_ns": 200, "baseline_decode_ns": 800,
            "tuned_ns": 900, "tuned_prefill_ns": 190, "tuned_decode_ns": 710,
        })
        self._install_fakes(profile)
        with mock.patch.object(worker, "_read_swap_bytes", side_effect=[0, 0]):
            code, result = worker.run_argv(["--spec-file", worker.SPEC_FILENAME])
        self.assertEqual(code, 0)
        self.assertEqual(result["outcome"], "ok")
        self.assertEqual(result["reason"], "calibration_complete")
        self.assertTrue(result["payload"]["calibration"]["complete"])
        self.assertIsNone(result["payload"]["confirmation"])
        self.assertEqual(len(result["payload"]["aa_baseline_samples"]), 6)
        self.assertIsInstance(MetricSample(**result["payload"]["aa_baseline_samples"][0]), MetricSample)
        self.assertEqual(__import__("sys").modules["ironmule.tune"].calls, [])
        self.assertFalse(result["payload"]["confirmed"])

    def test_actual_tune_calibration_shape_completes_without_top_level_token_or_ttft(self) -> None:
        self._set_stage("calibrate")
        profile = {
            "tokens": [1, 2, 3], "token_count": 3,
            "baseline_ns": 1000, "baseline_prefill_ns": 200, "baseline_decode_ns": 800,
            "tuned_ns": 900, "tuned_prefill_ns": 190, "tuned_decode_ns": 710,
            "trials": [{"knob": "readback_every", "value": 2, "ratio": 0.95,
                        "total_ns": 950, "prefill_ns": 190, "decode_ns": 760,
                        "disposition": "accepted", "verdict": "kept"}],
            "confirmation": None,
        }
        self._install_fakes(profile)
        with mock.patch.object(worker, "_read_swap_bytes", side_effect=[0, 0]):
            code, result = worker.run_argv(["--spec-file", worker.SPEC_FILENAME])
        self.assertEqual(code, 0)
        self.assertEqual(result["outcome"], "ok")
        self.assertEqual(result["reason"], "calibration_complete")
        self.assertIn("aa_baseline_samples", result["payload"])
        self.assertIsNone(result["payload"]["confirmation"])

    def test_actual_confirmation_shape_preserves_phase_pairs_and_qualifies_from_raw_evidence(self) -> None:
        profile = _profile()
        phase = lambda ratio: {"median_ratio": ratio, "ci_low": ratio - 0.01,
                               "ci_high": ratio + 0.01, "pairs": [ratio, ratio + 0.01, ratio - 0.01]}
        profile["confirmation"] = {
            "ratio": {"total_ns": phase(0.9), "prefill_ns": phase(0.95), "decode_ns": phase(0.88)},
            "token_identity": True,
        }
        profile.pop("ttft_ms", None)
        self._install_fakes(profile)
        with mock.patch.object(worker, "_read_swap_bytes", side_effect=[0, 0]):
            code, result = worker.run_argv(["--spec-file", worker.SPEC_FILENAME])
        self.assertEqual(code, 0)
        self.assertEqual(result["outcome"], "qualified")
        confirmation = result["payload"]["confirmation"]
        self.assertTrue(confirmation["confirmed"])
        self.assertTrue(result["payload"]["confirmed"])
        self.assertEqual(confirmation["pair_count"], 3)
        self.assertEqual(set(confirmation["ratios"]), {"total_ns", "prefill_ns", "decode_ns"})
        self.assertEqual(len(confirmation["ratios"]["decode_ns"]["pairs"]), 3)

    def test_test_ab_normalization_never_emits_aa_aliases(self) -> None:
        self._install_fakes(_profile())
        import sys
        raw = sys.modules["ironmule.ab"].run({"baseline": object(), "candidate": object()})
        spec = {"limits": {"processes": 6, "repeats": 7}, "session": {"session_id": "s1"},
                "expected": {"fingerprint": "d" * 64}, "workload": self.spec["workload"],
                "model": self.spec["model"]}
        normalized = worker._normalise_ab(raw, spec=spec, arm_names=("baseline", "candidate"))
        self.assertIn("baseline_samples", normalized)
        self.assertIn("candidate_samples", normalized)
        self.assertNotIn("aa_baseline_samples", normalized)
        self.assertNotIn("aa_control_samples", normalized)

    def test_rss_peak_includes_model_children_and_fails_closed(self) -> None:
        usage = [types.SimpleNamespace(ru_maxrss=100), types.SimpleNamespace(ru_maxrss=250)]
        with mock.patch.object(worker.resource, "getrusage", side_effect=usage):
            value = worker._rss_bytes()
        expected_multiplier = 1 if worker.sys.platform == "darwin" else 1024
        self.assertEqual(value, 250 * expected_multiplier)
        with mock.patch.object(worker.resource, "getrusage", side_effect=OSError("unavailable")):
            self.assertIsNone(worker._rss_bytes())

    def test_raw_sample_uses_median_not_outlier_sensitive_mean(self) -> None:
        self._install_fakes(_profile())
        import sys
        raw = sys.modules["ironmule.ab"].run({"aa_left": object(), "aa_right": object()})
        arm = raw["raw"][0]["arms"]["aa_left"]
        arm["prefill_ns"] = [100.0] * 6 + [10_000.0]
        arm["decode_ns"] = [100.0] * 6 + [10_000.0]
        spec = {"limits": {"processes": 6, "repeats": 7}, "session": {"session_id": "s1"},
                "expected": {"fingerprint": "d" * 64}, "workload": self.spec["workload"],
                "model": self.spec["model"]}
        sample, raw_arm = worker._raw_sample(raw["raw"][0], "aa_left", pair_id="p", order="AB", spec=spec, arm_names=("aa_left", "aa_right"))
        self.assertEqual(sample["engine_ttft_ns"], 100.0)
        self.assertEqual(sample["decode_tps"], 30_000_000.0)
        self.assertEqual(raw_arm["token_hash"], sample["token_hash"])

    def test_calibration_without_screening_stays_inconclusive(self) -> None:
        self._set_stage("calibrate")
        profile = _profile()
        profile.update({"token_identity": True, "trials": []})
        self._install_fakes(profile)
        import sys
        sys.modules["ironmule.ab"].run = lambda *args, **kwargs: {}
        with mock.patch.object(worker, "_read_swap_bytes", side_effect=[0, 0]):
            code, result = worker.run_argv(["--spec-file", worker.SPEC_FILENAME])
        self.assertEqual(code, 0)
        self.assertEqual(result["outcome"], "inconclusive")
        self.assertIn("raw_aa_evidence", result["reason"])

    def test_free_flags_absolute_path_and_duplicate_json_are_rejected(self) -> None:
        for argv in (["--model", "fake/gemma"], ["--spec-file", str(self.root / worker.SPEC_FILENAME)]):
            code, result = worker.run_argv(argv)
            self.assertEqual(code, 2)
            self.assertEqual(result["outcome"], "rejected")
        with self.assertRaises(worker.WorkerError):
            worker._strict_load(b'{"schema":1,"schema":2}')

    def test_model_source_accepts_only_org_name_hub_ids(self) -> None:
        self.assertEqual(worker.validate_hub_model_id("mlx-community/gemma-3-4b-it-4bit"), "mlx-community/gemma-3-4b-it-4bit")
        for value in ("local:gemma", "/tmp/gemma", "gemma", "org/name/extra", "org\\name", "../name", "org/..", "org/.", ". /name"):
            with self.subTest(value=value):
                with self.assertRaises(worker.WorkerError) as raised:
                    worker.validate_hub_model_id(value)
                self.assertEqual(str(raised.exception), "unsupported_model_source")

    def test_unknown_power_state_is_inconclusive(self) -> None:
        self._install_fakes(_profile())
        with mock.patch.object(worker, "_read_power_state", return_value=(None, False)):
            code, result = worker.run_argv(["--spec-file", worker.SPEC_FILENAME])
        self.assertEqual(code, 0)
        self.assertEqual(result["outcome"], "inconclusive")
        self.assertEqual(result["reason"], "ac_disconnected")
        with mock.patch.object(worker, "_read_power_state", return_value=(True, None)):
            code, result = worker.run_argv(["--spec-file", worker.SPEC_FILENAME])
        self.assertEqual(code, 0)
        self.assertEqual(result["reason"], "low_power_enabled")

    def test_tune_contract_mismatch_never_reaches_profile(self) -> None:
        self._install_fakes(_profile())
        import sys
        sys.modules["ironmule.tune"].SEARCH = []
        with mock.patch.object(worker, "_read_swap_bytes", side_effect=[0, 0]):
            code, result = worker.run_argv(["--spec-file", worker.SPEC_FILENAME])
        self.assertEqual(code, 0)
        self.assertEqual(result["outcome"], "inconclusive")
        self.assertEqual(result["reason"], "tune_contract_mismatch")

    def test_local_model_identity_mismatch_blocks_before_ab(self) -> None:
        self._set_stage("calibrate")
        self._install_fakes(_profile())
        import sys
        resolver = sys.modules["ironmule.tune"].resolve_local_model
        wrong = dict(resolver().identity)
        wrong["model_id"] = "other/model"
        sys.modules["ironmule.tune"].resolve_local_model = lambda *args, **kwargs: types.SimpleNamespace(identity=wrong)
        with mock.patch.object(worker, "_read_swap_bytes", side_effect=[0, 0]):
            code, result = worker.run_argv(["--spec-file", worker.SPEC_FILENAME])
        self.assertEqual(code, 0)
        self.assertEqual(result["outcome"], "inconclusive")
        self.assertEqual(result["reason"], "model_identity_mismatch")

    def test_source_mutation_is_rejected_before_import(self) -> None:
        (self.root / "ironmule" / "tune.py").write_bytes(b"changed")
        code, result = worker.run_argv(["--spec-file", worker.SPEC_FILENAME])
        self.assertEqual(code, 2)
        self.assertEqual(result["outcome"], "rejected")

    def test_timeout_is_an_explicit_envelope(self) -> None:
        self._install_fakes(_profile())
        import sys
        timeout_module = sys.modules["ironmule.tune"]
        timeout_module.tune = lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("synthetic"))
        with mock.patch.object(worker, "_read_swap_bytes", side_effect=[0, 0]):
            code, result = worker.run_argv(["--spec-file", worker.SPEC_FILENAME])
        self.assertEqual(code, 0)
        self.assertEqual(result["outcome"], "timeout")
        self.assertEqual(result["payload"]["status"], "timeout")


if __name__ == "__main__":
    unittest.main()
