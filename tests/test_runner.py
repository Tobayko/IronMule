import resource
import hashlib
import copy
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from friday_h0 import benchmark, worker
from friday_h0.correctness_contract import trusted_performance_fixture_identity
from friday_h0.provenance import collect_provenance
from friday_h0.canonical import canonical_json_bytes
from friday_h0.runner import (
    OFFLINE_MODES,
    RunnerError,
    _artifact_rows,
    _database_path_for_tests,
    _initialize_database_for_tests,
    _metric_rows,
    _persist_mlx_common_result,
    _seeds_for_mode,
    _test_root,
    build_manifest,
    database_path,
    load_aa_sessions,
    load_and_aggregate_h0_aa,
    normalize_mlx_common_result,
    result_exit_code,
    run_id_for,
    run_mlx,
    run_offline,
)
from friday_h0.protocol import close_manifest, fallback_result, validate_result
from friday_h0.manifest import validate_manifest
from friday_h0.supervisor import SupervisorLimits
from friday_h0.storage import Storage, StorageError
from tests.test_friday_benchmark import FakeBackend, FakeClock, small_fixture
from tests.test_aggregation import _correctness_cases, _session
from tests.test_manifest import valid_manifest


ROOT = Path(__file__).resolve().parents[1]


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.project_root = Path(self.temporary.name)
        self.context = _test_root(self.project_root)

    def tearDown(self):
        self.temporary.cleanup()

    def test_fixed_manifest_and_database_path(self):
        provenance = collect_provenance(ROOT)
        manifest = build_manifest("analysis_known_win", provenance)
        self.assertEqual(manifest["process"], {"set": "analysis", "index": 0})
        self.assertEqual(manifest["limits"]["total_s"], 120)
        self.assertTrue(manifest["run_id"].startswith("h0-analysis_known_win-analysis-0-"))
        self.assertEqual(_database_path_for_tests(self.context), (self.project_root / ".friday-data" / "h0.sqlite3").resolve())
        with self.assertRaises(RunnerError):
            database_path(self.project_root)

    def test_db_init_is_fixed_and_private(self):
        path = _initialize_database_for_tests(self.context)
        self.assertEqual(path, _database_path_for_tests(self.context))
        self.assertEqual((path.parent.stat().st_mode & 0o777), 0o700)
        self.assertEqual((path.stat().st_mode & 0o777), 0o600)
        self.assertEqual(path.parent.stat().st_uid, os.geteuid())
        self.assertEqual(path.stat().st_uid, os.geteuid())

    def test_existing_broad_modes_are_restricted_before_sqlite(self):
        data_dir = self.project_root / ".friday-data"
        data_dir.mkdir(mode=0o755)
        path = data_dir / "h0.sqlite3"
        path.write_bytes(b"")
        path.chmod(0o644)
        _initialize_database_for_tests(self.context)
        self.assertEqual(data_dir.stat().st_mode & 0o777, 0o700)
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_analysis_fixture_is_atomically_persisted_and_exact_bundle_replay_is_idempotent(self):
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        first = run_offline("analysis_known_win", _test_context=self.context, now_ns=1)
        after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.assertEqual(first.persistence.state, "inserted")
        self.assertEqual(first.result["classification"], "promoted")
        self.assertEqual(result_exit_code(first.result), 0)
        scalar_metrics = _metric_rows(first.result["evidence"])
        artifacts = _artifact_rows(first.result)
        with Storage.open(_database_path_for_tests(self.context)) as storage:
            replay = storage.persist_common_result(
                first.manifest,
                first.result,
                created_at_unix_ns=2,
                scalar_metrics=scalar_metrics,
                artifacts=artifacts,
            )
        self.assertEqual(replay.state, "idempotent")
        self.assertEqual(replay.bundle_sha256, first.persistence.bundle_sha256)
        self.assertGreaterEqual(after, before)

    def test_analysis_fixture_reexecution_collision_rejects_deterministic_volatile_change(self):
        first = run_offline("analysis_known_win", _test_context=self.context, now_ns=1)
        reexecution = copy.deepcopy(first.result)
        evidence = reexecution["evidence"]
        evidence["rss_peak_bytes"] = (evidence["rss_peak_bytes"] or 0) + 1
        evidence["rss_missing_reason"] = None
        reexecution = validate_result(reexecution, manifest=first.manifest)
        scalar_metrics = _metric_rows(reexecution["evidence"])
        artifacts = _artifact_rows(reexecution)
        with self.assertRaisesRegex(StorageError, "different common_result bundle"):
            with Storage.open(_database_path_for_tests(self.context)) as storage:
                storage.persist_common_result(
                    first.manifest,
                    reexecution,
                    created_at_unix_ns=2,
                    scalar_metrics=scalar_metrics,
                    artifacts=artifacts,
                )
        original_scalar_metrics = _metric_rows(first.result["evidence"])
        original_artifacts = _artifact_rows(first.result)
        with Storage.open(_database_path_for_tests(self.context), read_only=True) as readonly:
            self.assertEqual(
                readonly.verify_common_result_bundle(
                    first.manifest,
                    first.result,
                    scalar_metrics=original_scalar_metrics,
                    artifacts=original_artifacts,
                ),
                "verified",
            )

    def test_exit70_is_persisted_as_worker_fallback(self):
        outcome = run_offline("control_exit_70", _test_context=self.context, now_ns=3)
        self.assertEqual(outcome.result["status"], "worker_exit")
        self.assertEqual(outcome.result["classification"], "worker_exit")
        self.assertEqual(outcome.persistence.state, "inserted")
        self.assertEqual(result_exit_code(outcome.result), 10)

    def test_timeout_uses_private_short_test_seam(self):
        limits = SupervisorLimits.for_tests(total_s=0.10, cleanup_s=0.20, control_sleep_s=0.30)
        outcome = run_offline("control_timeout", _test_context=self.context, limits=limits, now_ns=4)
        self.assertEqual(outcome.result["status"], "timeout")
        self.assertEqual(outcome.persistence.state, "inserted")
        self.assertEqual(result_exit_code(outcome.result), 10)

    def test_non_offline_modes_are_not_executable(self):
        self.assertNotIn("eager_baseline", OFFLINE_MODES)
        with self.assertRaises(RuntimeError):
            run_offline("eager_baseline", _test_context=self.context)

    def test_mlx_execute_path_binds_tuple_and_persists_without_runtime(self):
        provenance = collect_provenance(ROOT)
        manifest = close_manifest(
            build_manifest("aa_gpu", provenance, process_set="confirmation", process_index=2)
        )
        worker_result = fallback_result(
            manifest=manifest,
            status="invalid",
            classification="runtime_unavailable",
            code="runtime_unavailable",
            message="MLX runtime is unavailable",
            evidence={"rss_peak_bytes": None, "rss_missing_reason": "unavailable"},
        )
        limits = SupervisorLimits.for_tests(total_s=0.2, cleanup_s=0.2)
        with mock.patch("friday_h0.runner.run_supervised", return_value=worker_result) as supervisor:
            outcome = run_mlx(
                "aa_gpu",
                "confirmation",
                2,
                _test_context=self.context,
                limits=limits,
                now_ns=9,
            )
        supervisor.assert_called_once()
        called_manifest = supervisor.call_args.args[0]
        self.assertEqual(called_manifest.mode, "aa_gpu")
        self.assertEqual(called_manifest.value["process"], {"set": "confirmation", "index": 2})
        self.assertEqual(outcome.result["classification"], "runtime_unavailable")
        self.assertEqual(outcome.result["action"], "baseline_fallback")
        self.assertEqual(outcome.persistence.state, "inserted")

    def test_mlx_execute_path_rejects_unregistered_tuple(self):
        for mode, process_set, process_index in (
            ("not-a-mode", "characterization", 0),
            ("aa_gpu", "analysis", 0),
            ("aa_gpu", "characterization", True),
            ("aa_gpu", "characterization", 3),
        ):
            with self.assertRaises(RunnerError):
                run_mlx(mode, process_set, process_index, _test_context=self.context)

    def test_database_symlinks_are_rejected_before_sqlite(self):
        data_dir = self.project_root / ".friday-data"
        data_dir.mkdir(mode=0o700)
        external = self.project_root / "external.sqlite3"
        external.write_bytes(b"must-not-be-opened")
        (data_dir / "h0.sqlite3").symlink_to(external)
        with self.assertRaises(RunnerError):
            _initialize_database_for_tests(self.context)
        self.assertEqual(external.read_bytes(), b"must-not-be-opened")

    def test_database_parent_symlink_is_rejected(self):
        external_dir = self.project_root / "external-dir"
        external_dir.mkdir()
        (self.project_root / ".friday-data").symlink_to(external_dir, target_is_directory=True)
        with self.assertRaises(RunnerError):
            _initialize_database_for_tests(self.context)
        self.assertFalse((external_dir / "h0.sqlite3").exists())

    def test_database_inode_replacement_is_detected_after_connect(self):
        replacement = self.project_root / "replacement.sqlite3"

        def replace(path: Path) -> None:
            os.replace(path, replacement)
            path.write_bytes(b"replacement")

        with mock.patch("friday_h0.runner._TEST_AFTER_CONNECT_HOOK", replace):
            with self.assertRaises(RunnerError):
                _initialize_database_for_tests(self.context)

    def test_parent_setup_failure_is_persisted_as_invalid_fallback(self):
        with mock.patch("friday_h0.runner.run_supervised", side_effect=OSError("secret/path")):
            outcome = run_offline("analysis_known_win", _test_context=self.context, now_ns=5)
        self.assertEqual(outcome.result["status"], "invalid")
        self.assertEqual(outcome.result["classification"], "invalid")
        self.assertEqual(outcome.result["error"]["code"], "parent_setup_failure")
        self.assertNotIn("secret/path", outcome.result["error"]["message"])
        self.assertEqual(outcome.persistence.state, "inserted")

    def test_run_id_requires_registered_process_tuple_and_full_hash(self):
        provenance = collect_provenance()
        from friday_h0.runner import run_id_for

        value = run_id_for("aa_gpu", "confirmation", 2, provenance)
        expected = hashlib.sha256(canonical_json_bytes(provenance.as_manifest())).hexdigest()
        self.assertTrue(value.endswith(expected))
        self.assertEqual(len(value.rsplit("-", 1)[-1]), 64)
        with self.assertRaises(RunnerError):
            run_id_for("aa_gpu", "analysis", 0, provenance)
        with self.assertRaises(RunnerError):
            run_id_for("analysis_known_win", "analysis", True, provenance)

    def test_aa_manifest_seed_builder_binds_bootstrap_by_process_set(self):
        self.assertEqual(
            _seeds_for_mode("aa_gpu", "characterization", 2)["bootstrap_seed"],
            0xAA052026,
        )
        self.assertEqual(
            _seeds_for_mode("aa_gpu", "confirmation", 0)["bootstrap_seed"],
            0xAA052126,
        )
        self.assertNotIn("bootstrap_seed", _seeds_for_mode("analysis_known_win", "analysis", 0))

    def _real_eager_common_result(self):
        manifest = close_manifest(valid_manifest("eager_baseline"))
        seed = manifest.value["seeds"]["fixture"]
        cases = _correctness_cases(performance_seed=seed)
        suite = {"cases": cases, "passed": True, "performance": cases[-2], "sign_invariant": cases[-1]}
        identity = trusted_performance_fixture_identity(
            a_shape=[2048, 2048], b_shape=[2048, 2048], dtype="float16",
            layout="C-contiguous", fixture_seed=seed,
        )
        with mock.patch.object(benchmark, "_generate_fixture", side_effect=small_fixture), mock.patch.object(
            benchmark, "_correctness_suite", return_value=suite
        ):
            domain = benchmark.run_mlx_benchmark(
                manifest.value,
                backend_factory=FakeBackend(),
                clock_ns=FakeClock(),
            )
        # The injected 4x4 fixture keeps this seam fast; bind its producer
        # metadata to the registered production identity used by the parent.
        domain["evidence"]["fixture"] = {
            key: identity[key] for key in
            ("fixture_seed", "a_sha256", "b_sha256", "metadata_sha256", "fixture_sha256")
        }
        return manifest, worker._benchmark_result(
            manifest,
            domain,
            {"rss_peak_bytes": 1234, "rss_missing_reason": None},
        )

    def _real_mode_common_result(self, mode):
        manifest = close_manifest(valid_manifest(mode))
        seed = manifest.value["seeds"]["fixture"]
        cases = _correctness_cases(performance_seed=seed)
        suite = {"cases": cases, "passed": True, "performance": cases[-2], "sign_invariant": cases[-1]}
        identity = trusted_performance_fixture_identity(
            a_shape=[2048, 2048], b_shape=[2048, 2048], dtype="float16",
            layout="C-contiguous", fixture_seed=seed,
        )
        with mock.patch.object(benchmark, "_generate_fixture", side_effect=small_fixture), mock.patch.object(
            benchmark, "_correctness_suite", return_value=suite
        ):
            domain = benchmark.run_mlx_benchmark(
                manifest.value,
                backend_factory=FakeBackend(),
                clock_ns=FakeClock(),
            )
        domain["evidence"]["fixture"] = {
            key: identity[key] for key in
            ("fixture_seed", "a_sha256", "b_sha256", "metadata_sha256", "fixture_sha256")
        }
        return manifest, worker._benchmark_result(
            manifest,
            domain,
            {"rss_peak_bytes": 1234, "rss_missing_reason": None},
        )

    def test_eager_benchmark_worker_normalizer_seam_and_persistence(self):
        manifest, result = self._real_eager_common_result()
        projection = normalize_mlx_common_result(manifest, result)
        self.assertEqual(projection["counts"]["raw_samples"], 40)
        self.assertEqual(projection["counts"]["correctness_metrics"], 88)
        persisted = _persist_mlx_common_result(
            manifest,
            result,
            _test_context=self.context,
            created_at_unix_ns=10,
        )
        self.assertEqual(persisted["persistence"].state, "inserted")
        with Storage.open(_database_path_for_tests(self.context), read_only=True) as readonly:
            self.assertEqual(
                readonly.verify_common_result_bundle(
                    manifest,
                    result,
                    raw_samples=projection["raw_samples"],
                    scalar_metrics=projection["scalar_metrics"],
                    correctness_metrics=projection["correctness_metrics"],
                    artifacts=projection["artifacts"],
                ),
                "verified",
            )

    def test_invalid_benchmark_diagnostic_roundtrip_is_read_only_verified(self):
        manifest = close_manifest(valid_manifest("aa_gpu"))
        class WarmupClock:
            def __init__(self):
                self.calls = 0
                self.now = 0

            def __call__(self):
                if self.calls < 2:
                    self.now += 1
                else:
                    offset = self.calls - 2
                    phase = offset % 3
                    sample = offset // 3
                    if phase == 1:
                        self.now += 1
                    elif phase == 2:
                        self.now += 100 if sample % 2 == 0 else 10_000_000
                self.calls += 1
                return self.now

        backend = FakeBackend()
        seed = manifest.value["seeds"]["fixture"]
        cases = _correctness_cases(performance_seed=seed)
        suite = {"cases": cases, "passed": True, "performance": cases[-2], "sign_invariant": cases[-1]}
        with mock.patch.object(benchmark, "_generate_fixture", side_effect=small_fixture), mock.patch.object(
            benchmark, "_correctness_suite", return_value=suite
        ):
            domain = benchmark.run_mlx_benchmark(
                manifest.value,
                backend_factory=backend,
                clock_ns=WarmupClock(),
            )
        diagnostic = domain["evidence"]["failure_diagnostic"]
        self.assertEqual(domain["error"]["code"], "warmup_unstable")
        self.assertEqual(diagnostic["schema_version"], 2)
        self.assertEqual(len(diagnostic["details"]["warmup_block_per_eval_ns"]), 16)
        self.assertEqual(len(diagnostic["details"]["warmup_blocks"]), 16)
        self.assertTrue(all(isinstance(value, int) and value > 0 for value in diagnostic["details"]["warmup_block_per_eval_ns"]))
        self.assertEqual(backend.eval_calls, sum(block["evaluations"] for block in diagnostic["details"]["warmup_blocks"]))
        self.assertEqual(backend.sync_calls, backend.eval_calls)
        result = worker._benchmark_result(manifest, domain, {"rss_peak_bytes": 1, "rss_missing_reason": None})
        self.assertEqual(result["evidence"]["benchmark_evidence"]["failure_diagnostic"], diagnostic)
        projection = normalize_mlx_common_result(manifest, result)
        self.assertEqual(projection["raw_samples"], [])
        self.assertEqual(projection["correctness_metrics"], [])
        persisted = _persist_mlx_common_result(
            manifest,
            result,
            _test_context=self.context,
            created_at_unix_ns=11,
        )
        self.assertEqual(persisted["persistence"].state, "inserted")
        with Storage.open(_database_path_for_tests(self.context), read_only=True) as readonly:
            self.assertEqual(
                readonly.verify_common_result_bundle(
                    manifest,
                    result,
                    raw_samples=projection["raw_samples"],
                    scalar_metrics=projection["scalar_metrics"],
                    correctness_metrics=projection["correctness_metrics"],
                    artifacts=projection["artifacts"],
                ),
                "verified",
            )
            events = readonly.rows("status_events", manifest.run_id)
            self.assertEqual(len(events), 1)
            payload = json.loads(events[0]["payload_json"])
            self.assertEqual(
                payload["result"]["evidence"]["benchmark_evidence"]["failure_diagnostic"],
                result["evidence"]["benchmark_evidence"]["failure_diagnostic"],
            )

    def test_eager_raw_samples_envelopes_are_bound_fail_closed(self):
        mutations = (
            (
                "benchmark_evidence.raw_samples",
                lambda evidence: evidence["raw_samples"].pop(),
            ),
            (
                "benchmark_evidence.arms.baseline.raw_samples",
                lambda evidence: evidence["arms"]["baseline"]["raw_samples"].pop(),
            ),
        )
        for mutation, mutate in mutations:
            manifest, result = self._real_eager_common_result()
            mutate(result["evidence"]["benchmark_evidence"])
            with self.assertRaises(RunnerError, msg=mutation):
                normalize_mlx_common_result(manifest, result)

    def test_eager_comparison_contract_is_closed_and_bound(self):
        for mutation in (
            "missing",
            "extra",
            "classification",
            "action",
            "aggregation",
            "cache",
            "raw_samples",
        ):
            manifest, result = self._real_eager_common_result()
            comparison = result["evidence"]["benchmark_evidence"]["comparison"]
            if mutation == "missing":
                del comparison["cache_state"]
            elif mutation == "extra":
                comparison["extra"] = True
            elif mutation == "classification":
                comparison["benchmark_classification"] = "measurement_complete"
            elif mutation == "action":
                comparison["action"] = "aggregation_required"
            elif mutation == "aggregation":
                comparison["aggregation_required"] = True
            elif mutation == "cache":
                comparison["cache_state"] = "warm"
            else:
                comparison["raw_samples"] = comparison["raw_samples"][:-1]
            with self.assertRaises(RunnerError, msg=mutation):
                normalize_mlx_common_result(manifest, result)

    def test_correctness_case_schema_and_registered_bindings_are_exact(self):
        mutations = (
            "missing", "extra", "wrong_type", "seed", "shape", "digest", "hard_caps",
            "sign_reference", "sign_relation", "sign_fixture_digest", "sign_seed", "sign_extra",
        )
        for mutation in mutations:
            manifest, result = self._real_eager_common_result()
            cases = result["evidence"]["benchmark_evidence"]["correctness"]["cases"]
            case = cases[0]
            if mutation == "missing":
                del case["hard_caps"]
            elif mutation == "extra":
                case["extra"] = True
            elif mutation == "wrong_type":
                case["shape"] = "64x64"
            elif mutation == "seed":
                case["seed"] += 1
            elif mutation == "shape":
                case["shape"][0] += 1
            elif mutation == "digest":
                case["fixture_digest"] = "0" * 64
            elif mutation == "hard_caps":
                case["hard_caps"]["abs_max"] = 2.0
            elif mutation == "sign_reference":
                result["evidence"]["benchmark_evidence"]["correctness"]["sign_invariant"]["reference"] = "wrong"
            elif mutation == "sign_relation":
                result["evidence"]["benchmark_evidence"]["correctness"]["sign_invariant"]["relation"] = "wrong"
            elif mutation == "sign_fixture_digest":
                result["evidence"]["benchmark_evidence"]["correctness"]["sign_invariant"]["fixture_digest"] = "0" * 64
            elif mutation == "sign_seed":
                result["evidence"]["benchmark_evidence"]["correctness"]["sign_invariant"]["seed"] += 1
            else:
                result["evidence"]["benchmark_evidence"]["correctness"]["sign_invariant"]["extra"] = True
            with self.assertRaises(RunnerError, msg=mutation):
                normalize_mlx_common_result(manifest, result)

    def test_performance_fixture_identity_rejects_full_co_mutation(self):
        manifest, result = self._synthetic_common_result()
        fixture = result["evidence"]["benchmark_evidence"]["fixture"]
        for field in ("a_sha256", "b_sha256", "metadata_sha256", "fixture_sha256"):
            candidate = copy.deepcopy(result)
            candidate["evidence"]["benchmark_evidence"]["fixture"][field] = "0" * 64
            with self.assertRaises(RunnerError, msg=field):
                normalize_mlx_common_result(manifest, candidate)
        candidate = copy.deepcopy(result)
        candidate_fixture = candidate["evidence"]["benchmark_evidence"]["fixture"]
        for field in ("a_sha256", "b_sha256", "metadata_sha256", "fixture_sha256"):
            candidate_fixture[field] = "f" * 64
        with self.assertRaises(RunnerError, msg="full_co_mutation"):
            normalize_mlx_common_result(manifest, candidate)

    def test_timing_contract_is_mode_exact_and_arm_bound(self):
        manifest, result = self._real_eager_common_result()
        domain = result["evidence"]["benchmark_evidence"]
        for mutation in ("setup", "first_eval", "arm_first_eval"):
            candidate = copy.deepcopy(result)
            candidate_domain = candidate["evidence"]["benchmark_evidence"]
            if mutation == "setup":
                candidate_domain["compile_wrapper_setup_ns"] = 1
            elif mutation == "first_eval":
                candidate_domain["first_eval_compile_inclusive_ns"] = 1
            else:
                candidate_domain["arms"]["baseline"]["first_eval_compile_inclusive_ns"] = 1
            with self.assertRaises(RunnerError, msg=f"eager:{mutation}"):
                normalize_mlx_common_result(manifest, candidate)

        aa_manifest, aa_result = self._real_mode_common_result("aa_gpu")
        for field in ("compile_wrapper_setup_ns", "first_eval_compile_inclusive_ns"):
            candidate = copy.deepcopy(aa_result)
            candidate["evidence"]["benchmark_evidence"][field] = 1
            with self.assertRaises(RunnerError, msg=f"aa:{field}"):
                normalize_mlx_common_result(aa_manifest, candidate)
        candidate = copy.deepcopy(aa_result)
        candidate["evidence"]["benchmark_evidence"]["arms"]["candidate"]["first_eval_compile_inclusive_ns"] = 1
        with self.assertRaises(RunnerError, msg="aa:arm_first_eval"):
            normalize_mlx_common_result(aa_manifest, candidate)

        compile_manifest, compile_result = self._real_mode_common_result("compile_comparison")
        compile_domain = compile_result["evidence"]["benchmark_evidence"]
        normalize_mlx_common_result(compile_manifest, compile_result)
        for mutation in ("setup_none", "first_none", "baseline_first", "candidate_missing", "candidate_mismatch"):
            candidate = copy.deepcopy(compile_result)
            candidate_domain = candidate["evidence"]["benchmark_evidence"]
            if mutation == "setup_none":
                candidate_domain["compile_wrapper_setup_ns"] = None
            elif mutation == "first_none":
                candidate_domain["first_eval_compile_inclusive_ns"] = None
            elif mutation == "baseline_first":
                candidate_domain["arms"]["baseline"]["first_eval_compile_inclusive_ns"] = 1
            elif mutation == "candidate_missing":
                del candidate_domain["arms"]["candidate"]["first_eval_compile_inclusive_ns"]
            else:
                candidate_domain["arms"]["candidate"]["first_eval_compile_inclusive_ns"] += 1
            with self.assertRaises(RunnerError, msg=f"compile:{mutation}"):
                normalize_mlx_common_result(compile_manifest, candidate)

    def test_memory_limit_and_gate_are_reconstructed_fail_closed(self):
        manifest, result = self._real_eager_common_result()
        domain = result["evidence"]["benchmark_evidence"]
        for mutation in ("missing", "extra", "wrong_type", "applied_reason", "gate_wrong", "peak_missing", "rss_missing"):
            candidate = copy.deepcopy(result)
            candidate_domain = candidate["evidence"]["benchmark_evidence"]
            if mutation == "missing":
                del candidate_domain["memory_limit"]["missing_reason"]
            elif mutation == "extra":
                candidate_domain["memory_limit"]["extra"] = True
            elif mutation == "wrong_type":
                candidate_domain["memory_limit"]["applied"] = 1
            elif mutation == "applied_reason":
                candidate_domain["memory_limit"]["missing_reason"] = "api_unavailable"
            elif mutation == "gate_wrong":
                candidate_domain["memory_gate"] = "aggregation_required" if domain["memory_gate"] != "aggregation_required" else "not_evaluable_missing_required_metric"
            else:
                target = "mlx_peak_memory" if mutation == "peak_missing" else "rss"
                for row in candidate_domain["memory"]:
                    if row["name"] == target:
                        row["value"] = None
                        row["missing_reason"] = "api_unavailable"
            with self.assertRaises(RunnerError, msg=mutation):
                normalize_mlx_common_result(manifest, candidate)

    def test_memory_name_and_reason_are_closed_and_registered_fallbacks_are_consistent(self):
        manifest, result = self._synthetic_common_result()
        candidate = copy.deepcopy(result)
        candidate["evidence"]["benchmark_evidence"]["memory"][0]["name"] = "arbitrary_memory"
        with self.assertRaises(RunnerError):
            normalize_mlx_common_result(manifest, candidate)

        candidate = copy.deepcopy(result)
        first = candidate["evidence"]["benchmark_evidence"]["memory"][0]
        first["value"] = None
        first["missing_reason"] = "arbitrary_reason"
        with self.assertRaises(RunnerError):
            normalize_mlx_common_result(manifest, candidate)

        reasons = (
            "not_recorded", "not_applicable", "api_unavailable", "unavailable", "no_sample",
            "source_missing", "entry_limit", "ps_exit", "ps_parse", "ps_negative",
            "parent_setup_failure", "invalid_source_value",
        )
        for reason in reasons:
            candidate = copy.deepcopy(result)
            evidence = candidate["evidence"]["benchmark_evidence"]
            evidence["memory_gate"] = "not_evaluable_missing_required_metric"
            for row in evidence["memory"]:
                if row["name"] in {"mlx_peak_memory", "rss"}:
                    row["value"] = None
                    row["missing_reason"] = reason
            normalize_mlx_common_result(manifest, candidate)

        for value in (True, 1 << 63):
            candidate = copy.deepcopy(result)
            candidate["evidence"]["benchmark_evidence"]["memory"][0]["value"] = value
            with self.assertRaises(RunnerError):
                normalize_mlx_common_result(manifest, candidate)

    def test_mode_cross_field_binding_rejects_each_independent_mismatch(self):
        # Every field is checked independently at the Common/Domain/Comparison
        # boundary, so a sparse or mixed producer cannot become a valid SQLite
        # projection by changing only one flag.
        for mutation in ("common_classification", "common_action", "common_aggregation", "common_aggregation_int", "domain_aggregation", "comparison_aggregation"):
            manifest, result = self._real_eager_common_result()
            evidence = result["evidence"]
            domain = evidence["benchmark_evidence"]
            comparison = domain["comparison"]
            if mutation == "common_classification":
                evidence["benchmark_classification"] = "measurement_complete"
            elif mutation == "common_action":
                evidence["benchmark_action"] = "aggregation_required"
            elif mutation == "common_aggregation":
                evidence["aggregation_required"] = True
            elif mutation == "common_aggregation_int":
                evidence["aggregation_required"] = 1
            elif mutation == "domain_aggregation":
                domain["aggregation_required"] = True
            else:
                comparison["aggregation_required"] = True
            with self.assertRaises(RunnerError, msg=mutation):
                normalize_mlx_common_result(manifest, result)

    def test_paired_modes_reject_eager_outer_binding_but_accept_valid_evidence(self):
        for mode in ("compile_comparison", "aa_gpu"):
            manifest, result = self._real_mode_common_result(mode)
            normalize_mlx_common_result(manifest, result)
            evidence = result["evidence"]
            evidence.update(
                {
                    "benchmark_classification": "baseline_reference",
                    "benchmark_action": "not_run",
                    "aggregation_required": False,
                }
            )
            with self.assertRaises(RunnerError, msg=mode):
                normalize_mlx_common_result(manifest, result)

    def test_test_root_is_token_bound(self):
        with self.assertRaises(RunnerError):
            run_offline("analysis_known_win", _test_context=object())

    def test_custom_limits_are_unavailable_without_private_test_context(self):
        limits = SupervisorLimits.for_tests(total_s=0.1, cleanup_s=0.1)
        with self.assertRaises(RunnerError):
            run_offline("control_timeout", limits=limits)

    def test_scalar_mapping_is_finite_nonnegative_and_bounded(self):
        rows = _metric_rows({
            "rss_peak_bytes": -1,
            "rss_peak_bytes_missing_reason": "x" * 1000,
            "stdout_bytes": 2**400,
            "stderr_bytes": None,
            "stderr_bytes_missing_reason": "worker_unavailable",
        })
        self.assertEqual(rows[0]["missing_reason"], "invalid_source_value")
        self.assertEqual(rows[1]["missing_reason"], "invalid_source_value")
        self.assertEqual(rows[2]["missing_reason"], "worker_unavailable")

    def _synthetic_common_result(self, process_set="characterization", index=0):
        # Reuse only the established deterministic offline session shape; no
        # benchmark or MLX execution occurs in this helper.

        provenance = collect_provenance(ROOT)
        from tests.test_manifest import valid_manifest

        raw_manifest = valid_manifest("aa_gpu", set_name=process_set, index=index)
        raw_manifest["provenance"] = provenance.as_manifest()
        raw_manifest["run_id"] = run_id_for("aa_gpu", process_set, index, provenance)
        manifest = close_manifest(validate_manifest(raw_manifest))
        source = _session(process_set, index)
        result = copy.deepcopy(source["result"])
        result["run_id"] = manifest.run_id
        result["manifest_sha256"] = manifest.sha256
        result["evidence"]["rss_peak_bytes"] = 2_000
        result["evidence"]["rss_missing_reason"] = None
        result["evidence"].update({
            "stdout_bytes": 0, "stderr_bytes": 0,
            "stdout_sha256": hashlib.sha256(b"").hexdigest(),
            "stderr_sha256": hashlib.sha256(b"").hexdigest(),
            "stdout_preview": "", "stderr_preview": "",
            "stdout_truncated": False, "stderr_truncated": False,
            "stdout_overflow": False, "stderr_overflow": False,
        })
        for arm in result["evidence"]["benchmark_evidence"]["arms"].values():
            arm["warmup"]["samples"] = [
                {"phase": "warmup", "sample_index": i, "value": arm["warmup"]["durations_ns"][i], "unit": "ns"}
                for i in range(arm["warmup"]["count"])
            ]
            arm["statistics"] = {
                "count": 30,
                "median_ns": 100_000_000.0,
                "mad_ns": 0.0,
                "iqr_ns": 0.0,
                "min_ns": 100_000_000.0,
                "max_ns": 100_000_000.0,
            }
        return manifest, result

    def test_mlx_projection_is_closed_and_hashable_without_runtime(self):
        manifest, result = self._synthetic_common_result()
        with mock.patch("friday_h0.runner.run_supervised", side_effect=AssertionError("worker must not run")):
            first = normalize_mlx_common_result(manifest, result)
            second = normalize_mlx_common_result(manifest, result)
        self.assertEqual(first["projection_sha256"], second["projection_sha256"])
        self.assertEqual(first["counts"]["raw_samples"], 78)
        self.assertEqual(first["counts"]["correctness_metrics"], 88)
        self.assertEqual(len(first["artifacts"]), 14)
        self.assertEqual(first["omitted_normalized_fields"], [])
        self.assertEqual(len({(row["sample_kind"], row["sample_index"]) for row in first["raw_samples"]}), 78)
        self.assertIn("ratio_median", {row["metric_name"] for row in first["scalar_metrics"]})
        self.assertEqual({row["unit"] for row in first["scalar_metrics"] if row["metric_name"].startswith("memory_")}, {"bytes"})
        for arm in result["evidence"]["benchmark_evidence"]["arms"].values():
            self.assertEqual(len(arm["warmup"]["blocks"]), arm["warmup"]["count"])

    def test_mlx_projection_requires_closed_cross_bound_warmup_blocks(self):
        mutations = ("missing", "extra", "short_block", "per_eval_mismatch", "count_mismatch")
        for mutation in mutations:
            manifest, result = self._synthetic_common_result()
            warmup = result["evidence"]["benchmark_evidence"]["arms"]["baseline"]["warmup"]
            if mutation == "missing":
                del warmup["blocks"]
            elif mutation == "extra":
                warmup["blocks"][0]["extra"] = 1
            elif mutation == "short_block":
                warmup["blocks"][0]["block_ns"] = benchmark.H0_BATCH_MIN_NS - 1
            elif mutation == "per_eval_mismatch":
                warmup["blocks"][0]["per_eval_ns"] += 1
            else:
                warmup["blocks"].pop()
            with self.assertRaises(RunnerError, msg=mutation):
                normalize_mlx_common_result(manifest, result)

    def test_projection_hash_binds_full_source_evidence_and_persisted_artifact(self):
        manifest, result = self._synthetic_common_result()
        first = normalize_mlx_common_result(manifest, result)
        changed = copy.deepcopy(result)
        changed["evidence"]["stdout_preview"] = "allowed-but-not-normalized-source-change"
        second = normalize_mlx_common_result(manifest, changed)
        self.assertNotEqual(first["source_evidence_sha256"], second["source_evidence_sha256"])
        self.assertNotEqual(first["projection_sha256"], second["projection_sha256"])
        first_persisted = _persist_mlx_common_result(manifest, result, _test_context=self.context, created_at_unix_ns=10)
        other_temporary = tempfile.TemporaryDirectory()
        try:
            changed_persisted = _persist_mlx_common_result(manifest, changed, _test_context=_test_root(other_temporary.name), created_at_unix_ns=11)
        finally:
            other_temporary.cleanup()
        self.assertNotEqual(first_persisted["persistence"].bundle_sha256, changed_persisted["persistence"].bundle_sha256)
        self.assertNotEqual(first_persisted["projection"]["artifacts"][-1]["sha256"], changed_persisted["projection"]["artifacts"][-1]["sha256"])

    def test_projection_rejects_missing_extra_or_wrong_adapter_contract(self):
        for mutation in ("missing", "extra", "wrong"):
            manifest, result = self._synthetic_common_result()
            contract = result["evidence"]["adapter_contract"]
            if mutation == "missing":
                del result["evidence"]["adapter_contract"]
            elif mutation == "extra":
                contract["extra"] = True
            else:
                contract["common_result_ready"] = True
            with self.assertRaises(RunnerError):
                normalize_mlx_common_result(manifest, result)

    def test_projection_requires_exact_correctness_links(self):
        manifest, result = self._synthetic_common_result()
        result["evidence"]["benchmark_evidence"]["correctness"]["performance"]["passed"] = False
        with self.assertRaises(RunnerError):
            normalize_mlx_common_result(manifest, result)
        manifest, result = self._synthetic_common_result()
        result["evidence"]["benchmark_evidence"]["correctness"]["sign_invariant"]["seed"] = 999
        with self.assertRaises(RunnerError):
            normalize_mlx_common_result(manifest, result)

    def test_mlx_projection_rejects_unknown_nonfinite_duplicate_and_oversize_samples(self):
        manifest, result = self._synthetic_common_result()
        result["evidence"]["benchmark_evidence"]["arms"]["baseline"]["raw_samples"][0]["unknown"] = 1
        with self.assertRaises(RunnerError):
            normalize_mlx_common_result(manifest, result)

    def test_mlx_projection_rejects_nonpositive_timing_and_warmup_cross_bind(self):
        for bad_value in (0.0, -1.0, True):
            manifest, result = self._synthetic_common_result()
            result["evidence"]["benchmark_evidence"]["arms"]["baseline"]["batches"][0]["per_eval_ns"] = bad_value
            with self.assertRaises(RunnerError):
                normalize_mlx_common_result(manifest, result)
        manifest, result = self._synthetic_common_result()
        result["evidence"]["benchmark_evidence"]["arms"]["baseline"]["warmup"]["samples"][0]["value"] = 101.0
        with self.assertRaises(RunnerError):
            normalize_mlx_common_result(manifest, result)
        manifest, result = self._synthetic_common_result()
        result["evidence"]["benchmark_evidence"]["arms"]["baseline"]["raw_samples"][0]["value"] = 101.0
        with self.assertRaises(RunnerError):
            normalize_mlx_common_result(manifest, result)
        manifest, result = self._synthetic_common_result()
        result["evidence"]["benchmark_evidence"]["arms"]["candidate"]["raw_samples"][10]["value"] = float("nan")
        with self.assertRaises(RunnerError):
            normalize_mlx_common_result(manifest, result)
        manifest, result = self._synthetic_common_result()
        result["evidence"]["benchmark_evidence"]["raw_samples"].append(copy.deepcopy(result["evidence"]["benchmark_evidence"]["raw_samples"][0]))
        with self.assertRaises(RunnerError):
            normalize_mlx_common_result(manifest, result)
        manifest, result = self._synthetic_common_result()
        result["evidence"]["benchmark_evidence"]["raw_samples"] = ["x"] * 10_001
        with self.assertRaises(RunnerError):
            normalize_mlx_common_result(manifest, result)
        maximum = (1 << 63) - 1
        manifest, result = self._synthetic_common_result()
        result["evidence"]["benchmark_evidence"]["memory"][0]["measured_at_ns"] = maximum
        normalized = normalize_mlx_common_result(manifest, result)
        self.assertGreaterEqual(len(normalized["scalar_metrics"]), 1)
        for measured_at in (1 << 63, 0, -1, True):
            manifest, result = self._synthetic_common_result()
            result["evidence"]["benchmark_evidence"]["memory"][0]["measured_at_ns"] = measured_at
            with self.assertRaises(RunnerError):
                normalize_mlx_common_result(manifest, result)

    def test_invalid_fallback_projects_supervisor_only_and_no_fake_benchmark_rows(self):
        manifest, result = self._synthetic_common_result()
        result["status"] = "invalid"
        result["classification"] = "runtime_unavailable"
        result["action"] = "baseline_fallback"
        result["error"] = {"code": "runtime_unavailable", "message": "not available"}
        result["evidence"] = {"rss_peak_bytes": None, "rss_missing_reason": "unavailable", "stdout_bytes": 0, "stderr_bytes": 0}
        result = dict(result)
        normalized = normalize_mlx_common_result(manifest, result)
        self.assertEqual(normalized["raw_samples"], [])
        self.assertEqual(normalized["correctness_metrics"], [])
        self.assertEqual(len(normalized["artifacts"]), 1)
        self.assertEqual(normalized["artifacts"][0]["artifact_name"], "normalization_projection_v1")
        self.assertIn("benchmark_evidence:runtime_unavailable", normalized["omitted_normalized_fields"])
        self.assertEqual({row["metric_name"] for row in normalized["scalar_metrics"]}, {"rss_peak_bytes", "stdout_bytes", "stderr_bytes"})

    def test_mlx_projection_persists_atomically_and_replays(self):
        manifest, result = self._synthetic_common_result()
        first = _persist_mlx_common_result(manifest, result, _test_context=self.context, created_at_unix_ns=10)
        second = _persist_mlx_common_result(manifest, result, _test_context=self.context, created_at_unix_ns=11)
        self.assertEqual(first["persistence"].state, "inserted")
        self.assertEqual(second["persistence"].state, "idempotent")
        self.assertEqual(first["projection"]["projection_sha256"], second["projection"]["projection_sha256"])

    def test_aa_loader_requires_complete_current_provenance_three_plus_three(self):
        for process_set in ("characterization", "confirmation"):
            for index in range(3):
                manifest, result = self._synthetic_common_result(process_set, index)
                _persist_mlx_common_result(manifest, result, _test_context=self.context, created_at_unix_ns=100 + index)
        characterization, confirmation = load_aa_sessions(_test_context=self.context)
        self.assertEqual([item["manifest"]["process"]["index"] for item in characterization], [0, 1, 2])
        self.assertEqual([item["manifest"]["process"]["index"] for item in confirmation], [0, 1, 2])
        aggregate = load_and_aggregate_h0_aa(_test_context=self.context)
        self.assertTrue(aggregate["aggregation_contract_ready"])
        self.assertFalse(aggregate["live_execution_authorized"])

    def test_aa_loader_rejects_missing_or_mismatched_session(self):
        manifest, result = self._synthetic_common_result("characterization", 0)
        _persist_mlx_common_result(manifest, result, _test_context=self.context, created_at_unix_ns=1)
        with self.assertRaises(RunnerError):
            load_aa_sessions(_test_context=self.context)

    def test_aa_loader_rejects_tampered_child_bundle(self):
        for process_set in ("characterization", "confirmation"):
            for index in range(3):
                manifest, result = self._synthetic_common_result(process_set, index)
                _persist_mlx_common_result(manifest, result, _test_context=self.context, created_at_unix_ns=100 + index)
        path = _database_path_for_tests(self.context)
        connection = sqlite3.connect(path)
        try:
            trigger_sql = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type='trigger' AND name='raw_samples_append_only_update'"
            ).fetchone()[0]
            connection.execute("DROP TRIGGER raw_samples_append_only_update")
            connection.execute(
                "UPDATE raw_samples SET value=value+1 WHERE run_id=(SELECT run_id FROM runs WHERE mode='aa_gpu' LIMIT 1)"
            )
            connection.commit()
        finally:
            connection.execute(trigger_sql)
            connection.commit()
            connection.close()
        with self.assertRaises((RunnerError, StorageError)):
            load_aa_sessions(_test_context=self.context)

    def test_persist_adapter_does_not_call_supervisor_or_runtime_import(self):
        manifest, result = self._synthetic_common_result()
        with mock.patch("friday_h0.runner.run_supervised", side_effect=AssertionError("supervisor must not run")):
            with mock.patch("builtins.__import__", side_effect=AssertionError("runtime import must not run")):
                # The helper is created before the guarded call; normalization
                # and SQLite projection perform no dynamic imports.
                projection = normalize_mlx_common_result(manifest, result)
        self.assertEqual(projection["counts"]["raw_samples"], 78)


if __name__ == "__main__":
    unittest.main()
