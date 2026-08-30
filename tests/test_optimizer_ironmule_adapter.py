from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from friday_optimizer.fingerprint import ExactFingerprint, EnvironmentFingerprint, ModelFingerprint, WorkloadFingerprint
from friday_optimizer.ironmule_adapter import (
    CheckoutValidationError,
    IronMuleAdapterError,
    IronMuleCheckoutBinding,
    IronMuleTuneAdapter,
    ResultValidationError,
    UnsupportedStage,
)


def fingerprint() -> ExactFingerprint:
    return ExactFingerprint(
        EnvironmentFingerprint(chip="Apple M1 Max", gpu="Apple GPU", ram_bytes=32_000_000_000,
                                cpu_cores=10, macos="15.6", mlx="0.32.0", mlx_lm="0.31.3",
                                python="3.12", runtime_commit="r" * 40),
        ModelFingerprint(model_id="mlx-community/gemma-3-4b-it-4bit", revision="main",
                          manifest="a" * 64, architecture="gemma3", quant_bits=4,
                          quant_group_size=64, tokenizer="b" * 64),
        WorkloadFingerprint(prompt_family="default", tokenizer="b" * 64, generator="g",
                            context_bucket="short", batch=1, concurrency=1, max_tokens=32,
                            greedy=True, prompt_logprobs=False, power_mode="ac", mode="interactive"),
    )


class AdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "ironmule"
        self.root.mkdir()
        (self.root / "ironmule_cli.py").write_text("print('offline')\n", encoding="utf-8")
        (self.root / "ironmule").mkdir()
        (self.root / "ironmule" / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
        subprocess.run(["/usr/bin/git", "init", "-q"], cwd=self.root, check=True)
        subprocess.run(["/usr/bin/git", "config", "user.email", "test@example.invalid"], cwd=self.root, check=True)
        subprocess.run(["/usr/bin/git", "config", "user.name", "Test"], cwd=self.root, check=True)
        subprocess.run(["/usr/bin/git", "add", "ironmule_cli.py", "ironmule/__init__.py"], cwd=self.root, check=True)
        subprocess.run(["/usr/bin/git", "commit", "-q", "-m", "fixture"], cwd=self.root, check=True)
        self.head = subprocess.check_output(["/usr/bin/git", "rev-parse", "HEAD"], cwd=self.root, text=True).strip()
        self.binding = IronMuleCheckoutBinding._for_testing(
            checkout=self.root,
            expected_head=self.head,
            interpreter=os.path.realpath(sys.executable),
            fingerprint=fingerprint(),
            fixed_execution_files=("ironmule_cli.py", "ironmule/__init__.py"),
            forbidden_checkouts=(),
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def adapter(self) -> IronMuleTuneAdapter:
        return IronMuleTuneAdapter(self.binding)

    def test_valid_binding_is_immutable_and_doctor_is_offline(self) -> None:
        adapter = self.adapter()
        report = adapter.doctor()
        self.assertTrue(report["ok"])
        self.assertTrue(report["offline"])
        self.assertFalse(report["supports_promotion"])
        with self.assertRaises((AttributeError, TypeError)):
            self.binding.checkout = "changed"  # type: ignore[misc]

    def test_production_constructor_cannot_override_registry_or_forbidden_policy(self) -> None:
        with self.assertRaises(TypeError):
            IronMuleCheckoutBinding(  # type: ignore[call-arg]
                checkout=self.root, expected_head=self.head, interpreter=os.path.realpath(sys.executable),
                fingerprint=fingerprint(), fixed_execution_files=("ironmule_cli.py", "ironmule/__init__.py"),
            )
        self.assertEqual(self.binding.execution_registry_hash, self.binding.execution_registry_hash)
        self.assertEqual(self.binding.forbidden_checkouts, ())

    def test_wrong_head_dirty_and_source_mutation_fail_closed(self) -> None:
        wrong = IronMuleCheckoutBinding._for_testing(
            checkout=self.root, expected_head="0" * 40, interpreter=os.path.realpath(sys.executable),
            fingerprint=fingerprint(), fixed_execution_files=("ironmule_cli.py", "ironmule/__init__.py"), forbidden_checkouts=(),
        )
        with self.assertRaises(CheckoutValidationError):
            IronMuleTuneAdapter(wrong).validate_checkout()
        (self.root / "untracked").write_text("x", encoding="utf-8")
        with self.assertRaises(CheckoutValidationError):
            self.adapter().validate_checkout()

        healthy = self.adapter()
        self.assertTrue(healthy.doctor()["ok"] is False)
        (self.root / "untracked").unlink()
        (self.root / "ironmule_cli.py").write_text("changed\n", encoding="utf-8")
        with self.assertRaises(CheckoutValidationError):
            self.adapter().validate_checkout()

    def test_symlink_and_forbidden_checkout_fail_closed(self) -> None:
        link = Path(self.tmp.name) / "link"
        link.symlink_to(self.root, target_is_directory=True)
        linked = IronMuleCheckoutBinding._for_testing(
            checkout=link, expected_head=self.head, interpreter=os.path.realpath(sys.executable),
            fingerprint=fingerprint(), fixed_execution_files=("ironmule_cli.py", "ironmule/__init__.py"), forbidden_checkouts=(),
        )
        with self.assertRaises(CheckoutValidationError):
            IronMuleTuneAdapter(linked).validate_checkout()
        forbidden = IronMuleCheckoutBinding._for_testing(
            checkout=self.root, expected_head=self.head, interpreter=os.path.realpath(sys.executable),
            fingerprint=fingerprint(), fixed_execution_files=("ironmule_cli.py", "ironmule/__init__.py"), forbidden_checkouts=(str(self.root),),
        )
        with self.assertRaises(CheckoutValidationError):
            IronMuleTuneAdapter(forbidden).validate_checkout()

    def test_interpreter_replacement_is_detected(self) -> None:
        # The fixture uses the running interpreter; replacing its inode cannot
        # be done safely, so a deliberately wrong bound hash covers the same
        # fail-closed identity gate.
        bad = IronMuleCheckoutBinding._for_testing(
            checkout=self.root, expected_head=self.head, interpreter=os.path.realpath(sys.executable),
            fingerprint=fingerprint(), fixed_execution_files=("ironmule_cli.py", "ironmule/__init__.py"), forbidden_checkouts=(),
            interpreter_sha256="0" * 64,
        )
        with self.assertRaises(CheckoutValidationError):
            IronMuleTuneAdapter(bad).validate_checkout()

    def test_unknown_candidate_and_free_stage_are_rejected(self) -> None:
        adapter = self.adapter()
        with self.assertRaises(Exception):
            adapter.plan_stage("test", candidate_id="not-registered")
        with self.assertRaises(IronMuleAdapterError):
            adapter.plan_stage("test", candidate_id="baseline", parameters=["--free"])  # type: ignore[arg-type]
        with self.assertRaises(UnsupportedStage):
            adapter.plan_stage("activate")

    def test_non_hub_model_sources_block_before_staging(self) -> None:
        from dataclasses import replace
        for model_id in ("local:gemma", "/tmp/gemma", "gemma", "org/name/extra", "org\\name", "../name"):
            with self.subTest(model_id=model_id):
                model = replace(self.binding.fingerprint.model, model_id=model_id)
                bad_fingerprint = replace(self.binding.fingerprint, model=model)
                binding = IronMuleCheckoutBinding._for_testing(
                    checkout=self.root, expected_head=self.head, interpreter=os.path.realpath(sys.executable),
                    fingerprint=bad_fingerprint, fixed_execution_files=("ironmule_cli.py", "ironmule/__init__.py"), forbidden_checkouts=(),
                )
                adapter = IronMuleTuneAdapter(binding)
                with self.assertRaisesRegex(IronMuleAdapterError, "unsupported_model_source"):
                    adapter.plan_stage("test", candidate_id="combined_core_profile", qualified=("fixed_compiled_cache", "head_skip_prefill"))
                self.assertEqual(tuple(adapter._staged_specs), ())

    def test_candidates_without_a_real_cli_mapping_are_blocked(self) -> None:
        adapter = self.adapter()
        for candidate in ("persistent_process", "fixed_compiled_cache", "head_skip_prefill",
                          "readback_every_2", "throughput_width_2", "throughput_width_3",
                          "throughput_width_4", "baseline"):
            with self.assertRaises(UnsupportedStage):
                adapter.plan_stage("test", candidate_id=candidate)

    def test_stage_is_exact_offline_and_execution_is_blocked_by_default(self) -> None:
        adapter = self.adapter()
        baseline = adapter.plan_stage("status", candidate_id="baseline")
        spec = adapter.plan_stage("test", candidate_id="combined_core_profile",
                                  qualified=("fixed_compiled_cache", "head_skip_prefill"))
        self.assertEqual(spec.executable, os.path.realpath(sys.executable))
        self.assertTrue(spec.cwd.startswith("/var/"))
        self.assertTrue(spec.cwd.endswith("/" + Path(spec.cwd).name))
        self.assertNotEqual(spec.cwd, str(self.root))
        self.assertEqual(baseline.args[1], "status")
        self.assertEqual(spec.args[1], "tune")
        worker_path = Path(spec.cwd) / "friday_ironmule_stage_worker.py"
        real_worker = Path(__file__).resolve().parents[1] / "friday_optimizer" / "ironmule_stage_worker.py"
        self.assertEqual(worker_path.read_bytes(), real_worker.read_bytes())
        self.assertTrue(os.stat(worker_path).st_mode & 0o111)
        staged_spec = __import__("json").loads((Path(spec.cwd) / "stage_spec.json").read_text(encoding="utf-8"))
        self.assertLessEqual(staged_spec["limits"]["max_peak_memory_bytes"], 12 * 1024**3)
        self.assertLessEqual(staged_spec["limits"]["max_rss_bytes"], 12 * 1024**3)
        self.assertLessEqual(staged_spec["limits"]["max_peak_memory_bytes"], fingerprint().environment.ram_bytes)
        self.assertLessEqual(staged_spec["limits"]["max_rss_bytes"], fingerprint().environment.ram_bytes)
        calibration = adapter.plan_stage("calibrate", candidate_id="combined_core_profile",
                                         qualified=("fixed_compiled_cache", "head_skip_prefill"))
        self.assertIn("--no-confirm", calibration.args)
        self.assertNotIn("--no-confirm", spec.args)
        self.assertEqual(spec.candidate_id, "combined_core_profile")
        self.assertFalse(spec.execute_authorized)
        self.assertEqual(dict(spec.parameters), {})
        self.assertEqual(spec.env["HF_HUB_OFFLINE"], "1")  # type: ignore[index]
        self.assertNotIn("--force", spec.args)
        result = adapter.test(deadline=10**9)
        self.assertEqual(result.payload["status"], "blocked")
        self.assertEqual(result.reason, "stage_authorization_rejected")
        self.assertFalse(Path(spec.cwd).exists())
        baseline.cleanup()
        calibration.cleanup()

        staged = adapter.plan_stage("test", candidate_id="combined_core_profile",
                                    qualified=("fixed_compiled_cache", "head_skip_prefill"))
        blocked = adapter.run_stage(staged, deadline=10**9, session_id="s1")
        self.assertEqual(blocked.payload["status"], "blocked")
        adapter.cleanup()

    def test_staged_spec_is_a_cleanup_context(self) -> None:
        adapter = self.adapter()
        with adapter.plan_stage("status") as spec:
            staged = Path(spec.cwd)
            self.assertTrue(staged.is_dir())
            self.assertTrue((staged / "ironmule_cli.py").is_file())
            self.assertNotEqual(staged, self.root)
        self.assertFalse(staged.exists())

    def test_authorization_returns_new_immutable_spec(self) -> None:
        adapter = self.adapter()
        planned = adapter.plan_stage("test", candidate_id="combined_core_profile",
                                     qualified=("fixed_compiled_cache", "head_skip_prefill"))
        authorized = adapter.authorize_stage(planned, "s1")
        self.assertIsNot(planned, authorized)
        self.assertFalse(planned.execute_authorized)
        self.assertTrue(authorized.execute_authorized)
        self.assertEqual(authorized.authorization_session_id, "s1")
        self.assertTrue(authorized.authorization_nonce)
        adapter.cleanup()

    def test_authorization_is_session_bound_and_one_time(self) -> None:
        adapter = self.adapter()
        planned = adapter.plan_stage("test", candidate_id="combined_core_profile",
                                     qualified=("fixed_compiled_cache", "head_skip_prefill"))
        authorized = adapter.authorize_stage(planned, "s1")
        self.assertFalse(adapter.verify_and_consume_authorization(authorized, "s2"))
        self.assertTrue(adapter.verify_and_consume_authorization(authorized, "s1"))
        self.assertFalse(adapter.verify_and_consume_authorization(authorized, "s1"))
        adapter.cleanup()

    def test_staged_manifest_blocks_cli_mutation_before_runner(self) -> None:
        adapter = self.adapter()
        planned = adapter.plan_stage("test", candidate_id="combined_core_profile",
                                     qualified=("fixed_compiled_cache", "head_skip_prefill"))
        authorized = adapter.authorize_stage(planned, "s1")
        staged_file = Path(authorized.cwd) / "ironmule_cli.py"
        os.chmod(staged_file, 0o600)
        staged_file.write_text("tampered\n", encoding="utf-8")
        adapter._runner_for = lambda _cwd: (_ for _ in ()).throw(AssertionError("runner must not run"))  # type: ignore[method-assign]
        result = adapter.run_stage(authorized, deadline=10**9, session_id="s1")
        self.assertEqual(result.payload["status"], "blocked")
        self.assertFalse(adapter.verify_and_consume_authorization(authorized, "s1"))
        adapter.cleanup()

    def test_staged_manifest_blocks_remove_and_replace_without_consuming_nonce(self) -> None:
        for action in ("remove", "replace"):
            adapter = self.adapter()
            planned = adapter.plan_stage("test", candidate_id="combined_core_profile",
                                         qualified=("fixed_compiled_cache", "head_skip_prefill"))
            authorized = adapter.authorize_stage(planned, "s1")
            staged_file = Path(authorized.cwd) / ("ironmule_cli.py" if action == "remove" else "ironmule/__init__.py")
            if action == "remove":
                staged_file.unlink()
            else:
                staged_file.unlink()
                staged_file.write_text("replacement\n", encoding="utf-8")
            result = adapter.run_stage(authorized, deadline=10**9, session_id="s1")
            self.assertEqual(result.payload["status"], "blocked")
            self.assertFalse(adapter.verify_and_consume_authorization(authorized, "s1"))
            adapter.cleanup()

    def test_unchanged_manifest_authorizes_once(self) -> None:
        adapter = self.adapter()
        planned = adapter.plan_stage("test", candidate_id="combined_core_profile",
                                     qualified=("fixed_compiled_cache", "head_skip_prefill"))
        authorized = adapter.authorize_stage(planned, "s1")
        self.assertTrue(adapter.verify_and_consume_authorization(authorized, "s1"))
        self.assertFalse(adapter.verify_and_consume_authorization(authorized, "s1"))
        adapter.cleanup()

    def test_tampered_authorization_is_rejected(self) -> None:
        from dataclasses import replace
        adapter = self.adapter()
        planned = adapter.plan_stage("test", candidate_id="combined_core_profile",
                                     qualified=("fixed_compiled_cache", "head_skip_prefill"))
        authorized = adapter.authorize_stage(planned, "s1")
        tampered = replace(authorized, candidate_id="baseline")
        self.assertFalse(adapter.verify_and_consume_authorization(tampered, "s1"))
        self.assertTrue(adapter.verify_and_consume_authorization(authorized, "s1"))
        adapter.cleanup()

    def test_promotion_has_no_specs_and_always_blocks(self) -> None:
        adapter = self.adapter()
        self.assertFalse(adapter.supports_promotion)
        self.assertNotIn("activate", adapter.stage_specs)
        self.assertEqual(adapter.activate(deadline=10**9).payload["status"], "blocked")
        adapter.cleanup()

    def valid_result(self, *, confirmation: dict | None = None, screening: dict | None = None) -> dict:
        return {
            "schema": "friday.ironmule.result.v1", "stage": "test", "commit": self.head,
            "fingerprint": fingerprint().fingerprint_hash, "candidate": "baseline",
            "correctness": {"token_identity": True, "token_count": 32, "stop_reason": "eos",
                             "response_hash": "c" * 64},
            "resources": {"ttft_ms": 1.0, "decode_tokens_per_second": 10.0,
                          "peak_memory_bytes": 100, "peak_rss_bytes": 100,
                          "swap_delta_bytes": 0, "resource_gate_passed": True},
            "screening": screening,
            "confirmation": confirmation,
        }

    def test_parser_requires_exact_schema_identity_and_resources(self) -> None:
        adapter = self.adapter()
        with self.assertRaises(ResultValidationError):
            adapter.parse_result({"schema": "wrong"})
        malformed = self.valid_result()
        del malformed["resources"]
        with self.assertRaises(ResultValidationError):
            adapter.parse_result(malformed)
        stale = self.valid_result()
        stale["commit"] = "f" * 40
        with self.assertRaises(ResultValidationError):
            adapter.parse_result(stale)

    def test_screening_does_not_count_as_confirmation(self) -> None:
        adapter = self.adapter()
        parsed = adapter.parse_result(self.valid_result(screening={"ratio": 0.5}))
        self.assertFalse(parsed.confirmed)
        self.assertIsNone(parsed.selected_ratio)

        envelope = {"outcome": "inconclusive", "reason": "screening_only",
                    "payload": self.valid_result(screening={"ratio": 0.5})}
        self.assertFalse(adapter.parse_result(envelope)["confirmed"])

    def test_q2_confirmation_ratio_is_the_only_selected_ratio(self) -> None:
        adapter = self.adapter()
        parsed = adapter.parse_result(self.valid_result(
            screening={"ratio": {"median_ratio": 0.5}},
            confirmation={"confirmed": True, "token_identity": True, "pair_count": 3,
                          "ratio": {"median_ratio": 0.9, "ci_low": 0.8, "ci_high": 0.95},
                          "resources": {"ttft_ms": 1.0, "decode_tokens_per_second": 10.0,
                                        "peak_memory_bytes": 100, "peak_rss_bytes": 100,
                                        "swap_delta_bytes": -1, "resource_gate_passed": True}},
        ))
        self.assertTrue(parsed.confirmed)
        self.assertEqual(parsed.selected_ratio, 0.9)

    def test_confirmation_cannot_use_top_level_token_or_resource_fallback(self) -> None:
        adapter = self.adapter()
        value = self.valid_result(confirmation={"confirmed": True, "pair_count": 3,
                                                  "ratio": {"median_ratio": 0.9, "ci_low": 0.8, "ci_high": 0.95},
                                                  "resources": self.valid_result()["resources"]})
        value["correctness"].pop("token_identity")
        value["token_identity"] = True
        with self.assertRaises(ResultValidationError):
            adapter.parse_result(value)

    def test_actual_tune_confirmation_shape_uses_phase_pairs_and_worker_resources(self) -> None:
        adapter = self.adapter()
        phase = lambda ratio: {"median_ratio": ratio, "ci_low": ratio - 0.01,
                               "ci_high": ratio + 0.01, "pairs": [ratio, ratio + 0.01, ratio - 0.01]}
        value = self.valid_result()
        value["candidate"] = "combined_core_profile"
        value["source_digest"] = self.binding.source_digest
        value["registry_hash"] = self.binding.execution_registry_hash
        value["worker_sha256"] = "e" * 64
        value["confirmation"] = {
            "ratio": {"total_ns": phase(0.9), "prefill_ns": phase(0.95), "decode_ns": phase(0.88)},
            "token_identity": True,
        }
        parsed = adapter.parse_result(value, candidate_id="combined_core_profile")
        self.assertTrue(parsed.confirmed)
        self.assertEqual(parsed.selected_ratio, 0.9)
        self.assertEqual(len(parsed.confirmation["ratio"]["decode_ns"]["pairs"]), 3)  # type: ignore[index]

    def test_git_policy_is_explicitly_isolated_from_repo_fsmonitor(self) -> None:
        marker = self.root / "fsmonitor-ran"
        config = self.root / ".git" / "config"
        with config.open("a", encoding="utf-8") as stream:
            stream.write("\n[core]\n\tfsmonitor = !touch " + str(marker) + "\n")
        self.assertTrue(self.adapter().doctor()["ok"])
        self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
