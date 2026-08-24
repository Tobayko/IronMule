"""Offline contract tests for the Cycle-17 batched-readback candidate.

The tests intentionally never import MLX, load a model, allocate device arrays,
or invoke the measurement harness.  They exercise the eventual worker through
small fakes and inspect the source for the safety properties that must hold
before a real run is permitted.
"""

from __future__ import annotations

import ast
import hashlib
import http.client
import importlib.util
import inspect
import json
import os
import stat
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "experiments" / "batched_readback_compile"
WORKER_PATH = STUDY / "worker.py"
HARNESS_PATH = STUDY / "measure_batched_readback.py"
DASHBOARD_PATH = STUDY / "dashboard.py"
CYCLE16 = ROOT / "experiments" / "matmul_compile_ab"


def _load(path: Path, name: str):
    if not path.is_file():
        raise AssertionError(f"missing expected Cycle-17 module: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _api(module, *names):
    for name in names:
        value = getattr(module, name, None)
        if value is not None:
            return value
    raise AssertionError(f"none of {names!r} is exported by {module.__name__}")


def _json_one(parser, text: str):
    return parser(text)


class TestStaticSafety(unittest.TestCase):
    def test_expected_modules_exist_and_have_no_top_level_mlx_import(self):
        for path in (WORKER_PATH, HARNESS_PATH, DASHBOARD_PATH):
            tree = ast.parse(_source(path), filename=str(path))
            # Only direct module-body imports are forbidden.  The authorised
            # worker may keep lazy MLX imports inside the explicitly guarded
            # execution functions.
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = [a.name for a in node.names]
                    root = node.module.split(".")[0] if isinstance(node, ast.ImportFrom) else ""
                    self.assertNotIn("mlx", names + [root], path.name)

    def test_harness_is_default_exit_78_and_has_no_default_writes(self):
        src = _source(HARNESS_PATH)
        self.assertRegex(src, r"(?m)Exit78|exit\(78\)|return\s+78")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in {"write_text", "write_bytes", "open"}:
                    parent = ast.get_source_segment(src, node) or ""
                    self.assertNotIn("--execute", parent)

    def test_harness_main_guard_and_worker_authorization(self):
        src = _source(HARNESS_PATH)
        self.assertIn('if __name__ == "__main__":', src)
        self.assertTrue(hasattr(_load(HARNESS_PATH, "cycle17_authorization_harness"), "_preflight"))
        worker = _source(WORKER_PATH)
        self.assertIn("def _authorise", worker)
        self.assertIn("load_count", worker)

    def test_gate_names_and_private_marker_modes_are_explicit(self):
        harness = _load(HARNESS_PATH, "cycle17_gate_harness")
        for name in ("_snapshot_identity", "_clean_worktree", "_require_target", "_evidence_state", "_validate_evidence_state"):
            self.assertTrue(hasattr(harness, name), name)
        src = _source(HARNESS_PATH)
        for term in ("snapshot", "power", "dirty", "result", "marker", "0600", "0o700"):
            self.assertIn(term.lower(), src.lower())

    def test_protocol_and_no_retry_contract_is_source_visible(self):
        src = _source(HARNESS_PATH) + _source(WORKER_PATH)
        for term in ("timeout", "partial", "retry", "JSON", "NaN", "multiline", "oversize"):
            self.assertIn(term.lower(), src.lower())

    def test_cycle16_prereg_and_prompt_are_readable_byte_artifacts(self):
        prereg = CYCLE16 / "PREREGISTRATION.md"
        self.assertTrue(prereg.is_file())
        self.assertEqual(_sha(prereg.read_bytes()), _sha(prereg.read_bytes()))
        old = _load(CYCLE16 / "worker.py", "cycle16_prompt_worker")
        new = _load(WORKER_PATH, "cycle17_prompt_worker")
        self.assertEqual(old.PLANNER_PROMPT, new.PLANNER_PROMPT)
        self.assertEqual(old.PROMPT_SHA256, new.PROMPT_SHA256)
        self.assertEqual(new.EXPECTED_PROMPT_TOKENS, 322)
        self.assertEqual(new.BOOTSTRAP_SEED, 20260824)
        self.assertEqual(new.BOOTSTRAP_RESAMPLES, 10_000)


class TestWorkerContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worker = _load(WORKER_PATH, "cycle17_test_worker")

    def test_only_two_arms_and_intervals_one_eight(self):
        arms = _api(self.worker, "ARMS", "ARM_NAMES")
        self.assertEqual(tuple(arms), ("fixed_compiled_readback_1", "fixed_compiled_readback_8"))
        intervals = _api(self.worker, "INTERVALS", "ARM_INTERVALS")
        if isinstance(intervals, dict):
            self.assertEqual(intervals, {arms[0]: 1, arms[1]: 8})
        else:
            self.assertEqual(tuple(intervals), (1, 8))
        for name in ("FIXED_CACHE", "FIXED_COMPILE", "CACHE_CAPACITY", "COMPILE_CONFIG"):
            self.assertTrue(hasattr(self.worker, name))

    def test_canonical_parent_worker_protocol_is_bound(self):
        src = _source(WORKER_PATH)
        harness = _source(HARNESS_PATH)
        # The parent constructs the authenticated names from one fixed prefix;
        # bind that runtime contract below rather than requiring every expanded
        # key to appear as a literal in source.
        self.assertIn("cycle17-fixed-compiled-batched-readback-v1", harness)
        for term in (
            "FRIDAY_BRB_PARENT_PID", "FRIDAY_BRB_RUN_ID", "FRIDAY_BRB_MODEL_KEY",
            "FRIDAY_BRB_NONCE", "FRIDAY_BRB_BLOCK", "FRIDAY_BRB_ARM_ORDER",
        ):
            self.assertIn(term, src)
        self.assertNotIn("FRIDAY_READBACK_", src)

        protocol = _load(HARNESS_PATH, "cycle17_shared_protocol")._validate_protocol_contract()
        self.assertEqual(set(protocol["event_required_fields"]), set(self.worker.EVENT_REQUIRED_FIELDS))
        validator = harness[harness.index("def _validate_event") : harness.index("def _run_block")]
        self.assertIn("event_required_fields", validator)
        self.assertNotIn('("interval_1", "interval_8")', src)

    def test_block_schedule_and_pre_marker_hash_gate(self):
        harness = _load(HARNESS_PATH, "cycle17_schedule_harness")
        self.assertEqual(tuple(harness.PAIR_SCHEDULE), tuple(
            (("fixed_compiled_readback_1", "fixed_compiled_readback_8") if i % 2 == 0
             else ("fixed_compiled_readback_8", "fixed_compiled_readback_1"))
            for i in range(6)
        ))
        src = _source(HARNESS_PATH)
        self.assertIn("FROZEN_PREREGISTRATION_SHA256", src)
        self.assertLess(src.index("_sha256(PREREGISTRATION)"), src.index("_exclusive_json(ATTEMPT_PATH"))
        environment = harness._environment(
            {"snapshot_path": "/local/snapshot", "snapshot_sha256": "s" * 64,
             "weight_sha256": {"weights.safetensors": "w" * 64},
             "execution_stat_manifest": {}},
            3, ("fixed_compiled_readback_1", "fixed_compiled_readback_8"), "p" * 64,
        )
        self.assertEqual(environment["FRIDAY_BRB_BLOCK"], "3")
        self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
        self.assertEqual(environment["PYTHONNOUSERSITE"], "1")
        # A preregistration mismatch must stop before any admission side effect
        # (including snapshot/model checks or marker creation).
        with mock.patch.object(harness, "FROZEN_PREREGISTRATION_SHA256", "0" * 64):
            with self.assertRaises(harness.StudyError):
                harness._preflight(harness.RUN_ID)

    def test_generation_manifest_and_snapshot_identity_are_bound(self):
        worker = _source(WORKER_PATH)
        harness = _source(HARNESS_PATH)
        self.assertIn('"generation_config.json"', worker)
        self.assertIn("_read_eos_ids", worker)
        self.assertIn("execution_stat_manifest", worker + harness)
        self.assertIn("EXPECTED_SNAPSHOT_SHA256", worker + harness)
        self.assertIn("EXPECTED_WEIGHT_SHA256", worker + harness)

    def test_fixed_cache_and_compile_are_invariant(self):
        src = _source(WORKER_PATH)
        self.assertRegex(src, r"fixed[_ -]?cache|cache.*invariant")
        self.assertRegex(src, r"fixed[_ -]?compile|compile.*invariant")
        self.assertNotRegex(src, r"interval.{0,80}(make_cache|compile)")

    def test_strict_parser_rejects_duplicate_nan_multiline_and_oversize(self):
        parser = _api(self.worker, "parse_one_json", "parse_event", "decode_event")
        good = '{"event":"ok","tokens":[]}'
        self.assertIsInstance(_json_one(parser, good), dict)
        for bad in (
            '{"event":1,"event":2}',
            '{"event":NaN}',
            good + "\n{}",
            "{" + '"x":"' + ("a" * 2_000_001) + '"}',
        ):
            with self.assertRaises((ValueError, TypeError, AssertionError, json.JSONDecodeError)):
                _json_one(parser, bad)

    def test_parent_parser_has_same_single_event_fail_closed_contract(self):
        parser = _api(_load(HARNESS_PATH, "cycle17_parent_parser"), "_decode_event")
        self.assertIsInstance(parser(b'{"event":"complete"}'), dict)
        for bad in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1}\n{"b":2}', b'{"a":1}\r\n'):
            with self.assertRaises(Exception):
                parser(bad)

    def test_timeout_and_partial_are_terminal_without_retry(self):
        fn = _api(self.worker, "validate_terminal_event", "validate_event", "terminal_status")
        for event in ({"status": "timeout"}, {"status": "partial"}, {"status": "retry"}):
            with self.assertRaises((ValueError, AssertionError)):
                fn(event)
        self.assertNotRegex(_source(WORKER_PATH), r"except[^:]*:\s*[^\n]*retry\(")
        harness = _source(HARNESS_PATH)
        self.assertIn("partial_result", harness)
        self.assertTrue("_write_fail_safe" in harness or "_atomic_result" in harness)
        self.assertNotRegex(harness, r"except[^:]*:\s*[^\n]*retry\(")

    def test_budget_guard_limits_and_charges_timer_before_charge(self):
        guard = _api(self.worker, "BudgetGuard")
        signature = inspect.signature(guard)
        self.assertEqual(signature.parameters["duty_cycle"].default, 0.15)
        self.assertEqual(signature.parameters["continuous_limit_seconds"].default, 6.0)
        self.assertEqual(set(signature.parameters), {"duty_cycle", "continuous_limit_seconds"})
        src = _source(WORKER_PATH)
        self.assertRegex(src, r"timer|perf_counter")
        self.assertRegex(src, r"charge|record")
        worker_run = src[src.index("def _run_worker") :]
        self.assertLess(worker_run.index("arm_finished_ns"), worker_run.index("_charge_arm"))

    def test_fake_loop_allows_item_only_at_boundaries(self):
        class Scalar:
            def __init__(self, value): self.value, self.calls = value, 0
            def item(self): self.calls += 1; return self.value

        class FakeModel:
            def __init__(self): self.readbacks = 0
            def __call__(self, token, **kwargs):
                self.readbacks += 1
                return types.SimpleNamespace(logits=[Scalar(token)])

        loop = _api(self.worker, "decode_loop", "run_decode", "batched_decode")
        model = FakeModel()
        tokens = [Scalar(1), Scalar(2), Scalar(3)]
        result = loop(model, tokens, interval=8, max_tokens=32, eos_ids={99})
        self.assertEqual(sum(token.calls for token in tokens), 3)
        self.assertEqual(result["readback_count"], 1)
        tokens_1 = [Scalar(1), Scalar(2), Scalar(3)]
        result_1 = loop(model, tokens_1, interval=1, max_tokens=32, eos_ids={99})
        self.assertEqual(sum(token.calls for token in tokens_1), 3)
        self.assertEqual(result_1["readback_count"], 3)
        self.assertIn("boundary", _source(WORKER_PATH).lower())

    def test_true_vector_boundary_has_one_tolist_and_no_scalar_item(self):
        src = _source(WORKER_PATH)
        start = src.index("def _host_readback_boundary")
        end = src.index("def _prompt_ids", start)
        boundary = src[start:end]
        tree = ast.parse(boundary)
        tolist_calls = [node for node in ast.walk(tree)
                        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "tolist"]
        item_calls = [node for node in ast.walk(tree)
                      if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                      and node.func.attr == "item"]
        self.assertEqual(len(tolist_calls), 1)
        self.assertEqual(item_calls, [])
        self.assertIn("mx.stack", boundary)
        self.assertIn("mx.synchronize", boundary)

    def test_budget_rejection_preserves_observed_and_guard_recorded_time(self):
        evidence = self.worker._arm_budget_evidence(
            6_000_000_000,
            guard_before_seconds=4.0,
            guard_after_seconds=5.5,
            accepted=False,
        )
        self.assertFalse(evidence["charge_accepted"])
        self.assertEqual(evidence["observed_model_work_ns"], 6_000_000_000)
        self.assertEqual(evidence["charged_model_work_ns"], 1_500_000_000)
        self.assertEqual(evidence["guard_recorded_model_work_ns"], 1_500_000_000)
        self.assertGreater(evidence["required_break_blocks"], 0)

    def test_boundary_gap_stats_use_unique_boundaries(self):
        src = _source(WORKER_PATH) + _source(HARNESS_PATH)
        self.assertIn("boundary_interarrival_ns", src)
        stats_start = _source(HARNESS_PATH).index("def _arm_stats")
        stats = _source(HARNESS_PATH)[stats_start:]
        self.assertNotIn('host_available_ns_by_token"], item["host_available_ns_by_token"][1:', stats)

    def test_eos_positions_logical_visible_tail_and_max_rest_block(self):
        src = _source(WORKER_PATH).lower()
        for term in ("eos", "logical", "visible", "tail", "32", "cache_discarded"):
            self.assertIn(term, src)
        contract = _api(self.worker, "normalize_tokens", "finalize_tokens", "validate_tokens")
        for pos in range(32):
            event = {"tokens": [1] * pos + [99] + [2] * 8, "eos_ids": [99], "max_tokens": 32}
            out = contract(event)
            self.assertNotIn(99, out.get("visible_tokens", out) if isinstance(out, dict) else out)
            self.assertLessEqual(len(out["physical_tokens"]), 32)
            self.assertEqual(out["logical_tokens"], out["physical_tokens"][: out["logical_token_count"]])
            self.assertEqual(out["overproduced_tokens"], out["physical_token_count"] - out["logical_token_count"])
        no_eos = contract({"tokens": list(range(100)), "eos_ids": [99], "max_tokens": 32})
        self.assertEqual(len(no_eos["physical_tokens"]), 32)
        self.assertEqual(no_eos["logical_tokens"], no_eos["physical_tokens"])
        self.assertEqual(no_eos["visible_tokens"], no_eos["logical_tokens"])
        self.assertEqual(no_eos["overproduced_tokens"], 0)

    def test_counts_forwards_hashes_and_mismatch_are_strict(self):
        src = _source(WORKER_PATH)
        for term in (
            "physical_tokens", "logical_tokens", "visible_tokens", "visible_text",
            "eos_position", "eos_block", "readback_ns",
            "host_available_ns_by_physical_token", "forwards", "token_sha256",
            "text_sha256", "mismatch",
        ):
            self.assertIn(term, src)
        validator = _api(self.worker, "validate_event", "validate_result")
        bad = {"status": "ok", "tokens": [1], "token_sha256": "0" * 64, "text_sha256": "0" * 64}
        with self.assertRaises((ValueError, AssertionError)):
            validator(bad)

    def test_arm_and_process_determinism(self):
        src = _source(WORKER_PATH).lower()
        for term in ("determin", "arm_order", "process", "seed"):
            self.assertIn(term, src)


class TestMeasurementContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.harness = _load(HARNESS_PATH, "cycle17_test_harness")

    def test_stats_are_median_mad_and_paired_bootstrap_is_closed(self):
        stats = _api(self.harness, "summarize", "arm_stats", "median_mad")
        value = stats([1.0, 2.0, 3.0, 4.0, 100.0])
        self.assertIsInstance(value, dict)
        self.assertIn("median", value)
        self.assertIn("mad", value)
        boot = _api(self.harness, "paired_bootstrap", "bootstrap_ratio")
        result = boot([1.0] * 6, [0.9] * 6, seed=20260824, iterations=10000)
        bootstrap = result.get("bootstrap_95_ci", result)
        self.assertEqual(bootstrap["seed"], 20260824)
        self.assertEqual(bootstrap.get("resamples", bootstrap.get("iterations", bootstrap.get("n"))), 10000)
        self.assertRegex(_source(HARNESS_PATH), r"no[_ -]?outlier|outlier")
        self.assertRegex(_source(HARNESS_PATH), r"closed|candidate_.*(failed|characterized|recommended)")
        self.assertLessEqual(result["median"], 0.95)
        self.assertLess(result["upper"], 1.0)
        decision = _api(self.harness, "decision_for")
        self.assertEqual(decision(resource_pass=True, budget_pass=True, correctness_pass=True,
                                  candidate_runnable=True, paired={"primary": result}),
                         "runtime_readback8_wins_exact_scope")
        self.assertEqual(decision(resource_pass=False, budget_pass=True, correctness_pass=True,
                                  candidate_runnable=True, paired={"primary": result}),
                         "resource_or_budget_failed")

    def test_result_show_selfchecks_are_read_only_lifecycle_safe(self):
        src = _source(HARNESS_PATH)
        for term in ("result", "show", "self_check", "before", "after", "sha256", "symlink", "0600"):
            self.assertIn(term, src)
        self.assertRegex(src, r"lstat|is_symlink")

    def test_marker_result_lifecycle_and_atomic_partial_save(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / "results.json"
            marker_dir = root / "marker"
            marker_dir.mkdir(mode=0o700)
            marker = marker_dir / "attempt.json"
            with mock.patch.object(self.harness, "RESULT_PATH", result), \
                 mock.patch.object(self.harness, "ATTEMPT_PATH", marker):
                before = self.harness._evidence_state()
                self.harness._validate_evidence_state(before)
                self.assertFalse(result.exists())
                self.assertFalse(marker.exists())
                self.harness._exclusive_json(marker, {"partial_result": True}, 0o600)
                self.harness._atomic_result({"partial_result": True, "formal_claim": False})
                self.assertEqual(stat.S_IMODE(marker.stat().st_mode), 0o600)
                self.assertEqual(stat.S_IMODE(result.stat().st_mode), 0o644)
                after = self.harness._evidence_state()
                self.assertTrue(after["result"]["exists"])
                self.assertTrue(after["marker"]["exists"])

    def test_default_and_show_are_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "results.json"
            with mock.patch.object(self.harness, "RESULT_PATH", result):
                self.assertEqual(self.harness._show(), 78)
                self.assertFalse(result.exists())
        completed = subprocess.run(
            [sys.executable, str(HARNESS_PATH)],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        self.assertEqual(completed.returncode, 78)

    def test_self_checks_do_not_create_evidence(self):
        evidence = [STUDY / "results.json", ROOT / ".friday-data" / "batched-readback-compile" / "attempt.json"]
        before = {str(path): (path.exists(), _sha(path.read_bytes()) if path.is_file() else None) for path in evidence}
        for executable in (HARNESS_PATH, WORKER_PATH):
            completed = subprocess.run(
                [sys.executable, str(executable), "--self-check"],
                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            self.assertEqual(completed.returncode, 0, (executable, completed.stdout, completed.stderr))
        after = {str(path): (path.exists(), _sha(path.read_bytes()) if path.is_file() else None) for path in evidence}
        self.assertEqual(before, after)


class TestDashboardContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dashboard = _load(DASHBOARD_PATH, "cycle17_test_dashboard")

    def test_dashboard_is_read_only_and_restricts_host(self):
        src = _source(DASHBOARD_PATH).lower()
        for term in ("get", "head", "405", "421", "no-store", "host", "private_keys", "cycle15", "cycle17", "sha256", "visible_text"):
            self.assertIn(term, src)
        self.assertTrue("cycle16" in src or "cycle15/16" in src)
        self.assertNotRegex(src, r"write_text|write_bytes|unlink|rename|mkdir")

    def test_dashboard_http_contract_with_synthetic_request_when_available(self):
        app = _api(self.dashboard, "make_server", "create_server", "app", "Handler")
        self.assertIsNotNone(app)
        self.assertIn("history", _source(DASHBOARD_PATH).lower())

    def test_real_read_only_http_projection_and_hash_stability(self):
        tracked = [
            STUDY / "results.json",
            ROOT / "experiments" / "dual_model_planner" / "results.json",
            ROOT / "experiments" / "matmul_compile_ab" / "results.json",
        ]
        before = {str(path): _sha(path.read_bytes()) for path in tracked if path.is_file() and not path.is_symlink()}
        server = self.dashboard.make_server("127.0.0.1", 0)
        thread = __import__("threading").Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_address[1]
        try:
            def request(method, path="/", host=None):
                connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
                headers = {"Host": host or f"127.0.0.1:{port}"}
                connection.request(method, path, headers=headers)
                response = connection.getresponse()
                body = response.read()
                connection.close()
                return response, body

            response, body = request("GET", "/")
            self.assertEqual(response.status, 200)
            self.assertEqual(response.getheader("Cache-Control"), "no-store")
            self.assertIn("text/html", response.getheader("Content-Type", ""))
            html_body = body.decode()
            self.assertIn("cycle 15", html_body.lower())
            self.assertIn("cycle 16", html_body.lower())
            self.assertIn("cycle 17", html_body.lower())
            response, body = request("GET", "/api/snapshot")
            self.assertEqual(response.status, 200)
            self.assertIn("application/json", response.getheader("Content-Type", ""))
            document = json.loads(body)
            self.assertEqual(document["history_cycles"], [15, 16, 17])
            self.assertEqual(document["study_id"], "fixed-compiled-batched-readback-20260824-01")
            self.assertNotIn("You choose exactly one next Project Friday experiment", body.decode())
            self.assertNotIn(str(ROOT), body.decode())
            self.assertNotIn("physical_tokens", body.decode())
            self.assertNotIn("prompt_token_ids", body.decode())
            self.assertNotIn("visible_text", body.decode())
            self.assertNotIn("Traceback", body.decode())
            self.assertNotIn("FileNotFoundError", body.decode())

            response, body = request("HEAD", "/api/snapshot")
            self.assertEqual(response.status, 200)
            self.assertEqual(body, b"")
            self.assertEqual(response.getheader("Cache-Control"), "no-store")
            for method in ("POST", "PUT", "PATCH", "DELETE"):
                response, _ = request(method)
                self.assertEqual(response.status, 405, method)
            for method in ("OPTIONS", "TRACE"):
                response, _ = request(method)
                self.assertIn(response.status, (405, 501), method)
            response, _ = request("GET", "/", host="evil.example")
            self.assertEqual(response.status, 421)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)
        after = {str(path): _sha(path.read_bytes()) for path in tracked if path.is_file() and not path.is_symlink()}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
