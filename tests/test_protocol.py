import json
import math
import os
import stat
import tempfile
import unittest
from pathlib import Path

from friday_h0.constants import AA_BOOTSTRAP_SEEDS, AA_SESSION_SEEDS
from friday_h0.protocol import (
    PRODUCTION_JSON_DEPTH,
    ProtocolError,
    close_manifest,
    parse_capped_json,
    read_capped_json,
    validate_result,
    write_json_atomic,
)


def valid_manifest(mode="eager_baseline", *, shape=2048, process_set="characterization", process_index=0):
    if mode == "aa_gpu":
        fixture = AA_SESSION_SEEDS[f"{process_set}_fixture"] + process_index
        order = AA_SESSION_SEEDS[f"{process_set}_order"] + process_index
    else:
        fixture = 0xF17A2026 + process_index
        order = 0xB10C2026 + process_index
    if mode.startswith("analysis_"):
        process_set, process_index = "analysis", 0
        fixture, order = 0, 0
    if mode.startswith("control_"):
        process_set, process_index = "control", 0
        fixture, order = 0, 0
    if mode == "analysis_wrong_fixture":
        shape, fixture = 64, 0xBAD02026
    return {
        "schema_version": 1,
        "phase": "H0",
        "run_id": "run-001",
        "mode": mode,
        "workload": {
            "operation": "matmul",
            "a_shape": [shape, shape],
            "b_shape": [shape, shape],
            "y_shape": [shape, shape],
            "dtype": "float16",
            "layout": "C-contiguous",
            "generator": "PCG64",
            "distribution": "uniform[-1,1)",
        },
        "seeds": {
            "fixture": fixture,
            "order": order,
            **({"bootstrap_seed": AA_BOOTSTRAP_SEEDS[process_set]} if mode == "aa_gpu" else {}),
        },
        "limits": {"first_eval_s": 10, "synchronize_s": 5, "total_s": 120},
        "process": {"set": process_set, "index": process_index},
        "provenance": {
            "code_sha256": "a" * 64,
            "spec_sha256": "b" * 64,
            "environment_sha256": "c" * 64,
            "revision": {"value": None, "missing_reason": "project root is not a Git repository"},
        },
    }


class ProtocolTests(unittest.TestCase):
    def test_strict_json_rejects_attacks(self):
        with self.assertRaises(ProtocolError):
            parse_capped_json(b'{"x":1,"x":2}', limit=100)
        with self.assertRaises(ProtocolError):
            parse_capped_json(b'{"x":NaN}', limit=100)
        with self.assertRaises(ProtocolError):
            parse_capped_json(b'{"x":1} trailing', limit=100)
        with self.assertRaises(ProtocolError):
            parse_capped_json(b"\xff", limit=100)
        deep = value = {}
        for _ in range(PRODUCTION_JSON_DEPTH + 2):
            value["x"] = {}
            value = value["x"]
        with self.assertRaises(ProtocolError):
            parse_capped_json(json.dumps(deep).encode(), limit=10000)
        with self.assertRaises(ProtocolError):
            parse_capped_json(b"{}", limit=1)

    def test_manifest_is_closed_and_replayable(self):
        first = close_manifest(valid_manifest())
        second = close_manifest(valid_manifest())
        self.assertEqual(first.canonical_bytes, second.canonical_bytes)
        self.assertEqual(first.sha256, second.sha256)
        value = first.value
        value["run_id"] = "mutated"
        self.assertEqual(first.run_id, "run-001")

    def test_atomic_regular_file_and_symlink_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "result.json"
            write_json_atomic(path, {"ok": True}, limit=1024)
            value, payload = read_capped_json(path, limit=1024)
            self.assertEqual(value, {"ok": True})
            self.assertEqual(payload, b'{"ok":true}')
            self.assertEqual(stat.S_IMODE(os.lstat(path).st_mode), 0o600)
            path.unlink()
            path.symlink_to(root / "missing")
            with self.assertRaises(ProtocolError):
                read_capped_json(path, limit=1024)

    def test_result_binds_hash_and_allowlists_fields(self):
        manifest = close_manifest(valid_manifest("analysis_known_win"))
        result = {
            "schema_version": 1,
            "run_id": manifest.run_id,
            "mode": manifest.mode,
            "manifest_sha256": manifest.sha256,
            "status": "completed",
            "classification": "promoted",
            "action": "promoted",
            "error": None,
            "evidence": {"rss_peak_bytes": None, "rss_missing_reason": "test"},
        }
        self.assertEqual(validate_result(result, manifest=manifest)["action"], "promoted")
        result["extra"] = True
        with self.assertRaises(ProtocolError):
            validate_result(result, manifest=manifest)

    def test_measurement_complete_is_neutral_and_h0_only(self):
        for mode in ("eager_baseline", "compile_comparison", "aa_gpu"):
            manifest = close_manifest(valid_manifest(mode))
            result = {
                "schema_version": 1,
                "run_id": manifest.run_id,
                "mode": manifest.mode,
                "manifest_sha256": manifest.sha256,
                "status": "completed",
                "classification": "measurement_complete",
                "action": "baseline_fallback",
                "error": None,
                "evidence": {"measurement_only": True},
            }
            validated = validate_result(result, manifest=manifest)
            self.assertEqual(validated["classification"], "measurement_complete")
            self.assertEqual(validated["action"], "baseline_fallback")
            self.assertEqual(set(validated), {
                "schema_version", "run_id", "mode", "manifest_sha256", "status",
                "classification", "action", "error", "evidence",
            })

            for invalid in (
                {**result, "status": "invalid", "error": {"code": "x", "message": "x"}},
                {**result, "action": "not_run"},
                {**result, "error": {"code": "x", "message": "x"}},
            ):
                with self.assertRaises(ProtocolError):
                    validate_result(invalid, manifest=manifest)

        for mode in ("analysis_known_win", "control_timeout"):
            manifest = close_manifest(valid_manifest(mode))
            result = {
                "schema_version": 1,
                "run_id": manifest.run_id,
                "mode": manifest.mode,
                "manifest_sha256": manifest.sha256,
                "status": "completed",
                "classification": "measurement_complete",
                "action": "baseline_fallback",
                "error": None,
                "evidence": {},
            }
            with self.assertRaises(ProtocolError):
                validate_result(result, manifest=manifest)

    def test_existing_promoted_regression_and_invalid_invariants_remain(self):
        promoted_manifest = close_manifest(valid_manifest("analysis_known_win"))
        promoted = {
            "schema_version": 1,
            "run_id": promoted_manifest.run_id,
            "mode": promoted_manifest.mode,
            "manifest_sha256": promoted_manifest.sha256,
            "status": "completed",
            "classification": "promoted",
            "action": "promoted",
            "error": None,
            "evidence": {},
        }
        self.assertEqual(validate_result(promoted, manifest=promoted_manifest)["classification"], "promoted")

        regression_manifest = close_manifest(valid_manifest("eager_baseline"))
        regression = {
            **promoted,
            "run_id": regression_manifest.run_id,
            "mode": regression_manifest.mode,
            "manifest_sha256": regression_manifest.sha256,
            "classification": "regression",
            "action": "baseline_fallback",
        }
        self.assertEqual(validate_result(regression, manifest=regression_manifest)["classification"], "regression")

        invalid = {
            **regression,
            "status": "invalid",
            "classification": "invalid",
            "error": {"code": "correctness_failed", "message": "correctness failed"},
        }
        self.assertEqual(validate_result(invalid, manifest=regression_manifest)["classification"], "invalid")

    def test_result_file_and_combination_failures_are_fail_closed(self):
        manifest = close_manifest(valid_manifest("analysis_known_win"))
        result = {
            "schema_version": 1,
            "run_id": manifest.run_id,
            "mode": manifest.mode,
            "manifest_sha256": manifest.sha256,
            "status": "completed",
            "classification": "promoted",
            "action": "promoted",
            "error": None,
            "evidence": {},
        }
        with self.assertRaises(ProtocolError):
            validate_result({**result, "schema_version": 2}, manifest=manifest)
        with self.assertRaises(ProtocolError):
            validate_result({**result, "status": "invalid", "error": {"code": "x", "message": "x"}}, manifest=manifest)
        with self.assertRaises(ProtocolError):
            validate_result({**result, "classification": "runtime_unavailable"}, manifest=manifest)
        with self.assertRaises(ProtocolError):
            validate_result({**result, "manifest_sha256": "0" * 64}, manifest=manifest)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            with self.assertRaises(ProtocolError):
                read_capped_json(path, limit=1024)
            path.write_bytes(b'{"schema_version":1')
            with self.assertRaises(ProtocolError):
                read_capped_json(path, limit=1024)
            path.write_bytes(b"{}" + b"x" * 2048)
            with self.assertRaises(ProtocolError):
                read_capped_json(path, limit=1024)


if __name__ == "__main__":
    unittest.main()
