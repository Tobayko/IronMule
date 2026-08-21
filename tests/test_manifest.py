import unittest

from friday_h0.constants import (
    AA_BOOTSTRAP_SEEDS,
    AA_SESSION_SEEDS,
    EAGER_COMPILE_SESSION_SEEDS,
    PHASE_H0,
    WRONG_FIXTURE_SEED,
)
from friday_h0.manifest import ManifestError, canonical_manifest_bytes, manifest_hash, validate_manifest


def valid_manifest(mode="eager_baseline", *, shape=None, fixture=None, order=None, set_name="characterization", index=0):
    if mode in {"eager_baseline", "compile_comparison"}:
        seed_table = EAGER_COMPILE_SESSION_SEEDS
        fixture = seed_table[f"{set_name}_fixture"] + index if fixture is None else fixture
        order = seed_table[f"{set_name}_order"] + index if order is None else order
    elif mode == "aa_gpu":
        seed_table = AA_SESSION_SEEDS
        fixture = seed_table[f"{set_name}_fixture"] + index if fixture is None else fixture
        order = seed_table[f"{set_name}_order"] + index if order is None else order
    else:
        if fixture is None:
            fixture = WRONG_FIXTURE_SEED if mode == "analysis_wrong_fixture" else 0
        order = 0 if order is None else order
    shape = 64 if mode == "analysis_wrong_fixture" else 2048 if shape is None else shape
    process_set = set_name if mode in {"eager_baseline", "compile_comparison", "aa_gpu"} else "analysis" if mode.startswith("analysis_") else "control"
    process_index = index if mode in {"eager_baseline", "compile_comparison", "aa_gpu"} else 0
    return {
        "schema_version": 1,
        "phase": PHASE_H0,
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
            **({"bootstrap_seed": AA_BOOTSTRAP_SEEDS[set_name]} if mode == "aa_gpu" else {}),
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


class ManifestTests(unittest.TestCase):
    def test_closed_manifest_is_hashable(self):
        manifest = valid_manifest()
        validated = validate_manifest(manifest)
        self.assertEqual(validated, manifest)
        self.assertEqual(len(manifest_hash(manifest)), 64)
        self.assertEqual(canonical_manifest_bytes(manifest), canonical_manifest_bytes(validated))

    def test_unknown_keys_and_bool_as_int_are_rejected(self):
        manifest = valid_manifest()
        manifest["source"] = "not-allowed"
        with self.assertRaises(ManifestError):
            validate_manifest(manifest)
        manifest = valid_manifest()
        manifest["seeds"]["fixture"] = True
        with self.assertRaises(ManifestError):
            validate_manifest(manifest)

    def test_wrong_fixture_is_explicitly_bound(self):
        manifest = valid_manifest("analysis_wrong_fixture", shape=64, fixture=0xBAD02026)
        self.assertEqual(validate_manifest(manifest)["mode"], "analysis_wrong_fixture")
        manifest["seeds"]["fixture"] = 1
        with self.assertRaises(ManifestError):
            validate_manifest(manifest)

    def test_nonfinite_and_nonregistered_shape_are_rejected(self):
        manifest = valid_manifest()
        manifest["limits"]["total_s"] = float("nan")
        with self.assertRaises(ManifestError):
            validate_manifest(manifest)
        manifest = valid_manifest(shape=64)
        with self.assertRaises(ManifestError):
            validate_manifest(manifest)

    def test_process_provenance_and_seed_contract_are_closed(self):
        manifest = valid_manifest("aa_gpu", set_name="confirmation", index=2)
        self.assertEqual(validate_manifest(manifest)["process"], {"set": "confirmation", "index": 2})
        self.assertEqual(manifest["seeds"]["bootstrap_seed"], AA_BOOTSTRAP_SEEDS["confirmation"])
        for invalid in (
            {**manifest["seeds"], "bootstrap_seed": AA_BOOTSTRAP_SEEDS["confirmation"] + 1},
            {**manifest["seeds"], "bootstrap_seed": True},
            {key: value for key, value in manifest["seeds"].items() if key != "bootstrap_seed"},
            {**manifest["seeds"], "extra": 1},
        ):
            manifest["seeds"] = invalid
            with self.assertRaises(ManifestError):
                validate_manifest(manifest)
        manifest = valid_manifest("aa_gpu", set_name="characterization", index=0)
        self.assertEqual(validate_manifest(manifest)["seeds"]["bootstrap_seed"], AA_BOOTSTRAP_SEEDS["characterization"])
        non_aa = valid_manifest("eager_baseline")
        self.assertNotIn("bootstrap_seed", non_aa["seeds"])
        self.assertEqual(validate_manifest(non_aa), non_aa)
        manifest["run_id"] = "bad:id"
        with self.assertRaises(ManifestError):
            validate_manifest(manifest)
        manifest = valid_manifest()
        manifest["provenance"]["code_sha256"] = "A" * 64
        with self.assertRaises(ManifestError):
            validate_manifest(manifest)
        manifest = valid_manifest()
        manifest["provenance"]["revision"] = {"value": "rev", "missing_reason": "why"}
        with self.assertRaises(ManifestError):
            validate_manifest(manifest)
        manifest = valid_manifest()
        manifest["seeds"]["fixture"] += 1
        with self.assertRaises(ManifestError):
            validate_manifest(manifest)
