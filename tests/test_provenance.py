import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path

from friday_h0.provenance import (
    REVISION_MISSING_REASON,
    ProvenanceError,
    _collect_provenance_for_tests,
    _test_root,
    collect_provenance,
)


ROOT = Path(__file__).resolve().parents[1]


class ProvenanceTests(unittest.TestCase):
    def test_provenance_is_deterministic_and_uses_registered_revision_reason(self):
        first = collect_provenance()
        second = collect_provenance()
        self.assertEqual(first, second)
        self.assertEqual(first.revision, {"value": None, "missing_reason": REVISION_MISSING_REASON})
        for value in (first.code_sha256, first.spec_sha256, first.environment_sha256):
            self.assertRegex(value, r"^[0-9a-f]{64}$")

    def test_spec_hash_is_exact_bytes(self):
        provenance = collect_provenance()
        expected = hashlib.sha256((ROOT / "docs/PHASE1_MATMUL_SPEC.md").read_bytes()).hexdigest()
        self.assertEqual(provenance.spec_sha256, expected)

    def test_missing_fixed_input_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ProvenanceError):
                _collect_provenance_for_tests(_test_root(temporary))

    def test_alternate_product_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ProvenanceError):
                collect_provenance(temporary)

    def _copy_project_fixture(self, temporary: str) -> Path:
        root = Path(temporary)
        shutil.copytree(ROOT / "friday_h0", root / "friday_h0")
        (root / "docs").mkdir()
        shutil.copy2(ROOT / "docs/PHASE1_MATMUL_SPEC.md", root / "docs/PHASE1_MATMUL_SPEC.md")
        return root

    def test_aggregation_is_in_the_code_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._copy_project_fixture(temporary)
            context = _test_root(fixture)
            before = _collect_provenance_for_tests(context)
            aggregation = fixture / "friday_h0/aggregation.py"
            aggregation.write_bytes(aggregation.read_bytes() + b"\n# isolated hash fixture\n")
            after = _collect_provenance_for_tests(context)
            self.assertNotEqual(before.code_sha256, after.code_sha256)

    def test_provenance_rejects_allowlisted_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._copy_project_fixture(temporary)
            target = fixture / "friday_h0/aggregation.py"
            replacement = fixture / "friday_h0/aggregation.real.py"
            target.rename(replacement)
            try:
                target.symlink_to(replacement)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {exc}")
            with self.assertRaises(ProvenanceError):
                _collect_provenance_for_tests(_test_root(fixture))


if __name__ == "__main__":
    unittest.main()
