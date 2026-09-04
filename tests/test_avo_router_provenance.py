from __future__ import annotations

import unittest
from unittest.mock import patch

from friday_avo_router.provenance import ProvenanceError, collect_provenance


class RouterProvenanceTests(unittest.TestCase):
    def test_current_dirty_tree_can_be_described_but_not_released(self) -> None:
        value = collect_provenance(require_clean=False)
        self.assertEqual(len(value["code_sha256"]), 64)
        self.assertEqual(len(value["spec_sha256"]), 64)
        self.assertEqual(len(value["provenance_sha256"]), 64)

    def test_require_clean_fails_before_hashing_when_status_is_dirty(self) -> None:
        def fake_git(*args: str) -> bytes:
            if args == ("rev-parse", "HEAD"):
                return b"1" * 40 + b"\n"
            return b" M friday_avo_router/router.py\n"

        with patch("friday_avo_router.provenance._git", side_effect=fake_git):
            with self.assertRaises(ProvenanceError):
                collect_provenance(require_clean=True)


if __name__ == "__main__":
    unittest.main()
