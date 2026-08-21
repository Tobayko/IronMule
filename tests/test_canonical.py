import math
import unittest

from friday_h0.canonical import CanonicalizationError, canonical_json_bytes, canonical_sha256


class CanonicalTests(unittest.TestCase):
    def test_sorted_utf8_json_is_reproducible(self):
        value = {"z": "ä", "a": [2, 1]}
        self.assertEqual(canonical_json_bytes(value), '{"a":[2,1],"z":"ä"}'.encode("utf-8"))
        self.assertEqual(canonical_sha256(value), canonical_sha256({"a": [2, 1], "z": "ä"}))

    def test_nonfinite_and_unsupported_values_are_rejected(self):
        with self.assertRaises(CanonicalizationError):
            canonical_json_bytes({"x": math.inf})
        with self.assertRaises(CanonicalizationError):
            canonical_json_bytes({"x": object()})
