from __future__ import annotations

import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from friday_optimizer.profiles import AtomicProfileStore, OptimizerProfile, ProfileError, ProfileMode


def profile(name="candidate", fp="hardware-v1", qualified=True):
    return OptimizerProfile(name, fp, "persistent_process", {"ttft": 1.0}, qualified).with_hash()


class ProfileStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "profiles.json"
        self.store = AtomicProfileStore(self.path)

    def tearDown(self): self.temp.cleanup()

    def test_atomic_store_and_exact_fingerprint(self):
        baseline = self.store.save(profile("base"), baseline=True, current_fingerprint="hardware-v1")
        candidate = self.store.save(profile(), current_fingerprint="hardware-v1")
        self.store.activate(candidate.profile_id, fingerprint="hardware-v1")
        selected = self.store.select("hardware-v1")
        self.assertEqual(selected.profile.profile_id, "candidate")
        self.assertFalse(self.store.select("other").profile)
        self.assertEqual(os.stat(self.path).st_mode & 0o777, 0o600)
        self.assertGreaterEqual(self.store.load()["version"], 3)

    def test_tamper_and_symlink_fail_closed(self):
        self.store.save(profile("base"), baseline=True)
        value = json.loads(self.path.read_text())
        value["profiles"]["base"]["candidate"] = "tampered"
        self.path.write_text(json.dumps(value))
        os.chmod(self.path, 0o600)
        with self.assertRaises(ProfileError): self.store.load()
        other = Path(self.temp.name) / "other"
        other.write_text("{}")
        self.path.unlink()
        self.path.symlink_to(other)
        with self.assertRaises(ProfileError): AtomicProfileStore(self.path)

    def test_rollback_latch_and_pinned_safety(self):
        self.store.save(profile("base"), baseline=True)
        self.store.save(profile("candidate"))
        self.store.activate("candidate", fingerprint="hardware-v1")
        self.store.rollback(reason="canary")
        self.assertTrue(self.store.select("hardware-v1").no_recommendation)
        with self.assertRaises(ProfileError): self.store.activate("candidate", fingerprint="hardware-v1")
        self.store.clear_rollback_latch(fingerprint="hardware-v1", qualified_profile_id="candidate")
        self.store.set_mode(ProfileMode.PINNED, pinned_id="candidate")
        self.assertEqual(self.store.select("hardware-v1").profile.profile_id, "candidate")
        self.assertTrue(self.store.select("different").no_recommendation)

    def test_parser_rejects_coercion_and_oversize(self):
        raw = profile("bad").as_dict()
        raw["qualified"] = 1
        with self.assertRaises(ProfileError): OptimizerProfile.from_dict(raw)
        raw = profile("deep").as_dict()
        raw["metrics"] = {"x": "x" * 5000}
        with self.assertRaises(ProfileError): OptimizerProfile.from_dict(raw)
        self.path.write_bytes(b"x" * 5000)
        os.chmod(self.path, 0o600)
        with self.assertRaises(ProfileError): AtomicProfileStore(self.path, max_bytes=4096).load()

    def test_duplicate_json_keys_are_rejected(self):
        self.path.write_text('{"schema_version":1,"schema_version":1}')
        os.chmod(self.path, 0o600)
        with self.assertRaises(ProfileError): self.store.load()

    def test_compare_and_swap_rejects_stale_writer(self):
        version = self.store.load()["version"]
        self.store.save(profile("one"), expected_version=version)
        with self.assertRaises(ProfileError):
            self.store.save(profile("two"), expected_version=version)

    def test_concurrent_writers_preserve_one_immutable_record(self):
        stores = [AtomicProfileStore(self.path), AtomicProfileStore(self.path)]
        values = [profile("same"), profile("same", qualified=False)]
        def write(args):
            store, value = args
            try:
                store.save(value)
                return "ok"
            except ProfileError:
                return "rejected"
        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(write, zip(stores, values)))
        self.assertEqual(sorted(outcomes), ["ok", "rejected"])
        self.assertIn(self.store.load()["profiles"]["same"]["qualified"], (True, False))


if __name__ == "__main__": unittest.main()
