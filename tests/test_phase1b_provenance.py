from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from friday_phase1b.provenance import ProvenanceError, _git, verify_source_snapshot
from tests.test_phase1b_history import provenance


class Phase1BProvenanceTests(unittest.TestCase):
    def test_git_identity_uses_closed_environment_and_disables_hooks(self) -> None:
        completed = Mock(returncode=0, stdout=b"ok", stderr=b"")
        with patch("friday_phase1b.provenance.subprocess.run", return_value=completed) as run:
            self.assertEqual(_git("rev-parse", "HEAD"), b"ok")
        argv = run.call_args.args[0]
        environment = run.call_args.kwargs["env"]
        self.assertIn("core.fsmonitor=false", argv)
        self.assertIn("core.hooksPath=/dev/null", argv)
        self.assertNotIn("GIT_DIR", environment)
        self.assertEqual(environment["GIT_CONFIG_GLOBAL"], "/dev/null")

    def test_frozen_snapshot_accepts_exact_provenance(self) -> None:
        expected = provenance()
        observed_keys = (
            "git_revision",
            "git_dirty",
            "git_status_sha256",
            "code_files",
            "code_sha256",
            "spec_files",
            "spec_sha256",
            "source",
            "source_binding_sha256",
        )
        observed = {key: expected[key] for key in observed_keys}
        with patch("friday_phase1b.provenance._source_snapshot", return_value=observed):
            verify_source_snapshot(expected)

    def test_frozen_snapshot_rejects_digest_and_live_source_changes(self) -> None:
        expected = provenance()
        changed_digest = dict(expected)
        changed_digest["code_sha256"] = "f" * 64
        with self.assertRaises(ProvenanceError):
            verify_source_snapshot(changed_digest)

        observed = {
            key: expected[key]
            for key in (
                "git_revision",
                "git_dirty",
                "git_status_sha256",
                "code_files",
                "code_sha256",
                "spec_files",
                "spec_sha256",
                "source",
                "source_binding_sha256",
            )
        }
        observed["code_sha256"] = "e" * 64
        with patch("friday_phase1b.provenance._source_snapshot", return_value=observed):
            with self.assertRaises(ProvenanceError):
                verify_source_snapshot(expected)


if __name__ == "__main__":
    unittest.main()
