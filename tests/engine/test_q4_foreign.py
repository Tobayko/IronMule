"""Unittest coverage for the offline foreign-evidence boundary."""

from __future__ import annotations

import base64
import hashlib
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from q4_offline_loader import load_offline_modules


_OFFLINE = load_offline_modules(
    "evidence", "q4_contracts", "q4_foreign", namespace="q4_foreign_test_modules"
)
_evidence = _OFFLINE["evidence"]
_contracts = _OFFLINE["q4_contracts"]
_foreign = _OFFLINE["q4_foreign"]

ArtifactRef = _evidence.ArtifactRef
EvidenceQuality = _evidence.EvidenceQuality
ForeignBundleMetadata = _contracts.ForeignBundleMetadata
ENVELOPE_SCHEMA = _foreign.ENVELOPE_SCHEMA
ForeignBundleEnvelope = _foreign.ForeignBundleEnvelope
ForeignBundleVerifier = _foreign.ForeignBundleVerifier
ForeignEvidenceStatus = _foreign.ForeignEvidenceStatus
ForeignIdentity = _foreign.ForeignIdentity
ForeignVerificationError = _foreign.ForeignVerificationError
VerifierUnavailable = _foreign.VerifierUnavailable
ReplayRegistry = _foreign.ReplayRegistry
TrustedPublicKey = _foreign.TrustedPublicKey
UserApprovedTrustStore = _foreign.UserApprovedTrustStore
bundle_id_sha256 = _foreign.bundle_id_sha256
canonical_bundle_payload = _foreign.canonical_bundle_payload
verify_foreign_bundle = _foreign.verify_foreign_bundle


HEX = "a" * 64
KEY_BYTES = bytes(range(32))
SIGNATURE = base64.b64encode(bytes(range(64))).decode("ascii")
NOW = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)


class ForeignBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "sample.json").write_text('{"raw": true}\n', encoding="utf-8")
        (self.root / "reviewer.json").write_text('{"evaluator": "independent", "gates": "complete"}\n', encoding="utf-8")
        self.raw_digest = hashlib.sha256((self.root / "sample.json").read_bytes()).hexdigest()
        self.reviewer_digest = hashlib.sha256((self.root / "reviewer.json").read_bytes()).hexdigest()
        key = TrustedPublicKey("foreign-test", KEY_BYTES, approved_by_user=True)
        self.store = UserApprovedTrustStore.from_keys((key,), approved_by_user=True)

        def verifier(public_key: bytes, signature: bytes, payload: bytes) -> bool:
            return public_key == KEY_BYTES and signature == bytes(range(64)) and payload == canonical_bundle_payload(
                self.metadata, nonce=self.envelope.nonce, expires_at_utc=self.envelope.expires_at_utc
            )

        self.signature_verifier = verifier

    def _verifier(self) -> ForeignBundleVerifier:
        return ForeignBundleVerifier(
            self.store,
            signature_verifier=self.signature_verifier,
            attestation_validator=lambda envelope, root, reviewer_path: reviewer_path.name == "reviewer.json",
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _metadata(self, *, artifact_id: str = "sample.json") -> ForeignBundleMetadata:
        metadata = ForeignBundleMetadata(
            bundle_id=HEX,
            exporter_id="foreign-exporter",
            host_class="Apple M2 Max",
            hardware_digest=HEX,
            model_digest="b" * 64,
            model_manifest_digest="c" * 64,
            runtime_digest="d" * 64,
            code_digest="e" * 64,
            workload_digest="f" * 64,
            preregistration_sha256="1" * 64,
            raw_artifacts=(ArtifactRef(artifact_id, self.raw_digest, EvidenceQuality.RAW_SAMPLES),),
            reviewer_record_sha256=self.reviewer_digest,
            signature_algorithm="Ed25519",
            signer_key_fingerprint=hashlib.sha256(KEY_BYTES).hexdigest(),
            signature=SIGNATURE,
            exported_at_utc="2026-09-01T10:00:00Z",
            public_key_id="foreign-test",
        )
        envelope = ForeignBundleEnvelope(metadata, "nonce-001", "2026-09-02T10:00:00Z")
        metadata = replace(metadata, bundle_id=bundle_id_sha256(metadata, nonce=envelope.nonce, expires_at_utc=envelope.expires_at_utc))
        self.metadata = metadata
        self.envelope = ForeignBundleEnvelope(metadata, envelope.nonce, envelope.expires_at_utc)
        return metadata

    def test_verified_bundle_is_calibration_only_and_requires_local_revalidation(self) -> None:
        self._metadata()
        result = self._verifier().verify(
            self.envelope, artifact_root=self.root, reviewer_record_path="reviewer.json", now=NOW
        )
        self.assertEqual(result.status, ForeignEvidenceStatus.REVALIDATION_REQUIRED)
        self.assertTrue(result.accepted)
        self.assertTrue(result.calibration_only)
        self.assertTrue(result.revalidation_required)
        self.assertFalse(result.can_replace_base)
        self.assertFalse(result.can_enter_q4_split)

    def test_bundle_id_is_derived_without_self_or_signature(self) -> None:
        self._metadata()
        first = bundle_id_sha256(self.metadata, nonce=self.envelope.nonce, expires_at_utc=self.envelope.expires_at_utc)
        altered_signature = replace(self.metadata, signature=base64.b64encode(b"x" * 64).decode("ascii"))
        second = bundle_id_sha256(altered_signature, nonce=self.envelope.nonce, expires_at_utc=self.envelope.expires_at_utc)
        self.assertEqual(first, self.metadata.bundle_id)
        self.assertEqual(first, second)

    def test_bad_bundle_id_fails_closed_before_signature(self) -> None:
        metadata = self._metadata()
        self.envelope = ForeignBundleEnvelope(replace(metadata, bundle_id=HEX), "nonce-002", "2026-09-02T10:00:00Z")
        result = self._verifier().verify(
            self.envelope, artifact_root=self.root, reviewer_record_path="reviewer.json", now=NOW
        )
        self.assertEqual(result.status, ForeignEvidenceStatus.UNTRUSTED_FOREIGN_EVIDENCE)
        self.assertFalse(result.accepted)

    def test_unknown_key_is_untrusted(self) -> None:
        self._metadata()
        self.envelope = ForeignBundleEnvelope(replace(self.metadata, public_key_id="other-key"), "nonce-003", "2026-09-02T10:00:00Z")
        # The changed nonce/key changes the derived ID; this test is about trust resolution.
        self.envelope = ForeignBundleEnvelope(
            replace(self.envelope.metadata, bundle_id=bundle_id_sha256(self.envelope.metadata, nonce=self.envelope.nonce, expires_at_utc=self.envelope.expires_at_utc)),
            self.envelope.nonce,
            self.envelope.expires_at_utc,
        )
        result = self._verifier().verify(
            self.envelope, artifact_root=self.root, reviewer_record_path="reviewer.json", now=NOW
        )
        self.assertEqual(result.status, ForeignEvidenceStatus.UNTRUSTED_FOREIGN_EVIDENCE)

    def test_missing_artifact_root_is_missing_not_accepted(self) -> None:
        self._metadata()
        result = self._verifier().verify(
            self.envelope, artifact_root=None, now=NOW
        )
        self.assertEqual(result.status, ForeignEvidenceStatus.MISSING)
        self.assertFalse(result.accepted)

    def test_missing_evaluator_attestation_is_incomplete(self) -> None:
        self._metadata()
        verifier = ForeignBundleVerifier(self.store, signature_verifier=self.signature_verifier)
        result = verifier.verify(
            self.envelope, artifact_root=self.root, reviewer_record_path="reviewer.json", now=NOW
        )
        self.assertEqual(result.status, ForeignEvidenceStatus.INCOMPLETE)
        self.assertFalse(result.can_calibrate)

    def test_unavailable_crypto_provider_fails_closed(self) -> None:
        self._metadata()

        class Unavailable:
            def verify(self, public_key: bytes, signature: bytes, message: bytes) -> bool:
                raise VerifierUnavailable("provider unavailable")

        result = ForeignBundleVerifier(
            self.store,
            signature_verifier=Unavailable(),
            attestation_validator=lambda envelope, root, reviewer_path: True,
        ).verify(self.envelope, artifact_root=self.root, reviewer_record_path="reviewer.json", now=NOW)
        self.assertEqual(result.status, ForeignEvidenceStatus.VERIFIER_UNAVAILABLE)
        self.assertFalse(result.accepted)

    def test_same_bundle_and_nonce_cannot_be_replayed(self) -> None:
        self._metadata()
        verifier = self._verifier()
        first = verifier.verify(self.envelope, artifact_root=self.root, reviewer_record_path="reviewer.json", now=NOW)
        second = verifier.verify(self.envelope, artifact_root=self.root, reviewer_record_path="reviewer.json", now=NOW)
        self.assertTrue(first.accepted)
        self.assertEqual(second.status, ForeignEvidenceStatus.DUPLICATE)

    def test_convenience_call_without_shared_replay_state_fails_closed(self) -> None:
        self._metadata()
        first = verify_foreign_bundle(
            self.envelope,
            self.store,
            artifact_root=self.root,
            reviewer_record_path="reviewer.json",
            signature_verifier=self.signature_verifier,
            attestation_validator=lambda envelope, root, reviewer_path: True,
            now=NOW,
        )
        second = verify_foreign_bundle(
            self.envelope,
            self.store,
            artifact_root=self.root,
            reviewer_record_path="reviewer.json",
            signature_verifier=self.signature_verifier,
            attestation_validator=lambda envelope, root, reviewer_path: True,
            now=NOW,
        )
        self.assertFalse(first.accepted)
        self.assertFalse(second.accepted)
        self.assertEqual(first.status, ForeignEvidenceStatus.REPLAY_REJECTED)

    def test_convenience_calls_share_explicit_registry(self) -> None:
        self._metadata()
        registry = ReplayRegistry()
        kwargs = dict(
            artifact_root=self.root,
            reviewer_record_path="reviewer.json",
            signature_verifier=self.signature_verifier,
            attestation_validator=lambda envelope, root, reviewer_path: True,
            replay_registry=registry,
            now=NOW,
        )
        first = verify_foreign_bundle(self.envelope, self.store, **kwargs)
        second = verify_foreign_bundle(self.envelope, self.store, **kwargs)
        self.assertTrue(first.accepted)
        self.assertEqual(second.status, ForeignEvidenceStatus.DUPLICATE)

    def test_separate_verifiers_share_registry_replay_state(self) -> None:
        self._metadata()
        registry = ReplayRegistry()
        first = ForeignBundleVerifier(
            self.store,
            signature_verifier=self.signature_verifier,
            attestation_validator=lambda envelope, root, reviewer_path: True,
            replay_registry=registry,
        ).verify(self.envelope, artifact_root=self.root, reviewer_record_path="reviewer.json", now=NOW)
        second = ForeignBundleVerifier(
            self.store,
            signature_verifier=self.signature_verifier,
            attestation_validator=lambda envelope, root, reviewer_path: True,
            replay_registry=registry,
        ).verify(self.envelope, artifact_root=self.root, reviewer_record_path="reviewer.json", now=NOW)
        self.assertTrue(first.accepted)
        self.assertEqual(second.status, ForeignEvidenceStatus.DUPLICATE)

    def test_expired_bundle_is_rejected(self) -> None:
        self._metadata()
        result = self._verifier().verify(
            self.envelope, artifact_root=self.root, reviewer_record_path="reviewer.json", now=datetime(2026, 9, 2, 10, 0, 1, tzinfo=timezone.utc)
        )
        self.assertEqual(result.status, ForeignEvidenceStatus.EXPIRED)

    def test_identity_mismatch_is_out_of_domain(self) -> None:
        self._metadata()
        expected = replace(ForeignIdentity.from_bundle(self.metadata), hardware_digest="9" * 64)
        result = self._verifier().verify(
            self.envelope, artifact_root=self.root, reviewer_record_path="reviewer.json", expected_identity=expected, now=NOW
        )
        self.assertEqual(result.status, ForeignEvidenceStatus.OUT_OF_DOMAIN)

    def test_path_traversal_and_hash_mismatch_fail_closed(self) -> None:
        metadata = self._metadata(artifact_id="../sample.json")
        self.envelope = ForeignBundleEnvelope(
            replace(metadata, bundle_id=bundle_id_sha256(metadata, nonce="nonce-path", expires_at_utc="2026-09-02T10:00:00Z")),
            "nonce-path",
            "2026-09-02T10:00:00Z",
        )
        result = self._verifier().verify(
            self.envelope, artifact_root=self.root, reviewer_record_path="reviewer.json", now=NOW
        )
        self.assertEqual(result.status, ForeignEvidenceStatus.UNTRUSTED_FOREIGN_EVIDENCE)

    def test_trust_store_requires_explicit_approval(self) -> None:
        with self.assertRaises(ForeignVerificationError):
            UserApprovedTrustStore.from_keys((TrustedPublicKey("id", KEY_BYTES, approved_by_user=True),), approved_by_user=False)

    def test_envelope_schema_is_strict(self) -> None:
        self._metadata()
        data = self.envelope.to_dict()
        self.assertEqual(data["schema"], ENVELOPE_SCHEMA)
        with self.assertRaises(ForeignVerificationError):
            ForeignBundleEnvelope.from_dict({**data, "extra": True})

    def test_prior_never_allows_base_or_q4_split(self) -> None:
        self._metadata()
        verifier = self._verifier()
        result = verifier.verify(self.envelope, artifact_root=self.root, reviewer_record_path="reviewer.json", now=NOW)
        prior = verifier.calibration_prior(result, ordered_action_ids=("action-a", "action-b"))
        self.assertEqual(prior.status, ForeignEvidenceStatus.REVALIDATION_REQUIRED)
        self.assertTrue(prior.requires_local_revalidation)
        self.assertFalse(prior.may_replace_base)
        self.assertFalse(prior.may_enter_q4_split)


if __name__ == "__main__":
    unittest.main()
