"""Metadata/lifecycle tests only; these make no native execution claim."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pytest

import ironmule_product.prefix_session as prefix_session
from ironmule_product.prefix_reuse import PrefixReuseTrace


BINDING = "7" * 64
SCOPE = "worker-private-scope"


@dataclass
class _SnapshotPart:
    nbytes: int = 8


class _MetadataAdapter:
    def __init__(self, model, tokenizer, cache, **identity):
        self.cache = cache
        self.identity = identity
        self.last_trace = PrefixReuseTrace(
            "prepared", "none", False, False, 0, "not_started", 0, 0, 0
        )

    def stream(self, prompt_ids, *, max_tokens):
        epoch, reason = self.cache.begin_cold(
            binding_sha256=self.identity["expected_binding_sha256"],
            scope_nonce=self.identity["scope_nonce"],
        )
        assert reason == "none"
        result = self.cache.commit(
            prompt_ids,
            [_SnapshotPart()],
            epoch=epoch,
            binding_sha256=self.identity["expected_binding_sha256"],
            scope_nonce=self.identity["scope_nonce"],
        )
        assert result == "deferred_commit"
        self.last_trace = PrefixReuseTrace(
            "completed", "none", False, False, 0, result, 1, 2, 0
        )
        return iter(())


@pytest.fixture
def metadata_adapter(monkeypatch):
    monkeypatch.setattr(prefix_session, "IdenticalPromptReuse", _MetadataAdapter)


def _session(**limits):
    return prefix_session.WorkerPrefixSession(
        object(),
        object(),
        binding_sha256=BINDING,
        scope_nonce=SCOPE,
        max_entries=limits.get("max_entries", 2),
        max_bytes=limits.get("max_bytes", 32),
    )


def test_constructor_is_metadata_only_and_repr_omits_identity_secrets():
    session = _session()
    rendered = repr(session)
    assert BINDING not in rendered
    assert SCOPE not in rendered
    assert "entries=0" in rendered


def test_accepted_finalize_is_the_only_cache_publication(metadata_adapter):
    session = _session()
    assert list(session.stream([11, 12], max_tokens=1)) == []
    assert session._cache.stats().entries == 0

    metadata = session.finalize(True)

    assert metadata.status == "accepted"
    assert metadata.commit_status == "stored"
    assert metadata.trace.cache_stored is True
    assert metadata.trace.fallback_reason == "none"
    assert metadata.stats.entries == 1
    assert session._proxy is None
    assert session._adapter is None


def test_rejected_finalize_discards_pending_without_changing_canonical(metadata_adapter):
    session = _session()
    list(session.stream([1, 2], max_tokens=1))
    assert session.finalize(True).stats.entries == 1
    list(session.stream([3, 4], max_tokens=1))

    metadata = session.finalize(False)

    assert metadata.status == "discarded"
    assert metadata.commit_status == "discarded"
    assert metadata.trace.cache_stored is False
    assert metadata.stats.entries == 1
    restored, reason = session._cache.restore(
        [1, 2], binding_sha256=BINDING, scope_nonce=SCOPE
    )
    assert restored is not None
    assert reason == "none"
    missing, reason = session._cache.restore(
        [3, 4], binding_sha256=BINDING, scope_nonce=SCOPE
    )
    assert missing is None
    assert reason == "cache_miss"


def test_only_one_generation_can_await_finalize(metadata_adapter):
    session = _session()
    session.stream([1, 2], max_tokens=1)
    with pytest.raises(RuntimeError, match="active generation"):
        session.stream([3, 4], max_tokens=1)
    session.finalize(False)
    session.stream([3, 4], max_tokens=1)


def test_clear_discards_pending_and_invalidates_its_epoch(metadata_adapter):
    session = _session()
    list(session.stream([1, 2], max_tokens=1))

    cleared = session.clear()
    finalized = session.finalize(True)

    assert cleared.status == "cleared"
    assert cleared.stats.entries == 0
    assert finalized.commit_status == "no_pending_commit"
    assert finalized.stats.entries == 0


def test_close_is_idempotent_and_never_publishes_pending(metadata_adapter):
    session = _session()
    list(session.stream([1, 2], max_tokens=1))

    first = session.close()
    second = session.close()

    assert first == second
    assert first.status == "closed"
    assert first.stats.entries == 0
    with pytest.raises(RuntimeError, match="closed"):
        session.stream([1, 2], max_tokens=1)


def test_auxiliary_commit_failures_remain_metadata_not_exceptions(metadata_adapter):
    session = _session(max_bytes=1)
    list(session.stream([1, 2], max_tokens=1))

    metadata = session.finalize(True)

    assert metadata.commit_status == "oversize"
    assert metadata.trace.cache_stored is False
    assert metadata.trace.fallback_reason == "oversize"
    assert metadata.stats.skipped_oversize == 1


def test_finalize_requires_an_actual_boolean():
    session = _session()
    with pytest.raises(ValueError, match="boolean"):
        session.finalize(1)  # type: ignore[arg-type]


def test_nonterminal_acceptance_is_rejected_and_serializes_as_safe_metadata(
    metadata_adapter,
):
    session = _session()
    session.stream([91, 92], max_tokens=1)
    session._adapter.last_trace = PrefixReuseTrace(
        "prepared", "none", False, False, 0, "not_started", 0, 0, 0
    )

    metadata = session.finalize(True)
    payload = asdict(metadata)

    assert metadata.status == "rejected"
    assert metadata.commit_status == "not_terminal"
    assert metadata.trace.status == "prepared"
    assert payload["trace"]["commit_host_ns"] == 0
    assert payload["stats"]["entries"] == 0
    assert BINDING not in repr(payload)
    assert SCOPE not in repr(payload)
