"""Metadata-only PROD11 checks; these make no GPU/cache correctness claim."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from ironmule_product.prefix_reuse import (
    PREFILL_STEP_SIZE,
    IdenticalPromptReuse,
    PrefixReuseOptions,
    PrefixReuseResponse,
    PrivatePrefixCache,
    _prompt_key,
)


SCOPE = "private-session-scope"


def test_candidate_options_are_closed_and_immutable():
    options = PrefixReuseOptions()
    assert options.prefill_step_size == PREFILL_STEP_SIZE
    assert options.decoding == "greedy"
    assert options.logits_processors is None
    assert options.draft_model is None
    assert options.kv_bits is None
    with pytest.raises(FrozenInstanceError):
        options.decoding = "sampling"  # type: ignore[misc]
    with pytest.raises(ValueError, match="unsupported"):
        PrefixReuseOptions(prefill_step_size=512)
    with pytest.raises(ValueError, match="unsupported"):
        PrefixReuseOptions(logits_processors=[])  # type: ignore[arg-type]


def test_private_metadata_never_exposes_binding_or_scope_secret():
    binding = "a" * 64
    secret = SCOPE
    cache = PrivatePrefixCache(
        binding_sha256=binding,
        scope_nonce=secret,
        max_entries=2,
        max_bytes=4096,
    )
    rendered = repr(cache)
    assert binding not in rendered
    assert secret not in rendered
    assert cache.stats().entries == 0
    assert cache.stats().bytes == 0


@pytest.mark.parametrize("binding", ["", "A" * 64, "0" * 63, "not-a-digest"])
def test_binding_must_be_an_opaque_lowercase_sha256(binding):
    with pytest.raises(ValueError, match="invalid cache binding"):
        PrivatePrefixCache(
            binding_sha256=binding,
            scope_nonce=SCOPE,
            max_entries=1,
            max_bytes=1,
        )


def test_clear_and_close_advance_lifecycle_without_exposing_contents():
    cache = PrivatePrefixCache(
        binding_sha256="b" * 64,
        scope_nonce=SCOPE,
        max_entries=1,
        max_bytes=1,
    )
    binding = "b" * 64
    old_epoch, reason = cache.begin_cold(binding_sha256=binding, scope_nonce=SCOPE)
    assert reason == "none"
    cache.clear()
    new_epoch, reason = cache.begin_cold(binding_sha256=binding, scope_nonce=SCOPE)
    assert reason == "none"
    assert new_epoch != old_epoch
    cache.close()
    epoch, reason = cache.begin_cold(binding_sha256=binding, scope_nonce=SCOPE)
    assert epoch is None
    assert reason == "cache_closed"


@pytest.mark.parametrize(
    "tokens",
    [None, (), [True], [1.0], ["1"], [1, -1]],
)
def test_prompt_keys_never_coerce_noncanonical_tokens(tokens):
    if tokens == ():
        assert _prompt_key(tokens) == ()
    else:
        with pytest.raises(ValueError, match="canonical"):
            _prompt_key(tokens)  # type: ignore[arg-type]


@pytest.mark.parametrize("name,value", [("max_entries", True), ("max_bytes", 1.0)])
def test_cache_limits_are_actual_positive_integers(name, value):
    kwargs = {"max_entries": 1, "max_bytes": 1}
    kwargs[name] = value
    with pytest.raises(ValueError, match="positive integers"):
        PrivatePrefixCache(
            binding_sha256="c" * 64,
            scope_nonce=SCOPE,
            **kwargs,
        )


def test_response_repr_omits_text_and_logprobs():
    response = PrefixReuseResponse(
        text="private response",
        token=7,
        logprobs="private tensor",
        from_draft=False,
        prompt_tokens=10,
        prompt_tps=None,
        suffix_prompt_tps=None,
        generation_tokens=1,
        generation_tps=2.0,
        peak_memory=3.0,
        finish_reason="stop",
    )
    rendered = repr(response)
    assert "private response" not in rendered
    assert "private tensor" not in rendered


def test_wrong_binding_and_scope_are_explicit_nonusable_epochs():
    cache = PrivatePrefixCache(
        binding_sha256="d" * 64,
        scope_nonce=SCOPE,
        max_entries=1,
        max_bytes=1,
    )
    epoch, reason = cache.begin_cold(
        binding_sha256="e" * 64, scope_nonce=SCOPE
    )
    assert epoch is None
    assert reason == "wrong_binding"
    epoch, reason = cache.begin_cold(
        binding_sha256="d" * 64, scope_nonce="another-private-scope"
    )
    assert epoch is None
    assert reason == "wrong_scope"
    restored, reason = cache.restore(
        [1, 2], binding_sha256="e" * 64, scope_nonce=SCOPE
    )
    assert restored is None
    assert reason == "wrong_binding"


@pytest.mark.parametrize("scope", [None, "", "too-short", "x" * 257])
def test_scope_nonce_has_bounded_string_metadata(scope):
    with pytest.raises(ValueError, match="invalid private scope"):
        PrivatePrefixCache(
            binding_sha256="f" * 64,
            scope_nonce=scope,  # type: ignore[arg-type]
            max_entries=1,
            max_bytes=1,
        )


def test_adapter_rejected_input_replaces_old_trace_without_importing_mlx():
    cache = PrivatePrefixCache(
        binding_sha256="1" * 64,
        scope_nonce=SCOPE,
        max_entries=1,
        max_bytes=1,
    )
    adapter = IdenticalPromptReuse(
        object(),
        object(),
        cache,
        expected_binding_sha256="1" * 64,
        scope_nonce=SCOPE,
    )
    with pytest.raises(ValueError, match="at least two"):
        adapter.stream([1], max_tokens=1)
    assert adapter.last_trace.status == "rejected"
    assert adapter.last_trace.failure_type == "input_validation"
    assert adapter.last_trace.fallback_reason == "invalid_request"


def test_busy_preparation_does_not_overwrite_active_trace():
    cache = PrivatePrefixCache(
        binding_sha256="2" * 64,
        scope_nonce=SCOPE,
        max_entries=1,
        max_bytes=1,
    )
    adapter = IdenticalPromptReuse(
        object(), object(), cache,
        expected_binding_sha256="2" * 64,
        scope_nonce=SCOPE,
    )
    adapter.last_trace = adapter.last_trace.__class__(
        "in_progress", "none", True, False, 1, "none", 1, 2, 3
    )
    before = adapter.last_trace
    assert adapter._generation_lock.acquire(blocking=False)
    try:
        with pytest.raises(RuntimeError, match="busy"):
            adapter.stream([1, 2], max_tokens=1)
    finally:
        adapter._generation_lock.release()
    assert adapter.last_trace == before
