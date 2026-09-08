"""Opt-in, session-private reuse of an identical MLX-LM prompt prefill.

This module is deliberately not wired into the product service.  It implements
the narrow PROD11 candidate using only MLX-LM's public generation seam.  MLX is
imported lazily when :meth:`IdenticalPromptReuse.stream` is iterated.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import dataclass
import re
import threading
import time
from typing import Any, Iterator, Sequence


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
PREFILL_STEP_SIZE = 2048


def _valid_scope_nonce(value: object) -> bool:
    return type(value) is str and 16 <= len(value) <= 256


@dataclass(frozen=True)
class PrefixReuseOptions:
    """The intentionally closed execution domain of the candidate."""

    prefill_step_size: int = PREFILL_STEP_SIZE
    decoding: str = "greedy"
    logits_processors: None = None
    draft_model: None = None
    kv_bits: None = None

    def __post_init__(self) -> None:
        if (
            type(self.prefill_step_size) is not int
            or self.prefill_step_size != PREFILL_STEP_SIZE
        ):
            raise ValueError("unsupported prefix-reuse options")
        if self.decoding != "greedy":
            raise ValueError("unsupported prefix-reuse options")
        if (
            self.logits_processors is not None
            or self.draft_model is not None
            or self.kv_bits is not None
        ):
            raise ValueError("unsupported prefix-reuse options")


@dataclass(frozen=True)
class PrefixReuseStats:
    hits: int
    misses: int
    inserts: int
    skipped_oversize: int
    evictions: int
    entries: int
    bytes: int


@dataclass(frozen=True, repr=False)
class PrefixReuseResponse:
    """Generation fields with honest full-prompt accounting.

    ``prompt_tps`` is intentionally ``None``: MLX-LM calculates it from the
    one-token suffix on a hit.  ``suffix_prompt_tps`` retains that diagnostic
    without relabelling it as full-prompt throughput.
    """

    text: str
    token: int
    logprobs: Any
    from_draft: bool
    prompt_tokens: int
    prompt_tps: None
    suffix_prompt_tps: float | None
    generation_tokens: int
    generation_tps: float
    peak_memory: float
    finish_reason: str | None

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(prompt_tokens={self.prompt_tokens}, "
            f"generation_tokens={self.generation_tokens}, finish_reason={self.finish_reason!r})"
        )


@dataclass(frozen=True)
class PrefixReuseTrace:
    status: str
    failure_type: str
    cache_hit: bool
    cache_stored: bool
    reused_tokens: int
    fallback_reason: str
    restore_host_ns: int
    capture_host_ns: int
    commit_host_ns: int


@dataclass(repr=False)
class _Entry:
    snapshot: list[Any]
    nbytes: int


class PrivatePrefixCache:
    """A bounded cache owned by exactly one backend session.

    The binding digest is supplied by the owner and must cover model snapshot
    and revision, tokenizer/chat-template identity, environment, code, and the
    fixed generation options.  The scope nonce is retained only as an ownership
    secret; neither it nor prompt tokens appear in repr, errors, or stats.
    """

    def __init__(
        self,
        *,
        binding_sha256: str,
        scope_nonce: str,
        max_entries: int,
        max_bytes: int,
    ) -> None:
        if type(binding_sha256) is not str or not _SHA256.fullmatch(binding_sha256):
            raise ValueError("invalid cache binding")
        if not _valid_scope_nonce(scope_nonce):
            raise ValueError("invalid private scope")
        if type(max_entries) is not int or type(max_bytes) is not int:
            raise ValueError("cache limits must be positive integers")
        if max_entries < 1 or max_bytes < 1:
            raise ValueError("cache limits must be positive")
        self._binding_sha256 = binding_sha256
        self._scope_nonce = scope_nonce
        self._max_entries = int(max_entries)
        self._max_bytes = int(max_bytes)
        self._entries: OrderedDict[tuple[int, ...], _Entry] = OrderedDict()
        self._bytes = 0
        self._epoch = 0
        self._closed = False
        self._hits = self._misses = self._inserts = 0
        self._skipped_oversize = self._evictions = 0
        self._lock = threading.Lock()

    @property
    def binding_sha256(self) -> str:
        return self._binding_sha256

    def __repr__(self) -> str:
        stats = self.stats()
        return (
            f"{type(self).__name__}(entries={stats.entries}, bytes={stats.bytes}, "
            f"closed={self._closed})"
        )

    def begin_cold(
        self, *, binding_sha256: str, scope_nonce: str
    ) -> tuple[int | None, str]:
        """Return an epoch token for a possible terminal commit."""
        with self._lock:
            if self._closed:
                return None, "cache_closed"
            mismatch = self._binding_mismatch(binding_sha256, scope_nonce)
            if mismatch is not None:
                return None, mismatch
            return self._epoch, "none"

    def restore(
        self,
        prompt_ids: Sequence[int],
        *,
        binding_sha256: str,
        scope_nonce: str,
    ) -> tuple[list[Any] | None, str]:
        key = _prompt_key(prompt_ids)
        with self._lock:
            if self._closed:
                return None, "cache_closed"
            mismatch = self._binding_mismatch(binding_sha256, scope_nonce)
            if mismatch is not None:
                self._misses += 1
                return None, mismatch
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None, "cache_miss"
            # MLX-LM itself uses deepcopy for LRUPromptCache and batch cache
            # isolation.  Copy under the metadata lock so clear/close cannot
            # overtake a successful restore.
            try:
                restored = copy.deepcopy(entry.snapshot)
            except Exception:
                return None, "restore_error"
            self._entries.move_to_end(key)
            self._hits += 1
            return restored, "none"

    def commit(
        self,
        prompt_ids: Sequence[int],
        snapshot: list[Any],
        *,
        epoch: int,
        binding_sha256: str,
        scope_nonce: str,
    ) -> str:
        """Commit only a terminal cold generation from the current epoch."""
        key = _prompt_key(prompt_ids)
        nbytes = _cache_nbytes(snapshot)
        with self._lock:
            if self._closed:
                return "cache_closed"
            mismatch = self._binding_mismatch(binding_sha256, scope_nonce)
            if mismatch is not None:
                return mismatch
            if epoch != self._epoch:
                return "stale_epoch"
            if nbytes > self._max_bytes:
                self._skipped_oversize += 1
                return "oversize"
            try:
                # The entry owns its object graph.  The callback's pending
                # snapshot remains caller-local and can never mutate it.
                owned_snapshot = copy.deepcopy(snapshot)
            except Exception:
                return "commit_copy_error"
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._bytes -= previous.nbytes
            self._entries[key] = _Entry(snapshot=owned_snapshot, nbytes=nbytes)
            self._bytes += nbytes
            self._inserts += 1
            while len(self._entries) > self._max_entries or self._bytes > self._max_bytes:
                _, removed = self._entries.popitem(last=False)
                self._bytes -= removed.nbytes
                self._evictions += 1
            return "stored"

    def _binding_mismatch(self, binding_sha256: str, scope_nonce: str) -> str | None:
        if binding_sha256 != self._binding_sha256:
            return "wrong_binding"
        if scope_nonce != self._scope_nonce:
            return "wrong_scope"
        return None

    def clear(self) -> None:
        with self._lock:
            self._epoch += 1
            self._entries.clear()
            self._bytes = 0

    def close(self) -> None:
        with self._lock:
            self._epoch += 1
            self._entries.clear()
            self._bytes = 0
            self._closed = True

    def stats(self) -> PrefixReuseStats:
        with self._lock:
            return PrefixReuseStats(
                hits=self._hits,
                misses=self._misses,
                inserts=self._inserts,
                skipped_oversize=self._skipped_oversize,
                evictions=self._evictions,
                entries=len(self._entries),
                bytes=self._bytes,
            )


class IdenticalPromptReuse:
    """Public-API adapter for identical canonical token sequences only."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        cache: PrivatePrefixCache,
        *,
        expected_binding_sha256: str,
        scope_nonce: str,
    ) -> None:
        if (
            type(expected_binding_sha256) is not str
            or not _SHA256.fullmatch(expected_binding_sha256)
        ):
            raise ValueError("invalid expected binding")
        if not _valid_scope_nonce(scope_nonce):
            raise ValueError("invalid expected scope")
        self._model = model
        self._tokenizer = tokenizer
        self._cache = cache
        self._expected_binding_sha256 = expected_binding_sha256
        self._scope_nonce = scope_nonce
        self.options = PrefixReuseOptions()
        self.last_trace = PrefixReuseTrace(
            "not_started", "none", False, False, 0, "not_started", 0, 0, 0
        )
        self._generation_lock = threading.Lock()

    def stream(
        self, prompt_ids: Sequence[int], *, max_tokens: int
    ) -> Iterator[PrefixReuseResponse]:
        """Validate and claim this adapter, returning one guarded generation.

        One adapter supports session owns at most one active generation.  The claim
        is non-blocking; independent cache restores may still be exercised via
        separate adapters that share a ``PrivatePrefixCache``.
        """
        if not self._generation_lock.acquire(blocking=False):
            # Do not let a concurrent rejection overwrite the active trace.
            raise RuntimeError("prefix-reuse adapter is busy")
        try:
            self.last_trace = PrefixReuseTrace(
                "prepared", "none", False, False, 0, "not_started", 0, 0, 0
            )
            try:
                ids = _prompt_key(prompt_ids)
                if len(ids) < 2:
                    raise ValueError("prefix reuse requires at least two prompt tokens")
                if type(max_tokens) is not int or max_tokens < 1:
                    raise ValueError("max_tokens must be positive")
            except Exception:
                self.last_trace = PrefixReuseTrace(
                    "rejected", "input_validation", False, False, 0,
                    "invalid_request", 0, 0, 0,
                )
                raise
        finally:
            self._generation_lock.release()
        return self._guarded_stream(ids, max_tokens=max_tokens)

    def _guarded_stream(
        self, ids: tuple[int, ...], *, max_tokens: int
    ) -> Iterator[PrefixReuseResponse]:
        if not self._generation_lock.acquire(blocking=False):
            raise RuntimeError("prefix-reuse adapter is busy")
        self.last_trace = PrefixReuseTrace(
            "in_progress", "none", False, False, 0, "in_progress", 0, 0, 0
        )
        try:
            yield from self._stream_validated(ids, max_tokens=max_tokens)
            if self.last_trace.status == "in_progress":
                self.last_trace = _trace_status(
                    self.last_trace, "incomplete", "missing_terminal"
                )
        except GeneratorExit:
            if self.last_trace.status != "completed":
                self.last_trace = _trace_status(
                    self.last_trace, "cancelled", "generator_closed"
                )
            raise
        except BaseException:
            self.last_trace = _trace_status(
                self.last_trace, "failed", "generation_error"
            )
            raise
        finally:
            self._generation_lock.release()

    def _stream_validated(
        self, ids: tuple[int, ...], *, max_tokens: int
    ) -> Iterator[PrefixReuseResponse]:
        """Generate from a cold prompt or a private clone of its N-1 snapshot.

        Cache insertion happens before the terminal response is yielded.  A
        cancelled or failed generation therefore cannot publish a pending cold
        snapshot.  Cache admission failure never stops generation.
        """
        # Deliberately lazy: importing this module is safe in metadata-only
        # processes and tests.
        from mlx_lm import stream_generate
        from mlx_lm.models.cache import make_prompt_cache

        setup_started = time.perf_counter_ns()
        try:
            prompt_cache, restore_reason = self._cache.restore(
                ids,
                binding_sha256=self._expected_binding_sha256,
                scope_nonce=self._scope_nonce,
            )
        except Exception:
            prompt_cache = None
            restore_reason = "restore_error"
        restore_ns = time.perf_counter_ns() - setup_started
        hit = prompt_cache is not None
        pending_snapshot: list[Any] | None = None
        epoch: int | None = None
        capture_ns = 0
        commit_ns = 0
        stored = False

        if hit:
            generation_prompt = [ids[-1]]
            callback = None
        else:
            try:
                epoch, begin_reason = self._cache.begin_cold(
                    binding_sha256=self._expected_binding_sha256,
                    scope_nonce=self._scope_nonce,
                )
            except Exception:
                epoch = None
                begin_reason = "cache_metadata_error"
            if restore_reason == "cache_miss" and begin_reason != "none":
                restore_reason = begin_reason
            prompt_cache = make_prompt_cache(self._model)
            generation_prompt = list(ids)

            def callback(processed: int, total: int) -> None:
                nonlocal pending_snapshot, capture_ns, restore_reason
                if (
                    epoch is not None
                    and processed > 0
                    and processed == total - 1
                    and pending_snapshot is None
                ):
                    capture_started = time.perf_counter_ns()
                    try:
                        pending_snapshot = copy.deepcopy(prompt_cache)
                    except Exception:
                        # Reuse is auxiliary: inability to snapshot must not
                        # convert a valid cold generation into an error.
                        pending_snapshot = None
                        restore_reason = "capture_error"
                    finally:
                        capture_ns += time.perf_counter_ns() - capture_started

        generated = stream_generate(
            self._model,
            self._tokenizer,
            generation_prompt,
            max_tokens=max_tokens,
            prompt_cache=prompt_cache,
            prefill_step_size=PREFILL_STEP_SIZE,
            logits_processors=None,
            draft_model=None,
            kv_bits=None,
            prompt_progress_callback=callback,
        )
        try:
            for response in generated:
                terminal = response.finish_reason in {"stop", "length"}
                if terminal and not hit and pending_snapshot is not None and epoch is not None:
                    # Admission/eviction is auxiliary and cannot fail inference.
                    try:
                        commit_started = time.perf_counter_ns()
                        commit_result = self._cache.commit(
                            ids,
                            pending_snapshot,
                            epoch=epoch,
                            binding_sha256=self._expected_binding_sha256,
                            scope_nonce=self._scope_nonce,
                        )
                        commit_ns += time.perf_counter_ns() - commit_started
                        stored = commit_result == "stored"
                        if not stored:
                            restore_reason = commit_result
                    except Exception:
                        commit_ns += time.perf_counter_ns() - commit_started
                        restore_reason = "commit_error"
                    pending_snapshot = None
                if terminal:
                    if (
                        not hit
                        and epoch is not None
                        and not stored
                        and restore_reason == "cache_miss"
                    ):
                        restore_reason = "snapshot_unavailable"
                    self.last_trace = PrefixReuseTrace(
                        status="completed",
                        failure_type="none",
                        cache_hit=hit,
                        cache_stored=stored,
                        reused_tokens=len(ids) - 1 if hit else 0,
                        fallback_reason="none" if hit or stored else restore_reason,
                        restore_host_ns=restore_ns,
                        capture_host_ns=capture_ns,
                        commit_host_ns=commit_ns,
                    )
                yield PrefixReuseResponse(
                    text=response.text,
                    token=response.token,
                    logprobs=response.logprobs,
                    from_draft=response.from_draft,
                    prompt_tokens=len(ids),
                    prompt_tps=None,
                    suffix_prompt_tps=getattr(response, "prompt_tps", None) if hit else None,
                    generation_tokens=response.generation_tokens,
                    generation_tps=response.generation_tps,
                    peak_memory=response.peak_memory,
                    finish_reason=response.finish_reason,
                )
        finally:
            close = getattr(generated, "close", None)
            if close is not None:
                close()


def _prompt_key(prompt_ids: Sequence[int]) -> tuple[int, ...]:
    if type(prompt_ids) not in (list, tuple):
        raise ValueError("canonical prompt tokens must be a list or tuple")
    if any(type(token) is not int or token < 0 for token in prompt_ids):
        raise ValueError("invalid canonical prompt tokens")
    return tuple(prompt_ids)


def _trace_status(
    trace: PrefixReuseTrace, status: str, failure_type: str
) -> PrefixReuseTrace:
    return PrefixReuseTrace(
        status=status,
        failure_type=failure_type,
        cache_hit=trace.cache_hit,
        cache_stored=trace.cache_stored,
        reused_tokens=trace.reused_tokens,
        fallback_reason=trace.fallback_reason,
        restore_host_ns=trace.restore_host_ns,
        capture_host_ns=trace.capture_host_ns,
        commit_host_ns=trace.commit_host_ns,
    )


def _cache_nbytes(snapshot: Sequence[Any]) -> int:
    try:
        values = [cache.nbytes for cache in snapshot]
    except Exception as exc:
        raise ValueError("cache snapshot has no byte accounting") from exc
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("cache snapshot has invalid byte accounting")
    return sum(values)


__all__ = [
    "IdenticalPromptReuse",
    "PREFILL_STEP_SIZE",
    "PrefixReuseOptions",
    "PrefixReuseResponse",
    "PrefixReuseStats",
    "PrefixReuseTrace",
    "PrivatePrefixCache",
]
