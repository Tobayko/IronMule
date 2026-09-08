"""Worker-private, transactional ownership for the opt-in prefix candidate.

This module adds no inference implementation of its own.  It delays the
existing adapter's terminal cache publication until the worker has made its
accept/cancel decision under the worker cancellation lock.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import threading
import time
from typing import Any, Iterator, Sequence

from .prefix_reuse import (
    IdenticalPromptReuse,
    PrefixReuseResponse,
    PrefixReuseStats,
    PrefixReuseTrace,
    PrivatePrefixCache,
)


@dataclass(frozen=True)
class WorkerPrefixSessionMetadata:
    """Secret-free lifecycle evidence returned to the worker.

    ``status`` records the worker/session decision.  ``trace.status`` remains
    the adapter's execution evidence, so discarding a completed generation
    never falsely claims that the model did not complete.
    """

    status: str
    commit_status: str
    trace: PrefixReuseTrace
    stats: PrefixReuseStats


class _DeferredCommitCache:
    """Cache-shaped proxy which records, but does not publish, a commit."""

    def __init__(self, cache: PrivatePrefixCache) -> None:
        self._cache = cache
        self._pending: tuple[tuple[int, ...], list[Any], int, str, str] | None = None
        self._lock = threading.Lock()

    def restore(
        self, prompt_ids: Sequence[int], **kwargs: Any
    ) -> tuple[list[Any] | None, str]:
        return self._cache.restore(prompt_ids, **kwargs)

    def begin_cold(self, **kwargs: Any) -> tuple[int | None, str]:
        return self._cache.begin_cold(**kwargs)

    def commit(
        self,
        prompt_ids: Sequence[int],
        snapshot: list[Any],
        *,
        epoch: int,
        binding_sha256: str,
        scope_nonce: str,
    ) -> str:
        # IdenticalPromptReuse already owns the captured snapshot.  Keep that
        # object only until finalize; PrivatePrefixCache performs the owning
        # copy if the transaction is accepted.
        pending = (
            tuple(prompt_ids), snapshot, epoch, binding_sha256, scope_nonce
        )
        with self._lock:
            self._pending = pending
        return "deferred_commit"

    def finalize(self, accepted: bool) -> tuple[str, int]:
        with self._lock:
            pending = self._pending
            self._pending = None
        if pending is None:
            return "no_pending_commit", 0
        if not accepted:
            return "discarded", 0
        started = time.perf_counter_ns()
        prompt_ids, snapshot, epoch, binding, scope = pending
        try:
            result = self._cache.commit(
                prompt_ids,
                snapshot,
                epoch=epoch,
                binding_sha256=binding,
                scope_nonce=scope,
            )
        except Exception:
            # Prefix reuse is auxiliary and must never turn valid inference
            # into a worker failure.
            result = "commit_error"
        finally:
            # Do not retain request tokens or cache state past linearization.
            del prompt_ids, snapshot, pending
        return result, time.perf_counter_ns() - started

    def discard(self) -> bool:
        with self._lock:
            existed = self._pending is not None
            self._pending = None
        return existed


class WorkerPrefixSession:
    """One worker's private, bounded prefix-reuse transaction coordinator."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        binding_sha256: str,
        scope_nonce: str,
        max_entries: int,
        max_bytes: int,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._binding_sha256 = binding_sha256
        self._scope_nonce = scope_nonce
        self._cache = PrivatePrefixCache(
            binding_sha256=binding_sha256,
            scope_nonce=scope_nonce,
            max_entries=max_entries,
            max_bytes=max_bytes,
        )
        self._lock = threading.Lock()
        self._adapter: IdenticalPromptReuse | None = None
        self._proxy: _DeferredCommitCache | None = None
        self._closed = False
        self._last_trace = PrefixReuseTrace(
            "not_started", "none", False, False, 0, "not_started", 0, 0, 0
        )
        self._last_commit_status = "not_started"

    def __repr__(self) -> str:
        with self._lock:
            active = self._adapter is not None
            closed = self._closed
        stats = self._cache.stats()
        return (
            f"{type(self).__name__}(active={active}, closed={closed}, "
            f"entries={stats.entries}, bytes={stats.bytes})"
        )

    def stream(
        self, prompt_ids: Sequence[int], *, max_tokens: int
    ) -> Iterator[PrefixReuseResponse]:
        with self._lock:
            if self._closed:
                raise RuntimeError("worker prefix session is closed")
            if self._adapter is not None:
                raise RuntimeError("worker prefix session already has an active generation")
            proxy = _DeferredCommitCache(self._cache)
            adapter = IdenticalPromptReuse(
                self._model,
                self._tokenizer,
                proxy,  # type: ignore[arg-type]
                expected_binding_sha256=self._binding_sha256,
                scope_nonce=self._scope_nonce,
            )
            self._proxy = proxy
            self._adapter = adapter
            self._last_commit_status = "pending_generation"
        try:
            return adapter.stream(prompt_ids, max_tokens=max_tokens)
        except BaseException:
            with self._lock:
                self._last_trace = adapter.last_trace
                self._last_commit_status = "no_pending_commit"
                self._adapter = None
                self._proxy = None
            raise

    def finalize(self, accepted: bool) -> WorkerPrefixSessionMetadata:
        """Linearize the worker decision; caller holds its cancellation lock."""
        if type(accepted) is not bool:
            raise ValueError("accepted must be a boolean")
        with self._lock:
            adapter = self._adapter
            proxy = self._proxy
            if adapter is None or proxy is None:
                return self._metadata_locked("idle", self._last_commit_status)
            trace = adapter.last_trace
            terminal_accept = accepted and trace.status == "completed"
            commit_status, commit_ns = proxy.finalize(terminal_accept)
            if accepted and not terminal_accept:
                commit_status = "not_terminal"
            stored = commit_status == "stored"
            trace = replace(trace, commit_host_ns=commit_ns)
            if stored:
                trace = replace(trace, cache_stored=True, fallback_reason="none")
            elif trace.cache_hit:
                trace = replace(trace, cache_stored=False, fallback_reason="none")
            else:
                trace = replace(
                    trace,
                    cache_stored=False,
                    fallback_reason=commit_status,
                )
            self._last_trace = trace
            self._last_commit_status = commit_status
            self._adapter = None
            self._proxy = None
            status = (
                "accepted"
                if terminal_accept
                else "rejected" if accepted else "discarded"
            )
            return self._metadata_locked(status, commit_status)

    def clear(self) -> WorkerPrefixSessionMetadata:
        with self._lock:
            if self._proxy is not None:
                self._proxy.discard()
            self._cache.clear()
            self._last_commit_status = "cleared"
            return self._metadata_locked("cleared", "cleared")

    def close(self) -> WorkerPrefixSessionMetadata:
        with self._lock:
            if not self._closed:
                if self._proxy is not None:
                    self._proxy.discard()
                self._cache.close()
                self._closed = True
                self._last_commit_status = "closed"
            return self._metadata_locked("closed", self._last_commit_status)

    def _metadata_locked(
        self, status: str, commit_status: str
    ) -> WorkerPrefixSessionMetadata:
        return WorkerPrefixSessionMetadata(
            status=status,
            commit_status=commit_status,
            trace=self._last_trace,
            stats=self._cache.stats(),
        )


__all__ = ["WorkerPrefixSession", "WorkerPrefixSessionMetadata"]
