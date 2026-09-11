"""A circuit breaker that survives the process it tripped in.

In the sealed runtimes the breaker is a single in-RAM latch
(``friday_head_skip_runtime/executor.py:192-195``): thread-safe, but gone at the
next start. For a manually launched measurement run that is defensible — the
operator sees the failure. For a service that dispatches without asking it is
not: a knob that failed once would silently come back on the next request.

Here the latch is written through to durable storage and read back at
construction. Two directions matter and both point the same way:

* a latch that cannot be **read** counts as tripped — an unreadable breaker
  must not authorise an optimised path;
* a latch that cannot be **written** still trips in memory, and the failure to
  persist is raised, because silently forgetting a failure is the one outcome
  worth refusing.

Falling back always means the baseline, which is correct and merely slower.
"""

from __future__ import annotations

import threading
from typing import Protocol


class BreakerError(RuntimeError):
    """A tripped state could not be made durable."""


class Latch(Protocol):
    """Durable storage for one runtime's first failure reason."""

    def read(self) -> str | None:
        """Return the persisted trip reason, or ``None`` if never tripped."""

    def write(self, reason: str) -> None:
        """Persist the first trip reason. Idempotent for a reason already stored."""


class MemoryLatch:
    """Process-local latch: the previous behaviour, kept for tests and dry runs."""

    def __init__(self, reason: str | None = None) -> None:
        self._reason = reason

    def read(self) -> str | None:
        return self._reason

    def write(self, reason: str) -> None:
        if self._reason is None:
            self._reason = reason


class PersistentLatch:
    """Latch backed by an append-only hash-chained runtime history.

    ``load`` returns the reason of the newest stored failure (or ``None``);
    ``append`` writes one. Both are supplied by the runtime package, because the
    record contract differs per package while this logic does not.
    """

    def __init__(self, load, append) -> None:
        if not callable(load) or not callable(append):
            raise BreakerError("persistent latch requires callable load and append")
        self._load = load
        self._append = append

    def read(self) -> str | None:
        try:
            reason = self._load()
        except Exception as exc:  # an unreadable history is a tripped breaker
            return f"latch_unreadable:{type(exc).__name__}"
        if reason is None:
            return None
        if not isinstance(reason, str) or not reason:
            return "latch_unreadable:malformed_reason"
        return reason

    def write(self, reason: str) -> None:
        try:
            self._append(reason)
        except Exception as exc:
            raise BreakerError("circuit breaker state could not be persisted") from exc


class CircuitBreaker:
    """Read the latch once, then latch in memory and write through on the first trip."""

    def __init__(self, latch: Latch | None = None) -> None:
        self._latch = latch if latch is not None else MemoryLatch()
        self._lock = threading.Lock()
        self._reason: str | None = self._latch.read()
        self._persisted = self._reason is not None

    @property
    def reason(self) -> str | None:
        with self._lock:
            return self._reason

    @property
    def tripped(self) -> bool:
        return self.reason is not None

    def trip(self, failure: BaseException | str) -> str:
        """Latch on the first failure and persist it. Later failures do not overwrite."""

        reason = failure if isinstance(failure, str) else type(failure).__name__
        if not reason:
            reason = "unspecified_failure"
        with self._lock:
            first = self._reason is None
            if first:
                self._reason = reason
            current = self._reason
            persist = first or not self._persisted
        if persist:
            self._latch.write(current)
            with self._lock:
                self._persisted = True
        return current


__all__ = ["BreakerError", "CircuitBreaker", "Latch", "MemoryLatch", "PersistentLatch"]
