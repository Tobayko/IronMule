"""Shared runtime substrate: provenance, hash-chained history, dispatch control.

The three runtime packages (``friday_runtime``, ``friday_runtime_n10``,
``friday_head_skip_runtime``) carry near-identical copies of this machinery —
``executor.py`` differs by a single docstring line, ``history.py`` by twelve.
Those packages are **not** refactored onto this one: every one of them hashes
its own ``*.py`` files into ``code_sha256`` and compares that against a frozen
or sealed constant, so editing them would change the very identity their scope
check verifies (``AGENTS.md``: sealed packages stay byte-identical).

This package is therefore the substrate for *new* runtime code — calibration,
serving, adaptive dispatch — and carries one behavioural fix the sealed copies
cannot receive: the circuit breaker survives process restarts.
"""

from __future__ import annotations

from .breaker import BreakerError, CircuitBreaker, MemoryLatch, PersistentLatch
from .controller import DispatchController, DispatchDecision, RuntimeExecutionError
from .history import HistoryConflict, HistoryError, PersistenceOutcome, RuntimeHistory, HistorySpec
from .provenance import ProvenanceError, ProvenanceSpec, collect_provenance

__all__ = [
    "BreakerError",
    "CircuitBreaker",
    "DispatchController",
    "DispatchDecision",
    "HistoryConflict",
    "HistoryError",
    "HistorySpec",
    "MemoryLatch",
    "PersistenceOutcome",
    "PersistentLatch",
    "ProvenanceError",
    "ProvenanceSpec",
    "RuntimeExecutionError",
    "RuntimeHistory",
    "collect_provenance",
]
