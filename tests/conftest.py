"""The single place test path setup happens.

Two facts collide here. The repository has no ``tests/__init__.py``, so ``tests``
is a namespace package whose ``__path__`` is rebuilt from ``sys.path`` on every
import. The IronMule engine lives in a git worktree
(``.worktrees/friday-optimizer-ironmule``) that has its **own** ``tests/`` tree.
When a test module did ``sys.path.insert(0, <worktree>)`` at import time, the
worktree's ``tests/test_benchmark.py`` shadowed the real one and
``from tests.test_benchmark import FakeBackend`` failed during collection,
aborting the whole suite under ``-n auto``.

Fix: the repo root goes first (``tests`` resolves to ``<repo>/tests``); the
IronMule worktree is only *appended*, so ``from ironmule.runtime import ...``
resolves while the worktree's stray ``tests/`` sorts last in ``__path__`` and
never wins.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IRONMULE = ROOT / ".worktrees" / "friday-optimizer-ironmule"

if sys.path[:1] != [str(ROOT)]:
    if str(ROOT) in sys.path:
        sys.path.remove(str(ROOT))
    sys.path.insert(0, str(ROOT))

if IRONMULE.is_dir() and str(IRONMULE) not in sys.path:
    sys.path.append(str(IRONMULE))
