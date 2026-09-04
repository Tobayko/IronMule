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

Since the two trees were merged there is a *third* ``test_benchmark.py``: the
engine package's own, which legitimately owns ``tests/test_benchmark.py``. The
research tree's copy was renamed to ``tests/test_friday_benchmark.py`` (likewise
``test_friday_cli.py`` and ``test_friday_evidence.py``), so a shared-helper
import has to name that file, not the engine's.
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


# -- collection on a machine that is not the target device --------------------
#
# The research suite asserts MLX, Metal and model behaviour, and AGENTS.md is
# explicit that such a test must have run on the target device -- a CI runner is
# not one. Those modules therefore need MLX present and the pinned IronMule
# worktree checked out (the worktree is gitignored, so a fresh clone never has
# it). Rather than maintaining a list of paths in the workflow file, a module
# whose precondition is missing removes itself from collection here, which keeps
# working when files are added or renamed.
#
# On this Mac both preconditions hold and nothing is skipped.

def _missing(name: str) -> bool:
    from importlib.util import find_spec

    try:
        return find_spec(name) is None
    except (ImportError, ValueError):
        return True


_REQUIRES_MLX = _missing("mlx")
_REQUIRES_ENGINE = not (IRONMULE / "ironmule" / "runtime.py").is_file()
# friday_optimizer uses MappingProxyType as a dataclass default
# (candidates.py:72, history.py:162). Python 3.12 relaxed the mutable-default
# check; 3.11 rejects it at import time. requirements-apple-silicon.txt already
# says 3.12, so this records the floor rather than inventing one -- the engine
# package keeps its own >=3.10 claim and its tests keep running on 3.11.
_REQUIRES_PY312 = sys.version_info < (3, 12)


def _needs(path: Path, tokens: tuple[str, ...]) -> bool:
    try:
        head = path.read_text(errors="ignore")
    except OSError:
        return False
    return any(token in head for token in tokens)


def collect_ignore_glob_hook(path: Path) -> bool:
    """True when *path* cannot be collected in this environment."""

    if _REQUIRES_MLX and _needs(path, ("import mlx", "from mlx")):
        return True
    if _REQUIRES_ENGINE and _needs(path, ("friday-optimizer-ironmule", "from ironmule", "import ironmule")):
        return True
    if _REQUIRES_PY312 and _needs(path, ("friday_optimizer",)):
        return True
    return False


def pytest_ignore_collect(collection_path, config):  # noqa: ARG001 - pytest hook
    path = Path(str(collection_path))
    if path.suffix != ".py" or not path.name.startswith("test_"):
        return None
    return True if collect_ignore_glob_hook(path) else None
