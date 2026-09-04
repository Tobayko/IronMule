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


# -- collection away from the target device -----------------------------------
#
# The research suite is bound to *this* machine by design, not by accident. Its
# evidence lives in gitignored SQLite databases under `.friday-data/`, its models
# in a validated local cache, its engine in a pinned worktree under
# `.worktrees/`, and several of its tests spawn `<repo>/.venv/bin/python` to
# drive a measurement script end to end. AGENTS.md is explicit that a test
# asserting MLX, Metal or model behaviour must have run on the target device --
# so a CI runner is not a place where that suite can say anything true.
#
# The first attempt enumerated what was missing, one precondition at a time, and
# each fix uncovered the next dependency: MLX, then the worktree, then `.venv`,
# then the evidence databases, then the model cache. The list was the wrong
# shape. One question replaces it: **is this the target device?** If it is not,
# the research tree is not collected at all, and CI checks the engine package --
# which is exactly what it can check honestly.
#
# The split is clean rather than approximate: 105 test modules import a
# `friday_*` package and none of them is one of the engine's 34.
#
# On this Mac every precondition holds and nothing is dropped.

def _missing(name: str) -> bool:
    from importlib.util import find_spec

    try:
        return find_spec(name) is None
    except (ImportError, ValueError):
        return True


#: The target device is the machine that carries the project's own environment
#: and its measured evidence. Both are gitignored, so no clone is one by default.
IS_TARGET_DEVICE = (ROOT / ".venv" / "bin" / "python").is_file() and (
    ROOT / ".friday-data"
).is_dir()

#: The engine package's own suite. It depends on nothing outside the repository,
#: so it runs anywhere -- that is what CI checks.
ENGINE_TESTS = Path(__file__).resolve().parent / "engine"

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

    if ENGINE_TESTS == path or ENGINE_TESTS in path.parents:
        return False  # the engine's suite is self-contained; it runs anywhere
    if not IS_TARGET_DEVICE:
        return True  # everything else is the research tree
    # On a target device the research tree can still be missing a piece.
    if _REQUIRES_MLX and _needs(path, ("import mlx", "from mlx")):
        return True
    if _REQUIRES_ENGINE and _needs(path, ("friday-optimizer-ironmule",)):
        return True
    if _REQUIRES_PY312 and _needs(path, ("friday_optimizer",)):
        return True
    return False


def pytest_ignore_collect(collection_path, config):  # noqa: ARG001 - pytest hook
    path = Path(str(collection_path))
    if path.is_dir():
        return None
    if path.suffix != ".py" or not path.name.startswith("test_"):
        return None
    return True if collect_ignore_glob_hook(path.resolve()) else None
