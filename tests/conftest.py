"""The single place test path setup happens.

The repository has no ``tests/__init__.py``, so ``tests`` is a namespace package
whose ``__path__`` is rebuilt from ``sys.path`` on every import. The repo root
therefore goes first and stays first, so ``tests`` always resolves to
``<repo>/tests`` and ``from ironmule.runtime import ...`` resolves to the engine
that ships with this checkout. There is no second engine tree to shadow it.

Since the two trees were merged there is a *third* ``test_benchmark.py``. The
engine package's own now lives in ``tests/engine/`` together with the rest of its
suite; the research tree's copy was renamed to ``tests/test_friday_benchmark.py``
(likewise ``test_friday_cli.py`` and ``test_friday_evidence.py``), so a
shared-helper import has to name that file rather than the engine's.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
#: The research tree is its own source root. Its packages keep flat names because
#: their provenance manifests hash paths relative to it, so renaming them would
#: change a run identity for no reason other than where the directory sits.
RESEARCH = ROOT / "research"

if sys.path[:1] != [str(ROOT)]:
    if str(ROOT) in sys.path:
        sys.path.remove(str(ROOT))
    sys.path.insert(0, str(ROOT))

if str(RESEARCH) not in sys.path:
    sys.path.insert(1, str(RESEARCH))


# -- collection away from the target device -----------------------------------
#
# The research suite is bound to *this* machine by design, not by accident. Its
# evidence lives in gitignored SQLite databases under `.friday-data/`, its models
# in a validated local cache, and several of its tests spawn
# `<repo>/.venv/bin/python` to drive a measurement script end to end. AGENTS.md
# is explicit that a test asserting MLX, Metal or model behaviour must have run
# on the target device --
# so a CI runner is not a place where that suite can say anything true.
#
# The first attempt enumerated what was missing, one precondition at a time, and
# each fix uncovered the next dependency: MLX, then `.venv`, then the evidence
# databases, then the model cache. The list was the wrong shape. One question
# replaces it: **is this the target device?** If it is not,
# the research tree is not collected at all, and CI checks the engine package --
# which is exactly what it can check honestly.
#
# The split is clean rather than approximate: 105 test modules import a
# `friday_*` package and none of them is one of the engine's 34.
#
# Two research modules are the exception, because they check public claims against
# committed files only: the numeric-plan table and the documented numbers (README,
# ledger, constants) against the runs they cite. They are collected everywhere; the two
# tests in them that need `.friday-data/` skip themselves when it is absent.
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

#: Research modules that read only committed files and guard public claims.
PORTABLE_RESEARCH_TESTS = frozenset({"test_numeric_plans.py", "test_documented_claims.py"})

_REQUIRES_MLX = _missing("mlx")


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
    if path.parent == ENGINE_TESTS.parent and path.name in PORTABLE_RESEARCH_TESTS:
        return _REQUIRES_MLX and _needs(path, ("import mlx", "from mlx"))
    if not IS_TARGET_DEVICE:
        return True  # everything else is the research tree
    # On a target device the research tree can still be missing a piece.
    if _REQUIRES_MLX and _needs(path, ("import mlx", "from mlx")):
        return True
    return False


def pytest_ignore_collect(collection_path, config):  # noqa: ARG001 - pytest hook
    path = Path(str(collection_path))
    if path.is_dir():
        return None
    if path.suffix != ".py" or not path.name.startswith("test_"):
        return None
    return True if collect_ignore_glob_hook(path.resolve()) else None


# -- process-wide variables ------------------------------------------------------
#
# `ironmule.hw.apply_cuda_graph_defaults` writes these before MLX's first kernel, and MLX
# reads them once per process. A test that loads through `load_engine` on a CUDA card, or
# exercises the function itself, used to leave `MLX_MAX_OPS_PER_BUFFER=400` behind for every
# later test on the same worker. Restore them after each test, whatever it did.
_PROCESS_WIDE_VARIABLES = ("MLX_MAX_OPS_PER_BUFFER", "MLX_MAX_MB_PER_BUFFER", "MLX_USE_CUDA_GRAPHS")


@pytest.fixture(autouse=True)
def _restore_process_wide_variables():
    before = {name: os.environ.get(name) for name in _PROCESS_WIDE_VARIABLES}
    yield
    for name, value in before.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


# -- the same-UID process table ---------------------------------------------------
#
# The Q3 cleanup checks read every process this user owns and refuse when one appeared
# that they cannot attribute. Under xdist, another worker's test that starts a process
# is exactly such a process: `test_real_macos_process_identity_and_cleanup_reap` failed
# whenever the q3f file ran beside it and passed alone or sequentially (R14, 2026-09-26).
# A test marked `process_table` therefore holds this lock exclusively and every other
# test holds it shared, so nothing this run starts appears while one of them looks.
@pytest.fixture(autouse=True)
def _process_table_lock(request, tmp_path_factory):
    if os.name != "posix":
        yield
        return
    import fcntl

    # The parent of the base temp dir is shared by every xdist worker of one run.
    with open(tmp_path_factory.getbasetemp().parent / "process-table.lock", "a") as handle:
        exclusive = request.node.get_closest_marker("process_table") is not None
        fcntl.flock(handle, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield
