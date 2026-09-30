"""The single place test path setup happens.

The repository has no ``tests/__init__.py``, so ``tests`` is a namespace package
whose ``__path__`` is rebuilt from ``sys.path`` on every import. The repo root
therefore goes first and stays first, so ``tests`` always resolves to
``<repo>/tests`` and ``from ironmule.runtime import ...`` resolves to the engine
that ships with this checkout.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

if sys.path[:1] != [str(ROOT)]:
    if str(ROOT) in sys.path:
        sys.path.remove(str(ROOT))
    sys.path.insert(0, str(ROOT))


# -- collection without an optional dependency ---------------------------------
#
# The whole suite ships with the package (ironmule, ironmule_product and
# friday_evidence), so every test file here depends on nothing outside this
# repository and runs wherever the package's own dependencies are installed.
# The one thing that can be missing is mlx itself, on a bare checkout before
# `pip install -e .`; a test file that needs it is left uncollected instead of
# failing at import time.

def _missing(name: str) -> bool:
    from importlib.util import find_spec

    try:
        return find_spec(name) is None
    except (ImportError, ValueError):
        return True


#: The engine package's own suite. It depends on nothing outside the repository,
#: so it is always collected -- an environment missing mlx also lacks a working
#: `ironmule` install, and that surfaces here rather than as a silent skip.
ENGINE_TESTS = Path(__file__).resolve().parent / "engine"

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
    return _REQUIRES_MLX and _needs(path, ("import mlx", "from mlx"))


def pytest_ignore_collect(collection_path, config):  # noqa: ARG001 - pytest hook
    path = Path(str(collection_path))
    if path.is_dir():
        return None
    if path.suffix != ".py" or not path.name.startswith("test_"):
        return None
    return True if collect_ignore_glob_hook(path.resolve()) else None


# -- measured data that stays on the machine that measured it ----------------------
#
# Raw measurements are gitignored and were purged from the published history on
# 2026-09-28. A claim check that reads one of those files skips where it is absent,
# and only for a path `.gitignore` keeps private: a typo in a path still fails. On the
# measuring machine every file is present and every check runs.
def _private(path: str) -> bool:
    import subprocess

    try:
        return subprocess.run(["git", "check-ignore", "-q", "--no-index", path],
                              cwd=ROOT, capture_output=True).returncode == 0
    except OSError:
        return False


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):  # noqa: ARG001 - pytest hook
    try:
        return (yield)
    except FileNotFoundError as error:
        if error.filename and _private(os.fsdecode(error.filename)):
            pytest.skip(f"private measured data: {os.path.relpath(error.filename, ROOT)}")
        raise


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
