#!/usr/bin/env python3
"""Run the complete H0.1 suite with reproducible import and socket guards.

The runner is intentionally stdlib-only.  HTTP/SSL support is imported before
the socket audit becomes active so the guard observes socket construction, not
class creation performed while importing ``ssl``.
"""

from __future__ import annotations

import http.server  # noqa: F401 - deliberately preloaded before the audit hook
import importlib.abc
import importlib.util
import json
import os
import platform
import resource
import socket  # noqa: F401 - deliberately retain the real socket class
import ssl  # noqa: F401 - SSLSocket must subclass the real socket class
import sys
import time
import unittest
from pathlib import Path
from typing import Any


BLOCKED_IMPORT_ROOTS = frozenset({"mlx", "numpy"})
H01_TEST_MODULES = (
    "tests.test_h01_schedule",
    "tests.test_h01_protocol",
    "tests.test_h01_analysis",
    "tests.test_h01_study",
    "tests.test_h01_storage",
    "tests.test_h01_dashboard",
    "tests.test_h01_import_h0",
)
# Modules deliberately outside the stdlib-only guarantee, each with the reason.
# The live execution path must reach real hardware, and its provenance must
# record the installed NumPy/MLX versions -- so it asks the import machinery
# whether they exist.  Under this guard that question is itself blocked, which
# would turn the run red for a contract the runner was never meant to hold.
# It is covered by the ordinary pytest suite instead.
H01_EXCLUDED_TEST_MODULES = {
    "tests.test_h01_runner": "live execution path; queries numpy/mlx for provenance",
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _unclassified_h01_test_modules() -> list[str]:
    """Name every tests/test_h01_*.py that is neither guarded nor excluded.

    Without this, a new H0.1 test module is silently absent from the guard and a
    green run reads as broader coverage than it has.
    """

    known = set(H01_TEST_MODULES) | set(H01_EXCLUDED_TEST_MODULES)
    found = {
        f"tests.{path.stem}"
        for path in sorted((PROJECT_ROOT / "tests").glob("test_h01_*.py"))
    }
    return sorted(found - known)


class _BlockedImportFinder(importlib.abc.MetaPathFinder):
    """Fail on import machinery lookup for either forbidden package root."""

    def __init__(self) -> None:
        self.attempts: list[str] = []

    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: object = None,
    ) -> None:
        del path, target
        if fullname.partition(".")[0] in BLOCKED_IMPORT_ROOTS:
            self.attempts.append(fullname)
            raise ImportError(f"H0.1 guard blocked import/discovery of {fullname}")
        return None


class _MeasuredResult(unittest.TextTestResult):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.subtests_run = 0

    def addSubTest(
        self,
        test: unittest.case.TestCase,
        subtest: unittest.case.TestCase,
        err: tuple[type[BaseException], BaseException, object] | None,
    ) -> None:
        self.subtests_run += 1
        super().addSubTest(test, subtest, err)


def _rss_bytes(value: int) -> int:
    # Darwin reports bytes; Linux and the BSD-derived Python documentation for
    # several other targets report KiB.  Record raw and normalized values.
    return value if platform.system() == "Darwin" else value * 1024


def main() -> int:
    root_text = str(PROJECT_ROOT)
    sys.path[:] = [entry for entry in sys.path if entry != root_text]
    sys.path.insert(0, root_text)
    os.chdir(PROJECT_ROOT)
    preloaded = sorted(
        name
        for name in sys.modules
        if name.partition(".")[0] in BLOCKED_IMPORT_ROOTS
    )
    finder = _BlockedImportFinder()
    sys.meta_path.insert(0, finder)

    configuration_errors: list[str] = []
    for module_name in _unclassified_h01_test_modules():
        configuration_errors.append(
            f"{module_name}: H0.1 test module is neither guarded nor explicitly excluded"
        )
    for module_name in H01_TEST_MODULES:
        expected = PROJECT_ROOT.joinpath(*module_name.split(".")).with_suffix(".py")
        try:
            spec = importlib.util.find_spec(module_name)
            origin = None if spec is None else spec.origin
            if origin is None or Path(origin).resolve() != expected.resolve():
                configuration_errors.append(
                    f"{module_name}: expected {expected}, resolved {origin!r}"
                )
        except (ImportError, AttributeError, OSError, ValueError) as exc:
            configuration_errors.append(f"{module_name}: {type(exc).__name__}: {exc}")
    if configuration_errors:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "runner": "tools/run_h01_guard.py",
                    "status": "configuration_error",
                    "project_root": root_text,
                    "test_modules": list(H01_TEST_MODULES),
                    "excluded_test_modules": dict(H01_EXCLUDED_TEST_MODULES),
                    "configuration_errors": configuration_errors,
                    "blocked_import_attempts": finder.attempts,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2

    suite = unittest.TestSuite(
        unittest.defaultTestLoader.loadTestsFromName(name)
        for name in H01_TEST_MODULES
    )

    socket_attempts: list[str] = []
    audit_armed = True

    def audit(event: str, args: tuple[Any, ...]) -> None:
        if audit_armed and event == "socket.__new__":
            socket_attempts.append(repr(args[:3]))
            raise RuntimeError("H0.1 guard blocked real socket construction")

    sys.addaudithook(audit)
    started = time.perf_counter()
    runner = unittest.TextTestRunner(verbosity=1, resultclass=_MeasuredResult)
    result = runner.run(suite)
    wall_seconds = time.perf_counter() - started
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    loaded_after = sorted(
        name
        for name in sys.modules
        if name.partition(".")[0] in BLOCKED_IMPORT_ROOTS
    )
    successful = (
        result.wasSuccessful()
        and not preloaded
        and not finder.attempts
        and not socket_attempts
        and not loaded_after
    )
    summary = {
        "schema_version": 1,
        "runner": "tools/run_h01_guard.py",
        "status": "pass" if successful else "fail",
        "test_modules": list(H01_TEST_MODULES),
        # A green status covers exactly these modules and no others.
        "excluded_test_modules": dict(H01_EXCLUDED_TEST_MODULES),
        "tests": result.testsRun,
        "subtests": result.subtests_run,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skips": len(result.skipped),
        "expected_failures": len(result.expectedFailures),
        "unexpected_successes": len(result.unexpectedSuccesses),
        "wall_seconds": wall_seconds,
        "self_user_seconds": own.ru_utime,
        "self_system_seconds": own.ru_stime,
        "child_user_seconds": children.ru_utime,
        "child_system_seconds": children.ru_stime,
        "self_max_rss_raw": own.ru_maxrss,
        "self_max_rss_bytes": _rss_bytes(own.ru_maxrss),
        "child_max_rss_raw": children.ru_maxrss,
        "child_max_rss_bytes": _rss_bytes(children.ru_maxrss),
        "blocked_import_roots": sorted(BLOCKED_IMPORT_ROOTS),
        "blocked_import_attempts": finder.attempts,
        "blocked_modules_preloaded": preloaded,
        "blocked_modules_loaded_after": loaded_after,
        "socket_construction_attempts": socket_attempts,
        "python_executable": sys.executable,
        "platform": platform.system(),
        "project_root": root_text,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
