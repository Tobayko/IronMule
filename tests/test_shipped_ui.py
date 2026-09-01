"""Every shipped dashboard must actually run, not merely be served.

Between 2026-08-30 and 2026-09-02 the optimizer dashboard shipped JavaScript
with one surplus closing parenthesis. The whole page is a single IIFE, so no
panel ever populated, and the existing tests missed it because they asserted
status codes, headers and byte counts for /assets/app.js — never that the
content parses. This module guards that class for every package at once, so a
new dashboard is covered the day it appears.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")

#: Ids the scripts resolve at runtime.
_ID_PATTERNS = (
    r"getElementById\(['\"]([\w-]+)['\"]\)",
    r"\bq\(['\"]([\w-]+)['\"]\)",
    r"\bput\(['\"]([\w-]+)['\"]",
    r"querySelectorAll\(['\"]#([\w-]+)",
)


def dashboard_sources() -> list[Path]:
    return sorted(
        path for path in ROOT.glob("friday_*/dashboard*.py")
        if path.is_file() and "__pycache__" not in path.parts
    )


def scripts_of(source: str) -> list[str]:
    """Inline <script> bodies plus module-level JavaScript constants."""

    found = [body.strip() for body in re.findall(r"<script[^>]*>(.*?)</script>", source, re.S)]
    for match in re.finditer(r'^(JS|SCRIPT|APP_JS) = r?"""(.*?)"""', source, re.S | re.M):
        found.append(match.group(2).strip())
    return [body for body in found if body]


def parse_javascript(body: str) -> tuple[int, str]:
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as handle:
        handle.write(body)
        path = handle.name
    try:
        completed = subprocess.run([NODE, "--check", path], capture_output=True, text=True)
        return completed.returncode, completed.stderr
    finally:
        Path(path).unlink(missing_ok=True)


def test_the_repository_still_ships_dashboards():
    sources = dashboard_sources()
    assert len(sources) >= 10, [path.name for path in sources]


@pytest.mark.parametrize("source_path", dashboard_sources(), ids=lambda p: p.parent.name)
def test_every_shipped_script_parses(source_path):
    if NODE is None:
        pytest.skip("no JavaScript engine available; a skip is not a pass")
    source = source_path.read_text()
    bodies = scripts_of(source)
    if not bodies:
        pytest.skip(f"{source_path.parent.name} ships no script")
    for index, body in enumerate(bodies):
        code, error = parse_javascript(body)
        assert code == 0, f"{source_path.parent.name} script {index}: {error.strip()[-300:]}"


@pytest.mark.parametrize("source_path", dashboard_sources(), ids=lambda p: p.parent.name)
def test_every_element_the_script_touches_exists_in_the_markup(source_path):
    source = source_path.read_text()
    bodies = scripts_of(source)
    if not bodies:
        pytest.skip(f"{source_path.parent.name} ships no script")
    identifiers: set[str] = set()
    for body in bodies:
        for pattern in _ID_PATTERNS:
            identifiers |= set(re.findall(pattern, body))
    # The markup may be written with escaped quotes inside a Python literal.
    missing = sorted(
        name for name in identifiers
        if f'id="{name}"' not in source and f'id=\\"{name}\\"' not in source
    )
    assert not missing, f"{source_path.parent.name} script fills ids absent from its markup: {missing}"
