"""Every SQL literal in the packages must compile against its own schema.

SQL lives in this repository the way JavaScript does: as a string that no
tool parses until a code path happens to execute it. A dashboard script with
one surplus parenthesis silenced a UI for months (2026-09-02); the same shape
of typo in a rarely-taken query would surface during a gated hardware run,
which is the most expensive moment available.

sqlite3 is in the standard library, so unlike the JavaScript guard this one
never skips.
"""

from __future__ import annotations

import ast
import re
import sqlite3
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_START = re.compile(r"^\s*(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|PRAGMA|WITH|REPLACE)\b", re.I)
_CLAUSE = re.compile(r"\b(FROM|INTO|TABLE|VALUES|SET|INDEX|TRIGGER|WHERE|VIEW)\b", re.I)
#: A literal that is concatenated or interpolated into a larger statement.
_FRAGMENT = re.compile(
    r"[(=,]\s*$"
    r"|^\s*PRAGMA\s+\w+\s*=\s*$"
    r"|\b(FROM|WHERE|INTO|SET|AND|OR|JOIN|EXISTS|BY|VALUES|LIKE|IN)\s*$",
    re.I,
)
_CREATE = re.compile(r"^\s*CREATE\b", re.I)

#: Queries that deliberately read another package's database, keyed by the
#: token SQLite names when the local schema lacks it. Each is a documented
#: cross-package read, not a schema mismatch.
_CROSS_PACKAGE = {
    ("friday_h01/import_h0.py", "runs"),                     # imports the H0 study database
    ("friday_h01/runner.py", "runs"),                        # same importer, reading H0 runs
    ("friday_head_skip_runtime/policy.py", "record_sha256"),  # reads the sealed head-skip study
}


def _docstring_ids(tree: ast.AST) -> set[int]:
    found = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and body:
            first = body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
               and isinstance(first.value.value, str):
                found.add(id(first.value))
    return found


def statements_of(path: Path) -> list[tuple[int, str]]:
    """Complete SQL statements written as literals in *path*."""

    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return []
    documentation = _docstring_ids(tree)
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        if id(node) in documentation:
            continue
        text = node.value.strip()
        if not _START.match(text) or len(text) <= 12:
            continue
        if not _CLAUSE.search(text) and not re.match(r"^\s*PRAGMA\s+\w+\s*$", text, re.I):
            continue
        if _FRAGMENT.search(text):
            continue
        found.append((node.lineno, text))
    return found


def packages() -> list[str]:
    return sorted({
        path.parent.name for path in ROOT.glob("friday_*/*.py")
        if "__pycache__" not in path.parts and statements_of(path)
    })


def test_the_sweep_still_finds_the_statements_it_is_meant_to_guard():
    total = sum(
        len(statements_of(path))
        for path in ROOT.glob("friday_*/*.py") if "__pycache__" not in path.parts
    )
    assert total >= 150, f"only {total} statements found; the extractor has drifted"


@pytest.mark.parametrize("package", packages())
def test_every_sql_literal_compiles_against_its_package_schema(package):
    connection = sqlite3.connect(":memory:")
    try:
        for migration in sorted((ROOT / package / "migrations").glob("*.sql")):
            try:
                connection.executescript(migration.read_text())
            except sqlite3.Error:
                pass
        sources = sorted(
            path for path in (ROOT / package).glob("*.py") if "__pycache__" not in path.parts
        )
        collected = [(path, line, text) for path in sources for line, text in statements_of(path)]
        for _, _, text in collected:
            if _CREATE.match(text):
                try:
                    connection.executescript(text)
                except sqlite3.Error:
                    pass
        failures = []
        for path, line, text in collected:
            try:
                connection.execute("EXPLAIN " + text, [None] * text.count("?"))
            except sqlite3.Error as exc:
                message = str(exc)
                # A binding count complaint proves the statement already parsed.
                if "bindings supplied" in message:
                    continue
                if _CREATE.match(text) and "already exists" in message:
                    continue
                relative = path.relative_to(ROOT).as_posix()
                if any(relative == name and token in message for name, token in _CROSS_PACKAGE):
                    continue
                failures.append(f"{relative}:{line}: {message}\n    {text[:120]}")
        assert not failures, "\n".join(failures)
    finally:
        connection.close()
