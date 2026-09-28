"""Every relative Markdown link must resolve to a file that ships in the repository.

Measured data stays local by `.gitignore` policy, so a link to it is dead for everybody
who clones. New text names those files as inline code; links that records already make
into that private data are accepted, because the file exists where it was measured.
"""

import hashlib
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
EXTERNAL = ("http://", "https://", "mailto:", "#")
#: `"<path>": "<sha256>"` as sealed evidence records a provenance input.
SEALED_INPUT = re.compile(r'"([A-Za-z0-9_./-]+\.md)":\s*"([0-9a-f]{64})"')


def _sealed_documents() -> dict[Path, str]:
    """Markdown files a sealed evidence record hashes, mapped to that hash.

    A preregistration describes the tree it was written against. Its links are a
    record of that tree, not navigation a reader is meant to follow today, and its
    bytes cannot be corrected without breaking the chain that cites them. The
    exemption is bound to the recorded hash: edit the document and it stops being
    exempt, so this can never be used to park a dead link in a living file.
    """

    out = subprocess.run(["git", "ls-files", "experiments/*.json", "research/*.json"],
                         cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    sealed: dict[Path, str] = {}
    for line in out.stdout.splitlines():
        if not line:
            continue
        try:
            text = (REPO_ROOT / line).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for relative, digest in SEALED_INPUT.findall(text):
            sealed[REPO_ROOT / relative] = digest
    # A seal whose record is now private measured data survives as a `<digest>  <name>`
    # sidecar next to the document, the format research/raw already uses.
    out = subprocess.run(["git", "ls-files", "*.sha256"],
                         cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    for line in out.stdout.splitlines():
        sidecar = REPO_ROOT / line
        for entry in sidecar.read_text(encoding="utf-8").splitlines():
            digest, _, name = entry.partition("  ")
            if name.endswith(".md"):
                sealed[sidecar.parent / name] = digest
    return sealed


SEALED = _sealed_documents()


def _is_sealed(md: Path) -> bool:
    digest = SEALED.get(md)
    return digest is not None and hashlib.sha256(md.read_bytes()).hexdigest() == digest


def _private(path: Path) -> bool:
    """True when `.gitignore` keeps this path on the machine that measured it."""
    return subprocess.run(["git", "check-ignore", "-q", "--no-index", str(path)],
                          cwd=REPO_ROOT).returncode == 0


def _tracked_markdown() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "*.md"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [REPO_ROOT / line for line in out.stdout.splitlines() if line]


@pytest.mark.parametrize("md", _tracked_markdown(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_relative_links_resolve(md: Path) -> None:
    if _is_sealed(md):
        pytest.skip("sealed provenance input; its links record the tree it was written against")
    broken = []
    for lineno, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
        for target in LINK.findall(line):
            if target.startswith(EXTERNAL):
                continue
            path = target.split("#", 1)[0]
            if not path:
                continue
            resolved = (md.parent / path).resolve()
            if not resolved.exists() and not _private(resolved):
                broken.append(f"{md.relative_to(REPO_ROOT)}:{lineno} -> {target}")
    assert not broken, "dead relative links:\n" + "\n".join(broken)
