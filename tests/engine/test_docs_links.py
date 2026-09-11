"""Every relative Markdown link must resolve to a file that ships in the repository.

Raw benchmark JSON stays local by `.gitignore` policy, so a link to one is dead for
everybody who clones. Name those files as inline code instead of linking them.
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
    return sealed


SEALED = _sealed_documents()


def _is_sealed(md: Path) -> bool:
    digest = SEALED.get(md)
    return digest is not None and hashlib.sha256(md.read_bytes()).hexdigest() == digest


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
            if not (md.parent / path).resolve().exists():
                broken.append(f"{md.relative_to(REPO_ROOT)}:{lineno} -> {target}")
    assert not broken, "dead relative links:\n" + "\n".join(broken)
