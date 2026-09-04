"""Every relative Markdown link must resolve to a file that ships in the repository.

Raw benchmark JSON stays local by `.gitignore` policy, so a link to one is dead for
everybody who clones. Name those files as inline code instead of linking them.
"""

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
EXTERNAL = ("http://", "https://", "mailto:", "#")


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
