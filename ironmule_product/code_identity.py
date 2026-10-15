"""The source files a process actually runs (RSI3-LIVE (f)).

A learner's experience and qualifications are bound to the code that produced them. A fixed list
of file names misses whatever else a process imports (the runtime, plans, executor), so each
process reports the digests of every module it has loaded from this checkout: the server when its
learner starts, the model worker once its engine is ready.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("ironmule", "ironmule_product", "friday_evidence")
TOP_LEVEL = ("ironmule_cli.py", "ironmule_inventory.py", "ironmule_controller.py")


def loaded_sources(root: Path = ROOT) -> dict[str, str]:
    """{path relative to the checkout: sha256} for every loaded module from its packages."""
    out: dict[str, str] = {}
    for module in list(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if not name:
            continue
        path = Path(name).resolve()
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if (relative.parts[0] in PACKAGES or str(relative) in TOP_LEVEL) and path.suffix == ".py" and path.is_file():
            out[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(out.items()))


def sources_digest(root: Path = ROOT) -> str:
    """One digest over `loaded_sources`; it changes when any loaded file changes or a new one loads."""
    lines = "".join(f"{path}\t{digest}\n" for path, digest in loaded_sources(root).items())
    return hashlib.sha256(lines.encode()).hexdigest()
