"""Load offline IronMule modules without touching the public package namespace.

The production package imports its MLX runtime from ``ironmule.__init__``.  Q4
contract tests must be able to import their stdlib-only modules on machines
without MLX, but must not install a fake ``ironmule`` package: doing so leaks
collection-time state into unrelated tests.  Each caller therefore gets a
private synthetic package namespace and all relative imports resolve within
that namespace.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_offline_modules(*names: str, namespace: str) -> dict[str, Any]:
    """Load ``ironmule/<name>.py`` modules under one private package name.

    ``namespace`` is intentionally caller-supplied so each test file can own
    one coherent module graph.  The namespace must not begin with ``ironmule``
    and is never used to replace an existing module.
    """
    if not names:
        raise ValueError("at least one offline module is required")
    if namespace.startswith("ironmule"):
        raise ValueError("offline namespace must not shadow the public package")
    if namespace in sys.modules:
        raise RuntimeError(f"offline namespace already loaded: {namespace}")

    package = types.ModuleType(namespace)
    package.__path__ = [str(ROOT / "ironmule")]
    package.__package__ = namespace
    package.__spec__ = importlib.util.spec_from_loader(namespace, loader=None, is_package=True)
    sys.modules[namespace] = package

    loaded: dict[str, Any] = {}
    try:
        for name in names:
            if not name.isidentifier() or name.startswith("_"):
                raise ValueError(f"invalid offline module name: {name!r}")
            full_name = f"{namespace}.{name}"
            if full_name in sys.modules:
                raise RuntimeError(f"offline module already loaded: {full_name}")
            path = ROOT / "ironmule" / f"{name}.py"
            spec = importlib.util.spec_from_file_location(full_name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load offline module {name}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[full_name] = module
            spec.loader.exec_module(module)
            loaded[name] = module
    except Exception:
        # Keep cleanup narrow: only synthetic keys owned by this invocation are
        # removed, and no public ``ironmule*`` key can be affected.
        for key in tuple(sys.modules):
            if key == namespace or key.startswith(f"{namespace}."):
                sys.modules.pop(key, None)
        raise
    return loaded
