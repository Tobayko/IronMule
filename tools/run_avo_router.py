#!/usr/bin/env python3
"""Run the closed AVO-lite shadow-router CLI from the project root."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from friday_avo_router.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
