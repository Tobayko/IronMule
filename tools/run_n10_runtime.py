#!/usr/bin/env python3
"""Repository entrypoint for the bounded N10 AVO-lite runtime prototype."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from friday_runtime_n10.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
