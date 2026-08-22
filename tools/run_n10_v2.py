#!/usr/bin/env python3
"""Repository entrypoint for the sealed formal N10-v2 study."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from friday_n10_v2.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
