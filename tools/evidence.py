#!/usr/bin/env python3
"""Read and verify the local H1/H2 evidence history; never touches MLX or the GPU."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from friday_evidence.cli import main  # noqa: E402,F401


if __name__ == "__main__":
    raise SystemExit(main())
