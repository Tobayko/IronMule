#!/usr/bin/env python3
"""Run the fixed Phase-1B CLI with the project interpreter."""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
if os.path.abspath(sys.executable) != str(PROJECT_PYTHON):
    os.execv(
        str(PROJECT_PYTHON),
        [str(PROJECT_PYTHON), "-P", "-s", "-B", __file__, *sys.argv[1:]],
    )

sys.path.insert(0, str(PROJECT_ROOT))

from friday_phase1b.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
