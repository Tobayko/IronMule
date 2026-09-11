#!/usr/bin/env python3
"""Build and query the single measurement index; never writes to a source."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from friday_evidence.ssot import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
