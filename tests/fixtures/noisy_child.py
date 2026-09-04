"""Emit bounded-test overflow on both streams, then remain alive briefly."""

from __future__ import annotations

import os
import time


def main() -> int:
    payload = b"x" * (256 * 1024)
    os.write(1, payload)
    os.write(2, payload)
    time.sleep(10.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
