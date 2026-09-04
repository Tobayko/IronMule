"""Frozen identifiers and gates for the AVO-lite shadow router."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROUTER_ID = "avo-shadow-router-20260822-01"
SCHEMA_VERSION = 1
SQLITE_APPLICATION_ID = 0x46525231  # ASCII FRR1
DEFAULT_DATABASE_PATH = PROJECT_ROOT / ".friday-data" / "avo-router.sqlite3"
DEFAULT_DASHBOARD_PORT = 8773

POLICY_RUN_ID = "avo-router-policy-20260822-01"
SHADOW_RUN_ID = "avo-router-shadow-20260822-01"

N8_RHS_COUNT = 8
N10_RHS_COUNT = 10
ENFORCED_PLAN = "serial_shadow_only"
SERIAL_PLAN = "serial_per_op_eval_and_sync"

COLD_LOAD_MAX_NS = 15_000_000_000
POLICY_MEDIAN_MAX_NS = 30_000
POLICY_P95_MAX_NS = 60_000
POLICY_INCREMENTAL_MEDIAN_MAX_NS = 15_000

POLICY_WARMUP_BLOCKS = 5
POLICY_BLOCKS = 21
POLICY_ITERATIONS = 10_000

HISTORY_KINDS = frozenset(
    {"policy_overhead", "shadow_validation", "router_failure"}
)
MAX_HISTORY_ROWS = 256
MAX_CANONICAL_BYTES = 4 * 1024 * 1024
MAX_DATABASE_BYTES = 32 * 1024 * 1024
