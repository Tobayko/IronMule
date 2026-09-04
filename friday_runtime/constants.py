"""Closed identities and resource bounds for the H1-derived runtime prototype."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_H1_DATABASE_PATH = PROJECT_ROOT / ".friday-data" / "h1-v2.sqlite3"
DEFAULT_RUNTIME_DATABASE_PATH = PROJECT_ROOT / ".friday-data" / "runtime.sqlite3"

SCHEMA_VERSION = 1
SQLITE_APPLICATION_ID = 0x46525254  # ASCII "FRRT"
RUNTIME_ID = "h1-runtime-dispatch-n8-20260821-01"

# These values bind the prototype to the single terminal H1-v2 result that
# authorized it. A different successful-looking database is not interchangeable.
H1_STUDY_ID = "h1v2-dispatch-n8-20260821-01"
H1_DECISION_RECORD_ID = "f508fc9e2b1f44a1b60084bdbeca581024f1f3599535b3dd662a9305c99a9357"
H1_DECISION_SHA256 = "5b022a1dcc127cba05dc86c427dafcc0b8a629e479cc1d29d742514555a5baa5"
H1_PREREGISTRATION_SHA256 = "50baafba71656e1786f120098e1d4f47933c9ab532c8891c39aa6d248561b550"
H1_PROVENANCE_SHA256 = "e08732640516712818fd1872411acdcbfdf7fb91849a588ee1101a8007e7d7e3"

OPERATION = "matmul"
DTYPE = "float16"
SHAPE = (2048, 2048)
OUTPUT_SHAPE = (2048, 2048)
RHS_COUNT = 8
SERIAL_PLAN = "serial_per_op_eval_and_sync"
BATCHED_PLAN = "enqueue_all_then_single_eval_and_sync"

POLICY_WARMUP_BLOCKS = 5
POLICY_MEASUREMENT_BLOCKS = 21
POLICY_ITERATIONS_PER_ARM = 20_000
POLICY_MAX_MEDIAN_NS = 25_000
POLICY_MAX_P95_NS = 50_000
POLICY_MAX_INCREMENTAL_NS = 20_000

GPU_WARMUP_PAIRS = 2
GPU_MEASUREMENT_BLOCKS = 12
GPU_MAX_RATIO = 0.95

MAX_CANONICAL_BYTES = 4 * 1024 * 1024
MAX_HISTORY_ROWS = 256
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_TARGET_BYTES = 2048
DEFAULT_DASHBOARD_PORT = 8769

HISTORY_KINDS = frozenset(
    {"policy_overhead", "runtime_validation", "runtime_failure"}
)
