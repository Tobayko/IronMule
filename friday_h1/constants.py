"""Frozen H1-v2 design constants; importing this module authorizes no run."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE_PATH = PROJECT_ROOT / ".friday-data" / "h1-v2.sqlite3"

SCHEMA_VERSION = 2
SQLITE_APPLICATION_ID = 0x48315632  # ASCII "H1V2"
PHASE = "H1"
STUDY_ID = "h1v2-dispatch-n8-20260821-01"
STUDY_NAME = "single-device-fp16-dispatch-confirmation"

CALIBRATION = "calibration"
CONFIRMATION = "confirmation"
STAGES = frozenset({CALIBRATION, CONFIRMATION})

SESSION_ORDER = ("C0", "V0", "C1", "V1", "C2", "V2")
SESSION_SPECS = {
    "C0": ("characterization", 0, 0x4831_5632_0000_0001),
    "V0": ("validation", 0, 0x4831_5632_0000_0002),
    "C1": ("characterization", 1, 0x4831_5632_0000_0003),
    "V1": ("validation", 1, 0x4831_5632_0000_0004),
    "C2": ("characterization", 2, 0x4831_5632_0000_0005),
    "V2": ("validation", 2, 0x4831_5632_0000_0006),
}

FIXTURE_SEED = 4_051_312_678
OPERAND_SEED = 2_026_082_101
SHAPE = 2048
N_MATMULS = 8
DTYPE = "float16"
WARMUP_PAIRS = 2
MEASUREMENT_BLOCKS = 24
INTER_SESSION_COOLDOWN_SECONDS = 20

BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEEDS = {
    "calibration_all": 0x4831_0000_0000_0001,
    "confirmation_characterization": 0x4831_0000_0000_0002,
    "confirmation_validation": 0x4831_0000_0000_0003,
    "confirmation_all": 0x4831_0000_0000_0004,
}
CONFIDENCE_LEVEL = 0.95
MINIMUM_EFFECT_FLOOR = 0.05
MAXIMUM_CALIBRATED_MDE = 0.15
CANDIDATE_COUNT = 1

GPU_WORK_LIMIT_SECONDS = 120.0
CONTINUOUS_GPU_LIMIT_SECONDS = 6.0
REQUIRED_BREAK_SECONDS = 4.0
DUTY_WINDOW_SECONDS = 60.0
DUTY_CYCLE_LIMIT = 0.25
WALL_LIMIT_SECONDS = 20.0 * 60.0
CANDIDATE_COOLDOWN_SECONDS = 60.0

MAX_CANONICAL_BYTES = 8 * 1024 * 1024
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_HISTORY_ROWS = 64
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1

RECORD_KINDS = frozenset(
    {
        "preregistration",
        "calibration_session",
        "calibration_summary",
        "confirmation_seal",
        "confirmation_session",
        "study_decision",
        "session_failure",
    }
)
