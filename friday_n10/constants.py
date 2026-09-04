"""Frozen N10-v1 design constants; importing this module authorizes no run."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE_PATH = PROJECT_ROOT / ".friday-data" / "n10-v1.sqlite3"

SCHEMA_VERSION = 1
SQLITE_APPLICATION_ID = 0x4E313056  # ASCII "N10V"
PHASE = "H2"
STUDY_ID = "h2n10-dispatch-confirmation-20260822-01"
STUDY_NAME = "single-device-fp16-n10-dispatch-confirmation"

CALIBRATION = "calibration"
CONFIRMATION = "confirmation"
STAGES = frozenset({CALIBRATION, CONFIRMATION})

SESSION_ORDER = ("C0", "V0", "C1", "V1", "C2", "V2")
SESSION_SPECS = {
    "C0": ("characterization", 0, 5_060_361_785_459_989_569),
    "V0": ("validation", 0, 883_950_215_809_699_703),
    "C1": ("characterization", 1, 2_323_802_873_345_837_297),
    "V1": ("validation", 1, 483_519_612_603_395_666),
    "C2": ("characterization", 2, 5_893_687_926_320_354_209),
    "V2": ("validation", 2, 5_188_879_407_004_767_969),
}

FIXTURE_SEED = 8_754_882_193_294_599_646
OPERAND_SEED = 7_421_913_553_926_890_024
SHAPE = 2048
N_MATMULS = 10
DTYPE = "float16"
WARMUP_PAIRS = 2
MEASUREMENT_BLOCKS = 24
INTER_SESSION_COOLDOWN_SECONDS = 20

BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEEDS = {
    "calibration_all": 777_255_143_216_008_523,
    "confirmation_characterization": 7_159_943_182_929_271_886,
    "confirmation_validation": 8_989_465_731_481_879_185,
    "confirmation_all": 4_114_342_224_181_825_282,
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
