"""Frozen N10-v2 design constants; importing this module authorizes no run."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE_PATH = PROJECT_ROOT / ".friday-data" / "n10-v2.sqlite3"
PREDECESSOR_DATABASE_PATH = PROJECT_ROOT / ".friday-data" / "n10-v1.sqlite3"

SCHEMA_VERSION = 1
SQLITE_APPLICATION_ID = 0x4E313057  # ASCII "N10W"
PHASE = "H2"
STUDY_ID = "h2n10-dispatch-confirmation-20260822-02"
STUDY_NAME = "single-device-fp16-n10-dispatch-confirmation-registered-fixture"

PREDECESSOR_STUDY_ID = "h2n10-dispatch-confirmation-20260822-01"
PREDECESSOR_FAILURE_RECORD_ID = (
    "3ce4477adf3ca13d30207f37d98f21e36c316c82e3d102abfacf61c091492e49"
)
PREDECESSOR_DATABASE_SHA256 = (
    "e0b5f4af62c128938e1e12e388c16b344a66e18eebf9e0568c7ebe34c5a4f0d5"
)
PREDECESSOR_SNAPSHOT_REVISION = (
    "bbc75d60b5cfc61a1037c0a104e117a89561ec63e13240ca9b84f1bc98c08976"
)

CALIBRATION = "calibration"
CONFIRMATION = "confirmation"
STAGES = frozenset({CALIBRATION, CONFIRMATION})

SESSION_ORDER = ("C0", "V0", "C1", "V1", "C2", "V2")
SESSION_SPECS = {
    "C0": ("characterization", 0, 5_694_182_798_642_334_346),
    "V0": ("validation", 0, 4_016_037_479_549_399_342),
    "C1": ("characterization", 1, 4_702_616_514_600_041_353),
    "V1": ("validation", 1, 5_448_993_668_583_962_080),
    "C2": ("characterization", 2, 6_937_834_284_092_508_076),
    "V2": ("validation", 2, 3_319_947_694_069_614_818),
}

FIXTURE_SEED = 4_051_312_678  # Registered H0 performance identity 0xF17A2026.
FIXTURE_A_SHA256 = "33043be0345487a8a41b522df292e5288914b9c6c6c4dc823dbec72b9146bf86"
FIXTURE_B_SHA256 = "dd40817873b24c2e6117e4e6eeebddccf89775bd4ee4453e7d5456a911670ac2"
FIXTURE_METADATA_SHA256 = (
    "1e26b28978e01ad0faaf296b48043e63803488cdb59e3aa84e79b9ab48a3bb20"
)
FIXTURE_SHA256 = "4776038d9500bad4374410fe2e4a167a6f834e80f0e4d19336592f4ff455dfa4"
OPERAND_SEED = 8_108_914_365_621_233_760
SHAPE = 2048
N_MATMULS = 10
DTYPE = "float16"
WARMUP_PAIRS = 2
MEASUREMENT_BLOCKS = 24
INTER_SESSION_COOLDOWN_SECONDS = 20

BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEEDS = {
    "calibration_all": 3_420_748_623_931_472_299,
    "confirmation_characterization": 968_347_539_867_383_741,
    "confirmation_validation": 2_471_101_842_785_840_228,
    "confirmation_all": 1_603_501_775_215_485_335,
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
