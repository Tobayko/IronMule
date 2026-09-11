"""Frozen Design A constants; this module authorizes no execution."""

from __future__ import annotations

PHASE = "H0.1"
STUDY = "paced_trajectory_design_a"
SCHEMA_VERSION = 2
SCHEDULE_ALGORITHM = "sha256_fisher_yates_v1"

INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1

SESSION_ORDER = ("C0", "V0", "C1", "V1", "C2", "V2")
SESSION_SPECS = {
    "C0": ("characterization", 0, 0x484F_3100_0000_0001),
    "V0": ("validation", 0, 0x484F_3100_0000_0002),
    "C1": ("characterization", 1, 0x484F_3100_0000_0003),
    "V1": ("validation", 1, 0x484F_3100_0000_0004),
    "C2": ("characterization", 2, 0x484F_3100_0000_0005),
    "V2": ("validation", 2, 0x484F_3100_0000_0006),
}

SAMPLES_PER_BLOCK = 4
BURN_IN_BLOCKS = 8
BURN_IN_SAMPLES = 32
MAIN_BLOCKS = 20
MAIN_SAMPLES = 80
TOTAL_SAMPLES = BURN_IN_SAMPLES + MAIN_SAMPLES

SHORT_LABEL = "short_50ms"
LONG_LABEL = "long_750ms"
SHORT_GAP_NS = 50_000_000
LONG_GAP_NS = 750_000_000
COOLDOWN_NS = 20_000_000_000
MAX_GAP_OVERSHOOT_NS = 250_000_000

CHANGEPOINT_MIN_SPLIT = 8
CHANGEPOINT_MAX_SPLIT = 72
ACF_LAGS = (1, 2, 3, 4)

TREND_EFFECT_ABS_MAX = 0.05
CHANGEPOINT_EFFECT_ABS_MAX = 0.05
ACF_ABS_MAX = 0.50
ESS_MIN = 40.0
PACE_EFFECT_ABS_MAX = 0.03
TAIL_RATIO_MAX = 1.20

GATE_LIMITS = {
    "trend": ("<=", TREND_EFFECT_ABS_MAX),
    "changepoint": ("<=", CHANGEPOINT_EFFECT_ABS_MAX),
    "acf": ("<=", ACF_ABS_MAX),
    "ess": (">=", ESS_MIN),
    "pacing": ("<=", PACE_EFFECT_ABS_MAX),
    "tail": ("<=", TAIL_RATIO_MAX),
}

SESSION_COMPLETE_STATUS = "h01_session_complete"
SESSION_INVALID_STATUS = "h01_invalid"
RESULT_STATUSES = frozenset({SESSION_COMPLETE_STATUS, SESSION_INVALID_STATUS})
STUDY_STATUSES = frozenset(
    {"h01_stationarity_supported", "h01_complete_unresolved", "h01_invalid"}
)
TELEMETRY_MISSING_REASONS = frozenset({"not_collected", "api_unavailable", "not_applicable"})
