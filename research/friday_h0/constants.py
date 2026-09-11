"""Frozen H0 constants; no runtime, MLX, or worker policy lives here."""

PHASE_H0 = "H0"
SCHEMA_VERSION = 1
STORAGE_SCHEMA_VERSION = 1
SQLITE_APPLICATION_ID = 0x46524830  # ASCII ``FRH0``; fixed DB identity signal.
BOOTSTRAP_REPLICATES = 10_000

ALLOWED_MODES = frozenset(
    {
        "eager_baseline",
        "compile_comparison",
        "aa_gpu",
        "analysis_slow",
        "analysis_known_win",
        "analysis_wrong_fixture",
        "analysis_missing_data",
        "control_timeout",
        "control_exit_70",
    }
)

MLX_MODES = frozenset({"eager_baseline", "compile_comparison", "aa_gpu"})
ANALYSIS_MODES = frozenset(
    {
        "analysis_slow",
        "analysis_known_win",
        "analysis_wrong_fixture",
        "analysis_missing_data",
    }
)
CONTROL_MODES = frozenset({"control_timeout", "control_exit_70"})

AA_SESSION_SEEDS = {
    "characterization_fixture": 0xAA1A2026,
    "characterization_order": 0xAA0D2026,
    "confirmation_fixture": 0xAA1A2126,
    "confirmation_order": 0xAA0D2126,
}
AA_BOOTSTRAP_SEEDS = {
    "characterization": 0xAA052026,
    "confirmation": 0xAA052126,
}
EAGER_COMPILE_SESSION_SEEDS = {
    "characterization_fixture": 0xF17A2026,
    "characterization_order": 0xB10C2026,
    "confirmation_fixture": 0xF17A2126,
    "confirmation_order": 0xB10C2126,
}

ENGINEERING_EQUIVALENCE_BAND = (0.98, 1.02)
SESSION_RATIO_BAND = (0.95, 1.05)
H0_BOOTSTRAP_SEEDS = AA_BOOTSTRAP_SEEDS
ANALYSIS_SLOW_FACTOR = 1.10
ANALYSIS_KNOWN_WIN_FACTOR = 0.90
ANALYSIS_BASELINE_NS = 1_000_000
ANALYSIS_STEP_NS = 1_000
ANALYSIS_CLUSTERS = 3
ANALYSIS_PAIRS_PER_CLUSTER = 30
WRONG_FIXTURE_SEED = 0xBAD02026
WRONG_FIXTURE_SIZE = 64
REQUIRED_MEMORY_FIELD = "rss_peak_bytes"
