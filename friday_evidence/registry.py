"""Closed evidence-tool registry and the approved H1/H2 safety policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
SQLITE_APPLICATION_ID = 0x46524945  # ASCII "FRIE"
DEFAULT_DATABASE_PATH = PROJECT_ROOT / ".friday-data" / "research.sqlite3"
MAX_REPORT_BYTES = 2 * 1024 * 1024
MAX_PROVENANCE_BYTES = 512 * 1024
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_RECENT_ROWS = 200
LEGACY_SOURCE_PATH = PROJECT_ROOT / "experiments" / "legacy_h1h2_summaries_v1.json"

REGISTERED_TOOLS = {
    "dispatch": "matmul-fp16-2048x2048-dispatch-plan",
    "cooldown": "matmul-fp16-2048x2048-cooldown",
    "loop": "matmul-fp16-2048x2048-optimization-loop",
    "model-loop": "gemma3-4b-proposed-matmul-dispatch-plans",
    "codegen": "gemma3-4b-generated-matmul-dispatch-plans",
    "roofline": "gemma3-1b-and-4b-inference-roofline",
    "fusion": "gemma3-1b-and-4b-cache-free-forward-fusion",
}

RAW_REPORT_FIELDS = {
    "dispatch": "replicates",
    "cooldown": "raw_timings",
    "loop": "rounds",
    "model-loop": "rounds",
    "codegen": "attempts",
    "roofline": "models",
    "fusion": "models",
}


@dataclass(frozen=True)
class BudgetPolicy:
    """Approved fail-closed limits shared by every H1/H2 measuring tool."""

    gpu_work_limit_s: float = 120.0
    continuous_gpu_limit_s: float = 6.0
    required_break_s: float = 4.0
    duty_window_s: float = 60.0
    duty_cycle_limit: float = 0.25
    wall_limit_s: float = 20.0 * 60.0
    candidate_cooldown_s: float = 60.0


DEFAULT_BUDGET_POLICY = BudgetPolicy()
