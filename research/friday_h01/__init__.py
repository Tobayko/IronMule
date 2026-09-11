"""Separate stdlib-only core for the preregistered H0.1 Design A study."""

from .analysis import analyze_trace
from .protocol import build_manifest, build_trace, validate_manifest, validate_result, validate_trace
from .schedule import materialize_schedule, validate_schedule
from .study import analyze_study, validate_study_result

__all__ = [
    "analyze_trace",
    "analyze_study",
    "build_manifest",
    "build_trace",
    "materialize_schedule",
    "validate_manifest",
    "validate_result",
    "validate_schedule",
    "validate_study_result",
    "validate_trace",
]
