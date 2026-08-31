"""IronMule — a local inference runtime with explicit execution plans.

    import ironmule

    rt = ironmule.Runtime.load()                                  # interactive by default

    # one request
    out = rt.generate("Summarise this paragraph: ...", max_tokens=64)

    # a session whose requests share a document, prefill reused and bit exact
    plan = rt.session_plan(document)
    out = rt.generate(document + "\\n\\nQuestion: ...", plan=plan)

    # several concurrent requests, throughput mode
    rt.mode = ironmule.ThroughputMode()
    results = rt.serve([ironmule.Request(prompt_ids=ids, plan=plan) for ids in batch])
    print(rt.telemetry.snapshot())

The caller chooses the execution plan and the service mode. Nothing in the runtime
switches either on its own, because both change observable behaviour: plans differ
in output, modes differ in the latency/throughput trade.

Every claim in the docstrings here is backed by a measurement in `research/LEDGER.md`
with raw data under `research/raw/`.
"""

from __future__ import annotations

from .executor import MAX_GROUP_WIDTH, AsyncGroupedB1Executor, SequentialExecutor
from .fingerprint import build as build_fingerprint, usable
from .plans import RUNTIME_VERSION, ExecutionPlan, ReusableSessionPlan, StrictOneShotPlan
from .runtime import BASELINE, Engine, Knobs, PrefixCache
from .service import InteractiveMode, Request, Result, Runtime, ThroughputMode
from .telemetry import RequestMetrics, Telemetry
from .tune import (
    DEFAULT_MODEL, _stored_confirmation_valid, knobs_for, load_profile, revalidate, stale, tune,
)

__version__ = RUNTIME_VERSION

__all__ = [
    # runtime
    "Runtime", "Request", "Result",
    # plans, chosen by the caller
    "ExecutionPlan", "StrictOneShotPlan", "ReusableSessionPlan",
    # service modes, chosen by the caller
    "InteractiveMode", "ThroughputMode",
    # executors, if a caller wants one directly
    "SequentialExecutor", "AsyncGroupedB1Executor", "MAX_GROUP_WIDTH",
    # observability and validity
    "Telemetry", "RequestMetrics", "build_fingerprint", "usable",
    # tuning, unchanged from the research phase
    "Engine", "Knobs", "BASELINE", "PrefixCache", "DEFAULT_MODEL",
    "tune", "load_profile", "knobs_for", "revalidate", "stale",
    "load", "status", "__version__",
]


def load(model_id: str = DEFAULT_MODEL, autotune: bool = True, force_retune: bool = False):
    """Legacy entry point: an `Engine` wearing this machine's tuned knobs.

    Prefer `Runtime.load()`, which adds plans, modes, telemetry and the fallback.
    Call ``close()`` on the returned engine (or use ``Runtime`` as a context
    manager) before releasing a wired profile or loading another engine.
    """
    from .tune import load_engine

    profile = None if force_retune else load_profile(model_id)
    if profile is None and autotune:
        profile = tune(model_id)
    knobs = Knobs(**profile["knobs"]) if profile else BASELINE
    engine, _tokenizer = load_engine(model_id, knobs)
    return engine


def status(model_id: str = DEFAULT_MODEL) -> str:
    """One line on what this machine has learned so far."""
    from .hw import static_facts

    facts = static_facts()
    profile = load_profile(model_id)
    where = f"{facts['chip']}, {facts['memory_bytes'] // 1024**3} GB, {facts['gpu_cores']} GPU cores"
    if profile is None:
        return f"{where}: not tuned yet"
    raw_confirmation = profile.get("confirmation")
    if raw_confirmation is None:
        confirmation = None
    elif not isinstance(raw_confirmation, dict):
        return f"{where}: BASELINE retained; confirmation invalid"
    else:
        confirmation = raw_confirmation
    if isinstance(confirmation, dict) and confirmation.get("accepted") is False:
        reason = confirmation.get("rejection_reason")
        if reason not in {
            "token_identity", "token_count_identity", "stop_reason_identity",
            "determinism", "ci_not_below_one", "invalid_confirmation",
        }:
            reason = "invalid_confirmation"
        return f"{where}: BASELINE retained; confirmation rejected ({reason})"
    if isinstance(confirmation, dict) and confirmation.get("accepted") is True:
        candidate_mapping = profile.get("confirmation_candidate_knobs")
        try:
            candidate_knobs = Knobs(**candidate_mapping)
        except (TypeError, ValueError):
            candidate_knobs = None
        if (candidate_knobs is None
                or candidate_knobs.as_dict() != candidate_mapping
                or not _stored_confirmation_valid(
                    confirmation, profile.get("confirmation_evidence"),
                    expected_candidate=candidate_knobs
                )):
            return f"{where}: BASELINE retained; confirmation invalid"
        ratio = confirmation.get("ratio")
        total = ratio.get("total_ns") if isinstance(ratio, dict) else None
        if not isinstance(total, dict):
            return f"{where}: BASELINE retained; confirmation invalid"
        try:
            paired_gain = (1.0 - float(total["median_ratio"])) * 100
            ci_low = (1.0 - float(total["ci_high"])) * 100
            ci_high = (1.0 - float(total["ci_low"])) * 100
        except (KeyError, TypeError, ValueError, OverflowError):
            return f"{where}: BASELINE retained; confirmation invalid"
        return (f"{where}: {paired_gain:.2f}% faster in paired confirmation, "
                f"tokens identical; paired 95% CI [{ci_low:.2f}%; {ci_high:.2f}%]")
    if confirmation is None:
        return f"{where}: screening-only; no confirmed speedup"
    return f"{where}: legacy/unconfirmed profile; no confirmed speedup"
