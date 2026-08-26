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
from .tune import DEFAULT_MODEL, knobs_for, load_profile, revalidate, stale, tune

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
    return (f"{where}: {profile['gain']*100:.2f}% faster than untuned "
            f"({profile['baseline_ns']/1e6:.1f} -> {profile['tuned_ns']/1e6:.1f} ms), "
            f"tokens identical")
