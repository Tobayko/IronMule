"""A profile may name a service strategy. Absent or incomplete, it names none.

The tuned profile already binds a `Knobs` set to a hardware fingerprint and a model
identity. A service mode is not a knob: it changes how requests are grouped, not how a
kernel computes. So this adds a separate, versioned record beside the knobs rather than
pretending the mode is one.

Three rules hold it shut:

* A profile without the record selects nothing. An older profile keeps its behaviour.
* The record must name the strategy, the range it was admitted for, the correctness
  contract it rests on, and the run ids that prove it. A record missing any of these is
  refused rather than trusted.
* Selection only happens after an explicit opt-in by the caller. A valid record on its
  own changes nothing.
"""

from __future__ import annotations

from typing import Any

SERVICE_STRATEGY_SCHEMA = "ironmule.tuned_profile.service_strategy.v1"
STRATEGY_PAIRED = "paired_throughput"
STRATEGY_THROUGHPUT = "throughput"
KNOWN_STRATEGIES = (STRATEGY_THROUGHPUT, STRATEGY_PAIRED)
REQUIRED_FIELDS = frozenset(
    {"schema", "strategy", "admitted_range", "correctness_contract", "evidence_run_ids"}
)
REQUIRED_RANGE_FIELDS = frozenset(
    {"min_ready_requests", "max_ready_requests", "model_identity_sha256",
     "hardware_fingerprint", "mlx", "mlx_lm"}
)


class StrategyRefused(ValueError):
    """The profile does not authorise this strategy here."""


def build_record(*, strategy: str, min_ready: int, max_ready: int,
                 identity_sha256: str, fingerprint: str, mlx: str, mlx_lm: str,
                 correctness_contract: str, evidence_run_ids: list[str]) -> dict[str, Any]:
    """Assemble a record. Every field is required; nothing is defaulted into existence."""

    if strategy not in KNOWN_STRATEGIES:
        raise StrategyRefused(f"unknown service strategy: {strategy!r}")
    if not evidence_run_ids:
        raise StrategyRefused("a strategy record needs the run ids that evidence it")
    if not 1 <= min_ready <= max_ready:
        raise StrategyRefused("the ready-request range is not ordered")
    return {
        "schema": SERVICE_STRATEGY_SCHEMA,
        "strategy": strategy,
        "admitted_range": {
            "min_ready_requests": min_ready,
            "max_ready_requests": max_ready,
            "model_identity_sha256": identity_sha256,
            "hardware_fingerprint": fingerprint,
            "mlx": mlx,
            "mlx_lm": mlx_lm,
        },
        "correctness_contract": correctness_contract,
        "evidence_run_ids": list(evidence_run_ids),
    }


def read_record(profile: dict[str, Any] | None) -> dict[str, Any] | None:
    """The record if it is complete and current, otherwise None. Never raises."""

    if not isinstance(profile, dict):
        return None
    record = profile.get("service_strategy")
    if not isinstance(record, dict):
        return None
    if record.get("schema") != SERVICE_STRATEGY_SCHEMA:
        return None
    if set(record) != REQUIRED_FIELDS:
        return None
    if record.get("strategy") not in KNOWN_STRATEGIES:
        return None
    admitted = record.get("admitted_range")
    if not isinstance(admitted, dict) or set(admitted) != REQUIRED_RANGE_FIELDS:
        return None
    if not record.get("evidence_run_ids") or not record.get("correctness_contract"):
        return None
    return record


def select(profile: dict[str, Any] | None, *, opt_in: bool, ready_requests: int,
           identity_sha256: str | None, fingerprint: str | None,
           mlx: str, mlx_lm: str) -> dict[str, Any]:
    """Which strategy to use, and why. The default answer is the established mode.

    Only facts available at decision time are used: how many requests are ready, and the
    identity of the machine, model and libraries. Nothing about the response a request
    will eventually produce enters here.
    """

    decision = {"strategy": STRATEGY_THROUGHPUT, "reason": "", "evidence_run_ids": [],
                "record_present": False}
    if not opt_in:
        decision["reason"] = "no opt-in: automatic selection is off"
        return decision
    record = read_record(profile)
    if record is None:
        decision["reason"] = "no complete service-strategy record in the profile"
        return decision
    decision["record_present"] = True
    if record["strategy"] != STRATEGY_PAIRED:
        decision["reason"] = f"profile names {record['strategy']!r}"
        return decision
    admitted = record["admitted_range"]
    checks = {
        "model_identity": admitted["model_identity_sha256"] == identity_sha256,
        "hardware": admitted["hardware_fingerprint"] == fingerprint,
        "mlx": admitted["mlx"] == mlx,
        "mlx_lm": admitted["mlx_lm"] == mlx_lm,
        "ready_requests": admitted["min_ready_requests"] <= ready_requests
        <= admitted["max_ready_requests"],
    }
    if not all(checks.values()):
        failed = sorted(name for name, ok in checks.items() if not ok)
        decision["reason"] = f"outside the admitted range: {failed}"
        return decision
    decision["strategy"] = STRATEGY_PAIRED
    decision["reason"] = (
        f"{ready_requests} ready requests inside the admitted range "
        f"{admitted['min_ready_requests']} to {admitted['max_ready_requests']}, "
        "on the qualified machine, model and libraries"
    )
    decision["evidence_run_ids"] = list(record["evidence_run_ids"])
    decision["correctness_contract"] = record["correctness_contract"]
    return decision


def attach(profile: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    """A copy of the profile carrying the record, or a refusal.

    The caller still writes through `tune.save_profile`, which is the only authorised
    write path and keeps its own identity checks. Nothing here touches a stored file, so
    a profile is never migrated as a side effect of reading it.
    """

    if not isinstance(profile, dict):
        raise StrategyRefused("a strategy record needs a profile to sit in")
    candidate = dict(profile)
    candidate["service_strategy"] = record
    if read_record(candidate) is None:
        raise StrategyRefused("the record would not read back as complete and current")
    return candidate
