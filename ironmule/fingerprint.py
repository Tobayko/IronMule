"""What a stored decision is valid for, and when it stops being valid.

A tuned or measured decision is never a universal truth. It was obtained on one
machine, one framework build, one model, one quantisation, one execution plan, one
service mode and one workload shape. All of those are recorded, and a change in any
of the identity fields invalidates the decision rather than silently carrying it
forward.
"""

from __future__ import annotations

import hashlib
import json
import platform
from typing import Any

from .model_identity import ModelIdentity, ModelIdentityError
from .plans import RUNTIME_VERSION

# Fields whose change makes a stored decision inapplicable rather than merely stale.
FINGERPRINT_SCHEMA = "ironmule.runtime_fingerprint.v2"
IDENTITY_FIELDS = (
    "fingerprint_schema", "hardware_fingerprint", "chip", "memory_bytes", "gpu_cores",
    "mlx", "mlx_lm", "runtime_version", "model_id", "model_revision",
    "model_manifest_sha256", "model_architecture", "quantisation",
    "quantisation_sha256", "tokenizer_sha256", "model_identity_sha256",
    "execution_plan", "service_mode",
)
# Workload fields compared in buckets: a 10% longer prompt is the same regime.
BUCKETED_FIELDS = {"prompt_tokens": 0.25, "max_tokens": 0.25, "concurrency": 0.5}


def build(model_id: str, quantisation: Any, plan_kind: str, service_mode: str,
          workload: dict[str, Any] | None = None, *,
          model_identity: ModelIdentity | dict[str, Any] | None = None) -> dict[str, Any]:
    from .bench import environment
    from .hw import fingerprint as hw_fingerprint, static_facts

    if model_identity is None:
        raise ModelIdentityError("runtime fingerprint requires exact model identity")
    identity = (model_identity if isinstance(model_identity, ModelIdentity)
                else ModelIdentity.from_dict(model_identity))
    if model_id != identity.model_id:
        raise ModelIdentityError("runtime model_id does not match exact model identity")
    if quantisation != identity.quantisation:
        raise ModelIdentityError("runtime quantisation does not match exact model identity")

    facts = static_facts()
    env = environment()
    record = {
        "fingerprint_schema": FINGERPRINT_SCHEMA,
        "hardware_fingerprint": hw_fingerprint(facts),
        "chip": facts.get("chip"), "memory_bytes": facts.get("memory_bytes"),
        "gpu_cores": facts.get("gpu_cores"), "machine": facts.get("machine"),
        "os": f"{platform.system()} {env.get('os') or platform.release()}",
        "mlx": env.get("mlx"), "mlx_lm": env.get("mlx_lm"),
        "runtime_version": RUNTIME_VERSION,
        "model_id": identity.model_id,
        "model_revision": identity.revision,
        "model_manifest_sha256": identity.model_manifest_sha256,
        "model_architecture": identity.architecture,
        "quantisation": identity.quantisation,
        "quantisation_sha256": identity.quantisation_sha256,
        "tokenizer_sha256": identity.tokenizer_sha256,
        "model_identity_sha256": identity.identity_sha256,
        "execution_plan": plan_kind, "service_mode": service_mode,
        "power_source": env.get("power_source"),
        "workload": dict(workload or {}),
    }
    record["digest"] = digest(record)
    return record


def digest(record: dict[str, Any]) -> str:
    payload = {k: record.get(k) for k in IDENTITY_FIELDS}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def incompatible(stored: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """Identity fields that changed. Non-empty means: do not reuse without revalidating."""
    return [f"{field}: {stored.get(field)!r} -> {current.get(field)!r}"
            for field in IDENTITY_FIELDS if stored.get(field) != current.get(field)]


def drifted(stored: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """Workload fields outside their bucket. Softer than `incompatible`."""
    out = []
    old, new = stored.get("workload") or {}, current.get("workload") or {}
    for field, tolerance in BUCKETED_FIELDS.items():
        if field in old and field in new and old[field]:
            if abs(new[field] - old[field]) / max(old[field], 1) > tolerance:
                out.append(f"{field}: {old[field]} -> {new[field]}")
    return out


def usable(stored: dict[str, Any], current: dict[str, Any]) -> tuple[bool, dict[str, list[str]]]:
    bad = incompatible(stored, current)
    soft = drifted(stored, current)
    return (not bad), {"incompatible": bad, "drifted": soft}


def _self_check() -> None:
    base = {"fingerprint_schema": FINGERPRINT_SCHEMA,
            "hardware_fingerprint": "abc", "chip": "Apple M1 Max", "memory_bytes": 1,
            "gpu_cores": 32, "mlx": "0.32.0", "mlx_lm": "0.31.3",
            "runtime_version": RUNTIME_VERSION, "model_id": "m",
            "model_revision": "r", "model_manifest_sha256": "a" * 64,
            "model_architecture": "arch", "quantisation": {"bits": 4},
            "quantisation_sha256": "b" * 64, "tokenizer_sha256": "c" * 64,
            "model_identity_sha256": "d" * 64,
            "execution_plan": "strict_one_shot", "service_mode": "interactive",
            "workload": {"prompt_tokens": 1000, "max_tokens": 32, "concurrency": 4}}
    assert digest(base) == digest(dict(base, power_source="battery")), \
        "power source is recorded but is not an identity field"

    ok, why = usable(base, base)
    assert ok and not why["incompatible"] and not why["drifted"]

    ok, why = usable(base, dict(base, mlx="0.33.0"))
    assert not ok and any("mlx" in x for x in why["incompatible"])

    ok, why = usable(base, dict(base, execution_plan="reusable_session"))
    assert not ok, "a different plan is a different decision"

    ok, why = usable(base, dict(base, workload={"prompt_tokens": 1100, "max_tokens": 32,
                                                "concurrency": 4}))
    assert ok and not why["drifted"], "10% longer prompt is the same regime"

    ok, why = usable(base, dict(base, workload={"prompt_tokens": 2000, "max_tokens": 32,
                                                "concurrency": 4}))
    assert ok and why["drifted"], "double the prompt is drift, not incompatibility"
    print("fingerprint self-check ok")


if __name__ == "__main__":
    _self_check()
