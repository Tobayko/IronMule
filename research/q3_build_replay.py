#!/usr/bin/env python3
"""Build the Q3 replay dataset from already-recorded JSON evidence.

The adapter is deliberately standalone: importing it never imports
``ironmule.__init__``, runtime, tune, MLX, or any model package.  It only
constructs immutable offline contracts and writes output when ``--execute`` is
explicitly supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import stat
import sys
import types
from pathlib import Path
from typing import Any, Iterable


B36_PREREGISTRATION_SHA256 = "7bf3997b19dc55d3b75be977c0da8d42d6ab554232ce2bf40617429c478897a4"
B36A_PREREGISTRATION_SHA256 = "ee5b3e9b250d75eb69ed6e38f9661f656da743098bef318966dc055099c9e492"
B36_CODE_DIGEST = "5566ee87f1656d9dcaceb05edf6a155ee2a35dd784c81a46fbb6dab30e499ddc"


def _load_contracts() -> tuple[Any, Any]:
    """Load evidence/adaptive under a synthetic package, bypassing __init__."""
    root = Path(__file__).resolve().parents[1]
    package = types.ModuleType("ironmule")
    package.__path__ = [str(root / "ironmule")]
    package.__package__ = "ironmule"
    sys.modules.setdefault("ironmule", package)
    for name in ("evidence", "adaptive"):
        full_name = f"ironmule.{name}"
        if full_name in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(full_name, root / "ironmule" / f"{name}.py")
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load offline contract {name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)
    return sys.modules["ironmule.evidence"], sys.modules["ironmule.adaptive"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON from {path}: {exc}") from exc


def _require_hash(path: Path, expected: str, label: str) -> None:
    if _sha256(path) != expected:
        raise ValueError(f"{label} SHA-256 mismatch")


def _digest_json(value: Any, canonical_sha256: Any) -> str:
    return canonical_sha256(value)


def _median(values: Iterable[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot take median of empty samples")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _stddev(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _artifact(evidence: Any, name: str, digest: str, quality: Any) -> Any:
    return evidence.ArtifactRef(name, digest, quality)


def _context(adaptive: Any, *, study: Any, model: Any, hardware: Any, framework: Any, workload: Any, time: Any) -> Any:
    return adaptive.AdaptiveContext(
        study_digest=study,
        model_digest=model,
        hardware_digest=hardware,
        framework_digest=framework,
        workload_digest=workload,
        time_digest=time,
    )


def _q2_actions(profile: dict[str, Any], adaptive: Any) -> list[Any]:
    trials = profile.get("trials")
    if not isinstance(trials, list) or len(trials) != sum(len(values) for _, values in adaptive.SEARCH_VALUES):
        raise ValueError("Q2 profile does not contain the complete registered trial sequence")
    expected = [(name, value) for name, values in adaptive.SEARCH_VALUES for value in values]
    current = adaptive.KnobAction.baseline()
    actions = [current]
    for trial, (expected_name, expected_value) in zip(trials, expected):
        if not isinstance(trial, dict) or trial.get("knob") != expected_name or trial.get("value") != expected_value:
            raise ValueError("Q2 trial sequence does not match SEARCH_VALUES")
        candidate_values = current.as_dict()
        candidate_values[expected_name] = expected_value
        candidate = adaptive.KnobAction(**candidate_values)
        actions.append(candidate)
        if trial.get("disposition") == "accepted":
            current = candidate
        elif trial.get("disposition") != "rejected":
            raise ValueError("Q2 trial disposition is not closed")
    if current.as_dict() != profile.get("knobs"):
        raise ValueError("Q2 final profile knobs do not match accepted trial path")
    if len({action.action_id for action in actions}) != 12:
        raise ValueError("Q2 sequential candidate actions are not unique")
    return actions


def _q2_observations(profile: dict[str, Any], profile_path: Path, log_path: Path, profile_sha: str, log_sha: str, adaptive: Any, evidence: Any) -> tuple[Any, ...]:
    actions = _q2_actions(profile, adaptive)
    conditions = profile.get("conditions")
    identity = profile.get("model_identity")
    if not isinstance(conditions, dict) or not isinstance(identity, dict):
        raise ValueError("Q2 profile is missing model conditions/identity")
    canonical_sha256 = evidence.canonical_sha256
    context = _context(
        adaptive,
        study=canonical_sha256("Q2"),
        model=identity["model_manifest_sha256"],
        hardware=canonical_sha256(profile.get("hardware")),
        framework=canonical_sha256({key: conditions.get(key) for key in ("mlx", "mlx_lm", "runtime_version", "os")} ),
        workload=canonical_sha256({"model_id": profile.get("model_id"), "prompt_tokens": conditions.get("prompt_tokens"), "max_tokens": conditions.get("max_tokens")}),
        time=canonical_sha256({"tuned_at": profile.get("tuned_at"), "source": log_sha}),
    )
    refs = (
        _artifact(evidence, profile_path.name, profile_sha, evidence.EvidenceQuality.SUMMARY_ONLY),
        _artifact(evidence, log_path.name, log_sha, evidence.EvidenceQuality.SUMMARY_ONLY),
    )
    values = [(profile["baseline_ns"], profile["baseline_prefill_ns"], profile["baseline_decode_ns"])]
    values.extend((trial["total_ns"], trial["prefill_ns"], trial["decode_ns"]) for trial in profile["trials"])
    rows = []
    for action, (total, prefill, decode) in zip(actions, values):
        outcome = adaptive.AdaptiveOutcome(
            raw_sample_refs=refs,
            raw_sample_count=0,
            total_ns=total,
            prefill_ns=prefill,
            decode_ns=decode,
            token_identity=None,
            stop_reason_identity=None,
            token_count_identity=None,
            state_identity=None,
            deterministic=None,
            mlx_active_memory_bytes=None,
            mlx_peak_memory_bytes=None,
            rss_peak_bytes=None,
            swap_delta_bytes=None,
            timeout=False,
            crash=False,
            fallbacks=0,
            hard_gates_passed=False,
            status=adaptive.OutcomeStatus.INCONCLUSIVE,
        )
        rows.append(adaptive.AdaptiveObservation(
            context=context,
            action=action,
            measurements={"total_ns": total, "prefill_ns": prefill, "decode_ns": decode},
            uncertainty={},
            outcome=outcome,
            rollback=adaptive.RollbackStatus.NOT_REQUIRED,
            evidence=refs,
            split=adaptive.ReplaySplit.VALIDATION,
        ))
    return tuple(rows)


def _validate_b36(raw: dict[str, Any], expected_b36_prereg_sha: str, expected_b36a_sha: str, expected_code_digest: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if raw.get("schema") != "ironmule.b36.v1" or raw.get("status") != "complete":
        raise ValueError("B36 schema/status is not exact v1 complete")
    constants = raw.get("constants")
    pairs = raw.get("pairs")
    if constants != {"max_tokens": 32, "no_retry": True, "pairs": 16, "repeats": 5, "warmups": 2}:
        raise ValueError("B36 constants do not match the frozen protocol")
    if not isinstance(pairs, list) or len(pairs) != 16:
        raise ValueError("B36 must contain exactly 16 pairs")
    orders = [tuple(pair.get("order", ())) for pair in pairs]
    if orders.count(("baseline", "candidate")) != 8 or orders.count(("candidate", "baseline")) != 8:
        raise ValueError("B36 must contain eight AB and eight BA pairs")
    if raw.get("preregistration", {}).get("sha256") != expected_b36_prereg_sha or raw.get("b36a_preregistration", {}).get("sha256") != expected_b36a_sha or raw.get("code_identity", {}).get("digest") != expected_code_digest:
        raise ValueError("B36 top-level source hash mismatch")
    children = [child for pair in pairs for child in pair.get("children", [])]
    if len(children) != 32 or sum(child.get("arm") == "baseline" for child in children) != 16 or sum(child.get("arm") == "candidate" for child in children) != 16:
        raise ValueError("B36 must contain exactly 32 children split 16/16")
    for pair in pairs:
        pair_result = pair.get("pair_result", {})
        if pair_result.get("status") != "ok" or not pair_result.get("token_identity"):
            raise ValueError("B36 pair status/token gate failed")
        if set(pair_result.get("hard_gates", {})) != {"complete", "identity", "no_crash", "peak_memory", "swap", "timings", "token_identity"} or not all(pair_result.get("hard_gates", {}).values()):
            raise ValueError("B36 pair hard gate failed")
        for child in pair.get("children", []):
            if child.get("schema") != "ironmule.b36.child.v1" or child.get("returncode") != 0 or child.get("crashed") or not child.get("no_crash") or not child.get("identity_gate") or not child.get("canonical_correctness_gate") or not child.get("post_evidence_complete"):
                raise ValueError("B36 child gate failed")
            if len(child.get("warmups", [])) != 2 or len(child.get("measured", [])) != 5:
                raise ValueError("B36 child repeat counts are incomplete")
            if child.get("preregistration_sha256") != expected_b36_prereg_sha or child.get("b36a_preregistration_sha256") != expected_b36a_sha or child.get("code_digest") != expected_code_digest:
                raise ValueError("B36 child source hash mismatch")
    identities = {child.get("model_manifest_digest") for child in children}
    workloads = {json.dumps(child.get("workload"), sort_keys=True, separators=(",", ":")) for child in children}
    environment_keys = ("chip", "hardware_fingerprint", "memory_bytes", "gpu_cores", "mlx", "mlx_lm", "python", "os", "git_commit", "power_source", "low_power_mode")
    environments = {json.dumps({key: child.get("environment", {}).get(key) for key in environment_keys}, sort_keys=True, separators=(",", ":")) for child in children}
    top_model_digest = raw.get("model_binding", {}).get("digest")
    if len(identities) != 1 or top_model_digest not in identities or len(workloads) != 1 or len(environments) != 1 or raw.get("summary", {}).get("valid_pairs") != 16:
        raise ValueError("B36 model/workload/environment identity is not stable")
    return children, pairs[0]["children"][0]


def _resource_max(children: list[dict[str, Any]], key: str, nested: str | None = None) -> int:
    values = []
    for child in children:
        for checkpoint in child.get("checkpoints", []):
            value = checkpoint.get("rss_bytes") if key == "rss_bytes" else checkpoint.get("mlx", {}).get(key)
            if value is not None:
                values.append(int(value))
    if not values:
        raise ValueError(f"B36 has no complete resource field {key}")
    return max(values)


def _b36_observations(raw: dict[str, Any], b36_path: Path, b36_sha: str, adaptive: Any, evidence: Any, expected_b36_prereg_sha: str, expected_b36a_sha: str, expected_code_digest: str) -> tuple[Any, ...]:
    children, first = _validate_b36(raw, expected_b36_prereg_sha, expected_b36a_sha, expected_code_digest)
    canonical_sha256 = evidence.canonical_sha256
    environment = first["environment"]
    workload = first["workload"]
    model_digest = first["model_manifest_digest"]
    context = _context(
        adaptive,
        study=canonical_sha256("B36"),
        model=model_digest,
        hardware=canonical_sha256({key: environment.get(key) for key in ("chip", "hardware_fingerprint", "memory_bytes", "gpu_cores")} ),
        framework=canonical_sha256({key: environment.get(key) for key in ("mlx", "mlx_lm", "python", "os", "git_commit")} ),
        workload=canonical_sha256(workload),
        time=canonical_sha256({"first_checkpoint": min(cp["timestamp_ns"] for child in children for cp in child["checkpoints"]), "last_checkpoint": max(cp["timestamp_ns"] for child in children for cp in child["checkpoints"])}),
    )
    ref = _artifact(evidence, b36_path.name, b36_sha, evidence.EvidenceQuality.RAW_SAMPLES)
    rows = []
    for arm in ("baseline", "candidate"):
        arm_children = [child for child in children if child["arm"] == arm]
        samples = {name: [float(measured[name]) for child in arm_children for measured in child["measured"]] for name in ("total_ns", "prefill_ns", "decode_ns")}
        if any(len(values) != 80 for values in samples.values()):
            raise ValueError("B36 must aggregate exactly 80 raw repeats per arm")
        measurements = {name: _median(values) for name, values in samples.items()}
        uncertainty = {}
        for name, values in samples.items():
            uncertainty.update({f"{name}_min": min(values), f"{name}_max": max(values), f"{name}_stddev": _stddev(values)})
        outcome = adaptive.AdaptiveOutcome(
            raw_sample_refs=(ref,), raw_sample_count=80,
            total_ns=measurements["total_ns"], prefill_ns=measurements["prefill_ns"], decode_ns=measurements["decode_ns"],
            token_identity=True, stop_reason_identity=True, token_count_identity=True, state_identity=True, deterministic=True,
            mlx_active_memory_bytes=_resource_max(arm_children, "active"), mlx_peak_memory_bytes=_resource_max(arm_children, "peak"),
            rss_peak_bytes=_resource_max(arm_children, "rss_bytes"), swap_delta_bytes=0,
            timeout=False, crash=False, fallbacks=0, hard_gates_passed=True, status=adaptive.OutcomeStatus.MEASURED,
        )
        action = adaptive.KnobAction() if arm == "baseline" else adaptive.KnobAction(compiled_fixed_cache=True, head_skip_prefill=True)
        rows.append(adaptive.AdaptiveObservation(
            context=context, action=action, measurements=measurements, uncertainty=uncertainty,
            outcome=outcome, rollback=adaptive.RollbackStatus.NOT_REQUIRED, evidence=(ref,), split=adaptive.ReplaySplit.SEALED_HOLDOUT,
        ))
    return tuple(rows)


def build_dataset(q2_profile: Path, q2_log: Path, b36_path: Path, expected_q2_profile_sha: str, expected_q2_log_sha: str, expected_b36_sha: str, expected_b36_prereg_sha: str = B36_PREREGISTRATION_SHA256, expected_b36a_sha: str = B36A_PREREGISTRATION_SHA256, expected_code_digest: str = B36_CODE_DIGEST) -> Any:
    evidence, adaptive = _load_contracts()
    _require_hash(q2_profile, expected_q2_profile_sha, "Q2 profile")
    _require_hash(q2_log, expected_q2_log_sha, "Q2 log")
    _require_hash(b36_path, expected_b36_sha, "B36 raw")
    profile_data = _read_json(q2_profile)
    if not isinstance(profile_data, dict) or len(profile_data) != 1:
        raise ValueError("Q2 profile must contain exactly one profile")
    profile = next(iter(profile_data.values()))
    if not isinstance(profile, dict):
        raise ValueError("Q2 profile payload must be an object")
    q2 = _q2_observations(profile, q2_profile, q2_log, expected_q2_profile_sha, expected_q2_log_sha, adaptive, evidence)
    b36 = _b36_observations(_read_json(b36_path), b36_path, expected_b36_sha, adaptive, evidence, expected_b36_prereg_sha, expected_b36a_sha, expected_code_digest)
    actions = [item.action for item in q2] + [item.action for item in b36 if item.action.action_id not in {row.action.action_id for row in q2}]
    if len(actions) != 12:
        raise ValueError("declared action pool must contain exactly twelve unique actions")
    return adaptive.ReplayDataset(observations=q2 + b36, action_pool=tuple(actions))


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q2-profile", type=Path, required=True)
    parser.add_argument("--q2-log", type=Path, required=True)
    parser.add_argument("--b36", dest="b36_path", type=Path, required=True)
    parser.add_argument("--q2-profile-sha256", required=True)
    parser.add_argument("--q2-log-sha256", required=True)
    parser.add_argument("--b36-sha256", required=True)
    parser.add_argument("--expected-b36-prereg-sha256", default=B36_PREREGISTRATION_SHA256)
    parser.add_argument("--expected-b36a-sha256", default=B36A_PREREGISTRATION_SHA256)
    parser.add_argument("--expected-code-sha256", default=B36_CODE_DIGEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="write output with O_EXCL and mode 0600")
    args = parser.parse_args(argv)
    try:
        dataset = build_dataset(args.q2_profile, args.q2_log, args.b36_path, args.q2_profile_sha256, args.q2_log_sha256, args.b36_sha256, args.expected_b36_prereg_sha256, args.expected_b36a_sha256, args.expected_code_sha256)
        evidence, _ = _load_contracts()
        payload = (evidence.canonical_json(dataset.to_dict()) + "\n").encode("utf-8")
        if args.execute:
            _write_exclusive(args.output, payload)
        else:
            sys.stdout.buffer.write(payload)
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"q3_build_replay: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
