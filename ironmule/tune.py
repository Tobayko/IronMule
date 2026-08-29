"""Self-tuning: find the fastest knob setting that still emits identical tokens.

The search is coordinate descent from the untuned baseline. Every candidate must
reproduce the baseline's greedy token sequence exactly, so a knob can only ever
buy speed, never a different answer. The winner is stored per hardware
fingerprint and model, so an unseen machine tunes itself once and every later
start reuses the result.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from .fast import FusionUnsupported
from .hw import STORE, fingerprint, probe
from .model_identity import (
    ModelIdentity, ModelIdentityError, ResolvedModelSource, build_model_identity,
    resolve_model_source,
)
from .runtime import BASELINE, Engine, Knobs

DEFAULT_MODEL = "mlx-community/gemma-3-4b-it-4bit"
PROFILES = STORE / "profiles.json"
PROFILE_CONDITIONS_SCHEMA = "ironmule.tuned_profile.conditions.v2"
PROFILE_CONDITION_FIELDS = {
    "conditions_schema", "fingerprint", "chip", "memory_bytes", "gpu_cores",
    "mlx", "mlx_lm", "runtime_version", "model_id", "model_revision",
    "model_manifest_sha256", "model_architecture", "quantisation",
    "quantisation_sha256", "tokenizer_sha256", "model_identity_sha256",
    "power_source", "prompt_tokens", "max_tokens", "execution_plan", "os",
}
MODEL_IDENTITY_CONDITION_FIELDS = {
    "model_id", "model_revision", "model_manifest_sha256", "model_architecture",
    "quantisation", "quantisation_sha256", "tokenizer_sha256",
    "model_identity_sha256",
}

# The predecessor project's planner request, reproduced word for word — including
# its internal project name — so token counts and therefore timings stay comparable
# with the cycles recorded in research/LEDGER.md. Do not reword it.
DEFAULT_PROMPT = """You choose exactly one next Project Friday experiment.

Hardware: Apple M1 Max, 32 GB unified memory. Use only the evidence below.

Measured evidence:
- persistent_service_qualification: keeping Gemma 4B loaded reduced paired time to first output by 65.3032%; all greedy outputs matched exactly. Multi-turn and parallel-request qualification are still missing.
- batched_readback: isolated decode readback accounts for 12.98% per output token, but batching the checks can emit extra tokens and therefore needs a later correctness study.
- host_readback_upper_bound: 15.3% is only an upper bound, not a directly usable implementation.
- kv_cache_preallocation_ab: 4.4263% of decode time is correlated with reallocations, but the first step is confounded and the cache change still requires separate architecture permission.

Fixed selection policy:
1. Prefer the largest already confirmed end-to-end lever that also closes a required missing workload.
2. Do not choose a diagnostic upper bound.
3. Do not choose a permission-blocked cache change.
4. Choose exactly one ID from this list: persistent_service_qualification, batched_readback, host_readback_upper_bound, kv_cache_preallocation_ab.

Return only a JSON object with exactly one key named candidate_id and no prose, markdown, or explanation."""

# Ordered so that the one knob needing a model reload (fusion) comes last.
SEARCH: list[tuple[str, list[Any]]] = [
    ("compiled_fixed_cache", [True]),
    ("fused_argmax", [True]),
    ("head_skip_prefill", [True]),
    ("prefill_into_fixed", [True]),
    ("readback_every", [2, 4, 8]),
    ("speculate_k", [4]),   # loses badly on MLX 0.32; kept so new hardware gets its own verdict
    ("capacity_slack", [128]),
    ("wired_fraction", [0.6]),
    ("fuse_projections", [True]),
]

KEEP_IF_RATIO_BELOW = 0.995   # a knob has to actually pay for itself
CONFIRM_PROCESSES = 6         # the screening search is cheap and single process;
CONFIRM_REPEATS = 7           # the winner still has to survive a paired A/B
REVALIDATE_PROCESSES = 3      # canaries are cheaper than a full confirmation
HYSTERESIS = 0.02             # a stored winner is only dropped when it clearly lost


def resolve_local_model(model_id: str, revision: str | None = None) -> ResolvedModelSource:
    """Resolve exactly one local source without downloading or changing global state."""
    local = Path(model_id).expanduser()
    if local.is_dir():
        return resolve_model_source(model_id, revision=revision)
    from huggingface_hub import scan_cache_dir
    return resolve_model_source(model_id, revision=revision, cache=scan_cache_dir())


def _identity_conditions(identity: ModelIdentity) -> dict[str, Any]:
    return {
        "model_id": identity.model_id,
        "model_revision": identity.revision,
        "model_manifest_sha256": identity.model_manifest_sha256,
        "model_architecture": identity.architecture,
        "quantisation": identity.quantisation,
        "quantisation_sha256": identity.quantisation_sha256,
        "tokenizer_sha256": identity.tokenizer_sha256,
        "model_identity_sha256": identity.identity_sha256,
    }


def _conditions_match_identity(conditions_record: Mapping[str, Any],
                               identity: ModelIdentity) -> bool:
    expected = _identity_conditions(identity)
    return all(conditions_record.get(key) == value for key, value in expected.items())


def verify_resolved_model(model_id: str, resolved: ResolvedModelSource) -> None:
    """Detect any source change between identity construction and actual model load."""
    verified = build_model_identity(model_id, resolved.path, resolved.identity.revision)
    if verified != resolved.identity:
        raise ModelIdentityError("model source changed during load")


def conditions(model_id: str, prompt_tokens: int, max_tokens: int, *,
               model_identity: ModelIdentity | None = None) -> dict[str, Any]:
    """What a stored winner is actually valid for.

    A tuned profile is not a universal truth. It was measured on one machine, one
    framework build, one model, one power state and one workload size. Recording
    those makes it possible to notice later that none of them still hold.
    """
    from .bench import environment
    from .hw import static_facts
    from .plans import RUNTIME_VERSION

    identity = model_identity or resolve_local_model(model_id).identity
    if not Path(model_id).expanduser().is_dir() and model_id != identity.model_id:
        raise ModelIdentityError("conditions model_id does not match exact identity")
    facts = static_facts()
    env = environment()
    return {
        "conditions_schema": PROFILE_CONDITIONS_SCHEMA,
        "fingerprint": fingerprint(facts),
        "chip": facts["chip"],
        "memory_bytes": facts["memory_bytes"],
        "gpu_cores": facts["gpu_cores"],
        "mlx": env.get("mlx"),
        "mlx_lm": env.get("mlx_lm"),
        "runtime_version": RUNTIME_VERSION,
        **_identity_conditions(identity),
        "power_source": env["power_source"],
        "prompt_tokens": prompt_tokens,
        "max_tokens": max_tokens,
        "execution_plan": "single_shot",   # the prefix-cache plan is caller declared
        "os": env.get("os"),
    }


def stale(profile: dict[str, Any], model_id: str, prompt_tokens: int,
          max_tokens: int, *, model_identity: ModelIdentity | None = None) -> list[str]:
    """Which recorded conditions no longer hold. Empty means the profile still applies."""
    stored = profile.get("conditions")
    if not stored:
        return ["no conditions recorded"]
    current = conditions(
        model_id, prompt_tokens, max_tokens, model_identity=model_identity
    )
    # Workload size is compared in buckets: a prompt 5% longer is not a new regime.
    drifted = []
    for key, value in stored.items():
        if key in ("prompt_tokens", "max_tokens"):
            if value and abs(current[key] - value) / max(value, 1) > 0.25:
                drifted.append(f"{key}: {value} -> {current[key]}")
        elif current.get(key) != value:
            drifted.append(f"{key}: {value} -> {current.get(key)}")
    return drifted


def gpu_busy() -> str | None:
    """Another heavy local job would poison every timing. Report it instead of measuring."""
    try:
        out = subprocess.run(["ps", "-Ao", "pid=,rss=,comm=,args="], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    mine = os.getpid()
    for line in out.stdout.splitlines():
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        pid, rss, _comm, args = parts
        if int(pid) == mine or int(rss) < 900_000:  # < ~900 MB is not a loaded model
            continue
        lowered = args.lower()
        if "python" in lowered and ("mlx" in lowered or "worker" in lowered or "measure" in lowered):
            return f"pid {pid} holds {int(rss)//1024} MB: {args[:80]}"
    return None


def load_engine(model_id: str, knobs: Knobs, *, offline: bool | None = True,
                revision: str | None = None,
                resolved_source: ResolvedModelSource | None = None) -> tuple[Engine, Any]:
    """Load a model under a caller-selectable local/offline policy.

    Offline loads resolve exactly one local source and attach its path-free identity
    to the Engine. Online mode and ``None`` preserve legacy caller-managed loading but
    leave identity unavailable, so fingerprints and profiles fail closed.
    No environment variable or imported-module global is changed.
    """
    from mlx_lm import load

    if resolved_source is not None and offline is not True:
        raise ValueError("resolved_source is valid only for offline loading")
    if revision is not None and offline is not True:
        raise ValueError("exact revision requires offline local resolution")
    resolved = resolved_source
    source: str = model_id
    if offline is True:
        resolved = resolved or resolve_local_model(model_id, revision)
        if (not Path(model_id).expanduser().is_dir()
                and resolved.identity.model_id != model_id):
            raise ModelIdentityError("resolved source belongs to a different model_id")
        if revision is not None and resolved.identity.revision != revision:
            raise ModelIdentityError("resolved source belongs to a different revision")
        source = str(resolved.path)
    model, tokenizer = load(source)
    if resolved is not None:
        verify_resolved_model(model_id, resolved)
    engine = Engine(model, tokenizer, knobs)
    engine.model_identity = resolved.identity if resolved is not None else None
    return engine, tokenizer


def _eos_ids(tokenizer) -> tuple[int, ...]:
    ids = set(getattr(tokenizer, "eos_token_ids", None) or ())
    if getattr(tokenizer, "eos_token_id", None) is not None:
        ids.add(tokenizer.eos_token_id)
    return tuple(sorted(ids)) or (1, 106)


def prompt_ids(tokenizer, prompt: str) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    values = tokenizer.encode(rendered, add_special_tokens=False)
    return list(values)


def measure(engine: Engine, ids: list[int], max_tokens: int, eos: tuple[int, ...],
            repeats: int = 5, warmup: int = 2) -> dict[str, Any]:
    for _ in range(warmup):
        engine.generate(ids, max_tokens, eos)
    runs = [engine.generate(ids, max_tokens, eos) for _ in range(repeats)]
    return {
        "total_ns": statistics.median(r["total_ns"] for r in runs),
        "prefill_ns": statistics.median(r["prefill_ns"] for r in runs),
        "decode_ns": statistics.median(r["decode_ns"] for r in runs),
        "logical_tokens": runs[0]["logical_tokens"],
        "deterministic": all(r["logical_tokens"] == runs[0]["logical_tokens"] for r in runs),
        "capacity": runs[0]["capacity"],
    }


def _is_unsupported_candidate(exc: BaseException) -> bool:
    """Recognize only typed fusion or explicitly unsupported cache failures."""
    if isinstance(exc, FusionUnsupported):
        return True
    return isinstance(exc, (ValueError, TypeError)) and "unsupported" in str(exc).lower()


def confirm(model_id: str, baseline: Knobs, candidate: Knobs, prompt: str,
            max_tokens: int) -> dict[str, Any]:
    """Screening found a candidate; a paired A/B decides whether it is real."""
    from . import ab
    return ab.run({"baseline": baseline, "candidate": candidate},
                  processes=CONFIRM_PROCESSES, repeats=CONFIRM_REPEATS, warmup=2,
                  max_tokens=max_tokens, model=model_id, prompt=prompt)


def revalidate(model_id: str = DEFAULT_MODEL, prompt: str = DEFAULT_PROMPT,
               max_tokens: int = 32) -> dict[str, Any]:
    """Canary: does this machine's stored winner still beat the untuned path?

    Hysteresis is deliberate. Measurement noise must not be able to flip the
    runtime back and forth between strategies on every check.
    """
    from . import ab

    resolved = resolve_local_model(model_id)
    profile = load_profile(
        model_id, require_compatible=False, model_identity=resolved.identity
    )
    if profile is None:
        return {"verdict": "no_profile"}
    result = ab.run({"baseline": BASELINE, "stored": Knobs(**profile["knobs"])},
                    processes=REVALIDATE_PROCESSES, repeats=5, warmup=2,
                    max_tokens=max_tokens, model=model_id, prompt=prompt)
    # The canary's child already tokenizes the exact prompt it measures.  Use
    # that observed length rather than the profile's old workload as the
    # comparison input; this catches materially changed prompts reliably.
    prompt_tokens = None
    for child in result.get("raw", []):
        for arm in child.get("arms", {}).values():
            if arm.get("prompt_tokens") is not None:
                prompt_tokens = int(arm["prompt_tokens"])
                break
        if prompt_tokens is not None:
            break
    if prompt_tokens is None:
        # Keep compatibility with a small test/dry-run harness that omits raw
        # child details, while still measuring the current prompt itself.
        engine, tokenizer = load_engine(
            model_id, BASELINE, resolved_source=resolved
        )
        try:
            prompt_tokens = len(prompt_ids(tokenizer, prompt))
        finally:
            del engine
    drifted = stale(
        profile, model_id, prompt_tokens, max_tokens,
        model_identity=resolved.identity,
    )
    ratio = result["ratios"]["stored/baseline"]["total_ns"]
    if not result["token_identity"]:
        verdict = "retune_required"
    elif ratio["median_ratio"] > 1.0 - HYSTERESIS:
        verdict = "retune_required"
    else:
        verdict = "still_valid_with_drift" if drifted else "still_valid"
    return {"verdict": verdict, "drifted_conditions": drifted, "ratio": ratio,
            "token_identity": result["token_identity"],
            "stored_gain": profile.get("gain")}


def tune(model_id: str = DEFAULT_MODEL, prompt: str = DEFAULT_PROMPT, max_tokens: int = 32,
         repeats: int = 5, force: bool = False, confirm_winner: bool = True) -> dict[str, Any]:
    busy = gpu_busy()
    if busy and not force:
        raise RuntimeError(f"another model process is running, refusing to measure ({busy})")

    hardware = probe()
    resolved = resolve_local_model(model_id)
    engine, tokenizer = load_engine(model_id, BASELINE, resolved_source=resolved)
    ids = prompt_ids(tokenizer, prompt)
    eos = _eos_ids(tokenizer)

    base = measure(engine, ids, max_tokens, eos, repeats=repeats)
    if not base["deterministic"]:
        raise RuntimeError("baseline is not deterministic, cannot gate on token identity")
    reference = base["logical_tokens"]
    print(f"baseline {base['total_ns']/1e6:.2f} ms  "
          f"(prefill {base['prefill_ns']/1e6:.2f}, decode {base['decode_ns']/1e6:.2f}) "
          f"{len(reference)} tokens, capacity {base['capacity']}")

    best, best_result, trials = BASELINE, base, []
    for name, values in SEARCH:
        for value in values:
            candidate = replace(best, **{name: value})
            if candidate == best:
                continue
            reload_needed = Engine.needs_reload(best, candidate)
            try:
                if reload_needed:
                    del engine
                    engine, tokenizer = load_engine(
                        model_id, candidate, resolved_source=resolved
                    )
                else:
                    engine.knobs = candidate
                    engine._compiled = None
                result = measure(engine, ids, max_tokens, eos, repeats=repeats)
            except (ValueError, RuntimeError, TypeError) as exc:
                # FusionUnsupported is typed; cache-specific ValueError and
                # TypeError paths must explicitly say unsupported.  Generic
                # runtime/type/value errors remain loud so programming bugs
                # cannot be misreported as tuning results.
                if not _is_unsupported_candidate(exc):
                    raise
                trials.append({"knob": name, "value": value,
                               "disposition": "unsupported",
                               "verdict": "unsupported",
                               "reason": f"{type(exc).__name__}: {exc}"})
                print(f"  {name}={value!r:>6}  unsupported ({exc})")
                if reload_needed:
                    engine, tokenizer = load_engine(
                        model_id, best, resolved_source=resolved
                    )
                else:
                    engine.knobs = best
                    engine._compiled = None
                continue
            ratio = result["total_ns"] / base["total_ns"]
            identical = result["logical_tokens"] == reference
            verdict = ("rejected: tokens differ" if not identical
                       else "rejected: not deterministic" if not result["deterministic"]
                       else "kept" if ratio < best_result["total_ns"] / base["total_ns"] * KEEP_IF_RATIO_BELOW
                       else "rejected: no gain")
            trials.append({"knob": name, "value": value, "ratio": ratio,
                           "disposition": "accepted" if verdict == "kept" else "rejected",
                           "verdict": verdict,
                           "total_ns": result["total_ns"], "decode_ns": result["decode_ns"],
                           "prefill_ns": result["prefill_ns"]})
            print(f"  {name}={value!r:>6}  ratio {ratio:.4f}  {verdict}")
            if verdict == "kept":
                best, best_result = candidate, result
            elif Engine.needs_reload(candidate, best):
                del engine
                engine, tokenizer = load_engine(
                    model_id, best, resolved_source=resolved
                )
            else:
                engine.knobs = best
                engine._compiled = None

    del engine
    confirmation = None
    if confirm_winner and best != BASELINE:
        print("confirming the screening winner with a paired A/B ...")
        confirmation = confirm(model_id, BASELINE, best, prompt, max_tokens)
        ratio = confirmation["ratios"]["candidate/baseline"]["total_ns"]
        ok = confirmation["token_identity"] and ratio["ci_high"] < 1.0
        print(f"  confirmed ratio {ratio['median_ratio']:.4f} "
              f"CI [{ratio['ci_low']:.4f}; {ratio['ci_high']:.4f}] "
              f"tokens identical {confirmation['token_identity']} -> {'accepted' if ok else 'rejected'}")
        if not ok:
            best, best_result = BASELINE, base

    gain = 1.0 - best_result["total_ns"] / base["total_ns"]
    profile = {
        "conditions": conditions(
            model_id, len(ids), max_tokens, model_identity=resolved.identity
        ),
        "confirmation": ({"ratio": confirmation["ratios"]["candidate/baseline"],
                          "token_identity": confirmation["token_identity"]}
                         if confirmation else None),
        "fingerprint": hardware["fingerprint"],
        "model_id": resolved.identity.model_id,
        "model_identity": resolved.identity.to_dict(),
        "knobs": best.as_dict(),
        "baseline_ns": base["total_ns"],
        "tuned_ns": best_result["total_ns"],
        "baseline_decode_ns": base["decode_ns"],
        "tuned_decode_ns": best_result["decode_ns"],
        "baseline_prefill_ns": base["prefill_ns"],
        "tuned_prefill_ns": best_result["prefill_ns"],
        "gain": gain,
        "token_count": len(reference),
        "tokens": reference,
        "trials": trials,
        "hardware": hardware,
        "tuned_at": time.time(),
    }
    save_profile(profile)
    print(f"tuned: {gain*100:.2f}% faster end to end, tokens identical, stored in {PROFILES}")
    return profile


def _all_profiles() -> dict[str, Any]:
    if not PROFILES.is_file():
        return {}
    try:
        loaded = json.loads(PROFILES.read_text())
        return loaded if isinstance(loaded, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_profile(profile: dict[str, Any]) -> None:
    identity = ModelIdentity.from_dict(profile["model_identity"])
    conditions_record = profile.get("conditions")
    if (profile.get("model_id") != identity.model_id
            or not isinstance(conditions_record, dict)
            or set(conditions_record) != PROFILE_CONDITION_FIELDS
            or conditions_record.get("conditions_schema") != PROFILE_CONDITIONS_SCHEMA
            or not _conditions_match_identity(conditions_record, identity)):
        raise ModelIdentityError("profile conditions do not match exact model identity")
    profiles = _all_profiles()
    profiles[f"{profile['fingerprint']}/{identity.identity_sha256}"] = profile
    STORE.mkdir(parents=True, exist_ok=True)
    PROFILES.write_text(json.dumps(profiles, indent=2, sort_keys=True))


def load_profile(model_id: str = DEFAULT_MODEL, *, require_compatible: bool = True,
                 revision: str | None = None,
                 model_identity: ModelIdentity | None = None) -> dict[str, Any] | None:
    """Return a valid profile, rejecting current identity drift by default.

    ``require_compatible=False`` is reserved for ``revalidate()``, which needs
    to load a schema-valid but stale profile in order to report its canary
    result rather than silently treating it as absent.
    """
    identity = model_identity or resolve_local_model(model_id, revision).identity
    if not Path(model_id).expanduser().is_dir() and model_id != identity.model_id:
        raise ModelIdentityError("profile model_id does not match exact identity")
    profile = _all_profiles().get(f"{fingerprint()}/{identity.identity_sha256}")
    if not isinstance(profile, dict) or profile.get("model_id") != identity.model_id:
        return None
    conditions_record = profile.get("conditions")
    required = PROFILE_CONDITION_FIELDS
    if (not isinstance(conditions_record, dict)
            or set(conditions_record) != required
            or conditions_record.get("conditions_schema") != PROFILE_CONDITIONS_SCHEMA
            or conditions_record.get("model_id") != identity.model_id
            or conditions_record.get("model_identity_sha256") != identity.identity_sha256
            or not _conditions_match_identity(conditions_record, identity)
            or not isinstance(profile.get("model_identity"), dict)
            or not isinstance(profile.get("knobs"), dict)):
        return None
    try:
        stored_identity = ModelIdentity.from_dict(profile["model_identity"])
    except (ModelIdentityError, TypeError, ValueError):
        return None
    if stored_identity != identity:
        return None
    try:
        Knobs(**profile["knobs"])
    except (TypeError, ValueError):
        return None
    try:
        prompt_tokens = int(conditions_record["prompt_tokens"])
        max_tokens = int(conditions_record["max_tokens"])
    except (TypeError, ValueError):
        return None
    if require_compatible and stale(
        profile, model_id, prompt_tokens, max_tokens, model_identity=identity
    ):
        return None
    return profile


def knobs_for(model_id: str = DEFAULT_MODEL) -> Knobs:
    profile = load_profile(model_id)
    return Knobs(**profile["knobs"]) if profile else BASELINE


def _self_check() -> None:
    from dataclasses import replace as _replace
    assert Knobs(**BASELINE.as_dict()) == BASELINE
    assert _replace(BASELINE, readback_every=8).readback_every == 8
    names = {name for name, _ in SEARCH}
    assert names <= set(BASELINE.as_dict()), f"unknown knob in search: {names - set(BASELINE.as_dict())}"
    assert SEARCH[-1][0] == "fuse_projections", "the reloading knob must be searched last"
    assert gpu_busy.__doc__
    fake = {"conditions": {"fingerprint": "x", "prompt_tokens": 100, "max_tokens": 32}}
    assert "no conditions recorded" in stale({}, DEFAULT_MODEL, 100, 32)
    drifted = stale(fake, DEFAULT_MODEL, 100, 32)
    assert any("fingerprint" in d for d in drifted), drifted
    near = {"conditions": {"prompt_tokens": 100, "max_tokens": 32}}
    assert stale(near, DEFAULT_MODEL, 110, 32) == [], "a 10% longer prompt is the same regime"
    assert any("prompt_tokens" in d for d in stale(near, DEFAULT_MODEL, 400, 32))
    print("tune self-check ok:", len(SEARCH), "knobs,", fingerprint(), "fingerprint")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="tune the local decode path against this hardware")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--force", action="store_true", help="measure even if another model process is up")
    parser.add_argument("--show", action="store_true", help="print the stored profile and exit")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--revalidate", action="store_true", help="canary-check the stored profile")
    parser.add_argument("--no-confirm", action="store_true", help="skip the paired A/B confirmation")
    args = parser.parse_args(argv)

    if args.self_check:
        _self_check()
        return 0
    if args.show:
        profile = load_profile(args.model)
        print(json.dumps(profile, indent=2, sort_keys=True) if profile else "no profile for this hardware yet")
        return 0
    if args.revalidate:
        print(json.dumps(revalidate(args.model, max_tokens=args.max_tokens), indent=2, default=str))
        return 0
    tune(args.model, max_tokens=args.max_tokens, repeats=args.repeats, force=args.force,
         confirm_winner=not args.no_confirm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
