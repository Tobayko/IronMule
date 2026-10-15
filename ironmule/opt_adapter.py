"""OPT1 adapter 1: IronMule under the Rust coordinator (`native/optimizer`, `imopt cycle -- python -m
ironmule.opt_adapter --model M`). Opt-in; nothing here runs unless the coordinator calls it.

Every call is its own process (the coordinator's isolation). Actions are bounded: the reference
(IronMule's untuned engine, ``BASELINE``) and one knob change from ``tune.SEARCH`` each; on a CUDA host
with ``IRONMULE_RUST2_BINARY`` set, ``engine=rust2`` too, as a ``tested_only`` action: it is measured,
but its tokens are not the MLX engine's (RUST2-P teacher forcing 126/128), so it can never qualify
automatically; numerically different execution needs its own quality and consent rules.

The evaluator is IronMule's own: ``tune.confirm`` (paired A/B over fresh processes with token, count,
stop-reason and determinism identity, RSI2's early stop) judged by ``tune._confirmation_decision``.
The coordinator cannot change it. Activation writes an ordinary tune profile through
``tune.save_profile`` (so ``--engine tuned`` serves it and ``tune.load_profile`` must accept it), and
returns the previous profile entry; ``revert`` restores exactly that entry or removes the key.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import importlib

from .hw import STORE, fingerprint, probe
from .runtime import BASELINE, Knobs

# The package re-exports the function `tune`, which shadows the module of the same name.
tune = importlib.import_module("ironmule.tune")
RUST2 = "engine=rust2"


def _actions() -> dict[str, Knobs | None]:
    actions: dict[str, Knobs | None] = {"baseline": BASELINE}
    for knob, values in tune.SEARCH:
        for value in values:
            actions[f"{knob}={value}"] = replace(BASELINE, **{knob: value})
    if os.environ.get("IRONMULE_RUST2_BINARY") and _cuda():
        actions[RUST2] = None
    return actions


def _cuda() -> bool:
    try:
        import mlx.core as mx
        return bool(mx.cuda.is_available())
    except Exception:  # noqa: BLE001 - absent CUDA is a capability state, not an error
        return False


def _state() -> Path:
    path = Path(os.environ.get("IMOPT_IRONMULE_STATE", str(STORE / "imopt")))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def describe(args) -> dict[str, Any]:
    identity = tune.resolve_local_model(args.model).identity
    from .bench import environment
    env = environment()
    acts = [{"id": name, "settings": (knobs.as_dict() if knobs is not None else {"engine": "rust2"}),
             "class": "auto" if knobs is not None else "tested_only"} for name, knobs in _actions().items()]
    return ({"name": "ironmule", "reference": "baseline", "actions": acts, "metric": "total_s", "evaluator": "adapter",
                 "identity": {"model_identity_sha256": identity.identity_sha256, "hardware_fingerprint": fingerprint(),
                              "mlx": env.get("mlx"), "mlx_lm": env.get("mlx_lm"), "prompt_sha256": _digest(args.prompt),
                              "max_tokens": args.max_tokens, "repeats": args.repeats}})


def _rust2_run(args) -> dict[str, Any]:
    from mlx_lm.utils import load_tokenizer
    from ironmule_product.rust2_bridge import Rust2Bridge
    resolved = tune.resolve_local_model(args.model)
    tokenizer = load_tokenizer(resolved.path)
    ids = tune.prompt_ids(tokenizer, args.prompt)
    eos = tune._eos_ids(tokenizer)
    bridge = Rust2Bridge(os.environ["IRONMULE_RUST2_BINARY"], str(resolved.path), eos,
                         os.environ.get("IRONMULE_RUST2_DRAFT") or None)
    try:
        for _ in range(2):
            bridge.generate(ids, args.max_tokens)
        times, tokens = [], None
        for _ in range(args.repeats):
            t0 = time.perf_counter_ns()
            result = bridge.generate(ids, args.max_tokens)
            times.append(time.perf_counter_ns() - t0)
            if tokens is not None and result.tokens != tokens:
                return {"ok": False, "error": "rust2 not deterministic"}
            tokens = result.tokens
    finally:
        bridge.close()
    times.sort()
    return {"ok": True, "cost": times[len(times) // 2] / 1e9, "correctness": _digest(tokens),
            "metrics": {"engine": "rust2", "tokens": len(tokens)}}


def run(args) -> dict[str, Any]:
    actions = _actions()
    if args.action not in actions:
        return ({"ok": False, "error": f"unknown action {args.action}"})
    if args.action == RUST2:
        return (_rust2_run(args))
    knobs = actions[args.action]
    try:
        engine, tokenizer = tune.load_engine(args.model, knobs, offline=True)
    except Exception as exc:  # noqa: BLE001 - an unsupported knob is a failed test, reported
        return ({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    try:
        ids = tune.prompt_ids(tokenizer, args.prompt)
        m = tune.measure(engine, ids, args.max_tokens, tune._eos_ids(tokenizer), repeats=args.repeats)
    except Exception as exc:  # noqa: BLE001
        return ({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        tune._close_engine(engine)
    return ({"ok": bool(m["deterministic"]), "cost": m["total_ns"] / 1e9, "correctness": _digest(m["logical_tokens"]),
                 "metrics": {"prefill_s": m["prefill_ns"] / 1e9, "decode_s": m["decode_ns"] / 1e9, "capacity": m["capacity"],
                             "tokens": len(m["logical_tokens"]), "prompt_tokens": len(ids)},
                 "error": None if m["deterministic"] else "not deterministic"})


def qualify(args) -> dict[str, Any]:
    knobs = _actions().get(args.action)
    if knobs is None:
        return ({"ok": True, "qualified": False, "reason": "not an automatically qualifiable action"})
    raw = tune.confirm(args.model, BASELINE, knobs, args.prompt, args.max_tokens)
    accepted, reason = tune._confirmation_decision(raw, expected_baseline=BASELINE, expected_candidate=knobs)
    ratio = (raw.get("ratios", {}).get("candidate/baseline", {}).get("total_ns", {}) if isinstance(raw, dict) else {})
    (_state() / f"qualify-{args.action}.json").write_text(json.dumps({"raw": raw, "accepted": accepted, "reason": reason,
                                                                       "binding": _binding(args)}))
    return ({"ok": True, "qualified": bool(accepted), "reason": reason, "mean_ratio": ratio.get("median_ratio"),
                 "ci": [ratio.get("ci_low"), ratio.get("ci_high")], "processes": raw.get("processes"),
                 "evidence_sha256": tune._confirmation_evidence_sha256(raw)})


def _profile_for(args, knobs: Knobs, raw: dict[str, Any]) -> dict[str, Any]:
    """The same profile `tune.tune` writes after an accepted confirmation (its validator must accept it)."""
    resolved = tune.resolve_local_model(args.model)
    per = raw["per_arm"]
    median = {arm: {m: per[arm][m]["median"] for m in ("total_ns", "prefill_ns", "decode_ns")} for arm in per}
    ratio = raw["ratios"]["candidate/baseline"]
    tokens = list(raw["reference_tokens"])
    record = {"ratio": ratio, "token_identity": raw.get("token_identity"), "token_count_identity": raw.get("token_count_identity"),
              "stop_reason_identity": raw.get("stop_reason_identity"), "deterministic": raw.get("deterministic"),
              "accepted": True, "rejection_reason": None, "evidence_sha256": tune._confirmation_evidence_sha256(raw)}
    hardware = probe()
    from mlx_lm.utils import load_tokenizer
    prompt_tokens = len(tune.prompt_ids(load_tokenizer(resolved.path), args.prompt))
    return {"conditions": tune.conditions(args.model, prompt_tokens, args.max_tokens, model_identity=resolved.identity),
            "confirmation": record, "confirmation_evidence": dict(raw), "confirmation_candidate_knobs": knobs.as_dict(),
            "fingerprint": hardware["fingerprint"], "model_id": resolved.identity.model_id,
            "model_identity": resolved.identity.to_dict(), "knobs": knobs.as_dict(),
            "baseline_ns": median["baseline"]["total_ns"], "tuned_ns": median["candidate"]["total_ns"],
            "baseline_decode_ns": median["baseline"]["decode_ns"], "tuned_decode_ns": median["candidate"]["decode_ns"],
            "baseline_prefill_ns": median["baseline"]["prefill_ns"], "tuned_prefill_ns": median["candidate"]["prefill_ns"],
            "gain": 1.0 - ratio["total_ns"]["median_ratio"], "token_count": len(tokens), "tokens": tokens,
            "trials": [], "knowledge": {}, "explorer": "imopt", "tree": None, "hardware": hardware, "tuned_at": time.time()}


def _binding(args) -> dict[str, Any]:
    """What a qualification is valid for: this model identity, hardware, software, prompt and length."""
    from .bench import environment
    env = environment()
    return {"model_identity_sha256": tune.resolve_local_model(args.model).identity.identity_sha256,
            "hardware_fingerprint": fingerprint(), "mlx": env.get("mlx"), "mlx_lm": env.get("mlx_lm"),
            "prompt_sha256": _digest(args.prompt), "max_tokens": args.max_tokens}


def _key(args) -> str:
    return tune._profile_key(fingerprint(), tune.resolve_local_model(args.model).identity)


def _action_of(knobs: Knobs | None) -> str | None:
    if knobs is None or knobs == BASELINE:
        return None
    return next((name for name, k in _actions().items() if k == knobs), "custom")


def apply(args) -> dict[str, Any]:
    knobs = _actions().get(args.action)
    path = _state() / f"qualify-{args.action}.json"
    if knobs is None or not path.is_file():
        return ({"ok": False, "error": "no qualification evidence for this action"})
    record = json.loads(path.read_text())
    if record.get("accepted") is not True:
        return ({"ok": False, "error": "the action was not qualified"})
    if record.get("binding") != _binding(args):
        return ({"ok": False, "error": "the qualification belongs to another model, machine, software or workload"})
    key = _key(args)
    previous_profile = tune._all_profiles().get(key)
    previous_valid = tune.load_profile(args.model)
    previous = {"key": key, "profile": previous_profile,
                "action": _action_of(Knobs(**previous_valid["knobs"]) if previous_valid else None)}
    tune.save_profile(_profile_for(args, knobs, record["raw"]))
    loaded = tune.load_profile(args.model, expected_candidate=knobs)
    if loaded is None or Knobs(**loaded["knobs"]) != knobs:
        _restore(previous)
        return ({"ok": False, "error": "the stored profile was not accepted by tune.load_profile; restored"})
    return ({"ok": True, "previous": previous})


def _restore(previous: dict[str, Any]) -> None:
    profiles = tune._all_profiles()
    key = previous.get("key")
    if previous.get("profile") is None:
        profiles.pop(key, None)
    else:
        profiles[key] = previous["profile"]
    STORE.mkdir(parents=True, exist_ok=True)
    tmp = tune.PROFILES.with_suffix(".tmp")
    tmp.write_text(json.dumps(profiles, indent=2, sort_keys=True))
    tmp.replace(tune.PROFILES)


def revert(args) -> dict[str, Any]:
    previous = json.loads(args.previous) if args.previous else None
    if not previous:
        # Nothing was recorded: remove this model's profile on this machine (the untuned default).
        previous = {"key": _key(args), "profile": None}
    _restore(previous)
    return ({"ok": True})


def active(args) -> dict[str, Any]:
    profile = tune.load_profile(args.model)
    return ({"ok": True, "action": _action_of(Knobs(**profile["knobs"]) if profile else None)})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ironmule.opt_adapter")
    parser.add_argument("command", choices=("describe", "run", "qualify", "apply", "revert", "active"))
    parser.add_argument("--model", default=tune.DEFAULT_MODEL)
    parser.add_argument("--prompt", default=tune.DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--action")
    parser.add_argument("--seed")
    parser.add_argument("--previous")
    args = parser.parse_args(argv)
    if args.command in ("run", "qualify", "apply") and not args.action:
        parser.error("--action is required")
    handler = {"describe": describe, "run": run, "qualify": qualify, "apply": apply, "revert": revert, "active": active}
    # The protocol's answer is the last stdout line; library output goes to stderr meanwhile.
    from contextlib import redirect_stdout
    with redirect_stdout(sys.stderr):
        result = handler[args.command](args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
