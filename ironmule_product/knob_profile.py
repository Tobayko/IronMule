"""KP1: the tuned engine's knobs for a setup where a hardware comparison qualified them.

Only knobs that leave every answer unchanged may be set this way: the ones HW4 found bit-identical to the
baseline on a Tesla T4 (prefill logits and state, every decode step, the final state) and the host-read interval,
which moves no arithmetic. A set is shipped with the comparison that qualified it, bound to the model, revision,
GPU name, engine and the MLX and mlx-lm versions it was measured on (a compiled step traces mlx-lm's model code);
anywhere else it is ignored and nothing changes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

KNOBS_SCHEMA = "ironmule.knob_evidence.v1"
KNOBS_FILE = "knobs.json"
# HW4: class A on the T4 (with MLX 0.32.4); readback_every only moves host reads.
CLASS_A_FIELDS = frozenset({"fuse_projections", "compiled_fixed_cache", "fused_argmax", "readback_every"})
BINDING_KEYS = frozenset({"model_id", "revision", "gpu", "engine", "mlx", "mlx_lm"})
# `ironmule.runtime.BASELINE.as_dict()`, kept in step by a test: this module runs in the server's own process,
# which does not import MLX.
BASELINE_KNOBS: dict[str, Any] = {
    "fuse_projections": False, "compiled_fixed_cache": False, "fused_argmax": False, "head_skip_prefill": False,
    "prefill_into_fixed": False, "readback_every": 1, "speculate_k": 0, "speculate_ngram": 3, "capacity_slack": 0,
    "wired_fraction": 0.0, "k3840_matvec": False,
}


def kp1_knobs() -> dict[str, Any]:
    """KP1's candidate, exactly as RL7's result named it: the compiled step and a host read every 4 tokens."""
    return {**BASELINE_KNOBS, "compiled_fixed_cache": True, "readback_every": 4}


def framework_versions() -> dict[str, str | None]:
    """The installed MLX and mlx-lm versions, read without importing either."""
    from importlib import metadata

    versions: dict[str, str | None] = {}
    for key, names in (("mlx", ("mlx",)), ("mlx_lm", ("mlx-lm", "mlx_lm"))):
        versions[key] = None
        for name in names:
            try:
                versions[key] = metadata.version(name)
                break
            except metadata.PackageNotFoundError:
                continue
    return versions


def class_a_knobs(value: Any) -> dict[str, Any] | None:
    """`value` as a complete knob mapping when it differs from the baseline only in class-A fields, else None.
    Plain Python: the server's own process never imports MLX; the worker builds `Knobs` from the result."""
    if not isinstance(value, Mapping) or set(value) - set(BASELINE_KNOBS):
        return None
    knobs = {**BASELINE_KNOBS, **value}
    if any(knobs[k] != v or type(knobs[k]) is not type(v) for k, v in BASELINE_KNOBS.items()
           if k not in CLASS_A_FIELDS):
        return None
    if any(type(knobs[k]) is not bool for k in CLASS_A_FIELDS - {"readback_every"}):
        return None
    if type(knobs["readback_every"]) is not int or not 1 <= knobs["readback_every"] <= 64:
        return None
    return knobs


def shipped_knobs(folder: Path | None = None) -> bool:
    """Whether any knob record ships at all, so a server without one looks nothing up."""
    folder = Path(__file__).with_name("dispatch") if folder is None else Path(folder)
    return any(folder.glob(f"*/{KNOBS_FILE}"))


def _digest(body: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def qualified_knobs(binding: Mapping[str, Any], folder: Path | None = None) -> tuple[dict, dict] | None:
    """A shipped class-A knob set whose binding names exactly this setup and whose record's digest, schema and
    verdict hold; None otherwise, and the engine's own default serves."""
    folder = Path(__file__).with_name("dispatch") if folder is None else Path(folder)
    for path in sorted(folder.glob(f"*/{KNOBS_FILE}")):
        try:
            record = json.loads(path.read_text())
            body = {k: v for k, v in record.items() if k != "sha256"}
            if _digest(body) != record.get("sha256") or body.get("schema") != KNOBS_SCHEMA \
                    or body.get("verdict") != "pass":
                continue
            wanted = body.get("binding") or {}
            if set(wanted) != BINDING_KEYS or any(wanted[key] != binding.get(key) for key in BINDING_KEYS):
                continue
            knobs = class_a_knobs(body.get("knobs"))
            if knobs is not None:
                return knobs, body
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


def knobs_record(experiment: str, binding: Mapping[str, Any], knobs: Mapping[str, Any], domain: list[str], rival: str,
                 gain: float, gain_ci95: list[float], floor: float, noise: float, no_harm: bool) -> dict[str, Any]:
    """The record `qualified_knobs` accepts. As KP1 fixed before its data, it passes when the gain's 2.5 % bound
    is above 0 and above `floor`, the A/A control (the answers are identical, so every resolved gain counts),
    and no state was harmed beyond the A/A noise, which `no_harm` reports."""
    knobs = class_a_knobs(knobs)
    if knobs is None:
        raise ValueError("only class-A knobs can be qualified this way")
    body = {"schema": KNOBS_SCHEMA, "experiment": experiment, "binding": dict(binding), "knobs": knobs,
            "domain": list(domain), "rival": rival, "gain": gain, "gain_ci95": list(gain_ci95), "floor": floor,
            "noise": noise, "no_harm": bool(no_harm),
            "verdict": "pass" if gain_ci95[0] > max(0.0, floor) and no_harm else "fail"}
    return {**body, "sha256": _digest(body)}


def compare_class_a(model: Any, tokenizer: Any, knobs: Any, prompts: Mapping[str, list[int]],
                    capacities: list[int], steps: int) -> dict[str, Any]:
    """KP1's kill 1, HW4's method: for each prompt at each capacity it fits, the baseline's prefill logits, its
    `steps` decode steps and its final state against `knobs` fed the baseline's tokens, bit for bit."""
    import mlx.core as mx

    from ironmule import runtime

    def run(engine: Any, ids: list[int], capacity: int, feed: list[int] | None):
        captured, original = [], runtime._project

        def capture(m: Any, hidden: Any) -> Any:
            out = original(m, hidden)
            captured.append(out)
            return out

        runtime._project = capture
        try:
            state, token = engine._prefill(list(ids), capacity)
        finally:
            runtime._project = original
        logits, picks = [captured[-1][:, -1, :]], [int(token.reshape((-1,)).item())]
        body = engine._body(capacity, 1)
        current = picks[0] if feed is None else feed[0]
        for k in range(steps):
            out = body(mx.array([[current]]), state)
            mx.eval(out[0], *runtime._leaves(out[1]))
            state = out[1]
            logits.append(out[0].reshape((1, -1)))
            picks.append(int(mx.argmax(logits[-1], axis=-1).item()))
            current = picks[-1] if feed is None else feed[k + 1]
        return logits, picks, state

    knobs = knobs if isinstance(knobs, runtime.Knobs) else runtime.Knobs(**knobs)
    base, candidate = runtime.Engine(model, tokenizer, runtime.BASELINE), runtime.Engine(model, tokenizer, knobs)
    rows, differ = [], []
    try:
        for capacity in sorted(set(capacities)):
            for name, ids in prompts.items():
                if len(ids) + steps + 8 > capacity:
                    continue
                reference, picks, reference_state = run(base, ids, capacity, None)
                got, _, state = run(candidate, ids, capacity, picks)
                logits_same = all(bool(mx.array_equal(a, b).item()) for a, b in zip(reference, got))
                state_same = all(bool(mx.array_equal(a, b).item()) for a, b in
                                 zip(runtime._leaves(reference_state), runtime._leaves(state)))
                rows.append({"capacity": capacity, "prompt": name, "logits_identical": logits_same,
                             "state_identical": state_same})
                if not (logits_same and state_same):
                    differ.append([capacity, name])
    finally:
        base.close()
        candidate.close()
    return {"rows": rows, "differ": differ, "ok": bool(rows) and not differ}
