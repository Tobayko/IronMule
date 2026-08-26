"""Paired A/B across fresh processes.

One process per pair, arm order alternating, so a machine that drifts warmer or
busier during the run cannot hand the win to whichever arm ran second. Each child
loads its own model per arm, because some knobs mutate the model in place and a
reused model would carry one arm's surgery into the next.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
from dataclasses import replace
from typing import Any

from .bench import interleave, paired_ratio, summarise
from .runtime import Knobs

CHILD_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONNOUSERSITE": "1"}


def _child(spec: dict[str, Any]) -> dict[str, Any]:
    """Runs inside the subprocess: one fresh model per arm, in the given order."""
    from .tune import DEFAULT_MODEL, DEFAULT_PROMPT, _eos_ids, load_engine, prompt_ids

    out: dict[str, Any] = {"pid": os.getpid(), "arms": {}, "order": spec["order"]}
    for name in spec["order"]:
        knobs = Knobs(**spec["arms"][name])
        engine, tok = load_engine(spec.get("model", DEFAULT_MODEL), knobs)
        ids = prompt_ids(tok, spec.get("prompt", DEFAULT_PROMPT))
        eos = _eos_ids(tok)
        for _ in range(spec["warmup"]):
            engine.generate(ids, spec["max_tokens"], eos)
        runs = [engine.generate(ids, spec["max_tokens"], eos) for _ in range(spec["repeats"])]
        out["arms"][name] = {
            "total_ns": [r["total_ns"] for r in runs],
            "prefill_ns": [r["prefill_ns"] for r in runs],
            "decode_ns": [r["decode_ns"] for r in runs],
            "logical_tokens": runs[0]["logical_tokens"],
            "deterministic": all(r["logical_tokens"] == runs[0]["logical_tokens"] for r in runs),
            "decode_steps": len(runs[0]["physical_tokens"]) - 1,
            "prompt_tokens": len(ids),
        }
        del engine
    import mlx.core as mx
    out["mlx_peak_bytes"] = mx.get_peak_memory()
    return out


def run(arms: dict[str, Knobs], processes: int = 6, repeats: int = 7, warmup: int = 2,
        max_tokens: int = 32, model: str | None = None, prompt: str | None = None) -> dict[str, Any]:
    """Spawn the children, collect raw samples, pair them per process."""
    from .tune import gpu_busy
    busy = gpu_busy()
    if busy:
        raise RuntimeError(f"another model process is running, refusing to measure ({busy})")

    names = list(arms)
    orders = interleave(names, processes)
    spec_base = {"arms": {n: k.as_dict() for n, k in arms.items()}, "repeats": repeats,
                 "warmup": warmup, "max_tokens": max_tokens}
    if model:
        spec_base["model"] = model
    if prompt is not None:
        spec_base["prompt"] = prompt

    children = []
    for order in orders:
        spec = dict(spec_base, order=order)
        proc = subprocess.run(
            [sys.executable, "-c",
             "import json,sys;from ironmule.ab import _child;"
             "print('@@'+json.dumps(_child(json.loads(sys.argv[1]))))", json.dumps(spec)],
            capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            env={**os.environ, **CHILD_ENV},
        )
        line = next((l for l in proc.stdout.splitlines() if l.startswith("@@")), None)
        if line is None:
            raise RuntimeError(f"child failed: {proc.stderr[-2000:]}")
        children.append(json.loads(line[2:]))

    per_arm: dict[str, dict[str, list[float]]] = {
        n: {"total_ns": [], "prefill_ns": [], "decode_ns": []} for n in names}
    tokens: dict[str, list[list[int]]] = {n: [] for n in names}
    for child in children:
        for name in names:
            arm = child["arms"][name]
            for metric in ("total_ns", "prefill_ns", "decode_ns"):
                per_arm[name][metric].append(statistics.median(arm[metric]))
            tokens[name].append(arm["logical_tokens"])

    reference = tokens[names[0]][0]
    identical = all(seq == reference for name in names for seq in tokens[name])
    deterministic = all(child["arms"][n]["deterministic"] for child in children for n in names)

    result: dict[str, Any] = {
        "arms": {n: k.as_dict() for n, k in arms.items()},
        "processes": processes, "repeats": repeats, "warmup": warmup,
        "raw": children,
        "per_arm": {n: {m: summarise(v) for m, v in metrics.items()} for n, metrics in per_arm.items()},
        "token_identity": identical,
        "deterministic": deterministic,
        "reference_tokens": reference,
        "ratios": {},
    }
    base = names[0]
    for name in names[1:]:
        result["ratios"][f"{name}/{base}"] = {
            metric: paired_ratio(per_arm[name][metric], per_arm[base][metric])
            for metric in ("total_ns", "prefill_ns", "decode_ns")
        }
    return result


def report(result: dict[str, Any]) -> str:
    lines = [f"token identity: {result['token_identity']}  deterministic: {result['deterministic']}"]
    for name, metrics in result["per_arm"].items():
        lines.append(f"  {name:34s} total {metrics['total_ns']['median']/1e6:8.2f} ms  "
                     f"prefill {metrics['prefill_ns']['median']/1e6:8.2f}  "
                     f"decode {metrics['decode_ns']['median']/1e6:8.2f}")
    for pair, metrics in result["ratios"].items():
        lines.append(f"  {pair}")
        for metric, r in metrics.items():
            lines.append(f"    {metric:11s} ratio {r['median_ratio']:.4f}  "
                         f"95% CI [{r['ci_low']:.4f}; {r['ci_high']:.4f}]  "
                         f"({(1-r['median_ratio'])*100:+.2f}%)")
    return "\n".join(lines)


def _self_check() -> None:
    assert interleave(["base", "cand"], 3) == [["base", "cand"], ["cand", "base"], ["base", "cand"]]
    k = Knobs(fuse_projections=True)
    assert Knobs(**k.as_dict()) == k
    assert replace(k, fuse_projections=False).fuse_projections is False
    print("ab self-check ok")


if __name__ == "__main__":
    _self_check()
