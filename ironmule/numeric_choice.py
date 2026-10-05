"""NUM1: choose an output-changing numeric plan only where its quality holds.

A numeric plan computes the checkpoint's weights in another type and changes the output, so
it is never chosen without the caller's consent. With consent, on a device class that has
measured plans (CUDA below compute capability 8, which emulates bfloat16):

1. plans the measured table marks `refused` are never taken;
2. the fastest `recommended` plan is taken on its recorded evidence;
3. any other plan that the table does not know to be slower is measured here (on Apple
   Silicon, which has no table, every plan but the CUDA-only `native` one; NUM2): the paired
   perplexity gate of PORT1/PORT2 on a fixed public-domain text (bootstrap upper bound of
   the ratio below `QUALITY_BOUND`) and a decode speed check. Every candidate is gated and the fastest that passes both is taken.

The decision and its evidence are stored per hardware fingerprint and model identity, so it
is made once; the reference path stays the default everywhere else.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Callable

from .hw import STORE
from .numeric_plans import CUDA_PRE_AMPERE, QUALITY_BOUND, architecture_of, device_class, measurements_for

TEXT = Path(__file__).with_name("quality_reference.txt")
PLANS = ("native", "float16", "float32")  # every plan the runtime accepts
CHUNKS, CHUNK_TOKENS = 16, 512
# What tune keeps on almost every machine and model; time is scaled to 128 answered tokens.
SPEED_KNOBS = {"compiled_fixed_cache": True, "head_skip_prefill": True, "readback_every": 4}
SPEED_TOKENS = 128
SHORT_PROMPT = "Explain in detail how rain forms, step by step."


def _store() -> Path:
    return STORE / "autopilot" / "numeric.json"


def _text_ids(tokenizer) -> list[int]:
    text = "".join(line for line in TEXT.read_text(encoding="utf-8").splitlines(True)
                   if not line.startswith("#"))
    return list(tokenizer.encode(text))


def nll_per_chunk(engine, ids: list[int], chunks: int = CHUNKS, size: int = CHUNK_TOKENS) -> list[float]:
    """Mean next-token negative log-likelihood per fixed chunk, as `quality.py` computes it."""
    import mlx.core as mx
    rows = []
    for index in range(chunks):
        chunk = ids[index * size:(index + 1) * size + 1]
        if len(chunk) < size + 1:
            break
        logits = engine.model(mx.array([chunk[:-1]]))[0].astype(mx.float32)
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        nll = -mx.mean(log_probs[mx.arange(size), mx.array(chunk[1:])])
        mx.eval(nll)
        rows.append(float(nll.item()))
        del logits, log_probs
        mx.clear_cache()
    return rows


def ratio_interval(reference: list[float], candidate: list[float], seed: int = 20261004) -> dict[str, Any]:
    """Perplexity ratio candidate/reference over paired chunks with a 95 % bootstrap interval."""
    if len(reference) != len(candidate) or len(reference) < 4:
        raise ValueError("the gate needs at least four paired chunks")
    pairs = list(zip(reference, candidate))
    rng = random.Random(seed)

    def ratio(sample):
        return math.exp(statistics.fmean(c for _, c in sample) - statistics.fmean(r for r, _ in sample))

    boot = sorted(ratio([rng.choice(pairs) for _ in pairs]) for _ in range(10000))
    return {"ratio": ratio(pairs), "interval": [boot[250], boot[9750]], "chunks": len(pairs),
            "passes": boot[9750] < QUALITY_BOUND}


def candidates(architecture: str | None, device: str | None) -> list[dict[str, Any]]:
    """Plans to consider, fastest known first; refused and known-slower plans are dropped."""
    if device is None or architecture is None:
        return []
    rows = {row.plan: row for row in measurements_for(architecture, device)}
    chosen = []
    for plan in PLANS:
        row = rows.get(plan)
        if plan == "native" and device != CUDA_PRE_AMPERE:
            continue  # IronMule's own CUDA kernels; refused on every other device
        if row is not None and row.verdict() in ("refused", "slower"):
            continue
        chosen.append({"plan": plan, "verdict": row.verdict() if row else "unmeasured",
                       "table_wall_ratio": row.wall_ratio if row else None})
    return sorted(chosen, key=lambda c: (c["verdict"] != "recommended",
                                          c["table_wall_ratio"] if c["table_wall_ratio"] is not None else 1.0))


def _checkpoint_dtype(model) -> str | None:
    """The floating type the checkpoint computes in, e.g. `bfloat16`; None when unknown."""
    import mlx.core as mx
    from mlx.utils import tree_flatten
    try:
        leaves = tree_flatten(model.parameters())
    except AttributeError:
        return None
    return next((str(value.dtype).rsplit(".", 1)[-1] for _, value in leaves
                 if mx.issubdtype(value.dtype, mx.floating)), None)


def _no_op(plan: str, dtype: str | None) -> bool:
    """A plan that cannot change this checkpoint: its own type, or `native` without bfloat16."""
    return dtype is not None and (plan == dtype or (plan == "native" and dtype != "bfloat16"))


def _speed(engine, tokenizer) -> dict[str, float]:
    """Seconds per 128 served tokens on the serving knobs, for a short and a long prompt.

    Served through `Runtime.serve`. One prompt cannot speak for both: wider arithmetic speeds a
    long prompt's compute-bound prefill, not a short one's bandwidth-bound decode (Gemma 3 4B
    float32: 0.807 of stock on tune's long prompt, 1.008 served on a short question; NUM6).
    """
    import statistics
    import time
    from dataclasses import replace

    from .plans import StrictOneShotPlan
    from .service import Request, Runtime
    from .tune import DEFAULT_PROMPT
    engine.knobs = replace(engine.knobs, **SPEED_KNOBS)
    runtime = Runtime(engine, tokenizer)  # the engine is closed by the caller
    out = {}
    for name, prompt in (("short", SHORT_PROMPT), ("long", DEFAULT_PROMPT)):
        ids, rows = runtime.encode(prompt), []
        for index in range(4):  # the first answer compiles; not recorded
            started = time.perf_counter()
            request = Request(prompt_ids=ids, max_tokens=SPEED_TOKENS, plan=StrictOneShotPlan())
            tokens = runtime.serve([request])[0].tokens
            if index:
                rows.append((time.perf_counter() - started) * SPEED_TOKENS / max(len(tokens), 1))
        out[name] = statistics.median(rows)
    return out


def _speed_ratios(stock: Any, plan: Any) -> dict[str, float]:
    if not isinstance(stock, dict):  # one number stands for both workloads
        stock, plan = {"short": stock, "long": stock}, {"short": plan, "long": plan}
    return {name: plan[name] / stock[name] for name in stock}


def _faster(ratios: dict[str, float]) -> bool:
    """Not slower on any workload and clearly faster on at least one."""
    return max(ratios.values()) < 1.0 and min(ratios.values()) < 0.95


def _stored(key: str) -> dict[str, Any] | None:
    try:
        return json.loads(_store().read_text()).get(key)
    except (OSError, ValueError):
        return None


def choose_out_of_process(model_id: str, fingerprint: str, identity_sha256: str, *,
                          log: Callable[[str], None] = print) -> dict[str, Any]:
    """`choose` in a child, so the caller starts tuning and serving with a clean device.

    A gate loads up to four engines. On a T4 the device memory they leave behind made the
    following tune and load fail (`cudaGraphInstantiate ... out of memory`, NUM1), and a
    closed hybrid model never gives its weights back (MEM1). A failed child chooses nothing.
    """
    import subprocess
    import sys
    key = f"{fingerprint}-{identity_sha256[:16]}"
    if (stored := _stored(key)) is not None:
        return stored
    code = "import sys; from ironmule.numeric_choice import choose; choose(*sys.argv[1:4])"
    try:
        subprocess.run([sys.executable, "-c", code, model_id, fingerprint, identity_sha256],
                       check=True, timeout=3600)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"plan": None, "reason": f"the gate did not finish ({type(exc).__name__}); stock kept"}
    return _stored(key) or {"plan": None, "reason": "the gate stored no decision; stock kept"}


def _options(architecture: str | None, dtype: str | None, device: str | None) -> list[dict[str, Any]]:
    return [o for o in candidates(architecture, device) if not _no_op(o["plan"], dtype)]


def measure_plan(model_id: str, plan: str | None, device: str | None, load: Callable | None = None) -> dict[str, Any]:
    """One load: the gate's text likelihood and served speed for this plan.

    For stock (`plan=None`) also the architecture and checkpoint type, and the likelihood and
    speed only if some candidate on this device needs gating.
    """
    from .tune import _close_engine, _release_device_memory
    if load is None:
        from .runtime import BASELINE
        from .tune import load_engine
        engine, tokenizer = load_engine(model_id, BASELINE, compute_dtype=plan)
    else:
        engine, tokenizer = load(plan)
    try:
        out: dict[str, Any] = {}
        if plan is None:
            out.update(architecture=architecture_of(engine.model), dtype=_checkpoint_dtype(engine.model))
            options = _options(out["architecture"], out["dtype"], device)
            if all(option["verdict"] == "recommended" for option in options):
                return out
        out.update(nll=nll_per_chunk(engine, _text_ids(tokenizer)), speed_s=_speed(engine, tokenizer))
        return out
    finally:
        _close_engine(engine)
        _release_device_memory()


def _measure_in_child(model_id: str, plan: str | None, device: str | None) -> dict[str, Any]:
    """`measure_plan` in its own process: a closed engine's compiled graph can keep a hybrid
    model's weights (MEM1), and three 27B loads in one process were killed on 32 GB (NUM4)."""
    import subprocess
    import sys
    code = ("import json, sys; from ironmule.numeric_choice import measure_plan; "
            "print('@@PLAN ' + json.dumps(measure_plan(sys.argv[1], sys.argv[2] or None, sys.argv[3] or None)), "
            "flush=True)")
    proc = subprocess.run([sys.executable, "-c", code, model_id, plan or "", device or ""],
                          stdout=subprocess.PIPE, text=True, timeout=3600)
    lines = [line for line in proc.stdout.splitlines() if line.startswith("@@PLAN ")]
    if proc.returncode or not lines:
        raise RuntimeError(f"the measurement child for {plan or 'stock'} exited with status {proc.returncode}")
    return json.loads(lines[-1][len("@@PLAN "):])


# CPU3: against the float32 plan that keeps the quantised weights, i.e. the same arithmetic;
# catches a wrong expansion (errors of order 1), not precision (CPU2: 0.0145 on the M1 Max CPU).
CPU_LOGIT_BOUND = 5e-2
CPU_SPEED_TOKENS = 16


def _served_seconds(engine, tokenizer, tokens: int) -> float:
    import time
    from dataclasses import replace

    from .plans import StrictOneShotPlan
    from .service import Request, Runtime
    engine.knobs = replace(engine.knobs, **SPEED_KNOBS)
    runtime = Runtime(engine, tokenizer)
    ids = runtime.encode(SHORT_PROMPT)
    started = time.perf_counter()
    answer = runtime.serve([Request(prompt_ids=ids, max_tokens=tokens, plan=StrictOneShotPlan())])[0].tokens
    return (time.perf_counter() - started) * tokens / max(len(answer), 1)


def cpu_check(model_id: str, load: Callable | None = None) -> dict[str, Any]:
    """CPU3: the bounded check of the `dequantize` plan on a CPU, in one process.

    The perplexity gate's stock reference is itself infeasible here (CPU1), and bf16 stock on a
    CPU differs from any float32 computation by about 0.15 (CPU2). So the dequantized model's
    last-position logits on a short prompt are compared with the float32 plan that keeps the
    quantised weights (same top-1, within CPU_LOGIT_BOUND relative), and one
    CPU_SPEED_TOKENS-token served answer must be faster than stock's.
    """
    import os

    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    from .runtime import BASELINE
    from .tune import _close_engine, _release_device_memory, load_engine
    load = load or (lambda plan: load_engine(model_id, BASELINE, compute_dtype=plan))
    out: dict[str, Any] = {}
    logits = {}
    for plan in (None, "float32", "dequantize"):
        engine, tokenizer = load(plan)
        try:
            if plan is None:
                quantized = [m for _, m in engine.model.named_modules()
                             if isinstance(m, (nn.QuantizedLinear, nn.QuantizedEmbedding))]
                dense_bytes = sum(m.weight.size * 32 // m.bits * 4 for m in quantized)
                other = sum(v.nbytes for _, v in tree_flatten(engine.model.parameters()))
                memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
                out.update(architecture=architecture_of(engine.model), dtype=_checkpoint_dtype(engine.model),
                           quantized_modules=len(quantized), dense_bytes=dense_bytes + other, memory_bytes=memory)
                if not quantized or dense_bytes + other > memory // 2:
                    return out
            ids = list(tokenizer.encode(SHORT_PROMPT))
            logits[plan] = engine.model(mx.array([ids]))[0, -1].astype(mx.float32)
            mx.eval(logits[plan])
            if plan != "float32":  # the reference for the logits only
                out[f"{plan or 'stock'}_s"] = _served_seconds(engine, tokenizer, CPU_SPEED_TOKENS)
        finally:
            _close_engine(engine)
            _release_device_memory()
    def relative(a, b):
        return float((mx.max(mx.abs(a - b)) / mx.max(mx.abs(b))).item())
    out["logit_rel_error"] = relative(logits["dequantize"], logits["float32"])
    out["logit_rel_error_vs_stock"] = relative(logits["dequantize"], logits[None])
    out["top1_equal"] = int(mx.argmax(logits["dequantize"]).item()) == int(mx.argmax(logits["float32"]).item())
    return out


def _cpu_check_in_child(model_id: str) -> dict[str, Any]:
    import subprocess
    import sys
    code = ("import json, sys; from ironmule.numeric_choice import cpu_check; "
            "print('@@CPU ' + json.dumps(cpu_check(sys.argv[1])), flush=True)")
    proc = subprocess.run([sys.executable, "-c", code, model_id], stdout=subprocess.PIPE, text=True, timeout=7200)
    lines = [line for line in proc.stdout.splitlines() if line.startswith("@@CPU ")]
    if proc.returncode or not lines:
        raise RuntimeError(f"the CPU check child exited with status {proc.returncode}")
    return json.loads(lines[-1][len("@@CPU "):])


def _choose_cpu(key: str, device: str, check: dict[str, Any]) -> dict[str, Any]:
    decision: dict[str, Any] = {"plan": None, "device": device, "architecture": check.get("architecture"),
                                "checkpoint_dtype": check.get("dtype"), "tried": [dict(check, plan="dequantize")]}
    if not check.get("quantized_modules"):
        decision["reason"] = "the checkpoint is not quantised; nothing to dequantize on this CPU"
    elif "logit_rel_error" not in check:
        decision["reason"] = (f"dequantized weights would need {check['dense_bytes'] / 2**30:.1f} GB, more than "
                              f"half of this machine's {check['memory_bytes'] / 2**30:.1f} GB")
    elif (not check.get("top1_equal") or check["logit_rel_error"] >= CPU_LOGIT_BOUND
          or check["dequantize_s"] >= check["stock_s"] * 0.95):
        decision["reason"] = (f"dequantize refused: logits {check['logit_rel_error']:.2e} relative, "
                              f"{check['dequantize_s'] / check['stock_s']:.3f} of stock time")
    else:
        decision.update(plan="dequantize", reason=(
            f"CPU check passed: logits within {check['logit_rel_error']:.2e} relative of the float32 plan, "
            f"{check['dequantize_s'] / check['stock_s']:.4f} of stock time for a {CPU_SPEED_TOKENS}-token answer"))
    return _save(key, decision)


def choose(model_id: str, fingerprint: str, identity_sha256: str, *, log: Callable[[str], None] = print,
           load: Callable | None = None) -> dict[str, Any]:
    """Decide once per machine and model; returns {"plan": name or None, "reason": ..., ...}.

    Each plan is measured in its own process unless a `load` is injected (tests).
    """
    key = f"{fingerprint}-{identity_sha256[:16]}"
    if (stored := _stored(key)) is not None:
        return stored
    import mlx.core as mx
    if mx.cuda.is_available():
        info = mx.device_info()
        # Below compute capability 8: the measured table; newer GPUs are gated like any device
        # without a table.
        device = device_class(info) or f"cuda:{info.get('device_name', '')}"
    elif mx.metal.is_available():  # NUM2: Apple GPUs have no table; every plan is gated on the device
        device = f"metal:{mx.device_info().get('architecture', '')}"
    else:  # CPU2: MLX's CPU backend; only `dequantize` helps, checked within a bounded time
        import platform
        device = f"cpu:{platform.machine()}"
        log("autopilot: checking dequantized weights on this CPU ...")
        try:
            check = cpu_check(model_id, load) if load else _cpu_check_in_child(model_id)
        except Exception as exc:  # noqa: BLE001 - no check, no plan
            check = {"outcome": f"unavailable: {type(exc).__name__}"}
        return _choose_cpu(key, device, check)
    def measure(plan):
        return measure_plan(model_id, plan, device, load) if load else _measure_in_child(model_id, plan, device)
    stock = measure(None)
    architecture, dtype = stock["architecture"], stock["dtype"]
    options = _options(architecture, dtype, device)
    decision: dict[str, Any] = {"plan": None, "device": device, "architecture": architecture,
                                "checkpoint_dtype": dtype, "tried": []}
    if not options:
        decision["reason"] = "no numeric plan can change this checkpoint on this device class"
        return _save(key, decision)
    stock_s = stock.get("speed_s")
    for option in options:
        if option["verdict"] == "recommended":
            decision.update(plan=option["plan"], reason=f"recommended by the measured table "
                            f"({option['table_wall_ratio']:.3f} of stock, quality inside {QUALITY_BOUND})")
            decision["tried"].append(option)
            return _save(key, decision)
        log(f"autopilot: gating numeric plan {option['plan']} on this device ...")
        try:
            result = measure(option["plan"])
        except Exception as exc:  # noqa: BLE001 - a plan this build cannot load is simply not taken
            decision["tried"].append(dict(option, outcome=f"unavailable: {type(exc).__name__}"))
            continue
        gate = ratio_interval(stock["nll"], result["nll"])
        ratios = _speed_ratios(stock_s, result["speed_s"])
        faster = _faster(ratios)
        decision["tried"].append(dict(option, gate=gate, stock_s=stock_s, plan_s=result["speed_s"],
                                      speed_ratios=ratios, faster=faster, passes=faster and gate["passes"]))
    passed = [t for t in decision["tried"] if t.get("passes")]
    if not passed:
        decision["reason"] = "no plan passed both the quality gate and the speed check here"
        return _save(key, decision)
    # Every candidate is gated; the fastest pass over both workloads wins.
    best = min(passed, key=lambda t: sum(t["speed_ratios"].values()))
    gate, ratios = best["gate"], best["speed_ratios"]
    decision.update(plan=best["plan"], reason=(
        f"on-device gate passed: perplexity ratio {gate['ratio']:.5f} "
        f"[{gate['interval'][0]:.5f}; {gate['interval'][1]:.5f}] and "
        + ", ".join(f"{ratios[name]:.3f} of stock time ({name} prompt)" for name in sorted(ratios))))
    return _save(key, decision)


def _save(key: str, decision: dict[str, Any]) -> dict[str, Any]:
    path = _store()
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        data = {}
    data[key] = decision
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, sort_keys=True, default=str) + "\n")
    return decision
