"""AG4: the agent chooses its plan and engine on this machine, once per hardware and model (BACKLOG AG4).

The owner decided that under `--agent on` the agent sets up by itself what earlier runs found by hand, including
output-changing plans that pass its own gate on this machine. Two steps:

1. NUM1 (`numeric_choice.choose`) picks the numeric plan, if any (`native`, `float16`, `float32`): each candidate is
   gated on the device by the perplexity gate and a served speed check. Agent mode implies the consent NUM1 asks for.
2. On that base path, a stacked engine (KB1's dense plan for Qwen 3 bf16, KB2's Gemma plan for 4-bit Gemma 3) is
   measured where its `refuse` lets the model through. A stacked plan only changes the decode path, so its gate runs
   there: the reference text teacher-forced through the base path's own decode and through the plan, eight rows
   stacked, against `numeric_plans.QUALITY_BOUND`. Tokens per second are measured at every width from 1 to the
   pool's slots (AG5), and the time to the first token on a short and a long prompt.

Decision rule (fixed in BACKLOG AG4 before data; amended by AG5 before AG5-G's data): the stacked engine serves if it
passed its load probe and its gate, makes more tokens per second at its best width (the most tokens per second on the
measured curve; it then serves at that width) than the base path does for one request, keeps at least `QUIET_FLOOR`
of the base path's single-request tokens per second, and its first token is at most `TTFT_SLACK` slower. Otherwise
the base path serves. The decision and every number behind it are stored per hardware fingerprint and model identity, and the
product shows them; nothing is chosen silently.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

from .hw import STORE

WIDTH = 8
#: Every lever the agent knows, built or not (BACKLOG AG4, HQ1). A built lever is measured and gated on this machine
#: by `choose_setup`; a planned one is listed so the agent's status says what is known and not yet available. When one
#: is built it joins the same gate and rule.
LEVERS: tuple[dict[str, str], ...] = (
    {"name": "numeric_plan", "status": "built", "backlog": "NUM1",
     "what": "compute the weights in another type (native, float16, float32, dequantize) where it is faster here"},
    {"name": "stacked_engine", "status": "built", "backlog": "KB1, KB2",
     "what": "up to eight requests in one decode step, each row's answer independent of the others"},
    {"name": "cpu_blas", "status": "built", "backlog": "CPU6",
     "what": "on a CPU, MLX's float32 products on an optimised BLAS (OpenBLAS) instead of the wheel's reference BLAS"},
    {"name": "routed_paths", "status": "built", "backlog": "AG4 step 2a",
     "what": "the single path and the stacked engine side by side, each request routed where it finishes first"},
    {"name": "tensor_core_rows", "status": "planned", "backlog": "HQ1 TC1",
     "what": "the stacked rows on fp16 or int8 tensor cores, so a further row costs almost nothing (HyperQwen)"},
    {"name": "context_speculation", "status": "planned", "backlog": "HQ1 SP1",
     "what": "drafts copied from the request's own context and verified by the model, the window size chosen per "
             "round by a policy learned online from measured round times and acceptance (HyperQwen, Strata)"},
    {"name": "chunked_prefill", "status": "planned", "backlog": "HQ1 CP1",
     "what": "a newcomer's prefill in chunks between decode steps, the chunk size learned (vLLM)"},
    {"name": "paged_kv", "status": "planned", "backlog": "HQ1 PK1",
     "what": "keys and values in blocks allocated as a request grows, more requests at the same precision (vLLM)"},
    {"name": "kv_cache_8bit", "status": "planned", "backlog": "HQ1 KV8",
     "what": "8-bit keys and values at run time, twice the slots on the same card; the weights are untouched "
             "(HyperQwen)"},
    {"name": "expert_cache", "status": "planned", "backlog": "HQ1 EC1",
     "what": "for a mixture-of-experts model, the experts this traffic uses most kept on the card, learned online "
             "(Strata)"},
    {"name": "tpu_path", "status": "planned", "backlog": "TP1",
     "what": "a TPU as one more device: a JAX decode path on the model's own bf16 weights, measured and chosen "
             "like any other path (Kaggle TPU v5e; TP0 probes it first)"},
)
#: The owner's boundary (BACKLOG HQ1): no lever changes or shrinks the model's weights.
MODEL_UNCHANGED = True
QUIET_FLOOR = 0.9       # the stacked engine's single-request tokens per second over the base path's, at least
TTFT_SLACK = 1.1        # its time to the first token over the base path's, at most
SPEED_STEPS = 32
FIRST_SAMPLES = 5  # base-path prefills behind each first-token time (their median); the plan's are its slots'
MEASURE_TIMEOUT_S = 3600  # a stacked measurement past this is a failure: KB2's probe gated Gemma 3 12B in 26 min
SHORT_PROMPT = "Explain in detail how rain forms, step by step."
#: CPU6 step 3: a CPU setup stored without these decisions (the BLAS, the CPU stacked plan) is measured again once
CPU_SETUP_VERSION = 1
BLAS_SPEED_RATIO = 0.95  # OpenBLAS's served answer at most this share of the stock BLAS's time (CPU3's threshold)


def _store() -> Path:
    return STORE / "plan_choice.json"


def _stored(key: str) -> dict[str, Any] | None:
    """The setup stored for `key`, or None when there is none or its gates ran under another gate version."""
    from .numeric_choice import GATE_VERSION

    try:
        data = json.loads(_store().read_text())
    except (OSError, ValueError):
        return None
    stored = data.get(key) if isinstance(data, dict) else None  # a store of another shape holds nothing usable
    return stored if isinstance(stored, dict) and stored.get("gate_version") == GATE_VERSION else None


def _save(key: str, decision: dict[str, Any]) -> dict[str, Any]:
    """Store `decision` under `key` and return it as stored; a store that cannot be written (a full or read-only
    disk) loses only the memory of it, and the next start measures again."""
    path = _store()
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    from .numeric_choice import GATE_VERSION

    data[key] = {**decision, "decided_unix": time.time(), "gate_version": GATE_VERSION}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1, sort_keys=True, default=str))
        tmp.replace(path)
    except OSError:
        pass
    return data[key]


def on_cpu() -> bool:
    """MLX computes on the CPU here: neither CUDA nor Metal is available."""
    import mlx.core as mx

    return not mx.cuda.is_available() and not mx.metal.is_available()


def stacked_refusal(model: Any, device_info: dict[str, Any] | None) -> str | None:
    """Why no stacked plan can serve this model on this device, found before anything is allocated or timed; None
    where one may (its install still probes the weights and may refuse)."""
    from . import dense_gemma, dense_plan
    from .numeric_plans import CUDA_PRE_AMPERE, device_class

    if device_info is None and on_cpu():  # CPU6: MLX's CPU device has its own stacked plan
        from . import dense_cpu

        return dense_cpu.refuse(model)
    if device_class(device_info) != CUDA_PRE_AMPERE:
        return "the stacked engines exist for CUDA GPUs below compute capability 8 only"
    if dense_gemma.refuse(model, native=False) is None:
        return dense_gemma.refuse(model)  # KB2's plan also needs the native row kernel on every linear
    return dense_plan.refuse(model)


def stacked_plan(model: Any, device_info: dict[str, Any] | None, *, slots: int = WIDTH,
                 capacity: int | None = None) -> tuple[Any, dict[str, Any]]:
    """The stacked plan this model takes (`ValueError` with the reason where none does). The base path's plan must
    already be on the model: KB2's plan needs the native row kernel, which NUM1 installs as `native`."""
    from . import dense_gemma, dense_plan

    if device_info is None and on_cpu():
        from . import dense_cpu

        return dense_cpu.install(model, slots=slots, capacity=capacity)
    if dense_gemma.refuse(model, native=False) is None:
        return dense_gemma.install(model, device_info, slots=slots, capacity=capacity)
    return dense_plan.install(model, device_info, slots=slots, capacity=capacity)


def _decode_nll(model: Any, ids: list[int], lead: list[int] | None = None) -> float:
    """The base path's own decode, teacher-forced: prefill the lead (the gate's BOS, if any) and the first token,
    then one step per token."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    logits = model(mx.array((lead or []) + ids[:1])[None], cache=cache)[0, -1].astype(mx.float32)
    total = 0.0
    for t in range(1, len(ids)):
        total += float((mx.logsumexp(logits) - logits[ids[t]]).item())
        if t + 1 < len(ids):
            logits = model(mx.array([[ids[t]]]), cache=cache)[0, -1].astype(mx.float32)
    return total / (len(ids) - 1)


def _plan_nll(plan: Any, chunks: list[list[int]], leads: list[list[int]] | None = None) -> list[float]:
    """The plan teacher-forced, `plan.slots` chunks stacked, each after its lead (the gate's BOS, if any); the
    first prediction is the plan's own prefill head."""
    import mlx.core as mx

    leads = leads or [[] for _ in chunks]
    means = []
    for start in range(0, len(chunks), plan.slots):
        group = chunks[start:start + plan.slots]
        offsets = [len(lead) for lead in leads[start:start + plan.slots]]
        sums, logits_rows = [0.0] * len(group), []
        for slot, ids in enumerate(group):
            new_cache = getattr(plan, "new_cache", None)
            if new_cache is not None:
                cache = new_cache()
            else:
                from mlx_lm.models.cache import make_prompt_cache
                cache = make_prompt_cache(plan.model)
            logits_rows.append(plan.prefill(mx.array(leads[start + slot] + ids[:1])[None], cache)[0])
            plan.write_prefill(slot, [c.state for c in cache])
        slots = mx.arange(len(group), dtype=mx.int32)
        for t in range(1, len(group[0])):
            for r, ids in enumerate(group):
                row = logits_rows[r]
                sums[r] += float((mx.logsumexp(row) - row[ids[t]]).item())
            if t + 1 < len(group[0]):
                logits = plan.forward(mx.array([ids[t] for ids in group]), slots,
                                      mx.array([offset + t for offset in offsets], dtype=mx.int32))
                mx.eval(logits)
                logits_rows = [logits[r] for r in range(len(group))]
        means.extend(total / (len(group[0]) - 1) for total in sums)
    return means


def _release_memory() -> None:
    """Return freed buffers to the device: before the stacked plan allocates its pool, and between stages."""
    import gc

    gc.collect()
    try:
        import mlx.core as mx

        getattr(mx, "synchronize", lambda: None)()  # frees queued on the stream land before the cache is cleared
        mx.clear_cache()
    except Exception:  # noqa: BLE001 - no MLX device to release; nothing held
        pass


def _memory_now() -> str:
    """MLX's allocator as one line: active, cached and peak gigabytes (empty where MLX does not report them)."""
    try:
        import mlx.core as mx

        return ", ".join(f"{name} {getattr(mx, f'get_{name}_memory')() / 1e9:.2f} GB"
                         for name in ("active", "cache", "peak"))
    except Exception:  # noqa: BLE001 - only a log line
        return ""


def _again_if_out_of_memory(step: Callable[[], Any], retried: list[str], name: str) -> Any:
    """`step()`, and once more after returning every freed buffer if the device ran out of memory (AG5-G run 3:
    the long prompt's first prefill on the stacked plan failed in `cudaMallocAsync` after the gate had passed).
    A second failure is raised, so the stacked engine is then not chosen."""
    try:
        return step()
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        retried.append(name)
        _release_memory()
        return step()


def _base_speed(model: Any, ids: list[int], steps: int, samples: list[float] | None = None) -> tuple[float, float]:
    """(seconds to the first token, seconds per decode step) for one request on the base path. The first token is
    the median of `FIRST_SAMPLES` prefills, each from a fresh cache (`samples` gets them all): AG5-G runs 4 and 5
    timed it once, and the same prefill read 0.742 s in one session and 0.685 s in the next."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    mx.eval(model(mx.array(ids)[None], cache=make_prompt_cache(model)))  # warm: compile outside the clock (AG4-Q 2)
    firsts = []
    for _ in range(FIRST_SAMPLES):
        cache = make_prompt_cache(model)
        started = time.perf_counter()
        token = int(mx.argmax(model(mx.array(ids)[None], cache=cache)[0, -1]).item())
        firsts.append(time.perf_counter() - started)
    if samples is not None:
        samples.extend(firsts)
    first = statistics.median(firsts)
    times = []
    for _ in range(steps):
        started = time.perf_counter()
        token = int(mx.argmax(model(mx.array([[token]]), cache=cache)[0, -1]).item())
        times.append(time.perf_counter() - started)
    return first, statistics.median(times[2:] or times)


def _plan_speed(plan: Any, ids: list[int], steps: int, samples: list[float] | None = None
                ) -> tuple[float, dict[int, float]]:
    """(seconds to the first token, {width: seconds per stacked step}) on the plan, every slot holding `ids`. The
    first token is the median of the slots' prefills (`samples` gets them all); AG5-G runs 4 and 5 kept the first
    slot's alone."""
    import mlx.core as mx

    new_cache = getattr(plan, "new_cache", None)
    firsts = []
    warm = new_cache() if new_cache is not None else None
    if warm is None:
        from mlx_lm.models.cache import make_prompt_cache
        warm = make_prompt_cache(plan.model)
    mx.eval(plan.prefill(mx.array(ids)[None], warm))  # warm: compile outside the clock
    for slot in range(plan.slots):
        if new_cache is not None:
            cache = new_cache()
        else:
            from mlx_lm.models.cache import make_prompt_cache
            cache = make_prompt_cache(plan.model)
        started = time.perf_counter()
        mx.eval(mx.argmax(plan.prefill(mx.array(ids)[None], cache), axis=-1))
        firsts.append(time.perf_counter() - started)
        plan.write_prefill(slot, [c.state for c in cache])
    widths = {}
    for width in range(1, plan.slots + 1):  # AG5: the whole curve; Gemma's plan is best at 6 and falls off at 8
        times = []
        for t in range(steps):
            started = time.perf_counter()
            mx.eval(mx.argmax(plan.forward(mx.array([(7 * r + t) % 1000 + 10 for r in range(width)]),
                                           mx.arange(width, dtype=mx.int32),
                                           mx.array([len(ids) + t] * width, dtype=mx.int32)), axis=-1))
            times.append(time.perf_counter() - started)
        widths[width] = statistics.median(times[2:] or times)
    if samples is not None:
        samples.extend(firsts)
    return statistics.median(firsts), widths


def measure_stacked(model: Any, tokenizer: Any, device_info: dict[str, Any] | None, *,
                    chunks: int | None = None, size: int | None = None, steps: int = SPEED_STEPS,
                    long_prompt: str | None = None,
                    make_plan: Callable[[Any, dict[str, Any] | None], tuple[Any, dict[str, Any]]] | None = None,
                    log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Everything AG4's rule reads about the stacked engine on this model, which already carries its base plan.

    Everything on the base path runs before the stacked plan exists: its timing and the gate's reference. Its
    prefill runs the head over the whole prompt through a float16 copy of the head (about 2 GB for Gemma 3 12B),
    which fits beside the weights but not beside the plan's pool as well. AG5-G run 1 ran out of memory timing it
    beside the pool; run 2, with the gate's reference beside the pool, did not finish within two hours, where KB2's
    probe, reference first, gated the same plan in 26 minutes. `log` gets each stage with the seconds so far."""
    from .numeric_choice import CHUNK_TOKENS, CHUNKS, gate_chunks, ratio_interval
    from .tune import DEFAULT_PROMPT

    started = time.perf_counter()
    retried: list[str] = []

    def note(stage: str) -> None:
        if log is not None:
            memory = _memory_now()
            log(f"agent: stacked measurement: {stage} ({time.perf_counter() - started:.0f} s"
                + (f"; {memory})" if memory else ")"))

    if make_plan is None and (reason := stacked_refusal(model, device_info)) is not None:
        return {"eligible": False, "reason": reason}  # nothing timed for a model no stacked plan serves
    prompts = (("short", SHORT_PROMPT), ("long", long_prompt or DEFAULT_PROMPT))
    base_firsts: dict[str, list[float]] = {name: [] for name, _ in prompts}
    base = {name: _base_speed(model, list(tokenizer.encode(prompt)), steps, base_firsts[name])
            for name, prompt in prompts}
    note("base path timed")
    items = gate_chunks(tokenizer, chunks or CHUNKS, size or CHUNK_TOKENS)
    leads, pieces = [lead for lead, _ in items], [piece for _, piece in items]
    reference = _again_if_out_of_memory(lambda: [_decode_nll(model, piece, lead) for lead, piece in items],
                                        retried, "gate reference")
    note(f"gate reference on the base path, {len(items)} chunks")
    _release_memory()
    try:
        plan, record = (make_plan or stacked_plan)(model, device_info)
    except ValueError as exc:
        return {"eligible": False, "reason": str(exc)}
    note("stacked plan installed")
    candidate = _again_if_out_of_memory(lambda: _plan_nll(plan, pieces, leads), retried, "gate")
    note("gate on the stacked plan")
    out: dict[str, Any] = {"eligible": True, "install": record, "gate": ratio_interval(reference, candidate),
                           "memory": _memory(model, plan, device_info)}
    speed = {}
    for name, prompt in prompts:
        prompt_ids = list(tokenizer.encode(prompt))
        base_first, base_step = base[name]
        _release_memory()
        plan_firsts: list[float] = []
        plan_first, plan_steps = _again_if_out_of_memory(
            lambda: _plan_speed(plan, prompt_ids, steps, plan_firsts), retried, f"speed, {name} prompt")
        speed[name] = {"base_first_s": base_first, "base_step_s": base_step, "plan_first_s": plan_first,
                       "plan_step_s": {str(w): s for w, s in plan_steps.items()},
                       "base_first_samples_s": base_firsts[name], "plan_first_samples_s": plan_firsts[-plan.slots:]}
        note(f"stacked steps timed at every width, {name} prompt")
    out["speed"] = speed
    out["retried_out_of_memory"] = retried
    return out


ROUTE_RESERVE = 3 * 1024 ** 3  # two workers' working memory beside two copies of the weights and the pool
ROUTE_HEADROOM = 0.9


def _memory(model: Any, plan: Any, device_info: dict[str, Any] | None) -> dict[str, Any]:
    """Whether two serving paths fit this card at once: two copies of the weights, the stacked pool and
    `ROUTE_RESERVE`, in `ROUTE_HEADROOM` of the card (unknown when the card reports no size)."""
    from mlx.utils import tree_flatten

    info = device_info or {}
    total = next((info[k] for k in ("total_memory", "memory_size") if isinstance(info.get(k), int) and info[k] > 0),
                 None)
    weights = sum(v.nbytes for _, v in tree_flatten(model.parameters()))
    pool = sum(a.nbytes for a in (*getattr(plan, "keys", []), *getattr(plan, "values", [])))
    need = 2 * weights + pool + ROUTE_RESERVE
    return {"weights": weights, "pool": pool, "total": total,
            "two_paths_fit": None if total is None else need <= ROUTE_HEADROOM * total}


def decide(stacked: dict[str, Any]) -> dict[str, Any]:
    """AG4's fixed rule over `measure_stacked`'s result: {"stacked": bool, "reason": ..., "tokens_per_second": ...}."""
    if not stacked.get("eligible"):
        return {"stacked": False, "reason": f"no stacked engine for this model: {stacked.get('reason')}"}
    if not stacked["gate"]["passes"]:
        return {"stacked": False, "reason": f"the stacked engine failed the quality gate "
                f"(ratio {stacked['gate']['ratio']:.5f}, bound {stacked['gate']['interval'][1]:.5f})"}
    rows = list(stacked["speed"].values())
    curve = {w: statistics.fmean(int(w) / r["plan_step_s"][w] for r in rows) for w in rows[0]["plan_step_s"]
             if all(w in r["plan_step_s"] for r in rows)}
    width = max(curve, key=lambda w: (curve[w], -int(w)))  # AG5: the best width; the narrower one on a tie
    base_tps = statistics.fmean(1 / r["base_step_s"] for r in rows)
    one_tps = statistics.fmean(1 / r["plan_step_s"]["1"] for r in rows)
    wide_tps = curve[width]
    ttft = max(r["plan_first_s"] / r["base_first_s"] for r in rows)
    tps = {"base_one": base_tps, "stacked_one": one_tps, f"stacked_{width}": wide_tps,
           "by_width": {w: curve[w] for w in sorted(curve, key=int)}}
    if wide_tps <= base_tps:
        return {"stacked": False, "tokens_per_second": tps, "width": int(width),
                "reason": "stacking makes no more tokens per second here at any width"}
    # AG4 step 2 (amended after AG4-Q run 2): a stacked engine that wins under load but not alone serves beside the
    # single path, and the router sends each request where it finishes first, if both fit the card.
    fits = (stacked.get("memory") or {}).get("two_paths_fit") is True
    if one_tps < QUIET_FLOOR * base_tps:
        why = f"one request alone would run at {one_tps / base_tps:.2f}x, below {QUIET_FLOOR}"
        return {"stacked": False, "route": fits, "tokens_per_second": tps, "width": int(width),
                "reason": why + ("; both paths serve and the agent routes each request" if fits else "")}
    if ttft > TTFT_SLACK:
        why = f"the first token would come {ttft:.2f}x later, above {TTFT_SLACK}"
        return {"stacked": False, "route": fits, "tokens_per_second": tps, "width": int(width),
                "reason": why + ("; both paths serve and the agent routes each request" if fits else "")}
    return {"stacked": True, "tokens_per_second": tps, "ttft_ratio": ttft, "width": int(width),
            "reason": f"{wide_tps / base_tps:.2f}x the tokens per second at width {width}, gate ratio "
                      f"{stacked['gate']['ratio']:.5f}, first token {ttft:.2f}x"}


def measure_base(model: Any, tokenizer: Any, *, steps: int = SPEED_STEPS, long_prompt: str | None = None
                 ) -> dict[str, dict[str, float]]:
    """The served path's own single-request speed: seconds to the first token and per decode step, short and long."""
    from .tune import DEFAULT_PROMPT

    out = {}
    for name, prompt in (("short", SHORT_PROMPT), ("long", long_prompt or DEFAULT_PROMPT)):
        first, step = _base_speed(model, list(tokenizer.encode(prompt)), steps)
        out[name] = {"base_first_s": first, "base_step_s": step}
    return out


def stacked_base(numeric_plan: str | None, cpu: bool = False) -> str | None:
    """The plan a stacked engine runs on: KB2's Gemma plan on `native`, KB1's dense plan on the checkpoint's own
    bf16. A float plan NUM1 chose (float16, float32) is not what either was built on, so it is compared against,
    not stacked on (AG4-Q run 1: the dense plan refused float16 weights). On a CPU (CPU6) the plan widens every
    weight to float32 itself, so it is measured on the plan that serves (`dequantize`, `float32` or none)."""
    if cpu:
        return numeric_plan
    return "native" if numeric_plan == "native" else None


def against(stacked: dict[str, Any], base: dict[str, dict[str, float]]) -> dict[str, Any]:
    """`stacked` with its speed rows compared against `base` (the path that serves without stacking); the stacked
    engine's own base path's numbers stay beside them as `stacked_base_*`."""
    if not stacked.get("eligible"):
        return stacked
    if any(name not in base for name in stacked["speed"]):  # its measurement failed: nothing to compare against
        return {"eligible": False, "technical": True,
                "reason": f"the path it competes with was not measured: {base.get('reason')}"}
    speed = {}
    for name, row in stacked["speed"].items():
        speed[name] = {**row, "stacked_base_first_s": row["base_first_s"], "stacked_base_step_s": row["base_step_s"],
                       **base[name]}
    return {**stacked, "speed": speed}


def measure_in_process(model_id: str, compute_dtype: str | None, base_only: bool = False) -> dict[str, Any]:
    """Load the model on its base path (the numeric plan NUM1 chose) and measure the stacked engine."""
    import mlx.core as mx

    from .runtime import BASELINE
    from .tune import _close_engine, _release_device_memory, load_engine

    import sys

    engine, tokenizer = load_engine(model_id, BASELINE, compute_dtype=compute_dtype)
    try:
        if base_only:
            return measure_base(engine.model, tokenizer)
        return measure_stacked(engine.model, tokenizer, mx.device_info() if mx.cuda.is_available() else None,
                               log=lambda line: print(line, file=sys.stderr, flush=True))
    finally:
        _close_engine(engine)
        _release_device_memory()


def _measure_in_child(model_id: str, compute_dtype: str | None, base_only: bool = False,
                      env: dict[str, str] | None = None) -> dict[str, Any]:
    """`measure_in_process` in a child process; `env` adds variables (CPU6: the chosen BLAS's)."""
    import os
    import subprocess
    import sys

    code = ("import json, sys; from ironmule.plan_choice import measure_in_process; "
            "print('@@STACKED ' + json.dumps(measure_in_process(sys.argv[1], sys.argv[2] or None, sys.argv[3] == '1'), "
            "default=str), flush=True)")
    try:
        proc = subprocess.run([sys.executable, "-c", code, model_id, compute_dtype or "", "1" if base_only else "0"],
                              stdout=subprocess.PIPE, text=True, timeout=MEASURE_TIMEOUT_S,
                              env={**os.environ, **env} if env else None)
    except subprocess.TimeoutExpired:  # the child is killed; the agent keeps the base path and says why
        return {"eligible": False, "technical": True,
                "reason": f"the measurement did not finish within {MEASURE_TIMEOUT_S} s"}
    lines = [line for line in proc.stdout.splitlines() if line.startswith("@@STACKED ")]
    if proc.returncode or not lines:
        return {"eligible": False, "technical": True,
                "reason": f"the measurement child exited with status {proc.returncode}"}
    try:
        measured = json.loads(lines[-1][len("@@STACKED "):])
    except ValueError:
        measured = None
    if not isinstance(measured, dict):
        return {"eligible": False, "technical": True, "reason": "the measurement child returned no readable result"}
    return measured


BLAS_CHILD = r"""
import json, sys
import mlx.core as mx
from ironmule import cpu_blas
from ironmule.numeric_choice import CPU_SPEED_TOKENS, SHORT_PROMPT, _served_seconds
from ironmule.runtime import BASELINE
from ironmule.tune import _close_engine, load_engine
engine, tokenizer = load_engine(sys.argv[1], BASELINE, compute_dtype=sys.argv[2] or None)
try:
    ids = list(tokenizer.encode(SHORT_PROMPT))
    logits = engine.model(mx.array([ids]))[0, -1].astype(mx.float32)
    mx.eval(logits)
    seconds = _served_seconds(engine, tokenizer, CPU_SPEED_TOKENS)
    print("@@BLAS " + json.dumps({"serving": cpu_blas.active()["serving"], "logits": logits.tolist(),
                                  "seconds": seconds}), flush=True)
finally:
    _close_engine(engine)
"""


def _blas_in_child(model_id: str, compute_dtype: str | None, env: dict[str, str]) -> dict[str, Any]:
    """The base path's short-prompt logits and one served answer's seconds, on the BLAS `env` names."""
    import os
    import subprocess
    import sys

    proc = subprocess.run([sys.executable, "-c", BLAS_CHILD, model_id, compute_dtype or ""], stdout=subprocess.PIPE,
                          text=True, timeout=MEASURE_TIMEOUT_S, env={**os.environ, **env})
    lines = [line for line in proc.stdout.splitlines() if line.startswith("@@BLAS ")]
    if proc.returncode or not lines:
        raise RuntimeError(f"the BLAS measurement child exited with status {proc.returncode}")
    return json.loads(lines[-1][len("@@BLAS "):])


def choose_blas(model_id: str, compute_dtype: str | None, *, environment: Callable[[], tuple] | None = None,
                measure: Callable[..., dict[str, Any]] | None = None) -> dict[str, Any]:
    """CPU6 step 3: OpenBLAS or the stock BLAS for the base path, by the rule BACKLOG CPU6 fixed before its data.
    {"cpu_blas": "openblas" | None, "reason", ...}; `env` holds the variables a child needs for the choice."""
    from .numeric_choice import CPU_LOGIT_BOUND

    if environment is None:
        from .cpu_blas import environment
    variables, record = environment()
    if not variables:
        return {"cpu_blas": None, "env": {}, "check": record,
                "reason": f"the optimised BLAS is not available here: {record.get('reason')}"}
    measure = measure or _blas_in_child
    stock, fast = measure(model_id, compute_dtype, {}), measure(model_id, compute_dtype, variables)
    import math

    gap = max(abs(a - b) for a, b in zip(fast["logits"], stock["logits"]))
    scale = max(abs(b) for b in stock["logits"]) or 1.0
    error = gap / scale
    top1 = max(range(len(fast["logits"])), key=fast["logits"].__getitem__) == max(
        range(len(stock["logits"])), key=stock["logits"].__getitem__)
    ratio = fast["seconds"] / stock["seconds"] if stock["seconds"] > 0 else math.inf
    out = {"check": record, "stock_s": stock["seconds"], "openblas_s": fast["seconds"], "speed_ratio": ratio,
           "logit_rel_error": error, "top1_equal": top1, "serving": [stock["serving"], fast["serving"]]}
    if fast["serving"] != "openblas":
        return {**out, "cpu_blas": None, "env": {}, "reason": "the optimised BLAS did not serve in its child"}
    if not top1 or error >= CPU_LOGIT_BOUND or ratio > BLAS_SPEED_RATIO:
        return {**out, "cpu_blas": None, "env": {},
                "reason": f"OpenBLAS refused: logits {error:.2e} relative, top token {'equal' if top1 else 'differs'}, "
                          f"{ratio:.3f} of the stock BLAS's time"}
    return {**out, "cpu_blas": "openblas", "env": variables,
            "reason": f"OpenBLAS: logits within {error:.2e} relative of the stock BLAS, same top token, {ratio:.3f} "
                      f"of its time for a 16-token answer"}


def levers(setup: dict[str, Any]) -> list[dict[str, str]]:
    """Each lever with what happened to it in this setup: used, measured and not chosen (and why), not eligible here,
    or not built yet."""
    stacked, decision = setup.get("stacked") or {}, setup.get("decision") or {}
    engine, plan = setup.get("engine"), setup.get("compute_dtype")
    numeric_reason = (setup.get("numeric") or {}).get("reason")
    if engine == "dense":
        stacked_here, routed_here = "used", "not needed: the stacked engine wins everywhere"
    elif engine == "route":
        stacked_here, routed_here = "used beside the single path", "used"
    else:
        stacked_here = (f"measured, not chosen: {decision.get('reason')}" if stacked.get("eligible")
                        else f"not eligible here: {stacked.get('reason')}")
        routed_here = f"not taken: {decision.get('reason')}"
    blas = setup.get("blas")
    blas_here = ("not a CPU here" if blas is None else "used: OpenBLAS" if setup.get("cpu_blas")
                 else f"measured, not chosen: {blas.get('reason')}")
    state = {"numeric_plan": f"used: {plan}" if plan else f"measured, none chosen: {numeric_reason}",
             "stacked_engine": stacked_here, "routed_paths": routed_here, "cpu_blas": blas_here}
    return [{**lever, "here": state.get(lever["name"], "not built yet")} for lever in LEVERS]


def choose_setup(model_id: str, fingerprint: str, identity_sha256: str, *, log: Callable[[str], None] = print,
                 numeric: Callable[..., dict[str, Any]] | None = None,
                 measure: Callable[..., dict[str, Any]] | None = None, cpu: bool | None = None,
                 blas: Callable[..., dict[str, Any]] | None = None) -> dict[str, Any]:
    """The agent's setup for this machine and model: {"compute_dtype", "engine": "dense" | "route" | "reference",
    "cpu_blas", "reason", "numeric", "stacked", "decision"}; made once and stored. On a CPU (`cpu`, found here when
    None) it also chooses the BLAS and measures CPU6's stacked plan on the chosen BLAS."""
    key = f"{fingerprint}-{identity_sha256[:16]}"
    if cpu is None:  # found here for the real measurements; a caller injecting its own says so itself
        cpu = on_cpu() if measure is None else False
    stored = _stored(key)
    if stored is not None and not (cpu and stored.get("cpu_setup") != CPU_SETUP_VERSION):
        return stored  # a CPU setup from before CPU6 step 3 is measured again once
    if numeric is None:
        from .numeric_choice import choose_out_of_process as numeric
    try:
        base = numeric(model_id, fingerprint, identity_sha256, log=log)
    except Exception as exc:  # noqa: BLE001 - a failed gate keeps the checkpoint's dtype, as a refused plan does
        base = {"plan": None, "technical": True,
                "reason": f"the numeric gate failed ({type(exc).__name__}: {exc})"[:300]}
    compute_dtype = base.get("plan")
    log(f"agent: numeric plan {compute_dtype or 'none (checkpoint dtype)'}: {base.get('reason')}")
    blas_choice, env = None, {}
    if cpu:
        log("agent: checking an optimised BLAS on this CPU ...")
        try:
            blas_choice = (blas or choose_blas)(model_id, compute_dtype)
        except Exception as exc:  # noqa: BLE001 - the stock BLAS serves, with the reason
            blas_choice = {"cpu_blas": None, "env": {}, "technical": True,
                           "reason": f"the BLAS measurement failed ({type(exc).__name__}: {exc})"[:300]}
        env = blas_choice.pop("env", {}) or {}
        log(f"agent: BLAS {blas_choice.get('cpu_blas') or 'stock'}: {blas_choice.get('reason')}")
    if measure is None:
        measure = (lambda m, c, base_only=False: _measure_in_child(m, c, base_only, env=env)) if env else _measure_in_child
    on = stacked_base(compute_dtype, cpu=cpu)
    log(f"agent: measuring the stacked engine on this {'CPU' if cpu else 'card'} (on {on or 'the checkpoint dtype'}) ...")
    try:
        stacked = measure(model_id, on)
        if stacked.get("eligible") and on != compute_dtype:  # compare against the path that would serve instead
            log(f"agent: measuring {compute_dtype or 'the checkpoint dtype'} alone, the path it competes with ...")
            stacked = against(stacked, measure(model_id, compute_dtype, base_only=True))
        decision = decide(stacked)
    except Exception as exc:  # noqa: BLE001 - a failed measurement or rule keeps the base path, with the reason
        stacked = {"eligible": False, "technical": True,
                   "reason": f"the stacked engine's measurement failed ({type(exc).__name__}: {exc})"[:300]}
        decision = decide(stacked)
    if cpu and decision.get("route"):  # two workers would share the cores (MG1 item 5): not offered on a CPU
        decision = {**decision, "route": False, "reason": decision["reason"].split("; both paths serve")[0]
                    + "; routing is not offered on a CPU, where both paths would share the cores"}
    log(f"agent: {'stacked engine' if decision['stacked'] else 'no stacked engine'}: {decision['reason']}")
    engine = "dense" if decision["stacked"] else "route" if decision.get("route") else "reference"
    setup = {"compute_dtype": compute_dtype, "engine": engine,
             "reason": decision["reason"], "numeric": base, "stacked": stacked, "decision": decision,
             "width": decision.get("width") if engine in ("dense", "route") else None}  # AG5: serves at this width
    if cpu:
        setup.update(cpu_blas=(blas_choice or {}).get("cpu_blas"), blas=blas_choice, cpu_setup=CPU_SETUP_VERSION)
    setup["levers"] = levers(setup)
    if base.get("technical") or stacked.get("technical") or (blas_choice or {}).get("technical"):
        # A measurement that failed for a technical reason (a child that crashed or timed out, a card out of
        # memory) serves the base path now and is measured again on the next start. Stored, it was reused for
        # good after the cause had gone (review of 2026-11-08). A refusal by a rule or a gate is stored.
        return {**setup, "measurement_failed": True}
    return _save(key, setup)
