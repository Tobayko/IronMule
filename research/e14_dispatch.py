"""E14: is the remaining decode latency fixed dispatch overhead?

Preregistered at commit e1c29f0, research/raw/E14_preregistration.md
(SHA-256 13f6d3589cfc5d83136f275fb63e0382f3b068ed8c1408abd460ac4f59a400a2).

Nothing in ironmule/ is modified and no scheduler is built. MLX exposes no kernel or
dispatch counter, so any kernel count here stays INFERRED and is labelled so.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import mlx.core as mx

from ironmule import bench
from ironmule.runtime import FixedKVCache, Knobs, _leaves, _project, _trunk
from ironmule.tune import DEFAULT_MODEL, gpu_busy, load_engine

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
SQUAD = HERE / "data" / "squad-dev-v1.1.json"

# --- frozen constants --------------------------------------------------------
CONTEXT_TOKENS = 1024
SEQUENCES = 8
WIDTHS = [1, 2, 4, 8]
BATCHES = [1, 2, 4, 8]
SEQ_STEPS = 8
CONTROL_K = [0, 64, 128, 256]
WARMUP, REPEATS = 2, 7
WEIGHT_BYTES = 2.18e9
BANDWIDTH_CEILING = 324e9          # E4, measured
FLOOR_MS = WEIGHT_BYTES / BANDWIDTH_CEILING * 1e3
KERNELS_INFERRED = 700             # INFERRED, see preregistration section 2
BOOTSTRAP_SEED = 20260825
KNOBS = Knobs(compiled_fixed_cache=True, fused_argmax=False, head_skip_prefill=True,
              fuse_projections=True)
CAPACITY = ((CONTEXT_TOKENS + SEQ_STEPS + 8 + 63) // 64) * 64
FIXED_TOKEN = 2                    # teacher-forcing token, never read back


def timed(build):
    """Total, CPU-side and GPU-side time for one submission.

    t_submit covers Python graph construction plus async submission, which is the
    runtime-side cost a scheduler could attack. It is not called 'dispatch'.
    """
    started = time.perf_counter_ns()
    outputs = build()
    flat = [o for o in outputs if isinstance(o, mx.array)]
    flat += [leaf for o in outputs if isinstance(o, dict) for leaf in _leaves(o)]
    mx.async_eval(*flat)
    submitted = time.perf_counter_ns()
    mx.eval(*flat)
    mx.synchronize()
    done = time.perf_counter_ns()
    return {"total_ns": done - started, "submit_ns": submitted - started,
            "gpu_ns": done - submitted}, outputs


class Harness:
    def __init__(self, model_id: str):
        self.engine, self.tok = load_engine(model_id, KNOBS)
        self.model = self.engine.model
        self.trunk = _trunk(self.engine.model)
        self._bodies: dict[tuple, object] = {}
        self._probe = None

    # -- prompts --------------------------------------------------------------
    def prompts(self) -> list[list[int]]:
        data = json.loads(SQUAD.read_text())
        articles = sorted(data["data"], key=lambda a: a["title"])[:SEQUENCES]
        out = []
        for article in articles:
            document = ""
            for paragraph in article["paragraphs"]:
                document = (document + "\n\n" + paragraph["context"]).strip()
                if len(self.tok.encode(document, add_special_tokens=False)) > CONTEXT_TOKENS + 200:
                    break
            rendered = self.tok.apply_chat_template(
                [{"role": "user", "content": document}], tokenize=False, add_generation_prompt=True)
            ids = list(self.tok.encode(rendered, add_special_tokens=False))
            assert len(ids) >= CONTEXT_TOKENS, f"{article['title']} too short: {len(ids)}"
            out.append(ids[:CONTEXT_TOKENS])          # exact token-level truncation
        return out

    # -- states ---------------------------------------------------------------
    def _shapes(self):
        if self._probe is None:
            cache = self.model.make_cache()
            self.trunk(mx.array([[FIXED_TOKEN]]), cache=cache)
            mx.eval([c.keys for c in cache])
            self._probe = [(c.keys.shape[1], c.keys.shape[3], c.keys.dtype) for c in cache]
        return self._probe

    def empty_state(self, batch: int):
        return {"position": {"offset": mx.array(0, dtype=mx.int32)},
                "layers": [{"keys": mx.zeros((batch, heads, CAPACITY, dim), dtype=dtype),
                            "values": mx.zeros((batch, heads, CAPACITY, dim), dtype=dtype)}
                           for heads, dim, dtype in self._shapes()]}

    def prefill(self, token_batch: list[list[int]]):
        """One prefill for a whole batch. All rows share an offset by construction."""
        state = self.empty_state(len(token_batch))
        caches = [FixedKVCache(layer, state["position"], CAPACITY) for layer in state["layers"]]
        hidden = self.trunk(mx.array(token_batch), cache=caches)
        logits = _project(self.model, hidden[:, -1:, :])
        state = {"position": {"offset": mx.array(len(token_batch[0]), dtype=mx.int32)},
                 "layers": [{"keys": c.keys, "values": c.values} for c in caches]}
        mx.eval(logits, *_leaves(state))
        mx.synchronize()
        return state, logits[:, -1, :]

    @staticmethod
    def reset(state):
        return {"position": {"offset": mx.array(CONTEXT_TOKENS, dtype=mx.int32)},
                "layers": [{"keys": l["keys"], "values": l["values"]} for l in state["layers"]]}

    # -- compiled bodies ------------------------------------------------------
    def body(self, batch: int, width: int, extra: int = 0):
        key = (batch, width, extra)
        if key in self._bodies:
            return self._bodies[key]
        model, trunk = self.model, self.trunk

        def fn(input_ids, state):
            caches = [FixedKVCache(layer, state["position"], CAPACITY) for layer in state["layers"]]
            hidden = trunk(input_ids, cache=caches)
            logits = _project(model, hidden[:, -1:, :])
            new_state = {"position": {"offset": state["position"]["offset"] + input_ids.shape[1]},
                         "layers": [{"keys": c.keys, "values": c.values} for c in caches]}
            if extra:
                # Seeded from the logits so it cannot be constant folded or hoisted,
                # returned as an extra output so it cannot be eliminated, and never
                # touching the model's own numbers.
                chain = mx.broadcast_to(mx.max(logits).reshape((1, 1)), (8, 8)) * 1e-3
                weight = mx.full((8, 8), 0.125, dtype=chain.dtype)
                for _ in range(extra):
                    chain = chain @ weight
                return logits, new_state, chain
            return logits, new_state

        compiled = mx.compile(fn, shapeless=False)
        self._bodies[key] = compiled
        return compiled


# --- arrangements ------------------------------------------------------------

def arrangement_S(h, state, steps):
    """batch 1, width 1, sequential steps. Teacher forced, so no readback."""
    body = h.body(1, 1)
    work = h.reset(state)
    inp = mx.array([[FIXED_TOKEN]])
    per_step = []
    started = time.perf_counter_ns()
    for _ in range(steps):
        sample, out = timed(lambda: body(inp, work))
        work = out[1]
        per_step.append(sample)
    return {"total_ns": time.perf_counter_ns() - started, "per_step": per_step,
            "logical_tokens": steps}


def arrangement_W(h, state, width):
    body = h.body(1, width)
    inp = mx.array([[FIXED_TOKEN] * width])
    sample, out = timed(lambda: body(inp, h.reset(state)))
    return {**sample, "logical_tokens": width, "logits": out[0]}


def arrangement_B(h, state, batch):
    body = h.body(batch, 1)
    inp = mx.array([[FIXED_TOKEN]] * batch)
    sample, out = timed(lambda: body(inp, h.reset(state)))
    return {**sample, "logical_tokens": batch, "logits": out[0]}


def arrangement_U(h, states, batch):
    """The same sequences, one at a time. Fair baseline for B(batch)."""
    body = h.body(1, 1)
    inp = mx.array([[FIXED_TOKEN]])
    per_step, logits = [], []
    started = time.perf_counter_ns()
    for index in range(batch):
        sample, out = timed(lambda: body(inp, h.reset(states[index])))
        per_step.append(sample)
        logits.append(out[0])
    return {"total_ns": time.perf_counter_ns() - started, "per_step": per_step,
            "logical_tokens": batch, "logits": logits}


def sync_probe(h, state, steps):
    """Per-step cost of synchronisation, measured rather than inferred."""
    body = h.body(1, 1)
    inp = mx.array([[FIXED_TOKEN]])

    work = h.reset(state)
    started = time.perf_counter_ns()
    for _ in range(steps):
        out = body(inp, work)
        work = out[1]
        mx.eval(out[0], *_leaves(work))
        mx.synchronize()
    per_step_sync_ns = time.perf_counter_ns() - started

    work = h.reset(state)
    started = time.perf_counter_ns()
    outs = []
    for _ in range(steps):
        out = body(inp, work)
        work = out[1]
        outs.append(out[0])
    mx.eval(*outs, *_leaves(work))
    mx.synchronize()
    amortised_ns = time.perf_counter_ns() - started

    return {"per_step_sync_ns": per_step_sync_ns, "amortised_ns": amortised_ns,
            "steps": steps,
            "delta_sync_ns_per_step": (per_step_sync_ns - amortised_ns) / steps}


def bits_equal(a, b) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    view = {2: mx.uint16, 4: mx.uint32, 1: mx.uint8, 8: mx.uint64}[a.dtype.size]
    return bool(mx.array_equal(a.view(view), b.view(view)).item())


# --- driver ------------------------------------------------------------------

def run_process(model_id: str, order: list[str], pilot: bool) -> dict:
    # MLX's peak is a process-wide high-water mark and these blocks share one
    # interpreter (see E15 limitation M2). Without this reset the reported peak is
    # the maximum over every block so far, which also makes the memory guard below
    # fire earlier the longer a run goes.
    mx.reset_peak_memory()
    h = Harness(model_id)
    prompts = h.prompts()
    sequences = 2 if pilot else SEQUENCES
    batches = [1, 2] if pilot else BATCHES
    widths = [1, 2] if pilot else WIDTHS
    controls = [0, 64] if pilot else CONTROL_K
    repeats = 2 if pilot else REPEATS

    single = [h.prefill([p]) for p in prompts[:sequences]]
    single_states = [s for s, _ in single]
    single_logits = [lg for _, lg in single]
    batched = {b: h.prefill(prompts[:b]) for b in batches if b <= sequences}

    # correctness of the batched primitive, against the unbatched rows
    identity = []
    for b, (bstate, blogits) in batched.items():
        rows = [bits_equal(blogits[i:i + 1], single_logits[i]) for i in range(b)]
        identity.append({"batch": b, "rows_bit_equal": rows, "all": all(rows)})

    out = {"order": order, "pid": os.getpid(), "capacity": CAPACITY,
           "context_tokens": CONTEXT_TOKENS, "prefill_logit_identity": identity,
           "arrangements": {}, "controls": {}, "sync_probe": None}

    def record(name, samples):
        out["arrangements"].setdefault(name, []).extend(samples)

    for _ in range(WARMUP):
        arrangement_S(h, single_states[0], 2)
        for b in batches:
            arrangement_B(h, batched[b][0], b)
        for w in widths:
            arrangement_W(h, single_states[0], w)

    for name in order:
        for _ in range(repeats):
            if name == "S":
                record("S", [arrangement_S(h, single_states[0], SEQ_STEPS)])
            elif name == "W":
                for w in widths:
                    r = arrangement_W(h, single_states[0], w)
                    r.pop("logits", None)
                    record(f"W{w}", [r])
            elif name == "B":
                for b in batches:
                    r = arrangement_B(h, batched[b][0], b)
                    r.pop("logits", None)
                    record(f"B{b}", [r])
            elif name == "U":
                for b in batches:
                    if b > sequences:
                        continue
                    r = arrangement_U(h, single_states, b)
                    r.pop("logits", None)
                    record(f"U{b}", [r])

    for k in controls:
        body = h.body(1, 1, extra=k)
        inp = mx.array([[FIXED_TOKEN]])
        for _ in range(WARMUP):
            timed(lambda: body(inp, h.reset(single_states[0])))
        out["controls"][str(k)] = [timed(lambda: body(inp, h.reset(single_states[0])))[0]
                                   for _ in range(repeats)]

    out["sync_probe"] = [sync_probe(h, single_states[0], SEQ_STEPS) for _ in range(repeats)]
    out["mlx_peak_bytes"] = mx.get_peak_memory()
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["pilot", "main"], required=True)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args(argv)

    busy = gpu_busy()
    if busy:
        print(f"ABORT gpu_busy: {busy}")
        return 2
    env = bench.environment()
    if env["power_source"] != "AC":
        print(f"ABORT power_source={env['power_source']}")
        return 2

    orders = [["S", "W", "B", "U"], ["U", "B", "W", "S"]]
    started = time.perf_counter()
    processes = 1 if args.stage == "pilot" else args.processes
    runs = []
    for index in range(processes):
        run = run_process(args.model, orders[index % 2], pilot=args.stage == "pilot")
        runs.append(run)
        print(f"process {index+1}/{processes} done, order {run['order']}, "
              f"peak {run['mlx_peak_bytes']/1e9:.2f} GB", flush=True)

    payload = {"experiment": "E14", "stage": args.stage, "runs": runs,
               "preregistration_sha256": "13f6d3589cfc5d83136f275fb63e0382f3b068ed8c1408abd460ac4f59a400a2",
               "prereg_commit": "e1c29f0", "floor_ms": FLOOR_MS,
               "kernels_inferred": KERNELS_INFERRED,
               "wall_seconds": time.perf_counter() - started, "environment": env}
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"E14_results_{args.stage}.json"
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str))
    print(f"\nwrote {path} ({payload['wall_seconds']:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
