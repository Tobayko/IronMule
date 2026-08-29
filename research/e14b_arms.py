"""E14b: separate submission/sync amortisation from true batched execution.

Preregistered at commit c6e7f69, research/raw/E14b_preregistration.md
(SHA-256 564c3906b3de856a5d641c0750a1e5493f8cc640794be1b72cd3c21fedcf8712).

Three arms on identical logical work:
  A  Sequential B1   b independent batch-1 executions, each synchronised on its own
  B  Async B1 Group  the same b executions, no intermediate barrier, one sync
  C  True Batch      the same b sequences in a real batch dimension

Arm B is the discriminator. A against C alone — which is what E14 measured — cannot
tell amortised submission from a shape effect.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import mlx.core as mx

from ironmule import bench
from ironmule.runtime import _leaves
from ironmule.tune import DEFAULT_MODEL, gpu_busy

from e14_dispatch import CAPACITY, CONTEXT_TOKENS, FIXED_TOKEN, Harness, RAW, bits_equal

HERE = Path(__file__).resolve().parent

BATCHES = [1, 2, 4, 8]
SEQUENCES = 8
WARMUP, REPEATS = 2, 7
GEN_TOKENS = 8
SEED = 20260825
now = time.perf_counter_ns


def flatten(outputs):
    flat = [o for o in outputs if isinstance(o, mx.array)]
    flat += [leaf for o in outputs if isinstance(o, dict) for leaf in _leaves(o)]
    return flat


_SENTINEL = None


def barrier():
    """Full barrier so no earlier work leaks into the next block (control 3)."""
    global _SENTINEL
    if _SENTINEL is None:
        _SENTINEL = mx.array([0.0])
    mx.eval(_SENTINEL)
    mx.synchronize()


def _split(t0, t_prep, t_submit, t_done):
    return {"host_prep_ns": t_prep - t0, "submission_ns": t_submit - t_prep,
            "completion_wait_ns": t_done - t_submit, "total_ns": t_done - t0}


def timer_control():
    """Control 1: the noise floor of the timing machinery on a trivial op."""
    x = mx.array([1.0])
    mx.eval(x)
    barrier()
    t0 = now()
    y = x + 1.0
    flat = [y]
    t_prep = now()
    mx.async_eval(*flat)
    t_submit = now()
    mx.eval(*flat)
    mx.synchronize()
    return _split(t0, t_prep, t_submit, now())


def arm_A(h, states, b):
    """Sequential B1: every execution submitted and synchronised on its own."""
    body = h.body(1, 1)
    inp = mx.array([[FIXED_TOKEN]])
    barrier()
    t0 = now()
    prep = submit = wait = 0
    for index in range(b):
        a = now()
        flat = flatten(body(inp, h.reset(states[index])))
        p = now()
        mx.async_eval(*flat)
        s = now()
        mx.eval(*flat)
        mx.synchronize()
        d = now()
        prep += p - a
        submit += s - p
        wait += d - s
    total = now() - t0
    return {"host_prep_ns": prep, "submission_ns": submit,
            "completion_wait_ns": wait, "total_ns": total}


def arm_B(h, states, b):
    """Async B1 Group: same shapes, no intermediate barrier, one synchronisation."""
    body = h.body(1, 1)
    inp = mx.array([[FIXED_TOKEN]])
    barrier()
    t0 = now()
    flat = []
    for index in range(b):
        flat += flatten(body(inp, h.reset(states[index])))
    t_prep = now()
    mx.async_eval(*flat)
    t_submit = now()
    mx.eval(*flat)
    mx.synchronize()
    return _split(t0, t_prep, t_submit, now())


def arm_C(h, batched_state, b):
    """True Batch: one execution in a real batch dimension."""
    body = h.body(b, 1)
    inp = mx.array([[FIXED_TOKEN]] * b)
    barrier()
    t0 = now()
    flat = flatten(body(inp, h.reset(batched_state)))
    t_prep = now()
    mx.async_eval(*flat)
    t_submit = now()
    mx.eval(*flat)
    mx.synchronize()
    return _split(t0, t_prep, t_submit, now())


ARMS = {"A": arm_A, "B": arm_B, "C": arm_C}


def generate(h, state, batch, steps):
    """Real greedy generation, for the correctness block only."""
    body = h.body(batch, 1)
    work = h.reset(state)
    current = mx.array([[FIXED_TOKEN]] * batch)
    rows = [[] for _ in range(batch)]
    for _ in range(steps):
        out = body(current, work)
        logits, work = out[0][:, -1, :], out[1]
        mx.eval(logits, *_leaves(work))
        mx.synchronize()
        picks = mx.argmax(logits.astype(mx.float32), axis=-1)
        for r in range(batch):
            rows[r].append(int(picks[r].item()))
        current = picks.reshape((batch, 1))
    return rows


def run_process(model_id: str, index: int, pilot: bool) -> dict:
    # Since R12 each block is its own process, so the process-wide high-water mark is
    # already this block's alone. Kept because this function is still callable in-process
    # and the reset costs nothing: it makes the peak correct either way rather than
    # correct only as long as nobody calls it twice.
    mx.reset_peak_memory()
    h = Harness(model_id)
    prompts = h.prompts()
    sequences = 2 if pilot else SEQUENCES
    batches = [1, 2] if pilot else BATCHES
    repeats = 2 if pilot else REPEATS
    rng = random.Random(SEED + index)

    controls = {"timer_control": [timer_control() for _ in range(5)]}
    prefilled = [h.prefill([p]) for p in prompts[:sequences]]
    single_states = [state for state, _ in prefilled]
    single_logits_all = [logits for _, logits in prefilled]
    mx.clear_cache()

    out = {"pid": os.getpid(), "process_index": index, "capacity": CAPACITY,
           "context_tokens": CONTEXT_TOKENS, "controls": controls,
           "blocks": [], "correctness": [], "batches": batches}

    order = list(batches)
    rng.shuffle(order)
    for b in order:
        batched_state, batched_logits = h.prefill(prompts[:b])
        # reuse the single prefills already computed; re-running them would cost
        # another full prefill per block and measure nothing new
        rows = [bits_equal(batched_logits[i:i + 1], single_logits_all[i])
                for i in range(min(b, sequences))]

        for _ in range(WARMUP):
            for name in ("A", "B", "C"):
                (ARMS[name](h, single_states, b) if name != "C"
                 else ARMS[name](h, batched_state, b))

        for repeat in range(repeats):
            names = ["A", "B", "C"]
            rng.shuffle(names)
            for name in names:
                sample = (ARMS[name](h, single_states, b) if name != "C"
                          else ARMS[name](h, batched_state, b))
                out["blocks"].append({"arm": name, "batch": b, "repeat": repeat, **sample})

        if b > 1 and not pilot:
            batched_rows = generate(h, batched_state, b, GEN_TOKENS)
            singles = [generate(h, single_states[i], 1, GEN_TOKENS)[0] for i in range(b)]
            out["correctness"].append({
                "batch": b, "prefill_logits_bit_equal": rows,
                "batched_tokens": batched_rows, "single_tokens": singles,
                "tokens_equal": [batched_rows[i] == singles[i] for i in range(b)],
                "counts_equal": [len(batched_rows[i]) == len(singles[i]) for i in range(b)]})
        del batched_state, batched_logits
        mx.clear_cache()

    out["mlx_peak_bytes"] = mx.get_peak_memory()
    return out


def spawn(model_id: str, index: int, pilot: bool, timeout: int = 1800) -> dict:
    """One block, one OS process, gone before the next begins.

    R12, following `e16_replication.spawn`. The preregistration says "four fresh
    processes" and the loop this replaces gave four blocks in one interpreter (E15
    limitation M2), so the fork delivers the frozen document rather than departing from
    it. It also makes each reported peak the block's own: run in one interpreter, block 2
    of a 12B run peaked at 23.14 GB against block 1's 17.51 GB because the allocator had
    not returned block 1's buffers, and the machine swapped.
    """
    spec = json.dumps({"model": model_id, "index": index, "pilot": pilot})
    try:
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--child", spec],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(HERE.parent), env={**os.environ, "PYTHONPATH": str(HERE.parent),
                                       "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    except subprocess.TimeoutExpired:
        # One slow block must not take the run down with it. `e16_replication.spawn`
        # leaves this uncaught; a 27B block would sit closest to the limit.
        return {"index": index, "crashed": True, "stderr": f"timeout after {timeout}s",
                "mlx_peak_bytes": 0, "blocks": [], "correctness": []}
    line = next((l for l in proc.stdout.splitlines() if l.startswith("@@")), None)
    if line is None:
        return {"index": index, "crashed": True, "stderr": proc.stderr[-2000:],
                "mlx_peak_bytes": 0, "blocks": [], "correctness": []}
    return json.loads(line[2:])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    # Not `required`: a child is invoked with `--child` alone and carries its stage
    # inside the spec. The parent path still insists on it, just below.
    parser.add_argument("--stage", choices=["pilot", "main"])
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--child", help="internal: run one block and print it as JSON")
    args = parser.parse_args(argv)

    if args.child:
        spec = json.loads(args.child)
        run = run_process(spec["model"], spec["index"], pilot=spec["pilot"])
        print("@@" + json.dumps(run, default=str))
        return 0

    if args.stage is None:
        parser.error("--stage is required")

    busy = gpu_busy()
    if busy:
        print(f"ABORT gpu_busy: {busy}")
        return 2
    env = bench.environment()
    if env["power_source"] != "AC":
        print(f"ABORT power_source={env['power_source']}")
        return 2

    started = time.perf_counter()
    processes = 1 if args.stage == "pilot" else args.processes
    runs = []
    # R11: the condition is swap, not a byte count. The old 12 GiB literal refused 12B
    # at a 17.51 GB peak that measured cleanly, and passed a run that had to be
    # discarded for swapping. `gate.record` carries what it saw; putting that into the
    # payload is R10, which needs a schema decision this experiment cannot make alone.
    gate = bench.MemoryGate()
    for index in range(processes):
        run = spawn(args.model, index, args.stage == "pilot")
        runs.append(run)
        if run.get("crashed"):
            print(f"ABORT child {index} crashed:\n{run['stderr']}")
            break
        print(f"process {index+1}/{processes} done, peak {run['mlx_peak_bytes']/1e9:.2f} GB",
              flush=True)
        stop = gate.check(index, run["mlx_peak_bytes"])
        if stop:
            print(f"ABORT {stop}")
            break

    # R10, additively: this diff already extends the schema with `crashed`/`stderr`,
    # so withholding the abort record was inconsistent rather than careful. An
    # unknown key costs existing readers nothing; a silent truncation costs them the
    # difference between a short run and a stopped one.
    payload = {"experiment": "E14b", "stage": args.stage, "runs": runs,
               "memory_gate": gate.record,
               "preregistration_sha256": "564c3906b3de856a5d641c0750a1e5493f8cc640794be1b72cd3c21fedcf8712",
               "prereg_commit": "c6e7f69", "wall_seconds": time.perf_counter() - started,
               "environment": env}
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"E14b_results_{args.stage}.json"
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str))
    print(f"\nwrote {path} ({payload['wall_seconds']:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
