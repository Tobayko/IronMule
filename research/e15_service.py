"""E15: does grouped async B1 survive a real service workload?

Preregistered at commit c2c8a59, research/raw/E15_preregistration.md
(SHA-256 939a3c40683433e6fc2e24c4409304a4a762fbae52c7b528b6a4de1216b70a92).

A minimal experimental executor. Not adaptive, no controller, no tensor batch
dimension — every execution stays batch 1 with unchanged shapes. The caller keeps
the execution plan; the executor never switches or alters one. Nothing in ironmule/
is modified.

`completion_wait` is a wait, not GPU time. No kernel count is used anywhere.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics as st
import time
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import mlx.core as mx
import numpy as np

from ironmule import bench
from ironmule.runtime import Knobs, PrefixCache, _leaves
from ironmule.tune import DEFAULT_MODEL, _eos_ids, gpu_busy, load_engine

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
SQUAD = HERE / "data" / "squad-dev-v1.1.json"

WIDTHS = [1, 2, 4, 8]
PRIMARY_WIDTH = 4
REQUESTS = 8
WARMUP, REPEATS = 2, 3
SEED = 20260825
KNOBS = Knobs(compiled_fixed_cache=True, fused_argmax=False, head_skip_prefill=True,
              fuse_projections=True)
# The pilot showed the terse instruction below yields 2-3 token answers, which
# cannot exercise a decode scheduler at all. The main workloads therefore ask for a
# complete sentence, which generates naturally ragged 10-30 token answers. The terse
# variant is kept as its own workload because it is the case where grouping loses.
INSTRUCTION = ("Read the document and answer the question in one complete sentence, "
               "quoting the relevant wording from the document.\n\nDocument:\n")
TERSE_INSTRUCTION = ("Read the document and answer the question using the shortest exact span "
                     "from the document. Give only that span, with no explanation.\n\n"
                     "Document:\n")

WORKLOADS = {
    "homogeneous":   {"contexts": [1024] * 8, "caps": [16] * 8,
                      "arrivals": [0.0] * 8},
    "heterogeneous": {"contexts": [320, 512, 768, 1024, 1200, 320, 768, 1024],
                      "caps": [8, 12, 16, 24, 8, 24, 12, 16], "arrivals": [0.0] * 8},
    "staggered":     {"contexts": [320, 512, 768, 1024, 1200, 320, 768, 1024],
                      "caps": [8, 12, 16, 24, 8, 24, 12, 16],
                      "arrivals": [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0]},
    # Added after the pilot, before freezing: the degenerate short-answer service.
    "terse":         {"contexts": [1024] * 8, "caps": [16] * 8, "arrivals": [0.0] * 8,
                      "instruction": "terse"},
}

now = time.perf_counter_ns


@dataclass
class Request:
    rid: int
    prompt_ids: list[int]
    arrival_ms: float
    cap: int
    prompt_tokens: int
    base_state: dict
    first_token: int
    state: dict | None = None
    tokens: list[int] = field(default_factory=list)
    done: bool = False
    stop_reason: str = ""
    admitted_ns: int = 0
    first_step_ns: int = 0
    finished_ns: int = 0
    token_times_ns: list[int] = field(default_factory=list)


def reset_state(state, offset: int):
    return {"position": {"offset": mx.array(offset, dtype=mx.int32)},
            "layers": [{"keys": l["keys"], "values": l["values"]} for l in state["layers"]]}


def valid_kv_hash(state, offset: int) -> str:
    import hashlib
    digest = hashlib.sha256()
    for layer in state["layers"]:
        for name in ("keys", "values"):
            arr = layer[name][..., :offset, :]
            view = {2: mx.uint16, 4: mx.uint32}[arr.dtype.size]
            digest.update(np.asarray(arr.view(view)).tobytes())
    return digest.hexdigest()


_SENTINEL = None


def barrier():
    global _SENTINEL
    if _SENTINEL is None:
        _SENTINEL = mx.array([0.0])
    mx.eval(_SENTINEL)
    mx.synchronize()


def _prepare(requests, capacity):
    for r in requests:
        r.state = reset_state(r.base_state, r.prompt_tokens)
        r.tokens = [r.first_token]
        r.done = False
        r.stop_reason = ""
        r.admitted_ns = r.first_step_ns = r.finished_ns = 0
        r.token_times_ns = []


def _step_graph(body, request):
    out = body(mx.array([[request.tokens[-1]]]), request.state)
    pick = mx.argmax(out[0][:, -1, :].astype(mx.float32), axis=-1)
    return out, pick


def _finish(request, token, eos, when):
    request.tokens.append(token)
    request.token_times_ns.append(when)
    if token in eos:
        request.done, request.stop_reason = True, "eos"
    elif len(request.tokens) - 1 >= request.cap:
        request.done, request.stop_reason = True, "cap"


def run_grouped(engine, requests, capacity, eos, width, reverse=False):
    """AsyncGroupedB1(W). Never waits to fill a group; serves what is ready."""
    body = engine._body(capacity, 1)
    _prepare(requests, capacity)
    order = sorted(requests, key=lambda r: (r.arrival_ms, r.rid), reverse=reverse)
    pending, active, rounds = list(order), [], []
    idle_ns = 0
    barrier()
    t0 = now()
    while pending or active:
        while pending and (now() - t0) / 1e6 >= pending[0].arrival_ms:
            admitted = pending.pop(0)
            admitted.admitted_ns = now()
            active.append(admitted)
        if not active:
            wait_ms = pending[0].arrival_ms - (now() - t0) / 1e6
            if wait_ms > 0:
                idle_start = now()
                time.sleep(wait_ms / 1000.0)
                idle_ns += now() - idle_start
            continue

        group = active[:width]
        ta = now()
        graphs = [_step_graph(body, r) for r in group]
        flat = [pick for _, pick in graphs]
        flat += [leaf for out, _ in graphs for leaf in _leaves(out[1])]
        t_prep = now()
        mx.async_eval(*flat)
        t_submit = now()
        mx.eval(*flat)
        mx.synchronize()
        t_done = now()

        for (out, pick), r in zip(graphs, group):
            if not r.first_step_ns:
                r.first_step_ns = t_done
            r.state = out[1]
            _finish(r, int(pick.item()), eos, t_done)
            if r.done:
                r.finished_ns = t_done
        rounds.append({"width_requested": width, "width_realised": len(group),
                       "host_prep_ns": t_prep - ta, "submission_ns": t_submit - t_prep,
                       "completion_wait_ns": t_done - t_submit, "total_ns": t_done - ta})
        active = active[len(group):] + [r for r in group if not r.done]
    return {"wall_ns": now() - t0, "idle_ns": idle_ns, "rounds": rounds,
            "t0": t0, "strategy": f"grouped{width}"}


def run_sequential(engine, requests, capacity, eos, reverse=False):
    """SequentialService: each request run to completion, synchronised per step."""
    body = engine._body(capacity, 1)
    _prepare(requests, capacity)
    order = sorted(requests, key=lambda r: (r.arrival_ms, r.rid), reverse=reverse)
    rounds, idle_ns = [], 0
    barrier()
    t0 = now()
    for r in order:
        wait_ms = r.arrival_ms - (now() - t0) / 1e6
        if wait_ms > 0:
            idle_start = now()
            time.sleep(wait_ms / 1000.0)
            idle_ns += now() - idle_start
        r.admitted_ns = now()
        while not r.done:
            ta = now()
            out, pick = _step_graph(body, r)
            flat = [pick] + _leaves(out[1])
            t_prep = now()
            mx.async_eval(*flat)
            t_submit = now()
            mx.eval(*flat)
            mx.synchronize()
            t_done = now()
            if not r.first_step_ns:
                r.first_step_ns = t_done
            r.state = out[1]
            _finish(r, int(pick.item()), eos, t_done)
            rounds.append({"width_requested": 0, "width_realised": 1,
                           "host_prep_ns": t_prep - ta, "submission_ns": t_submit - t_prep,
                           "completion_wait_ns": t_done - t_submit, "total_ns": t_done - ta})
        r.finished_ns = now()
    return {"wall_ns": now() - t0, "idle_ns": idle_ns, "rounds": rounds,
            "t0": t0, "strategy": "sequential"}


def snapshot(requests, run, capacity, with_hashes: bool):
    t0 = run["t0"]
    out = []
    for r in sorted(requests, key=lambda r: r.rid):
        gaps = [b - a for a, b in zip(r.token_times_ns, r.token_times_ns[1:])]
        out.append({
            "rid": r.rid, "arrival_ms": r.arrival_ms, "cap": r.cap,
            "prompt_tokens": r.prompt_tokens, "tokens": r.tokens,
            "token_count": len(r.tokens), "stop_reason": r.stop_reason,
            "queue_wait_ms": (r.first_step_ns - max(r.admitted_ns, t0)) / 1e6,
            "ttft_ms": (r.token_times_ns[0] - t0) / 1e6 if r.token_times_ns else None,
            "latency_ms": (r.finished_ns - max(r.admitted_ns, t0)) / 1e6,
            "inter_token_ms": [g / 1e6 for g in gaps],
            "kv_hash": valid_kv_hash(r.state, r.prompt_tokens + len(r.tokens) - 1)
                       if with_hashes else None,
        })
    return out


def build_requests(engine, tok, workload: str, plan: str):
    spec = WORKLOADS[workload]
    instruction = TERSE_INSTRUCTION if spec.get("instruction") == "terse" else INSTRUCTION
    data = json.loads(SQUAD.read_text())
    articles = sorted(data["data"], key=lambda a: a["title"])
    encode = lambda t: list(tok.encode(t, add_special_tokens=False))

    def document(article, target):
        text = ""
        for paragraph in article["paragraphs"]:
            text = (text + "\n\n" + paragraph["context"]).strip()
            if len(encode(text)) >= target:
                break
        return text

    def render(doc, question):
        return tok.apply_chat_template(
            [{"role": "user", "content": instruction + doc + "\n\nQuestion: " + question}],
            tokenize=False, add_generation_prompt=True)

    if plan == "reusable":
        # A reusable session shares one prefix by definition, so context variation
        # lives in the questions rather than the documents. Recorded, not hidden.
        article = articles[0]
        doc = document(article, max(spec["contexts"]))
        questions = [qa["question"] for p in article["paragraphs"] for qa in p["qas"]][:REQUESTS]
        docs = [doc] * REQUESTS
        prefix_text = tok.apply_chat_template(
            [{"role": "user", "content": instruction + doc + "\n\n@@CUT@@"}],
            tokenize=False, add_generation_prompt=True).split("@@CUT@@")[0]
        engine.prefix_cache = PrefixCache(encode(prefix_text))
    else:
        engine.prefix_cache = None
        docs, questions = [], []
        for index in range(REQUESTS):
            article = articles[index + 3]
            docs.append(document(article, spec["contexts"][index]))
            questions.append(article["paragraphs"][0]["qas"][0]["question"])

    prompts = [encode(render(docs[i], questions[i])) for i in range(REQUESTS)]
    capacity = ((max(len(p) for p in prompts) + max(spec["caps"]) + 8 + 63) // 64) * 64

    requests, prefill_ns = [], []
    for index, ids in enumerate(prompts):
        started = now()
        state, token = engine._prefill(ids, capacity)
        prefill_ns.append(now() - started)
        requests.append(Request(rid=index, prompt_ids=ids, arrival_ms=spec["arrivals"][index],
                                cap=spec["caps"][index], prompt_tokens=len(ids),
                                base_state=state, first_token=int(token.reshape((-1,)).item())))
    engine.prefix_cache = None
    return requests, capacity, prefill_ns


def run_process(model_id: str, index: int, pilot: bool) -> dict:
    engine, tok = load_engine(model_id, KNOBS)
    eos = _eos_ids(tok)
    rng = random.Random(SEED + index)
    workloads = ["homogeneous"] if pilot else list(WORKLOADS)
    plans = ["strict"] if pilot else ["strict", "reusable"]
    widths = [1, 2, 4] if pilot else WIDTHS
    repeats = 1 if pilot else REPEATS

    out = {"pid": os.getpid(), "process_index": index, "runs": [],
           "reference": [], "controls": {}, "prefill_ms": {}}

    noise = []
    for _ in range(5):
        barrier()
        ta = now()
        y = mx.array([1.0]) + 1.0
        t_prep = now()
        mx.async_eval(y)
        t_submit = now()
        mx.eval(y)
        mx.synchronize()
        noise.append({"host_prep_ns": t_prep - ta, "submission_ns": t_submit - t_prep,
                      "completion_wait_ns": now() - t_submit, "total_ns": now() - ta})
    out["controls"]["timer_noise"] = noise

    # No prefill cache across workloads. The first main run aborted on the
    # preregistered 12 GiB guard at 20.97 GB, because caching six request sets holds
    # six times eight KV states of 187 MB each. Correct behaviour from the guard, a
    # defect in the harness: every set is now built, used and freed.
    for workload in workloads:
        for plan in plans:
            spec = WORKLOADS[workload]
            requests, capacity, prefill_ns = build_requests(engine, tok, workload, plan)
            cold_key = f"{workload}/{plan}"
            if cold_key not in out.setdefault("cold_start", {}):
                cold = run_grouped(engine, requests, capacity, eos, PRIMARY_WIDTH)
                out["cold_start"][cold_key] = {
                    "first_rounds_ms": [r["total_ns"] / 1e6 for r in cold["rounds"][:5]],
                    "steady_median_ms": st.median(r["total_ns"] for r in cold["rounds"][5:]) / 1e6
                                        if len(cold["rounds"]) > 5 else None}
            for r, arrival, cap in zip(requests, spec["arrivals"], spec["caps"]):
                r.arrival_ms, r.cap = arrival, cap    # same prefills, different schedule
            out["prefill_ms"][f"{workload}/{plan}"] = [p / 1e6 for p in prefill_ns]

            # Warm every width that will be measured, not only the primary one. The
            # pilot showed the first three grouped rounds at W=4 cost 412/315/252 ms
            # against a 43.5 ms steady state: a one-time allocator build-up for W
            # simultaneous KV states. That is a real startup cost a service pays once,
            # not a per-request cost, and letting it land inside a measured block
            # would report it as strategy cost. It is reported separately instead.
            for _ in range(WARMUP):
                run_sequential(engine, requests, capacity, eos)
                for width in widths:
                    run_grouped(engine, requests, capacity, eos, width)
                run_grouped(engine, requests, capacity, eos, PRIMARY_WIDTH, reverse=True)

            base = run_sequential(engine, requests, capacity, eos)
            reference = snapshot(requests, base, capacity, with_hashes=True)
            out["reference"].append({"workload": workload, "plan": plan,
                                     "capacity": capacity, "requests": reference})

            blocks = [("sequential", None)] + [("grouped", w) for w in widths]
            hashed: set[str] = set()
            for _ in range(repeats):
                rng.shuffle(blocks)
                for kind, width in blocks:
                    run = (run_sequential(engine, requests, capacity, eos) if kind == "sequential"
                           else run_grouped(engine, requests, capacity, eos, width))
                    # hashing 1.5 GB of KV costs about a second; once per strategy is
                    # enough because repeats are deterministic. Tokens are compared every time.
                    want = run["strategy"] not in hashed
                    hashed.add(run["strategy"])
                    snap = snapshot(requests, run, capacity, with_hashes=want)
                    out["runs"].append({
                        "workload": workload, "plan": plan, "strategy": run["strategy"],
                        "width": width or 0, "wall_ns": run["wall_ns"], "idle_ns": run["idle_ns"],
                        "rounds": run["rounds"], "requests": snap,
                        "tokens_generated": sum(r["token_count"] - 1 for r in snap),
                        "mean_realised_width": st.mean(x["width_realised"] for x in run["rounds"]),
                    })

            # order independence: reversed arrival order at the primary width
            rev = run_grouped(engine, requests, capacity, eos, PRIMARY_WIDTH, reverse=True)
            out["runs"].append({
                "workload": workload, "plan": plan, "strategy": "grouped_reversed",
                "width": PRIMARY_WIDTH, "wall_ns": rev["wall_ns"], "idle_ns": rev["idle_ns"],
                "rounds": rev["rounds"], "requests": snapshot(requests, rev, capacity, True),
                "tokens_generated": 0, "mean_realised_width":
                    st.mean(x["width_realised"] for x in rev["rounds"])})
            for r in requests:
                r.base_state = None
                r.state = None
            del requests
            mx.clear_cache()

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

    started = time.perf_counter()
    processes = 1 if args.stage == "pilot" else args.processes
    runs, aborted = [], None
    for index in range(processes):
        if time.perf_counter() - started > 45 * 60:
            aborted = "wall_limit"
            print("ABORT main-run wall limit, partial evidence preserved")
            break
        run = run_process(args.model, index, pilot=args.stage == "pilot")
        runs.append(run)
        print(f"process {index+1}/{processes} done, {len(run['runs'])} runs, "
              f"peak {run['mlx_peak_bytes']/1e9:.2f} GB", flush=True)
        if run["mlx_peak_bytes"] > 12 * 1024**3:
            print("ABORT memory limit")
            break

    payload = {"experiment": "E15", "stage": args.stage, "runs": runs, "aborted": aborted,
               "preregistration_sha256": "939a3c40683433e6fc2e24c4409304a4a762fbae52c7b528b6a4de1216b70a92",
               "prereg_commit": "c2c8a5931cb2c67097fed9f435c5af52c7196abe",
               "wall_seconds": time.perf_counter() - started, "environment": env}
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"E15_results_{args.stage}.json"
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str))
    print(f"\nwrote {path} ({payload['wall_seconds']:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
