"""E12: try to break E9's plan-internal bit identity at Gemma 3's 1024 sliding window.

Preregistered at commit 750be38, research/raw/E12_preregistration.md
(SHA-256 5d0dbc3ccc66084a237f9bf2af051f0643d179a3a05afdbb744bba30dff4890e).

Comparison A  chunked-no-reuse  vs  chunked-reuse   -- identical plan, decides the result
Comparison B  single-shot       vs  chunked-no-reuse -- different plans, documentation only

Equality is tested on raw bits, never on values: mx.array_equal reports -0.0 == 0.0
as true while the bit patterns differ, which would let a real difference through.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import mlx.core as mx
import numpy as np

from ironmule import bench
from ironmule.runtime import FixedKVCache, Knobs, _empty_fixed_state, _fixed_state_from_standard, _leaves, _project, _trunk
from ironmule.tune import DEFAULT_MODEL, _eos_ids, gpu_busy, load_engine

RAW = Path(__file__).resolve().parent / "raw"

# --- frozen constants from the preregistration -------------------------------
SLIDING_WINDOW = 1024
MAX_NEW_TOKENS = 32
WARMUP = 2
PREFIX_LENGTHS = [276, 768, 870, 896, 1000, 1023, 1024, 1025, 1048, 1152, 1280, 1536, 2048]
CONFIRM_LENGTHS = [1023, 1024, 1025, 1152, 1280, 2048]
STAGE_WALL_LIMIT_S = 45 * 60

KNOBS = Knobs(compiled_fixed_cache=True, fused_argmax=False, head_skip_prefill=True,
              fuse_projections=True)


# --- bit-exact comparison primitives -----------------------------------------

def _bits(x: mx.array) -> mx.array:
    """Reinterpret as unsigned integers so comparison is on bits, not on values."""
    return x.view({2: mx.uint16, 4: mx.uint32, 1: mx.uint8, 8: mx.uint64}[x.dtype.size])


def bits_equal(a: mx.array, b: mx.array) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(mx.array_equal(_bits(a), _bits(b)).item())


def bit_hash(arrays: list[mx.array]) -> str:
    digest = hashlib.sha256()
    for arr in arrays:
        digest.update(np.asarray(_bits(arr)).tobytes())
    return digest.hexdigest()


def first_difference(a: mx.array, b: mx.array) -> dict:
    """Locate and describe the first differing element. Only called on failure."""
    ba, bb = _bits(a).reshape(-1), _bits(b).reshape(-1)
    diff = mx.argmax((ba != bb).astype(mx.uint32)).item()
    fa, fb = a.reshape(-1).astype(mx.float32), b.reshape(-1).astype(mx.float32)
    delta = mx.abs(fa - fb)
    scale = mx.maximum(mx.abs(fa), mx.array(1e-30, mx.float32))
    return {
        "first_differing_flat_index": int(diff),
        "value_baseline": float(fa[diff].item()),
        "value_candidate": float(fb[diff].item()),
        "bits_baseline": int(ba[diff].item()),
        "bits_candidate": int(bb[diff].item()),
        "max_abs_diff": float(mx.max(delta).item()),
        "max_rel_diff": float(mx.max(delta / scale).item()),
        "differing_elements": int(mx.sum((ba != bb).astype(mx.uint32)).item()),
    }


def top_two(logits: mx.array) -> tuple[int, int, float]:
    f = logits.astype(mx.float32).reshape(-1)
    order = mx.argsort(-f)
    i1, i2 = int(order[0].item()), int(order[1].item())
    return i1, i2, float((f[i1] - f[i2]).item())


# --- workload ----------------------------------------------------------------

NATURAL_BLOCK = """You are the execution planner for a local inference research programme. \
Your task is to read the measured evidence for a set of candidate optimisations and to \
select exactly one of them for the next experiment cycle. The hardware is an Apple M1 Max \
with thirty-two gigabytes of unified memory and thirty-two graphics cores, running a \
four-bit quantised language model through the MLX framework. The model weights, the model \
architecture and the quantisation scheme are fixed for the entire programme and may never \
be altered in order to produce a performance gain; only the execution layer beneath the \
model is permitted to change. Every candidate you consider has already been measured on \
this exact machine under alternating arm order across several fresh processes, and every \
reported effect carries a bootstrap confidence interval derived from ten thousand paired \
resamples of the underlying medians. When you weigh the candidates against one another you \
must apply the following selection policy without exception and in the order given. First, \
prefer the candidate whose confirmed end to end effect is largest and which additionally \
closes a workload gap that the programme has not previously covered, because an effect that \
only repeats existing coverage buys less than one that extends it. Second, never select a \
candidate whose only supporting evidence is a diagnostic upper bound, because an upper \
bound describes what an ideal implementation might achieve and is not itself an \
implementation that can be shipped or measured again. Third, never select a candidate that \
is blocked on an authorisation you do not hold, because scheduling work that cannot legally \
begin wastes an entire cycle. Fourth, never select a candidate whose correctness gate has \
already failed, unless the evidence in front of you states explicitly that the failure was \
diagnosed and resolved, since a faster answer that is a different answer is not a faster \
answer at all. Fifth, when two candidates remain otherwise indistinguishable, prefer the one \
with the lower implementation risk and the shorter expected measurement cost, so that the \
cycle produces evidence sooner rather than later. You must also respect the reporting \
contract that governs this programme. Speed is never permitted to be purchased with \
unexplained changes in output, correctness gates are always evaluated before performance is \
considered at all, warm up passes are never mixed into steady state measurements, negative \
results are recorded permanently rather than discarded, and no performance claim is ever \
made without the raw samples that support it being written to disk first. """

SYNTHETIC_WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]


def build_preamble(kind: str, target_tokens: int, tok) -> str:
    """A preamble that just exceeds `target_tokens`, built per length.

    Grown in small units so the overshoot stays under ~50 tokens. Sizing it to the
    longest length in the grid instead would make every short-prefix case carry a
    2000-token prompt, which inflates cost and destroys the prefix share the case is
    supposed to represent.
    """
    if kind == "natural":
        units = [s.strip() + ". " for s in NATURAL_BLOCK.split(". ") if s.strip()]
    else:
        # Deterministic, token-controlled: a fixed rotation of ordinary vocabulary words.
        units = [" ".join(SYNTHETIC_WORDS[(i + k) % len(SYNTHETIC_WORDS)]
                          for k in range(24)) + " " for i in range(len(SYNTHETIC_WORDS))]
    text, index = "", 0
    while len(tok.encode(text, add_special_tokens=False)) < target_tokens:
        text += units[index % len(units)]
        index += 1
    return text


def load_corpus() -> tuple[str, list[str]]:
    """The frozen twelve-request corpus, exactly as used in E8, E9 and E10."""
    src = (Path(__file__).resolve().parent / "e8_reuse_risk.py").read_text()
    ns = {"statistics": statistics}
    exec(compile(src[src.index('PREFIX = """'):src.index("MAX_TOKENS = 32")], "corpus", "exec"), ns)
    return ns["PREFIX"], ns["SUFFIXES"]


# --- arms --------------------------------------------------------------------

class Harness:
    def __init__(self, model_id: str):
        self.engine, self.tok = load_engine(model_id, KNOBS)
        self.model = self.engine.model
        self.trunk = _trunk(self.engine.model)
        self.eos = _eos_ids(self.tok)

    def render(self, text: str) -> list[int]:
        rendered = self.tok.apply_chat_template([{"role": "user", "content": text}],
                                                tokenize=False, add_generation_prompt=True)
        return list(self.tok.encode(rendered, add_special_tokens=False))

    def empty_state(self, capacity: int):
        probe = self.model.make_cache()
        self.trunk(mx.array([[2]]), cache=probe)
        mx.eval([c.keys for c in probe])
        return _empty_fixed_state(capacity, probe)

    def feed(self, state, ids: list[int], capacity: int):
        caches = [FixedKVCache(layer, state["position"], capacity) for layer in state["layers"]]
        hidden = self.trunk(mx.array(ids)[None, :], cache=caches)
        used = int(state["position"]["offset"].item()) + len(ids)
        return ({"position": {"offset": mx.array(used, dtype=mx.int32)},
                 "layers": [{"keys": c.keys, "values": c.values} for c in caches]}, hidden)

    def project(self, hidden):
        logits = _project(self.model, hidden[:, -1:, :])
        return logits[:, -1, :]

    def prefill_chunked(self, ids: list[int], split: int, capacity: int, snapshot=None):
        """Plan `chunked@L`. Chunk one either recomputed or restored; chunk two identical."""
        started = time.perf_counter_ns()
        if snapshot is None:
            state, _ = self.feed(self.empty_state(capacity), ids[:split], capacity)
            mx.eval(*_leaves(state))
        else:
            state = {"position": {"offset": mx.array(split, dtype=mx.int32)},
                     "layers": [{"keys": l["keys"], "values": l["values"]} for l in snapshot]}
        state, hidden = self.feed(state, ids[split:], capacity)
        logits = self.project(hidden)
        mx.eval(logits, *_leaves(state))
        mx.synchronize()
        return state, logits, time.perf_counter_ns() - started

    def prefill_single_shot(self, ids: list[int], capacity: int):
        """Plan `single_shot`. One forward into a standard cache, then converted."""
        started = time.perf_counter_ns()
        cache = self.model.make_cache()
        hidden = self.trunk(mx.array(ids)[None, :], cache=cache)
        logits = self.project(hidden)
        mx.eval(logits)
        state = _fixed_state_from_standard(cache, len(ids), capacity)
        mx.eval(*_leaves(state))
        mx.synchronize()
        return state, logits, time.perf_counter_ns() - started

    def snapshot(self, ids: list[int], split: int, capacity: int):
        started = time.perf_counter_ns()
        state, _ = self.feed(self.empty_state(capacity), ids[:split], capacity)
        mx.eval(*_leaves(state))
        mx.synchronize()
        return ([{"keys": l["keys"], "values": l["values"]} for l in state["layers"]],
                time.perf_counter_ns() - started)


def valid_kv(state, offset: int) -> list[mx.array]:
    out = []
    for layer in state["layers"]:
        out.append(layer["keys"][..., :offset, :])
        out.append(layer["values"][..., :offset, :])
    return out


def mask_signature(state, capacity: int, n_tokens: int) -> dict:
    """Masks and admitted-position counts, for both the global and the sliding layers."""
    cache = FixedKVCache(state["layers"][0], state["position"], capacity)
    glob = cache.make_mask(n_tokens)
    slide = cache.make_mask(n_tokens, window_size=SLIDING_WINDOW)
    mx.eval(glob, slide)
    return {
        "global_admitted": int(mx.sum(glob.astype(mx.uint32)).item()),
        "sliding_admitted": int(mx.sum(slide.astype(mx.uint32)).item()),
        "global_hash": bit_hash([glob]),
        "sliding_hash": bit_hash([slide]),
        "window_clips": bool(mx.sum(glob.astype(mx.uint32)).item()
                             != mx.sum(slide.astype(mx.uint32)).item()),
    }


# --- one case ----------------------------------------------------------------

def run_case(h: Harness, prefix_kind: str, length: int, preamble: str,
             corpus: list[str], model_id: str) -> dict:
    prompts = [h.render(preamble + "\n\nMeasured evidence for this request:\n" + s)
               for s in corpus]
    prefix_ids = prompts[0][:length]
    for i, ids in enumerate(prompts):
        if ids[:length] != prefix_ids:
            return {"aborted": "tokenisation_gate", "request": i,
                    "prefix_kind": prefix_kind, "prefix_length": length}
        if len(ids) <= length:
            return {"aborted": "prompt_shorter_than_prefix", "request": i,
                    "prefix_kind": prefix_kind, "prefix_length": length}
    if len(prefix_ids) != length:
        return {"aborted": "realised_prefix_length", "prefix_kind": prefix_kind,
                "prefix_length": length, "realised": len(prefix_ids)}

    max_len = max(len(p) for p in prompts)
    capacity = ((max_len + MAX_NEW_TOKENS + 63) // 64) * 64

    snapshot, snapshot_ns = h.snapshot(prompts[0], length, capacity)
    body = h.engine._body(capacity, 1)

    case = {
        "prefix_kind": prefix_kind, "prefix_length": length, "realised_prefix_tokens": len(prefix_ids),
        "capacity": capacity, "sliding_window": SLIDING_WINDOW,
        "prompt_tokens": [len(p) for p in prompts],
        "suffix_tokens": [len(p) - length for p in prompts],
        "window_crossed_at_prefill": max_len > SLIDING_WINDOW,
        "window_crossed_only_in_decode": max_len <= SLIDING_WINDOW
                                         and max_len + MAX_NEW_TOKENS > SLIDING_WINDOW,
        "snapshot_build_ns": snapshot_ns,
        "aligned_64": length % 64 == 0, "aligned_256": length % 256 == 0,
        "requests": [],
    }

    for index, ids in enumerate(prompts):
        cb_state, cb_logits, cb_ns = h.prefill_chunked(ids, length, capacity)
        cr_state, cr_logits, cr_ns = h.prefill_chunked(ids, length, capacity, snapshot=snapshot)
        ss_state, ss_logits, ss_ns = h.prefill_single_shot(ids, capacity)

        offset = len(ids)
        rec = {
            "request": index, "prompt_tokens": offset,
            "chunked_cold_ttft_ns": cb_ns, "chunked_reuse_ttft_ns": cr_ns,
            "single_shot_ttft_ns": ss_ns,
            "A_prefill_logits_bit_equal": bits_equal(cb_logits, cr_logits),
            "A_prefill_kv_bit_equal": all(
                bits_equal(x, y) for x, y in zip(valid_kv(cb_state, offset), valid_kv(cr_state, offset))),
            "A_prefill_offset_equal": int(cb_state["position"]["offset"].item())
                                      == int(cr_state["position"]["offset"].item()),
            "mask_baseline": mask_signature(cb_state, capacity, 1),
            "mask_candidate": mask_signature(cr_state, capacity, 1),
            "B_prefill_max_abs_diff": float(mx.max(mx.abs(
                ss_logits.astype(mx.float32) - cb_logits.astype(mx.float32))).item()),
            "B_prefill_bit_equal": bits_equal(ss_logits, cb_logits),
            "steps": [], "failures": [],
        }
        rec["A_prefill_masks_equal"] = rec["mask_baseline"] == rec["mask_candidate"]

        if not rec["A_prefill_logits_bit_equal"]:
            rec["failures"].append({"where": "prefill_logits", "step": 0,
                                    **first_difference(cb_logits, cr_logits)})
        if not rec["A_prefill_kv_bit_equal"]:
            for layer_index, (x, y) in enumerate(zip(valid_kv(cb_state, offset),
                                                     valid_kv(cr_state, offset))):
                if not bits_equal(x, y):
                    rec["failures"].append({
                        "where": "prefill_kv", "layer": layer_index // 2,
                        "tensor": "keys" if layer_index % 2 == 0 else "values",
                        "token_position": None, "step": 0, **first_difference(x, y)})
                    break

        cb_tokens, cr_tokens, ss_tokens = [], [], []
        cb_done = cr_done = ss_done = False
        step = 0
        while step < MAX_NEW_TOKENS and not (cb_done and cr_done):
            equal = bits_equal(cb_logits, cr_logits)
            cb_top = int(mx.argmax(cb_logits.astype(mx.float32), -1).item())
            cr_top = int(mx.argmax(cr_logits.astype(mx.float32), -1).item())
            ss_top = int(mx.argmax(ss_logits.astype(mx.float32), -1).item())
            b_diff = float(mx.max(mx.abs(ss_logits.astype(mx.float32)
                                         - cb_logits.astype(mx.float32))).item())
            rec["steps"].append({
                "step": step, "A_logits_bit_equal": equal,
                "token_baseline": cb_top, "token_candidate": cr_top,
                "offset_baseline": int(cb_state["position"]["offset"].item()),
                "offset_candidate": int(cr_state["position"]["offset"].item()),
                "B_max_abs_diff": b_diff, "token_single_shot": ss_top,
            })
            if not equal:
                i1, i2, gap = top_two(cb_logits)
                rec["failures"].append({
                    "where": "decode_logits", "step": step, "layer": None, "tensor": "logits",
                    "token_position": offset + step, "affected_token_decision": cb_top != cr_top,
                    "leading_pair": [i1, i2], "logit_gap": gap,
                    **first_difference(cb_logits, cr_logits)})

            if not cb_done:
                cb_tokens.append(cb_top)
                cb_done = cb_top in h.eos
            if not cr_done:
                cr_tokens.append(cr_top)
                cr_done = cr_top in h.eos
            if not ss_done:
                ss_tokens.append(ss_top)
                ss_done = ss_top in h.eos
            if cb_done and cr_done:
                break

            out = body(mx.array([[cb_top]]), cb_state); cb_state, cb_logits = out[1], out[0][:, -1, :]
            out = body(mx.array([[cr_top]]), cr_state); cr_state, cr_logits = out[1], out[0][:, -1, :]
            out = body(mx.array([[ss_top]]), ss_state); ss_state, ss_logits = out[1], out[0][:, -1, :]
            mx.eval(cb_logits, cr_logits, ss_logits,
                    *_leaves(cb_state), *_leaves(cr_state), *_leaves(ss_state))
            mx.synchronize()
            step += 1

        final = int(cb_state["position"]["offset"].item())
        cb_kv, cr_kv = valid_kv(cb_state, final), valid_kv(cr_state, final)
        rec["A_final_kv_bit_equal"] = all(bits_equal(x, y) for x, y in zip(cb_kv, cr_kv))
        if not rec["A_final_kv_bit_equal"]:
            for layer_index, (x, y) in enumerate(zip(cb_kv, cr_kv)):
                if not bits_equal(x, y):
                    rec["failures"].append({
                        "where": "final_kv", "layer": layer_index // 2,
                        "tensor": "keys" if layer_index % 2 == 0 else "values",
                        "step": step, **first_difference(x, y)})
                    break

        rec["tokens_baseline"] = cb_tokens
        rec["tokens_candidate"] = cr_tokens
        rec["tokens_single_shot"] = ss_tokens
        rec["A_tokens_equal"] = cb_tokens == cr_tokens
        rec["A_token_count_equal"] = len(cb_tokens) == len(cr_tokens)
        rec["A_stop_reason_equal"] = (
            (cb_tokens[-1] if cb_tokens else None) in h.eos) == (
            (cr_tokens[-1] if cr_tokens else None) in h.eos)
        rec["stop_reason_baseline"] = "eos" if cb_tokens and cb_tokens[-1] in h.eos else "length"
        rec["stop_reason_candidate"] = "eos" if cr_tokens and cr_tokens[-1] in h.eos else "length"
        rec["B_tokens_equal"] = cb_tokens == ss_tokens
        rec["B_max_abs_diff_overall"] = max([rec["B_prefill_max_abs_diff"]]
                                            + [s["B_max_abs_diff"] for s in rec["steps"]])

        rec["exact_equal"] = all([
            rec["A_prefill_logits_bit_equal"], rec["A_prefill_kv_bit_equal"],
            rec["A_prefill_offset_equal"], rec["A_prefill_masks_equal"],
            rec["A_final_kv_bit_equal"], rec["A_tokens_equal"],
            rec["A_token_count_equal"], rec["A_stop_reason_equal"],
            all(s["A_logits_bit_equal"] for s in rec["steps"]),
        ])
        if rec["exact_equal"]:
            rec["hash_baseline"] = bit_hash(cb_kv)
            rec["hash_candidate"] = bit_hash(cr_kv)
            rec["hashes_equal"] = rec["hash_baseline"] == rec["hash_candidate"]
        case["requests"].append(rec)
        mx.clear_cache()

    case["A_pass"] = all(r["exact_equal"] for r in case["requests"])
    case["A_failures"] = sum(len(r["failures"]) for r in case["requests"])
    case["B_max_abs_diff"] = max(r["B_max_abs_diff_overall"] for r in case["requests"])
    case["B_requests_with_different_tokens"] = sum(not r["B_tokens_equal"] for r in case["requests"])
    case["mlx_peak_bytes"] = mx.get_peak_memory()
    return case


# --- driver ------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--lengths", default="")
    parser.add_argument("--types", default="natural,synthetic")
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

    lengths = [int(x) for x in args.lengths.split(",")] if args.lengths else PREFIX_LENGTHS
    kinds = args.types.split(",")

    h = Harness(args.model)
    _, corpus = load_corpus()
    preambles = {(k, n): build_preamble(k, n, h.tok) for k in kinds for n in lengths}

    # warmup, on the shortest configuration, before anything is measured
    warm_ids = h.render(preambles[(kinds[0], min(lengths))]
                        + "\n\nMeasured evidence for this request:\n" + corpus[0])
    warm_cap = ((len(warm_ids) + MAX_NEW_TOKENS + 63) // 64) * 64
    for _ in range(WARMUP):
        h.prefill_chunked(warm_ids, min(lengths), warm_cap)
        h.prefill_single_shot(warm_ids, warm_cap)

    started = time.perf_counter()
    cases, aborted = [], None
    gate = bench.MemoryGate()  # R11: swap, not a byte count
    for kind in kinds:
        for length in lengths:
            if time.perf_counter() - started > STAGE_WALL_LIMIT_S:
                aborted = "stage_wall_limit"
                break
            case = run_case(h, kind, length, preambles[(kind, length)], corpus, args.model)
            cases.append(case)
            if case.get("aborted"):
                aborted = case["aborted"]
                break
            flag = "PASS" if case["A_pass"] else "FAIL"
            print(f"{kind:9s} L={length:5d} cap={case['capacity']:5d} "
                  f"win_prefill={str(case['window_crossed_at_prefill']):5s} "
                  f"A={flag} A_failures={case['A_failures']:3d} "
                  f"B_maxdiff={case['B_max_abs_diff']:7.4f} "
                  f"B_difftok={case['B_requests_with_different_tokens']:2d}/12 "
                  f"peak={case['mlx_peak_bytes']/1e9:.2f}GB", flush=True)
            if not case["A_pass"]:
                print("  -> Comparison A failed; stopping the broad matrix per plan section 11.",
                      flush=True)
                aborted = "comparison_a_failure"
                break
            stop = gate.check(len(cases) - 1, case["mlx_peak_bytes"])
            if stop:
                print(f"  -> ABORT {stop}", flush=True)
                aborted = f"memory_gate: {stop}"
                break
        if aborted:
            break

    payload = {
        "experiment": "E12", "tag": args.tag, "stage_aborted": aborted,
        "preregistration_sha256": "5d0dbc3ccc66084a237f9bf2af051f0643d179a3a05afdbb744bba30dff4890e",
        "prereg_commit": "750be38377d11b1285b54f48e6403c9959e78ae9",
        "lengths": lengths, "types": kinds, "corpus_size": len(corpus),
        "sliding_window": SLIDING_WINDOW, "max_new_tokens": MAX_NEW_TOKENS,
        "wall_seconds": time.perf_counter() - started,
        "pid": os.getpid(), "cases": cases,
    }
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"E12_results_{args.tag}.json"
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str))
    print(f"\nwrote {path}  ({time.perf_counter()-started:.0f}s, aborted={aborted})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
