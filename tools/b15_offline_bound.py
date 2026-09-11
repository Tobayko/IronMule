#!/usr/bin/env python3
"""B15 offline screen: how many `lm_head` rows survive a correct Cauchy-Schwarz bound.

Greedy decoding needs the argmax, not the logit vector. Cluster the vocabulary, bound
each cluster with `<h,c> + ||h|| * r`, evaluate clusters in bound order and stop once
the best bound can no longer beat the incumbent. The surviving token is exact by
construction; this script also checks it against the full argmax on every step.

Offline diagnosis only. It reads a real model and real hidden states, reports the
surviving fraction, and makes no performance claim: bytes, not seconds.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPTS = (
    "Explain in three sentences why unified memory changes how a laptop runs a language model.",
    "Write a short paragraph about the difference between latency and throughput.",
    "List four reasons a matrix multiplication can be memory bound.",
)


def _dequantised_head(head: nn.Module) -> mx.array:
    return mx.dequantize(
        head.weight, head.scales, head.biases, group_size=head.group_size, bits=head.bits
    ).astype(mx.float32)


def _hidden_states(model, tokenizer, prompt: str, steps: int) -> tuple[list[mx.array], list[int]]:
    """Real greedy decode; the trunk output of each step is the vector B15 would bound."""

    language = getattr(model, "language_model", model)
    tokens = tokenizer.encode(prompt)
    cache = language.make_cache()
    hidden, chosen = [], []
    ids = mx.array(tokens)[None]
    for _ in range(steps):
        out = language.model(ids, cache=cache)
        last = out[:, -1, :]
        logits = language.lm_head(last)
        token = int(mx.argmax(logits, axis=-1).item())
        mx.eval(last, logits)
        hidden.append(last[0].astype(mx.float32))
        chosen.append(token)
        if token == tokenizer.eos_token_id:
            break
        ids = mx.array([[token]])
    return hidden, chosen


def _kmeans(rows: mx.array, k: int, iterations: int, block: int, seed: int) -> mx.array:
    """Plain Lloyd iterations. Euclidean, because the bound uses a Euclidean radius."""

    count = rows.shape[0]
    mx.random.seed(seed)
    start = mx.random.randint(0, count, shape=(k,))
    centroids = rows[start]
    for _ in range(iterations):
        totals = mx.zeros((k, rows.shape[1]), dtype=mx.float32)
        counts = mx.zeros((k,), dtype=mx.float32)
        centroid_norms = mx.sum(centroids * centroids, axis=1)
        for begin in range(0, count, block):
            chunk = rows[begin : begin + block]
            scores = chunk @ centroids.T - 0.5 * centroid_norms
            nearest = mx.argmax(scores, axis=1)
            totals = totals.at[nearest].add(chunk)
            counts = counts.at[nearest].add(mx.ones(chunk.shape[0], dtype=mx.float32))
            mx.eval(totals, counts)
        empty = counts == 0
        centroids = totals / mx.maximum(counts, 1)[:, None] + centroids * empty[:, None]
        mx.eval(centroids)
    return centroids


def _assign_and_radii(
    rows: mx.array, centroids: mx.array, block: int
) -> tuple[mx.array, mx.array]:
    k = centroids.shape[0]
    centroid_norms = mx.sum(centroids * centroids, axis=1)
    labels, radii = [], mx.zeros((k,), dtype=mx.float32)
    for begin in range(0, rows.shape[0], block):
        chunk = rows[begin : begin + block]
        scores = chunk @ centroids.T - 0.5 * centroid_norms
        nearest = mx.argmax(scores, axis=1)
        gap = chunk - centroids[nearest]
        distance = mx.sqrt(mx.maximum(mx.sum(gap * gap, axis=1), 0.0))
        radii = radii.at[nearest].maximum(distance)
        labels.append(nearest)
        mx.eval(radii, nearest)
    return mx.concatenate(labels), radii


def _norm_screen(rows: mx.array, hidden: list[mx.array]) -> dict:
    """The cheapest correct bound: `<h,w> <= ||h|| * ||w||`, one scalar per row.

    Sort rows by norm, walk from the largest, stop when `||h|| * ||w||` cannot beat the
    incumbent. Reading a norm costs 4 bytes against 1280 for a 2560-wide 4-bit row.
    """

    norms = mx.sqrt(mx.sum(rows * rows, axis=1))
    order = mx.argsort(-norms)
    sorted_rows, sorted_norms = rows[order], norms[order]
    mx.eval(sorted_rows, sorted_norms)
    norm_list = sorted_norms.tolist()
    survived, matches = [], 0
    for vector in hidden:
        length = float(mx.sqrt(mx.sum(vector * vector)).item())
        logits = sorted_rows @ vector
        mx.eval(logits)
        values = logits.tolist()
        best, rows_read = -float("inf"), 0
        for index, value in enumerate(values):
            if length * norm_list[index] <= best:
                break
            if value > best:
                best = value
            rows_read += 1
        survived.append(rows_read)
        matches += int(best == float(mx.max(logits).item()))
    total = int(rows.shape[0])
    return {
        "clusters": 0,
        "bound": "norm_only",
        "steps": len(hidden),
        "vocabulary_rows": total,
        "exact_token_matches": matches,
        "surviving_rows_mean": sum(survived) / len(survived),
        "surviving_rows_max": max(survived),
        "surviving_fraction_mean": sum(survived) / len(survived) / total,
        "surviving_fraction_max": max(survived) / total,
        "row_norm_min": float(mx.min(norms).item()),
        "row_norm_median": float(mx.median(norms).item()) if hasattr(mx, "median") else None,
        "row_norm_max": float(mx.max(norms).item()),
    }


def _geometry(rows: mx.array, hidden: list[mx.array]) -> dict:
    """Why the bound does or does not bite: hidden norm against the logit gap."""

    norms = mx.sqrt(mx.sum(rows * rows, axis=1))
    gaps, tops, lengths = [], [], []
    for vector in hidden:
        logits = rows @ vector
        top = float(mx.max(logits).item())
        second = float(mx.max(mx.where(logits == top, -mx.inf, logits)).item())
        gaps.append(top - second)
        tops.append(top)
        lengths.append(float(mx.sqrt(mx.sum(vector * vector)).item()))
    return {
        "hidden_norm_mean": sum(lengths) / len(lengths),
        "top_logit_mean": sum(tops) / len(tops),
        "top_to_second_gap_mean": sum(gaps) / len(gaps),
        "row_norm_mean": float(mx.mean(norms).item()),
        "row_norm_max": float(mx.max(norms).item()),
        "cauchy_schwarz_ceiling_mean": (sum(lengths) / len(lengths))
        * float(mx.max(norms).item()),
    }


def _neighbourhood(rows: mx.array, radius: float, samples: int, seed: int) -> dict:
    """Is the required radius reachable by any clustering at all?

    A cluster of radius `r` needs its rows within `r` of one centre. Sample rows and
    count how many neighbours fall inside that ball. If the answer is one, the row
    itself, then no clustering achieves the radius and the entry closes on geometry
    rather than on this script's k-means.
    """

    mx.random.seed(seed)
    picks = mx.random.randint(0, rows.shape[0], shape=(samples,))
    probes = rows[picks]
    norms = mx.sum(rows * rows, axis=1)
    counts, nearest = [], []
    for index in range(samples):
        probe = probes[index]
        squared = norms - 2.0 * (rows @ probe) + float(mx.sum(probe * probe).item())
        distance = mx.sqrt(mx.maximum(squared, 0.0))
        counts.append(int(mx.sum(distance <= radius).item()))
        second = mx.min(mx.where(distance <= 0.0, mx.inf, distance))
        nearest.append(float(second.item()))
        mx.eval(distance)
    return {
        "required_radius": radius,
        "samples": samples,
        "rows_within_required_radius_mean": sum(counts) / len(counts),
        "rows_within_required_radius_max": max(counts),
        "nearest_neighbour_distance_mean": sum(nearest) / len(nearest),
        "nearest_neighbour_distance_min": min(nearest),
    }


def _screen(
    rows: mx.array,
    labels: mx.array,
    sizes: mx.array,
    centroids: mx.array,
    radii: mx.array,
    hidden: list[mx.array],
    margin: float,
) -> dict:
    """Evaluate clusters in bound order until the bound cannot beat the incumbent.

    Vectorised, but the same rule: with every cluster's exact maximum known, the search
    stops at the first cluster whose bound is no better than the running incumbent, and
    everything up to that point counts as read.
    """

    k = int(centroids.shape[0])
    survived, groups_read, matches, needed = [], [], 0, []
    for vector in hidden:
        norm = float(mx.sqrt(mx.sum(vector * vector)).item())
        bounds = centroids @ vector + norm * radii + margin
        logits = rows @ vector
        cluster_max = mx.full((k,), -mx.inf, dtype=mx.float32).at[labels].maximum(logits)
        order = mx.argsort(-bounds)
        ordered_bounds = bounds[order]
        ordered_max = cluster_max[order]
        ordered_sizes = sizes[order].astype(mx.int32)
        running = mx.cummax(ordered_max) if hasattr(mx, "cummax") else None
        mx.eval(ordered_bounds, ordered_max, ordered_sizes)
        bound_list, max_list, size_list = (
            ordered_bounds.tolist(),
            ordered_max.tolist(),
            ordered_sizes.tolist(),
        )
        best, rows_read, evaluated = -float("inf"), 0, 0
        for index in range(k):
            if bound_list[index] <= best:
                break
            best = max(best, max_list[index])
            rows_read += size_list[index]
            evaluated += 1
        true_max = float(mx.max(logits).item())
        matches += int(best == true_max)
        survived.append(rows_read)
        groups_read.append(evaluated)
        needed.append((true_max - float(mx.max(centroids @ vector).item())) / norm)
    total = int(rows.shape[0])
    return {
        "bound": "cluster_cauchy_schwarz",
        "steps": len(hidden),
        "vocabulary_rows": total,
        "exact_token_matches": matches,
        "surviving_rows_mean": sum(survived) / len(survived),
        "surviving_rows_max": max(survived),
        "surviving_fraction_mean": sum(survived) / len(survived) / total,
        "surviving_fraction_max": max(survived) / total,
        "clusters_evaluated_mean": sum(groups_read) / len(groups_read),
        "required_radius_mean": sum(needed) / len(needed),
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", default="mlx-community/gemma-3-4b-it-4bit")
    parser.add_argument("--clusters", type=int, nargs="+", default=[256, 1024, 4096])
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--block", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--neighbour-samples", type=int, default=256)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    started = time.monotonic()
    model, tokenizer = load(args.model)
    language = getattr(model, "language_model", model)
    head = language.lm_head
    rows = _dequantised_head(head)
    mx.eval(rows)

    hidden, tokens = [], []
    for prompt in PROMPTS:
        vectors, chosen = _hidden_states(model, tokenizer, prompt, args.steps)
        hidden.extend(vectors)
        tokens.extend(chosen)

    geometry = _geometry(rows, hidden)
    print(json.dumps(geometry, sort_keys=True))
    results = [_norm_screen(rows, hidden)]
    print(json.dumps(results[0], sort_keys=True))
    for k in args.clusters:
        centroids = _kmeans(rows, k, args.iterations, args.block, args.seed)
        labels, radii = _assign_and_radii(rows, centroids, args.block)
        sizes = (
            mx.zeros((k,), dtype=mx.float32)
            .at[labels]
            .add(mx.ones(labels.shape[0], dtype=mx.float32))
            .astype(mx.int32)
        )
        screened = _screen(rows, labels, sizes, centroids, radii, hidden, args.margin)
        screened["clusters"] = k
        screened["centroid_rows_read"] = k
        screened["effective_fraction_mean"] = (
            screened["surviving_rows_mean"] + k
        ) / screened["vocabulary_rows"]
        screened["radius_mean"] = float(mx.mean(radii).item())
        screened["radius_max"] = float(mx.max(radii).item())
        screened["cluster_size_max"] = int(mx.max(sizes).item())
        results.append(screened)
        print(json.dumps({key: value for key, value in screened.items()
                          if key != "surviving_rows_per_step"}, sort_keys=True))

    required = min(result.get("required_radius_mean", float("inf")) for result in results)
    neighbourhood = _neighbourhood(rows, required, args.neighbour_samples, args.seed)
    print(json.dumps(neighbourhood, sort_keys=True))

    revision = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    record = {
        "schema": "ironmule.b15_offline_screen.v1",
        "experiment_id": args.out.stem,
        "agent": "claude",
        "kind": "offline_bound_screen",
        "status": "measured",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "question": "what fraction of lm_head rows survives a correct Cauchy-Schwarz bound",
        "kill_criterion": "surviving fraction not below 0.25",
        "model": args.model,
        "decoding": "greedy",
        "prompts": list(PROMPTS),
        "steps_per_prompt": args.steps,
        "kmeans_iterations": args.iterations,
        "seed": args.seed,
        "bound_margin": args.margin,
        "device": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "mlx": mx.__version__ if hasattr(mx, "__version__") else None,
        },
        "git_revision": revision,
        "wall_seconds": time.monotonic() - started,
        "tokens_generated": len(hidden),
        "geometry": geometry,
        "neighbourhood": neighbourhood,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"written": str(args.out), "wall_seconds": record["wall_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
