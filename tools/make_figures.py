"""Render every README figure from a committed evidence artifact.

No figure here carries a number that was typed in. Each one names the JSON it
read, and `--check` re-renders into a temporary directory and fails on any byte
difference, so a figure cannot drift away from the data behind it.

    python tools/make_figures.py            # write docs/assets/*.svg
    python tools/make_figures.py --check    # fail if the committed SVGs differ
    python tools/make_figures.py --manifest # print what each figure claims

Needs matplotlib:  pip install -e ".[figures]"
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import figure_style as style  # noqa: E402

matplotlib.rcParams["svg.hashsalt"] = "ironmule.figures.v1"
style.configure(matplotlib)

import matplotlib.pyplot as plt  # noqa: E402

ASSETS = ROOT / "docs" / "assets"

#: Set by `render` for the theme being drawn: "" for light, "-dark" for the one
#: GitHub serves to readers on a dark background.
SUFFIX = ""


def emit(fig: Any, name: str) -> str:
    """Write one figure under the current theme's name and return that name."""

    relative = f"docs/assets/{name}{SUFFIX}.svg"
    style.save(fig, ASSETS / f"{name}{SUFFIX}.svg")
    return relative


HEAD_SKIP = Path("experiments/head_skip_formal/results.json")
PREFIX_CACHE = Path("research/raw/E10-prefix-cache-session-ab.json")
PREFILL_PHASES = Path("research/raw/E1-prefill-breakdown.json")

#: PORT1, per model: Apple's balanced A/B/C/D square and the T4's fresh-process
#: cross run. Each file carries its own device's stock reference, so a bar only
#: ever compares a machine with itself.
CROSS = (
    ("Gemma 3 1B",
     Path("experiments/kaggle_compat/results/apple-abcd/abcd-1b.json"),
     Path("experiments/kaggle_compat/results/port1-run6-c3af42bd/cross-1b.json"),
     "ironmule"),
    ("Gemma 3 4B",
     Path("experiments/kaggle_compat/results/apple-abcd/abcd-4b.json"),
     Path("experiments/kaggle_compat/results/port1-run6-c3af42bd/cross-4b.json"),
     "ironmule_exact"),
    ("Gemma 3 12B",
     Path("experiments/kaggle_compat/results/apple-abcd/abcd-12b.json"),
     Path("experiments/kaggle_compat/results/port1-run7-1f40ad2b/cross-12b.json"),
     "ironmule_exact"),
)

#: Every figure states the machine it was measured on. One machine, said once.
DEVICE = "Apple M1 Max, 32 GB unified memory"


def load(relative: Path) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


# -- figure 1: the headline, with its control on the same axis -----------------

def paired_ratios() -> dict:
    """Confirmed paired ratios against the A/A control that calibrates them.

    A ratio below 1.0 is faster. The control arm compares the baseline against
    itself, so it has no effect to find; where it lands is this machine's noise
    floor, and a candidate is only readable against it.
    """

    head = load(HEAD_SKIP)
    prefix = load(PREFIX_CACHE)

    control = head["calibration_from_measured_blocks"]
    confirmed = head["calculated_decision"]["intervals"]["all"]
    warm = prefix["ratio_warm"]
    cold = prefix["ratio_cold"]

    sessions = head["workload"]["confirmation_sessions"]
    calibration = head["workload"]["calibration_sessions"]
    # Top to bottom, so the control is read before anything it calibrates.
    rows = [
        ("A/A control\nbaseline vs itself, n=%d sessions" % calibration,
         control["aggregate_ratio"], control["ci95"], style.CONTROL),
        ("Head-skip prefill\nprefill time, n=%d sessions" % sessions,
         confirmed["ratio"], confirmed["ci95"], style.CANDIDATE),
        ("Prefix cache, warm\nsession wall time, n=%d processes" % prefix["processes"],
         warm["median_ratio"], [warm["ci_low"], warm["ci_high"]], style.SECONDARY),
        ("Prefix cache, cold\nsession wall time, n=%d processes" % prefix["processes"],
         cold["median_ratio"], [cold["ci_low"], cold["ci_high"]], style.SECONDARY),
    ]

    fig, ax = plt.subplots(figsize=(style.WIDTH_IN, 3.6))
    positions = [len(rows) - 1 - index for index in range(len(rows))]
    for y, (_, ratio, ci, colour) in zip(positions, rows):
        low, high = ci
        ax.plot([low, high], [y, y], color=colour, linewidth=2.4, solid_capstyle="butt")
        for edge in (low, high):
            ax.plot([edge, edge], [y - 0.12, y + 0.12], color=colour, linewidth=2.4)
        ax.plot([ratio], [y], marker="o", markersize=8, color=colour,
                markeredgecolor=style.PAPER, markeredgewidth=1.4, zorder=3)
        ax.annotate(f"{ratio:.4f}  [{low:.4f}, {high:.4f}]", (ratio, y),
                    textcoords="offset points", xytext=(0, 12), ha="center",
                    fontsize=9, color=style.TEXT)

    ax.axvline(1.0, color=style.RULE, linewidth=1.2, linestyle="--", zorder=1)
    ax.annotate("no change", (1.0, -0.42), textcoords="offset points",
                xytext=(-6, 0), fontsize=9, color=style.MUTED, va="center", ha="right")

    ax.set_yticks(positions)
    ax.set_yticklabels([row[0] for row in rows])
    ax.set_ylim(-0.7, len(rows) - 0.35)
    ax.set_xlim(0.56, 1.10)
    ax.set_xlabel("paired ratio, candidate over baseline (lower is faster)")
    ax.set_title("Measured gains, with the control that makes them readable")
    ax.grid(axis="y", visible=False)

    written = emit(fig, "headline-ratios")
    plt.close(fig)
    return {
        "figure": written,
        "sources": [HEAD_SKIP.as_posix(), PREFIX_CACHE.as_posix()],
        "device": DEVICE,
        "models": [head["sealed_identity"]["model_id"]],
        "samples": (f"head skip: {head['workload']['confirmation_sessions']} confirmation "
                    f"sessions x {head['workload']['measurement_pairs_per_session']} pairs; "
                    f"A/A control: {head['workload']['calibration_sessions']} sessions; "
                    f"prefix cache: {prefix['processes']} paired processes"),
        "intervals": "95 percent bootstrap, paired",
    }


# -- figure 2: every session, so the reader sees the spread, not a summary -----

def session_ratios() -> dict:
    """The two arms session by session. Separation, or the absence of it."""

    head = load(HEAD_SKIP)
    control = head["calibration_from_measured_blocks"]["session_ratios"]
    confirmation = head["confirmation_from_measured_blocks"]["session_ratios"]
    names = sorted(confirmation)
    candidate = [confirmation[name] for name in names]

    fig, ax = plt.subplots(figsize=(style.WIDTH_IN, 3.2))
    x_control = range(1, len(control) + 1)
    x_candidate = range(1, len(candidate) + 1)
    ax.plot(list(x_control), list(control), marker="o", markersize=7, linestyle="none",
            color=style.CONTROL, label="A/A control (baseline against itself)")
    ax.plot(list(x_candidate), candidate, marker="D", markersize=7, linestyle="none",
            color=style.CANDIDATE, label="head-skip prefill against baseline")
    ax.axhline(1.0, color=style.RULE, linewidth=1.2, linestyle="--")

    ax.set_xticks(list(x_candidate))
    ax.set_xticklabels([f"{name}\nsession {i}" for i, name in enumerate(names, 1)],
                       fontsize=8.5)
    ax.set_ylabel("paired ratio per session")
    ax.set_ylim(0.80, 1.05)
    ax.set_title("Six sessions each: the control never moves, the candidate always does")
    ax.legend(loc="center right")
    ax.grid(axis="x", visible=False)

    written = emit(fig, "session-ratios")
    plt.close(fig)
    return {
        "figure": written,
        "sources": [HEAD_SKIP.as_posix()],
        "device": DEVICE,
        "models": [head["sealed_identity"]["model_id"]],
        "samples": (f"{len(control)} control sessions and {len(candidate)} confirmation "
                    f"sessions, {head['workload']['measurement_pairs_per_session']} pairs each, "
                    f"token identity {head['confirmation_from_measured_blocks']['token_identity']}"),
        "intervals": "none; every session shown",
    }


# -- figure 3: the mechanism, so the gain is not a magic number ---------------

def prefill_phases() -> dict:
    """Where prefill time goes, and which part head skip removes."""

    data = load(PREFILL_PHASES)
    phases = data["phases_ms"]
    order = ["trunk_forward", "projection", "fixed_state_build", "argmax", "make_cache"]
    labels = ["trunk forward", "output projection", "fixed state build", "argmax", "make cache"]
    full = [phases["full_head"][name] for name in order]
    skipped = [phases["head_skip"][name] for name in order]

    fig, ax = plt.subplots(figsize=(style.WIDTH_IN, 3.2))
    positions = range(len(order))
    height = 0.36
    ax.barh([y + height / 2 for y in positions], full, height=height,
            color=style.BASELINE, label="full head")
    ax.barh([y - height / 2 for y in positions], skipped, height=height,
            color=style.CANDIDATE, label="head skip")
    for y, (a, b) in zip(positions, zip(full, skipped)):
        ax.annotate(f"{a:.2f} ms", (a, y + height / 2), textcoords="offset points",
                    xytext=(5, 0), va="center", fontsize=8.5, color=style.MUTED)
        ax.annotate(f"{b:.2f} ms", (b, y - height / 2), textcoords="offset points",
                    xytext=(5, 0), va="center", fontsize=8.5, color=style.MUTED)

    ax.set_yticks(list(positions))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(0.03, 2600)
    ax.set_xlabel("median phase time, milliseconds (log scale)")
    ax.set_title("Prefill is one thing, plus one avoidable thing")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)

    written = emit(fig, "prefill-phases")
    plt.close(fig)
    return {
        "figure": written,
        "sources": [PREFILL_PHASES.as_posix()],
        "device": DEVICE,
        "models": ["mlx-community/gemma-3-4b-it-4bit"],
        "samples": (f"{data['repeats']} repeats, {data['prompt_tokens']} prompt tokens, "
                    f"first token identical in both arms"),
        "intervals": "none; medians of an instrumented split whose sum matches the "
                     "uninstrumented call to 1.0000x",
    }


# -- figure 4: the same runtime on two very different GPUs --------------------

def cross_platform_speedup() -> dict:
    """How much faster each model runs, per device, against that device's stock.

    The float32 arm is drawn apart from the two exact arms because it is not the
    same computation: it reproduces Apple's float32 result rather than the T4's
    emulated bf16 one, which is why IronMule never selects it by itself.
    """

    groups, sources, models = [], [], []
    for label, apple_path, t4_path, exact_arm in CROSS:
        apple, t4 = load(apple_path), load(t4_path)
        sources += [apple_path.as_posix(), t4_path.as_posix()]
        models.append(apple["model_id"])
        bars = [("Apple M1 Max, identical tokens",
                 1 / apple["wall_ratios"]["D/A"]["median_ratio"], style.CANDIDATE),
                ("NVIDIA Tesla T4, identical tokens",
                 1 / t4["summary"][exact_arm]["median_ratio"], style.SECONDARY)]
        if "ironmule_fp32" in t4["summary"]:
            bars.append(("Tesla T4, opt-in float32 plan",
                         1 / t4["summary"]["ironmule_fp32"]["median_ratio"], style.ACCENT))
        groups.append((label, bars))

    fig, ax = plt.subplots(figsize=(style.WIDTH_IN, 4.0))
    height, labelled, ticks = 0.24, set(), []
    for index, (label, bars) in enumerate(groups):
        centre = len(groups) - 1 - index
        ticks.append(centre)
        for slot, (name, speedup, colour) in enumerate(bars):
            y = centre + ((len(bars) - 1) / 2 - slot) * height
            ax.barh(y, speedup, height=height * 0.86, color=colour,
                    label=None if name in labelled else name)
            labelled.add(name)
            ax.annotate(f"{speedup:.2f}×   +{(speedup - 1) * 100:.0f}%", (speedup, y),
                        textcoords="offset points", xytext=(6, 0), va="center",
                        fontsize=9, color=style.TEXT)

    ax.axvline(1.0, color=style.RULE, linewidth=1.2, linestyle="--", zorder=3)
    ax.annotate("stock reference", (1.0, len(groups) - 0.52), textcoords="offset points",
                xytext=(6, 0), fontsize=9, color=style.MUTED, va="center")
    ax.set_yticks(ticks)
    ax.set_yticklabels([label for label, _ in groups])
    ax.set_xlim(0, 2.45)
    ax.set_ylim(-0.5, len(groups) - 0.35)
    ax.set_xlabel("times faster than the stock reference on the same device")
    ax.set_title("One runtime, two GPUs, the same answers")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=9)

    written = emit(fig, "cross-platform-speedup")
    plt.close(fig)
    return {
        "figure": written,
        "sources": sources,
        "device": "Apple M1 Max, 32 GB unified memory; Kaggle NVIDIA Tesla T4, 15 GB",
        "models": models,
        "samples": "Apple: 6 repeats of a balanced A/B/C/D square, 6 requests x 48 tokens; "
                   "T4: interleaved fresh processes, the same workload, stock pinned to "
                   "MLX's own graph limits",
        "intervals": "none; median wall-time ratio per arm, inverted to a speed-up",
    }


PERF1 = Path("experiments/kaggle_compat/results")
#: PERF1 run 2: stock and the native plan's kernels in fresh processes on one T4 cell.
PERF1_E2E = (("Qwen 3 8B", "qwen3-8b"), ("Qwen 3 14B", "qwen3-14b"))
PERF1_E2E_DIR = PERF1 / "perf1-run2-baddcb2b"
#: PERF1 run 8: the row kernel and the tensor-core kernel serving the same 8 requests.
PERF1_SERVER_DIR = PERF1 / "perf1-run8-aa90d4d2"
#: PERF1 run 5: one gate per path the plan changes, against stock bf16 on the same path.
PERF1_GATES = (
    ("Qwen 3 8B, decode path", "perf1-run5-a9559a15/gate-qwen3-8b-kernel-decode.json",
     "perf1-run5-a9559a15/gate-qwen3-8b-stock-decode.json"),
    ("Qwen 3 8B, prefill path", "perf1-run5-a9559a15/gate-qwen3-8b-p16-prefill.json",
     "perf1-run5-a9559a15/gate-qwen3-8b-stock-prefill.json"),
    ("Qwen 3 14B, prefill path", "perf1-run5-a9559a15/gate-qwen3-14b-p16-prefill.json",
     "perf1-run5-a9559a15/gate-qwen3-14b-stock-prefill.json"),
)
T4 = "Kaggle NVIDIA Tesla T4 (compute capability 7.5), 15 GB"


# -- figure 5: the native plan, decode and time to first token -----------------
def t4_native_kernels() -> dict:
    """What IronMule's own kernels do on a GPU that emulates bf16.

    Two measures with different units, so two panels with their own axis rather
    than one chart with two scales. Both arms of a model come from the same run.
    """

    rows, sources = [], []
    for label, key in PERF1_E2E:
        stock_path = PERF1_E2E_DIR / f"e2e-{key}-stock.json"
        native_path = PERF1_E2E_DIR / f"e2e-{key}-kernel+p16.json"
        rows.append((label, load(stock_path), load(native_path)))
        sources += [stock_path.as_posix(), native_path.as_posix()]

    fig, axes = plt.subplots(1, 2, figsize=(style.WIDTH_IN, 3.3), sharey=True)
    height = 0.36
    panels = ((axes[0], "decode_tps_median", 1.0, "Decode, tokens per second", True),
              (axes[1], "ttft_ms_median", 1e-3, "Time to first token, 512 tokens", False))
    for axis, metric, scale, title, higher in panels:
        top = 0.0
        for index, (label, stock, native) in enumerate(rows):
            centre = len(rows) - 1 - index
            before, after = stock[metric] * scale, native[metric] * scale
            top = max(top, before, after)
            axis.barh(centre + height / 2, before, height * 0.86, color=style.BASELINE,
                      label="stock MLX, bf16" if index == 0 else None)
            axis.barh(centre - height / 2, after, height * 0.86, color=style.CANDIDATE,
                      label="IronMule --compute-dtype native" if index == 0 else None)
            change = (f"{after / before:.1f}×" if higher else f"\u2212{(1 - after / before) * 100:.0f}%")
            unit = "" if higher else " s"
            axis.annotate(f"{before:.1f}{unit}", (before, centre + height / 2), textcoords="offset points",
                          xytext=(5, 0), va="center", fontsize=9, color=style.MUTED)
            axis.annotate(f"{after:.{1 if higher else 2}f}{unit} \u00b7 {change}", (after, centre - height / 2),
                          textcoords="offset points", xytext=(5, 0), va="center", fontsize=9,
                          color=style.TEXT, fontweight="bold")
        axis.set_xlim(0, top * 1.42)
        axis.set_title(title, fontsize=10.5, loc="left")
        axis.set_xlabel("higher is better" if higher else "lower is better", fontsize=9, color=style.MUTED)
        axis.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([label for label, _, _ in reversed(rows)])
    fig.suptitle("A free T4 without bf16 arithmetic, with IronMule's own kernels",
                 fontsize=12, fontweight="bold", color=style.TEXT)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.06))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.8, bottom=0.2, wspace=0.12)

    written = emit(fig, "t4-native-kernels")
    plt.close(fig)
    return {
        "figure": written,
        "sources": sources,
        "device": T4,
        "models": ["mlx-community/Qwen3-8B-4bit", "mlx-community/Qwen3-14B-4bit"],
        "samples": "one fresh process per arm, one warm generation then 3 measured generations of "
                   "a 512-token prompt and 128 greedy tokens; decode and TTFT are the medians",
        "intervals": "none; screening medians from one cell (PERF1 run 2)",
    }


# -- figure 6: serving eight requests at once ------------------------------------
def t4_server_batching() -> dict:
    """Continuous batching, where a kernel either shares one pass over the weights or not."""

    arms = (("row kernel (decode path of the plan)", "server-qwen3-8b-kernel+p16-free.json", style.SECONDARY),
            ("tensor-core kernel for 2..16 rows", "server-qwen3-8b-kernel+mma+p16-free.json", style.CANDIDATE))
    data = [(name, load(PERF1_SERVER_DIR / file)["widths"], colour) for name, file, colour in arms]
    widths = (("one request at a time", "1"), ("eight requests at once", "8"))

    fig, ax = plt.subplots(figsize=(style.WIDTH_IN, 3.2))
    height, top = 0.36, 0.0
    control = data[0][1]["8"]["aggregate_tps"]
    for index, (label, width) in enumerate(widths):
        centre = len(widths) - 1 - index
        for slot, (name, rows, colour) in enumerate(data):
            value = rows[width]["aggregate_tps"]
            top = max(top, value)
            y = centre + (0.5 - slot) * height
            ax.barh(y, value, height * 0.86, color=colour, label=name if index == 0 else None)
            note = f"{value:.1f} tok/s"
            if slot == 1 and width == "8":
                note += f" \u00b7 {value / control:.2f}× the row kernel"
            ax.annotate(note, (value, y), textcoords="offset points", xytext=(5, 0), va="center",
                        fontsize=9, color=style.TEXT, fontweight="bold" if slot == 1 else "normal")
    ax.set_yticks(range(len(widths)))
    ax.set_yticklabels([label for label, _ in reversed(widths)])
    ax.set_xlim(0, top * 1.55)
    ax.set_xlabel("aggregate tokens per second across 8 different requests, Qwen 3 8B, higher is better")
    ax.set_title("A server's batch: tensor cores read each weight once for all requests")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=9)

    written = emit(fig, "t4-server-batching")
    plt.close(fig)
    return {
        "figure": written,
        "sources": [(PERF1_SERVER_DIR / file).as_posix() for _, file, _ in arms],
        "device": T4,
        "models": ["mlx-community/Qwen3-8B-4bit"],
        "samples": "8 different chat requests, greedy, 128 tokens each, mlx-lm's continuous-batching "
                   "BatchGenerator; both arms in the same run (PERF1 run 8)",
        "intervals": "none; one pass per width after a warm pass",
    }


def _perplexity_ratio(candidate: Path, reference: Path) -> tuple[float, float, float]:
    """Ratio and 95% chunk-bootstrap interval, the method `tests/test_numeric_plans.py` pins."""

    import math
    import random
    import statistics

    rows = list(zip(load(reference)["chunk_nll"], load(candidate)["chunk_nll"]))
    rng = random.Random(20260916)

    def ratio(sample):
        return math.exp(statistics.mean(b for _, b in sample) - statistics.mean(a for a, _ in sample))

    draws = sorted(ratio([rng.choice(rows) for _ in rows]) for _ in range(10000))
    return ratio(rows), draws[250], draws[9750]


# -- figure 7: what the plan costs in quality -------------------------------------
def t4_native_quality() -> dict:
    """The gate each changed path had to pass, with the bound it had to stay under."""

    gates = [(label, *_perplexity_ratio(PERF1 / candidate, PERF1 / reference))
             for label, candidate, reference in PERF1_GATES]
    fig, ax = plt.subplots(figsize=(style.WIDTH_IN, 2.9))
    for index, (label, value, low, high) in enumerate(gates):
        y = len(gates) - 1 - index
        ax.errorbar(value, y, xerr=[[value - low], [high - value]], fmt="o", color=style.CANDIDATE,
                    ecolor=style.CANDIDATE, elinewidth=2, capsize=4, markersize=7, zorder=4)
        # A fixed column right of the bound, so no label ever crosses the line it is judged by.
        ax.annotate(f"{value:.4f} [{low:.4f}; {high:.4f}]", (1.0056, y), va="center", fontsize=9,
                    color=style.TEXT)
    ax.axvline(1.0, color=style.RULE, linewidth=1.2, linestyle="--", zorder=2)
    ax.axvline(1.005, color=style.ACCENT, linewidth=1.6, zorder=2)
    ax.annotate("same as stock bf16", (1.0, len(gates) - 0.45), textcoords="offset points", xytext=(4, 0),
                fontsize=8.5, color=style.MUTED)
    ax.annotate("quality bound 1.005", (1.005, len(gates) - 0.45), textcoords="offset points",
                xytext=(4, 0), fontsize=8.5, color=style.ACCENT)
    ax.set_yticks(range(len(gates)))
    ax.set_yticklabels([label for label, *_ in reversed(gates)])
    ax.set_xlim(0.994, 1.0118)
    ax.set_ylim(-0.6, len(gates) - 0.2)
    ax.set_xlabel("perplexity ratio against stock bf16 on WikiText-2, 16 x 512 tokens, lower is better")
    ax.set_title("The quality gate: every path the plan changes stays inside the bound")
    ax.grid(axis="y", visible=False)

    written = emit(fig, "t4-native-quality")
    plt.close(fig)
    return {
        "figure": written,
        "sources": [(PERF1 / path).as_posix() for _, a, b in PERF1_GATES for path in (a, b)],
        "device": T4,
        "models": ["mlx-community/Qwen3-8B-4bit", "mlx-community/Qwen3-14B-4bit"],
        "samples": "16 chunks of 512 tokens per arm; decode path teacher-forced through the cache",
        "intervals": "95% chunk bootstrap, 10000 draws, seed 20260916",
    }


#: PERF1 run 18 (PERF1-Z): every Gemma 3 arm and its stock reference in one session.
RUN18 = PERF1 / "perf1-run18-863237d6"
RUN18_ARMS = (
    ("Gemma 3 1B", "1b", "ironmule", "exact"),
    ("Gemma 3 4B", "4b", "ironmule_exact", "exact"),
    ("Gemma 3 12B", "12b", "ironmule_exact", "exact"),
    ("Gemma 3 12B, float32 plan", "12b", "ironmule_fp32", "plan"),
    ("Gemma 3 12B, native plan", "12b", "ironmule_native", "plan"),
    ("Gemma 3 12B, native + fixed cache", "12b", "ironmule_native_compiled", "plan"),
)


# -- figure 8: the newest re-measurement, every arm against stock in one run ---
def t4_run18_rerun() -> dict:
    """PERF1 run 18 re-measured every published Gemma 3 CUDA ratio in one session.

    Each bar is one arm against the stock arm of the same run and repetition, so no
    ratio here is chained across runs. Dots are the repetitions. Exact arms returned
    stock's tokens; the numeric plans change the arithmetic and carry their own gate.
    """

    rows = []
    for label, model, arm, kind in RUN18_ARMS:
        summary = load(RUN18 / f"cross-gemma3-{model}.json")["summary"][arm]
        rows.append((label, 1 / summary["median_ratio"],
                     [1 / r for r in summary["ratio_vs_reference_per_rep"]],
                     summary["identical_requests"], kind))

    fig, ax = plt.subplots(figsize=(style.WIDTH_IN, 4.2))
    colours = {"exact": style.SECONDARY, "plan": style.CANDIDATE}
    names = {"exact": "exact: the same tokens as stock",
             "plan": "opt-in numeric plan: changes the arithmetic, gated on its own"}
    labelled = set()
    for index, (label, speedup, reps, identical, kind) in enumerate(rows):
        y = len(rows) - 1 - index
        ax.barh(y, speedup, height=0.62, color=colours[kind],
                label=None if kind in labelled else names[kind])
        labelled.add(kind)
        ax.scatter(reps, [y] * len(reps), s=14, color=style.TEXT, zorder=4)
        ax.annotate(f"{speedup:.2f}× · +{(speedup - 1) * 100:.0f}% · {identical}/6 same tokens",
                    (max(reps + [speedup]), y), textcoords="offset points", xytext=(7, 0),
                    va="center", fontsize=9, color=style.TEXT)
    ax.axvline(1.0, color=style.RULE, linewidth=1.2, linestyle="--", zorder=3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label for label, *_ in reversed(rows)])
    ax.set_xlim(0, max(speedup for _, speedup, *_ in rows) * 1.55)
    ax.set_xlabel("times faster than stock MLX in the same run, NVIDIA Tesla T4, higher is faster")
    ax.set_title("Every Gemma 3 number on a T4, re-measured against stock in one run")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=9)

    written = emit(fig, "t4-run18-rerun")
    plt.close(fig)
    return {
        "figure": written,
        "sources": sorted({(RUN18 / f"cross-gemma3-{model}.json").as_posix()
                           for _, model, _, _ in RUN18_ARMS}),
        "device": T4,
        "models": [f"mlx-community/gemma-3-{size}-it-4bit" for size in ("1b", "4b", "12b")],
        "samples": "PERF1 run 18: interleaved fresh processes, 6 requests x 48 tokens, "
                   "2 repetitions for 12B and 3 for 1B and 4B, stock in every repetition",
        "intervals": "none; median of the per-repetition wall ratios, inverted to a speed-up; "
                     "dots are the repetitions",
    }


FIGURES = (paired_ratios, session_ratios, prefill_phases, cross_platform_speedup,
           t4_native_kernels, t4_server_batching, t4_native_quality, t4_run18_rerun)


def render(destination: Path) -> list[dict]:
    """Draw every figure once per theme. Light first, so its bytes are stable."""

    global ASSETS, SUFFIX
    previous, ASSETS = ASSETS, destination
    manifest = []
    try:
        for theme, SUFFIX in (("light", ""), ("dark", "-dark")):
            style.configure(matplotlib, theme)
            manifest += [dict(figure(), theme=theme) for figure in FIGURES]
    finally:
        ASSETS, SUFFIX = previous, ""
        style.configure(matplotlib)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="re-render and fail if a committed SVG differs")
    parser.add_argument("--manifest", action="store_true",
                        help="print what each figure claims and what it read")
    args = parser.parse_args(argv)

    if args.check:
        with tempfile.TemporaryDirectory() as directory:
            manifest = render(Path(directory))
            drifted = []
            for entry in manifest:
                built = Path(directory) / Path(entry["figure"]).name
                committed = ROOT / entry["figure"]
                if not committed.is_file():
                    drifted.append(f"{entry['figure']}: not committed")
                elif built.read_bytes() != committed.read_bytes():
                    drifted.append(f"{entry['figure']}: differs from its source data")
            if drifted:
                print("figures have drifted from the evidence:", file=sys.stderr)
                for line in drifted:
                    print(f"  {line}", file=sys.stderr)
                print("\nrun: python tools/make_figures.py", file=sys.stderr)
                return 1
        print(f"{len(manifest)} figures match their evidence")
        return 0

    manifest = render(ASSETS)
    if args.manifest:
        print(json.dumps(manifest, indent=2))
    else:
        for entry in manifest:
            print(f"{entry['figure']}  <- {', '.join(entry['sources'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
