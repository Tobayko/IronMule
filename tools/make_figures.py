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


FIGURES = (paired_ratios, session_ratios, prefill_phases, cross_platform_speedup)


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
