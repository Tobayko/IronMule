"""One place the figures agree on what things look like.

Two rules make this file worth having. Colour carries meaning rather than
decoration: a baseline, a candidate and a control keep the same colour in every
figure, so a reader who learns the legend once never has to read it again. And
output is byte-reproducible, because a figure that drifts against unchanged data
is a figure nobody can trust.

The palette is Okabe-Ito, which stays distinguishable under the common forms of
colour vision deficiency and in greyscale print.
"""

from __future__ import annotations

from typing import Any

#: Okabe-Ito, assigned by role rather than by order of appearance.
CONTROL = "#0072B2"      # blue: an A/A arm, which must land on 1.0
CANDIDATE = "#D55E00"    # vermillion: the arm under test
BASELINE = "#767676"     # grey: the reference the candidate is measured against
SECONDARY = "#009E73"    # green: a second candidate in the same figure
ACCENT = "#CC79A7"       # magenta: an annotation that is not an arm
RULE = "#B0B0B0"         # the 1.0 line, axis rules, error-bar caps

TEXT = "#1A1A1A"
MUTED = "#5A5A5A"
GRID = "#E4E4E4"

#: Every figure is this wide so they stack in a README without jumping.
WIDTH_IN = 8.6


def configure(matplotlib: Any) -> None:
    """Apply the shared look and make SVG output deterministic."""

    matplotlib.use("Agg")
    # Without a fixed salt, matplotlib derives clip-path and gradient ids from
    # object identity, so the same data renders to different bytes each run.
    matplotlib.rcParams["svg.hashsalt"] = "ironmule.figures.v1"
    # Text stays text: no font outlines embedded, so the file does not change
    # when the machine's font set does.
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams.update({
        "figure.dpi": 100,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": RULE,
        "axes.labelcolor": TEXT,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "font.family": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
        "font.size": 10,
    })


def save(fig: Any, path: Any) -> None:
    """Write one SVG, with nothing in it that changes between runs."""

    path.parent.mkdir(parents=True, exist_ok=True)
    # `Date: None` drops the only timestamp matplotlib writes into an SVG.
    fig.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None})


__all__ = ["ACCENT", "BASELINE", "CANDIDATE", "CONTROL", "GRID", "MUTED", "RULE",
           "SECONDARY", "TEXT", "WIDTH_IN", "configure", "save"]
