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
PAPER = "white"

#: The two palettes, by the same role names. GitHub serves whichever theme the
#: reader chose, so a figure exists twice; the dark values keep Okabe-Ito's hues
#: and only lift them off a dark ground, so both renderings read as one figure.
LIGHT = {name: globals()[name] for name in
         ("CONTROL", "CANDIDATE", "BASELINE", "SECONDARY", "ACCENT", "RULE",
          "TEXT", "MUTED", "GRID", "PAPER")}
DARK = {
    "CONTROL": "#56B4E9",
    "CANDIDATE": "#F0873C",
    "BASELINE": "#9AA4AE",
    "SECONDARY": "#1FC99B",
    "ACCENT": "#E8A0C4",
    "RULE": "#6E7681",
    "TEXT": "#E6EDF3",
    "MUTED": "#9198A1",
    "GRID": "#2A3038",
    "PAPER": "#0D1117",  # GitHub's own dark canvas, so nothing frames the figure
}

#: Every figure is this wide so they stack in a README without jumping.
WIDTH_IN = 8.6


def configure(matplotlib: Any, theme: str = "light") -> None:
    """Apply the shared look for one theme and make SVG output deterministic."""

    globals().update(DARK if theme == "dark" else LIGHT)

    matplotlib.use("Agg")
    # Without a fixed salt, matplotlib derives clip-path and gradient ids from
    # object identity, so the same data renders to different bytes each run.
    matplotlib.rcParams["svg.hashsalt"] = "ironmule.figures.v1"
    # Text stays text: no font outlines embedded, so the file does not change
    # when the machine's font set does.
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams.update({
        "figure.dpi": 100,
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "axes.edgecolor": RULE,
        # Legend labels and any unstyled text follow the theme too, or the dark
        # rendering writes black on black.
        "text.color": TEXT,
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


__all__ = ["ACCENT", "BASELINE", "CANDIDATE", "CONTROL", "DARK", "GRID", "LIGHT", "MUTED",
           "PAPER", "RULE", "SECONDARY", "TEXT", "WIDTH_IN", "configure", "save"]
