"""Render the 1280x640 card GitHub shows wherever the repository is shared.

Every number on the card is read from the same committed run the README cites,
so the card cannot promise something the evidence does not.

    python tools/make_social_preview.py

Needs matplotlib:  pip install -e ".[figures]"
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import figure_style as style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import make_figures as figures  # noqa: E402

# After `make_figures`, which configures the light theme as it imports: the card
# is dark whatever a reader's own theme is, because a link preview has no theme.
style.configure(matplotlib, "dark")

ASSETS = ROOT / "docs" / "assets"
CARD = ASSETS / "social-preview.png"
BADGE = ASSETS / "ironmule-badge.jpg"


def speedup(ratio: float) -> str:
    return f"+{(1 / ratio - 1) * 100:.0f}%"


def main() -> int:
    _, apple_path, t4_path, _ = figures.CROSS[0]
    apple = figures.load(apple_path)["wall_ratios"]["D/A"]["median_ratio"]
    t4 = figures.load(t4_path)["summary"]["ironmule"]["median_ratio"]

    fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
    fig.patch.set_facecolor(style.PAPER)

    badge = fig.add_axes((0.055, 0.595, 0.17, 0.34))
    badge.imshow(plt.imread(BADGE))
    badge.axis("off")

    fig.text(0.25, 0.835, "IronMule", fontsize=62, fontweight="bold",
             color=style.TEXT, va="center")
    fig.text(0.252, 0.715, "Local LLM inference on Apple Silicon and NVIDIA",
             fontsize=23, color=style.MUTED, va="center")

    fig.add_artist(plt.Line2D([0.055, 0.945], [0.50, 0.50], color=style.GRID,
                              linewidth=1.5, transform=fig.transFigure))

    columns = [
        (speedup(t4), "faster", "Gemma 3 1B on a Tesla T4", style.SECONDARY),
        (speedup(apple), "faster", "the same model on an M1 Max", style.CANDIDATE),
        ("0", "tokens changed", "every gain is proved, or it is not used", style.CONTROL),
    ]
    for index, (number, unit, caption, colour) in enumerate(columns):
        x = 0.055 + index * 0.31
        fig.text(x, 0.335, number, fontsize=54, fontweight="bold", color=colour, va="center")
        fig.text(x, 0.215, unit, fontsize=20, color=style.TEXT, va="center")
        fig.text(x, 0.135, caption, fontsize=15, color=style.MUTED, va="center")

    fig.savefig(CARD, format="png", facecolor=style.PAPER, metadata={"Software": None})
    plt.close(fig)
    print(f"{CARD.relative_to(ROOT)}  <- "
          f"{apple_path.as_posix()}, {t4_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
