"""Generate Figure 4.1 and Figure 5.1 for the project report.

Both are drawn in Times New Roman to match the report body, and in a greyscale
safe palette so they remain legible when the report is printed in black and
white. Run from the project root:

    venv/Scripts/python.exe -I report/make_diagrams.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 9,
    "axes.linewidth": 0.8,
})

# Greyscale safe: these four fills stay distinct when desaturated.
STAGE = "#e8e8e8"
SOURCE = "#cfcfcf"
ARTEFACT = "#f7f7f7"
ACCENT = "#9a9a9a"
INK = "#1a1a1a"


def box(ax, x, y, w, h, text, *, fill=STAGE, bold=False, size=9,
        style="round,pad=0.015,rounding_size=0.012", lw=1.0, ls="-"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=style,
        facecolor=fill, edgecolor=INK, linewidth=lw, linestyle=ls, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=size, fontweight="bold" if bold else "normal",
            color=INK, zorder=3, linespacing=1.45)


def arrow(ax, x1, y1, x2, y2, lw=1.1, ls="-"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
        linewidth=lw, linestyle=ls, color=INK, zorder=4,
        shrinkA=0, shrinkB=0))


# ---------------------------------------------------------------------------
# Figure 4.1: end to end system architecture
# ---------------------------------------------------------------------------

def figure_4_1() -> None:
    fig, ax = plt.subplots(figsize=(6.6, 7.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 116)
    ax.axis("off")

    cx, cw = 16, 46          # pipeline column
    ax_, aw = 68, 29         # artefact column
    h = 7.2

    # Process boundary for the batch pipeline
    ax.add_patch(Rectangle((2.5, 22.5), 96, 89.5, fill=False, edgecolor=ACCENT,
                           linewidth=0.9, linestyle=(0, (5, 4)), zorder=1))
    ax.text(6.0, 112.0, "BATCH PIPELINE  (src/run_pipeline.py)", fontsize=8,
            style="italic", color="#555555", ha="left", va="center", zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", pad=2.2))

    # Data source
    box(ax, 20, 99.0, 60, h + 3.0,
        "YAHOO FINANCE\n"
        "Instrument OHLCV  ·  India VIX\n"
        "USD/INR  ·  NIFTY 50",
        fill=SOURCE, bold=True, size=8.2)

    stages = [
        (88.0, "ACQUISITION\ndaily bars, 2015 to present, split adjusted",
         "data/raw/\n{slug}_raw.csv"),
        (78.0, "CLEANING\nheader normalisation, deduplication, forward fill",
         "data/processed/\n{slug}_clean.csv"),
        (68.0, "FEATURE ENGINEERING\n26 index / 28 stock, all scale invariant",
         "data/processed/\n{slug}_features.csv"),
        (58.0, "LABELLING\nUP / NEUTRAL / DOWN at ±0.25% or ±0.50%",
         "data/processed/\n{slug}_labeled.csv"),
        (48.0, "WALK FORWARD TRAINING\n4 expanding folds, then one sealed test",
         "data/predictions/\n{slug}_{model}_*.csv"),
    ]

    prev_y = 99.0
    for y, label, artefact in stages:
        box(ax, cx, y, cw, h, label, size=8.4)
        box(ax, ax_, y + 0.4, aw, h - 0.8, artefact, fill=ARTEFACT, size=7.6)
        arrow(ax, cx + cw / 2, prev_y, cx + cw / 2, y + h)
        ax.plot([cx + cw, ax_], [y + h / 2, y + h / 2], color=ACCENT,
                linewidth=0.8, linestyle=(0, (2, 2)), zorder=1)
        prev_y = y

    # The dual model split
    split_y = 34.5
    arrow(ax, cx + cw / 2, 48, cx + cw / 2, 45.2)
    ax.plot([20, 72], [45.2, 45.2], color=INK, linewidth=1.1, zorder=4)
    arrow(ax, 20, 45.2, 20, split_y + h + 1.2)
    arrow(ax, 72, 45.2, 72, split_y + h + 1.2)

    box(ax, 5, split_y, 32, h + 1.2,
        "FINAL TEST MODEL\ntrained 2015 to 2023\nscored exactly once",
        fill=STAGE, bold=False, size=8.2)
    box(ax, 56, split_y, 32, h + 1.2,
        "LIVE MODEL\nrefit on all history\nnever scored",
        fill=STAGE, bold=False, size=8.2)

    box(ax, 5, 24.5, 32, 5.6, "results/{slug}_metrics.csv\nhonest performance",
        fill=ARTEFACT, size=7.6)
    box(ax, 56, 24.5, 32, 5.6, "saved_models/{slug}_{model}_live\ntomorrow's call",
        fill=ARTEFACT, size=7.6)
    arrow(ax, 21, split_y, 21, 30.1)
    arrow(ax, 72, split_y, 72, 30.1)

    # Dashboard, a separate process reading the artefacts
    box(ax, 12, 7.5, 74, 9.0,
        "INTERACTIVE DASHBOARD  (dashboard/app.py)\n"
        "reads artefacts only  ·  trains nothing in process\n"
        "launches the pipeline as a child process on request",
        fill=SOURCE, bold=True, size=8.4)
    arrow(ax, 21, 24.5, 21, 16.5, ls=(0, (4, 3)))
    arrow(ax, 72, 24.5, 72, 16.5, ls=(0, (4, 3)))
    ax.text(49, 20.5, "files on disk are the only interface",
            fontsize=7.6, style="italic", color="#555555", ha="center",
            zorder=6, bbox=dict(facecolor="white", edgecolor="none", pad=2.2))

    fig.savefig(OUT / "fig_architecture.png", dpi=220,
                bbox_inches="tight", facecolor="white", pad_inches=0.06)
    plt.close(fig)
    print("wrote fig_architecture.png")


# ---------------------------------------------------------------------------
# Figure 5.1: expanding window walk forward validation
# ---------------------------------------------------------------------------

def figure_5_1() -> None:
    fig, ax = plt.subplots(figsize=(7.6, 3.5))

    start, end = 2015, 2027
    rows = [
        ("Fold 1", 2015, 2020, 2021, "2020\nCOVID-19 crash", False),
        ("Fold 2", 2015, 2021, 2022, "2021\nRecovery rally", False),
        ("Fold 3", 2015, 2022, 2023, "2022\nRate tightening", False),
        ("Fold 4", 2015, 2023, 2024, "2023\nStabilisation", False),
        ("Final test", 2015, 2024, 2027, "2024 onward\nSealed, scored once", True),
    ]

    bar_h = 0.56
    for i, (label, t0, t1, v1, note, sealed) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, t1 - t0, left=t0, height=bar_h,
                color=STAGE, edgecolor=INK, linewidth=0.9, zorder=3)
        ax.barh(y, v1 - t1, left=t1, height=bar_h,
                color=ACCENT if not sealed else "#5f5f5f",
                edgecolor=INK, linewidth=0.9, zorder=3,
                hatch="" if not sealed else "///")
        ax.text((t0 + t1) / 2, y, f"train  {t0} to {t1 - 1}",
                ha="center", va="center", fontsize=8.2, color=INK, zorder=4)
        if sealed:
            ax.text((t1 + v1) / 2, y, note, ha="center", va="center",
                    fontsize=7.6, color="white", zorder=4, linespacing=1.3,
                    fontweight="bold")
        else:
            # A one year bar is too narrow to hold the regime name, so the year
            # sits inside it and the description runs alongside.
            year, regime = note.split("\n")
            ax.text((t1 + v1) / 2, y, year, ha="center",
                    va="center", fontsize=7.6, color=INK, zorder=4)
            # White backing so the sealed boundary rule does not cross the text
            ax.text(v1 + 0.18, y, regime, ha="left", va="center",
                    fontsize=7.6, style="italic", color="#444444", zorder=6,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.4))

    ax.axvline(2024, color=INK, linewidth=1.1, linestyle=(0, (5, 3)), zorder=5)
    ax.text(2024.08, len(rows) - 0.38, "sealed test boundary",
            fontsize=7.8, style="italic", rotation=90,
            va="top", ha="left", color=INK, zorder=6)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=9)
    ax.set_xlim(start - 0.3, end + 2.6)
    ax.set_ylim(-0.75, len(rows) - 0.25)
    ax.set_xticks(range(start, end + 1, 2))
    ax.set_xlabel("Calendar year", fontsize=9)
    ax.tick_params(axis="x", labelsize=8.4)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color="#dddddd", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=STAGE, edgecolor=INK, linewidth=0.8),
        plt.Rectangle((0, 0), 1, 1, facecolor=ACCENT, edgecolor=INK, linewidth=0.8),
        plt.Rectangle((0, 0), 1, 1, facecolor="#5f5f5f", edgecolor=INK,
                      linewidth=0.8, hatch="///"),
    ]
    ax.legend(handles, ["Training window", "Validation year", "Sealed test window"],
              loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=3,
              frameon=False, fontsize=8.2)

    fig.savefig(OUT / "fig_walkforward.png", dpi=220,
                bbox_inches="tight", facecolor="white", pad_inches=0.06)
    plt.close(fig)
    print("wrote fig_walkforward.png")


if __name__ == "__main__":
    figure_4_1()
    figure_5_1()
