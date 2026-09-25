"""Figure 1 - Heaven's Eye device (abstracted, no moving parts).

Vertical stack: a fixed camera (the "eye") + lens sit ABOVE a stationary glass
platter; the controller PCB / motherboard (the "brain": AI inference model) sits
BELOW the glass and drives the eye over a high-bandwidth data link (fast transfer).
The same fixed optic both reads (capture + register on the glass's intrinsic
imperfections) and drives writes (illuminate / modify, Mode B/C). Nothing moves.

Spaced so no labels overlap; Arial via _style. Run: python3 device.py -> device.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Ellipse
import matplotlib.pyplot as plt

from _style import (apply_style, INK, SLATE, GLASS, GLASS_EDGE, WRITE,
                    WRITE_FILL, READ, READ_FILL, GOLD, PANEL, PANEL_EDGE, MUTE, GREEN)

HERE = Path(__file__).resolve().parent


def _arrow(ax, xy0, xy1, color, lw=2.4, style="-|>", mut=16, ls="-"):
    ax.add_patch(FancyArrowPatch(xy0, xy1, arrowstyle=style, mutation_scale=mut,
                 lw=lw, color=color, linestyle=ls, shrinkA=3, shrinkB=3,
                 zorder=8, capstyle="round"))


def main() -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(8.4, 8.8))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    cx = 48

    ax.text(cx - 8, 97, "Heaven's Eye device", ha="center", va="center",
            fontsize=15, color=INK, fontweight="bold")
    ax.text(cx - 8, 93.4, "fixed camera above · stationary glass · brain below",
            ha="center", va="center", fontsize=9.2, color=SLATE)

    # no-moving-parts badge (top-right, clear)
    ax.add_patch(FancyBboxPatch((78, 90.5), 20, 4.6,
                 boxstyle="round,pad=0.2,rounding_size=0.7",
                 facecolor="#eaf5ee", edgecolor=GREEN, lw=1.3, zorder=3))
    ax.text(88, 92.8, "NO MOVING PARTS", ha="center", va="center",
            fontsize=9, color=GREEN, fontweight="bold")

    # ---- the eye: fixed camera (top) ---------------------------------------
    ax.add_patch(FancyBboxPatch((cx - 25, 79.6), 50, 9.2,
                 boxstyle="round,pad=0.3,rounding_size=1.0",
                 facecolor=READ_FILL, edgecolor=READ, lw=2.0, zorder=6))
    ax.text(cx, 85.7, "Fixed camera  (the eye)", ha="center", va="center",
            fontsize=12.0, color=READ, fontweight="bold", zorder=7)
    ax.text(cx, 82.3, "global-shutter CMOS + telecentric optics",
            ha="center", va="center", fontsize=8.0, color=INK, zorder=7)

    ax.add_patch(Ellipse((cx, 76.5), 16, 3.6, facecolor="#eef4f8",
                 edgecolor=READ, lw=1.5, zorder=5))
    ax.text(cx, 76.5, "lens", ha="center", va="center", fontsize=8.5,
            color=SLATE, zorder=6)
    ax.text(cx - 18, 77.6, "multispectral · polarized\n· coherent illumination",
            ha="right", va="center", fontsize=8.2, color=SLATE, zorder=6)

    # ---- stationary glass platter (layered) --------------------------------
    gy = 52
    n_layers, lw_, lh_, skew, gap = 6, 28, 3.0, 6.5, 2.9
    base_y = gy - (n_layers * gap) / 2
    for k in range(n_layers):
        y = base_y + k * gap
        pts = np.array([[cx - lw_/2, y], [cx + lw_/2, y],
                        [cx + lw_/2 + skew, y + lh_], [cx - lw_/2 + skew, y + lh_]])
        ax.add_patch(mpatches.Polygon(pts, closed=True, facecolor=GLASS,
                     edgecolor=GLASS_EDGE, lw=1.1, alpha=0.55 + 0.07*k, zorder=3))
        rng = np.random.default_rng(k + 3)
        for j in range(11):
            fx = cx - lw_/2 + 1.4 + j * (lw_ - 2.8) / 10
            if rng.random() < 0.8:
                ax.add_patch(Ellipse((fx + skew*0.5, y + lh_/2), 0.6, 0.4, angle=25,
                             facecolor=GOLD, edgecolor="none", alpha=0.85, zorder=4))
    top_y = base_y + n_layers * gap
    # Platter caption sits LOW, right above the platter itself, clear of the
    # WRITE/READ row which sits higher (near the lens).
    ax.text(cx + skew/2, top_y + 1.1, "stationary glass platter · 8×8 blocks · 16,384 cells",
            ha="center", va="bottom", fontsize=10, color=INK, fontweight="bold")

    # WRITE (down) / READ (up) through the fixed optic -- routed to the sides
    # of the platter so neither arrow crosses the label above it.
    _arrow(ax, (cx - 10, 74.4), (cx - 10, top_y + 4.4), WRITE, lw=2.6)
    _arrow(ax, (cx + 10, top_y + 4.4), (cx + 10, 74.4), READ, lw=2.6)
    ax.text(cx - 15.5, 71.0, "WRITE", ha="right", va="center", fontsize=10.5,
            color=WRITE, fontweight="bold")
    ax.text(cx - 15.5, 68.7, "illuminate / modify\n(Mode B)", ha="right",
            va="center", fontsize=8.2, color=SLATE)
    ax.text(cx + 15.5, 71.0, "READ", ha="left", va="center", fontsize=10.5,
            color=READ, fontweight="bold")
    ax.text(cx + 15.5, 68.7, "capture + register\non imperfections", ha="left",
            va="center", fontsize=8.2, color=SLATE)

    # landmark callout (left of platter)
    ax.add_patch(FancyBboxPatch((2, 47.5), 20, 9,
                 boxstyle="round,pad=0.25,rounding_size=0.7",
                 facecolor="#fff9e6", edgecolor=GOLD, lw=1.3, zorder=3))
    ax.text(12, 53.5, "intrinsic imperfections", ha="center", va="center",
            fontsize=9, color=INK, fontweight="bold")
    ax.text(12, 50.0, "= fiducial lattice\n(coordinate frame\nfor read + write)",
            ha="center", va="center", fontsize=8.0, color=SLATE)
    _arrow(ax, (22.2, 51.5), (cx - lw_/2 + 1, base_y + 3*gap), MUTE, lw=1.1,
           style="-|>", mut=10)

    # ---- the brain: controller PCB / motherboard (BELOW the glass) ---------
    # Box extended downward and the long spec line split across rows so no text
    # exceeds the box width.
    ax.add_patch(FancyBboxPatch((cx - 26, 16.5), 52, 16.5,
                 boxstyle="round,pad=0.3,rounding_size=1.0",
                 facecolor=PANEL, edgecolor=INK, lw=2.0, zorder=6))
    ax.text(cx, 30.4, "Controller PCB / motherboard  (the brain)", ha="center",
            va="center", fontsize=11.5, color=INK, fontweight="bold", zorder=7)
    ax.text(cx, 27.1, "AI inference: registration · CNN decode", ha="center",
            va="center", fontsize=8.4, color=SLATE, zorder=7)
    ax.text(cx, 24.3, "· addressing · learned read/write codec · ECC", ha="center",
            va="center", fontsize=8.4, color=SLATE, zorder=7)
    ax.text(cx, 20.4, "optical PCIe / CXL host link   ·   12 V power", ha="center",
            va="center", fontsize=8.4, color=SLATE, zorder=7)
    # PCB sits under the glass: connect glass -> PCB
    _arrow(ax, (cx + skew/2, base_y - 0.4), (cx + skew/2, 33.2), MUTE, lw=1.2,
           style="-", mut=1)

    # ---- high-bandwidth data link: camera <-> PCB (right side) -------------
    # Routed to enter the BRAIN box through its top edge (off-center, x=68,
    # inside the box's 22..74 span) so the arrowhead lands on empty panel
    # space, never on the box's text lines.
    link_x = 78
    _arrow(ax, (cx + 15, 84.2), (link_x, 84.2), READ, lw=2.2, style="-", mut=1)
    _arrow(ax, (link_x, 84.2), (link_x, 36), READ, lw=2.2, style="-", mut=1)
    _arrow(ax, (link_x, 36), (68, 36), READ, lw=2.2, style="-", mut=1)
    _arrow(ax, (68, 36), (68, 33.2), READ, lw=2.2, style="-|>", mut=13)
    ax.text(80.5, 58, "high-bandwidth\ndata link\n(fast transfer)", ha="left",
            va="center", fontsize=8.6, color=READ, fontweight="bold")

    fig.savefig(HERE / "device.png")
    print("wrote", HERE / "device.png")


if __name__ == "__main__":
    main()
