"""Shared style for GlassFleet schematic figures.

Registers Arial from ~/.local/share/fonts (per repo figure rules) with a
Liberation Sans metric-compatible fallback, and defines a restrained palette so
all three diagrams look like one figure set. Import and call apply_style().
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless render
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# --- Arial registration (repo rule) --------------------------------------- #
_FONT_DIR = Path.home() / ".local" / "share" / "fonts"
for _name in ("Arial.ttf", "Arial Bold.ttf", "Arial Italic.ttf"):
    _p = _FONT_DIR / _name
    if _p.exists():
        try:
            fm.fontManager.addfont(str(_p))
        except Exception:
            pass


def apply_style() -> None:
    installed = {f.name for f in fm.fontManager.ttflist}
    family = "Arial" if "Arial" in installed else "Liberation Sans"
    plt.rcParams["font.family"] = family
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.linewidth"] = 0.0
    plt.rcParams["figure.dpi"] = 200
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


# --- Palette (calm, print-legible; not the AI-default rainbow) ------------- #
INK = "#1f2430"        # near-black text / strokes
SLATE = "#5b6473"      # secondary text
GLASS = "#cfe6ef"      # quartz body fill (pale cyan)
GLASS_EDGE = "#7fb4c6"
WRITE = "#c2410c"      # write path (warm, "laser")
WRITE_FILL = "#fde3d3"
READ = "#1d6fb8"       # read path (cool, "camera")
READ_FILL = "#dbecf9"
GOLD = "#b8860b"       # voxel / accent
PANEL = "#f4f6f8"      # soft panel background
PANEL_EDGE = "#c8ced7"
GREEN = "#2f7d4f"      # "viable" / success accents
MUTE = "#9aa3b0"
