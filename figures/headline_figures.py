"""Headline quantitative figures for the fixed-optics glass paper, built directly
from the simulator CSVs (sim/results/*.csv) so every point is traceable. Arial per
TTT-Research CLAUDE.md. MODEL results, not measured hardware.

Figures:
  fig_latency_ladder.png  -- random-read latency: fixed optics vs robotic mount vs NVMe
  fig_frontier.png        -- capacity + useful read throughput vs feature size (FOV x res)
  fig_ber_snr.png         -- BER vs SNR (camera-noise, M-PAM) with ECC waterfall
  fig_tpw.png             -- throughput-per-watt vs camera pixel-rate, NVMe crossover (Q4)
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "sim" / "results"
RW_RES = HERE.parent / "results" / "rw_final"
for ttf in ("Arial.ttf", "Arial Bold.ttf"):
    p = Path.home() / ".local/share/fonts" / ttf
    if p.exists():
        font_manager.fontManager.addfont(str(p))
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
BLUE, RED, GREEN, INK = "#4c72b0", "#c44e52", "#55a868", "#22242a"


def _read(name):
    with (RES / name).open() as f:
        return list(csv.DictReader(f))


def _read_rw(name):
    with (RW_RES / name).open() as f:
        return list(csv.DictReader(f))


def latency_ladder():
    # From frontier.csv (fixed-optics random_block_latency_ms) + module constants.
    fo = _read("fixed_optics_frontier.csv")
    fo_lat = float(fo[0]["random_block_latency_ms"])   # ~constant across features
    labels = ["NVMe SSD", "HDD\n(seek)", "Heaven's Eye\n(one camera frame)"]
    vals = [0.1, 7.0, fo_lat]
    # Grayscale + hatching: stays legible in black-and-white print, where a
    # green/blue/red triplet collapses to three near-identical grays.
    # Single-column figure: sized to the USENIX 3.4in column so it is placed at
    # ~1:1 rather than scaled down, which is what keeps bar width and type size
    # matching the body text. No axes title -- the caption carries it, per the
    # convention in the accepted-paper corpus.
    fills = ["white", "0.62", "0.25"]
    hatches = ["///", "...", ""]
    fig, ax = plt.subplots(figsize=(3.4, 2.15))
    bars = ax.bar(labels, vals, color=fills, width=0.52,
                  edgecolor=INK, linewidth=0.8)
    for b, h in zip(bars, hatches):
        b.set_hatch(h)
    ax.set_yscale("log")
    ax.set_ylabel("Random-read latency (ms)", fontsize=8)
    ax.tick_params(axis="both", labelsize=7.5)
    ax.set_ylim(0.03, 300)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v*1.15, f"{v:g}", ha="center",
                va="bottom", fontsize=7.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    fig.tight_layout(); fig.savefig(HERE / "fig_latency_ladder.png", dpi=200)
    plt.close(fig)


def rw_ber_snr():
    """Print-safe learned-vs-identity curve from the committed codec sweep."""
    rows = _read_rw("codec_sweep.csv")
    fig, ax = plt.subplots(figsize=(3.4, 2.25))
    shades = {2: "#1b4f8c", 3: "#c9622b", 4: "#2e8b57"}  # colorblind-safe blue/orange/green
    for bits in (2, 3, 4):
        for learned, marker, line, name in (
            (False, "s", "--", "identity"),
            (True, "o", "-", "learned"),
        ):
            selected = sorted(
                (r for r in rows
                 if int(r["bits"]) == bits
                 and r["use_encoder"].lower() == str(learned).lower()),
                key=lambda r: float(r["signal_e"]),
            )
            ax.plot(
                [float(r["signal_e"]) for r in selected],
                [max(float(r["ber"]), 1e-6) for r in selected],
                marker=marker, linestyle=line, color=shades[bits],
                linewidth=1.0, markersize=2.8,
                label=f"{bits} bit, {name}",
            )
    ax.axhline(1e-2, color="0.1", linestyle=":", linewidth=0.9)
    ax.text(13500, 1.35e-2, "BER threshold", ha="right", va="bottom", fontsize=6.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Signal electrons", fontsize=8)
    ax.set_ylabel("Bit-error rate", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    # Legend was overlapping the plotted curves at "lower left"; anchor it
    # below the axes instead so no label sits on top of data.
    ax.legend(fontsize=5.7, ncol=3, frameon=False, columnspacing=0.9,
              handlelength=1.8, loc="upper center", bbox_to_anchor=(0.5, -0.30))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.22, linewidth=0.45)
    fig.tight_layout()
    fig.savefig(HERE / "rw_ber_snr.png", dpi=220)
    plt.close(fig)


def frontier():
    r = _read("fixed_optics_frontier.csv")
    feat = [float(x["feature_um"]) for x in r]
    cap = [float(x["capacity_TB"]) for x in r]
    thr = [float(x["read_throughput_MBps"]) for x in r]
    fig, ax1 = plt.subplots(figsize=(5.6, 3.6))
    ax1.plot(feat, cap, "o-", color=BLUE, label="Capacity (TB)")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.invert_xaxis()
    ax1.set_xlabel("Feature pitch (µm)  — finer → higher capacity, more pixels")
    ax1.set_ylabel("Capacity per platter (TB)", color=BLUE)
    ax1.tick_params(axis="y", labelcolor=BLUE)
    ax2 = ax1.twinx(); ax2.grid(False)
    ax2.plot(feat, thr, "s--", color=RED, label="Useful read (MB/s)")
    ax2.set_ylabel("Useful read throughput (MB/s)", color=RED)
    ax2.tick_params(axis="y", labelcolor=RED)
    ax1.set_title("Capacity / throughput vs feature size (FOV × resolution)")
    fig.tight_layout(); fig.savefig(HERE / "fig_frontier.png", dpi=200)
    plt.close(fig)


def ber_snr():
    r = _read("ber_snr.csv")
    snr = [float(x["snr_dB"]) for x in r]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for b, style in zip((1, 2, 3, 4), ("o-", "s-", "^-", "d-")):
        y = [max(float(x[f"ber_{b}bit"]), 1e-12) for x in r]
        ax.plot(snr, y, style, ms=3, label=f"{b} bit/voxel")
    ax.axhline(1e-2, color=INK, ls=":", lw=1.2)
    ax.text(snr[-1], 3.5e-3, "ECC waterfall (raw BER ≤ 1e-2)", fontsize=8,
            color=INK, ha="right", va="center")
    ax.set_yscale("log"); ax.set_ylim(1e-12, 1)
    ax.set_xlabel("Read SNR (dB)"); ax.set_ylabel("Raw bit-error rate")
    ax.set_title("BER vs SNR under the camera-noise model")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(HERE / "fig_ber_snr.png", dpi=200)
    plt.close(fig)


def tpw():
    r = _read("throughput_per_watt.csv")
    x = [float(v["pixel_rate_x"]) for v in r]
    y = [float(v["MBps_per_W"]) for v in r]
    nvme = float(r[0]["nvme_MBps_per_W"])
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot(x, y, "o-", color=BLUE, label="Fixed-optics glass")
    ax.axhline(nvme, color=GREEN, ls="--", label=f"NVMe (~{nvme:g} MB/s/W)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Camera pixel-rate multiplier (×)")
    ax.set_ylabel("Throughput per watt (MB/s/W)")
    ax.set_title("Throughput-per-watt vs NVMe crossover (research Q4)")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(HERE / "fig_tpw.png", dpi=200)
    plt.close(fig)


def landmarks():
    d = _read("landmark_density.csv")
    dens = [float(x["landmark_density_per_cell"]) for x in d]
    med = [float(x["median_err_cells"]) for x in d]
    p99 = [float(x["p99_err_cells"]) for x in d]
    er = [float(x["addressing_error_rate"]) for x in d]
    fig, ax1 = plt.subplots(figsize=(5.6, 3.6))
    ax1.plot(dens, med, "o-", color=BLUE, label="median residual")
    ax1.plot(dens, p99, "s--", color=RED, label="p99 residual")
    ax1.axhline(0.5, color=INK, ls=":", lw=1.2)
    ax1.text(dens[1], 0.55, "0.5 cell = addressing threshold", fontsize=8, color=INK)
    ax1.set_xlabel("Landmark density (per cell)")
    ax1.set_ylabel("Residual localization error (cells)")
    ax1.set_yscale("log"); ax1.legend(fontsize=8, loc="upper right")
    ax1.set_title("Imperfections as fiducials: sub-cell addressing")
    fig.tight_layout(); fig.savefig(HERE / "fig_landmarks.png", dpi=200)
    plt.close(fig)


def main():
    latency_ladder(); rw_ber_snr(); frontier(); ber_snr(); tpw(); landmarks()
    print("wrote:", *(p.name for p in sorted(HERE.glob("fig_*.png"))))


if __name__ == "__main__":
    main()
