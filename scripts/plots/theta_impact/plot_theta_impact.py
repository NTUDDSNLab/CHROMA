#!/usr/bin/env python3
"""Render the θ-impact figure (paper Fig. 6 redraw).

One row of subplots (default datasets: as-skitter, cit-Patents,
europe_osm). Per subplot: x = θ; left y = runtime (ms) as bars coloured
by #colors used (discrete per-subplot legend `color = N`); right y =
iteration count (line + markers). A star marks CEP θ (v0_paper) and a
diamond marks AEP θ (v3_raw), drawn near y=0 on the left axis; a marker
is omitted when its predicted θ is null.

Examples:
    python3 scripts/plots/theta_impact/plot_theta_impact.py
    python3 scripts/plots/theta_impact/plot_theta_impact.py \\
        --in scripts/plots/theta_impact/theta_impact_results.json \\
        --figsize 16 4
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np

SUBPLOT_TAGS = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
# Single colour family, sequential shades: darker = more colors used.
BAR_CMAP = "Blues"
CEP_COLOR, AEP_COLOR = "#D62728", "#17BECF"
TICK_FS, LABEL_FS, LEGEND_FS, TAG_FS = 10, 12, 9, 12


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("--in", dest="in_path",
                    default=str(here / "theta_impact_results.json"))
    ap.add_argument("--out-prefix", default=str(here / "theta_impact"))
    ap.add_argument("--figsize", nargs=2, type=float, default=[15.0, 4.0],
                    metavar=("W", "H"))
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        print(f"ERROR: {in_path} not found. Run theta_impact.py first.",
              file=sys.stderr)
        return 1

    d = json.loads(in_path.read_text())
    datasets = d["datasets"]
    theta_max = d["theta_max"]
    thetas = list(range(0, theta_max + 1))

    fig, axes = plt.subplots(1, len(datasets), figsize=tuple(args.figsize),
                             constrained_layout=True)
    if len(datasets) == 1:
        axes = [axes]

    for si, stem in enumerate(datasets):
        ax = axes[si]
        ax2 = ax.twinx()
        entry = d["data"].get(stem, {})
        sweep = entry.get("sweep", {})

        runtimes, iters, colors = [], [], []
        for t in thetas:
            cell = sweep.get(str(t), {})
            if not cell or "error" in cell:
                runtimes.append(0.0)
                iters.append(np.nan)
                colors.append(None)
            else:
                runtimes.append(cell["runtime_ms"])
                ic = cell.get("iter_count")
                iters.append(ic if ic is not None else np.nan)
                colors.append(cell["color"])

        # uniq ascending: fewest colors -> lightest, most -> darkest.
        uniq = sorted({c for c in colors if c is not None})
        cm = matplotlib.colormaps[BAR_CMAP]
        if len(uniq) <= 1:
            shades = [cm(0.65)] * len(uniq)
        else:
            lo, hi = 0.30, 0.95
            shades = [cm(lo + (hi - lo) * i / (len(uniq) - 1))
                      for i in range(len(uniq))]
        cmap = {c: s for c, s in zip(uniq, shades)}
        bar_colors = [cmap.get(c, "#CCCCCC") for c in colors]
        ax.bar(thetas, runtimes, width=0.8, color=bar_colors,
               edgecolor="black", linewidth=0.4, zorder=2)
        ax2.plot(thetas, iters, color="#1f1f1f", marker="o",
                 markersize=3, linewidth=1.3, zorder=3)

        cep = entry.get("cep_theta")
        aep = entry.get("aep_theta")
        ymax = max([r for r in runtimes if r > 0] or [1.0])
        y0 = ymax * 0.02
        if cep is not None:
            ax.scatter([cep], [y0], marker="*", s=240, color=CEP_COLOR,
                       edgecolor="black", linewidth=0.5, zorder=5)
        if aep is not None:
            ax.scatter([aep], [y0], marker="D", s=90, color=AEP_COLOR,
                       edgecolor="black", linewidth=0.5, zorder=5)

        ax.set_xlim(-0.6, theta_max + 0.6)
        ax.set_xticks(range(0, theta_max + 1, max(1, theta_max // 10)))
        ax.set_ylim(bottom=0.0)
        ax.tick_params(axis="both", labelsize=TICK_FS)
        ax2.tick_params(axis="y", labelsize=TICK_FS)
        ax.set_xlabel(r"$\theta$", fontsize=LABEL_FS)
        if si == 0:
            ax.set_ylabel("runtime (ms)", fontsize=LABEL_FS)
        if si == len(datasets) - 1:
            ax2.set_ylabel("iteration count", fontsize=LABEL_FS)
        ax.set_title(f"{SUBPLOT_TAGS[si]} {stem}", fontsize=TAG_FS)

        handles = [mpatches.Patch(facecolor=cmap[c], edgecolor="black",
                                  linewidth=0.4, label=f"color = {c}")
                   for c in uniq]
        if cep is not None:
            handles.append(mlines.Line2D(
                [], [], linestyle="none", marker="*", markersize=13,
                color=CEP_COLOR, markeredgecolor="black",
                label="CEP theta (v0_paper)"))
        if aep is not None:
            handles.append(mlines.Line2D(
                [], [], linestyle="none", marker="D", markersize=8,
                color=AEP_COLOR, markeredgecolor="black",
                label="AEP theta (v3_raw)"))
        if handles:
            ax.legend(handles=handles, fontsize=LEGEND_FS, frameon=True,
                      loc="upper right", handlelength=1.2, borderpad=0.4)

    pdf = Path(args.out_prefix + ".pdf")
    png = Path(args.out_prefix + ".png")
    pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    print(f"# wrote {pdf}\n# wrote {png}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
