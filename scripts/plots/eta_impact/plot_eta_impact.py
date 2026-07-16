#!/usr/bin/env python3
"""Render the η-impact figure from eta_impact_results.json.

A grid of subplots (one per dataset; default all datasets in the
results JSON, subset via --datasets): x = η (categorical, the swept
values); left y = PA-best runtime (ms) as a line + markers (NOT
zero-based so the turnover around the optimum stays visible); right
y = share of the warp-centric (SDC) path in % — one line for
iterations, one for peeled nodes (CTA-centric share = 100 − shown).
A dashed vertical line marks the default η = 2048 and a star marks
the best (min-runtime) η. X tick labels show η in units of 1024. A
shared legend sits above the grid. A failed η cell becomes a gap in
the lines.

Examples:
    python3 scripts/plots/eta_impact/plot_eta_impact.py
    python3 scripts/plots/eta_impact/plot_eta_impact.py \\
        --datasets as-skitter cit-Patents europe_osm --figsize 15 3
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

# Shorter display names for subplot titles (data keys stay unchanged).
DISPLAY_NAMES = {"soc-pokec-relationships": "soc-pokec"}
LINE_COLOR = "#E6550D"         # matches theta_impact's Oranges family
BEST_COLOR = "#17BECF"
ITER_COLOR = "#3182BD"         # % warp-centric iterations
NODE_COLOR = "#31A354"         # % warp-centric nodes
DEFAULT_ETA = 2048             # 4 * block_size(512), the compiled-in default
TICK_FS, LABEL_FS, LEGEND_FS, TAG_FS = 13, 15, 12, 15


def eta_label(e: int) -> str:
    """η in units of 512 = block_size (e.g. 2048 -> 4, 64M -> 131072)."""
    return f"{e / 512:g}"


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("--in", dest="in_path",
                    default=str(here / "eta_impact_results.json"))
    ap.add_argument("--out-prefix", default=str(here / "eta_impact"))
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="Subset of datasets to draw "
                         "(default: all in the results JSON)")
    ap.add_argument("--figsize", nargs=2, type=float, default=None,
                    metavar=("W", "H"),
                    help="Figure size (default: 5x3 per subplot)")
    ap.add_argument("--ncols", type=int, default=3,
                    help="Subplots per row (default 3)")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        print(f"ERROR: {in_path} not found. Run eta_impact.py first.",
              file=sys.stderr)
        return 1

    d = json.loads(in_path.read_text())
    datasets = args.datasets or d["datasets"]
    missing = [s for s in datasets if s not in d["data"]]
    if missing:
        print(f"ERROR: not in results JSON: {', '.join(missing)} "
              f"(have: {', '.join(d['data'])})", file=sys.stderr)
        return 1
    etas = d["etas"]
    xs = list(range(len(etas)))

    ncols = min(args.ncols, len(datasets))
    nrows = (len(datasets) + ncols - 1) // ncols
    figsize = tuple(args.figsize) if args.figsize else (5.0 * ncols,
                                                        3.0 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False,
                             constrained_layout=True)
    for ax in axes.flat[len(datasets):]:
        ax.set_visible(False)

    for si, stem in enumerate(datasets):
        ax = axes.flat[si]
        sweep = d["data"].get(stem, {}).get("sweep", {})
        runtimes, iter_pct, node_pct = [], [], []
        for e in etas:
            cell = sweep.get(str(e), {})
            bad = not cell or "error" in cell
            runtimes.append(np.nan if bad else cell["runtime_ms"])
            iter_pct.append(np.nan if bad
                            else cell.get("warp_iter_pct", np.nan))
            node_pct.append(np.nan if bad
                            else cell.get("warp_node_pct", np.nan))

        ok = [(i, r) for i, r in zip(xs, runtimes) if not np.isnan(r)]
        best_i = min(ok, key=lambda t: t[1])[0] if ok else None

        ax.plot(xs, runtimes, color=LINE_COLOR, marker="o", markersize=4,
                linewidth=1.5, zorder=2)
        ax2 = ax.twinx()
        ax2.plot(xs, iter_pct, color=ITER_COLOR, marker="^", markersize=4,
                 linewidth=1.2, linestyle="--", zorder=2)
        ax2.plot(xs, node_pct, color=NODE_COLOR, marker="s", markersize=3.5,
                 linewidth=1.2, linestyle="-.", zorder=2)
        ax2.set_ylim(-3, 103)
        ax2.tick_params(axis="y", labelsize=TICK_FS)
        if si % ncols == ncols - 1 or si == len(datasets) - 1:
            ax2.set_ylabel("warp-centric share (%)", fontsize=LABEL_FS)
        else:
            ax2.set_yticklabels([])
        if DEFAULT_ETA in etas:
            ax.axvline(etas.index(DEFAULT_ETA), color="#666666",
                       linestyle="--", linewidth=1.0, zorder=1)
        if best_i is not None:
            ax.scatter([best_i], [runtimes[best_i]], marker="*", s=220,
                       color=BEST_COLOR, edgecolor="black", linewidth=0.5,
                       zorder=3)

        ax.set_xticks(xs)
        ax.set_xticklabels([eta_label(e) for e in etas],
                           fontsize=TICK_FS - 2, rotation=60)
        ax.tick_params(axis="y", labelsize=TICK_FS)
        ax.grid(axis="y", linewidth=0.3, alpha=0.5)
        if si // ncols == nrows - 1:
            ax.set_xlabel(r"$\eta\ (\times 512)$", fontsize=LABEL_FS)
        if si % ncols == 0:
            ax.set_ylabel("runtime (ms)", fontsize=LABEL_FS)
        ax.set_title(DISPLAY_NAMES.get(stem, stem), fontsize=TAG_FS)

    handles = [
        mlines.Line2D([], [], color=LINE_COLOR, marker="o",
                      markersize=4, label="runtime (ms)"),
        mlines.Line2D([], [], color=ITER_COLOR, marker="^",
                      markersize=4, linestyle="--",
                      label="warp iters (%)"),
        mlines.Line2D([], [], color=NODE_COLOR, marker="s",
                      markersize=3.5, linestyle="-.",
                      label="warp nodes (%)"),
        mlines.Line2D([], [], color="#666666", linestyle="--",
                      linewidth=1.0,
                      label=rf"default $\eta$={eta_label(DEFAULT_ETA)}"),
        mlines.Line2D([], [], linestyle="none", marker="*",
                      markersize=12, color=BEST_COLOR,
                      markeredgecolor="black", label=r"best $\eta$"),
    ]
    # Shared legend above the grid; bbox_inches="tight" grows the canvas
    # to include it (matplotlib 3.6 has no constrained "outside" loc yet).
    fig.legend(handles=handles, fontsize=LEGEND_FS, frameon=True,
               ncol=len(handles), loc="lower center",
               bbox_to_anchor=(0.5, 1.0), handlelength=1.6, borderpad=0.4)

    pdf = Path(args.out_prefix + ".pdf")
    png = Path(args.out_prefix + ".png")
    pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    print(f"# wrote {pdf}\n# wrote {png}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
