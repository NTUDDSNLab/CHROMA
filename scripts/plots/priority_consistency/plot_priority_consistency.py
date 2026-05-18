#!/usr/bin/env python3
"""Render the priority-consistency figure (paper Fig. 5 redraw).

Single axes; one grouped cluster per dataset; one solid-colour bar per
framework; y-axis "Consistency" as percent with a configurable floor
(default 10%, low enough to show every bar); 19 datasets on the x-axis
sorted alphabetically by
displayed name (deterministic across servers); one horizontal legend on
top; no title.

Examples:
    python3 scripts/plots/priority_consistency/plot_priority_consistency.py
    python3 scripts/plots/priority_consistency/plot_priority_consistency.py \\
        --in scripts/plots/priority_consistency/consistency_results.json \\
        --ymin 0 --figsize 16 4
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
import matplotlib.ticker as mticker
import numpy as np

# Framework key -> (display label, fill colour). Order = bar order.
FRAMEWORK_STYLE = [
    ("JP-SL^A",   r"JP-SL$^{A}$ (self)", "#4D4D4D"),
    ("JP-SLL",    "JP-SLL",     "#4C72B0"),
    ("JP-ADG",    "JP-ADG",     "#DD8452"),
    ("CHROMA",    "CHROMA",     "#C44E52"),
    ("CHROMA+",        r"CHROMA$^{+}$",            "#8172B3"),
    ("CHROMA*",        r"CHROMA$^{*}$",            "#937860"),
    ("CHROMA_v2_raw",  r"CHROMA$_{v2}$-b",  "#DA8BC3"),
    ("CHROMA_v2_bump", r"CHROMA$_{v2}$",    "#CCB974"),
]

DATASET_LABELS = {
    "wiki-Vote.col":               "wiki-Vote",
    "Email-Enron.col":             "Email-Enron",
    "soc-Epinions1.col":           "soc-Epinions1",
    "wiki-Talk.col":               "wiki-Talk",
    "twitter_combined":            "twitter",
    "soc-pokec-relationships.col": "soc-Pokec",
}

TICK_FS, LABEL_FS, LEGEND_FS = 11, 13, 12


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path",
                    default=str(here / "consistency_results.json"))
    ap.add_argument("--out-prefix",
                    default=str(here / "priority_consistency"))
    ap.add_argument("--ymin", type=float, default=10.0,
                    help="Y-axis floor in percent (default 10 — shows every "
                         "bar; global min is europe_osm CHROMA ~14.7%)")
    ap.add_argument("--figsize", nargs=2, type=float, default=[14.0, 3.2],
                    metavar=("W", "H"))
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        print(f"ERROR: {in_path} not found. Run "
              f"sweep_priority_consistency.py first.", file=sys.stderr)
        sys.exit(1)

    d = json.loads(in_path.read_text())
    json_fw = set(d["frameworks"])
    fw_style = [(k, lbl, c) for (k, lbl, c) in FRAMEWORK_STYLE
                if k in json_fw]
    by_key = {(r["framework"], r["dataset"]): r for r in d["rows"]}

    datasets = sorted(
        d["datasets"],
        key=lambda x: DATASET_LABELS.get(x["name"], x["name"]).lower())
    names = [x["name"] for x in datasets]
    disp = [DATASET_LABELS.get(n, n) for n in names]
    n_ds = len(datasets)
    n_fw = len(fw_style)

    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    centers = np.arange(n_ds)
    bar_w = 0.85 / max(1, n_fw)
    offsets = (np.arange(n_fw) - (n_fw - 1) / 2) * bar_w

    for i, (key, _lbl, color) in enumerate(fw_style):
        # Percent; missing/errored cells render as a 0-height gap.
        ys = np.array([
            (by_key.get((key, nm), {}).get("consistency_ratio") or 0.0) * 100.0
            for nm in names])
        ax.bar(centers + offsets[i], ys, width=bar_w, color=color,
               edgecolor="black", linewidth=0.6)

    ax.set_xticks(centers)
    ax.set_xticklabels(disp, rotation=35, ha="right", fontsize=TICK_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.set_ylabel("Consistency", fontsize=LABEL_FS)
    ax.set_ylim(args.ymin, 100.0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.set_xlim(-0.5, n_ds - 0.5)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    handles = [mpatches.Patch(facecolor=c, edgecolor="black",
                              linewidth=0.6, label=lbl)
               for (_k, lbl, c) in fw_style]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.00), ncol=len(handles),
               frameon=False, fontsize=LEGEND_FS, columnspacing=1.8)

    fig.subplots_adjust(top=0.88, bottom=0.28, left=0.06, right=0.995)

    pdf = Path(args.out_prefix + ".pdf")
    png = Path(args.out_prefix + ".png")
    pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    print(f"# wrote {pdf}\n# wrote {png}", file=sys.stderr)


if __name__ == "__main__":
    main()
