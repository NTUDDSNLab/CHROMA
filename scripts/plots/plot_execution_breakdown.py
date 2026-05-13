#!/usr/bin/env python3
"""Render the execution-time breakdown figure from batch_profile_results.json.

Default layout: datasets are split into multiple panels by total execution
time so each panel has its own y-axis scale and small bars stay readable
alongside europe_osm. Panels are laid out horizontally; within each panel
datasets are sorted by edge count ascending. Each bar is a stack of CA
(bottom) / PA scan / PA decrement (top); each framework gets a distinct
hatch. Two horizontal legends sit above the figure: stack colours and
framework hatches. No title.

Use --bins "" to render a single panel (legacy single-axes mode).

Examples:
    python3 scripts/plots/plot_execution_breakdown.py
    python3 scripts/plots/plot_execution_breakdown.py --bins 100,300,1000
    python3 scripts/plots/plot_execution_breakdown.py --bins ""    # single panel
    python3 scripts/plots/plot_execution_breakdown.py --log
    python3 scripts/plots/plot_execution_breakdown.py \\
        --in scripts/batch_profile_results.json \\
        --out-prefix /tmp/figure --figsize 18 5
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
import numpy as np


SEGMENTS_BREAKDOWN = [
    ("ca_ms",            "CA time",           "#4C72B0"),  # blue
    ("pa_scan_ms",       "PA scan time",      "#DD8452"),  # orange
    ("pa_decrement_ms",  "PA decrement time", "#55A467"),  # green
]
SEGMENTS_UNIFIED = [
    ("ca_ms",  "CA time", "#4C72B0"),  # blue
    ("pa_ms",  "PA time", "#DD8452"),  # orange
]
HATCHES = ["",  "//", "xx", ".."]  # one per framework, in input order

# Font sizes (pt). Tuned for a paper-figure-sized PDF.
TICK_FS   = 12
LABEL_FS  = 13
LEGEND_FS = 13

FRAMEWORK_LABELS = {
    # Unified cooperative kernels. CHROMA^* (superscript asterisk via
    # mathtext) marks the SDC family in paper figures; the suffix encodes
    # the workload-balancing strategy:
    #   * (no suffix) baseline warp-per-vertex SDC
    #   -cta         CTA-balanced decrement (BlockScan)
    #   -adw         adaptive dispatched warp/CTA (a.k.a. CTA_S)
    #   -adw-d       same as -adw, with the on-device dynamic-θ
    #                 bumping controller enabled
    "cuSL_ELS_SDC":          r"CHROMA$^{*}$",
    "cuSL_ELS_SDC_CTA":      r"CHROMA$^{*}$-cta",
    "cuSL_ELS_SDC_CTA_S":    r"CHROMA$^{*}$-adw",
    "cuSL_ELS_SDC_CTA_S_D":  r"CHROMA$^{*}$-adw-d",
    # Diagnostic SPLIT-mode variants reuse the same paper labels.
    "cuSL_ELS_SDC_SPLIT":        r"CHROMA$^{*}$",
    "cuSL_ELS_SDC_CTA_SPLIT":    r"CHROMA$^{*}$-cta",
    "cuSL_ELS_SDC_CTA_S_SPLIT":  r"CHROMA$^{*}$-adw",
}

# X-axis display names. The .egr basenames carry historical suffixes
# (.col, _combined, …) and the relationships graph has a long name; the
# paper's results table uses the shorter forms below, so the figure
# aligns by mapping at the display layer only.
DATASET_LABELS = {
    "wiki-Vote.col":               "wiki-Vote",
    "Email-Enron.col":             "Email-Enron",
    "soc-Epinions1.col":           "soc-Epinions1",
    "wiki-Talk.col":               "wiki-Talk",
    "twitter_combined":            "twitter",
    "soc-pokec-relationships.col": "soc-Pokec",
}


def load(path: Path):
    d = json.loads(path.read_text())
    by_key = {(r["framework"], r["dataset"]): r for r in d["rows"]}
    return d["frameworks"], d["datasets"], by_key


def pick_segments(by_key):
    """If every row has the SPLIT per-phase fields, render the
    three-segment breakdown; otherwise fall back to two segments
    (CA + unified PA)."""
    for r in by_key.values():
        if "error" in r:
            continue
        if "pa_scan_ms" not in r or "pa_decrement_ms" not in r:
            return SEGMENTS_UNIFIED
    return SEGMENTS_BREAKDOWN


def max_total_ms(name, frameworks, by_key, segments) -> float:
    seg_keys = [k for k, _l, _c in segments]
    best = 0.0
    for fw in frameworks:
        r = by_key.get((fw, name), {})
        t = sum(r.get(k, 0.0) for k in seg_keys)
        if t > best:
            best = t
    return best


def bin_datasets(datasets, frameworks, by_key, segments, bin_edges):
    """Partition datasets into len(bin_edges)+1 bins by max total time.

    bin_edges is a sorted ascending list of millisecond thresholds. Returns
    a list of (label, ds_list) pairs in display order; empty bins are
    dropped. Within each bin, ds_list is sorted by edge count ascending.
    """
    n_bins = len(bin_edges) + 1
    buckets = [[] for _ in range(n_bins)]
    labels = []
    if not bin_edges:
        labels = ["all"]
    else:
        labels = [f"<{bin_edges[0]:g} ms"]
        for i in range(len(bin_edges) - 1):
            labels.append(f"{bin_edges[i]:g}–{bin_edges[i+1]:g} ms")
        labels.append(f">{bin_edges[-1]:g} ms")

    for ds in datasets:
        t = max_total_ms(ds["name"], frameworks, by_key, segments)
        idx = 0
        for j, edge in enumerate(bin_edges):
            if t >= edge:
                idx = j + 1
        buckets[idx].append(ds)

    out = []
    for label, ds_list in zip(labels, buckets):
        if not ds_list:
            continue
        ds_list = sorted(ds_list, key=lambda d: d["edges"])
        out.append((label, ds_list))
    return out


def draw_panel(ax, panel_datasets, frameworks, by_key, segments, bar_w):
    n_ds = len(panel_datasets)
    n_fw = len(frameworks)
    group_centers = np.arange(n_ds)
    offsets = (np.arange(n_fw) - (n_fw - 1) / 2) * bar_w
    names = [d["name"] for d in panel_datasets]

    for fw_idx, fw in enumerate(frameworks):
        xs = group_centers + offsets[fw_idx]
        bottoms = np.zeros(n_ds)
        for seg_key, _seg_label, seg_color in segments:
            heights = np.array([by_key.get((fw, name), {}).get(seg_key, 0.0)
                                for name in names])
            ax.bar(xs, heights, width=bar_w, bottom=bottoms,
                   color=seg_color,
                   edgecolor="black", linewidth=0.8,
                   hatch=HATCHES[fw_idx])
            bottoms += heights

    ax.set_xticks(group_centers)
    ax.set_xticklabels([DATASET_LABELS.get(n, n) for n in names],
                       rotation=35, ha="right", fontsize=TICK_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.set_xlim(-0.5, n_ds - 0.5)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main():
    repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path",
                    default=str(repo / "scripts" / "batch_profile_results.json"))
    ap.add_argument("--out-prefix",
                    default=str(repo / "scripts" / "plots" / "execution_time_breakdown"))
    ap.add_argument("--bins", default="20,100,300,1000",
                    help="Comma-separated ascending ms thresholds for splitting "
                         "datasets across horizontal panels. Pass --bins \"\" for "
                         "a single panel. Default: 20,100,300,1000 (5 panels).")
    ap.add_argument("--log", action="store_true",
                    help="Use symlog y-axis on every panel (rarely needed once "
                         "panels are split by magnitude).")
    ap.add_argument("--figsize", nargs=2, type=float, default=[16.0, 4.0],
                    metavar=("W", "H"),
                    help="Figure size in inches. Default 16 4.")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        print(f"ERROR: {in_path} not found. Run batch_profile.py first.", file=sys.stderr)
        sys.exit(1)

    frameworks, datasets, by_key = load(in_path)
    segments = pick_segments(by_key)

    bin_edges = []
    if args.bins.strip():
        try:
            bin_edges = sorted(float(x) for x in args.bins.split(","))
        except ValueError:
            print(f"ERROR: --bins must be comma-separated numbers, got {args.bins!r}",
                  file=sys.stderr)
            sys.exit(1)

    panels = bin_datasets(datasets, frameworks, by_key, segments, bin_edges)
    if not panels:
        print("ERROR: no datasets to plot.", file=sys.stderr)
        sys.exit(1)

    # Width ratios reflect the dataset count in each panel so bar widths
    # look consistent across the figure. Single-panel mode just stretches.
    width_ratios = [len(ds_list) for _, ds_list in panels]

    fig, axes = plt.subplots(
        nrows=1, ncols=len(panels),
        figsize=tuple(args.figsize),
        gridspec_kw={"width_ratios": width_ratios, "wspace": 0.18},
        squeeze=False,
    )
    axes = axes[0]

    # bar_w scales so the bars in a group span ~85% of the unit slot,
    # leaving ~15% white space between adjacent dataset groups regardless
    # of framework count (3 fw → ~0.28, 4 fw → ~0.21).
    bar_w = 0.85 / max(1, len(frameworks))
    for ax, (_label, ds_list) in zip(axes, panels):
        draw_panel(ax, ds_list, frameworks, by_key, segments, bar_w)
        if args.log:
            ax.set_yscale("symlog", linthresh=0.1)

    # Only the leftmost panel carries the y-axis title; other panels
    # keep their own tick numbers (independent scale) but drop the
    # repeated "Time (ms)" label.
    axes[0].set_ylabel("Time (ms)", fontsize=LABEL_FS)

    # Single combined horizontal legend above the axes: segment colours
    # first, then framework hatches. Single row keeps the top strip thin
    # so the figure stays short without legend rows colliding.
    seg_handles = [mpatches.Patch(facecolor=c, edgecolor="black",
                                   linewidth=0.8, label=lbl)
                   for _k, lbl, c in segments]
    fw_handles = [mpatches.Patch(facecolor="lightgrey", edgecolor="black",
                                  linewidth=0.8, hatch=HATCHES[i],
                                  label=FRAMEWORK_LABELS.get(fw, fw))
                  for i, fw in enumerate(frameworks)]
    combined_handles = seg_handles + fw_handles

    fig.legend(handles=combined_handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.02),
               ncol=len(combined_handles),
               frameon=False, handlelength=2.0, columnspacing=2.5,
               fontsize=LEGEND_FS)

    # Margins: drop the axes top to 0.84 so the legend sits well above
    # the bars with clear whitespace; rotated dataset labels need ~25%
    # at the bottom; side gutters as small as practical.
    fig.subplots_adjust(top=0.84, bottom=0.27, left=0.035, right=0.997,
                        wspace=0.16)

    pdf_path = Path(args.out_prefix + ".pdf")
    png_path = Path(args.out_prefix + ".png")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=200)
    print(f"# wrote {pdf_path}\n# wrote {png_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
