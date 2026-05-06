#!/usr/bin/env python3
"""Compare two batch_test.py sweep JSONs (BB vs reference) and print
per-dataset table: colors_diff, PA_speedup, CA_delta_pct, total_speedup.

Usage:
  python3 scripts/bb_diff_report.py bb_sweep.json cusl_sweep.json
"""
import argparse
import json
import sys
from typing import Any, Dict, Optional


def load_datasets(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict) or "datasets" not in data:
        raise ValueError(f"{path}: expected top-level dict with 'datasets' key")
    ds = data["datasets"]
    if not isinstance(ds, dict):
        raise ValueError(f"{path}: 'datasets' must be a dict")
    return ds


def fmt_int(v: Optional[int]) -> str:
    if v is None:
        return "—"
    return f"{v:+d}"


def fmt_speedup(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"{v:.2f}x"


def fmt_pct(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"{v:+.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser(description="Diff BB vs reference sweep JSONs")
    ap.add_argument("bb_json", help="BB-cuSL sweep JSON (e.g. bb_sweep.json)")
    ap.add_argument("ref_json", help="Reference sweep JSON (e.g. cusl_sweep.json)")
    args = ap.parse_args()

    bb  = load_datasets(args.bb_json)
    ref = load_datasets(args.ref_json)

    keys = sorted(set(bb) & set(ref))
    if not keys:
        print("No overlapping datasets between the two JSONs.", file=sys.stderr)
        return 1

    header = (
        f"{'dataset':<40} {'theta':>6} {'colors_diff':>12} "
        f"{'PA_speedup':>12} {'CA_delta':>10} {'total_speedup':>14}"
    )
    print(header)
    print("-" * len(header))

    pa_speedups = []
    color_diffs = []
    ca_regressions = []

    for ds in keys:
        b = bb[ds]
        r = ref[ds]

        bc = b.get("best_color")
        rc = r.get("best_color")
        cdiff = (bc - rc) if (bc is not None and rc is not None) else None

        bpa = b.get("best_pa_runtime_ms")
        rpa = r.get("best_pa_runtime_ms")
        pa_spd = (rpa / bpa) if (bpa and rpa and bpa > 0) else None

        bca = b.get("best_ca_runtime_ms")
        rca = r.get("best_ca_runtime_ms")
        ca_delta = ((bca - rca) / rca * 100.0) if (bca is not None and rca and rca > 0) else None

        btot = b.get("best_runtime_ms")
        rtot = r.get("best_runtime_ms")
        tot_spd = (rtot / btot) if (btot and rtot and btot > 0) else None

        theta = b.get("theta")

        if pa_spd is not None: pa_speedups.append((ds, pa_spd))
        if cdiff is not None:  color_diffs.append((ds, cdiff))
        if ca_delta is not None and abs(ca_delta) > 10.0:
            ca_regressions.append((ds, ca_delta))

        print(
            f"{ds:<40} {str(theta) if theta is not None else '—':>6} "
            f"{fmt_int(cdiff):>12} "
            f"{fmt_speedup(pa_spd):>12} "
            f"{fmt_pct(ca_delta):>10} "
            f"{fmt_speedup(tot_spd):>14}"
        )

    # Summary
    print()
    print("=== Summary ===")
    if pa_speedups:
        best_pa = max(pa_speedups, key=lambda x: x[1])
        worst_pa = min(pa_speedups, key=lambda x: x[1])
        gmean = 1.0
        for _, s in pa_speedups: gmean *= s
        gmean **= (1.0 / len(pa_speedups))
        print(f"PA speedup: best {best_pa[1]:.2f}x ({best_pa[0]}), "
              f"worst {worst_pa[1]:.2f}x ({worst_pa[0]}), "
              f"geomean {gmean:.2f}x")
    if color_diffs:
        better = sum(1 for _, d in color_diffs if d < 0)
        same   = sum(1 for _, d in color_diffs if d == 0)
        worse  = sum(1 for _, d in color_diffs if d > 0)
        print(f"Colors: BB better on {better}, same on {same}, worse on {worse}")
    if ca_regressions:
        print(f"CA regressions > 10%: {len(ca_regressions)} datasets")
        for ds, d in ca_regressions:
            print(f"  {ds}: {d:+.1f}%")
    else:
        print("CA regressions > 10%: none")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
