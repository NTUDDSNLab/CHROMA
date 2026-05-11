"""
4-way EGR sweep: baseline vs v1_combined vs v2_combined vs v3_combined.

Uses the freshly-built CHROMA_v1, CHROMA_v2, CHROMA_v3 binaries from
.claude/worktrees/theta-predictor-v2/CHROMA/. Reports per-graph theta,
average runtime, and average colors over N runs. Output JSON written to
model/v2_data/egr_sweep_combined.json.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

# ---- Regexes (match the CHROMA stdout format) -----------------------------
EGC_RE       = re.compile(r"EGC θ:\s*(-?\d+)", re.IGNORECASE)
AVG_TOTAL_RE = re.compile(r"^\s*Total\s+time\s*:\s*avg=\s*([0-9.]+)",
                          re.IGNORECASE | re.MULTILINE)
AVG_COLOR_RE = re.compile(r"^\s*colors\s+used\s*:\s*avg=\s*([0-9.]+)",
                          re.IGNORECASE | re.MULTILINE)
RUN_RE       = re.compile(r"\[Run \d+/\d+\][^\n]*colors:\s*(\d+)\s+iters:\s*(\d+)")

ROOT = Path('/home/chsieh45/PunchShadow/CHROMA/.claude/worktrees/theta-predictor-v2')
BINS = {
    'baseline': str(ROOT / 'CHROMA' / 'CHROMA_v1'),     # baseline = v1 binary with -e 0
    'v1':       str(ROOT / 'CHROMA' / 'CHROMA_v1'),
    'v2':       str(ROOT / 'CHROMA' / 'CHROMA_v2'),
    'v3':       str(ROOT / 'CHROMA' / 'CHROMA_v3'),
}
ARGS = {
    'baseline': ['-e', '0'],
    'v1':       ['--predict'],
    'v2':       ['--predict', '--v2-model'],
    'v3':       ['--predict', '--v2-model'],
}

# Holdout vs trained-overlap split
TRAINED_OVERLAP = {
    'Email-Enron.col.egr', 'Slashdot0811.egr', 'Slashdot0902.egr', 'Stanford.egr',
    'as-skitter.egr', 'cit-Patents.egr', 'delaunay_n24.egr', 'soc-Epinions1.col.egr',
    'wiki-Talk.col.egr', 'wiki-Vote.col.egr', 'youtube.egr',
}


def parse(out: str):
    egc = EGC_RE.search(out)
    t = AVG_TOTAL_RE.search(out)
    c = AVG_COLOR_RE.search(out)
    runs = RUN_RE.findall(out)
    if not (egc and t and c and runs):
        return None
    return {
        'theta':        int(egc.group(1)),
        'avg_color':    float(c.group(1)),
        'avg_total_ms': float(t.group(1)),
    }


def run_one(binary: str, graph: Path, mode: str, runs: int = 5, timeout: int = 600):
    cmd = [binary, '-f', str(graph), '-a', 'cuSL_ELS_SDC_CTA_S',
           '--runs', str(runs)] + ARGS[mode]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return parse(proc.stdout)
    except subprocess.TimeoutExpired:
        return None


def fmt(rec):
    if rec is None:
        return '?'
    return f"θ={rec['theta']:>2} {rec['avg_total_ms']:8.1f}ms {rec['avg_color']:6.1f}c"


def main():
    egr_dir = Path('/home/chsieh45/PunchShadow/CHROMA/Datasets/EGR')
    graphs = sorted(egr_dir.glob('*.egr'))
    rows = []
    for g in graphs:
        rec = {'graph': g.name, 'in_train_overlap': g.name in TRAINED_OVERLAP}
        for mode, binary in BINS.items():
            rec[mode] = run_one(binary, g, mode)
        rows.append(rec)
        flag = 'trained' if rec['in_train_overlap'] else 'holdout'
        print(f"{g.name:36s} [{flag}]"
              f" | base {fmt(rec['baseline']):28s}"
              f" | v1 {fmt(rec['v1']):28s}"
              f" | v2 {fmt(rec['v2']):28s}"
              f" | v3 {fmt(rec['v3']):28s}",
              file=sys.stderr, flush=True)

    out_path = ROOT / 'model' / 'v2_data' / 'egr_sweep_combined.json'
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out_path}", file=sys.stderr)


if __name__ == '__main__':
    main()
