"""3-way sweep: baseline / v0_paper (pre-T21 model.cpp) / v3_bump (current default).

Uses CHROMA_v0_paper (built with model/model.cpp from commit 55b19d2 — the
2-feature decision tree on raw V/E that was active before T21 retrained
v1/v2/v3 on 456 graphs) and current CHROMA (v3 + dynamic-θ bump default).
Reports per-graph theta, avg runtime, avg colors over N runs.
"""
import argparse, json, re, subprocess, sys
from pathlib import Path

EGC_RE       = re.compile(r"EGC θ:\s*(-?\d+)", re.IGNORECASE)
AVG_TOTAL_RE = re.compile(r"^\s*Total\s+time\s*:\s*avg=\s*([0-9.]+)", re.IGNORECASE | re.MULTILINE)
AVG_COLOR_RE = re.compile(r"^\s*colors\s+used\s*:\s*avg=\s*([0-9.]+)",  re.IGNORECASE | re.MULTILINE)
RUN_RE       = re.compile(r"\[Run \d+/\d+\][^\n]*colors:\s*(\d+)\s+iters:\s*(\d+)")

ROOT = Path('/home/chsieh45/PunchShadow/CHROMA/.claude/worktrees/theta-predictor-v2')
BINS = {
    'baseline': str(ROOT / 'CHROMA' / 'CHROMA_v0_paper'),
    'v0_paper': str(ROOT / 'CHROMA' / 'CHROMA_v0_paper'),
    'v3_bump':  str(ROOT / 'CHROMA' / 'CHROMA'),
}
ARGS = {
    'baseline': ['-e', '0', '--no-dynamic-theta'],
    'v0_paper': ['--predict'],          # paper-era binary built with DYNAMIC_THETA=0
    'v3_bump':  ['--predict'],          # current default = v3 + bump
}

TRAINED_OVERLAP = {
    'Email-Enron.col.egr', 'Slashdot0811.egr', 'Slashdot0902.egr', 'Stanford.egr',
    'as-skitter.egr', 'cit-Patents.egr', 'delaunay_n24.egr', 'soc-Epinions1.col.egr',
    'wiki-Talk.col.egr', 'wiki-Vote.col.egr', 'youtube.egr',
}


def parse(out):
    egc, t, c, runs = (EGC_RE.search(out), AVG_TOTAL_RE.search(out),
                        AVG_COLOR_RE.search(out), RUN_RE.findall(out))
    if not (egc and t and c and runs):
        return None
    return {'theta': int(egc.group(1)), 'avg_color': float(c.group(1)),
            'avg_total_ms': float(t.group(1)), 'n_runs': len(runs)}


def run_one(binary, graph, mode, runs, timeout):
    cmd = [binary, '-f', str(graph), '-a', 'cuSL_ELS_SDC_CTA_S',
           '--runs', str(runs)] + ARGS[mode]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return parse(proc.stdout)
    except subprocess.TimeoutExpired:
        return None


def fmt(rec):
    if rec is None: return '?'
    return f"θ={rec['theta']:>2} {rec['avg_total_ms']:8.1f}ms {rec['avg_color']:6.1f}c"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=int, default=5)
    ap.add_argument('--timeout', type=int, default=600)
    ap.add_argument('--out', default='model/v2_data/egr_v0_paper.json')
    ap.add_argument('--egr-dir', default='/home/chsieh45/PunchShadow/CHROMA/Datasets/EGR')
    args = ap.parse_args()

    graphs = sorted(Path(args.egr_dir).glob('*.egr'))
    print(f"# {len(graphs)} graphs, runs={args.runs}", file=sys.stderr)

    rows = []
    for g in graphs:
        rec = {'graph': g.name, 'in_train_overlap': g.name in TRAINED_OVERLAP}
        for mode in ('baseline', 'v0_paper', 'v3_bump'):
            rec[mode] = run_one(BINS[mode], g, mode, args.runs, args.timeout)
        flag = 'trained' if rec['in_train_overlap'] else 'holdout'
        print(f"{g.name:36s} [{flag}]"
              f" | base {fmt(rec['baseline']):26s}"
              f" | v0_paper {fmt(rec['v0_paper']):26s}"
              f" | v3_bump {fmt(rec['v3_bump']):26s}",
              file=sys.stderr, flush=True)
        rows.append(rec)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out_path}", file=sys.stderr)


if __name__ == '__main__':
    main()
