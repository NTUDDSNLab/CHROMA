import json, re, subprocess, sys
from pathlib import Path

EGC_RE       = re.compile(r"EGC θ:\s*(-?\d+)", re.IGNORECASE)
AVG_TOTAL_RE = re.compile(r"^\s*Total\s+time\s*:\s*avg=\s*([0-9.]+)", re.IGNORECASE | re.MULTILINE)
AVG_COLOR_RE = re.compile(r"^\s*colors\s+used\s*:\s*avg=\s*([0-9.]+)",  re.IGNORECASE | re.MULTILINE)
RUN_RE       = re.compile(r"\[Run \d+/\d+\][^\n]*colors:\s*(\d+)\s+iters:\s*(\d+)")

BINS = {
    'baseline':       '/home/chsieh45/PunchShadow/CHROMA/.claude/worktrees/theta-predictor-v2/CHROMA/CHROMA',
    'v1':             '/home/chsieh45/PunchShadow/CHROMA/CHROMA/CHROMA',
    'v2_old_dep':     '/tmp/CHROMA_v2_deployed',
    'v2_clean_dep':   '/tmp/CHROMA_v2_clean',
    'v2_clean_raw':   '/tmp/CHROMA_v2_clean_raw',
}
ARGS = {
    'baseline':     ['-e', '0'],
    'v1':           ['--predict'],
    'v2_old_dep':   ['--predict', '--v2-model'],
    'v2_clean_dep': ['--predict', '--v2-model'],
    'v2_clean_raw': ['--predict', '--v2-model'],
}

def parse(out):
    egc, t, c, runs = (EGC_RE.search(out), AVG_TOTAL_RE.search(out),
                        AVG_COLOR_RE.search(out), RUN_RE.findall(out))
    if not (egc and t and c and runs): return None
    return {'theta': int(egc.group(1)), 'avg_color': float(c.group(1)),
            'avg_total_ms': float(t.group(1))}

def run_one(binary, graph, mode, runs=5, timeout=600):
    cmd = [binary, '-f', str(graph), '-a', 'cuSL_ELS_SDC_CTA_S', '--runs', str(runs)] + ARGS[mode]
    try: return parse(subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout)
    except subprocess.TimeoutExpired: return None

graphs = sorted(Path('/home/chsieh45/PunchShadow/CHROMA/Datasets/EGR').glob('*.egr'))
overlap_graphs = {'Email-Enron.col.egr', 'Slashdot0811.egr', 'Slashdot0902.egr',
                  'Stanford.egr', 'as-skitter.egr', 'cit-Patents.egr',
                  'delaunay_n24.egr', 'soc-Epinions1.col.egr',
                  'wiki-Talk.col.egr', 'wiki-Vote.col.egr', 'youtube.egr'}

rows = []
for g in graphs:
    rec = {'graph': g.name, 'in_train_overlap': g.name in overlap_graphs}
    for mode, binary in BINS.items():
        rec[mode] = run_one(binary, g, mode)
    rows.append(rec)

    def fmt(m):
        r = rec[m]
        if r is None: return '?'
        return f'θ={r["theta"]:>2} {r["avg_total_ms"]:7.1f}ms {r["avg_color"]:5.1f}c'

    flag = '⚠ trained' if rec['in_train_overlap'] else '· holdout'
    print(f'{g.name:32s} {flag} | base {fmt("baseline"):28s} | v1 {fmt("v1"):28s} | old {fmt("v2_old_dep"):28s} | clean_dep {fmt("v2_clean_dep"):28s} | clean_raw {fmt("v2_clean_raw"):28s}',
          file=sys.stderr)

Path('/tmp/egr_5way.json').write_text(json.dumps(rows, indent=2))
print(f'\nWrote /tmp/egr_5way.json', file=sys.stderr)
