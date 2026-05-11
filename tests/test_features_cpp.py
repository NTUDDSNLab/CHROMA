# tests/test_features_cpp.py
import json
import subprocess
from pathlib import Path

import pytest
from model import features

REPO    = Path(__file__).parent.parent
EXTRACT = REPO / "tools" / "feature_extractor" / "extract"


@pytest.mark.skipif(not EXTRACT.is_file(),
                     reason="run `make` in tools/feature_extractor first")
@pytest.mark.parametrize("name", ["K_5", "P_10", "C_10", "star_K_1_9"])
def test_python_cpp_parity(canonical_egr_fixtures, name):
    egr = canonical_egr_fixtures[name]
    cpp = json.loads(subprocess.run(
        [str(EXTRACT), str(egr)],
        capture_output=True, text=True, check=True, timeout=10,
    ).stdout)
    py  = features.compute_features(features.load_ecl_graph(egr))
    for key in ("V", "E", "d", "s", "R", "GI", "H_er", "kcore", "assort"):
        assert cpp[key] == pytest.approx(py[key], abs=1e-9, rel=1e-9), (key, cpp, py)
