# tests/test_train_pipeline.py
import subprocess
import json
from pathlib import Path

FIXDIR = Path(__file__).parent / "fixtures"
REPO   = Path(__file__).parent.parent


def test_train_v2_smoke(tmp_path):
    """End-to-end: tiny JSON + .egr fixtures → model_v2.cpp + scaler.h + meta."""
    out_cpp    = tmp_path / "model_v2.cpp"
    out_scaler = tmp_path / "scaler.h"
    out_meta   = tmp_path / "meta.json"
    proc = subprocess.run(
        ["python3", "-m", "model.train",
         "--v2",
         "--input",       str(FIXDIR / "tiny_training.json"),
         "--graph-dir",   str(FIXDIR),
         "--model",       "linear",
         "--no-grid",
         "--cv-folds",    "2",         # tiny dataset
         "--out-cpp",     str(out_cpp),
         "--out-scaler",  str(out_scaler),
         "--out-meta",    str(out_meta)],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert out_cpp.exists()
    assert "score" in out_cpp.read_text()
    assert "FEATURE_COUNT = 7" in out_scaler.read_text()
    meta = json.loads(out_meta.read_text())
    assert meta["feature_count"] == 7
    assert meta["model_class"] == "linear"
    assert "cv_results" in meta
