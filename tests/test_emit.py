import re
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from model import emit


def test_emit_scaler_header(tmp_path):
    X = np.array([[1, 2, 3, 4, 5, 6, 7],
                  [2, 4, 6, 8, 10, 12, 14],
                  [3, 6, 9, 12, 15, 18, 21]], dtype=float)
    scaler = StandardScaler().fit(X)
    out = tmp_path / "scaler.h"
    emit.write_scaler_header(scaler, model_class="linear",
                              feature_names=("V","E","d","s","R","GI","H_er"),
                              path=out)
    text = out.read_text()
    # Required tokens
    assert "FEATURE_COUNT = 7" in text
    assert "SCALER_MEAN[" in text
    assert "SCALER_STD["  in text
    assert 'MODEL_CLASS = "linear"' in text
    # 7 numbers in MEAN, 7 in STD
    nums = re.findall(r"-?\d+\.\d+(?:[eE][-+]?\d+)?", text)
    assert len(nums) >= 14
    # mean[0] should be 2.0, std[0] = sqrt(((1-2)²+0+(3-2)²)/3) ≈ 0.8165
    mean_block = text.split("SCALER_MEAN")[1].split("};")[0]
    assert "2.0" in mean_block
