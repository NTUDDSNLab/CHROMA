"""Loss / scorer functions for the θ predictor.

The headline scorer is `weighted_vr_scorer`: a hinge on a violation-ratio
cap plus a reward on mean predicted θ. Selects models that stay under the
VR ceiling AND push predictions as high as possible.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import make_scorer


def violation_ratio(y_true, y_pred) -> float:
    """Fraction of samples where ŷ > y. ŷ may legitimately equal y."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_pred > y_true))


def mean_theta_pred(y_true, y_pred) -> float:
    """Mean of predictions. y_true ignored (kept for sklearn signature)."""
    return float(np.mean(np.asarray(y_pred)))


def weighted_vr_loss(y_true, y_pred,
                     alpha: float = 10.0,
                     beta:  float = 1.0,
                     vr_cap: float = 0.05) -> float:
    """Lower-is-better scalar:
        loss = α · max(0, VR − vr_cap) − β · mean(ŷ)
    """
    vr = violation_ratio(y_true, y_pred)
    m  = mean_theta_pred(y_true, y_pred)
    return alpha * max(0.0, vr - vr_cap) - beta * m


def weighted_vr_scorer(alpha: float = 10.0,
                       beta:  float = 1.0,
                       vr_cap: float = 0.05):
    """Return a sklearn-compatible scorer (greater_is_better=False)."""
    def _score(y_true, y_pred):
        return weighted_vr_loss(y_true, y_pred, alpha, beta, vr_cap)
    return make_scorer(_score, greater_is_better=False)
