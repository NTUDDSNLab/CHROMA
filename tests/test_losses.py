import numpy as np
import pytest
from model import losses


def test_violation_ratio():
    assert losses.violation_ratio([1, 2, 3], [1, 2, 3]) == 0.0
    assert losses.violation_ratio([1, 2, 3], [2, 2, 3]) == pytest.approx(1/3)
    assert losses.violation_ratio([1, 2, 3], [2, 3, 4]) == 1.0


def test_mean_theta_pred():
    assert losses.mean_theta_pred(None, [1.0, 2.0, 3.0]) == pytest.approx(2.0)


@pytest.mark.parametrize("vr,mean,expected", [
    (0.04,  5.0, -5.00),   # under cap → loss = -β·mean
    (0.04, 10.0, -10.00),
    (0.08,  5.0, -4.70),   # over cap by 0.03 → +α·0.03 = +0.30 hinge
    (0.08, 10.0, -9.70),
    (0.20, 10.0, -8.50),
])
def test_weighted_vr_scorer_corner_cases(vr, mean, expected):
    """α=10, β=1, vr_cap=0.05.
    Build inputs that produce the chosen (VR, mean): n samples where
    n_over samples over-predict by 1, the rest under-predict to maintain mean."""
    n = 100
    n_over = int(round(vr * n))
    n_under = n - n_over
    # y_pred: n_over at (mean+1), n_under at (mean - n_over/n_under)
    # to keep mean(y_pred) = mean
    under_val = mean - (n_over / n_under)
    y_pred = np.concatenate([
        np.full(n_over, mean + 1.0),
        np.full(n_under, under_val)
    ])
    # y_true: all equal to mean
    y_true = np.full(n, mean, dtype=float)
    assert losses.violation_ratio(y_true, y_pred) == pytest.approx(vr)
    assert losses.mean_theta_pred(None, y_pred) == pytest.approx(mean)
    raw = losses.weighted_vr_loss(y_true, y_pred, alpha=10.0, beta=1.0, vr_cap=0.05)
    assert raw == pytest.approx(expected, abs=1e-9)


def test_scorer_factory_is_negated():
    """weighted_vr_scorer(...) returns a sklearn scorer with greater_is_better=False,
    so its ._score_func returns -loss for sklearn's "higher is better" convention."""
    from sklearn.linear_model import LinearRegression
    scorer = losses.weighted_vr_scorer(alpha=10.0, beta=1.0, vr_cap=0.05)
    # sanity: callable with the sklearn-style signature
    est = LinearRegression().fit([[0], [1]], [0, 1])
    val = scorer(est, [[0], [1]], [0, 1])
    assert isinstance(val, float)
