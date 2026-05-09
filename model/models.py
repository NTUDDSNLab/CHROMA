"""Registry of trainable θ-predictor model classes + their hyperparameter
grids. `deployable` flags the ones m2cgen can emit to C++ (everything but
SVR, currently).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict

from sklearn.linear_model import LinearRegression
from sklearn.ensemble    import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm         import SVR


def _xgb_factory():
    import xgboost as xgb
    return xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                             objective="reg:squarederror", random_state=42)


def _xgb_pinball_factory(q: float = 0.15):
    """Return a 0-arg factory that builds an XGBRegressor with pinball/quantile
    custom objective targeting the q-th quantile of y|x. q < 0.5 → predictions
    are biased low (conservative). The custom objective is per-sample with
    constant hessian = 1.0 (linear loss approximation that XGBoost handles well).
    """
    import numpy as np
    import xgboost as xgb

    def _pinball_obj(y_true, y_pred):
        err = y_pred - y_true
        grad = np.where(err > 0, 1.0 - q, -q)
        hess = np.ones_like(err)
        return grad, hess

    def factory():
        return xgb.XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            base_score=0.0, random_state=42, verbosity=0,
            objective=_pinball_obj,
        )
    return factory


def set_xgb_pinball_quantile(q: float) -> None:
    """Override the q baked into the xgb_pinball spec's factory. Lets train.py
    expose --xgb-quantile without having to re-register the spec."""
    spec = get_spec("xgb_pinball")
    spec.factory = _xgb_pinball_factory(q)


def _lgbm_factory():
    import lightgbm as lgb
    return lgb.LGBMRegressor(n_estimators=100, max_depth=-1, learning_rate=0.1,
                              random_state=42, verbose=-1)


@dataclass
class ModelSpec:
    name: str
    factory: Callable[[], Any]
    grid: Dict[str, list] = field(default_factory=dict)
    deployable: bool = True


SPECS = [
    ModelSpec("linear",
              lambda: LinearRegression(),
              {}, deployable=True),
    ModelSpec("rf",
              lambda: RandomForestRegressor(random_state=42, n_jobs=-1),
              {"n_estimators":     [100, 300, 500, 800],
               "max_depth":        [5, 10, 20, None],
               "min_samples_leaf": [1, 2, 3, 5, 10],
               "max_features":     ["sqrt", 0.5, 1.0]},
              deployable=True),
    ModelSpec("gbr",
              lambda: GradientBoostingRegressor(random_state=42),
              {"n_estimators":  [100, 300, 500],
               "max_depth":     [3, 5, 7],
               "learning_rate": [0.03, 0.05, 0.1],
               "subsample":     [0.7, 1.0]},
              deployable=False),  # m2cgen does not support sklearn GBR
    ModelSpec("xgb",
              _xgb_factory,
              {"n_estimators": [100, 300],
               "max_depth":    [3, 6],
               "learning_rate":[0.05, 0.1]},
              deployable=True),
    ModelSpec("xgb_pinball",
              _xgb_pinball_factory(0.15),  # default q; override via --xgb-quantile
              {"n_estimators":  [200, 300, 500],
               "max_depth":     [3, 5, 7],
               "learning_rate": [0.05, 0.1]},
              deployable=True),
    ModelSpec("lgbm",
              _lgbm_factory,
              {"n_estimators": [100, 300],
               "num_leaves":   [15, 31, 63],
               "learning_rate":[0.05, 0.1]},
              deployable=True),
    ModelSpec("svr",
              lambda: SVR(),
              {"C":       [0.1, 1, 10],
               "epsilon": [0.01, 0.1],
               "kernel":  ["rbf", "linear"]},
              deployable=False),
]


def get_spec(name: str) -> ModelSpec:
    for sp in SPECS:
        if sp.name == name:
            return sp
    raise KeyError(f"Unknown model spec: {name!r}. Choices: {[s.name for s in SPECS]}")
