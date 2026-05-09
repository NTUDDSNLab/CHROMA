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
              {"n_estimators": [100, 300, 500],
               "max_depth":    [None, 10, 20],
               "min_samples_leaf": [1, 3, 5]},
              deployable=True),
    ModelSpec("gbr",
              lambda: GradientBoostingRegressor(random_state=42),
              {"n_estimators": [100, 300],
               "max_depth":    [3, 5, 7],
               "learning_rate":[0.05, 0.1, 0.2]},
              deployable=False),  # m2cgen does not support sklearn GBR
    ModelSpec("xgb",
              _xgb_factory,
              {"n_estimators": [100, 300],
               "max_depth":    [3, 6],
               "learning_rate":[0.05, 0.1]},
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
