import pytest
from sklearn.linear_model import LinearRegression
from sklearn.ensemble    import RandomForestRegressor
from model import models


def test_specs_registered():
    names = {s.name for s in models.SPECS}
    assert names >= {"linear", "rf", "gbr", "xgb", "lgbm", "svr"}


def test_get_spec_round_trip():
    sp = models.get_spec("linear")
    assert sp.deployable is True
    inst = sp.factory()
    assert isinstance(inst, LinearRegression)


def test_svr_marked_non_deployable():
    sp = models.get_spec("svr")
    assert sp.deployable is False


def test_unknown_name_errors():
    with pytest.raises(KeyError):
        models.get_spec("doesnotexist")
