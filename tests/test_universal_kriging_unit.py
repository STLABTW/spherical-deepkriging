import importlib
import sys
import types

import numpy as np
import pandas as pd
import pytest


class FakeGPModel:
    instances = []
    next_cov_pars = []
    next_predictions = []
    next_coefs = []
    fit_error = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fit_calls = []
        FakeGPModel.instances.append(self)

    def fit(self, y, X, params):
        if FakeGPModel.fit_error is not None:
            raise FakeGPModel.fit_error
        self.fit_calls.append({"y": y, "X": X, "params": params})

    def get_cov_pars(self):
        return FakeGPModel.next_cov_pars.pop(0)

    def predict(self, X_pred, gp_coords_pred, predict_var):
        self.predict_call = {
            "X_pred": X_pred,
            "gp_coords_pred": gp_coords_pred,
            "predict_var": predict_var,
        }
        return {"mu": FakeGPModel.next_predictions.pop(0)}

    def get_coef(self):
        coef = FakeGPModel.next_coefs.pop(0)
        if isinstance(coef, Exception):
            raise coef
        return coef


def load_universal_module(monkeypatch):
    FakeGPModel.instances = []
    FakeGPModel.next_cov_pars = []
    FakeGPModel.next_predictions = []
    FakeGPModel.next_coefs = []
    FakeGPModel.fit_error = None

    fake_gpboost = types.ModuleType("gpboost")
    fake_gpboost.GPModel = FakeGPModel
    monkeypatch.setitem(sys.modules, "gpboost", fake_gpboost)
    sys.modules.pop("spherical_deepkriging.models.universal_kriging", None)

    module = importlib.import_module("spherical_deepkriging.models.universal_kriging")
    return importlib.reload(module)


def make_cov_pars(nu=0.5, rho=1.5, sigma2=2.0, nugget=0.1, include_nu=True):
    data = {
        "GP_range": [rho],
        "GP_var": [sigma2],
        "Error_term": [nugget],
    }
    if include_nu:
        data["Matern_nu"] = [nu]
    return pd.DataFrame(data)


def test_static_helpers_cover_distance_and_parameter_extraction(monkeypatch):
    module = load_universal_module(monkeypatch)
    uk = module.UniversalKriging

    coords = np.array([[0.0, 0.0], [0.0, 90.0]], dtype=np.float32)
    coords_rad = uk.coords_to_radians(coords)
    distances = uk.compute_spherical_distance_matrix(coords_rad)

    assert coords_rad.dtype == np.float32
    assert distances.shape == (2, 2)
    assert np.isclose(distances[0, 1], np.pi / 2, atol=1e-5)
    assert uk.extract_gp_params(
        types.SimpleNamespace(get_cov_pars=lambda: make_cov_pars(include_nu=False))
    ) == (0.5, 1.5, 2.0, 0.1)


def test_fit_with_fixed_covariance_and_covariates(monkeypatch):
    module = load_universal_module(monkeypatch)
    FakeGPModel.next_cov_pars = [make_cov_pars(nu=0.5)]
    FakeGPModel.next_coefs = [pd.Series([1.0, -2.0])]
    model = module.UniversalKriging(num_neighbors=4, cov_function="exponential")

    coords = np.array([[10.0, 20.0], [11.0, 21.0]], dtype=np.float32)
    phi = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([[2.0], [4.0]], dtype=np.float32)

    fitted = model.fit(coords, phi, y, center_y=True)

    assert fitted is model
    assert model.has_covariates is True
    assert model.nu_was_refitted is False
    assert np.isclose(model.y_mean, 3.0)
    assert model.params["cov_function"] == "exponential"
    assert model.params["beta"].shape == (2, 1)
    assert FakeGPModel.instances[0].kwargs["cov_function"] == "matern"
    assert FakeGPModel.instances[0].fit_calls[0]["X"].dtype == np.float32


def test_fit_uses_zero_beta_when_get_coef_fails(monkeypatch):
    module = load_universal_module(monkeypatch)
    FakeGPModel.next_cov_pars = [make_cov_pars()]
    FakeGPModel.next_coefs = [RuntimeError("coef failure")]
    model = module.UniversalKriging(num_neighbors=2, cov_function="gaussian")

    coords = np.array([[10.0, 20.0], [11.0, 21.0]], dtype=np.float32)
    phi = np.array([[1.0], [3.0]], dtype=np.float32)
    y = np.array([[2.0], [4.0]], dtype=np.float32)

    model.fit(coords, phi, y)

    assert np.array_equal(model.params["beta"], np.zeros((1, 1)))
    assert FakeGPModel.instances[0].kwargs["cov_function"] == "gaussian"
    assert "cov_fct_shape" not in FakeGPModel.instances[0].kwargs


def test_fit_wraps_fixed_covariance_errors(monkeypatch):
    module = load_universal_module(monkeypatch)
    FakeGPModel.fit_error = ValueError("boom")
    model = module.UniversalKriging(num_neighbors=2, cov_function="gaussian")

    coords = np.array([[10.0, 20.0], [11.0, 21.0]], dtype=np.float32)
    y = np.array([[1.0], [2.0]], dtype=np.float32)

    with pytest.raises(
        RuntimeError, match="Model fitting failed with gaussian covariance: boom"
    ):
        model.fit(coords, None, y)


def test_fit_refits_when_estimated_nu_is_too_large(monkeypatch):
    module = load_universal_module(monkeypatch)
    FakeGPModel.next_cov_pars = [
        make_cov_pars(nu=0.9),
        make_cov_pars(nu=0.5, rho=3.0),
    ]
    FakeGPModel.next_coefs = [np.array([0.25])]

    monkeypatch.setattr(
        module.UniversalKriging,
        "_get_gpboost_cov_params",
        lambda self: ("matern_estimate_shape", None, True),
    )

    model = module.UniversalKriging(num_neighbors=3, cov_function="matern_auto")
    coords = np.array([[10.0, 20.0], [11.0, 21.0]], dtype=np.float32)
    y = np.array([[1.0], [2.0]], dtype=np.float32)

    model.fit(coords, None, y)

    assert model.nu_was_refitted is True
    assert model.params["was_refitted"] is True
    assert model.params["nu_initial_estimate"] == 0.9
    assert model.params["rho_rad"] == 3.0
    assert len(FakeGPModel.instances) == 2


def test_fit_keeps_estimated_model_when_nu_is_valid(monkeypatch):
    module = load_universal_module(monkeypatch)
    FakeGPModel.next_cov_pars = [make_cov_pars(nu=0.4, rho=2.2)]
    FakeGPModel.next_coefs = [np.array([1.0])]

    monkeypatch.setattr(
        module.UniversalKriging,
        "_get_gpboost_cov_params",
        lambda self: ("matern_estimate_shape", None, True),
    )

    model = module.UniversalKriging(num_neighbors=3, cov_function="matern_auto")
    coords = np.array([[10.0, 20.0], [11.0, 21.0]], dtype=np.float32)
    phi = np.array([[1.0], [2.0]], dtype=np.float32)
    y = np.array([[1.0], [2.0]], dtype=np.float32)

    model.fit(coords, phi, y, center_y=False)

    assert model.nu_was_refitted is False
    assert model.params["nu"] == 0.4
    assert model.y_mean == 0.0
    assert len(FakeGPModel.instances) == 1


def test_predict_get_coef_decompose_and_cleanup(monkeypatch):
    module = load_universal_module(monkeypatch)
    FakeGPModel.next_cov_pars = [make_cov_pars()]
    FakeGPModel.next_coefs = [
        np.array([2.0, 3.0]),
        np.array([2.0, 3.0]),
        np.array([2.0, 3.0]),
        np.array([2.0, 3.0, 4.0]),
        np.array([1.0]),
    ]
    FakeGPModel.next_predictions = [
        np.array([10.0, 20.0], dtype=np.float32),
        np.array([10.0, 20.0], dtype=np.float32),
        np.array([10.0, 20.0], dtype=np.float32),
        np.array([5.0, 6.0], dtype=np.float32),
        np.array([7.0, 8.0], dtype=np.float32),
    ]

    model = module.UniversalKriging(num_neighbors=2, cov_function="gaussian")
    coords = np.array([[10.0, 20.0], [11.0, 21.0]], dtype=np.float32)
    phi = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    y = np.array([[2.0], [4.0]], dtype=np.float32)
    model.fit(coords, phi, y)

    pred_centered = model.predict(coords, phi_new=phi, return_centered=True)
    pred_uncentered = model.predict(coords, phi_new=phi, return_centered=False)
    coef = model.get_coef()
    total, fixed, random = model.decompose_prediction(coords, phi)

    assert np.array_equal(pred_centered, np.array([10.0, 20.0], dtype=np.float32))
    assert np.array_equal(pred_uncentered, np.array([13.0, 23.0], dtype=np.float32))
    assert np.array_equal(coef, np.array([2.0, 3.0]))
    assert np.array_equal(fixed, np.array([2.0, 3.0]))
    assert np.array_equal(random, total - fixed)

    total_i, fixed_i, random_i = model.decompose_prediction(coords, phi)
    assert np.array_equal(fixed_i, np.array([6.0, 7.0]))
    assert np.array_equal(random_i, total_i - fixed_i)

    with pytest.raises(ValueError, match="Coefficient shape mismatch"):
        model.decompose_prediction(coords, phi)

    model.cleanup()
    assert model.gp_model is None
    assert model.params is None


