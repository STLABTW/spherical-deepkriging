import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")

from spherical_deepkriging.basis_functions.mrts.mrts import mrts0
from spherical_deepkriging.basis_functions.mrts.utils import (
    build_extended_matrix,
    compute_h,
    dist,
    predict_rabf,
)


def test_dist_and_compute_h_cover_supported_dimensions():
    a = jnp.array([[0.0], [3.0]])
    b = jnp.array([[0.0], [4.0]])
    distances = np.array(dist(a, b))

    assert np.allclose(distances, np.array([[0.0, 4.0], [3.0, 1.0]]))

    d = jnp.array([[0.0, 2.0]], dtype=jnp.float32)
    assert np.allclose(np.array(compute_h(d, 1)), np.array([[0.0, 8.0 / 12.0]]))
    expected_2d = np.array([[0.0, (1.0 / 8.0) * 4.0 * np.log(2.0 + 1e-8)]])
    assert np.allclose(np.array(compute_h(d, 2)), expected_2d)
    assert np.allclose(np.array(compute_h(d, 3)), np.array([[0.0, -0.25]]))


def test_build_extended_matrix_and_predict_rabf_cover_both_branches():
    xu = jnp.array([[0.0], [1.0], [2.0]], dtype=jnp.float32)
    aha, uz = build_extended_matrix(xu, k=3, ndims=1, slice_size=1)

    assert aha.shape == (3, 3)
    assert uz.shape == (5, 3)
    assert np.all(np.isfinite(np.array(aha)))
    assert np.all(np.isfinite(np.array(uz)))

    _, obj, k = mrts0(xu, k=3)
    predicted = predict_rabf(
        obj, newx=jnp.array([[0.5], [1.5]], dtype=jnp.float32), k=k
    )
    assert predicted.shape == (2, 3)

    _, obj_small, k_small = mrts0(xu, k=2)
    predicted_small = predict_rabf(
        obj_small, newx=jnp.array([[0.5], [1.5]], dtype=jnp.float32), k=k_small
    )
    assert predicted_small.shape == (2, 2)
    assert np.allclose(np.array(predicted_small[:, 0]), 1.0)

    returned_obj = predict_rabf(obj, newx=None, k=k)
    assert set(returned_obj) == set(obj)
    assert np.allclose(np.array(returned_obj["Xu"]), np.array(obj["Xu"]))


def test_mrts0_returns_tuple_or_prediction_and_validates_k():
    knot = jnp.array([[0.0], [1.0], [2.0]], dtype=jnp.float32)

    xu, obj, used_k = mrts0(knot, k=3)
    assert used_k == 3
    assert xu.shape == knot.shape
    assert set(obj) == {"S", "UZ", "Xu", "BBBH", "ndims"}

    prediction = mrts0(knot, k=3, x=jnp.array([[0.25], [1.25]], dtype=jnp.float32))
    assert prediction.shape == (2, 3)

    with pytest.raises(ValueError, match="Invalid k"):
        mrts0(knot, k=1)
