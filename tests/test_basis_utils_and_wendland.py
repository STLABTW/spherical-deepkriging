import numpy as np
import pytest

from spherical_deepkriging.basis_functions.utils import create_knots_grid
from spherical_deepkriging.basis_functions.wendland.wenland import (
    wendland,
    wendland_core,
)


def test_create_knots_grid_returns_cartesian_grid():
    grid = create_knots_grid(resolution=3, start=-1.0, end=1.0)

    expected = np.array(
        [
            [-1.0, -1.0],
            [0.0, -1.0],
            [1.0, -1.0],
            [-1.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [-1.0, 1.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )

    assert np.allclose(grid, expected)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"resolution": 0}, "resolution must be a positive integer"),
        ({"resolution": 2, "start": 1.0, "end": 1.0}, "start must be less than end"),
    ],
)
def test_create_knots_grid_validates_inputs(kwargs, message):
    with pytest.raises(ValueError, match=message):
        create_knots_grid(**kwargs)


@pytest.mark.parametrize("k", [0, 1, 2])
def test_wendland_matches_single_knot_core(k):
    points = np.array([[0.0, 0.0], [0.5, 0.0], [1.5, 0.0]])
    knot = np.array([0.0, 0.0])

    combined = wendland(points, knots=np.array([knot]), theta=1.0, k=k)
    core = wendland_core(points, knot=knot, theta=1.0, k=k)

    assert combined.shape == (3, 1)
    assert np.allclose(combined[:, 0], core)
    assert core[-1] == 0.0


@pytest.mark.parametrize(
    ("points", "knot", "theta", "k", "message"),
    [
        (np.array([0.0, 1.0]), np.array([0.0]), 1.0, 0, "Points must be a 2D array"),
        (
            np.array([[0.0], [1.0]]),
            np.array([[0.0]]),
            1.0,
            0,
            "Knot must be a 1D array",
        ),
        (
            np.array([[0.0, 0.0, 0.0]]),
            np.array([0.0, 0.0, 0.0]),
            1.0,
            0,
            "Only 1D and 2D spaces are supported",
        ),
        (
            np.array([[0.0, 0.0]]),
            np.array([0.0]),
            1.0,
            0,
            "Points and knot dimensions must match",
        ),
        (
            np.array([[0.0, 0.0]]),
            np.array([0.0, 0.0]),
            0.0,
            0,
            "Theta must be a positive value",
        ),
        (
            np.array([[0.0, 0.0]]),
            np.array([0.0, 0.0]),
            1.0,
            3,
            "Unsupported value of k",
        ),
    ],
)
def test_wendland_core_validates_inputs(points, knot, theta, k, message):
    with pytest.raises(ValueError, match=message):
        wendland_core(points, knot=knot, theta=theta, k=k)
