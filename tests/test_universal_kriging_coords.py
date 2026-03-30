import numpy as np

import pytest


pytest.importorskip("gpboost")


def test_universal_kriging_coords_to_radians_dtype_and_shape():
    from spherical_deepkriging.models.universal_kriging import UniversalKriging

    coords = np.array([[30.0, 120.0], [-45.0, 10.0]], dtype=np.float32)
    out = UniversalKriging.coords_to_radians(coords)

    assert out.shape == coords.shape
    assert out.dtype == np.float32

