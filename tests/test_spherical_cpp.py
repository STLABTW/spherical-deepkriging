"""Unit tests for spherical basis functions (mrts_sphere)."""

import numpy as np
import pytest

# Try to import spherical basis functions and check if C++ extensions are available
SPHERICAL_AVAILABLE = False
try:
    from spherical_deepkriging.basis_functions.mrts_sphere.sphere_cpp import (
        CPP_EXTENSIONS_AVAILABLE,
        mrts_sphere,
    )

    SPHERICAL_AVAILABLE = CPP_EXTENSIONS_AVAILABLE
except ImportError:
    SPHERICAL_AVAILABLE = False

if not SPHERICAL_AVAILABLE:
    pytestmark = pytest.mark.skip(
        "Spherical basis functions not available. Build C++ extensions with: make build-cpp"
    )


@pytest.mark.skipif(
    not SPHERICAL_AVAILABLE, reason="Spherical basis functions not available"
)
class TestMRTSSphere:
    """Test suite for mrts_sphere function."""

    def test_basic_no_x(self):
        """Test basic mrts_sphere with no X (uses knots as prediction points)."""
        knot = np.array([[45.0, -120.0], [46.0, -121.0], [47.0, -122.0]])
        k = 3

        result = mrts_sphere(knot, k=k)

        assert "mrts" in result
        basis = result["mrts"]
        assert basis.shape == (3, 3)  # (n, k)
        # First basis function should be constant at sqrt(1/n)
        assert np.allclose(basis[:, 0], np.sqrt(1.0 / len(knot)))

    def test_basic_with_x(self):
        """Test basic mrts_sphere with prediction points."""
        knot = np.array([[45.0, -120.0], [46.0, -121.0], [47.0, -122.0]])
        X = np.array([[45.5, -120.5], [46.5, -121.5]])
        k = 3

        result = mrts_sphere(knot, k=k, X=X)

        assert "mrts" in result
        basis = result["mrts"]
        assert basis.shape == (2, 3)  # (N, k)
        # First basis function should be constant at sqrt(1/n)
        assert np.allclose(basis[:, 0], np.sqrt(1.0 / len(knot)))

    def test_different_k_values(self):
        """Test mrts_sphere with different k values."""
        knot = np.array(
            [
                [45.0, -120.0],
                [46.0, -121.0],
                [47.0, -122.0],
                [48.0, -123.0],
                [49.0, -124.0],
            ]
        )

        for k in [1, 2, 3, 5]:
            result = mrts_sphere(knot, k=k)
            basis = result["mrts"]
            assert basis.shape == (len(knot), k)
            # First basis function should be constant
            assert np.allclose(basis[:, 0], np.sqrt(1.0 / len(knot)))

    def test_larger_grid(self):
        """Test mrts_sphere with a larger grid of knots."""
        # Create a 4x4 grid of knots
        lat_min, lat_max = 30.0, 50.0
        lon_min, lon_max = -120.0, -100.0
        lat_knots = np.linspace(lat_min, lat_max, 4)
        lon_knots = np.linspace(lon_min, lon_max, 4)
        lat_grid, lon_grid = np.meshgrid(lat_knots, lon_knots)
        knot = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])

        k = 10
        result = mrts_sphere(knot, k=k)

        basis = result["mrts"]
        assert basis.shape == (16, k)
        # First basis function should be constant
        assert np.allclose(basis[:, 0], np.sqrt(1.0 / len(knot)))

    def test_prediction_grid(self):
        """Test mrts_sphere with a prediction grid."""
        knot = np.array([[45.0, -120.0], [46.0, -121.0], [47.0, -122.0]])
        # Create prediction grid
        lat_pred = np.linspace(45.0, 47.0, 5)
        lon_pred = np.linspace(-122.0, -120.0, 5)
        lat_grid, lon_grid = np.meshgrid(lat_pred, lon_pred)
        X = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])

        k = 3
        result = mrts_sphere(knot, k=k, X=X)

        basis = result["mrts"]
        assert basis.shape == (len(X), k)

    def test_invalid_knot_shape(self):
        """Test that invalid knot shapes raise ValueError."""
        # Wrong number of dimensions
        with pytest.raises(ValueError, match="knot must be a 2D array"):
            mrts_sphere(np.array([45.0, -120.0]), k=3)

        # Wrong number of columns
        with pytest.raises(ValueError, match="knot must be a 2D array"):
            mrts_sphere(np.array([[45.0]]), k=3)

    def test_invalid_x_shape(self):
        """Test that invalid X shapes raise ValueError."""
        knot = np.array([[45.0, -120.0], [46.0, -121.0]])
        # Wrong number of dimensions
        with pytest.raises(ValueError, match="X must be a 2D array"):
            mrts_sphere(knot, k=2, X=np.array([45.0, -120.0]))

        # Wrong number of columns
        with pytest.raises(ValueError, match="X must be a 2D array"):
            mrts_sphere(knot, k=2, X=np.array([[45.0]]))

    def test_k_too_large(self):
        """Test that k > n raises ValueError."""
        knot = np.array([[45.0, -120.0], [46.0, -121.0]])
        with pytest.raises(
            ValueError, match="k.*cannot be greater than the number of knots"
        ):
            mrts_sphere(knot, k=3)

    def test_k_too_small(self):
        """Test that k < 1 raises ValueError."""
        knot = np.array([[45.0, -120.0], [46.0, -121.0]])
        with pytest.raises(ValueError, match="k must be at least 1"):
            mrts_sphere(knot, k=0)

    def test_empty_knot(self):
        """Test that empty knot array raises ValueError."""
        with pytest.raises(ValueError, match="knot must have at least one point"):
            mrts_sphere(np.array([]).reshape(0, 2), k=1)

    def test_first_basis_constant(self):
        """Test that the first basis function is constant."""
        knot = np.array(
            [
                [45.0, -120.0],
                [46.0, -121.0],
                [47.0, -122.0],
                [48.0, -123.0],
            ]
        )
        X = np.random.uniform([44.0, -121.0], [49.0, -119.0], size=(20, 2))

        result = mrts_sphere(knot, k=4, X=X)
        basis = result["mrts"]

        # First basis function should be constant across all prediction points
        first_basis = basis[:, 0]
        assert np.allclose(first_basis, first_basis[0])
        assert np.isclose(first_basis[0], np.sqrt(1.0 / len(knot)))

    def test_basis_finite(self):
        """Test that all basis function values are finite."""
        knot = np.array([[45.0, -120.0], [46.0, -121.0], [47.0, -122.0]])
        X = np.array([[45.5, -120.5], [46.5, -121.5], [47.5, -122.5]])

        result = mrts_sphere(knot, k=3, X=X)
        basis = result["mrts"]

        # All values should be finite (no NaN or Inf)
        assert np.all(np.isfinite(basis))

    def test_import_error_when_extensions_unavailable(self):
        """Test that ImportError is raised when C++ extensions are unavailable."""
        from unittest.mock import patch

        import spherical_deepkriging.basis_functions.mrts_sphere.sphere_cpp as spherical_module

        # Patch the CPP_EXTENSIONS_AVAILABLE flag directly
        with patch.object(spherical_module, "CPP_EXTENSIONS_AVAILABLE", False):
            knot = np.array([[45.0, -120.0], [46.0, -121.0]])
            # Escape the + in "C++" for regex
            with pytest.raises(ImportError, match="C\\+\\+ extensions not available"):
                spherical_module.mrts_sphere(knot, k=2)

    def test_floating_point_error_for_non_finite_k(self):
        """Test that FloatingPointError is raised when K matrix contains non-finite values."""
        from unittest.mock import MagicMock, patch

        import spherical_deepkriging.basis_functions.mrts_sphere.sphere_cpp as spherical_module

        knot = np.array([[45.0, -120.0], [46.0, -121.0]])
        # Create a K matrix with NaN
        K_with_nan = np.array([[1.0, np.nan], [np.nan, 1.0]])

        # Need to patch CPP_EXTENSIONS_AVAILABLE and provide mock functions
        mock_getEigenTopK = MagicMock(
            return_value=(np.array([1.0, 0.5]), np.ones((2, 2)))
        )
        mock_cpp_Kmatrix = MagicMock(return_value=np.ones((2, 2)))

        with (
            patch.object(spherical_module, "CPP_EXTENSIONS_AVAILABLE", True),
            patch.object(
                spherical_module,
                "cpp_K",
                lambda lat, lon, n: K_with_nan,
                create=True,
            ),
            patch.object(
                spherical_module,
                "getEigenTopK",
                mock_getEigenTopK,
                create=True,
            ),
            patch.object(
                spherical_module,
                "cpp_Kmatrix",
                mock_cpp_Kmatrix,
                create=True,
            ),
        ):
            with pytest.raises(
                FloatingPointError, match="K contains non-finite values"
            ):
                spherical_module.mrts_sphere(knot, k=2)
