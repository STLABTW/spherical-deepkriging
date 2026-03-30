import numpy as np
import pytest

import spherical_deepkriging.basis_functions.mrts_sphere.sphere_cpp as sphere_cpp


def test_sphere_cpp_validates_inputs_and_unavailable_extensions(monkeypatch):
    monkeypatch.setattr(sphere_cpp, "CPP_EXTENSIONS_AVAILABLE", False)
    with pytest.raises(ImportError, match=r"C\+\+ extensions not available"):
        sphere_cpp.mrts_sphere(np.array([[1.0, 2.0]]), k=1)

    monkeypatch.setattr(sphere_cpp, "CPP_EXTENSIONS_AVAILABLE", True)

    with pytest.raises(ValueError, match="knot must be a 2D array"):
        sphere_cpp.mrts_sphere(np.array([1.0, 2.0]), k=1)

    with pytest.raises(ValueError, match="knot must have at least one point"):
        sphere_cpp.mrts_sphere(np.empty((0, 2)), k=1)

    with pytest.raises(ValueError, match="k must be at least 1"):
        sphere_cpp.mrts_sphere(np.array([[1.0, 2.0]]), k=0)

    with pytest.raises(ValueError, match="cannot be greater than the number of knots"):
        sphere_cpp.mrts_sphere(np.array([[1.0, 2.0]]), k=2)

    with pytest.raises(ValueError, match="X must be a 2D array"):
        sphere_cpp.mrts_sphere(
            np.array([[1.0, 2.0], [3.0, 4.0]]), k=1, X=np.array([1.0, 2.0])
        )


def test_sphere_cpp_builds_cache_and_reuses_precomputed_values(monkeypatch):
    sphere_cpp.clear_cache()
    monkeypatch.setattr(sphere_cpp, "CPP_EXTENSIONS_AVAILABLE", True)

    calls = {"cpp_K": 0, "cpp_Kmatrix": 0}

    def fake_cpp_K(lat, lon, n):
        calls["cpp_K"] += 1
        return np.array(
            [[2.0, 1.0, 0.5], [1.0, 2.0, 0.5], [0.5, 0.5, 2.0]], dtype=float
        )

    def fake_get_eigen_top_k(matrix, k):
        return np.array([0.0, 2.0, 1.0], dtype=float), np.eye(3, dtype=float)

    def fake_cpp_Kmatrix(k, knot, X, Konev, eiKvecmval, n, N):
        calls["cpp_Kmatrix"] += 1
        assert np.allclose(Konev, np.array([1.16666667, 1.16666667, 1.0]))
        assert eiKvecmval.shape == (3, 2)
        assert np.allclose(eiKvecmval[:, 0], 0.0)
        return np.full((N, k), 7.0)

    monkeypatch.setattr(sphere_cpp, "cpp_K", fake_cpp_K, raising=False)
    monkeypatch.setattr(sphere_cpp, "getEigenTopK", fake_get_eigen_top_k, raising=False)
    monkeypatch.setattr(sphere_cpp, "cpp_Kmatrix", fake_cpp_Kmatrix, raising=False)

    knot = np.array([[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]], dtype=float)
    out1 = sphere_cpp.mrts_sphere(knot, k=3)
    assert out1["mrts"].shape == (3, 3)
    assert len(sphere_cpp._cache) == 1

    monkeypatch.setattr(
        sphere_cpp,
        "cpp_K",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cache miss")),
        raising=False,
    )
    out2 = sphere_cpp.mrts_sphere(knot, k=3, X=np.array([[13.0, 23.0]], dtype=float))
    assert out2["mrts"].shape == (1, 3)
    assert calls["cpp_K"] == 1
    assert calls["cpp_Kmatrix"] == 2


def test_sphere_cpp_raises_on_non_finite_kernel_and_can_clear_cache(monkeypatch):
    sphere_cpp.clear_cache()
    monkeypatch.setattr(sphere_cpp, "CPP_EXTENSIONS_AVAILABLE", True)
    monkeypatch.setattr(
        sphere_cpp,
        "cpp_K",
        lambda lat, lon, n: np.array([[1.0, np.nan], [np.nan, 1.0]], dtype=float),
        raising=False,
    )

    with pytest.raises(FloatingPointError, match="K contains non-finite values"):
        sphere_cpp.mrts_sphere(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float), k=2)

    sphere_cpp._cache[(b"x", 1)] = {
        "K": np.eye(1),
        "Konev": np.ones(1),
        "eiKvecmval": np.ones((1, 1)),
    }
    sphere_cpp.clear_cache()
    assert sphere_cpp._cache == {}
