"""Spherical basis functions using C++ extensions."""

from __future__ import annotations

import hashlib
from typing import Dict, Optional

import numpy as np

try:
    from .cpp_extensions import cpp_K, cpp_Kmatrix, getEigenTopK

    CPP_EXTENSIONS_AVAILABLE = True
except ImportError:
    CPP_EXTENSIONS_AVAILABLE = False


# Cache for K matrix, Konev, and eigenpairs keyed by (knot hash, k)
_cache: Dict[tuple[bytes, int], Dict[str, np.ndarray]] = {}


def _knot_hash(knot: np.ndarray) -> bytes:
    """Compute a hash of the knot array for caching purposes."""
    return hashlib.sha256(knot.tobytes()).digest()


def mrts_sphere(
    knot: np.ndarray,
    k: int,
    X: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Multi-Resolution Thin Plate Splines for spherical coordinates.

    This function computes spherical basis functions using the C++ extensions.
    It is equivalent to the R function mrts_sphere.
    """

    if not CPP_EXTENSIONS_AVAILABLE:
        raise ImportError(
            "C++ extensions not available. Please build them using 'make build-cpp'."
        )

    # Validate inputs
    if knot.ndim != 2 or knot.shape[1] != 2:
        raise ValueError(
            "knot must be a 2D array with shape (n, 2) where columns are [latitude, longitude]"
        )

    n = knot.shape[0]
    if n < 1:
        raise ValueError("knot must have at least one point")

    if k < 1:
        raise ValueError("k must be at least 1")

    if k > n:
        raise ValueError(f"k ({k}) cannot be greater than the number of knots ({n})")

    # Use knot locations for prediction if X is not provided
    if X is None:
        X = knot.copy()

    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(
            "X must be a 2D array with shape (N, 2) where columns are [latitude, longitude]"
        )

    N = X.shape[0]

    # Check cache for this knot configuration
    knot_hash_bytes = _knot_hash(knot)
    cache_key = (knot_hash_bytes, k)

    # Try to get from cache
    if cache_key in _cache:
        cache_data = _cache[cache_key]
        K = cache_data["K"]
        Konev = cache_data["Konev"]
        eiKvecmval = cache_data["eiKvecmval"]
    else:
        # Step 1: Compute K matrix using C++ function
        K = cpp_K(knot[:, 0], knot[:, 1], n)

        # Check for NaN/inf in K matrix (indicates bug in cpp_Kf/integration)
        if not np.isfinite(K).all():
            bad = np.argwhere(~np.isfinite(K))
            raise FloatingPointError(
                f"K contains non-finite values at {bad[:5].tolist()} (showing up to 5). "
                "This likely indicates a bug in cpp_Kf when handling identical points."
            )

        # Step 2: Compute Q @ K @ Q more efficiently
        row_mean = K.mean(axis=1, keepdims=True)
        col_mean = K.mean(axis=0, keepdims=True)
        grand_mean = K.mean()
        QKQ = K - row_mean - col_mean + grand_mean

        # Step 3: Compute top k eigenvalues and eigenvectors using Spectra
        top_eigvals, top_eigvecs = getEigenTopK(QKQ, k)

        # Step 4: Compute eiKvecmval: eigenvectors divided by eigenvalues
        eps = 1e-12
        eiKvecmval = top_eigvecs[:, : (k - 1)] / np.where(
            np.abs(top_eigvals[: (k - 1)]) > eps,
            top_eigvals[: (k - 1)],
            np.inf,
        )
        eiKvecmval = np.nan_to_num(
            eiKvecmval, nan=0.0, posinf=0.0, neginf=0.0
        )

        # Step 5: Compute Konev = K @ onev where onev = ones(n) / n
        onev = np.ones(n) / n
        Konev = K @ onev

        _cache[cache_key] = {
            "K": K,
            "Konev": Konev,
            "eiKvecmval": eiKvecmval,
        }

    # Step 6: Compute basis functions at prediction locations
    dm_train = cpp_Kmatrix(k, knot, X, Konev, eiKvecmval, n, N)

    return {"mrts": dm_train}


def clear_cache() -> None:
    """Clear the cache for K matrix, Konev, and eigenpairs."""
    global _cache
    _cache.clear()

