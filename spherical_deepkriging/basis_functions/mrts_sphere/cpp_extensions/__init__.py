"""C++ extensions for spherical basis functions."""

__all__ = []

# Try to import spherical basis extensions
try:
    from .spherical_basis import (
        cpp_exp,
        cpp_fk,
        cpp_K,
        cpp_Kf,
        cpp_Kmatrix,
        getEigen,
        getEigenTopK,
    )

    __all__.extend(
        [
            "cpp_Kf",
            "cpp_K",
            "cpp_fk",
            "cpp_Kmatrix",
            "cpp_exp",
            "getEigen",
            "getEigenTopK",
        ]
    )
except ImportError:
    pass

if not __all__:
    import warnings

    warnings.warn(
        "C++ extensions not available. "
        "Please build the extensions using 'make build-cpp'.",
        ImportWarning,
    )

