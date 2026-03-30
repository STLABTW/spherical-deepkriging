from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

import numpy as np

try:
    import rpy2.robjects as ro
    from rpy2.robjects import default_converter, numpy2ri

    _RPY2_AVAILABLE = True
except ImportError:  # pragma: no cover
    ro = None  # type: ignore[assignment]
    default_converter = None  # type: ignore[assignment]
    numpy2ri = None  # type: ignore[assignment]
    _RPY2_AVAILABLE = False

numpy2ri_converter = None
_mrts_sphere_r = None

if _RPY2_AVAILABLE:
    conda_root = sys.prefix
    conda_bin = Path(conda_root) / "bin"
    r_bin = conda_bin / "R"

    if "R_HOME" not in os.environ:
        os.environ["R_HOME"] = (
            subprocess.check_output([str(r_bin), "RHOME"]).decode().strip()
        )
        os.environ["PATH"] = f"{conda_bin}:{os.environ.get('PATH', '')}"

    # Keep this variable for notebook compatibility (callers use localconverter()).
    numpy2ri_converter = default_converter + numpy2ri.converter

    r_script_path = Path(__file__).parent / "fn_cpp.R"
    ro.r(f'Sys.setenv(PATH="{os.environ["PATH"]}")')
    ro.r(f'source("{r_script_path.as_posix()}")')
    _mrts_sphere_r = ro.globalenv["mrts_sphere"]


try:
    # C++/pybind implementation (deep-frk style).
    from .sphere_cpp import CPP_EXTENSIONS_AVAILABLE, mrts_sphere as _mrts_sphere_cpp
except Exception:  # pragma: no cover
    CPP_EXTENSIONS_AVAILABLE = False
    _mrts_sphere_cpp = None  # type: ignore[assignment]


class _NamedValues:
    """Minimal shim to mimic rpy2's named-list behaviour.

    Existing notebooks do:
        res_dict = dict(zip(res_r.names(), res_r))

    and then access `res_dict["mrts"]`.
    """

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def names(self) -> list[str]:
        return list(self._data.keys())

    def __iter__(self) -> Iterator[Any]:
        # rpy2 named-list iteration yields values (not keys).
        return iter(self._data.values())

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def keys(self) -> Iterable[str]:
        return self._data.keys()

    def items(self):
        return self._data.items()


def mrts_sphere(
    knot: np.ndarray,
    k: int,
    X: Optional[np.ndarray] = None,
):
    """Compute spherical MRTS basis functions.

    Prefers the C++/pybind (deep-frk) version when available; otherwise falls
    back to the existing R/rpy2 implementation.
    """

    if CPP_EXTENSIONS_AVAILABLE:
        knot64 = np.asarray(knot, dtype=np.float64)
        X64 = None if X is None else np.asarray(X, dtype=np.float64)
        res = _mrts_sphere_cpp(knot=knot64, k=k, X=X64)
        return _NamedValues(res)

    # R fallback.
    if not _RPY2_AVAILABLE or _mrts_sphere_r is None:
        raise ImportError(
            "Neither C++ extensions nor rpy2/R implementation are available."
        )

    # The R version requires X; if caller omits it, use knots as prediction points.
    X_r = knot if X is None else X
    return _mrts_sphere_r(knot, k, X_r)