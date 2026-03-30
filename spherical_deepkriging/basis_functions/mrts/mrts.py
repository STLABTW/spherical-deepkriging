from typing import Dict, Optional, Tuple, Union

import jax.numpy as jnp
from jax.scipy.linalg import solve

from spherical_deepkriging.basis_functions.mrts.utils import (
    build_extended_matrix,
    compute_h,
    dist,
    predict_rabf,
)


def mrts0(
    knot: jnp.ndarray, k: int, x: Optional[jnp.ndarray] = None
) -> Union[Tuple[jnp.ndarray, Dict[str, jnp.ndarray], int], jnp.ndarray]:
    """Main function for MRTS with static pre-processing."""
    Xu = jnp.asarray(knot)
    ndims = Xu.shape[1]

    if k < (ndims + 1):
        raise ValueError(f"Invalid k: {k}. It must be >= {ndims + 1}")

    slice_size = k - ndims - 1
    AHA, UZ = build_extended_matrix(Xu, k=k, ndims=ndims, slice_size=slice_size)

    BBBH = solve(
        jnp.hstack([jnp.ones((Xu.shape[0], 1)), Xu]).T
        @ jnp.hstack([jnp.ones((Xu.shape[0], 1)), Xu]),
        jnp.hstack([jnp.ones((Xu.shape[0], 1)), Xu]).T,
    ) @ compute_h(dist(Xu, Xu), ndims)
    obj_attrs = {
        "S": AHA,
        "UZ": UZ,
        "Xu": Xu,
        "BBBH": BBBH,
        "ndims": ndims,
    }
    return predict_rabf(obj_attrs, x, k) if x is not None else (Xu, obj_attrs, k)
