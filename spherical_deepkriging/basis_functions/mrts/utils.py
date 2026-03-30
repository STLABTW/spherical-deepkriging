from functools import partial
from typing import Dict, Optional, Tuple

import jax.numpy as jnp
from jax import jit, lax


@jit
def dist(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(
        jnp.sum((A[:, jnp.newaxis, :] - B[jnp.newaxis, :, :]) ** 2, axis=-1)
    )


def compute_h(d: jnp.ndarray, ndims: int) -> jnp.ndarray:
    """Computes the H matrix based on the dimension."""

    def case_ndims_1(_: None) -> jnp.ndarray:
        return (1 / 12) * d**3

    def case_ndims_2(_: None) -> jnp.ndarray:
        return (1 / 8) * d**2 * jnp.log(d + 1e-8)

    def case_ndims_3(_: None) -> jnp.ndarray:
        return (-1 / 8) * d

    cases = [case_ndims_1, case_ndims_2, case_ndims_3]
    return lax.switch(ndims - 1, cases, None)


@partial(jit, static_argnames=["ndims", "slice_size"])
def build_extended_matrix(
    Xu: jnp.ndarray, k: int, ndims: int, slice_size: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Builds the S and UZ matrices."""
    n = Xu.shape[0]
    B = jnp.hstack([jnp.ones((n, 1)), Xu])
    BBB = jnp.linalg.solve(B.T @ B, B.T)
    A = jnp.eye(n) - B @ BBB

    d = jnp.sqrt(jnp.sum((Xu[:, None, :] - Xu[None, :, :]) ** 2, axis=-1))
    H = compute_h(d, ndims)
    AH = H - (H @ B) @ BBB
    AHA = AH - BBB.T @ (B.T @ AH)

    gamma0, _, _ = jnp.linalg.svd(AHA, full_matrices=False)

    gamma0 = lax.dynamic_slice(
        gamma0, start_indices=(0, 0), slice_sizes=(gamma0.shape[0], slice_size)
    )

    trueBS = AHA @ gamma0
    rho = jnp.sqrt(jnp.sum(trueBS**2, axis=0))
    gammas = A @ gamma0 / rho * jnp.sqrt(n)

    extension_dim = ndims + 1

    UZ = jnp.vstack(
        [
            jnp.hstack([gammas, jnp.zeros((gammas.shape[0], extension_dim))]),
            jnp.zeros((extension_dim, gammas.shape[1] + extension_dim)),
        ]
    )
    pad_size = ndims + 1

    def compute_valid_indices() -> jnp.ndarray:
        max_valid_index = min(pad_size, Xu.shape[1])
        return jnp.arange(max_valid_index)

    valid_indices = lax.stop_gradient(compute_valid_indices())

    updates = 1 / jnp.std(Xu[:, valid_indices], axis=0) / jnp.sqrt((n - 1) / n)
    UZ = UZ.at[n + valid_indices, k - pad_size - 1 + valid_indices].set(updates)

    return AHA, UZ


@partial(jit, static_argnames=["k"])
def predict_rabf(
    obj: Dict[str, jnp.ndarray],
    newx: Optional[jnp.ndarray] = None,
    k: Optional[int] = None,
) -> jnp.ndarray:
    """Predicts new values based on the provided model object."""
    if newx is None:
        return obj

    x0 = newx
    d = dist(x0, obj["Xu"])
    ndims = x0.shape[1] if x0.ndim > 1 else 1
    H = compute_h(d, ndims)

    kstar = k - ndims - 1

    def true_branch(_: None) -> jnp.ndarray:
        slice_UZ = lax.dynamic_slice(
            obj["UZ"], start_indices=(0, 0), slice_sizes=(obj["Xu"].shape[0], kstar)
        )
        X1 = H @ slice_UZ

        B = jnp.hstack(
            [jnp.ones((x0.shape[0], 1)), x0 if x0.ndim > 1 else x0[:, jnp.newaxis]]
        )
        BBBH_UZ = lax.dynamic_slice(
            obj["UZ"], start_indices=(0, 0), slice_sizes=(obj["Xu"].shape[0], kstar)
        )
        X1 -= B @ obj["BBBH"] @ BBBH_UZ
        X1 /= jnp.sqrt(obj["Xu"].shape[0])
        return X1

    def false_branch(_: None) -> jnp.ndarray:
        return jnp.zeros((x0.shape[0], kstar))

    X1 = lax.cond(kstar > 0, true_branch, false_branch, operand=None)

    X2 = jnp.hstack(
        [jnp.ones((x0.shape[0], 1)), x0 if x0.ndim > 1 else x0[:, jnp.newaxis]]
    )

    def concat_branch(_: None) -> jnp.ndarray:
        return jnp.hstack([X2, X1])

    def no_concat_branch(_: None) -> jnp.ndarray:
        padding = jnp.zeros((x0.shape[0], X1.shape[1]))
        return jnp.hstack([X2, padding])

    return lax.cond(kstar > 0, concat_branch, no_concat_branch, operand=None)
