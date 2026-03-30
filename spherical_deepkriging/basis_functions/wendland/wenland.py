import numpy as np


def wendland(points: np.ndarray, knots: np.ndarray, theta: float, k: int) -> np.ndarray:
    return np.array([wendland_core(points, knot, theta, k) for knot in knots]).T


def wendland_core(
    points: np.ndarray, knot: np.ndarray, theta: float, k: int
) -> np.ndarray:
    # Validate inputs
    if points.ndim != 2:
        raise ValueError("Points must be a 2D array (n_points, n_dimensions).")
    if knot.ndim != 1:
        raise ValueError("Knot must be a 1D array representing a single point.")
    if points.shape[1] not in [1, 2]:
        raise ValueError("Only 1D and 2D spaces are supported.")
    if points.shape[1] != knot.shape[0]:
        raise ValueError("Points and knot dimensions must match.")
    if theta <= 0:
        raise ValueError("Theta must be a positive value.")
    if k not in [0, 1, 2]:
        raise ValueError("Unsupported value of k. Choose k as 0, 1, or 2.")

    # Compute normalized distances
    distances = np.linalg.norm(points - knot, axis=1) / theta

    # Wendland function based on k
    if k == 0:
        return (1 - distances) ** 2 * (distances <= 1)
    elif k == 1:
        return (1 - distances) ** 4 * (4 * distances + 1) * (distances <= 1)
    elif k == 2:
        return (
            (1 - distances) ** 6
            * (35 * distances**2 + 18 * distances + 3)
            / 3
            * (distances <= 1)
        )
