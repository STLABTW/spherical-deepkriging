import numpy as np


def create_knots_grid(
    resolution: int, start: float = 0.0, end: float = 1.0
) -> np.ndarray:
    if resolution <= 0:
        raise ValueError("resolution must be a positive integer.")
    if start >= end:
        raise ValueError("start must be less than end.")

    # Generate linearly spaced grid points
    grid = np.linspace(start, end, resolution)
    grid_X, grid_Y = np.meshgrid(grid, grid)

    # Combine into [x, y] coordinates
    return np.column_stack((grid_X.ravel(), grid_Y.ravel()))
