import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

from spherical_deepkriging.basis_functions.mrts.mrts import mrts0

logger = logging.getLogger(__name__)


def plot_1d_basis_functions(
    basis_values: np.ndarray, r: np.ndarray, num_basis: int
) -> None:
    cols = 5
    rows = (num_basis + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
    axes = axes.ravel()

    for i in range(num_basis):
        axes[i].plot(r, basis_values[:, i], color="blue", linewidth=1)
        axes[i].set_title(f"Basis {i + 1}")
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    for j in range(num_basis, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"MRTS Basis Functions ({num_basis} bases, 1D)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_2d_basis_functions(
    basis_values: np.ndarray, grid_size: int, num_basis: int, start: float, end: float
) -> None:
    cols = 5
    rows = (num_basis + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(15, rows * 3), constrained_layout=True
    )
    axes = axes.ravel()

    for i in range(num_basis):
        basis_function = basis_values[:, i].reshape((grid_size, grid_size))
        cax = axes[i].imshow(
            basis_function,
            cmap="coolwarm",
            norm=colors.Normalize(vmin=-1, vmax=1),
            extent=[start, end, start, end],
            origin="lower",
        )
        axes[i].set_title(f"Basis {i + 1}")
        axes[i].axis("off")

    for j in range(num_basis, len(axes)):
        axes[j].axis("off")

    fig.colorbar(cax, ax=axes.ravel().tolist(), shrink=0.7)
    fig.suptitle(f"MRTS Basis Functions ({num_basis} bases, 2D grid)", fontsize=16)
    plt.show()


def plot_3d_basis_functions(
    basis_values: np.ndarray, grid_points: np.ndarray, grid_size: int, num_basis: int
) -> None:
    cols = 5
    rows = (num_basis + cols - 1) // cols
    fig = plt.figure(figsize=(15, rows * 3))

    x, y, z = grid_points[:, 0], grid_points[:, 1], grid_points[:, 2]
    x = x.reshape(grid_size, grid_size, grid_size)
    y = y.reshape(grid_size, grid_size, grid_size)
    z = z.reshape(grid_size, grid_size, grid_size)

    for i in range(num_basis):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        basis_function = (
            basis_values[:, i].reshape((grid_size, grid_size, grid_size)).mean(axis=2)
        )
        surf = ax.plot_surface(
            x[:, :, 0],
            y[:, :, 0],
            basis_function,
            cmap="coolwarm",
            norm=colors.Normalize(vmin=-1, vmax=1),
            edgecolor="none",
        )
        ax.set_title(f"Basis {i + 1}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    fig.colorbar(surf, ax=ax, shrink=0.5, orientation="horizontal", pad=0.1)
    fig.suptitle(f"MRTS Basis Functions ({num_basis} bases, 3D grid)", fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_mrts_basis_functions(
    num_basis: int,
    resolution: int = 30,
    ndims: int = 1,
    start: float = 0.0,
    end: float = 1.0,
) -> None:
    if num_basis < ndims + 1:
        raise ValueError(f"Invalid k: {num_basis}. It must be >= {ndims + 1}")

    if ndims == 1:
        r = np.linspace(start, end, resolution).reshape(-1, 1)
        basis_values = mrts0(r, k=num_basis, x=r)
        plot_1d_basis_functions(basis_values, r, num_basis)

    elif ndims == 2:
        grid_size = resolution
        x_grid = np.linspace(start, end, grid_size)
        y_grid = np.linspace(start, end, grid_size)
        x, y = np.meshgrid(x_grid, y_grid)
        grid_points = np.column_stack([x.ravel(), y.ravel()])

        x_knots = np.linspace(start, end, int(np.sqrt(num_basis)))
        y_knots = np.linspace(start, end, int(np.sqrt(num_basis)))
        xk, yk = np.meshgrid(x_knots, y_knots)
        knot = np.column_stack([xk.ravel(), yk.ravel()])

        basis_values = mrts0(knot, k=num_basis, x=grid_points)
        plot_2d_basis_functions(basis_values, grid_size, num_basis, start, end)

    elif ndims == 3:
        grid_size = resolution

        # Handle non-cubic number of basis functions
        approx_dim = int(np.round(np.cbrt(num_basis)))
        actual_num_basis = approx_dim**3

        if actual_num_basis != num_basis:
            logger.warning(
                "Adjusting num_basis from %s to nearest cube %s.",
                num_basis,
                actual_num_basis,
            )
            input_num_basis = actual_num_basis
        else:
            input_num_basis = num_basis

        # Create grid points
        x_grid = np.linspace(start, end, grid_size)
        y_grid = np.linspace(start, end, grid_size)
        z_grid = np.linspace(start, end, grid_size)
        x, y, z = np.meshgrid(x_grid, y_grid, z_grid)
        grid_points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])

        # Generate knots
        x_knots = np.linspace(start, end, approx_dim)
        y_knots = np.linspace(start, end, approx_dim)
        z_knots = np.linspace(start, end, approx_dim)
        xk, yk, zk = np.meshgrid(x_knots, y_knots, z_knots)
        knot = np.column_stack([xk.ravel(), yk.ravel(), zk.ravel()])

        # Compute basis functions
        basis_values = mrts0(knot, k=input_num_basis, x=grid_points)
        plot_3d_basis_functions(basis_values, grid_points, grid_size, num_basis)

    else:
        raise ValueError(
            f"Unsupported ndims={ndims}. Only ndims=1, 2, or 3 are supported."
        )
