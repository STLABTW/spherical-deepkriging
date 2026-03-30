import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from spherical_deepkriging.basis_functions.wendland.wenland import wendland_core


def visualize_2d_basis_functions(
    knots: np.ndarray, theta: float, k: int, num_basis_to_show: int = 6
) -> None:
    # Define grid points for visualization
    grid_x, grid_y = np.linspace(0, 1, 50), np.linspace(0, 1, 50)
    grid_X, grid_Y = np.meshgrid(grid_x, grid_y)
    grid_points = np.column_stack([grid_X.ravel(), grid_Y.ravel()])

    # Determine the number of rows needed
    num_columns = 3
    num_rows = -(-num_basis_to_show // num_columns)  # Ceiling division

    # Create subplots with a 3-column layout
    fig = make_subplots(
        rows=num_rows,
        cols=num_columns,
        subplot_titles=[
            f"Basis {i + 1}" for i in range(min(num_basis_to_show, len(knots)))
        ],
        horizontal_spacing=0.05,
        vertical_spacing=0.1,
    )

    # Add heatmaps for each knot
    for i in range(min(num_basis_to_show, len(knots))):
        # Calculate basis function values
        z_values = wendland_core(grid_points, knots[i], theta, k).reshape(grid_X.shape)

        # Add heatmap to the corresponding subplot
        row, col = divmod(i, num_columns)
        fig.add_trace(
            go.Heatmap(
                z=z_values,
                x=grid_x,
                y=grid_y,
                colorscale="Viridis",
                showscale=(col == 0),  # Only show colorbar on the first column
                colorbar=dict(title="Value") if col == 0 else None,
            ),
            row=row + 1,
            col=col + 1,
        )

    # Update layout
    fig.update_layout(
        title=f"Wendland Basis Functions (First {num_basis_to_show})",
        height=300 * num_rows,
        width=900,
        margin=dict(l=0, r=0, b=0, t=50),
    )
    fig.update_xaxes(title_text="X")
    fig.update_yaxes(title_text="Y")

    fig.show()


def visualize_1d_basis_functions(
    knots: np.ndarray, theta: float, k: int, num_basis_to_show: int = 3
) -> None:
    d_values = np.linspace(0, 1.5, 1000)
    fig = go.Figure()

    for knot_index in range(min(num_basis_to_show, len(knots))):
        wendland_vals = [
            wendland_core(np.array([[d, 0]]), knots[knot_index], theta=theta, k=k)[0]
            for d in d_values
        ]
        fig.add_trace(
            go.Scatter(
                x=d_values,
                y=wendland_vals,
                mode="lines",
                name=f"Basis {knot_index + 1}",
                line=dict(width=2),
            )
        )

    # Update layout
    fig.update_layout(
        title=f"1D Wendland Basis Functions (First {num_basis_to_show})",
        xaxis_title="Distance (r)",
        yaxis_title="Function Value",
        height=500,
        width=800,
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
    )
    fig.show()
