import numpy as np
import pytest

from spherical_deepkriging.basis_functions.mrts import visualization as mrts_vis
from spherical_deepkriging.basis_functions.wendland import visualization as w_vis


class FakePlotlyFigure:
    def __init__(self):
        self.traces = []
        self.layout_updates = []
        self.x_updates = []
        self.y_updates = []
        self.show_called = False

    def add_trace(self, trace, row=None, col=None):
        self.traces.append((trace, row, col))

    def update_layout(self, **kwargs):
        self.layout_updates.append(kwargs)

    def update_xaxes(self, **kwargs):
        self.x_updates.append(kwargs)

    def update_yaxes(self, **kwargs):
        self.y_updates.append(kwargs)

    def show(self):
        self.show_called = True


def test_mrts_plot_helpers_run_without_display(monkeypatch):
    monkeypatch.setattr(mrts_vis.plt, "show", lambda: None)
    monkeypatch.setattr(mrts_vis.plt, "tight_layout", lambda *args, **kwargs: None)

    r = np.linspace(0.0, 1.0, 10)
    basis_1d = np.column_stack([r, r**2, r**3])
    mrts_vis.plot_1d_basis_functions(basis_1d, r, num_basis=3)

    basis_2d = np.arange(16, dtype=float).reshape(4, 4)
    mrts_vis.plot_2d_basis_functions(
        basis_2d, grid_size=2, num_basis=4, start=0.0, end=1.0
    )

    grid_points = np.array(
        [[x, y, z] for x in range(2) for y in range(2) for z in range(2)],
        dtype=float,
    )
    basis_3d = np.arange(16, dtype=float).reshape(8, 2)
    mrts_vis.plot_3d_basis_functions(
        basis_3d, grid_points=grid_points, grid_size=2, num_basis=2
    )


def test_plot_mrts_basis_functions_dispatches_by_dimension(monkeypatch):
    calls = []

    monkeypatch.setattr(
        mrts_vis,
        "plot_1d_basis_functions",
        lambda basis_values, r, num_basis: calls.append(
            ("1d", basis_values.shape, r.shape, num_basis)
        ),
    )
    monkeypatch.setattr(
        mrts_vis,
        "plot_2d_basis_functions",
        lambda basis_values, grid_size, num_basis, start, end: calls.append(
            ("2d", basis_values.shape, grid_size, num_basis, start, end)
        ),
    )
    monkeypatch.setattr(
        mrts_vis,
        "plot_3d_basis_functions",
        lambda basis_values, grid_points, grid_size, num_basis: calls.append(
            ("3d", basis_values.shape, grid_points.shape, grid_size, num_basis)
        ),
    )

    def fake_mrts0(knot, k, x=None):
        rows = len(x) if x is not None else len(knot)
        return np.ones((rows, k), dtype=float)

    monkeypatch.setattr(mrts_vis, "mrts0", fake_mrts0)

    mrts_vis.plot_mrts_basis_functions(num_basis=3, resolution=4, ndims=1)
    mrts_vis.plot_mrts_basis_functions(num_basis=4, resolution=3, ndims=2)
    mrts_vis.plot_mrts_basis_functions(num_basis=10, resolution=2, ndims=3)

    assert [call[0] for call in calls] == ["1d", "2d", "3d"]

    with pytest.raises(ValueError, match="Invalid k"):
        mrts_vis.plot_mrts_basis_functions(num_basis=1, ndims=1)

    with pytest.raises(ValueError, match="Unsupported ndims"):
        mrts_vis.plot_mrts_basis_functions(num_basis=6, ndims=4)


def test_wendland_visualization_builds_expected_traces(monkeypatch):
    figures = []

    monkeypatch.setattr(
        w_vis.go,
        "Figure",
        lambda: figures.append(FakePlotlyFigure()) or figures[-1],
    )
    monkeypatch.setattr(
        w_vis,
        "make_subplots",
        lambda **kwargs: figures.append(FakePlotlyFigure()) or figures[-1],
    )

    knots_2d = np.array([[0.25, 0.25], [0.75, 0.75]])
    w_vis.visualize_2d_basis_functions(knots_2d, theta=0.5, k=1, num_basis_to_show=2)
    fig2d = figures[0]
    assert len(fig2d.traces) == 2
    assert fig2d.show_called is True

    knots_1d = np.array([[0.0, 0.0], [0.5, 0.0]])
    w_vis.visualize_1d_basis_functions(knots_1d, theta=1.0, k=0, num_basis_to_show=2)
    fig1d = figures[1]
    assert len(fig1d.traces) == 2
    assert fig1d.show_called is True
