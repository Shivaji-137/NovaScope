from __future__ import annotations

import pandas as pd
import matplotlib
import pytest

matplotlib.use("Agg", force=True)

pytest.importorskip("plotly.express")

from plotter.plotting import PlotRequest, build_plot


def test_build_plot_pie_returns_plotly_backend():
    df = pd.DataFrame({"category": ["a", "b"], "value": [10, 20]})
    request = PlotRequest(
        data=df,
        x="category",
        y=["value"],
        color=None,
        plot_type="pie",
    )
    result = build_plot(request)
    assert result.backend == "plotly"
    assert result.figure is not None


def test_build_plot_kde_returns_matplotlib_backend():
    df = pd.DataFrame({"measure": [1, 2, 3, 4]})
    request = PlotRequest(
        data=df,
        x=None,
        y=["measure"],
        color=None,
        plot_type="kde",
    )
    result = build_plot(request)
    assert result.backend == "matplotlib"
    assert result.figure is not None


def test_build_plot_scatter_matrix_returns_plotly_backend():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "z": [7, 8, 9]})
    request = PlotRequest(
        data=df,
        x=None,
        y=["x", "y", "z"],
        color=None,
        plot_type="scatter_matrix",
    )
    result = build_plot(request)
    assert result.backend == "plotly"
    assert result.figure is not None


def test_build_plot_pairplot_returns_matplotlib_backend():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]})
    request = PlotRequest(
        data=df,
        x=None,
        y=["a", "b"],
        color=None,
        plot_type="pairplot",
    )
    result = build_plot(request)
    assert result.backend == "matplotlib"
    assert result.figure is not None


def test_build_plot_3d_returns_plotly_backend():
    df = pd.DataFrame({"time": [1, 2, 3, 4], "signal": [0.1, 0.5, 0.2, 0.9], "depth": [5, 6, 7, 8]})
    request = PlotRequest(
        data=df,
        x="time",
        y=["signal", "depth"],
        color=None,
        plot_type="3dplot",
    )
    result = build_plot(request)
    assert result.backend == "plotly"
    assert result.figure is not None
