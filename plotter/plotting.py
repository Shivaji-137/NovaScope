from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

PLOT_GROUPS: Dict[str, Tuple[str, ...]] = {
    "Trend & Temporal": ("line", "area"),
    "Comparisons": ("scatter", "bar", "violin", "pie"),
    "Distributions": ("histogram", "kde", "pdf", "ecdf"),
    "Correlation & Density": ("heatmap", "density_contour", "density_heatmap", "regression"),
    "Multivariate": ("scatter_matrix", "pairplot"),
    "3D plot": ("3dplot",),
}

PLOT_TYPES: Tuple[str, ...] = tuple(
    plot_type for group in PLOT_GROUPS.values() for plot_type in group
)


@dataclass
class PlotRequest:
    data: pd.DataFrame
    x: Optional[str]
    y: List[str]
    color: Optional[str]
    plot_type: str
    use_log_y: bool = False
    aggregate: Optional[str] = None  # e.g., "sum", "mean"

    def validate(self) -> None:
        missing = [col for col in self.y if col not in self.data.columns]
        if missing:
            raise ValueError(f"Missing columns in data: {missing}")
        if self.x and self.x not in self.data.columns:
            raise ValueError(f"X column '{self.x}' not found in data.")
        if self.color and self.color not in self.data.columns:
            raise ValueError(f"Color column '{self.color}' not found in data.")
        if self.plot_type not in PLOT_TYPES:
            raise ValueError(f"Plot type '{self.plot_type}' is not supported.")
        if self.plot_type == "regression" and not self.x:
            raise ValueError("Regression plot requires an X-axis column.")
        if self.plot_type == "pie" and not self.y:
            raise ValueError("Pie chart requires at least one numeric column for values.")
        if self.plot_type in {"kde", "pdf", "ecdf"} and not self.y:
            raise ValueError("Density plots require at least one numeric column.")
        if self.plot_type in {"density_contour", "density_heatmap"}:
            if not self.x or not self.y:
                raise ValueError("Contour/heatmap density plots require X and Y columns.")
        if self.plot_type in {"scatter_matrix", "pairplot"} and len(self.y) < 2:
            raise ValueError("Scatter matrix/Pair plot require at least two columns.")
        if self.plot_type == "3dplot":
            if not self.x:
                raise ValueError("3D plot requires an X-axis column.")
            if len(self.y) < 2:
                raise ValueError("3D plot requires two numeric value columns (Y and Z).")


@dataclass
class PlotResult:
    backend: Literal["plotly", "matplotlib"]
    figure: Any


def _aggregate_if_needed(df: pd.DataFrame, request: PlotRequest) -> pd.DataFrame:
    if not request.aggregate or not request.x:
        return df
    agg_map = {col: request.aggregate for col in request.y}
    grouped = df.groupby(request.x, dropna=False).agg(agg_map).reset_index()
    return grouped


def build_plot(request: PlotRequest) -> PlotResult:
    request.validate()
    df = _aggregate_if_needed(request.data, request)

    plot_kwargs = dict(
        data_frame=df,
        x=request.x,
        y=request.y if len(request.y) > 1 else request.y[0],
        color=request.color,
        log_y=request.use_log_y,
        title=f"{request.plot_type.title()} plot",
    )

    if request.plot_type == "line":
        fig = px.line(**plot_kwargs)
    elif request.plot_type == "scatter":
        fig = px.scatter(**plot_kwargs)
    elif request.plot_type == "area":
        fig = px.area(**plot_kwargs)
    elif request.plot_type == "bar":
        fig = px.bar(**plot_kwargs, barmode="group")
    elif request.plot_type == "histogram":
        histogram_kwargs = plot_kwargs.copy()
        histogram_kwargs["nbins"] = min(50, len(df))
        fig = px.histogram(**histogram_kwargs)
    elif request.plot_type == "3dplot":
        y_axis = request.y[0]
        z_axis = request.y[1]
        scatter_kwargs = dict(
            data_frame=df,
            x=request.x,
            y=y_axis,
            z=z_axis,
            color=request.color,
            title="3D Plot",
            opacity=0.8,
        )
        if len(request.y) > 2:
            scatter_kwargs["size"] = request.y[2]
        fig = px.scatter_3d(**scatter_kwargs)
        fig.update_traces(marker=dict(size=4, line=dict(width=0)))
        fig.update_layout(
            scene=dict(
                xaxis_title=request.x,
                yaxis_title=y_axis,
                zaxis_title=z_axis,
                bgcolor="rgba(0,0,0,0)",
            ),
        )
    elif request.plot_type == "violin":
        fig = px.violin(**plot_kwargs, box=True, points="all")
    elif request.plot_type == "heatmap":
        heat_data = df[request.y]
        if heat_data.shape[1] == 1:
            heat_data = heat_data.T
        fig = px.imshow(heat_data, aspect="auto", color_continuous_scale="Viridis")
    elif request.plot_type == "scatter_matrix":
        fig = px.scatter_matrix(df, dimensions=request.y, color=request.color)
    elif request.plot_type == "density_contour":
        fig = px.density_contour(df, x=request.x, y=request.y[0], color=request.color)
    elif request.plot_type == "density_heatmap":
        fig = px.density_heatmap(
            df,
            x=request.x,
            y=request.y[0],
            color_continuous_scale="Viridis",
        )
    elif request.plot_type == "pie":
        pie_kwargs = dict(data_frame=df, values=request.y[0])
        if request.x:
            pie_kwargs["names"] = request.x
        else:
            pie_kwargs["names"] = df.index.astype(str)
        fig = px.pie(**pie_kwargs)
        fig.update_traces(textposition="inside", textinfo="percent+label")
    elif request.plot_type in {"kde", "regression", "pdf", "pairplot", "ecdf"}:
        return _build_seaborn_plot(df, request)
    else:
        raise ValueError(f"Unknown plot type: {request.plot_type}")

    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return PlotResult(backend="plotly", figure=fig)


def _build_seaborn_plot(df: pd.DataFrame, request: PlotRequest) -> PlotResult:
    sns.set_theme(style="whitegrid")
    if request.plot_type == "pairplot":
        data = df[request.y]
        grid = sns.pairplot(data=data, hue=request.color)
        grid.fig.tight_layout()
        return PlotResult(backend="matplotlib", figure=grid.fig)

    fig, ax = plt.subplots(figsize=(7, 4))

    if request.plot_type == "kde":
        if len(request.y) == 1:
            sns.kdeplot(data=df, x=request.y[0], hue=request.color, ax=ax, fill=True)
        else:
            for column in request.y:
                sns.kdeplot(data=df, x=column, ax=ax, label=column, fill=False)
        ax.set_title("Kernel Density Estimate")
    elif request.plot_type == "regression":
        sns.regplot(data=df, x=request.x, y=request.y[0], ax=ax)
        ax.set_title("Regression Plot")
    elif request.plot_type == "pdf":
        sns.histplot(data=df, x=request.y[0], stat="density", kde=True, ax=ax)
        ax.set_title("Probability Density Function")
    elif request.plot_type == "ecdf":
        sns.ecdfplot(data=df, x=request.y[0], hue=request.color, ax=ax)
        ax.set_title("Empirical CDF")
    else:
        raise ValueError(f"Unsupported seaborn plot type: {request.plot_type}")

    if request.use_log_y:
        ax.set_yscale("log")

    fig.tight_layout()
    return PlotResult(backend="matplotlib", figure=fig)
