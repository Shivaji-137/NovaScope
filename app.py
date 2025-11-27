from __future__ import annotations

import streamlit as st
import numpy as np
import pandas as pd
import plotly.io as pio
from matplotlib import cm

from pathlib import Path
from typing import Any, Dict
from io import BytesIO

from plotter import SUPPORTED_FILES, LoadedData, load_data
from plotter.plotting import PLOT_GROUPS, PlotRequest, PlotResult, build_plot

st.set_page_config(page_title="NovaScope", layout="wide")
pio.templates.default = "plotly_white"

ROW_INDEX_COLUMN = "row_index"

DS9_COLORMAPS = {
    "Grey": "gray",
    "Red": "Reds",
    "Green": "Greens",
    "Blue": "Blues",
    "Heat": "hot",
    "Cool": "cool",
    "Rainbow": "rainbow",
    "A": "viridis",
    "B": "plasma",
}

DS9_SCALE_MODES = {
    "Linear": "linear",
    "Log": "log",
    "Sqrt": "sqrt",
    "Asinh": "asinh",
    "Power": "power",
}


def _scale_image_values(image: np.ndarray, mode: str) -> np.ndarray:
    arr = np.asarray(image, dtype="float64")
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return np.zeros_like(arr, dtype=float)

    valid = arr[finite_mask]
    min_val = valid.min()
    shifted = valid - min_val
    span = shifted.max()
    normalized = shifted / span if span > 0 else np.zeros_like(shifted)

    mode = mode.lower()
    if mode == "log":
        scaled_valid = np.log10(normalized * 9 + 1)
    elif mode == "sqrt":
        scaled_valid = np.sqrt(normalized)
    elif mode == "asinh":
        scaled_valid = np.arcsinh(normalized * 10) / np.arcsinh(10)
    elif mode == "power":
        gamma = 2.0
        scaled_valid = np.power(normalized, gamma)
    else:
        scaled_valid = normalized

    scaled = np.zeros_like(arr, dtype=float)
    scaled[finite_mask] = scaled_valid
    return np.clip(scaled, 0.0, 1.0)


def _stylize_fits_image(image: np.ndarray, colormap_label: str, scale_label: str) -> np.ndarray:
    cmap_name = DS9_COLORMAPS.get(colormap_label, "gray")
    scale_mode = DS9_SCALE_MODES.get(scale_label, "linear")
    scaled = _scale_image_values(image, scale_mode)
    mapped = cm.get_cmap(cmap_name)(scaled, bytes=True)
    return np.asarray(mapped)[..., :3]

CUSTOM_CSS = """
<style>
:root {
    --brand-primary: #6366f1;
    --brand-secondary: #14b8a6;
    --brand-muted: #94a3b8;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top, #0f172a, #020617 55%);
}

.hero-card {
    padding: 2rem 2.4rem;
    border-radius: 1.5rem;
    background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(20,184,166,0.2));
    border: 1px solid rgba(148,163,184,0.3);
    margin-bottom: 1.5rem;
    color: #e2e8f0;
}

.hero-card h1 {
    margin-bottom: 0.3rem;
    font-size: 2.3rem;
}

.hero-card .eyebrow {
    text-transform: uppercase;
    letter-spacing: 0.2em;
    color: var(--brand-muted);
    font-size: 0.85rem;
}

.hero-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    margin-top: 0.8rem;
}

.hero-badges span {
    background: rgba(2,6,23,0.4);
    border-radius: 999px;
    padding: 0.35rem 0.9rem;
    font-size: 0.9rem;
    border: 1px solid rgba(148,163,184,0.35);
}

.glass-card {
    background: rgba(15,23,42,0.6);
    border-radius: 1rem;
    padding: 1.2rem;
    border: 1px solid rgba(148,163,184,0.2);
}

.glass-card h4 {
    margin-bottom: 0.4rem;
    color: #f8fafc;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
}

.stTabs [data-baseweb="tab"] {
    padding: 0.65rem 1.5rem;
    border-radius: 999px;
    background: rgba(15,23,42,0.3);
    color: #e2e8f0;
}

.stMetricDelta {
    color: var(--brand-secondary) !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(15,23,42,0.95), rgba(2,6,23,0.98));
    border-right: 1px solid rgba(148,163,184,0.2);
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
    color: #f8fafc;
    padding: 0.5rem 0;
    border-bottom: 1px solid rgba(99,102,241,0.3);
    margin-bottom: 1rem;
}

[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stCheckbox label {
    color: #e2e8f0;
    font-weight: 500;
}

[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: rgba(99,102,241,0.1);
    border: 1px dashed rgba(99,102,241,0.4);
    border-radius: 0.75rem;
    padding: 1rem;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown(
    """
    <div class="hero-card">
        <p class="eyebrow">Streamlit • Multi-format plotting</p>
        <h1>NovaScope</h1>
        <p>
            Drop in scientific data files, explore curated plot families, and publish ready-to-host visuals.
            Configure columns from the sidebar or directly inside the interactive table editor.
        </p>
        <div class="hero-badges">
            <span>Plotly + Seaborn powered</span>
            <span>Plot families for faster discovery</span>
            <span>3D rendering</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("Supported file types", expanded=False):
    st.markdown(
        "\n".join(f"- **{spec.extension}** — {spec.description}" for spec in SUPPORTED_FILES)
    )

sidebar = st.sidebar
sidebar.header("1. Upload a file")

types = [spec.extension.replace(".", "") for spec in SUPPORTED_FILES]
uploaded_file = sidebar.file_uploader(
    "Choose a data file",
    type=types,
    help="Accepted formats: " + ", ".join(ext for ext in types),
)


def _format_bytes(num_bytes: int | None) -> str:
    if not num_bytes:
        return "—"
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _format_option(option, none_label: str) -> str:
    return none_label if option is None else str(option)


def _format_plot_label(plot_key: str) -> str:
    if plot_key == "3dplot":
        return "3D Plot"
    return plot_key.replace("_", " ").title()


def _render_footer() -> None:
    st.divider()
    st.caption("Copyright © 2025 Shivaji Chaulagain")


def _prepare_preview(df: pd.DataFrame, limit: int = 200) -> pd.DataFrame:
    preview = df.head(limit).copy()
    object_cols = preview.select_dtypes(include=["object"]).columns
    if len(object_cols) > 0:
        preview[object_cols] = preview[object_cols].astype("string")
    return preview


def _sanitize_for_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats_df = df.copy()
    object_cols = stats_df.select_dtypes(include=["object"]).columns
    if len(object_cols) > 0:
        stats_df[object_cols] = stats_df[object_cols].astype("string")
    return stats_df


def _describe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    kwargs = {"include": "all"}

    version_parts = pd.__version__.split(".")
    try:
        major = int(version_parts[0])
        minor = int(version_parts[1]) if len(version_parts) > 1 else 0
    except ValueError:
        major = minor = 0

    if major > 1 or (major == 1 and minor >= 1):
        kwargs["datetime_is_numeric"] = True

    try:
        return _sanitize_for_stats(df).describe(**kwargs)
    except TypeError:
        kwargs.pop("datetime_is_numeric", None)
        return _sanitize_for_stats(df).describe(**kwargs)


def _normalize_dataset_path(key: str, frame: pd.DataFrame) -> str:
    attr_path = frame.attrs.get("dataset_path") if hasattr(frame, "attrs") else None
    if isinstance(attr_path, str) and attr_path.strip():
        normalized = attr_path.strip().strip("/")
        return normalized or key
    normalized = key.replace("\\", "/").strip("/")
    return normalized or key


def _build_dataset_tree(frames: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    tree: Dict[str, Any] = {}
    for key, frame in frames.items():
        display_path = _normalize_dataset_path(key, frame)
        parts = [part for part in display_path.split("/") if part] or [display_path]
        cursor = tree
        built_path: list[str] = []
        for part in parts[:-1]:
            built_path.append(part)
            node = cursor.setdefault(
                part,
                {"children": {}, "frame": None, "key": None, "path": "/".join(built_path)},
            )
            cursor = node["children"]
        built_path.append(parts[-1])
        leaf = cursor.setdefault(
            parts[-1],
            {"children": {}, "frame": None, "key": None, "path": "/".join(built_path)},
        )
        leaf["frame"] = frame
        leaf["key"] = key
    return tree


def _render_dataset_tree(
    tree: Dict[str, Any],
    selected_path: str,
    parent_expanded: bool = True,
    depth: int = 0,
    max_depth: int = 64,
) -> None:
    if depth >= max_depth:
        st.warning(
            f"Dataset tree truncated beyond depth {max_depth}. Use the sidebar selector to access deeply nested datasets."
        )
        return
    for name in sorted(tree.keys()):
        node = tree[name]
        children = node["children"]
        frame = node["frame"]
        node_path = node["path"]
        icon = "📂" if children else "📄"
        title = f"{icon} {name}"
        if selected_path == node_path:
            title += " (selected)"
        expanded = parent_expanded and (selected_path.startswith(node_path) or not children)
        with st.expander(title, expanded=expanded):
            st.caption(f"Path: `{node_path}`")
            if frame is not None:
                st.markdown(f"**Shape:** {frame.shape[0]} × {frame.shape[1]}")
                st.dataframe(_prepare_preview(frame), width="stretch")
            if children:
                _render_dataset_tree(
                    children,
                    selected_path,
                    parent_expanded=expanded,
                    depth=depth + 1,
                    max_depth=max_depth,
                )

if not uploaded_file:
    st.info("Upload a file to begin plotting.")
    _render_footer()
    st.stop()

try:
    file_bytes: bytes
    if hasattr(uploaded_file, "getvalue"):
        file_bytes = uploaded_file.getvalue()
    else:
        uploaded_file.seek(0)
        file_bytes = uploaded_file.read()
    bundle = load_data(BytesIO(file_bytes), uploaded_file.name)
    if isinstance(bundle, pd.DataFrame):
        inferred_ext = Path(uploaded_file.name).suffix or ""
        bundle = LoadedData.from_single(bundle, source=inferred_ext or "unknown")
except Exception as exc:
    st.error(f"Failed to load file: {exc}")
    _render_footer()
    st.stop()

is_hdf_bundle = bundle.source in {".h5", ".hdf5"}
is_fits_bundle = bundle.source in {".fits", ".fit"}

dataset_names = bundle.dataset_names()
default_index = dataset_names.index(bundle.default_key)
if bundle.has_multiple():
    dataset_key = sidebar.selectbox(
        "Dataset",
        dataset_names,
        index=default_index,
        help="Select which dataset inside the file should be used for previewing and plotting.",
    )
    sidebar.caption(
        f"Dataset '{dataset_key}' selected ({len(dataset_names)} total datasets detected)."
    )
else:
    dataset_key = bundle.default_key

data = bundle.select(dataset_key)
multi_dataset = bundle.has_multiple()

if is_hdf_bundle or is_fits_bundle:
    if ROW_INDEX_COLUMN not in data.columns:
        data = data.copy()
        data[ROW_INDEX_COLUMN] = pd.RangeIndex(start=0, stop=len(data))

image_array = data.attrs.get("image_array") if hasattr(data, "attrs") else None

sidebar.success("Dataset ready")
sidebar.caption(
    f"📄 {uploaded_file.name} • {_format_bytes(getattr(uploaded_file, 'size', None))}"
)

sidebar.header("2. Configure plot")
numeric_cols = data.select_dtypes(include=["number", "bool"]).columns.tolist()
all_cols = data.columns.tolist()

if not numeric_cols:
    st.warning("No numeric columns detected. Plotting may not work as expected.")
    numeric_cols = all_cols

plot_group_names = list(PLOT_GROUPS.keys())
selected_group = sidebar.selectbox("Plot family", plot_group_names, index=0)
plot_type = sidebar.selectbox(
    "Plot style",
    PLOT_GROUPS[selected_group],
    index=0,
    format_func=_format_plot_label,
)

x_options = [None] + all_cols

if (is_hdf_bundle or is_fits_bundle) and ROW_INDEX_COLUMN in all_cols:
    sidebar.info(
        "Multi-dataset scientific files expose a synthetic row index. Select the desired axis columns below or use the table editor for advanced control."
    )
    default_x_option = ROW_INDEX_COLUMN
else:
    default_x_option = None

default_x_index = x_options.index(default_x_option) if default_x_option in x_options else 0
x_col = sidebar.selectbox(
    "X axis column",
    x_options,
    index=default_x_index,
    format_func=lambda value: _format_option(value, "Index"),
)

multi_select_options = numeric_cols if plot_type != "heatmap" else all_cols
selected_y_default: list[str] = []
if numeric_cols:
    candidate_cols = [col for col in numeric_cols if col != ROW_INDEX_COLUMN]
    selected_y_default = candidate_cols[:1] if candidate_cols else numeric_cols[:1]
selected_y = sidebar.multiselect(
    "Y axis / value columns",
    options=multi_select_options,
    default=selected_y_default,
)

color_col = None
if plot_type not in {"histogram", "heatmap"}:
    color_options = [None] + all_cols
    color_col = sidebar.selectbox(
        "Color grouping",
        color_options,
        format_func=lambda value: _format_option(value, "None"),
    )

aggregate = None
if plot_type in {"line", "area", "bar"} and x_col:
    aggregate = sidebar.selectbox("Aggregate", [None, "mean", "sum", "min", "max"])

use_log_y = sidebar.checkbox("Log scale on Y", value=False)

selection_mode = sidebar.radio(
    "Column selection mode",
    ("Sidebar controls", "Table editor"),
    index=0,
    help="Use the sidebar widgets or assign roles directly from the table preview.",
)

sidebar.header("3. Plot")
sidebar.caption("Plots update automatically when options change.")

row_count = len(data)
column_count = len(all_cols)
missing_values = int(data.isna().sum().sum())
numeric_count = len(numeric_cols)
missing_pct = (
    (missing_values / (row_count * column_count) * 100)
    if row_count and column_count and missing_values
    else 0
)

metric_cols = st.columns(4, gap="large")
metric_cols[0].metric("Rows", f"{row_count:,}")
metric_cols[1].metric("Columns", f"{column_count:,}")
metric_cols[2].metric("Numeric fields", str(numeric_count))
metric_cols[3].metric("Missing values", f"{missing_values:,}", f"{missing_pct:.1f}%")

styled_fits_image = None

preview_tab, chart_tab, stats_tab = st.tabs([
    "Preview & selection",
    "Chart",
    "Statistics",
])

with preview_tab:
    st.markdown("#### Data preview")
    if multi_dataset:
        st.caption(
            "Browse datasets via the explorer or switch to a flat list. Use the search box to quickly locate paths inside large HDF5/FITS files."
        )

        dataset_rows = [
            {
                "Dataset": name,
                "Path": _normalize_dataset_path(name, frame),
                "Rows": len(frame),
                "Columns": len(frame.columns),
            }
            for name, frame in bundle.frames.items()
        ]
        dataset_filter = st.text_input(
            "Filter datasets",
            value="",
            placeholder="Type part of a name or path (case-insensitive)...",
        ).strip()
        if dataset_filter:
            dataset_rows = [
                row
                for row in dataset_rows
                if dataset_filter.lower() in row["Dataset"].lower()
                or dataset_filter.lower() in row["Path"].lower()
            ]
            if dataset_key not in {row["Dataset"] for row in dataset_rows}:
                selected_frame = bundle.frames[dataset_key]
                dataset_rows.append(
                    {
                        "Dataset": dataset_key,
                        "Path": _normalize_dataset_path(dataset_key, selected_frame),
                        "Rows": len(selected_frame),
                        "Columns": len(selected_frame.columns),
                    }
                )

        overview_df = pd.DataFrame(dataset_rows)
        if overview_df.empty:
            st.warning("No datasets match the current filter.")
        else:
            st.dataframe(overview_df.sort_values("Path"), width="stretch", hide_index=True)

        st.markdown("#### Dataset explorer")
        explorer_mode = st.radio(
            "Explorer layout",
            ("Tree", "Flat list"),
            horizontal=True,
            key="dataset_explorer_mode",
        )

        filtered_keys = [row["Dataset"] for row in dataset_rows]
        filtered_frames = {key: bundle.frames[key] for key in filtered_keys if key in bundle.frames}
        if explorer_mode == "Tree":
            dataset_tree = _build_dataset_tree(filtered_frames or bundle.frames)
            selected_path = _normalize_dataset_path(dataset_key, data)
            _render_dataset_tree(dataset_tree, selected_path)
        else:
            if not filtered_frames:
                st.info("No datasets available in the flat list view for the current filter.")
            for key in sorted(filtered_frames.keys()):
                frame = filtered_frames[key]
                path = _normalize_dataset_path(key, frame)
                title = f"📄 {path}"
                if key == dataset_key:
                    title += " (selected)"
                with st.expander(title, expanded=key == dataset_key):
                    st.markdown(f"**Shape:** {frame.shape[0]} × {frame.shape[1]}")
                    st.dataframe(_prepare_preview(frame), width="stretch")
    else:
        st.caption(
            "First 200 rows are shown for speed. Use search/filter controls within the table preview."
        )
    st.dataframe(_prepare_preview(data), width="stretch")

    if image_array is not None:
        st.markdown("#### FITS image preview")
        st.caption("Full-resolution image rendered directly from the FITS HDU.")
        control_cols = st.columns(2)
        color_choice = control_cols[0].selectbox(
            "Color map",
            list(DS9_COLORMAPS.keys()),
            key="fits_preview_colormap",
        )
        scale_choice = control_cols[1].selectbox(
            "Intensity scale",
            list(DS9_SCALE_MODES.keys()),
            key="fits_preview_scale",
        )
        styled_fits_image = _stylize_fits_image(image_array, color_choice, scale_choice)
        st.image(styled_fits_image, width=550)

    if selection_mode == "Table editor":
        st.markdown("#### Column selector")
        st.caption(
            "Assign plotting roles without leaving the table. First selected column becomes the X-axis by default."
        )

        selected_flags = []
        role_labels = []
        for column in all_cols:
            is_x = column == x_col
            is_y = column in selected_y
            is_color = bool(color_col and column == color_col)
            selected_flags.append(is_x or is_y)
            if is_x:
                role_labels.append("X-axis")
            elif is_y:
                role_labels.append("Y-axis")
            elif is_color:
                role_labels.append("Color")
            else:
                role_labels.append("Ignore")

        selector_df = pd.DataFrame({
            "Column": all_cols,
            "Selected": selected_flags,
            "Role": role_labels,
        })

        edited_roles = st.data_editor(
            selector_df,
            key="column_role_editor",
            hide_index=True,
            width="stretch",
            column_config={
                "Column": st.column_config.Column(disabled=True),
                "Selected": st.column_config.CheckboxColumn(
                    "Selected",
                    help="Quick-pick columns for plotting (first = X, rest = Y).",
                ),
                "Role": st.column_config.SelectboxColumn(
                    "Role",
                    options=("Ignore", "X-axis", "Y-axis", "Color"),
                ),
            },
        )

        selected_by_checkbox = edited_roles.loc[edited_roles["Selected"], "Column"].tolist()
        role_defined_x = edited_roles.loc[edited_roles["Role"] == "X-axis", "Column"].tolist()
        role_defined_y = edited_roles.loc[edited_roles["Role"] == "Y-axis", "Column"].tolist()
        role_defined_color = edited_roles.loc[edited_roles["Role"] == "Color", "Column"].tolist()

        if role_defined_x:
            x_col = role_defined_x[0]
        elif selected_by_checkbox:
            x_col = selected_by_checkbox[0]

        if role_defined_y:
            selected_y = role_defined_y
        elif len(selected_by_checkbox) >= 2:
            selected_y = selected_by_checkbox[1:]

        if role_defined_color:
            color_col = role_defined_color[0]

plot_ready = bool(selected_y)
request: PlotRequest | None = None
if plot_ready:
    request = PlotRequest(
        data=data,
        x=x_col,
        y=selected_y,
        color=color_col,
        plot_type=plot_type,
        use_log_y=use_log_y,
        aggregate=aggregate,
    )

with chart_tab:
    st.markdown("#### Visualization")
    image_shown = False
    display_image = styled_fits_image if styled_fits_image is not None else image_array
    if display_image is not None:
        st.markdown("##### FITS image")
        st.image(display_image, width=600)
        image_shown = True

    if not plot_ready:
        if image_shown:
            st.info("Select columns to generate additional plots, or rely on the FITS image above.")
        else:
            st.info("Select at least one column to start plotting via the sidebar or the table editor.")
    else:
        with st.spinner("Rendering chart..."):
            try:
                result = build_plot(request)
                if result.backend == "plotly":
                    st.plotly_chart(result.figure, width="stretch")
                else:
                    st.pyplot(result.figure)
            except Exception as exc:
                st.error(f"Unable to build plot: {exc}")

with stats_tab:
    st.markdown("#### Dataset profile")
    st.caption("Quick look at dtypes and descriptive statistics to understand the spread of your data.")
    st.write(pd.DataFrame({"column": data.columns, "dtype": data.dtypes}).set_index("column"))
    st.write(_describe_dataframe(data))

_render_footer()
