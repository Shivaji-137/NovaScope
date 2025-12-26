from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError as PandasEmptyDataError


@dataclass(frozen=True)
class SupportedFile:
    extension: str
    description: str


SUPPORTED_FILES: Tuple[SupportedFile, ...] = (
    SupportedFile(".csv", "Comma separated values"),
    SupportedFile(".txt", "Plain text (delimiters auto-detected)"),
    SupportedFile(".dat", "Generic whitespace delimited data"),
    SupportedFile(".npz", "NumPy compressed archive"),
    SupportedFile(".fits", "Flexible Image Transport System"),
    SupportedFile(".fit", "Flexible Image Transport System"),
    SupportedFile(".hdf5", "Hierarchical Data Format"),
    SupportedFile(".h5", "Hierarchical Data Format"),
)

SUPPORTED_EXTENSIONS = tuple(spec.extension for spec in SUPPORTED_FILES)
_SUPPORTED_EXTENSION_SET = set(SUPPORTED_EXTENSIONS)


_TEXT_EXTS = {".csv", ".txt", ".dat"}
_NUMPY_EXTS = {".npz"}
_FITS_EXTS = {".fits", ".fit"}
_HDF_EXTS = {".hdf5", ".h5"}


@dataclass
class LoadedData:
    frames: Dict[str, pd.DataFrame]
    default_key: str
    source: str

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("LoadedData requires at least one DataFrame.")
        if self.default_key not in self.frames:
            raise ValueError("Default key must exist in frames.")

    @property
    def primary(self) -> pd.DataFrame:
        return self.frames[self.default_key]

    def dataset_names(self) -> List[str]:
        return list(self.frames.keys())

    def has_multiple(self) -> bool:
        return len(self.frames) > 1

    def select(self, key: str) -> pd.DataFrame:
        return self.frames[key]

    @classmethod
    def from_single(cls, df: pd.DataFrame, source: str, key: str = "main") -> "LoadedData":
        return cls(frames={key: df}, default_key=key, source=source)

    def __getattr__(self, name: str) -> Any:
        try:
            return getattr(self.primary, name)
        except AttributeError as exc:
            raise AttributeError(f"'LoadedData' object has no attribute '{name}'") from exc

    def __getitem__(self, item: Any) -> Any:
        return self.primary.__getitem__(item)

    def __len__(self) -> int:
        return len(self.primary)


class UnsupportedFileError(ValueError):
    """Raised when the user uploads an unsupported file type."""


class EmptyDataError(ValueError):
    """Raised when no tabular data can be extracted from the file."""


def iter_supported_extensions() -> Iterable[str]:
    return (spec.extension for spec in SUPPORTED_FILES)


def _to_dataframe_from_array(array_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Convert a mapping of array names to arrays into a DataFrame."""

    columns: Dict[str, List[Any]] = {}
    for key, array in array_dict.items():
        flattened = np.atleast_2d(array)
        if flattened.shape[0] == 1:
            # Single row vectors become columns
            columns[key] = flattened.flatten().tolist()
        else:
            # Multi-row arrays become multiple columns with suffixes
            for idx in range(flattened.shape[0]):
                columns[f"{key}_{idx}"] = flattened[idx].flatten().tolist()

    if not columns:
        raise EmptyDataError("No numeric arrays found in uploaded file.")

    # Align lengths by padding with NaN
    max_len = max(len(values) for values in columns.values())
    for key, values in columns.items():
        if len(values) < max_len:
            values.extend([np.nan] * (max_len - len(values)))

    return pd.DataFrame(columns)

def _looks_like_numeric(token: str) -> bool:
    candidate = token.replace("D", "E").replace("d", "E")
    try:
        float(candidate)
        return True
    except ValueError:
        return False


def _detect_header_line(file_buffer: BytesIO) -> Tuple[Optional[List[str]], int]:
    """Return header names and number of rows to skip for text files."""

    pos = file_buffer.tell()
    file_buffer.seek(0)
    first_line = file_buffer.readline()
    file_buffer.seek(pos)

    if not first_line:
        return None, 0

    stripped = first_line.strip()
    if not stripped:
        return None, 0

    is_comment = stripped.startswith(b"#")
    if is_comment:
        stripped = stripped.lstrip(b"#")

    try:
        decoded = stripped.decode("utf-8")
    except UnicodeDecodeError:
        decoded = stripped.decode("latin-1", errors="ignore")

    tokens = decoded.strip().split()
    if not tokens:
        return (None, 1) if is_comment else (None, 0)

    has_non_numeric = any(not _looks_like_numeric(token) for token in tokens)
    if is_comment or has_non_numeric:
        return tokens, 1

    return None, 0


def _sanitize_object_array(array: np.ndarray) -> np.ndarray:
    try:
        return np.array(array, dtype=float)
    except (TypeError, ValueError):
        try:
            return np.array(array.tolist())
        except Exception:
            return np.array(array, dtype=object)


def _load_text(file_buffer: BytesIO, extension: str) -> pd.DataFrame:
    file_buffer.seek(0)
    header_names: Optional[List[str]] = None
    skiprows = 0
    text_kwargs: Dict[str, object] = {}

    if extension != ".csv":
        header_names, skiprows = _detect_header_line(file_buffer)
        text_kwargs["header"] = None
        text_kwargs.setdefault("sep", r"\s+")
        text_kwargs.setdefault("engine", "python")
        if header_names:
            text_kwargs["names"] = header_names
            text_kwargs["skiprows"] = skiprows

    try:
        if extension == ".csv":
            df = pd.read_csv(file_buffer)
        else:
            df = pd.read_csv(file_buffer, **text_kwargs)
    except PandasEmptyDataError as exc:
        raise EmptyDataError("File does not contain tabular data.") from exc
    except Exception:
        file_buffer.seek(0)
        try:
            fallback_kwargs = dict(text_kwargs)
            fallback_kwargs["sep"] = r"[,\s]+"
            fallback_kwargs["engine"] = "python"
            df = pd.read_csv(file_buffer, **fallback_kwargs)
        except PandasEmptyDataError as exc:
            raise EmptyDataError("File does not contain tabular data.") from exc

    if df.empty:
        raise EmptyDataError("File does not contain any rows of data.")
    return df


def _load_npz(file_buffer: BytesIO) -> pd.DataFrame:
    file_buffer.seek(0)
    try:
        with np.load(file_buffer, allow_pickle=False) as archive:
            return _to_dataframe_from_array({key: archive[key] for key in archive.files})
    except ValueError as exc:
        error_msg = str(exc)
        if "Object arrays cannot be loaded when allow_pickle=False" not in error_msg:
            raise

    file_buffer.seek(0)
    with np.load(file_buffer, allow_pickle=True) as archive:
        cleaned: Dict[str, np.ndarray] = {}
        for key in archive.files:
            array = archive[key]
            if array.dtype.kind == "O":
                array = _sanitize_object_array(array)
            cleaned[key] = array
        return _to_dataframe_from_array(cleaned)


def _fits_dataset_label(hdu, index: int) -> str:
    name = getattr(hdu, "name", None)
    if isinstance(name, str) and name.strip() and name.upper() != "PRIMARY":
        return name.strip()
    return f"hdu_{index}"


def _load_fits(file_buffer: BytesIO) -> Dict[str, pd.DataFrame]:
    try:
        from astropy.io import fits
    except ImportError as exc:  # pragma: no cover - only hit when missing dep
        raise ImportError(
            "astropy is required to read FITS files. Install it via 'pip install astropy'."
        ) from exc

    file_buffer.seek(0)
    frames: Dict[str, pd.DataFrame] = {}
    with fits.open(file_buffer) as hdus:
        for idx, hdu in enumerate(hdus):
            data = hdu.data
            header = getattr(hdu, "header", None)
            if data is None:
                continue
            if hasattr(hdu, "columns"):
                to_pandas = getattr(hdu, "to_pandas", None)
                if callable(to_pandas):
                    df = to_pandas()
                else:
                    # Handle multi-dimensional columns in FITS tables
                    records = {}
                    for col_name in hdu.columns.names:
                        col_data = data[col_name]
                        if isinstance(col_data[0] if len(col_data) > 0 else None, np.ndarray):
                            # Multi-dimensional column - flatten or expand
                            if col_data[0].ndim == 1:
                                # 1D array: create separate columns for each element
                                for i in range(len(col_data[0])):
                                    records[f"{col_name}_{i}"] = [row[i] for row in col_data]
                            else:
                                # Higher dimensional: convert to string representation
                                records[col_name] = [str(arr) for arr in col_data]
                        else:
                            # Scalar column
                            records[col_name] = col_data.tolist()
                    df = pd.DataFrame(records)
                if not df.empty:
                    label = _fits_dataset_label(hdu, idx)
                    df.attrs["dataset_path"] = label
                    frames[label] = df
                    continue
            array = np.asarray(data)
            if array.ndim >= 2:
                label = _fits_dataset_label(hdu, idx)
                df = _to_dataframe_from_array({label: array})
                df.attrs["dataset_path"] = label
                if array.ndim == 2:
                    df.attrs["image_array"] = array
                frames[label] = df
            if header and len(header) > 0:
                header_df = pd.DataFrame(list(header.items()), columns=["keyword", "value"])
                # Convert mixed-type value column to string for Arrow compatibility
                header_df["value"] = header_df["value"].apply(lambda x: str(x) if x is not None else "").astype("string")
                label = _fits_dataset_label(hdu, idx)
                header_df.attrs["dataset_path"] = f"{label}/header"
                frames[f"{label}/header"] = header_df
    if not frames:
        raise EmptyDataError("Could not extract tabular data from FITS file.")
    return frames


def _load_hdf5(file_buffer: BytesIO) -> Dict[str, pd.DataFrame]:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - only hit when missing dep
        raise ImportError(
            "h5py is required to read HDF5 files. Install it via 'pip install h5py'."
        ) from exc

    file_buffer.seek(0)
    with h5py.File(file_buffer, "r") as handle:
        frames: Dict[str, pd.DataFrame] = {}
        stack: List[Tuple[str, h5py.Group | h5py.Dataset]] = [("", handle)]
        seen_refs: set[Any] = set()

        while stack:
            path, node = stack.pop()
            node_ref = getattr(node, "ref", None)
            if node_ref is not None:
                if node_ref in seen_refs:
                    continue
                seen_refs.add(node_ref)

            if isinstance(node, h5py.Dataset):
                dataset_path = path or getattr(node, "name", None) or "dataset"
                dataset_path = dataset_path.strip("/") or "dataset"
                df_label = dataset_path.replace("/", "_") or "dataset"
                df = _to_dataframe_from_array({df_label: node[()]})
                df.attrs["dataset_path"] = dataset_path
                frames[dataset_path] = df
                continue

            if isinstance(node, h5py.Group):
                for key, child in node.items():
                    child_path = f"{path}/{key}".strip("/")
                    stack.append((child_path, child))

        if not frames:
            raise EmptyDataError("No datasets found inside HDF5 file.")
        return frames


def load_data(uploaded_file, filename: str) -> LoadedData:
    """Return a LoadedData bundle from the uploaded file."""

    extension = Path(filename).suffix.lower()
    if extension not in _SUPPORTED_EXTENSION_SET:
        raise UnsupportedFileError(f"Unsupported file extension: {extension}")

    if isinstance(uploaded_file, BytesIO):
        buffer = uploaded_file
        buffer.seek(0)
    else:
        if hasattr(uploaded_file, "seek"):
            try:
                uploaded_file.seek(0)
            except Exception:
                pass

        file_bytes = uploaded_file.read()
        if not file_bytes and hasattr(uploaded_file, "getvalue"):
            try:
                file_bytes = uploaded_file.getvalue()
            except Exception:
                pass
        buffer = BytesIO(file_bytes)

    if extension in _TEXT_EXTS:
        return LoadedData.from_single(_load_text(buffer, extension), source=extension)
    if extension in _NUMPY_EXTS:
        return LoadedData.from_single(_load_npz(buffer), source=extension)
    if extension in _FITS_EXTS:
        fits_frames = _load_fits(buffer)
        default_key = next(iter(fits_frames.keys()))
        return LoadedData(frames=fits_frames, default_key=default_key, source=extension)
    if extension in _HDF_EXTS:
        hdf_frames = _load_hdf5(buffer)
        default_key = next(iter(hdf_frames.keys()))
        return LoadedData(frames=hdf_frames, default_key=default_key, source=extension)

    raise UnsupportedFileError(f"No loader found for extension {extension}")
