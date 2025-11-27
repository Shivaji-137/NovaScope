from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

from plotter.data_loader import (
    EmptyDataError,
    LoadedData,
    UnsupportedFileError,
    load_data,
)


class MemoryUpload:
    """Mimic Streamlit's UploadedFile interface for tests."""

    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


def test_load_csv_returns_dataframe():
    csv_bytes = b"time,value\n0,10\n1,20\n"
    upload = MemoryUpload(csv_bytes)
    bundle = load_data(upload, "sample.csv")
    assert isinstance(bundle, LoadedData)
    df = bundle.primary
    assert list(df.columns) == ["time", "value"]
    assert df["value"].tolist() == [10, 20]


def test_load_npz_converts_arrays():
    buffer = BytesIO()
    np.savez(buffer, sensor=np.array([[1, 2, 3], [4, 5, 6]]))
    upload = MemoryUpload(buffer.getvalue())
    bundle = load_data(upload, "data.npz")
    df = bundle.primary
    assert sorted(df.columns) == ["sensor_0", "sensor_1"]
    assert len(df) == 3


def test_load_npz_object_array_numeric():
    buffer = BytesIO()
    np.savez(buffer, readings=np.array([1, 2, 3], dtype=object))
    upload = MemoryUpload(buffer.getvalue())
    bundle = load_data(upload, "object_numeric.npz")
    df = bundle.primary
    assert list(df.columns) == ["readings"]
    assert df["readings"].tolist() == [1.0, 2.0, 3.0]


def test_load_npz_object_array_strings():
    buffer = BytesIO()
    np.savez(buffer, labels=np.array(["a", "b"], dtype=object))
    upload = MemoryUpload(buffer.getvalue())
    bundle = load_data(upload, "object_strings.npz")
    df = bundle.primary
    assert list(df.columns) == ["labels"]
    assert df["labels"].tolist() == ["a", "b"]


def test_load_dat_uses_hash_header():
    dat_bytes = b"# col1 col2 col3\n1 2 3\n4 5 6\n"
    upload = MemoryUpload(dat_bytes)
    bundle = load_data(upload, "sample.dat")
    df = bundle.primary
    assert list(df.columns) == ["col1", "col2", "col3"]
    assert df.iloc[0, 0] == 1


def test_load_dat_infers_text_header():
    dat_bytes = b"col_a col_b\n10 20\n30 40\n"
    upload = MemoryUpload(dat_bytes)
    bundle = load_data(upload, "sample2.dat")
    df = bundle.primary
    assert list(df.columns) == ["col_a", "col_b"]
    assert df.iloc[0, 0] == 10


def test_load_hdf5_extracts_datasets():
    h5py = pytest.importorskip("h5py")
    with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as tmp:
        with h5py.File(tmp.name, "w") as handle:
            handle.create_dataset("values", data=np.arange(4))
        tmp.flush()
        data = Path(tmp.name).read_bytes()
    upload = MemoryUpload(data)
    bundle = load_data(upload, "values.hdf5")
    assert isinstance(bundle, LoadedData)
    assert "values" in bundle.dataset_names()
    df = bundle.primary
    assert "values_0" in df.columns or "values" in df.columns
    assert df.shape[0] == 4
    Path(tmp.name).unlink(missing_ok=True)


def test_load_hdf5_preserves_dataset_path_metadata():
    h5py = pytest.importorskip("h5py")
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        with h5py.File(tmp.name, "w") as handle:
            grp = handle.create_group("run1")
            grp.create_dataset("measurements", data=np.arange(3))
        tmp.flush()
        data = Path(tmp.name).read_bytes()

    upload = MemoryUpload(data)
    bundle = load_data(upload, "nested.h5")
    df = bundle.primary
    assert df.attrs.get("dataset_path") == "run1/measurements"
    Path(tmp.name).unlink(missing_ok=True)


def test_load_fits_table():
    fits = pytest.importorskip("astropy.io.fits")
    cols = [
        fits.Column(name="x", array=np.array([1, 2, 3]), format="E"),
        fits.Column(name="y", array=np.array([4, 5, 6]), format="E"),
    ]
    hdu = fits.BinTableHDU.from_columns(cols)
    buffer = BytesIO()
    hdul = fits.HDUList([fits.PrimaryHDU(), hdu])
    hdul.writeto(buffer)
    upload = MemoryUpload(buffer.getvalue())
    bundle = load_data(upload, "stars.fits")
    assert isinstance(bundle, LoadedData)
    df = bundle.primary
    assert list(df.columns) == ["x", "y"]
    assert len(df) == 3


def test_load_fits_table_without_to_pandas(monkeypatch):
    fits = pytest.importorskip("astropy.io.fits")
    cols = [
        fits.Column(name="a", array=np.array([10, 20]), format="E"),
        fits.Column(name="b", array=np.array([30, 40]), format="E"),
    ]
    hdu = fits.BinTableHDU.from_columns(cols)

    class DummyHdu(fits.BinTableHDU):
        pass

    monkeypatch.setattr(hdu.__class__, "to_pandas", None, raising=False)

    buffer = BytesIO()
    hdul = fits.HDUList([fits.PrimaryHDU(), hdu])
    hdul.writeto(buffer)
    upload = MemoryUpload(buffer.getvalue())
    bundle = load_data(upload, "stars_no_pandas.fits")
    df = bundle.primary
    assert list(df.columns) == ["a", "b"]
    assert df.iloc[0].tolist() == [10, 30]


def test_load_fits_image_exposes_image_array():
    fits = pytest.importorskip("astropy.io.fits")
    image_data = (np.arange(16)).reshape(4, 4).astype(float)
    hdu = fits.PrimaryHDU(data=image_data)
    buffer = BytesIO()
    hdul = fits.HDUList([hdu])
    hdul.writeto(buffer)
    upload = MemoryUpload(buffer.getvalue())
    bundle = load_data(upload, "image.fits")
    df = bundle.primary
    image_array = df.attrs.get("image_array")
    assert image_array is not None
    assert image_array.shape == (4, 4)


def test_loaded_data_delegates_dataframe_attributes():
    df = pd.DataFrame({"value": [1, 2, 3]})
    bundle = LoadedData.from_single(df, source=".csv")

    assert list(bundle.columns) == ["value"]
    assert bundle["value"].tolist() == [1, 2, 3]
    assert len(bundle) == 3


def test_unsupported_extension_raises():
    upload = MemoryUpload(b"")
    with pytest.raises(UnsupportedFileError):
        load_data(upload, "data.xlsx")


def test_empty_file_raises():
    upload = MemoryUpload(b"")
    with pytest.raises(EmptyDataError):
        load_data(upload, "empty.csv")
