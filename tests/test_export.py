"""Tests for export helpers."""

import numpy as np
import pandas as pd

from bioimage_pipeline.export import (
    export_intensity_tiff,
    export_label_tiff,
    export_mask_tiff,
    export_measurements_csv,
)
from bioimage_pipeline.fiji_tiff import TiffExportMetadata, read_fiji_tiff_metadata
from bioimage_pipeline.io import read_tiff


def test_export_mask_tiff_saves_uint8_0_255(tmp_path) -> None:
    mask = np.array([[True, False], [False, True]])
    path = tmp_path / "mask.tif"

    export_mask_tiff(path, mask)
    loaded = read_tiff(path)

    assert loaded.dtype == np.uint8
    assert set(np.unique(loaded)).issubset({0, 255})
    np.testing.assert_array_equal(loaded, [[255, 0], [0, 255]])


def test_export_mask_tiff_writes_imagej_metadata(tmp_path) -> None:
    mask = np.array([[True, False], [False, True]])
    metadata = TiffExportMetadata(
        pixel_size_x=0.5,
        pixel_size_y=0.5,
        unit="um",
        channel_name="Mask",
        description="segmentation mask",
    )

    export_mask_tiff(tmp_path / "mask_meta.tif", mask, metadata=metadata)
    loaded_metadata = read_fiji_tiff_metadata(tmp_path / "mask_meta.tif")

    assert loaded_metadata["channel_name"] == "Mask"
    assert loaded_metadata["description"] == "segmentation mask"


def test_export_label_tiff_preserves_integer_labels_as_uint16(tmp_path) -> None:
    labels = np.array([[0, 1], [1, 2]], dtype=np.int32)
    path = tmp_path / "labels.tif"

    export_label_tiff(path, labels)
    loaded = read_tiff(path)

    assert loaded.dtype == np.uint16
    np.testing.assert_array_equal(loaded, labels.astype(np.uint16))


def test_export_label_tiff_uses_uint32_for_large_ids(tmp_path) -> None:
    labels = np.array([[0, 70000]], dtype=np.int32)

    export_label_tiff(tmp_path / "labels32.tif", labels)
    loaded = read_tiff(tmp_path / "labels32.tif")

    assert loaded.dtype == np.uint32
    assert int(loaded.max()) == 70000


def test_export_intensity_tiff_preserves_uint16_dtype(tmp_path) -> None:
    image = np.arange(9, dtype=np.uint16).reshape(3, 3)

    export_intensity_tiff(tmp_path / "intensity.tif", image)
    loaded = read_tiff(tmp_path / "intensity.tif")

    assert loaded.dtype == np.uint16
    np.testing.assert_array_equal(loaded, image)


def test_export_measurements_csv_creates_readable_file(tmp_path) -> None:
    dataframe = pd.DataFrame({"label": [1, 2], "area": [10, 20]})
    path = tmp_path / "nested" / "measurements.csv"

    export_measurements_csv(path, dataframe)
    loaded = pd.read_csv(path)

    pd.testing.assert_frame_equal(loaded, dataframe)
