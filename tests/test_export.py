"""Tests for export helpers."""

import numpy as np
import pandas as pd

from bioimage_pipeline.export import (
    export_label_tiff,
    export_mask_tiff,
    export_measurements_csv,
)
from bioimage_pipeline.io import read_tiff


def test_export_mask_tiff_saves_uint8_0_255(tmp_path) -> None:
    mask = np.array([[True, False], [False, True]])
    path = tmp_path / "mask.tif"

    export_mask_tiff(path, mask)
    loaded = read_tiff(path)

    assert loaded.dtype == np.uint8
    np.testing.assert_array_equal(loaded, [[255, 0], [0, 255]])


def test_export_label_tiff_preserves_integer_labels(tmp_path) -> None:
    labels = np.array([[0, 1], [1, 2]], dtype=np.int32)
    path = tmp_path / "labels.tif"

    export_label_tiff(path, labels)
    loaded = read_tiff(path)

    np.testing.assert_array_equal(loaded, labels)


def test_export_measurements_csv_creates_readable_file(tmp_path) -> None:
    dataframe = pd.DataFrame({"label": [1, 2], "area": [10, 20]})
    path = tmp_path / "nested" / "measurements.csv"

    export_measurements_csv(path, dataframe)
    loaded = pd.read_csv(path)

    pd.testing.assert_frame_equal(loaded, dataframe)
