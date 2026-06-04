"""Tests for object measurements."""

import numpy as np
import pandas as pd

from bioimage_pipeline.measure import measure_objects


def test_measure_objects_one_row_per_object() -> None:
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[2:4, 2:4] = 1
    labels[6:9, 6:9] = 2

    table = measure_objects(labels)

    assert isinstance(table, pd.DataFrame)
    assert len(table) == 2
    assert set(table["label"]) == {1, 2}


def test_measure_objects_includes_area_and_centroid() -> None:
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[1:4, 1:4] = 1

    table = measure_objects(labels)

    assert table.loc[0, "area"] == 9
    assert "centroid-0" in table.columns
    assert "centroid-1" in table.columns


def test_measure_objects_intensity_from_original_image() -> None:
    labels = np.zeros((6, 6), dtype=np.int32)
    labels[1:4, 1:4] = 1

    intensity = np.zeros((6, 6), dtype=np.uint8)
    intensity[1:4, 1:4] = 200

    table = measure_objects(labels, intensity_image=intensity)

    assert "mean_intensity" in table.columns
    assert "max_intensity" in table.columns
    assert table.loc[0, "mean_intensity"] == 200
    assert table.loc[0, "max_intensity"] == 200


def test_measure_objects_without_intensity_omits_intensity_columns() -> None:
    labels = np.zeros((5, 5), dtype=np.int32)
    labels[2, 2] = 1

    table = measure_objects(labels)

    assert "mean_intensity" not in table.columns
    assert "max_intensity" not in table.columns
