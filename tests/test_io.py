"""Tests for TIFF image input/output."""

import numpy as np
import pytest

from bioimage_pipeline.io import read_tiff, save_tiff


def test_save_and_read_tiff_round_trip(tmp_path) -> None:
    image = np.arange(25, dtype=np.uint16).reshape((5, 5))
    path = tmp_path / "image.tif"

    save_tiff(path, image)
    loaded = read_tiff(path)

    np.testing.assert_array_equal(loaded, image)
    assert loaded.dtype == image.dtype


def test_save_and_read_tiff_accepts_string_paths(tmp_path) -> None:
    image = np.arange(9, dtype=np.uint8).reshape((3, 3))
    path = tmp_path / "string-path.tif"

    save_tiff(str(path), image)
    loaded = read_tiff(str(path))

    np.testing.assert_array_equal(loaded, image)


def test_save_and_read_mask_preserves_shape_and_dtype(tmp_path) -> None:
    mask = np.array([[True, False], [False, True]])
    path = tmp_path / "mask.tif"

    save_tiff(path, mask)
    loaded = read_tiff(path)

    np.testing.assert_array_equal(loaded, mask)
    assert loaded.shape == mask.shape
    assert loaded.dtype == mask.dtype


def test_save_and_read_labels_preserves_shape_and_dtype(tmp_path) -> None:
    labels = np.array([[0, 1, 1], [0, 2, 2]], dtype=np.int32)
    path = tmp_path / "labels.tiff"

    save_tiff(path, labels)
    loaded = read_tiff(path)

    np.testing.assert_array_equal(loaded, labels)
    assert loaded.shape == labels.shape
    assert loaded.dtype == labels.dtype


def test_save_tiff_creates_parent_directories(tmp_path) -> None:
    image = np.zeros((3, 3), dtype=np.uint8)
    path = tmp_path / "nested" / "mask.tif"

    save_tiff(path, image)

    assert path.exists()


def test_read_tiff_missing_file_raises_file_not_found(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        read_tiff(tmp_path / "missing.tif")


def test_read_tiff_directory_raises_value_error(tmp_path) -> None:
    with pytest.raises(ValueError):
        read_tiff(tmp_path)


def test_save_tiff_requires_numpy_array(tmp_path) -> None:
    with pytest.raises(ValueError):
        save_tiff(tmp_path / "image.tif", [[1, 2], [3, 4]])
