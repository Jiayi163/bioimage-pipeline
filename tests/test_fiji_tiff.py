"""Tests for Fiji/ImageJ-compatible TIFF export."""

import numpy as np
import pytest

from bioimage_pipeline.fiji_tiff import (
    TiffExportMetadata,
    label_export_dtype,
    prepare_intensity_for_export,
    prepare_labels_for_export,
    prepare_mask_for_export,
    read_fiji_tiff_metadata,
    save_fiji_compatible_tiff,
)
from bioimage_pipeline.io import read_tiff


def test_prepare_mask_for_export_uses_uint8_binary_values() -> None:
    mask = np.array([[True, False], [False, True]])
    exported = prepare_mask_for_export(mask)

    assert exported.dtype == np.uint8
    np.testing.assert_array_equal(exported, [[255, 0], [0, 255]])


def test_prepare_labels_for_export_uses_uint16_for_small_label_ids() -> None:
    labels = np.array([[0, 1], [2, 65535]], dtype=np.int32)
    exported = prepare_labels_for_export(labels)

    assert exported.dtype == np.uint16
    np.testing.assert_array_equal(exported, labels.astype(np.uint16))


def test_label_export_dtype_uses_uint32_for_large_label_ids() -> None:
    labels = np.array([[0, 70000]], dtype=np.int32)

    assert label_export_dtype(labels) == np.uint32


def test_prepare_intensity_for_export_preserves_uint16() -> None:
    image = np.arange(9, dtype=np.uint16).reshape(3, 3)
    exported = prepare_intensity_for_export(image)

    assert exported.dtype == np.uint16
    np.testing.assert_array_equal(exported, image)


def test_save_and_read_fiji_compatible_tiff_preserves_shape_and_dtype(
    tmp_path,
) -> None:
    image = np.arange(16, dtype=np.uint16).reshape(4, 4)
    path = save_fiji_compatible_tiff(tmp_path / "intensity.tif", image)

    loaded = read_tiff(path)

    assert loaded.shape == image.shape
    assert loaded.dtype == image.dtype
    np.testing.assert_array_equal(loaded, image)


def test_export_mask_round_trip_keeps_binary_semantics(tmp_path) -> None:
    mask = np.array([[True, False], [False, True]], dtype=bool)
    path = save_fiji_compatible_tiff(
        tmp_path / "mask.tif",
        prepare_mask_for_export(mask),
    )

    loaded = read_tiff(path)

    assert loaded.dtype == np.uint8
    assert set(np.unique(loaded)).issubset({0, 255})
    np.testing.assert_array_equal(loaded, prepare_mask_for_export(mask))


def test_export_labels_round_trip_preserves_label_ids(tmp_path) -> None:
    labels = np.array([[0, 1, 2], [0, 3, 4]], dtype=np.int32)
    exported = prepare_labels_for_export(labels)
    path = save_fiji_compatible_tiff(tmp_path / "labels.tif", exported)

    loaded = read_tiff(path)

    assert loaded.dtype == np.uint16
    np.testing.assert_array_equal(loaded, exported)
    assert int(loaded.max()) == 4


def test_save_fiji_compatible_tiff_writes_metadata(tmp_path) -> None:
    image = np.zeros((8, 8), dtype=np.uint8)
    metadata = TiffExportMetadata(
        pixel_size_x=0.65,
        pixel_size_y=0.65,
        unit="um",
        channel_name="DAPI",
        description="mask overlay",
    )

    path = save_fiji_compatible_tiff(tmp_path / "meta.tif", image, metadata=metadata)
    loaded_metadata = read_fiji_tiff_metadata(path)

    assert loaded_metadata["channel_name"] == "DAPI"
    assert loaded_metadata["description"] == "mask overlay"
    assert loaded_metadata["unit"] == "um"
    assert loaded_metadata["pixel_size"] == pytest.approx(0.65, rel=1e-3)


def test_save_fiji_compatible_tiff_invalid_pixel_size_raises(tmp_path) -> None:
    image = np.zeros((4, 4), dtype=np.uint8)
    metadata = TiffExportMetadata(pixel_size_x=0.0, pixel_size_y=0.65)

    with pytest.raises(ValueError, match="pixel_size"):
        save_fiji_compatible_tiff(tmp_path / "bad.tif", image, metadata=metadata)
