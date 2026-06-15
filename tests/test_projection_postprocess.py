"""Tests for projected TIFF post-processing before CellProfiler."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from bioimage_pipeline.projection_postprocess import (
    compare_projected_tiffs,
    normalize_projected_image_for_cellprofiler,
    validate_and_normalize_projected_tiff,
    validate_projected_tiffs_for_cellprofiler,
)


def test_normalize_float01_to_uint16() -> None:
    image = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32)
    normalized, action = normalize_projected_image_for_cellprofiler(
        image,
        projection_method="average",
    )
    assert action == "float01_scaled_to_uint16"
    assert normalized.dtype == np.uint16
    assert normalized.max() == 65535
    assert normalized[0, 0] == 0


def test_normalize_float_microscopy_intensity_clips_to_uint16() -> None:
    image = np.array([[0.0, 1234.5], [4095.0, 65000.0]], dtype=np.float32)
    normalized, action = normalize_projected_image_for_cellprofiler(
        image,
        projection_method="average",
    )
    assert action == "float_intensity_clipped_to_uint16"
    assert normalized.dtype == np.uint16
    np.testing.assert_array_equal(normalized, [[0, 1234], [4095, 65000]])


def test_normalize_float_large_values_scale_to_uint16() -> None:
    image = np.array([[0.0, 50000.0], [100000.0, 200000.0]], dtype=np.float64)
    normalized, action = normalize_projected_image_for_cellprofiler(
        image,
        projection_method="sum",
    )
    assert action == "float_intensity_scaled_to_uint16"
    assert normalized.dtype == np.uint16
    assert normalized.max() == 65535
    assert normalized[1, 0] == 32767


def test_normalize_uint16_is_unchanged() -> None:
    image = np.array([[1, 2], [3, 4000]], dtype=np.uint16)
    normalized, action = normalize_projected_image_for_cellprofiler(image)
    assert action == "unchanged_uint16"
    np.testing.assert_array_equal(normalized, image)


def test_validate_and_normalize_projected_tiff_rewrites_float_file(
    tmp_path: Path,
) -> None:
    tiff_path = tmp_path / "sample.tif"
    tifffile.imwrite(tiff_path, np.array([[10.5, 20.0]], dtype=np.float32))

    record = validate_and_normalize_projected_tiff(
        tiff_path,
        engine="python",
        projection_method="average",
    )

    assert record.rewritten is True
    assert record.dtype_before == "float32"
    assert record.dtype_after == "uint16"
    reloaded = tifffile.imread(tiff_path)
    assert reloaded.dtype == np.uint16


def test_validate_and_normalize_projected_tiff_preserves_fiji_output(
    tmp_path: Path,
) -> None:
    tiff_path = tmp_path / "sample.tif"
    original = np.array([[10, 20], [30, 4000]], dtype=np.uint16)
    tifffile.imwrite(tiff_path, original)

    record = validate_and_normalize_projected_tiff(
        tiff_path,
        engine="fiji",
        projection_method="max",
    )

    assert record.rewritten is False
    assert record.action == "fiji_preserved"
    np.testing.assert_array_equal(tifffile.imread(tiff_path), original)


def test_validate_fiji_float_output_is_not_rewritten(tmp_path: Path) -> None:
    tiff_path = tmp_path / "sample.tif"
    original = np.array([[10.5, 20.0], [30.0, 4000.0]], dtype=np.float32)
    tifffile.imwrite(tiff_path, original)

    record = validate_and_normalize_projected_tiff(
        tiff_path,
        engine="fiji",
        projection_method="max",
    )

    assert record.rewritten is False
    assert record.action == "fiji_preserved"
    np.testing.assert_array_equal(tifffile.imread(tiff_path), original)


def test_compare_projected_tiffs_reports_difference(tmp_path: Path) -> None:
    reference = tmp_path / "reference.tif"
    candidate = tmp_path / "candidate.tif"
    tifffile.imwrite(reference, np.array([[1, 2], [3, 4]], dtype=np.uint16))
    tifffile.imwrite(candidate, np.array([[1, 2], [3, 5]], dtype=np.uint16))

    comparison = compare_projected_tiffs(reference, candidate, projection_method="max")

    assert comparison.identical is False
    assert comparison.max_abs_diff == 1.0
    assert comparison.mean_abs_diff == 0.25


def test_validate_projected_tiffs_for_cellprofiler_writes_logs(tmp_path: Path) -> None:
    tiff_a = tmp_path / "a.tif"
    tiff_b = tmp_path / "b.tif"
    logs_dir = tmp_path / "logs"
    tifffile.imwrite(tiff_a, np.array([[0.0, 0.5]], dtype=np.float32))
    tifffile.imwrite(tiff_b, np.array([[100, 200]], dtype=np.uint16))

    records = validate_projected_tiffs_for_cellprofiler(
        [tiff_a, tiff_b],
        engine="python",
        projection_method="median",
        logs_dir=logs_dir,
    )

    assert len(records) == 2
    assert (logs_dir / "projection_postprocess.txt").is_file()
    assert (logs_dir / "projection_postprocess.json").is_file()
    text = (logs_dir / "projection_postprocess.txt").read_text(encoding="utf-8")
    assert "engine: python" in text
    assert "projection_method: median" in text
    assert "dtype_before: float32" in text


def test_normalize_rejects_non_2d_projection() -> None:
    image = np.zeros((2, 3, 4, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="2D projected plane"):
        normalize_projected_image_for_cellprofiler(image)
