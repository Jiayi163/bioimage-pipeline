"""Tests for Z-max projection utilities."""

from pathlib import Path

import numpy as np
import pytest
import tifffile

from bioimage_pipeline.z_projection import (
    Z_PROJECTION_METHODS,
    format_oir_read_dependency_error,
    iter_oir_files,
    normalize_projection_method,
    oir_output_filename,
    oir_output_path,
    project_stack,
    zmax_intensity,
)


def test_oir_output_filename_replaces_oir_with_tif() -> None:
    assert oir_output_filename("sample.oir") == "sample.tif"
    assert oir_output_filename("negative control_0004.oir") == "negative control_0004.tif"
    assert oir_output_filename("my.stack.oir") == "my.stack.tif"


def test_zmax_intensity_collapses_z_axis() -> None:
    stack = np.array(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
            [[2, 9], [1, 0]],
        ],
        dtype=np.uint16,
    )
    projected = zmax_intensity(stack, axis=0)
    np.testing.assert_array_equal(projected, [[5, 9], [7, 8]])


def test_zmax_intensity_passthrough_2d() -> None:
    image = np.arange(6, dtype=np.uint8).reshape(2, 3)
    np.testing.assert_array_equal(zmax_intensity(image), image)


@pytest.mark.parametrize("method", Z_PROJECTION_METHODS)
def test_project_stack_supports_all_fiji_methods(method: str) -> None:
    stack = np.array(
        [
            [[0, 1], [2, 3]],
            [[3, 4], [5, 6]],
            [[9, 8], [7, 0]],
        ],
        dtype=np.float64,
    )
    reducers = {
        "average": np.mean,
        "max": np.max,
        "min": np.min,
        "sum": np.sum,
        "standard": np.std,
        "median": np.median,
    }
    projected = project_stack(stack, method, axis=0)
    expected = reducers[method](stack, axis=0)
    np.testing.assert_allclose(projected, expected)


def test_normalize_projection_method_accepts_aliases() -> None:
    assert normalize_projection_method("Max Intensity") == "max"
    assert normalize_projection_method("standard_deviation") == "standard"


def test_normalize_projection_method_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unsupported Z projection method"):
        normalize_projection_method("mode")


def test_all_gui_methods_are_supported() -> None:
    assert set(Z_PROJECTION_METHODS) == {
        "average",
        "max",
        "min",
        "sum",
        "standard",
        "median",
    }


def test_iter_oir_files_finds_nested_files(tmp_path: Path) -> None:
    root = tmp_path / "input"
    sub = root / "nested"
    sub.mkdir(parents=True)
    (root / "a.oir").write_bytes(b"")
    (sub / "b.oir").write_bytes(b"")
    (root / "skip.tif").write_bytes(b"")

    found = [path.name for path in iter_oir_files(root)]
    assert found == ["a.oir", "b.oir"]


def test_zmax_on_multipage_tiff_matches_numpy_max(tmp_path: Path) -> None:
    planes = [np.full((4, 5), i, dtype=np.uint8) for i in (1, 5, 3)]
    stack_path = tmp_path / "stack.tif"
    tifffile.imwrite(stack_path, np.stack(planes), photometric="minisblack")

    loaded = tifffile.imread(stack_path)
    projected = zmax_intensity(loaded, axis=0)
    np.testing.assert_array_equal(projected, np.full((4, 5), 5, dtype=np.uint8))


def test_iter_oir_files_requires_directory(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        list(iter_oir_files(tmp_path / "missing"))


def test_oir_output_path_uses_tif_extension(tmp_path: Path) -> None:
    out = oir_output_path(tmp_path / "nested" / "sample.oir", tmp_path / "results")
    assert out == tmp_path / "results" / "sample.tif"


def test_format_oir_read_dependency_error_mentions_fiji() -> None:
    message = format_oir_read_dependency_error(
        RuntimeError("Java backend is not available")
    )
    assert "aicsimageio/bfio" in message
    assert "Fiji" in message
