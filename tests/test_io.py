"""Tests for TIFF image input/output."""

import numpy as np
import pytest
import tifffile

from bioimage_pipeline.io import (
    AxisInfo,
    StackFrame,
    extract_2d_plane,
    interpret_tiff_axes,
    iter_stack_frames,
    read_tiff,
    save_tiff,
)


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


# ---------------------------------------------------------------------------
# interpret_tiff_axes
# ---------------------------------------------------------------------------


def test_interpret_tiff_axes_2d_returns_single_frame() -> None:
    info = interpret_tiff_axes((64, 80))
    assert info.height == 64
    assert info.width == 80
    assert info.frame_count == 1
    assert info.z_count == 1
    assert info.t_count == 1
    assert info.c_count == 1


def test_interpret_tiff_axes_3d_treats_leading_dim_as_z() -> None:
    info = interpret_tiff_axes((5, 64, 80))
    assert info.z_count == 5
    assert info.t_count == 1
    assert info.c_count == 1
    assert info.frame_count == 5
    assert info.height == 64
    assert info.width == 80


def test_interpret_tiff_axes_4d_treats_as_z_c() -> None:
    info = interpret_tiff_axes((3, 2, 64, 80))
    assert info.z_count == 3
    assert info.c_count == 2
    assert info.frame_count == 6


def test_interpret_tiff_axes_5d_treats_as_t_z_c() -> None:
    info = interpret_tiff_axes((4, 3, 2, 64, 80))
    assert info.t_count == 4
    assert info.z_count == 3
    assert info.c_count == 2
    assert info.frame_count == 24


def test_interpret_tiff_axes_uses_imagej_metadata_when_provided() -> None:
    meta = {"slices": 7, "frames": 2, "channels": 3}
    info = interpret_tiff_axes((42, 64, 80), imagej_metadata=meta)
    assert info.z_count == 7
    assert info.t_count == 2
    assert info.c_count == 3
    assert info.frame_count == 42
    assert info.source == "imagej_metadata"


def test_interpret_tiff_axes_raises_for_1d() -> None:
    with pytest.raises(ValueError):
        interpret_tiff_axes((10,))


# ---------------------------------------------------------------------------
# extract_2d_plane
# ---------------------------------------------------------------------------


def test_extract_2d_plane_returns_2d_array_unchanged() -> None:
    image = np.zeros((16, 20), dtype=np.uint8)
    result = extract_2d_plane(image)
    assert result.shape == (16, 20)
    np.testing.assert_array_equal(result, image)


def test_extract_2d_plane_first_frame_of_stack() -> None:
    stack = np.arange(3 * 4 * 5, dtype=np.uint16).reshape(3, 4, 5)
    result = extract_2d_plane(stack, frame_index=0)
    assert result.shape == (4, 5)
    np.testing.assert_array_equal(result, stack[0])


def test_extract_2d_plane_selects_correct_frame() -> None:
    stack = np.zeros((4, 8, 8), dtype=np.uint8)
    stack[2] = 42
    result = extract_2d_plane(stack, frame_index=2)
    assert result.shape == (8, 8)
    assert result[0, 0] == 42


def test_extract_2d_plane_out_of_range_raises() -> None:
    stack = np.zeros((2, 4, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="out of range"):
        extract_2d_plane(stack, frame_index=5)


def test_extract_2d_plane_1d_raises() -> None:
    with pytest.raises(ValueError):
        extract_2d_plane(np.zeros(10))


# ---------------------------------------------------------------------------
# iter_stack_frames
# ---------------------------------------------------------------------------


def _write_multipage_tiff(path, arrays: list[np.ndarray]) -> None:
    """Write a list of 2D arrays as a multi-page grayscale TIFF."""
    tifffile.imwrite(path, np.stack(arrays), photometric="minisblack")


def test_iter_stack_frames_single_image_yields_one_frame(tmp_path) -> None:
    image = np.zeros((8, 10), dtype=np.uint8)
    path = tmp_path / "single.tif"
    tifffile.imwrite(path, image)

    frames = list(iter_stack_frames(path))
    assert len(frames) == 1
    assert isinstance(frames[0], StackFrame)
    assert frames[0].index == 0
    assert frames[0].array.shape == (8, 10)


def test_iter_stack_frames_multipage_yields_all_frames(tmp_path) -> None:
    planes = [np.full((6, 6), i, dtype=np.uint8) for i in range(5)]
    path = tmp_path / "stack.tif"
    _write_multipage_tiff(path, planes)

    frames = list(iter_stack_frames(path))
    assert len(frames) == 5
    for i, frame in enumerate(frames):
        assert frame.index == i
        assert frame.array.shape == (6, 6)
        assert int(frame.array[0, 0]) == i


def test_iter_stack_frames_frame_arrays_are_2d(tmp_path) -> None:
    planes = [np.zeros((4, 5), dtype=np.uint16) for _ in range(4)]
    path = tmp_path / "stack.tif"
    _write_multipage_tiff(path, planes)

    for frame in iter_stack_frames(path):
        assert frame.array.ndim == 2


def test_iter_stack_frames_attaches_source_path(tmp_path) -> None:
    path = tmp_path / "image.tif"
    tifffile.imwrite(path, np.zeros((4, 4), dtype=np.uint8))

    frames = list(iter_stack_frames(path))
    assert frames[0].source_path == path


def test_iter_stack_frames_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        list(iter_stack_frames(tmp_path / "no_such_file.tif"))


def test_iter_stack_frames_directory_raises(tmp_path) -> None:
    with pytest.raises(ValueError):
        list(iter_stack_frames(tmp_path))


def test_iter_stack_frames_z_index_set_for_multipage(tmp_path) -> None:
    planes = [np.zeros((4, 4), dtype=np.uint8) for _ in range(3)]
    path = tmp_path / "stack.tif"
    _write_multipage_tiff(path, planes)

    frames = list(iter_stack_frames(path))
    z_indices = [f.z_index for f in frames]
    assert z_indices == [0, 1, 2]
