"""Tests for the ImageStack data model and loaders (Phase S.2)."""

from pathlib import Path

import numpy as np
import pytest
import tifffile

from bioimage_pipeline.io import StackFrame
from bioimage_pipeline.stack import (
    ImageStack,
    load_stack,
    load_stack_from_folder,
    load_stack_from_tiff,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_single(path: Path, shape=(8, 10), dtype=np.uint8) -> np.ndarray:
    arr = np.random.randint(0, 200, size=shape, dtype=dtype)
    tifffile.imwrite(path, arr)
    return arr


def _write_multipage(path: Path, n: int, shape=(6, 8)) -> list[np.ndarray]:
    planes = [np.full(shape, i, dtype=np.uint8) for i in range(n)]
    tifffile.imwrite(path, np.stack(planes), photometric="minisblack")
    return planes


# ---------------------------------------------------------------------------
# ImageStack: properties and container behaviour
# ---------------------------------------------------------------------------


def test_image_stack_frame_count_matches_frames() -> None:
    frames = [
        StackFrame(index=i, array=np.zeros((4, 4), dtype=np.uint8))
        for i in range(3)
    ]
    stack = ImageStack(frames=frames, source=Path("dummy"))
    assert stack.frame_count == 3
    assert len(stack) == 3


def test_image_stack_shape_returns_hw_of_first_frame() -> None:
    frames = [StackFrame(index=0, array=np.zeros((5, 7), dtype=np.uint8))]
    stack = ImageStack(frames=frames, source=Path("dummy"))
    assert stack.shape == (5, 7)


def test_image_stack_shape_is_none_for_empty_stack() -> None:
    stack = ImageStack(frames=[], source=Path("dummy"))
    assert stack.shape is None


def test_image_stack_iteration_yields_frames() -> None:
    frames = [
        StackFrame(index=i, array=np.zeros((4, 4), dtype=np.uint8))
        for i in range(4)
    ]
    stack = ImageStack(frames=frames, source=Path("dummy"))
    indices = [f.index for f in stack]
    assert indices == [0, 1, 2, 3]


def test_image_stack_getitem_returns_correct_frame() -> None:
    frames = [
        StackFrame(index=i, array=np.full((2, 2), i, dtype=np.uint8))
        for i in range(3)
    ]
    stack = ImageStack(frames=frames, source=Path("dummy"))
    assert int(stack[2].array[0, 0]) == 2


# ---------------------------------------------------------------------------
# load_stack_from_tiff
# ---------------------------------------------------------------------------


def test_load_stack_from_tiff_single_page(tmp_path: Path) -> None:
    path = tmp_path / "image.tif"
    _write_single(path)
    stack = load_stack_from_tiff(path)
    assert stack.frame_count == 1
    assert isinstance(stack.frames[0], StackFrame)
    assert stack.frames[0].array.ndim == 2


def test_load_stack_from_tiff_multipage(tmp_path: Path) -> None:
    path = tmp_path / "stack.tif"
    _write_multipage(path, n=5)
    stack = load_stack_from_tiff(path)
    assert stack.frame_count == 5


def test_load_stack_from_tiff_frame_values_correct(tmp_path: Path) -> None:
    path = tmp_path / "stack.tif"
    planes = _write_multipage(path, n=3)
    stack = load_stack_from_tiff(path)
    for i, frame in enumerate(stack):
        assert int(frame.array[0, 0]) == i


def test_load_stack_from_tiff_source_is_file_path(tmp_path: Path) -> None:
    path = tmp_path / "image.tif"
    _write_single(path)
    stack = load_stack_from_tiff(path)
    assert stack.source == path


def test_load_stack_from_tiff_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_stack_from_tiff(tmp_path / "no_such.tif")


def test_load_stack_from_tiff_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_stack_from_tiff(tmp_path)


# ---------------------------------------------------------------------------
# load_stack_from_folder
# ---------------------------------------------------------------------------


def test_load_stack_from_folder_one_frame_per_file(tmp_path: Path) -> None:
    for i in range(4):
        _write_single(tmp_path / f"img_{i:02d}.tif")
    stack = load_stack_from_folder(tmp_path)
    assert stack.frame_count == 4


def test_load_stack_from_folder_alphabetical_order(tmp_path: Path) -> None:
    names = ["c.tif", "a.tif", "b.tif"]
    for name in names:
        _write_single(tmp_path / name)
    stack = load_stack_from_folder(tmp_path)
    stems = [f.source_path.stem for f in stack]
    assert stems == sorted(stems)


def test_load_stack_from_folder_frame_indices_sequential(tmp_path: Path) -> None:
    for i in range(3):
        _write_single(tmp_path / f"img_{i}.tif")
    stack = load_stack_from_folder(tmp_path)
    assert [f.index for f in stack] == [0, 1, 2]


def test_load_stack_from_folder_source_paths_attached(tmp_path: Path) -> None:
    for i in range(2):
        _write_single(tmp_path / f"img_{i}.tif")
    stack = load_stack_from_folder(tmp_path)
    for frame in stack:
        assert frame.source_path is not None
        assert frame.source_path.is_file()


def test_load_stack_from_folder_frame_arrays_are_2d(tmp_path: Path) -> None:
    _write_single(tmp_path / "img.tif")
    stack = load_stack_from_folder(tmp_path)
    for frame in stack:
        assert frame.array.ndim == 2


def test_load_stack_from_folder_missing_folder_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_stack_from_folder(tmp_path / "no_such_folder")


def test_load_stack_from_folder_empty_folder_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No TIFF files"):
        load_stack_from_folder(tmp_path)


def test_load_stack_from_folder_file_path_raises(tmp_path: Path) -> None:
    path = tmp_path / "image.tif"
    _write_single(path)
    with pytest.raises(ValueError):
        load_stack_from_folder(path)


# ---------------------------------------------------------------------------
# load_stack — auto-detection
# ---------------------------------------------------------------------------


def test_load_stack_auto_detects_file(tmp_path: Path) -> None:
    path = tmp_path / "stack.tif"
    _write_multipage(path, n=3)
    stack = load_stack(path)
    assert stack.frame_count == 3


def test_load_stack_auto_detects_folder(tmp_path: Path) -> None:
    for i in range(3):
        _write_single(tmp_path / f"img_{i}.tif")
    stack = load_stack(tmp_path)
    assert stack.frame_count == 3


def test_load_stack_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_stack(tmp_path / "nowhere")


def test_load_stack_file_and_folder_same_frame_count(tmp_path: Path) -> None:
    n = 4
    file_path = tmp_path / "stack.tif"
    folder = tmp_path / "folder"
    folder.mkdir()

    _write_multipage(file_path, n=n)
    for i in range(n):
        _write_single(folder / f"img_{i:02d}.tif")

    file_stack = load_stack(file_path)
    folder_stack = load_stack(folder)
    assert file_stack.frame_count == folder_stack.frame_count == n
