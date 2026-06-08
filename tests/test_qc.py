"""Tests for QC visualization helpers."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from bioimage_pipeline.io import read_tiff, save_tiff
from bioimage_pipeline.qc import (
    create_label_overlay,
    create_mask_overlay,
    export_qc_artifacts,
    generate_qc_for_folder,
    generate_qc_for_stack,
    normalize_for_display,
    save_qc_figure,
    view_in_napari,
)
from bioimage_pipeline.stack import load_stack_from_folder


def test_normalize_for_display_scales_to_uint8() -> None:
    image = np.array([[0, 100], [200, 400]], dtype=np.uint16)
    display = normalize_for_display(image)

    assert display.dtype == np.uint8
    assert display.min() >= 0
    assert display.max() <= 255


def test_create_mask_overlay_returns_rgb_uint8() -> None:
    image = np.zeros((40, 40), dtype=np.uint8)
    image[10:20, 10:20] = 200
    mask = np.zeros((40, 40), dtype=bool)
    mask[12:18, 12:18] = True

    overlay = create_mask_overlay(image, mask)

    assert overlay.shape == (40, 40, 3)
    assert overlay.dtype == np.uint8
    assert overlay[12, 12, 0] > overlay[0, 0, 0]


def test_create_mask_overlay_requires_matching_shapes() -> None:
    image = np.zeros((20, 20), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=bool)

    with pytest.raises(ValueError, match="shape"):
        create_mask_overlay(image, mask)


def test_create_label_overlay_returns_rgb_uint8() -> None:
    image = np.zeros((30, 30), dtype=np.uint8)
    image[5:15, 5:15] = 180
    labels = np.zeros((30, 30), dtype=np.int32)
    labels[6:14, 6:14] = 1

    overlay = create_label_overlay(image, labels)

    assert overlay.shape == (30, 30, 3)
    assert overlay.dtype == np.uint8


def test_save_qc_figure_writes_png(tmp_path: Path) -> None:
    figure = np.zeros((20, 20, 3), dtype=np.uint8)
    figure[5:10, 5:10] = [255, 0, 0]
    path = save_qc_figure(tmp_path / "overlay.png", figure)

    assert path.exists()
    assert path.suffix == ".png"


def test_save_qc_figure_writes_tiff(tmp_path: Path) -> None:
    figure = np.zeros((20, 20, 3), dtype=np.uint8)
    figure[5:10, 5:10] = [0, 255, 0]
    path = save_qc_figure(tmp_path / "overlay.tif", figure)

    loaded = read_tiff(path)
    assert loaded.shape == (20, 20, 3)


def test_export_qc_artifacts_creates_mask_and_label_files(tmp_path: Path) -> None:
    image = np.zeros((25, 25), dtype=np.uint8)
    image[5:15, 5:15] = 200
    mask = np.zeros((25, 25), dtype=bool)
    mask[6:14, 6:14] = True
    labels = np.zeros((25, 25), dtype=np.int32)
    labels[6:14, 6:14] = 1

    artifacts = export_qc_artifacts(
        image,
        tmp_path,
        "sample",
        mask=mask,
        labels=labels,
    )

    assert artifacts["mask_overlay"].exists()
    assert artifacts["label_overlay"].exists()


def test_generate_qc_for_folder_reads_batch_outputs(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    image = np.zeros((30, 30), dtype=np.uint8)
    image[:, 15:] = 200
    mask = image > 100
    labels = np.zeros((30, 30), dtype=np.int32)
    labels[mask] = 1

    save_tiff(input_dir / "cell.tif", image)
    save_tiff(output_dir / "cell_mask.tif", mask.astype(np.uint8) * 255)
    save_tiff(output_dir / "cell_labels.tif", labels)

    result = generate_qc_for_folder(input_dir, output_dir)

    assert result["generated"] == ["cell.tif"]
    assert result["skipped"] == []
    assert (output_dir / "cell_qc_mask_overlay.png").exists()
    assert (output_dir / "cell_qc_label_overlay.png").exists()


def test_view_in_napari_raises_without_optional_dependency() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(RuntimeError, match="Napari is not installed"):
        view_in_napari(image)


@patch("bioimage_pipeline.qc.napari", create=True)
def test_view_in_napari_opens_viewer(mock_napari_module: MagicMock) -> None:
    mock_viewer = MagicMock()
    mock_napari_module.Viewer.return_value = mock_viewer
    mock_napari_module.run = MagicMock()

    image = np.zeros((10, 10), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True

    import sys

    sys.modules["napari"] = mock_napari_module
    try:
        viewer = view_in_napari(image, mask=mask)
    finally:
        del sys.modules["napari"]

    assert viewer is mock_viewer
    mock_viewer.add_image.assert_called_once()
    mock_viewer.add_labels.assert_called_once()
    mock_napari_module.run.assert_called_once()


# ---------------------------------------------------------------------------
# generate_qc_for_stack (Phase S.7)
# ---------------------------------------------------------------------------


def _make_stack_with_outputs(tmp_path: Path, n_frames: int = 3):
    """Create a folder-based stack and matching mask/label TIFFs in output_dir."""
    src = tmp_path / "src"
    src.mkdir()
    out = tmp_path / "out"
    out.mkdir()

    for i in range(n_frames):
        img = np.zeros((30, 40), dtype=np.uint8)
        img[5:15, 10:25] = 180
        save_tiff(src / f"img_{i:02d}.tif", img)

        mask = img > 100
        mask_uint8 = mask.astype(np.uint8) * 255
        save_tiff(out / f"img_{i:02d}_f{i:03d}_mask.tif", mask_uint8)

        import skimage.measure
        labels = skimage.measure.label(mask).astype(np.int32)
        save_tiff(out / f"img_{i:02d}_f{i:03d}_labels.tif", labels)

    stack = load_stack_from_folder(src)
    return stack, out


def test_generate_qc_for_stack_creates_overlays_for_each_frame(tmp_path: Path) -> None:
    stack, out = _make_stack_with_outputs(tmp_path, n_frames=3)
    qc_dir = tmp_path / "qc"

    artifacts = generate_qc_for_stack(stack, out, image_format="png")

    assert len(artifacts) == 3
    for frame_idx, frame_artifacts in artifacts.items():
        assert "mask_overlay" in frame_artifacts
        assert "label_overlay" in frame_artifacts
        assert frame_artifacts["mask_overlay"].exists()
        assert frame_artifacts["label_overlay"].exists()


def test_generate_qc_for_stack_returns_frame_index_keys(tmp_path: Path) -> None:
    stack, out = _make_stack_with_outputs(tmp_path, n_frames=2)
    artifacts = generate_qc_for_stack(stack, out)
    assert set(artifacts.keys()) == {0, 1}


def test_generate_qc_for_stack_skips_frames_without_outputs(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    out = tmp_path / "out"
    out.mkdir()

    for i in range(3):
        img = np.zeros((20, 20), dtype=np.uint8)
        save_tiff(src / f"img_{i}.tif", img)

    stack = load_stack_from_folder(src)
    artifacts = generate_qc_for_stack(stack, out)

    assert artifacts == {}
