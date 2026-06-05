"""Tests for self-adaptive import thresholding."""

from pathlib import Path

import numpy as np
import pytest

from bioimage_pipeline.adaptive_import import (
    estimate_block_size,
    estimate_vignette_score,
    extract_2d_plane,
    run_self_adaptive_threshold,
    run_self_adaptive_threshold_on_folder,
)
from bioimage_pipeline.io import save_tiff
from bioimage_pipeline.validation import compare_masks

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "real_data"


def _make_vignetted_nuclei_image(shape: tuple[int, int] = (384, 384)) -> np.ndarray:
    rng = np.random.default_rng(0)
    image = rng.integers(80, 140, size=shape, dtype=np.uint16)
    rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]]
    vignette = 1.0 - 0.25 * (
        ((rows / shape[0]) - 0.5) ** 2 + ((cols / shape[1]) - 0.5) ** 2
    )
    image = (image.astype(np.float32) * vignette).astype(np.uint16)
    for center_y, center_x, radius, intensity in (
        (90, 95, 18, 1800),
        (95, 250, 16, 1650),
        (180, 150, 20, 1900),
        (250, 80, 14, 1500),
        (260, 290, 17, 1750),
        (310, 190, 15, 1600),
    ):
        circle = (rows - center_y) ** 2 + (cols - center_x) ** 2 <= radius**2
        image[circle] = np.maximum(image[circle], intensity)
    noise = rng.normal(0, 25, size=shape)
    return np.clip(image.astype(np.float32) + noise, 0, 4095).astype(np.uint16)


def _make_reference_mask(shape: tuple[int, int] = (384, 384)) -> np.ndarray:
    rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]]
    mask = np.zeros(shape, dtype=bool)
    for center_y, center_x, radius, _ in (
        (90, 95, 18, 1800),
        (95, 250, 16, 1650),
        (180, 150, 20, 1900),
        (250, 80, 14, 1500),
        (260, 290, 17, 1750),
        (310, 190, 15, 1600),
    ):
        circle = (rows - center_y) ** 2 + (cols - center_x) ** 2 <= radius**2
        mask[circle] = True
    return mask


def _make_low_snr_image(shape: tuple[int, int] = (256, 256)) -> np.ndarray:
    rng = np.random.default_rng(1)
    image = rng.integers(960, 1005, size=shape, dtype=np.uint16)
    rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]]
    for center_y, center_x, radius, intensity in (
        (70, 70, 12, 1045),
        (150, 170, 10, 1035),
        (200, 90, 11, 1040),
    ):
        circle = (rows - center_y) ** 2 + (cols - center_x) ** 2 <= radius**2
        image[circle] = np.maximum(image[circle], intensity)
    noise = rng.normal(0, 90, size=shape)
    return np.clip(image.astype(np.float32) + noise, 0, 4095).astype(np.uint16)


def test_extract_2d_plane_from_grayscale() -> None:
    image = np.zeros((64, 64), dtype=np.uint16)
    plane = extract_2d_plane(image)
    assert plane.shape == (64, 64)


def test_estimate_vignette_score_detects_corners_darker() -> None:
    image = np.full((128, 128), 200, dtype=np.uint16)
    image[:20, :] = 80
    image[-20:, :] = 80
    score = estimate_vignette_score(image)
    assert score > 0.1


def test_run_self_adaptive_threshold_returns_mask_and_labels() -> None:
    image = _make_vignetted_nuclei_image()
    result = run_self_adaptive_threshold(image)

    assert result.mask.dtype == bool
    assert result.mask.shape == (384, 384)
    assert result.labels.shape == (384, 384)
    assert result.decision.method in {"otsu", "local", "sauvola"}
    assert result.decision.confidence in {"high", "medium", "low"}
    assert result.decision.object_count >= 1


def test_self_adaptive_threshold_improves_dense_fixture_iou() -> None:
    image = _make_vignetted_nuclei_image()
    reference_mask = _make_reference_mask()

    result = run_self_adaptive_threshold(image)
    comparison = compare_masks(result.mask, reference_mask)

    assert comparison.iou >= 0.5
    assert comparison.object_count_a >= 4


def test_self_adaptive_threshold_runs_on_low_snr_fixture() -> None:
    image = _make_low_snr_image()
    result = run_self_adaptive_threshold(image)

    assert result.decision.method in {"otsu", "local", "sauvola"}
    assert result.mask.any()


def test_estimate_block_size_returns_odd_value() -> None:
    image = _make_vignetted_nuclei_image()
    block_size = estimate_block_size(image.astype(np.float32))
    assert block_size % 2 == 1
    assert 15 <= block_size <= 101


def test_run_self_adaptive_threshold_on_folder_writes_staging(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    staging_dir = tmp_path / "staging"
    input_dir.mkdir()
    save_tiff(input_dir / "cells_dense.tif", _make_vignetted_nuclei_image())

    summary = run_self_adaptive_threshold_on_folder(
        input_dir,
        staging_dir,
        logs_dir=tmp_path / "logs",
    )

    assert summary["processed"] == ["cells_dense.tif"]
    assert (staging_dir / "masks" / "cells_dense.tif").exists()
    assert (staging_dir / "labels" / "cells_dense.tif").exists()
    assert "cells_dense.tif" in summary["decisions"]
    assert Path(summary["summary_path"]).exists()


def test_run_self_adaptive_threshold_on_folder_missing_input_raises(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        run_self_adaptive_threshold_on_folder(
            tmp_path / "missing",
            tmp_path / "staging",
        )
