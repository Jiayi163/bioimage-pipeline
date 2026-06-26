"""Tests for training image/label pair discovery."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bioimage_pipeline.io import save_tiff
from bioimage_pipeline.training_data import (
    LABEL_BACKGROUND,
    LABEL_EV,
    discover_training_pairs,
    validate_training_pairs,
)


def _write_pair(root: Path, name: str = "img_a") -> None:
    images = root / "images"
    labels = root / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    image = np.zeros((32, 32), dtype=np.uint8)
    image[5:10, 5:10] = 180
    label = np.zeros((32, 32), dtype=np.uint8)
    label[5:10, 5:10] = LABEL_EV
    label[20:25, 20:25] = LABEL_BACKGROUND
    save_tiff(images / f"{name}.tif", image)
    save_tiff(labels / f"{name}_labels.tif", label)


def test_discover_training_pairs(tmp_path: Path) -> None:
    _write_pair(tmp_path / "training_data")
    pairs = discover_training_pairs(tmp_path / "training_data")
    assert len(pairs) == 1
    assert pairs[0].image_path.name == "img_a.tif"
    assert pairs[0].label_path.name == "img_a_labels.tif"


def test_validate_training_pairs_reports_shape_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "training_data"
    images = root / "images"
    labels = root / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    save_tiff(images / "bad.tif", np.zeros((20, 20), dtype=np.uint8))
    save_tiff(labels / "bad_labels.tif", np.zeros((24, 24), dtype=np.uint8))
    pairs = discover_training_pairs(root)
    warnings = validate_training_pairs(pairs)
    assert any("Shape mismatch" in message for message in warnings)
