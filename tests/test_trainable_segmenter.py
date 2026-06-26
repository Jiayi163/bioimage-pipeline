"""Tests for trainable segmentation (scikit-image gallery pattern)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bioimage_pipeline.io import save_tiff
from bioimage_pipeline.trainable_segmenter import (
    TrainableSegmenterConfig,
    load_model_bundle,
    predict_segmenter,
    train_segmenter,
)
from bioimage_pipeline.training_data import LABEL_BACKGROUND, LABEL_EV


def _make_training_pair(tmp_path: Path, shape: tuple[int, int] = (48, 48)) -> Path:
    training_root = tmp_path / "training_data"
    images_dir = training_root / "images"
    labels_dir = training_root / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    image = np.zeros(shape, dtype=np.uint8)
    labels = np.zeros(shape, dtype=np.uint8)
    image[10:20, 10:20] = 200
    labels[10:20, 10:20] = LABEL_EV
    image[30:40, 30:40] = 40
    labels[30:40, 30:40] = LABEL_BACKGROUND

    save_tiff(images_dir / "sample.tif", image)
    save_tiff(labels_dir / "sample_labels.tif", labels)
    return training_root


def test_train_and_predict_segmenter_round_trip(tmp_path: Path) -> None:
    training_root = _make_training_pair(tmp_path)
    model_path = tmp_path / "model.joblib"
    bundle = train_segmenter(
        training_root,
        model_path,
        config=TrainableSegmenterConfig(sigma_max=4, n_estimators=10, max_depth=5),
    )
    assert bundle.training_pair_count == 1
    assert LABEL_EV in bundle.classifier.classes_

    loaded = load_model_bundle(model_path)
    image = np.zeros((48, 48), dtype=np.uint8)
    image[12:18, 12:18] = 210
    result = predict_segmenter(image, loaded)
    assert result["mask"].dtype == bool
    assert result["ev_probability"].shape == image.shape
    assert result["labels"].shape == image.shape
    assert result["mask"][15, 15]
    assert not result["mask"][35, 35]


def test_save_model_bundle_writes_metadata(tmp_path: Path) -> None:
    training_root = _make_training_pair(tmp_path)
    model_path = tmp_path / "segmenter.joblib"
    train_segmenter(
        training_root,
        model_path,
        config=TrainableSegmenterConfig(sigma_max=4, n_estimators=5, max_depth=3),
    )
    metadata_path = model_path.with_suffix(".json")
    assert metadata_path.is_file()
    assert "classes" in metadata_path.read_text(encoding="utf-8")


def test_train_segmenter_requires_labeled_pixels(tmp_path: Path) -> None:
    training_root = tmp_path / "training_data"
    images_dir = training_root / "images"
    labels_dir = training_root / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    save_tiff(images_dir / "empty.tif", np.zeros((16, 16), dtype=np.uint8))
    save_tiff(labels_dir / "empty_labels.tif", np.zeros((16, 16), dtype=np.uint8))
    with pytest.raises(ValueError, match="No labeled training pixels"):
        train_segmenter(training_root, tmp_path / "model.joblib")
