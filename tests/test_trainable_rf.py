"""Tests for spec-compliant trainable RF segmentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bioimage_pipeline.io import save_tiff
from bioimage_pipeline.trainable_rf import (
    ClassifierSettings,
    FeatureSettings,
    batch_predict,
    compute_features,
    load_grayscale_tiff,
    load_model,
    normalize_image,
    train_model_from_folder,
)
from bioimage_pipeline.training_data import LABEL_BACKGROUND, LABEL_EV


def _make_training_pair(tmp_path: Path, shape: tuple[int, int] = (48, 48)) -> Path:
    training_root = tmp_path / "training_data"
    images_dir = training_root / "images"
    labels_dir = training_root / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    image = np.zeros(shape, dtype=np.uint16)
    labels = np.zeros(shape, dtype=np.uint8)
    image[10:20, 10:20] = 5000
    labels[10:20, 10:20] = LABEL_EV
    image[30:40, 30:40] = 500
    labels[30:40, 30:40] = LABEL_BACKGROUND
    save_tiff(images_dir / "EV_001_zmax.tif", image)
    save_tiff(labels_dir / "EV_001_zmax_labels.tif", labels)
    return training_root


def test_load_grayscale_tiff_and_normalization(tmp_path: Path) -> None:
    from bioimage_pipeline.trainable_rf import NormalizationSettings

    path = tmp_path / "img.tif"
    save_tiff(path, np.array([[0, 100], [200, 400]], dtype=np.uint16))
    image = load_grayscale_tiff(path)
    assert image.dtype == np.float32
    normalized, stats = normalize_image(image, NormalizationSettings())
    assert normalized.max() <= 1.0
    assert "p_low" in stats


def test_compute_features_grayscale_shape(tmp_path: Path) -> None:
    image = np.random.randint(0, 1000, (32, 32), dtype=np.uint16).astype(np.float32)
    features = compute_features(image, FeatureSettings(sigma_max=4))
    assert features.ndim == 3
    assert features.shape[:2] == (32, 32)


def test_train_and_batch_predict(tmp_path: Path) -> None:
    training_root = _make_training_pair(tmp_path)
    model_path = tmp_path / "ev_rf_segmenter.joblib"
    bundle = train_model_from_folder(
        training_root,
        model_path,
        feature_settings=FeatureSettings(sigma_max=4),
        classifier_settings=ClassifierSettings(
            n_estimators=10,
            max_depth=5,
            max_pixels_per_class=1000,
        ),
    )
    assert bundle.training_metadata["training_image_count"] == 1
    input_dir = training_root / "images"
    output_dir = tmp_path / "classifier_output"
    exports = batch_predict(model_path, input_dir, output_dir, threshold=0.5)
    assert len(exports) == 1
    assert exports[0].probability_map_path.name.endswith("_ev_probability.tif")
    assert exports[0].mask_path.name.endswith("_ev_mask.tif")
    assert exports[0].overlay_path is not None
    loaded = load_model(model_path)
    assert LABEL_EV in loaded.classifier.classes_
