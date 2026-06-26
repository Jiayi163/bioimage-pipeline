"""Trainable pixel segmentation using scikit-image features and RandomForest.

Follows the scikit-image gallery example for trainable segmentation:
``multiscale_basic_features`` + ``future.fit_segmenter`` /
``future.predict_segmenter`` (same pattern as ImageJ Trainable Weka
Segmentation / ilastik pixel classification).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
from skimage import feature, future
from sklearn.ensemble import RandomForestClassifier

from bioimage_pipeline.export import export_label_tiff, export_mask_tiff, export_intensity_tiff
from bioimage_pipeline.io import read_tiff
from bioimage_pipeline.segment import label_objects, remove_small_objects_from_mask
from bioimage_pipeline.training_data import (
    LABEL_EV,
    TrainingPair,
    discover_training_pairs,
    validate_training_pairs,
)

DEFAULT_MODEL_FILENAME = "trainable_segmenter.joblib"


@dataclass
class TrainableSegmenterConfig:
    """Configuration mirroring the scikit-image trainable segmentation example."""

    sigma_min: float = 1.0
    sigma_max: float = 16.0
    intensity: bool = True
    edges: bool = False
    texture: bool = True
    n_estimators: int = 50
    max_depth: int = 10
    max_samples: float = 0.05
    n_jobs: int = -1
    ev_class_label: int = LABEL_EV
    ev_probability_threshold: float = 0.5
    min_object_size: int = 3
    image_pattern: str = "*.tif"


@dataclass
class TrainableSegmenterBundle:
    """Serialized trainable segmenter state."""

    classifier: RandomForestClassifier
    config: TrainableSegmenterConfig
    training_pair_count: int = 0
    training_image_names: list[str] = field(default_factory=list)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "training_pair_count": self.training_pair_count,
            "training_image_names": list(self.training_image_names),
            "config": asdict(self.config),
            "classes": [int(value) for value in self.classifier.classes_],
        }


@dataclass
class PredictionExport:
    """Paths written for one predicted image."""

    image_name: str
    probability_map_path: Path
    mask_path: Path
    labels_path: Path


def _as_feature_image(image: np.ndarray) -> np.ndarray:
    """Return a 2D or H×W×C array suitable for ``multiscale_basic_features``."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[-1] in (1, 2, 3, 4):
        if arr.shape[-1] == 1:
            return arr[..., 0]
        return arr
    raise ValueError(
        f"Unsupported image shape {arr.shape}; expected 2D or H×W×C."
    )


def build_features_func(
    config: TrainableSegmenterConfig,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a feature extractor matching the gallery example."""
    kwargs: dict[str, Any] = {
        "intensity": config.intensity,
        "edges": config.edges,
        "texture": config.texture,
        "sigma_min": config.sigma_min,
        "sigma_max": config.sigma_max,
    }
    return partial(feature.multiscale_basic_features, **kwargs)


def extract_features(
    image: np.ndarray,
    config: TrainableSegmenterConfig | None = None,
    *,
    features_func: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Compute multiscale pixel features for one image."""
    cfg = config or TrainableSegmenterConfig()
    fn = features_func or build_features_func(cfg)
    feature_image = _as_feature_image(image)
    if feature_image.ndim == 3:
        return fn(feature_image, channel_axis=-1)
    return fn(feature_image)


def _new_classifier(config: TrainableSegmenterConfig) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=config.n_estimators,
        n_jobs=config.n_jobs,
        max_depth=config.max_depth,
        max_samples=config.max_samples,
    )


def _fit_classifier_from_pairs(
    pairs: list[TrainingPair],
    config: TrainableSegmenterConfig,
) -> tuple[RandomForestClassifier, Callable[[np.ndarray], np.ndarray]]:
    """Train a classifier from labeled pixels across all image/label pairs."""
    features_func = build_features_func(config)
    labeled_features: list[np.ndarray] = []
    labeled_values: list[np.ndarray] = []

    for pair in pairs:
        image = read_tiff(pair.image_path)
        labels = np.asarray(read_tiff(pair.label_path))
        if labels.ndim != 2:
            raise ValueError(f"Label mask must be 2D: {pair.label_path}")
        features = extract_features(image, config, features_func=features_func)
        mask = labels > 0
        if not np.any(mask):
            continue
        labeled_features.append(features[mask])
        labeled_values.append(labels[mask])

    if not labeled_features:
        raise ValueError("No labeled training pixels found across training pairs.")

    training_features = np.concatenate(labeled_features, axis=0)
    training_labels = np.concatenate(labeled_values, axis=0)
    classifier = _new_classifier(config)
    classifier.fit(training_features, training_labels)
    return classifier, features_func


def train_segmenter(
    training_data_dir: str | Path,
    model_path: str | Path,
    *,
    config: TrainableSegmenterConfig | None = None,
    image_pattern: str | None = None,
) -> TrainableSegmenterBundle:
    """Train a random-forest pixel classifier from labeled training images."""
    cfg = config or TrainableSegmenterConfig()
    if image_pattern is not None:
        cfg.image_pattern = image_pattern

    pairs = discover_training_pairs(training_data_dir, image_pattern=cfg.image_pattern)
    warnings = validate_training_pairs(pairs)
    blocking = [msg for msg in warnings if "Missing" in msg or "Could not read" in msg]
    if blocking:
        raise ValueError("Training data validation failed:\n- " + "\n- ".join(blocking))

    classifier, _ = _fit_classifier_from_pairs(pairs, cfg)
    bundle = TrainableSegmenterBundle(
        classifier=classifier,
        config=cfg,
        training_pair_count=len(pairs),
        training_image_names=[pair.image_path.name for pair in pairs],
    )
    save_model_bundle(bundle, model_path)
    return bundle


def save_model_bundle(bundle: TrainableSegmenterBundle, model_path: str | Path) -> Path:
    """Persist classifier bundle with joblib."""
    destination = Path(model_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination)
    metadata_path = destination.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(bundle.to_metadata(), indent=2),
        encoding="utf-8",
    )
    return destination


def load_model_bundle(model_path: str | Path) -> TrainableSegmenterBundle:
    """Load a classifier bundle saved by :func:`save_model_bundle`."""
    bundle = joblib.load(Path(model_path))
    if not isinstance(bundle, TrainableSegmenterBundle):
        raise TypeError(f"Expected TrainableSegmenterBundle, got {type(bundle)!r}.")
    return bundle


def _ev_class_index(classifier: RandomForestClassifier, ev_class: int) -> int:
    classes = [int(value) for value in classifier.classes_]
    if ev_class not in classes:
        raise ValueError(
            f"EV class label {ev_class} not present in trained classes {classes}."
        )
    return classes.index(ev_class)


def predict_segmenter(
    image: np.ndarray,
    bundle: TrainableSegmenterBundle,
) -> dict[str, np.ndarray]:
    """Predict class labels, EV probability, and binary mask for one image."""
    cfg = bundle.config
    features = extract_features(image, cfg)
    predicted = future.predict_segmenter(features, bundle.classifier)
    probabilities = bundle.classifier.predict_proba(
        features.reshape(-1, features.shape[-1])
    )
    ev_index = _ev_class_index(bundle.classifier, cfg.ev_class_label)
    ev_probability = probabilities[:, ev_index].reshape(predicted.shape)
    mask = ev_probability >= cfg.ev_probability_threshold
    mask = remove_small_objects_from_mask(mask, min_size=cfg.min_object_size)
    labels = label_objects(mask)
    return {
        "predicted": predicted.astype(np.int32),
        "ev_probability": ev_probability.astype(np.float32),
        "mask": mask,
        "labels": labels,
    }


def predict_folder(
    input_dir: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    *,
    pattern: str = "*.tif",
    config_override: TrainableSegmenterConfig | None = None,
) -> list[PredictionExport]:
    """Batch-predict classifier outputs for all images in a folder."""
    bundle = load_model_bundle(model_path)
    if config_override is not None:
        bundle.config = config_override

    input_path = Path(input_dir).resolve()
    root = Path(output_dir).resolve()
    probability_dir = root / "probability_maps"
    masks_dir = root / "masks"
    labels_dir = root / "labels"
    for directory in (probability_dir, masks_dir, labels_dir):
        directory.mkdir(parents=True, exist_ok=True)

    exports: list[PredictionExport] = []
    for image_path in sorted(input_path.glob(pattern)):
        if not image_path.is_file():
            continue
        image = read_tiff(image_path)
        result = predict_segmenter(image, bundle)
        stem = image_path.stem
        probability_path = probability_dir / f"{stem}_prob.tif"
        mask_path = masks_dir / f"{stem}_mask.tif"
        label_path = labels_dir / f"{stem}_labels.tif"
        export_intensity_tiff(probability_path, result["ev_probability"])
        export_mask_tiff(mask_path, result["mask"])
        export_label_tiff(label_path, result["labels"])
        exports.append(
            PredictionExport(
                image_name=image_path.name,
                probability_map_path=probability_path,
                mask_path=mask_path,
                labels_path=label_path,
            )
        )
    return exports
