"""Native Python trainable RF segmentation for grayscale EV fluorescence TIFFs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from skimage import feature
from sklearn.ensemble import RandomForestClassifier

from bioimage_pipeline.export import export_intensity_tiff, export_label_tiff, export_mask_tiff
from bioimage_pipeline.io import read_tiff
from bioimage_pipeline.segment import label_objects, remove_small_objects_from_mask
from bioimage_pipeline.segmentation_qc import (
    save_mask_qc_overlay,
    save_segmentation_qc_report,
    summarize_predicted_mask,
)
from bioimage_pipeline.training_data import (
    LABEL_EV,
    LABEL_UNLABELED,
    TrainingPair,
    discover_training_pairs,
    validate_training_pairs,
)

DEFAULT_MODEL_FILENAME = "ev_rf_segmenter.joblib"
MAX_FEATURE_BYTES = 8 * 1024**3


@dataclass
class NormalizationSettings:
    method: str = "percentile"
    p_low: float = 1.0
    p_high: float = 99.8
    fixed_max: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FeatureSettings:
    intensity: bool = True
    edges: bool = True
    texture: bool = True
    sigma_min: float = 1.0
    sigma_max: float = 8.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClassifierSettings:
    n_estimators: int = 100
    max_depth: int | None = None
    n_jobs: int = -1
    class_weight: str | None = "balanced"
    random_state: int = 42
    max_pixels_per_class: int | None = 50_000
    ev_class_label: int = LABEL_EV
    ev_probability_threshold: float = 0.5
    min_object_size: int = 3

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RFModelBundle:
    classifier: RandomForestClassifier
    normalization: NormalizationSettings
    features: FeatureSettings
    classifier_settings: ClassifierSettings
    label_mapping: dict[str, int] = field(
        default_factory=lambda: {
            "unlabeled": LABEL_UNLABELED,
            "ev": LABEL_EV,
            "background": 2,
            "artifact": 3,
        }
    )
    training_metadata: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "normalization": self.normalization.to_dict(),
            "features": self.features.to_dict(),
            "classifier_settings": self.classifier_settings.to_dict(),
            "label_mapping": dict(self.label_mapping),
            "training_metadata": dict(self.training_metadata),
            "classes": [int(value) for value in self.classifier.classes_],
        }


@dataclass
class PredictionExport:
    image_name: str
    probability_map_path: Path
    mask_path: Path
    labels_path: Path
    overlay_path: Path | None = None
    qc_summary: dict[str, Any] | None = None


def load_grayscale_tiff(path: str | Path) -> np.ndarray:
    """Load a grayscale TIFF as float32, preserving spatial shape."""
    array = np.asarray(read_tiff(path))
    if array.ndim == 3:
        if array.shape[-1] in (1, 2, 3, 4):
            array = array[..., 0] if array.shape[-1] == 1 else array.mean(axis=-1)
        elif array.shape[0] <= 4:
            array = array[0]
        else:
            raise ValueError(f"Unsupported TIFF shape {array.shape} in {path}.")
    if array.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image in {path}, got {array.shape}.")
    return array.astype(np.float32, copy=False)


def normalize_image(
    image: np.ndarray,
    settings: NormalizationSettings,
) -> tuple[np.ndarray, dict[str, float]]:
    arr = np.asarray(image, dtype=np.float32)
    if settings.method == "fixed_16bit" and settings.fixed_max is not None:
        scale = float(settings.fixed_max)
        return np.clip(arr / scale, 0.0, 1.0), {"method": "fixed_16bit", "fixed_max": scale}
    low = float(np.percentile(arr, settings.p_low))
    high = float(np.percentile(arr, settings.p_high))
    if high <= low:
        high = low + 1.0
    normalized = np.clip((arr - low) / (high - low), 0.0, 1.0)
    return normalized, {"p_low": low, "p_high": high, "method": settings.method}


def compute_features(
    image: np.ndarray,
    settings: FeatureSettings,
    *,
    normalization: NormalizationSettings | None = None,
) -> np.ndarray:
    """Compute multiscale local features for grayscale fluorescence."""
    normalized, _ = normalize_image(image, normalization or NormalizationSettings())
    return feature.multiscale_basic_features(
        normalized,
        intensity=settings.intensity,
        edges=settings.edges,
        texture=settings.texture,
        sigma_min=settings.sigma_min,
        sigma_max=settings.sigma_max,
        channel_axis=None,
    )


def _estimate_feature_bytes(height: int, width: int, settings: FeatureSettings) -> int:
    probe = np.zeros((min(height, 8), min(width, 8)), dtype=np.float32)
    channels = compute_features(probe, settings).shape[-1]
    return height * width * channels * 4


def _check_feature_memory(image_shape: tuple[int, int], settings: FeatureSettings) -> None:
    estimated = _estimate_feature_bytes(image_shape[0], image_shape[1], settings)
    if estimated > MAX_FEATURE_BYTES:
        raise MemoryError(
            f"Estimated feature array size is {estimated / (1024**3):.2f} GiB "
            f"for shape {image_shape}. Reduce image size or add tiled prediction later."
        )


def _sample_class_pixels(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    max_pixels_per_class: int | None,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    flat_features = features.reshape(-1, features.shape[-1])
    flat_labels = labels.reshape(-1)
    mask = flat_labels > 0
    selected_features = flat_features[mask]
    selected_labels = flat_labels[mask]
    if max_pixels_per_class is None:
        return selected_features, selected_labels
    rng = np.random.default_rng(random_state)
    chunks_f: list[np.ndarray] = []
    chunks_y: list[np.ndarray] = []
    for class_id in np.unique(selected_labels):
        class_mask = selected_labels == class_id
        xf = selected_features[class_mask]
        yf = selected_labels[class_mask]
        if len(xf) > max_pixels_per_class:
            idx = rng.choice(len(xf), max_pixels_per_class, replace=False)
            xf = xf[idx]
            yf = yf[idx]
        chunks_f.append(xf)
        chunks_y.append(yf)
    return np.concatenate(chunks_f, axis=0), np.concatenate(chunks_y, axis=0)


def _pairs_from_paths(
    image_paths: Sequence[str | Path],
    label_paths: Sequence[str | Path],
) -> list[TrainingPair]:
    if len(image_paths) != len(label_paths):
        raise ValueError("image_paths and label_paths must have the same length.")
    return [TrainingPair(Path(i), Path(l)) for i, l in zip(image_paths, label_paths)]


def train_model(
    image_paths: Sequence[str | Path],
    label_paths: Sequence[str | Path],
    output_model_path: str | Path,
    *,
    normalization: NormalizationSettings | None = None,
    feature_settings: FeatureSettings | None = None,
    classifier_settings: ClassifierSettings | None = None,
) -> RFModelBundle:
    norm = normalization or NormalizationSettings()
    features_cfg = feature_settings or FeatureSettings()
    clf_cfg = classifier_settings or ClassifierSettings()
    pairs = _pairs_from_paths(image_paths, label_paths)
    all_features: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    image_names: list[str] = []
    for pair in pairs:
        image = load_grayscale_tiff(pair.image_path)
        label_mask = np.asarray(read_tiff(pair.label_path))
        if label_mask.ndim != 2 or label_mask.shape != image.shape:
            raise ValueError(f"Invalid label mask for {pair.image_path.name}.")
        _check_feature_memory(image.shape, features_cfg)
        normalized, _ = normalize_image(image, norm)
        feature_map = feature.multiscale_basic_features(
            normalized,
            intensity=features_cfg.intensity,
            edges=features_cfg.edges,
            texture=features_cfg.texture,
            sigma_min=features_cfg.sigma_min,
            sigma_max=features_cfg.sigma_max,
            channel_axis=None,
        )
        xf, yf = _sample_class_pixels(
            feature_map,
            label_mask.astype(np.int32),
            max_pixels_per_class=clf_cfg.max_pixels_per_class,
            random_state=clf_cfg.random_state,
        )
        if len(xf) == 0:
            continue
        all_features.append(xf)
        all_labels.append(yf)
        image_names.append(pair.image_path.name)
    if not all_features:
        raise ValueError("No labeled training pixels found.")
    classifier = RandomForestClassifier(
        n_estimators=clf_cfg.n_estimators,
        max_depth=clf_cfg.max_depth,
        n_jobs=clf_cfg.n_jobs,
        class_weight=clf_cfg.class_weight,
        random_state=clf_cfg.random_state,
    )
    classifier.fit(np.concatenate(all_features), np.concatenate(all_labels))
    bundle = RFModelBundle(
        classifier=classifier,
        normalization=norm,
        features=features_cfg,
        classifier_settings=clf_cfg,
        training_metadata={
            "training_image_count": len(image_names),
            "training_image_names": image_names,
            "labeled_pixel_count": int(sum(len(y) for y in all_labels)),
            "scope": (
                "EV-like fluorescence images acquired and preprocessed under "
                "comparable lab conditions."
            ),
        },
    )
    save_model(bundle, output_model_path)
    return bundle


def train_model_from_folder(
    training_data_dir: str | Path,
    output_model_path: str | Path,
    *,
    image_pattern: str = "*.tif",
    normalization: NormalizationSettings | None = None,
    feature_settings: FeatureSettings | None = None,
    classifier_settings: ClassifierSettings | None = None,
) -> RFModelBundle:
    pairs = discover_training_pairs(training_data_dir, image_pattern=image_pattern)
    warnings = validate_training_pairs(pairs)
    blocking = [m for m in warnings if "Missing" in m or "Could not read" in m]
    if blocking:
        raise ValueError("Training data validation failed:\n- " + "\n- ".join(blocking))
    return train_model(
        [p.image_path for p in pairs],
        [p.label_path for p in pairs],
        output_model_path,
        normalization=normalization,
        feature_settings=feature_settings,
        classifier_settings=classifier_settings,
    )


def save_model(bundle: RFModelBundle, output_model_path: str | Path) -> Path:
    destination = Path(output_model_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination)
    destination.with_suffix(".json").write_text(
        json.dumps(bundle.to_metadata(), indent=2), encoding="utf-8"
    )
    return destination


def load_model(model_path: str | Path) -> RFModelBundle:
    bundle = joblib.load(Path(model_path))
    if not isinstance(bundle, RFModelBundle):
        raise TypeError(f"Expected RFModelBundle, got {type(bundle)!r}.")
    return bundle


def _ev_class_index(classifier: RandomForestClassifier, ev_class: int) -> int:
    classes = [int(v) for v in classifier.classes_]
    if ev_class not in classes:
        raise ValueError(f"EV class {ev_class} not in trained classes {classes}.")
    return classes.index(ev_class)


def predict_image(
    model_path: str | Path,
    image_path: str | Path,
    output_dir: str | Path,
    *,
    threshold: float = 0.5,
    is_negative_control: bool = False,
) -> PredictionExport:
    bundle = load_model(model_path)
    image = load_grayscale_tiff(image_path)
    _check_feature_memory(image.shape, bundle.features)
    normalized, _ = normalize_image(image, bundle.normalization)
    feature_map = feature.multiscale_basic_features(
        normalized,
        intensity=bundle.features.intensity,
        edges=bundle.features.edges,
        texture=bundle.features.texture,
        sigma_min=bundle.features.sigma_min,
        sigma_max=bundle.features.sigma_max,
        channel_axis=None,
    )
    flat = feature_map.reshape(-1, feature_map.shape[-1])
    predicted = bundle.classifier.predict(flat).reshape(image.shape).astype(np.int32)
    probabilities = bundle.classifier.predict_proba(flat)
    ev_index = _ev_class_index(bundle.classifier, bundle.classifier_settings.ev_class_label)
    ev_probability = probabilities[:, ev_index].reshape(image.shape).astype(np.float32)
    mask = remove_small_objects_from_mask(
        ev_probability >= threshold,
        min_size=bundle.classifier_settings.min_object_size,
    )
    labels = label_objects(mask)
    root = Path(output_dir)
    for sub in ("probability_maps", "masks", "labels", "qc"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    prob_path = root / "probability_maps" / f"{stem}_ev_probability.tif"
    mask_path = root / "masks" / f"{stem}_ev_mask.tif"
    label_path = root / "labels" / f"{stem}_predicted_labels.tif"
    overlay_path = root / "qc" / f"{stem}_overlay.png"
    export_intensity_tiff(prob_path, ev_probability)
    export_mask_tiff(mask_path, mask)
    export_label_tiff(label_path, labels)
    save_mask_qc_overlay(normalized, mask, overlay_path)
    qc = summarize_predicted_mask(
        mask, image_name=Path(image_path).name, is_negative_control=is_negative_control
    )
    return PredictionExport(
        image_name=Path(image_path).name,
        probability_map_path=prob_path,
        mask_path=mask_path,
        labels_path=label_path,
        overlay_path=overlay_path,
        qc_summary=qc.to_dict(),
    )


def batch_predict(
    model_path: str | Path,
    input_folder: str | Path,
    output_dir: str | Path,
    *,
    pattern: str = "*.tif",
    threshold: float = 0.5,
    negative_control_names: set[str] | None = None,
) -> list[PredictionExport]:
    negatives = {n.lower() for n in (negative_control_names or set())}
    exports: list[PredictionExport] = []
    for image_path in sorted(Path(input_folder).glob(pattern)):
        if image_path.is_file():
            exports.append(
                predict_image(
                    model_path,
                    image_path,
                    output_dir,
                    threshold=threshold,
                    is_negative_control=any(t in image_path.name.lower() for t in negatives),
                )
            )
    if exports:
        save_segmentation_qc_report(
            [
                summarize_predicted_mask(
                    read_tiff(e.mask_path) > 0,
                    image_name=e.image_name,
                    is_negative_control=any(t in e.image_name.lower() for t in negatives),
                )
                for e in exports
            ],
            Path(output_dir) / "qc",
        )
    return exports


def _build_train_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train an EV RF segmenter.")
    p.add_argument("--images", type=Path)
    p.add_argument("--labels", type=Path)
    p.add_argument("--training-data", type=Path)
    p.add_argument("--output-model", type=Path, required=True)
    p.add_argument("--sigma-min", type=float, default=1.0)
    p.add_argument("--sigma-max", type=float, default=8.0)
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--max-depth", type=int, default=None)
    p.add_argument("--max-pixels-per-class", type=int, default=50_000)
    p.add_argument("--image-pattern", default="*.tif")
    return p


def _build_predict_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Predict with a trained EV RF segmenter.")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--pattern", default="*.tif")
    return p


def train_cli_main(argv: Sequence[str] | None = None) -> int:
    args = _build_train_parser().parse_args(argv)
    fs = FeatureSettings(sigma_min=args.sigma_min, sigma_max=args.sigma_max)
    cs = ClassifierSettings(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        max_pixels_per_class=args.max_pixels_per_class,
    )
    if args.training_data:
        bundle = train_model_from_folder(
            args.training_data, args.output_model,
            image_pattern=args.image_pattern, feature_settings=fs, classifier_settings=cs,
        )
    elif args.images and args.labels:
        imgs = sorted(args.images.glob(args.image_pattern))
        lbls = [args.labels / f"{p.stem}_labels{p.suffix}" for p in imgs]
        bundle = train_model(imgs, lbls, args.output_model, feature_settings=fs, classifier_settings=cs)
    else:
        raise SystemExit("Provide --training-data or both --images and --labels.")
    print(f"Trained on {bundle.training_metadata.get('training_image_count', 0)} image(s).")
    print(f"Saved: {args.output_model.resolve()}")
    return 0


def predict_cli_main(argv: Sequence[str] | None = None) -> int:
    args = _build_predict_parser().parse_args(argv)
    exports = batch_predict(args.model, args.input, args.output, pattern=args.pattern, threshold=args.threshold)
    print(f"Predicted {len(exports)} image(s) -> {args.output.resolve()}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EV trainable RF segmentation CLI.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train", parents=[_build_train_parser()], add_help=False)
    sub.add_parser("predict", parents=[_build_predict_parser()], add_help=False)
    args, extras = parser.parse_known_args(argv)
    return train_cli_main(extras) if args.command == "train" else predict_cli_main(extras)


if __name__ == "__main__":
    raise SystemExit(main())
