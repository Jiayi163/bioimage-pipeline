"""Training image/label pair discovery for trainable segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bioimage_pipeline.io import read_tiff

LABEL_UNLABELED = 0
LABEL_EV = 1
LABEL_BACKGROUND = 2
LABEL_ARTIFACT = 3

KNOWN_LABEL_VALUES = frozenset({LABEL_UNLABELED, LABEL_EV, LABEL_BACKGROUND, LABEL_ARTIFACT})


@dataclass(frozen=True)
class TrainingPair:
    """One projected image and its multi-class label mask."""

    image_path: Path
    label_path: Path


def _labels_dir_for(training_data_dir: Path) -> Path:
    return training_data_dir / "labels"


def _images_dir_for(training_data_dir: Path) -> Path:
    return training_data_dir / "images"


def _label_path_for_image(image_path: Path, labels_dir: Path) -> Path:
    stem = image_path.stem
    candidates = (
        labels_dir / f"{stem}_labels.tif",
        labels_dir / f"{stem}_labels.tiff",
        labels_dir / f"{stem}.tif",
        labels_dir / f"{stem}.tiff",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def discover_training_pairs(
    training_data_dir: str | Path,
    *,
    image_pattern: str = "*.tif",
) -> list[TrainingPair]:
    """Discover image/label pairs under ``training_data/images`` and ``labels``."""
    root = Path(training_data_dir).resolve()
    images_dir = _images_dir_for(root)
    labels_dir = _labels_dir_for(root)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Training images directory not found: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Training labels directory not found: {labels_dir}")

    pairs: list[TrainingPair] = []
    for image_path in sorted(images_dir.glob(image_pattern)):
        if not image_path.is_file():
            continue
        label_path = _label_path_for_image(image_path, labels_dir)
        pairs.append(TrainingPair(image_path=image_path, label_path=label_path))
    return pairs


def validate_training_pairs(pairs: list[TrainingPair]) -> list[str]:
    """Return human-readable validation warnings for training pairs."""
    warnings: list[str] = []
    if not pairs:
        warnings.append("No training image/label pairs were discovered.")
        return warnings

    for pair in pairs:
        if not pair.image_path.is_file():
            warnings.append(f"Missing training image: {pair.image_path}")
            continue
        if not pair.label_path.is_file():
            warnings.append(
                f"Missing label mask for {pair.image_path.name}: {pair.label_path}"
            )
            continue
        try:
            image = read_tiff(pair.image_path)
            labels = read_tiff(pair.label_path)
        except OSError as exc:
            warnings.append(f"Could not read {pair.image_path.name}: {exc}")
            continue

        if image.ndim not in (2, 3):
            warnings.append(
                f"{pair.image_path.name}: expected 2D or multi-channel image, "
                f"got shape {image.shape}."
            )
        if labels.ndim != 2:
            warnings.append(
                f"{pair.label_path.name}: expected 2D label mask, got shape {labels.shape}."
            )
            continue
        if image.ndim >= 2 and labels.shape != image.shape[:2]:
            warnings.append(
                f"Shape mismatch for {pair.image_path.name}: image {image.shape[:2]} "
                f"vs labels {labels.shape}."
            )

        unique_values = set(np.unique(labels).tolist())
        labeled_pixels = unique_values - {LABEL_UNLABELED}
        if not labeled_pixels:
            warnings.append(f"{pair.label_path.name}: no labeled pixels (all zeros).")
        unknown = unique_values - KNOWN_LABEL_VALUES
        if unknown:
            warnings.append(
                f"{pair.label_path.name}: unknown label values {sorted(unknown)}."
            )
        if LABEL_EV not in unique_values:
            warnings.append(
                f"{pair.label_path.name}: no EV pixels (class {LABEL_EV}) found."
            )
    return warnings


SPLIT_MANIFEST_FILENAME = "split_manifest.json"


def load_split_manifest(training_data_dir: str | Path) -> dict[str, list[str]]:
    """Load optional train/val/test split manifest from training data root."""
    path = Path(training_data_dir) / SPLIT_MANIFEST_FILENAME
    if not path.is_file():
        return {}
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        key: [str(value) for value in payload.get(key, [])]
        for key in ("train", "val", "test")
    }


def pairs_for_split(
    pairs: list[TrainingPair],
    split_manifest: dict[str, list[str]],
    split: str,
) -> list[TrainingPair]:
    """Return pairs whose image filename is listed in the requested split."""
    allowed = {name for name in split_manifest.get(split, [])}
    if not allowed:
        raise ValueError(f"No filenames listed for split {split!r}.")
    selected = [pair for pair in pairs if pair.image_path.name in allowed]
    if not selected:
        raise ValueError(f"No training pairs matched split {split!r}.")
    return selected
