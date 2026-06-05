"""Real-data validation and cross-engine comparison helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bioimage_pipeline.io import read_tiff
from bioimage_pipeline.measure import measure_objects
from bioimage_pipeline.segment import label_objects


@dataclass
class ImageProperties:
    """Summary of microscopy image characteristics."""

    shape: tuple[int, ...]
    dtype: str
    ndim: int
    min_value: float
    max_value: float
    dynamic_range: float
    mean_intensity: float
    background_std: float
    estimated_snr: float | None
    channel_count: int
    limitations: list[str] = field(default_factory=list)


@dataclass
class MaskComparison:
    """Agreement metrics between two binary masks."""

    iou: float
    dice: float
    pixel_agreement: float
    foreground_pixels_a: int
    foreground_pixels_b: int
    object_count_a: int
    object_count_b: int


@dataclass
class MeasurementComparison:
    """Summary comparison between two measurement tables."""

    object_count_a: int
    object_count_b: int
    count_difference: int
    count_ratio: float | None
    mean_area_a: float | None
    mean_area_b: float | None
    mean_area_difference: float | None
    area_column_a: str | None
    area_column_b: str | None


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask).astype(bool)


def _count_objects(mask: np.ndarray) -> int:
    labels = label_objects(_normalize_mask(mask))
    return int(labels.max())


def inspect_image(image: np.ndarray) -> ImageProperties:
    """Summarize image properties and flag known analysis limitations."""
    array = np.asarray(image)
    limitations: list[str] = []

    if array.ndim not in (2, 3):
        limitations.append(f"Unsupported dimensionality: {array.ndim}D")

    if array.ndim == 3:
        channel_count = array.shape[0] if array.shape[0] <= 4 else array.shape[-1]
        if channel_count > 1:
            limitations.append(
                "Multi-channel image: lightweight Python mode uses one channel only."
            )
    else:
        channel_count = 1

    working = array[0] if array.ndim == 3 and array.shape[0] <= 4 else array
    if working.ndim != 2:
        working = np.squeeze(working)
        if working.ndim != 2:
            limitations.append("Could not reduce image to a single 2D plane.")

    min_value = float(np.min(array))
    max_value = float(np.max(array))
    dynamic_range = max_value - min_value
    mean_intensity = float(np.mean(working))

    background = working[working <= np.percentile(working, 25)]
    background_std = float(np.std(background)) if background.size else 0.0
    contrast = float(np.percentile(working, 90) - np.percentile(working, 50))

    estimated_snr = None
    if background_std > 0:
        estimated_snr = contrast / background_std

    if estimated_snr is not None and estimated_snr < 3.0:
        limitations.append("Low estimated SNR: thresholding may be unstable.")
    if dynamic_range < 100:
        limitations.append("Low dynamic range: weak foreground/background separation.")
    if max(working.shape) > 4096:
        limitations.append("Large image size: processing may be slow or memory-heavy.")

    return ImageProperties(
        shape=tuple(int(value) for value in array.shape),
        dtype=str(array.dtype),
        ndim=array.ndim,
        min_value=min_value,
        max_value=max_value,
        dynamic_range=dynamic_range,
        mean_intensity=mean_intensity,
        background_std=background_std,
        estimated_snr=estimated_snr,
        channel_count=channel_count,
        limitations=limitations,
    )


def compare_masks(mask_a: np.ndarray, mask_b: np.ndarray) -> MaskComparison:
    """Compare two binary masks using IoU, Dice, and pixel agreement."""
    a = _normalize_mask(mask_a)
    b = _normalize_mask(mask_b)

    if a.shape != b.shape:
        raise ValueError(f"Mask shapes do not match: {a.shape} vs {b.shape}")

    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    total = a.size
    agreement = (a == b).sum()

    iou = float(intersection / union) if union else 1.0
    dice_denominator = a.sum() + b.sum()
    dice = float(2 * intersection / dice_denominator) if dice_denominator else 1.0
    pixel_agreement = float(agreement / total) if total else 1.0

    return MaskComparison(
        iou=iou,
        dice=dice,
        pixel_agreement=pixel_agreement,
        foreground_pixels_a=int(a.sum()),
        foreground_pixels_b=int(b.sum()),
        object_count_a=_count_objects(a),
        object_count_b=_count_objects(b),
    )


def _pick_area_column(dataframe: pd.DataFrame) -> str | None:
    candidates = ("area", "AreaShape_Area")
    for column in candidates:
        if column in dataframe.columns:
            return column
    return None


def compare_measurements(
    measurements_a: pd.DataFrame,
    measurements_b: pd.DataFrame,
) -> MeasurementComparison:
    """Compare object counts and mean area between measurement tables."""
    area_column_a = _pick_area_column(measurements_a)
    area_column_b = _pick_area_column(measurements_b)

    count_a = len(measurements_a)
    count_b = len(measurements_b)
    count_difference = count_a - count_b
    count_ratio = float(count_a / count_b) if count_b else None

    mean_area_a = (
        float(measurements_a[area_column_a].mean()) if area_column_a else None
    )
    mean_area_b = (
        float(measurements_b[area_column_b].mean()) if area_column_b else None
    )
    mean_area_difference = None
    if mean_area_a is not None and mean_area_b is not None:
        mean_area_difference = mean_area_a - mean_area_b

    return MeasurementComparison(
        object_count_a=count_a,
        object_count_b=count_b,
        count_difference=count_difference,
        count_ratio=count_ratio,
        mean_area_a=mean_area_a,
        mean_area_b=mean_area_b,
        mean_area_difference=mean_area_difference,
        area_column_a=area_column_a,
        area_column_b=area_column_b,
    )


def measurements_from_mask(
    mask: np.ndarray,
    intensity_image: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build a measurement table from a binary mask."""
    labels = label_objects(_normalize_mask(mask))
    return measure_objects(labels, intensity_image=intensity_image)


def build_validation_report(
    *,
    image_path: str | Path,
    python_mask_path: str | Path | None = None,
    reference_mask_path: str | Path | None = None,
    python_measurements: pd.DataFrame | None = None,
    reference_measurements: pd.DataFrame | None = None,
    cellprofiler_measurements: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Build a structured validation report for one image."""
    image = read_tiff(image_path)
    properties = inspect_image(image)

    report: dict[str, Any] = {
        "image_path": str(Path(image_path)),
        "image_properties": asdict(properties),
        "mask_comparison": None,
        "measurement_comparison": None,
        "cellprofiler_measurement_comparison": None,
    }

    if python_mask_path is not None and reference_mask_path is not None:
        python_mask = read_tiff(python_mask_path) > 0
        reference_mask = read_tiff(reference_mask_path) > 0
        report["mask_comparison"] = asdict(
            compare_masks(python_mask, reference_mask)
        )

    if python_measurements is None and python_mask_path is not None:
        python_mask = read_tiff(python_mask_path) > 0
        python_measurements = measurements_from_mask(python_mask, image)

    if reference_measurements is None and reference_mask_path is not None:
        reference_mask = read_tiff(reference_mask_path) > 0
        reference_measurements = measurements_from_mask(reference_mask, image)

    if python_measurements is not None and reference_measurements is not None:
        report["measurement_comparison"] = asdict(
            compare_measurements(python_measurements, reference_measurements)
        )

    if python_measurements is not None and cellprofiler_measurements is not None:
        report["cellprofiler_measurement_comparison"] = asdict(
            compare_measurements(python_measurements, cellprofiler_measurements)
        )

    return report


def write_validation_report(report: dict[str, Any], output_path: str | Path) -> Path:
    """Write a validation report as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path.resolve()
