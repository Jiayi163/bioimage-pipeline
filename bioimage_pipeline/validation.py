"""Real-data validation and cross-engine comparison helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

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
class ObjectLevelComparison:
    """Object-level agreement between predicted and reference segmentations."""

    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    count_error: int
    count_ratio: float | None
    predicted_object_count: int
    reference_object_count: int
    match_iou_threshold: float


@dataclass
class SegmentationComparison:
    """Combined pixel- and object-level segmentation agreement."""

    pixel: MaskComparison
    object_level: ObjectLevelComparison


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


def _object_masks_from_labels(
    labels: np.ndarray,
    object_ids: Sequence[int],
) -> dict[int, np.ndarray]:
    mapping: dict[int, np.ndarray] = {}
    for object_id in object_ids:
        mapping[int(object_id)] = labels == object_id
    return mapping


def _label_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection / union) if union else 0.0


def compare_objects(
    predicted_mask: np.ndarray,
    reference_mask: np.ndarray,
    *,
    match_iou_threshold: float = 0.3,
) -> ObjectLevelComparison:
    """Match labeled objects and compute precision, recall, and F1."""
    pred = _normalize_mask(predicted_mask)
    ref = _normalize_mask(reference_mask)
    if pred.shape != ref.shape:
        raise ValueError(f"Mask shapes do not match: {pred.shape} vs {ref.shape}")

    pred_labels = label_objects(pred)
    ref_labels = label_objects(ref)
    pred_ids = [int(value) for value in np.unique(pred_labels) if value > 0]
    ref_ids = [int(value) for value in np.unique(ref_labels) if value > 0]

    if not pred_ids and not ref_ids:
        return ObjectLevelComparison(
            true_positives=0,
            false_positives=0,
            false_negatives=0,
            precision=1.0,
            recall=1.0,
            f1=1.0,
            count_error=0,
            count_ratio=1.0,
            predicted_object_count=0,
            reference_object_count=0,
            match_iou_threshold=match_iou_threshold,
        )

    if not pred_ids:
        return ObjectLevelComparison(
            true_positives=0,
            false_positives=0,
            false_negatives=len(ref_ids),
            precision=0.0,
            recall=0.0,
            f1=0.0,
            count_error=len(ref_ids),
            count_ratio=0.0,
            predicted_object_count=0,
            reference_object_count=len(ref_ids),
            match_iou_threshold=match_iou_threshold,
        )

    if not ref_ids:
        return ObjectLevelComparison(
            true_positives=0,
            false_positives=len(pred_ids),
            false_negatives=0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            count_error=len(pred_ids),
            count_ratio=None,
            predicted_object_count=len(pred_ids),
            reference_object_count=0,
            match_iou_threshold=match_iou_threshold,
        )

    pred_masks = _object_masks_from_labels(pred_labels, pred_ids)
    ref_masks = _object_masks_from_labels(ref_labels, ref_ids)

    iou_matrix = np.zeros((len(pred_ids), len(ref_ids)), dtype=float)
    for row, pred_id in enumerate(pred_ids):
        for col, ref_id in enumerate(ref_ids):
            iou_matrix[row, col] = _label_iou(
                pred_masks[pred_id],
                ref_masks[ref_id],
            )

    from scipy.optimize import linear_sum_assignment

    row_indices, col_indices = linear_sum_assignment(-iou_matrix)
    matched_pairs = 0
    for row, col in zip(row_indices, col_indices, strict=True):
        if iou_matrix[row, col] >= match_iou_threshold:
            matched_pairs += 1

    false_positives = len(pred_ids) - matched_pairs
    false_negatives = len(ref_ids) - matched_pairs
    true_positives = matched_pairs

    precision = (
        float(true_positives / (true_positives + false_positives))
        if (true_positives + false_positives)
        else 0.0
    )
    recall = (
        float(true_positives / (true_positives + false_negatives))
        if (true_positives + false_negatives)
        else 0.0
    )
    f1 = (
        float(2 * precision * recall / (precision + recall))
        if (precision + recall)
        else 0.0
    )
    count_error = abs(len(pred_ids) - len(ref_ids))
    count_ratio = float(len(pred_ids) / len(ref_ids)) if len(ref_ids) else None

    return ObjectLevelComparison(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
        count_error=count_error,
        count_ratio=count_ratio,
        predicted_object_count=len(pred_ids),
        reference_object_count=len(ref_ids),
        match_iou_threshold=match_iou_threshold,
    )


def compare_segmentation(
    predicted_mask: np.ndarray,
    reference_mask: np.ndarray,
    *,
    match_iou_threshold: float = 0.3,
) -> SegmentationComparison:
    """Compare predicted and reference masks at pixel and object level."""
    pixel = compare_masks(predicted_mask, reference_mask)
    object_level = compare_objects(
        predicted_mask,
        reference_mask,
        match_iou_threshold=match_iou_threshold,
    )
    return SegmentationComparison(pixel=pixel, object_level=object_level)


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
