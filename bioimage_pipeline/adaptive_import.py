"""Self-adaptive thresholding for fluorescence nuclei at import time.

.. warning:: Phase 17 — DEFERRED / experimental prototype

   This module is **not** used by the default Python pipeline or CellProfiler
   workflow. Logic may need substantial tuning before production use. See
   ``DEVELOPMENT_PLAN.md`` Phase 17.

   Opt-in: ``run_cellprofiler_workflow(..., adaptive_threshold=True)`` or call
   :func:`run_self_adaptive_threshold` directly for experiments.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from skimage import morphology

from bioimage_pipeline.export import export_label_tiff, export_mask_tiff
from bioimage_pipeline.io import read_tiff
from bioimage_pipeline.preprocess import gaussian_blur, rolling_ball_subtract
from bioimage_pipeline.segment import (
    clean_mask,
    label_objects,
    split_touching_objects,
)
from bioimage_pipeline.threshold import (
    adaptive_threshold,
    otsu_threshold,
    sauvola_threshold,
)
from bioimage_pipeline.validation import inspect_image

ThresholdMethod = Literal["otsu", "local", "sauvola"]
ConfidenceLevel = Literal["high", "medium", "low"]

STAGING_MASKS_DIR = "masks"
STAGING_LABELS_DIR = "labels"
STAGING_CORRECTED_DIR = "corrected"


@dataclass
class AdaptiveThresholdDecision:
    """Record of automatic threshold choices for one image."""

    method: ThresholdMethod
    block_size: int
    offset: float
    rolling_ball_radius: int
    blur_sigma: float
    estimated_snr: float | None
    vignette_score: float
    split_touching: bool
    confidence: ConfidenceLevel
    foreground_fraction: float
    object_count: int
    warnings: list[str] = field(default_factory=list)
    fallback_used: bool = False


@dataclass
class SelfAdaptiveThresholdResult:
    """Output of the self-adaptive import threshold pipeline."""

    mask: np.ndarray
    labels: np.ndarray
    corrected_image: np.ndarray
    decision: AdaptiveThresholdDecision


def extract_2d_plane(image: np.ndarray) -> np.ndarray:
    """Reduce a microscopy image to a single 2D plane for analysis."""
    array = np.asarray(image)
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        if array.shape[0] <= 4:
            return array[0]
        return array[..., 0]
    raise ValueError(f"Unsupported image dimensionality: {array.ndim}D")


def estimate_vignette_score(image: np.ndarray) -> float:
    """Estimate uneven illumination from dark corners relative to the field."""
    plane = extract_2d_plane(image).astype(np.float32)
    height, width = plane.shape
    margin = max(4, min(height, width) // 16)
    corners = np.concatenate(
        [
            plane[:margin, :margin].ravel(),
            plane[:margin, -margin:].ravel(),
            plane[-margin:, :margin].ravel(),
            plane[-margin:, -margin:].ravel(),
        ]
    )
    corner_mean = float(corners.mean())
    field_median = float(np.median(plane))
    if field_median <= 0:
        return 0.0
    if corner_mean >= field_median * 0.95:
        return 0.0
    return (field_median - corner_mean) / field_median


def _odd_clamped(value: int, minimum: int, maximum: int) -> int:
    clamped = int(np.clip(value, minimum, maximum))
    if clamped % 2 == 0:
        clamped += 1 if clamped < maximum else -1
    return max(minimum, clamped)


def estimate_block_size(corrected_image: np.ndarray) -> int:
    """Estimate a local threshold window from coarse object scale."""
    plane = extract_2d_plane(corrected_image)
    height, width = plane.shape
    max_block = _odd_clamped(min(height, width) // 4, 15, 101)

    try:
        coarse_mask = otsu_threshold(plane)
    except ValueError:
        return _odd_clamped(max_block // 2, 15, max_block)

    if not coarse_mask.any():
        return _odd_clamped(max_block // 2, 15, max_block)

    from bioimage_pipeline.segment import distance_transform

    distances = distance_transform(coarse_mask)
    positive = distances[coarse_mask]
    if positive.size == 0:
        return _odd_clamped(max_block // 2, 15, max_block)

    nucleus_radius = float(np.percentile(positive, 75))
    diameter = max(6.0, nucleus_radius * 2.0)
    block_size = _odd_clamped(int(round(diameter * 3.0)), 15, max_block)
    return block_size


def _estimate_offset(corrected_image: np.ndarray) -> float:
    plane = extract_2d_plane(corrected_image).astype(np.float32)
    background = plane[plane <= np.percentile(plane, 30)]
    if background.size == 0:
        return 0.0
    return max(0.0, float(np.std(background)) * 0.25)


def _estimate_min_object_size(corrected_image: np.ndarray, default: int) -> int:
    block_size = estimate_block_size(corrected_image)
    radius = max(3, block_size // 6)
    area_estimate = int(np.pi * radius * radius * 0.35)
    max_pixels = max(default, corrected_image.size // 20)
    return min(max(default, area_estimate), max_pixels)


def _normalize_corrected(corrected_image: np.ndarray) -> np.ndarray:
    """Scale corrected intensities to [0, 1] using robust percentiles."""
    array = corrected_image.astype(np.float32)
    low = float(np.percentile(array, 1))
    high = float(np.percentile(array, 99.5))
    if high <= low:
        return np.zeros_like(array, dtype=np.float32)
    scaled = (array - low) / (high - low)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def _choose_primary_method(
    *,
    estimated_snr: float | None,
    vignette_score: float,
) -> ThresholdMethod:
    if estimated_snr is not None and estimated_snr < 3.0:
        return "sauvola"
    if vignette_score < 0.05 and (estimated_snr is None or estimated_snr >= 4.0):
        return "otsu"
    if estimated_snr is not None and estimated_snr >= 6.0 and vignette_score < 0.12:
        return "otsu"
    return "local"


def _threshold_with_method(
    corrected_image: np.ndarray,
    *,
    method: ThresholdMethod,
    block_size: int,
    offset: float,
) -> np.ndarray:
    plane = extract_2d_plane(corrected_image)
    if method == "otsu":
        return otsu_threshold(plane)
    if method == "sauvola":
        # Sauvola expects original intensity scale; rescale block window only.
        scaled = (plane * 255).astype(np.float32) if plane.max() <= 1.0 else plane
        dynamic_range = float(scaled.max()) if scaled.max() > 0 else 255.0
        return sauvola_threshold(
            scaled,
            block_size=block_size,
            r=dynamic_range,
        )
    # Local adaptive on normalized [0, 1] images uses a small offset.
    local_offset = min(offset, 0.05) if plane.max() <= 1.0 else offset
    return adaptive_threshold(plane, block_size=block_size, offset=local_offset)


def _mask_sanity(mask: np.ndarray) -> tuple[bool, list[str]]:
    warnings: list[str] = []
    foreground_fraction = float(mask.mean())
    if foreground_fraction <= 0.0005:
        warnings.append("Threshold produced almost no foreground.")
        return False, warnings
    if foreground_fraction >= 0.55:
        warnings.append("Threshold produced excessive foreground.")
        return False, warnings
    return True, warnings


def _should_split_touching(mask: np.ndarray) -> bool:
    labels = label_objects(mask)
    object_count = int(labels.max())
    if object_count == 0:
        return False
    if object_count == 1:
        return float(mask.mean()) > 0.01

    areas = np.bincount(labels.ravel())
    areas = areas[1:]
    if areas.size == 0:
        return False
    median_area = float(np.median(areas))
    largest_area = float(areas.max())
    if median_area > 0 and largest_area > 3.0 * median_area:
        return True
    return object_count <= 2 and float(mask.mean()) > 0.015


def _score_confidence(
    *,
    estimated_snr: float | None,
    warnings: list[str],
    fallback_used: bool,
    object_count: int,
) -> ConfidenceLevel:
    if object_count == 0 or any("almost no foreground" in w for w in warnings):
        return "low"
    if fallback_used or (estimated_snr is not None and estimated_snr < 3.0):
        return "medium"
    if estimated_snr is not None and estimated_snr >= 5.0 and not warnings:
        return "high"
    return "medium"


def run_self_adaptive_threshold(
    image: np.ndarray,
    *,
    min_object_size: int = 20,
    watershed_min_distance: int = 8,
    watershed_min_peak_ratio: float = 0.5,
) -> SelfAdaptiveThresholdResult:
    """Run the self-adaptive fluorescence nuclei threshold pipeline on one image."""
    plane = extract_2d_plane(image)
    properties = inspect_image(plane)
    vignette_score = estimate_vignette_score(plane)
    use_rolling_ball = vignette_score >= 0.05 or (
        properties.estimated_snr is not None and properties.estimated_snr < 6.0
    )
    rolling_ball_radius = max(8, min(plane.shape) // 8)

    if use_rolling_ball:
        corrected = rolling_ball_subtract(plane, radius=rolling_ball_radius)
    else:
        corrected = plane.astype(np.float32)

    blur_sigma = 0.0
    if properties.estimated_snr is not None and properties.estimated_snr < 4.0:
        blur_sigma = 1.0
        corrected = gaussian_blur(corrected, sigma=blur_sigma)

    block_size = estimate_block_size(corrected)
    offset = _estimate_offset(corrected)
    threshold_image = _normalize_corrected(corrected)
    effective_min_size = _estimate_min_object_size(corrected, min_object_size)
    primary_method = _choose_primary_method(
        estimated_snr=properties.estimated_snr,
        vignette_score=vignette_score,
    )

    warnings: list[str] = list(properties.limitations)
    fallback_used = False
    method = primary_method
    mask = _threshold_with_method(
        threshold_image,
        method=method,
        block_size=block_size,
        offset=offset,
    )
    sane, sanity_warnings = _mask_sanity(mask)
    warnings.extend(sanity_warnings)

    if not sane:
        fallback_chain: list[ThresholdMethod] = ["otsu", "local", "sauvola"]
        if primary_method in fallback_chain:
            fallback_chain.remove(primary_method)
        for fallback_method in fallback_chain:
            candidate = _threshold_with_method(
                threshold_image,
                method=fallback_method,
                block_size=block_size,
                offset=offset,
            )
            candidate_sane, candidate_warnings = _mask_sanity(candidate)
            if candidate_sane:
                mask = candidate
                method = fallback_method
                fallback_used = True
                warnings.extend(candidate_warnings)
                warnings.append(f"Fell back to {fallback_method} threshold.")
                break
        else:
            otsu_mask = _threshold_with_method(
                threshold_image,
                method="otsu",
                block_size=block_size,
                offset=offset,
            )
            if otsu_mask.any():
                mask = otsu_mask
                method = "otsu"
                fallback_used = True
                warnings.append("Used Otsu fallback after sanity checks failed.")

    mask = morphology.opening(mask, morphology.disk(1))
    mask = clean_mask(mask, min_size=effective_min_size, clear_border=False)
    if not mask.any() and method != "otsu":
        mask = clean_mask(
            _threshold_with_method(
                threshold_image,
                method="otsu",
                block_size=block_size,
                offset=offset,
            ),
            min_size=effective_min_size,
            clear_border=False,
        )
        method = "otsu"
        fallback_used = True
        warnings.append("Recovered empty mask using Otsu after morphology.")
    split_touching = _should_split_touching(mask)
    if split_touching:
        labels = split_touching_objects(
            mask,
            min_distance=watershed_min_distance,
            min_peak_ratio=watershed_min_peak_ratio,
        )
    else:
        labels = label_objects(mask)

    object_count = int(labels.max())
    decision = AdaptiveThresholdDecision(
        method=method,
        block_size=block_size,
        offset=offset,
        rolling_ball_radius=rolling_ball_radius,
        blur_sigma=blur_sigma,
        estimated_snr=properties.estimated_snr,
        vignette_score=vignette_score,
        split_touching=split_touching,
        confidence=_score_confidence(
            estimated_snr=properties.estimated_snr,
            warnings=warnings,
            fallback_used=fallback_used,
            object_count=object_count,
        ),
        foreground_fraction=float(mask.mean()),
        object_count=object_count,
        warnings=warnings,
        fallback_used=fallback_used,
    )
    return SelfAdaptiveThresholdResult(
        mask=mask,
        labels=labels,
        corrected_image=corrected,
        decision=decision,
    )


def _collect_image_paths(input_folder: Path, pattern: str) -> list[Path]:
    paths = sorted(input_folder.glob(pattern))
    if pattern == "*.tif":
        paths.extend(sorted(input_folder.glob("*.tiff")))
    return [path for path in paths if path.is_file()]


def run_self_adaptive_threshold_on_folder(
    input_dir: str | Path,
    staging_dir: str | Path,
    *,
    pattern: str = "*.tif",
    min_object_size: int = 20,
    export_corrected: bool = False,
    logs_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Threshold every image in a folder and write masks/labels to staging."""
    input_path = Path(input_dir)
    staging_path = Path(staging_dir)
    masks_dir = staging_path / STAGING_MASKS_DIR
    labels_dir = staging_path / STAGING_LABELS_DIR
    corrected_dir = staging_path / STAGING_CORRECTED_DIR

    masks_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    if export_corrected:
        corrected_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input folder not found: {input_path}")

    processed: list[str] = []
    failed: list[dict[str, str]] = []
    decisions: dict[str, dict[str, Any]] = {}

    for image_path in _collect_image_paths(input_path, pattern):
        try:
            image = read_tiff(image_path)
            result = run_self_adaptive_threshold(
                image,
                min_object_size=min_object_size,
            )
            stem = image_path.stem
            export_mask_tiff(masks_dir / f"{stem}.tif", result.mask)
            export_label_tiff(labels_dir / f"{stem}.tif", result.labels)
            if export_corrected:
                from bioimage_pipeline.io import save_tiff

                save_tiff(
                    corrected_dir / f"{stem}.tif",
                    result.corrected_image.astype(np.float32),
                )
            decisions[image_path.name] = asdict(result.decision)
            processed.append(image_path.name)
        except Exception as exc:
            failed.append({"filename": image_path.name, "error": str(exc)})

    summary = {
        "input_dir": str(input_path.resolve()),
        "staging_dir": str(staging_path.resolve()),
        "processed": processed,
        "failed": failed,
        "decisions": decisions,
    }
    log_root = Path(logs_dir) if logs_dir is not None else staging_path.parent / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    summary_path = log_root / "adaptive_threshold_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path.resolve())
    return summary
