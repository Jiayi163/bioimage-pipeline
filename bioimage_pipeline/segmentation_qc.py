"""QC metrics and overlays for classifier-predicted segmentation masks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from skimage import io as skio

from bioimage_pipeline.io import read_tiff
from bioimage_pipeline.qc import create_mask_overlay
from bioimage_pipeline.segment import label_objects

DEFAULT_TINY_AREA = 9
DEFAULT_HUGE_AREA = 500


@dataclass
class MaskQcSummary:
    """Per-image QC summary for a predicted EV mask."""

    image_name: str
    object_count: int
    foreground_fraction: float
    median_object_area: float
    tiny_object_fraction: float
    huge_object_fraction: float
    negative_control_false_positive_count: int | None = None
    warnings: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_predicted_mask(
    mask: np.ndarray,
    *,
    image_name: str,
    tiny_area: int = DEFAULT_TINY_AREA,
    huge_area: int = DEFAULT_HUGE_AREA,
    is_negative_control: bool = False,
) -> MaskQcSummary:
    """Compute QC metrics for one binary EV mask."""
    mask_arr = np.asarray(mask).astype(bool)
    labels = label_objects(mask_arr)
    object_ids = [int(value) for value in np.unique(labels) if value > 0]
    areas = [int(np.sum(labels == obj_id)) for obj_id in object_ids]
    pixel_count = mask_arr.size
    foreground_pixels = int(mask_arr.sum())
    foreground_fraction = foreground_pixels / pixel_count if pixel_count else 0.0

    tiny_count = sum(1 for area in areas if area < tiny_area)
    huge_count = sum(1 for area in areas if area > huge_area)
    object_count = len(areas)
    median_area = float(np.median(areas)) if areas else 0.0
    tiny_fraction = tiny_count / object_count if object_count else 0.0
    huge_fraction = huge_count / object_count if object_count else 0.0

    warnings: list[str] = []
    if object_count == 0:
        warnings.append("No foreground objects detected.")
    if foreground_fraction > 0.5:
        warnings.append("Foreground fraction exceeds 50%.")
    if tiny_fraction > 0.5 and object_count > 0:
        warnings.append("More than half of objects are tiny.")
    false_positive_count = object_count if is_negative_control else None
    if is_negative_control and object_count > 0:
        warnings.append("Negative-control image has detected EV objects.")

    return MaskQcSummary(
        image_name=image_name,
        object_count=object_count,
        foreground_fraction=foreground_fraction,
        median_object_area=median_area,
        tiny_object_fraction=tiny_fraction,
        huge_object_fraction=huge_fraction,
        negative_control_false_positive_count=false_positive_count,
        warnings=warnings or None,
    )


def summarize_mask_folder(
    masks_dir: str | Path,
    *,
    pattern: str = "*_ev_mask.tif",
    negative_control_names: set[str] | None = None,
) -> list[MaskQcSummary]:
    """Summarize all predicted masks in a folder."""
    root = Path(masks_dir)
    negatives = {name.lower() for name in (negative_control_names or set())}
    summaries: list[MaskQcSummary] = []
    for mask_path in sorted(root.glob(pattern)):
        if not mask_path.is_file():
            continue
        mask = read_tiff(mask_path)
        is_negative = any(token in mask_path.name.lower() for token in negatives)
        summaries.append(
            summarize_predicted_mask(
                mask,
                image_name=mask_path.name,
                is_negative_control=is_negative,
            )
        )
    return summaries


def save_mask_qc_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    output_path: str | Path,
) -> Path:
    """Write a PNG overlay for visual review."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    overlay = create_mask_overlay(image, mask)
    skio.imsave(destination, overlay)
    return destination


def save_segmentation_qc_report(
    summaries: list[MaskQcSummary],
    logs_dir: str | Path,
    *,
    basename: str = "segmentation_qc",
) -> dict[str, Path]:
    """Persist QC summaries as JSON and a plain-text report."""
    logs_path = Path(logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)
    json_path = logs_path / f"{basename}.json"
    text_path = logs_path / f"{basename}.txt"
    payload = [summary.to_dict() for summary in summaries]
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["Segmentation QC summary", ""]
    for summary in summaries:
        lines.append(f"Image: {summary.image_name}")
        lines.append(f"  object_count: {summary.object_count}")
        lines.append(f"  foreground_fraction: {summary.foreground_fraction:.4f}")
        lines.append(f"  median_object_area: {summary.median_object_area:.1f}")
        lines.append(f"  tiny_object_fraction: {summary.tiny_object_fraction:.4f}")
        lines.append(f"  huge_object_fraction: {summary.huge_object_fraction:.4f}")
        if summary.negative_control_false_positive_count is not None:
            lines.append(
                "  negative_control_false_positive_count: "
                f"{summary.negative_control_false_positive_count}"
            )
        if summary.warnings:
            for warning in summary.warnings:
                lines.append(f"  warning: {warning}")
        lines.append("")
    text_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json": json_path, "text": text_path}
