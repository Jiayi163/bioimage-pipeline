"""Local background correction for puncta ROI patches."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.types import ObjectInfo, ObjectPatch


def expand_bbox(
    bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
    margin: int,
) -> tuple[int, int, int, int]:
    """Expand a bbox by margin pixels, clipped to image bounds."""
    min_row, min_col, max_row, max_col = bbox
    height, width = shape
    return (
        max(0, min_row - margin),
        max(0, min_col - margin),
        min(height, max_row + margin),
        min(width, max_col + margin),
    )


def estimate_ring_background(
    image_patch: np.ndarray,
    object_mask_patch: np.ndarray,
    *,
    ring_width: int,
) -> float:
    """Estimate background from a ring around the object mask inside the patch."""
    if not object_mask_patch.any():
        return float(np.median(image_patch))

    structure = ndimage.generate_binary_structure(2, 1)
    dilated = object_mask_patch.copy()
    for _ in range(max(1, ring_width)):
        dilated = ndimage.binary_dilation(dilated, structure=structure)

    ring = dilated & ~object_mask_patch
    if ring.any():
        return float(np.median(image_patch[ring]))

    outside = ~object_mask_patch
    if outside.any():
        return float(np.median(image_patch[outside]))
    return float(np.median(image_patch))


def build_object_patch(
    image: np.ndarray,
    object_mask: np.ndarray,
    obj: ObjectInfo,
    config: PunctaDeclumpConfig,
) -> ObjectPatch:
    """Extract and background-correct a patch for one mask object."""
    image_arr = np.asarray(image, dtype=np.float64)
    global_bbox = expand_bbox(obj.bbox, image_arr.shape, config.background_margin)
    min_row, min_col, max_row, max_col = global_bbox

    raw_patch = image_arr[min_row:max_row, min_col:max_col]
    patch_mask = object_mask[min_row:max_row, min_col:max_col].astype(bool)
    background = estimate_ring_background(
        raw_patch,
        patch_mask,
        ring_width=config.background_ring_width,
    )
    corrected = np.clip(raw_patch - background, 0.0, None)

    return ObjectPatch(
        object_id=obj.label,
        row_offset=min_row,
        col_offset=min_col,
        corrected=corrected,
        object_mask=patch_mask,
        background_level=background,
        global_bbox=global_bbox,
        raw=raw_patch.copy(),
    )


def patch_to_global(row: float, col: float, patch: ObjectPatch) -> tuple[float, float]:
    return row + patch.row_offset, col + patch.col_offset


def global_to_patch(row: float, col: float, patch: ObjectPatch) -> tuple[float, float]:
    return row - patch.row_offset, col - patch.col_offset
