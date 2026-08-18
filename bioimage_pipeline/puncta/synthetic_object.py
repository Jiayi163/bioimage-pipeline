"""Synthetic ObjectInfo and ObjectPatch builders for image-only peak groups."""

from __future__ import annotations

import math

import numpy as np

from bioimage_pipeline.puncta.background import build_object_patch, expand_bbox
from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.types import ObjectInfo, ObjectPatch, PeakCandidate, PeakGroup


def make_peak_object_info(
    peak: PeakCandidate,
    *,
    label: int,
    raw: np.ndarray,
    disk_radius: float = 2.0,
) -> ObjectInfo:
    """Build a minimal ObjectInfo for a single direct-accept peak."""
    row = int(round(peak.row))
    col = int(round(peak.col))
    height, width = raw.shape
    row = max(0, min(height - 1, row))
    col = max(0, min(width - 1, col))
    area = math.pi * disk_radius * disk_radius
    margin = max(2, int(math.ceil(disk_radius)) + 1)
    bbox = expand_bbox((row, col, row + 1, col + 1), raw.shape, margin)
    return ObjectInfo(
        label=label,
        area=area,
        equivalent_diameter=2.0 * disk_radius,
        bbox=bbox,
        centroid=(peak.row, peak.col),
        brightest_row=peak.row,
        brightest_col=peak.col,
        brightest_intensity=float(raw[row, col]),
    )


def build_peak_disk_mask(
    shape: tuple[int, int],
    peaks: list[PeakCandidate],
    *,
    disk_radius: float,
    global_support: np.ndarray | None = None,
    row_offset: int = 0,
    col_offset: int = 0,
) -> np.ndarray:
    """Union of disks around peaks, optionally intersected with global support."""
    mask = np.zeros(shape, dtype=bool)
    for peak in peaks:
        local_row = int(round(peak.row)) - row_offset
        local_col = int(round(peak.col)) - col_offset
        rr, cc = np.ogrid[: shape[0], : shape[1]]
        dist_sq = (rr - local_row) ** 2 + (cc - local_col) ** 2
        mask |= dist_sq <= disk_radius * disk_radius
    if global_support is not None:
        support_patch = np.asarray(global_support)[
            row_offset : row_offset + shape[0],
            col_offset : col_offset + shape[1],
        ]
        mask &= support_patch.astype(bool)
    return mask


def make_group_object_info(
    group: PeakGroup,
    raw: np.ndarray,
    config: PunctaDeclumpConfig,
) -> ObjectInfo:
    """Build ObjectInfo for an ambiguous peak group."""
    peaks = group.peaks
    brightest = max(peaks, key=lambda p: p.intensity)
    row = int(round(brightest.row))
    col = int(round(brightest.col))
    height, width = raw.shape
    row = max(0, min(height - 1, row))
    col = max(0, min(width - 1, col))

    disk_area = math.pi * config.image_only_peak_disk_radius ** 2
    area = disk_area * len(peaks)
    centroid_row = float(np.mean([p.row for p in peaks]))
    centroid_col = float(np.mean([p.col for p in peaks]))

    if len(peaks) >= 2:
        rows = [p.row for p in peaks]
        cols = [p.col for p in peaks]
        spread_row = max(rows) - min(rows)
        spread_col = max(cols) - min(cols)
        major = max(spread_row, spread_col, config.image_only_peak_disk_radius)
        minor = max(min(spread_row, spread_col), config.image_only_peak_disk_radius)
        elongation = major / max(minor, 1e-6)
    else:
        elongation = 1.0

    return ObjectInfo(
        label=group.group_id,
        area=area,
        equivalent_diameter=2.0 * math.sqrt(area / math.pi),
        bbox=group.bbox,
        centroid=(centroid_row, centroid_col),
        brightest_row=brightest.row,
        brightest_col=brightest.col,
        brightest_intensity=float(raw[row, col]),
        elongation=elongation,
    )


def build_group_patch_and_mask(
    image: np.ndarray,
    group: PeakGroup,
    support: np.ndarray,
    config: PunctaDeclumpConfig,
) -> tuple[ObjectPatch, np.ndarray, ObjectInfo]:
    """Build ObjectPatch and full-image synthetic mask for a peak group."""
    obj = make_group_object_info(group, image, config)
    min_row, min_col, max_row, max_col = obj.bbox
    full_mask = np.zeros(image.shape, dtype=bool)
    patch_mask = build_peak_disk_mask(
        (max_row - min_row, max_col - min_col),
        group.peaks,
        disk_radius=config.image_only_peak_disk_radius,
        global_support=support,
        row_offset=min_row,
        col_offset=min_col,
    )
    full_mask[min_row:max_row, min_col:max_col] = patch_mask
    patch = build_object_patch(image, full_mask, obj, config)
    return patch, full_mask, obj
