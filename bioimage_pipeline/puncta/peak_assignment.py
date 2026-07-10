"""Assign image-level candidate peaks to connected mask objects."""

from __future__ import annotations

import numpy as np

from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.types import ImagePeakTable, ObjectInfo, PeakCandidate


def assign_peaks_to_objects(
    labels: np.ndarray,
    objects: list[ObjectInfo],
    peak_table: ImagePeakTable,
    config: PunctaDeclumpConfig,
) -> dict[int, list[PeakCandidate]]:
    """Map global peaks to object labels by mask containment."""
    if not objects:
        return {}

    assigned: dict[int, list[PeakCandidate]] = {obj.label: [] for obj in objects}
    label_arr = np.asarray(labels)

    for peak in peak_table.peaks:
        row = int(round(peak.row))
        col = int(round(peak.col))
        if row < 0 or col < 0 or row >= label_arr.shape[0] or col >= label_arr.shape[1]:
            continue
        label = int(label_arr[row, col])
        if label <= 0:
            continue
        if label not in assigned:
            continue
        assigned[label].append(peak)

    for obj in objects:
        peaks = assigned[obj.label]
        if not peaks:
            continue
        peaks.sort(key=lambda p: p.intensity, reverse=True)
        assigned[obj.label] = _deduplicate_peaks(peaks, config.min_center_separation)

    return assigned


def count_reliable_assigned_peaks(
    peaks: list[PeakCandidate],
    config: PunctaDeclumpConfig,
) -> int:
    """Count peaks that remain after separation-aware deduplication."""
    if not peaks:
        return 0
    return len(_deduplicate_peaks(peaks, config.min_center_separation))


def _deduplicate_peaks(
    peaks: list[PeakCandidate],
    min_separation: float,
) -> list[PeakCandidate]:
    if len(peaks) <= 1:
        return peaks
    kept: list[PeakCandidate] = []
    for peak in peaks:
        if all(
            float(np.hypot(peak.row - other.row, peak.col - other.col)) >= min_separation
            for other in kept
        ):
            kept.append(peak)
    return kept
