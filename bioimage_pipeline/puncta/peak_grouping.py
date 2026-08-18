"""Spatial grouping of validated peaks for image-only puncta detection."""

from __future__ import annotations

from typing import Literal

import numpy as np

from bioimage_pipeline.puncta.background import expand_bbox
from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.types import ImageOnlyRoutingReason, PeakCandidate, PeakGroup


def _min_pairwise_separation(peaks: list[PeakCandidate]) -> float | None:
    if len(peaks) < 2:
        return None
    min_sep = float("inf")
    for i, peak_a in enumerate(peaks):
        for peak_b in peaks[i + 1 :]:
            dist = float(np.hypot(peak_a.row - peak_b.row, peak_a.col - peak_b.col))
            min_sep = min(min_sep, dist)
    return min_sep if min_sep < float("inf") else None


def _single_linkage_clusters(
    peaks: list[PeakCandidate],
    link_distance: float,
) -> list[list[int]]:
    """Cluster peak indices by single-linkage with given link distance."""
    n = len(peaks)
    if n == 0:
        return []
    parent = list(range(n))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.hypot(peaks[i].row - peaks[j].row, peaks[i].col - peaks[j].col))
            if dist <= link_distance:
                union(i, j)

    clusters: dict[int, list[int]] = {}
    for index in range(n):
        root = find(index)
        clusters.setdefault(root, []).append(index)
    return list(clusters.values())


def _group_bbox(
    peaks: list[PeakCandidate],
    image_shape: tuple[int, int],
    margin: int,
) -> tuple[int, int, int, int]:
    rows = [int(round(p.row)) for p in peaks]
    cols = [int(round(p.col)) for p in peaks]
    raw_bbox = (min(rows), min(cols), max(rows) + 1, max(cols) + 1)
    return expand_bbox(raw_bbox, image_shape, margin)


def _peaks_pass_direct_quality_gate(
    peaks: list[PeakCandidate],
    config: PunctaDeclumpConfig,
) -> bool:
    """Conservative gate: validated peaks must look plausible for direct acceptance."""
    min_amplitude = float(config.min_amplitude)
    return all(float(peak.intensity) >= min_amplitude for peak in peaks)


def _route_group(
    peaks: list[PeakCandidate],
    config: PunctaDeclumpConfig,
) -> tuple[Literal["direct", "gmm"], ImageOnlyRoutingReason]:
    """Route a validated peak cluster using separation, not peak count alone."""
    n = len(peaks)
    if n <= 1:
        return "direct", "direct_single"

    if not _peaks_pass_direct_quality_gate(peaks, config):
        return "gmm", "gmm_unresolved_multi_peak"

    min_sep = _min_pairwise_separation(peaks)
    if min_sep is not None and min_sep >= config.min_center_separation:
        return "direct", "direct_resolved_multi_peak"

    return "gmm", "gmm_unresolved_multi_peak"


def group_peaks(
    peaks: list[PeakCandidate],
    image_shape: tuple[int, int],
    config: PunctaDeclumpConfig,
) -> list[PeakGroup]:
    """Cluster validated peaks and assign direct vs GMM routing."""
    if not peaks:
        return []

    cluster_indices = _single_linkage_clusters(
        peaks,
        config.image_only_group_link_distance,
    )
    groups: list[PeakGroup] = []
    for group_id, indices in enumerate(cluster_indices, start=1):
        group_peaks = [peaks[i] for i in indices]
        route, routing_reason = _route_group(group_peaks, config)
        bbox = _group_bbox(group_peaks, image_shape, config.image_only_patch_margin)
        groups.append(
            PeakGroup(
                group_id=-group_id,
                peak_indices=tuple(indices),
                peaks=group_peaks,
                route=route,
                bbox=bbox,
                min_pairwise_separation=_min_pairwise_separation(group_peaks),
                routing_reason=routing_reason,
            )
        )
    return groups
