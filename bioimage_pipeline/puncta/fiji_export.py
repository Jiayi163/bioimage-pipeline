"""Fiji-friendly TIFF exports for puncta declumping results."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bioimage_pipeline.export import export_intensity_tiff, export_label_tiff
from bioimage_pipeline.fiji_tiff import save_fiji_compatible_tiff
from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.overlay import OverlayRenderer
from bioimage_pipeline.puncta.types import DeclumpResult, PunctumCandidate


def _candidate_center(candidate: PunctumCandidate) -> tuple[float, float]:
    if candidate.fit_status == "fit_ok" and candidate.fitted_row is not None:
        return float(candidate.fitted_row), float(candidate.fitted_col)
    return candidate.final_row, candidate.final_col


def _stamp_disk(
    image: np.ndarray,
    row: float,
    col: float,
    radius: int,
    value: int | float,
) -> None:
    """Stamp a filled disk using array slicing (much faster than skimage.draw.disk on full image)."""
    height, width = image.shape[:2]
    r = int(round(row))
    c = int(round(col))
    rad = max(1, int(radius))
    min_row = max(0, r - rad)
    max_row = min(height, r + rad + 1)
    min_col = max(0, c - rad)
    max_col = min(width, c + rad + 1)
    if min_row >= max_row or min_col >= max_col:
        return

    local_rows = np.arange(min_row, max_row)[:, None] - r
    local_cols = np.arange(min_col, max_col)[None, :] - c
    mask = local_rows * local_rows + local_cols * local_cols <= rad * rad
    image[min_row:max_row, min_col:max_col][mask] = value


def _stamp_cross(
    image: np.ndarray,
    row: float,
    col: float,
    *,
    half_size: int,
    value: int | float,
) -> None:
    height, width = image.shape[:2]
    r = int(round(row))
    c = int(round(col))
    half = max(1, half_size)
    for delta in range(-half, half + 1):
        rr, cc = r + delta, c
        if 0 <= rr < height and 0 <= cc < width:
            image[rr, cc] = value
        rr, cc = r, c + delta
        if 0 <= rr < height and 0 <= cc < width:
            image[rr, cc] = value


def _label_radius(config: PunctaDeclumpConfig) -> int:
    return max(1, int(round(config.fiji_label_disk_radius)))


def build_fit_ok_centers_image(
    result: DeclumpResult,
    image_shape: tuple[int, int],
    config: PunctaDeclumpConfig,
) -> np.ndarray:
    """Grayscale image with disk+cross markers at fit_ok (and optional fallback) centers."""
    height, width = image_shape
    centers = np.zeros((height, width), dtype=np.uint16)
    marker_value = np.iinfo(np.uint16).max
    radius = max(1, int(config.fiji_center_disk_radius))

    for candidate in result.candidates:
        if not candidate.accepted:
            continue
        if candidate.fit_status == "fit_ok":
            pass
        elif candidate.fit_status == "fit_failed_fallback" and config.include_fallback_in_centers:
            pass
        else:
            continue

        row, col = _candidate_center(candidate)
        _stamp_disk(centers, row, col, radius, marker_value)
        _stamp_cross(centers, row, col, half_size=radius, value=marker_value)
    return centers


def build_component_labels_image(
    result: DeclumpResult,
    image_shape: tuple[int, int],
    config: PunctaDeclumpConfig,
    *,
    gmm_only: bool = False,
) -> np.ndarray:
    """Label image: one ID per accepted Gaussian component with a small disk at its center."""
    height, width = image_shape
    labels = np.zeros((height, width), dtype=np.uint16)
    label_id = 1
    radius = _label_radius(config)

    for candidate in result.candidates:
        if not candidate.accepted:
            continue
        if candidate.fit_status != "fit_ok":
            continue
        if gmm_only and candidate.path != "gmm":
            continue
        if candidate.fitted_row is None or candidate.fitted_col is None:
            continue

        _stamp_disk(labels, candidate.fitted_row, candidate.fitted_col, radius, label_id)
        label_id += 1

    return labels


class FijiResultExporter:
    """Write puncta results as Fiji-friendly TIFF files."""

    def __init__(self, config: PunctaDeclumpConfig | None = None) -> None:
        self.config = config or PunctaDeclumpConfig()
        self.overlay_renderer = OverlayRenderer(
            cross_half_size=max(2, self.config.fit_roi_radius // 2),
        )

    def export_all(
        self,
        output_dir: str | Path,
        result: DeclumpResult,
        image: np.ndarray,
        *,
        stem: str = "puncta",
        show_rejected: bool = True,
    ) -> dict[str, Path]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        shape = image.shape[:2]

        paths: dict[str, Path] = {}
        paths["fit_ok_centers"] = self.export_fit_ok_centers(
            output_path / f"{stem}_fit_ok_centers.tif",
            result,
            shape,
        )
        paths["component_labels"] = self.export_component_labels(
            output_path / f"{stem}_component_labels.tif",
            result,
            shape,
        )
        paths["overlay_tiff"] = self.export_overlay_tiff(
            output_path / f"{stem}_overlay.tif",
            image,
            result,
            show_rejected=show_rejected,
        )
        paths["gmm_object_labels"] = self.export_gmm_object_labels(
            output_path / f"{stem}_gmm_object_labels.tif",
            result,
            shape,
        )
        return paths

    def export_fit_ok_centers(
        self,
        path: str | Path,
        result: DeclumpResult,
        image_shape: tuple[int, int],
    ) -> Path:
        image = build_fit_ok_centers_image(result, image_shape, self.config)
        return export_intensity_tiff(path, image)

    def export_component_labels(
        self,
        path: str | Path,
        result: DeclumpResult,
        image_shape: tuple[int, int],
    ) -> Path:
        labels = build_component_labels_image(result, image_shape, self.config, gmm_only=False)
        return export_label_tiff(path, labels)

    def export_gmm_object_labels(
        self,
        path: str | Path,
        result: DeclumpResult,
        image_shape: tuple[int, int],
    ) -> Path:
        labels = build_component_labels_image(result, image_shape, self.config, gmm_only=True)
        return export_label_tiff(path, labels)

    def export_overlay_tiff(
        self,
        path: str | Path,
        image: np.ndarray,
        result: DeclumpResult,
        *,
        show_rejected: bool = True,
    ) -> Path:
        figure = self.overlay_renderer.render_fiji_overlay(
            image,
            result,
            show_rejected=show_rejected,
        )
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return save_fiji_compatible_tiff(output_path, figure, imagej=True)
