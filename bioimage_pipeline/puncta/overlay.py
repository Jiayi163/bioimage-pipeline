"""Overlay rendering for puncta declumping."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from skimage import draw, io as skio

from bioimage_pipeline.puncta.types import DeclumpResult, PunctumCandidate
from bioimage_pipeline.qc import normalize_for_display


class OverlayRenderer:
    """Draw accepted/rejected puncta with fit-quality visualization."""

    def __init__(self, *, cross_half_size: int = 3) -> None:
        self.cross_half_size = cross_half_size

    def render_fiji_overlay(
        self,
        image: np.ndarray,
        result: DeclumpResult,
        *,
        show_rejected: bool = True,
    ) -> np.ndarray:
        """RGB overlay for Fiji: green=fit_ok, yellow=fallback, red=rejected/suspicious."""
        base = normalize_for_display(image)
        overlay = np.stack([base, base, base], axis=-1).astype(np.float32)

        for candidate in result.accepted:
            if candidate.fit_status == "fit_ok":
                if candidate.under_split_suspect:
                    color = (255, 0, 0)
                else:
                    color = (0, 255, 0)
            elif candidate.fit_status == "fit_failed_fallback":
                color = (255, 220, 0)
            else:
                color = (255, 128, 0)
            self._draw_cross(overlay, candidate, color=color)

        if show_rejected:
            for candidate in result.rejected:
                self._draw_cross(overlay, candidate, color=(255, 0, 0))

        return np.clip(overlay, 0, 255).astype(np.uint8)

    def render(
        self,
        image: np.ndarray,
        result: DeclumpResult,
        *,
        show_rejected: bool = False,
        show_sigma_circles: bool = True,
        show_seed_shifts: bool = True,
    ) -> np.ndarray:
        """Create an RGB overlay with puncta markers and optional fit annotations."""
        base = normalize_for_display(image)
        overlay = np.stack([base, base, base], axis=-1).astype(np.float32)

        for candidate in result.accepted:
            if candidate.fit_status == "fit_ok":
                color = (0, 255, 0)
                if show_sigma_circles and candidate.sigma is not None:
                    self._draw_sigma_circle(overlay, candidate, color=(0, 200, 0))
                if show_seed_shifts and candidate.center_shift is not None and candidate.center_shift > 0.25:
                    self._draw_shift_line(overlay, candidate, color=(0, 180, 255))
            elif candidate.fit_status == "fit_failed_fallback":
                color = (255, 220, 0)
            else:
                color = (255, 128, 0)
            self._draw_cross(overlay, candidate, color=color)

        if show_rejected:
            for candidate in result.rejected:
                self._draw_cross(overlay, candidate, color=(255, 0, 0))

        return np.clip(overlay, 0, 255).astype(np.uint8)

    def render_image_only_diagnostic(
        self,
        image: np.ndarray,
        result: DeclumpResult,
        *,
        show_rejected: bool = False,
    ) -> np.ndarray:
        """Diagnostic overlay for image-only mode."""
        base = normalize_for_display(image)
        overlay = np.stack([base, base, base], axis=-1).astype(np.float32)
        diag = result.image_only_diagnostics

        if diag is not None and diag.signal_support is not None:
            self._draw_support_contour(overlay, diag.signal_support)

        if diag is not None:
            for peak in diag.validated_peaks:
                self._draw_peak_marker(
                    overlay,
                    int(round(peak.row)),
                    int(round(peak.col)),
                    color=(255, 255, 0),
                )
            for group in diag.peak_groups:
                if group.route == "gmm":
                    self._draw_group_bbox(overlay, group.bbox, color=(255, 128, 0))

        for candidate in result.accepted:
            color = (0, 255, 0) if candidate.fit_status == "fit_ok" else (255, 220, 0)
            self._draw_cross(overlay, candidate, color=color)

        if show_rejected:
            for candidate in result.rejected:
                self._draw_cross(overlay, candidate, color=(255, 0, 0))

        return np.clip(overlay, 0, 255).astype(np.uint8)

    def save_image_only_diagnostic(
        self,
        path: str | Path,
        image: np.ndarray,
        result: DeclumpResult,
        *,
        show_rejected: bool = False,
    ) -> Path:
        """Save image-only diagnostic overlay."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure = self.render_image_only_diagnostic(
            image,
            result,
            show_rejected=show_rejected,
        )
        skio.imsave(output_path, figure, check_contrast=False)
        return output_path

    def save(
        self,
        path: str | Path,
        image: np.ndarray,
        result: DeclumpResult,
        *,
        show_rejected: bool = False,
        show_sigma_circles: bool = True,
        show_seed_shifts: bool = True,
    ) -> Path:
        """Render and save an overlay image."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure = self.render(
            image,
            result,
            show_rejected=show_rejected,
            show_sigma_circles=show_sigma_circles,
            show_seed_shifts=show_seed_shifts,
        )
        skio.imsave(output_path, figure, check_contrast=False)
        return output_path

    def _draw_cross(
        self,
        overlay: np.ndarray,
        candidate: PunctumCandidate,
        *,
        color: tuple[int, int, int],
    ) -> None:
        row = int(round(candidate.final_row))
        col = int(round(candidate.final_col))
        half = self.cross_half_size
        height, width = overlay.shape[:2]
        color_arr = np.array(color, dtype=np.float32)

        for delta in range(-half, half + 1):
            cross_row = row + delta
            cross_col = col + delta
            if 0 <= cross_row < height and 0 <= col < width:
                overlay[cross_row, col] = color_arr
            if 0 <= row < height and 0 <= cross_col < width:
                overlay[row, cross_col] = color_arr

    def _draw_sigma_circle(
        self,
        overlay: np.ndarray,
        candidate: PunctumCandidate,
        *,
        color: tuple[int, int, int],
    ) -> None:
        if candidate.sigma is None:
            return
        radius = max(1.0, float(candidate.sigma or 1.0) * 2.355 / 2.0)
        row = candidate.final_row
        col = candidate.final_col
        height, width = overlay.shape[:2]
        rr, cc = draw.circle_perimeter(
            int(round(row)),
            int(round(col)),
            radius=int(round(radius)),
            shape=(height, width),
        )
        color_arr = np.array(color, dtype=np.float32)
        overlay[rr, cc] = color_arr

    def _draw_shift_line(
        self,
        overlay: np.ndarray,
        candidate: PunctumCandidate,
        *,
        color: tuple[int, int, int],
    ) -> None:
        if candidate.fitted_row is None or candidate.fitted_col is None:
            return
        height, width = overlay.shape[:2]
        row0 = int(round(candidate.initial_row))
        col0 = int(round(candidate.initial_col))
        row1 = int(round(candidate.fitted_row))
        col1 = int(round(candidate.fitted_col))
        rr, cc = draw.line(row0, col0, row1, col1)
        valid = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
        color_arr = np.array(color, dtype=np.float32)
        overlay[rr[valid], cc[valid]] = color_arr

    def _draw_peak_marker(
        self,
        overlay: np.ndarray,
        row: int,
        col: int,
        *,
        color: tuple[int, int, int],
        half: int = 2,
    ) -> None:
        height, width = overlay.shape[:2]
        color_arr = np.array(color, dtype=np.float32)
        for delta in range(-half, half + 1):
            r = row + delta
            c = col + delta
            if 0 <= r < height and 0 <= col < width:
                overlay[r, col] = color_arr
            if 0 <= row < height and 0 <= c < width:
                overlay[row, c] = color_arr

    def _draw_support_contour(
        self,
        overlay: np.ndarray,
        support: np.ndarray,
        *,
        color: tuple[int, int, int] = (0, 255, 255),
    ) -> None:
        from skimage import measure

        contours = measure.find_contours(support.astype(float), 0.5)
        color_arr = np.array(color, dtype=np.float32)
        height, width = overlay.shape[:2]
        for contour in contours:
            rows = np.clip(contour[:, 0].astype(int), 0, height - 1)
            cols = np.clip(contour[:, 1].astype(int), 0, width - 1)
            overlay[rows, cols] = color_arr

    def _draw_group_bbox(
        self,
        overlay: np.ndarray,
        bbox: tuple[int, int, int, int],
        *,
        color: tuple[int, int, int],
    ) -> None:
        min_row, min_col, max_row, max_col = bbox
        height, width = overlay.shape[:2]
        color_arr = np.array(color, dtype=np.float32)
        for col in range(min_col, max_col):
            if 0 <= min_row < height and 0 <= col < width:
                overlay[min_row, col] = color_arr
            if 0 <= max_row - 1 < height and 0 <= col < width:
                overlay[max_row - 1, col] = color_arr
        for row in range(min_row, max_row):
            if 0 <= row < height and 0 <= min_col < width:
                overlay[row, min_col] = color_arr
            if 0 <= row < height and 0 <= max_col - 1 < width:
                overlay[row, max_col - 1] = color_arr
