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
            if candidate.sigma is not None:
                color = (0, 255, 0)
                if show_sigma_circles and candidate.sigma is not None:
                    self._draw_sigma_circle(overlay, candidate, color=(0, 200, 0))
                if show_seed_shifts and candidate.center_shift is not None and candidate.center_shift > 0.25:
                    self._draw_shift_line(overlay, candidate, color=(0, 180, 255))
            else:
                color = (255, 220, 0)
            self._draw_cross(overlay, candidate, color=color)

        if show_rejected:
            for candidate in result.rejected:
                self._draw_cross(overlay, candidate, color=(255, 0, 0))

        return np.clip(overlay, 0, 255).astype(np.uint8)

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
        radius = max(1.0, float(candidate.sigma) * 2.355 / 2.0)
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
