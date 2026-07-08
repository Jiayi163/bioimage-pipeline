"""Intensity-based local maxima detection inside object regions."""

from __future__ import annotations

import numpy as np
from skimage.feature import peak_local_max

from bioimage_pipeline.preprocess import gaussian_blur
from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.types import PeakCandidate


class MaximaDetector:
    """Find local intensity maxima on smoothed raw image inside a mask patch."""

    def __init__(self, config: PunctaDeclumpConfig) -> None:
        self.config = config

    def find_in_patch(
        self,
        patch_raw: np.ndarray,
        patch_mask: np.ndarray,
    ) -> list[PeakCandidate]:
        """Detect local maxima within a cropped object patch."""
        patch_raw_arr = np.asarray(patch_raw, dtype=np.float64)
        patch_mask_arr = np.asarray(patch_mask).astype(bool)
        if patch_raw_arr.shape != patch_mask_arr.shape:
            raise ValueError("patch_raw and patch_mask must have the same shape")
        if not patch_mask_arr.any():
            return []

        smoothed = gaussian_blur(patch_raw_arr, sigma=self.config.smoothing_sigma)
        labels = patch_mask_arr.astype(np.int32)

        threshold_abs = self._compute_threshold_abs(smoothed, patch_mask_arr)

        coordinates = peak_local_max(
            smoothed,
            labels=labels,
            min_distance=max(1, self.config.min_peak_distance),
            threshold_abs=threshold_abs,
            exclude_border=True,
        )

        peaks: list[PeakCandidate] = []
        for row, col in coordinates:
            peaks.append(
                PeakCandidate(
                    row=float(row),
                    col=float(col),
                    intensity=float(smoothed[int(row), int(col)]),
                )
            )
        return peaks

    def find_in_full_image(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> list[PeakCandidate]:
        """Detect maxima in a bbox crop and return full-image coordinates."""
        min_row, min_col, max_row, max_col = bbox
        patch_raw = np.asarray(image)[min_row:max_row, min_col:max_col]
        patch_mask = np.asarray(mask)[min_row:max_row, min_col:max_col]

        patch_peaks = self.find_in_patch(patch_raw, patch_mask)
        return [
            PeakCandidate(
                row=peak.row + min_row,
                col=peak.col + min_col,
                intensity=peak.intensity,
            )
            for peak in patch_peaks
        ]

    def _compute_threshold_abs(
        self,
        smoothed: np.ndarray,
        patch_mask: np.ndarray,
    ) -> float | None:
        """Compute absolute intensity threshold for peak detection."""
        masked_values = smoothed[patch_mask]
        if masked_values.size == 0:
            return None

        background = float(np.median(masked_values))
        tolerance = self.config.peak_noise_tolerance
        if tolerance <= 0:
            return background

        return background + tolerance
