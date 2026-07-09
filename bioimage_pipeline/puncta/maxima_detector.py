"""Intensity-based local maxima detection inside object regions."""

from __future__ import annotations

import numpy as np
from skimage.feature import peak_local_max
from skimage.filters import difference_of_gaussians

from bioimage_pipeline.preprocess import gaussian_blur
from bioimage_pipeline.puncta.config import PunctaDeclumpConfig
from bioimage_pipeline.puncta.types import PeakCandidate, PeakDetectionResult


class MaximaDetector:
    """Find local intensity maxima on smoothed / DoG image inside a mask patch."""

    def __init__(self, config: PunctaDeclumpConfig) -> None:
        self.config = config

    def detect(
        self,
        patch_raw: np.ndarray,
        patch_mask: np.ndarray,
    ) -> PeakDetectionResult:
        """Detect raw and filtered local maxima within a cropped object patch."""
        patch_raw_arr = np.asarray(patch_raw, dtype=np.float64)
        patch_mask_arr = np.asarray(patch_mask).astype(bool)
        if patch_raw_arr.shape != patch_mask_arr.shape:
            raise ValueError("patch_raw and patch_mask must have the same shape")
        if not patch_mask_arr.any():
            return PeakDetectionResult(raw_peaks=[], filtered_peaks=[], method="empty")

        response, method = self._build_response(patch_raw_arr)
        labels = patch_mask_arr.astype(np.int32)
        threshold_abs = self._compute_threshold_abs(response, patch_mask_arr)

        # Raw peaks: allow close peaks with minimal filtering.
        raw_coords = peak_local_max(
            response,
            labels=labels,
            min_distance=1,
            threshold_abs=threshold_abs,
            exclude_border=False,
        )
        raw_peaks = self._coords_to_peaks(raw_coords, response)

        # Filtered peaks: enforce min distance + relative height / prominence.
        filtered_coords = peak_local_max(
            response,
            labels=labels,
            min_distance=max(1, self.config.min_peak_distance),
            threshold_abs=threshold_abs,
            exclude_border=False,
        )
        filtered_peaks = self._coords_to_peaks(filtered_coords, response)
        filtered_peaks = self._apply_relative_filters(filtered_peaks, response, patch_mask_arr)

        return PeakDetectionResult(
            raw_peaks=raw_peaks,
            filtered_peaks=filtered_peaks,
            method=method,
        )

    def find_in_patch(
        self,
        patch_raw: np.ndarray,
        patch_mask: np.ndarray,
    ) -> list[PeakCandidate]:
        """Backward-compatible API returning filtered peaks only."""
        return self.detect(patch_raw, patch_mask).filtered_peaks

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

    def _build_response(self, patch: np.ndarray) -> tuple[np.ndarray, str]:
        if self.config.use_dog_peaks:
            try:
                dog = difference_of_gaussians(
                    patch,
                    low_sigma=self.config.dog_sigma_small,
                    high_sigma=self.config.dog_sigma_large,
                )
                # Keep only positive DoG response (bright spots).
                response = np.clip(dog, 0.0, None)
                if float(response.max()) > 0:
                    return response, "dog"
            except Exception:
                pass

        smoothed = gaussian_blur(patch, sigma=self.config.smoothing_sigma)
        return smoothed, "gaussian_blur"

    def _compute_threshold_abs(
        self,
        response: np.ndarray,
        patch_mask: np.ndarray,
    ) -> float | None:
        masked_values = response[patch_mask]
        if masked_values.size == 0:
            return None

        background = float(np.median(masked_values))
        peak = float(np.max(masked_values))
        tolerance = self.config.peak_noise_tolerance
        relative = self.config.peak_relative_prominence * max(peak - background, 0.0)
        threshold = background + max(tolerance, relative)
        if threshold <= background:
            return background
        return threshold

    def _apply_relative_filters(
        self,
        peaks: list[PeakCandidate],
        response: np.ndarray,
        patch_mask: np.ndarray,
    ) -> list[PeakCandidate]:
        if not peaks:
            return []

        masked = response[patch_mask]
        if masked.size == 0:
            return peaks

        peak_max = float(np.max(masked))
        if peak_max <= 0:
            return peaks

        min_height = self.config.peak_min_relative_height * peak_max
        kept = [peak for peak in peaks if peak.intensity >= min_height]
        if not kept:
            # Keep the strongest peak even if below relative height.
            return [max(peaks, key=lambda p: p.intensity)]

        # Prefer spatially separated peaks: if two are closer than min_distance,
        # keep the brighter one (peak_local_max already does this, but re-check).
        min_dist = max(1, self.config.min_peak_distance)
        kept_sorted = sorted(kept, key=lambda p: p.intensity, reverse=True)
        selected: list[PeakCandidate] = []
        for peak in kept_sorted:
            if all(
                ((peak.row - other.row) ** 2 + (peak.col - other.col) ** 2) ** 0.5 >= min_dist
                for other in selected
            ):
                selected.append(peak)
        return selected

    @staticmethod
    def _coords_to_peaks(coordinates: np.ndarray, response: np.ndarray) -> list[PeakCandidate]:
        peaks: list[PeakCandidate] = []
        for row, col in coordinates:
            peaks.append(
                PeakCandidate(
                    row=float(row),
                    col=float(col),
                    intensity=float(response[int(row), int(col)]),
                )
            )
        return peaks
